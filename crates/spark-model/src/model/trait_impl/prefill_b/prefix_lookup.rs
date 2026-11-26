// SPDX-License-Identifier: AGPL-3.0-only

//! Phase 2: prefix-cache lookup + multi-rank (EP/TP) sync of matched count and
//! Marconi SSM snapshot restore depth. Returns `(kv_write_start, marconi_skip)`.

#![allow(unused_imports, dead_code, clippy::too_many_arguments)]

use anyhow::Result;
use spark_runtime::kv_cache::PagedKvCache;

use super::super::super::block_mgmt::reuse_prefix_match_disk_ids;
use super::super::super::types::TransformerModel;
use crate::traits::SequenceState;

/// Which SSM snapshot a rank restores after [`agree_restore`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::model) enum Agreed {
    /// No rank restores an SSM snapshot.
    None,
    /// Restore this rank's own lookup choice at its local depth.
    Local,
    /// Restore the snapshot with this ID at the (shallower) agreed depth.
    At(usize),
}

/// Cross-rank restore-depth agreement over `depth` (`0` = no usable snapshot),
/// a cross-rank min-reduction `min`, and `probe(agreed_depth)` returning this
/// rank's restorable snapshot ID at a shallower depth.
///
/// Every rank observes the exact same agreed depth (or `(0, Agreed::None)` on
/// all ranks when any rank lacks a restorable snapshot at that boundary), and
/// every rank executes the exact same collective count (`1` when `agreed == 0`,
/// `2` when `agreed > 0`).
pub(in crate::model) fn agree_restore(
    depth: usize,
    mut min: impl FnMut(u32) -> Result<u32>,
    probe: impl FnOnce(usize) -> Option<usize>,
) -> Result<(usize, Agreed)> {
    let agreed = min(u32::try_from(depth)?)? as usize;
    if agreed == 0 {
        return Ok((0, Agreed::None));
    }
    let mine = if agreed == depth {
        Some(Agreed::Local)
    } else {
        probe(agreed).map(Agreed::At)
    };
    let all = min(u32::from(mine.is_some()))? == 1;
    Ok(match mine {
        Some(r) if all => (agreed, r),
        _ => (0, Agreed::None),
    })
}

impl TransformerModel {
    /// True when prefill runs across multiple NCCL ranks (EP, pure TP, or
    /// overlapping TP+EP). Rank-local prefix caches and SSM snapshot pools can
    /// diverge in any multi-rank topology.
    pub(in crate::model) fn multi_rank_prefill_active(&self) -> bool {
        self.comm.as_ref().is_some_and(|c| {
            c.world_size() > 1
                || self.config.ep_world_size > 1
                || self.config.tp_world_size > 1
                || self.config.pp_world_size > 1
        })
    }

    /// Cross-rank `u32` min-reduction spanning the full communicator world
    /// (`comm.world_size()`, `ep_world_size`, `tp_world_size`, and `pp_world_size`).
    fn prefill_rank_min_u32(&self, val: u32) -> Result<u32> {
        let Some(comm) = self.comm.as_ref() else {
            return Ok(val);
        };
        let stream = self.gpu.default_stream();
        let world = comm
            .world_size()
            .max(self.config.ep_world_size)
            .max(self.config.tp_world_size)
            .max(self.config.pp_world_size);
        let mut min_val = val;
        for root in 0..world {
            let v = if comm.rank() == root {
                self.gpu.copy_h2d(&val.to_le_bytes(), self.ep_cmd_buf)?;
                self.gpu.synchronize(stream)?;
                comm.broadcast(self.ep_cmd_buf.0, 4, root)?;
                val
            } else {
                comm.broadcast(self.ep_cmd_buf.0, 4, root)?;
                self.gpu.synchronize(stream)?;
                let mut buf = [0u8; 4];
                self.gpu.copy_d2h(self.ep_cmd_buf, &mut buf)?;
                u32::from_le_bytes(buf)
            };
            min_val = min_val.min(v);
        }
        Ok(min_val)
    }

    /// True when this request requires populated Qwen4 QSA side-cache pages
    /// (`prompt + max_tokens >= 2048` on a Qwen4 QSA model) and must not reuse
    /// cached KV blocks inserted by shorter turns that skipped QSA indexing.
    fn seq_requires_qsa_populated_prefix(&self, seq: &SequenceState) -> bool {
        self.config.is_qwen4_exp() && self.config.qwen4_qsa && seq.qwen4_qsa_required
    }

    /// Model-aware prefix lookup requiring QSA-populated nodes, mirroring
    /// [`Self::lookup_prefill_prefix`].
    fn lookup_prefill_prefix_qsa(
        &self,
        tokens: &[u32],
        block_size: usize,
        session_hash: u64,
    ) -> spark_runtime::prefix_cache::PrefixMatch {
        if self.config.num_ssm_layers() > 0 {
            self.prefix_cache
                .lookup_paired_qsa(tokens, block_size, session_hash)
        } else {
            self.prefix_cache
                .lookup_qsa(tokens, block_size, session_hash)
        }
    }

    pub(in crate::model) fn prefill_b_prefix_lookup(
        &self,
        tokens: &[u32],
        seq: &mut SequenceState,
        chunk_start: usize,
        total: usize,
        kv_cache: &mut PagedKvCache,
        stream: u64,
    ) -> Result<(usize, bool)> {
        let bs = kv_cache.block_size();
        if chunk_start == 0 {
            let require_qsa = self.seq_requires_qsa_populated_prefix(seq);
            let mut prefix_match = if self.tokens_have_vision_pad(tokens) {
                spark_runtime::prefix_cache::PrefixMatch::empty()
            } else if require_qsa {
                self.lookup_prefill_prefix_qsa(tokens, bs, seq.session_hash)
            } else {
                self.lookup_prefill_prefix(tokens, bs, seq.session_hash)
            };
            // F83 (2026-04-30): on multi-rank (EP or TP), head and worker have
            // independent local prefix caches whose match counts can
            // diverge (eviction order differences, async insert
            // timing). If we proceed with different `matched` per
            // rank, the chunked prefill computes different proc_count
            // values → MoE/TP collective sizes mismatch → collective
            // deadlock. Sync via rooted broadcasts (one per rank,
            // accumulating min): all ranks agree on the min match.
            // If `matched_min < local_matched`, release the extra
            // matched blocks (the lookup inc_ref'd them — undo so
            // they're not leaked) and re-walk for the agreed count.
            // UNCONDITIONAL on multi-rank active, even if local
            // matched_tokens == 0, so rooted broadcasts on each rank
            // always find a matching receiver.
            let multi_rank = self.multi_rank_prefill_active();
            let has_ssm = self.config.num_ssm_layers() > 0;
            if multi_rank {
                let local = u32::try_from(prefix_match.matched_tokens)?;
                let agreed = self.prefill_rank_min_u32(local)? as usize;
                if agreed < prefix_match.matched_tokens {
                    self.prefix_cache
                        .release_matched(tokens, bs, prefix_match.matched_tokens);
                    if agreed > 0 {
                        prefix_match = if require_qsa {
                            self.lookup_prefill_prefix_qsa(&tokens[..agreed], bs, seq.session_hash)
                        } else {
                            self.lookup_prefill_prefix(&tokens[..agreed], bs, seq.session_hash)
                        };
                    } else {
                        prefix_match = spark_runtime::prefix_cache::PrefixMatch::empty();
                    }
                    tracing::info!(
                        "F83 EP-cache-sync: local_matched={local} agreed_matched={agreed} \
                         (cap to min across ranks)"
                    );
                } else if local > 0 || agreed > 0 {
                    tracing::debug!(
                        "F83 EP-cache-sync: local_matched={local} agreed_matched={agreed} (no cap)"
                    );
                }
            }
            // Re-pairing at the rank-agreed KV depth can demote one rank to a
            // miss when that exact prefix has no SSM checkpoint. Take a second
            // min so every rank either reuses the paired prefix or fully
            // prefills; divergent proc_count values deadlock collectives.
            if multi_rank && has_ssm {
                let local = u32::try_from(prefix_match.matched_tokens)?;
                let agreed = self.prefill_rank_min_u32(local)? as usize;
                if agreed < prefix_match.matched_tokens {
                    self.prefix_cache
                        .release_matched(tokens, bs, prefix_match.matched_tokens);
                    prefix_match = spark_runtime::prefix_cache::PrefixMatch::empty();
                    tracing::info!(
                        "F83 EP-cache-sync: hybrid paired miss local={local} agreed={agreed}"
                    );
                }
            }
            let matched = prefix_match.matched_tokens;
            seq.cached_prefix_tokens = matched;
            seq.prompt_len = total;
            for &block_idx in &prefix_match.matched_blocks {
                kv_cache.inc_ref(block_idx);
                seq.block_table.push(block_idx);
            }
            reuse_prefix_match_disk_ids(
                &prefix_match.matched_disk_block_ids,
                &mut seq.disk_block_ids,
            );
            // Issue #31: the prefix cache stores per-layer K/V on disk for every
            // matched block (that's the radix-tree invariant — blocks with a
            // non-MAX `disk_block_id` are fully offloaded across every attention
            // layer). Advance every layer's offload cursor to match the new
            // `disk_block_ids.len()` so the slide-before-alloc loop in
            // `block_mgmt::ensure_blocks_through_prefill` doesn't bail later
            // when it discovers `disk_last_offloaded[L] < window_start`.
            let new_total = seq.disk_block_ids.len() as u32;
            for cursor in seq.disk_last_offloaded_per_layer.iter_mut() {
                if *cursor < new_total {
                    *cursor = new_total;
                }
            }

            // Marconi: restore SSM snapshot if available, agreeing on the exact
            // `ssm_snapshot_tokens` restore boundary across all EP/TP ranks so
            // `seq.marconi_skip_to` (`proc_start` / `proc_count`) never diverges
            // when e.g. Rank 0 holds a 16K intermediate snapshot and Rank 1
            // holds an 8K snapshot on a 24K KV hit.
            let ssm_restorable = |snap_id: usize, snap_tok: usize| -> bool {
                snap_tok > 0
                    && snap_tok <= matched
                    && matched <= total
                    && super::should_restore_ssm_snapshot(snap_tok, matched, total)
                    && self
                        .ssm_snapshots
                        .session_matches(snap_id, seq.session_hash)
            };
            let local_ssm = match prefix_match.ssm_snapshot {
                Some(id) if ssm_restorable(id, prefix_match.ssm_snapshot_tokens) => {
                    (Some(id), prefix_match.ssm_snapshot_tokens)
                }
                _ => (None, 0),
            };
            let (resolved_snap_id, resolved_snap_tok) = if multi_rank && has_ssm {
                let (agreed_tok, choice) = agree_restore(
                    local_ssm.1,
                    |v| self.prefill_rank_min_u32(v),
                    |at| {
                        let probe = if require_qsa {
                            self.lookup_prefill_prefix_qsa(&tokens[..at], bs, seq.session_hash)
                        } else {
                            self.lookup_prefill_prefix(&tokens[..at], bs, seq.session_hash)
                        };
                        if probe.matched_tokens > 0 {
                            self.prefix_cache.release_matched(
                                &tokens[..at],
                                bs,
                                probe.matched_tokens,
                            );
                        }
                        match probe.ssm_snapshot {
                            Some(id)
                                if probe.ssm_snapshot_tokens == at && ssm_restorable(id, at) =>
                            {
                                Some(id)
                            }
                            _ => None,
                        }
                    },
                )?;
                let out = match choice {
                    Agreed::None => (None, 0),
                    Agreed::Local => (local_ssm.0, agreed_tok),
                    Agreed::At(id) => (Some(id), agreed_tok),
                };
                if out.1 != local_ssm.1 {
                    tracing::info!(
                        "pc rank-agree: local_ssm_tokens={} agreed_ssm_tokens={}",
                        local_ssm.1,
                        out.1
                    );
                }
                out
            } else {
                local_ssm
            };

            let mut skip = if let Some(snap_id) = resolved_snap_id {
                self.ssm_snapshots.restore(
                    snap_id,
                    seq.slot_idx,
                    &self.ssm_pool,
                    self.gpu.as_ref(),
                    stream,
                )?;
                if resolved_snap_tok < matched {
                    tracing::info!(
                        "Marconi intermediate hit: restored from checkpoint at token {} \
                         (skipping {} tokens, recomputing {} SSM tokens to match point {})",
                        resolved_snap_tok,
                        resolved_snap_tok,
                        matched - resolved_snap_tok,
                        matched,
                    );
                } else {
                    tracing::info!(
                        "Marconi SSM cache hit: {} tokens skipped ({} blocks), snapshot {}",
                        matched,
                        prefix_match.matched_blocks.len(),
                        snap_id,
                    );
                }
                true
            } else {
                if let Some(_raw_id) = prefix_match.ssm_snapshot {
                    let raw_tok = prefix_match.ssm_snapshot_tokens;
                    if raw_tok == matched
                        && matched == total
                        && !super::should_restore_ssm_snapshot(raw_tok, matched, total)
                    {
                        tracing::info!(
                            "Exact full-prompt SSM snapshot bypassed; recomputing recurrent state \
                             (ATLAS_MARCONI_EXACT=1 re-enables the unsafe shortcut)"
                        );
                    }
                }
                false
            };
            if matched > 0 && !skip && has_ssm {
                tracing::info!(
                    "Prefix cache hit: {} tokens ({} blocks) but no SSM snapshot — recomputing recurrent state",
                    matched,
                    prefix_match.matched_blocks.len(),
                );
            } else if super::attention_only_prefix_skip(matched, skip, has_ssm) {
                // F82 (2026-04-30): attention-only cache-hit skip. A hybrid
                // snapshot can be present but deliberately bypassed (exact
                // full-prompt policy); it must full-prefill recurrent state.
                skip = true;
                tracing::info!(
                    "Prefix cache hit: {} tokens ({} blocks) reused (F82+F83: attention-only skip)",
                    matched,
                    prefix_match.matched_blocks.len(),
                );
            }
            // SSM execution resumes at the agreed restored checkpoint, which may
            // be shallower than both the KV match and this rank's deepest local
            // snapshot. Using `resolved_snap_tok` guarantees `marconi_skip_to`
            // (and thus `proc_count`) is identical across every EP/TP rank.
            let skip_tokens = if skip {
                super::restored_prefix_skip_tokens(has_ssm, resolved_snap_tok, matched)
            } else {
                0
            };
            seq.marconi_skip_to = skip_tokens;
            Ok((skip_tokens, skip))
        } else if seq.marconi_skip_to > 0 {
            // Chunk 1+: inherit skip info from chunk 0's prefix cache lookup.
            Ok((seq.marconi_skip_to, true))
        } else {
            Ok((0, false))
        }
    }
}

#[cfg(test)]
mod qsa_prefix_tests {
    use spark_runtime::prefix_cache::PrefixCache;
    use spark_runtime::radix_tree::RadixTree;

    #[test]
    fn turn1_without_qsa_cannot_poison_turn2_qsa_lookup() {
        let cache = RadixTree::new();
        let bs = 16usize;
        let session = 0xDEAD_BEEFu64;
        let prompt_turn1: Vec<u32> = (0..64u32).collect();
        let prompt_turn2: Vec<u32> = (0..128u32).collect();

        // Turn 1 (`< 2048`, `qwen4_qsa_required = false`) caches blocks without
        // QSA side-cache population (`qsa_populated = false`).
        let turn1_blocks = [0u32, 1, 2, 3];
        let _ = cache.insert_with_snapshot(&prompt_turn1, &turn1_blocks, &[], bs, 1, session, 0);
        cache.release(&prompt_turn1, bs);

        // Turn 2 (`>= 2048`, `qwen4_qsa_required = true`) MUST stop before any
        // `!qsa_populated` block and miss instead of reading stale QSA pages.
        let turn2_hit = cache.lookup_paired_qsa(&prompt_turn2, bs, session);
        assert!(turn2_hit.is_empty());
        assert_eq!(turn2_hit.ssm_snapshot, None);

        // Turn 1's blocks remain reusable for another short `< 2048` turn.
        let short_hit = cache.lookup_paired(&prompt_turn2, bs, session);
        assert_eq!(short_hit.matched_tokens, 64);
        assert_eq!(short_hit.matched_blocks, turn1_blocks);
        cache.release_matched(&prompt_turn2, bs, short_hit.matched_tokens);
    }
}
