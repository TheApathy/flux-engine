// SPDX-License-Identifier: AGPL-3.0-only

#![allow(unused_imports, dead_code, clippy::too_many_arguments)]

//! Post-prefill snapshot + prefix-cache insert helper.
//!
//! Hoisted from `prefill_c.rs` (`prefill_twophase_dispatch`) to keep that
//! file under the 500 LoC cap. The single helper here mirrors the original
//! "section 8" block 1:1 — Marconi snapshot save (with reclaim retry on
//! pool full) followed by a prefix-cache insert that prefers the snapshot
//! flavour when one was successfully saved.

use spark_runtime::kv_cache::PagedKvCache;

use super::super::types::TransformerModel;
use crate::traits::SequenceState;

impl TransformerModel {
    /// Save a Marconi SSM snapshot and insert the prefilled sequence into the
    /// prefix cache, skipping both when the prompt contains vision-pad tokens
    /// so image-tainted recurrent state never causes spurious pool eviction or
    /// cross-image prefix collisions.
    pub(super) fn prefill_save_snapshot_with_vision_gate(
        &self,
        tokens: &[u32],
        seq: &SequenceState,
        kv_cache: &mut PagedKvCache,
        bs: usize,
        stream: u64,
    ) {
        if self.tokens_have_vision_pad(tokens) {
            // Vision prefill: snapshot would be image-tainted and token IDs
            // collide across distinct images, so skip both snapshot save
            // (avoiding spurious cache eviction + GPU D2D copies) and radix
            // prefix insertion.
            return;
        }
        let noncausal_spans = seq.rotary_positions.noncausal_spans();
        let save_aligned_ssm = bs > 0 && tokens.len() >= bs && tokens.len().is_multiple_of(bs);
        let mut snap_saved = None;
        if self.prefix_cache.is_active() && self.ssm_snapshots.is_enabled() && save_aligned_ssm {
            let snap_result = match self.ssm_snapshots.save(
                seq.slot_idx,
                seq.session_hash,
                Self::seq_h_is_f16(seq),
                &self.ssm_pool,
                self.gpu.as_ref(),
                stream,
            ) {
                Ok(Some(id)) => Some(id),
                Ok(None) => {
                    // Pool exhausted — evict LRU entries to reclaim a slot
                    if self
                        .ssm_snapshots
                        .reclaim_from_cache(self.prefix_cache.as_ref(), kv_cache)
                    {
                        self.ssm_snapshots
                            .save(
                                seq.slot_idx,
                                seq.session_hash,
                                Self::seq_h_is_f16(seq),
                                &self.ssm_pool,
                                self.gpu.as_ref(),
                                stream,
                            )
                            .ok()
                            .flatten()
                    } else {
                        None
                    }
                }
                Err(_) => None,
            };
            snap_saved = snap_result;
        }
        if let Some(snap_id) = snap_saved {
            let (displaced, acquired) = self.prefix_cache.insert_with_snapshot_and_noncausal_spans(
                tokens,
                &seq.block_table,
                &seq.disk_block_ids,
                bs,
                snap_id,
                seq.session_hash,
                seq.cached_prefix_tokens,
                &noncausal_spans,
            );
            super::super::block_mgmt::cache_acquires_disk_refs(&acquired);
            if let Some(old) = displaced {
                self.ssm_snapshots.free(old);
            }
        } else {
            let acquired = self.prefix_cache.insert_with_noncausal_spans(
                tokens,
                &seq.block_table,
                &seq.disk_block_ids,
                bs,
                seq.cached_prefix_tokens,
                &noncausal_spans,
            );
            super::super::block_mgmt::cache_acquires_disk_refs(&acquired);
        }
    }

    /// Save a Marconi SSM snapshot for the prefilled sequence (reclaiming
    /// from the prefix cache on pool exhaustion) and insert into the
    /// prefix cache. Skips both snapshot save and prefix-cache insertion when
    /// vision-pad tokens are present; falls back to snapshot-less insert when
    /// the snapshot pool is unavailable or the prompt length is not block-aligned.
    pub(super) fn prefill_save_snapshot_and_insert(
        &self,
        tokens: &[u32],
        seq: &SequenceState,
        kv_cache: &mut PagedKvCache,
        bs: usize,
        stream: u64,
    ) {
        if self.tokens_have_vision_pad(tokens) {
            // Vision prefill: snapshot would be image-tainted and token IDs
            // collide across distinct images, so skip both snapshot save and
            // radix prefix insertion.
            return;
        }
        let noncausal_spans = seq.rotary_positions.noncausal_spans();
        let save_aligned_ssm = bs > 0 && tokens.len() >= bs && tokens.len().is_multiple_of(bs);
        let mut snap_saved = None;
        if self.prefix_cache.is_active() && self.ssm_snapshots.is_enabled() && save_aligned_ssm {
            let snap_result = match self.ssm_snapshots.save(
                seq.slot_idx,
                seq.session_hash,
                Self::seq_h_is_f16(seq),
                &self.ssm_pool,
                self.gpu.as_ref(),
                stream,
            ) {
                Ok(Some(id)) => Some(id),
                Ok(None) => {
                    tracing::debug!("Snapshot pool full, reclaiming...");
                    if self
                        .ssm_snapshots
                        .reclaim_from_cache(self.prefix_cache.as_ref(), kv_cache)
                    {
                        self.ssm_snapshots
                            .save(
                                seq.slot_idx,
                                seq.session_hash,
                                Self::seq_h_is_f16(seq),
                                &self.ssm_pool,
                                self.gpu.as_ref(),
                                stream,
                            )
                            .ok()
                            .flatten()
                    } else {
                        tracing::debug!("Reclaim failed — no evictable snapshots");
                        None
                    }
                }
                Err(e) => {
                    tracing::warn!("SSM snapshot save error: {e}");
                    None
                }
            };
            snap_saved = snap_result;
        }
        if let Some(snap_id) = snap_saved {
            tracing::info!(
                "Saved SSM snapshot {} for {} tokens ({} blocks) [twophase]",
                snap_id,
                tokens.len(),
                seq.block_table.len(),
            );
            let (displaced, acquired) = self.prefix_cache.insert_with_snapshot_and_noncausal_spans(
                tokens,
                &seq.block_table,
                &seq.disk_block_ids,
                bs,
                snap_id,
                seq.session_hash,
                seq.cached_prefix_tokens,
                &noncausal_spans,
            );
            super::super::block_mgmt::cache_acquires_disk_refs(&acquired);
            if let Some(old) = displaced {
                self.ssm_snapshots.free(old);
            }
        } else {
            let acquired = self.prefix_cache.insert_with_noncausal_spans(
                tokens,
                &seq.block_table,
                &seq.disk_block_ids,
                bs,
                seq.cached_prefix_tokens,
                &noncausal_spans,
            );
            super::super::block_mgmt::cache_acquires_disk_refs(&acquired);
        }
    }
}
