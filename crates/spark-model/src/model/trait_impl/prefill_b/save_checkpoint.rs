// SPDX-License-Identifier: AGPL-3.0-only

//! Phase 9 — non-last chunk: save SSM snapshot at chunked-prefill block
//! boundaries (Marconi intermediate checkpoint). On future partial
//! prefix hits, restoring from the deepest intermediate checkpoint
//! avoids recomputing SSM for the entire prefix.

#![allow(unused_imports, dead_code, clippy::too_many_arguments)]

use anyhow::Result;
use spark_runtime::kv_cache::PagedKvCache;

use super::super::super::types::TransformerModel;
use crate::traits::SequenceState;

#[path = "ckpt_interval.rs"]
mod ckpt_interval;

impl TransformerModel {
    pub(in crate::model) fn prefill_b_save_checkpoint(
        &self,
        tokens: &[u32],
        seq: &mut SequenceState,
        kv_cache: &mut PagedKvCache,
        chunk_start: usize,
        chunk_len: usize,
        stream: u64,
    ) -> Result<()> {
        if self.ssm_checkpoint_interval == 0
            || !self.prefix_cache.is_active()
            || !self.ssm_snapshots.is_enabled()
        {
            return Ok(());
        }
        let bs = kv_cache.block_size();
        let end_token = chunk_start + chunk_len;
        let end_block = end_token / bs;
        // `--ssm-checkpoint-interval` is a filter over chunk ends (this runs
        // only at a chunk end; splitting chunks at interval boundaries would
        // cost prefill throughput and is not done). A chunk end fires when the
        // chunk CROSSES an interval boundary, not only when it lands on one:
        // an idle server's first chunk is the whole arena (budget + decode
        // rows), which put every later end off the interval and left cold
        // prefills with no interval checkpoint (`ckpt_interval`).
        if end_block == 0
            || !ckpt_interval::crosses_interval(
                chunk_start,
                end_token,
                bs,
                self.ssm_checkpoint_interval,
            )
        {
            return Ok(());
        }

        // Check HSS sliding-window and vision-pad skip conditions BEFORE
        // saving an SSM snapshot so we never evict live cached prefixes or run
        // GPU D2D copies only to immediately discard the snapshot.
        let boundary_tokens = &tokens[..end_token];
        let skip_boundary_insert = seq.hss_window_start() > 0 || end_block > seq.block_table.len();
        if skip_boundary_insert || self.tokens_have_vision_pad(boundary_tokens) {
            return Ok(());
        }

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
                // Pool exhausted — try to reclaim from cache
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
                    tracing::warn!(
                        "SSM snapshot pool exhausted and no evictable cached entries — \
                         dropping checkpoint for this chunk. Long-context prefix-cache \
                         hits will recompute SSM state. Consider raising \
                         --ssm-cache-slots."
                    );
                    None
                }
            }
            Err(e) => {
                tracing::warn!("SSM snapshot save error: {e}");
                None
            }
        };
        let Some(snap_id) = snap_result else {
            return Ok(());
        };

        let boundary_blocks = &seq.block_table[..end_block];
        let boundary_disk = if seq.disk_block_ids.len() >= end_block {
            &seq.disk_block_ids[..end_block]
        } else {
            &[][..]
        };
        // Intermediate checkpoint inserts tree nodes as a placeholder for
        // the snapshot boundary — the final chunk's insert will bump
        // ref_counts for seq-ownership over the full tokens range. Passing
        // matched_tokens = end_token here makes is_seq_owned=false for every
        // block in this checkpoint, avoiding a double-bump that would leak
        // the cache's baseline ref by +1 per checkpointed block after the
        // eventual release().
        let noncausal_spans = seq.rotary_positions.noncausal_spans();
        let acquired = self.prefix_cache.insert_with_noncausal_spans(
            boundary_tokens,
            boundary_blocks,
            boundary_disk,
            bs,
            end_token,
            &noncausal_spans,
        );
        super::super::super::block_mgmt::cache_acquires_disk_refs(&acquired);
        if let Some(old) = self
            .prefix_cache
            .insert_intermediate_snapshot_with_noncausal_spans(
                boundary_tokens,
                boundary_blocks,
                boundary_disk,
                bs,
                snap_id,
                seq.session_hash,
                end_token,
                &noncausal_spans,
            )
        {
            self.ssm_snapshots.free(old);
        }
        tracing::info!(
            "Intermediate SSM checkpoint saved at token {} (snapshot_id {}, block {})",
            end_token,
            snap_id,
            end_block,
        );
        Ok(())
    }
}
