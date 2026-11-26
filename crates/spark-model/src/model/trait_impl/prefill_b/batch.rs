// SPDX-License-Identifier: AGPL-3.0-only

//! `prefill_batch_chunk_dispatch` — batched prefill orchestrator (Q12).
//!
//! Dispatches `N` concurrent prefill slices through three paths:
//!   1. **Qwen4 short-batch (`qwen4_short_batch.rs`)**: packs `N > 1` Qwen4
//!      prefill slices into a shared short-batch forward when eligible, with a
//!      distinct-logits sequential fallback (`prefill_qwen4_sequential_fallback`).
//!   2. **Kernel-batched (`batch_kernel.rs`)**: when `ATLAS_Q12_BATCHED != 0`
//!      and `kernel_batched_eligible` passes, lays out uniform non-first chunks
//!      at stacked offsets and runs one outer layer loop via
//!      `prefill_attn_batched_layer` and `prefill_ssm_batched_layer`.
//!   3. **Sequential per-stream fallback (below)**: holds `self.kv_cache` once
//!      and runs the per-stream phase helpers in sequence, preserving distinct
//!      disjoint rows in `self.buffers.logits()` whenever two or more streams
//!      complete in the same batch.

#![allow(unused_imports, dead_code, clippy::too_many_arguments)]

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;

use super::super::super::types::TransformerModel;
use super::proc_range::ProcRange;
use super::upload_meta::MetaLayout;
use crate::traits::{Model, PrefillSlice, SequenceState};

#[path = "qwen4_short_batch.rs"]
mod qwen4_short_batch;

impl TransformerModel {
    /// Batched-prefill dispatch for N concurrent streams. See module docs.
    ///
    /// Returns `Vec<DevicePtr>` parallel to `streams`: each entry is the
    /// last-token logits pointer for that stream when its chunk is the last,
    /// or `DevicePtr::NULL` otherwise. Order matches `streams`.
    pub(in crate::model) fn prefill_batch_chunk_dispatch(
        &self,
        streams: &mut [PrefillSlice<'_>],
        stream: u64,
    ) -> Result<Vec<DevicePtr>> {
        let n = streams.len();
        // Q12 diagnostic: dispatch entry. Useful to confirm scheduler is
        // funneling concurrent prefills here. Debug-level by default;
        // promote with `RUST_LOG=atlas::q12=debug`.
        tracing::debug!(
            target: "atlas::q12",
            n = n,
            "prefill_batch_chunk_dispatch entry"
        );
        if n == 0 {
            return Ok(Vec::new());
        }
        let multi_chunk = crate::layers::qwen4_prefill_moe::multi_chunk_selected()?;
        if n > 1 && self.config.is_qwen4_exp() {
            if let Some(logits) = self.try_prefill_qwen4_short_batch(streams, stream)? {
                return Ok(logits);
            }
            return self.prefill_qwen4_sequential_fallback(streams, multi_chunk, stream);
        }
        if n > 1 && multi_chunk {
            // The experimental continuation path owns one exact prompt chain
            // and one persistent absolute page table at a time. Route every
            // stream through the guarded C1 entry before batched setup can
            // mutate buffers, KV ownership, or recurrent state. Changing from
            // the singular route is terminal for every live/selected lineage.
            for slice in streams.iter_mut() {
                crate::layers::qwen4_prefill_moe::attn16::mark_ordinary_for_selected_route(
                    &mut slice.seq.qwen4_prefill_lineage,
                    slice.prompt_tokens.len(),
                );
            }
            let completing_count = streams.iter().filter(|s| s.is_last_chunk).count();
            let last_completing = streams.iter().rposition(|s| s.is_last_chunk);
            let preserve_distinct = completing_count >= 2 && self.config.pp_rank == 0;
            let vocab_row_bytes = if preserve_distinct {
                self.validate_distinct_bf16_logits_rows(n, self.config.vocab_size)?
            } else {
                0
            };
            let mut saved_logits: Vec<Option<Vec<u8>>> = (0..n).map(|_| None).collect();
            let mut logits = Vec::with_capacity(n);
            for (b, slice) in streams.iter_mut().enumerate() {
                let ptr = self.prefill_chunk_dispatch(
                    slice.prompt_tokens,
                    slice.seq,
                    slice.chunk_start,
                    slice.chunk_len,
                    slice.is_last_chunk,
                    stream,
                )?;
                if preserve_distinct
                    && slice.is_last_chunk
                    && !ptr.is_null()
                    && Some(b) != last_completing
                {
                    self.gpu.synchronize(stream)?;
                    let mut host_row = vec![0u8; vocab_row_bytes];
                    self.gpu.copy_d2h(ptr, &mut host_row)?;
                    saved_logits[b] = Some(host_row);
                }
                logits.push(ptr);
            }
            if preserve_distinct && let Some(last_idx) = last_completing {
                self.restore_batched_prefill_logits_rows(
                    &mut logits,
                    &saved_logits,
                    last_idx,
                    vocab_row_bytes,
                    stream,
                )?;
            }
            return Ok(logits);
        }
        anyhow::ensure!(
            n == 1
                || streams
                    .iter()
                    .all(|s| !self.vision_prompt_present(s.prompt_tokens)
                        && s.seq.rotary_positions.is_identity()),
            "image prefill batching requires the C1 canonical path"
        );
        if n == 1 {
            // Fast path: N=1 has no batching to do. Delegate to the
            // single-stream dispatch and skip the per-stream-loop bookkeeping.
            let s = &mut streams[0];
            let logits = self.prefill_chunk_dispatch(
                s.prompt_tokens,
                s.seq,
                s.chunk_start,
                s.chunk_len,
                s.is_last_chunk,
                stream,
            )?;
            return Ok(vec![logits]);
        }

        // Buffer-arena fit check (per-stream sequential layout still respects
        // arena cap on each call). Bail loud if any stream's chunk exceeds
        // the arena — CUDA 700 territory.
        let arena_cap = self.buffers.max_batch_tokens();
        for (i, s) in streams.iter().enumerate() {
            if s.chunk_len > arena_cap {
                anyhow::bail!(
                    "Batched prefill stream {i} chunk_len={} exceeds arena \
                     capacity {arena_cap}. Reduce --max-prefill-tokens.",
                    s.chunk_len
                );
            }
        }

        // Q12 Path B: kernel-batched fast path. Check eligibility upfront
        // (cheap) and, when viable, dispatch to the outer-layer-loop
        // implementation that uses BatchedAttnMetadata + per-layer batched
        // dispatchers. Eligibility excludes every first-chunk or inherited
        // prefix-replay case before lookup, ref acquisition, block allocation,
        // or model-state mutation. Once the kernel path starts, every error is
        // propagated: falling through would repeat lookup/allocation against
        // partially-mutated streams.
        //
        // Runtime kill switch: set `ATLAS_Q12_BATCHED=0` to force-disable
        // the kernel-batched path without rebuilding. Default is enabled
        // (any unset / non-"0" value). Useful for the kernel-validation
        // session when isolating a regression to the batched path.
        let q12_batched_enabled = std::env::var("ATLAS_Q12_BATCHED")
            .map(|v| v != "0" && v.to_lowercase() != "false")
            .unwrap_or(true);
        if q12_batched_enabled && self.kernel_batched_eligible(streams) {
            tracing::debug!(
                target: "atlas::q12",
                n = n,
                chunk_len = streams[0].chunk_len,
                is_last_chunk = streams[0].is_last_chunk,
                "Q12 kernel-batched dispatch attempt"
            );
            return self.prefill_batch_chunk_kernel_batched(streams, stream);
        } else if !q12_batched_enabled {
            tracing::trace!(
                target: "atlas::q12",
                "Q12 kernel-batched disabled via ATLAS_Q12_BATCHED=0"
            );
        } else {
            // Observability: eligibility failed. Surface why so operators
            // can diagnose silent fallback. Logged at debug to avoid log
            // floods on hot paths.
            let chunk_lens: Vec<usize> = streams.iter().map(|s| s.chunk_len).collect();
            let chunk_starts: Vec<usize> = streams.iter().map(|s| s.chunk_start).collect();
            let total: usize = chunk_lens.iter().sum();
            tracing::debug!(
                target: "atlas::q12",
                n = n,
                chunk_lens = ?chunk_lens,
                chunk_starts = ?chunk_starts,
                total = total,
                arena_cap = self.buffers.max_batch_tokens(),
                head_dim = self.config.head_dim,
                model_type = self.config.model_type.as_str(),
                "Q12 kernel-batched ineligible — falling back to per-stream"
            );
        }

        // Multi-rank (EP or TP) active → NCCL needs the default stream.
        let stream = if self.multi_rank_prefill_active() {
            self.gpu.default_stream()
        } else {
            stream
        };

        // Lock KV cache once for the whole batched dispatch.
        let mut kv_cache = self.kv_cache.lock();

        let completing_count = streams.iter().filter(|s| s.is_last_chunk).count();
        let last_completing = streams.iter().rposition(|s| s.is_last_chunk);
        let preserve_distinct = completing_count >= 2 && self.config.pp_rank == 0;
        let vocab_row_bytes = if preserve_distinct {
            self.validate_distinct_bf16_logits_rows(n, self.config.vocab_size)?
        } else {
            0
        };
        let mut saved_logits: Vec<Option<Vec<u8>>> = (0..n).map(|_| None).collect();
        let mut logits_out: Vec<DevicePtr> = Vec::with_capacity(n);

        for (b, slice) in streams.iter_mut().enumerate() {
            let tokens = slice.prompt_tokens;
            let chunk_start = slice.chunk_start;
            let chunk_len = slice.chunk_len;
            let is_last_chunk = slice.is_last_chunk;
            let total = tokens.len();
            let seq = &mut *slice.seq;

            // Zero shared arena buffers on multi-rank runs (every chunk) or at
            // chunk_start == 0 to clear stale state from prior requests.
            if self.comm.is_some() || chunk_start == 0 {
                self.buffers.zero_all(self.gpu.as_ref(), stream)?;
            }

            // Phase 1+1b: embed at the shared hidden-buffer offset 0.
            // (Per-stream offsets in the buffer are deferred to Phase 2b/3
            // when the kernel-batched path actually reads N streams' worth
            // of hidden at once; today each stream's layer-loop consumes
            // offset 0 before the next stream overwrites it.)
            self.prefill_b_embed_chunk(tokens, chunk_start, chunk_len, stream)?;

            // Phase 2: prefix-cache + EP-sync + Marconi.
            let (kv_write_start, marconi_skip) = self.prefill_b_prefix_lookup(
                tokens,
                seq,
                chunk_start,
                total,
                &mut kv_cache,
                stream,
            )?;

            // Block allocation through end of chunk.
            let bs = kv_cache.block_size();
            let end_pos = chunk_start + chunk_len;
            let blocks_needed = (end_pos - 1) / bs + 1;
            super::super::super::block_mgmt::ensure_blocks_through_prefill(
                seq,
                blocks_needed - 1,
                &mut kv_cache,
                self.prefix_cache.as_ref(),
                self.gpu.as_ref(),
                stream,
            )?;

            // Phase 2b: proc range (may early-return on full prefix hit
            // of an intermediate chunk).
            let (proc_start, proc_count, effective_seq_len_start) = match self
                .prefill_b_proc_range(
                    tokens,
                    seq,
                    chunk_start,
                    chunk_len,
                    is_last_chunk,
                    kv_write_start,
                    marconi_skip,
                    stream,
                )? {
                ProcRange::Compute {
                    proc_start,
                    proc_count,
                    effective_seq_len_start,
                } => (proc_start, proc_count, effective_seq_len_start),
                ProcRange::EarlyReturn(ptr) => {
                    logits_out.push(ptr);
                    continue;
                }
            };

            // Phase 3+3b: positions / MRoPE / paged metadata.
            let MetaLayout {
                meta_base,
                slot_offset,
                pos_stream_bytes,
                use_mrope,
                needs_paged,
            } = self.prefill_b_upload_meta(
                tokens,
                seq,
                chunk_start,
                chunk_len,
                proc_start,
                proc_count,
                effective_seq_len_start,
                &kv_cache,
                stream,
            )?;

            if needs_paged {
                self.prefill_b_upload_paged(
                    seq,
                    total,
                    proc_start,
                    proc_count,
                    meta_base,
                    slot_offset,
                    &kv_cache,
                    stream,
                )?;
            }

            // Synchronise H2D before layer compute (GB10 DMA quirk —
            // see prefill_chunk_dispatch comment).
            self.gpu.synchronize(stream)?;

            // Phase 4: forward through all layers (per-stream — Phase 2b/3
            // will hoist this out of the loop with `layer.prefill_batched`).
            self.prefill_b_forward_layers(
                tokens,
                seq,
                &mut kv_cache,
                chunk_start,
                chunk_len,
                is_last_chunk,
                proc_count,
                effective_seq_len_start,
                kv_write_start,
                marconi_skip,
                meta_base,
                slot_offset,
                pos_stream_bytes,
                use_mrope,
                needs_paged,
                stream,
                None,
                false,
            )?;

            // Phase 5: update sequence state.
            seq.tokens
                .extend_from_slice(&tokens[chunk_start..chunk_start + chunk_len]);
            seq.seq_len = chunk_start + chunk_len;

            let logits = if is_last_chunk {
                let ptr = self.prefill_b_finalize_last(
                    tokens,
                    seq,
                    &mut kv_cache,
                    chunk_start,
                    chunk_len,
                    proc_count,
                    stream,
                )?;
                if preserve_distinct && !ptr.is_null() && Some(b) != last_completing {
                    self.gpu.synchronize(stream)?;
                    let mut host_row = vec![0u8; vocab_row_bytes];
                    self.gpu.copy_d2h(ptr, &mut host_row)?;
                    saved_logits[b] = Some(host_row);
                }
                ptr
            } else {
                self.prefill_b_save_checkpoint(
                    tokens,
                    seq,
                    &mut kv_cache,
                    chunk_start,
                    chunk_len,
                    stream,
                )?;
                DevicePtr::NULL
            };
            logits_out.push(logits);
        }

        if preserve_distinct && let Some(last_idx) = last_completing {
            self.restore_batched_prefill_logits_rows(
                &mut logits_out,
                &saved_logits,
                last_idx,
                vocab_row_bytes,
                stream,
            )?;
        }

        Ok(logits_out)
    }

    /// Validate that `n` completing prefill streams can each hold a distinct
    /// BF16 logits row inside `self.buffers.logits()` and return the per-row
    /// byte stride.
    pub(super) fn validate_distinct_bf16_logits_rows(
        &self,
        n: usize,
        vocab: usize,
    ) -> Result<usize> {
        anyhow::ensure!(
            !self.use_fp32_logits,
            "batched multi-stream prefill completion requires the BF16 batched logits arena \
             (use_fp32_logits scratch holds only 1 row)"
        );
        let vocab_row_bytes = vocab
            .checked_mul(2)
            .ok_or_else(|| anyhow::anyhow!("batched prefill logits row byte overflow"))?;
        let required_logits = n
            .checked_mul(vocab_row_bytes)
            .ok_or_else(|| anyhow::anyhow!("batched prefill logits total byte overflow"))?;
        anyhow::ensure!(
            required_logits <= self.buffers.sizes().logits,
            "batched prefill logits footprint ({required_logits} B) exceeds logits arena ({} B)",
            self.buffers.sizes().logits
        );
        Ok(vocab_row_bytes)
    }

    /// Copy the final completing stream's row-0 logits to its disjoint offset
    /// and upload earlier host-saved streams' logits to their own disjoint rows
    /// inside `self.buffers.logits()`.
    ///
    /// Synchronizes `stream` before cumulative H2D uploads exceed half of the
    /// 8 MiB pinned staging ring (4 MiB) so wide batches never overwrite an
    /// in-flight async DMA transfer.
    pub(super) fn restore_batched_prefill_logits_rows(
        &self,
        logits_out: &mut [DevicePtr],
        saved_logits: &[Option<Vec<u8>>],
        last_completing: usize,
        vocab_row_bytes: usize,
        stream: u64,
    ) -> Result<()> {
        use spark_runtime::gpu::HostToDeviceCopy;
        const RING_SYNC_BYTES: usize = 4 * 1024 * 1024;

        let logits_base = self.buffers.logits();
        let last_src = logits_out[last_completing];
        if !last_src.is_null() {
            let last_dst = logits_base.offset(last_completing * vocab_row_bytes);
            if last_dst != last_src {
                self.gpu
                    .copy_d2d_async(last_src, last_dst, vocab_row_bytes, stream)?;
            }
            logits_out[last_completing] = last_dst;
        }
        let mut uploaded_ring_bytes = 0usize;
        for (b, saved) in saved_logits.iter().enumerate() {
            if let Some(host_row) = saved {
                if uploaded_ring_bytes > 0
                    && uploaded_ring_bytes.saturating_add(host_row.len()) > RING_SYNC_BYTES
                {
                    self.gpu.synchronize(stream)?;
                    uploaded_ring_bytes = 0;
                }
                let dst = logits_base.offset(b * vocab_row_bytes);
                self.copy_h2d_group_fast(&[HostToDeviceCopy::new(host_row, dst)], stream)?;
                uploaded_ring_bytes = uploaded_ring_bytes.saturating_add(host_row.len());
                logits_out[b] = dst;
            }
        }
        if uploaded_ring_bytes > 0 {
            self.gpu.synchronize(stream)?;
        }
        Ok(())
    }
}
