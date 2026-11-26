// SPDX-License-Identifier: AGPL-3.0-only

//! Prefill phase C — long-context fallback / continuation.
//!
//! Uses inline-documented `unsafe` blocks for POD-slice byte views
//! (`std::slice::from_raw_parts`) and page-locked metadata staging
//! (`std::ptr::copy_nonoverlapping` + `copy_h2d_pinned_async`).

#![allow(unused_imports, dead_code, clippy::too_many_arguments)]

use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Result, bail};
use atlas_core::config::{LayerType, ModelConfig};
use spark_runtime::buffers::BufferArena;
use spark_runtime::gpu::{DevicePtr, GpuBackend, GraphHandle, HostToDeviceCopy, KernelHandle};
use spark_runtime::kv_cache::PagedKvCache;

use super::super::block_mgmt::{
    apply_evicted_blocks, ensure_blocks_through_decode, ensure_blocks_through_prefill,
    extract_layer_refs, reuse_prefix_match_disk_ids,
};
use super::super::ssm_pool::SsmStatePool;
use super::super::ssm_snapshot::SsmSnapshotPool;
use super::super::types::{PinnedMetaStaging, TransformerModel};
use crate::layer::{
    AttnMetadataDev, ForwardContext, GdnPrefillBuffers, LayerState, SsmLayerState, TransformerLayer,
};
use crate::layers::ops;
use crate::speculative::DraftProposer;
use crate::traits::{ChunkedPrefillPageMetadata, Model, SequenceState};
use crate::weight_map::{DenseWeight, MtpWeights, QuantizedWeight};

impl TransformerModel {
    pub(super) fn prefill_twophase_dispatch(
        &self,
        tokens: &[u32],
        seq: &mut SequenceState,
        chunk_size: usize,
        stream: u64,
    ) -> Result<DevicePtr> {
        crate::model::verify_failure::check()?;
        // Qwen4's direct two-phase path slices SSM phase 3 independently of
        // the request-scoped fast-prefill receipt and can carry a deferred HC
        // combine across arena-reusing layers. The scheduler capability-gates
        // Qwen4 before entry; keep this pre-effect rejection as a fail-closed
        // backstop rather than replaying any two-phase error.
        anyhow::ensure!(
            !self.config.is_qwen4_exp(),
            "Qwen4 direct two-phase prefill is unsupported; use chunked prefill"
        );
        let total_len = tokens.len();
        self.validate_vision_prompt(tokens, 0, total_len)?;
        if total_len == 0 {
            self.admit_rotary_prompt(tokens, seq, 0)?;
            return Ok(DevicePtr::NULL);
        }
        anyhow::ensure!(chunk_size > 0, "prefill chunk size must be positive");
        let multi_chunk = crate::layers::qwen4_prefill_moe::multi_chunk_selected()?;
        if multi_chunk {
            // Two-phase setup mutates prompt-wide buffers and recurrent state
            // before it reaches a layer-local admission check. Keep the
            // experimental continuation family on the guarded C1 route, but
            // never allow this alternate entrypoint to establish or resume a
            // singular fast lineage.
            crate::layers::qwen4_prefill_moe::attn16::mark_ordinary_for_split_route(
                &mut seq.qwen4_prefill_lineage,
                total_len,
                chunk_size,
            );
            let mut offset = 0;
            while offset < total_len {
                let remaining = total_len - offset;
                let chunk_len = remaining.min(chunk_size);
                let is_last = chunk_len == remaining;
                let logits = self.prefill_chunk(tokens, seq, offset, chunk_len, is_last, stream)?;
                offset += chunk_len;
                if is_last {
                    return Ok(logits);
                }
            }
            return Ok(DevicePtr::NULL);
        }
        self.admit_rotary_prompt(tokens, seq, 0)?;

        // Fall back to standard chunked prefill if the arena cannot hold all
        // tokens at once (hidden_states buffer is sized for max_batch_tokens).
        // With chunked SSM prefill, this is the normal path for long prompts —
        // the monolithic SSM prefill() carries h_state/conv_state between chunks.
        let arena_cap = self.buffers.max_batch_tokens();
        if total_len > arena_cap
            || (self.config.mrope_interleaved && self.vision_prompt_present(tokens))
        {
            anyhow::ensure!(chunk_size > 0, "vision prefill chunk size must be positive");
            tracing::info!(
                "Chunked SSM prefill: {total_len} tokens in {} chunks of {chunk_size} \
                 (arena_cap={arena_cap})",
                total_len.div_ceil(chunk_size),
            );
            let mut offset = 0;
            while offset < total_len {
                let remaining = total_len - offset;
                let chunk_len = remaining.min(chunk_size);
                let is_last = offset + chunk_len >= total_len;
                let logits = self.prefill_chunk(tokens, seq, offset, chunk_len, is_last, stream)?;
                offset += chunk_len;
                if is_last {
                    return Ok(logits);
                }
            }
            return Ok(DevicePtr::NULL);
        }

        // Fall back if GDN buffers were not allocated (no SSM layers).
        if self.gdn_buf_qkv.is_null() {
            return self.prefill_chunk(tokens, seq, 0, total_len, true, stream);
        }

        // Fall back if total_len exceeds GDN buffer capacity.
        if total_len > self.gdn_buf_max_len {
            tracing::info!(
                "prefill_twophase: total_len ({total_len}) > GDN buffer max ({}) \
                 falling back to chunked prefill",
                self.gdn_buf_max_len,
            );
            return self.prefill_chunk(tokens, seq, 0, total_len, true, stream);
        }

        // Use the caller-provided stream for compute-copy overlap,
        // unless EP is active (NCCL requires the default stream).
        let stream = if self.comm.is_some() && self.config.ep_world_size > 1 {
            self.gpu.default_stream()
        } else {
            stream
        };
        let h = self.config.hidden_size;
        let _bf16 = 2usize;
        let fp32 = if self.config.use_fp32_residual() {
            4usize
        } else {
            2usize
        };
        let hidden = self.buffers.hidden_states();
        let residual = self.buffers.residual();

        // Zero arena buffers to clear stale data from prior request.
        self.buffers.zero_all(self.gpu.as_ref(), stream)?;

        let mut kv_cache = self.kv_cache.lock();

        // ── 1. Embed ALL tokens → [total_len, H] contiguous ──
        {
            // SAFETY: `tokens` is a valid initialized `&[u32]` slice; `u8` has
            // alignment 1 and `size_of_val` matches its exact initialized byte
            // length for `copy_h2d_group_fast`.
            let token_ids_bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(
                    tokens.as_ptr() as *const u8,
                    std::mem::size_of_val(tokens),
                )
            };
            let token_ids_dev = self.buffers.scratch();
            self.copy_h2d_group_fast(
                &[HostToDeviceCopy::new(token_ids_bytes, token_ids_dev)],
                stream,
            )?;
            ops::batched_embed(
                self.gpu.as_ref(),
                self.batched_embed_kernel,
                token_ids_dev,
                self.embed_tokens.weight,
                hidden,
                total_len as u32,
                h as u32,
                stream,
            )?;
            self.scale_embeddings(hidden, total_len, stream)?;
        }

        self.splice_vision_embeddings(tokens, 0, total_len, hidden, stream)?;

        // ── 2. Prefix cache lookup + block allocation for full sequence ──
        let bs = kv_cache.block_size();
        let prefix_match = if self.tokens_have_vision_pad(tokens) {
            spark_runtime::prefix_cache::PrefixMatch::empty()
        } else {
            self.lookup_prefill_prefix(tokens, bs, seq.session_hash)
        };
        let matched = prefix_match.matched_tokens;
        seq.cached_prefix_tokens = matched;
        // Record the original prompt length for cache_sequence bookkeeping.
        seq.prompt_len = tokens.len();
        for &block_idx in &prefix_match.matched_blocks {
            kv_cache.inc_ref(block_idx);
            seq.block_table.push(block_idx);
        }
        reuse_prefix_match_disk_ids(
            &prefix_match.matched_disk_block_ids,
            &mut seq.disk_block_ids,
        );

        // Marconi: restore SSM snapshot if available (session-gated).
        let (kv_write_start, marconi_skip) = if let Some(snap_id) = prefix_match.ssm_snapshot {
            let snap_tok = prefix_match.ssm_snapshot_tokens;
            if snap_tok > 0
                && matched <= total_len
                && super::prefill_b::should_restore_ssm_snapshot(snap_tok, matched, total_len)
                && self
                    .ssm_snapshots
                    .session_matches(snap_id, seq.session_hash)
            {
                self.ssm_snapshots.restore(
                    snap_id,
                    seq.slot_idx,
                    &self.ssm_pool,
                    self.gpu.as_ref(),
                    stream,
                )?;
                tracing::info!(
                    "Marconi two-phase: restored SSM snapshot at token {snap_tok} \
                         ({matched} tokens, {} KV blocks cached)",
                    prefix_match.matched_blocks.len(),
                );
                // KV may match deeper than the surviving recurrent checkpoint.
                // Resume from the state actually restored, never from the raw
                // KV depth. This is the same paired invariant as prefill A/B.
                let skip = super::prefill_b::restored_prefix_skip_tokens(true, snap_tok, matched);
                (skip, true)
            } else {
                if snap_tok == matched && matched == total_len {
                    tracing::info!(
                        "Exact full-prompt SSM snapshot bypassed; recomputing recurrent state \
                         (ATLAS_MARCONI_EXACT=1 re-enables the unsafe shortcut)"
                    );
                }
                (0, false)
            }
        } else {
            if matched > 0 {
                tracing::info!(
                    "Prefix cache hit: {} tokens ({} blocks) but no SSM snapshot — \
                         recomputing all KV",
                    matched,
                    prefix_match.matched_blocks.len(),
                );
            }
            (0, false)
        };
        seq.marconi_skip_to = kv_write_start;

        // Allocate all KV blocks upfront for the full sequence.
        let blocks_needed = (total_len - 1) / bs + 1;
        ensure_blocks_through_prefill(
            seq,
            blocks_needed - 1,
            &mut kv_cache,
            self.prefix_cache.as_ref(),
            self.gpu.as_ref(),
            stream,
        )?;

        // Determine effective processing range (skip Marconi-cached tokens).
        // Even on an exact full-prompt hit (`ATLAS_MARCONI_EXACT=1`), process
        // at least the last token through all transformer layers so final_norm
        // and lm_head receive real post-layer hidden states rather than raw
        // input embeddings (matching `prefill_a.rs` and `prefill_b/proc_range.rs`).
        let (proc_start, proc_count) = if marconi_skip && kv_write_start >= total_len {
            (total_len - 1, 1)
        } else if marconi_skip && kv_write_start > 0 {
            (kv_write_start, total_len - kv_write_start)
        } else {
            (0, total_len)
        };

        // Re-embed only the uncached portion at hidden[0..proc_count].
        if proc_start > 0 {
            let uncached_tokens = &tokens[proc_start..];
            // SAFETY: `uncached_tokens` is a valid initialized `&[u32]` slice;
            // `u8` has alignment 1 and `size_of_val` matches its exact byte length.
            let token_ids_bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(
                    uncached_tokens.as_ptr() as *const u8,
                    std::mem::size_of_val(uncached_tokens),
                )
            };
            let token_ids_dev = self.buffers.scratch();
            self.copy_h2d_group_fast(
                &[HostToDeviceCopy::new(token_ids_bytes, token_ids_dev)],
                stream,
            )?;
            ops::batched_embed(
                self.gpu.as_ref(),
                self.batched_embed_kernel,
                token_ids_dev,
                self.embed_tokens.weight,
                hidden,
                proc_count as u32,
                h as u32,
                stream,
            )?;
            self.scale_embeddings(hidden, proc_count, stream)?;
        }

        // ── 3. Upload attention metadata for full sequence ──
        // Positions and slots for the processed range.
        let moe_scratch_bytes = proc_count * self.config.num_experts_per_tok * 4 * 2;
        let meta_offset = (moe_scratch_bytes + 7) & !7;
        let meta_base = self.buffers.scratch().offset(meta_offset);
        let slot_offset = (proc_count * 4 + 7) & !7;

        // For two-phase, we process the full uncached range — attention layers see
        // the full sequence from proc_start. If proc_start > 0, use paged attention.
        // (`!self.config.is_qwen4_exp()` is already ensured at entry.)
        let needs_paged = proc_start > 0;

        {
            let mut stg = self.pinned_staging.lock();
            stg.positions.clear();
            stg.positions
                .extend(proc_start as u32..(proc_start + proc_count) as u32);

            let pos_bytes = std::mem::size_of_val(stg.positions.as_slice());
            let slots_bytes = proc_count * std::mem::size_of::<i64>();
            let total_bytes = if !needs_paged {
                slot_offset + slots_bytes
            } else {
                pos_bytes
            };
            let scratch_end = meta_offset
                .checked_add(total_bytes)
                .ok_or_else(|| anyhow::anyhow!("prefill_twophase metadata offset overflow"))?;
            anyhow::ensure!(
                scratch_end <= self.buffers.sizes().scratch,
                "prefill_twophase metadata ends at byte {scratch_end}, beyond the {}-byte scratch arena",
                self.buffers.sizes().scratch
            );
            // Validate pinned host buffer capacity BEFORE any copy_nonoverlapping write.
            stg.buffer.pinned_slice(total_bytes)?;

            let pinned = stg.buffer.as_mut_ptr();
            // SAFETY: `total_bytes <= stg.buffer.len()` was verified above and covers
            // `[0..pos_bytes)`; `stg.positions` holds `pos_bytes` initialized bytes and
            // does not overlap `stg.buffer`.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    stg.positions.as_ptr() as *const u8,
                    pinned,
                    pos_bytes,
                );
            }

            if !needs_paged {
                stg.slots.clear();
                stg.slots
                    .extend((proc_start..proc_start + proc_count).map(|i| {
                        let block_idx = seq
                            .physical_block_for(i / bs)
                            .unwrap_or(self.dummy_kv_block);
                        (block_idx as i64) * (bs as i64) + ((i % bs) as i64)
                    }));
                // SAFETY: `total_bytes <= stg.buffer.len()` covers
                // `[slot_offset..slot_offset + slots_bytes)`; `stg.slots` holds
                // `slots_bytes` initialized bytes and does not overlap `stg.buffer`.
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        stg.slots.as_ptr() as *const u8,
                        pinned.add(slot_offset),
                        slots_bytes,
                    );
                }
            }

            let pinned_slice = stg.buffer.pinned_slice(total_bytes)?;
            // SAFETY: `pinned_slice` points into page-locked `stg.buffer` (model
            // lifetime), `[meta_offset..scratch_end)` fits in `buffers.scratch()`,
            // and `self.gpu.synchronize(stream)` below drains DMA before reuse.
            unsafe {
                self.gpu
                    .copy_h2d_pinned_async(pinned_slice, meta_base, stream)?;
            }
        }

        if needs_paged {
            let current_blocks = seq.block_table.len();
            let upload_start = self
                .ensure_chunked_prefill_meta(seq, total_len, bs)?
                .uploaded_blocks;
            anyhow::ensure!(
                upload_start <= current_blocks,
                "two-phase prefill uploaded block count {upload_start} exceeds host block table {current_blocks}"
            );
            // Phase 6.3: when HSS sliding has occurred, the rolling window
            // can't be uploaded by absolute index without re-mapping. The
            // orchestrator path bypasses the production paged kernel that
            // reads this metadata, so skip the upload entirely in HSS mode.
            let seq_len_val = (proc_start + proc_count) as u32;
            let seq_len_bytes = seq_len_val.to_ne_bytes();
            let seq_len_base = seq.chunked_prefill_meta.as_ref().unwrap().seq_len;
            if upload_start < current_blocks && seq.hss_window_start() == 0 {
                let new_blocks = &seq.block_table[upload_start..];
                // SAFETY: `new_blocks` is a valid initialized `&[u32]` slice;
                // `u8` has alignment 1 and `size_of_val` matches its exact byte length.
                let bt_bytes = unsafe {
                    std::slice::from_raw_parts(
                        new_blocks.as_ptr() as *const u8,
                        std::mem::size_of_val(new_blocks),
                    )
                };
                let block_table_base = seq.chunked_prefill_meta.as_ref().unwrap().block_table;
                let copies = [
                    HostToDeviceCopy::new(
                        bt_bytes,
                        block_table_base.offset(upload_start * std::mem::size_of::<u32>()),
                    ),
                    HostToDeviceCopy::new(&seq_len_bytes, seq_len_base),
                ];
                self.copy_h2d_group_fast(&copies, stream)?;
                seq.chunked_prefill_meta.as_mut().unwrap().uploaded_blocks = current_blocks;
            } else {
                self.copy_h2d_group_fast(
                    &[HostToDeviceCopy::new(&seq_len_bytes, seq_len_base)],
                    stream,
                )?;
            }

            let block_table_base = seq.chunked_prefill_meta.as_ref().unwrap().block_table;
            ops::fill_slots_from_block_table(
                self.gpu.as_ref(),
                self.fill_slots_kernel,
                meta_base.offset(slot_offset),
                block_table_base,
                proc_start as u32,
                proc_count as u32,
                bs as u32,
                stream,
            )?;
        }

        // Force H2D metadata copy to complete before layer forward.
        self.gpu.synchronize(stream)?;

        let (block_table_dev, seq_len_dev) = if needs_paged {
            let page_meta = seq.chunked_prefill_meta.as_ref().unwrap();
            (page_meta.block_table, page_meta.seq_len)
        } else {
            (DevicePtr::NULL, DevicePtr::NULL)
        };

        let attn_metadata = AttnMetadataDev {
            qwen4_qsa_required: seq.qwen4_qsa_required,
            positions: meta_base,
            positions_h: meta_base,
            positions_w: meta_base,
            slot: meta_base.offset(slot_offset),
            seq_len: seq_len_dev,
            block_table: block_table_dev,
            max_blocks_per_seq: seq.block_table.len() as u32,
            num_seqs: 1,
        };

        let ctx = ForwardContext {
            buffers: &self.buffers,
            gpu: self.gpu.as_ref(),
            config: &self.config,
            attn_metadata: Some(attn_metadata),
            profile: self.profile,
            comm: self.comm_ref(),
            graph_capture: false,
            ddtree_parent_ids_dev: None,
            tree_aware_attn: None,
            ssm_multi_seq_ptr_table_override: None,
            self_spec_sparse_draft: None,
            ffn_defer: None,
        };

        // ── 4. Per-layer forward: SSM uses three-phase, attention uses standard ──
        let layer_kv_write_start = if marconi_skip {
            super::prefill_b::cached_kv_rows_in_slice(matched, proc_start, proc_count)
        } else {
            kv_write_start
        };
        let gdn_bufs = GdnPrefillBuffers {
            qkv: self.gdn_buf_qkv,
            gate_beta: self.gdn_buf_gate_beta,
            output: self.gdn_buf_out,
            z: self.gdn_buf_z,
            total_len: proc_count,
        };

        for (i, layer) in self.layers.iter().enumerate() {
            if layer.is_ssm_layer() {
                // Phase 1: chunked projections → GDN input buffers.
                for chunk_start in (0..proc_count).step_by(chunk_size) {
                    let chunk_len = chunk_size.min(proc_count - chunk_start);
                    let hidden_chunk = hidden.offset(chunk_start * h * fp32);
                    let residual_chunk = residual.offset(chunk_start * h * fp32);
                    layer.prefill_phase1(
                        hidden_chunk,
                        residual_chunk,
                        chunk_len,
                        seq.layer_states[i].as_mut(),
                        &mut kv_cache,
                        proc_start + chunk_start,
                        &mut seq.block_table,
                        &mut seq.disk_block_ids,
                        &mut seq.disk_last_offloaded_per_layer,
                        layer_kv_write_start,
                        &gdn_bufs,
                        chunk_start,
                        &ctx,
                        stream,
                    )?;
                }

                // Phase 2: single GDN launch on full sequence.
                layer.prefill_gdn_full(seq.layer_states[i].as_mut(), &gdn_bufs, &ctx, stream)?;

                // Phase 3: chunked post-processing (gated RMS norm, out proj, MoE).
                for chunk_start in (0..proc_count).step_by(chunk_size) {
                    let chunk_len = chunk_size.min(proc_count - chunk_start);
                    let hidden_chunk = hidden.offset(chunk_start * h * fp32);
                    let residual_chunk = residual.offset(chunk_start * h * fp32);
                    layer.prefill_phase3(
                        hidden_chunk,
                        residual_chunk,
                        chunk_len,
                        &gdn_bufs,
                        chunk_start,
                        &ctx,
                        stream,
                    )?;
                }
            } else {
                // Attention layer: process all tokens at once.
                layer
                    .prefill(
                        hidden,
                        residual,
                        proc_count,
                        seq.layer_states[i].as_mut(),
                        &mut kv_cache,
                        proc_start,
                        &mut seq.block_table,
                        &mut seq.disk_block_ids,
                        &mut seq.disk_last_offloaded_per_layer,
                        layer_kv_write_start,
                        &ctx,
                        stream,
                    )
                    .map_err(|e| {
                        anyhow::anyhow!("Two-phase prefill attention layer {i} failed: {e}")
                    })?;
            }
        }

        // ── 5. Update sequence state ──
        seq.tokens.extend_from_slice(tokens);
        seq.seq_len = total_len;

        crate::layers::moe::qwen4_moe_status::flush(self.gpu.as_ref(), stream)?;

        // ── 6. Final norm on LAST token only ──
        let last_hidden = hidden.offset((proc_count - 1) * h * fp32);
        let normed = self.buffers.norm_output();
        self.final_norm_rows(last_hidden, normed, 1, stream)?;

        // ── 7. LM head on last token → logits ──
        self.lm_head(normed, stream)?;

        // ── 8. Insert into prefix cache + Marconi snapshot ──
        self.prefill_save_snapshot_and_insert(tokens, seq, &mut kv_cache, bs, stream);

        Ok(self.decode_logits_ptr())
    }
}
