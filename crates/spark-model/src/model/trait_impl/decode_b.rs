// SPDX-License-Identifier: AGPL-3.0-only
//! Decode phase B — fused mixed prefill + multi-sequence decode (`mixed_forward_dispatch`).
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

use super::mixed_scratch::MixedPrefillScratchLayout;

const MIXED_DECODE_META_OFFSET: usize = 64 * 1024;

fn validate_mixed_forward_arena_bounds(
    layout: spark_runtime::buffers::DecodeMetaLayout,
    max_blocks_per_seq: usize,
    logits_arena_bytes: usize,
    padded_n: usize,
    vocab_size: usize,
    logits_elem_bytes: usize,
    prefill_is_last: bool,
) -> Result<()> {
    anyhow::ensure!(
        padded_n <= layout.rows(),
        "mixed_forward padded decode width {padded_n} exceeds the {}-row decode metadata layout",
        layout.rows()
    );

    let metadata_bytes = layout
        .rows()
        .checked_mul(max_blocks_per_seq)
        .and_then(|entries| entries.checked_mul(std::mem::size_of::<u32>()))
        .and_then(|table_bytes| layout.block_table_off().checked_add(table_bytes))
        .ok_or_else(|| anyhow::anyhow!("mixed_forward decode metadata extent overflow"))?;
    let metadata_end = MIXED_DECODE_META_OFFSET
        .checked_add(metadata_bytes)
        .ok_or_else(|| anyhow::anyhow!("mixed_forward decode metadata extent overflow"))?;
    anyhow::ensure!(
        metadata_end <= logits_arena_bytes,
        "mixed_forward decode metadata ends at byte {metadata_end}, beyond the {logits_arena_bytes}-byte logits arena"
    );

    let logits_rows = padded_n
        .checked_add(if prefill_is_last { 1 } else { 0 })
        .ok_or_else(|| anyhow::anyhow!("mixed_forward logits row count overflow"))?;
    let logits_bytes = logits_rows
        .checked_mul(vocab_size)
        .and_then(|elements| elements.checked_mul(logits_elem_bytes))
        .ok_or_else(|| anyhow::anyhow!("mixed_forward logits extent overflow"))?;
    anyhow::ensure!(
        logits_bytes <= logits_arena_bytes,
        "mixed_forward needs {logits_bytes} bytes for {logits_rows} logits rows, beyond the {logits_arena_bytes}-byte logits arena"
    );
    Ok(())
}

/// RAII guard that returns taken per-sequence `layer_states` on every exit path
/// (`Ok`, `Err`, or panic unwind), dropping any appended dummy pad states.
struct MixedDecodeLayerStatesGuard<'a, 'b> {
    seqs: &'a mut [&'b mut SequenceState],
    states: Vec<Vec<Box<dyn LayerState>>>,
    n: usize,
}

impl<'a, 'b> MixedDecodeLayerStatesGuard<'a, 'b> {
    fn take(seqs: &'a mut [&'b mut SequenceState]) -> Self {
        let n = seqs.len();
        let states = seqs
            .iter_mut()
            .map(|s| std::mem::take(&mut s.layer_states))
            .collect();
        Self { seqs, states, n }
    }
}

impl Drop for MixedDecodeLayerStatesGuard<'_, '_> {
    fn drop(&mut self) {
        let restore_n = self.n.min(self.states.len());
        for (seq, ls) in self.seqs.iter_mut().zip(self.states.drain(..restore_n)) {
            seq.layer_states = ls;
        }
    }
}

impl TransformerModel {
    pub(super) fn mixed_forward_dispatch(
        &self,
        decode_tokens: &[u32],
        decode_seqs: &mut [&mut SequenceState],
        prefill_tokens: &[u32],
        prefill_seq: &mut SequenceState,
        prefill_chunk_start: usize,
        prefill_chunk_len: usize,
        prefill_is_last: bool,
        stream: u64,
    ) -> Result<crate::traits::MixedForwardResult> {
        anyhow::ensure!(
            !self.vision_prompt_present(prefill_tokens)
                && prefill_seq.rotary_positions.is_identity()
                && decode_seqs
                    .iter()
                    .all(|seq| seq.rotary_positions.is_identity()),
            "image sequences require C1 ordinary/chunked prefill, not mixed forward"
        );
        anyhow::ensure!(
            prefill_chunk_len > 0,
            "mixed_forward requires non-empty prefill chunk (got prefill_chunk_len=0)"
        );
        prefill_chunk_start
            .checked_add(prefill_chunk_len)
            .filter(|&end| end <= prefill_tokens.len())
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "mixed prefill chunk [{prefill_chunk_start}, +{prefill_chunk_len}) exceeds prompt length {}",
                    prefill_tokens.len()
                )
            })?;
        let multi_chunk = crate::layers::qwen4_prefill_moe::multi_chunk_selected()?;
        let _multi_chunk_suppressed = if multi_chunk {
            // Mixed decode mutates unrelated sequence state before its prefill
            // half completes. Permanently route this selected prompt ordinary
            // before either half performs work; a later singular call may not
            // resurrect the fast family.
            crate::layers::qwen4_prefill_moe::attn16::mark_ordinary_for_selected_route(
                &mut prefill_seq.qwen4_prefill_lineage,
                prefill_tokens.len(),
            );
            Some(crate::layers::qwen4_prefill_moe::attn16::suppress_family())
        } else {
            None
        };
        let n_decode = decode_tokens.len();
        let n_prefill = prefill_chunk_len;

        // Padded decode count for batched decode kernel compatibility.
        let padded_n_guard = crate::traits::padded_batch_n(n_decode);

        // Guard: fall back to default (sequential) for EP, oversized, no decode,
        // or MLA. MLA models route the decode portion through `decode_batch`,
        // whose `decode_batch_dispatch` dispatches the batched MLA branch
        // (`ms_mla_decode`, issue #84). The fused `decode_multi_seq` body
        // inlined below is NOT used for MLA here — it shares a single layer
        // loop with the prefill chunk and that interleaving has not been
        // validated for the absorbed-MLA path — so MLA stays on the
        // dedicated `decode_batch` route.
        // Use padded_n (not n_decode) because padding slots consume hidden buffer space.
        // Qwen4 / mHC models use expanded residual width (`residual_width() != hidden_size`),
        // which this fixed-stride fused loop does not support — fall back sequentially.
        if self.comm.is_some()
            || self.is_mla_dispatch()
            || self.config.is_qwen4_exp()
            || self.config.hc_mult > 0
            || (padded_n_guard + n_prefill) > self.buffers.max_batch_tokens()
            || n_decode == 0
            || multi_chunk
        {
            let decode_logits = if !decode_tokens.is_empty() {
                self.decode_batch(decode_tokens, decode_seqs, stream)?
            } else {
                DevicePtr::NULL
            };
            let prefill_logits = self.prefill_chunk(
                prefill_tokens,
                prefill_seq,
                prefill_chunk_start,
                prefill_chunk_len,
                prefill_is_last,
                stream,
            )?;
            return Ok(crate::traits::MixedForwardResult {
                decode_logits,
                prefill_logits,
            });
        }

        // PREFIX_CACHE_POLICY: BYPASS.
        // The fused active+prefill route cannot safely splice paired KV+SSM
        // replay into its fixed stacked-buffer layout: a restored checkpoint
        // changes the prefill processing range inside the shared layer loop.
        // It therefore performs no prefix lookup or insertion. Every supplied
        // prefill row, including every recurrent row, is computed below.
        // ── Fused mixed forward: single layer loop, weights loaded once per layer ──
        //
        // Layout in hidden/residual buffers (contiguous):
        //   [0 .. N*H*fp32)           = decode tokens (N sequences × 1 token each)
        //   [N*H*fp32 .. (N+M)*H*fp32) = prefill chunk tokens (1 sequence × M tokens)
        //
        // Per layer: decode_multi_seq on [0..N), then prefill on [N..N+M).
        // Both use non-overlapping hidden/residual regions. Intermediate scratch
        // buffers (norm_output, qkv_output, etc.) are overwritten by each sub-call,
        // safe because same CUDA stream guarantees sequential execution.

        let stream = self.gpu.default_stream();
        let h = self.config.hidden_size;
        let bf16 = 2usize;
        let fp32 = if self.config.use_fp32_residual() {
            4usize
        } else {
            2usize
        };
        let hidden = self.buffers.hidden_states();
        let residual = self.buffers.residual();

        // Pad decode count to the shared C128 ladder.
        let padded_n = crate::traits::padded_batch_n(n_decode);
        validate_mixed_forward_arena_bounds(
            self.buffers.decode_meta(),
            self.max_blocks_per_seq as usize,
            self.buffers.sizes().logits,
            padded_n,
            self.config.vocab_size,
            bf16,
            prefill_is_last,
        )?;
        let prefill_scratch = MixedPrefillScratchLayout::new(
            n_prefill,
            padded_n,
            self.config.num_experts_per_tok,
            self.buffers.sizes().scratch,
        )?;
        let prefill_upload_bytes = prefill_scratch.upload_bytes(prefill_chunk_start > 0);
        let staging_bytes = self.pinned_staging.lock().buffer.len();
        anyhow::ensure!(
            prefill_upload_bytes <= staging_bytes,
            "mixed_forward prefill metadata needs {prefill_upload_bytes} staging bytes, beyond the {staging_bytes}-byte pinned buffer"
        );

        // ── 1. Embed all tokens contiguously ──

        // 1a. Decode tokens → hidden[0..n_decode*H)
        for (i, &tok) in decode_tokens.iter().enumerate() {
            self.embed(tok, hidden.offset(i * h * fp32), stream)?;
        }
        // 1b. Zero padding for decode [n_decode..padded_n)
        for i in n_decode..padded_n {
            self.gpu.memset(hidden.offset(i * h * fp32), 0, h * fp32)?;
        }
        // 1c. Prefill chunk tokens → hidden[padded_n*H..(padded_n+M)*H)
        //     Use batched embed for efficiency (single kernel launch for M tokens)
        let prefill_hidden = hidden.offset(padded_n * h * fp32);
        let prefill_residual = residual.offset(padded_n * h * fp32);
        {
            let chunk_tokens =
                &prefill_tokens[prefill_chunk_start..prefill_chunk_start + n_prefill];
            if self.batched_embed_kernel.0 != 0 {
                // SAFETY: `chunk_tokens` is a valid initialized `&[u32]` slice;
                // `u8` has alignment 1 and `size_of_val` matches its exact
                // initialized byte length for `copy_h2d_group_fast`.
                let token_ids_bytes: &[u8] = unsafe {
                    std::slice::from_raw_parts(
                        chunk_tokens.as_ptr() as *const u8,
                        std::mem::size_of_val(chunk_tokens),
                    )
                };
                // Use norm_output as temporary staging for token IDs (overwritten by first layer)
                let token_ids_dev = self.buffers.norm_output();
                self.copy_h2d_group_fast(
                    &[HostToDeviceCopy::new(token_ids_bytes, token_ids_dev)],
                    stream,
                )?;
                ops::batched_embed(
                    self.gpu.as_ref(),
                    self.batched_embed_kernel,
                    token_ids_dev,
                    self.embed_tokens.weight,
                    prefill_hidden,
                    n_prefill as u32,
                    h as u32,
                    stream,
                )?;
                self.scale_embeddings(prefill_hidden, n_prefill, stream)?;
            } else {
                for (i, &tok) in chunk_tokens.iter().enumerate() {
                    self.embed(tok, prefill_hidden.offset(i * h * fp32), stream)?;
                }
            }
        }

        // ── 2. Lock KV cache once for both decode and prefill ──
        let mut kv_cache = self.kv_cache.lock();
        let bs = kv_cache.block_size();

        // 2a. Allocate KV blocks for decode sequences
        for seq in decode_seqs.iter_mut() {
            let blocks_needed = (seq.seq_len / bs) + 1;
            ensure_blocks_through_decode(
                seq,
                blocks_needed - 1,
                &mut kv_cache,
                self.prefix_cache.as_ref(),
                self.gpu.as_ref(),
                stream,
            )?;
        }

        // 2b. Allocate KV blocks for prefill sequence (`n_prefill > 0` checked above).
        let prefill_end_pos = prefill_chunk_start + n_prefill;
        let prefill_blocks_needed = prefill_end_pos.div_ceil(bs);
        ensure_blocks_through_prefill(
            prefill_seq,
            prefill_blocks_needed - 1,
            &mut kv_cache,
            self.prefix_cache.as_ref(),
            self.gpu.as_ref(),
            stream,
        )?;

        // ── 3. Upload decode metadata ──
        //
        // Place decode metadata in the logits buffer (not used until step 7).
        // This avoids conflicts with prefill MoE routing scratch at scratch[0..].
        // Its derived layout is sized for the admitted decode width and is
        // bounds-checked above against this logits arena. The logits overwrite
        // the metadata only in step 7, after the layer loop has consumed it.
        //
        // BUG FIX 2026-05-10: offset by 64KB to avoid being overwritten by MoE
        // forward's `shared_gate_scratch` which also uses `logits` as scratch
        // (moe/forward.rs:211, forward_batched.rs:61, forward_k2.rs:91, etc.).
        // Without this offset, the first MoE call during the layer loop
        // overwrites decode_meta's positions/slots/seq_lens/block_table at
        // logits[0..16K], causing subsequent attention kernels to read
        // corrupted block_table → CUDA-700 illegal memory access. Reproducer:
        // Qwen3-Next-80B + 2 streams + chunked prefill, when one finishes
        // first and `mixed_forward` runs decode+prefill fused. 64KB offset
        // leaves room for the largest known shared-expert scratch
        // (shared_expert_intermediate_size × 2 ≤ 32KB observed for any
        // current Atlas model — 64KB is 2× safety margin).
        let decode_meta_base = self.buffers.logits().offset(MIXED_DECODE_META_OFFSET);

        let decode_metadata = self.upload_batch_metadata_at(
            decode_seqs,
            padded_n,
            &mut kv_cache,
            decode_meta_base,
            stream,
        )?;

        // ── 4. Upload prefill metadata ──
        //
        // Prefill metadata at scratch[moe_scratch..], same layout as prefill_chunk.
        let proc_start = prefill_chunk_start;
        let proc_count = n_prefill;
        let effective_seq_len_start = prefill_chunk_start;
        let prefill_meta_base = self.buffers.scratch().offset(prefill_scratch.meta_offset);
        let slot_offset = prefill_scratch.slot_offset;
        let needs_paged = effective_seq_len_start > 0;

        {
            let mut stg = self.pinned_staging.lock();
            stg.positions.clear();
            stg.positions
                .extend(proc_start as u32..(proc_start + proc_count) as u32);

            let pinned = stg.buffer.as_mut_ptr();
            // SAFETY: `prefill_upload_bytes <= stg.buffer.len()` was verified
            // above and covers `positions_bytes`; `stg.positions` holds
            // `proc_count` initialized `u32` elements (`positions_bytes` bytes),
            // and source/destination buffers are non-overlapping.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    stg.positions.as_ptr() as *const u8,
                    pinned,
                    prefill_scratch.positions_bytes,
                );
            }

            if !needs_paged {
                stg.slots.clear();
                stg.slots
                    .extend((proc_start..proc_start + proc_count).map(|i| {
                        let block_idx = prefill_seq
                            .physical_block_for(i / bs)
                            .unwrap_or(self.dummy_kv_block);
                        (block_idx as i64) * (bs as i64) + ((i % bs) as i64)
                    }));
                // SAFETY: `prefill_upload_bytes <= stg.buffer.len()` covers
                // `slot_offset + slots_bytes`; `stg.slots` holds `proc_count`
                // initialized `i64` elements (`slots_bytes` bytes), and
                // source/destination buffers are non-overlapping.
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        stg.slots.as_ptr() as *const u8,
                        pinned.add(slot_offset),
                        prefill_scratch.slots_bytes,
                    );
                }
            }

            let pinned_slice = stg.buffer.pinned_slice(prefill_upload_bytes)?;
            // SAFETY: `pinned_slice` points into page-locked `stg.buffer` (model
            // lifetime), `prefill_meta_base` fits within `buffers.scratch()` per
            // `MixedPrefillScratchLayout::new`, and `self.gpu.synchronize(stream)`
            // below drains DMA before any reuse of `stg.buffer`.
            unsafe {
                self.gpu
                    .copy_h2d_pinned_async(pinned_slice, prefill_meta_base, stream)?;
            }
        }

        if needs_paged {
            let current_blocks = prefill_seq.block_table.len();
            let upload_start = self
                .ensure_chunked_prefill_meta(prefill_seq, prefill_tokens.len(), bs)?
                .uploaded_blocks;
            anyhow::ensure!(
                upload_start <= current_blocks,
                "mixed prefill uploaded block count {upload_start} exceeds host block table {current_blocks}"
            );
            let seq_len_val = (proc_start + proc_count) as u32;
            let seq_len_bytes = seq_len_val.to_le_bytes();
            let seq_len_base = prefill_seq.chunked_prefill_meta.as_ref().unwrap().seq_len;
            // Phase 6.3: skip upload in HSS mode (orchestrator bypasses kernel).
            if upload_start < current_blocks && prefill_seq.hss_window_start() == 0 {
                let new_blocks = &prefill_seq.block_table[upload_start..];
                // SAFETY: `new_blocks` is a valid initialized `&[u32]` slice;
                // `u8` has alignment 1 and `size_of_val` matches its exact
                // initialized byte length for `copy_h2d_group_fast`.
                let bt_bytes = unsafe {
                    std::slice::from_raw_parts(
                        new_blocks.as_ptr() as *const u8,
                        std::mem::size_of_val(new_blocks),
                    )
                };
                let block_table_base = prefill_seq
                    .chunked_prefill_meta
                    .as_ref()
                    .unwrap()
                    .block_table;
                let copies = [
                    HostToDeviceCopy::new(
                        bt_bytes,
                        block_table_base.offset(upload_start * std::mem::size_of::<u32>()),
                    ),
                    HostToDeviceCopy::new(&seq_len_bytes, seq_len_base),
                ];
                self.copy_h2d_group_fast(&copies, stream)?;
                prefill_seq
                    .chunked_prefill_meta
                    .as_mut()
                    .unwrap()
                    .uploaded_blocks = current_blocks;
            } else {
                self.copy_h2d_group_fast(
                    &[HostToDeviceCopy::new(&seq_len_bytes, seq_len_base)],
                    stream,
                )?;
            }

            let block_table_base = prefill_seq
                .chunked_prefill_meta
                .as_ref()
                .unwrap()
                .block_table;
            ops::fill_slots_from_block_table(
                self.gpu.as_ref(),
                self.fill_slots_kernel,
                prefill_meta_base.offset(slot_offset),
                block_table_base,
                proc_start as u32,
                proc_count as u32,
                bs as u32,
                stream,
            )?;
        }

        // Force H2D metadata copies to complete before layer forward.
        self.gpu.synchronize(stream)?;

        let (prefill_bt_dev, prefill_sl_dev) = if needs_paged {
            let page_meta = prefill_seq.chunked_prefill_meta.as_ref().unwrap();
            (page_meta.block_table, page_meta.seq_len)
        } else {
            (DevicePtr::NULL, DevicePtr::NULL)
        };

        let prefill_metadata = AttnMetadataDev {
            qwen4_qsa_required: prefill_seq.qwen4_qsa_required,
            positions: prefill_meta_base,
            positions_h: prefill_meta_base,
            positions_w: prefill_meta_base,
            slot: prefill_meta_base.offset(slot_offset),
            seq_len: prefill_sl_dev,
            block_table: prefill_bt_dev,
            max_blocks_per_seq: prefill_seq.block_table.len() as u32,
            num_seqs: 1,
        };

        // ── 5. Build decode layer states ──
        let seq_lens: Vec<usize> = (0..padded_n)
            .map(|i| {
                if i < n_decode {
                    decode_seqs[i].seq_len
                } else {
                    0
                }
            })
            .collect();
        let block_tables: Vec<Vec<u32>> = (0..padded_n)
            .map(|i| {
                if i < n_decode {
                    decode_seqs[i].block_table.clone()
                } else {
                    vec![self.dummy_kv_block]
                }
            })
            .collect();

        // Guard taken `layer_states` so any error or unwind during dummy-pad
        // allocation or the fused layer loop restores every decode sequence.
        {
            let mut layer_states_guard = MixedDecodeLayerStatesGuard::take(decode_seqs);

            // Build dummy layer_states for padding positions. Use the
            // dedicated `dummy_slot()` (see SsmStatePool) so pad SSM kernel
            // writes can never collide with another claimed sequence.
            let dummy_ssm_slot = self.ssm_pool.dummy_slot();
            for _pad_pos in n_decode..padded_n {
                let mut dummy: Vec<Box<dyn LayerState>> = Vec::with_capacity(self.layers.len());
                let mut ssm_idx = 0usize;
                for (li, layer) in self.layers.iter().enumerate() {
                    if self.config.layer_type(li) == LayerType::LinearAttention {
                        dummy.push(Box::new(SsmLayerState {
                            h_state: self.ssm_pool.h_state(ssm_idx, dummy_ssm_slot),
                            conv_state: self.ssm_pool.conv_state(ssm_idx, dummy_ssm_slot),
                            h_state_checkpoint: None,
                            conv_state_checkpoint: None,
                            h_state_intermediates: Vec::new(),
                            conv_state_intermediates: Vec::new(),
                            wy17_kv_retain: None,
                            wy17_gate_retain: None,
                            h_is_f16: false,
                        }));
                        ssm_idx += 1;
                    } else {
                        dummy.push(layer.alloc_state(self.gpu.as_ref())?);
                    }
                }
                layer_states_guard.states.push(dummy);
            }

            // ── 6. Fused layer loop ──
            //
            // For each layer: process decode portion then prefill portion.
            // Weights are loaded once by the first sub-call and remain in L2
            // cache for the second sub-call. This halves memory bandwidth vs
            // the sequential decode_batch + prefill_chunk approach.
            let decode_ctx = ForwardContext {
                buffers: &self.buffers,
                gpu: self.gpu.as_ref(),
                config: &self.config,
                attn_metadata: Some(decode_metadata),
                profile: false,
                comm: self.comm_ref(),
                graph_capture: false,
                ddtree_parent_ids_dev: None,
                tree_aware_attn: None,
                ssm_multi_seq_ptr_table_override: None,
                self_spec_sparse_draft: None,
                ffn_defer: None,
            };

            let prefill_ctx = ForwardContext {
                buffers: &self.buffers,
                gpu: self.gpu.as_ref(),
                config: &self.config,
                attn_metadata: Some(prefill_metadata),
                profile: false,
                comm: self.comm_ref(),
                graph_capture: false,
                ddtree_parent_ids_dev: None,
                tree_aware_attn: None,
                ssm_multi_seq_ptr_table_override: None,
                self_spec_sparse_draft: None,
                ffn_defer: None,
            };

            for (layer_idx, layer) in self.layers.iter().enumerate() {
                // 6a. Decode: N sequences × 1 token each on hidden[0..padded_n*H)
                let mut layer_state_refs =
                    extract_layer_refs(&mut layer_states_guard.states, layer_idx);
                layer.decode_multi_seq(
                    hidden,
                    residual,
                    padded_n,
                    &mut layer_state_refs,
                    &mut kv_cache,
                    &seq_lens,
                    &block_tables,
                    &decode_ctx,
                    stream,
                )?;

                // 6b. Prefill: 1 sequence × M tokens on hidden[padded_n*H..]
                layer.prefill(
                    prefill_hidden,
                    prefill_residual,
                    proc_count,
                    prefill_seq.layer_states[layer_idx].as_mut(),
                    &mut kv_cache,
                    effective_seq_len_start,
                    &mut prefill_seq.block_table,
                    &mut prefill_seq.disk_block_ids,
                    &mut prefill_seq.disk_last_offloaded_per_layer,
                    0, // kv_write_start: cache bypass writes all current rows
                    &prefill_ctx,
                    stream,
                )?;
            }
        } // Drop restores decode layer_states on Ok, Err, and unwind.

        // ── 7. Final norm + LM head ──
        let head_out = self.mixed_final_norm_lm_head(
            hidden,
            prefill_hidden,
            padded_n,
            proc_count,
            prefill_is_last,
            h,
            bf16,
            fp32,
            stream,
        )?;
        let decode_logits = head_out.decode_logits;
        let prefill_logits = head_out.prefill_logits;

        // ── 8. Update sequence states (after all computation) ──
        for (i, seq) in decode_seqs.iter_mut().enumerate() {
            seq.tokens.push(decode_tokens[i]);
            seq.seq_len += 1;
        }
        prefill_seq.tokens.extend_from_slice(
            &prefill_tokens[prefill_chunk_start..prefill_chunk_start + n_prefill],
        );
        prefill_seq.seq_len = prefill_chunk_start + n_prefill;

        Ok(crate::traits::MixedForwardResult {
            decode_logits,
            prefill_logits,
        })
    }
}

#[cfg(test)]
mod mixed_forward_bounds_tests {
    use super::validate_mixed_forward_arena_bounds;
    use spark_runtime::buffers::DecodeMetaLayout;

    const BLOCKS: usize = 8_193;
    const VOCAB: usize = 1_000;
    const BF16: usize = 2;

    #[test]
    fn c128_metadata_and_prefill_logits_fit_when_both_extents_are_provisioned() {
        let layout = DecodeMetaLayout::for_max_batch_size(128);
        let arena = (64 * 1024 + layout.meta_bytes(BLOCKS)).max(129 * VOCAB * BF16);
        validate_mixed_forward_arena_bounds(layout, BLOCKS, arena, 128, VOCAB, BF16, true).unwrap();
    }

    #[test]
    fn rejects_width_metadata_and_extra_logits_row_overruns() {
        let layout = DecodeMetaLayout::for_max_batch_size(128);
        let full_arena = (64 * 1024 + layout.meta_bytes(BLOCKS)).max(129 * VOCAB * BF16);
        assert!(
            validate_mixed_forward_arena_bounds(
                layout, BLOCKS, full_arena, 129, VOCAB, BF16, false,
            )
            .is_err()
        );
        assert!(
            validate_mixed_forward_arena_bounds(
                layout,
                BLOCKS,
                64 * 1024 + layout.meta_bytes(BLOCKS) - 1,
                128,
                VOCAB,
                BF16,
                false,
            )
            .is_err()
        );
        assert!(
            validate_mixed_forward_arena_bounds(
                layout,
                1,
                128 * VOCAB * BF16,
                128,
                VOCAB,
                BF16,
                true,
            )
            .is_err()
        );
    }
}
