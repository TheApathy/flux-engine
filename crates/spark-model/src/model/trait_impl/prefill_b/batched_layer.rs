// SPDX-License-Identifier: AGPL-3.0-only

//! Q12 Path B model-level batched per-layer dispatchers.
//!
//! Two methods on `TransformerModel`:
//!   - `prefill_attn_batched_layer` — runs one attention layer over N
//!     stacked-input streams via `layer.prefill_inner_batched_q12`, using the
//!     batched paged-prefill kernel for the attention compute step.
//!   - `prefill_ssm_batched_layer` — runs one SSM layer over N streams via
//!     per-stream phase 1/3 and one batched GDN kernel call in phase 2.
//!
//! Both are called from `prefill_batch_chunk_kernel_batched`'s outer layer
//! loop after `stage_batched_attn_metadata` has built `BatchedAttnMetadata`.

#![allow(unused_imports, dead_code, clippy::too_many_arguments)]

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;
use spark_runtime::kv_cache::PagedKvCache;

use super::super::super::types::TransformerModel;
use crate::layer::{
    AttnMetadataDev, BatchedAttnMetadata, ForwardContext, GdnPrefillBuffers, LayerState,
    TransformerLayer,
};
use crate::traits::SequenceState;

impl TransformerModel {
    /// Run one attention layer over N stacked-input streams.
    ///
    /// `hidden_stacked` and `residual_stacked` are at the arena's
    /// `hidden_states()` / `residual()` pointers respectively, and
    /// contain N streams' tokens at offsets `b * chunk_len * H * dtype`.
    /// `seqs` provides per-stream `SequenceState` for KV-write routing
    /// and per-stream layer state (which is `EmptyLayerState` for
    /// attention but kept in the slice for symmetry with SSM).
    /// `meta` is the per-call `BatchedAttnMetadata` from
    /// `stage_batched_attn_metadata`.
    pub(in crate::model) fn prefill_attn_batched_layer(
        &self,
        layer: &dyn TransformerLayer,
        layer_idx: usize,
        hidden_stacked: DevicePtr,
        residual_stacked: DevicePtr,
        seqs: &mut [&mut SequenceState],
        kv_cache: &mut PagedKvCache,
        kv_write_starts: &[usize],
        seq_lens_start: usize,
        meta: &BatchedAttnMetadata,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        // Q12 Path B: dispatch to the layer's batched-attention method.
        // `Qwen3AttentionLayer::prefill_inner_batched_q12` performs the full
        // attention-layer prefill (rms_norm + residual + qkv_proj + RoPE +
        // KV-write + batched attention compute + o_proj + post-attn norm +
        // FFN + final residual) on the stacked input. The kernel-batched
        // attention step uses `block_table_ptrs[N]` from `meta`.
        //
        // The layer override bails Err for unsupported paths (MLA,
        // HDIM=512, HSS engaged, seq_len_start == 0); eligibility checks
        // screen these upfront in `kernel_batched_eligible` and any runtime
        // error propagates directly to the caller.
        debug_assert_eq!(seqs.len() as u32, meta.batch_size);
        let _ = (layer_idx, kv_write_starts);
        let num_tokens = meta.total_tokens as usize;
        // Mut borrow on seqs is unused inside this branch (the batched
        // attention call routes block-table info through meta.block_table_ptrs
        // and seq mutations are not needed for attention prefill). Drop
        // the borrow before calling.
        let _ = seqs;
        layer.prefill_inner_batched_q12(
            hidden_stacked,
            residual_stacked,
            num_tokens,
            kv_cache,
            seq_lens_start,
            meta,
            ctx,
            stream,
        )
    }

    /// Run one SSM layer over N stacked-input streams.
    ///
    /// Same args as `prefill_attn_batched_layer` plus access to the
    /// model's SSM layer state pool via `seqs[b].layer_states[layer_idx]`.
    pub(in crate::model) fn prefill_ssm_batched_layer(
        &self,
        layer: &dyn TransformerLayer,
        layer_idx: usize,
        hidden_stacked: DevicePtr,
        residual_stacked: DevicePtr,
        seqs: &mut [&mut SequenceState],
        kv_cache: &mut PagedKvCache,
        seqs_proc_start: &[usize],
        meta: &BatchedAttnMetadata,
        gdn_bufs: &GdnPrefillBuffers,
        h_state_ptrs_scratch_offset: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let n = seqs.len();
        debug_assert_eq!(n as u32, meta.batch_size);
        debug_assert_eq!(n, seqs_proc_start.len());
        let chunk_len = meta.chunk_len as usize;
        let h = ctx.config.hidden_size;
        let dtype_bytes = if ctx.config.use_fp32_residual() {
            4usize
        } else {
            2usize
        };

        // ── Phase 1: per-stream projections + conv1d + L2 norm ──
        // Each stream's data lands in gdn_bufs at offset `b * chunk_len`.
        // The `_kv_write_start` arg is ignored by SSM layers (recurrent
        // state requires all tokens, no skip), so we pass 0.
        for (b, seq) in seqs.iter_mut().enumerate() {
            let h_b = hidden_stacked.offset(b * chunk_len * h * dtype_bytes);
            let r_b = residual_stacked.offset(b * chunk_len * h * dtype_bytes);
            // Split borrow so we can pass multiple &mut fields of the same seq.
            let (block_table, disk_block_ids, disk_last, layer_state) = {
                let SequenceState {
                    block_table,
                    disk_block_ids,
                    disk_last_offloaded_per_layer,
                    layer_states,
                    ..
                } = &mut **seq;
                (
                    block_table,
                    disk_block_ids,
                    disk_last_offloaded_per_layer,
                    layer_states[layer_idx].as_mut(),
                )
            };
            layer.prefill_phase1(
                h_b,
                r_b,
                chunk_len,
                layer_state,
                kv_cache,
                seqs_proc_start[b],
                block_table,
                disk_block_ids,
                disk_last,
                0,
                gdn_bufs,
                b * chunk_len,
                ctx,
                stream,
            )?;
        }

        // ── Phase 2: ONE batched GDN kernel call ──
        // Stage h_state_ptrs[N] device array at the dedicated scratch offset
        // (caller computes this offset to avoid colliding with the
        // BatchedAttnMetadata staging at scratch[0..]).
        let h_state_ptrs_dev =
            self.stage_h_state_ptrs(layer_idx, seqs, h_state_ptrs_scratch_offset, stream)?;

        layer.prefill_gdn_full_batched(
            h_state_ptrs_dev,
            gdn_bufs,
            meta.batch_size,
            meta.chunk_len,
            ctx,
            stream,
        )?;

        // ── Phase 3: per-stream gated-RMS-norm + out-proj + MoE ──
        for (b, seq) in seqs.iter_mut().enumerate() {
            let h_b = hidden_stacked.offset(b * chunk_len * h * dtype_bytes);
            let r_b = residual_stacked.offset(b * chunk_len * h * dtype_bytes);
            layer.prefill_phase3(h_b, r_b, chunk_len, gdn_bufs, b * chunk_len, ctx, stream)?;
            // Suppress unused-binding warnings — seq could be queried for
            // future per-stream metadata in phase3 but currently isn't.
            let _ = seq;
        }

        // meta is consumed for chunk_len/batch_size above.
        let _ = meta;
        Ok(())
    }
}
