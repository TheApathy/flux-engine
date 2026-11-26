// SPDX-License-Identifier: AGPL-3.0-only

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kv_cache::PagedKvCache;

use super::Qwen3AttentionLayer;
use crate::layer::{
    BatchedAttnMetadata, EmptyLayerState, ForwardContext, LayerState, TransformerLayer,
};
use crate::layers::FfnComponent;

mod cross_seq_verify;
mod decode_inner;
mod multi_seq;
mod prefill_attn_prep_fused;
mod prefill_fast_attn;
mod prefill_inner;
mod prefill_moe_admit;
mod prefill_moe_attn16;
mod prefill_moe_attn16_device;
mod prefill_moe_only;
mod prefill_qsa_sparse;
mod prefill_suppressed;

fn ddtree_indirection_supported(
    dtype: spark_runtime::kv_cache::KvCacheDtype,
    qwen_bf16_tree_kernel_loaded: bool,
) -> bool {
    match dtype {
        spark_runtime::kv_cache::KvCacheDtype::Fp8
        | spark_runtime::kv_cache::KvCacheDtype::Nvfp4 => true,
        spark_runtime::kv_cache::KvCacheDtype::Bf16 => qwen_bf16_tree_kernel_loaded,
        _ => false,
    }
}

/// Debug: read back BF16 GPU tensor and compute L2 norm + first 4 values.
pub(super) fn diag_norm(
    gpu: &dyn GpuBackend,
    ptr: DevicePtr,
    n_elements: usize,
    stream: u64,
    label: &str,
) {
    if gpu.synchronize(stream).is_err() {
        return;
    }
    let mut buf = vec![0u16; n_elements];
    // SAFETY: `u16` has no padding, any byte pattern is valid, `buf.len() == n_elements`,
    // and the mutable byte view does not outlive `buf`.
    let bytes =
        unsafe { std::slice::from_raw_parts_mut(buf.as_mut_ptr().cast::<u8>(), n_elements * 2) };
    if gpu.copy_d2h(ptr, bytes).is_err() {
        return;
    }
    let vals: Vec<f32> = buf
        .iter()
        .map(|&b| f32::from_bits((b as u32) << 16))
        .collect();
    let norm: f32 = vals.iter().map(|v| v * v).sum::<f32>().sqrt();
    let max_abs: f32 = vals.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    let f4 = if vals.len() >= 4 {
        format!(
            "[{:.4},{:.4},{:.4},{:.4}]",
            vals[0], vals[1], vals[2], vals[3]
        )
    } else {
        format!("{:?}", &vals[..vals.len().min(4)])
    };
    tracing::info!("DIAG {label}: norm={norm:.4} max={max_abs:.4} first4={f4} n={n_elements}");
}

/// Debug: read back FP32 GPU tensor and compute L2 norm + first 4 values.
pub fn diag_norm_f32(
    gpu: &dyn GpuBackend,
    ptr: DevicePtr,
    n_elements: usize,
    stream: u64,
    label: &str,
) {
    if gpu.synchronize(stream).is_err() {
        return;
    }
    let mut buf = vec![0f32; n_elements];
    // SAFETY: `f32` has no padding, any byte pattern is valid, `buf.len() == n_elements`,
    // and the mutable byte view does not outlive `buf`.
    let bytes =
        unsafe { std::slice::from_raw_parts_mut(buf.as_mut_ptr().cast::<u8>(), n_elements * 4) };
    if gpu.copy_d2h(ptr, bytes).is_err() {
        return;
    }
    let norm: f32 = buf.iter().map(|v| v * v).sum::<f32>().sqrt();
    let max_abs: f32 = buf.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    let f4 = if buf.len() >= 4 {
        format!("[{:.4},{:.4},{:.4},{:.4}]", buf[0], buf[1], buf[2], buf[3])
    } else {
        format!("{:?}", &buf[..buf.len().min(4)])
    };
    tracing::info!(
        "DIAG {label}: norm={norm:.4} max={max_abs:.4} first4={f4} n={n_elements} (FP32)"
    );
}

/// Gemma-4 diagnostic gate. Set ATLAS_DIAG_GEMMA4=1 to enable per-layer
/// hidden-state norm dumps in the decode path. Heavy (one d2h copy per
pub(super) fn gemma4_diag_enabled() -> bool {
    static CACHED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| {
        matches!(
            std::env::var("ATLAS_DIAG_GEMMA4").ok().as_deref(),
            Some("1") | Some("true")
        )
    })
}

impl TransformerLayer for Qwen3AttentionLayer {
    fn qwen4_attn_hyper(&self) -> Option<&crate::layers::qwen4_hyper::Qwen4HyperConnection> {
        self.qwen4_attn_hyper.as_ref()
    }

    fn qwen4_prefill_finish_moe(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        rows: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let mlp = self.qwen4_mlp_hyper.as_ref().ok_or_else(|| {
            anyhow::anyhow!("expert-major prefill needs the Qwen4 MLP hyperconnection")
        })?;
        crate::layers::qwen4_prefill_moe::finish(
            mlp, &self.ffn, hidden, residual, rows, ctx, stream,
        )?;
        crate::model::qwen4_prefill_engagement::engage(
            crate::model::qwen4_prefill_engagement::PrefillPath::Attention,
            rows,
        )
    }
    fn preflight_qwen4_attn16(
        &self,
        kv_cache: &PagedKvCache,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        self.preflight_moe_attn16(kv_cache, gpu, stream)
    }

    fn prepare_qwen4_qsa_verify(
        &self,
        kv_cache: &PagedKvCache,
        gpu: &dyn GpuBackend,
    ) -> Result<()> {
        let qsa = self
            .qwen4_qsa
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Qwen4 row-serial QSA verify requires an indexer"))?;
        qsa.prepare_verify_k5_transaction(kv_cache, gpu)
    }

    fn qwen4_qsa_seqstate_block_regions(
        &self,
        gpu: &dyn GpuBackend,
        kv_cache: &PagedKvCache,
        block: u32,
        create: bool,
    ) -> Result<Option<[(DevicePtr, usize); 2]>> {
        match self.qwen4_qsa.as_ref() {
            Some(qsa) => qsa.seqstate_block_regions(gpu, kv_cache, block, create),
            None => Ok(None),
        }
    }

    fn qwen4_batch_verify_lane_ranges(
        &self,
        slot_idx: usize,
        gpu: &dyn GpuBackend,
        kv_cache: &PagedKvCache,
        config: &atlas_core::config::ModelConfig,
        writable_blocks: &[u32],
    ) -> Result<Vec<(DevicePtr, usize, DevicePtr, usize)>> {
        anyhow::ensure!(
            config.is_qwen4_exp(),
            "batch journal export requires the Qwen4 model"
        );
        let qsa = self
            .qwen4_qsa
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("batch verify attention has no physical QSA journal"))?;
        qsa.batch_verify_lane_ranges(slot_idx, gpu, kv_cache, writable_blocks)
    }

    fn snapshot_qwen4_qsa_verify_row(
        &self,
        kv_cache: &PagedKvCache,
        slot: DevicePtr,
        row: usize,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        let qsa = self
            .qwen4_qsa
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Qwen4 row-serial QSA verify requires an indexer"))?;
        qsa.snapshot_verify_k5_row(kv_cache, slot, row, gpu, stream)
    }

    fn commit_qwen4_qsa_k5(
        &self,
        accepted: usize,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        if let Some(qsa) = self.qwen4_qsa.as_ref() {
            qsa.commit_verify_k5_prefix(accepted, gpu, stream)?;
        }
        Ok(())
    }

    fn commit_qwen4_qsa_rows(
        &self,
        accepted: usize,
        rows: usize,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        if let Some(qsa) = self.qwen4_qsa.as_ref() {
            qsa.commit_verify_prefix(accepted, rows, gpu, stream)?;
        }
        Ok(())
    }

    fn has_qsa_side_cache(&self) -> bool {
        self.qwen4_qsa.is_some()
    }

    fn read_qsa_swap_block(
        &self,
        block_idx: u32,
        kv_cache: &PagedKvCache,
        gpu: &dyn GpuBackend,
    ) -> Result<(Vec<u8>, Vec<u8>)> {
        let qsa = self
            .qwen4_qsa
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("attention layer has no QSA indexer"))?;
        qsa.read_swap_block(block_idx, kv_cache, gpu)
    }

    fn write_qsa_swap_block(
        &self,
        block_idx: u32,
        raw_data: &[u8],
        comp_data: &[u8],
        kv_cache: &PagedKvCache,
        gpu: &dyn GpuBackend,
    ) -> Result<()> {
        let qsa = self
            .qwen4_qsa
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("attention layer has no QSA indexer"))?;
        qsa.write_swap_block(block_idx, raw_data, comp_data, kv_cache, gpu)
    }

    fn set_qwen4_hyperconnections(
        &mut self,
        attn: crate::layers::Qwen4HyperConnection,
        mlp: crate::layers::Qwen4HyperConnection,
    ) -> Result<()> {
        Qwen3AttentionLayer::set_qwen4_hyperconnections(self, attn, mlp);
        Ok(())
    }

    fn ddtree_ancestor_attention_exact(&self) -> bool {
        // BF16 certifies only when this exact Qwen target resolved the
        // dedicated tree ABI. Turbo variants still ignore indirection.
        ddtree_indirection_supported(self.kv_dtype, self.paged_decode_bf16_qwen_tree_k.0 != 0)
    }

    fn ddtree_attention_certificate(
        &self,
    ) -> Option<(spark_runtime::kv_cache::KvCacheDtype, bool)> {
        Some((self.kv_dtype, self.ddtree_ancestor_attention_exact()))
    }

    fn decode(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        state: &mut dyn LayerState,
        kv_cache: &mut PagedKvCache,
        seq_len: usize,
        block_table: &mut Vec<u32>,
        disk_block_ids: &mut Vec<u32>,
        disk_last_offloaded_per_layer: &mut Vec<u32>,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        self.decode_inner(
            hidden,
            residual,
            state,
            kv_cache,
            seq_len,
            false,
            block_table,
            disk_block_ids,
            disk_last_offloaded_per_layer,
            ctx,
            stream,
        )
    }

    fn decode_qwen4_verified_row(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        state: &mut dyn LayerState,
        kv_cache: &mut PagedKvCache,
        seq_len: usize,
        block_table: &mut Vec<u32>,
        disk_block_ids: &mut Vec<u32>,
        disk_last_offloaded_per_layer: &mut Vec<u32>,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        self.decode_inner(
            hidden,
            residual,
            state,
            kv_cache,
            seq_len,
            true,
            block_table,
            disk_block_ids,
            disk_last_offloaded_per_layer,
            ctx,
            stream,
        )
    }

    fn decode_kv_only(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        state: &mut dyn LayerState,
        kv_cache: &mut PagedKvCache,
        seq_len: usize,
        block_table: &mut Vec<u32>,
        disk_block_ids: &mut Vec<u32>,
        disk_last_offloaded_per_layer: &mut Vec<u32>,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        self.decode_kv_only_inner(
            hidden,
            residual,
            state,
            kv_cache,
            seq_len,
            block_table,
            disk_block_ids,
            disk_last_offloaded_per_layer,
            ctx,
            stream,
        )
    }

    fn decode_qwen4_batched(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        num_tokens: usize,
        state: &mut dyn LayerState,
        kv_cache: &mut PagedKvCache,
        seq_len: usize,
        block_table: &mut Vec<u32>,
        disk_block_ids: &mut Vec<u32>,
        disk_last_offloaded_per_layer: &mut Vec<u32>,
        h_intermediate: DevicePtr,
        conv_intermediate: DevicePtr,
        h_intermediate_stride: usize,
        conv_intermediate_stride: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        self.decode_qwen4_batched_inner(
            hidden,
            residual,
            num_tokens,
            state,
            kv_cache,
            seq_len,
            block_table,
            disk_block_ids,
            disk_last_offloaded_per_layer,
            h_intermediate,
            conv_intermediate,
            h_intermediate_stride,
            conv_intermediate_stride,
            ctx,
            stream,
        )
    }

    fn decode_qwen4_batched_mtp(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        num_tokens: usize,
        emit_last_row: bool,
        state: &mut dyn LayerState,
        kv_cache: &mut PagedKvCache,
        seq_len: usize,
        block_table: &mut Vec<u32>,
        disk_block_ids: &mut Vec<u32>,
        disk_last_offloaded_per_layer: &mut Vec<u32>,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        self.decode_qwen4_batched_mtp_inner(
            hidden,
            residual,
            num_tokens,
            emit_last_row,
            state,
            kv_cache,
            seq_len,
            block_table,
            disk_block_ids,
            disk_last_offloaded_per_layer,
            ctx,
            stream,
        )
    }

    fn decode_qwen4_cross_seq_mtp(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        seq_lens: &[usize],
        block_tables: &mut [&mut Vec<u32>],
        disk_block_ids: &mut [&mut Vec<u32>],
        disk_last_offloaded_per_layer: &mut [&mut Vec<u32>],
        kv_cache: &mut PagedKvCache,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        self.decode_qwen4_cross_seq_mtp_inner(
            hidden,
            residual,
            seq_lens,
            block_tables,
            disk_block_ids,
            disk_last_offloaded_per_layer,
            kv_cache,
            ctx,
            stream,
        )
    }

    fn preflight_qwen4_native_target(
        &self,
        num_seqs: usize,
        states: &[&(dyn LayerState + 'static)],
        ctx: &ForwardContext,
    ) -> Result<()> {
        self.preflight_qwen4_native_target_inner(num_seqs, states, ctx)
    }

    fn qwen4_native_capability(
        &self,
        num_seqs: usize,
        ctx: &ForwardContext,
    ) -> std::result::Result<(), crate::qwen4_batch_policy::NativeGap> {
        use crate::qwen4_batch_policy::{NativeGap, SerialReason};
        // The MoE half first, so a MoE gap (including a bounded-transpose
        // layout) is named as such rather than as an attention gap.
        self.ffn
            .preflight_qwen4_native_exact_rows(num_seqs, ctx)
            .map_err(|error| NativeGap::new(SerialReason::MoeExactRowsUnavailable, error))?;
        self.preflight_qwen4_native_target_core(num_seqs, ctx)
            .map_err(|error| NativeGap::new(SerialReason::AttentionLayer, error))
    }

    fn decode_qwen4_native_target_attention(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        num_seqs: usize,
        kv_cache: &mut PagedKvCache,
        seq_lens: &[usize],
        block_tables: &[Vec<u32>],
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        self.decode_qwen4_native_target_attention_inner(
            hidden,
            residual,
            num_seqs,
            kv_cache,
            seq_lens,
            block_tables,
            ctx,
            stream,
        )
    }

    fn decode_qwen4_native_target_mlp(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        num_seqs: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        self.decode_qwen4_native_target_mlp_inner(hidden, residual, num_seqs, ctx, stream)
    }

    fn preflight_qwen4_native_target_qsa(
        &self,
        routes: &[crate::layers::qwen4_qsa::native_batch::NativeAttnRoute],
        seq_lens: &[usize],
        kv_cache: &PagedKvCache,
    ) -> Result<()> {
        self.preflight_qwen4_native_qsa_rows(routes, seq_lens, kv_cache)
    }

    fn decode_qwen4_native_target_attention_qsa(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        num_seqs: usize,
        kv_cache: &mut PagedKvCache,
        seq_lens: &[usize],
        block_tables: &[Vec<u32>],
        routes: &[crate::layers::qwen4_qsa::native_batch::NativeAttnRoute],
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        self.decode_qwen4_native_target_attention_routed(
            hidden,
            residual,
            num_seqs,
            kv_cache,
            seq_lens,
            block_tables,
            Some(routes),
            ctx,
            stream,
        )
    }

    fn preflight_qwen4_cross_seq_verify(
        &self,
        frame: crate::layer::Qwen4CrossSeqFrame,
        qsa_required: &[bool],
        ctx: &ForwardContext,
    ) -> Result<()> {
        self.preflight_qwen4_cross_seq_verify_inner(frame, qsa_required, ctx)
    }

    fn decode_qwen4_cross_seq_verify(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        frame: crate::layer::Qwen4CrossSeqFrame,
        seqs: &mut [crate::layer::Qwen4CrossSeqRows<'_>],
        kv_cache: &mut PagedKvCache,
        staging: crate::layer::Qwen4CrossSeqStaging,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        self.decode_qwen4_cross_seq_verify_inner(
            hidden, residual, frame, seqs, kv_cache, staging, ctx, stream,
        )
    }

    fn preflight_qwen4_cross_seq_mixer_only(
        &self,
        frame: crate::layer::Qwen4CrossSeqFrame,
        qsa_required: &[bool],
        ctx: &ForwardContext,
    ) -> Result<()> {
        self.preflight_qwen4_cross_seq_mixer_only_inner(frame, qsa_required, ctx)
    }

    fn decode_qwen4_cross_seq_mixer_only(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        frame: crate::layer::Qwen4CrossSeqFrame,
        seqs: &mut [crate::layer::Qwen4CrossSeqRows<'_>],
        kv_cache: &mut PagedKvCache,
        staging: crate::layer::Qwen4CrossSeqStaging,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        self.decode_qwen4_cross_seq_mixer_only_inner(
            hidden, residual, frame, seqs, kv_cache, staging, ctx, stream,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn prefill(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        num_tokens: usize,
        state: &mut dyn LayerState,
        kv_cache: &mut PagedKvCache,
        seq_len_start: usize,
        block_table: &mut Vec<u32>,
        disk_block_ids: &mut Vec<u32>,
        disk_last_offloaded_per_layer: &mut Vec<u32>,
        kv_write_start: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        self.prefill_inner(
            hidden,
            residual,
            num_tokens,
            state,
            kv_cache,
            seq_len_start,
            block_table,
            disk_block_ids,
            disk_last_offloaded_per_layer,
            kv_write_start,
            None, // batched_meta — single-stream
            ctx,
            stream,
        )
    }

    /// Q12 Path B: batched-mode attention prefill via `prefill_inner` with
    /// `batched_meta = Some`. The model-level `prefill_attn_batched_layer`
    /// calls this method. Per-stream block_table is unused under batched
    /// mode (block_table_ptrs from batched_meta carries them); we still
    /// pass an empty Vec to satisfy the signature.
    fn prefill_inner_batched_q12(
        &self,
        hidden_stacked: DevicePtr,
        residual_stacked: DevicePtr,
        num_tokens: usize,
        kv_cache: &mut PagedKvCache,
        seq_len_start: usize,
        batched_meta: &BatchedAttnMetadata,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let mut empty_state = EmptyLayerState;
        let mut empty_block_table: Vec<u32> = Vec::new();
        let mut empty_disk_block_ids: Vec<u32> = Vec::new();
        let mut empty_disk_last: Vec<u32> = Vec::new();
        self.prefill_inner(
            hidden_stacked,
            residual_stacked,
            num_tokens,
            &mut empty_state,
            kv_cache,
            seq_len_start,
            &mut empty_block_table,
            &mut empty_disk_block_ids,
            &mut empty_disk_last,
            0,
            Some(batched_meta),
            ctx,
            stream,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn decode_multi_seq<'a, 'b: 'a>(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        num_seqs: usize,
        states: &'a mut [&'b mut (dyn LayerState + 'static)],
        kv_cache: &mut PagedKvCache,
        seq_lens: &[usize],
        block_tables: &[Vec<u32>],
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        self.decode_multi_seq_inner(
            hidden,
            residual,
            num_seqs,
            states,
            kv_cache,
            seq_lens,
            block_tables,
            ctx,
            stream,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn decode_multi_seq_qkv_batched(
        &self,
        hidden: DevicePtr,
        num_rows: usize,
        qkv_out_base: DevicePtr,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        self.decode_multi_seq_qkv_batched_inner(hidden, num_rows, qkv_out_base, ctx, stream)
    }

    #[allow(clippy::too_many_arguments)]
    fn decode_multi_seq_attn_from_qkv<'a, 'b: 'a>(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        num_seqs: usize,
        states: &'a mut [&'b mut (dyn LayerState + 'static)],
        kv_cache: &mut PagedKvCache,
        seq_lens: &[usize],
        block_tables: &[Vec<u32>],
        ctx: &ForwardContext,
        stream: u64,
        qkv_base: DevicePtr,
    ) -> Result<()> {
        self.decode_multi_seq_attn_from_qkv(
            hidden,
            residual,
            num_seqs,
            states,
            kv_cache,
            seq_lens,
            block_tables,
            ctx,
            stream,
            qkv_base,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn decode_multi_seq_attn_core_from_qkv(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        num_seqs: usize,
        kv_cache: &mut PagedKvCache,
        seq_lens: &[usize],
        ctx: &ForwardContext,
        stream: u64,
        qkv_base: DevicePtr,
        attn_out_dst: DevicePtr,
    ) -> Result<()> {
        self.decode_multi_seq_attn_core_from_qkv_inner(
            hidden,
            residual,
            num_seqs,
            kv_cache,
            seq_lens,
            ctx,
            stream,
            qkv_base,
            attn_out_dst,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn decode_multi_seq_o_proj_post_norm_ffn_batched(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        total_rows: usize,
        qkv_base: DevicePtr,
        attn_out_base: DevicePtr,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        self.decode_multi_seq_o_proj_post_norm_ffn_batched_inner(
            hidden,
            residual,
            total_rows,
            qkv_base,
            attn_out_base,
            ctx,
            stream,
        )
    }

    fn run_deferred_ffn(
        &self,
        ffn_input: DevicePtr,
        hidden: DevicePtr,
        total_rows: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        use crate::layers::ops;
        // Attention layers may have no FFN (Nemotron-H) — nothing to add.
        if self.ffn.is_none() {
            return Ok(());
        }
        let h = ctx.config.hidden_size;
        let bf16 = 2usize;
        let residual_elem = if ctx.config.use_fp32_residual() {
            4usize
        } else {
            2usize
        };
        let try_kgamma = (total_rows as u32) > 3
            && (self.ffn.wide_kgamma_ready(total_rows as u32)
                || crate::layers::ffn_kgamma_m16_enabled()
                || self.ffn.exact_kgamma_applicable(total_rows as u32));
        let used_kgamma = if try_kgamma {
            let serviced = self
                .ffn
                .forward_kgamma(ffn_input, total_rows as u32, ctx, stream)?;
            if serviced {
                let moe_out = ctx.buffers.moe_output();
                ops::residual_add(
                    ctx.gpu,
                    self.residual_add_k,
                    hidden,
                    moe_out,
                    (total_rows as u32) * h as u32,
                    stream,
                )?;
            }
            serviced
        } else {
            false
        };
        if !used_kgamma {
            for t in 0..total_rows {
                let in_t = ffn_input.offset(t * h * bf16);
                let moe_out = self.ffn.forward(in_t, ctx, stream)?;
                let hidden_t = hidden.offset(t * h * residual_elem);
                ops::residual_add(
                    ctx.gpu,
                    self.residual_add_k,
                    hidden_t,
                    moe_out,
                    h as u32,
                    stream,
                )?;
            }
        }
        Ok(())
    }

    fn alloc_state(&self, _gpu: &dyn GpuBackend) -> Result<Box<dyn LayerState>> {
        Ok(Box::new(EmptyLayerState))
    }

    fn transpose_moe_for_prefill(
        &mut self,
        gpu: &dyn GpuBackend,
        config: &atlas_core::config::ModelConfig,
    ) -> Result<()> {
        if let FfnComponent::Moe(moe) = &mut self.ffn {
            moe.transpose_for_prefill(gpu, config)?;
        }
        if let Some(FfnComponent::Moe(moe)) = self.moe_ffn.as_mut() {
            moe.transpose_for_prefill(gpu, config)?;
        }
        Ok(())
    }

    fn transpose_moe_gate_up_for_prefill(
        &mut self,
        gpu: &dyn GpuBackend,
        config: &atlas_core::config::ModelConfig,
    ) -> Result<()> {
        if let FfnComponent::Moe(moe) = &mut self.ffn {
            moe.transpose_gate_up_for_prefill(gpu, config)?;
        }
        if let Some(FfnComponent::Moe(moe)) = self.moe_ffn.as_mut() {
            moe.transpose_gate_up_for_prefill(gpu, config)?;
        }
        Ok(())
    }

    fn set_moe_down_transpose_scratch(
        &mut self,
        scratch_packed: DevicePtr,
        scratch_scale: DevicePtr,
        packed_ptrs_t: DevicePtr,
        scale_ptrs_t: DevicePtr,
    ) {
        if let FfnComponent::Moe(moe) = &mut self.ffn {
            moe.set_down_transpose_scratch(
                scratch_packed,
                scratch_scale,
                packed_ptrs_t,
                scale_ptrs_t,
            );
        }
        if let Some(FfnComponent::Moe(moe)) = self.moe_ffn.as_mut() {
            moe.set_down_transpose_scratch(
                scratch_packed,
                scratch_scale,
                packed_ptrs_t,
                scale_ptrs_t,
            );
        }
    }

    fn set_moe_stream_transpose_scratch(
        &mut self,
        scratch: crate::layer::MoeStreamTransposeScratch,
    ) {
        if let FfnComponent::Moe(moe) = &mut self.ffn {
            moe.set_moe_stream_transpose_scratch(scratch);
        }
        if let Some(FfnComponent::Moe(moe)) = self.moe_ffn.as_mut() {
            moe.set_moe_stream_transpose_scratch(scratch);
        }
    }

    fn transpose_moe_for_prefill_unified(
        &mut self,
        gpu: &dyn GpuBackend,
        config: &atlas_core::config::ModelConfig,
    ) -> Result<()> {
        if let FfnComponent::Moe(moe) = &mut self.ffn {
            moe.transpose_for_prefill_unified(gpu, config)?;
        }
        if let Some(FfnComponent::Moe(moe)) = self.moe_ffn.as_mut() {
            moe.transpose_for_prefill_unified(gpu, config)?;
        }
        Ok(())
    }

    fn transpose_moe_for_prefill_hybrid(
        &mut self,
        gpu: &dyn GpuBackend,
        config: &atlas_core::config::ModelConfig,
    ) -> Result<()> {
        if let FfnComponent::Moe(moe) = &mut self.ffn {
            moe.transpose_for_prefill_hybrid(gpu, config)?;
        }
        if let Some(FfnComponent::Moe(moe)) = self.moe_ffn.as_mut() {
            moe.transpose_for_prefill_hybrid(gpu, config)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use spark_runtime::gpu::mock::MockGpuBackend;
    use spark_runtime::kv_cache::KvCacheDtype;

    #[test]
    fn ddtree_indirection_support_fails_closed_for_unwired_kv_dtypes() {
        assert!(ddtree_indirection_supported(KvCacheDtype::Fp8, false));
        assert!(ddtree_indirection_supported(KvCacheDtype::Nvfp4, false));
        assert!(!ddtree_indirection_supported(KvCacheDtype::Bf16, false));
        assert!(ddtree_indirection_supported(KvCacheDtype::Bf16, true));
        assert!(!ddtree_indirection_supported(KvCacheDtype::Turbo4, true));
    }

    #[test]
    fn test_alloc_state_returns_empty() {
        let gpu = MockGpuBackend::new();
        assert!(gpu.kernel("norm", "rms_norm").is_ok());
        assert!(gpu.kernel("rope", "rope_forward").is_ok());
        assert!(
            gpu.kernel("paged_decode_fp8", "paged_decode_attn_fp8")
                .is_ok()
        );
    }
}
