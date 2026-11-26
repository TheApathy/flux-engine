// SPDX-License-Identifier: AGPL-3.0-only
//! Call site of the QSA-sparse prefill core (`ATLAS_QWEN4_PREFILL_QSA=1`).

use anyhow::{Result, ensure};
use spark_runtime::gpu::DevicePtr;
use spark_runtime::kv_cache::PagedKvCache;

use super::super::Qwen3AttentionLayer;
use crate::layer::ForwardContext;
use crate::layers::qwen4_qsa::Qwen4QsaIndexer;
use crate::layers::qwen4_qsa::prefill_fp8;
use crate::layers::qwen4_qsa::prefill_sparse::{self, SparseChunk};

impl Qwen3AttentionLayer {
    /// True when this chunk's attention core must be the QSA-sparse one: the
    /// selector is on and some row of the chunk lies past the dense window.
    pub(super) fn qsa_sparse_prefill_selected(
        &self,
        seq_len_start: usize,
        rows: usize,
    ) -> Result<bool> {
        Ok(prefill_sparse::selected()?
            && self.qwen4_qsa.is_some()
            && prefill_sparse::chunk_is_sparse(seq_len_start, rows))
    }

    /// Kick off QSA `iq_proj` + `iq_norm_rope` on the secondary stream ahead of
    /// `qkv_proj` + `kv_prep` when `ATLAS_QWEN4_PREFILL_QSA_IQ_OVERLAP` is active
    /// (auto-ON on `active_sm_count() > 64`, OFF on GB10 `<= 64`).
    ///
    /// Returns an RAII guard that joins `done_event` back onto `stream` before
    /// `packed_inputs` is overwritten by `o_proj` or on any early error path.
    pub(super) fn try_start_prefill_qsa_iq_overlap<'a>(
        &self,
        packed_inputs: DevicePtr,
        rows: usize,
        seq_len_start: usize,
        ctx: &ForwardContext<'a>,
        stream: u64,
    ) -> Result<Option<impl Drop + use<'a>>> {
        if ctx.profile
            || ctx.graph_capture
            || spark_runtime::gpu_capture_region::active_capture_region().is_some()
            || ctx.gpu.stream_is_capturing(stream)
            || !Qwen4QsaIndexer::prefill_iq_overlap_enabled()
        {
            return Ok(None);
        }
        if !self.qsa_sparse_prefill_selected(seq_len_start, rows)? {
            return Ok(None);
        }
        let Some(qsa) = self.qwen4_qsa.as_ref() else {
            return Ok(None);
        };
        let Some(metadata) = ctx.attn_metadata else {
            return Ok(None);
        };
        if !metadata.qwen4_qsa_required
            || metadata.num_seqs != 1
            || metadata.block_table.is_null()
            || metadata.positions.is_null()
        {
            return Ok(None);
        }
        let hidden_size = ctx.config.hidden_size;
        let hidden_bytes = rows.saturating_mul(hidden_size).saturating_mul(2);
        qsa.try_launch_prefill_iq_overlap(
            packed_inputs,
            hidden_bytes,
            ctx.buffers.qkv_output(),
            ctx.buffers.sizes().qkv_output,
            rows,
            seq_len_start,
            self.attn_layer_idx,
            u32::try_from(hidden_size)?,
            ctx.config.rms_norm_eps as f32,
            ctx.config.rope_theta as f32,
            ctx.config.rotary_dim() as u32,
            metadata,
            ctx.gpu,
            stream,
        )
    }

    /// Run the sparse core over the chunk's RoPE'd strided QKV rows (K/V and
    /// QSA index keys already stored) into the GATED `[rows][nq*hd]` output:
    /// the sigmoid output gate is fused, so the caller must not apply it again.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn prefill_qsa_sparse_core(
        &self,
        packed_inputs: DevicePtr,
        qkv: DevicePtr,
        row_stride: usize,
        raw_outputs: DevicePtr,
        out_row_elems: usize,
        rows: usize,
        seq_len_start: usize,
        prep_fused: bool,
        kv_cache: &PagedKvCache,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let metadata = ctx.attn_metadata.ok_or_else(|| {
            anyhow::anyhow!("{}: requires paged metadata", prefill_sparse::SELECTOR)
        })?;
        let qsa = self.qwen4_qsa.as_ref().ok_or_else(|| {
            anyhow::anyhow!("{}: requires the QSA indexer", prefill_sparse::SELECTOR)
        })?;
        ensure!(
            metadata.qwen4_qsa_required,
            "{}: requires --qwen4-qsa (the side cache must hold this chunk's index keys)",
            prefill_sparse::SELECTOR
        );
        let nq = ctx.config.num_attention_heads;
        let hd = ctx.config.head_dim;
        ensure!(
            out_row_elems == nq * hd,
            "{}: the sparse core writes a contiguous [rows, nq*hd] output",
            prefill_sparse::SELECTOR
        );
        // The E4M3 core reads the Q8 rows and K8/V8 shadow the BF16 unscaled fused prep wrote.
        // Under NVFP4 paged KV (1M single-Spark) or YaRN scaling, the sparse core streams
        // directly from the native paged KV cache without allocating an FP8 shadow.
        let e4m3 = prefill_fp8::mode()?.is_some()
            && nq == 24
            && ctx.config.num_key_value_heads == 2
            && kv_cache.dtype_for_layer(self.attn_layer_idx)
                == spark_runtime::kv_cache::KvCacheDtype::Bf16
            && self.qwen4_yarn_inv_freq.is_null();
        ensure!(
            !e4m3 || prep_fused,
            "{}: requires the fused full-row attention prep ({})",
            prefill_fp8::SELECTOR,
            crate::layers::qwen4_attn_prep::SELECTOR
        );
        let chunk = SparseChunk {
            hidden: packed_inputs,
            // Q is read in place from the strided QKV rows (normed + RoPE'd by the prep).
            q: qkv,
            q_stride: u32::try_from(row_stride)?,
            out: raw_outputs,
            gate: qkv.offset(nq * hd * 2),
            gate_stride: u32::try_from(row_stride)?,
            rows,
            start: seq_len_start,
            attn_layer_idx: self.attn_layer_idx,
            hidden_size: ctx.config.hidden_size as u32,
            eps: ctx.config.rms_norm_eps as f32,
            rope_theta: ctx.config.rope_theta as f32,
            rotary_dim: ctx.config.rotary_dim() as u32,
            softmax_scale: self.effective_attn_scale(hd as u32),
            e4m3,
            num_query_heads: nq,
        };
        qsa.prefill_sparse_attention(&chunk, kv_cache, metadata, ctx.gpu, stream)
    }
}
