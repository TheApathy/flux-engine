// SPDX-License-Identifier: AGPL-3.0-only

//! Cross-sequence batched DFlash verify: hoisted Q/K/V projection over all
//! `c * K` rows.
//!
//! Runs ONE input RMS norm + ONE `ms_phase_qkv` across all `c * K` rows into
//! the shared `qkv_output` buffer in the per-row `per_seq_qkv`-strided layout:
//! - On SM120/SM103 (`sm_count > 64`), `ms_phase_qkv` dispatches
//!   `try_ms_qkv_sm120_fast` (`M <= 16` coalesced `v2_n16_sk4` + 1-launch
//!   `scatter_qkv_multi_seq`; `M >= 24` direct-strided `try_cublaslt_nvfp4_multiseq_qkv`
//!   + 1-launch `deinterleave_qg`) followed by 1-launch batched `ms_qk_norm_rows`.
//! - On GB10 (`sm_count <= 64`), `ms_phase_qkv` preserves the exact K1-order
//!   QG + dual-KV path (`ms_qkv_exact`) reading each Q/K/V weight once per
//!   `<= 32`-row sweep instead of once per sequence.

use anyhow::{Result, bail};
use spark_runtime::gpu::DevicePtr;

use super::ctx::MultiSeqCtx;
use crate::layer::ForwardContext;
use crate::layers::ops;
use crate::layers::qwen3_attention::Qwen3AttentionLayer;

impl Qwen3AttentionLayer {
    /// RMS-norm (`input_norm`) + batched Q/K/V projection + QK-norm over
    /// `num_rows` contiguous rows of `hidden_in`, writing into `qkv_out_base`
    /// in the `per_seq_qkv`-strided `[Q | (Gate) | K | V]` layout.
    ///
    /// Bails if the layer configuration or active diagnostic controls require
    /// the per-sequence fallback path.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::layers::qwen3_attention) fn decode_multi_seq_qkv_batched_inner(
        &self,
        hidden_in: DevicePtr,
        num_rows: usize,
        qkv_out_base: DevicePtr,
        ctx: &ForwardContext<'_>,
        stream: u64,
    ) -> Result<()> {
        self.validate_yarn_multi_seq_controls()?;
        if num_rows == 0 || num_rows > ctx.buffers.max_batch_tokens() {
            bail!(
                "decode_multi_seq_qkv_batched: invalid num_rows={num_rows} (max={})",
                ctx.buffers.max_batch_tokens()
            );
        }
        if ctx.config.use_fp32_residual() {
            bail!("decode_multi_seq_qkv_batched: FP32 residual stays on per-seq path");
        }
        if self.mla.is_some() {
            bail!("decode_multi_seq_qkv_batched: MLA attention stays on per-seq path");
        }
        if crate::model::env_diag::DflashSerialControls::current().layer_norms {
            bail!("decode_multi_seq_qkv_batched: serial layer_norms diagnostic active");
        }

        let mut c = MultiSeqCtx::new(self, ctx, hidden_in, hidden_in, num_rows, 1, stream, 0);
        c.qkv_buf = qkv_out_base;
        let kv_bytes = (c.nkv * c.hd) as usize * c.bf16;
        let sizes = ctx.buffers.sizes();
        if num_rows * c.h * c.bf16 > sizes.norm_output
            || num_rows * c.per_seq_qkv > sizes.qkv_output
            || num_rows * c.q_proj_bytes > sizes.ssm_qkvz
            || 2 * num_rows * kv_bytes > sizes.attn_output
        {
            bail!(
                "decode_multi_seq_qkv_batched: num_rows={num_rows} exceeds scratch buffer capacity"
            );
        }

        // 1-launch input RMS norm over all `num_rows` rows into `norm_output`.
        // `residual` is untouched here because post-attn `residual_add_rms_norm`
        // adds `o_out` directly into `hidden`.
        ops::rms_norm(
            c.fwd.gpu,
            self.rms_norm_k,
            c.hidden,
            &self.input_norm,
            c.normed,
            num_rows as u32,
            c.h as u32,
            c.eps,
            c.stream,
        )?;

        // Delegate to the proven multi-seq QKV dispatcher (SM120 coalesced /
        // cuBLASLt NVFP4 + 1-launch scatter/deinterleave + 1-launch QK norm;
        // GB10 exact K1-order QG + dual-KV).
        self.ms_phase_qkv(&c)
    }
}
