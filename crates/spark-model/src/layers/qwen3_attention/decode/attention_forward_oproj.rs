// SPDX-License-Identifier: AGPL-3.0-only

//! O-projection (GEMV) branch of `attention_forward` decode path.
//! Picks one of: MLA NVFP4 wo, BF16 dense (Gemma-4), W8A16 (FP8 native),
//! or default w4a16. Extracted from `attention_forward.rs` to keep that
//! file under 500 LoC. (MLA decode actually returns through
//! `attention_forward_mla.rs`, but the standard chain still has its own
//! MLA fallback for layers that didn't take the absorbed path.)

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;

use super::super::Qwen3AttentionLayer;
use crate::layer::ForwardContext;
use crate::layers::ops;

impl Qwen3AttentionLayer {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn attention_forward_oproj(
        &self,
        attn_out: DevicePtr,
        nq: u32,
        hd: u32,
        h: u32,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<DevicePtr> {
        let o_out = ctx.buffers.norm_output();
        if let Some(o_bf16) = self.o_dense_bf16_decode.as_ref() {
            // `ATLAS_QWEN4_ATTN_BF16=1` (decode-only). Deliberately a
            // separate field from `o_dense_bf16`: that field also gates
            // FlashInfer prefill-projection eligibility
            // (`flashinfer_projection.rs`), so reusing it would change a
            // prefill admission check, not just decode arithmetic.
            ops::dense_gemv(
                ctx.gpu,
                self.dense_gemv_k,
                attn_out,
                o_bf16,
                o_out,
                h,
                nq * hd,
                stream,
            )?;
        } else if let Some(fp8_decode) = self.qwen4_fp8_o(ctx, attn_out, o_out, h, nq * hd, stream)
        {
            fp8_decode?;
        } else if let Some(ref mla) = self.mla {
            let mla_v_dim = mla.v_dim as u32;
            if let Some(ref wo_nvfp4) = mla.wo_nvfp4 {
                ops::w4a16_gemv(
                    ctx.gpu,
                    self.w4a16_gemv_k,
                    attn_out,
                    wo_nvfp4,
                    o_out,
                    h,
                    nq * mla_v_dim,
                    stream,
                )?;
            } else {
                ops::dense_gemv(
                    ctx.gpu,
                    self.dense_gemv_k,
                    attn_out,
                    &mla.wo,
                    o_out,
                    h,
                    nq * mla_v_dim,
                    stream,
                )?;
            }
        } else if let Some(o_bf16) = self.o_dense_bf16.as_ref() {
            ops::dense_gemv(
                ctx.gpu,
                self.dense_gemv_k,
                attn_out,
                o_bf16,
                o_out,
                h,
                nq * hd,
                stream,
            )?;
        } else if let Some(fp8) = self.o_weight.as_ref().and_then(|w| w.as_fp8()) {
            ops::w8a16_gemv(
                ctx.gpu,
                self.w8a16_gemv_k,
                attn_out,
                fp8.weight,
                fp8.row_scale,
                o_out,
                h,
                nq * hd,
                stream,
            )?;
        } else {
            ops::w4a16_decode_gemv(
                ctx.gpu,
                self.w4a16_gemv_k,
                self.w4a16_gemv_sw_k,
                self.gemv_sw,
                attn_out,
                &self.attn.o_proj,
                o_out,
                h,
                nq * hd,
                stream,
            )?;
        }
        if ctx.config.tp_world_size > 1
            && let Some(comm) = ctx.comm
        {
            comm.all_reduce_async(o_out.0, (h as usize) * 2, stream)?;
        }
        Ok(o_out)
    }
}
