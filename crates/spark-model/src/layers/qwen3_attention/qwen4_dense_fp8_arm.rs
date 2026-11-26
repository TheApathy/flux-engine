// SPDX-License-Identifier: AGPL-3.0-only

//! `ATLAS_QWEN4_DENSE_FP8=1` decode arm of the attention layer: q, k/v and
//! o run on native MXFP8 checkpoint tensors inside the decode scope. Each accessor
//! returns `None` when the arm does not apply, so the existing NVFP4 chain
//! runs unchanged.

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;

use super::Qwen3AttentionLayer;
use crate::layer::ForwardContext;
use crate::layers::ops;
use crate::layers::qwen4_dense_fp8::proj::AttnFp8;

impl Qwen3AttentionLayer {
    pub(crate) fn set_qwen4_dense_fp8(&mut self, fp8: AttnFp8) {
        self.qwen4_dense_fp8 = Some(fp8);
    }

    /// q (or q+gate) projection. The gated checkpoint layout is per-head
    /// `[q | gate]`; the GEMV writes it as is and `deinterleave_qg` splits it
    /// exactly like the BF16 dense arm does.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn qwen4_fp8_q(
        &self,
        ctx: &ForwardContext,
        normed: DevicePtr,
        q_out: DevicePtr,
        q_proj_dim: u32,
        h: u32,
        nq: u32,
        hd: u32,
        stream: u64,
    ) -> Option<Result<()>> {
        let fp8 = self.qwen4_dense_fp8.as_ref()?;
        let projected = fp8.q(ctx.gpu, normed, q_out, q_proj_dim, h, stream)?;
        Some(projected.and_then(|()| {
            if !self.gated {
                return Ok(());
            }
            ops::deinterleave_qg(
                ctx.gpu,
                self.deinterleave_qg_k,
                q_out,
                1,
                nq,
                hd,
                nq * hd * 2,
                stream,
            )
        }))
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn qwen4_fp8_kv(
        &self,
        ctx: &ForwardContext,
        normed: DevicePtr,
        k_out: DevicePtr,
        v_out: DevicePtr,
        n: u32,
        h: u32,
        stream: u64,
    ) -> Option<Result<()>> {
        self.qwen4_dense_fp8
            .as_ref()?
            .kv(ctx.gpu, normed, k_out, v_out, n, h, stream)
    }

    pub(super) fn qwen4_fp8_o(
        &self,
        ctx: &ForwardContext,
        attn_out: DevicePtr,
        o_out: DevicePtr,
        h: u32,
        k: u32,
        stream: u64,
    ) -> Option<Result<()>> {
        self.qwen4_dense_fp8
            .as_ref()?
            .o(ctx.gpu, attn_out, o_out, h, k, stream)
    }
}
