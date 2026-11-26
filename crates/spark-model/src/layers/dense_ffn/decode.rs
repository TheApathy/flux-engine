// SPDX-License-Identifier: AGPL-3.0-only

//! Single-token (K=1) and small fixed-width (K=2, K=3) decode forward passes for `DenseFfnLayer`.

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;

use super::layer::{DenseFfnLayer, FfnActivation};
use crate::layer::ForwardContext;
use crate::layers::{ffn_dual_tuned_enabled, ops};

impl DenseFfnLayer {
    /// Single-token decode: 2-3 kernel launches depending on activation.
    /// SiLU: dual GEMV + SiLU-fused down GEMV (2 launches).
    /// GELU: dual GEMV + gelu_mul + down GEMV (3 launches, no fused GELU down kernel).
    pub fn forward(
        &self,
        input: DevicePtr,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<DevicePtr> {
        let h = ctx.config.hidden_size as u32;
        let inter = ctx.config.intermediate_size as u32;

        let gate_out = ctx.buffers.expert_gate_out();
        let up_out = ctx.buffers.expert_up_out();

        // BF16 dispatch: per-projection GEMV via `dense_gemv_bf16`. We
        // don't have a fused dual-BF16-GEMV kernel today; two sequential
        // launches are still BF16-precision-correct and only ~10% slower
        // than the fused w4a16 path on Gemma-4-31B (the cost is dominated
        // by the bigger BF16 weight reads, not launch overhead).
        if let Some(ref bf16w) = self.bf16_weights {
            ops::dense_gemv(
                ctx.gpu,
                self.dense_gemv_bf16_k,
                input,
                &bf16w.gate_proj,
                gate_out,
                inter,
                h,
                stream,
            )?;
            ops::dense_gemv(
                ctx.gpu,
                self.dense_gemv_bf16_k,
                input,
                &bf16w.up_proj,
                up_out,
                inter,
                h,
                stream,
            )?;
            ops::silu_mul(
                ctx.gpu,
                self.act_mul,
                gate_out,
                up_out,
                gate_out,
                inter,
                stream,
            )?;
            let output = ctx.buffers.moe_output();
            ops::dense_gemv(
                ctx.gpu,
                self.dense_gemv_bf16_k,
                gate_out,
                &bf16w.down_proj,
                output,
                h,
                inter,
                stream,
            )?;
            return Ok(output);
        }

        // W3 (3-bit) single-token decode GEMV path when W3 weights and kernels
        // are installed on this SiLU layer (`ATLAS_FFN_W3_LAYERS`).
        if self.has_w3_gemv()
            && let Some(ref w3) = self.w3_weights
        {
            crate::kprof!(ctx.gpu, stream, "ffn_gate_up_dual_w3_m1", {
                ops::w3a16_gemv_dual(
                    ctx.gpu,
                    self.w3a16_gemv_dual_k,
                    input,
                    &w3.gate_proj,
                    gate_out,
                    &w3.up_proj,
                    up_out,
                    inter,
                    h,
                    stream,
                )?;
                anyhow::Result::<()>::Ok(())
            })?;
            if self.sparsity_measure_active() && !ctx.graph_capture {
                let (hist_g, count_g, hist_d, count_d, meas_silu) =
                    self.ensure_sparsity_meas(ctx.gpu, inter as usize)?;
                self.measure_sparsity_site(ctx, input, hist_g, count_g, h, stream)?;
                ops::silu_mul(
                    ctx.gpu,
                    self.act_mul,
                    gate_out,
                    up_out,
                    meas_silu,
                    inter,
                    stream,
                )?;
                self.measure_sparsity_site(ctx, meas_silu, hist_d, count_d, inter, stream)?;
                self.maybe_dump_sparsity(ctx, "dense_ffn", stream)?;
            }
            let output = ctx.buffers.moe_output();
            crate::kprof!(ctx.gpu, stream, "ffn_down_silu_w3_m1", {
                ops::w3a16_gemv_silu_input(
                    ctx.gpu,
                    self.w3a16_gemv_silu_input_k,
                    gate_out,
                    up_out,
                    &w3.down_proj,
                    output,
                    h,
                    inter,
                    stream,
                )?;
                anyhow::Result::<()>::Ok(())
            })?;
            return Ok(output);
        }

        // Fused gate_proj + up_proj: [1, H] → [1, inter] × 2
        crate::kprof!(ctx.gpu, stream, "ffn_gate_up_dual_m1", {
            ops::w4a16_decode_gemv_dual(
                ctx.gpu,
                self.w4a16_gemv_dual,
                self.w4a16_gemv_dual_sw,
                self.gemv_sw,
                input,
                &self.weights.gate_proj,
                gate_out,
                &self.weights.up_proj,
                up_out,
                inter,
                h,
                stream,
            )?;
            anyhow::Result::<()>::Ok(())
        })?;

        // ── FFN activation-sparsity MEASUREMENT (observer, default-off) ──
        // Runs ONLY when ATLAS_MEASURE_FFN_SPARSITY=1 and the kernel symbol is
        // present. Pure reader: measures `input` (gate/up in, K=hidden) and a
        // freshly-recomputed `silu(gate)*up` copy (down in, K=inter) into
        // dedicated counter buffers. It never touches `input`, `gate_out`,
        // `up_out`, or `output`, so the token stream stays byte-identical
        // whether the gate is on or off (deterministic greedy token parity preserved).
        //
        // Skipped under CUDA graph capture: the observer lazily `gpu.alloc`s its
        // counter buffers on first use, which is illegal mid-capture. The
        // measurement run is intended for eager decode (the operator sets the
        // gate for a measurement window); skipping graphed steps only omits a
        // subset of rows from the average and never perturbs the token stream.
        if self.sparsity_measure_active() && !ctx.graph_capture {
            let (hist_g, count_g, hist_d, count_d, meas_silu) =
                self.ensure_sparsity_meas(ctx.gpu, inter as usize)?;
            // Site 0: gate/up input (residual stream, K=hidden).
            self.measure_sparsity_site(ctx, input, hist_g, count_g, h, stream)?;
            // Site 1: down input = silu(gate)*up (K=intermediate). Recompute
            // into the observer's OWN scratch so gate_out/up_out (read by the
            // fused down GEMV below) are untouched.
            ops::silu_mul(
                ctx.gpu,
                self.act_mul,
                gate_out,
                up_out,
                meas_silu,
                inter,
                stream,
            )?;
            self.measure_sparsity_site(ctx, meas_silu, hist_d, count_d, inter, stream)?;
            self.maybe_dump_sparsity(ctx, "dense_ffn", stream)?;
        }

        let output = ctx.buffers.moe_output();
        match self.activation {
            FfnActivation::SiLU => {
                // Fused SiLU(gate)*up + down_proj: [1, inter] → [1, H]
                crate::kprof!(ctx.gpu, stream, "ffn_down_silu_m1", {
                    ops::w4a16_decode_gemv_silu_input(
                        ctx.gpu,
                        self.w4a16_gemv_silu_input,
                        self.w4a16_gemv_silu_input_sw,
                        self.gemv_sw,
                        gate_out,
                        up_out,
                        &self.weights.down_proj,
                        output,
                        h,
                        inter,
                        stream,
                    )?;
                    anyhow::Result::<()>::Ok(())
                })?;
            }
            FfnActivation::GeLU => {
                // GELU(gate)*up → gate_out, then down_proj GEMV
                crate::kprof!(ctx.gpu, stream, "ffn_gelu_mul_m1", {
                    ops::silu_mul(
                        ctx.gpu,
                        self.act_mul,
                        gate_out,
                        up_out,
                        gate_out,
                        inter,
                        stream,
                    )?;
                    anyhow::Result::<()>::Ok(())
                })?;
                crate::kprof!(ctx.gpu, stream, "ffn_down_m1", {
                    ops::w4a16_decode_gemv(
                        ctx.gpu,
                        self.w4a16_gemv,
                        self.w4a16_gemv_sw,
                        self.gemv_sw,
                        gate_out,
                        &self.weights.down_proj,
                        output,
                        h,
                        inter,
                        stream,
                    )?;
                    anyhow::Result::<()>::Ok(())
                })?;
            }
        }

        Ok(output)
    }

    /// K=2 speculative: batched GEMV for 2 tokens.
    /// 3 launches: dual batch2 (gate+up) + silu_mul + batch2 (down).
    pub fn forward_k2(&self, input: DevicePtr, ctx: &ForwardContext, stream: u64) -> Result<()> {
        let h = ctx.config.hidden_size as u32;
        let inter = ctx.config.intermediate_size as u32;

        let gate_out = ctx.buffers.expert_gate_out();
        let up_out = ctx.buffers.expert_up_out();

        // Fused gate+up for 2 tokens
        ops::w4a16_gemv_dual_batch2(
            ctx.gpu,
            self.w4a16_gemv_dual_batch2,
            input,
            &self.weights.gate_proj,
            gate_out,
            &self.weights.up_proj,
            up_out,
            inter,
            h,
            stream,
        )?;
        ops::silu_mul(
            ctx.gpu,
            self.act_mul,
            gate_out,
            up_out,
            gate_out,
            2 * inter,
            stream,
        )?;
        let output = ctx.buffers.moe_output();
        ops::w4a16_gemv_batch2(
            ctx.gpu,
            self.w4a16_gemv_batch2,
            gate_out,
            &self.weights.down_proj,
            output,
            h,
            inter,
            stream,
        )?;

        Ok(())
    }

    /// K=3 speculative: batched GEMV for 3 tokens.
    /// 3 launches: dual batch3 (gate+up) + silu_mul + batch3 (down).
    pub fn forward_k3(&self, input: DevicePtr, ctx: &ForwardContext, stream: u64) -> Result<()> {
        // Tensor-core M=3 fast path (ATLAS_TC_NVFP4_K3=1):
        //
        // Routes through `w4a16_gemm_t_m16_n64` (small-M, N_TILE=64) for
        // gate / up / down. Dispatches 3 GEMM launches (gate, up, down)
        // instead of the GEMV path's 3 launches (dual gate+up, silu, down)
        // but each GEMM runs on tensor cores via m16n8k32 e4m3 MMA.
        //
        // The first attempt (routing to `forward_kgamma(n=3)` which used
        // `w4a16_gemm_t_m16` with N_TILE=128) measured −23% mean tok/s
        // on Qwen3.6-27B because the 136-CTA grid under-saturated GB10's
        // 48 SMs (~2.8 CTAs/SM).
        // The N_TILE=64 variant fields 272 CTAs/projection (~5.7 CTAs/SM)
        // at half the per-CTA work — designed to keep the tensor-core
        // pipeline fed at M=3.
        //
        // Bounds: requires transposed weights + the n64 kernel symbol
        // loaded; falls through to the GEMV path otherwise.
        if crate::layers::tc_nvfp4_k3_enabled()
            && self.has_transposed_ffn()
            && self.w4a16_gemm_t_m16_n64.0 != 0
        {
            let h = ctx.config.hidden_size as u32;
            let inter = ctx.config.intermediate_size as u32;
            let gate_out_buf = ctx.buffers.expert_gate_out();
            let up_out_buf = ctx.buffers.expert_up_out();
            let gt = self.gate_proj_t.as_ref().unwrap();
            let ut = self.up_proj_t.as_ref().unwrap();
            let dt = self.down_proj_t.as_ref().unwrap();

            crate::kprof!(ctx.gpu, stream, "ffn_gate_up_dual_batch3", {
                ops::w4a16_gemm_n64_m16(
                    ctx.gpu,
                    self.w4a16_gemm_t_m16_n64,
                    input,
                    gt,
                    gate_out_buf,
                    3,
                    inter,
                    h,
                    stream,
                )?;
                ops::w4a16_gemm_n64_m16(
                    ctx.gpu,
                    self.w4a16_gemm_t_m16_n64,
                    input,
                    ut,
                    up_out_buf,
                    3,
                    inter,
                    h,
                    stream,
                )?;
                anyhow::Result::<()>::Ok(())
            })?;
            crate::kprof!(ctx.gpu, stream, "ffn_silu_mul", {
                ops::silu_mul(
                    ctx.gpu,
                    self.act_mul,
                    gate_out_buf,
                    up_out_buf,
                    gate_out_buf,
                    3 * inter,
                    stream,
                )?;
                anyhow::Result::<()>::Ok(())
            })?;
            let output = ctx.buffers.moe_output();
            crate::kprof!(ctx.gpu, stream, "ffn_down_batch3", {
                ops::w4a16_gemm_n64_m16(
                    ctx.gpu,
                    self.w4a16_gemm_t_m16_n64,
                    gate_out_buf,
                    dt,
                    output,
                    3,
                    h,
                    inter,
                    stream,
                )?;
                anyhow::Result::<()>::Ok(())
            })?;
            return Ok(());
        }

        let h = ctx.config.hidden_size as u32;
        let inter = ctx.config.intermediate_size as u32;

        let gate_out = ctx.buffers.expert_gate_out();
        let up_out = ctx.buffers.expert_up_out();

        // Fused gate+up for 3 tokens. Tuned variant (gated by
        // `ATLAS_FFN_DUAL_TUNED=1`) fuses both projections into the SAME CTA
        // so the 3-token activation vector is loaded once per CTA instead of
        // twice. Falls back to the baseline kernel when the env var is unset
        // OR the tuned kernel symbol was not present in the loaded cache.
        let use_tuned = ffn_dual_tuned_enabled() && self.w4a16_gemv_dual_batch3_tuned.0 != 0;
        let dual_kernel = if use_tuned {
            self.w4a16_gemv_dual_batch3_tuned
        } else {
            self.w4a16_gemv_dual_batch3
        };
        crate::kprof!(ctx.gpu, stream, "ffn_gate_up_dual_batch3", {
            if use_tuned {
                ops::w4a16_gemv_dual_batch3_tuned(
                    ctx.gpu,
                    dual_kernel,
                    input,
                    &self.weights.gate_proj,
                    gate_out,
                    &self.weights.up_proj,
                    up_out,
                    inter,
                    h,
                    stream,
                )?;
            } else {
                ops::w4a16_gemv_dual_batch3(
                    ctx.gpu,
                    dual_kernel,
                    input,
                    &self.weights.gate_proj,
                    gate_out,
                    &self.weights.up_proj,
                    up_out,
                    inter,
                    h,
                    stream,
                )?;
            }
            anyhow::Result::<()>::Ok(())
        })?;
        crate::kprof!(ctx.gpu, stream, "ffn_silu_mul", {
            ops::silu_mul(
                ctx.gpu,
                self.act_mul,
                gate_out,
                up_out,
                gate_out,
                3 * inter,
                stream,
            )?;
            anyhow::Result::<()>::Ok(())
        })?;
        let output = ctx.buffers.moe_output();
        crate::kprof!(ctx.gpu, stream, "ffn_down_batch3", {
            ops::w4a16_gemv_batch3(
                ctx.gpu,
                self.w4a16_gemv_batch3,
                gate_out,
                &self.weights.down_proj,
                output,
                h,
                inter,
                stream,
            )?;
            anyhow::Result::<()>::Ok(())
        })?;

        Ok(())
    }
}
