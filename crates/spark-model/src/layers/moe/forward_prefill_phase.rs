// SPDX-License-Identifier: AGPL-3.0-only

//! Shared-expert phase of `MoeLayer::forward_prefill`.
//!
//! Hoisted from `forward_prefill.rs` to keep that file under the 500 LoC
//! cap. The single entry point [`MoeLayer::run_shared_expert_prefill`]
//! mirrors the original block 1:1 — same control flow, same kernel
//! launches, same buffer wiring.

use super::*;

impl MoeLayer {
    /// Shared-expert path of the prefill pipeline (gate + up GEMM → SiLU →
    /// down GEMM). Runs sequentially on the supplied `aux` stream when
    /// `use_overlap == false`; otherwise issues an event so the routed
    /// path can wait on completion.
    ///
    /// Skips entirely when `shared_inter == 0` (e.g. Qwen3-VL-30B has no
    /// shared expert). Launching kernels with N=0 returns
    /// CUDA_ERROR_INVALID_VALUE (grid.x=0).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn run_shared_expert_prefill(
        &self,
        input: DevicePtr,
        n: u32,
        h: u32,
        shared_inter: u32,
        aux: u64,
        stream: u64,
        use_overlap: bool,
        ctx: &ForwardContext,
    ) -> Result<()> {
        if shared_inter == 0 {
            return Ok(());
        }
        if !use_overlap
            && n >= 64
            && shared_gemm_selected()
            && let (Some(sg), Some(su), Some(sd)) =
                (&self.shared_gate_t, &self.shared_up_t, &self.shared_down_t)
        {
            return self.run_shared_expert_prefill_gemm(
                input,
                n,
                h,
                shared_inter,
                sg,
                su,
                sd,
                ctx,
                stream,
            );
        }
        if use_overlap {
            // Ensure secondary stream sees `input` (produced by prior default-stream work)
            ctx.gpu.record_event(self.event_a, stream)?;
            ctx.gpu.stream_wait_event(aux, self.event_a)?;
        }

        // Shared gate + up GEMM, activation, and down GEMM on `aux` stream.
        // Wrapped so that when `use_overlap` is true, any partial-launch error
        // still orders/drains `aux` before returning `Err` to the caller.
        let launch_res: Result<()> = (|| {
            let shared_gate_out = ctx.buffers.ssm_deinterleaved();
            let shared_up_out = ctx.buffers.ssm_qkvz();
            if let (Some(sg_fp8), Some(su_fp8)) = (self.shared_gate_fp8, self.shared_up_fp8) {
                ops::fp8_gemm_n128(
                    ctx.gpu,
                    self.fp8_gemm_k,
                    input,
                    sg_fp8,
                    shared_gate_out,
                    n,
                    shared_inter,
                    h,
                    aux,
                )?;
                ops::fp8_gemm_n128(
                    ctx.gpu,
                    self.fp8_gemm_k,
                    input,
                    su_fp8,
                    shared_up_out,
                    n,
                    shared_inter,
                    h,
                    aux,
                )?;
            } else if let (Some(sg), Some(su)) = (&self.shared_gate_t, &self.shared_up_t) {
                ops::w4a16_gemm_n128(
                    ctx.gpu,
                    self.w4a16_gemm_t,
                    input,
                    sg,
                    shared_gate_out,
                    n,
                    shared_inter,
                    h,
                    aux,
                )?;
                ops::w4a16_gemm_n128(
                    ctx.gpu,
                    self.w4a16_gemm_t,
                    input,
                    su,
                    shared_up_out,
                    n,
                    shared_inter,
                    h,
                    aux,
                )?;
            } else {
                ops::w4a16_gemm(
                    ctx.gpu,
                    self.w4a16_gemm,
                    input,
                    &self.weights.shared_expert.gate_proj,
                    shared_gate_out,
                    n,
                    shared_inter,
                    h,
                    aux,
                )?;
                ops::w4a16_gemm(
                    ctx.gpu,
                    self.w4a16_gemm,
                    input,
                    &self.weights.shared_expert.up_proj,
                    shared_up_out,
                    n,
                    shared_inter,
                    h,
                    aux,
                )?;
            }

            // Shared activation (SiLU or GeGLU) + down GEMM on aux stream
            ops::silu_mul(
                ctx.gpu,
                self.moe_act_mul,
                shared_gate_out,
                shared_up_out,
                shared_gate_out,
                n * shared_inter,
                aux,
            )?;
            let shared_down_out = ctx.buffers.attn_output();
            if let Some(sd_fp8) = self.shared_down_fp8 {
                ops::fp8_gemm_n128(
                    ctx.gpu,
                    self.fp8_gemm_k,
                    shared_gate_out,
                    sd_fp8,
                    shared_down_out,
                    n,
                    h,
                    shared_inter,
                    aux,
                )?;
            } else if let Some(sd) = &self.shared_down_t {
                ops::w4a16_gemm_n128(
                    ctx.gpu,
                    self.w4a16_gemm_t,
                    shared_gate_out,
                    sd,
                    shared_down_out,
                    n,
                    h,
                    shared_inter,
                    aux,
                )?;
            } else {
                ops::w4a16_gemm(
                    ctx.gpu,
                    self.w4a16_gemm,
                    shared_gate_out,
                    &self.weights.shared_expert.down_proj,
                    shared_down_out,
                    n,
                    h,
                    shared_inter,
                    aux,
                )?;
            }
            Ok(())
        })();

        if use_overlap {
            let rec_res = ctx.gpu.record_event(self.event_b, aux);
            if launch_res.is_err() || rec_res.is_err() {
                // Drain/order any already-launched kernels on `aux` before returning `Err`
                // so the caller never observes in-flight side-stream accesses to arena buffers.
                let joined = match &rec_res {
                    Ok(()) => ctx.gpu.stream_wait_event(stream, self.event_b),
                    Err(_) => Err(anyhow::anyhow!("event_b record failed")),
                };
                if let Err(wait_err) = joined {
                    tracing::error!(
                        "run_shared_expert_prefill error-path stream_wait_event failed (falling back to synchronize): {wait_err:#}"
                    );
                    if let Err(sync_err) = ctx.gpu.synchronize(aux) {
                        tracing::error!(
                            "run_shared_expert_prefill error-path synchronize(aux) failed: {sync_err:#}"
                        );
                    }
                }
                launch_res?;
                rec_res?;
            }
        } else {
            launch_res?;
        }
        Ok(())
    }

    /// `ATLAS_QWEN4_PREFILL_SHARED_GEMM=1`: the shared expert's three
    /// projections as dequant (transposed NVFP4 -> BF16 [K, N], scale2 folded,
    /// the W4A16 operand) + cuBLASLt BF16 GEMMs instead of `w4a16_gemm_n128`.
    /// Default off; not bit-exact (accumulation order).
    #[allow(clippy::too_many_arguments)]
    fn run_shared_expert_prefill_gemm(
        &self,
        input: DevicePtr,
        n: u32,
        h: u32,
        shared_inter: u32,
        sg: &QuantizedWeight,
        su: &QuantizedWeight,
        sd: &QuantizedWeight,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        use spark_runtime::kernel_args::{KernelLaunch, div_ceil};
        static K: std::sync::OnceLock<Result<spark_runtime::gpu::KernelHandle, String>> =
            std::sync::OnceLock::new();
        let k = K
            .get_or_init(|| {
                ctx.gpu
                    .kernel("qwen4_wide_glue", "qwen4_dequant_t_bf16")
                    .map_err(|e| e.to_string())
            })
            .clone()
            .map_err(|e| {
                anyhow::anyhow!("ATLAS_QWEN4_PREFILL_SHARED_GEMM: kernel unavailable: {e}")
            })?;
        anyhow::ensure!(
            h.is_multiple_of(64) && shared_inter.is_multiple_of(64),
            "ATLAS_QWEN4_PREFILL_SHARED_GEMM requires unpadded transposed weights"
        );
        let elems = (h as usize) * (shared_inter as usize);
        let w = shared_gemm_scratch(ctx.gpu, elems * 2 * 3)?;
        let (wg, wu, wd) = (w, w.offset(elems * 2), w.offset(elems * 4));
        for (q, out, nn, kk) in [
            (sg, wg, shared_inter, h),
            (su, wu, shared_inter, h),
            (sd, wd, h, shared_inter),
        ] {
            KernelLaunch::new(ctx.gpu, k)
                .grid([div_ceil(kk / 2 * nn, 256), 1, 1])
                .block([256, 1, 1])
                .arg_ptr(q.weight)
                .arg_ptr(q.weight_scale)
                .arg_f32(q.weight_scale_2)
                .arg_ptr(out)
                .arg_u32(nn)
                .arg_u32(kk)
                .launch(stream)?;
        }
        let gate_out = ctx.buffers.ssm_deinterleaved();
        let up_out = ctx.buffers.ssm_qkvz();
        spark_runtime::cublaslt::bf16_gemm_act_weight(
            input.0,
            wg.0,
            gate_out.0,
            n,
            shared_inter,
            h,
            stream,
        )?;
        spark_runtime::cublaslt::bf16_gemm_act_weight(
            input.0,
            wu.0,
            up_out.0,
            n,
            shared_inter,
            h,
            stream,
        )?;
        ops::silu_mul(
            ctx.gpu,
            self.moe_act_mul,
            gate_out,
            up_out,
            gate_out,
            n * shared_inter,
            stream,
        )?;
        let down_out = ctx.buffers.attn_output();
        spark_runtime::cublaslt::bf16_gemm_act_weight(
            gate_out.0,
            wd.0,
            down_out.0,
            n,
            h,
            shared_inter,
            stream,
        )?;
        static LOGGED: std::sync::Once = std::sync::Once::new();
        LOGGED.call_once(|| {
            tracing::info!("MOE_SHARED_PREFILL_GEMM_ENGAGED selector=ATLAS_QWEN4_PREFILL_SHARED_GEMM rows={n} non_bit_exact");
        });
        Ok(())
    }
}

fn shared_gemm_selected() -> bool {
    static SEL: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *SEL.get_or_init(|| {
        std::env::var("ATLAS_QWEN4_PREFILL_SHARED_GEMM")
            .ok()
            .as_deref()
            == Some("1")
    })
}

/// Grow-only BF16 weight scratch for the three shared-expert projections.
fn shared_gemm_scratch(gpu: &dyn GpuBackend, bytes: usize) -> Result<DevicePtr> {
    static S: std::sync::Mutex<Option<(DevicePtr, usize)>> = std::sync::Mutex::new(None);
    let mut guard = S
        .lock()
        .map_err(|_| anyhow::anyhow!("shared GEMM scratch poisoned"))?;
    if let Some((ptr, cap)) = *guard
        && cap >= bytes
    {
        return Ok(ptr);
    }
    let ptr = gpu.alloc(bytes)?;
    *guard = Some((ptr, bytes));
    Ok(ptr)
}
