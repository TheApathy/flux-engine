// SPDX-License-Identifier: AGPL-3.0-only

//! N-token prefill and batched forward dispatch for `DenseFfnLayer`.

use anyhow::{Result, bail, ensure};
use spark_runtime::gpu::DevicePtr;

use super::e2m1_flashinfer::{
    E2M1_KMAJOR_M256_MIN_M, E2m1PrefillRoute, E2m1Scope, E2m1SelectedKernel,
    e2m1_kmajor_projection_shape, e2m1_prefill_route, e2m1_selected_kernel,
};
use super::layer::{DenseFfnLayer, FfnActivation};
use super::policy::{
    E2m1RouteProof, PrefillFfnFastRoute, PrefillFusedEpilogueRoute, log_e2m1_route_once,
    log_prefill_dual_fused_route_once, log_prefill_ffn_fast_route_once,
    log_prefill_pipe_route_once, log_prefill_up_fused_route_once, prefill_ffn_fast_route,
    prefill_fused_epilogue_route,
};
use crate::layer::ForwardContext;
use crate::layers::ops;

impl DenseFfnLayer {
    /// N-token prefill: GEMM for all projections.
    pub fn forward_prefill(
        &self,
        input: DevicePtr,
        num_tokens: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let h = ctx.config.hidden_size as u32;
        let inter = ctx.config.intermediate_size as u32;
        let m = num_tokens as u32;

        let gate_out = ctx.buffers.expert_gate_out();
        let up_out = ctx.buffers.expert_up_out();

        // BF16 prefill dispatch: dense_gemm_bf16 for all three projections.
        // (Gemma-4 dense path — bypasses NVFP4 entirely; not affected by
        // ATLAS_PREFILL_FFN_FAST.)
        if let Some(ref bf16w) = self.bf16_weights {
            ops::dense_gemm(
                ctx.gpu,
                self.dense_gemm_bf16_k,
                input,
                &bf16w.gate_proj,
                gate_out,
                m,
                inter,
                h,
                stream,
            )?;
            ops::dense_gemm(
                ctx.gpu,
                self.dense_gemm_bf16_k,
                input,
                &bf16w.up_proj,
                up_out,
                m,
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
                m * inter,
                stream,
            )?;
            let output = ctx.buffers.moe_output();
            ops::dense_gemm(
                ctx.gpu,
                self.dense_gemm_bf16_k,
                gate_out,
                &bf16w.down_proj,
                output,
                m,
                h,
                inter,
                stream,
            )?;
            return Ok(());
        }

        #[cfg(all(feature = "cuda", target_os = "linux"))]
        if self.try_forward_flashinfer_prefill(input, m, h, inter, gate_out, up_out, ctx, stream)? {
            return Ok(());
        }

        #[cfg(feature = "cuda")]
        if self
            .try_forward_cublaslt_nvfp4_prefill(input, m, h, inter, gate_out, up_out, ctx, stream)?
        {
            return Ok(());
        }

        // Large-M W4A4 native NVFP4×NVFP4 fast path: route through the
        // CUTLASS-style `nvfp4_nvfp4_gemm_t_m64` kernel (native E2M1
        // tensor-core MMA) when:
        //   - ATLAS_E2M1_GEMM=1 was set at startup
        //   - all three required kernel symbols loaded (nvfp4_cutlass +
        //     quantize_nvfp4)
        //   - M >= 128 (kernel's intended window — matches the other
        //     prefill fast paths)
        // Each GEMM prequantizes BF16 activations to NVFP4 inline
        // (absmax + per-group E2M1) before dispatching the native MMA.
        // Theoretical 2× MFU lift over `w4a16_gemm_t_m128` (BF16×NVFP4)
        // since we eliminate the inner-loop dequant AND halve activation
        // DRAM traffic (0.5 B/elt vs 2 B/elt BF16). Takes precedence
        // over the fp8/v2/m128 paths when active. Disabled or geometrically
        // ineligible requests retain those fallbacks; eligible incomplete
        // requests fail before projection.
        let e2m1_full_requested = crate::layers::prefill_ffn_e2m1_enabled();
        let e2m1_down_requested = crate::layers::prefill_ffn_e2m1_down_only_enabled();
        let e2m1_kmajor_requested = crate::layers::prefill_ffn_e2m1_kmajor_enabled();
        let e2m1_kmajor_m256_requested = crate::layers::prefill_ffn_e2m1_kmajor_m256_enabled();
        let e2m1_silu_quant_requested = crate::layers::prefill_ffn_e2m1_silu_quant_enabled();
        if m >= 128 && e2m1_silu_quant_requested {
            ensure!(
                e2m1_full_requested || e2m1_down_requested,
                "ATLAS_E2M1_SILU_QUANT=1 requires ATLAS_E2M1_GEMM=1 or ATLAS_E2M1_GEMM_DOWN_ONLY=1"
            );
            ensure!(
                self.e2m1_checkpoint_scales.is_some(),
                "ATLAS_E2M1_SILU_QUANT=1 requires ATLAS_E2M1_STATIC_SCALE=1 and valid checkpoint input scales"
            );
            ensure!(
                self.activation == FfnActivation::SiLU,
                "ATLAS_E2M1_SILU_QUANT=1 is valid only for SiLU-gated FFNs"
            );
            ensure!(
                self.nvfp4_silu_quantize_k.0 != 0,
                "ATLAS_E2M1_SILU_QUANT=1 requires quantize_silu_mul_bf16_to_nvfp4 in the quantize_nvfp4 module"
            );
        }
        if m >= 128 && e2m1_kmajor_requested {
            ensure!(
                e2m1_full_requested || e2m1_down_requested,
                "ATLAS_E2M1_KMAJOR=1 requires ATLAS_E2M1_GEMM=1 or ATLAS_E2M1_GEMM_DOWN_ONLY=1"
            );
            ensure!(
                self.has_e2m1_kmajor_runtime(),
                "ATLAS_E2M1_KMAJOR=1 requires nvfp4_nvfp4_gemm_kmajor_m128 plus the activation quantizer"
            );
            ensure!(
                self.has_transposed_ffn(),
                "ATLAS_E2M1_KMAJOR=1 requires transformed gate/up/down weights (ATLAS_FFN_M16_TRANSPOSED=1)"
            );
            ensure!(
                e2m1_kmajor_projection_shape(m, inter, h)
                    && e2m1_kmajor_projection_shape(m, h, inter),
                "ATLAS_E2M1_KMAJOR=1 requires M>=128, N divisible by 128, and K divisible by 64 for every FFN projection; got M={m} hidden={h} intermediate={inter}"
            );
        }
        if e2m1_kmajor_m256_requested {
            ensure!(
                e2m1_kmajor_requested,
                "ATLAS_E2M1_KMAJOR_M256=1 requires ATLAS_E2M1_KMAJOR=1"
            );
            if m >= E2M1_KMAJOR_M256_MIN_M {
                ensure!(
                    self.has_e2m1_kmajor_m256_runtime(),
                    "ATLAS_E2M1_KMAJOR_M256=1 requires nvfp4_nvfp4_gemm_kmajor_m256 plus the M128 K-major runtime"
                );
            }
        }
        let kmajor_geometry_ready =
            e2m1_kmajor_projection_shape(m, inter, h) && e2m1_kmajor_projection_shape(m, h, inter);
        let selected_kernel =
            e2m1_selected_kernel(e2m1_kmajor_requested, e2m1_kmajor_m256_requested, m);
        let e2m1_runtime_ready = if e2m1_kmajor_requested {
            self.has_e2m1_kmajor_runtime()
                && self.has_transposed_ffn()
                && kmajor_geometry_ready
                && (selected_kernel != E2m1SelectedKernel::KmajorM256
                    || self.has_e2m1_kmajor_m256_runtime())
        } else {
            self.has_e2m1_ffn()
        };
        let silu_quant_ready = !e2m1_silu_quant_requested
            || (self.e2m1_checkpoint_scales.is_some()
                && self.activation == FfnActivation::SiLU
                && self.nvfp4_silu_quantize_k.0 != 0);

        // Per-shape E2M1 dispatch: gate/up stay on `w4a16_gemm_t_m128`,
        // down_proj routes through native E2M1 hardware MMA. Per the
        // shape table at the `prefill_ffn_e2m1_down_only_enabled` doc
        // site: gate (K=5120,N=17408) + up (same) are faster on the
        // BF16×NVFP4 w4a16 m128 path, while down (K=17408,N=5120) is
        // 1.31× faster via E2M1 MMA — net ~30% down savings with no
        // gate/up regression. Mutually exclusive with the all-three
        // `e2m1_fast_path`.
        let e2m1_route = e2m1_prefill_route(
            e2m1_full_requested,
            e2m1_down_requested,
            m >= 128,
            e2m1_runtime_ready && silu_quant_ready,
            e2m1_runtime_ready
                && silu_quant_ready
                && self.has_transposed_ffn()
                && self.w4a16_gemm_t_m128.0 != 0,
        );
        let (e2m1_fast_path, e2m1_down_only_path) = match e2m1_route {
            E2m1PrefillRoute::Complete(E2m1Scope::Full) => (true, false),
            E2m1PrefillRoute::Complete(E2m1Scope::DownOnly) => (false, true),
            E2m1PrefillRoute::Missing(E2m1Scope::Full) => {
                bail!(
                    "ATLAS_E2M1_GEMM=1 requires the complete selected W4A4 runtime before gate projection"
                );
            }
            E2m1PrefillRoute::Missing(E2m1Scope::DownOnly) => {
                bail!(
                    "ATLAS_E2M1_GEMM_DOWN_ONLY=1 requires the complete selected W4A4 runtime plus transformed W4A16 M128 gate/up before gate projection"
                );
            }
            E2m1PrefillRoute::Disabled | E2m1PrefillRoute::Ineligible => (false, false),
        };
        if let E2m1PrefillRoute::Complete(scope) = e2m1_route {
            log_e2m1_route_once(E2m1RouteProof {
                scope,
                kernel: selected_kernel,
                checkpoint_static: self.e2m1_checkpoint_scales.is_some(),
                fused_silu_input: e2m1_silu_quant_requested,
                activation: self.activation,
                m,
                h,
                inter,
            });
        }

        // Large-M FP8 predequant fast path: route through the
        // `fp8_gemm_t_m128` kernel (BF16 A × pre-dequanted FP8 B) when:
        //   - ATLAS_FFN_PREDEQUANT_FP8=1 was set at startup
        //   - `predequant_for_prefill` ran successfully (gate/up/down
        //     FP8 buffers installed)
        //   - the `fp8_gemm_t_m128` kernel symbol is present
        //   - M >= 128
        // Saves the entire DEQUANT phase + one __syncthreads per
        // K-step inside w4a16_gemm_t_m128. Empirically a REGRESSION
        // on Qwen3.6-27B (5.68s → 8.11s TTFT at ISL=3603) because the
        // 2× B DRAM traffic (FP8 1 byte vs NVFP4 0.5 byte/elt) at
        // K=5120 + N=17408 exceeds the dequant savings — the FFN GEMM
        // is memory-bound, not compute-bound. Falls back to the w4a16
        // m128 path when not active. Mirrors the attention
        // `predequant_for_prefill` + `fp8_gemm_n128_m128` pattern.
        let fp8_fast_path = !e2m1_fast_path
            && !e2m1_down_only_path
            && m >= 128
            && crate::layers::prefill_ffn_fp8_enabled()
            && self.has_fp8_ffn();

        // Large-M v2 fast path: route through the 8-warp shadow
        // `w4a16_gemm_t_m128_v2` kernel when:
        //   - ATLAS_FFN_M128_V2=1
        //   - transposed (`nvfp4_t`) FFN weights installed
        //   - the v2 kernel symbol is present
        //   - M >= 128
        // Same SMEM footprint + 2-stage pipeline as v1 but doubles the
        // active warp count (256 threads/CTA, 4 warps per chunk run
        // chunk 0 and chunk 1 MMAs in parallel instead of serial). Net
        // result on compute-bound GEMMs: ~10-20% kernel-time win. On
        // memory-bound GEMMs: roughly neutral. Falls back to v1 when
        // not active. Original kernel:
        // kernels/gb10/minimax-m2-229b/nvfp4/w4a16_gemm_v2.cu —
        // copied verbatim into qwen3.6-27b/.
        let v2_fast_path = !e2m1_fast_path
            && !e2m1_down_only_path
            && !fp8_fast_path
            && m >= 128
            && crate::layers::prefill_ffn_m128_v2_enabled()
            && self.has_transposed_ffn()
            && self.w4a16_gemm_t_m128_v2.0 != 0;

        // Large-M fast path: route through the transposed-weight
        // M_TILE=128 kernel (`w4a16_gemm_t_m128`) when:
        //   - ATLAS_PREFILL_FFN_FAST is enabled (default ON unless `=0`)
        //   - transposed (`nvfp4_t`) FFN weights installed at load
        //     (controlled by ATLAS_FFN_M16_TRANSPOSED, default ON unless `=0`)
        //   - the kernel symbol is present in the loaded module
        //   - M >= 128 (kernel's intended window — small-M would waste
        //     128-row CTAs the same way the M_TILE=64 path wastes 64-row
        //     CTAs at M < 64)
        // Mirrors the attention `w4a16_gemm_m128_dispatch` pattern in
        // `qwen3_attention/prefill_weights.rs`. Disabled or ineligible
        // requests retain the lower-priority route. An eligible explicit
        // request fails closed when any required weight/kernel is absent so
        // a stale bundle cannot be mislabeled as a FAST benchmark.
        let fast_shape_eligible =
            !e2m1_fast_path && !e2m1_down_only_path && !fp8_fast_path && !v2_fast_path && m >= 128;
        let fast_route = prefill_ffn_fast_route(
            crate::layers::prefill_ffn_fast_enabled(),
            fast_shape_eligible,
            self.gate_proj_t.is_some() && self.up_proj_t.is_some() && self.down_proj_t.is_some(),
            self.w4a16_gemm_t_m16.0 != 0 || self.prefill_t_m128_only,
            self.w4a16_gemm_t_m128.0 != 0,
        );
        let fast_path = match fast_route {
            PrefillFfnFastRoute::Complete => true,
            PrefillFfnFastRoute::Missing if !crate::layers::prefill_ffn_fast_explicit() => {
                static DEFAULT_MISSING: std::sync::Once = std::sync::Once::new();
                DEFAULT_MISSING.call_once(|| {
                    tracing::warn!(
                        "default-on ATLAS_PREFILL_FFN_FAST skipped: transposed FFN weights or \
                         w4a16_gemm_t_m16/m128 are absent for this checkpoint; using the \
                         M_TILE=64 route (set ATLAS_PREFILL_FFN_FAST=1 to fail closed instead)"
                    );
                });
                false
            }
            PrefillFfnFastRoute::Missing => {
                bail!(
                    "ATLAS_PREFILL_FFN_FAST=1 requires transformed gate/up/down weights plus w4a16_gemm_t_m16 and w4a16_gemm_t_m128 before gate projection"
                );
            }
            PrefillFfnFastRoute::Disabled | PrefillFfnFastRoute::Ineligible => false,
        };

        // One-shot info log on first prefill so the bench harness can
        // verify which path is firing. The atomic ensures we log once
        // across all 64 layers per process. Uses both tracing::info! and
        // eprintln! — the latter bypasses tracing's BufWriter when the
        // server is launched under `nohup ... > file 2>&1` (stderr is
        // line-buffered to a regular file, so a single `\n`-terminated
        // write hits disk immediately, even before process teardown).
        static LOGGED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        if !LOGGED.swap(true, std::sync::atomic::Ordering::Relaxed) {
            let has_t = self.has_transposed_ffn();
            let has_fp8 = self.has_fp8_ffn();
            let has_e2m1 = self.has_e2m1_ffn();
            let has_e2m1_kmajor = self.has_e2m1_kmajor_runtime();
            let has_e2m1_kmajor_m256 = self.has_e2m1_kmajor_m256_runtime();
            let m128_ok = self.w4a16_gemm_t_m128.0 != 0;
            let m128_v2_ok = self.w4a16_gemm_t_m128_v2.0 != 0;
            let fp8_m128_ok = self.fp8_gemm_t_m128_k.0 != 0;
            let e2m1_ok = self.nvfp4_gemm_k.0 != 0;
            let gate_on = crate::layers::prefill_ffn_fast_enabled();
            let fp8_gate_on =
                std::env::var("ATLAS_FFN_PREDEQUANT_FP8").ok().as_deref() == Some("1");
            let v2_gate_on = std::env::var("ATLAS_FFN_M128_V2").ok().as_deref() == Some("1");
            let e2m1_gate_on = std::env::var("ATLAS_E2M1_GEMM").ok().as_deref() == Some("1");
            let e2m1_down_only_gate_on =
                std::env::var("ATLAS_E2M1_GEMM_DOWN_ONLY").ok().as_deref() == Some("1");
            let has_e2m1_checkpoint_scales = self.e2m1_checkpoint_scales.is_some();
            let e2m1_static_scale_env =
                std::env::var("ATLAS_E2M1_STATIC_SCALE").ok().as_deref() == Some("1");
            tracing::info!(
                m,
                inter,
                h,
                e2m1_fast_path,
                e2m1_down_only_path,
                fp8_fast_path,
                v2_fast_path,
                fast_path,
                has_e2m1_ffn = has_e2m1,
                has_fp8_ffn = has_fp8,
                has_transposed = has_t,
                e2m1_kernel = e2m1_ok,
                fp8_m128_kernel = fp8_m128_ok,
                m128_v2_kernel = m128_v2_ok,
                m128_kernel = m128_ok,
                gate = gate_on,
                fp8_gate = fp8_gate_on,
                v2_gate = v2_gate_on,
                e2m1_gate = e2m1_gate_on,
                e2m1_down_only_gate = e2m1_down_only_gate_on,
                has_e2m1_checkpoint_scales,
                e2m1_static_scale_env,
                e2m1_kmajor_requested,
                e2m1_kmajor_m256_requested,
                e2m1_silu_quant_requested,
                has_e2m1_kmajor,
                has_e2m1_kmajor_m256,
                "dense_ffn forward_prefill dispatch (one-shot)"
            );
            eprintln!(
                "[atlas-prefill-ffn] dispatch: M={m} inter={inter} h={h} \
                 e2m1_fast_path={e2m1_fast_path} \
                 e2m1_down_only_path={e2m1_down_only_path} \
                 fp8_fast_path={fp8_fast_path} v2_fast_path={v2_fast_path} \
                 fast_path={fast_path} has_e2m1={has_e2m1} has_fp8={has_fp8} \
                 has_transposed={has_t} e2m1_kernel={e2m1_ok} \
                 fp8_m128_kernel={fp8_m128_ok} \
                 m128_v2_kernel={m128_v2_ok} m128_kernel={m128_ok} \
                 e2m1_gate=ATLAS_E2M1_GEMM={e2m1_gate_on} \
                 e2m1_down_only_gate=ATLAS_E2M1_GEMM_DOWN_ONLY={e2m1_down_only_gate_on} \
                 has_e2m1_checkpoint_scales={has_e2m1_checkpoint_scales} \
                 e2m1_static_scale=ATLAS_E2M1_STATIC_SCALE={e2m1_static_scale_env} \
                 e2m1_kmajor=ATLAS_E2M1_KMAJOR={e2m1_kmajor_requested} \
                 e2m1_kmajor_m256=ATLAS_E2M1_KMAJOR_M256={e2m1_kmajor_m256_requested} \
                 e2m1_silu_quant=ATLAS_E2M1_SILU_QUANT={e2m1_silu_quant_requested} \
                 e2m1_kmajor_kernel={has_e2m1_kmajor} \
                 e2m1_kmajor_m256_kernel={has_e2m1_kmajor_m256} \
                 gate=ATLAS_PREFILL_FFN_FAST={gate_on} \
                 fp8_gate=ATLAS_FFN_PREDEQUANT_FP8={fp8_gate_on} \
                 v2_gate=ATLAS_FFN_M128_V2={v2_gate_on}"
            );
        }

        if e2m1_fast_path {
            let _scratch_use = self.lock_e2m1_use(stream)?;
            // Native W4A4 NVFP4×NVFP4 dispatch: prequant activations to
            // NVFP4 in-place, then issue the CUTLASS-style E2M1×E2M1 MMA
            // GEMM. Uses the standard `[N, K/2]` HuggingFace weight
            // layout (NOT the `nvfp4_t` transposed layout) — matches the
            // kernel's coalesced gmem read pattern.
            //
            // gate_proj and up_proj share the exact same [M, H] BF16 input.
            // Prepare it once: the old path repeated an identical global
            // absmax, D2H synchronization/readback, and quantization before
            // each projection even though both consumed identical bytes.
            let gate_up_scale = self.e2m1_checkpoint_scales.map(|scales| scales.gate_up);
            let gate_weight = if e2m1_kmajor_requested {
                self.gate_proj_t.as_ref().unwrap()
            } else {
                &self.weights.gate_proj
            };
            let up_weight = if e2m1_kmajor_requested {
                self.up_proj_t.as_ref().unwrap()
            } else {
                &self.weights.up_proj
            };
            let down_weight = if e2m1_kmajor_requested {
                self.down_proj_t.as_ref().unwrap()
            } else {
                &self.weights.down_proj
            };
            let (a_packed, a_scale, a_scale2) =
                self.prepare_e2m1_input(ctx, input, m, h, gate_up_scale, stream)?;
            self.forward_e2m1_prepared(
                ctx,
                a_packed,
                a_scale,
                a_scale2,
                gate_weight,
                gate_out,
                m,
                inter,
                h,
                e2m1_kmajor_requested,
                stream,
            )?;
            self.forward_e2m1_prepared(
                ctx,
                a_packed,
                a_scale,
                a_scale2,
                up_weight,
                up_out,
                m,
                inter,
                h,
                e2m1_kmajor_requested,
                stream,
            )?;
            // down_proj: [M, inter] BF16 → [M, H] BF16
            let output = ctx.buffers.moe_output();
            if e2m1_silu_quant_requested {
                let down_scale = self.e2m1_checkpoint_scales.unwrap().down;
                let (a_packed, a_scale, a_scale2) = self
                    .prepare_e2m1_silu_input(ctx, gate_out, up_out, m, inter, down_scale, stream)?;
                self.forward_e2m1_prepared(
                    ctx,
                    a_packed,
                    a_scale,
                    a_scale2,
                    down_weight,
                    output,
                    m,
                    h,
                    inter,
                    e2m1_kmajor_requested,
                    stream,
                )?;
            } else {
                // SiLU/GELU(gate) * up for all M tokens
                ops::silu_mul(
                    ctx.gpu,
                    self.act_mul,
                    gate_out,
                    up_out,
                    gate_out,
                    m * inter,
                    stream,
                )?;
                self.forward_e2m1_proj(
                    ctx,
                    gate_out,
                    down_weight,
                    output,
                    m,
                    h,
                    inter,
                    self.e2m1_checkpoint_scales.map(|scales| scales.down),
                    e2m1_kmajor_requested,
                    stream,
                )?;
            }
            return Ok(());
        }

        if e2m1_down_only_path {
            let _scratch_use = self.lock_e2m1_use(stream)?;
            // gate/up stay on the w4a16 m128 fast path; only down_proj
            // routes through E2M1 hardware MMA. Matches the shape table
            // documented at `prefill_ffn_e2m1_down_only_enabled` — net
            // ~30% down_proj savings, no gate/up regression vs the
            // standard `fast_path`.
            let gt = self.gate_proj_t.as_ref().unwrap();
            let ut = self.up_proj_t.as_ref().unwrap();
            ops::w4a16_gemm_n128_m128(
                ctx.gpu,
                self.w4a16_gemm_t_m128,
                input,
                gt,
                gate_out,
                m,
                inter,
                h,
                stream,
            )?;
            ops::w4a16_gemm_n128_m128(
                ctx.gpu,
                self.w4a16_gemm_t_m128,
                input,
                ut,
                up_out,
                m,
                inter,
                h,
                stream,
            )?;
            let output = ctx.buffers.moe_output();
            let down_weight = if e2m1_kmajor_requested {
                self.down_proj_t.as_ref().unwrap()
            } else {
                &self.weights.down_proj
            };
            if e2m1_silu_quant_requested {
                let down_scale = self.e2m1_checkpoint_scales.unwrap().down;
                let (a_packed, a_scale, a_scale2) = self
                    .prepare_e2m1_silu_input(ctx, gate_out, up_out, m, inter, down_scale, stream)?;
                self.forward_e2m1_prepared(
                    ctx,
                    a_packed,
                    a_scale,
                    a_scale2,
                    down_weight,
                    output,
                    m,
                    h,
                    inter,
                    e2m1_kmajor_requested,
                    stream,
                )?;
            } else {
                ops::silu_mul(
                    ctx.gpu,
                    self.act_mul,
                    gate_out,
                    up_out,
                    gate_out,
                    m * inter,
                    stream,
                )?;
                self.forward_e2m1_proj(
                    ctx,
                    gate_out,
                    down_weight,
                    output,
                    m,
                    h,
                    inter,
                    self.e2m1_checkpoint_scales.map(|scales| scales.down),
                    e2m1_kmajor_requested,
                    stream,
                )?;
            }
            return Ok(());
        }

        if v2_fast_path {
            let gt = self.gate_proj_t.as_ref().unwrap();
            let ut = self.up_proj_t.as_ref().unwrap();
            let dt = self.down_proj_t.as_ref().unwrap();
            // gate_proj GEMM via 8-warp v2 kernel
            ops::w4a16_gemm_n128_m128_v2(
                ctx.gpu,
                self.w4a16_gemm_t_m128_v2,
                input,
                gt,
                gate_out,
                m,
                inter,
                h,
                stream,
            )?;
            ops::w4a16_gemm_n128_m128_v2(
                ctx.gpu,
                self.w4a16_gemm_t_m128_v2,
                input,
                ut,
                up_out,
                m,
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
                m * inter,
                stream,
            )?;
            let output = ctx.buffers.moe_output();
            ops::w4a16_gemm_n128_m128_v2(
                ctx.gpu,
                self.w4a16_gemm_t_m128_v2,
                gate_out,
                dt,
                output,
                m,
                h,
                inter,
                stream,
            )?;
            return Ok(());
        }

        if fp8_fast_path {
            // Pre-dequanted FP8 weights — single sync per K-step in
            // the kernel, no DEQUANT phase. Same A×B GEMM math, just
            // bypassing the inner-loop NVFP4 → FP8 dequant.
            let gfp8 = self.gate_fp8.unwrap();
            let ufp8 = self.up_fp8.unwrap();
            let dfp8 = self.down_fp8.unwrap();

            // gate_proj GEMM: [M, H] BF16 × [inter, H] FP8 → [M, inter] BF16
            ops::fp8_gemm_n128_m128(
                ctx.gpu,
                self.fp8_gemm_t_m128_k,
                input,
                gfp8,
                gate_out,
                m,
                inter,
                h,
                stream,
            )?;
            ops::fp8_gemm_n128_m128(
                ctx.gpu,
                self.fp8_gemm_t_m128_k,
                input,
                ufp8,
                up_out,
                m,
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
                m * inter,
                stream,
            )?;
            let output = ctx.buffers.moe_output();
            ops::fp8_gemm_n128_m128(
                ctx.gpu,
                self.fp8_gemm_t_m128_k,
                gate_out,
                dfp8,
                output,
                m,
                h,
                inter,
                stream,
            )?;
            return Ok(());
        }

        if fast_path {
            let gt = self.gate_proj_t.as_ref().unwrap();
            let ut = self.up_proj_t.as_ref().unwrap();
            let dt = self.down_proj_t.as_ref().unwrap();

            // gate_proj GEMM: [M, H] → [M, inter]
            ops::w4a16_gemm_n128_m128(
                ctx.gpu,
                self.w4a16_gemm_t_m128,
                input,
                gt,
                gate_out,
                m,
                inter,
                h,
                stream,
            )?;
            // up_proj GEMM: [M, H] → [M, inter]
            ops::w4a16_gemm_n128_m128(
                ctx.gpu,
                self.w4a16_gemm_t_m128,
                input,
                ut,
                up_out,
                m,
                inter,
                h,
                stream,
            )?;
            // activation(gate) * up for all M tokens (SiLU or GELU)
            ops::silu_mul(
                ctx.gpu,
                self.act_mul,
                gate_out,
                up_out,
                gate_out,
                m * inter,
                stream,
            )?;
            // down_proj GEMM: [M, inter] → [M, H]
            let output = ctx.buffers.moe_output();
            ops::w4a16_gemm_n128_m128(
                ctx.gpu,
                self.w4a16_gemm_t_m128,
                gate_out,
                dt,
                output,
                m,
                h,
                inter,
                stream,
            )?;
            log_prefill_ffn_fast_route_once(m, h, inter);
            return Ok(());
        }

        // Byte-exact cp.async pipelined path (ATLAS_PREFILL_FFN_PIPE=1):
        // `w4a16_gemm_pipe` is a load-pipelined byte-exact shadow of the
        // baseline M_TILE=64 kernel below (same B dequant arithmetic, same
        // m16n8k16 MMA sequence/accumulation order) — only the weight-load
        // pipeline differs. Small-M prefills pay a latency-bound fixed cost
        // per layer on the baseline kernel (~8ms/layer at M=18); the pipe
        // kernel overlaps the stream and should land near the ~190 GB/s the
        // verify-path kernels achieve. Default off; an eligible explicit
        // request with handle 0 fails before launching the gate projection.
        // `w4a16_gemm_pipe` is byte-exact ONLY when the K reduction is a
        // multiple of its 64-row stage; a partial final stage would load past
        // the row's packed-weight boundary and silently corrupt output. The
        // FFN shapes (h=5120, inter=17408) satisfy this, but guard explicitly
        // so any future topology change falls back to the exact baseline.
        let pipe_requested = crate::layers::prefill_ffn_pipe_enabled();
        let pipe_shape_eligible = h.is_multiple_of(64) && inter.is_multiple_of(64);
        if pipe_requested && pipe_shape_eligible && self.w4a16_gemm_pipe.0 == 0 {
            bail!(
                "ATLAS_PREFILL_FFN_PIPE requested for an eligible shape but w4a16_gemm_pipe is missing"
            );
        }
        let pipe_ready = pipe_requested && pipe_shape_eligible && self.w4a16_gemm_pipe.0 != 0;
        let dual_route = prefill_fused_epilogue_route(
            crate::layers::prefill_ffn_dual_fused_enabled()?,
            pipe_ready,
            self.w4a16_gemm_pipe_dual.0 != 0,
            self.activation,
            m,
            h,
            inter,
        );
        if dual_route == PrefillFusedEpilogueRoute::Missing {
            bail!(
                "ATLAS_PREFILL_FFN_DUAL_FUSED requested for an eligible shape but its parent pipe or dual kernel is missing"
            );
        }
        let dual_fused = dual_route == PrefillFusedEpilogueRoute::Complete;
        let fused_route = if dual_fused {
            // Dual is authoritative when both explicit candidate flags are set.
            PrefillFusedEpilogueRoute::Disabled
        } else {
            prefill_fused_epilogue_route(
                crate::layers::prefill_ffn_fused_epilogue_enabled(),
                pipe_ready,
                self.w4a16_gemm_pipe_silu_mul.0 != 0,
                self.activation,
                m,
                h,
                inter,
            )
        };
        if fused_route == PrefillFusedEpilogueRoute::Missing {
            bail!(
                "ATLAS_PREFILL_FFN_FUSED_EPILOGUE requested for an eligible shape but its parent pipe or fused kernel is missing"
            );
        }
        let fused_epilogue = fused_route == PrefillFusedEpilogueRoute::Complete;

        if pipe_ready {
            if !dual_fused && fused_epilogue {
                log_prefill_up_fused_route_once();
            } else if !dual_fused {
                log_prefill_pipe_route_once();
            }
            if dual_fused {
                ops::w4a16_gemm_pipe_dual(
                    ctx.gpu,
                    self.w4a16_gemm_pipe_dual,
                    input,
                    &self.weights.gate_proj,
                    &self.weights.up_proj,
                    gate_out,
                    DevicePtr::NULL,
                    true,
                    m,
                    inter,
                    h,
                    stream,
                )?;
                log_prefill_dual_fused_route_once();
            } else {
                ops::w4a16_gemm_pipe(
                    ctx.gpu,
                    self.w4a16_gemm_pipe,
                    input,
                    &self.weights.gate_proj,
                    gate_out,
                    m,
                    inter,
                    h,
                    stream,
                )?;
            }
            if !dual_fused && fused_epilogue {
                ops::w4a16_gemm_pipe_silu_mul(
                    ctx.gpu,
                    self.w4a16_gemm_pipe_silu_mul,
                    input,
                    &self.weights.up_proj,
                    gate_out,
                    m,
                    inter,
                    h,
                    stream,
                )?;
            } else if !dual_fused {
                ops::w4a16_gemm_pipe(
                    ctx.gpu,
                    self.w4a16_gemm_pipe,
                    input,
                    &self.weights.up_proj,
                    up_out,
                    m,
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
                    m * inter,
                    stream,
                )?;
            }
            let output = ctx.buffers.moe_output();
            ops::w4a16_gemm_pipe(
                ctx.gpu,
                self.w4a16_gemm_pipe,
                gate_out,
                &self.weights.down_proj,
                output,
                m,
                h,
                inter,
                stream,
            )?;
            return Ok(());
        }

        // Baseline M_TILE=64 path (unchanged).
        // gate_proj GEMM: [M, H] → [M, inter]
        ops::w4a16_gemm(
            ctx.gpu,
            self.w4a16_gemm,
            input,
            &self.weights.gate_proj,
            gate_out,
            m,
            inter,
            h,
            stream,
        )?;

        // up_proj GEMM: [M, H] → [M, inter]
        ops::w4a16_gemm(
            ctx.gpu,
            self.w4a16_gemm,
            input,
            &self.weights.up_proj,
            up_out,
            m,
            inter,
            h,
            stream,
        )?;

        // activation(gate) * up for all M tokens (SiLU or GELU)
        ops::silu_mul(
            ctx.gpu,
            self.act_mul,
            gate_out,
            up_out,
            gate_out,
            m * inter,
            stream,
        )?;

        // down_proj GEMM: [M, inter] → [M, H]
        let output = ctx.buffers.moe_output();
        ops::w4a16_gemm(
            ctx.gpu,
            self.w4a16_gemm,
            gate_out,
            &self.weights.down_proj,
            output,
            m,
            h,
            inter,
            stream,
        )?;

        Ok(())
    }

    /// Batched forward (delegates to `forward_prefill`). Used by `forward_batched` in model loop.
    pub fn forward_batched(
        &self,
        input: DevicePtr,
        num_tokens: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        self.forward_prefill(input, num_tokens, ctx, stream)
    }
}
