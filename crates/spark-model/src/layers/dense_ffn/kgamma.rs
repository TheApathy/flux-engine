// SPDX-License-Identifier: AGPL-3.0-only

//! Speculative K-gamma / multi-sequence batched verification forward passes for `DenseFfnLayer`.

use anyhow::{Context, Result};
use spark_runtime::gpu::DevicePtr;

use super::layer::{DenseFfnLayer, FfnActivation};
#[cfg(feature = "cuda")]
use super::policy::{CUBLASLT_NVFP4_MIN_M, kgamma_cublaslt_opt_in};
use super::policy::{
    ExactFfnDispatch, ExactFfnMaterializedAvailability, ExactFfnMaterializedRoute,
    ExactFfnPhysicalRoute, FFN_KGAMMA_AUTO_TC_SM_THRESHOLD, FFN_KGAMMA_WIDE_MAX_ROWS, FfnTcPolicy,
    exact_ffn_auto_kgamma_applicable, exact_ffn_dispatch, exact_ffn_lowreg_gate_up_m16_enabled,
    exact_ffn_materialized_route, exact_ffn_physical_route_with_policy, exact_ffn_row_offsets,
    exact_ffn_split_m8_enabled, exact_ffn_tc_override, ffn_tc_active_for_rows, ffn_tc_policy,
    kgamma_fused_gateup_active, kgamma_m32_allowed, kgamma_m128_active, kgamma_wide_active,
    w3_kgamma_applicable,
};
use super::sm120_coalesced::{
    sm120_coalesced_m16_env_enabled, sm120_coalesced_max_m, try_coalesced_nvfp4_sm120_small_m,
    try_coalesced_nvfp4_sm120_small_m_gateup_silu,
};
use crate::layer::ForwardContext;
use crate::layers::ops;

impl DenseFfnLayer {
    fn measure_exact_kgamma_inputs(
        &self,
        input: DevicePtr,
        gate_out: DevicePtr,
        up_out: DevicePtr,
        ctx: &ForwardContext,
        hidden: u32,
        intermediate: u32,
        stream: u64,
    ) -> Result<()> {
        if !self.sparsity_measure_active() || ctx.graph_capture {
            return Ok(());
        }

        let (hist_g, count_g, hist_d, count_d, meas_silu) =
            self.ensure_sparsity_meas(ctx.gpu, intermediate as usize)?;
        self.measure_sparsity_site(ctx, input, hist_g, count_g, hidden, stream)?;
        ops::silu_mul(
            ctx.gpu,
            self.act_mul,
            gate_out,
            up_out,
            meas_silu,
            intermediate,
            stream,
        )?;
        self.measure_sparsity_site(ctx, meas_silu, hist_d, count_d, intermediate, stream)?;
        self.maybe_dump_sparsity(ctx, "dense_ffn_kgamma_exact", stream)
    }

    fn forward_kgamma_exact_split_m8_down(
        &self,
        ctx: &ForwardContext,
        hidden: u32,
        intermediate: u32,
        stream: u64,
    ) -> Result<()> {
        const GROUP_ROWS: u32 = 8;
        let preactivation = ctx.buffers.ssm_conv_out_f32();
        let output = ctx.buffers.moe_output();

        for first_row in [0, GROUP_ROWS] {
            let offsets = exact_ffn_row_offsets(first_row, hidden, intermediate)
                .ok_or_else(|| anyhow::anyhow!("exact FFN M8 down row-offset overflow"))?;
            ops::w4a16_gemv_f32_input_exact(
                ctx.gpu,
                self.exact_ffn_kernels,
                preactivation.offset(offsets.preactivation),
                &self.weights.down_proj,
                output.offset(offsets.output),
                GROUP_ROWS,
                hidden,
                intermediate,
                stream,
            )?;
        }
        Ok(())
    }

    fn forward_kgamma_exact_lowreg_gate_up_m16(
        &self,
        input: DevicePtr,
        ctx: &ForwardContext,
        hidden: u32,
        intermediate: u32,
        stream: u64,
    ) -> Result<()> {
        const ROWS: u32 = 16;
        let gate_out = ctx.buffers.expert_gate_out();
        let up_out = ctx.buffers.expert_up_out();
        let preactivation = ctx.buffers.ssm_conv_out_f32();

        ops::w4a16_gemv_gate_exact_m16_lowreg(
            ctx.gpu,
            self.exact_ffn_lowreg_m16,
            input,
            &self.weights.gate_proj,
            gate_out,
            ROWS,
            intermediate,
            hidden,
            stream,
        )?;
        ops::w4a16_gemv_up_exact_m16_lowreg(
            ctx.gpu,
            self.exact_ffn_lowreg_m16,
            input,
            &self.weights.up_proj,
            up_out,
            ROWS,
            intermediate,
            hidden,
            stream,
        )?;
        self.measure_exact_kgamma_inputs(
            input,
            gate_out,
            up_out,
            ctx,
            hidden,
            intermediate,
            stream,
        )?;
        ops::w4a16_gate_up_materialize_f32_m16(
            ctx.gpu,
            self.exact_ffn_lowreg_m16,
            gate_out,
            up_out,
            preactivation,
            ROWS,
            intermediate,
            stream,
        )?;
        self.forward_kgamma_exact_split_m8_down(ctx, hidden, intermediate, stream)
    }

    fn forward_kgamma_exact_split_m8(
        &self,
        input: DevicePtr,
        ctx: &ForwardContext,
        hidden: u32,
        intermediate: u32,
        stream: u64,
    ) -> Result<()> {
        const GROUP_ROWS: u32 = 8;
        if ops::exact_ffn_lowreg_m16_route(
            2 * GROUP_ROWS,
            exact_ffn_lowreg_gate_up_m16_enabled(),
            self.exact_ffn_lowreg_m16,
        ) == Some(ops::ExactFfnLowregM16Route::Lowreg)
        {
            return self.forward_kgamma_exact_lowreg_gate_up_m16(
                input,
                ctx,
                hidden,
                intermediate,
                stream,
            );
        }
        let gate_out = ctx.buffers.expert_gate_out();
        let up_out = ctx.buffers.expert_up_out();
        let preactivation = ctx.buffers.ssm_conv_out_f32();
        let output = ctx.buffers.moe_output();

        for first_row in [0, GROUP_ROWS] {
            let offsets = exact_ffn_row_offsets(first_row, hidden, intermediate)
                .ok_or_else(|| anyhow::anyhow!("exact FFN M8 row-offset overflow"))?;
            let input_group = input.offset(offsets.input);
            let gate_group = gate_out.offset(offsets.gate_up);
            let up_group = up_out.offset(offsets.gate_up);
            let preactivation_group = preactivation.offset(offsets.preactivation);
            let output_group = output.offset(offsets.output);

            ops::w4a16_gemv_dual_exact(
                ctx.gpu,
                self.exact_ffn_kernels,
                input_group,
                &self.weights.gate_proj,
                gate_group,
                &self.weights.up_proj,
                up_group,
                GROUP_ROWS,
                intermediate,
                hidden,
                stream,
            )?;
            self.measure_exact_kgamma_inputs(
                input_group,
                gate_group,
                up_group,
                ctx,
                hidden,
                intermediate,
                stream,
            )?;
            ops::w4a16_gemv_dual_silu_f32_exact(
                ctx.gpu,
                self.exact_ffn_kernels,
                gate_group,
                up_group,
                preactivation_group,
                GROUP_ROWS,
                intermediate,
                stream,
            )?;
            ops::w4a16_gemv_f32_input_exact(
                ctx.gpu,
                self.exact_ffn_kernels,
                preactivation_group,
                &self.weights.down_proj,
                output_group,
                GROUP_ROWS,
                hidden,
                intermediate,
                stream,
            )?;
        }
        Ok(())
    }

    fn forward_kgamma_exact(
        &self,
        input: DevicePtr,
        ctx: &ForwardContext,
        rows: u32,
        stream: u64,
    ) -> Result<()> {
        let hidden = ctx.config.hidden_size as u32;
        let intermediate = ctx.config.intermediate_size as u32;
        let gate_out = ctx.buffers.expert_gate_out();
        let up_out = ctx.buffers.expert_up_out();

        if exact_ffn_physical_route_with_policy(
            rows,
            intermediate,
            exact_ffn_split_m8_enabled(),
            self.exact_ffn_kernels
                .tier_is_complete(ops::ExactFfnTier::M8),
            self.exact_ffn_kernels.materialized_m8_is_complete(),
            ctx.buffers.sizes().ssm_conv_out_f32,
            crate::layers::deterministic_batch::deterministic_batch_policy_or_disabled(),
        ) == ExactFfnPhysicalRoute::SplitM8x2
        {
            return self.forward_kgamma_exact_split_m8(input, ctx, hidden, intermediate, stream);
        }

        // Attention/SSM has completed before FFN on this stream, so its FP32
        // recurrence workspace is dead until the next layer. Reuse it only
        // when the arena's authoritative byte capacity covers every row.
        let route = exact_ffn_materialized_route(
            rows,
            intermediate,
            ExactFfnMaterializedAvailability {
                m8_dual_silu: self.exact_ffn_kernels.dual_silu_f32_m8_is_present(),
                m8_f32_down: self.exact_ffn_kernels.f32_input_m8_is_present(),
                m17_dual_silu: self.exact_ffn_kernels.dual_silu_f32_m17_is_present(),
                m17_f32_down: self.exact_ffn_kernels.f32_input_m17_is_present(),
                // The fused stage does not expose gate/up tensors required by
                // sparsity diagnostics; retain the split exact route then.
                m17_fused_dual_silu: self.exact_ffn_kernels.fused_materialized_m17_is_present()
                    && (!self.sparsity_measure_active() || ctx.graph_capture),
            },
            ctx.buffers.sizes().ssm_conv_out_f32,
        );

        let preactivation_f32 = ctx.buffers.ssm_conv_out_f32();
        match route {
            ExactFfnMaterializedRoute::FusedM17 => {
                ops::w4a16_gemv_dual_materialize_f32_exact_m17(
                    ctx.gpu,
                    self.exact_ffn_kernels,
                    input,
                    &self.weights.gate_proj,
                    &self.weights.up_proj,
                    preactivation_f32,
                    rows,
                    intermediate,
                    hidden,
                    stream,
                )?;
            }
            ExactFfnMaterializedRoute::Split | ExactFfnMaterializedRoute::Inline => {
                ops::w4a16_gemv_dual_exact(
                    ctx.gpu,
                    self.exact_ffn_kernels,
                    input,
                    &self.weights.gate_proj,
                    gate_out,
                    &self.weights.up_proj,
                    up_out,
                    rows,
                    intermediate,
                    hidden,
                    stream,
                )?;
                self.measure_exact_kgamma_inputs(
                    input,
                    gate_out,
                    up_out,
                    ctx,
                    hidden,
                    intermediate,
                    stream,
                )?;
            }
        }

        match route {
            ExactFfnMaterializedRoute::Split => ops::w4a16_gemv_dual_silu_f32_exact(
                ctx.gpu,
                self.exact_ffn_kernels,
                gate_out,
                up_out,
                preactivation_f32,
                rows,
                intermediate,
                stream,
            )?,
            ExactFfnMaterializedRoute::Inline => {
                return ops::w4a16_gemv_silu_input_exact(
                    ctx.gpu,
                    self.exact_ffn_kernels,
                    gate_out,
                    up_out,
                    &self.weights.down_proj,
                    ctx.buffers.moe_output(),
                    rows,
                    hidden,
                    intermediate,
                    stream,
                );
            }
            ExactFfnMaterializedRoute::FusedM17 => {}
        }
        ops::w4a16_gemv_f32_input_exact(
            ctx.gpu,
            self.exact_ffn_kernels,
            preactivation_f32,
            &self.weights.down_proj,
            ctx.buffers.moe_output(),
            rows,
            hidden,
            intermediate,
            stream,
        )
    }

    fn forward_kgamma_k1_rows(
        &self,
        input: DevicePtr,
        ctx: &ForwardContext,
        rows: u32,
        stream: u64,
    ) -> Result<()> {
        let hidden = ctx.config.hidden_size as u32;
        let intermediate = ctx.config.intermediate_size as u32;
        let gate_out = ctx.buffers.expert_gate_out();
        let up_out = ctx.buffers.expert_up_out();
        let output = ctx.buffers.moe_output();

        for row in 0..rows as usize {
            let input_row = input.offset(row * hidden as usize * 2);
            let gate_row = gate_out.offset(row * intermediate as usize * 2);
            let up_row = up_out.offset(row * intermediate as usize * 2);
            let output_row = output.offset(row * hidden as usize * 2);
            ops::w4a16_decode_gemv_dual(
                ctx.gpu,
                self.w4a16_gemv_dual,
                self.w4a16_gemv_dual_sw,
                self.gemv_sw,
                input_row,
                &self.weights.gate_proj,
                gate_row,
                &self.weights.up_proj,
                up_row,
                intermediate,
                hidden,
                stream,
            )?;
            ops::w4a16_decode_gemv_silu_input(
                ctx.gpu,
                self.w4a16_gemv_silu_input,
                self.w4a16_gemv_silu_input_sw,
                self.gemv_sw,
                gate_row,
                up_row,
                &self.weights.down_proj,
                output_row,
                hidden,
                intermediate,
                stream,
            )?;
        }

        self.measure_exact_kgamma_inputs(input, gate_out, up_out, ctx, hidden, intermediate, stream)
    }

    /// K=γ verify batch (DFlash γ ≥ 16, typical n=17 with γ=16).
    ///
    /// Replaces the per-token loop that calls `forward()` n times (n=γ+1).
    /// Each `forward()` call runs 2 M=1 GEMVs that re-read 134 MB of NVFP4
    /// FFN weights from LPDDR5X; per-step this costs `64 layers × 17 tokens
    /// × 134 MB = 145 GB` of redundant weight bandwidth.
    ///
    /// This path issues 3 `w4a16_gemm` calls with M=n. The standard
    /// (non-transposed) NVFP4 GEMM has M_TILE=64; M=17 fits inside a single
    /// CTA-row with some accumulator waste but loads the weight tile once
    /// per layer instead of n times. Expected: ~64 × 134 MB = 8.6 GB per
    /// step, an ~18× reduction in FFN-loop bandwidth.
    ///
    /// Sequence: gate_proj (GEMM M=n) → up_proj (GEMM M=n) → silu_mul
    /// (n × intermediate) → down_proj (GEMM M=n).
    /// W4+SiLU rows 2..=32 instead use exact dynamic-M dual and fused-down
    /// GEMVs that preserve K1 accumulation order. If either selected-tier
    /// symbol is missing, every row runs through the ordinary K1 kernels;
    /// BF16, W3, GELU, and rows above 32 retain their existing routes.
    ///
    /// Reuses `ctx.buffers.expert_gate_out` / `expert_up_out` /
    /// `moe_output`, which are sized for `max_batch_tokens × intermediate`
    /// (always ≥ n, see `buffers/sizes.rs`).
    ///
    /// Output is written to `ctx.buffers.moe_output()`; callers downstream
    /// already consume from there (see `ms_phase_ffn` for the n==3 branch
    /// which is the contract this matches at higher n).
    ///
    /// When `ATLAS_FFN_M16_TRANSPOSED` is active (default ON unless `=0`) and
    /// the loader installed transposed (`nvfp4_t`) FFN weights via
    /// `set_transposed_weights`, this path dispatches gate/up/down through
    /// `w4a16_gemm_t_m16` (M_TILE=16, near-zero MMA accumulator waste at
    /// M ≤ 32). Otherwise the
    /// non-transposed `w4a16_gemm` (M_TILE=64) is used as the fallback —
    /// layout-compatible with the standard HuggingFace `[N, K/2]` weights
    /// but discards ~73% of accumulator writes at M=17. The transposed
    /// fast path is the K=γ verify analogue of the existing SSM `qkvz` /
    /// `out_proj` and DFlash drafter routings.
    /// BF16-weight fallback is not supported on this path (Gemma-4 dense
    /// has no MTP / DFlash, so n is always 1 there).
    /// Opt-in (`ATLAS_W4A4_DECODE_SK3=1`, default OFF) 48-SM roofline W4A4
    /// decode/verify GEMM route (`w4a4_gemm_m16_sk3_sm120` +
    /// `reduce_w4a4_sk3_bf16_sm120`) for `M in 1..=16`.
    ///
    /// Preserves bit-exact `W=1 == W=8 == W=16` and `B=1 == B=4/8` invariance
    /// by using row-independent activation scaling (checkpoint scale or `1.0`,
    /// never batch-dependent absmax) and an `M`-independent `split_k` factor
    /// (`3` on `<= 64`-SM GB10). Returns `Ok(false)` immediately when disabled
    /// or when any precondition is not met, leaving the baseline 100% unchanged.
    #[allow(clippy::too_many_arguments)]
    fn try_forward_kgamma_w4a4_sk3(
        &self,
        input: DevicePtr,
        ctx: &ForwardContext,
        n: u32,
        h: u32,
        inter: u32,
        gate_out: DevicePtr,
        up_out: DevicePtr,
        stream: u64,
    ) -> Result<bool> {
        let output = ctx.buffers.moe_output();
        if crate::layers::deterministic_batch::deterministic_batch_enabled()
            || !ops::w4a4_decode_sk3_enabled()
            || n == 0
            || n > ops::W4A4_DECODE_SK3_MAX_M
            || self.bf16_weights.is_some()
            || self.has_w3_gemm()
            || self.activation != FfnActivation::SiLU
            || h < ops::W4A4_DECODE_SK3_K_STEP
            || !h.is_multiple_of(ops::W4A4_DECODE_SK3_K_STEP)
            || inter < ops::W4A4_DECODE_SK3_K_STEP
            || !inter.is_multiple_of(ops::W4A4_DECODE_SK3_K_STEP)
            || !ops::w4a4_decode_sk3_ffn_preflight_ok(
                &self.weights.gate_proj,
                &self.weights.up_proj,
                &self.weights.down_proj,
                self.act_mul,
                self.nvfp4_quantize_k,
                input,
                gate_out,
                up_out,
                output,
            )
        {
            return Ok(false);
        }

        let a_scale2_gu = self
            .e2m1_checkpoint_scales
            .map(|s| s.gate_up)
            .filter(|s| s.is_finite() && *s > 0.0)
            .unwrap_or(1.0);
        let a_scale2_down = self
            .e2m1_checkpoint_scales
            .map(|s| s.down)
            .filter(|s| s.is_finite() && *s > 0.0)
            .unwrap_or(1.0);
        let scale2_gate = a_scale2_gu * self.weights.gate_proj.weight_scale_2;
        let scale2_up = a_scale2_gu * self.weights.up_proj.weight_scale_2;
        let scale2_down = a_scale2_down * self.weights.down_proj.weight_scale_2;
        if !(scale2_gate.is_finite()
            && scale2_gate > 0.0
            && scale2_up.is_finite()
            && scale2_up > 0.0
            && scale2_down.is_finite()
            && scale2_down > 0.0)
        {
            return Ok(false);
        }

        let kernels = ops::cached_w4a4_decode_sk3_kernels(ctx.gpu);
        if !kernels.is_ready() {
            return Ok(false);
        }

        let sm_count = atlas_core::device::active_sm_count();
        let split_k_gu = ops::default_w4a4_decode_split_k(sm_count, h);
        let split_k_down = ops::default_w4a4_decode_split_k(sm_count, inter);
        let plan_gu = ops::plan_w4a4_decode_sk3(n, inter, h, split_k_gu)?;
        let plan_down = ops::plan_w4a4_decode_sk3(n, h, inter, split_k_down)?;

        let capturing = ctx.graph_capture || ctx.gpu.stream_is_capturing(stream);
        let req_ws_bytes = plan_gu.workspace_bytes.max(plan_down.workspace_bytes);
        let (ws_ptr, ws_bytes) = if req_ws_bytes == 0 {
            (DevicePtr::NULL, 0)
        } else if ctx.buffers.sizes().ssm_conv_out_f32 >= req_ws_bytes
            && !ctx.buffers.ssm_conv_out_f32().is_null()
            && ctx.buffers.ssm_conv_out_f32().0.is_multiple_of(4)
        {
            (
                ctx.buffers.ssm_conv_out_f32(),
                ctx.buffers.sizes().ssm_conv_out_f32,
            )
        } else {
            let alloc_policy = if capturing {
                ops::W4a4WorkspaceAllocPolicy::CaptureNoAlloc
            } else {
                ops::W4a4WorkspaceAllocPolicy::AllowAlloc
            };
            let Some(ws) =
                ops::ensure_w4a4_decode_sk3_workspace(ctx.gpu, stream, req_ws_bytes, alloc_policy)?
            else {
                return Ok(false);
            };
            ws
        };
        if req_ws_bytes > 0
            && (ws_ptr.is_null() || !ws_ptr.0.is_multiple_of(4) || ws_bytes < req_ws_bytes)
        {
            return Ok(false);
        }

        let _e2m1_stream_guard = self.lock_e2m1_use(stream)?;
        let req_k = (h.max(inter)) as usize;
        // Match `e2m1_scratch_layout`'s exact `cap_m = m.max(128)` floor so
        // stream capture never triggers `ensure_e2m1_scratch` reallocation/sync.
        let min_cap_m = (n as usize).max(128);
        let (a_packed, a_scale) = if capturing {
            let slot = self
                .e2m1_scratch
                .lock()
                .map_err(|_| anyhow::anyhow!("W4A4 activation scratch lock poisoned"))?;
            let Some(s) = slot
                .as_ref()
                .filter(|s| s.cap_m >= min_cap_m && s.cap_k >= req_k)
            else {
                return Ok(false);
            };
            (s.a_packed, s.a_scale)
        } else {
            let (ap, asc, _amax) = self.ensure_e2m1_scratch(ctx.gpu, n as usize, req_k)?;
            (ap, asc)
        };
        if !ops::w4a4_decode_sk3_activation_scratch_ok(a_packed, a_scale) {
            return Ok(false);
        }

        // Quantize shared [n, H] BF16 input ONCE with row-independent scale2.
        ops::quantize_bf16_to_nvfp4(
            ctx.gpu,
            self.nvfp4_quantize_k,
            input,
            a_packed,
            a_scale,
            a_scale2_gu,
            n,
            h,
            stream,
        )?;
        crate::kprof!(ctx.gpu, stream, "ffn_gate_w4a4_sk3_kgamma", {
            ops::w4a4_decode_roofline_gemm(
                ctx.gpu,
                kernels,
                &plan_gu,
                a_packed,
                a_scale,
                &self.weights.gate_proj,
                ws_ptr,
                ws_bytes,
                gate_out,
                scale2_gate,
                stream,
            )?;
            anyhow::Result::<()>::Ok(())
        })?;
        crate::kprof!(ctx.gpu, stream, "ffn_up_w4a4_sk3_kgamma", {
            ops::w4a4_decode_roofline_gemm(
                ctx.gpu,
                kernels,
                &plan_gu,
                a_packed,
                a_scale,
                &self.weights.up_proj,
                ws_ptr,
                ws_bytes,
                up_out,
                scale2_up,
                stream,
            )?;
            anyhow::Result::<()>::Ok(())
        })?;
        crate::kprof!(ctx.gpu, stream, "ffn_silu_mul_w4a4_sk3_kgamma", {
            ops::silu_mul(
                ctx.gpu,
                self.act_mul,
                gate_out,
                up_out,
                gate_out,
                n * inter,
                stream,
            )?;
            anyhow::Result::<()>::Ok(())
        })?;

        ops::quantize_bf16_to_nvfp4(
            ctx.gpu,
            self.nvfp4_quantize_k,
            gate_out,
            a_packed,
            a_scale,
            a_scale2_down,
            n,
            inter,
            stream,
        )?;
        crate::kprof!(ctx.gpu, stream, "ffn_down_w4a4_sk3_kgamma", {
            ops::w4a4_decode_roofline_gemm(
                ctx.gpu,
                kernels,
                &plan_down,
                a_packed,
                a_scale,
                &self.weights.down_proj,
                ws_ptr,
                ws_bytes,
                output,
                scale2_down,
                stream,
            )?;
            anyhow::Result::<()>::Ok(())
        })?;

        static LOGGED: std::sync::Once = std::sync::Once::new();
        LOGGED.call_once(|| {
            tracing::info!(
                rows = n,
                split_k_gu,
                split_k_down,
                sm_count,
                graph_capture = capturing,
                "FFN_KGAMMA_W4A4_DECODE_SK3_ENGAGED"
            );
        });
        Ok(true)
    }

    pub fn forward_kgamma(
        &self,
        input: DevicePtr,
        ctx: &ForwardContext,
        n: u32,
        stream: u64,
    ) -> Result<()> {
        debug_assert!(
            n > 1 || self.decode_tc_parity_route() || ops::w4a4_decode_sk3_enabled(),
            "forward_kgamma is for batched verify; use forward() at n=1"
        );
        let h = ctx.config.hidden_size as u32;
        let inter = ctx.config.intermediate_size as u32;

        let gate_out = ctx.buffers.expert_gate_out();
        let up_out = ctx.buffers.expert_up_out();

        if self.try_forward_kgamma_w4a4_sk3(input, ctx, n, h, inter, gate_out, up_out, stream)? {
            return Ok(());
        }

        let exact_w4_silu_eligible = self.exact_kgamma_internal_eligible(n);
        let exact_tier_present = matches!(
            self.exact_ffn_kernels.route_for_rows(n),
            Some(ops::ExactFfnRoute::Exact(_))
        );
        match exact_ffn_dispatch(n, exact_w4_silu_eligible, exact_tier_present) {
            ExactFfnDispatch::Batched => {
                return self.forward_kgamma_exact(input, ctx, n, stream);
            }
            ExactFfnDispatch::PerRowK1 => {
                return self.forward_kgamma_k1_rows(input, ctx, n, stream);
            }
            ExactFfnDispatch::Existing => {}
        }

        if w3_kgamma_applicable(
            n,
            self.activation == FfnActivation::SiLU,
            self.has_w3_gemm(),
        ) {
            let w3 = self
                .w3_weights_t
                .as_ref()
                .context("W3 K-gamma route selected without transposed weights")?;
            crate::kprof!(ctx.gpu, stream, "ffn_gate_w3_kgamma", {
                ops::w3a16_gemm_n64_m32(
                    ctx.gpu,
                    self.w3a16_gemm_t_m32_n64_k,
                    input,
                    &w3.gate_proj,
                    gate_out,
                    n,
                    inter,
                    h,
                    stream,
                )?;
                anyhow::Result::<()>::Ok(())
            })?;
            crate::kprof!(ctx.gpu, stream, "ffn_up_w3_kgamma", {
                ops::w3a16_gemm_n64_m32(
                    ctx.gpu,
                    self.w3a16_gemm_t_m32_n64_k,
                    input,
                    &w3.up_proj,
                    up_out,
                    n,
                    inter,
                    h,
                    stream,
                )?;
                anyhow::Result::<()>::Ok(())
            })?;
            crate::kprof!(ctx.gpu, stream, "ffn_silu_mul_w3_kgamma", {
                ops::silu_mul(
                    ctx.gpu,
                    self.act_mul,
                    gate_out,
                    up_out,
                    gate_out,
                    n * inter,
                    stream,
                )?;
                anyhow::Result::<()>::Ok(())
            })?;
            let output = ctx.buffers.moe_output();
            crate::kprof!(ctx.gpu, stream, "ffn_down_w3_kgamma", {
                ops::w3a16_gemm_n64_m32(
                    ctx.gpu,
                    self.w3a16_gemm_t_m32_n64_k,
                    gate_out,
                    &w3.down_proj,
                    output,
                    n,
                    h,
                    inter,
                    stream,
                )?;
                anyhow::Result::<()>::Ok(())
            })?;
            static LOGGED: std::sync::Once = std::sync::Once::new();
            LOGGED.call_once(|| {
                tracing::info!(
                    "ENGAGED W3 FFN K-gamma: approximate ghost/quality-gated target route"
                );
            });
            return Ok(());
        }

        // Route through M_TILE=16 + transposed weights when:
        //   - ATLAS_FFN_M16_TRANSPOSED=1 OR ATLAS_TC_NVFP4_M16=1
        //   - the loader installed transposed copies of all 3 FFN projections
        //   - the `w4a16_gemm_t_m16` kernel symbol is present (try_kernel)
        //   - n ≤ 32 (the kernel's intended small-M window)
        // The combined gate matches the SSM `qkvz` / drafter pattern.
        // WIDE window (SASS audit 2026-07-08): at c>=2 batched verify,
        // M = 17c exceeds 32 and this whole transposed family used to
        // disengage — silent fallback to legacy `w4a16_gemm` (no cp.async,
        // scalar LDG.U8, ~4x sector overfetch, issue-capped ~47% of DRAM
        // BW): the measured cause of the concurrency ceiling. Route
        // 32 < n <= 256 through `w4a16_gemm_t_m128` (y-tiled; at M=136 two
        // weight reads still beat legacy's 3 sweeps x overfetch). Requires
        // the m128 kernel — the m16 kernel's small-M window is NOT widened.
        let kgamma_m128_on = kgamma_m128_active();
        let kgamma_wide_on = kgamma_wide_active();
        let wide_m128 = n as usize > crate::layers::TC_VERIFY_MAX_ROWS
            && n <= FFN_KGAMMA_WIDE_MAX_ROWS
            && kgamma_wide_on
            && kgamma_m128_on
            && self.w4a16_gemm_t_m128.0 != 0
            && (crate::layers::ffn_m16_transposed_enabled()
                || crate::layers::tc_nvfp4_m16_enabled())
            && self.has_transposed_ffn();
        // On wide discrete Blackwell GPUs (`sm_count > 64`) for small/medium-M
        // (`n <= sm120_coalesced_max_m()`, up to 64 rows: `C=1..64` multi-seq
        // and `C=1..8` DFlash2 verify), prefer single-launch BF16xBF16 MMA
        // intra-CTA warp-split-K fused `gate+up+silu` + `down_proj` directly on
        // row-major NVFP4 weights (zero activation quantize, zero absmax/alpha
        // launches, zero transposed-weight dependency, full 188-SM 600W saturation).
        // Always skipped on GB10 (`sm_count <= 64`).
        let sm120_max_m = sm120_coalesced_max_m();
        if (n as usize) <= sm120_max_m
            && self.activation == FfnActivation::SiLU
            && !self.weights.gate_proj.is_null()
            && !self.weights.up_proj.is_null()
            && !self.weights.down_proj.is_null()
            && crate::kprof!(ctx.gpu, stream, "ffn_gateup_fused_kgamma", {
                try_coalesced_nvfp4_sm120_small_m_gateup_silu(
                    ctx.gpu,
                    input,
                    &self.weights.gate_proj,
                    &self.weights.up_proj,
                    gate_out,
                    n,
                    inter,
                    h,
                    sm120_max_m,
                    stream,
                )
            })?
        {
            let output = ctx.buffers.moe_output();
            let down_ok = crate::kprof!(ctx.gpu, stream, "ffn_down_kgamma", {
                try_coalesced_nvfp4_sm120_small_m(
                    ctx.gpu,
                    gate_out,
                    &self.weights.down_proj,
                    output,
                    n,
                    h,
                    inter,
                    sm120_max_m,
                    stream,
                )
            })?;
            if down_ok {
                return Ok(());
            }
            // Fallback if down_proj kernel symbol was absent (rare).
            ops::w4a16_gemm(
                ctx.gpu,
                self.w4a16_gemm,
                gate_out,
                &self.weights.down_proj,
                output,
                n,
                h,
                inter,
                stream,
            )?;
            return Ok(());
        }

        // Native cuBLASLt NVFP4 (`VEC16_UE4M3`) for wide M >= 32 (both eager
        // and inside captured multi-seq CUDA graphs once warmed on step 1) when
        // enabled (`ATLAS_FFN_KGAMMA_CUBLASLT=1`, auto-ON on `sm_count > 64`,
        // default-OFF on GB10) or when transposed `w4a16_gemm_t_m128` is
        // unavailable (e.g. SM103). Uses a per-(gpu, stream) decode scratch so
        // it never collides with prefill's stream-bound `e2m1_scratch`.
        #[cfg(feature = "cuda")]
        if n >= CUBLASLT_NVFP4_MIN_M
            && (kgamma_cublaslt_opt_in(ctx.gpu) || !wide_m128)
            && self.try_forward_cublaslt_nvfp4_kgamma(
                input, n, h, inter, gate_out, up_out, ctx, stream,
            )?
        {
            return Ok(());
        }
        let m16_path = (n as usize <= crate::layers::TC_VERIFY_MAX_ROWS
            && (crate::layers::ffn_m16_transposed_enabled()
                || crate::layers::tc_nvfp4_m16_enabled())
            && self.has_transposed_ffn())
            || wide_m128;
        // m128 upgrade of the m16 path: ONE M-tile at n ≤ 128 → single
        // weight read (m16 re-reads B per 16-row tile: 2× traffic at
        // n=17 on a memory-bound GEMM). See ffn_kgamma_m128_enabled.
        //
        // FAIL-CLOSED GUARD (2026-08-19): ATLAS_FFN_KGAMMA_M128=0 selects
        // the M_TILE=16-only route (`w4a16_gemm_t_m16`, 2 M-tile rows at
        // n=17). That route is NON-DETERMINISTIC in this binary — three
        // identical 1,500-token MinHeap requests returned three different
        // completions (513/1500/1500 tokens, three distinct SHAs). The
        // m128/m32 route is deterministic (verified: 3/3 identical,
        // 41.7 tok/s). Fail closed: when the transposed m16 family would
        // otherwise be the only route, force the deterministic m128/m32
        // path regardless of the flag and warn once.
        let m128_forced = m16_path && !kgamma_m128_on && self.w4a16_gemm_t_m128.0 != 0;
        if m128_forced {
            static FORCED: std::sync::Once = std::sync::Once::new();
            FORCED.call_once(|| {
                tracing::warn!(
                    "ATLAS_FFN_KGAMMA_M128=0 requests the M_TILE=16 verify FFN route, \
                     which is NON-DETERMINISTIC in this binary — forcing the \
                     deterministic m128/m32 route instead (fail-closed)."
                );
            });
        }
        let m128_path =
            (m16_path && kgamma_m128_on && self.w4a16_gemm_t_m128.0 != 0) || m128_forced;
        if crate::layers::ffn_kgamma_m128_enabled() && self.w4a16_gemm_t_m128.0 == 0 {
            static WARNED: std::sync::Once = std::sync::Once::new();
            crate::layers::warn_kernel_fallback(
                &WARNED,
                "ATLAS_FFN_KGAMMA_M128=1",
                "w4a16_gemm_t_m128",
                "the K=γ verify FFN stays on the M_TILE=64 path, and every knob \
                 downstream of m128_path (ATLAS_FFN_KGAMMA_M32, \
                 ATLAS_FFN_GATEUP_SPLITK, ATLAS_FFN_FUSED_GATEUP) is disabled with it",
            );
        }
        // m32_n64: single B read AND full SM occupancy — strictly better
        // than m128 at n ≤ 32 when the kernel symbol is present. The n<=32
        // bound keeps the WIDE window off m32/fused/split-K variants (their
        // M_TILE=32 shapes don't cover wide M).
        // ATLAS_FFN_KGAMMA_M32=0 opts out for bisection.
        let m32_path = m128_path
            && n as usize <= crate::layers::TC_VERIFY_MAX_ROWS
            && self.w4a16_gemm_t_m32_n64.0 != 0
            && kgamma_m32_allowed();

        // gate_proj GEMM: [n, H] → [n, inter]
        // Split-K [M=n, N=inter, K=h] when ATLAS_FFN_GATEUP_SPLITK is set —
        // slices K across gridDim.z into the shared FP32 workspace (lossless,
        // token-exact). Falls through to the single-slice m32_n64 path below.
        let gateup_splitk = crate::layers::ffn_gateup_splitk();
        let gateup_ws = if gateup_splitk > 0 {
            *self.splitk_workspace.lock().unwrap()
        } else {
            None
        };
        let gateup_splitk_ok = m32_path
            && gateup_splitk > 0
            && self.w4a16_gemm_t_m32_n64_splitk.0 != 0
            && self.reduce_splitk_k.0 != 0
            && gateup_ws.is_some();

        // FUSED gate+up+silu (ATLAS_FFN_FUSED_GATEUP=1 or auto on >64-SM GPUs):
        // one launch reads the shared [n,H] input once, streams both transposed
        // weights, and writes silu(gate)*up into `gate_out` (the same buffer
        // moe_silu_mul targets) — replacing the gate GEMM + up GEMM + silu_mul
        // below. Superseded by explicit ATLAS_FFN_GATEUP_SPLITK when active. Requires the m32 transposed path,
        // SiLU activation, and a fused kernel symbol; byte-exact with unfused.
        let fused_gateup_on = kgamma_fused_gateup_active();
        let fused_gateup = m32_path
            && !gateup_splitk_ok
            && self.activation == FfnActivation::SiLU
            && fused_gateup_on
            && self.w4a16_gemm_t_m32_n64_gateup_silu.0 != 0;
        if fused_gateup_on
            && crate::layers::ffn_fused_gateup_enabled()
            && self.w4a16_gemm_t_m32_n64_gateup_silu.0 == 0
        {
            static WARNED: std::sync::Once = std::sync::Once::new();
            crate::layers::warn_kernel_fallback(
                &WARNED,
                "ATLAS_FFN_FUSED_GATEUP=1",
                "w4a16_gemm_t_m32_n64_gateup_silu",
                "gate/up stay on two separate GEMMs plus a silu_mul launch",
            );
        }
        if matches!(ffn_tc_policy(), FfnTcPolicy::Auto)
            && atlas_core::device::active_sm_count() > FFN_KGAMMA_AUTO_TC_SM_THRESHOLD
            && (m32_path || wide_m128)
        {
            static SM120_TC_LOGGED: std::sync::Once = std::sync::Once::new();
            SM120_TC_LOGGED.call_once(|| {
                tracing::info!(
                    rows = n,
                    m32_path,
                    fused_gateup,
                    wide_m128,
                    graph_capture = ctx.graph_capture,
                    sm_count = atlas_core::device::active_sm_count(),
                    "FFN_KGAMMA_SM120_TC_ENGAGED"
                );
            });
        }
        if fused_gateup {
            // On wide discrete GPUs (`sm_count > 64`) at small `M <= 16`, prefer the
            // coalesced 4-way warp-split-K BF16xBF16 MMA fused gate+up+silu kernel
            // (`w4a16_gemm_coalesced_v2_n16_sk4_m16_gateup_silu`, 1,088 CTAs at
            // `N=17408`): reads A once, streams row-major 128-bit NVFP4 weights with
            // `L2::evict_first`, eliminates M_TILE=32 waste and FP8 activation clipping.
            // Always skipped on GB10 (`sm_count <= 64`).
            let coalesced_gateup_done = n <= 16
                && !self.weights.gate_proj.is_null()
                && !self.weights.up_proj.is_null()
                && crate::kprof!(ctx.gpu, stream, "ffn_gateup_fused_kgamma", {
                    try_coalesced_nvfp4_sm120_small_m_gateup_silu(
                        ctx.gpu,
                        input,
                        &self.weights.gate_proj,
                        &self.weights.up_proj,
                        gate_out,
                        n,
                        inter,
                        h,
                        16,
                        stream,
                    )
                })?;
            if coalesced_gateup_done {
                // Dispatched via SM120 BF16xBF16 MMA coalesced warp-split-K fused gate+up+silu.
            } else {
                let gt = self.gate_proj_t.as_ref().unwrap();
                let ut = self.up_proj_t.as_ref().unwrap();
                // Kernel selection priority (highest first, all 0-ULP byte-exact):
                //   ATLAS_GATEUP_N32    → n32_pipe_k64 (544 CTAs on N=17408, auto >128 SMs)
                //   ATLAS_GATEUP_K64    → n64_pipe_k64 (K_STEP=64, reg-dequant, auto >64 SMs)
                //   ATLAS_DEQUANT_PIPE  → n64_pipe     (K_STEP=32, reg-dequant, auto >64 SMs)
                //   default             → n64 staged   (smem_B_fp8 staging)
                if crate::layers::gateup_n32_enabled()
                    && self.w4a16_gemm_t_m32_n32_gateup_silu_pipe_k64.0 != 0
                    && h.is_multiple_of(64)
                    && inter.is_multiple_of(32)
                {
                    crate::kprof!(ctx.gpu, stream, "ffn_gateup_fused_kgamma", {
                        ops::w4a16_gemm_n32_m32_gateup_silu(
                            ctx.gpu,
                            self.w4a16_gemm_t_m32_n32_gateup_silu_pipe_k64,
                            input,
                            gt,
                            ut,
                            gate_out,
                            n,
                            inter,
                            h,
                            stream,
                        )?;
                        anyhow::Result::<()>::Ok(())
                    })?;
                } else {
                    let fused_kernel = if crate::layers::gateup_k64_enabled()
                        && self.w4a16_gemm_t_m32_n64_gateup_silu_pipe_k64.0 != 0
                        && h.is_multiple_of(64)
                    // K must be divisible by 64
                    {
                        self.w4a16_gemm_t_m32_n64_gateup_silu_pipe_k64
                    } else if crate::layers::dequant_pipe_enabled()
                        && self.w4a16_gemm_t_m32_n64_gateup_silu_pipe.0 != 0
                    {
                        self.w4a16_gemm_t_m32_n64_gateup_silu_pipe
                    } else {
                        self.w4a16_gemm_t_m32_n64_gateup_silu
                    };
                    crate::kprof!(ctx.gpu, stream, "ffn_gateup_fused_kgamma", {
                        ops::w4a16_gemm_n64_m32_gateup_silu(
                            ctx.gpu,
                            fused_kernel,
                            input,
                            gt,
                            ut,
                            gate_out,
                            n,
                            inter,
                            h,
                            stream,
                        )?;
                        anyhow::Result::<()>::Ok(())
                    })?;
                }
            }
        } else {
            crate::kprof!(ctx.gpu, stream, "ffn_gate_kgamma", {
                if gateup_splitk_ok {
                    let gt = self.gate_proj_t.as_ref().unwrap();
                    ops::w4a16_gemm_n64_m32_splitk(
                        ctx.gpu,
                        self.w4a16_gemm_t_m32_n64_splitk,
                        self.reduce_splitk_k,
                        input,
                        gt,
                        gate_out,
                        gateup_ws.unwrap(),
                        n,
                        inter,
                        h,
                        inter, // ldb == N for tightly-packed T-weight
                        gateup_splitk,
                        stream,
                    )?;
                } else if m32_path {
                    let gt = self.gate_proj_t.as_ref().unwrap();
                    ops::w4a16_gemm_n64_m32(
                        ctx.gpu,
                        self.w4a16_gemm_t_m32_n64,
                        input,
                        gt,
                        gate_out,
                        n,
                        inter,
                        h,
                        stream,
                    )?;
                } else if m128_path {
                    let gt = self.gate_proj_t.as_ref().unwrap();
                    ops::w4a16_gemm_n128_m128(
                        ctx.gpu,
                        self.w4a16_gemm_t_m128,
                        input,
                        gt,
                        gate_out,
                        n,
                        inter,
                        h,
                        stream,
                    )?;
                } else if m16_path {
                    let gt = self.gate_proj_t.as_ref().unwrap();
                    ops::w4a16_gemm_n128_m16(
                        ctx.gpu,
                        self.w4a16_gemm_t_m16,
                        input,
                        gt,
                        gate_out,
                        n,
                        inter,
                        h,
                        stream,
                    )?;
                } else {
                    ops::w4a16_gemm(
                        ctx.gpu,
                        self.w4a16_gemm,
                        input,
                        &self.weights.gate_proj,
                        gate_out,
                        n,
                        inter,
                        h,
                        stream,
                    )?;
                }
                anyhow::Result::<()>::Ok(())
            })?;

            // up_proj GEMM: [n, H] → [n, inter]. Same split-K treatment as gate;
            // reuses the shared workspace (gate's reduce already consumed it).
            crate::kprof!(ctx.gpu, stream, "ffn_up_kgamma", {
                if gateup_splitk_ok {
                    let ut = self.up_proj_t.as_ref().unwrap();
                    ops::w4a16_gemm_n64_m32_splitk(
                        ctx.gpu,
                        self.w4a16_gemm_t_m32_n64_splitk,
                        self.reduce_splitk_k,
                        input,
                        ut,
                        up_out,
                        gateup_ws.unwrap(),
                        n,
                        inter,
                        h,
                        inter, // ldb == N for tightly-packed T-weight
                        gateup_splitk,
                        stream,
                    )?;
                } else if m32_path {
                    let ut = self.up_proj_t.as_ref().unwrap();
                    ops::w4a16_gemm_n64_m32(
                        ctx.gpu,
                        self.w4a16_gemm_t_m32_n64,
                        input,
                        ut,
                        up_out,
                        n,
                        inter,
                        h,
                        stream,
                    )?;
                } else if m128_path {
                    let ut = self.up_proj_t.as_ref().unwrap();
                    ops::w4a16_gemm_n128_m128(
                        ctx.gpu,
                        self.w4a16_gemm_t_m128,
                        input,
                        ut,
                        up_out,
                        n,
                        inter,
                        h,
                        stream,
                    )?;
                } else if m16_path {
                    let ut = self.up_proj_t.as_ref().unwrap();
                    ops::w4a16_gemm_n128_m16(
                        ctx.gpu,
                        self.w4a16_gemm_t_m16,
                        input,
                        ut,
                        up_out,
                        n,
                        inter,
                        h,
                        stream,
                    )?;
                } else {
                    ops::w4a16_gemm(
                        ctx.gpu,
                        self.w4a16_gemm,
                        input,
                        &self.weights.up_proj,
                        up_out,
                        n,
                        inter,
                        h,
                        stream,
                    )?;
                }
                anyhow::Result::<()>::Ok(())
            })?;

            // activation(gate) * up for n tokens
            crate::kprof!(ctx.gpu, stream, "ffn_silu_mul_kgamma", {
                ops::silu_mul(
                    ctx.gpu,
                    self.act_mul,
                    gate_out,
                    up_out,
                    gate_out,
                    n * inter,
                    stream,
                )?;
                anyhow::Result::<()>::Ok(())
            })?;
        } // end !fused_gateup

        // FFN activation-sparsity MEASUREMENT (ATLAS_MEASURE_FFN_SPARSITY).
        // This is the K=γ verify FFN — the path DFlash decode ACTUALLY uses
        // (forward() M=1 is not hit per decode step). Row 0 of the [n, *]
        // buffers is a representative decode activation. `input` is the
        // gate/up input (K=hidden); `gate_out` now holds silu(gate)*up, the
        // down_proj input (K=intermediate) — no recompute needed here (unlike
        // forward(), which consumes gate_out in its fused path). PURE READER:
        // measures row 0 into dedicated accumulators, never mutates the token
        // stream. Skipped under graph capture (lazy alloc illegal mid-capture)
        // → run with ATLAS_DFLASH_DEBUG_NO_GRAPH=1 to observe eager decode.
        if self.sparsity_measure_active() && !ctx.graph_capture {
            let (hist_g, count_g, hist_d, count_d, _meas) =
                self.ensure_sparsity_meas(ctx.gpu, inter as usize)?;
            self.measure_sparsity_site(ctx, input, hist_g, count_g, h, stream)?;
            self.measure_sparsity_site(ctx, gate_out, hist_d, count_d, inter, stream)?;
            self.maybe_dump_sparsity(ctx, "dense_ffn_kgamma", stream)?;
        }

        // down_proj GEMM: [n, inter] → [n, H]
        let output = ctx.buffers.moe_output();
        crate::kprof!(ctx.gpu, stream, "ffn_down_kgamma", {
            // On wide discrete GPUs (`sm_count > 64`) at small `M <= 16`, route
            // `down_proj` directly through intra-CTA warp-split-K BF16xBF16 MMA
            // (`w4a16_gemm_coalesced_{v2_}n16_sk4_m16`, 320 CTAs at `H=5120`):
            // eliminates global FP32 Split-K partial traffic, eliminates the
            // `reduce_splitk_f32_to_bf16` launch across all 64 layers, removes
            // unscaled FP8 E4M3 activation clipping, and halves wasted M-tile
            // rows vs M32. Always skipped on GB10 (`<= 64` SMs).
            let coalesced_done = m32_path
                && n <= 16
                && !self.weights.down_proj.is_null()
                && try_coalesced_nvfp4_sm120_small_m(
                    ctx.gpu,
                    gate_out,
                    &self.weights.down_proj,
                    output,
                    n,
                    h,
                    inter,
                    16,
                    stream,
                )?;
            if !coalesced_done {
                // Split-K down_proj: [M=n, N=h, K=inter]. The single-slice
                // m32_n64 kernel is occupancy-starved here (N=h=5120 → 80 CTAs
                // vs gate/up's 256 at N=inter=16384) and grinds a long K-loop.
                // Split-K multiplies CTAs by `splits` into an FP32 workspace,
                // then reduces → BF16. Gated by ATLAS_FFN_DOWN_SPLITK; falls
                // through to the single-slice path when disabled or unallocated.
                let splitk = crate::layers::ffn_down_splitk();
                let ws = if splitk > 0 {
                    *self.splitk_workspace.lock().unwrap()
                } else {
                    None
                };
                if let (true, Some(ws)) = (
                    m32_path
                        && splitk > 0
                        && self.w4a16_gemm_t_m32_n64_splitk.0 != 0
                        && self.reduce_splitk_k.0 != 0,
                    ws,
                ) {
                    let dt = self.down_proj_t.as_ref().unwrap();
                    ops::w4a16_gemm_n64_m32_splitk(
                        ctx.gpu,
                        self.w4a16_gemm_t_m32_n64_splitk,
                        self.reduce_splitk_k,
                        gate_out,
                        dt,
                        output,
                        ws,
                        n,
                        h,
                        inter,
                        h, // ldb == N for tightly-packed T-weight
                        splitk,
                        stream,
                    )?;
                } else if m32_path {
                    let dt = self.down_proj_t.as_ref().unwrap();
                    ops::w4a16_gemm_n64_m32(
                        ctx.gpu,
                        self.w4a16_gemm_t_m32_n64,
                        gate_out,
                        dt,
                        output,
                        n,
                        h,
                        inter,
                        stream,
                    )?;
                } else if m128_path {
                    let dt = self.down_proj_t.as_ref().unwrap();
                    ops::w4a16_gemm_n128_m128(
                        ctx.gpu,
                        self.w4a16_gemm_t_m128,
                        gate_out,
                        dt,
                        output,
                        n,
                        h,
                        inter,
                        stream,
                    )?;
                } else if m16_path {
                    let dt = self.down_proj_t.as_ref().unwrap();
                    ops::w4a16_gemm_n128_m16(
                        ctx.gpu,
                        self.w4a16_gemm_t_m16,
                        gate_out,
                        dt,
                        output,
                        n,
                        h,
                        inter,
                        stream,
                    )?;
                } else {
                    ops::w4a16_gemm(
                        ctx.gpu,
                        self.w4a16_gemm,
                        gate_out,
                        &self.weights.down_proj,
                        output,
                        n,
                        h,
                        inter,
                        stream,
                    )?;
                }
            }
            anyhow::Result::<()>::Ok(())
        })?;

        Ok(())
    }

    /// Internal exact-scalar eligibility inside `forward_kgamma`: stays true
    /// for `rows in 2..=32` on GB10 (`<= 64` SMs) or when `ATLAS_FFN_TC=0`, and
    /// yields to batched tensor-core kernels for `rows >= 4` on `> 64`-SM GPUs.
    fn exact_kgamma_internal_eligible(&self, rows: u32) -> bool {
        !ffn_tc_active_for_rows(rows)
            && exact_ffn_auto_kgamma_applicable(
                rows,
                self.bf16_weights.is_none()
                    && !self.has_w3_gemm()
                    && self.activation == FfnActivation::SiLU,
            )
    }

    /// True when a `> 64`-SM GPU should auto-route `rows in 4..=256` into
    /// batched tensor-core `forward_kgamma` (even when `ATLAS_FFN_KGAMMA_M16`
    /// is unset), preventing attention layers from falling back to a per-token
    /// `ffn.forward()` loop at `C = 16..64`. Always `false` on GB10 (`<= 64` SMs)
    /// and whenever `ATLAS_FFN_TC=0`.
    fn sm120_auto_tc_kgamma_ready(&self, rows: u32) -> bool {
        if atlas_core::device::active_sm_count() <= FFN_KGAMMA_AUTO_TC_SM_THRESHOLD
            || !ffn_tc_active_for_rows(rows)
            || rows < crate::layers::TC_VERIFY_MIN_ROWS as u32
            || rows > FFN_KGAMMA_WIDE_MAX_ROWS
            || self.bf16_weights.is_some()
            || self.has_w3_gemm()
            || self.activation != FfnActivation::SiLU
        {
            return false;
        }
        if (rows as usize) <= sm120_coalesced_max_m()
            && sm120_coalesced_m16_env_enabled()
            && !self.weights.gate_proj.is_null()
            && !self.weights.up_proj.is_null()
            && !self.weights.down_proj.is_null()
        {
            return true;
        }
        if rows as usize <= crate::layers::TC_VERIFY_MAX_ROWS {
            self.has_transposed_ffn() && self.w4a16_gemm_t_m128.0 != 0
        } else {
            self.wide_kgamma_ready(rows)
        }
    }

    /// Whether `forward_kgamma` should service `rows` without requiring an
    /// explicit `ATLAS_FFN_KGAMMA_M16=1` opt-in.
    ///
    /// On GB10 (`<= 64` SMs) this is strictly the exact W4+SiLU dispatcher for
    /// `rows in 2..=32`. On `> 64`-SM GPUs (unless `ATLAS_FFN_TC=0`) this also
    /// admits `rows in 4..=256` when the transposed tensor-core FFN kernels are
    /// ready so `M = 4, 8, 16, 32, 64` never fall back to per-token loops.
    pub fn exact_kgamma_applicable(&self, rows: u32) -> bool {
        self.exact_kgamma_internal_eligible(rows) || self.sm120_auto_tc_kgamma_ready(rows)
    }

    /// Whether single-token decode should run `forward_kgamma(n = 1)` so it
    /// shares the K=γ verify's tensor-core FFN route
    /// (`ATLAS_DECODE_TC_PARITY=1`). Engages only when verify itself is on
    /// that route: `ATLAS_FFN_TC=1` bypasses the exact dispatcher and
    /// `ATLAS_FFN_KGAMMA_M16=1` sends verify rows through `forward_kgamma`.
    pub fn decode_tc_parity_route(&self) -> bool {
        crate::layers::decode_tc_parity_enabled()
            && exact_ffn_tc_override()
            && crate::layers::ffn_kgamma_m16_enabled()
            && self.bf16_weights.is_none()
            && !self.has_w3_gemm()
            && self.activation == FfnActivation::SiLU
            && self.has_transposed_ffn()
    }
}
