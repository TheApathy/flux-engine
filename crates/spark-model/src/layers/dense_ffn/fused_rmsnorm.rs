// SPDX-License-Identifier: AGPL-3.0-only

//! Opt-in (`ATLAS_FUSED_RMSNORM_NVFP4=1`, default OFF) Blackwell
//! `PreparedRmsNormNvfp4` integration hooks for [`DenseFfnLayer`].
//!
//! Replaces separate `rms_norm` / `residual_add_rms_norm` +
//! `quantize_bf16_to_nvfp4*` and `silu_mul` + `quantize_bf16_to_nvfp4*`
//! launches with the single-pass SM120/SM121 fused kernels in
//! `kernels/gb10/common/rms_norm_nvfp4_sm120.cu` across FlashInfer, cuBLASLt,
//! and W4A4 dense FFN prefill paths. Fails closed when explicitly enabled on an
//! incompatible configuration.

use anyhow::{Context, Result, bail, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};

#[cfg(feature = "cuda")]
use super::cublaslt_proj::{
    clear_precomputed_proj_absmax, ensure_shared_merged_gate_up_out, nvfp4_weight_ready,
};
use super::e2m1_flashinfer::{
    E2m1PrefillRoute, E2m1Scope, E2m1SelectedKernel, e2m1_kmajor_projection_shape,
    e2m1_prefill_route, e2m1_selected_kernel,
};
#[cfg(all(feature = "cuda", target_os = "linux"))]
use super::e2m1_flashinfer::{
    FlashinferFfnPrefillRoute, ensure_flashinfer_ffn_disjoint,
    flashinfer_ffn_activation_scale_extent, flashinfer_ffn_extent, flashinfer_ffn_prefill_route,
};
use super::layer::{DenseFfnLayer, FfnActivation};
use super::policy::supports_warp_nvfp4_mma;
#[cfg(feature = "cuda")]
use super::policy::{
    FlashInferFfnAvailability, cublaslt_nvfp4_route_enabled, ffn_prefill_l2_strip_rows,
    prefill_ffn_cublaslt_nvfp4_policy,
};
use crate::layer::ForwardContext;
use crate::layers::ops::{
    self, FusedRmsNormNvfp4Policy, Nvfp4ScaleLayout, RmsNormNvfp4LaunchArgs,
    RmsNormResidualNvfp4LaunchArgs, RmsNormWeightMode, SiluMulNvfp4LaunchArgs, SwiGluInputLayout,
};
use crate::weight_map::DenseWeight;

/// Prepared NVFP4 activation slice produced by `rms_norm_nvfp4_sm120` or
/// `rms_norm_residual_nvfp4_sm120`, ready for direct consumption by FlashInfer,
/// cuBLASLt (`Swizzled128x4`), or W4A4 (`Linear`) dense FFN projections.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PreparedRmsNormNvfp4 {
    pub packed: DevicePtr,
    pub scales: DevicePtr,
    pub scale2: f32,
    pub rows: u32,
    pub cols: u32,
    pub layout: Nvfp4ScaleLayout,
}

/// Pure admission decision for the opt-in `ATLAS_FUSED_RMSNORM_NVFP4=1` path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FusedRmsNormNvfp4Admission {
    Disabled,
    Incompatible(&'static str),
    Admitted,
}

/// Inputs to [`evaluate_fused_rmsnorm_nvfp4_admission`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct FusedRmsNormNvfp4AdmissionInput {
    pub policy: FusedRmsNormNvfp4Policy,
    pub activation: FfnActivation,
    pub has_bf16_weights: bool,
    pub has_required_kernels: bool,
    pub has_static_scales: bool,
    pub rows: u32,
    pub cols: u32,
    pub scale_layout: Nvfp4ScaleLayout,
}

/// Pure, side-effect-free admission predicate for `ATLAS_FUSED_RMSNORM_NVFP4=1`.
#[must_use]
pub(crate) const fn evaluate_fused_rmsnorm_nvfp4_admission(
    input: FusedRmsNormNvfp4AdmissionInput,
) -> FusedRmsNormNvfp4Admission {
    match input.policy {
        FusedRmsNormNvfp4Policy::Disabled => FusedRmsNormNvfp4Admission::Disabled,
        FusedRmsNormNvfp4Policy::Enabled => {
            if input.has_bf16_weights {
                return FusedRmsNormNvfp4Admission::Incompatible(
                    "ATLAS_FUSED_RMSNORM_NVFP4=1 is incompatible with BF16 dense FFN weights",
                );
            }
            if !matches!(input.activation, FfnActivation::SiLU) {
                return FusedRmsNormNvfp4Admission::Incompatible(
                    "ATLAS_FUSED_RMSNORM_NVFP4=1 requires a SiLU-gated dense FFN",
                );
            }
            if !input.has_required_kernels {
                return FusedRmsNormNvfp4Admission::Incompatible(
                    "ATLAS_FUSED_RMSNORM_NVFP4=1 requires rms_norm_nvfp4_sm120 kernels in the loaded bundle",
                );
            }
            if !input.has_static_scales {
                return FusedRmsNormNvfp4Admission::Incompatible(
                    "ATLAS_FUSED_RMSNORM_NVFP4=1 requires valid ModelOpt static activation scales",
                );
            }
            if input.rows == 0 {
                return FusedRmsNormNvfp4Admission::Incompatible(
                    "ATLAS_FUSED_RMSNORM_NVFP4=1 requires rows > 0",
                );
            }
            let align = match input.scale_layout {
                Nvfp4ScaleLayout::Linear => ops::RMSNORM_NVFP4_GROUP_SIZE,
                Nvfp4ScaleLayout::Swizzled128x4 => ops::RMSNORM_NVFP4_128X4_COLS_ALIGN,
            };
            if input.cols < align || !input.cols.is_multiple_of(align) {
                return FusedRmsNormNvfp4Admission::Incompatible(
                    "ATLAS_FUSED_RMSNORM_NVFP4=1 requires cols aligned to NVFP4 scale layout",
                );
            }
            FusedRmsNormNvfp4Admission::Admitted
        }
    }
}

/// Pure context guardrails shared by both fused prefill entry points.
#[must_use]
pub(crate) fn fused_prefill_context_guards_allow(
    graph_capture: bool,
    stream_capturing: bool,
    use_fp32_residual: bool,
    model_type: &str,
    tp_world_size: usize,
    has_bf16_weights: bool,
) -> bool {
    !graph_capture
        && !stream_capturing
        && !use_fp32_residual
        && model_type != "gemma4"
        && tp_world_size <= 1
        && !has_bf16_weights
}

impl DenseFfnLayer {
    fn fused_prefill_context_eligible(&self, ctx: &ForwardContext, stream: u64) -> bool {
        fused_prefill_context_guards_allow(
            ctx.graph_capture,
            ctx.gpu.stream_is_capturing(stream),
            ctx.config.use_fp32_residual(),
            &ctx.config.model_type,
            ctx.config.tp_world_size,
            self.bf16_weights.is_some(),
        )
    }

    /// Launch `silu_mul_nvfp4_sm120` for either split `(gate, up)` or merged
    /// `[gate | up]` down-projection inputs. Fails closed if the fused kernel
    /// or SiLU activation contract is not satisfied.
    #[expect(
        clippy::too_many_arguments,
        reason = "dense FFN SwiGLU->NVFP4 helper forwarding device buffers, layout, scale, and stream"
    )]
    pub(super) fn launch_fused_silu_nvfp4(
        &self,
        gpu: &dyn GpuBackend,
        gate: DevicePtr,
        up: DevicePtr,
        packed_out: DevicePtr,
        scale_out: DevicePtr,
        scale2: f32,
        rows: u32,
        cols: u32,
        input_layout: SwiGluInputLayout,
        scale_layout: Nvfp4ScaleLayout,
        stream: u64,
    ) -> Result<()> {
        ensure!(
            self.activation == FfnActivation::SiLU,
            "ATLAS_FUSED_RMSNORM_NVFP4=1 SwiGLU fusion requires a SiLU-gated dense FFN"
        );
        ensure!(
            self.fused_rmsnorm_nvfp4_k.has_silu_mul(),
            "ATLAS_FUSED_RMSNORM_NVFP4=1 requires silu_mul_nvfp4_sm120 in rms_norm_nvfp4_sm120"
        );
        ops::launch_silu_mul_nvfp4_sm120(
            gpu,
            self.fused_rmsnorm_nvfp4_k.silu_mul,
            &SiluMulNvfp4LaunchArgs {
                gate,
                up,
                packed_out,
                scale_out,
                global_scale: Some(scale2),
                rows,
                cols,
                input_layout,
                scale_layout,
            },
            stream,
        )
    }

    /// Run `rms_norm_nvfp4_sm120` into caller-owned NVFP4 activation scratch
    /// buffers, returning a [`PreparedRmsNormNvfp4`] receipt.
    #[expect(
        clippy::too_many_arguments,
        reason = "prepares fused RMSNorm->NVFP4 activation directly into caller scratch"
    )]
    pub(super) fn prepare_rms_norm_nvfp4_into(
        &self,
        gpu: &dyn GpuBackend,
        input: DevicePtr,
        norm_weight: &DenseWeight,
        packed_out: DevicePtr,
        scale_out: DevicePtr,
        scale2: f32,
        eps: f32,
        weight_mode: RmsNormWeightMode,
        rows: u32,
        cols: u32,
        scale_layout: Nvfp4ScaleLayout,
        stream: u64,
    ) -> Result<PreparedRmsNormNvfp4> {
        ensure!(
            self.fused_rmsnorm_nvfp4_k.has_rms_norm(),
            "ATLAS_FUSED_RMSNORM_NVFP4=1 requires rms_norm_nvfp4_sm120 in rms_norm_nvfp4_sm120"
        );
        ops::launch_rms_norm_nvfp4_sm120(
            gpu,
            self.fused_rmsnorm_nvfp4_k.rms_norm,
            &RmsNormNvfp4LaunchArgs {
                input,
                weight: norm_weight.weight,
                packed_out,
                scale_out,
                global_scale: Some(scale2),
                eps,
                weight_mode,
                rows,
                cols,
                scale_layout,
            },
            stream,
        )?;
        Ok(PreparedRmsNormNvfp4 {
            packed: packed_out,
            scales: scale_out,
            scale2,
            rows,
            cols,
            layout: scale_layout,
        })
    }

    /// Run `rms_norm_residual_nvfp4_sm120` (in-place `hidden += residual` +
    /// RMSNorm -> NVFP4) into caller-owned NVFP4 activation scratch buffers.
    #[expect(
        clippy::too_many_arguments,
        reason = "prepares fused residual+RMSNorm->NVFP4 activation directly into caller scratch"
    )]
    pub(super) fn prepare_rms_norm_residual_nvfp4_into(
        &self,
        gpu: &dyn GpuBackend,
        hidden: DevicePtr,
        residual: DevicePtr,
        norm_weight: &DenseWeight,
        packed_out: DevicePtr,
        scale_out: DevicePtr,
        scale2: f32,
        eps: f32,
        weight_mode: RmsNormWeightMode,
        rows: u32,
        cols: u32,
        scale_layout: Nvfp4ScaleLayout,
        stream: u64,
    ) -> Result<PreparedRmsNormNvfp4> {
        ensure!(
            self.fused_rmsnorm_nvfp4_k.has_rms_norm_residual(),
            "ATLAS_FUSED_RMSNORM_NVFP4=1 requires rms_norm_residual_nvfp4_sm120 in rms_norm_nvfp4_sm120"
        );
        ops::launch_rms_norm_residual_nvfp4_sm120(
            gpu,
            self.fused_rmsnorm_nvfp4_k.rms_norm_residual,
            &RmsNormResidualNvfp4LaunchArgs {
                hidden,
                residual,
                weight: norm_weight.weight,
                packed_out,
                scale_out,
                global_scale: Some(scale2),
                eps,
                weight_mode,
                rows,
                cols,
                scale_layout,
            },
            stream,
        )?;
        Ok(PreparedRmsNormNvfp4 {
            packed: packed_out,
            scales: scale_out,
            scale2,
            rows,
            cols,
            layout: scale_layout,
        })
    }

    /// Opt-in (`ATLAS_FUSED_RMSNORM_NVFP4=1`) prefill FFN entry taking a
    /// pre-norm BF16 `input` + `norm_weight` directly, fusing `rms_norm` +
    /// `quantize_bf16_to_nvfp4*` into `rms_norm_nvfp4_sm120` and `silu_mul` +
    /// `quantize` into `silu_mul_nvfp4_sm120`.
    ///
    /// Returns `Ok(false)` when `ATLAS_FUSED_RMSNORM_NVFP4` is OFF or no
    /// quantized NVFP4 FFN path (FlashInfer / W4A4) is active. Fails closed
    /// when `ATLAS_FUSED_RMSNORM_NVFP4=1` is requested on an active quantized
    /// FFN path whose kernel/scale/alignment invariants cannot be satisfied.
    #[expect(
        clippy::too_many_arguments,
        reason = "fused pre-norm + quantized FFN prefill dispatcher"
    )]
    pub fn try_forward_prefill_with_rms_norm_nvfp4(
        &self,
        pre_norm_input: DevicePtr,
        norm_weight: &DenseWeight,
        num_tokens: usize,
        eps: f32,
        weight_mode: RmsNormWeightMode,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<bool> {
        let policy = ops::fused_rmsnorm_nvfp4_policy()?;
        if policy == FusedRmsNormNvfp4Policy::Disabled {
            return Ok(false);
        }
        if !self.fused_prefill_context_eligible(ctx, stream) {
            return Ok(false);
        }
        let Ok(m) = u32::try_from(num_tokens) else {
            bail!("ATLAS_FUSED_RMSNORM_NVFP4=1 token count {num_tokens} exceeds u32");
        };
        let h = ctx.config.hidden_size as u32;
        let inter = ctx.config.intermediate_size as u32;

        #[cfg(all(feature = "cuda", target_os = "linux"))]
        if self.try_flashinfer_with_fused_norm(
            FusedNormInput::NonResidual {
                input: pre_norm_input,
            },
            norm_weight,
            m,
            h,
            inter,
            eps,
            weight_mode,
            ctx,
            stream,
        )? {
            return Ok(true);
        }

        if self.try_w4a4_with_fused_norm(
            FusedNormInput::NonResidual {
                input: pre_norm_input,
            },
            norm_weight,
            m,
            h,
            inter,
            eps,
            weight_mode,
            ctx,
            stream,
        )? {
            return Ok(true);
        }

        Ok(false)
    }

    /// Opt-in (`ATLAS_FUSED_RMSNORM_NVFP4=1`) fused residual-add + RMSNorm +
    /// NVFP4 quantization + quantized FFN prefill block.
    ///
    /// Invoked from [`DenseFfnLayer::try_forward_prefill_fused_residual_norm`]
    /// when `ATLAS_FUSED_RMSNORM_NVFP4=1` is explicitly enabled. Supports
    /// FlashInfer (`Swizzled128x4`), cuBLASLt (`Swizzled128x4`), and linear
    /// W4A4 (`Linear`), failing closed when any required kernel, scale, or
    /// alignment contract is violated.
    #[expect(
        clippy::too_many_arguments,
        reason = "fused residual + RMSNorm + NVFP4 FFN prefill entry point"
    )]
    pub(super) fn try_forward_prefill_fused_residual_rmsnorm_nvfp4(
        &self,
        hidden: DevicePtr,
        src: DevicePtr,
        norm_weight: &DenseWeight,
        num_tokens: usize,
        eps: f32,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<bool> {
        let policy = ops::fused_rmsnorm_nvfp4_policy()?;
        if policy == FusedRmsNormNvfp4Policy::Disabled {
            return Ok(false);
        }
        if !self.fused_prefill_context_eligible(ctx, stream) {
            return Ok(false);
        }
        let Ok(m) = u32::try_from(num_tokens) else {
            bail!("ATLAS_FUSED_RMSNORM_NVFP4=1 token count {num_tokens} exceeds u32");
        };
        let h = ctx.config.hidden_size as u32;
        let inter = ctx.config.intermediate_size as u32;
        let weight_mode = RmsNormWeightMode::UnitOffset;

        #[cfg(all(feature = "cuda", target_os = "linux"))]
        if self.try_flashinfer_with_fused_norm(
            FusedNormInput::Residual {
                hidden,
                residual: src,
            },
            norm_weight,
            m,
            h,
            inter,
            eps,
            weight_mode,
            ctx,
            stream,
        )? {
            return Ok(true);
        }

        #[cfg(feature = "cuda")]
        if self.try_cublaslt_with_fused_residual_norm(
            hidden,
            src,
            norm_weight,
            m,
            h,
            inter,
            eps,
            weight_mode,
            ctx,
            stream,
        )? {
            return Ok(true);
        }

        if self.try_w4a4_with_fused_norm(
            FusedNormInput::Residual {
                hidden,
                residual: src,
            },
            norm_weight,
            m,
            h,
            inter,
            eps,
            weight_mode,
            ctx,
            stream,
        )? {
            return Ok(true);
        }

        Ok(false)
    }

    #[cfg(all(feature = "cuda", target_os = "linux"))]
    #[expect(
        clippy::too_many_arguments,
        reason = "internal FlashInfer fused norm+quantize+SwiGLU dispatcher"
    )]
    fn try_flashinfer_with_fused_norm(
        &self,
        norm_input: FusedNormInput,
        norm_weight: &DenseWeight,
        m: u32,
        h: u32,
        inter: u32,
        eps: f32,
        weight_mode: RmsNormWeightMode,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<bool> {
        use crate::weight_map::flashinfer_ffn_admission::{
            QWEN38_HIDDEN, QWEN38_INTERMEDIATE, Qwen38FfnOperation, select_qwen38_ffn_launch,
        };
        use ops::flashinfer_sm121::{FlashInferSm121Buffers, FlashInferSm121Shape};

        let requested = crate::layers::prefill_ffn_flashinfer_enabled()?;
        let exact_qwen38_dense = ctx.config.model_type == "qwen3_5"
            && ctx.config.num_experts == 0
            && h as usize == QWEN38_HIDDEN
            && inter as usize == QWEN38_INTERMEDIATE
            && ctx.config.tp_world_size.max(1) == 1;
        let m_qualified = matches!(m, 2_079 | 8_192)
            || crate::weight_map::flashinfer_ffn_admission::qwen38_ffn_launch_candidate(
                Qwen38FfnOperation::MergedGateUp,
                m as usize,
            )
            .is_ok();
        match flashinfer_ffn_prefill_route(
            requested,
            exact_qwen38_dense,
            m_qualified,
            self.flashinfer_ffn_prefill.is_some(),
        ) {
            FlashinferFfnPrefillRoute::Disabled | FlashinferFfnPrefillRoute::Ineligible => {
                return Ok(false);
            }
            FlashinferFfnPrefillRoute::Missing => bail!(
                "ATLAS_PREFILL_FFN_FLASHINFER=1 + ATLAS_FUSED_RMSNORM_NVFP4=1 requires prepared FlashInfer FFN operands for M={m}"
            ),
            FlashinferFfnPrefillRoute::Complete => {}
        }

        let has_required = match norm_input {
            FusedNormInput::NonResidual { .. } => {
                self.fused_rmsnorm_nvfp4_k.has_rms_norm()
                    && self.fused_rmsnorm_nvfp4_k.has_silu_mul()
            }
            FusedNormInput::Residual { .. } => {
                self.fused_rmsnorm_nvfp4_k.has_rms_norm_residual()
                    && self.fused_rmsnorm_nvfp4_k.has_silu_mul()
                    && self.residual_add_k.0 != 0
            }
        };
        match evaluate_fused_rmsnorm_nvfp4_admission(FusedRmsNormNvfp4AdmissionInput {
            policy: FusedRmsNormNvfp4Policy::Enabled,
            activation: self.activation,
            has_bf16_weights: self.bf16_weights.is_some(),
            has_required_kernels: has_required,
            has_static_scales: true,
            rows: m,
            cols: h,
            scale_layout: Nvfp4ScaleLayout::Swizzled128x4,
        }) {
            FusedRmsNormNvfp4Admission::Admitted => {}
            FusedRmsNormNvfp4Admission::Incompatible(reason) => bail!("{reason}"),
            FusedRmsNormNvfp4Admission::Disabled => return Ok(false),
        }

        let route = self
            .flashinfer_ffn_prefill
            .as_ref()
            .context("FlashInfer FFN route missing after admission")?;
        ensure!(
            route.merged_gate_up.layer == route.layer
                && route.merged_gate_up.source_gate_weight == self.weights.gate_proj.weight
                && route.merged_gate_up.source_up_weight == self.weights.up_proj.weight
                && route.down.weight == self.weights.down_proj.weight,
            "FlashInfer FFN checkpoint operand identity changed after preparation"
        );
        let merged_plan = select_qwen38_ffn_launch(Qwen38FfnOperation::MergedGateUp, m as usize)?;
        let down_plan = select_qwen38_ffn_launch(Qwen38FfnOperation::Down, m as usize)?;
        ensure!(
            merged_plan.workspace_bytes == 0 && down_plan.workspace_bytes == 0,
            "FlashInfer FFN runtime selected a nonzero-workspace plan"
        );

        let rows = m as usize;
        let merged_bytes = flashinfer_ffn_extent(rows, 2 * QWEN38_INTERMEDIATE, 2)?;
        let output_bytes = flashinfer_ffn_extent(rows, QWEN38_HIDDEN, 2)?;
        let activation_fp4_bytes = flashinfer_ffn_extent(rows, QWEN38_INTERMEDIATE, 1)? / 2;
        let activation_scale_bytes =
            flashinfer_ffn_activation_scale_extent(rows, QWEN38_INTERMEDIATE)?;
        ensure!(
            ctx.buffers.sizes().ffn_gate_up_bf16 >= merged_bytes
                && ctx.buffers.sizes().moe_output >= output_bytes,
            "FlashInfer FFN arena undersized for ATLAS_FUSED_RMSNORM_NVFP4=1"
        );

        let merged_out = ctx.buffers.ffn_gate_up_bf16();
        let output = ctx.buffers.moe_output();
        ensure!(
            !merged_out.is_null()
                && merged_out.0.is_multiple_of(16)
                && !output.is_null()
                && output.0.is_multiple_of(16),
            "FlashInfer FFN output buffers must be non-null and 16-byte aligned"
        );
        let residual_total_elems = match norm_input {
            FusedNormInput::Residual { .. } => Some(
                m.checked_mul(h)
                    .context("FlashInfer fused residual add element overflow")?,
            ),
            FusedNormInput::NonResidual { .. } => None,
        };
        let merged_shape = FlashInferSm121Shape::new(
            u8::try_from(merged_plan.tactic).context("FlashInfer tactic exceeds u8")?,
            merged_plan.m,
            merged_plan.n,
            merged_plan.k,
            1,
        )?;
        let down_shape = FlashInferSm121Shape::new(
            u8::try_from(down_plan.tactic).context("FlashInfer tactic exceeds u8")?,
            down_plan.m,
            down_plan.n,
            down_plan.k,
            1,
        )?;
        let mut merged_launch =
            route
                .library
                .prepare_borrowed_zero_workspace(ctx.gpu, merged_shape, stream)?;
        let mut down_launch = route
            .library
            .prepare_borrowed_zero_workspace(ctx.gpu, down_shape, stream)?;

        let _scratch_use = self.lock_e2m1_use(stream)?;
        let (activation_fp4, activation_scales, _) =
            self.ensure_e2m1_scratch(ctx.gpu, rows, inter as usize)?;
        ensure_flashinfer_ffn_disjoint(
            activation_fp4,
            activation_fp4_bytes,
            activation_scales,
            activation_scale_bytes,
        )?;
        ensure_flashinfer_ffn_disjoint(
            activation_fp4,
            activation_fp4_bytes,
            merged_out,
            merged_bytes,
        )?;
        ensure_flashinfer_ffn_disjoint(
            activation_scales,
            activation_scale_bytes,
            merged_out,
            merged_bytes,
        )?;

        let gate_up_scale = f32::from_bits(route.merged_gate_up.input_scale_bits);
        let prepared = match norm_input {
            FusedNormInput::NonResidual { input } => self.prepare_rms_norm_nvfp4_into(
                ctx.gpu,
                input,
                norm_weight,
                activation_fp4,
                activation_scales,
                gate_up_scale,
                eps,
                weight_mode,
                m,
                h,
                Nvfp4ScaleLayout::Swizzled128x4,
                stream,
            )?,
            FusedNormInput::Residual { hidden, residual } => self
                .prepare_rms_norm_residual_nvfp4_into(
                    ctx.gpu,
                    hidden,
                    residual,
                    norm_weight,
                    activation_fp4,
                    activation_scales,
                    gate_up_scale,
                    eps,
                    weight_mode,
                    m,
                    h,
                    Nvfp4ScaleLayout::Swizzled128x4,
                    stream,
                )?,
        };
        merged_launch.launch_eager(
            FlashInferSm121Buffers {
                output_bf16: merged_out,
                activation_fp4: prepared.packed,
                weight_fp4: route.merged_gate_up.weight,
                activation_scales: prepared.scales,
                weight_scales: route.merged_gate_up.weight_scales_128x4,
                global_scale_f32: route.merged_gate_up.alpha_f32,
            },
            stream,
        )?;
        self.launch_fused_silu_nvfp4(
            ctx.gpu,
            merged_out,
            merged_out.offset((inter as usize) * 2),
            activation_fp4,
            activation_scales,
            f32::from_bits(route.down.input_scale_bits),
            m,
            inter,
            SwiGluInputLayout::MergedGateUp,
            Nvfp4ScaleLayout::Swizzled128x4,
            stream,
        )?;
        down_launch.launch_eager(
            FlashInferSm121Buffers {
                output_bf16: output,
                activation_fp4,
                weight_fp4: route.down.weight,
                activation_scales,
                weight_scales: route.down.weight_scales_128x4,
                global_scale_f32: route.down.alpha_f32,
            },
            stream,
        )?;
        if let (FusedNormInput::Residual { hidden, .. }, Some(total_elems)) =
            (norm_input, residual_total_elems)
        {
            ops::residual_add(
                ctx.gpu,
                self.residual_add_k,
                hidden,
                output,
                total_elems,
                stream,
            )?;
        }

        static LOGGED: std::sync::Once = std::sync::Once::new();
        LOGGED.call_once(|| {
            tracing::info!(
                layer = route.layer,
                m,
                h,
                inter,
                "ENGAGED ATLAS_FUSED_RMSNORM_NVFP4=1 FlashInfer fused norm+SwiGLU NVFP4 prefill"
            );
        });
        Ok(true)
    }

    #[cfg(feature = "cuda")]
    #[expect(
        clippy::too_many_arguments,
        reason = "internal cuBLASLt fused residual+RMSNorm->NVFP4 strip-mined dispatcher"
    )]
    fn try_cublaslt_with_fused_residual_norm(
        &self,
        hidden: DevicePtr,
        src: DevicePtr,
        norm_weight: &DenseWeight,
        m: u32,
        h: u32,
        inter: u32,
        eps: f32,
        weight_mode: RmsNormWeightMode,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<bool> {
        let cc = ctx.gpu.compute_capability();
        #[cfg(target_os = "linux")]
        let flashinfer_ffn = if self.flashinfer_ffn_prefill.is_some() {
            FlashInferFfnAvailability::Prepared
        } else {
            FlashInferFfnAvailability::Absent
        };
        #[cfg(not(target_os = "linux"))]
        let flashinfer_ffn = FlashInferFfnAvailability::Absent;

        let scales = self
            .cublaslt_checkpoint_scales
            .or(self.e2m1_checkpoint_scales);
        if !cublaslt_nvfp4_route_enabled(
            prefill_ffn_cublaslt_nvfp4_policy(),
            cc,
            flashinfer_ffn,
            m,
            h,
            inter,
            true,
            scales.is_some(),
        ) {
            return Ok(false);
        }
        let weights_ready = [
            &self.weights.gate_proj,
            &self.weights.up_proj,
            &self.weights.down_proj,
        ]
        .into_iter()
        .all(nvfp4_weight_ready);
        if !weights_ready {
            return Ok(false);
        }
        let has_required = self.fused_rmsnorm_nvfp4_k.has_rms_norm_residual()
            && self.fused_rmsnorm_nvfp4_k.has_silu_mul();
        match evaluate_fused_rmsnorm_nvfp4_admission(FusedRmsNormNvfp4AdmissionInput {
            policy: FusedRmsNormNvfp4Policy::Enabled,
            activation: self.activation,
            has_bf16_weights: self.bf16_weights.is_some(),
            has_required_kernels: has_required,
            has_static_scales: scales.is_some(),
            rows: m,
            cols: h,
            scale_layout: Nvfp4ScaleLayout::Swizzled128x4,
        }) {
            FusedRmsNormNvfp4Admission::Admitted => {}
            FusedRmsNormNvfp4Admission::Incompatible(reason) => bail!("{reason}"),
            FusedRmsNormNvfp4Admission::Disabled => return Ok(false),
        }
        let Some(scales) = scales else {
            bail!("ATLAS_FUSED_RMSNORM_NVFP4=1 requires static activation scales for cuBLASLt");
        };
        if self.weights.gate_proj.weight_scale_2.to_bits()
            != self.weights.up_proj.weight_scale_2.to_bits()
        {
            bail!(
                "ATLAS_FUSED_RMSNORM_NVFP4=1 cuBLASLt fused norm requires matching gate/up weight_scale_2"
            );
        }

        let _scratch_use = self.lock_e2m1_use(stream)?;
        let (_gate_s, _up_s, down_s128x4, _dyn, merged_gate_up) =
            self.ensure_cublaslt_nvfp4_scales(ctx.gpu, h, inter, true, stream)?;
        let Some((merged_w, merged_s128x4)) = merged_gate_up else {
            return Ok(false);
        };

        let strip = ffn_prefill_l2_strip_rows(ctx.gpu).unwrap_or(m).min(m);
        let scratch_rows = strip.max(1);
        let proj_bytes = (scratch_rows as usize)
            .checked_mul(inter as usize)
            .and_then(|e| e.checked_mul(2))
            .context("fused cuBLASLt projection byte overflow")?;
        let merged_bytes = proj_bytes
            .checked_mul(2)
            .context("fused cuBLASLt merged output byte overflow")?;
        let merged_out = if ctx.buffers.sizes().ffn_gate_up_bf16 >= merged_bytes {
            ctx.buffers.ffn_gate_up_bf16()
        } else {
            let Some(ptr) = ensure_shared_merged_gate_up_out(ctx.gpu, merged_bytes)? else {
                return Ok(false);
            };
            ptr
        };

        let (a_packed, a_scale, _) =
            self.ensure_e2m1_scratch(ctx.gpu, scratch_rows as usize, h.max(inter) as usize)?;
        let merged_n = inter
            .checked_mul(2)
            .context("fused cuBLASLt merged gate/up N overflow")?;
        let merged_alpha = scales.gate_up * self.weights.gate_proj.weight_scale_2;
        let down_alpha = scales.down * self.weights.down_proj.weight_scale_2;
        ensure!(
            merged_alpha.is_finite() && merged_alpha > 0.0,
            "fused cuBLASLt invalid merged alpha {merged_alpha}"
        );
        ensure!(
            down_alpha.is_finite() && down_alpha > 0.0,
            "fused cuBLASLt invalid down alpha {down_alpha}"
        );
        let merged_scalars = spark_runtime::cublaslt::BlockScaleScalars::host(merged_alpha, 0.0);
        let down_scalars = spark_runtime::cublaslt::BlockScaleScalars::host(down_alpha, 1.0);

        clear_precomputed_proj_absmax();
        static LOGGED: std::sync::Once = std::sync::Once::new();
        LOGGED.call_once(|| {
            tracing::info!(
                cc = ?cc,
                m,
                h,
                inter,
                strip_rows = strip,
                "ENGAGED ATLAS_FUSED_RMSNORM_NVFP4=1 cuBLASLt fused residual_norm + SwiGLU NVFP4 prefill"
            );
        });

        let row_bytes_bf16 = (h as usize)
            .checked_mul(2)
            .context("fused cuBLASLt row byte stride overflow")?;
        let mut row0 = 0u32;
        while row0 < m {
            let m_s = (m - row0).min(strip);
            let row_off = (row0 as usize)
                .checked_mul(row_bytes_bf16)
                .context("fused cuBLASLt strip byte offset overflow")?;
            let h_s = hidden.offset(row_off);
            let s_s = src.offset(row_off);
            let prepared = self.prepare_rms_norm_residual_nvfp4_into(
                ctx.gpu,
                h_s,
                s_s,
                norm_weight,
                a_packed,
                a_scale,
                scales.gate_up,
                eps,
                weight_mode,
                m_s,
                h,
                Nvfp4ScaleLayout::Swizzled128x4,
                stream,
            )?;
            spark_runtime::cublaslt::nvfp4_gemm_block_scaled(
                prepared.packed.0,
                prepared.scales.0,
                merged_w.0,
                merged_s128x4.0,
                merged_out.0,
                m_s,
                merged_n,
                h,
                merged_n,
                merged_scalars,
                stream,
            )?;
            self.launch_fused_silu_nvfp4(
                ctx.gpu,
                merged_out,
                merged_out.offset((inter as usize) * 2),
                a_packed,
                a_scale,
                scales.down,
                m_s,
                inter,
                SwiGluInputLayout::MergedGateUp,
                Nvfp4ScaleLayout::Swizzled128x4,
                stream,
            )?;
            spark_runtime::cublaslt::nvfp4_gemm_block_scaled(
                a_packed.0,
                a_scale.0,
                self.weights.down_proj.weight.0,
                down_s128x4.0,
                h_s.0,
                m_s,
                h,
                inter,
                h,
                down_scalars,
                stream,
            )?;
            row0 += m_s;
        }
        Ok(true)
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "internal W4A4 linear fused norm+quantize+SwiGLU dispatcher"
    )]
    fn try_w4a4_with_fused_norm(
        &self,
        norm_input: FusedNormInput,
        norm_weight: &DenseWeight,
        m: u32,
        h: u32,
        inter: u32,
        eps: f32,
        weight_mode: RmsNormWeightMode,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<bool> {
        // Defense-in-depth context guardrails so W4A4 fused norm can never
        // bypass TP all-reduce, read FP32 residual buffers as BF16, or allocate
        // scratch during CUDA graph capture.
        if !self.fused_prefill_context_eligible(ctx, stream) {
            return Ok(false);
        }

        let e2m1_full = crate::layers::prefill_ffn_e2m1_enabled();
        let e2m1_kmajor = crate::layers::prefill_ffn_e2m1_kmajor_enabled();
        let e2m1_kmajor_m256 = crate::layers::prefill_ffn_e2m1_kmajor_m256_enabled();
        if e2m1_kmajor_m256 {
            ensure!(
                e2m1_kmajor,
                "ATLAS_E2M1_KMAJOR_M256=1 requires ATLAS_E2M1_KMAJOR=1"
            );
        }
        let selected_kernel = e2m1_selected_kernel(e2m1_kmajor, e2m1_kmajor_m256, m);
        let kmajor_ready = !e2m1_kmajor
            || (self.has_e2m1_kmajor_runtime()
                && self.has_transposed_ffn()
                && e2m1_kmajor_projection_shape(m, inter, h)
                && e2m1_kmajor_projection_shape(m, h, inter)
                && (selected_kernel != E2m1SelectedKernel::KmajorM256
                    || self.has_e2m1_kmajor_m256_runtime()));
        let route = e2m1_prefill_route(
            e2m1_full,
            false,
            m >= 128,
            if e2m1_kmajor {
                kmajor_ready
            } else {
                self.has_e2m1_ffn()
            },
            false,
        );
        match route {
            E2m1PrefillRoute::Disabled | E2m1PrefillRoute::Ineligible => return Ok(false),
            E2m1PrefillRoute::Missing(_) => bail!(
                "ATLAS_E2M1_GEMM=1 + ATLAS_FUSED_RMSNORM_NVFP4=1 requires complete W4A4 runtime"
            ),
            E2m1PrefillRoute::Complete(E2m1Scope::Full) => {}
            E2m1PrefillRoute::Complete(E2m1Scope::DownOnly) => return Ok(false),
        }

        // Preflight every compute-capability, dimension, scale, pointer, and
        // overflow invariant BEFORE issuing `prepare_rms_norm_residual_nvfp4_into`
        // (which mutates `hidden += residual` in place on GPU).
        ensure!(
            supports_warp_nvfp4_mma(ctx.gpu.compute_capability()),
            "warp-level SM120 NVFP4 MMA (cutlass_nvfp4_gemm.cu) is unsupported on {:?}; cannot run ATLAS_E2M1_GEMM=1 + ATLAS_FUSED_RMSNORM_NVFP4=1",
            ctx.gpu.compute_capability()
        );
        let has_required = match norm_input {
            FusedNormInput::NonResidual { .. } => {
                self.fused_rmsnorm_nvfp4_k.has_rms_norm()
                    && self.fused_rmsnorm_nvfp4_k.has_silu_mul()
            }
            FusedNormInput::Residual { .. } => {
                self.fused_rmsnorm_nvfp4_k.has_rms_norm_residual()
                    && self.fused_rmsnorm_nvfp4_k.has_silu_mul()
                    && self.residual_add_k.0 != 0
            }
        };
        match evaluate_fused_rmsnorm_nvfp4_admission(FusedRmsNormNvfp4AdmissionInput {
            policy: FusedRmsNormNvfp4Policy::Enabled,
            activation: self.activation,
            has_bf16_weights: self.bf16_weights.is_some(),
            has_required_kernels: has_required,
            has_static_scales: self.e2m1_checkpoint_scales.is_some(),
            rows: m,
            cols: h,
            scale_layout: Nvfp4ScaleLayout::Linear,
        }) {
            FusedRmsNormNvfp4Admission::Admitted => {}
            FusedRmsNormNvfp4Admission::Incompatible(reason) => bail!("{reason}"),
            FusedRmsNormNvfp4Admission::Disabled => return Ok(false),
        }
        let _ = ops::nvfp4_activation_extent(m, h, Nvfp4ScaleLayout::Linear)?;
        let _ = ops::nvfp4_activation_extent(m, inter, Nvfp4ScaleLayout::Linear)?;
        let residual_total_elems = match norm_input {
            FusedNormInput::Residual { .. } => Some(
                m.checked_mul(h)
                    .context("W4A4 fused residual add element overflow")?,
            ),
            FusedNormInput::NonResidual { .. } => None,
        };

        let scales = self
            .e2m1_checkpoint_scales
            .context("ATLAS_FUSED_RMSNORM_NVFP4=1 W4A4 requires static checkpoint scales")?;
        ensure!(
            scales.gate_up.is_finite()
                && scales.gate_up > 0.0
                && scales.down.is_finite()
                && scales.down > 0.0,
            "ATLAS_FUSED_RMSNORM_NVFP4=1 W4A4 requires finite positive static activation scales"
        );
        let gate_out = ctx.buffers.expert_gate_out();
        let up_out = ctx.buffers.expert_up_out();
        let output = ctx.buffers.moe_output();
        ensure!(
            !gate_out.is_null()
                && gate_out.0.is_multiple_of(16)
                && !up_out.is_null()
                && up_out.0.is_multiple_of(16)
                && !output.is_null()
                && output.0.is_multiple_of(16),
            "ATLAS_FUSED_RMSNORM_NVFP4=1 W4A4 requires non-null 16-byte aligned gate/up/output buffers"
        );
        let gate_weight = if e2m1_kmajor {
            self.gate_proj_t
                .as_ref()
                .context("W4A4 kmajor missing transposed gate_proj")?
        } else {
            &self.weights.gate_proj
        };
        let up_weight = if e2m1_kmajor {
            self.up_proj_t
                .as_ref()
                .context("W4A4 kmajor missing transposed up_proj")?
        } else {
            &self.weights.up_proj
        };
        let down_weight = if e2m1_kmajor {
            self.down_proj_t
                .as_ref()
                .context("W4A4 kmajor missing transposed down_proj")?
        } else {
            &self.weights.down_proj
        };

        let _scratch_use = self.lock_e2m1_use(stream)?;
        let (a_packed, a_scale, _) =
            self.ensure_e2m1_scratch(ctx.gpu, m as usize, h.max(inter) as usize)?;
        let prepared = match norm_input {
            FusedNormInput::NonResidual { input } => self.prepare_rms_norm_nvfp4_into(
                ctx.gpu,
                input,
                norm_weight,
                a_packed,
                a_scale,
                scales.gate_up,
                eps,
                weight_mode,
                m,
                h,
                Nvfp4ScaleLayout::Linear,
                stream,
            )?,
            FusedNormInput::Residual { hidden, residual } => self
                .prepare_rms_norm_residual_nvfp4_into(
                    ctx.gpu,
                    hidden,
                    residual,
                    norm_weight,
                    a_packed,
                    a_scale,
                    scales.gate_up,
                    eps,
                    weight_mode,
                    m,
                    h,
                    Nvfp4ScaleLayout::Linear,
                    stream,
                )?,
        };

        self.forward_e2m1_prepared(
            ctx,
            prepared.packed,
            prepared.scales,
            prepared.scale2,
            gate_weight,
            gate_out,
            m,
            inter,
            h,
            e2m1_kmajor,
            stream,
        )?;
        self.forward_e2m1_prepared(
            ctx,
            prepared.packed,
            prepared.scales,
            prepared.scale2,
            up_weight,
            up_out,
            m,
            inter,
            h,
            e2m1_kmajor,
            stream,
        )?;
        self.launch_fused_silu_nvfp4(
            ctx.gpu,
            gate_out,
            up_out,
            a_packed,
            a_scale,
            scales.down,
            m,
            inter,
            SwiGluInputLayout::Split,
            Nvfp4ScaleLayout::Linear,
            stream,
        )?;
        self.forward_e2m1_prepared(
            ctx,
            a_packed,
            a_scale,
            scales.down,
            down_weight,
            output,
            m,
            h,
            inter,
            e2m1_kmajor,
            stream,
        )?;
        if let (FusedNormInput::Residual { hidden, .. }, Some(total_elems)) =
            (norm_input, residual_total_elems)
        {
            ops::residual_add(
                ctx.gpu,
                self.residual_add_k,
                hidden,
                output,
                total_elems,
                stream,
            )?;
        }
        Ok(true)
    }
}

#[derive(Debug, Clone, Copy)]
enum FusedNormInput {
    NonResidual {
        input: DevicePtr,
    },
    Residual {
        hidden: DevicePtr,
        residual: DevicePtr,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_input() -> FusedRmsNormNvfp4AdmissionInput {
        FusedRmsNormNvfp4AdmissionInput {
            policy: FusedRmsNormNvfp4Policy::Enabled,
            activation: FfnActivation::SiLU,
            has_bf16_weights: false,
            has_required_kernels: true,
            has_static_scales: true,
            rows: 128,
            cols: 5120,
            scale_layout: Nvfp4ScaleLayout::Swizzled128x4,
        }
    }

    #[test]
    fn admission_is_disabled_by_default_and_fails_closed_on_incompatibilities() {
        let mut input = valid_input();
        assert_eq!(
            evaluate_fused_rmsnorm_nvfp4_admission(input),
            FusedRmsNormNvfp4Admission::Admitted
        );

        input.policy = FusedRmsNormNvfp4Policy::Disabled;
        assert_eq!(
            evaluate_fused_rmsnorm_nvfp4_admission(input),
            FusedRmsNormNvfp4Admission::Disabled
        );

        assert!(matches!(
            evaluate_fused_rmsnorm_nvfp4_admission(FusedRmsNormNvfp4AdmissionInput {
                has_bf16_weights: true,
                ..valid_input()
            }),
            FusedRmsNormNvfp4Admission::Incompatible(_)
        ));
        assert!(matches!(
            evaluate_fused_rmsnorm_nvfp4_admission(FusedRmsNormNvfp4AdmissionInput {
                activation: FfnActivation::GeLU,
                ..valid_input()
            }),
            FusedRmsNormNvfp4Admission::Incompatible(_)
        ));
        assert!(matches!(
            evaluate_fused_rmsnorm_nvfp4_admission(FusedRmsNormNvfp4AdmissionInput {
                has_required_kernels: false,
                ..valid_input()
            }),
            FusedRmsNormNvfp4Admission::Incompatible(_)
        ));
        assert!(matches!(
            evaluate_fused_rmsnorm_nvfp4_admission(FusedRmsNormNvfp4AdmissionInput {
                has_static_scales: false,
                ..valid_input()
            }),
            FusedRmsNormNvfp4Admission::Incompatible(_)
        ));
        assert!(matches!(
            evaluate_fused_rmsnorm_nvfp4_admission(FusedRmsNormNvfp4AdmissionInput {
                rows: 0,
                ..valid_input()
            }),
            FusedRmsNormNvfp4Admission::Incompatible(_)
        ));
        assert!(matches!(
            evaluate_fused_rmsnorm_nvfp4_admission(FusedRmsNormNvfp4AdmissionInput {
                cols: 32,
                scale_layout: Nvfp4ScaleLayout::Swizzled128x4,
                ..valid_input()
            }),
            FusedRmsNormNvfp4Admission::Incompatible(_)
        ));
        // Linear layout permits cols = 32 (multiple of 16).
        assert_eq!(
            evaluate_fused_rmsnorm_nvfp4_admission(FusedRmsNormNvfp4AdmissionInput {
                cols: 32,
                scale_layout: Nvfp4ScaleLayout::Linear,
                ..valid_input()
            }),
            FusedRmsNormNvfp4Admission::Admitted
        );
    }

    #[test]
    fn context_guards_reject_capture_fp32_residual_gemma4_tp_and_bf16() {
        assert!(fused_prefill_context_guards_allow(
            false, false, false, "qwen3_5", 1, false
        ));
        assert!(!fused_prefill_context_guards_allow(
            true, false, false, "qwen3_5", 1, false
        ));
        assert!(!fused_prefill_context_guards_allow(
            false, true, false, "qwen3_5", 1, false
        ));
        assert!(!fused_prefill_context_guards_allow(
            false, false, true, "qwen3_5", 1, false
        ));
        assert!(!fused_prefill_context_guards_allow(
            false, false, false, "gemma4", 1, false
        ));
        assert!(!fused_prefill_context_guards_allow(
            false, false, false, "qwen3_5", 2, false
        ));
        assert!(!fused_prefill_context_guards_allow(
            false, false, false, "qwen3_5", 1, true
        ));
    }

    #[test]
    fn w4a4_fused_preflight_order_checks_mma_m256_and_inter_before_gpu_mutation() {
        let source = include_str!("fused_rmsnorm.rs");
        let start = source
            .find("fn try_w4a4_with_fused_norm(")
            .expect("try_w4a4_with_fused_norm missing");
        let body = &source[start..];

        let ctx_guard = body.find("self.fused_prefill_context_eligible(").unwrap();
        let m256_check = body.find("self.has_e2m1_kmajor_m256_runtime()").unwrap();
        let warp_mma = body.find("supports_warp_nvfp4_mma(").unwrap();
        let inter_extent = body
            .find("ops::nvfp4_activation_extent(m, inter, Nvfp4ScaleLayout::Linear)")
            .unwrap();
        let residual_mutate = body.find("prepare_rms_norm_residual_nvfp4_into(").unwrap();
        assert!(
            ctx_guard < m256_check
                && m256_check < warp_mma
                && warp_mma < inter_extent
                && inter_extent < residual_mutate,
            "all W4A4 preflight invariants must run before in-place hidden += residual mutation"
        );
    }
}
