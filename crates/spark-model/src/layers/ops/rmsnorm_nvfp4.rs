// SPDX-License-Identifier: AGPL-3.0-only

//! Fused Blackwell (SM120/SM121 + SM103 fallback) `RMSNorm -> E2M1` and
//! `SwiGLU -> E2M1` single-pass NVFP4 activation preparation launchers and
//! exact CPU mathematical reference.
//!
//! All 3 CUDA entrypoints live in `kernels/gb10/common/rms_norm_nvfp4_sm120.cu`
//! (auto-discovered by `atlas-kernels` under module `"rms_norm_nvfp4_sm120"`):
//!
//! - `rms_norm_nvfp4_sm120`: BF16 input + BF16 weight -> RMSNorm -> block-16
//!   UE4M3 scales + packed E2M1 NVFP4 activations.
//! - `rms_norm_residual_nvfp4_sm120`: in-place BF16 `hidden += residual` +
//!   RMSNorm -> block-16 UE4M3 scales + packed E2M1 NVFP4 activations in one
//!   global pass.
//! - `silu_mul_nvfp4_sm120`: fused SwiGLU `silu(gate) * up` -> block-16 UE4M3
//!   scales + packed E2M1 NVFP4 activations (supporting both split `(gate, up)`
//!   and merged `[gate | up]` layouts).
//!
//! Opt-in gate: `ATLAS_FUSED_RMSNORM_NVFP4=1` (default OFF for bit-identical
//! baseline; fails closed on invalid or non-UTF-8 values).

use std::sync::OnceLock;

use anyhow::{Context, Result, bail, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

/// Auto-discovered `atlas-kernels` module name for
/// `kernels/gb10/common/rms_norm_nvfp4_sm120.cu`.
pub const RMSNORM_NVFP4_MODULE: &str = "rms_norm_nvfp4_sm120";

/// Environment variable gating the opt-in fused `RMSNorm/SwiGLU -> E2M1` path.
pub const RMSNORM_NVFP4_ENV_VAR: &str = "ATLAS_FUSED_RMSNORM_NVFP4";

/// NVFP4 block-scale group width in elements.
pub const RMSNORM_NVFP4_GROUP_SIZE: u32 = 16;

/// Column multiple required for `128x4` swizzled UE4M3 scales (`4 * 16 = 64`).
pub const RMSNORM_NVFP4_128X4_COLS_ALIGN: u32 = 64;

/// Row-tile padding multiple for `128x4` swizzled UE4M3 scales.
pub const RMSNORM_NVFP4_128X4_ROW_TILE: u32 = 128;

/// RMSNorm weight parameterization convention.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RmsNormWeightMode {
    /// Qwen3 / Qwen3.5 / Qwen3.8 offset-from-one convention: `y = norm(x) * (1 + w)`.
    UnitOffset,
    /// Standard Llama / Mistral convention: `y = norm(x) * w`.
    Direct,
}

impl RmsNormWeightMode {
    /// Additive weight offset passed to the CUDA kernel (`1.0` or `0.0`).
    #[must_use]
    pub const fn weight_offset(self) -> f32 {
        match self {
            Self::UnitOffset => 1.0,
            Self::Direct => 0.0,
        }
    }
}

/// Physical memory layout of emitted block-16 `UE4M3` (`e4m3fn`) scales.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Nvfp4ScaleLayout {
    /// Row-major `[rows, cols / 16]` for linear W4A4 (`nvfp4_cutlass`) GEMMs.
    Linear,
    /// CUTLASS / FlashInfer / cuBLASLt `128x4` swizzled layout with zero-padded
    /// rows to `ceil(rows / 128) * 128` (`[padded_rows, cols / 16]`).
    Swizzled128x4,
}

impl Nvfp4ScaleLayout {
    /// Raw `u32` selector expected by `rms_norm_nvfp4_sm120.cu`.
    #[must_use]
    pub const fn as_kernel_arg(self) -> u32 {
        match self {
            Self::Linear => 0,
            Self::Swizzled128x4 => 1,
        }
    }
}

/// Input layout for [`launch_silu_mul_nvfp4_sm120`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SwiGluInputLayout {
    /// Independent contiguous `[rows, cols]` BF16 buffers for `gate` and `up`.
    Split,
    /// Single merged `[rows, 2 * cols]` BF16 buffer where each row stores
    /// `[gate[0..cols] | up[0..cols]]` and `up == gate.offset(cols * 2)`.
    MergedGateUp,
}

/// Runtime policy resolved from `ATLAS_FUSED_RMSNORM_NVFP4`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FusedRmsNormNvfp4Policy {
    /// Default OFF: keep the existing unfused baseline bit-identical.
    Disabled,
    /// Explicit `ATLAS_FUSED_RMSNORM_NVFP4=1`: use fused single-pass Blackwell
    /// `RMSNorm/SwiGLU -> E2M1` kernels and fail closed when incompatible.
    Enabled,
}

/// Parse an `ATLAS_FUSED_RMSNORM_NVFP4` environment lookup strictly.
///
/// - Unset (`NotPresent`) or `"0"` -> `Ok(Disabled)`.
/// - `"1"` -> `Ok(Enabled)`.
/// - Any other UTF-8 value or `NotUnicode` -> `Err(...)` (fail closed).
pub fn parse_fused_rmsnorm_nvfp4_env(
    value: std::result::Result<String, std::env::VarError>,
) -> std::result::Result<FusedRmsNormNvfp4Policy, &'static str> {
    match value {
        Err(std::env::VarError::NotPresent) => Ok(FusedRmsNormNvfp4Policy::Disabled),
        Err(std::env::VarError::NotUnicode(_)) => {
            Err("ATLAS_FUSED_RMSNORM_NVFP4 must be valid UTF-8 and exactly 0 or 1")
        }
        Ok(raw) => match raw.as_str() {
            "0" => Ok(FusedRmsNormNvfp4Policy::Disabled),
            "1" => Ok(FusedRmsNormNvfp4Policy::Enabled),
            _ => Err("ATLAS_FUSED_RMSNORM_NVFP4 must be exactly 0 or 1"),
        },
    }
}

/// Read and cache `ATLAS_FUSED_RMSNORM_NVFP4` fail-closed.
pub fn fused_rmsnorm_nvfp4_policy() -> Result<FusedRmsNormNvfp4Policy> {
    static GATE: OnceLock<std::result::Result<FusedRmsNormNvfp4Policy, &'static str>> =
        OnceLock::new();
    (*GATE.get_or_init(|| parse_fused_rmsnorm_nvfp4_env(std::env::var(RMSNORM_NVFP4_ENV_VAR))))
        .map_err(anyhow::Error::msg)
}

/// Effective fused `RMSNorm/SwiGLU -> E2M1` policy resolving
/// `ATLAS_FUSED_RMSNORM_NVFP4` together with the `ATLAS_GB10_FAST_PREFILL`
/// umbrella.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EffectiveFusedRmsNormNvfp4Policy {
    /// Disabled (unset with `ATLAS_GB10_FAST_PREFILL` off, or explicit `=0`).
    Disabled,
    /// Auto-enabled by `ATLAS_GB10_FAST_PREFILL=1` while
    /// `ATLAS_FUSED_RMSNORM_NVFP4` is unset: engages whenever the layer
    /// satisfies all kernel, scale, and alignment contracts, and falls back
    /// cleanly on ineligible layers (e.g. dynamic-scale or BF16 FFN).
    AutoGb10FastPrefill,
    /// Explicitly enabled via `ATLAS_FUSED_RMSNORM_NVFP4=1`: fails closed on
    /// any incompatible configuration.
    ExplicitEnabled,
}

impl EffectiveFusedRmsNormNvfp4Policy {
    /// True when the fused `RMSNorm/SwiGLU -> E2M1` path should be attempted.
    #[must_use]
    pub const fn is_enabled(self) -> bool {
        !matches!(self, Self::Disabled)
    }

    /// True when incompatibilities must return an error rather than falling back.
    #[must_use]
    pub const fn is_fail_closed(self) -> bool {
        matches!(self, Self::ExplicitEnabled)
    }
}

/// Resolve `ATLAS_FUSED_RMSNORM_NVFP4` together with `ATLAS_GB10_FAST_PREFILL`.
///
/// Explicit `"0"` always disables even when `gb10_fast_prefill` is `true`;
/// explicit `"1"` stays fail-closed (`ExplicitEnabled`); unset promotes to
/// `AutoGb10FastPrefill` when `gb10_fast_prefill` is `true` and stays
/// `Disabled` otherwise.
pub fn resolve_effective_fused_rmsnorm_nvfp4_policy(
    value: std::result::Result<String, std::env::VarError>,
    gb10_fast_prefill: bool,
) -> std::result::Result<EffectiveFusedRmsNormNvfp4Policy, &'static str> {
    match value {
        Err(std::env::VarError::NotPresent) => Ok(if gb10_fast_prefill {
            EffectiveFusedRmsNormNvfp4Policy::AutoGb10FastPrefill
        } else {
            EffectiveFusedRmsNormNvfp4Policy::Disabled
        }),
        Err(std::env::VarError::NotUnicode(_)) => {
            Err("ATLAS_FUSED_RMSNORM_NVFP4 must be valid UTF-8 and exactly 0 or 1")
        }
        Ok(raw) => match raw.as_str() {
            "0" => Ok(EffectiveFusedRmsNormNvfp4Policy::Disabled),
            "1" => Ok(EffectiveFusedRmsNormNvfp4Policy::ExplicitEnabled),
            _ => Err("ATLAS_FUSED_RMSNORM_NVFP4 must be exactly 0 or 1"),
        },
    }
}

/// Read and cache the effective `ATLAS_FUSED_RMSNORM_NVFP4` policy fail-closed.
pub fn effective_fused_rmsnorm_nvfp4_policy() -> Result<EffectiveFusedRmsNormNvfp4Policy> {
    static GATE: OnceLock<std::result::Result<EffectiveFusedRmsNormNvfp4Policy, &'static str>> =
        OnceLock::new();
    let gb10_fast = super::gb10_fast_prefill_policy()?;
    (*GATE.get_or_init(|| {
        resolve_effective_fused_rmsnorm_nvfp4_policy(
            std::env::var(RMSNORM_NVFP4_ENV_VAR),
            gb10_fast == super::Gb10FastPrefillPolicy::Enabled,
        )
    }))
    .map_err(anyhow::Error::msg)
}

/// Resolved CUDA kernel handles for `rms_norm_nvfp4_sm120.cu`.
#[derive(Debug, Clone, Copy)]
pub struct FusedRmsNormNvfp4Kernels {
    pub rms_norm: KernelHandle,
    pub rms_norm_residual: KernelHandle,
    pub silu_mul: KernelHandle,
}

impl FusedRmsNormNvfp4Kernels {
    /// Probe all 3 optional entrypoints from `"rms_norm_nvfp4_sm120"`.
    #[must_use]
    pub fn load_optional(gpu: &dyn GpuBackend) -> Self {
        Self {
            rms_norm: crate::layers::try_kernel(gpu, RMSNORM_NVFP4_MODULE, "rms_norm_nvfp4_sm120"),
            rms_norm_residual: crate::layers::try_kernel(
                gpu,
                RMSNORM_NVFP4_MODULE,
                "rms_norm_residual_nvfp4_sm120",
            ),
            silu_mul: crate::layers::try_kernel(gpu, RMSNORM_NVFP4_MODULE, "silu_mul_nvfp4_sm120"),
        }
    }

    /// True when `rms_norm_nvfp4_sm120` resolved to a non-null handle.
    #[must_use]
    pub const fn has_rms_norm(&self) -> bool {
        self.rms_norm.0 != 0
    }

    /// True when `rms_norm_residual_nvfp4_sm120` resolved to a non-null handle.
    #[must_use]
    pub const fn has_rms_norm_residual(&self) -> bool {
        self.rms_norm_residual.0 != 0
    }

    /// True when `silu_mul_nvfp4_sm120` resolved to a non-null handle.
    #[must_use]
    pub const fn has_silu_mul(&self) -> bool {
        self.silu_mul.0 != 0
    }

    /// True when all 3 fused kernels are loaded.
    #[must_use]
    pub const fn is_complete(&self) -> bool {
        self.has_rms_norm() && self.has_rms_norm_residual() && self.has_silu_mul()
    }
}

/// Validated physical byte extents for one NVFP4 activation tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PreparedNvfp4Extent {
    pub packed_bytes: usize,
    pub scale_bytes: usize,
    pub padded_rows: u32,
}

/// Compute checked packed-E2M1 and UE4M3-scale byte extents for `(rows, cols, layout)`.
pub fn nvfp4_activation_extent(
    rows: u32,
    cols: u32,
    layout: Nvfp4ScaleLayout,
) -> Result<PreparedNvfp4Extent> {
    ensure!(rows > 0, "NVFP4 activation requires rows > 0");
    ensure!(
        cols >= RMSNORM_NVFP4_GROUP_SIZE && cols.is_multiple_of(RMSNORM_NVFP4_GROUP_SIZE),
        "NVFP4 activation requires cols >= 16 and cols % 16 == 0; got {cols}"
    );
    if layout == Nvfp4ScaleLayout::Swizzled128x4 {
        ensure!(
            cols >= RMSNORM_NVFP4_128X4_COLS_ALIGN
                && cols.is_multiple_of(RMSNORM_NVFP4_128X4_COLS_ALIGN),
            "NVFP4 128x4 activation requires cols >= 64 and cols % 64 == 0; got {cols}"
        );
    }

    let padded_rows = match layout {
        Nvfp4ScaleLayout::Linear => rows,
        Nvfp4ScaleLayout::Swizzled128x4 => rows
            .checked_add(RMSNORM_NVFP4_128X4_ROW_TILE - 1)
            .and_then(|v| {
                (v / RMSNORM_NVFP4_128X4_ROW_TILE).checked_mul(RMSNORM_NVFP4_128X4_ROW_TILE)
            })
            .context("NVFP4 128x4 row padding overflow")?,
    };
    let packed_bytes = (rows as usize)
        .checked_mul((cols / 2) as usize)
        .context("NVFP4 packed activation byte extent overflow")?;
    let groups = (cols / RMSNORM_NVFP4_GROUP_SIZE) as usize;
    let scale_bytes = (padded_rows as usize)
        .checked_mul(groups)
        .context("NVFP4 scale activation byte extent overflow")?;
    Ok(PreparedNvfp4Extent {
        packed_bytes,
        scale_bytes,
        padded_rows,
    })
}

/// Compute the byte offset of `(row, group)` inside `scale_out` for `layout`.
pub fn nvfp4_scale_offset(
    row: u32,
    group: u32,
    groups: u32,
    layout: Nvfp4ScaleLayout,
) -> Result<usize> {
    ensure!(groups > 0, "nvfp4_scale_offset requires groups > 0");
    ensure!(
        group < groups,
        "nvfp4_scale_offset group {group} out of range {groups}"
    );
    let offset_u64 = match layout {
        Nvfp4ScaleLayout::Linear => u64::from(row)
            .checked_mul(u64::from(groups))
            .and_then(|v| v.checked_add(u64::from(group)))
            .context("linear scale offset overflow")?,
        Nvfp4ScaleLayout::Swizzled128x4 => {
            let group_blocks = u64::from(groups.div_ceil(4));
            let row64 = u64::from(row);
            let group64 = u64::from(group);
            let tile = (row64 / 128)
                .checked_mul(group_blocks)
                .and_then(|v| v.checked_add(group64 / 4))
                .context("128x4 scale tile overflow")?;
            tile.checked_mul(32)
                .and_then(|v| v.checked_add(row64 % 32))
                .and_then(|v| v.checked_mul(4))
                .and_then(|v| v.checked_add((row64 % 128) / 32))
                .and_then(|v| v.checked_mul(4))
                .and_then(|v| v.checked_add(group64 % 4))
                .context("128x4 scale byte offset overflow")?
        }
    };
    usize::try_from(offset_u64).context("scale offset exceeds usize")
}

/// Arguments for [`launch_rms_norm_nvfp4_sm120`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RmsNormNvfp4LaunchArgs {
    pub input: DevicePtr,
    pub weight: DevicePtr,
    pub packed_out: DevicePtr,
    pub scale_out: DevicePtr,
    /// Optional static global activation scale (`scale2` / `alpha`). `None` uses `1.0`.
    pub global_scale: Option<f32>,
    pub eps: f32,
    pub weight_mode: RmsNormWeightMode,
    pub rows: u32,
    pub cols: u32,
    pub scale_layout: Nvfp4ScaleLayout,
}

/// Arguments for [`launch_rms_norm_residual_nvfp4_sm120`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RmsNormResidualNvfp4LaunchArgs {
    /// In-place residual stream updated as `hidden += residual` in BF16.
    pub hidden: DevicePtr,
    /// Read-only BF16 residual addend (e.g. attention or SSM output projection).
    pub residual: DevicePtr,
    pub weight: DevicePtr,
    pub packed_out: DevicePtr,
    pub scale_out: DevicePtr,
    /// Optional static global activation scale (`scale2` / `alpha`). `None` uses `1.0`.
    pub global_scale: Option<f32>,
    pub eps: f32,
    pub weight_mode: RmsNormWeightMode,
    pub rows: u32,
    pub cols: u32,
    pub scale_layout: Nvfp4ScaleLayout,
}

/// Arguments for [`launch_silu_mul_nvfp4_sm120`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SiluMulNvfp4LaunchArgs {
    pub gate: DevicePtr,
    pub up: DevicePtr,
    pub packed_out: DevicePtr,
    pub scale_out: DevicePtr,
    /// Optional static global activation scale (`scale2` / `alpha`). `None` uses `1.0`.
    pub global_scale: Option<f32>,
    pub rows: u32,
    pub cols: u32,
    pub input_layout: SwiGluInputLayout,
    pub scale_layout: Nvfp4ScaleLayout,
}

fn ensure_disjoint_ranges(
    left_label: &str,
    left: DevicePtr,
    left_bytes: usize,
    right_label: &str,
    right: DevicePtr,
    right_bytes: usize,
) -> Result<()> {
    let left_end = left
        .0
        .checked_add(u64::try_from(left_bytes).context("left buffer extent exceeds u64")?)
        .context("left buffer address overflow")?;
    let right_end = right
        .0
        .checked_add(u64::try_from(right_bytes).context("right buffer extent exceeds u64")?)
        .context("right buffer address overflow")?;
    ensure!(
        left_end <= right.0 || right_end <= left.0,
        "rms_norm_nvfp4_sm120 buffers {left_label} and {right_label} overlap"
    );
    Ok(())
}

fn validate_global_scale(global_scale: Option<f32>, kernel_name: &str) -> Result<f32> {
    let scale = global_scale.unwrap_or(1.0);
    ensure!(
        scale.is_finite() && scale > 0.0,
        "{kernel_name} requires finite positive global_scale; got {scale}"
    );
    let denom = scale * 6.0;
    ensure!(
        denom.is_finite() && denom > 0.0,
        "{kernel_name} global_scale * 6.0 overflowed ({scale})"
    );
    Ok(scale)
}

fn bf16_matrix_bytes(rows: u32, cols: u32, label: &str) -> Result<usize> {
    (rows as usize)
        .checked_mul(cols as usize)
        .and_then(|elems| elems.checked_mul(2))
        .with_context(|| format!("{label} BF16 byte extent overflow"))
}

fn rmsnorm_block_x(cols: u32) -> u32 {
    let groups = cols / RMSNORM_NVFP4_GROUP_SIZE;
    if groups <= 512 {
        div_ceil(groups, 32) * 32
    } else {
        256
    }
}

fn active_sm_count_for_gpu(gpu: &dyn GpuBackend) -> u32 {
    gpu.sm_count()
        .filter(|&s| s > 0)
        .unwrap_or_else(atlas_core::device::active_sm_count)
}

/// Launch `rms_norm_nvfp4_sm120` (fused RMSNorm -> block-16 NVFP4 quantization).
pub fn launch_rms_norm_nvfp4_sm120(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    args: &RmsNormNvfp4LaunchArgs,
    stream: u64,
) -> Result<()> {
    const NAME: &str = "rms_norm_nvfp4_sm120";
    ensure!(kernel.0 != 0, "{NAME} kernel handle is missing");
    ensure!(
        args.eps.is_finite() && args.eps > 0.0,
        "{NAME} requires finite positive eps; got {}",
        args.eps
    );
    let scale2 = validate_global_scale(args.global_scale, NAME)?;
    let extent = nvfp4_activation_extent(args.rows, args.cols, args.scale_layout)?;
    let input_bytes = bf16_matrix_bytes(args.rows, args.cols, "input")?;
    let weight_bytes = bf16_matrix_bytes(1, args.cols, "weight")?;

    ensure!(
        !args.input.is_null()
            && !args.weight.is_null()
            && !args.packed_out.is_null()
            && !args.scale_out.is_null(),
        "{NAME} received a null device pointer"
    );
    ensure!(
        args.input.0.is_multiple_of(16) && args.weight.0.is_multiple_of(16),
        "{NAME} requires 16-byte aligned BF16 input and weight pointers"
    );
    ensure!(
        args.packed_out.0.is_multiple_of(8),
        "{NAME} requires 8-byte aligned packed_out pointer"
    );
    ensure_disjoint_ranges(
        "packed_out",
        args.packed_out,
        extent.packed_bytes,
        "scale_out",
        args.scale_out,
        extent.scale_bytes,
    )?;
    for (label, ptr, bytes) in [
        ("input", args.input, input_bytes),
        ("weight", args.weight, weight_bytes),
    ] {
        ensure_disjoint_ranges(
            "packed_out",
            args.packed_out,
            extent.packed_bytes,
            label,
            ptr,
            bytes,
        )?;
        ensure_disjoint_ranges(
            "scale_out",
            args.scale_out,
            extent.scale_bytes,
            label,
            ptr,
            bytes,
        )?;
    }

    let sm_count = active_sm_count_for_gpu(gpu);
    let grid_x = extent
        .padded_rows
        .min(sm_count.saturating_mul(4).max(96))
        .max(1);
    let block_x = rmsnorm_block_x(args.cols);

    KernelLaunch::new(gpu, kernel)
        .grid([grid_x, 1, 1])
        .block([block_x, 1, 1])
        .arg_ptr(args.input)
        .arg_ptr(args.weight)
        .arg_ptr(args.packed_out)
        .arg_ptr(args.scale_out)
        .arg_f32(scale2)
        .arg_f32(args.eps)
        .arg_f32(args.weight_mode.weight_offset())
        .arg_u32(args.rows)
        .arg_u32(args.cols)
        .arg_u32(args.scale_layout.as_kernel_arg())
        .launch(stream)
}

/// Launch `rms_norm_residual_nvfp4_sm120` (in-place `hidden += residual` +
/// RMSNorm -> block-16 NVFP4 quantization in one global pass).
pub fn launch_rms_norm_residual_nvfp4_sm120(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    args: &RmsNormResidualNvfp4LaunchArgs,
    stream: u64,
) -> Result<()> {
    const NAME: &str = "rms_norm_residual_nvfp4_sm120";
    ensure!(kernel.0 != 0, "{NAME} kernel handle is missing");
    ensure!(
        args.eps.is_finite() && args.eps > 0.0,
        "{NAME} requires finite positive eps; got {}",
        args.eps
    );
    let scale2 = validate_global_scale(args.global_scale, NAME)?;
    let extent = nvfp4_activation_extent(args.rows, args.cols, args.scale_layout)?;
    let mat_bytes = bf16_matrix_bytes(args.rows, args.cols, "hidden/residual")?;
    let weight_bytes = bf16_matrix_bytes(1, args.cols, "weight")?;

    ensure!(
        !args.hidden.is_null()
            && !args.residual.is_null()
            && !args.weight.is_null()
            && !args.packed_out.is_null()
            && !args.scale_out.is_null(),
        "{NAME} received a null device pointer"
    );
    ensure!(
        args.hidden.0.is_multiple_of(16)
            && args.residual.0.is_multiple_of(16)
            && args.weight.0.is_multiple_of(16),
        "{NAME} requires 16-byte aligned BF16 hidden, residual, and weight pointers"
    );
    ensure!(
        args.packed_out.0.is_multiple_of(8),
        "{NAME} requires 8-byte aligned packed_out pointer"
    );
    ensure_disjoint_ranges(
        "hidden",
        args.hidden,
        mat_bytes,
        "residual",
        args.residual,
        mat_bytes,
    )?;
    ensure_disjoint_ranges(
        "hidden",
        args.hidden,
        mat_bytes,
        "weight",
        args.weight,
        weight_bytes,
    )?;
    ensure_disjoint_ranges(
        "packed_out",
        args.packed_out,
        extent.packed_bytes,
        "scale_out",
        args.scale_out,
        extent.scale_bytes,
    )?;
    for (label, ptr, bytes) in [
        ("hidden", args.hidden, mat_bytes),
        ("residual", args.residual, mat_bytes),
        ("weight", args.weight, weight_bytes),
    ] {
        ensure_disjoint_ranges(
            "packed_out",
            args.packed_out,
            extent.packed_bytes,
            label,
            ptr,
            bytes,
        )?;
        ensure_disjoint_ranges(
            "scale_out",
            args.scale_out,
            extent.scale_bytes,
            label,
            ptr,
            bytes,
        )?;
    }

    let sm_count = active_sm_count_for_gpu(gpu);
    let grid_x = extent
        .padded_rows
        .min(sm_count.saturating_mul(4).max(96))
        .max(1);
    let block_x = rmsnorm_block_x(args.cols);

    KernelLaunch::new(gpu, kernel)
        .grid([grid_x, 1, 1])
        .block([block_x, 1, 1])
        .arg_ptr(args.hidden)
        .arg_ptr(args.residual)
        .arg_ptr(args.weight)
        .arg_ptr(args.packed_out)
        .arg_ptr(args.scale_out)
        .arg_f32(scale2)
        .arg_f32(args.eps)
        .arg_f32(args.weight_mode.weight_offset())
        .arg_u32(args.rows)
        .arg_u32(args.cols)
        .arg_u32(args.scale_layout.as_kernel_arg())
        .launch(stream)
}

/// Launch `silu_mul_nvfp4_sm120` (fused SwiGLU `silu(gate) * up` -> block-16
/// NVFP4 quantization).
pub fn launch_silu_mul_nvfp4_sm120(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    args: &SiluMulNvfp4LaunchArgs,
    stream: u64,
) -> Result<()> {
    const NAME: &str = "silu_mul_nvfp4_sm120";
    ensure!(kernel.0 != 0, "{NAME} kernel handle is missing");
    let scale2 = validate_global_scale(args.global_scale, NAME)?;
    let extent = nvfp4_activation_extent(args.rows, args.cols, args.scale_layout)?;

    ensure!(
        !args.gate.is_null()
            && !args.up.is_null()
            && !args.packed_out.is_null()
            && !args.scale_out.is_null(),
        "{NAME} received a null device pointer"
    );
    ensure!(
        args.gate.0.is_multiple_of(16) && args.up.0.is_multiple_of(16),
        "{NAME} requires 16-byte aligned BF16 gate and up pointers"
    );
    ensure!(
        args.packed_out.0.is_multiple_of(8),
        "{NAME} requires 8-byte aligned packed_out pointer"
    );

    let row_stride_elems = match args.input_layout {
        SwiGluInputLayout::Split => {
            let proj_bytes = bf16_matrix_bytes(args.rows, args.cols, "gate/up")?;
            for (label, ptr) in [("gate", args.gate), ("up", args.up)] {
                ensure_disjoint_ranges(
                    "packed_out",
                    args.packed_out,
                    extent.packed_bytes,
                    label,
                    ptr,
                    proj_bytes,
                )?;
                ensure_disjoint_ranges(
                    "scale_out",
                    args.scale_out,
                    extent.scale_bytes,
                    label,
                    ptr,
                    proj_bytes,
                )?;
            }
            args.cols
        }
        SwiGluInputLayout::MergedGateUp => {
            let row_stride = args
                .cols
                .checked_mul(2)
                .context("{NAME} merged row stride overflow")?;
            let expected_up_offset = (args.cols as usize)
                .checked_mul(2)
                .context("{NAME} merged up offset overflow")?;
            let expected_up = args
                .gate
                .0
                .checked_add(u64::try_from(expected_up_offset).context("up offset exceeds u64")?)
                .context("{NAME} merged up pointer overflow")?;
            if args.up.0 != expected_up {
                bail!(
                    "{NAME} MergedGateUp requires up == gate + {} bytes ({:#x}); got {:#x}",
                    expected_up_offset,
                    expected_up,
                    args.up.0
                );
            }
            let merged_bytes = bf16_matrix_bytes(args.rows, row_stride, "merged_gate_up")?;
            ensure_disjoint_ranges(
                "packed_out",
                args.packed_out,
                extent.packed_bytes,
                "merged_gate_up",
                args.gate,
                merged_bytes,
            )?;
            ensure_disjoint_ranges(
                "scale_out",
                args.scale_out,
                extent.scale_bytes,
                "merged_gate_up",
                args.gate,
                merged_bytes,
            )?;
            row_stride
        }
    };
    ensure_disjoint_ranges(
        "packed_out",
        args.packed_out,
        extent.packed_bytes,
        "scale_out",
        args.scale_out,
        extent.scale_bytes,
    )?;

    let sm_count = active_sm_count_for_gpu(gpu);
    let grid_x = extent
        .padded_rows
        .min(sm_count.saturating_mul(2).max(96))
        .max(1);

    KernelLaunch::new(gpu, kernel)
        .grid([grid_x, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(args.gate)
        .arg_ptr(args.up)
        .arg_ptr(args.packed_out)
        .arg_ptr(args.scale_out)
        .arg_f32(scale2)
        .arg_u32(args.rows)
        .arg_u32(args.cols)
        .arg_u32(row_stride_elems)
        .arg_u32(args.scale_layout.as_kernel_arg())
        .launch(stream)
}

// ── Exact CPU Mathematical Reference ──────────────────────────────────────────

/// Convert IEEE-754 `f32` to BF16 bits using round-to-nearest-even (matching
/// CUDA `__float2bfloat16`).
#[must_use]
pub fn nvfp4_f32_to_bf16_rne(value: f32) -> u16 {
    let bits = value.to_bits();
    let exp = (bits >> 23) & 0xff;
    let mant = bits & 0x007f_ffff;
    if exp == 0xff && mant != 0 {
        // Canonical quiet NaN payload in BF16.
        return ((bits >> 16) as u16) | 0x0040;
    }
    let lsb = (bits >> 16) & 1;
    let rounding_bias = 0x7fffu32 + lsb;
    (bits.wrapping_add(rounding_bias) >> 16) as u16
}

/// Convert BF16 bits to IEEE-754 `f32` (exact bit shift, matching `__bfloat162float`).
#[must_use]
pub fn nvfp4_bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

/// Encode a non-negative `f32` into unsigned E4M3 (`e4m3fn`, clamped to `[0, 448]`)
/// with exact IEEE-754 round-to-nearest-even, matching CUDA `__nv_fp8_e4m3`.
#[must_use]
pub fn encode_ue4m3_rne(value: f32) -> u8 {
    if !value.is_finite() || value <= 0.0 {
        return 0;
    }
    let clamped = value.min(448.0);
    if clamped < 0.015_625 {
        // Subnormal range [0, 2^-6): step is 2^-9 = 1/512.
        let scaled = clamped * 512.0;
        let floor = scaled.floor();
        let frac = scaled - floor;
        let base = floor as u8;
        let rounded = if frac > 0.5 || (frac == 0.5 && (base & 1) == 1) {
            base + 1
        } else {
            base
        };
        return if rounded >= 8 { 0x08 } else { rounded };
    }
    let bits = clamped.to_bits();
    let f32_exp = ((bits >> 23) & 0xff) as i32 - 127;
    let f32_mant = bits & 0x007f_ffff;
    let mut exp = (f32_exp + 7) as u8;
    let mut mant = (f32_mant >> 20) as u8;
    let rem = f32_mant & 0x000f_ffff;
    if rem > 0x0008_0000 || (rem == 0x0008_0000 && (mant & 1) == 1) {
        mant += 1;
        if mant == 8 {
            mant = 0;
            exp += 1;
        }
    }
    if exp > 15 || (exp == 15 && mant >= 7) {
        return 0x7e; // 448.0
    }
    (exp << 3) | mant
}

/// Decode an `e4m3fn` byte back to `f32` (exact, matching `static_cast<float>(__nv_fp8_e4m3)`).
#[must_use]
pub fn decode_ue4m3(byte: u8) -> f32 {
    let sign = (byte >> 7) & 1;
    let exp = u32::from((byte >> 3) & 0x0f);
    let mant = u32::from(byte & 0x07);
    let mag = if exp == 0 {
        (mant as f32) * (1.0 / 512.0)
    } else if exp == 15 && mant == 7 {
        0.0
    } else {
        f32::from_bits(((exp + 120) << 23) | (mant << 20))
    };
    if sign != 0 { -mag } else { mag }
}

/// Branchless 2-op SWAR canonicalizer mapping `-0.0` (`0x8`) nibbles to `+0.0` (`0x0`)
/// across 8 packed E2M1 nibbles, matching `canonicalize_e2m1x8_neg_zero` in CUDA.
#[must_use]
pub const fn canonicalize_e2m1x8_neg_zero(packed: u32) -> u32 {
    let neg_zero = packed & !((packed & 0x7777_7777) + 0x7777_7777) & 0x8888_8888;
    packed ^ neg_zero
}

/// Encode one `f32` value into a 4-bit E2M1 nibble using exact IEEE-754
/// round-to-nearest-even with `-0.0` (`0x8`) canonicalized to `+0.0` (`0x0`)
/// and non-finite inputs clamped to `0x0`.
#[must_use]
pub fn quantize_e2m1_rne_canonical(value: f32) -> u8 {
    if !value.is_finite() {
        return 0;
    }
    let a = value.abs();
    let idx: u8 = if a <= 0.25 {
        0
    } else if a < 0.75 {
        1
    } else if a <= 1.25 {
        2
    } else if a < 1.75 {
        3
    } else if a <= 2.5 {
        4
    } else if a < 3.5 {
        5
    } else if a <= 5.0 {
        6
    } else {
        7
    };
    if idx == 0 {
        0
    } else if value.is_sign_negative() {
        0x8 | idx
    } else {
        idx
    }
}

/// Decode one 4-bit E2M1 nibble to its exact `f32` value.
#[must_use]
pub const fn dequantize_e2m1(nibble: u8) -> f32 {
    const LUT: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    let mag = LUT[(nibble & 0x7) as usize];
    if (nibble & 0x8) != 0 { -mag } else { mag }
}

fn sanitize_finite_f32(x: f32) -> f32 {
    if x.is_finite() { x } else { 0.0 }
}

fn cpu_quantize_row_groups(
    row_values: &[f32],
    row: u32,
    groups: u32,
    global_scale: f32,
    layout: Nvfp4ScaleLayout,
    packed_out: &mut [u8],
    scale_out: &mut [u8],
) -> Result<()> {
    let cols = (groups * RMSNORM_NVFP4_GROUP_SIZE) as usize;
    let inv_scale2_over_6 = 1.0_f32 / (global_scale * 6.0_f32);
    let row_packed_base = (row as usize) * (cols / 2);

    for g in 0..groups {
        let base = (g * RMSNORM_NVFP4_GROUP_SIZE) as usize;
        let slice = &row_values[base..base + 16];
        let mut group_max = 0.0_f32;
        for &v in slice {
            let a = v.abs();
            if a > group_max {
                group_max = a;
            }
        }
        let fp8_raw = if group_max.is_finite() && group_max > 0.0 {
            (group_max * inv_scale2_over_6).min(448.0)
        } else {
            0.0
        };
        let scale_byte = encode_ue4m3_rne(fp8_raw);
        let scale_idx = nvfp4_scale_offset(row, g, groups, layout)?;
        scale_out[scale_idx] = scale_byte;

        let decoded = if scale_byte == 0 {
            0.0
        } else {
            decode_ue4m3(scale_byte)
        };
        let effective_scale = decoded * global_scale;
        let inv_eff = if effective_scale.is_finite() && effective_scale > 0.0 {
            1.0_f32 / effective_scale
        } else {
            0.0
        };

        let mut words = [0u32; 2];
        for (word_idx, word_slot) in words.iter_mut().enumerate() {
            let mut word = 0u32;
            for pair in 0..4 {
                let idx0 = word_idx * 8 + pair * 2;
                let n0 = u32::from(quantize_e2m1_rne_canonical(slice[idx0] * inv_eff));
                let n1 = u32::from(quantize_e2m1_rne_canonical(slice[idx0 + 1] * inv_eff));
                let byte = (n1 << 4) | n0;
                word |= byte << (pair * 8);
            }
            *word_slot = canonicalize_e2m1x8_neg_zero(word);
        }
        let dst = row_packed_base + (g as usize) * 8;
        packed_out[dst..dst + 4].copy_from_slice(&words[0].to_le_bytes());
        packed_out[dst + 4..dst + 8].copy_from_slice(&words[1].to_le_bytes());
    }
    Ok(())
}

/// Exact CPU reference for `rms_norm_nvfp4_sm120`.
///
/// Returns `(packed_e2m1, ue4m3_scales)` matching the CUDA kernel's math,
/// `-0.0` canonicalization, `NaN`/zero-row clamp, global scale, weight mode,
/// and scale layout.
#[expect(
    clippy::too_many_arguments,
    reason = "CPU reference matching CUDA kernel operands for parity testing"
)]
pub fn cpu_rms_norm_nvfp4(
    input_bf16: &[u16],
    weight_bf16: &[u16],
    rows: u32,
    cols: u32,
    global_scale: Option<f32>,
    eps: f32,
    weight_mode: RmsNormWeightMode,
    layout: Nvfp4ScaleLayout,
) -> Result<(Vec<u8>, Vec<u8>)> {
    let scale2 = validate_global_scale(global_scale, "cpu_rms_norm_nvfp4")?;
    ensure!(eps.is_finite() && eps > 0.0, "eps must be finite positive");
    let extent = nvfp4_activation_extent(rows, cols, layout)?;
    let expected_elems = (rows as usize)
        .checked_mul(cols as usize)
        .context("input element count overflow")?;
    ensure!(
        input_bf16.len() == expected_elems,
        "input_bf16 len {} != rows*cols {}",
        input_bf16.len(),
        expected_elems
    );
    ensure!(
        weight_bf16.len() == cols as usize,
        "weight_bf16 len {} != cols {}",
        weight_bf16.len(),
        cols
    );

    let mut packed_out = vec![0u8; extent.packed_bytes];
    let mut scale_out = vec![0u8; extent.scale_bytes];
    let groups = cols / RMSNORM_NVFP4_GROUP_SIZE;
    let weight_offset = weight_mode.weight_offset();
    let mut normed_row = vec![0.0_f32; cols as usize];

    for row in 0..rows {
        let base = (row as usize) * (cols as usize);
        let x_row = &input_bf16[base..base + cols as usize];
        let mut sum_sq = 0.0_f32;
        for &bits in x_row {
            let x = nvfp4_bf16_to_f32(bits);
            sum_sq += x * x;
        }
        let denom = if sum_sq.is_finite() && sum_sq >= 0.0 {
            sum_sq / (cols as f32) + eps
        } else {
            0.0
        };
        let rms = if denom.is_finite() && denom > 0.0 {
            1.0_f32 / denom.sqrt()
        } else {
            0.0
        };
        for c in 0..(cols as usize) {
            let x = sanitize_finite_f32(nvfp4_bf16_to_f32(x_row[c]));
            let w = sanitize_finite_f32(nvfp4_bf16_to_f32(weight_bf16[c]));
            normed_row[c] = sanitize_finite_f32((x * rms) * (weight_offset + w));
        }
        cpu_quantize_row_groups(
            &normed_row,
            row,
            groups,
            scale2,
            layout,
            &mut packed_out,
            &mut scale_out,
        )?;
    }
    Ok((packed_out, scale_out))
}

/// Exact CPU reference for `rms_norm_residual_nvfp4_sm120`.
///
/// Returns `(updated_hidden_bf16, packed_e2m1, ue4m3_scales)`.
#[expect(
    clippy::too_many_arguments,
    reason = "CPU reference matching CUDA kernel operands for parity testing"
)]
pub fn cpu_rms_norm_residual_nvfp4(
    hidden_bf16: &[u16],
    residual_bf16: &[u16],
    weight_bf16: &[u16],
    rows: u32,
    cols: u32,
    global_scale: Option<f32>,
    eps: f32,
    weight_mode: RmsNormWeightMode,
    layout: Nvfp4ScaleLayout,
) -> Result<(Vec<u16>, Vec<u8>, Vec<u8>)> {
    let expected_elems = (rows as usize)
        .checked_mul(cols as usize)
        .context("hidden element count overflow")?;
    ensure!(
        hidden_bf16.len() == expected_elems && residual_bf16.len() == expected_elems,
        "hidden/residual lengths must equal rows*cols ({expected_elems})"
    );
    let mut updated_hidden = Vec::with_capacity(expected_elems);
    for (&h_bits, &r_bits) in hidden_bf16.iter().zip(residual_bf16.iter()) {
        let sum = nvfp4_bf16_to_f32(h_bits) + nvfp4_bf16_to_f32(r_bits);
        updated_hidden.push(nvfp4_f32_to_bf16_rne(sum));
    }
    let (packed, scales) = cpu_rms_norm_nvfp4(
        &updated_hidden,
        weight_bf16,
        rows,
        cols,
        global_scale,
        eps,
        weight_mode,
        layout,
    )?;
    Ok((updated_hidden, packed, scales))
}

/// Exact CPU reference for `silu_mul_nvfp4_sm120` (both split and merged layouts).
///
/// For [`SwiGluInputLayout::MergedGateUp`], pass the full `[rows, 2 * cols]`
/// slice as `gate_bf16` and an empty slice for `up_bf16`.
pub fn cpu_silu_mul_nvfp4(
    gate_bf16: &[u16],
    up_bf16: &[u16],
    rows: u32,
    cols: u32,
    global_scale: Option<f32>,
    input_layout: SwiGluInputLayout,
    scale_layout: Nvfp4ScaleLayout,
) -> Result<(Vec<u8>, Vec<u8>)> {
    let scale2 = validate_global_scale(global_scale, "cpu_silu_mul_nvfp4")?;
    let extent = nvfp4_activation_extent(rows, cols, scale_layout)?;
    let groups = cols / RMSNORM_NVFP4_GROUP_SIZE;

    match input_layout {
        SwiGluInputLayout::Split => {
            let expected = (rows as usize)
                .checked_mul(cols as usize)
                .context("split SwiGLU element count overflow")?;
            ensure!(
                gate_bf16.len() == expected && up_bf16.len() == expected,
                "split gate/up lengths must equal rows*cols ({expected})"
            );
        }
        SwiGluInputLayout::MergedGateUp => {
            let expected = (rows as usize)
                .checked_mul(cols as usize)
                .and_then(|v| v.checked_mul(2))
                .context("merged SwiGLU element count overflow")?;
            ensure!(
                gate_bf16.len() == expected,
                "merged gate_up length {} != rows*2*cols {}",
                gate_bf16.len(),
                expected
            );
        }
    }

    let mut packed_out = vec![0u8; extent.packed_bytes];
    let mut scale_out = vec![0u8; extent.scale_bytes];
    let mut act_row = vec![0.0_f32; cols as usize];

    for row in 0..rows {
        let (g_row, u_row) = match input_layout {
            SwiGluInputLayout::Split => {
                let base = (row as usize) * (cols as usize);
                (
                    &gate_bf16[base..base + cols as usize],
                    &up_bf16[base..base + cols as usize],
                )
            }
            SwiGluInputLayout::MergedGateUp => {
                let base = (row as usize) * (2 * cols as usize);
                (
                    &gate_bf16[base..base + cols as usize],
                    &gate_bf16[base + cols as usize..base + 2 * cols as usize],
                )
            }
        };
        for c in 0..(cols as usize) {
            let g = nvfp4_bf16_to_f32(g_row[c]);
            let u = nvfp4_bf16_to_f32(u_row[c]);
            act_row[c] = if !g.is_finite() || !u.is_finite() {
                0.0
            } else {
                let denom = 1.0_f32 + (-g).exp();
                let sig = if denom.is_finite() { g / denom } else { 0.0 };
                sanitize_finite_f32(sig * u)
            };
        }
        cpu_quantize_row_groups(
            &act_row,
            row,
            groups,
            scale2,
            scale_layout,
            &mut packed_out,
            &mut scale_out,
        )?;
    }
    Ok((packed_out, scale_out))
}

#[cfg(test)]
mod tests {
    use std::ffi::OsString;
    use std::os::unix::ffi::OsStringExt;

    use spark_runtime::gpu::mock::MockGpuBackend;

    use super::*;

    fn bf16_vec(vals: &[f32]) -> Vec<u16> {
        vals.iter().copied().map(nvfp4_f32_to_bf16_rne).collect()
    }

    fn dequant_row(
        packed: &[u8],
        scales: &[u8],
        row: u32,
        cols: u32,
        global_scale: f32,
        layout: Nvfp4ScaleLayout,
    ) -> Vec<f32> {
        let groups = cols / RMSNORM_NVFP4_GROUP_SIZE;
        let row_base = (row as usize) * (cols as usize / 2);
        let mut out = Vec::with_capacity(cols as usize);
        for g in 0..groups {
            let s_idx = nvfp4_scale_offset(row, g, groups, layout).unwrap();
            let eff_scale = decode_ue4m3(scales[s_idx]) * global_scale;
            for b in 0..8usize {
                let byte = packed[row_base + (g as usize) * 8 + b];
                out.push(dequantize_e2m1(byte & 0x0f) * eff_scale);
                out.push(dequantize_e2m1((byte >> 4) & 0x0f) * eff_scale);
            }
        }
        out
    }

    #[test]
    fn env_parser_defaults_off_and_fails_closed_on_invalid_or_non_utf8() {
        assert_eq!(
            parse_fused_rmsnorm_nvfp4_env(Err(std::env::VarError::NotPresent)).unwrap(),
            FusedRmsNormNvfp4Policy::Disabled
        );
        assert_eq!(
            parse_fused_rmsnorm_nvfp4_env(Ok("0".to_string())).unwrap(),
            FusedRmsNormNvfp4Policy::Disabled
        );
        assert_eq!(
            parse_fused_rmsnorm_nvfp4_env(Ok("1".to_string())).unwrap(),
            FusedRmsNormNvfp4Policy::Enabled
        );
        for invalid in ["", " ", "true", "false", "2", "yes", "1 "] {
            assert!(
                parse_fused_rmsnorm_nvfp4_env(Ok(invalid.to_string())).is_err(),
                "invalid env value {invalid:?} must fail closed"
            );
        }
        let non_utf8 = OsString::from_vec(vec![0xff, 0xfe]);
        assert!(
            parse_fused_rmsnorm_nvfp4_env(Err(std::env::VarError::NotUnicode(non_utf8))).is_err(),
            "non-UTF-8 env value must fail closed"
        );
    }

    #[test]
    fn neg_zero_canonicalization_clears_only_0x8_nibbles() {
        // Test every nibble value 0x0..=0xF in all 8 nibble positions.
        for nib in 0u32..16 {
            let mut word = 0u32;
            for pos in 0..8 {
                word |= nib << (pos * 4);
            }
            let canon = canonicalize_e2m1x8_neg_zero(word);
            let expected_nib = if nib == 0x8 { 0x0 } else { nib };
            let mut expected = 0u32;
            for pos in 0..8 {
                expected |= expected_nib << (pos * 4);
            }
            assert_eq!(canon, expected, "nibble {nib:#x} mismatch");
        }
        // Mixed word with adjacent 0x8 and 0x7/0xF neighbors must have zero nibble carry.
        let mixed = 0x8F87_8081u32;
        assert_eq!(canonicalize_e2m1x8_neg_zero(mixed), 0x0F07_0001u32);

        // Scalar RNE encoder maps negative values in [-0.25, -0.0] to +0.0 (0x0), never 0x8.
        for &v in &[
            -0.0_f32,
            -1e-7,
            -0.1,
            -0.25,
            0.0,
            0.25,
            f32::NAN,
            f32::INFINITY,
        ] {
            assert_eq!(
                quantize_e2m1_rne_canonical(v),
                0x0,
                "value {v} must encode to canonical +0.0"
            );
        }
        // Exact RNE ties-to-even midpoints:
        assert_eq!(quantize_e2m1_rne_canonical(0.75), 0x2); // ties to 1.0 (even)
        assert_eq!(quantize_e2m1_rne_canonical(-0.75), 0xa); // ties to -1.0 (even)
        assert_eq!(quantize_e2m1_rne_canonical(1.25), 0x2); // ties to 1.0 (even)
        assert_eq!(quantize_e2m1_rne_canonical(1.75), 0x4); // ties to 2.0 (even)
        assert_eq!(quantize_e2m1_rne_canonical(2.5), 0x4); // ties to 2.0 (even)
        assert_eq!(quantize_e2m1_rne_canonical(3.5), 0x6); // ties to 4.0 (even)
        assert_eq!(quantize_e2m1_rne_canonical(5.0), 0x6); // ties to 4.0 (even)
        assert_eq!(quantize_e2m1_rne_canonical(5.01), 0x7); // 6.0
        assert_eq!(quantize_e2m1_rne_canonical(-100.0), 0xf); // -6.0 saturated
    }

    #[test]
    fn ue4m3_rne_roundtrip_handles_zero_subnormals_normals_and_clamp() {
        assert_eq!(encode_ue4m3_rne(0.0), 0x00);
        assert_eq!(encode_ue4m3_rne(-1.0), 0x00);
        assert_eq!(encode_ue4m3_rne(f32::NAN), 0x00);
        assert_eq!(encode_ue4m3_rne(f32::INFINITY), 0x00);
        assert_eq!(encode_ue4m3_rne(1.0 / 512.0), 0x01);
        assert_eq!(decode_ue4m3(0x01), 1.0 / 512.0);
        assert_eq!(encode_ue4m3_rne(1.0), 0x38);
        assert_eq!(decode_ue4m3(0x38), 1.0);
        assert_eq!(encode_ue4m3_rne(448.0), 0x7e);
        assert_eq!(encode_ue4m3_rne(9999.0), 0x7e);
        assert_eq!(decode_ue4m3(0x7e), 448.0);
        // Every non-NaN positive UE4M3 encoding must round-trip losslessly.
        for byte in 0u8..=0x7e {
            let v = decode_ue4m3(byte);
            assert_eq!(
                encode_ue4m3_rne(v),
                byte,
                "UE4M3 byte {byte:#x} failed roundtrip"
            );
        }
    }

    #[test]
    fn cpu_rms_norm_nvfp4_matches_exact_math_and_clamps_nan_and_zero_rows() {
        let rows = 3u32;
        let cols = 64u32;
        let eps = 1e-6_f32;
        let global_scale = 0.25_f32;
        let weight = bf16_vec(
            &(0..cols)
                .map(|c| ((c as f32) * 0.01) - 0.3)
                .collect::<Vec<_>>(),
        );

        let mut input = vec![0.0_f32; (rows * cols) as usize];
        // Row 0: non-trivial signed activations including tiny negatives that hit -0.0.
        for c in 0..cols as usize {
            input[c] = if c % 16 == 0 {
                -1e-4
            } else {
                ((c as f32) - 31.5) * 0.25
            };
        }
        // Row 1: exact zero row (including -0.0).
        for c in 0..cols as usize {
            input[cols as usize + c] = if c % 2 == 0 { -0.0 } else { 0.0 };
        }
        // Row 2: contains a NaN -> entire row must clamp cleanly to 0x00 scales & 0x00 nibbles.
        for c in 0..cols as usize {
            input[2 * cols as usize + c] = 1.5;
        }
        input[2 * cols as usize + 17] = f32::NAN;
        let input_bf16 = bf16_vec(&input);

        for (weight_mode, layout) in [
            (RmsNormWeightMode::UnitOffset, Nvfp4ScaleLayout::Linear),
            (RmsNormWeightMode::Direct, Nvfp4ScaleLayout::Swizzled128x4),
        ] {
            let (packed, scales) = cpu_rms_norm_nvfp4(
                &input_bf16,
                &weight,
                rows,
                cols,
                Some(global_scale),
                eps,
                weight_mode,
                layout,
            )
            .unwrap();

            // Row 0: verify reconstruction accuracy and zero `-0.0` (0x8) nibbles.
            for &b in &packed {
                assert_ne!(b & 0x0f, 0x8, "low nibble must never be -0.0");
                assert_ne!((b >> 4) & 0x0f, 0x8, "high nibble must never be -0.0");
            }
            let recon0 = dequant_row(&packed, &scales, 0, cols, global_scale, layout);
            let x0: Vec<f32> = input_bf16[..cols as usize]
                .iter()
                .map(|&b| nvfp4_bf16_to_f32(b))
                .collect();
            let mean_sq = x0.iter().map(|v| v * v).sum::<f32>() / (cols as f32);
            let rms = 1.0 / (mean_sq + eps).sqrt();
            let mut dot = 0.0_f64;
            let mut norm_e = 0.0_f64;
            let mut norm_r = 0.0_f64;
            let groups = cols / RMSNORM_NVFP4_GROUP_SIZE;
            for g in 0..groups {
                let s_idx = nvfp4_scale_offset(0, g, groups, layout).unwrap();
                let eff_scale = decode_ue4m3(scales[s_idx]) * global_scale;
                for i in 0..16usize {
                    let c = (g as usize) * 16 + i;
                    let w = weight_mode.weight_offset() + nvfp4_bf16_to_f32(weight[c]);
                    let expected = x0[c] * rms * w;
                    let err = (recon0[c] - expected).abs();
                    // Maximum E2M1 quantization gap is between 4.0 and 6.0 (half-step = 1.0 * eff_scale).
                    assert!(
                        err <= eff_scale * 1.0001 + 1e-5,
                        "element {c} error {err} exceeded E2M1 half-step bound {eff_scale} for {weight_mode:?}/{layout:?}"
                    );
                    dot += f64::from(expected) * f64::from(recon0[c]);
                    norm_e += f64::from(expected) * f64::from(expected);
                    norm_r += f64::from(recon0[c]) * f64::from(recon0[c]);
                }
            }
            let cos_sim = dot / (norm_e.sqrt() * norm_r.sqrt());
            assert!(
                cos_sim > 0.99,
                "NVFP4 cosine similarity too low for {weight_mode:?}/{layout:?}: {cos_sim}"
            );

            // Row 1 (zero) and Row 2 (NaN) must produce all-zero bytes and scales.
            for zero_row in [1u32, 2u32] {
                let p_base = (zero_row as usize) * (cols as usize / 2);
                assert!(
                    packed[p_base..p_base + cols as usize / 2]
                        .iter()
                        .all(|&b| b == 0)
                );
                for g in 0..(cols / 16) {
                    let s_idx = nvfp4_scale_offset(zero_row, g, cols / 16, layout).unwrap();
                    assert_eq!(scales[s_idx], 0);
                }
            }
            // Padding rows in Swizzled128x4 must be zeroed.
            if layout == Nvfp4ScaleLayout::Swizzled128x4 {
                for pad_row in rows..128 {
                    for g in 0..(cols / 16) {
                        let s_idx = nvfp4_scale_offset(pad_row, g, cols / 16, layout).unwrap();
                        assert_eq!(scales[s_idx], 0);
                    }
                }
            }
        }
    }

    #[test]
    fn cpu_rms_norm_residual_matches_two_step_residual_add_then_rms_norm_bit_for_bit() {
        let rows = 2u32;
        let cols = 64u32;
        let hidden = bf16_vec(
            &(0..(rows * cols))
                .map(|i| ((i as f32) * 0.13).sin() * 2.0)
                .collect::<Vec<_>>(),
        );
        let residual = bf16_vec(
            &(0..(rows * cols))
                .map(|i| ((i as f32) * 0.29).cos() * 1.5)
                .collect::<Vec<_>>(),
        );
        let weight = bf16_vec(&(0..cols).map(|i| (i as f32) * 0.005).collect::<Vec<_>>());

        let (updated_h, packed_fused, scales_fused) = cpu_rms_norm_residual_nvfp4(
            &hidden,
            &residual,
            &weight,
            rows,
            cols,
            Some(0.5),
            1e-6,
            RmsNormWeightMode::UnitOffset,
            Nvfp4ScaleLayout::Swizzled128x4,
        )
        .unwrap();

        let expected_h: Vec<u16> = hidden
            .iter()
            .zip(residual.iter())
            .map(|(&h, &r)| nvfp4_f32_to_bf16_rne(nvfp4_bf16_to_f32(h) + nvfp4_bf16_to_f32(r)))
            .collect();
        assert_eq!(updated_h, expected_h);

        let (packed_ref, scales_ref) = cpu_rms_norm_nvfp4(
            &expected_h,
            &weight,
            rows,
            cols,
            Some(0.5),
            1e-6,
            RmsNormWeightMode::UnitOffset,
            Nvfp4ScaleLayout::Swizzled128x4,
        )
        .unwrap();
        assert_eq!(packed_fused, packed_ref);
        assert_eq!(scales_fused, scales_ref);
    }

    #[test]
    fn cpu_silu_mul_nvfp4_split_and_merged_match_bit_for_bit() {
        let rows = 3u32;
        let cols = 64u32;
        let gate_f32: Vec<f32> = (0..(rows * cols))
            .map(|i| ((i as f32) * 0.17 - 8.0) * 0.4)
            .collect();
        let up_f32: Vec<f32> = (0..(rows * cols))
            .map(|i| ((i as f32) * 0.23).cos() * 3.0)
            .collect();
        let gate = bf16_vec(&gate_f32);
        let up = bf16_vec(&up_f32);

        let mut merged = Vec::with_capacity((rows * cols * 2) as usize);
        for r in 0..rows as usize {
            let base = r * cols as usize;
            merged.extend_from_slice(&gate[base..base + cols as usize]);
            merged.extend_from_slice(&up[base..base + cols as usize]);
        }

        for layout in [Nvfp4ScaleLayout::Linear, Nvfp4ScaleLayout::Swizzled128x4] {
            let (p_split, s_split) = cpu_silu_mul_nvfp4(
                &gate,
                &up,
                rows,
                cols,
                Some(0.125),
                SwiGluInputLayout::Split,
                layout,
            )
            .unwrap();
            let (p_merged, s_merged) = cpu_silu_mul_nvfp4(
                &merged,
                &[],
                rows,
                cols,
                Some(0.125),
                SwiGluInputLayout::MergedGateUp,
                layout,
            )
            .unwrap();
            assert_eq!(p_split, p_merged);
            assert_eq!(s_split, s_merged);
            for &b in &p_split {
                assert_ne!(b & 0x0f, 0x8);
                assert_ne!((b >> 4) & 0x0f, 0x8);
            }
        }

        // Extreme negative gate (-100.0 -> exp(+100) = +inf), NaN, and Inf inputs must clamp cleanly to 0.
        let mut extreme_gate = vec![0.0_f32; 64];
        let mut extreme_up = vec![1.0_f32; 64];
        extreme_gate[0] = -100.0;
        extreme_gate[1] = f32::NAN;
        extreme_gate[2] = f32::INFINITY;
        extreme_up[3] = f32::NEG_INFINITY;
        extreme_gate[4] = -1e-4;
        extreme_up[4] = 1.0;
        let (p_ext, s_ext) = cpu_silu_mul_nvfp4(
            &bf16_vec(&extreme_gate),
            &bf16_vec(&extreme_up),
            1,
            64,
            None,
            SwiGluInputLayout::Split,
            Nvfp4ScaleLayout::Linear,
        )
        .unwrap();
        assert_eq!(s_ext[0], 0);
        assert!(p_ext[..8].iter().all(|&b| b == 0));
    }

    #[test]
    fn launchers_validate_pointers_alignment_overlap_and_dispatch_on_mock_gpu() {
        let gpu = MockGpuBackend::new();
        let h_in = DevicePtr(0x0010_0000);
        let h_res = DevicePtr(0x0020_0000);
        let h_w = DevicePtr(0x0030_0000);
        let h_packed = DevicePtr(0x0040_0000);
        let h_scale = DevicePtr(0x0050_0000);

        let rms_args = RmsNormNvfp4LaunchArgs {
            input: h_in,
            weight: h_w,
            packed_out: h_packed,
            scale_out: h_scale,
            global_scale: Some(0.25),
            eps: 1e-6,
            weight_mode: RmsNormWeightMode::UnitOffset,
            rows: 32,
            cols: 64,
            scale_layout: Nvfp4ScaleLayout::Swizzled128x4,
        };
        launch_rms_norm_nvfp4_sm120(&gpu, KernelHandle(1), &rms_args, 0x11).unwrap();

        let res_args = RmsNormResidualNvfp4LaunchArgs {
            hidden: h_in,
            residual: h_res,
            weight: h_w,
            packed_out: h_packed,
            scale_out: h_scale,
            global_scale: None,
            eps: 1e-5,
            weight_mode: RmsNormWeightMode::Direct,
            rows: 16,
            cols: 64,
            scale_layout: Nvfp4ScaleLayout::Linear,
        };
        launch_rms_norm_residual_nvfp4_sm120(&gpu, KernelHandle(2), &res_args, 0x11).unwrap();

        let merged_silu_args = SiluMulNvfp4LaunchArgs {
            gate: h_in,
            up: h_in.offset(64 * 2),
            packed_out: h_packed,
            scale_out: h_scale,
            global_scale: Some(0.5),
            rows: 32,
            cols: 64,
            input_layout: SwiGluInputLayout::MergedGateUp,
            scale_layout: Nvfp4ScaleLayout::Swizzled128x4,
        };
        launch_silu_mul_nvfp4_sm120(&gpu, KernelHandle(3), &merged_silu_args, 0x11).unwrap();
        assert_eq!(gpu.launch_count(), 3);

        // Fail-closed validation checks: zero handle, misaligned ptr, overlap, bad eps/scale, bad merged up.
        assert!(launch_rms_norm_nvfp4_sm120(&gpu, KernelHandle(0), &rms_args, 0).is_err());
        assert!(
            launch_rms_norm_nvfp4_sm120(
                &gpu,
                KernelHandle(1),
                &RmsNormNvfp4LaunchArgs {
                    input: DevicePtr(0x0010_0008),
                    ..rms_args
                },
                0,
            )
            .is_err()
        );
        assert!(
            launch_rms_norm_nvfp4_sm120(
                &gpu,
                KernelHandle(1),
                &RmsNormNvfp4LaunchArgs {
                    global_scale: Some(0.0),
                    ..rms_args
                },
                0,
            )
            .is_err()
        );
        assert!(
            launch_rms_norm_nvfp4_sm120(
                &gpu,
                KernelHandle(1),
                &RmsNormNvfp4LaunchArgs {
                    eps: f32::NAN,
                    ..rms_args
                },
                0,
            )
            .is_err()
        );
        assert!(
            launch_rms_norm_residual_nvfp4_sm120(
                &gpu,
                KernelHandle(2),
                &RmsNormResidualNvfp4LaunchArgs {
                    residual: h_in,
                    ..res_args
                },
                0,
            )
            .is_err()
        );
        assert!(
            launch_silu_mul_nvfp4_sm120(
                &gpu,
                KernelHandle(3),
                &SiluMulNvfp4LaunchArgs {
                    up: h_res,
                    ..merged_silu_args
                },
                0,
            )
            .is_err()
        );
        assert!(
            launch_silu_mul_nvfp4_sm120(
                &gpu,
                KernelHandle(3),
                &SiluMulNvfp4LaunchArgs {
                    cols: 32,
                    up: h_in.offset(32 * 2),
                    scale_layout: Nvfp4ScaleLayout::Swizzled128x4,
                    ..merged_silu_args
                },
                0,
            )
            .is_err()
        );
        // No invalid call may enqueue a kernel.
        assert_eq!(gpu.launch_count(), 3);
    }

    #[test]
    fn cuda_source_contains_required_entrypoints_and_sm120_ptx_contract() {
        let cuda = include_str!("../../../../../kernels/gb10/common/rms_norm_nvfp4_sm120.cu");
        for entry in [
            "rms_norm_nvfp4_sm120(",
            "rms_norm_residual_nvfp4_sm120(",
            "silu_mul_nvfp4_sm120(",
        ] {
            assert!(cuda.contains(entry), "missing CUDA entry {entry}");
        }
        assert!(cuda.contains("cvt.rn.satfinite.e2m1x2.f32"));
        assert!(cuda.contains("__CUDA_ARCH__ >= 1200"));
        assert!(cuda.contains("canonicalize_e2m1x8_neg_zero"));
    }
}
