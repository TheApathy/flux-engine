// SPDX-License-Identifier: AGPL-3.0-only

//! Scalar Tensor-FP8 (E4M3FN) prefill quantization, `[0.125, 8.0]` bounded
//! FP32 epilogue deferral (`TensorFp8EpilogueDeferral`), and the opt-in
//! `ATLAS_GB10_FAST_PREFILL=1` umbrella policy.
//!
//! CUDA entrypoints live in `kernels/gb10/common/tensor_fp8_quant_sm120.cu`
//! (auto-discovered by `atlas-kernels` under module `"tensor_fp8_quant_sm120"`):
//! - `tensor_fp8_absmax_bf16_sm120`: global finite absmax over `[rows, cols]` BF16.
//! - `bf16_to_tensor_fp8_sm120`: BF16 -> strict `[-448.0, 448.0]` E4M3FN with
//!   zero `0x7F`/`0xFF` NaN bytes and device scale writeback.
//! - `rms_norm_tensor_fp8_sm120`: fused RMSNorm -> explicit `cvt.rn.bf16.f32`
//!   BF16 boundary rounding -> scalar Tensor-FP8 E4M3FN.
//!
//! Opt-in gates (both default OFF for byte-identical baseline):
//! - `ATLAS_TENSOR_FP8_PREFILL=1`: explicitly enables scalar Tensor-FP8 prefill
//!   projections (fails closed when required kernels are absent).
//! - `ATLAS_GB10_FAST_PREFILL=1`: opt-in GB10 fast prefill umbrella enabling
//!   SM12.1 cuBLASLt NVFP4 FFN/projections, fused RMSNorm->NVFP4, scalar
//!   Tensor-FP8 prefill, `10 MiB` L2 GDN windowing, and fast conv1d when each
//!   individual knob is unset (explicit `=0` on any sub-knob always disables it).

use std::sync::{Mutex, OnceLock};

use anyhow::{Context, Result, bail, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

use super::RmsNormWeightMode;

/// Auto-discovered `atlas-kernels` module name for
/// `kernels/gb10/common/tensor_fp8_quant_sm120.cu`.
pub const TENSOR_FP8_QUANT_MODULE: &str = "tensor_fp8_quant_sm120";

/// Opt-in gate enabling scalar Tensor-FP8 prefill projections (`0` or `1`, default OFF).
pub const TENSOR_FP8_PREFILL_ENV_VAR: &str = "ATLAS_TENSOR_FP8_PREFILL";

/// Opt-in umbrella gate for the GB10 fast prefill stack (`0` or `1`, default OFF).
pub const GB10_FAST_PREFILL_ENV_VAR: &str = "ATLAS_GB10_FAST_PREFILL";

/// Maximum finite magnitude representable in FP8 E4M3FN (`0x7E` = `448.0`).
pub const TENSOR_FP8_E4M3_MAX: f32 = 448.0;

/// Minimum dynamic tensor scale floor to prevent division by zero on near-zero tensors.
pub const TENSOR_FP8_MIN_SCALE: f32 = 1.0e-12;

/// Minimum partition scale eligible for FP32 epilogue deferral (`2^-3 = 0.125`).
pub const TENSOR_FP8_DEFERRAL_MIN_SCALE: f32 = 0.125;

/// Maximum partition scale eligible for FP32 epilogue deferral (`2^+3 = 8.0`).
pub const TENSOR_FP8_DEFERRAL_MAX_SCALE: f32 = 8.0;

/// Required `N` and `K` multiple for `cublaslt::fp8_gemm_device_scaled`.
pub const TENSOR_FP8_GEMM_ALIGN: u32 = 16;

const TENSOR_FP8_BLOCK_THREADS: u32 = 256;
const TENSOR_FP8_ABSMAX_MAX_BLOCKS: u32 = 256;
const TENSOR_FP8_QUANT_MAX_BLOCKS: u32 = 65_535;
const TENSOR_FP8_SCALAR_SCRATCH_BYTES: usize = 64;
const TENSOR_FP8_ABSMAX_OFF: usize = 0;
const TENSOR_FP8_ACT_SCALE_OFF: usize = 4;
const TENSOR_FP8_UNIT_WEIGHT_SCALE_OFF: usize = 8;
const TENSOR_FP8_GB10_FENCE_OFF: usize = 12;

/// Runtime policy resolved from `ATLAS_GB10_FAST_PREFILL`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Gb10FastPrefillPolicy {
    /// Default OFF: keep all GB10 prefill defaults byte-identical.
    Disabled,
    /// Explicit `ATLAS_GB10_FAST_PREFILL=1`: engage the GB10 fast prefill stack
    /// on all sub-knobs that are not explicitly disabled with `=0`.
    Enabled,
}

/// Strictly parse `ATLAS_GB10_FAST_PREFILL` (`NotPresent` or `"0"` -> `Disabled`,
/// `"1"` -> `Enabled`, any other UTF-8 value or `NotUnicode` -> `Err`).
pub fn parse_gb10_fast_prefill_env(
    value: std::result::Result<String, std::env::VarError>,
) -> std::result::Result<Gb10FastPrefillPolicy, &'static str> {
    match value {
        Err(std::env::VarError::NotPresent) => Ok(Gb10FastPrefillPolicy::Disabled),
        Err(std::env::VarError::NotUnicode(_)) => {
            Err("ATLAS_GB10_FAST_PREFILL must be valid UTF-8 and exactly 0 or 1")
        }
        Ok(raw) => match raw.as_str() {
            "0" => Ok(Gb10FastPrefillPolicy::Disabled),
            "1" => Ok(Gb10FastPrefillPolicy::Enabled),
            _ => Err("ATLAS_GB10_FAST_PREFILL must be exactly 0 or 1"),
        },
    }
}

/// Read and cache `ATLAS_GB10_FAST_PREFILL` fail-closed.
pub fn gb10_fast_prefill_policy() -> Result<Gb10FastPrefillPolicy> {
    static GATE: OnceLock<std::result::Result<Gb10FastPrefillPolicy, &'static str>> =
        OnceLock::new();
    (*GATE.get_or_init(|| parse_gb10_fast_prefill_env(std::env::var(GB10_FAST_PREFILL_ENV_VAR))))
        .map_err(anyhow::Error::msg)
}

/// Infallible convenience wrapper over [`gb10_fast_prefill_policy`] for pure
/// policy sites: malformed or non-UTF-8 values log a warning once and fail
/// closed to `false` (disabled).
#[must_use]
pub fn gb10_fast_prefill_enabled() -> bool {
    match gb10_fast_prefill_policy() {
        Ok(Gb10FastPrefillPolicy::Enabled) => true,
        Ok(Gb10FastPrefillPolicy::Disabled) => false,
        Err(err) => {
            static WARNED: std::sync::Once = std::sync::Once::new();
            WARNED.call_once(|| {
                tracing::warn!("{err}; failing closed with {GB10_FAST_PREFILL_ENV_VAR}=0");
            });
            false
        }
    }
}

/// Runtime policy resolved from `ATLAS_TENSOR_FP8_PREFILL` together with
/// `ATLAS_GB10_FAST_PREFILL`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TensorFp8PrefillPolicy {
    /// Disabled (unset with `ATLAS_GB10_FAST_PREFILL` off, or explicit `=0`).
    Disabled,
    /// Auto-enabled via `ATLAS_GB10_FAST_PREFILL=1` while
    /// `ATLAS_TENSOR_FP8_PREFILL` is unset: engages whenever kernels, shape,
    /// and deferral invariants are satisfied, and falls back cleanly otherwise.
    AutoGb10FastPrefill,
    /// Explicitly enabled via `ATLAS_TENSOR_FP8_PREFILL=1`: fails closed if
    /// required kernels are missing on an eligible projection.
    ExplicitEnabled,
}

impl TensorFp8PrefillPolicy {
    /// True when scalar Tensor-FP8 prefill is active (either explicit or auto).
    #[must_use]
    pub const fn is_enabled(self) -> bool {
        !matches!(self, Self::Disabled)
    }

    /// True when missing kernels or scratch capture conflicts must return an
    /// error rather than falling back to the legacy FP8 GEMM kernel.
    #[must_use]
    pub const fn is_fail_closed(self) -> bool {
        matches!(self, Self::ExplicitEnabled)
    }
}

/// Strictly parse `ATLAS_TENSOR_FP8_PREFILL` with `ATLAS_GB10_FAST_PREFILL` cascade.
///
/// - Explicit `"0"` ALWAYS disables even when `gb10_fast_prefill` is `true`.
/// - Explicit `"1"` returns `ExplicitEnabled`.
/// - `NotPresent` returns `AutoGb10FastPrefill` if `gb10_fast_prefill`, else `Disabled`.
/// - Any other UTF-8 value or `NotUnicode` returns `Err(...)` (fail-closed).
pub fn parse_tensor_fp8_prefill_env(
    value: std::result::Result<String, std::env::VarError>,
    gb10_fast_prefill: bool,
) -> std::result::Result<TensorFp8PrefillPolicy, &'static str> {
    match value {
        Err(std::env::VarError::NotPresent) => Ok(if gb10_fast_prefill {
            TensorFp8PrefillPolicy::AutoGb10FastPrefill
        } else {
            TensorFp8PrefillPolicy::Disabled
        }),
        Err(std::env::VarError::NotUnicode(_)) => {
            Err("ATLAS_TENSOR_FP8_PREFILL must be valid UTF-8 and exactly 0 or 1")
        }
        Ok(raw) => match raw.as_str() {
            "0" => Ok(TensorFp8PrefillPolicy::Disabled),
            "1" => Ok(TensorFp8PrefillPolicy::ExplicitEnabled),
            _ => Err("ATLAS_TENSOR_FP8_PREFILL must be exactly 0 or 1"),
        },
    }
}

/// Read and cache the effective `ATLAS_TENSOR_FP8_PREFILL` policy fail-closed.
pub fn tensor_fp8_prefill_policy() -> Result<TensorFp8PrefillPolicy> {
    static GATE: OnceLock<std::result::Result<TensorFp8PrefillPolicy, &'static str>> =
        OnceLock::new();
    let gb10_fast = gb10_fast_prefill_policy()?;
    (*GATE.get_or_init(|| {
        parse_tensor_fp8_prefill_env(
            std::env::var(TENSOR_FP8_PREFILL_ENV_VAR),
            gb10_fast == Gb10FastPrefillPolicy::Enabled,
        )
    }))
    .map_err(anyhow::Error::msg)
}

/// Resolved CUDA kernel handles for `tensor_fp8_quant_sm120.cu`.
#[derive(Debug, Clone, Copy)]
pub struct TensorFp8QuantKernels {
    pub absmax_bf16: KernelHandle,
    pub bf16_to_tensor_fp8: KernelHandle,
    pub rms_norm_tensor_fp8: KernelHandle,
}

impl TensorFp8QuantKernels {
    /// Probe all 3 optional entrypoints from `"tensor_fp8_quant_sm120"`.
    #[must_use]
    pub fn load_optional(gpu: &dyn GpuBackend) -> Self {
        Self {
            absmax_bf16: crate::layers::try_kernel(
                gpu,
                TENSOR_FP8_QUANT_MODULE,
                "tensor_fp8_absmax_bf16_sm120",
            ),
            bf16_to_tensor_fp8: crate::layers::try_kernel(
                gpu,
                TENSOR_FP8_QUANT_MODULE,
                "bf16_to_tensor_fp8_sm120",
            ),
            rms_norm_tensor_fp8: crate::layers::try_kernel(
                gpu,
                TENSOR_FP8_QUANT_MODULE,
                "rms_norm_tensor_fp8_sm120",
            ),
        }
    }

    /// True when dynamic BF16 -> Tensor-FP8 quantization (`absmax` + `bf16_to_tensor_fp8`)
    /// is loaded.
    #[must_use]
    pub const fn has_bf16_dynamic(&self) -> bool {
        self.absmax_bf16.0 != 0 && self.bf16_to_tensor_fp8.0 != 0
    }

    /// True when `bf16_to_tensor_fp8_sm120` is loaded.
    #[must_use]
    pub const fn has_bf16_static(&self) -> bool {
        self.bf16_to_tensor_fp8.0 != 0
    }

    /// True when `rms_norm_tensor_fp8_sm120` is loaded.
    #[must_use]
    pub const fn has_rms_norm(&self) -> bool {
        self.rms_norm_tensor_fp8.0 != 0
    }

    /// True when all 3 entrypoints are loaded.
    #[must_use]
    pub const fn is_complete(&self) -> bool {
        self.has_bf16_dynamic() && self.has_rms_norm()
    }
}

/// Checked byte extent of a `[rows, cols]` BF16 buffer, bounded by `i64::MAX`
/// to match CUDA `int64_t` indexing.
pub fn tensor_fp8_bf16_bytes(rows: u32, cols: u32) -> Result<usize> {
    ensure!(
        rows > 0 && cols > 0,
        "tensor_fp8 requires rows > 0 and cols > 0"
    );
    let elems = (rows as u64)
        .checked_mul(cols as u64)
        .context("tensor_fp8 element count overflow")?;
    let bytes_u64 = elems
        .checked_mul(2)
        .context("tensor_fp8 BF16 byte extent overflow")?;
    ensure!(
        i64::try_from(bytes_u64).is_ok(),
        "tensor_fp8 BF16 byte extent {bytes_u64} exceeds i64::MAX"
    );
    usize::try_from(bytes_u64).context("tensor_fp8 BF16 byte extent exceeds usize")
}

/// Checked byte extent of a `[rows, cols]` FP8 E4M3FN buffer, bounded by `i64::MAX`
/// to match CUDA `int64_t` indexing.
pub fn tensor_fp8_e4m3_bytes(rows: u32, cols: u32) -> Result<usize> {
    ensure!(
        rows > 0 && cols > 0,
        "tensor_fp8 requires rows > 0 and cols > 0"
    );
    let bytes_u64 = (rows as u64)
        .checked_mul(cols as u64)
        .context("tensor_fp8 E4M3 byte extent overflow")?;
    ensure!(
        i64::try_from(bytes_u64).is_ok(),
        "tensor_fp8 E4M3 byte extent {bytes_u64} exceeds i64::MAX"
    );
    usize::try_from(bytes_u64).context("tensor_fp8 E4M3 byte extent exceeds usize")
}

/// Verify that two device byte ranges `[a, a + a_bytes)` and `[b, b + b_bytes)`
/// do not overlap and do not wrap `u64`.
pub fn ensure_tensor_fp8_disjoint(
    a: DevicePtr,
    a_bytes: usize,
    a_label: &str,
    b: DevicePtr,
    b_bytes: usize,
    b_label: &str,
) -> Result<()> {
    let a_end =
        a.0.checked_add(a_bytes as u64)
            .with_context(|| format!("tensor_fp8 buffer {a_label} address range overflow"))?;
    let b_end =
        b.0.checked_add(b_bytes as u64)
            .with_context(|| format!("tensor_fp8 buffer {b_label} address range overflow"))?;
    ensure!(
        a_end <= b.0 || b_end <= a.0,
        "tensor_fp8 buffers {a_label} and {b_label} overlap (aliasing forbidden)"
    );
    Ok(())
}

/// Launch `tensor_fp8_absmax_bf16_sm120` over `[rows, cols]` BF16 input.
///
/// Caller must zero `d_absmax_out` (4 bytes) prior to launch.
pub fn launch_tensor_fp8_absmax_bf16_sm120(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input_bf16: DevicePtr,
    d_absmax_out: DevicePtr,
    rows: u32,
    cols: u32,
    stream: u64,
) -> Result<()> {
    const NAME: &str = "tensor_fp8_absmax_bf16_sm120";
    ensure!(kernel.0 != 0, "{NAME}: null kernel handle");
    ensure!(!input_bf16.is_null(), "{NAME}: input_bf16 is NULL");
    ensure!(!d_absmax_out.is_null(), "{NAME}: d_absmax_out is NULL");
    ensure!(
        input_bf16.0.is_multiple_of(2),
        "{NAME}: input_bf16 must be 2-byte aligned"
    );
    ensure!(
        d_absmax_out.0.is_multiple_of(4),
        "{NAME}: d_absmax_out must be 4-byte aligned"
    );
    let in_bytes = tensor_fp8_bf16_bytes(rows, cols)?;
    ensure_tensor_fp8_disjoint(
        input_bf16,
        in_bytes,
        "input_bf16",
        d_absmax_out,
        4,
        "d_absmax_out",
    )?;

    let total_elems = (rows as u64) * (cols as u64);
    let work_units = if (total_elems & 3) == 0 && input_bf16.0.is_multiple_of(8) {
        total_elems / 4
    } else {
        total_elems
    };
    let blocks = work_units
        .div_ceil(u64::from(TENSOR_FP8_BLOCK_THREADS))
        .clamp(1, u64::from(TENSOR_FP8_ABSMAX_MAX_BLOCKS)) as u32;

    KernelLaunch::new(gpu, kernel)
        .grid([blocks, 1, 1])
        .block([TENSOR_FP8_BLOCK_THREADS, 1, 1])
        .arg_ptr(input_bf16)
        .arg_ptr(d_absmax_out)
        .arg_u32(rows)
        .arg_u32(cols)
        .launch(stream)
}

/// Validated launch arguments for [`launch_bf16_to_tensor_fp8_sm120`].
#[derive(Debug, Clone, Copy)]
pub struct Bf16ToTensorFp8LaunchArgs {
    pub input_bf16: DevicePtr,
    pub out_fp8: DevicePtr,
    /// Optional device FP32 absmax pointer (`DevicePtr::NULL` uses `static_scale`).
    pub d_absmax_in: DevicePtr,
    /// Optional device FP32 scale output (`DevicePtr::NULL` skips writeback).
    pub d_scale_out: DevicePtr,
    /// Static tensor scale used when `d_absmax_in` is `DevicePtr::NULL`.
    pub static_scale: f32,
    pub rows: u32,
    pub cols: u32,
}

/// Launch `bf16_to_tensor_fp8_sm120` with strict non-aliasing and scale validation.
pub fn launch_bf16_to_tensor_fp8_sm120(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    args: &Bf16ToTensorFp8LaunchArgs,
    stream: u64,
) -> Result<()> {
    const NAME: &str = "bf16_to_tensor_fp8_sm120";
    ensure!(kernel.0 != 0, "{NAME}: null kernel handle");
    ensure!(!args.input_bf16.is_null(), "{NAME}: input_bf16 is NULL");
    ensure!(!args.out_fp8.is_null(), "{NAME}: out_fp8 is NULL");
    ensure!(
        args.input_bf16.0.is_multiple_of(2),
        "{NAME}: input_bf16 must be 2-byte aligned"
    );
    if args.d_absmax_in.is_null() {
        ensure!(
            args.static_scale.is_finite() && args.static_scale > 0.0,
            "{NAME}: static_scale must be finite and positive when d_absmax_in is NULL, got {}",
            args.static_scale
        );
    } else {
        ensure!(
            args.d_absmax_in.0.is_multiple_of(4),
            "{NAME}: d_absmax_in must be 4-byte aligned"
        );
    }
    if !args.d_scale_out.is_null() {
        ensure!(
            args.d_scale_out.0.is_multiple_of(4),
            "{NAME}: d_scale_out must be 4-byte aligned"
        );
    }

    let in_bytes = tensor_fp8_bf16_bytes(args.rows, args.cols)?;
    let out_bytes = tensor_fp8_e4m3_bytes(args.rows, args.cols)?;
    ensure_tensor_fp8_disjoint(
        args.input_bf16,
        in_bytes,
        "input_bf16",
        args.out_fp8,
        out_bytes,
        "out_fp8",
    )?;
    if !args.d_absmax_in.is_null() {
        ensure_tensor_fp8_disjoint(
            args.out_fp8,
            out_bytes,
            "out_fp8",
            args.d_absmax_in,
            4,
            "d_absmax_in",
        )?;
    }
    if !args.d_scale_out.is_null() {
        ensure_tensor_fp8_disjoint(
            args.input_bf16,
            in_bytes,
            "input_bf16",
            args.d_scale_out,
            4,
            "d_scale_out",
        )?;
        ensure_tensor_fp8_disjoint(
            args.out_fp8,
            out_bytes,
            "out_fp8",
            args.d_scale_out,
            4,
            "d_scale_out",
        )?;
        if !args.d_absmax_in.is_null() {
            ensure_tensor_fp8_disjoint(
                args.d_absmax_in,
                4,
                "d_absmax_in",
                args.d_scale_out,
                4,
                "d_scale_out",
            )?;
        }
    }

    let total_elems = (args.rows as u64) * (args.cols as u64);
    let work_units = if (total_elems & 3) == 0
        && args.input_bf16.0.is_multiple_of(8)
        && args.out_fp8.0.is_multiple_of(4)
    {
        total_elems / 4
    } else {
        total_elems
    };
    let blocks = work_units
        .div_ceil(u64::from(TENSOR_FP8_BLOCK_THREADS))
        .clamp(1, u64::from(TENSOR_FP8_QUANT_MAX_BLOCKS)) as u32;

    KernelLaunch::new(gpu, kernel)
        .grid([blocks, 1, 1])
        .block([TENSOR_FP8_BLOCK_THREADS, 1, 1])
        .arg_ptr(args.input_bf16)
        .arg_ptr(args.out_fp8)
        .arg_ptr(args.d_absmax_in)
        .arg_ptr(args.d_scale_out)
        .arg_f32(args.static_scale)
        .arg_u32(args.rows)
        .arg_u32(args.cols)
        .launch(stream)
}

/// Validated launch arguments for [`launch_rms_norm_tensor_fp8_sm120`].
#[derive(Debug, Clone, Copy)]
pub struct RmsNormTensorFp8LaunchArgs {
    pub input_bf16: DevicePtr,
    pub weight_bf16: DevicePtr,
    pub out_fp8: DevicePtr,
    /// Optional mirrored BF16 RMSNorm output (`DevicePtr::NULL` to omit).
    pub out_bf16_opt: DevicePtr,
    /// Optional device FP32 absmax (`DevicePtr::NULL` uses `static_scale`).
    pub d_absmax_in: DevicePtr,
    /// Optional device FP32 scale output (`DevicePtr::NULL` to omit).
    pub d_scale_out: DevicePtr,
    pub eps: f32,
    pub weight_mode: RmsNormWeightMode,
    pub static_scale: f32,
    pub rows: u32,
    pub cols: u32,
}

/// Launch `rms_norm_tensor_fp8_sm120` (fused RMSNorm -> explicit BF16 rounding
/// -> scalar Tensor-FP8 E4M3FN).
pub fn launch_rms_norm_tensor_fp8_sm120(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    args: &RmsNormTensorFp8LaunchArgs,
    stream: u64,
) -> Result<()> {
    const NAME: &str = "rms_norm_tensor_fp8_sm120";
    ensure!(kernel.0 != 0, "{NAME}: null kernel handle");
    ensure!(!args.input_bf16.is_null(), "{NAME}: input_bf16 is NULL");
    ensure!(!args.weight_bf16.is_null(), "{NAME}: weight_bf16 is NULL");
    ensure!(!args.out_fp8.is_null(), "{NAME}: out_fp8 is NULL");
    ensure!(
        args.input_bf16.0.is_multiple_of(2) && args.weight_bf16.0.is_multiple_of(2),
        "{NAME}: BF16 pointers must be 2-byte aligned"
    );
    ensure!(
        args.eps.is_finite() && args.eps > 0.0,
        "{NAME}: eps must be finite and strictly positive, got {}",
        args.eps
    );
    if args.d_absmax_in.is_null() {
        ensure!(
            args.static_scale.is_finite() && args.static_scale > 0.0,
            "{NAME}: static_scale must be finite and positive when d_absmax_in is NULL, got {}",
            args.static_scale
        );
    } else {
        ensure!(
            args.d_absmax_in.0.is_multiple_of(4),
            "{NAME}: d_absmax_in must be 4-byte aligned"
        );
    }
    if !args.d_scale_out.is_null() {
        ensure!(
            args.d_scale_out.0.is_multiple_of(4),
            "{NAME}: d_scale_out must be 4-byte aligned"
        );
    }

    let in_bytes = tensor_fp8_bf16_bytes(args.rows, args.cols)?;
    let weight_bytes = tensor_fp8_bf16_bytes(1, args.cols)?;
    let out_fp8_bytes = tensor_fp8_e4m3_bytes(args.rows, args.cols)?;
    ensure_tensor_fp8_disjoint(
        args.input_bf16,
        in_bytes,
        "input_bf16",
        args.out_fp8,
        out_fp8_bytes,
        "out_fp8",
    )?;
    ensure_tensor_fp8_disjoint(
        args.weight_bf16,
        weight_bytes,
        "weight_bf16",
        args.out_fp8,
        out_fp8_bytes,
        "out_fp8",
    )?;
    if !args.out_bf16_opt.is_null() {
        ensure!(
            args.out_bf16_opt.0.is_multiple_of(2),
            "{NAME}: out_bf16_opt must be 2-byte aligned"
        );
        ensure_tensor_fp8_disjoint(
            args.input_bf16,
            in_bytes,
            "input_bf16",
            args.out_bf16_opt,
            in_bytes,
            "out_bf16_opt",
        )?;
        ensure_tensor_fp8_disjoint(
            args.weight_bf16,
            weight_bytes,
            "weight_bf16",
            args.out_bf16_opt,
            in_bytes,
            "out_bf16_opt",
        )?;
        ensure_tensor_fp8_disjoint(
            args.out_fp8,
            out_fp8_bytes,
            "out_fp8",
            args.out_bf16_opt,
            in_bytes,
            "out_bf16_opt",
        )?;
    }
    if !args.d_absmax_in.is_null() {
        ensure_tensor_fp8_disjoint(
            args.out_fp8,
            out_fp8_bytes,
            "out_fp8",
            args.d_absmax_in,
            4,
            "d_absmax_in",
        )?;
        if !args.out_bf16_opt.is_null() {
            ensure_tensor_fp8_disjoint(
                args.out_bf16_opt,
                in_bytes,
                "out_bf16_opt",
                args.d_absmax_in,
                4,
                "d_absmax_in",
            )?;
        }
    }
    if !args.d_scale_out.is_null() {
        ensure_tensor_fp8_disjoint(
            args.input_bf16,
            in_bytes,
            "input_bf16",
            args.d_scale_out,
            4,
            "d_scale_out",
        )?;
        ensure_tensor_fp8_disjoint(
            args.weight_bf16,
            weight_bytes,
            "weight_bf16",
            args.d_scale_out,
            4,
            "d_scale_out",
        )?;
        ensure_tensor_fp8_disjoint(
            args.out_fp8,
            out_fp8_bytes,
            "out_fp8",
            args.d_scale_out,
            4,
            "d_scale_out",
        )?;
        if !args.out_bf16_opt.is_null() {
            ensure_tensor_fp8_disjoint(
                args.out_bf16_opt,
                in_bytes,
                "out_bf16_opt",
                args.d_scale_out,
                4,
                "d_scale_out",
            )?;
        }
        if !args.d_absmax_in.is_null() {
            ensure_tensor_fp8_disjoint(
                args.d_absmax_in,
                4,
                "d_absmax_in",
                args.d_scale_out,
                4,
                "d_scale_out",
            )?;
        }
    }

    KernelLaunch::new(gpu, kernel)
        .grid([args.rows, 1, 1])
        .block([TENSOR_FP8_BLOCK_THREADS, 1, 1])
        .arg_ptr(args.input_bf16)
        .arg_ptr(args.weight_bf16)
        .arg_ptr(args.out_fp8)
        .arg_ptr(args.out_bf16_opt)
        .arg_ptr(args.d_absmax_in)
        .arg_ptr(args.d_scale_out)
        .arg_f32(args.eps)
        .arg_f32(args.weight_mode.weight_offset())
        .arg_f32(args.static_scale)
        .arg_u32(args.rows)
        .arg_u32(args.cols)
        .launch(stream)
}

/// Reason why a partition scale cannot be deferred to the FP32 GEMM epilogue.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TensorFp8DeferralRefusal {
    /// Per-K-block scales are not uniform across the reduction dimension, so
    /// factoring a single scalar into the FP32 epilogue is algebraically invalid.
    NonUniformKBlocks,
    /// Partition scale is NaN, infinite, zero, or negative.
    NonFiniteOrNonPositiveScale,
    /// Partition scale is strictly below [`TENSOR_FP8_DEFERRAL_MIN_SCALE`] (`0.125`).
    ScaleBelowMin,
    /// Partition scale is strictly above [`TENSOR_FP8_DEFERRAL_MAX_SCALE`] (`8.0`).
    ScaleAboveMax,
}

/// Decision produced by [`TensorFp8EpilogueDeferral`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TensorFp8DeferralDecision {
    /// Safe to defer `epilogue_alpha` into the FP32 cuBLASLt GEMM epilogue:
    /// `per_block_k_uniform` holds and `epilogue_alpha` is in `[0.125, 8.0]`.
    DeferToEpilogue { epilogue_alpha: f32 },
    /// Deferral refused: caller must fold scales directly or use a block-scaled kernel.
    DirectRescale { reason: TensorFp8DeferralRefusal },
}

/// Bounded `[0.125, 8.0]` FP32 epilogue deferral guard for scalar Tensor-FP8
/// GEMMs (`cublaslt::fp8_gemm_device_scaled`).
///
/// Deferring a partition scale `s` into the FP32 GEMM epilogue (`alpha = s`)
/// is valid **if and only if**:
/// 1. `per_block_k_uniform == true`: every K-block along the reduction
///    dimension shares the same scale factor so `s` factors out of the dot
///    product exactly.
/// 2. `s` is finite and within `[0.125, 8.0]` (`[2^-3, 2^+3]`), bounding E4M3FN
///    dynamic-range shift to at most 3 exponent steps so operands neither
///    underflow into subnormals nor saturate at `448.0`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TensorFp8EpilogueDeferral {
    partition_scale: f32,
    per_block_k_uniform: bool,
    decision: TensorFp8DeferralDecision,
}

impl TensorFp8EpilogueDeferral {
    /// Evaluate epilogue deferral eligibility for a scalar `partition_scale`
    /// and explicit `per_block_k_uniform` proof.
    #[must_use]
    pub const fn new(partition_scale: f32, per_block_k_uniform: bool) -> Self {
        let decision = if !per_block_k_uniform {
            TensorFp8DeferralDecision::DirectRescale {
                reason: TensorFp8DeferralRefusal::NonUniformKBlocks,
            }
        } else if !partition_scale.is_finite() || partition_scale <= 0.0 {
            TensorFp8DeferralDecision::DirectRescale {
                reason: TensorFp8DeferralRefusal::NonFiniteOrNonPositiveScale,
            }
        } else if partition_scale < TENSOR_FP8_DEFERRAL_MIN_SCALE {
            TensorFp8DeferralDecision::DirectRescale {
                reason: TensorFp8DeferralRefusal::ScaleBelowMin,
            }
        } else if partition_scale > TENSOR_FP8_DEFERRAL_MAX_SCALE {
            TensorFp8DeferralDecision::DirectRescale {
                reason: TensorFp8DeferralRefusal::ScaleAboveMax,
            }
        } else {
            TensorFp8DeferralDecision::DeferToEpilogue {
                epilogue_alpha: partition_scale,
            }
        };
        Self {
            partition_scale,
            per_block_k_uniform,
            decision,
        }
    }

    /// Derive a [`TensorFp8EpilogueDeferral`] directly from a slice of per-K-block
    /// scales. Requires a non-empty slice where every entry is bit-identical
    /// (`per_block_k_uniform`) and lies inside `[0.125, 8.0]`.
    #[must_use]
    pub fn from_k_block_scales(k_block_scales: &[f32]) -> Self {
        let Some(&first) = k_block_scales.first() else {
            return Self::new(0.0, false);
        };
        let first_bits = first.to_bits();
        let uniform = k_block_scales
            .iter()
            .all(|&s| s.is_finite() && s.to_bits() == first_bits);
        Self::new(first, uniform)
    }

    /// Raw partition scale supplied at construction.
    #[must_use]
    pub const fn partition_scale(&self) -> f32 {
        self.partition_scale
    }

    /// Whether the `per_block_k_uniform` invariant was satisfied.
    #[must_use]
    pub const fn is_per_block_k_uniform(&self) -> bool {
        self.per_block_k_uniform
    }

    /// Resolved deferral decision.
    #[must_use]
    pub const fn decision(&self) -> TensorFp8DeferralDecision {
        self.decision
    }

    /// Returns `Some(epilogue_alpha)` iff deferral is admitted.
    #[must_use]
    pub const fn deferred_alpha(&self) -> Option<f32> {
        match self.decision {
            TensorFp8DeferralDecision::DeferToEpilogue { epilogue_alpha } => Some(epilogue_alpha),
            TensorFp8DeferralDecision::DirectRescale { .. } => None,
        }
    }

    /// Launch `spark_runtime::cublaslt::fp8_gemm_device_scaled` (`outer = false`)
    /// using the deferred FP32 epilogue `alpha`.
    ///
    /// Fails closed if [`Self::decision`] is `DirectRescale`, if any shape/pointer
    /// constraint is violated, or if `out_bf16` aliases any input buffer.
    pub fn launch_gemm_device_scaled(
        &self,
        gpu: &dyn GpuBackend,
        args: &TensorFp8GemmArgs,
        stream: u64,
    ) -> Result<()> {
        let epilogue_alpha = match self.decision {
            TensorFp8DeferralDecision::DeferToEpilogue { epilogue_alpha } => epilogue_alpha,
            TensorFp8DeferralDecision::DirectRescale { reason } => {
                bail!(
                    "TensorFp8EpilogueDeferral refused deferral (reason={reason:?}, scale={}, per_block_k_uniform={})",
                    self.partition_scale,
                    self.per_block_k_uniform
                );
            }
        };
        validate_tensor_fp8_gemm_args(args)?;

        // GB10 ordering fence: when a device-scale producer kernel immediately
        // precedes a scalar-scaled cuBLASLt FP8 GEMM, a 4-byte async memset on
        // a dedicated unused fence slot prevents GB10 CTAs from reading stale
        // device scale values across back-to-back calls.
        if let Some(fence_slot) = args.gb10_scale_fence_slot {
            ensure!(
                !fence_slot.is_null() && fence_slot.0.is_multiple_of(4),
                "tensor_fp8 GEMM GB10 fence slot must be non-null and 4-byte aligned"
            );
            ensure_tensor_fp8_disjoint(
                fence_slot,
                4,
                "gb10_scale_fence_slot",
                args.act_scale,
                4,
                "act_scale",
            )?;
            ensure_tensor_fp8_disjoint(
                fence_slot,
                4,
                "gb10_scale_fence_slot",
                args.weight_scale,
                4,
                "weight_scale",
            )?;
            gpu.memset_async(fence_slot, 0, 4, stream)?;
        }

        spark_runtime::cublaslt::fp8_gemm_device_scaled(
            args.act_e4m3.0,
            args.act_scale.0,
            args.weight_e4m3.0,
            args.weight_scale.0,
            args.out_bf16.0,
            args.m,
            args.n,
            args.k,
            args.ldc,
            epilogue_alpha,
            false,
            stream,
        )
    }
}

/// Validated operand bundle for [`TensorFp8EpilogueDeferral::launch_gemm_device_scaled`].
#[derive(Debug, Clone, Copy)]
pub struct TensorFp8GemmArgs {
    /// `[m, k]` row-major FP8 E4M3FN activation.
    pub act_e4m3: DevicePtr,
    /// `[1]` device FP32 scalar activation scale.
    pub act_scale: DevicePtr,
    /// `[n, k]` row-major FP8 E4M3FN weight.
    pub weight_e4m3: DevicePtr,
    /// `[1]` device FP32 scalar weight scale.
    pub weight_scale: DevicePtr,
    /// `[m, ldc]` row-major BF16 output (`ldc >= n`).
    pub out_bf16: DevicePtr,
    /// Optional 4-byte unused device slot for the GB10 scale-visibility fence.
    pub gb10_scale_fence_slot: Option<DevicePtr>,
    pub m: u32,
    pub n: u32,
    pub k: u32,
    pub ldc: u32,
}

/// Required `ldc` alignment in BF16 elements for cuBLASLt FP8 Tensor Core GEMM
/// (`8 * 2 B = 16 B`).
pub const TENSOR_FP8_GEMM_LDC_ALIGN: u32 = 8;

/// Required device-pointer alignment in bytes for cuBLASLt FP8 matrix operands
/// (`act_e4m3`, `weight_e4m3`, `out_bf16`).
pub const TENSOR_FP8_GEMM_PTR_ALIGN: u64 = 16;

/// True when the GPU compute capability supports native FP8 Tensor Core GEMM
/// (`SM89` Ada, `SM90` Hopper, `SM100`/`SM103`/`SM120`/`SM121` Blackwell).
///
/// `None` (mock backends without a CUDA device) returns `true` so unit tests
/// retain deterministic behavior.
#[must_use]
pub const fn supports_tensor_fp8_gemm(cc: Option<(u32, u32)>) -> bool {
    match cc {
        Some((major, minor)) => major > 8 || (major == 8 && minor >= 9),
        None => true,
    }
}

fn validate_tensor_fp8_gemm_args(args: &TensorFp8GemmArgs) -> Result<()> {
    ensure!(
        args.m > 0 && args.n > 0 && args.k > 0,
        "tensor_fp8 GEMM requires positive dimensions, got m={} n={} k={}",
        args.m,
        args.n,
        args.k
    );
    ensure!(
        args.n.is_multiple_of(TENSOR_FP8_GEMM_ALIGN)
            && args.k.is_multiple_of(TENSOR_FP8_GEMM_ALIGN)
            && args.ldc >= args.n
            && args.ldc.is_multiple_of(TENSOR_FP8_GEMM_LDC_ALIGN),
        "tensor_fp8 GEMM requires n,k % 16 == 0, ldc >= n, and ldc % 8 == 0; got m={} n={} k={} ldc={}",
        args.m,
        args.n,
        args.k,
        args.ldc
    );
    ensure!(
        !args.act_e4m3.is_null()
            && !args.act_scale.is_null()
            && !args.weight_e4m3.is_null()
            && !args.weight_scale.is_null()
            && !args.out_bf16.is_null(),
        "tensor_fp8 GEMM received a NULL device pointer"
    );
    ensure!(
        args.act_e4m3.0.is_multiple_of(TENSOR_FP8_GEMM_PTR_ALIGN)
            && args.weight_e4m3.0.is_multiple_of(TENSOR_FP8_GEMM_PTR_ALIGN)
            && args.out_bf16.0.is_multiple_of(TENSOR_FP8_GEMM_PTR_ALIGN)
            && args.act_scale.0.is_multiple_of(4)
            && args.weight_scale.0.is_multiple_of(4),
        "tensor_fp8 GEMM requires 16-byte alignment on act/weight/out and 4-byte alignment on scales"
    );

    let act_bytes = tensor_fp8_e4m3_bytes(args.m, args.k)?;
    let weight_bytes = tensor_fp8_e4m3_bytes(args.n, args.k)?;
    let out_bytes = tensor_fp8_bf16_bytes(args.m, args.ldc)?;
    ensure_tensor_fp8_disjoint(
        args.out_bf16,
        out_bytes,
        "out_bf16",
        args.act_e4m3,
        act_bytes,
        "act_e4m3",
    )?;
    ensure_tensor_fp8_disjoint(
        args.out_bf16,
        out_bytes,
        "out_bf16",
        args.weight_e4m3,
        weight_bytes,
        "weight_e4m3",
    )?;
    ensure_tensor_fp8_disjoint(
        args.out_bf16,
        out_bytes,
        "out_bf16",
        args.act_scale,
        4,
        "act_scale",
    )?;
    ensure_tensor_fp8_disjoint(
        args.out_bf16,
        out_bytes,
        "out_bf16",
        args.weight_scale,
        4,
        "weight_scale",
    )?;
    Ok(())
}

/// Per-`(gpu, stream)` scratch workspace for opt-in scalar Tensor-FP8 prefill
/// projections so concurrent streams never race on activation/scale buffers.
#[derive(Debug, Clone, Copy)]
struct TensorFp8PrefillScratch {
    act_e4m3: DevicePtr,
    act_capacity: usize,
    /// 64-byte scalar block:
    /// - `[0..4]`: `d_absmax` (f32)
    /// - `[4..8]`: `d_act_scale` (f32)
    /// - `[8..12]`: `d_unit_weight_scale` (constant `1.0f32`)
    /// - `[12..16]`: GB10 4-byte scale-ordering fence slot
    scalars: DevicePtr,
}

type TensorFp8ScratchKey = (usize, u64);
type TensorFp8ScratchMap = std::collections::HashMap<TensorFp8ScratchKey, TensorFp8PrefillScratch>;

fn tensor_fp8_scratch_lock() -> &'static Mutex<TensorFp8ScratchMap> {
    static SCRATCH: OnceLock<Mutex<TensorFp8ScratchMap>> = OnceLock::new();
    SCRATCH.get_or_init(|| Mutex::new(std::collections::HashMap::new()))
}

fn ensure_tensor_fp8_scratch(
    map: &mut TensorFp8ScratchMap,
    gpu: &dyn GpuBackend,
    needed_act_bytes: usize,
    stream: u64,
) -> Result<Option<(DevicePtr, DevicePtr, DevicePtr, DevicePtr, DevicePtr)>> {
    let aligned_need = needed_act_bytes
        .checked_add(255)
        .map(|v| v & !255)
        .context("tensor_fp8 activation scratch alignment overflow")?;
    let key = (gpu as *const dyn GpuBackend as *const () as usize, stream);

    let existing = map.get(&key).copied();
    let needs_alloc = match existing {
        Some(s) => s.act_capacity < aligned_need,
        None => true,
    };
    if needs_alloc {
        if gpu.stream_is_capturing(stream)
            || spark_runtime::gpu_capture_region::active_capture_region().is_some()
        {
            return Ok(None);
        }
        let (scalars, fresh_scalars) = match existing {
            Some(prev) => (prev.scalars, false),
            None => {
                let s = gpu
                    .alloc(TENSOR_FP8_SCALAR_SCRATCH_BYTES)
                    .context("allocate tensor_fp8 scalar scratch")?;
                let mut init = [0u8; TENSOR_FP8_SCALAR_SCRATCH_BYTES];
                init[TENSOR_FP8_UNIT_WEIGHT_SCALE_OFF..TENSOR_FP8_UNIT_WEIGHT_SCALE_OFF + 4]
                    .copy_from_slice(&1.0f32.to_le_bytes());
                if let Err(err) = gpu
                    .copy_h2d(&init, s)
                    .context("initialize tensor_fp8 scalar scratch")
                {
                    let _ = gpu.free(s);
                    return Err(err);
                }
                (s, true)
            }
        };
        let new_act = match gpu
            .alloc(aligned_need)
            .context("allocate tensor_fp8 activation scratch")
        {
            Ok(ptr) => ptr,
            Err(err) => {
                if fresh_scalars {
                    let _ = gpu.free(scalars);
                }
                return Err(err);
            }
        };
        if let Some(prev) = existing {
            // Synchronize stream BEFORE replacing/freeing `prev.act_e4m3`. On sync
            // failure, free `new_act` and preserve `prev` in `map` (zero leak).
            if let Err(err) = gpu.synchronize(stream) {
                let _ = gpu.free(new_act);
                return Err(err);
            }
            let _ = gpu.free(prev.act_e4m3);
        }
        map.insert(
            key,
            TensorFp8PrefillScratch {
                act_e4m3: new_act,
                act_capacity: aligned_need,
                scalars,
            },
        );
    }

    let scratch = map
        .get(&key)
        .context("tensor_fp8 scratch unexpectedly empty after allocation")?;
    Ok(Some((
        scratch.act_e4m3,
        scratch.scalars.offset(TENSOR_FP8_ABSMAX_OFF),
        scratch.scalars.offset(TENSOR_FP8_ACT_SCALE_OFF),
        scratch.scalars.offset(TENSOR_FP8_UNIT_WEIGHT_SCALE_OFF),
        scratch.scalars.offset(TENSOR_FP8_GB10_FENCE_OFF),
    )))
}

/// Pure eligibility predicate for scalar Tensor-FP8 prefill projections.
#[must_use]
pub const fn tensor_fp8_prefill_shape_ok(m: u32, n: u32, k: u32) -> bool {
    m > 0
        && n >= TENSOR_FP8_GEMM_ALIGN
        && k >= TENSOR_FP8_GEMM_ALIGN
        && n.is_multiple_of(TENSOR_FP8_GEMM_ALIGN)
        && k.is_multiple_of(TENSOR_FP8_GEMM_ALIGN)
}

/// Route a single pre-dequanted row-major `[n, k]` E4M3FN prefill projection
/// (`ssm_qkvz`, `ssm_out_proj`, `attn_q/k/v`, `attn_o_proj`) through
/// `tensor_fp8_quant_sm120` + [`TensorFp8EpilogueDeferral`] +
/// `spark_runtime::cublaslt::fp8_gemm_device_scaled`.
///
/// Returns `Ok(false)` immediately when `ATLAS_TENSOR_FP8_PREFILL` (and
/// `ATLAS_GB10_FAST_PREFILL`) is disabled, leaving the baseline byte-identical.
#[expect(
    clippy::too_many_arguments,
    reason = "projection dispatcher forwarding activation, weight, output, shape, deferral metadata, label, and stream"
)]
pub fn try_tensor_fp8_prefill_proj_single(
    gpu: &dyn GpuBackend,
    act_bf16: DevicePtr,
    weight_e4m3: DevicePtr,
    out_bf16: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    partition_scale: f32,
    per_block_k_uniform: bool,
    label: &'static str,
    stream: u64,
) -> Result<bool> {
    let policy = tensor_fp8_prefill_policy()?;
    if !policy.is_enabled() {
        return Ok(false);
    }
    let cc = gpu.compute_capability();
    if !supports_tensor_fp8_gemm(cc) {
        if policy.is_fail_closed() {
            bail!(
                "{TENSOR_FP8_PREFILL_ENV_VAR}=1 requires compute capability >= SM89, got {cc:?} ({label})"
            );
        }
        return Ok(false);
    }
    if !tensor_fp8_prefill_shape_ok(m, n, k)
        || act_bf16.is_null()
        || weight_e4m3.is_null()
        || out_bf16.is_null()
    {
        return Ok(false);
    }
    if !act_bf16.0.is_multiple_of(2)
        || !weight_e4m3.0.is_multiple_of(TENSOR_FP8_GEMM_PTR_ALIGN)
        || !out_bf16.0.is_multiple_of(TENSOR_FP8_GEMM_PTR_ALIGN)
    {
        if policy.is_fail_closed() {
            bail!(
                "{TENSOR_FP8_PREFILL_ENV_VAR}=1 requires 2B-aligned act_bf16 and 16B-aligned weight/out ({label})"
            );
        }
        return Ok(false);
    }
    let deferral = TensorFp8EpilogueDeferral::new(partition_scale, per_block_k_uniform);
    if deferral.deferred_alpha().is_none() {
        return Ok(false);
    }

    let ks = TensorFp8QuantKernels::load_optional(gpu);
    if !ks.has_bf16_dynamic() {
        if policy.is_fail_closed() {
            bail!(
                "{TENSOR_FP8_PREFILL_ENV_VAR}=1 requires {TENSOR_FP8_QUANT_MODULE} kernels ({label})"
            );
        }
        return Ok(false);
    }

    let act_bytes = tensor_fp8_e4m3_bytes(m, k)?;
    let mut guard = tensor_fp8_scratch_lock()
        .lock()
        .map_err(|_| anyhow::anyhow!("tensor_fp8 prefill scratch lock poisoned"))?;
    let Some((act_e4m3, d_absmax, d_act_scale, d_unit_weight_scale, fence_slot)) =
        ensure_tensor_fp8_scratch(&mut guard, gpu, act_bytes, stream)?
    else {
        if policy.is_fail_closed() {
            bail!(
                "{TENSOR_FP8_PREFILL_ENV_VAR}=1 cannot allocate activation scratch during CUDA graph capture ({label})"
            );
        }
        return Ok(false);
    };

    let gemm_args = TensorFp8GemmArgs {
        act_e4m3,
        act_scale: d_act_scale,
        weight_e4m3,
        weight_scale: d_unit_weight_scale,
        out_bf16,
        gb10_scale_fence_slot: Some(fence_slot),
        m,
        n,
        k,
        ldc: n,
    };
    // Preflight all GEMM pointer, alignment, and non-aliasing invariants BEFORE
    // issuing any quantizer launch so invalid outputs cannot mutate scratch.
    if let Err(err) = validate_tensor_fp8_gemm_args(&gemm_args) {
        if policy.is_fail_closed() {
            return Err(err);
        }
        return Ok(false);
    }

    gpu.memset_async(d_absmax, 0, 4, stream)?;
    launch_tensor_fp8_absmax_bf16_sm120(gpu, ks.absmax_bf16, act_bf16, d_absmax, m, k, stream)?;
    launch_bf16_to_tensor_fp8_sm120(
        gpu,
        ks.bf16_to_tensor_fp8,
        &Bf16ToTensorFp8LaunchArgs {
            input_bf16: act_bf16,
            out_fp8: act_e4m3,
            d_absmax_in: d_absmax,
            d_scale_out: d_act_scale,
            static_scale: 1.0,
            rows: m,
            cols: k,
        },
        stream,
    )?;

    deferral.launch_gemm_device_scaled(gpu, &gemm_args, stream)?;

    static LOGGED: std::sync::Once = std::sync::Once::new();
    LOGGED.call_once(|| {
        tracing::info!(
            label,
            m,
            n,
            k,
            policy = ?policy,
            epilogue_alpha = partition_scale,
            "ENGAGED ATLAS_TENSOR_FP8_PREFILL: scalar Tensor-FP8 projection via cublaslt::fp8_gemm_device_scaled"
        );
    });
    Ok(true)
}

/// Route Attention `q_proj`, `k_proj`, and `v_proj` pre-dequanted E4M3FN
/// prefill projections through a **single shared activation quantization**
/// (`tensor_fp8_quant_sm120`) + [`TensorFp8EpilogueDeferral`] +
/// `spark_runtime::cublaslt::fp8_gemm_device_scaled`.
#[expect(
    clippy::too_many_arguments,
    reason = "shared-activation Q/K/V Tensor-FP8 dispatcher forwarding three weight/output pairs"
)]
pub fn try_tensor_fp8_prefill_proj_qkv(
    gpu: &dyn GpuBackend,
    act_bf16: DevicePtr,
    q_weight_e4m3: DevicePtr,
    k_weight_e4m3: DevicePtr,
    v_weight_e4m3: DevicePtr,
    q_out: DevicePtr,
    k_out: DevicePtr,
    v_out: DevicePtr,
    m: u32,
    q_n: u32,
    kv_n: u32,
    k: u32,
    stream: u64,
) -> Result<bool> {
    let policy = tensor_fp8_prefill_policy()?;
    if !policy.is_enabled() {
        return Ok(false);
    }
    let cc = gpu.compute_capability();
    if !supports_tensor_fp8_gemm(cc) {
        if policy.is_fail_closed() {
            bail!(
                "{TENSOR_FP8_PREFILL_ENV_VAR}=1 requires compute capability >= SM89, got {cc:?} (attn_qkv)"
            );
        }
        return Ok(false);
    }
    if !tensor_fp8_prefill_shape_ok(m, q_n, k)
        || !tensor_fp8_prefill_shape_ok(m, kv_n, k)
        || act_bf16.is_null()
        || q_weight_e4m3.is_null()
        || k_weight_e4m3.is_null()
        || v_weight_e4m3.is_null()
        || q_out.is_null()
        || k_out.is_null()
        || v_out.is_null()
    {
        return Ok(false);
    }
    let aligned_ptrs = act_bf16.0.is_multiple_of(2)
        && [
            q_weight_e4m3,
            k_weight_e4m3,
            v_weight_e4m3,
            q_out,
            k_out,
            v_out,
        ]
        .into_iter()
        .all(|p| p.0.is_multiple_of(TENSOR_FP8_GEMM_PTR_ALIGN));
    if !aligned_ptrs {
        if policy.is_fail_closed() {
            bail!(
                "{TENSOR_FP8_PREFILL_ENV_VAR}=1 requires 2B-aligned act_bf16 and 16B-aligned Q/K/V weights and outputs (attn_qkv)"
            );
        }
        return Ok(false);
    }

    let deferral = TensorFp8EpilogueDeferral::new(1.0, true);
    let ks = TensorFp8QuantKernels::load_optional(gpu);
    if !ks.has_bf16_dynamic() {
        if policy.is_fail_closed() {
            bail!(
                "{TENSOR_FP8_PREFILL_ENV_VAR}=1 requires {TENSOR_FP8_QUANT_MODULE} kernels (attn_qkv)"
            );
        }
        return Ok(false);
    }

    let act_bytes = tensor_fp8_e4m3_bytes(m, k)?;
    let mut guard = tensor_fp8_scratch_lock()
        .lock()
        .map_err(|_| anyhow::anyhow!("tensor_fp8 prefill scratch lock poisoned"))?;
    let Some((act_e4m3, d_absmax, d_act_scale, d_unit_weight_scale, fence_slot)) =
        ensure_tensor_fp8_scratch(&mut guard, gpu, act_bytes, stream)?
    else {
        if policy.is_fail_closed() {
            bail!(
                "{TENSOR_FP8_PREFILL_ENV_VAR}=1 cannot allocate activation scratch during CUDA graph capture (attn_qkv)"
            );
        }
        return Ok(false);
    };

    let qkv_gemms = [
        TensorFp8GemmArgs {
            act_e4m3,
            act_scale: d_act_scale,
            weight_e4m3: q_weight_e4m3,
            weight_scale: d_unit_weight_scale,
            out_bf16: q_out,
            gb10_scale_fence_slot: Some(fence_slot),
            m,
            n: q_n,
            k,
            ldc: q_n,
        },
        TensorFp8GemmArgs {
            act_e4m3,
            act_scale: d_act_scale,
            weight_e4m3: k_weight_e4m3,
            weight_scale: d_unit_weight_scale,
            out_bf16: k_out,
            gb10_scale_fence_slot: None,
            m,
            n: kv_n,
            k,
            ldc: kv_n,
        },
        TensorFp8GemmArgs {
            act_e4m3,
            act_scale: d_act_scale,
            weight_e4m3: v_weight_e4m3,
            weight_scale: d_unit_weight_scale,
            out_bf16: v_out,
            gb10_scale_fence_slot: None,
            m,
            n: kv_n,
            k,
            ldc: kv_n,
        },
    ];
    // Preflight all three Q/K/V GEMM descriptors BEFORE mutating device state.
    for args in &qkv_gemms {
        if let Err(err) = validate_tensor_fp8_gemm_args(args) {
            if policy.is_fail_closed() {
                return Err(err);
            }
            return Ok(false);
        }
    }

    gpu.memset_async(d_absmax, 0, 4, stream)?;
    launch_tensor_fp8_absmax_bf16_sm120(gpu, ks.absmax_bf16, act_bf16, d_absmax, m, k, stream)?;
    launch_bf16_to_tensor_fp8_sm120(
        gpu,
        ks.bf16_to_tensor_fp8,
        &Bf16ToTensorFp8LaunchArgs {
            input_bf16: act_bf16,
            out_fp8: act_e4m3,
            d_absmax_in: d_absmax,
            d_scale_out: d_act_scale,
            static_scale: 1.0,
            rows: m,
            cols: k,
        },
        stream,
    )?;

    // Single GB10 ordering fence before Q; K and V reuse the same fenced scale.
    for args in &qkv_gemms {
        deferral.launch_gemm_device_scaled(gpu, args, stream)?;
    }

    static LOGGED_QKV: std::sync::Once = std::sync::Once::new();
    LOGGED_QKV.call_once(|| {
        tracing::info!(
            m,
            q_n,
            kv_n,
            k,
            policy = ?policy,
            "ENGAGED ATLAS_TENSOR_FP8_PREFILL: shared-quant Attention Q/K/V via cublaslt::fp8_gemm_device_scaled"
        );
    });
    Ok(true)
}

// ===========================================================================
// Bit-Exact CPU Mathematical References
// ===========================================================================

/// Convert raw BF16 bits to `f32`.
#[inline]
#[must_use]
pub fn bf16_bits_to_f32_cpu(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

/// Exact IEEE-754 round-to-nearest-even `f32 -> BF16` matching PTX `cvt.rn.bf16.f32`.
#[inline]
#[must_use]
pub fn f32_to_bf16_rn_bits_cpu(v: f32) -> u16 {
    let u = v.to_bits();
    let exp = u & 0x7F80_0000;
    let mant = u & 0x007F_FFFF;
    if exp == 0x7F80_0000 {
        if mant != 0 {
            return ((u >> 16) as u16) | 0x0040;
        }
        return (u >> 16) as u16;
    }
    let lsb = (u >> 16) & 1;
    let rounded = u.wrapping_add(0x0000_7FFF).wrapping_add(lsb);
    (rounded >> 16) as u16
}

/// Exact IEEE-754 round-to-nearest-even `f32 -> FP8 E4M3FN` with strict
/// `[-448.0, 448.0]` clamping and guaranteed zero `0x7F`/`0xFF` NaN bytes.
///
/// Matches `f32_to_e4m3_strict` / `cvt.rn.satfinite.e4m3x2.f32` + `scrub_e4m3_byte`
/// bit-for-bit across all finite, subnormal, extreme, signed-zero, and NaN/Inf inputs.
#[must_use]
pub fn f32_to_e4m3fn_strict_cpu(v: f32) -> u8 {
    if !v.is_finite() {
        return 0x00;
    }
    let clamped = v.clamp(-TENSOR_FP8_E4M3_MAX, TENSOR_FP8_E4M3_MAX);
    let bits = clamped.to_bits();
    let sign = ((bits >> 24) & 0x80) as u8;
    let ax = clamped.abs();
    if ax == 0.0 {
        return 0x00;
    }
    if ax >= TENSOR_FP8_E4M3_MAX {
        return sign | 0x7E;
    }

    let abits = bits & 0x7FFF_FFFF;
    let exp_unbiased = ((abits >> 23) & 0xFF) as i32 - 127;
    let mant23 = abits & 0x007F_FFFF;

    let raw = if exp_unbiased < -6 {
        // Subnormal E4M3FN grid: step = 2^-9, tie-to-even.
        let scaled = ax * 512.0;
        let q = scaled.round_ties_even() as u32;
        if q == 0 {
            0x00
        } else if q >= 8 {
            sign | 0x08
        } else {
            sign | (q as u8)
        }
    } else {
        let e4 = exp_unbiased + 7;
        let mant_hi = mant23 >> 20;
        let rem = mant23 & 0x000F_FFFF;
        let mut rounded_m = mant_hi;
        if rem > 0x0008_0000 || (rem == 0x0008_0000 && (mant_hi & 1) != 0) {
            rounded_m += 1;
        }
        let mut final_e = e4;
        if rounded_m == 8 {
            rounded_m = 0;
            final_e += 1;
        }
        if final_e > 15 || (final_e == 15 && rounded_m > 6) {
            sign | 0x7E
        } else {
            sign | (((final_e as u8) << 3) | (rounded_m as u8))
        }
    };

    // Defense-in-depth scrub matching CUDA `scrub_e4m3_byte`.
    if (raw & 0x7F) == 0x7F || raw == 0x80 {
        0x00
    } else {
        raw
    }
}

/// Decode a finite FP8 E4M3FN byte to `f32` (`0x7F`/`0xFF` decode to `f32::NAN`).
#[must_use]
pub fn e4m3fn_to_f32_cpu(b: u8) -> f32 {
    if (b & 0x7F) == 0x7F {
        return f32::NAN;
    }
    let sign = if (b & 0x80) != 0 { -1.0f32 } else { 1.0f32 };
    let exp = ((b >> 3) & 0x0F) as i32;
    let mant = (b & 0x07) as f32;
    let mag = if exp == 0 {
        mant * 2.0f32.powi(-9)
    } else {
        (1.0 + mant / 8.0) * 2.0f32.powi(exp - 7)
    };
    sign * mag
}

/// CPU reference matching `tensor_fp8_absmax_bf16_sm120` (ignores non-finite lanes).
#[must_use]
pub fn cpu_tensor_fp8_absmax_bf16(input_bf16: &[u16]) -> f32 {
    let mut amax = 0.0f32;
    for &bits in input_bf16 {
        let v = bf16_bits_to_f32_cpu(bits);
        if v.is_finite() {
            amax = amax.max(v.abs());
        }
    }
    amax
}

/// Resolve scalar tensor scale matching CUDA `resolve_tensor_fp8_scale`.
#[must_use]
pub fn resolve_tensor_fp8_scale_cpu(absmax_opt: Option<f32>, static_scale: f32) -> f32 {
    let mut scale = static_scale;
    if let Some(amax) = absmax_opt {
        if amax.is_finite() && amax > 0.0 {
            scale = (amax / TENSOR_FP8_E4M3_MAX).max(TENSOR_FP8_MIN_SCALE);
        } else {
            scale = 1.0;
        }
    }
    if !scale.is_finite() || scale <= 0.0 {
        scale = 1.0;
    }
    scale
}

/// Bit-exact CPU reference for `bf16_to_tensor_fp8_sm120`.
///
/// When `static_scale` is `None`, computes the global finite absmax dynamically
/// matching `tensor_fp8_absmax_bf16_sm120` + `bf16_to_tensor_fp8_sm120`.
pub fn cpu_bf16_to_tensor_fp8(
    input_bf16: &[u16],
    rows: usize,
    cols: usize,
    static_scale: Option<f32>,
) -> Result<(Vec<u8>, f32)> {
    let expected = rows
        .checked_mul(cols)
        .context("cpu_bf16_to_tensor_fp8 shape overflow")?;
    ensure!(
        rows > 0 && cols > 0 && input_bf16.len() == expected,
        "cpu_bf16_to_tensor_fp8 expected {expected} elements for {rows}x{cols}, got {}",
        input_bf16.len()
    );
    let scale = match static_scale {
        Some(s) => {
            ensure!(
                s.is_finite() && s > 0.0,
                "static_scale must be finite and positive"
            );
            resolve_tensor_fp8_scale_cpu(None, s)
        }
        None => resolve_tensor_fp8_scale_cpu(Some(cpu_tensor_fp8_absmax_bf16(input_bf16)), 1.0),
    };
    let inv_scale = 1.0 / scale;
    let out = input_bf16
        .iter()
        .map(|&bits| f32_to_e4m3fn_strict_cpu(bf16_bits_to_f32_cpu(bits) * inv_scale))
        .collect();
    Ok((out, scale))
}

/// Bit-exact CPU reference for `rms_norm_tensor_fp8_sm120`.
///
/// Returns `(out_fp8, out_bf16, scale)` where each normalized element crosses the
/// exact `cvt.rn.bf16.f32` boundary before quantization.
pub fn cpu_rms_norm_tensor_fp8(
    input_bf16: &[u16],
    weight_bf16: &[u16],
    rows: usize,
    cols: usize,
    eps: f32,
    weight_mode: RmsNormWeightMode,
    static_scale: Option<f32>,
) -> Result<(Vec<u8>, Vec<u16>, f32)> {
    let expected = rows
        .checked_mul(cols)
        .context("cpu_rms_norm_tensor_fp8 shape overflow")?;
    ensure!(
        rows > 0 && cols > 0 && input_bf16.len() == expected && weight_bf16.len() == cols,
        "cpu_rms_norm_tensor_fp8 shape mismatch"
    );
    ensure!(
        eps.is_finite() && eps > 0.0,
        "eps must be finite and strictly positive"
    );

    let weight_offset = weight_mode.weight_offset();
    let mut normed_bf16 = vec![0u16; expected];
    for r in 0..rows {
        let base = r * cols;
        let mut sum_sq = 0.0f32;
        for c in 0..cols {
            let x = bf16_bits_to_f32_cpu(input_bf16[base + c]);
            sum_sq += x * x;
        }
        let inv_rms = 1.0f32 / (sum_sq / (cols as f32) + eps).sqrt();
        for c in 0..cols {
            let x = bf16_bits_to_f32_cpu(input_bf16[base + c]);
            let w = bf16_bits_to_f32_cpu(weight_bf16[c]) + weight_offset;
            normed_bf16[base + c] = f32_to_bf16_rn_bits_cpu(x * inv_rms * w);
        }
    }

    let (out_fp8, scale) = cpu_bf16_to_tensor_fp8(&normed_bf16, rows, cols, static_scale)?;
    Ok((out_fp8, normed_bf16, scale))
}

/// CPU reference for scalar-scaled FP8 GEMM with bounded [`TensorFp8EpilogueDeferral`]:
/// `out[m, n] = bf16(epilogue_alpha * (act_scale * weight_scale) * sum_k act[m, k] * weight[n, k])`.
#[expect(
    clippy::too_many_arguments,
    reason = "CPU reference GEMM mirroring device-scaled FP8 GEMM operand arity"
)]
pub fn cpu_tensor_fp8_gemm_deferred(
    act_e4m3: &[u8],
    act_scale: f32,
    weight_e4m3: &[u8],
    weight_scale: f32,
    m: usize,
    n: usize,
    k: usize,
    deferral: TensorFp8EpilogueDeferral,
) -> Result<Vec<u16>> {
    let epilogue_alpha = deferral
        .deferred_alpha()
        .ok_or_else(|| anyhow::anyhow!("cpu_tensor_fp8_gemm_deferred: deferral refused"))?;
    ensure!(
        act_e4m3.len() == m * k && weight_e4m3.len() == n * k,
        "cpu_tensor_fp8_gemm_deferred buffer length mismatch"
    );
    ensure!(
        act_scale.is_finite() && act_scale > 0.0 && weight_scale.is_finite() && weight_scale > 0.0,
        "cpu_tensor_fp8_gemm_deferred requires finite positive tensor scales"
    );
    let combined_scale = epilogue_alpha * act_scale * weight_scale;
    let mut out = vec![0u16; m * n];
    for row in 0..m {
        for col in 0..n {
            let mut acc = 0.0f32;
            for kk in 0..k {
                let a = e4m3fn_to_f32_cpu(act_e4m3[row * k + kk]);
                let w = e4m3fn_to_f32_cpu(weight_e4m3[col * k + kk]);
                acc += a * w;
            }
            out[row * n + col] = f32_to_bf16_rn_bits_cpu(acc * combined_scale);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use std::env::VarError;
    use std::ffi::OsString;
    use std::sync::Mutex;

    use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

    use super::*;

    #[derive(Debug, Clone)]
    struct RecordedLaunch {
        kernel: KernelHandle,
        grid: [u32; 3],
        block: [u32; 3],
        arg_count: usize,
        stream: u64,
    }

    struct MockGpu {
        launches: Mutex<Vec<RecordedLaunch>>,
    }

    impl MockGpu {
        fn new() -> Self {
            Self {
                launches: Mutex::new(Vec::new()),
            }
        }

        fn recorded(&self) -> Vec<RecordedLaunch> {
            self.launches.lock().unwrap().clone()
        }
    }

    impl GpuBackend for MockGpu {
        fn total_memory(&self) -> Result<usize> {
            Ok(128 << 30)
        }
        fn free_memory(&self) -> Result<usize> {
            Ok(64 << 30)
        }
        fn alloc(&self, _bytes: usize) -> Result<DevicePtr> {
            Ok(DevicePtr(0x1000))
        }
        fn alloc_managed(&self, _bytes: usize) -> Result<DevicePtr> {
            Ok(DevicePtr(0x1000))
        }
        fn free(&self, _ptr: DevicePtr) -> Result<()> {
            Ok(())
        }
        fn copy_h2d(&self, _src: &[u8], _dst: DevicePtr) -> Result<()> {
            Ok(())
        }
        fn copy_h2d_async(&self, _src: &[u8], _dst: DevicePtr, _stream: u64) -> Result<()> {
            Ok(())
        }
        fn copy_d2h(&self, _src: DevicePtr, _dst: &mut [u8]) -> Result<()> {
            Ok(())
        }
        fn copy_d2d(&self, _src: DevicePtr, _dst: DevicePtr, _bytes: usize) -> Result<()> {
            Ok(())
        }
        fn copy_d2d_async(
            &self,
            _src: DevicePtr,
            _dst: DevicePtr,
            _bytes: usize,
            _stream: u64,
        ) -> Result<()> {
            Ok(())
        }
        fn memset(&self, _ptr: DevicePtr, _val: u8, _bytes: usize) -> Result<()> {
            Ok(())
        }
        fn memset_async(
            &self,
            _ptr: DevicePtr,
            _val: u8,
            _bytes: usize,
            _stream: u64,
        ) -> Result<()> {
            Ok(())
        }
        fn kernel(&self, _module: &str, _name: &str) -> Result<KernelHandle> {
            Ok(KernelHandle(1))
        }
        fn launch(
            &self,
            kernel: KernelHandle,
            grid: [u32; 3],
            block: [u32; 3],
            _shared_mem_bytes: u32,
            stream: u64,
            args: &mut [*mut std::ffi::c_void],
        ) -> Result<()> {
            self.launches.lock().unwrap().push(RecordedLaunch {
                kernel,
                grid,
                block,
                arg_count: args.len(),
                stream,
            });
            Ok(())
        }
        fn synchronize(&self, _stream: u64) -> Result<()> {
            Ok(())
        }
        fn default_stream(&self) -> u64 {
            0
        }
        fn create_stream(&self) -> Result<u64> {
            Ok(1)
        }
    }

    #[test]
    fn env_parsers_are_strict_default_off_and_respect_explicit_opt_out() {
        assert_eq!(
            parse_gb10_fast_prefill_env(Err(VarError::NotPresent)),
            Ok(Gb10FastPrefillPolicy::Disabled)
        );
        assert_eq!(
            parse_gb10_fast_prefill_env(Ok("0".to_string())),
            Ok(Gb10FastPrefillPolicy::Disabled)
        );
        assert_eq!(
            parse_gb10_fast_prefill_env(Ok("1".to_string())),
            Ok(Gb10FastPrefillPolicy::Enabled)
        );
        for bad in ["", "true", "false", "2", " 1", "01"] {
            assert!(parse_gb10_fast_prefill_env(Ok(bad.to_string())).is_err());
        }
        assert!(parse_gb10_fast_prefill_env(Err(VarError::NotUnicode(OsString::new()))).is_err());

        // ATLAS_TENSOR_FP8_PREFILL defaults OFF when umbrella is OFF, auto-enables
        // when umbrella is ON and knob is unset, and explicit "0" ALWAYS disables.
        assert_eq!(
            parse_tensor_fp8_prefill_env(Err(VarError::NotPresent), false),
            Ok(TensorFp8PrefillPolicy::Disabled)
        );
        assert_eq!(
            parse_tensor_fp8_prefill_env(Err(VarError::NotPresent), true),
            Ok(TensorFp8PrefillPolicy::AutoGb10FastPrefill)
        );
        assert_eq!(
            parse_tensor_fp8_prefill_env(Ok("0".to_string()), true),
            Ok(TensorFp8PrefillPolicy::Disabled)
        );
        assert_eq!(
            parse_tensor_fp8_prefill_env(Ok("1".to_string()), false),
            Ok(TensorFp8PrefillPolicy::ExplicitEnabled)
        );
        assert_eq!(
            parse_tensor_fp8_prefill_env(Ok("1".to_string()), true),
            Ok(TensorFp8PrefillPolicy::ExplicitEnabled)
        );
        for bad in ["", "true", "2", "01"] {
            assert!(parse_tensor_fp8_prefill_env(Ok(bad.to_string()), false).is_err());
        }
        assert!(
            parse_tensor_fp8_prefill_env(Err(VarError::NotUnicode(OsString::new())), true).is_err()
        );

        // Verify EffectiveFusedRmsNormNvfp4Policy cascade under ATLAS_GB10_FAST_PREFILL.
        use crate::layers::ops::{
            EffectiveFusedRmsNormNvfp4Policy, resolve_effective_fused_rmsnorm_nvfp4_policy,
        };
        assert_eq!(
            resolve_effective_fused_rmsnorm_nvfp4_policy(Err(VarError::NotPresent), false),
            Ok(EffectiveFusedRmsNormNvfp4Policy::Disabled)
        );
        assert_eq!(
            resolve_effective_fused_rmsnorm_nvfp4_policy(Err(VarError::NotPresent), true),
            Ok(EffectiveFusedRmsNormNvfp4Policy::AutoGb10FastPrefill)
        );
        assert_eq!(
            resolve_effective_fused_rmsnorm_nvfp4_policy(Ok("0".to_string()), true),
            Ok(EffectiveFusedRmsNormNvfp4Policy::Disabled)
        );
        assert_eq!(
            resolve_effective_fused_rmsnorm_nvfp4_policy(Ok("1".to_string()), false),
            Ok(EffectiveFusedRmsNormNvfp4Policy::ExplicitEnabled)
        );
        assert!(resolve_effective_fused_rmsnorm_nvfp4_policy(Ok("2".to_string()), true).is_err());
        assert!(
            resolve_effective_fused_rmsnorm_nvfp4_policy(
                Err(VarError::NotUnicode(OsString::new())),
                true
            )
            .is_err()
        );
    }

    #[test]
    fn e4m3fn_strict_conversion_clamps_to_448_and_never_emits_nan_byte() {
        let edge_inputs = [
            0.0f32,
            -0.0,
            f32::NAN,
            -f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            448.0,
            -448.0,
            449.0,
            -449.0,
            1e30,
            -1e30,
            1e-30,
            -1e-30,
            1.0 / 512.0,
            7.0 / 512.0,
            7.5 / 512.0,
            1.0,
            -1.0,
            1.0625, // tie to even -> 1.0 (mantissa 0, even)
            1.1875, // tie to even -> 1.25 (mantissa 2, even)
            432.0,  // tie between 416 (mant=5 odd) and 448 (mant=6 even) -> 448 (0x7E)
            -432.0,
        ];
        for v in edge_inputs {
            let b = f32_to_e4m3fn_strict_cpu(v);
            assert_ne!(b & 0x7F, 0x7F, "NaN byte 0x{b:02X} emitted for input {v}");
            assert_ne!(b, 0x80, "-0.0 byte emitted for input {v}");
        }
        assert_eq!(f32_to_e4m3fn_strict_cpu(448.0), 0x7E);
        assert_eq!(f32_to_e4m3fn_strict_cpu(1e20), 0x7E);
        assert_eq!(f32_to_e4m3fn_strict_cpu(-448.0), 0xFE);
        assert_eq!(f32_to_e4m3fn_strict_cpu(-1e20), 0xFE);
        assert_eq!(f32_to_e4m3fn_strict_cpu(f32::NAN), 0x00);
        assert_eq!(f32_to_e4m3fn_strict_cpu(f32::INFINITY), 0x00);
        assert_eq!(f32_to_e4m3fn_strict_cpu(-0.0), 0x00);
        assert_eq!(e4m3fn_to_f32_cpu(f32_to_e4m3fn_strict_cpu(1.0625)), 1.0);
        assert_eq!(e4m3fn_to_f32_cpu(f32_to_e4m3fn_strict_cpu(1.1875)), 1.25);
        assert_eq!(e4m3fn_to_f32_cpu(f32_to_e4m3fn_strict_cpu(432.0)), 448.0);
    }

    #[test]
    fn epilogue_deferral_enforces_0125_to_8_bounds_and_k_block_uniformity() {
        // Admitted boundary endpoints and interior values.
        for s in [0.125f32, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0] {
            let d = TensorFp8EpilogueDeferral::new(s, true);
            assert_eq!(
                d.decision(),
                TensorFp8DeferralDecision::DeferToEpilogue { epilogue_alpha: s }
            );
            assert_eq!(d.deferred_alpha(), Some(s));
        }

        // Below 0.125 and above 8.0 must refuse deferral.
        assert_eq!(
            TensorFp8EpilogueDeferral::new(f32::from_bits(0.125f32.to_bits() - 1), true).decision(),
            TensorFp8DeferralDecision::DirectRescale {
                reason: TensorFp8DeferralRefusal::ScaleBelowMin
            }
        );
        assert_eq!(
            TensorFp8EpilogueDeferral::new(f32::from_bits(8.0f32.to_bits() + 1), true).decision(),
            TensorFp8DeferralDecision::DirectRescale {
                reason: TensorFp8DeferralRefusal::ScaleAboveMax
            }
        );

        // Non-finite / non-positive scales refuse deferral.
        for bad in [
            0.0f32,
            -0.0,
            -1.0,
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
        ] {
            assert_eq!(
                TensorFp8EpilogueDeferral::new(bad, true).decision(),
                TensorFp8DeferralDecision::DirectRescale {
                    reason: TensorFp8DeferralRefusal::NonFiniteOrNonPositiveScale
                }
            );
        }

        // Non-uniform K-blocks MUST refuse deferral even for scale inside [0.125, 8.0].
        assert_eq!(
            TensorFp8EpilogueDeferral::new(1.0, false).decision(),
            TensorFp8DeferralDecision::DirectRescale {
                reason: TensorFp8DeferralRefusal::NonUniformKBlocks
            }
        );
        assert_eq!(
            TensorFp8EpilogueDeferral::from_k_block_scales(&[1.0, 1.0, 1.0]).deferred_alpha(),
            Some(1.0)
        );
        assert_eq!(
            TensorFp8EpilogueDeferral::from_k_block_scales(&[1.0, 1.0, 2.0]).decision(),
            TensorFp8DeferralDecision::DirectRescale {
                reason: TensorFp8DeferralRefusal::NonUniformKBlocks
            }
        );
        assert_eq!(
            TensorFp8EpilogueDeferral::from_k_block_scales(&[]).decision(),
            TensorFp8DeferralDecision::DirectRescale {
                reason: TensorFp8DeferralRefusal::NonUniformKBlocks
            }
        );
    }

    #[test]
    fn cpu_rms_norm_tensor_fp8_matches_two_step_bf16_boundary_bit_for_bit() {
        let rows = 4usize;
        let cols = 64usize;
        let mut input = Vec::with_capacity(rows * cols);
        for i in 0..(rows * cols) {
            let v = (((i.wrapping_mul(1_664_525).wrapping_add(1_013_904_223) >> 16) & 0xFF) as f32
                / 64.0
                - 2.0)
                * 1.75;
            input.push(f32_to_bf16_rn_bits_cpu(v));
        }
        let weight: Vec<u16> = (0..cols)
            .map(|c| f32_to_bf16_rn_bits_cpu(0.05 * ((c as f32) - 32.0) / 32.0))
            .collect();

        let (fused_fp8, fused_bf16, fused_scale) = cpu_rms_norm_tensor_fp8(
            &input,
            &weight,
            rows,
            cols,
            1e-6,
            RmsNormWeightMode::UnitOffset,
            None,
        )
        .unwrap();
        let (two_step_fp8, two_step_scale) =
            cpu_bf16_to_tensor_fp8(&fused_bf16, rows, cols, None).unwrap();

        assert_eq!(fused_fp8, two_step_fp8);
        assert_eq!(fused_scale.to_bits(), two_step_scale.to_bits());
        assert!(fused_fp8.iter().all(|&b| (b & 0x7F) != 0x7F));
    }

    #[test]
    fn cpu_deferred_gemm_matches_direct_scaled_reference() {
        let (m, n, k) = (2usize, 16usize, 16usize);
        let act_f32: Vec<f32> = (0..m * k).map(|i| ((i as f32) - 15.5) * 0.25).collect();
        let w_f32: Vec<f32> = (0..n * k)
            .map(|i| (((i * 7) % 19) as f32 - 9.0) * 0.125)
            .collect();
        let act_bf16: Vec<u16> = act_f32
            .iter()
            .map(|&v| f32_to_bf16_rn_bits_cpu(v))
            .collect();
        let w_bf16: Vec<u16> = w_f32.iter().map(|&v| f32_to_bf16_rn_bits_cpu(v)).collect();

        let (act_fp8, act_scale) = cpu_bf16_to_tensor_fp8(&act_bf16, m, k, None).unwrap();
        let (w_fp8, w_scale) = cpu_bf16_to_tensor_fp8(&w_bf16, n, k, None).unwrap();

        let deferral = TensorFp8EpilogueDeferral::new(2.5, true);
        let out_deferred =
            cpu_tensor_fp8_gemm_deferred(&act_fp8, act_scale, &w_fp8, w_scale, m, n, k, deferral)
                .unwrap();

        // Compare against folded-scale reference (alpha = 1.0, weight_scale *= 2.5).
        let out_folded = cpu_tensor_fp8_gemm_deferred(
            &act_fp8,
            act_scale,
            &w_fp8,
            w_scale * 2.5,
            m,
            n,
            k,
            TensorFp8EpilogueDeferral::new(1.0, true),
        )
        .unwrap();
        assert_eq!(out_deferred, out_folded);

        // Refused deferral errors cleanly.
        assert!(
            cpu_tensor_fp8_gemm_deferred(
                &act_fp8,
                act_scale,
                &w_fp8,
                w_scale,
                m,
                n,
                k,
                TensorFp8EpilogueDeferral::new(16.0, true),
            )
            .is_err()
        );
    }

    #[test]
    fn launchers_enforce_non_aliasing_alignment_and_cuda_source_contracts() {
        let gpu = MockGpu::new();
        // Valid launches record expected grid/block/arity.
        launch_tensor_fp8_absmax_bf16_sm120(
            &gpu,
            KernelHandle(10),
            DevicePtr(0x10_0000),
            DevicePtr(0x20_0000),
            8,
            64,
            0x7,
        )
        .unwrap();
        launch_bf16_to_tensor_fp8_sm120(
            &gpu,
            KernelHandle(11),
            &Bf16ToTensorFp8LaunchArgs {
                input_bf16: DevicePtr(0x10_0000),
                out_fp8: DevicePtr(0x30_0000),
                d_absmax_in: DevicePtr(0x20_0000),
                d_scale_out: DevicePtr(0x20_0010),
                static_scale: 1.0,
                rows: 8,
                cols: 64,
            },
            0x7,
        )
        .unwrap();
        launch_rms_norm_tensor_fp8_sm120(
            &gpu,
            KernelHandle(12),
            &RmsNormTensorFp8LaunchArgs {
                input_bf16: DevicePtr(0x10_0000),
                weight_bf16: DevicePtr(0x18_0000),
                out_fp8: DevicePtr(0x30_0000),
                out_bf16_opt: DevicePtr(0x40_0000),
                d_absmax_in: DevicePtr::NULL,
                d_scale_out: DevicePtr(0x20_0010),
                eps: 1e-6,
                weight_mode: RmsNormWeightMode::UnitOffset,
                static_scale: 1.0,
                rows: 8,
                cols: 64,
            },
            0x7,
        )
        .unwrap();

        let rec = gpu.recorded();
        assert_eq!(rec.len(), 3);
        assert_eq!(rec[0].kernel.0, 10);
        assert_eq!(rec[0].stream, 0x7);
        assert_eq!(rec[0].arg_count, 4);
        assert_eq!(rec[1].kernel.0, 11);
        assert_eq!(rec[1].arg_count, 7);
        assert_eq!(rec[2].kernel.0, 12);
        assert_eq!(rec[2].arg_count, 11);
        assert_eq!(rec[2].grid, [8, 1, 1]);
        assert_eq!(rec[2].block, [256, 1, 1]);

        // Aliasing / overlap MUST fail closed.
        assert!(
            launch_bf16_to_tensor_fp8_sm120(
                &gpu,
                KernelHandle(11),
                &Bf16ToTensorFp8LaunchArgs {
                    input_bf16: DevicePtr(0x10_0000),
                    out_fp8: DevicePtr(0x10_0100), // overlaps input_bf16 (1024 bytes)
                    d_absmax_in: DevicePtr::NULL,
                    d_scale_out: DevicePtr::NULL,
                    static_scale: 1.0,
                    rows: 8,
                    cols: 64,
                },
                0,
            )
            .is_err()
        );
        assert!(
            launch_rms_norm_tensor_fp8_sm120(
                &gpu,
                KernelHandle(12),
                &RmsNormTensorFp8LaunchArgs {
                    input_bf16: DevicePtr(0x10_0000),
                    weight_bf16: DevicePtr(0x18_0000),
                    out_fp8: DevicePtr(0x30_0000),
                    out_bf16_opt: DevicePtr(0x10_0000), // aliases input_bf16
                    d_absmax_in: DevicePtr::NULL,
                    d_scale_out: DevicePtr::NULL,
                    eps: 1e-6,
                    weight_mode: RmsNormWeightMode::Direct,
                    static_scale: 1.0,
                    rows: 8,
                    cols: 64,
                },
                0,
            )
            .is_err()
        );

        // Refused deferral refuses GEMM launch before touching cuBLASLt.
        assert!(
            TensorFp8EpilogueDeferral::new(0.0625, true)
                .launch_gemm_device_scaled(
                    &gpu,
                    &TensorFp8GemmArgs {
                        act_e4m3: DevicePtr(0x10_0000),
                        act_scale: DevicePtr(0x20_0000),
                        weight_e4m3: DevicePtr(0x30_0000),
                        weight_scale: DevicePtr(0x40_0000),
                        out_bf16: DevicePtr(0x50_0000),
                        gb10_scale_fence_slot: Some(DevicePtr(0x60_0000)),
                        m: 32,
                        n: 64,
                        k: 64,
                        ldc: 64,
                    },
                    0,
                )
                .is_err()
        );

        // Verify CUDA source contains every required ISA & safety contract.
        let cuda = include_str!("../../../../../kernels/gb10/common/tensor_fp8_quant_sm120.cu");
        for token in [
            "tensor_fp8_absmax_bf16_sm120(",
            "bf16_to_tensor_fp8_sm120(",
            "rms_norm_tensor_fp8_sm120(",
            "cvt.rn.bf16.f32",
            "cvt.rn.satfinite.e4m3x2.f32",
            "448.0f",
            "int64_t",
            "__restrict__",
        ] {
            assert!(
                cuda.contains(token),
                "tensor_fp8_quant_sm120.cu missing contract token `{token}`"
            );
        }
    }

    #[test]
    fn scratch_growth_capture_and_sync_error_preserve_state_without_leaks() {
        struct TrackingGpu {
            next_ptr: std::sync::atomic::AtomicU64,
            live: Mutex<std::collections::HashSet<u64>>,
            fail_sync: std::sync::atomic::AtomicBool,
            capturing: std::sync::atomic::AtomicBool,
        }

        impl TrackingGpu {
            fn new() -> Self {
                Self {
                    next_ptr: std::sync::atomic::AtomicU64::new(0x10_0000),
                    live: Mutex::new(std::collections::HashSet::new()),
                    fail_sync: std::sync::atomic::AtomicBool::new(false),
                    capturing: std::sync::atomic::AtomicBool::new(false),
                }
            }
            fn live_count(&self) -> usize {
                self.live.lock().unwrap().len()
            }
        }

        impl GpuBackend for TrackingGpu {
            fn total_memory(&self) -> Result<usize> {
                Ok(128 << 30)
            }
            fn free_memory(&self) -> Result<usize> {
                Ok(64 << 30)
            }
            fn alloc(&self, bytes: usize) -> Result<DevicePtr> {
                let step = bytes.max(256) as u64;
                let ptr = self
                    .next_ptr
                    .fetch_add(step, std::sync::atomic::Ordering::Relaxed);
                assert!(self.live.lock().unwrap().insert(ptr));
                Ok(DevicePtr(ptr))
            }
            fn alloc_managed(&self, bytes: usize) -> Result<DevicePtr> {
                self.alloc(bytes)
            }
            fn free(&self, ptr: DevicePtr) -> Result<()> {
                assert!(
                    self.live.lock().unwrap().remove(&ptr.0),
                    "double free or unknown pointer 0x{:x}",
                    ptr.0
                );
                Ok(())
            }
            fn copy_h2d(&self, _src: &[u8], _dst: DevicePtr) -> Result<()> {
                Ok(())
            }
            fn copy_h2d_async(&self, _src: &[u8], _dst: DevicePtr, _stream: u64) -> Result<()> {
                Ok(())
            }
            fn copy_d2h(&self, _src: DevicePtr, _dst: &mut [u8]) -> Result<()> {
                Ok(())
            }
            fn copy_d2d(&self, _src: DevicePtr, _dst: DevicePtr, _bytes: usize) -> Result<()> {
                Ok(())
            }
            fn copy_d2d_async(
                &self,
                _src: DevicePtr,
                _dst: DevicePtr,
                _bytes: usize,
                _stream: u64,
            ) -> Result<()> {
                Ok(())
            }
            fn memset(&self, _ptr: DevicePtr, _val: u8, _bytes: usize) -> Result<()> {
                Ok(())
            }
            fn memset_async(
                &self,
                _ptr: DevicePtr,
                _val: u8,
                _bytes: usize,
                _stream: u64,
            ) -> Result<()> {
                Ok(())
            }
            fn kernel(&self, _module: &str, _name: &str) -> Result<KernelHandle> {
                Ok(KernelHandle(1))
            }
            fn launch(
                &self,
                _kernel: KernelHandle,
                _grid: [u32; 3],
                _block: [u32; 3],
                _shared_mem_bytes: u32,
                _stream: u64,
                _args: &mut [*mut std::ffi::c_void],
            ) -> Result<()> {
                Ok(())
            }
            fn synchronize(&self, _stream: u64) -> Result<()> {
                if self.fail_sync.load(std::sync::atomic::Ordering::Relaxed) {
                    bail!("injected stream sync failure");
                }
                Ok(())
            }
            fn default_stream(&self) -> u64 {
                0
            }
            fn create_stream(&self) -> Result<u64> {
                Ok(1)
            }
            fn stream_is_capturing(&self, _stream: u64) -> bool {
                self.capturing.load(std::sync::atomic::Ordering::Relaxed)
            }
        }

        let gpu = TrackingGpu::new();
        let mut map = TensorFp8ScratchMap::new();

        // Capture refuses initial allocation with 0 allocations performed.
        gpu.capturing
            .store(true, std::sync::atomic::Ordering::Relaxed);
        assert!(
            ensure_tensor_fp8_scratch(&mut map, &gpu, 1024, 1)
                .unwrap()
                .is_none()
        );
        assert_eq!(gpu.live_count(), 0);
        gpu.capturing
            .store(false, std::sync::atomic::Ordering::Relaxed);

        // Initial allocation creates scalars + activation buffer (2 live allocations).
        let (act1, _, _, _, _) = ensure_tensor_fp8_scratch(&mut map, &gpu, 1024, 1)
            .unwrap()
            .unwrap();
        assert_eq!(gpu.live_count(), 2);

        // Reuse within capacity allocates nothing and works during capture.
        gpu.capturing
            .store(true, std::sync::atomic::Ordering::Relaxed);
        let (act1_reuse, _, _, _, _) = ensure_tensor_fp8_scratch(&mut map, &gpu, 512, 1)
            .unwrap()
            .unwrap();
        assert_eq!(act1_reuse, act1);
        assert_eq!(gpu.live_count(), 2);
        // Growth during capture refuses without leaking or replacing `act1`.
        assert!(
            ensure_tensor_fp8_scratch(&mut map, &gpu, 4096, 1)
                .unwrap()
                .is_none()
        );
        assert_eq!(gpu.live_count(), 2);
        gpu.capturing
            .store(false, std::sync::atomic::Ordering::Relaxed);

        // Sync error during growth frees newly allocated buffer and preserves `act1`.
        gpu.fail_sync
            .store(true, std::sync::atomic::Ordering::Relaxed);
        assert!(ensure_tensor_fp8_scratch(&mut map, &gpu, 4096, 1).is_err());
        assert_eq!(gpu.live_count(), 2);
        gpu.fail_sync
            .store(false, std::sync::atomic::Ordering::Relaxed);
        let (act1_still, _, _, _, _) = ensure_tensor_fp8_scratch(&mut map, &gpu, 1024, 1)
            .unwrap()
            .unwrap();
        assert_eq!(act1_still, act1);

        // Successful growth frees `act1` and installs `act2` (still exactly 2 live).
        let (act2, _, _, _, _) = ensure_tensor_fp8_scratch(&mut map, &gpu, 4096, 1)
            .unwrap()
            .unwrap();
        assert_ne!(act2, act1);
        assert_eq!(gpu.live_count(), 2);
    }

    #[test]
    fn gemm_alignment_ldc_compute_capability_and_strict_eps_guards() {
        // Compute capability gate: SM89+ and None (mock) admitted; pre-Ada rejected.
        assert!(!supports_tensor_fp8_gemm(Some((7, 5))));
        assert!(!supports_tensor_fp8_gemm(Some((8, 0))));
        assert!(!supports_tensor_fp8_gemm(Some((8, 6))));
        assert!(!supports_tensor_fp8_gemm(Some((8, 8))));
        for cc in [
            None,
            Some((8, 9)),
            Some((9, 0)),
            Some((10, 0)),
            Some((10, 3)),
            Some((12, 0)),
            Some((12, 1)),
        ] {
            assert!(supports_tensor_fp8_gemm(cc), "expected {cc:?} supported");
        }

        let valid = TensorFp8GemmArgs {
            act_e4m3: DevicePtr(0x10_0000),
            act_scale: DevicePtr(0x20_0000),
            weight_e4m3: DevicePtr(0x30_0000),
            weight_scale: DevicePtr(0x40_0000),
            out_bf16: DevicePtr(0x50_0000),
            gb10_scale_fence_slot: Some(DevicePtr(0x60_0000)),
            m: 32,
            n: 64,
            k: 64,
            ldc: 72, // >= n and multiple of 8
        };
        assert!(validate_tensor_fp8_gemm_args(&valid).is_ok());

        // Non-16B-aligned act/weight/out and non-8-aligned ldc MUST fail closed.
        assert!(
            validate_tensor_fp8_gemm_args(&TensorFp8GemmArgs {
                act_e4m3: DevicePtr(0x10_0008),
                ..valid
            })
            .is_err()
        );
        assert!(
            validate_tensor_fp8_gemm_args(&TensorFp8GemmArgs {
                weight_e4m3: DevicePtr(0x30_0004),
                ..valid
            })
            .is_err()
        );
        assert!(
            validate_tensor_fp8_gemm_args(&TensorFp8GemmArgs {
                out_bf16: DevicePtr(0x50_0002),
                ..valid
            })
            .is_err()
        );
        assert!(validate_tensor_fp8_gemm_args(&TensorFp8GemmArgs { ldc: 68, ..valid }).is_err());

        // Strict eps > 0.0 on both GPU launcher and CPU reference.
        let gpu = MockGpu::new();
        for bad_eps in [0.0f32, -0.0, -1e-6, f32::NAN, f32::INFINITY] {
            assert!(
                launch_rms_norm_tensor_fp8_sm120(
                    &gpu,
                    KernelHandle(12),
                    &RmsNormTensorFp8LaunchArgs {
                        input_bf16: DevicePtr(0x10_0000),
                        weight_bf16: DevicePtr(0x18_0000),
                        out_fp8: DevicePtr(0x30_0000),
                        out_bf16_opt: DevicePtr::NULL,
                        d_absmax_in: DevicePtr::NULL,
                        d_scale_out: DevicePtr::NULL,
                        eps: bad_eps,
                        weight_mode: RmsNormWeightMode::UnitOffset,
                        static_scale: 1.0,
                        rows: 4,
                        cols: 16,
                    },
                    0,
                )
                .is_err()
            );
            assert!(
                cpu_rms_norm_tensor_fp8(
                    &[0u16; 64],
                    &[0u16; 16],
                    4,
                    16,
                    bad_eps,
                    RmsNormWeightMode::UnitOffset,
                    Some(1.0),
                )
                .is_err()
            );
        }
    }
}
