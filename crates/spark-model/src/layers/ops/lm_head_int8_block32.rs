// SPDX-License-Identifier: AGPL-3.0-only

//! Native Blackwell SM120/SM121 (and portable SM103) Fused `INT8 Block-32
//! BF16-Scale (8.50 bpw)` LM-Head GEMM + Online Warp/Block Argmax, weight
//! quantizer, fail-closed env gate, and bit-exact CPU reference oracles.
//!
//! CUDA entrypoints live in `kernels/gb10/common/lm_head_int8_block32_argmax_sm120.cu`
//! (auto-discovered by `atlas-kernels` under module
//! `"lm_head_int8_block32_argmax_sm120"`):
//! - `lm_head_int8_block32_gemm_sm120`: full BF16 logit mode (`M = 1..=64`,
//!   `K % 32 == 0`, arbitrary `N`) with `LDG.E.EF.128` weights, coalesced
//!   `u32` `LDG.E.EF` scale-pair loads, `LDG.E.128.CONSTANT` activations,
//!   0-ULP lossless `int8 -> bf16` unpack into portable
//!   `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`, FP32 per-block-32
//!   scale factorization, and explicit inline PTX `cvt.rn.bf16.f32` epilogue.
//! - `lm_head_int8_block32_argmax_stage1_sm120`: fused `M = 1..=16`,
//!   `V_TILE = 256` online warp/block argmax skipping `[M, V]` BF16 logit
//!   materialization (`<= 62 KiB` partials at `M=8, V=248k`) with deterministic
//!   strict lowest-vocab-index tie-break and `NaN` exclusion.
//! - `lm_head_int8_block32_argmax_stage2_sm120`: 1-warp-per-row final reduction.
//! - `quantize_bf16_to_int8_block32_sm120`: coalesced 1-warp-per-block-32
//!   BF16 -> INT8 (`[-127, 127]`) + BF16 scale packer.
//!
//! Opt-in gate: `ATLAS_LM_HEAD_INT8_BLOCK32=1` (strict `ZERO_OR_ONE`, default
//! OFF for 100% bit-identical baseline and zero extra VRAM when unset).

use std::collections::HashMap;
use std::ffi::OsStr;
use std::fmt;
use std::sync::{Mutex, Once, OnceLock};

use anyhow::{Context, Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

/// Auto-discovered `atlas-kernels` module name for
/// `kernels/gb10/common/lm_head_int8_block32_argmax_sm120.cu`.
pub const LM_HEAD_INT8_BLOCK32_MODULE: &str = "lm_head_int8_block32_argmax_sm120";

/// Environment variable opting into INT8 Block-32 BF16-scale LM-head GEMM and
/// fused online argmax. Strict `ZERO_OR_ONE` (`"0"` or `"1"`, unset = disabled).
pub const LM_HEAD_INT8_BLOCK32_ENV: &str = "ATLAS_LM_HEAD_INT8_BLOCK32";

/// Number of weights sharing each BF16 block scale (`8.50 bpw`).
pub const LM_HEAD_INT8_BLOCK_SIZE: u32 = 32;

/// Maximum rows `M` supported by `lm_head_int8_block32_gemm_sm120`.
pub const LM_HEAD_INT8_GEMM_MAX_M: u32 = 64;

/// Output columns per CTA in `lm_head_int8_block32_gemm_sm120`.
pub const LM_HEAD_INT8_GEMM_N_TILE: u32 = 64;

/// Row tile height for `lm_head_int8_block32_gemm_sm120` (`M=1..=16` fits 1 tile).
pub const LM_HEAD_INT8_GEMM_M_TILE: u32 = 16;

/// Maximum rows `M` supported by `lm_head_int8_block32_argmax_stage1_sm120`.
pub const LM_HEAD_INT8_ARGMAX_MAX_M: u32 = 16;

/// Vocab columns covered per CTA in `lm_head_int8_block32_argmax_stage1_sm120`.
pub const LM_HEAD_INT8_ARGMAX_V_TILE: u32 = 256;

/// Sentinel vocab index indicating an out-of-bounds or all-NaN reduction slot.
pub const LM_HEAD_INT8_INVALID_ID: u32 = 0xFFFF_FFFF;

/// Error returned when parsing [`LM_HEAD_INT8_BLOCK32_ENV`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LmHeadInt8Block32ConfigError {
    /// Environment value is not valid UTF-8.
    NonUtf8,
    /// Environment value is not strictly `"0"` or `"1"`.
    InvalidToken(String),
}

impl fmt::Display for LmHeadInt8Block32ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonUtf8 => write!(
                f,
                "{LM_HEAD_INT8_BLOCK32_ENV} must contain valid UTF-8 and be strictly 0 or 1"
            ),
            Self::InvalidToken(raw) => write!(
                f,
                "{LM_HEAD_INT8_BLOCK32_ENV} must be strictly 0 or 1, got {raw:?}"
            ),
        }
    }
}

impl std::error::Error for LmHeadInt8Block32ConfigError {}

/// Opt-in runtime policy parsed from [`LM_HEAD_INT8_BLOCK32_ENV`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum LmHeadInt8Block32Policy {
    /// Default OFF: existing LM-head path is 100% bit-identical.
    #[default]
    Disabled,
    /// Explicit `ATLAS_LM_HEAD_INT8_BLOCK32=1`.
    Enabled,
}

impl LmHeadInt8Block32Policy {
    /// Strictly parse [`LM_HEAD_INT8_BLOCK32_ENV`] from the process environment.
    ///
    /// # Errors
    /// Fails closed on non-UTF-8 or any value outside `{"0", "1"}`.
    pub fn from_env() -> std::result::Result<Self, LmHeadInt8Block32ConfigError> {
        let raw = std::env::var_os(LM_HEAD_INT8_BLOCK32_ENV);
        Self::parse_os(raw.as_deref())
    }

    /// Parse an optional OS string (used by [`Self::from_env`] and tests).
    ///
    /// # Errors
    /// Fails closed on non-UTF-8 or any value outside `{"0", "1"}`.
    pub fn parse_os(
        raw: Option<&OsStr>,
    ) -> std::result::Result<Self, LmHeadInt8Block32ConfigError> {
        let Some(os_val) = raw else {
            return Ok(Self::Disabled);
        };
        let utf8 = os_val
            .to_str()
            .ok_or(LmHeadInt8Block32ConfigError::NonUtf8)?;
        Self::parse_optional(Some(utf8))
    }

    /// Parse an optional UTF-8 string strictly against `ZERO_OR_ONE`.
    ///
    /// # Errors
    /// Fails closed on any token other than `"0"` or `"1"`.
    pub fn parse_optional(
        raw: Option<&str>,
    ) -> std::result::Result<Self, LmHeadInt8Block32ConfigError> {
        let Some(val) = raw else {
            return Ok(Self::Disabled);
        };
        match val {
            "0" => Ok(Self::Disabled),
            "1" => Ok(Self::Enabled),
            other => Err(LmHeadInt8Block32ConfigError::InvalidToken(other.to_owned())),
        }
    }
}

/// Cached fail-closed runtime policy for `ATLAS_LM_HEAD_INT8_BLOCK32`.
///
/// Logs a warning once and stays [`LmHeadInt8Block32Policy::Disabled`] if the
/// environment variable contains non-UTF-8 bytes or a token outside `{"0", "1"}`.
#[must_use]
pub fn lm_head_int8_block32_policy_fail_closed() -> LmHeadInt8Block32Policy {
    static POLICY: OnceLock<LmHeadInt8Block32Policy> = OnceLock::new();
    *POLICY.get_or_init(|| match LmHeadInt8Block32Policy::from_env() {
        Ok(policy) => policy,
        Err(err) => {
            static WARNED: Once = Once::new();
            WARNED.call_once(|| {
                tracing::warn!(
                    env = LM_HEAD_INT8_BLOCK32_ENV,
                    error = %err,
                    "invalid ATLAS_LM_HEAD_INT8_BLOCK32 value — failing closed to Disabled"
                );
            });
            LmHeadInt8Block32Policy::Disabled
        }
    })
}

/// Convenience predicate returning `true` only when `ATLAS_LM_HEAD_INT8_BLOCK32=1`.
#[must_use]
pub fn lm_head_int8_block32_enabled() -> bool {
    lm_head_int8_block32_policy_fail_closed() == LmHeadInt8Block32Policy::Enabled
}

/// Device-resident `INT8 Block-32 BF16-Scale` (`8.50 bpw`) LM-head weight.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Int8Block32DeviceWeight {
    /// `[rows, cols]` row-major `int8` (`[-127, 127]`), 16B-aligned.
    pub w_i8: DevicePtr,
    /// `[rows, cols / 32]` row-major BF16 scales (`u16` bits), 4B-aligned when `cols % 64 == 0`.
    pub scales_bf16: DevicePtr,
    /// Total quantized vocabulary rows (`N`).
    pub rows: u32,
    /// Hidden width (`K`, divisible by 32).
    pub cols: u32,
    /// Byte length of `w_i8`.
    pub weight_bytes: usize,
    /// Byte length of `scales_bf16`.
    pub scales_bytes: usize,
}

/// Checked byte sizes for an `INT8 Block-32 BF16-Scale` `[rows, cols]` weight.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Int8Block32WeightPlan {
    pub rows: u32,
    pub cols: u32,
    pub num_blocks: u32,
    pub weight_bytes: usize,
    pub scales_bytes: usize,
    pub src_bf16_bytes: usize,
}

/// Validate `[rows, cols]` weight shape and compute checked buffer extents.
///
/// # Errors
/// Fails closed if `rows == 0`, `cols < 32`, `cols % 32 != 0`, or byte sizing overflows.
pub fn plan_lm_head_int8_block32_weight(rows: u32, cols: u32) -> Result<Int8Block32WeightPlan> {
    ensure!(rows > 0, "lm_head_int8_block32: rows (vocab) must be > 0");
    ensure!(
        cols >= LM_HEAD_INT8_BLOCK_SIZE && cols.is_multiple_of(LM_HEAD_INT8_BLOCK_SIZE),
        "lm_head_int8_block32: cols={cols} must be >= 32 and divisible by 32"
    );
    let num_blocks = cols / LM_HEAD_INT8_BLOCK_SIZE;
    let weight_bytes = (rows as usize)
        .checked_mul(cols as usize)
        .context("lm_head_int8_block32: weight_i8 byte count overflow")?;
    let scales_bytes = (rows as usize)
        .checked_mul(num_blocks as usize)
        .and_then(|v| v.checked_mul(std::mem::size_of::<u16>()))
        .context("lm_head_int8_block32: scales_bf16 byte count overflow")?;
    let src_bf16_bytes = weight_bytes
        .checked_mul(std::mem::size_of::<u16>())
        .context("lm_head_int8_block32: src_bf16 byte count overflow")?;
    Ok(Int8Block32WeightPlan {
        rows,
        cols,
        num_blocks,
        weight_bytes,
        scales_bytes,
        src_bf16_bytes,
    })
}

/// Validated launch geometry and buffer extents for full BF16 logit GEMM.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LmHeadInt8Block32GemmPlan {
    pub rows: u32,
    pub vocab: u32,
    pub hidden: u32,
    pub grid_x: u32,
    pub grid_y: u32,
    pub a_bytes: usize,
    pub c_bytes: usize,
    pub weight_plan: Int8Block32WeightPlan,
}

/// Validate dimensions for `lm_head_int8_block32_gemm_sm120`.
///
/// # Errors
/// Fails closed if `rows` is outside `1..=64`, `vocab == 0`, `hidden < 32` or
/// not divisible by 32, or any byte extent overflows `usize`.
pub fn plan_lm_head_int8_block32_gemm(
    rows: u32,
    vocab: u32,
    hidden: u32,
) -> Result<LmHeadInt8Block32GemmPlan> {
    ensure!(
        (1..=LM_HEAD_INT8_GEMM_MAX_M).contains(&rows),
        "lm_head_int8_block32_gemm: rows={rows} must lie in 1..={LM_HEAD_INT8_GEMM_MAX_M}"
    );
    let weight_plan = plan_lm_head_int8_block32_weight(vocab, hidden)?;
    let a_bytes = (rows as usize)
        .checked_mul(hidden as usize)
        .and_then(|v| v.checked_mul(std::mem::size_of::<u16>()))
        .context("lm_head_int8_block32_gemm: A_bf16 byte count overflow")?;
    let c_bytes = (rows as usize)
        .checked_mul(vocab as usize)
        .and_then(|v| v.checked_mul(std::mem::size_of::<u16>()))
        .context("lm_head_int8_block32_gemm: C_bf16 byte count overflow")?;
    Ok(LmHeadInt8Block32GemmPlan {
        rows,
        vocab,
        hidden,
        grid_x: div_ceil(vocab, LM_HEAD_INT8_GEMM_N_TILE),
        grid_y: div_ceil(rows, LM_HEAD_INT8_GEMM_M_TILE),
        a_bytes,
        c_bytes,
        weight_plan,
    })
}

/// Validated launch geometry and compact partial-scratch layout for two-stage
/// fused online argmax (`lm_head_int8_block32_argmax_stage{1,2}_sm120`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LmHeadInt8Block32ArgmaxPlan {
    pub rows: u32,
    pub vocab: u32,
    pub hidden: u32,
    pub num_v_tiles: u32,
    pub a_bytes: usize,
    pub partial_vals_bytes: usize,
    pub partial_ids_offset: usize,
    pub partial_ids_bytes: usize,
    pub total_scratch_bytes: usize,
    pub out_tokens_bytes: usize,
    pub weight_plan: Int8Block32WeightPlan,
}

#[inline]
#[must_use]
fn align_up(val: usize, align: usize) -> Option<usize> {
    let rem = val % align;
    if rem == 0 {
        Some(val)
    } else {
        val.checked_add(align - rem)
    }
}

/// Validate dimensions and compute checked compact scratch sizing for fused
/// online argmax (`M in 1..=16`, `K % 32 == 0`, arbitrary `V > 0`).
///
/// # Errors
/// Fails closed on out-of-range dimensions or byte-extent overflow.
pub fn plan_lm_head_int8_block32_argmax(
    rows: u32,
    vocab: u32,
    hidden: u32,
) -> Result<LmHeadInt8Block32ArgmaxPlan> {
    ensure!(
        (1..=LM_HEAD_INT8_ARGMAX_MAX_M).contains(&rows),
        "lm_head_int8_block32_argmax: rows={rows} must lie in 1..={LM_HEAD_INT8_ARGMAX_MAX_M}"
    );
    let weight_plan = plan_lm_head_int8_block32_weight(vocab, hidden)?;
    let num_v_tiles = div_ceil(vocab, LM_HEAD_INT8_ARGMAX_V_TILE);
    let a_bytes = (rows as usize)
        .checked_mul(hidden as usize)
        .and_then(|v| v.checked_mul(std::mem::size_of::<u16>()))
        .context("lm_head_int8_block32_argmax: A_bf16 byte count overflow")?;
    let partial_elems = (rows as usize)
        .checked_mul(num_v_tiles as usize)
        .context("lm_head_int8_block32_argmax: partial element count overflow")?;
    let partial_vals_bytes = partial_elems
        .checked_mul(std::mem::size_of::<f32>())
        .context("lm_head_int8_block32_argmax: partial_vals byte count overflow")?;
    let partial_ids_offset = align_up(partial_vals_bytes, 16)
        .context("lm_head_int8_block32_argmax: partial_ids_offset alignment overflow")?;
    let partial_ids_bytes = partial_elems
        .checked_mul(std::mem::size_of::<u32>())
        .context("lm_head_int8_block32_argmax: partial_ids byte count overflow")?;
    let total_scratch_bytes = partial_ids_offset
        .checked_add(partial_ids_bytes)
        .context("lm_head_int8_block32_argmax: total_scratch_bytes overflow")?;
    let out_tokens_bytes = (rows as usize)
        .checked_mul(std::mem::size_of::<u32>())
        .context("lm_head_int8_block32_argmax: out_tokens byte count overflow")?;

    Ok(LmHeadInt8Block32ArgmaxPlan {
        rows,
        vocab,
        hidden,
        num_v_tiles,
        a_bytes,
        partial_vals_bytes,
        partial_ids_offset,
        partial_ids_bytes,
        total_scratch_bytes,
        out_tokens_bytes,
        weight_plan,
    })
}

/// Resolved CUDA kernel handles for `lm_head_int8_block32_argmax_sm120.cu`.
#[derive(Clone, Copy, Debug)]
pub struct LmHeadInt8Block32Kernels {
    pub gemm: KernelHandle,
    pub argmax_stage1: KernelHandle,
    pub argmax_stage2: KernelHandle,
    pub quantize: KernelHandle,
}

impl Default for LmHeadInt8Block32Kernels {
    fn default() -> Self {
        Self {
            gemm: KernelHandle(0),
            argmax_stage1: KernelHandle(0),
            argmax_stage2: KernelHandle(0),
            quantize: KernelHandle(0),
        }
    }
}

impl PartialEq for LmHeadInt8Block32Kernels {
    fn eq(&self, other: &Self) -> bool {
        self.gemm.0 == other.gemm.0
            && self.argmax_stage1.0 == other.argmax_stage1.0
            && self.argmax_stage2.0 == other.argmax_stage2.0
            && self.quantize.0 == other.quantize.0
    }
}

impl Eq for LmHeadInt8Block32Kernels {}

impl LmHeadInt8Block32Kernels {
    /// Probe all 4 entrypoints from module `"lm_head_int8_block32_argmax_sm120"`.
    #[must_use]
    pub fn probe(gpu: &dyn GpuBackend) -> Self {
        Self {
            gemm: crate::layers::try_kernel(
                gpu,
                LM_HEAD_INT8_BLOCK32_MODULE,
                "lm_head_int8_block32_gemm_sm120",
            ),
            argmax_stage1: crate::layers::try_kernel(
                gpu,
                LM_HEAD_INT8_BLOCK32_MODULE,
                "lm_head_int8_block32_argmax_stage1_sm120",
            ),
            argmax_stage2: crate::layers::try_kernel(
                gpu,
                LM_HEAD_INT8_BLOCK32_MODULE,
                "lm_head_int8_block32_argmax_stage2_sm120",
            ),
            quantize: crate::layers::try_kernel(
                gpu,
                LM_HEAD_INT8_BLOCK32_MODULE,
                "quantize_bf16_to_int8_block32_sm120",
            ),
        }
    }

    /// True when the full BF16 logit GEMM kernel is loaded.
    #[must_use]
    pub const fn gemm_ready(self) -> bool {
        self.gemm.0 != 0
    }

    /// True when both fused online argmax stages are loaded.
    #[must_use]
    pub const fn argmax_ready(self) -> bool {
        self.argmax_stage1.0 != 0 && self.argmax_stage2.0 != 0
    }

    /// True when the GPU BF16 -> INT8 Block-32 weight quantizer is loaded.
    #[must_use]
    pub const fn quantize_ready(self) -> bool {
        self.quantize.0 != 0
    }
}

/// Process-wide per-GPU cache of [`LmHeadInt8Block32Kernels`] handles.
#[must_use]
pub fn cached_lm_head_int8_block32_kernels(gpu: &dyn GpuBackend) -> LmHeadInt8Block32Kernels {
    static CACHE: OnceLock<Mutex<HashMap<usize, LmHeadInt8Block32Kernels>>> = OnceLock::new();
    let key = std::ptr::from_ref(gpu) as *const () as usize;
    let map = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let Ok(mut guard) = map.lock() else {
        return LmHeadInt8Block32Kernels::probe(gpu);
    };
    *guard
        .entry(key)
        .or_insert_with(|| LmHeadInt8Block32Kernels::probe(gpu))
}

/// Whether lazy GPU weight quantization/allocation is allowed on this call.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Int8WeightAllocPolicy {
    /// Eager execution: may allocate and quantize the weight once into cache.
    AllowAlloc,
    /// CUDA graph capture: return cached weight if present, else `Ok(None)`
    /// with zero allocations, zero syncs, and zero launches.
    CaptureNoAlloc,
}

fn ensure_disjoint(
    a_label: &str,
    a_ptr: DevicePtr,
    a_bytes: usize,
    b_label: &str,
    b_ptr: DevicePtr,
    b_bytes: usize,
) -> Result<()> {
    if a_bytes == 0 || b_bytes == 0 {
        return Ok(());
    }
    let a_end = a_ptr
        .0
        .checked_add(a_bytes as u64)
        .with_context(|| format!("lm_head_int8_block32: {a_label} address range overflow"))?;
    let b_end = b_ptr
        .0
        .checked_add(b_bytes as u64)
        .with_context(|| format!("lm_head_int8_block32: {b_label} address range overflow"))?;
    let disjoint = a_end <= b_ptr.0 || b_end <= a_ptr.0;
    ensure!(
        disjoint,
        "lm_head_int8_block32: buffers {a_label} ({:#x}..{:#x}) and {b_label} ({:#x}..{:#x}) overlap",
        a_ptr.0,
        a_end,
        b_ptr.0,
        b_end
    );
    Ok(())
}

fn validate_device_weight(
    weight: &Int8Block32DeviceWeight,
    req_vocab: u32,
    req_hidden: u32,
) -> Result<()> {
    ensure!(
        weight.rows >= req_vocab && weight.cols == req_hidden,
        "lm_head_int8_block32: weight shape [{}, {}] incompatible with requested [{req_vocab}, {req_hidden}]",
        weight.rows,
        weight.cols
    );
    let plan = plan_lm_head_int8_block32_weight(req_vocab, req_hidden)?;
    ensure!(
        !weight.w_i8.is_null() && !weight.scales_bf16.is_null(),
        "lm_head_int8_block32: w_i8 or scales_bf16 pointer is NULL"
    );
    ensure!(
        weight.w_i8.0.is_multiple_of(16),
        "lm_head_int8_block32: w_i8 pointer must be 16B-aligned for LDG.E.EF.128"
    );
    let req_scale_align = if req_hidden.is_multiple_of(64) { 4 } else { 2 };
    ensure!(
        weight.scales_bf16.0.is_multiple_of(req_scale_align),
        "lm_head_int8_block32: scales_bf16 pointer must be {req_scale_align}B-aligned"
    );
    ensure!(
        weight.weight_bytes >= plan.weight_bytes && weight.scales_bytes >= plan.scales_bytes,
        "lm_head_int8_block32: weight buffer sizes ({} / {} B) smaller than required ({} / {} B)",
        weight.weight_bytes,
        weight.scales_bytes,
        plan.weight_bytes,
        plan.scales_bytes
    );
    ensure_disjoint(
        "w_i8",
        weight.w_i8,
        plan.weight_bytes,
        "scales_bf16",
        weight.scales_bf16,
        plan.scales_bytes,
    )
}

/// Launch `quantize_bf16_to_int8_block32_sm120` to quantize a `[rows, cols]`
/// BF16 weight matrix into `w_i8` (`[rows, cols]`) + `scales_bf16`
/// (`[rows, cols / 32]`).
///
/// Grid: `(rows, 1, 1)`  Block: `(256, 1, 1)`
///
/// # Errors
/// Fails closed before launch on invalid handle, dimensions, null/misaligned
/// pointers, or overlapping buffers.
#[allow(clippy::too_many_arguments)]
pub fn quantize_bf16_to_int8_block32_sm120(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    src_bf16: DevicePtr,
    dst_i8: DevicePtr,
    dst_scales: DevicePtr,
    rows: u32,
    cols: u32,
    stream: u64,
) -> Result<()> {
    ensure!(
        kernel.0 != 0,
        "quantize_bf16_to_int8_block32_sm120 requires a non-zero kernel handle"
    );
    let plan = plan_lm_head_int8_block32_weight(rows, cols)?;
    ensure!(
        !src_bf16.is_null() && src_bf16.0.is_multiple_of(2),
        "quantize_bf16_to_int8_block32_sm120: src_bf16 must be non-null and 2B-aligned"
    );
    ensure!(
        !dst_i8.is_null() && dst_i8.0.is_multiple_of(16),
        "quantize_bf16_to_int8_block32_sm120: dst_i8 must be non-null and 16B-aligned"
    );
    let req_scale_align = if cols.is_multiple_of(64) { 4 } else { 2 };
    ensure!(
        !dst_scales.is_null() && dst_scales.0.is_multiple_of(req_scale_align),
        "quantize_bf16_to_int8_block32_sm120: dst_scales must be non-null and {req_scale_align}B-aligned"
    );
    ensure_disjoint(
        "src_bf16",
        src_bf16,
        plan.src_bf16_bytes,
        "dst_i8",
        dst_i8,
        plan.weight_bytes,
    )?;
    ensure_disjoint(
        "src_bf16",
        src_bf16,
        plan.src_bf16_bytes,
        "dst_scales",
        dst_scales,
        plan.scales_bytes,
    )?;
    ensure_disjoint(
        "dst_i8",
        dst_i8,
        plan.weight_bytes,
        "dst_scales",
        dst_scales,
        plan.scales_bytes,
    )?;

    KernelLaunch::new(gpu, kernel)
        .grid([rows, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(src_bf16)
        .arg_ptr(dst_i8)
        .arg_ptr(dst_scales)
        .arg_u32(rows)
        .arg_u32(cols)
        .launch(stream)
}

/// Lazily quantize and cache a device BF16 LM-head weight `[vocab, hidden]`
/// into `INT8 Block-32 BF16-Scale` (`8.50 bpw`).
///
/// - Allocates ZERO VRAM unless called (never called when `ATLAS_LM_HEAD_INT8_BLOCK32`
///   is disabled).
/// - Under [`Int8WeightAllocPolicy::CaptureNoAlloc`], returns `Ok(None)` on
///   cache miss with zero allocations, zero syncs, and zero launches.
/// - On any allocation, launch, or synchronization error, frees all partially
///   allocated buffers before returning `Err` (zero device memory leak).
///
/// # Errors
/// Propagates validation, allocation, launch, or synchronization failures.
#[allow(clippy::too_many_arguments)]
pub fn ensure_cached_lm_head_int8_block32_weight(
    gpu: &dyn GpuBackend,
    quantize_kernel: KernelHandle,
    src_bf16: DevicePtr,
    vocab: u32,
    hidden: u32,
    stream: u64,
    policy: Int8WeightAllocPolicy,
) -> Result<Option<Int8Block32DeviceWeight>> {
    let plan = plan_lm_head_int8_block32_weight(vocab, hidden)?;
    ensure!(
        !src_bf16.is_null() && src_bf16.0.is_multiple_of(2),
        "ensure_cached_lm_head_int8_block32_weight: src_bf16 must be non-null and 2B-aligned"
    );

    static CACHE: OnceLock<Mutex<HashMap<(usize, u64, u32, u32), Int8Block32DeviceWeight>>> =
        OnceLock::new();
    let key = (
        std::ptr::from_ref(gpu) as *const () as usize,
        src_bf16.0,
        vocab,
        hidden,
    );
    let map = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let mut guard = map
        .lock()
        .map_err(|_| anyhow::anyhow!("lm_head_int8_block32 weight cache lock poisoned"))?;
    if let Some(existing) = guard.get(&key).copied() {
        return Ok(Some(existing));
    }
    if policy == Int8WeightAllocPolicy::CaptureNoAlloc || quantize_kernel.0 == 0 {
        return Ok(None);
    }

    let w_i8 = gpu.alloc(plan.weight_bytes)?;
    let scales_bf16 = match gpu.alloc(plan.scales_bytes) {
        Ok(ptr) => ptr,
        Err(err) => {
            let _ = gpu.free(w_i8);
            return Err(err);
        }
    };

    if let Err(err) = quantize_bf16_to_int8_block32_sm120(
        gpu,
        quantize_kernel,
        src_bf16,
        w_i8,
        scales_bf16,
        vocab,
        hidden,
        stream,
    )
    .and_then(|()| gpu.synchronize(stream))
    {
        let _ = gpu.free(w_i8);
        let _ = gpu.free(scales_bf16);
        return Err(err);
    }

    let entry = Int8Block32DeviceWeight {
        w_i8,
        scales_bf16,
        rows: vocab,
        cols: hidden,
        weight_bytes: plan.weight_bytes,
        scales_bytes: plan.scales_bytes,
    };
    guard.insert(key, entry);
    Ok(Some(entry))
}

/// Launch `lm_head_int8_block32_gemm_sm120` (full BF16 logit GEMM for
/// `M in 1..=64`, `K % 32 == 0`, arbitrary `N`).
///
/// Grid: `(ceil(N / 64), ceil(M / 16), 1)`  Block: `(128, 1, 1)`
///
/// # Errors
/// Fails closed before launch on invalid handle, dimensions, alignment,
/// null pointers, or buffer aliasing.
#[allow(clippy::too_many_arguments)]
pub fn lm_head_int8_block32_gemm_sm120(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    plan: &LmHeadInt8Block32GemmPlan,
    input_bf16: DevicePtr,
    weight: &Int8Block32DeviceWeight,
    out_logits_bf16: DevicePtr,
    out_logits_bytes: usize,
    stream: u64,
) -> Result<()> {
    ensure!(
        kernel.0 != 0,
        "lm_head_int8_block32_gemm_sm120 requires a non-zero kernel handle"
    );
    let checked = plan_lm_head_int8_block32_gemm(plan.rows, plan.vocab, plan.hidden)?;
    validate_device_weight(weight, checked.vocab, checked.hidden)?;
    ensure!(
        !input_bf16.is_null() && input_bf16.0.is_multiple_of(16),
        "lm_head_int8_block32_gemm_sm120: input_bf16 must be non-null and 16B-aligned"
    );
    ensure!(
        !out_logits_bf16.is_null() && out_logits_bf16.0.is_multiple_of(2),
        "lm_head_int8_block32_gemm_sm120: out_logits_bf16 must be non-null and 2B-aligned"
    );
    ensure!(
        out_logits_bytes >= checked.c_bytes,
        "lm_head_int8_block32_gemm_sm120: out_logits ({out_logits_bytes} B) smaller than required {} B",
        checked.c_bytes
    );
    ensure_disjoint(
        "input_bf16",
        input_bf16,
        checked.a_bytes,
        "out_logits_bf16",
        out_logits_bf16,
        checked.c_bytes,
    )?;
    ensure_disjoint(
        "w_i8",
        weight.w_i8,
        checked.weight_plan.weight_bytes,
        "out_logits_bf16",
        out_logits_bf16,
        checked.c_bytes,
    )?;
    ensure_disjoint(
        "scales_bf16",
        weight.scales_bf16,
        checked.weight_plan.scales_bytes,
        "out_logits_bf16",
        out_logits_bf16,
        checked.c_bytes,
    )?;

    KernelLaunch::new(gpu, kernel)
        .grid([checked.grid_x, checked.grid_y, 1])
        .block([128, 1, 1])
        .arg_ptr(input_bf16)
        .arg_ptr(weight.w_i8)
        .arg_ptr(weight.scales_bf16)
        .arg_ptr(out_logits_bf16)
        .arg_u32(checked.rows)
        .arg_u32(checked.vocab)
        .arg_u32(checked.hidden)
        .launch(stream)
}

/// Stage 1: Launch `lm_head_int8_block32_argmax_stage1_sm120`.
///
/// Grid: `(ceil(V / 256), 1, 1)`  Block: `(128, 1, 1)`
///
/// # Errors
/// Fails closed before launch on invalid handle, dimensions, alignment,
/// null pointers, or buffer aliasing.
#[allow(clippy::too_many_arguments)]
pub fn lm_head_int8_block32_argmax_stage1_sm120(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    plan: &LmHeadInt8Block32ArgmaxPlan,
    input_bf16: DevicePtr,
    weight: &Int8Block32DeviceWeight,
    partial_vals: DevicePtr,
    partial_ids: DevicePtr,
    stream: u64,
) -> Result<()> {
    ensure!(
        kernel.0 != 0,
        "lm_head_int8_block32_argmax_stage1_sm120 requires a non-zero kernel handle"
    );
    let checked = plan_lm_head_int8_block32_argmax(plan.rows, plan.vocab, plan.hidden)?;
    validate_device_weight(weight, checked.vocab, checked.hidden)?;
    ensure!(
        !input_bf16.is_null() && input_bf16.0.is_multiple_of(16),
        "lm_head_int8_block32_argmax_stage1_sm120: input_bf16 must be non-null and 16B-aligned"
    );
    ensure!(
        !partial_vals.is_null() && partial_vals.0.is_multiple_of(4),
        "lm_head_int8_block32_argmax_stage1_sm120: partial_vals must be non-null and 4B-aligned"
    );
    ensure!(
        !partial_ids.is_null() && partial_ids.0.is_multiple_of(4),
        "lm_head_int8_block32_argmax_stage1_sm120: partial_ids must be non-null and 4B-aligned"
    );
    ensure_disjoint(
        "partial_vals",
        partial_vals,
        checked.partial_vals_bytes,
        "partial_ids",
        partial_ids,
        checked.partial_ids_bytes,
    )?;
    ensure_disjoint(
        "input_bf16",
        input_bf16,
        checked.a_bytes,
        "partial_vals",
        partial_vals,
        checked.partial_vals_bytes,
    )?;
    ensure_disjoint(
        "input_bf16",
        input_bf16,
        checked.a_bytes,
        "partial_ids",
        partial_ids,
        checked.partial_ids_bytes,
    )?;
    ensure_disjoint(
        "w_i8",
        weight.w_i8,
        checked.weight_plan.weight_bytes,
        "partial_vals",
        partial_vals,
        checked.partial_vals_bytes,
    )?;
    ensure_disjoint(
        "w_i8",
        weight.w_i8,
        checked.weight_plan.weight_bytes,
        "partial_ids",
        partial_ids,
        checked.partial_ids_bytes,
    )?;

    KernelLaunch::new(gpu, kernel)
        .grid([checked.num_v_tiles, 1, 1])
        .block([128, 1, 1])
        .arg_ptr(input_bf16)
        .arg_ptr(weight.w_i8)
        .arg_ptr(weight.scales_bf16)
        .arg_ptr(partial_vals)
        .arg_ptr(partial_ids)
        .arg_u32(checked.rows)
        .arg_u32(checked.vocab)
        .arg_u32(checked.hidden)
        .launch(stream)
}

/// Stage 2: Launch `lm_head_int8_block32_argmax_stage2_sm120`.
///
/// Grid: `(rows, 1, 1)`  Block: `(32, 1, 1)`
///
/// # Errors
/// Fails closed before launch on invalid handle, dimensions, alignment,
/// null pointers, or buffer aliasing.
#[allow(clippy::too_many_arguments)]
pub fn lm_head_int8_block32_argmax_stage2_sm120(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    plan: &LmHeadInt8Block32ArgmaxPlan,
    partial_vals: DevicePtr,
    partial_ids: DevicePtr,
    out_tokens: DevicePtr,
    stream: u64,
) -> Result<()> {
    ensure!(
        kernel.0 != 0,
        "lm_head_int8_block32_argmax_stage2_sm120 requires a non-zero kernel handle"
    );
    let checked = plan_lm_head_int8_block32_argmax(plan.rows, plan.vocab, plan.hidden)?;
    ensure!(
        !partial_vals.is_null() && partial_vals.0.is_multiple_of(4),
        "lm_head_int8_block32_argmax_stage2_sm120: partial_vals must be non-null and 4B-aligned"
    );
    ensure!(
        !partial_ids.is_null() && partial_ids.0.is_multiple_of(4),
        "lm_head_int8_block32_argmax_stage2_sm120: partial_ids must be non-null and 4B-aligned"
    );
    ensure!(
        !out_tokens.is_null() && out_tokens.0.is_multiple_of(4),
        "lm_head_int8_block32_argmax_stage2_sm120: out_tokens must be non-null and 4B-aligned"
    );
    ensure_disjoint(
        "partial_vals",
        partial_vals,
        checked.partial_vals_bytes,
        "out_tokens",
        out_tokens,
        checked.out_tokens_bytes,
    )?;
    ensure_disjoint(
        "partial_ids",
        partial_ids,
        checked.partial_ids_bytes,
        "out_tokens",
        out_tokens,
        checked.out_tokens_bytes,
    )?;

    KernelLaunch::new(gpu, kernel)
        .grid([checked.rows, 1, 1])
        .block([32, 1, 1])
        .arg_ptr(partial_vals)
        .arg_ptr(partial_ids)
        .arg_ptr(out_tokens)
        .arg_u32(checked.rows)
        .arg_u32(checked.num_v_tiles)
        .launch(stream)
}

/// Combined two-stage fused INT8 Block-32 LM-head + online argmax dispatch.
///
/// Validates ALL stage-1 and stage-2 preconditions before issuing any GPU
/// launch so any failure leaves the stream untouched.
///
/// # Errors
/// Fails closed with zero launches on invalid handles, scratch size,
/// pointers, alignment, or buffer aliasing.
#[allow(clippy::too_many_arguments)]
pub fn lm_head_int8_block32_fused_argmax(
    gpu: &dyn GpuBackend,
    kernels: LmHeadInt8Block32Kernels,
    plan: &LmHeadInt8Block32ArgmaxPlan,
    input_bf16: DevicePtr,
    weight: &Int8Block32DeviceWeight,
    scratch: DevicePtr,
    scratch_bytes: usize,
    out_tokens: DevicePtr,
    stream: u64,
) -> Result<()> {
    let checked = plan_lm_head_int8_block32_argmax(plan.rows, plan.vocab, plan.hidden)?;
    ensure!(
        kernels.argmax_ready(),
        "lm_head_int8_block32_fused_argmax requires both stage1 and stage2 kernel handles"
    );
    ensure!(
        !scratch.is_null() && scratch.0.is_multiple_of(4),
        "lm_head_int8_block32_fused_argmax: scratch must be non-null and 4B-aligned"
    );
    ensure!(
        scratch_bytes >= checked.total_scratch_bytes,
        "lm_head_int8_block32_fused_argmax: scratch ({scratch_bytes} B) smaller than required {} B",
        checked.total_scratch_bytes
    );
    ensure!(
        !out_tokens.is_null() && out_tokens.0.is_multiple_of(4),
        "lm_head_int8_block32_fused_argmax: out_tokens must be non-null and 4B-aligned"
    );
    let partial_vals = scratch;
    let partial_ids = scratch.offset(checked.partial_ids_offset);
    ensure_disjoint(
        "partial_vals",
        partial_vals,
        checked.partial_vals_bytes,
        "out_tokens",
        out_tokens,
        checked.out_tokens_bytes,
    )?;
    ensure_disjoint(
        "partial_ids",
        partial_ids,
        checked.partial_ids_bytes,
        "out_tokens",
        out_tokens,
        checked.out_tokens_bytes,
    )?;

    lm_head_int8_block32_argmax_stage1_sm120(
        gpu,
        kernels.argmax_stage1,
        &checked,
        input_bf16,
        weight,
        partial_vals,
        partial_ids,
        stream,
    )?;
    lm_head_int8_block32_argmax_stage2_sm120(
        gpu,
        kernels.argmax_stage2,
        &checked,
        partial_vals,
        partial_ids,
        out_tokens,
        stream,
    )
}

// ── Bit-Exact Host Packer & Reference Oracles ──────────────────────────────

/// Exact IEEE-754 FP32 -> BF16 round-to-nearest-even mirroring PTX
/// `cvt.rn.bf16.f32`.
#[inline]
#[must_use]
pub const fn f32_to_bf16_rn_cpu(x: f32) -> u16 {
    let bits = x.to_bits();
    let exp = bits & 0x7F80_0000;
    let mant = bits & 0x007F_FFFF;
    if exp == 0x7F80_0000 {
        if mant != 0 {
            return ((bits >> 16) as u16) | 0x0040;
        }
        return (bits >> 16) as u16;
    }
    let lsb = (bits >> 16) & 1;
    let rounding_bias = 0x7FFF_u32.wrapping_add(lsb);
    (bits.wrapping_add(rounding_bias) >> 16) as u16
}

/// Convert raw BF16 `u16` bits to FP32.
#[inline]
#[must_use]
pub const fn bf16_to_f32_cpu(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// Lossless 0-ULP conversion of `int8` (`-127..=127`) to BF16 `u16` bits.
#[inline]
#[must_use]
pub const fn int8_to_bf16_lossless_cpu(q: i8) -> u16 {
    f32_to_bf16_rn_cpu(q as f32)
}

/// Host-packed `INT8 Block-32 BF16-Scale` (`8.50 bpw`) weight matrix.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Int8Block32HostWeight {
    pub w_i8: Vec<i8>,
    pub scales_bf16: Vec<u16>,
    pub rows: usize,
    pub cols: usize,
}

/// Quantize a row-major `[rows, cols]` BF16 weight matrix (`u16` bits) into
/// `INT8 Block-32 BF16-Scale` (`8.50 bpw`), matching
/// `quantize_bf16_to_int8_block32_sm120` bit-for-bit:
/// - Non-finite inputs are sanitized to `0.0`.
/// - `scale_bf16 = rne_bf16(amax_32 / 127.0)`.
/// - `q = round_ties_even(val / bf16_to_f32(scale_bf16)).clamp(-127, 127) as i8`.
///
/// # Errors
/// Fails closed if `rows == 0`, `cols < 32`, `cols % 32 != 0`, or slice length
/// does not equal `rows * cols`.
pub fn quantize_bf16_weight_to_int8_block32(
    weights_bf16: &[u16],
    rows: usize,
    cols: usize,
) -> Result<Int8Block32HostWeight> {
    ensure!(rows > 0, "quantize_bf16_weight_to_int8_block32: rows == 0");
    ensure!(
        cols >= (LM_HEAD_INT8_BLOCK_SIZE as usize)
            && cols.is_multiple_of(LM_HEAD_INT8_BLOCK_SIZE as usize),
        "quantize_bf16_weight_to_int8_block32: cols={cols} must be >= 32 and divisible by 32"
    );
    let expected = rows
        .checked_mul(cols)
        .context("quantize_bf16_weight_to_int8_block32: rows * cols overflow")?;
    ensure!(
        weights_bf16.len() == expected,
        "quantize_bf16_weight_to_int8_block32: len {} != {expected}",
        weights_bf16.len()
    );

    let num_blocks = cols / (LM_HEAD_INT8_BLOCK_SIZE as usize);
    let mut w_i8 = vec![0i8; expected];
    let mut scales_bf16 = vec![0u16; rows * num_blocks];

    for r in 0..rows {
        for b in 0..num_blocks {
            let base = r * cols + b * 32;
            let mut amax = 0.0_f32;
            let mut block_vals = [0.0_f32; 32];
            for j in 0..32 {
                let raw = bf16_to_f32_cpu(weights_bf16[base + j]);
                let v = if raw.is_finite() { raw } else { 0.0_f32 };
                block_vals[j] = v;
                let av = v.abs();
                if av > amax {
                    amax = av;
                }
            }
            let raw_scale = if amax > 0.0 { amax / 127.0 } else { 0.0 };
            let scale_bits = f32_to_bf16_rn_cpu(raw_scale);
            let scale_rt = bf16_to_f32_cpu(scale_bits);
            scales_bf16[r * num_blocks + b] = scale_bits;

            for j in 0..32 {
                let q = if scale_rt > 0.0 {
                    (block_vals[j] / scale_rt)
                        .round_ties_even()
                        .clamp(-127.0, 127.0) as i8
                } else {
                    0i8
                };
                w_i8[base + j] = q;
            }
        }
    }

    Ok(Int8Block32HostWeight {
        w_i8,
        scales_bf16,
        rows,
        cols,
    })
}

/// Deterministic tie-breaking comparator matching `w8_better` in CUDA:
/// - `LM_HEAD_INT8_INVALID_ID` never wins.
/// - `NaN` never wins.
/// - Strictly larger value wins; exact ties pick strictly lower vocab index.
#[inline]
#[must_use]
pub fn int8_lmh_better_cpu(val: f32, idx: u32, best_val: f32, best_idx: u32) -> bool {
    if idx == LM_HEAD_INT8_INVALID_ID || val.is_nan() {
        return false;
    }
    if best_idx == LM_HEAD_INT8_INVALID_ID {
        return true;
    }
    if val > best_val {
        return true;
    }
    val == best_val && idx < best_idx
}

/// Compute FP32 accumulated logits matching the exact block-32 factorized MMA
/// accumulation order of `lm_head_int8_block32_{gemm,argmax_stage1}_sm120`.
fn compute_int8_block32_logits_f32_cpu(
    a_bf16: &[u16],
    weight: &Int8Block32HostWeight,
    m: usize,
    v: usize,
    k: usize,
) -> Result<Vec<f32>> {
    ensure!(m > 0 && v > 0, "m={m} and v={v} must be > 0");
    ensure!(
        k >= 32 && k.is_multiple_of(32),
        "k={k} must be >= 32 and divisible by 32"
    );
    ensure!(
        weight.rows >= v && weight.cols == k,
        "weight shape [{}, {}] incompatible with [{v}, {k}]",
        weight.rows,
        weight.cols
    );
    ensure!(
        a_bf16.len() == m * k,
        "a_bf16 len {} != {}",
        a_bf16.len(),
        m * k
    );

    let num_blocks = k / 32;
    let mut logits = vec![0.0_f32; m * v];
    for r in 0..m {
        for c in 0..v {
            let mut acc = 0.0_f32;
            for b in 0..num_blocks {
                let k_base = b * 32;
                let mut blk_dot = 0.0_f32;
                // Two k=16 halves matching two `m16n8k16` MMA instructions.
                for half in 0..2 {
                    let h_off = k_base + half * 16;
                    let mut h_dot = 0.0_f32;
                    for j in 0..16 {
                        let a_val = bf16_to_f32_cpu(a_bf16[r * k + h_off + j]);
                        let w_val = bf16_to_f32_cpu(int8_to_bf16_lossless_cpu(
                            weight.w_i8[c * k + h_off + j],
                        ));
                        h_dot += a_val * w_val;
                    }
                    blk_dot += h_dot;
                }
                let sc = bf16_to_f32_cpu(weight.scales_bf16[c * num_blocks + b]);
                acc += blk_dot * sc;
            }
            logits[r * v + c] = acc;
        }
    }
    Ok(logits)
}

/// Exact CPU mirror of `lm_head_int8_block32_gemm_sm120`, returning BF16 `u16`
/// logits `[m, v]`.
///
/// # Errors
/// Fails closed if `m` is outside `1..=64` or dimensions/slice sizes mismatch.
pub fn lm_head_int8_block32_gemm_cpu(
    a_bf16: &[u16],
    weight: &Int8Block32HostWeight,
    m: usize,
    v: usize,
    k: usize,
) -> Result<Vec<u16>> {
    ensure!(
        (1..=(LM_HEAD_INT8_GEMM_MAX_M as usize)).contains(&m),
        "lm_head_int8_block32_gemm_cpu: m={m} must lie in 1..={LM_HEAD_INT8_GEMM_MAX_M}"
    );
    let f32_logits = compute_int8_block32_logits_f32_cpu(a_bf16, weight, m, v, k)?;
    Ok(f32_logits.into_iter().map(f32_to_bf16_rn_cpu).collect())
}

/// Exact CPU mirror of `lm_head_int8_block32_argmax_stage1_sm120`.
///
/// # Errors
/// Fails closed if `m` is outside `1..=16` or dimensions/slice sizes mismatch.
pub fn lm_head_int8_block32_argmax_stage1_cpu(
    a_bf16: &[u16],
    weight: &Int8Block32HostWeight,
    m: usize,
    v: usize,
    k: usize,
) -> Result<(Vec<f32>, Vec<u32>)> {
    ensure!(
        (1..=(LM_HEAD_INT8_ARGMAX_MAX_M as usize)).contains(&m),
        "lm_head_int8_block32_argmax_stage1_cpu: m={m} must lie in 1..={LM_HEAD_INT8_ARGMAX_MAX_M}"
    );
    let f32_logits = compute_int8_block32_logits_f32_cpu(a_bf16, weight, m, v, k)?;
    let v_tile = LM_HEAD_INT8_ARGMAX_V_TILE as usize;
    let num_v_tiles = v.div_ceil(v_tile);
    let mut partial_vals = vec![f32::NEG_INFINITY; m * num_v_tiles];
    let mut partial_ids = vec![LM_HEAD_INT8_INVALID_ID; m * num_v_tiles];

    for r in 0..m {
        for tile in 0..num_v_tiles {
            let v_start = tile * v_tile;
            let v_end = (v_start + v_tile).min(v);
            let mut best_v = f32::NEG_INFINITY;
            let mut best_i = LM_HEAD_INT8_INVALID_ID;
            for col in v_start..v_end {
                let val = f32_logits[r * v + col];
                if int8_lmh_better_cpu(val, col as u32, best_v, best_i) {
                    best_v = val;
                    best_i = col as u32;
                }
            }
            partial_vals[r * num_v_tiles + tile] = best_v;
            partial_ids[r * num_v_tiles + tile] = best_i;
        }
    }
    Ok((partial_vals, partial_ids))
}

/// Exact CPU mirror of `lm_head_int8_block32_argmax_stage2_sm120`.
///
/// # Errors
/// Fails closed if slice lengths mismatch `m * num_v_tiles` or `m == 0` / `num_v_tiles == 0`.
pub fn lm_head_int8_block32_argmax_stage2_cpu(
    partial_vals: &[f32],
    partial_ids: &[u32],
    m: usize,
    num_v_tiles: usize,
) -> Result<Vec<u32>> {
    ensure!(
        m > 0 && num_v_tiles > 0,
        "lm_head_int8_block32_argmax_stage2_cpu: m={m}, num_v_tiles={num_v_tiles} must be > 0"
    );
    let expected = m
        .checked_mul(num_v_tiles)
        .context("lm_head_int8_block32_argmax_stage2_cpu: m * num_v_tiles overflow")?;
    ensure!(
        partial_vals.len() == expected && partial_ids.len() == expected,
        "lm_head_int8_block32_argmax_stage2_cpu: partial lens ({}, {}) != {expected}",
        partial_vals.len(),
        partial_ids.len()
    );

    let mut out = Vec::with_capacity(m);
    for r in 0..m {
        let mut best_v = f32::NEG_INFINITY;
        let mut best_i = LM_HEAD_INT8_INVALID_ID;
        for t in 0..num_v_tiles {
            let val = partial_vals[r * num_v_tiles + t];
            let id = partial_ids[r * num_v_tiles + t];
            if int8_lmh_better_cpu(val, id, best_v, best_i) {
                best_v = val;
                best_i = id;
            }
        }
        out.push(if best_i == LM_HEAD_INT8_INVALID_ID {
            0
        } else {
            best_i
        });
    }
    Ok(out)
}

/// Combined two-stage CPU reference for fused INT8 Block-32 LM-head argmax.
///
/// # Errors
/// Propagates validation errors from stage 1 and stage 2.
pub fn lm_head_int8_block32_fused_argmax_cpu(
    a_bf16: &[u16],
    weight: &Int8Block32HostWeight,
    m: usize,
    v: usize,
    k: usize,
) -> Result<Vec<u32>> {
    let num_v_tiles = v.div_ceil(LM_HEAD_INT8_ARGMAX_V_TILE as usize);
    let (pv, pi) = lm_head_int8_block32_argmax_stage1_cpu(a_bf16, weight, m, v, k)?;
    lm_head_int8_block32_argmax_stage2_cpu(&pv, &pi, m, num_v_tiles)
}

#[cfg(test)]
mod tests {
    use super::*;
    use spark_runtime::gpu::mock::MockGpuBackend;
    use std::sync::atomic::{AtomicBool, Ordering};

    fn synthetic_gaussian_bf16(len: usize, std_dev: f32, seed: u64) -> Vec<u16> {
        let mut s = seed ^ 0xA076_1D64_78BD_642F;
        let mut next_u = || -> f64 {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let bits = (s >> 11) as f64 / ((1u64 << 53) as f64);
            bits.clamp(1.0e-12, 1.0 - 1.0e-12)
        };
        let mut out = Vec::with_capacity(len);
        while out.len() < len {
            let u1 = next_u();
            let u2 = next_u();
            let r = (-2.0 * u1.ln()).sqrt();
            let theta = 2.0 * std::f64::consts::PI * u2;
            let z0 = (r * theta.cos()) as f32 * std_dev;
            let z1 = (r * theta.sin()) as f32 * std_dev;
            out.push(f32_to_bf16_rn_cpu(z0));
            if out.len() < len {
                out.push(f32_to_bf16_rn_cpu(z1));
            }
        }
        out
    }

    fn softmax_f64(logits: &[f64]) -> Vec<f64> {
        let max_v = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<f64> = logits.iter().map(|&x| (x - max_v).exp()).collect();
        let sum: f64 = exps.iter().sum();
        exps.into_iter().map(|e| e / sum).collect()
    }

    #[test]
    fn int8_to_bf16_is_zero_ulp_lossless_for_all_256_values_and_magic_bias_identity() {
        for q in i8::MIN..=i8::MAX {
            let u = q as u8;
            let hi_bits = 0x4300u16 | u16::from(u & 0x7F);
            let bias_bits = 0x4300u16 | u16::from(u & 0x80);
            let magic_bf16 =
                f32_to_bf16_rn_cpu(bf16_to_f32_cpu(hi_bits) - bf16_to_f32_cpu(bias_bits));
            let direct_bf16 = int8_to_bf16_lossless_cpu(q);
            assert_eq!(
                magic_bf16, direct_bf16,
                "magic-bias 0x4300 mismatch at q={q}: magic={magic_bf16:#06x} direct={direct_bf16:#06x}"
            );
            let rt = bf16_to_f32_cpu(magic_bf16);
            assert_eq!(
                rt,
                f32::from(q),
                "int8 -> BF16 roundtrip drift at q={q}: bf16={magic_bf16:#06x} rt={rt}"
            );
        }
    }

    #[test]
    fn strict_zero_or_one_env_parser_and_non_utf8_fail_closed() {
        assert_eq!(
            LmHeadInt8Block32Policy::parse_os(None).unwrap(),
            LmHeadInt8Block32Policy::Disabled
        );
        assert_eq!(
            LmHeadInt8Block32Policy::parse_optional(Some("0")).unwrap(),
            LmHeadInt8Block32Policy::Disabled
        );
        assert_eq!(
            LmHeadInt8Block32Policy::parse_optional(Some("1")).unwrap(),
            LmHeadInt8Block32Policy::Enabled
        );
        for bad in [
            "", " ", "0 ", " 1", "true", "false", "on", "off", "yes", "2", "-1",
        ] {
            let err = LmHeadInt8Block32Policy::parse_optional(Some(bad)).unwrap_err();
            assert_eq!(
                err,
                LmHeadInt8Block32ConfigError::InvalidToken(bad.to_owned())
            );
            assert!(err.to_string().contains(LM_HEAD_INT8_BLOCK32_ENV));
        }

        #[cfg(unix)]
        {
            use std::os::unix::ffi::OsStrExt;
            let non_utf8 = OsStr::from_bytes(&[0xFF, 0xFE]);
            let err = LmHeadInt8Block32Policy::parse_os(Some(non_utf8)).unwrap_err();
            assert_eq!(err, LmHeadInt8Block32ConfigError::NonUtf8);
            assert!(err.to_string().contains("UTF-8"));
        }
    }

    #[test]
    fn verify_width_invariance_m1_m8_m16_and_non_tile_vocab_bit_exact() {
        // Test BOTH odd-block K=96 (K % 64 == 32) and even-block K=128 with non-tile V=277.
        for &(v, k) in &[(277usize, 96usize), (277usize, 128usize)] {
            let w_bf16 = synthetic_gaussian_bf16(v * k, 0.04, 0x1010_2020 + k as u64);
            let w_q = quantize_bf16_weight_to_int8_block32(&w_bf16, v, k).unwrap();
            let a_16 = synthetic_gaussian_bf16(16 * k, 1.2, 0x3030_4040 + k as u64);

            let logits_m16 = lm_head_int8_block32_gemm_cpu(&a_16, &w_q, 16, v, k).unwrap();
            let argmax_m16 = lm_head_int8_block32_fused_argmax_cpu(&a_16, &w_q, 16, v, k).unwrap();

            // M=1 row-by-row MUST match M=16 bit-for-bit (0 ULP) in BOTH logits and argmax.
            for r in 0..16 {
                let a_row = &a_16[r * k..(r + 1) * k];
                let logits_m1 = lm_head_int8_block32_gemm_cpu(a_row, &w_q, 1, v, k).unwrap();
                let argmax_m1 =
                    lm_head_int8_block32_fused_argmax_cpu(a_row, &w_q, 1, v, k).unwrap();
                assert_eq!(
                    logits_m1.as_slice(),
                    &logits_m16[r * v..(r + 1) * v],
                    "M=1 vs M=16 BF16 logit drift at row {r} (V={v}, K={k})"
                );
                assert_eq!(
                    argmax_m1[0], argmax_m16[r],
                    "M=1 vs M=16 argmax drift at row {r} (V={v}, K={k})"
                );
            }

            // M=8 halves MUST match M=16 bit-for-bit (0 ULP).
            for half in 0..2 {
                let r0 = half * 8;
                let a_8 = &a_16[r0 * k..(r0 + 8) * k];
                let logits_m8 = lm_head_int8_block32_gemm_cpu(a_8, &w_q, 8, v, k).unwrap();
                let argmax_m8 = lm_head_int8_block32_fused_argmax_cpu(a_8, &w_q, 8, v, k).unwrap();
                assert_eq!(logits_m8.as_slice(), &logits_m16[r0 * v..(r0 + 8) * v]);
                assert_eq!(argmax_m8.as_slice(), &argmax_m16[r0..r0 + 8]);
            }
        }
    }

    #[test]
    fn fused_argmax_enforces_lowest_vocab_tie_break_and_nan_exclusion() {
        let vals = vec![
            // Row 0: exact tie across 4 tiles -> lowest vocab ID (17) wins.
            3.25,
            3.25,
            3.25,
            3.25,
            // Row 1: NaNs ignored, tie on -1.5 -> lowest vocab ID (260) wins.
            f32::NAN,
            -1.5,
            f32::NAN,
            -1.5,
            // Row 2: -Inf tie -> lowest vocab ID (12) wins (not INVALID_ID).
            f32::NEG_INFINITY,
            f32::NEG_INFINITY,
            f32::NEG_INFINITY,
            f32::NEG_INFINITY,
            // Row 3: all NaN -> safe fallback token 0.
            f32::NAN,
            f32::NAN,
            f32::NAN,
            f32::NAN,
        ];
        let ids = vec![
            512, 17, 1024, 99, // row 0 -> 17
            0, 400, 1, 260, // row 1 -> 260
            900, 12, 300, 12, // row 2 -> 12
            5, 6, 7, 8, // row 3 -> 0
        ];
        let out = lm_head_int8_block32_argmax_stage2_cpu(&vals, &ids, 4, 4).unwrap();
        assert_eq!(out, vec![17, 260, 12, 0]);
    }

    #[test]
    fn gaussian_lm_head_certifies_sqnr_above_44db_kl_below_1e_6_and_zero_top1_flips() {
        let v = 512usize;
        let k = 256usize;
        let m = 8usize;
        // Realistic RMSNormed LLM hidden state (std ~ 0.5) with mild top-token
        // alignment (as in real autoregressive decoding) and Gaussian lm_head weights (std ~ 0.02).
        let w_bf16 = synthetic_gaussian_bf16(v * k, 0.02, 0x5555_6666);
        let mut a_bf16 = synthetic_gaussian_bf16(m * k, 0.5, 0x7777_8888);
        for r in 0..m {
            let target_tok = (r * 59 + 17) % v;
            for j in 0..k {
                let base = bf16_to_f32_cpu(a_bf16[r * k + j]);
                let w_t = bf16_to_f32_cpu(w_bf16[target_tok * k + j]);
                a_bf16[r * k + j] = f32_to_bf16_rn_cpu(base + 4.0 * w_t);
            }
        }

        let w_q = quantize_bf16_weight_to_int8_block32(&w_bf16, v, k).unwrap();

        // 1. Weight SQNR >= 44.0 dB.
        let num_blocks = k / 32;
        let mut w_sig_pow = 0.0_f64;
        let mut w_err_pow = 0.0_f64;
        for r in 0..v {
            for b in 0..num_blocks {
                let sc = f64::from(bf16_to_f32_cpu(w_q.scales_bf16[r * num_blocks + b]));
                for j in 0..32 {
                    let idx = r * k + b * 32 + j;
                    let orig = f64::from(bf16_to_f32_cpu(w_bf16[idx]));
                    let deq = f64::from(w_q.w_i8[idx]) * sc;
                    w_sig_pow += orig * orig;
                    let d = orig - deq;
                    w_err_pow += d * d;
                }
            }
        }
        let w_sqnr_db = 10.0 * (w_sig_pow / w_err_pow).log10();
        assert!(
            w_sqnr_db >= 44.0,
            "INT8 block-32 weight SQNR {w_sqnr_db:.2} dB < 44.0 dB"
        );

        // 2. Exact FP64 reference logits vs INT8 block-32 FP32/BF16 logits + fused argmax.
        let mut ref_logits = vec![0.0_f64; m * v];
        for r in 0..m {
            for c in 0..v {
                let mut dot = 0.0_f64;
                for j in 0..k {
                    dot += f64::from(bf16_to_f32_cpu(a_bf16[r * k + j]))
                        * f64::from(bf16_to_f32_cpu(w_bf16[c * k + j]));
                }
                ref_logits[r * v + c] = dot;
            }
        }
        let q_logits_f32 = compute_int8_block32_logits_f32_cpu(&a_bf16, &w_q, m, v, k).unwrap();
        let fused_ids = lm_head_int8_block32_fused_argmax_cpu(&a_bf16, &w_q, m, v, k).unwrap();

        let mut l_sig_pow = 0.0_f64;
        let mut l_err_pow = 0.0_f64;
        let mut total_kl = 0.0_f64;
        let mut top1_flips = 0usize;
        for r in 0..m {
            let ref_row = &ref_logits[r * v..(r + 1) * v];
            let cand_row: Vec<f64> = q_logits_f32[r * v..(r + 1) * v]
                .iter()
                .map(|&x| f64::from(x))
                .collect();
            for (&rv, &cv) in ref_row.iter().zip(&cand_row) {
                l_sig_pow += rv * rv;
                let d = rv - cv;
                l_err_pow += d * d;
            }
            let ref_top1 = ref_row
                .iter()
                .enumerate()
                .max_by(|(ia, a), (ib, b)| a.total_cmp(b).then_with(|| ib.cmp(ia)))
                .map(|(i, _)| i as u32)
                .unwrap();
            if fused_ids[r] != ref_top1 {
                top1_flips += 1;
            }
            let p = softmax_f64(ref_row);
            let q = softmax_f64(&cand_row);
            let kl: f64 = p
                .iter()
                .zip(&q)
                .map(|(&pi, &qi)| {
                    if pi > 0.0 {
                        pi * (pi / qi.max(1e-300)).ln()
                    } else {
                        0.0
                    }
                })
                .sum();
            total_kl += kl;
        }
        let l_sqnr_db = 10.0 * (l_sig_pow / l_err_pow).log10();
        let mean_kl = total_kl / (m as f64);
        assert!(
            l_sqnr_db >= 44.0,
            "INT8 block-32 logit SQNR {l_sqnr_db:.2} dB < 44.0 dB"
        );
        assert!(
            mean_kl < 1.0e-6,
            "INT8 block-32 mean D_KL {mean_kl:.3e} >= 1e-6"
        );
        assert_eq!(top1_flips, 0, "unexpected top-1 flips: {top1_flips}/{m}");
    }

    #[test]
    fn mock_gpu_launchers_validate_alignment_aliasing_and_emit_expected_launches() {
        let kernels = LmHeadInt8Block32Kernels {
            gemm: KernelHandle(1),
            argmax_stage1: KernelHandle(2),
            argmax_stage2: KernelHandle(3),
            quantize: KernelHandle(4),
        };
        let w_plan = plan_lm_head_int8_block32_weight(248_077, 2560).unwrap();
        let dev_w = Int8Block32DeviceWeight {
            w_i8: DevicePtr(0x1_0000_0000),
            scales_bf16: DevicePtr(0x2_0000_0000),
            rows: 248_077,
            cols: 2560,
            weight_bytes: w_plan.weight_bytes,
            scales_bytes: w_plan.scales_bytes,
        };

        // 1. Full BF16 logit GEMM -> 1 launch.
        let gpu_gemm = MockGpuBackend::new();
        let gplan = plan_lm_head_int8_block32_gemm(16, 248_077, 2560).unwrap();
        lm_head_int8_block32_gemm_sm120(
            &gpu_gemm,
            kernels.gemm,
            &gplan,
            DevicePtr(0x3_0000_0000),
            &dev_w,
            DevicePtr(0x4_0000_0000),
            gplan.c_bytes,
            0x11,
        )
        .unwrap();
        assert_eq!(gpu_gemm.launch_count(), 1);

        // 2. Fused two-stage argmax -> 2 launches, compact scratch <= 62 KiB at M=8.
        let gpu_arg = MockGpuBackend::new();
        let aplan8 = plan_lm_head_int8_block32_argmax(8, 248_077, 2560).unwrap();
        assert!(aplan8.total_scratch_bytes <= 64 * 1024);
        lm_head_int8_block32_fused_argmax(
            &gpu_arg,
            kernels,
            &aplan8,
            DevicePtr(0x3_0000_0000),
            &dev_w,
            DevicePtr(0x4_0000_0000),
            aplan8.total_scratch_bytes,
            DevicePtr(0x5_0000_0000),
            0x22,
        )
        .unwrap();
        assert_eq!(gpu_arg.launch_count(), 2);

        // 3. Fail-closed checks MUST reject with 0 launches.
        let gpu_err = MockGpuBackend::new();
        assert!(plan_lm_head_int8_block32_gemm(0, 256, 64).is_err());
        assert!(plan_lm_head_int8_block32_gemm(65, 256, 64).is_err());
        assert!(plan_lm_head_int8_block32_argmax(0, 256, 64).is_err());
        assert!(plan_lm_head_int8_block32_argmax(17, 256, 64).is_err());
        assert!(plan_lm_head_int8_block32_weight(0, 64).is_err());
        assert!(plan_lm_head_int8_block32_weight(64, 16).is_err());
        assert!(plan_lm_head_int8_block32_weight(64, 48).is_err());

        // Missing stage2 handle rejects BEFORE launching stage1.
        assert!(
            lm_head_int8_block32_fused_argmax(
                &gpu_err,
                LmHeadInt8Block32Kernels {
                    argmax_stage2: KernelHandle(0),
                    ..kernels
                },
                &aplan8,
                DevicePtr(0x3_0000_0000),
                &dev_w,
                DevicePtr(0x4_0000_0000),
                aplan8.total_scratch_bytes,
                DevicePtr(0x5_0000_0000),
                0,
            )
            .is_err()
        );
        // Overlapping scratch / out_tokens rejects BEFORE launching stage1.
        assert!(
            lm_head_int8_block32_fused_argmax(
                &gpu_err,
                kernels,
                &aplan8,
                DevicePtr(0x3_0000_0000),
                &dev_w,
                DevicePtr(0x4_0000_0000),
                aplan8.total_scratch_bytes,
                DevicePtr(0x4_0000_0000),
                0,
            )
            .is_err()
        );
        // Undersized scratch rejects BEFORE launching stage1.
        assert!(
            lm_head_int8_block32_fused_argmax(
                &gpu_err,
                kernels,
                &aplan8,
                DevicePtr(0x3_0000_0000),
                &dev_w,
                DevicePtr(0x4_0000_0000),
                aplan8.total_scratch_bytes - 4,
                DevicePtr(0x5_0000_0000),
                0,
            )
            .is_err()
        );
        // Misaligned A (non-16B), misaligned scales (non-4B for K%64==0), or aliasing C with A.
        assert!(
            lm_head_int8_block32_gemm_sm120(
                &gpu_err,
                kernels.gemm,
                &gplan,
                DevicePtr(0x3_0000_0008),
                &dev_w,
                DevicePtr(0x4_0000_0000),
                gplan.c_bytes,
                0,
            )
            .is_err()
        );
        assert!(
            lm_head_int8_block32_gemm_sm120(
                &gpu_err,
                kernels.gemm,
                &gplan,
                DevicePtr(0x3_0000_0000),
                &Int8Block32DeviceWeight {
                    scales_bf16: DevicePtr(0x2_0000_0002),
                    ..dev_w
                },
                DevicePtr(0x4_0000_0000),
                gplan.c_bytes,
                0,
            )
            .is_err()
        );
        assert!(
            lm_head_int8_block32_gemm_sm120(
                &gpu_err,
                kernels.gemm,
                &gplan,
                DevicePtr(0x3_0000_0000),
                &dev_w,
                DevicePtr(0x3_0000_0000),
                gplan.c_bytes,
                0,
            )
            .is_err()
        );
        assert_eq!(gpu_err.launch_count(), 0);
    }

    #[test]
    fn cached_weight_quantizer_respects_capture_no_alloc_and_frees_on_error() {
        struct SyncFailGpu {
            inner: MockGpuBackend,
            fail_sync: AtomicBool,
        }

        impl GpuBackend for SyncFailGpu {
            fn alloc(&self, bytes: usize) -> Result<DevicePtr> {
                self.inner.alloc(bytes)
            }
            fn alloc_managed(&self, bytes: usize) -> Result<DevicePtr> {
                self.inner.alloc_managed(bytes)
            }
            fn free(&self, ptr: DevicePtr) -> Result<()> {
                self.inner.free(ptr)
            }
            fn copy_h2d(&self, src: &[u8], dst: DevicePtr) -> Result<()> {
                self.inner.copy_h2d(src, dst)
            }
            fn copy_d2h(&self, src: DevicePtr, dst: &mut [u8]) -> Result<()> {
                self.inner.copy_d2h(src, dst)
            }
            fn copy_d2d(&self, src: DevicePtr, dst: DevicePtr, bytes: usize) -> Result<()> {
                self.inner.copy_d2d(src, dst, bytes)
            }
            fn launch(
                &self,
                func: KernelHandle,
                grid: [u32; 3],
                block: [u32; 3],
                shared_mem: u32,
                stream: u64,
                params: &mut [*mut std::ffi::c_void],
            ) -> Result<()> {
                self.inner
                    .launch(func, grid, block, shared_mem, stream, params)
            }
            fn synchronize(&self, stream: u64) -> Result<()> {
                if self.fail_sync.load(Ordering::Relaxed) {
                    anyhow::bail!("injected sync failure");
                }
                self.inner.synchronize(stream)
            }
            fn default_stream(&self) -> u64 {
                self.inner.default_stream()
            }
            fn kernel(&self, module: &str, func_name: &str) -> Result<KernelHandle> {
                self.inner.kernel(module, func_name)
            }
            fn memset(&self, ptr: DevicePtr, value: u8, bytes: usize) -> Result<()> {
                self.inner.memset(ptr, value, bytes)
            }
            fn memset_async(
                &self,
                ptr: DevicePtr,
                value: u8,
                bytes: usize,
                stream: u64,
            ) -> Result<()> {
                self.inner.memset_async(ptr, value, bytes, stream)
            }
            fn total_memory(&self) -> Result<usize> {
                self.inner.total_memory()
            }
            fn free_memory(&self) -> Result<usize> {
                self.inner.free_memory()
            }
        }

        let gpu = Box::new(SyncFailGpu {
            inner: MockGpuBackend::new(),
            fail_sync: AtomicBool::new(false),
        });
        let src = DevicePtr(0x6_0000_0000);

        // 1. CaptureNoAlloc on cache miss -> Ok(None), 0 allocs, 0 launches.
        assert_eq!(
            ensure_cached_lm_head_int8_block32_weight(
                gpu.as_ref(),
                KernelHandle(4),
                src,
                128,
                64,
                0,
                Int8WeightAllocPolicy::CaptureNoAlloc,
            )
            .unwrap(),
            None
        );
        assert_eq!(gpu.inner.alloc_count(), 0);
        assert_eq!(gpu.inner.launch_count(), 0);

        // 2. Sync failure during quantization MUST free both allocated buffers (0 leaks).
        gpu.fail_sync.store(true, Ordering::Relaxed);
        assert!(
            ensure_cached_lm_head_int8_block32_weight(
                gpu.as_ref(),
                KernelHandle(4),
                src,
                128,
                64,
                0,
                Int8WeightAllocPolicy::AllowAlloc,
            )
            .is_err()
        );
        assert_eq!(gpu.inner.alloc_count(), 0);
        assert_eq!(gpu.inner.invalid_free_count(), 0);

        // 3. Second alloc failure frees first buffer cleanly (0 leaks).
        gpu.fail_sync.store(false, Ordering::Relaxed);
        gpu.inner.fail_alloc_after(1);
        assert!(
            ensure_cached_lm_head_int8_block32_weight(
                gpu.as_ref(),
                KernelHandle(4),
                src,
                128,
                64,
                0,
                Int8WeightAllocPolicy::AllowAlloc,
            )
            .is_err()
        );
        assert_eq!(gpu.inner.alloc_count(), 0);
        assert_eq!(gpu.inner.invalid_free_count(), 0);

        // 4. Successful quantization caches weight; CaptureNoAlloc hits cache with 0 extra allocs.
        let cached = ensure_cached_lm_head_int8_block32_weight(
            gpu.as_ref(),
            KernelHandle(4),
            src,
            128,
            64,
            0,
            Int8WeightAllocPolicy::AllowAlloc,
        )
        .unwrap()
        .unwrap();
        assert_eq!(gpu.inner.alloc_count(), 2);
        let hit = ensure_cached_lm_head_int8_block32_weight(
            gpu.as_ref(),
            KernelHandle(4),
            src,
            128,
            64,
            0,
            Int8WeightAllocPolicy::CaptureNoAlloc,
        )
        .unwrap()
        .unwrap();
        assert_eq!(hit, cached);
        assert_eq!(gpu.inner.alloc_count(), 2);
    }

    #[test]
    fn cuda_source_preserves_isa_and_determinism_contracts() {
        let cu =
            include_str!("../../../../../kernels/gb10/common/lm_head_int8_block32_argmax_sm120.cu");
        assert!(cu.contains("SPDX-License-Identifier: AGPL-3.0-only"));
        assert!(cu.contains("void lm_head_int8_block32_gemm_sm120("));
        assert!(cu.contains("void lm_head_int8_block32_argmax_stage1_sm120("));
        assert!(cu.contains("void lm_head_int8_block32_argmax_stage2_sm120("));
        assert!(cu.contains("void quantize_bf16_to_int8_block32_sm120("));
        assert!(cu.contains("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32"));
        assert!(cu.contains("ld.global.cs.L2::cache_hint.v4.u32"));
        assert!(cu.contains("ld.global.nc.u32"));
        assert!(cu.contains("ld.global.nc.v4.u32"));
        assert!(cu.contains("cvt.rn.bf16.f32"));
        assert!(cu.contains("sub.bf16x2"));
        assert!(cu.contains("prmt.b32"));
        for line in cu.lines() {
            let code = line.split("//").next().unwrap_or("");
            assert!(
                !code.contains("atomicAdd"),
                "non-deterministic atomicAdd found in CUDA kernel: {line}"
            );
        }
    }
}
