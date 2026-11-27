// SPDX-License-Identifier: AGPL-3.0-only

//! Fused two-stage Blackwell NVFP4 LM-Head + Online Warp/Block Argmax dispatch,
//! fail-closed admission gate, and bit-exact CPU reference mirror.
//!
//! Eliminates the `[M, V]` BF16 logits global-memory write+read slab (`~4 MiB`
//! on greedy `K=8, V=248,077` speculative verify) by computing logits strictly
//! in registers/SMEM (`kernels/gb10/common/lm_head_nvfp4_argmax_sm120.cu`) and
//! writing only compact `[M, num_v_tiles]` partial `(f32 max_val, u32 argmax_id)`
//! reductions (`~62 KiB` at `M=8, V_TILE=256`), followed by a 1-warp-per-row
//! reduction to `[M]` `u32` token IDs.
//!
//! Opt-in only via `ATLAS_LM_HEAD_FUSED_ARGMAX=1` (default OFF). Fails closed
//! to standard full-logits + argmax whenever sampling is non-greedy
//! (`temperature != 0` or non-finite), penalties/grammar/logprobs/softcap are
//! active, LM-head weights are not NVFP4, or shapes exceed `M in 1..=16` /
//! `K % 64 == 0`.

use std::ffi::OsStr;
use std::fmt;

use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use crate::weight_map::QuantizedWeight;
use crate::weight_map::dual_plane::{
    DUAL_PLANE_E2M1_LUT, DUAL_PLANE_LO_SCALE, e4m3_byte_to_f32, float_to_e2m1, float_to_ue4m3,
};

/// Environment variable opting into fused NVFP4 LM-head + online argmax.
///
/// Unset or empty keeps the baseline logits + argmax path (default OFF).
pub const LM_HEAD_FUSED_ARGMAX_ENV: &str = "ATLAS_LM_HEAD_FUSED_ARGMAX";

/// Maximum batch rows `M` supported by `lm_head_nvfp4_argmax_partial_sm120`.
pub const LM_HEAD_ARGMAX_MAX_M: u32 = 16;

/// Vocab columns covered per CTA in `lm_head_nvfp4_argmax_partial_sm120`.
pub const LM_HEAD_ARGMAX_V_TILE: u32 = 256;

/// Required hidden-dimension alignment for Blackwell `m16n8k64` FP4 MMA.
pub const LM_HEAD_ARGMAX_K_ALIGN: u32 = 64;

/// Sentinel vocab index indicating an out-of-bounds or all-NaN slot.
pub const LM_HEAD_ARGMAX_INVALID_ID: u32 = 0xFFFF_FFFF;

/// Error returned when parsing `ATLAS_LM_HEAD_FUSED_ARGMAX`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FusedLmHeadArgmaxConfigError {
    /// Environment value is not valid UTF-8.
    NonUtf8,
    /// Environment value is not a recognized boolean token.
    InvalidToken(String),
}

impl fmt::Display for FusedLmHeadArgmaxConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonUtf8 => {
                write!(f, "{LM_HEAD_FUSED_ARGMAX_ENV} must contain valid UTF-8")
            }
            Self::InvalidToken(raw) => write!(
                f,
                "{LM_HEAD_FUSED_ARGMAX_ENV} must be one of 0/1/false/true/off/on, got {raw:?}"
            ),
        }
    }
}

impl std::error::Error for FusedLmHeadArgmaxConfigError {}

/// Opt-in policy parsed from `ATLAS_LM_HEAD_FUSED_ARGMAX`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum FusedLmHeadArgmaxPolicy {
    /// Disabled (default): baseline logits + argmax path is unchanged.
    #[default]
    Disabled,
    /// Opt-in enabled (still subject to all greedy/shape/weight admission guards).
    Enabled,
}

impl FusedLmHeadArgmaxPolicy {
    /// Parse `ATLAS_LM_HEAD_FUSED_ARGMAX` from the process environment.
    ///
    /// Returns [`Self::Disabled`] when unset or empty.
    ///
    /// # Errors
    /// Fails closed on non-UTF-8 or unrecognized non-empty tokens.
    pub fn from_env() -> std::result::Result<Self, FusedLmHeadArgmaxConfigError> {
        let raw = std::env::var_os(LM_HEAD_FUSED_ARGMAX_ENV);
        Self::parse_os(raw.as_deref())
    }

    /// Parse an optional OS string (used by [`Self::from_env`] and unit tests).
    ///
    /// # Errors
    /// Fails closed on non-UTF-8 or unrecognized non-empty tokens.
    pub fn parse_os(
        raw: Option<&OsStr>,
    ) -> std::result::Result<Self, FusedLmHeadArgmaxConfigError> {
        let Some(os_val) = raw else {
            return Ok(Self::Disabled);
        };
        let utf8 = os_val
            .to_str()
            .ok_or(FusedLmHeadArgmaxConfigError::NonUtf8)?;
        Self::parse_optional(Some(utf8))
    }

    /// Parse an optional UTF-8 string slice.
    ///
    /// # Errors
    /// Fails closed on unrecognized non-empty tokens.
    pub fn parse_optional(
        raw: Option<&str>,
    ) -> std::result::Result<Self, FusedLmHeadArgmaxConfigError> {
        let Some(trimmed) = raw.map(str::trim).filter(|s| !s.is_empty()) else {
            return Ok(Self::Disabled);
        };
        match trimmed.to_ascii_lowercase().as_str() {
            "0" | "false" | "off" => Ok(Self::Disabled),
            "1" | "true" | "on" => Ok(Self::Enabled),
            _ => Err(FusedLmHeadArgmaxConfigError::InvalidToken(
                trimmed.to_owned(),
            )),
        }
    }
}

/// Whether sampling strictly selects greedy argmax (`temperature == 0.0`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LmHeadSamplingMode {
    /// Strict greedy argmax (`temperature == 0.0` finite).
    GreedyArgmax,
    /// Non-greedy or invalid/non-finite temperature (must materialize logits).
    NonGreedy,
}

impl LmHeadSamplingMode {
    /// Classify a request temperature strictly fail-closed: only finite
    /// `temperature == 0.0` (`+0.0` or `-0.0`) qualifies as greedy argmax.
    #[must_use]
    pub fn from_temperature(temperature: f32) -> Self {
        if temperature.is_finite() && temperature == 0.0 {
            Self::GreedyArgmax
        } else {
            Self::NonGreedy
        }
    }
}

/// Whether logit penalties (repetition, frequency, presence, DRY, LZ) are active.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LmHeadPenaltyMode {
    Inactive,
    Active,
}

/// Whether structured-output grammar masking or logit bias is active.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LmHeadGrammarMode {
    Inactive,
    Active,
}

/// Whether full or top-K logprobs were requested by the caller.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LmHeadLogprobsMode {
    Omitted,
    Requested,
}

/// Whether final logit softcapping (e.g. Gemma-4) is enabled.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LmHeadSoftcapMode {
    Disabled,
    Enabled,
}

/// Active target LM-head weight format.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LmHeadWeightFormat {
    Nvfp4,
    Fp8,
    Bf16,
}

/// Whether both fused LM-head argmax CUDA kernels are loaded.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LmHeadKernelAvailability {
    Unavailable,
    Ready,
}

/// All preconditions checked by [`admit_fused_lm_head_argmax`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LmHeadArgmaxAdmissionRequest {
    pub sampling: LmHeadSamplingMode,
    pub penalties: LmHeadPenaltyMode,
    pub grammar: LmHeadGrammarMode,
    pub logprobs: LmHeadLogprobsMode,
    pub softcap: LmHeadSoftcapMode,
    pub weight_format: LmHeadWeightFormat,
    pub kernels: LmHeadKernelAvailability,
    pub rows: u32,
    pub vocab: u32,
    pub hidden: u32,
}

/// Explicit fail-closed reason why fused NVFP4 LM-head argmax was not admitted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FusedLmHeadArgmaxRefusal {
    /// `ATLAS_LM_HEAD_FUSED_ARGMAX` is unset or off (default OFF).
    DisabledByConfig,
    /// Request temperature is non-zero or non-finite.
    NonGreedySampling,
    /// Repetition/frequency/presence/DRY/LZ penalties require materialized logits.
    PenaltiesActive,
    /// Grammar/JSON/tool masking requires materialized logits.
    GrammarActive,
    /// Caller requested logprobs (requires full logit distribution).
    LogprobsRequested,
    /// Final logit softcap is enabled (must run full logit path).
    LogitSoftcapActive,
    /// Target LM-head is FP8 or BF16 rather than NVFP4.
    UnsupportedWeightFormat,
    /// One or both fused CUDA kernels are unavailable on this bundle.
    KernelsUnavailable,
    /// Dimensions violate `rows in 1..=16`, `vocab > 0`, or `hidden % 64 == 0`.
    InvalidShape { rows: u32, vocab: u32, hidden: u32 },
}

impl fmt::Display for FusedLmHeadArgmaxRefusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::DisabledByConfig => {
                write!(
                    f,
                    "fused LM-head argmax disabled ({LM_HEAD_FUSED_ARGMAX_ENV} unset/off)"
                )
            }
            Self::NonGreedySampling => {
                write!(
                    f,
                    "fused LM-head argmax requires strict greedy temperature == 0"
                )
            }
            Self::PenaltiesActive => {
                write!(
                    f,
                    "fused LM-head argmax refused while logit penalties are active"
                )
            }
            Self::GrammarActive => {
                write!(
                    f,
                    "fused LM-head argmax refused while grammar masking is active"
                )
            }
            Self::LogprobsRequested => {
                write!(
                    f,
                    "fused LM-head argmax refused when logprobs are requested"
                )
            }
            Self::LogitSoftcapActive => {
                write!(
                    f,
                    "fused LM-head argmax refused when final logit softcap is enabled"
                )
            }
            Self::UnsupportedWeightFormat => {
                write!(f, "fused LM-head argmax requires NVFP4 LM-head weights")
            }
            Self::KernelsUnavailable => {
                write!(f, "fused LM-head argmax CUDA kernels are not loaded")
            }
            Self::InvalidShape {
                rows,
                vocab,
                hidden,
            } => write!(
                f,
                "fused LM-head argmax invalid shape rows={rows} (1..={LM_HEAD_ARGMAX_MAX_M}), \
                 vocab={vocab} (>0), hidden={hidden} (>0, multiple of {LM_HEAD_ARGMAX_K_ALIGN})"
            ),
        }
    }
}

impl std::error::Error for FusedLmHeadArgmaxRefusal {}

/// Validated two-stage launch geometry and scratch-buffer layout.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FusedLmHeadArgmaxPlan {
    pub rows: u32,
    pub vocab: u32,
    pub hidden: u32,
    pub num_v_tiles: u32,
    /// Byte length of `[rows, num_v_tiles]` FP32 partial max values.
    pub partial_vals_bytes: usize,
    /// 16-byte aligned byte offset of `[rows, num_v_tiles]` U32 partial argmax IDs.
    pub partial_ids_offset: usize,
    /// Byte length of `[rows, num_v_tiles]` U32 partial argmax IDs.
    pub partial_ids_bytes: usize,
    /// Total contiguous device scratch bytes needed for both partial buffers.
    pub total_scratch_bytes: usize,
}

/// Fail-closed admission gate for fused two-stage NVFP4 LM-head + online argmax.
///
/// # Errors
/// Returns a specific [`FusedLmHeadArgmaxRefusal`] whenever any precondition fails.
pub fn admit_fused_lm_head_argmax(
    policy: FusedLmHeadArgmaxPolicy,
    req: &LmHeadArgmaxAdmissionRequest,
) -> std::result::Result<FusedLmHeadArgmaxPlan, FusedLmHeadArgmaxRefusal> {
    if policy == FusedLmHeadArgmaxPolicy::Disabled {
        return Err(FusedLmHeadArgmaxRefusal::DisabledByConfig);
    }
    if req.sampling != LmHeadSamplingMode::GreedyArgmax {
        return Err(FusedLmHeadArgmaxRefusal::NonGreedySampling);
    }
    if req.penalties != LmHeadPenaltyMode::Inactive {
        return Err(FusedLmHeadArgmaxRefusal::PenaltiesActive);
    }
    if req.grammar != LmHeadGrammarMode::Inactive {
        return Err(FusedLmHeadArgmaxRefusal::GrammarActive);
    }
    if req.logprobs != LmHeadLogprobsMode::Omitted {
        return Err(FusedLmHeadArgmaxRefusal::LogprobsRequested);
    }
    if req.softcap != LmHeadSoftcapMode::Disabled {
        return Err(FusedLmHeadArgmaxRefusal::LogitSoftcapActive);
    }
    if req.weight_format != LmHeadWeightFormat::Nvfp4 {
        return Err(FusedLmHeadArgmaxRefusal::UnsupportedWeightFormat);
    }
    if req.kernels != LmHeadKernelAvailability::Ready {
        return Err(FusedLmHeadArgmaxRefusal::KernelsUnavailable);
    }
    if !(1..=LM_HEAD_ARGMAX_MAX_M).contains(&req.rows)
        || req.vocab == 0
        || req.hidden == 0
        || !req.hidden.is_multiple_of(LM_HEAD_ARGMAX_K_ALIGN)
    {
        return Err(FusedLmHeadArgmaxRefusal::InvalidShape {
            rows: req.rows,
            vocab: req.vocab,
            hidden: req.hidden,
        });
    }

    let num_v_tiles = div_ceil(req.vocab, LM_HEAD_ARGMAX_V_TILE);
    let elems = (req.rows as usize)
        .checked_mul(num_v_tiles as usize)
        .ok_or(FusedLmHeadArgmaxRefusal::InvalidShape {
            rows: req.rows,
            vocab: req.vocab,
            hidden: req.hidden,
        })?;
    let partial_vals_bytes =
        elems
            .checked_mul(4)
            .ok_or(FusedLmHeadArgmaxRefusal::InvalidShape {
                rows: req.rows,
                vocab: req.vocab,
                hidden: req.hidden,
            })?;
    let partial_ids_bytes = partial_vals_bytes;
    let partial_ids_offset = partial_vals_bytes.checked_add(15).map(|v| v & !15).ok_or(
        FusedLmHeadArgmaxRefusal::InvalidShape {
            rows: req.rows,
            vocab: req.vocab,
            hidden: req.hidden,
        },
    )?;
    let total_scratch_bytes = partial_ids_offset.checked_add(partial_ids_bytes).ok_or(
        FusedLmHeadArgmaxRefusal::InvalidShape {
            rows: req.rows,
            vocab: req.vocab,
            hidden: req.hidden,
        },
    )?;

    Ok(FusedLmHeadArgmaxPlan {
        rows: req.rows,
        vocab: req.vocab,
        hidden: req.hidden,
        num_v_tiles,
        partial_vals_bytes,
        partial_ids_offset,
        partial_ids_bytes,
        total_scratch_bytes,
    })
}

// ── CUDA Kernel Launchers ───────────────────────────────────────────────────

/// Stage 1: Launch `lm_head_nvfp4_argmax_partial_sm120`.
///
/// Grid: `(ceil(V / 256), 1, 1)`  Block: `(128, 1, 1)`
///
/// # Errors
/// Fails closed before launch on null kernel handle, null device pointers,
/// non-finite/non-positive `weight_scale_2`, or invalid dimensions.
pub fn lm_head_nvfp4_argmax_partial_sm120(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input_bf16: DevicePtr,
    weight: &QuantizedWeight,
    partial_vals: DevicePtr,
    partial_ids: DevicePtr,
    rows: u32,
    vocab: u32,
    hidden: u32,
    stream: u64,
) -> Result<()> {
    ensure!(
        kernel.0 != 0,
        "lm_head_nvfp4_argmax_partial_sm120 requires a non-zero kernel handle"
    );
    ensure!(
        (1..=LM_HEAD_ARGMAX_MAX_M).contains(&rows),
        "lm_head_nvfp4_argmax_partial_sm120: rows={rows} must lie in 1..={LM_HEAD_ARGMAX_MAX_M}"
    );
    ensure!(
        vocab > 0 && hidden > 0 && hidden.is_multiple_of(LM_HEAD_ARGMAX_K_ALIGN),
        "lm_head_nvfp4_argmax_partial_sm120: invalid vocab={vocab}, hidden={hidden} (must be >0, hidden % 64 == 0)"
    );
    ensure!(
        input_bf16 != DevicePtr::NULL,
        "lm_head_nvfp4_argmax_partial_sm120: input_bf16 pointer is NULL"
    );
    ensure!(
        !weight.is_null() && weight.weight_scale != DevicePtr::NULL,
        "lm_head_nvfp4_argmax_partial_sm120: NVFP4 weight or block scale pointer is NULL"
    );
    ensure!(
        weight.weight.0.is_multiple_of(16) && weight.weight_scale.0.is_multiple_of(4),
        "lm_head_nvfp4_argmax_partial_sm120: B_packed must be 16B-aligned and B_scale 4B-aligned"
    );
    ensure!(
        weight.weight_scale_2.is_finite() && weight.weight_scale_2 > 0.0,
        "lm_head_nvfp4_argmax_partial_sm120: weight_scale_2={} must be finite and positive",
        weight.weight_scale_2
    );
    ensure!(
        partial_vals != DevicePtr::NULL && partial_ids != DevicePtr::NULL,
        "lm_head_nvfp4_argmax_partial_sm120: partial output buffer pointer is NULL"
    );

    let num_v_tiles = div_ceil(vocab, LM_HEAD_ARGMAX_V_TILE);
    KernelLaunch::new(gpu, kernel)
        .grid([num_v_tiles, 1, 1])
        .block([128, 1, 1])
        .arg_ptr(input_bf16)
        .arg_ptr(weight.weight)
        .arg_ptr(weight.weight_scale)
        .arg_f32(weight.weight_scale_2)
        .arg_ptr(partial_vals)
        .arg_ptr(partial_ids)
        .arg_u32(rows)
        .arg_u32(vocab)
        .arg_u32(hidden)
        .launch(stream)
}

/// Stage 2: Launch `lm_head_argmax_reduce_sm120`.
///
/// Grid: `(rows, 1, 1)`  Block: `(32, 1, 1)` (1 warp per row)
///
/// # Errors
/// Fails closed before launch on null handle/pointers or zero `rows`/`num_v_tiles`.
pub fn lm_head_argmax_reduce_sm120(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    partial_vals: DevicePtr,
    partial_ids: DevicePtr,
    out_tokens: DevicePtr,
    rows: u32,
    num_v_tiles: u32,
    stream: u64,
) -> Result<()> {
    ensure!(
        kernel.0 != 0,
        "lm_head_argmax_reduce_sm120 requires a non-zero kernel handle"
    );
    ensure!(
        (1..=LM_HEAD_ARGMAX_MAX_M).contains(&rows) && num_v_tiles > 0,
        "lm_head_argmax_reduce_sm120: invalid rows={rows}, num_v_tiles={num_v_tiles}"
    );
    ensure!(
        partial_vals != DevicePtr::NULL
            && partial_ids != DevicePtr::NULL
            && out_tokens != DevicePtr::NULL,
        "lm_head_argmax_reduce_sm120: null device pointer"
    );

    KernelLaunch::new(gpu, kernel)
        .grid([rows, 1, 1])
        .block([32, 1, 1])
        .arg_ptr(partial_vals)
        .arg_ptr(partial_ids)
        .arg_ptr(out_tokens)
        .arg_u32(rows)
        .arg_u32(num_v_tiles)
        .launch(stream)
}

/// Combined two-stage fused NVFP4 LM-head + online argmax dispatch writing
/// `[plan.rows]` `u32` token IDs to `out_tokens` using `scratch` (sized
/// `>= plan.total_scratch_bytes`) for the compact partial reductions.
///
/// # Errors
/// Fails closed before issuing any launch if either handle is null, `scratch`
/// is too small, or pointers/scales are invalid.
pub fn lm_head_nvfp4_fused_argmax(
    gpu: &dyn GpuBackend,
    partial_kernel: KernelHandle,
    reduce_kernel: KernelHandle,
    plan: &FusedLmHeadArgmaxPlan,
    input_bf16: DevicePtr,
    weight: &QuantizedWeight,
    scratch: DevicePtr,
    scratch_bytes: usize,
    out_tokens: DevicePtr,
    stream: u64,
) -> Result<()> {
    ensure!(
        partial_kernel.0 != 0 && reduce_kernel.0 != 0,
        "lm_head_nvfp4_fused_argmax requires both partial and reduce kernel handles"
    );
    ensure!(
        scratch != DevicePtr::NULL && scratch_bytes >= plan.total_scratch_bytes,
        "lm_head_nvfp4_fused_argmax: scratch ({scratch_bytes} B) smaller than required {} B",
        plan.total_scratch_bytes
    );
    ensure!(
        out_tokens != DevicePtr::NULL,
        "lm_head_nvfp4_fused_argmax: out_tokens pointer is NULL"
    );

    let partial_vals = scratch;
    let partial_ids = scratch.offset(plan.partial_ids_offset);
    lm_head_nvfp4_argmax_partial_sm120(
        gpu,
        partial_kernel,
        input_bf16,
        weight,
        partial_vals,
        partial_ids,
        plan.rows,
        plan.vocab,
        plan.hidden,
        stream,
    )?;
    lm_head_argmax_reduce_sm120(
        gpu,
        reduce_kernel,
        partial_vals,
        partial_ids,
        out_tokens,
        plan.rows,
        plan.num_v_tiles,
        stream,
    )
}

// ── Bit-Exact Host Reference Mirror (matches lm_head_nvfp4_argmax_sm120.cu) ─

/// Deterministic tie-breaking comparator matching `lmh_better` in CUDA:
///   - `LM_HEAD_ARGMAX_INVALID_ID` never wins.
///   - `NaN` never wins.
///   - On equal values, the strictly lower vocab index wins.
#[must_use]
pub fn lmh_better_cpu(cand_val: f32, cand_idx: u32, cur_val: f32, cur_idx: u32) -> bool {
    if cand_idx == LM_HEAD_ARGMAX_INVALID_ID {
        return false;
    }
    if cand_val.is_nan() {
        return false;
    }
    if cur_idx == LM_HEAD_ARGMAX_INVALID_ID {
        return true;
    }
    cand_val > cur_val || (cand_val == cur_val && cand_idx < cur_idx)
}

/// Host reference for stage 1 (`lm_head_nvfp4_argmax_partial_sm120`):
/// computes dual-plane E2M1 quantized activations dot NVFP4 weights and returns
/// `(partial_vals, partial_ids)` of shape `[m, num_v_tiles]`.
///
/// # Errors
/// Returns an error on invalid dimensions or mismatched slice extents.
pub fn lm_head_nvfp4_argmax_partial_cpu(
    a_f32: &[f32],
    b_packed: &[u8],
    b_scale: &[u8],
    scale2: f32,
    m: usize,
    v: usize,
    k: usize,
) -> Result<(Vec<f32>, Vec<u32>)> {
    ensure!(
        (1..=LM_HEAD_ARGMAX_MAX_M as usize).contains(&m),
        "m={m} must lie in 1..={LM_HEAD_ARGMAX_MAX_M}"
    );
    ensure!(
        v > 0 && k > 0 && k.is_multiple_of(LM_HEAD_ARGMAX_K_ALIGN as usize),
        "invalid vocab={v}, hidden={k}"
    );
    let a_len = m
        .checked_mul(k)
        .ok_or_else(|| anyhow::anyhow!("activation size overflow"))?;
    let half_k = k / 2;
    let num_groups = k / 16;
    let bp_len = v
        .checked_mul(half_k)
        .ok_or_else(|| anyhow::anyhow!("B_packed size overflow"))?;
    let bs_len = v
        .checked_mul(num_groups)
        .ok_or_else(|| anyhow::anyhow!("B_scale size overflow"))?;
    ensure!(a_f32.len() == a_len, "A len {} != {a_len}", a_f32.len());
    ensure!(
        b_packed.len() == bp_len,
        "B_packed len {} != {bp_len}",
        b_packed.len()
    );
    ensure!(
        b_scale.len() == bs_len,
        "B_scale len {} != {bs_len}",
        b_scale.len()
    );

    // Dual-plane quantize A per 16-element group identically to lmh_quantize_group16.
    let mut a_hi = vec![0_u8; m * half_k];
    let mut a_lo = vec![0_u8; m * half_k];
    let mut a_sc = vec![0_u8; m * num_groups];
    for r in 0..m {
        for g in 0..num_groups {
            let group = &a_f32[r * k + g * 16..r * k + (g + 1) * 16];
            let gmax = group.iter().fold(0.0_f32, |acc, &x| acc.max(x.abs()));
            let sc = float_to_ue4m3(gmax * (1.0 / 6.0));
            a_sc[r * num_groups + g] = sc;
            let eff = e4m3_byte_to_f32(sc);
            let inv_eff = if eff > 0.0 { 1.0 / eff } else { 0.0 };
            for p in 0..8 {
                let n0 = group[p * 2] * inv_eff;
                let n1 = group[p * 2 + 1] * inv_eff;
                let hi0 = float_to_e2m1(n0);
                let hi1 = float_to_e2m1(n1);
                let lo0 = float_to_e2m1(8.0 * (n0 - DUAL_PLANE_E2M1_LUT[hi0 as usize]));
                let lo1 = float_to_e2m1(8.0 * (n1 - DUAL_PLANE_E2M1_LUT[hi1 as usize]));
                a_hi[r * half_k + g * 8 + p] = (hi1 << 4) | (hi0 & 0x0F);
                a_lo[r * half_k + g * 8 + p] = (lo1 << 4) | (lo0 & 0x0F);
            }
        }
    }

    let v_tile = LM_HEAD_ARGMAX_V_TILE as usize;
    let num_v_tiles = v.div_ceil(v_tile);
    let mut partial_vals = vec![f32::NEG_INFINITY; m * num_v_tiles];
    let mut partial_ids = vec![LM_HEAD_ARGMAX_INVALID_ID; m * num_v_tiles];

    for tile in 0..num_v_tiles {
        let v_start = tile * v_tile;
        let v_end = (v_start + v_tile).min(v);
        for r in 0..m {
            let mut best_v = f32::NEG_INFINITY;
            let mut best_i = LM_HEAD_ARGMAX_INVALID_ID;
            for col in v_start..v_end {
                let mut acc_hi = 0.0_f32;
                let mut acc_lo = 0.0_f32;
                for g in 0..num_groups {
                    let sa = e4m3_byte_to_f32(a_sc[r * num_groups + g]);
                    let sb = e4m3_byte_to_f32(b_scale[col * num_groups + g] & 0x7F);
                    let sab = sa * sb;
                    let mut d_hi = 0.0_f32;
                    let mut d_lo = 0.0_f32;
                    for p in 0..8 {
                        let bp = b_packed[col * half_k + g * 8 + p];
                        let ah = a_hi[r * half_k + g * 8 + p];
                        let al = a_lo[r * half_k + g * 8 + p];
                        let wb0 = DUAL_PLANE_E2M1_LUT[(bp & 0x0F) as usize];
                        let wb1 = DUAL_PLANE_E2M1_LUT[(bp >> 4) as usize];
                        d_hi += DUAL_PLANE_E2M1_LUT[(ah & 0x0F) as usize] * wb0
                            + DUAL_PLANE_E2M1_LUT[(ah >> 4) as usize] * wb1;
                        d_lo += DUAL_PLANE_E2M1_LUT[(al & 0x0F) as usize] * wb0
                            + DUAL_PLANE_E2M1_LUT[(al >> 4) as usize] * wb1;
                    }
                    acc_hi += d_hi * sab;
                    acc_lo += d_lo * sab;
                }
                let logit = (acc_hi + DUAL_PLANE_LO_SCALE * acc_lo) * scale2;
                if lmh_better_cpu(logit, col as u32, best_v, best_i) {
                    best_v = logit;
                    best_i = col as u32;
                }
            }
            partial_vals[r * num_v_tiles + tile] = best_v;
            partial_ids[r * num_v_tiles + tile] = best_i;
        }
    }
    Ok((partial_vals, partial_ids))
}

/// Host reference for stage 2 (`lm_head_argmax_reduce_sm120`):
/// reduces `[m, num_v_tiles]` partials to `[m]` `u32` argmax token IDs.
///
/// # Errors
/// Returns an error if dimensions are zero or partial buffer lengths mismatch.
pub fn lm_head_argmax_reduce_cpu(
    partial_vals: &[f32],
    partial_ids: &[u32],
    m: usize,
    num_v_tiles: usize,
) -> Result<Vec<u32>> {
    ensure!(m > 0 && num_v_tiles > 0, "m and num_v_tiles must be > 0");
    let expected = m
        .checked_mul(num_v_tiles)
        .ok_or_else(|| anyhow::anyhow!("partial extent overflow"))?;
    ensure!(
        partial_vals.len() == expected && partial_ids.len() == expected,
        "partial slice lengths must equal m * num_v_tiles ({expected})"
    );
    let mut out = Vec::with_capacity(m);
    for r in 0..m {
        let mut best_v = f32::NEG_INFINITY;
        let mut best_i = LM_HEAD_ARGMAX_INVALID_ID;
        for t in 0..num_v_tiles {
            let cv = partial_vals[r * num_v_tiles + t];
            let ci = partial_ids[r * num_v_tiles + t];
            if lmh_better_cpu(cv, ci, best_v, best_i) {
                best_v = cv;
                best_i = ci;
            }
        }
        out.push(if best_i == LM_HEAD_ARGMAX_INVALID_ID {
            0
        } else {
            best_i
        });
    }
    Ok(out)
}

/// End-to-end host reference for fused two-stage NVFP4 LM-head argmax.
///
/// # Errors
/// Propagates validation errors from stage 1 or stage 2.
pub fn lm_head_nvfp4_fused_argmax_cpu(
    a_f32: &[f32],
    b_packed: &[u8],
    b_scale: &[u8],
    scale2: f32,
    m: usize,
    v: usize,
    k: usize,
) -> Result<Vec<u32>> {
    let num_v_tiles = v.div_ceil(LM_HEAD_ARGMAX_V_TILE as usize);
    let (pv, pi) = lm_head_nvfp4_argmax_partial_cpu(a_f32, b_packed, b_scale, scale2, m, v, k)?;
    lm_head_argmax_reduce_cpu(&pv, &pi, m, num_v_tiles)
}

#[cfg(test)]
mod tests {
    use spark_runtime::gpu::mock::MockGpuBackend;

    use super::*;
    use crate::weight_map::dual_plane::quantize_f32_to_dual_plane_host;

    fn deterministic_vals(len: usize, seed: u64) -> Vec<f32> {
        let mut state = seed;
        (0..len)
            .map(|_| {
                state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
                let mut z = state;
                z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
                z ^= z >> 31;
                let u = ((z >> 11) as f64) * (1.0 / ((1_u64 << 53) as f64));
                ((u - 0.5) * 2.0) as f32
            })
            .collect()
    }

    #[test]
    fn tie_breaking_picks_strictly_lowest_vocab_index_and_ignores_nan() {
        // Equal scores across tiles: tile 0 id=17 vs tile 1 id=300 vs tile 3 id=17 (or id=5).
        let vals = vec![
            1.5,
            1.5,
            1.5,
            1.5, // row 0: all equal -> lowest id wins
            f32::NAN,
            -2.0,
            f32::NAN,
            -2.0, // row 1: NaN ignored, tie on -2.0 -> lowest id
            f32::NEG_INFINITY,
            f32::NEG_INFINITY,
            f32::NEG_INFINITY,
            f32::NEG_INFINITY, // row 2: -inf tie
            f32::NAN,
            f32::NAN,
            f32::NAN,
            f32::NAN, // row 3: all NaN -> safe 0 fallback
        ];
        let ids = vec![
            512,
            17,
            1024,
            99, // row 0 -> 17
            0,
            400,
            1,
            260, // row 1 -> 260
            900,
            12,
            300,
            45, // row 2 -> 12
            LM_HEAD_ARGMAX_INVALID_ID,
            LM_HEAD_ARGMAX_INVALID_ID,
            LM_HEAD_ARGMAX_INVALID_ID,
            LM_HEAD_ARGMAX_INVALID_ID,
        ];
        let reduced = lm_head_argmax_reduce_cpu(&vals, &ids, 4, 4).unwrap();
        assert_eq!(reduced, vec![17, 260, 12, 0]);

        // Also test exact tie inside NVFP4 weight matrix: rows 5 and 270 identical.
        let (m, v, k) = (2, 300, 64);
        let a = vec![1.0_f32; m * k];
        let mut w = vec![0.0_f32; v * k];
        for c in 0..k {
            w[5 * k + c] = 2.0;
            w[270 * k + c] = 2.0;
            w[299 * k + c] = 1.0;
        }
        let q = quantize_f32_to_dual_plane_host(&w, v, k).unwrap();
        let out =
            lm_head_nvfp4_fused_argmax_cpu(&a, &q.w_hi, &q.scales, q.scale2, m, v, k).unwrap();
        assert_eq!(
            out,
            vec![5, 5],
            "equal logits across tiles must select lowest vocab index 5"
        );
    }

    #[test]
    fn non_multiple_of_128_tail_v248077_plan_and_v269_masking() {
        // Exact production Qwen3.8 vocab V=248,077 (neither %128 nor %256 == 0).
        let req = LmHeadArgmaxAdmissionRequest {
            sampling: LmHeadSamplingMode::from_temperature(0.0),
            penalties: LmHeadPenaltyMode::Inactive,
            grammar: LmHeadGrammarMode::Inactive,
            logprobs: LmHeadLogprobsMode::Omitted,
            softcap: LmHeadSoftcapMode::Disabled,
            weight_format: LmHeadWeightFormat::Nvfp4,
            kernels: LmHeadKernelAvailability::Ready,
            rows: 8,
            vocab: 248_077,
            hidden: 5120,
        };
        let plan = admit_fused_lm_head_argmax(FusedLmHeadArgmaxPolicy::Enabled, &req).unwrap();
        assert_eq!(plan.num_v_tiles, 970);
        assert_eq!(plan.partial_vals_bytes, 8 * 970 * 4);
        assert_eq!(plan.partial_ids_offset % 16, 0);
        // Compact partial footprint (~62 KiB) vs full BF16 logits slab (~3.97 MiB).
        assert!(plan.total_scratch_bytes < 64 * 1024);

        // V = 269 = 1 full 256-tile + 13 tail vocab entries (same tail length as 248,077 % 256 == 13).
        assert_eq!(248_077 % LM_HEAD_ARGMAX_V_TILE, 13);
        let (m, v, k) = (3, 269, 64);
        let a = vec![1.0_f32; m * k];
        let mut w = deterministic_vals(v * k, 0xCAFE_1234);
        // Make the final valid tail vocab entry (id=268) dominate for row 0,
        // and id=256 (first entry of tail tile) dominate for row 1,
        // and id=0 dominate for row 2.
        for c in 0..k {
            w[268 * k + c] = 5.0;
        }
        let mut a_rows = a;
        for c in 0..k {
            // Row 1 strongly prefers col 256, row 2 strongly prefers col 0.
            w[256 * k + c] = if c % 2 == 0 { 5.0 } else { -5.0 };
            a_rows[k + c] = if c % 2 == 0 { 1.0 } else { -1.0 };
            w[c] = if c < 32 { 5.0 } else { -5.0 };
            a_rows[2 * k + c] = if c < 32 { 1.0 } else { -1.0 };
        }
        let q = quantize_f32_to_dual_plane_host(&w, v, k).unwrap();
        let out =
            lm_head_nvfp4_fused_argmax_cpu(&a_rows, &q.w_hi, &q.scales, q.scale2, m, v, k).unwrap();
        assert_eq!(out, vec![268, 256, 0]);
        assert!(out.iter().all(|&tok| (tok as usize) < v));
    }

    #[test]
    fn multi_row_m1_through_m16_matches_naive_fp32_reference() {
        let (v, k) = (531, 128); // spans 3 V_TILE=256 tiles with 19-col tail
        let w = deterministic_vals(v * k, 0xBEEF_0042);
        let q = quantize_f32_to_dual_plane_host(&w, v, k).unwrap();
        let w_deq = crate::weight_map::dual_plane::dequantize_dual_plane_hi_host(&q);

        for m in [1_usize, 2, 4, 8, 16] {
            let mut a = deterministic_vals(m * k, 0xABCD_0000 + m as u64);
            // Plant a distinct clear winner per row across different tiles so
            // quantization noise never creates accidental near-ties.
            let mut expected = Vec::with_capacity(m);
            for r in 0..m {
                let target_v = (r * 37 + 11) % v;
                for c in 0..k {
                    a[r * k + c] = w_deq[target_v * k + c];
                }
                // Compute naive full FP32 matmul against exact NVFP4 dequant weights.
                let mut best_val = f32::NEG_INFINITY;
                let mut best_id = 0_u32;
                for col in 0..v {
                    let mut dot = 0.0_f32;
                    for c in 0..k {
                        dot += a[r * k + c] * w_deq[col * k + c];
                    }
                    if dot > best_val {
                        best_val = dot;
                        best_id = col as u32;
                    }
                }
                expected.push(best_id);
            }

            let actual =
                lm_head_nvfp4_fused_argmax_cpu(&a, &q.w_hi, &q.scales, q.scale2, m, v, k).unwrap();
            assert_eq!(actual, expected, "mismatch at M={m}");
        }
    }

    #[test]
    fn admission_gate_defaults_off_and_fails_closed_on_every_disqualifier() {
        assert_eq!(
            FusedLmHeadArgmaxPolicy::parse_optional(None).unwrap(),
            FusedLmHeadArgmaxPolicy::Disabled
        );
        assert_eq!(
            FusedLmHeadArgmaxPolicy::parse_optional(Some("   ")).unwrap(),
            FusedLmHeadArgmaxPolicy::Disabled
        );
        assert_eq!(
            FusedLmHeadArgmaxPolicy::parse_optional(Some("0")).unwrap(),
            FusedLmHeadArgmaxPolicy::Disabled
        );
        assert_eq!(
            FusedLmHeadArgmaxPolicy::parse_optional(Some("1")).unwrap(),
            FusedLmHeadArgmaxPolicy::Enabled
        );
        assert!(FusedLmHeadArgmaxPolicy::parse_optional(Some("yes")).is_err());

        #[cfg(unix)]
        {
            use std::os::unix::ffi::OsStrExt;
            let bad = OsStr::from_bytes(&[0xFF, 0xFE]);
            assert_eq!(
                FusedLmHeadArgmaxPolicy::parse_os(Some(bad)),
                Err(FusedLmHeadArgmaxConfigError::NonUtf8)
            );
        }

        let base = LmHeadArgmaxAdmissionRequest {
            sampling: LmHeadSamplingMode::from_temperature(0.0),
            penalties: LmHeadPenaltyMode::Inactive,
            grammar: LmHeadGrammarMode::Inactive,
            logprobs: LmHeadLogprobsMode::Omitted,
            softcap: LmHeadSoftcapMode::Disabled,
            weight_format: LmHeadWeightFormat::Nvfp4,
            kernels: LmHeadKernelAvailability::Ready,
            rows: 8,
            vocab: 248_077,
            hidden: 2560,
        };

        assert_eq!(
            admit_fused_lm_head_argmax(FusedLmHeadArgmaxPolicy::Disabled, &base).unwrap_err(),
            FusedLmHeadArgmaxRefusal::DisabledByConfig
        );

        for bad_temp in [
            0.7_f32,
            1e-6,
            -0.1,
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
        ] {
            let mut r = base;
            r.sampling = LmHeadSamplingMode::from_temperature(bad_temp);
            assert_eq!(
                admit_fused_lm_head_argmax(FusedLmHeadArgmaxPolicy::Enabled, &r).unwrap_err(),
                FusedLmHeadArgmaxRefusal::NonGreedySampling,
                "temperature={bad_temp} must fail closed"
            );
        }

        let mut r = base;
        r.penalties = LmHeadPenaltyMode::Active;
        assert_eq!(
            admit_fused_lm_head_argmax(FusedLmHeadArgmaxPolicy::Enabled, &r).unwrap_err(),
            FusedLmHeadArgmaxRefusal::PenaltiesActive
        );

        let mut r = base;
        r.grammar = LmHeadGrammarMode::Active;
        assert_eq!(
            admit_fused_lm_head_argmax(FusedLmHeadArgmaxPolicy::Enabled, &r).unwrap_err(),
            FusedLmHeadArgmaxRefusal::GrammarActive
        );

        let mut r = base;
        r.logprobs = LmHeadLogprobsMode::Requested;
        assert_eq!(
            admit_fused_lm_head_argmax(FusedLmHeadArgmaxPolicy::Enabled, &r).unwrap_err(),
            FusedLmHeadArgmaxRefusal::LogprobsRequested
        );

        let mut r = base;
        r.softcap = LmHeadSoftcapMode::Enabled;
        assert_eq!(
            admit_fused_lm_head_argmax(FusedLmHeadArgmaxPolicy::Enabled, &r).unwrap_err(),
            FusedLmHeadArgmaxRefusal::LogitSoftcapActive
        );

        for fmt in [LmHeadWeightFormat::Fp8, LmHeadWeightFormat::Bf16] {
            let mut r = base;
            r.weight_format = fmt;
            assert_eq!(
                admit_fused_lm_head_argmax(FusedLmHeadArgmaxPolicy::Enabled, &r).unwrap_err(),
                FusedLmHeadArgmaxRefusal::UnsupportedWeightFormat
            );
        }

        let mut r = base;
        r.kernels = LmHeadKernelAvailability::Unavailable;
        assert_eq!(
            admit_fused_lm_head_argmax(FusedLmHeadArgmaxPolicy::Enabled, &r).unwrap_err(),
            FusedLmHeadArgmaxRefusal::KernelsUnavailable
        );

        for (rows, vocab, hidden) in [(0, 100, 64), (17, 100, 64), (4, 0, 64), (4, 100, 48)] {
            let mut r = base;
            r.rows = rows;
            r.vocab = vocab;
            r.hidden = hidden;
            assert!(matches!(
                admit_fused_lm_head_argmax(FusedLmHeadArgmaxPolicy::Enabled, &r).unwrap_err(),
                FusedLmHeadArgmaxRefusal::InvalidShape { .. }
            ));
        }
    }

    #[test]
    fn mock_gpu_launcher_validates_bounds_and_launches_two_stages() {
        let gpu = MockGpuBackend::new();
        let req = LmHeadArgmaxAdmissionRequest {
            sampling: LmHeadSamplingMode::GreedyArgmax,
            penalties: LmHeadPenaltyMode::Inactive,
            grammar: LmHeadGrammarMode::Inactive,
            logprobs: LmHeadLogprobsMode::Omitted,
            softcap: LmHeadSoftcapMode::Disabled,
            weight_format: LmHeadWeightFormat::Nvfp4,
            kernels: LmHeadKernelAvailability::Ready,
            rows: 8,
            vocab: 248_077,
            hidden: 2560,
        };
        let plan = admit_fused_lm_head_argmax(FusedLmHeadArgmaxPolicy::Enabled, &req).unwrap();
        let weight = QuantizedWeight {
            weight: DevicePtr(0x10_0000),
            weight_scale: DevicePtr(0x20_0000),
            weight_scale_2: 0.25,
            input_scale: DevicePtr::NULL,
        };

        // Null handle or undersized scratch must fail before launching anything.
        assert!(
            lm_head_nvfp4_fused_argmax(
                &gpu,
                KernelHandle(0),
                KernelHandle(2),
                &plan,
                DevicePtr(0x30_0000),
                &weight,
                DevicePtr(0x40_0000),
                plan.total_scratch_bytes,
                DevicePtr(0x50_0000),
                0,
            )
            .is_err()
        );
        assert!(
            lm_head_nvfp4_fused_argmax(
                &gpu,
                KernelHandle(1),
                KernelHandle(2),
                &plan,
                DevicePtr(0x30_0000),
                &weight,
                DevicePtr(0x40_0000),
                plan.total_scratch_bytes - 1,
                DevicePtr(0x50_0000),
                0,
            )
            .is_err()
        );
        assert_eq!(gpu.launch_count(), 0);

        // Valid call dispatches exactly 2 kernels (stage 1 partial + stage 2 reduce).
        lm_head_nvfp4_fused_argmax(
            &gpu,
            KernelHandle(1),
            KernelHandle(2),
            &plan,
            DevicePtr(0x30_0000),
            &weight,
            DevicePtr(0x40_0000),
            plan.total_scratch_bytes,
            DevicePtr(0x50_0000),
            7,
        )
        .unwrap();
        assert_eq!(gpu.launch_count(), 2);
    }
}
