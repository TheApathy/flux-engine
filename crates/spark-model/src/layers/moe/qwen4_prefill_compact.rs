// SPDX-License-Identifier: AGPL-3.0-only

//! Default-off F8-only original-layout BF16-MMA compact scheduling.

#[path = "qwen4_moe_w4a4.rs"]
pub(crate) mod qwen4_moe_w4a4;

use anyhow::{Context, Result, bail, ensure};
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::{GpuBackend, KernelHandle};

pub(crate) use qwen4_moe_w4a4::{
    W4A4_SELECTOR, W4a4ScratchLayout, W4a4V32Kernels, load_w4a4_v32_kernels, w4a4_mode,
};

use super::qwen4_compact_contract as abi;
use super::qwen4_plan_order_rule::{
    PLAN_ORDER_MAX_ROWS_SELECTOR, PLAN_ORDER_SELECTOR, PLAN_ORDER_TRIGGER_SELECTOR, PLAN_V8_KERNEL,
    PLAN_V8_MODULE, PlanOrder, parse_plan_order,
};
use super::qwen4_xq_quant::xq_quant_kernel;
use crate::layer::ForwardContext;

pub(crate) const SELECTOR: &str = "ATLAS_QWEN4_PREFILL_MOE_COMPACT";
pub(crate) const CHECK_SELECTOR: &str = "ATLAS_QWEN4_PREFILL_MOE_COMPACT_CHECK";
pub(crate) const K32_SELECTOR: &str = "ATLAS_QWEN4_PREFILL_MOE_COMPACT_K32";
pub(crate) const TRANSPOSED_SELECTOR: &str = "ATLAS_QWEN4_PREFILL_MOE_COMPACT_T";
pub(crate) const STREAM_SELECTOR: &str = "ATLAS_QWEN4_PREFILL_MOE_STREAM_T";
pub(crate) const V2_SELECTOR: &str = "ATLAS_QWEN4_PREFILL_MOE_COMPACT_V2";
/// Default-off: routed experts through the V10 W4A8 kernels (b12x w4a8 numerics: MXFP8
/// activations, NVFP4 weights rounded to E4M3 payloads under UE8M0/32; QMMA.SF). Needs a V2 path.
pub(crate) const W4A8_SELECTOR: &str = "ATLAS_QWEN4_PREFILL_MOE_W4A8";
/// Default-off with W4A8: the V12 kernels (V10 with two k32 steps per barrier; bit-identical to V10).
pub(crate) const W4A8_K64_SELECTOR: &str = "ATLAS_QWEN4_PREFILL_MOE_W4A8_K64";
/// Default-off with W4A8 (takes precedence over K64): the V14 kernels (V12 on a 128 x 256 tile,
/// 512-thread CTAs, 89 KB dynamic shared memory; halves the activation re-reads; bit-identical to V12).
pub(crate) const W4A8_WIDE_SELECTOR: &str = "ATLAS_QWEN4_PREFILL_MOE_W4A8_WIDE";
/// V14 dynamic shared memory per CTA (both kernels; the kernels fail closed on less).
const V14_SMEM_BYTES: u32 = 89_088;
/// V16 (ATLAS_QWEN4_PREFILL_MOE_W4A8_WIDE=2): V14's layout (same dynamic shared memory).
const V16_SMEM_BYTES: u32 = 89_088;
/// V28 (ATLAS_QWEN4_PREFILL_MOE_W4A8_WIDE=7): V16 with packed B-column scales (one LDS.64 per k32 step
/// instead of eight LDS.U8, QMMA B scale byte selector); bit-identical rows, V16's launch and smem.
const V28_SMEM_BYTES: u32 = V16_SMEM_BYTES;

/// Small-batch threshold for MoE low-latency routing and adaptive grid sizing.
pub const SMALL_BATCH_THRESHOLD: usize = super::SMALL_BATCH_MOE_THRESHOLD;
/// Legacy GB10 SM121 baseline small-batch grid (48 SMs). Runtime callers use
/// [`get_optimal_grid_dim`], which scales dynamically with `active_sm_count()`.
pub const SMALL_BATCH_GRID: u32 = atlas_core::device::sm121::NUM_SMS;
/// Full persistent worklist grid for large prefill chunks.
pub const DEFAULT_PREFILL_GRID: u32 = 4096;

#[inline]
#[must_use]
const fn get_optimal_grid_dim_for_sm(rows: usize, sm_count: u32) -> u32 {
    if rows <= SMALL_BATCH_THRESHOLD {
        let sm = if sm_count == 0 { 1 } else { sm_count };
        if sm > DEFAULT_PREFILL_GRID {
            DEFAULT_PREFILL_GRID
        } else {
            sm
        }
    } else {
        DEFAULT_PREFILL_GRID
    }
}

#[inline]
#[must_use]
pub fn get_optimal_grid_dim(rows: usize) -> u32 {
    get_optimal_grid_dim_for_sm(rows, atlas_core::device::active_sm_count())
}

/// Default-off with `W4A8_WIDE=2` or `7`: the routed down GEMM writes its slot rows as MXFP8
/// (`1` = E4M3 payload + UE8M0 per 32 columns; `2` = `Mxfp8R2` adding FP32 residual corrections)
/// and the unpermute reads them back, halving slot-row traffic. A quantisation site (gated), not bit-identical.
pub(crate) const SLOT_MXFP8_SELECTOR: &str = "ATLAS_QWEN4_PREFILL_MOE_SLOT_MXFP8";

const SLOT_PAYLOAD_BYTES_PER_ROW: usize = 2560;
const SLOT_SCALE_BYTES_PER_ROW: usize = SLOT_PAYLOAD_BYTES_PER_ROW / 32;
const SLOT_R2_BYTES_PER_ROW: usize = SLOT_SCALE_BYTES_PER_ROW * std::mem::size_of::<u32>();

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) enum SlotStorage {
    #[default]
    Bf16,
    Mxfp8,
    Mxfp8R2,
}

impl SlotStorage {
    pub(super) const fn is_mxfp8(self) -> bool {
        !matches!(self, Self::Bf16)
    }

    pub(super) const fn has_corrections(self) -> bool {
        matches!(self, Self::Mxfp8R2)
    }
}

pub(super) const fn dispatch_slot_storage(
    configured: SlotStorage,
    r2_prefill_active: bool,
) -> SlotStorage {
    if matches!(configured, SlotStorage::Mxfp8R2) && !r2_prefill_active {
        SlotStorage::Bf16
    } else {
        configured
    }
}

const fn request_active_from(loaded: bool, family_suppressed: bool) -> bool {
    loaded && !family_suppressed
}

/// Whether a load-time compact bundle is active for this request.  Admission
/// may suppress the complete fast-prefill family without unloading its
/// kernels; every request-time dispatch must honor that scoped decision.
pub(super) fn request_active(loaded: bool) -> bool {
    request_active_from(
        loaded,
        crate::layers::qwen4_prefill_moe::family_suppressed(),
    )
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct SlotLayout {
    pub scales_offset: Option<usize>,
    pub corrections_offset: Option<usize>,
    pub used_bytes: usize,
}

fn parse_slot_storage(value: Option<&str>) -> std::result::Result<SlotStorage, &'static str> {
    match value {
        None | Some("0") => Ok(SlotStorage::Bf16),
        Some("1") => Ok(SlotStorage::Mxfp8),
        Some("2") => Ok(SlotStorage::Mxfp8R2),
        _ => Err("must be exactly 0, 1 or 2"),
    }
}

pub(super) fn slot_storage_selected() -> Result<SlotStorage> {
    static SEL: std::sync::OnceLock<std::result::Result<SlotStorage, String>> =
        std::sync::OnceLock::new();
    SEL.get_or_init(|| {
        let value = match std::env::var(SLOT_MXFP8_SELECTOR) {
            Ok(value) => Some(value),
            Err(std::env::VarError::NotPresent) => None,
            Err(error) => return Err(format!("invalid {SLOT_MXFP8_SELECTOR}: {error}")),
        };
        parse_slot_storage(value.as_deref())
            .map_err(|error| format!("{SLOT_MXFP8_SELECTOR}: {error}"))
    })
    .clone()
    .map_err(anyhow::Error::msg)
}

pub(crate) fn slot_mxfp8_r2_selected() -> Result<bool> {
    Ok(slot_storage_selected()? == SlotStorage::Mxfp8R2)
}

pub(super) fn slot_layout(pairs: usize, storage: SlotStorage) -> Result<SlotLayout> {
    let payload_bytes = pairs
        .checked_mul(SLOT_PAYLOAD_BYTES_PER_ROW)
        .context("MXFP8 slot payload size overflow")?;
    if !storage.is_mxfp8() {
        return Ok(SlotLayout {
            scales_offset: None,
            corrections_offset: None,
            used_bytes: pairs
                .checked_mul(SLOT_PAYLOAD_BYTES_PER_ROW * 2)
                .context("BF16 slot size overflow")?,
        });
    }
    let scale_bytes = pairs
        .checked_mul(SLOT_SCALE_BYTES_PER_ROW)
        .context("MXFP8 slot scale size overflow")?;
    let corrections_offset = payload_bytes
        .checked_add(scale_bytes)
        .context("MXFP8 slot correction offset overflow")?;
    let correction_bytes = if storage.has_corrections() {
        pairs
            .checked_mul(SLOT_R2_BYTES_PER_ROW)
            .context("MXFP8 slot correction size overflow")?
    } else {
        0
    };
    Ok(SlotLayout {
        scales_offset: Some(payload_bytes),
        corrections_offset: storage.has_corrections().then_some(corrections_offset),
        used_bytes: corrections_offset
            .checked_add(correction_bytes)
            .context("MXFP8 slot layout size overflow")?,
    })
}

/// Default-off with SHARED_W4A8 + W4A8: the routed dispatch reuses the MXFP8 x the shared expert
/// just wrote into expert_up_out (same input, same rows, same quant kernel) instead of
/// re-quantising it. Bit-identical.
pub(crate) const XQ_REUSE_SELECTOR: &str = "ATLAS_QWEN4_PREFILL_MOE_XQ_REUSE";

/// Default-off with W4A8 + WIDE=2 (V16) or 7 (V28): the quad-lane coalesced bf16 -> MXFP8 x
/// quantiser (`qwen4_mxfp8_quant_rows_v16q4`: four lanes per MX32 block, one 16-byte load and one
/// 8-byte store per lane, quad-shuffle absmax) replaces `qwen4_mxfp8_quant_rows_v16` (V28: the same
/// code under the `_v28` tag; the V28 module includes the V16 source, twin included) for the
/// shared and routed x quant. Bit-identical: fmaxf over the same 32 magnitudes, the same frexpf/pow2ceil
/// scale rule, the same e4m3 RN-satfinite per value. The MoE input is bf16 (`norm_output`) under
/// every GDN norm mode, so this is the only bf16 -> MXFP8 pass of the slot/XQ path; the slot rows
/// themselves are quantised inside the v17q down epilogue. Off, every W4A8 set loads its own
/// `qwen4_mxfp8_quant_rows_<tag>` (`super::qwen4_xq_quant::xq_quant_kernel`).
pub(crate) const SLOT_BF16_FUSED_SELECTOR: &str = "ATLAS_QWEN4_PREFILL_MOE_SLOT_BF16_FUSED";

pub(crate) fn slot_bf16_fused_selected() -> Result<bool> {
    static SEL: std::sync::OnceLock<std::result::Result<bool, String>> = std::sync::OnceLock::new();
    SEL.get_or_init(|| flag(SLOT_BF16_FUSED_SELECTOR).map_err(|e| e.to_string()))
        .clone()
        .map_err(anyhow::Error::msg)
}

pub(crate) fn xq_reuse_selected() -> Result<bool> {
    static SEL: std::sync::OnceLock<Result<bool, String>> = std::sync::OnceLock::new();
    SEL.get_or_init(|| flag(XQ_REUSE_SELECTOR).map_err(|e| e.to_string()))
        .clone()
        .map_err(anyhow::Error::msg)
}

/// ATLAS_QWEN4_PREFILL_MOE_W4A8_WIDE: absent/0 off, 1 = V14, 2 = V16, 7 = V28.
fn w4a8_wide_level() -> Result<u32> {
    static LEVEL: std::sync::OnceLock<std::result::Result<u32, String>> =
        std::sync::OnceLock::new();
    LEVEL
        .get_or_init(|| match std::env::var(W4A8_WIDE_SELECTOR) {
            Err(std::env::VarError::NotPresent) => Ok(0),
            Ok(v) if v == "0" => Ok(0),
            Ok(v) if v == "1" => Ok(1),
            Ok(v) if v == "2" => Ok(2),
            Ok(v) if v == "7" => Ok(7),
            Ok(v) => Err(format!(
                "{W4A8_WIDE_SELECTOR} must be absent, 0, 1, 2 or 7; got {v:?}"
            )),
            Err(e) => Err(format!("invalid {W4A8_WIDE_SELECTOR}: {e}")),
        })
        .clone()
        .map_err(anyhow::Error::msg)
}

/// Default-off with W4A8 + WIDE=7 (V28): selects the V28q routed MoE kernels
/// (512-thread prologue transcode, tid>=256 in-loop transcode warp-half swap, and quad-lane
/// coalesced MXFP8 epilogue for both GATEUP and DOWN MXFP8 slot output):
///   0 / absent = off (v28 gateup + v17q down)
///   1 = v28q4 gateup + v28q down (or v28_m64 + v28q_m64 when V28_M64=1)
///   2 = v28_twarp gateup + v28q_twarp down (or v28_m64 + v28q_m64 when V28_M64=1)
///   3 = v28q4 gateup + v17q down (gateup-only ablation)
///   4 = v28 gateup + v28q down (down-only ablation)
pub(crate) const V28Q_SELECTOR: &str = "ATLAS_QWEN4_PREFILL_MOE_V28Q";

fn parse_v28q_level(value: Option<&str>) -> std::result::Result<u32, &'static str> {
    match value {
        None | Some("0") => Ok(0),
        Some("1") => Ok(1),
        Some("2") => Ok(2),
        Some("3") => Ok(3),
        Some("4") => Ok(4),
        _ => Err("must be absent, 0, 1, 2, 3 or 4"),
    }
}

pub(crate) fn v28q_level() -> Result<u32> {
    static LEVEL: std::sync::OnceLock<std::result::Result<u32, String>> =
        std::sync::OnceLock::new();
    LEVEL
        .get_or_init(|| {
            let value = match std::env::var(V28Q_SELECTOR) {
                Ok(value) => Some(value),
                Err(std::env::VarError::NotPresent) => None,
                Err(error) => return Err(format!("invalid {V28Q_SELECTOR}: {error}")),
            };
            parse_v28q_level(value.as_deref()).map_err(|error| format!("{V28Q_SELECTOR}: {error}"))
        })
        .clone()
        .map_err(anyhow::Error::msg)
}

/// Default-off with W4A8 + WIDE=7 (V28): the M64 N-split V28 kernels (`_v28_m64` / `_v28q_m64`),
/// remapping the 16 warps to 2x8 across N when `valid_rows <= 64`. Bit-identical to V28.
pub(crate) const V28_M64_SELECTOR: &str = "ATLAS_QWEN4_PREFILL_MOE_V28_M64";
pub(crate) const MOE_SMALL_TILE_SELECTOR: &str = "ATLAS_QWEN4_MOE_SMALL_TILE";

fn parse_v28_m64(v1: Option<&str>, v2: Option<&str>) -> std::result::Result<bool, &'static str> {
    if let Some(v) = v1 {
        abi::parse(Some(v))
    } else if let Some(v) = v2 {
        abi::parse(Some(v))
    } else {
        Ok(false)
    }
}

pub(crate) fn v28_m64_selected() -> Result<bool> {
    static SEL: std::sync::OnceLock<std::result::Result<bool, String>> = std::sync::OnceLock::new();
    SEL.get_or_init(|| {
        let v1 = match std::env::var(V28_M64_SELECTOR) {
            Ok(v) => Some(v),
            Err(std::env::VarError::NotPresent) => None,
            Err(e) => return Err(format!("invalid {V28_M64_SELECTOR}: {e}")),
        };
        let v2 = match std::env::var(MOE_SMALL_TILE_SELECTOR) {
            Ok(v) => Some(v),
            Err(std::env::VarError::NotPresent) => None,
            Err(e) => return Err(format!("invalid {MOE_SMALL_TILE_SELECTOR}: {e}")),
        };
        let active_name = if v1.is_some() {
            V28_M64_SELECTOR
        } else {
            MOE_SMALL_TILE_SELECTOR
        };
        parse_v28_m64(v1.as_deref(), v2.as_deref())
            .map_err(|error| format!("{active_name}: {error}"))
    })
    .clone()
    .map_err(anyhow::Error::msg)
}

/// `0-ULP` `TILE_M=256 x BC=128` MoE prefill selector (`ATLAS_QWEN4_PREFILL_MOE_V29_M256`):
///   unset / `0` = disabled (keep `v28_twarp`);
///   `auto`      = auto-engage on V28 when `rows >= 2048`, `!has_corrections`,
///                 `l2_windows.is_none()`, `!v28_m64`, and `v29_m256` kernels are loaded;
///   `1`         = force `v29_m256` across all prefill row counts (fails closed if unavailable).
pub(crate) const V29_M256_SELECTOR: &str = "ATLAS_QWEN4_PREFILL_MOE_V29_M256";
/// Minimum prompt rows for automatic `v29_m256` (`M=256`) prefill dispatch (`rows >= 2048`).
pub(super) const V29_M256_AUTO_MIN_ROWS: usize = 2048;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum V29M256Policy {
    Disabled,
    Auto,
    Force,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct V29M256Eligibility {
    pub rows: usize,
    pub w4a8_is_v28: bool,
    pub v28_m64_forced: bool,
    pub has_corrections: bool,
    pub l2_windowed: bool,
    pub kernels_loaded: bool,
}

impl V29M256Policy {
    #[must_use]
    pub(super) const fn admits(self, eligibility: V29M256Eligibility) -> bool {
        let common = eligibility.w4a8_is_v28
            && !eligibility.v28_m64_forced
            && !eligibility.has_corrections
            && !eligibility.l2_windowed
            && eligibility.kernels_loaded;
        match self {
            Self::Disabled => false,
            Self::Auto => common && eligibility.rows >= V29_M256_AUTO_MIN_ROWS,
            Self::Force => common,
        }
    }
}

fn parse_v29_m256_policy(value: Option<&str>) -> std::result::Result<V29M256Policy, &'static str> {
    match value {
        None | Some("0") => Ok(V29M256Policy::Disabled),
        Some("auto") => Ok(V29M256Policy::Auto),
        Some("1") => Ok(V29M256Policy::Force),
        _ => Err("must be unset, 0, 1, or auto"),
    }
}

pub(super) fn v29_m256_policy() -> Result<V29M256Policy> {
    static POLICY: std::sync::OnceLock<std::result::Result<V29M256Policy, String>> =
        std::sync::OnceLock::new();
    POLICY
        .get_or_init(|| {
            let value = match std::env::var(V29_M256_SELECTOR) {
                Ok(v) => Some(v),
                Err(std::env::VarError::NotPresent) => None,
                Err(e) => return Err(format!("invalid {V29_M256_SELECTOR}: {e}")),
            };
            parse_v29_m256_policy(value.as_deref())
                .map_err(|error| format!("{V29_M256_SELECTOR}: {error}"))
        })
        .clone()
        .map_err(anyhow::Error::msg)
}

/// Wave-21 2-CTA/SM (`0-ULP` `TILE_M=128 x BC=128`, 8 warps = 256 threads, `<= 48 KiB` SMEM)
/// MoE prefill selector (`ATLAS_QWEN4_PREFILL_MOE_V30_2CTA`):
///   unset / `0`    = disabled (opt-in only; fall back to `v31_ws` / `v28_twarp`);
///   `auto`         = auto-engage on V28 when `rows >= 2048`, `!has_corrections`,
///                    `l2_windows.is_none()`, `!v28_m64`, `!v29_m256_active`,
///                    `v28q_level == 0 || v28q_level == 2`, and `v30_2cta` kernels are loaded;
///   `1`            = force `v30_2cta` across all prefill row counts (fails closed if unavailable).
pub(crate) const V30_2CTA_SELECTOR: &str = "ATLAS_QWEN4_PREFILL_MOE_V30_2CTA";
/// Minimum prompt rows for automatic `v30_2cta` (2 CTAs/SM) prefill dispatch (`rows >= 2048`).
pub(super) const V30_2CTA_AUTO_MIN_ROWS: usize = 2048;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum V302CtaPolicy {
    Disabled,
    Auto,
    Force,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct V302CtaEligibility {
    pub rows: usize,
    pub w4a8_is_v28: bool,
    pub v28_m64_forced: bool,
    pub has_corrections: bool,
    pub l2_windowed: bool,
    pub v29_m256_active: bool,
    pub v28q_level: u32,
    pub kernels_loaded: bool,
}

impl V302CtaPolicy {
    #[must_use]
    pub(super) const fn admits(self, eligibility: V302CtaEligibility) -> bool {
        let common = eligibility.w4a8_is_v28
            && !eligibility.v28_m64_forced
            && !eligibility.has_corrections
            && !eligibility.l2_windowed
            && !eligibility.v29_m256_active
            && eligibility.kernels_loaded;
        match self {
            Self::Disabled => false,
            Self::Auto => {
                common
                    && eligibility.rows >= V30_2CTA_AUTO_MIN_ROWS
                    && (eligibility.v28q_level == 0 || eligibility.v28q_level == 2)
            }
            Self::Force => common,
        }
    }
}

fn parse_v30_2cta_policy(value: Option<&str>) -> std::result::Result<V302CtaPolicy, &'static str> {
    match value {
        None | Some("0") => Ok(V302CtaPolicy::Disabled),
        Some("auto") => Ok(V302CtaPolicy::Auto),
        Some("1") => Ok(V302CtaPolicy::Force),
        _ => Err("must be unset, 0, 1, or auto"),
    }
}

pub(super) fn v30_2cta_policy() -> Result<V302CtaPolicy> {
    static POLICY: std::sync::OnceLock<std::result::Result<V302CtaPolicy, String>> =
        std::sync::OnceLock::new();
    POLICY
        .get_or_init(|| {
            let value = match std::env::var(V30_2CTA_SELECTOR) {
                Ok(v) => Some(v),
                Err(std::env::VarError::NotPresent) => None,
                Err(e) => return Err(format!("invalid {V30_2CTA_SELECTOR}: {e}")),
            };
            parse_v30_2cta_policy(value.as_deref())
                .map_err(|error| format!("{V30_2CTA_SELECTOR}: {error}"))
        })
        .clone()
        .map_err(anyhow::Error::msg)
}

/// Wave-22 Warp-Specialized (`0-ULP` `TILE_M=128 x BC=256`, 16 warps = 512 threads:
/// WG0+WG1 256 Producer/Transcode threads @ 72 regs + WG2+WG3 256 Consumer MMA threads @ 184 regs,
/// full `TN=128 (GU) / 256 (DN)` geometry, `<= 99 KiB` SMEM) MoE prefill selector
/// (`ATLAS_QWEN4_PREFILL_MOE_V31_WS`):
///   unset / `auto` = auto-engage on V28 when `rows >= v31_ws_auto_min_rows_for_sm(sm_count)`
///                    (`2048` on `<= 64` SMs; `96` on `> 64` SMs such as SM120 188 SMs,
///                    covering both `C=24..64` batched verify frames and prefill chunks while
///                    leaving `rows < 96` on 4-CTA/SM `v28_m16`),
///                    `!has_corrections`, `l2_windows.is_none()`, `!v28_m64`, `!v29_m256_active`,
///                    `!v30_2cta_active`, `v28q_level == 0 || v28q_level == 2`,
///                    and `v31_ws` kernels are loaded;
///   `0`            = disabled (fall back to `v28_twarp`);
///   `1`            = force `v31_ws` across all prefill row counts (fails closed if unavailable).
pub(crate) const V31_WS_SELECTOR: &str = "ATLAS_QWEN4_PREFILL_MOE_V31_WS";
/// Minimum prompt rows for automatic `v31_ws` warp-specialized prefill dispatch
/// on GB10-class GPUs (`<= 64` SMs: `rows >= 2048`).
pub(super) const V31_WS_AUTO_MIN_ROWS: usize = 2048;
/// Minimum rows for automatic `v31_ws` warp-specialized dispatch on high-SM
/// Blackwell GPUs (`sm_count > 64`, e.g. SM120 188 SMs: `rows >= 33`, where
/// 188-CTA persistent warp-specialization outperforms `v33_thin_sk` and
/// `v28_m32` across `C=8..64` ragged `p_cum=0.25` verify frames and prefill).
pub(super) const V31_WS_AUTO_MIN_ROWS_HIGH_SM: usize = 33;
const V31_WS_HIGH_SM_THRESHOLD: u32 = 64;

#[inline]
#[must_use]
pub(super) const fn v31_ws_auto_min_rows_for_sm(sm_count: u32) -> usize {
    if sm_count > V31_WS_HIGH_SM_THRESHOLD {
        V31_WS_AUTO_MIN_ROWS_HIGH_SM
    } else {
        V31_WS_AUTO_MIN_ROWS
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum V31WsPolicy {
    Disabled,
    Auto,
    Force,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct V31WsEligibility {
    pub rows: usize,
    pub w4a8_is_v28: bool,
    pub v28_m64_forced: bool,
    pub has_corrections: bool,
    pub l2_windowed: bool,
    pub v29_m256_active: bool,
    pub v30_2cta_active: bool,
    pub v28q_level: u32,
    pub kernels_loaded: bool,
}

impl V31WsPolicy {
    #[must_use]
    pub(super) const fn admits_for_sm(self, eligibility: V31WsEligibility, sm_count: u32) -> bool {
        let common = eligibility.w4a8_is_v28
            && !eligibility.v28_m64_forced
            && !eligibility.has_corrections
            && !eligibility.l2_windowed
            && !eligibility.v29_m256_active
            && !eligibility.v30_2cta_active
            && eligibility.kernels_loaded;
        match self {
            Self::Disabled => false,
            Self::Auto => {
                common
                    && eligibility.rows >= v31_ws_auto_min_rows_for_sm(sm_count)
                    && (eligibility.v28q_level == 0 || eligibility.v28q_level == 2)
            }
            Self::Force => common,
        }
    }

    #[must_use]
    pub(super) fn admits(self, eligibility: V31WsEligibility) -> bool {
        self.admits_for_sm(eligibility, atlas_core::device::active_sm_count())
    }
}

fn parse_v31_ws_policy(value: Option<&str>) -> std::result::Result<V31WsPolicy, &'static str> {
    match value {
        None | Some("auto") => Ok(V31WsPolicy::Auto),
        Some("0") => Ok(V31WsPolicy::Disabled),
        Some("1") => Ok(V31WsPolicy::Force),
        _ => Err("must be unset, 0, 1, or auto"),
    }
}

pub(super) fn v31_ws_policy() -> Result<V31WsPolicy> {
    static POLICY: std::sync::OnceLock<std::result::Result<V31WsPolicy, String>> =
        std::sync::OnceLock::new();
    POLICY
        .get_or_init(|| {
            let value = match std::env::var(V31_WS_SELECTOR) {
                Ok(v) => Some(v),
                Err(std::env::VarError::NotPresent) => None,
                Err(e) => return Err(format!("invalid {V31_WS_SELECTOR}: {e}")),
            };
            parse_v31_ws_policy(value.as_deref())
                .map_err(|error| format!("{V31_WS_SELECTOR}: {error}"))
        })
        .clone()
        .map_err(anyhow::Error::msg)
}

/// Batched-verify W4A8 tensor-core MoE selector (`ATLAS_QWEN4_BATCHED_VERIFY_TC_MOE`).
/// Unset / `1` / `auto` => ON when `rows >= min_rows` (default `DEFAULT_BATCHED_VERIFY_TC_MOE_MIN_ROWS` = 4),
/// the compact W4A8 + shared expert path is ready, and either execution is eager or
/// graph-capture scratch is pre-allocated; `0` => disabled (forces exact `t8`).
pub(crate) const BATCHED_VERIFY_TC_MOE_SELECTOR: &str = "ATLAS_QWEN4_BATCHED_VERIFY_TC_MOE";
/// Optional minimum row count for batched-verify W4A8 TC MoE (default `4`, must be `>= 2`).
pub(crate) const BATCHED_VERIFY_TC_MOE_MIN_ROWS_SELECTOR: &str =
    "ATLAS_QWEN4_BATCHED_VERIFY_TC_MOE_MIN_ROWS";
/// Default minimum verify batch rows to route onto W4A8 `QMMA.SF` compact MoE.
/// Routes all `C >= 2` batched verify cohorts (`rows >= 4`, including ragged
/// `C=2 K=2..5` and `C=4..16`) onto 2-CTA/SM `v28_m32_ts4e` + in-graph shared
/// overlap (`C=1` `0-ULP` verify does not arm `BatchedVerifyTcDenseGuard`).
pub(crate) const DEFAULT_BATCHED_VERIFY_TC_MOE_MIN_ROWS: usize = 4;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct BatchedVerifyTcMoePolicy {
    pub enabled: bool,
    pub min_rows: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct BatchedVerifyTcMoeEligibility {
    pub rows: usize,
    pub graph_capture: bool,
    /// True when all per-layer shared pointer tables, identity indices, and sticky status
    /// constants are already pre-allocated on device so CUDA graph capture performs zero
    /// allocations, H2D uploads, or host D2H syncs.
    pub capture_scratch_ready: bool,
    pub compact_active: bool,
    pub w4a8_ready: bool,
    pub shared_ready: bool,
}

impl BatchedVerifyTcMoePolicy {
    #[must_use]
    pub(super) const fn admits(self, eligibility: BatchedVerifyTcMoeEligibility) -> bool {
        self.enabled
            && eligibility.rows >= self.min_rows
            && (!eligibility.graph_capture || eligibility.capture_scratch_ready)
            && eligibility.compact_active
            && eligibility.w4a8_ready
            && eligibility.shared_ready
    }
}

fn parse_batched_verify_tc_moe(value: Option<&str>) -> std::result::Result<bool, &'static str> {
    match value {
        None | Some("1") | Some("auto") => Ok(true),
        Some("0") => Ok(false),
        _ => Err("must be unset, 0, 1, or auto"),
    }
}

fn parse_batched_verify_tc_moe_min_rows(
    value: Option<&str>,
) -> std::result::Result<usize, &'static str> {
    let Some(raw) = value else {
        return Ok(DEFAULT_BATCHED_VERIFY_TC_MOE_MIN_ROWS);
    };
    match raw.parse::<usize>() {
        Ok(rows) if rows >= 2 => Ok(rows),
        _ => Err("must be an integer >= 2"),
    }
}

fn parse_batched_verify_tc_moe_policy(
    raw_mode: Option<&str>,
    raw_min_rows: Option<&str>,
) -> std::result::Result<BatchedVerifyTcMoePolicy, String> {
    let enabled = parse_batched_verify_tc_moe(raw_mode)
        .map_err(|e| format!("{BATCHED_VERIFY_TC_MOE_SELECTOR}: {e}"))?;
    let min_rows = parse_batched_verify_tc_moe_min_rows(raw_min_rows)
        .map_err(|e| format!("{BATCHED_VERIFY_TC_MOE_MIN_ROWS_SELECTOR}: {e}"))?;
    Ok(BatchedVerifyTcMoePolicy { enabled, min_rows })
}

pub(super) fn batched_verify_tc_moe_policy() -> Result<BatchedVerifyTcMoePolicy> {
    static POLICY: std::sync::OnceLock<std::result::Result<BatchedVerifyTcMoePolicy, String>> =
        std::sync::OnceLock::new();
    POLICY
        .get_or_init(|| {
            let mode = match std::env::var(BATCHED_VERIFY_TC_MOE_SELECTOR) {
                Ok(v) => Some(v),
                Err(std::env::VarError::NotPresent) => None,
                Err(e) => return Err(format!("invalid {BATCHED_VERIFY_TC_MOE_SELECTOR}: {e}")),
            };
            let min_rows = match std::env::var(BATCHED_VERIFY_TC_MOE_MIN_ROWS_SELECTOR) {
                Ok(v) => Some(v),
                Err(std::env::VarError::NotPresent) => None,
                Err(e) => {
                    return Err(format!(
                        "invalid {BATCHED_VERIFY_TC_MOE_MIN_ROWS_SELECTOR}: {e}"
                    ));
                }
            };
            parse_batched_verify_tc_moe_policy(mode.as_deref(), min_rows.as_deref())
        })
        .clone()
        .map_err(anyhow::Error::msg)
}

/// 0 = shipping kernel, 1 = pipelined v2 (64x64), 2 = v3 (64x128, 256 threads),
/// 3 = v3 down + fused gate/up/silu (64x64, 128 threads),
/// 4 = v4 (v3 tile, single-buffered sBf: 3 CTAs/SM = 24 warps, bit-exact with v3),
/// 5 = v5 (v2 tile, single-buffered sBf: 4 CTAs/SM = 16 warps, bit-exact with v2),
/// 6 = v6 (v4 with the MMAs of all-padding 16-row warp slices skipped; stored
///     outputs run v4's exact instruction sequence),
/// 7 = v7 (wide chunks: 128x128 tile over its own 128-row planner, B fragments
///     dequantized in registers with F2FP, scale2 in the epilogue; weights exact,
///     so NOT bit-identical to v2..v6),
/// 8 = v7 down + v7 fused gate/up/SwiGLU (one shared A tile, SiLU on the FP32
///     accumulators, only the activated intermediate is stored).
pub(crate) fn v2_level() -> Result<u32> {
    static LEVEL: std::sync::OnceLock<std::result::Result<u32, String>> =
        std::sync::OnceLock::new();
    LEVEL
        .get_or_init(|| match std::env::var(V2_SELECTOR) {
            Err(std::env::VarError::NotPresent) => Ok(0),
            Ok(v) if v == "0" => Ok(0),
            Ok(v) if v == "1" => Ok(1),
            Ok(v) if v == "2" => Ok(2),
            Ok(v) if v == "3" => Ok(3),
            Ok(v) if v == "4" => Ok(4),
            Ok(v) if v == "5" => Ok(5),
            Ok(v) if v == "6" => Ok(6),
            Ok(v) if v == "7" => Ok(7),
            Ok(v) if v == "8" => Ok(8),
            Ok(v) => Err(format!(
                "{V2_SELECTOR} must be absent, 0, 1, 2, 3, 4, 5, 6, 7 or 8; got {v:?}"
            )),
            Err(e) => Err(format!("invalid {V2_SELECTOR}: {e}")),
        })
        .clone()
        .map_err(anyhow::Error::msg)
}

fn flag(name: &str) -> Result<bool> {
    let value = match std::env::var(name) {
        Ok(value) => Some(value),
        Err(std::env::VarError::NotPresent) => None,
        Err(error) => return Err(error).with_context(|| format!("invalid {name}")),
    };
    abi::parse(value.as_deref()).map_err(|error| anyhow::anyhow!("{name}: {error}"))
}

pub(crate) fn selected() -> Result<bool> {
    static SEL: std::sync::OnceLock<std::result::Result<bool, String>> = std::sync::OnceLock::new();
    SEL.get_or_init(|| flag(SELECTOR).map_err(|e| e.to_string()))
        .clone()
        .map_err(anyhow::Error::msg)
}

pub(crate) fn w4a8_selected() -> Result<bool> {
    static SEL: std::sync::OnceLock<std::result::Result<bool, String>> = std::sync::OnceLock::new();
    SEL.get_or_init(|| flag(W4A8_SELECTOR).map_err(|e| e.to_string()))
        .clone()
        .map_err(anyhow::Error::msg)
}

fn prefill_moe_batch_selected() -> Result<bool> {
    static SEL: std::sync::OnceLock<std::result::Result<bool, String>> = std::sync::OnceLock::new();
    SEL.get_or_init(|| flag("ATLAS_QWEN4_PREFILL_MOE_BATCH").map_err(|e| e.to_string()))
        .clone()
        .map_err(anyhow::Error::msg)
}

fn unified_moe_layout_selected() -> Result<bool> {
    static SEL: std::sync::OnceLock<std::result::Result<bool, String>> = std::sync::OnceLock::new();
    SEL.get_or_init(|| flag("ATLAS_UNIFIED_MOE_LAYOUT").map_err(|e| e.to_string()))
        .clone()
        .map_err(anyhow::Error::msg)
}

fn oracle_dump_present() -> bool {
    static PRESENT: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *PRESENT.get_or_init(|| std::env::var_os("ATLAS_QWEN4_ORACLE_DUMP").is_some())
}

fn validate_compact_conflicts() -> Result<()> {
    static CHECKED: std::sync::OnceLock<std::result::Result<(), String>> =
        std::sync::OnceLock::new();
    CHECKED
        .get_or_init(|| {
            for name in [
                "ATLAS_NVFP4_MOE_WORKLIST",
                "ATLAS_MOE_EXACT_PREFILL_GRID",
                "ATLAS_NVFP4_GATE_UP_M128",
                "ATLAS_HYBRID_MOE_LAYOUT",
            ] {
                match flag(name) {
                    Ok(true) => return Err(format!("{SELECTOR} conflicts with {name}")),
                    Ok(false) => {}
                    Err(e) => return Err(e.to_string()),
                }
            }
            Ok(())
        })
        .clone()
        .map_err(anyhow::Error::msg)
}

pub(crate) fn check_selected() -> Result<bool> {
    static SEL: std::sync::OnceLock<std::result::Result<bool, String>> = std::sync::OnceLock::new();
    SEL.get_or_init(|| {
        let check = flag(CHECK_SELECTOR).map_err(|e| e.to_string())?;
        if check {
            let sel = selected().map_err(|e| e.to_string())?;
            let batch = prefill_moe_batch_selected().map_err(|e| e.to_string())?;
            super::qwen4_compact_compare::admit(check, sel, batch).map_err(str::to_string)?;
        }
        Ok(check)
    })
    .clone()
    .map_err(anyhow::Error::msg)
}

pub(crate) fn stream_selected() -> Result<bool> {
    static SEL: std::sync::OnceLock<std::result::Result<bool, String>> = std::sync::OnceLock::new();
    SEL.get_or_init(|| flag(STREAM_SELECTOR).map_err(|e| e.to_string()))
        .clone()
        .map_err(anyhow::Error::msg)
}

pub(crate) fn transposed_selected() -> Result<bool> {
    static SEL: std::sync::OnceLock<std::result::Result<bool, String>> = std::sync::OnceLock::new();
    SEL.get_or_init(|| {
        let persistent = flag(TRANSPOSED_SELECTOR).map_err(|e| e.to_string())?;
        let streaming = stream_selected().map_err(|e| e.to_string())?;
        if persistent && streaming {
            return Err(format!(
                "{TRANSPOSED_SELECTOR} conflicts with {STREAM_SELECTOR}"
            ));
        }
        Ok(persistent || streaming)
    })
    .clone()
    .map_err(anyhow::Error::msg)
}

fn selected_step_k() -> Result<u32> {
    static STEP_K: std::sync::OnceLock<std::result::Result<u32, String>> =
        std::sync::OnceLock::new();
    STEP_K
        .get_or_init(|| {
            Ok(if flag(K32_SELECTOR).map_err(|e| e.to_string())? {
                32
            } else {
                16
            })
        })
        .clone()
        .map_err(anyhow::Error::msg)
}

/// Safe to call before embedding or any model/layer effects; no recursion into F8.
pub(crate) fn admit_request(config: &ModelConfig, rows: usize, start: usize) -> Result<()> {
    check_selected()?;
    let step_k = selected_step_k()?;
    let transposed = transposed_selected()?;
    let streaming = stream_selected()?;
    let slot_storage = slot_storage_selected()?;
    ensure!(
        step_k == 16 || selected()?,
        "K32 compact schedule requires {SELECTOR}=1"
    );
    ensure!(
        !transposed || selected()?,
        "{TRANSPOSED_SELECTOR} requires {SELECTOR}=1"
    );
    ensure!(
        !transposed || step_k == 32,
        "{TRANSPOSED_SELECTOR} requires {K32_SELECTOR}=1"
    );
    ensure!(
        !transposed || !check_selected()?,
        "{TRANSPOSED_SELECTOR} rejects replay after original retirement"
    );
    let unified = unified_moe_layout_selected()?;
    ensure!(
        !transposed || streaming || unified,
        "{TRANSPOSED_SELECTOR} requires ATLAS_UNIFIED_MOE_LAYOUT=1"
    );
    ensure!(
        !streaming || !unified,
        "{STREAM_SELECTOR} conflicts with ATLAS_UNIFIED_MOE_LAYOUT"
    );
    if slot_storage.is_mxfp8() {
        ensure!(
            selected()?
                && transposed
                && step_k == 32
                && v2_level()? > 0
                && w4a8_selected()?
                && matches!(w4a8_wide_level()?, 2 | 7),
            "{SLOT_MXFP8_SELECTOR} requires compact T/K32/V2, {W4A8_SELECTOR}=1 and \
             {W4A8_WIDE_SELECTOR}=2 or 7"
        );
    }
    if slot_bf16_fused_selected()? {
        ensure!(
            selected()?
                && transposed
                && step_k == 32
                && v2_level()? > 0
                && w4a8_selected()?
                && matches!(w4a8_wide_level()?, 2 | 7),
            "{SLOT_BF16_FUSED_SELECTOR} requires compact T/K32/V2, {W4A8_SELECTOR}=1 and \
             {W4A8_WIDE_SELECTOR}=2 or 7"
        );
    }
    if v28q_level()? > 0 {
        ensure!(
            selected()?
                && transposed
                && step_k == 32
                && v2_level()? > 0
                && w4a8_selected()?
                && w4a8_wide_level()? == 7,
            "{V28Q_SELECTOR} requires compact T/K32/V2, {W4A8_SELECTOR}=1 and \
             {W4A8_WIDE_SELECTOR}=7"
        );
    }
    if v28_m64_selected()? {
        ensure!(
            selected()?
                && transposed
                && step_k == 32
                && v2_level()? > 0
                && w4a8_selected()?
                && w4a8_wide_level()? == 7,
            "{V28_M64_SELECTOR} requires compact T/K32/V2, {W4A8_SELECTOR}=1 and \
             {W4A8_WIDE_SELECTOR}=7"
        );
    }
    if v29_m256_policy()? == V29M256Policy::Force {
        ensure!(
            selected()?
                && transposed
                && step_k == 32
                && v2_level()? > 0
                && w4a8_selected()?
                && w4a8_wide_level()? == 7,
            "{V29_M256_SELECTOR}=1 requires compact T/K32/V2, {W4A8_SELECTOR}=1 and \
             {W4A8_WIDE_SELECTOR}=7"
        );
        ensure!(
            !v28_m64_selected()?,
            "{V29_M256_SELECTOR}=1 conflicts with {V28_M64_SELECTOR}=1"
        );
    }
    if v30_2cta_policy()? == V302CtaPolicy::Force {
        ensure!(
            selected()?
                && transposed
                && step_k == 32
                && v2_level()? > 0
                && w4a8_selected()?
                && w4a8_wide_level()? == 7,
            "{V30_2CTA_SELECTOR}=1 requires compact T/K32/V2, {W4A8_SELECTOR}=1 and \
             {W4A8_WIDE_SELECTOR}=7"
        );
        ensure!(
            !v28_m64_selected()?,
            "{V30_2CTA_SELECTOR}=1 conflicts with {V28_M64_SELECTOR}=1"
        );
        ensure!(
            v29_m256_policy()? != V29M256Policy::Force,
            "{V30_2CTA_SELECTOR}=1 conflicts with {V29_M256_SELECTOR}=1"
        );
    }
    if v31_ws_policy()? == V31WsPolicy::Force {
        ensure!(
            selected()?
                && transposed
                && step_k == 32
                && v2_level()? > 0
                && w4a8_selected()?
                && w4a8_wide_level()? == 7,
            "{V31_WS_SELECTOR}=1 requires compact T/K32/V2, {W4A8_SELECTOR}=1 and \
             {W4A8_WIDE_SELECTOR}=7"
        );
        ensure!(
            !v28_m64_selected()?,
            "{V31_WS_SELECTOR}=1 conflicts with {V28_M64_SELECTOR}=1"
        );
        ensure!(
            v29_m256_policy()? != V29M256Policy::Force,
            "{V31_WS_SELECTOR}=1 conflicts with {V29_M256_SELECTOR}=1"
        );
        ensure!(
            v30_2cta_policy()? != V302CtaPolicy::Force,
            "{V31_WS_SELECTOR}=1 conflicts with {V30_2CTA_SELECTOR}=1"
        );
    }
    let w4a4 = w4a4_mode()?;
    if w4a4.is_enabled() {
        ensure!(
            selected()?
                && transposed
                && step_k == 32
                && v2_level()? > 0
                && w4a8_selected()?
                && w4a8_wide_level()? == 7,
            "{W4A4_SELECTOR}={} requires compact T/K32/V2, {W4A8_SELECTOR}=1 and \
             {W4A8_WIDE_SELECTOR}=7",
            w4a4.as_str(),
        );
        ensure!(
            !v28_m64_selected()?,
            "{W4A4_SELECTOR} conflicts with {V28_M64_SELECTOR}=1"
        );
        ensure!(
            v29_m256_policy()? != V29M256Policy::Force,
            "{W4A4_SELECTOR} conflicts with {V29_M256_SELECTOR}=1"
        );
        ensure!(
            v30_2cta_policy()? != V302CtaPolicy::Force,
            "{W4A4_SELECTOR} conflicts with {V30_2CTA_SELECTOR}=1"
        );
        ensure!(
            !slot_storage.has_corrections(),
            "{W4A4_SELECTOR} conflicts with {SLOT_MXFP8_SELECTOR}=2 residual corrections"
        );
    }
    if !selected()? {
        return Ok(());
    }
    ensure!(
        prefill_moe_batch_selected()?,
        "{SELECTOR} requires ATLAS_QWEN4_PREFILL_MOE_BATCH=1"
    );
    let pp = config.pp_world_size.max(1);
    ensure!(
        config.is_qwen4_exp()
            && config.hidden_size == 2560
            && config.residual_width() == 10240
            && config.num_hidden_layers * pp == 48
            && config.num_attention_layers() * pp == 12
            && config.num_ssm_layers() * pp == 36
            && config.num_experts == 512
            && config.num_experts_per_tok == 10
            && config.moe_intermediate_size == 640
            && config.shared_expert_intermediate_size == 640
            && !config.use_fp32_residual()
            && config.ep_world_size <= 4
            && config.tp_world_size <= 4,
        "{SELECTOR} requires canonical Flash-Next (TP<=4, EP<=4)"
    );
    // Singleton chunks retain shipping decode and do not engage compact GEMM.
    ensure!(
        crate::layers::qwen4_prefill_moe::chunk_window_admitted(rows, start)?,
        "{SELECTOR} requires an admitted nonempty dense-window chunk"
    );
    validate_compact_conflicts()?;
    ensure!(
        transposed || !unified,
        "{SELECTOR} original layout conflicts with ATLAS_UNIFIED_MOE_LAYOUT"
    );
    Ok(())
}

#[derive(Clone, Copy)]
pub(super) struct Kernels {
    pub plan: KernelHandle,
    pub gemm: KernelHandle,
    pub block: u32,
    pub gateup: Option<KernelHandle>,
    pub gateup_block: u32,
    /// The V7 set (128-row planner, fused gate/up, b12x weight-rounding twins), loaded on
    /// every V2 path so ATLAS_QWEN4_PREFILL_FAKEQUANT=v7|b12x can switch to it per dispatch.
    pub wide: Option<WideKernels>,
    pub step_k: u32,
    pub transposed: bool,
    pub streamed: bool,
    /// Zero-based trunk layer index assigned in sequential layer-load order (`0..num_layers`).
    pub layer_idx: usize,
}

/// V7 kernels (all 256-thread CTAs over the V7 planner's 128-row work items).
#[derive(Clone, Copy)]
pub(super) struct WideKernels {
    pub plan: KernelHandle,
    pub gemm: KernelHandle,
    pub gateup: KernelHandle,
    pub gemm_b12xw: KernelHandle,
    pub gateup_b12xw: KernelHandle,
    /// V10 W4A8 set: MXFP8 row quantiser, fused gate/up/SwiGLU writing the MXFP8 down
    /// operand, and down (bf16 out).
    pub w4a8_quant: KernelHandle,
    /// ATLAS_QWEN4_PREFILL_MOE_SLOT_BF16_FUSED at load: `w4a8_quant` is the quad-lane twin
    /// `qwen4_mxfp8_quant_rows_v16q4` (V16 or V28 set) instead of the set's own `..._<tag>`.
    pub xq_fused: bool,
    /// Fused MXFP8 row quantiser + shared-expert gate scalar (`qwen4_mxfp8_quant_rows_v16q4_gate`),
    /// loaded on V16 and V28 sets for `ATLAS_QWEN4_PREFILL_MHC_FUSED=...,gatepre`.
    pub w4a8_quant_gate: Option<KernelHandle>,
    pub w4a8_gateup: KernelHandle,
    pub w4a8_gemm: KernelHandle,
    /// "v10", "v12", "v14", "v16", or "v28", for the receipt.
    pub w4a8_tag: &'static str,
    /// ATLAS_QWEN4_PREFILL_MOE_V28Q level (0=off, 1=v28q4+v28q, 2=v28_twarp+v28q_twarp, 3=v28q4, 4=v28q; or m64 variants when V28_M64=1).
    pub v28q_level: u32,
    /// ATLAS_QWEN4_PREFILL_MOE_V28_M64 / ATLAS_QWEN4_MOE_SMALL_TILE hybrid M64 dispatch.
    pub v28_m64: bool,
    /// CTA size and dynamic shared memory of the W4A8 gate/up and down kernels.
    pub w4a8_block: u32,
    pub w4a8_smem: u32,
    pub w4a8_smem_nopf: u32,
    /// Dynamic shared memory of the W4A8 down kernel.
    pub w4a8_smem_gemm: u32,
    /// ATLAS_QWEN4_PREFILL_MOE_SLOT_MXFP8: the V17/V28q down kernel writing MXFP8 slot rows, and the
    /// matching unpermute.
    pub slot_storage: SlotStorage,
    pub w4a8_down_mxfp8: Option<KernelHandle>,
    pub w4a8_smem_down_mxfp8: u32,
    pub unpermute_mxfp8: Option<KernelHandle>,
    /// Vectorized 128-bit fused MXFP8 unpermute-reduce + shared-expert blend reusing `ss_ptr`
    /// from `v16q4_gate` and folding sticky status checks (`qwen4_moe_compact_t_k32_v17`).
    pub unpermute_blend_mxfp8: Option<KernelHandle>,
    pub w4a8_gateup_nopf: Option<KernelHandle>,
    pub w4a8_down_mxfp8_nopf: Option<KernelHandle>,
    pub w4a8_smem_down_mxfp8_nopf: u32,
    /// ATLAS_QWEN4_PREFILL_MOE_PLAN_ORDER=inter: the interleaved planner (same items as `plan`, multi-tile
    /// experts contiguous, single-tile experts spread between them) and its row gate. W4A8 dispatches only.
    pub plan_order: Option<(KernelHandle, PlanOrder)>,
    /// Wave-17 Path A (`TILE_M=256 x BC=128`) 0-ULP V29 kernels + M256 planners, loaded on V28.
    pub v29_m256: Option<super::qwen4_compact_ops::V29M256Kernels>,
    /// Wave-21 2-CTA/SM (`TILE_M=128 x BC=128`, `<= 48 KiB` SMEM) 0-ULP V30 kernels, loaded on V28.
    pub v30_2cta: Option<super::qwen4_compact_ops::V302CtaKernels>,
    /// Wave-22 Warp-Specialized (`TILE_M=128 x BC=256`, 512 threads, `<= 99 KiB` SMEM) 0-ULP V31 kernels, loaded on V28.
    pub v31_ws: Option<super::qwen4_compact_ops::V31WsKernels>,
    /// Native Blackwell W4A4 (`NVFP4 x NVFP4`, `m16n8k64` `4X`, 2 CTAs/SM, `<= 48 KiB` SMEM)
    /// V32 kernels, loaded on V28 when `ATLAS_QWEN4_PREFILL_MOE_W4A4` is enabled.
    pub w4a4_v32: Option<W4a4V32Kernels>,
}

static PLAN_ORDER_RECEIPT_LOGGED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);
static V28Q_RECEIPT_LOGGED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);
static V28_M64_RECEIPT_LOGGED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

fn env_value(name: &str) -> Result<Option<String>> {
    match std::env::var(name) {
        Ok(value) => Ok(Some(value)),
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(error) => Err(error).with_context(|| format!("invalid {name}")),
    }
}

/// One MOE_PREFILL_PLAN_ORDER_REFUSED line per process, then the error that refuses the load.
fn plan_order_refused(reason: String) -> anyhow::Error {
    if !PLAN_ORDER_RECEIPT_LOGGED.swap(true, std::sync::atomic::Ordering::Relaxed) {
        tracing::warn!(
            "MOE_PREFILL_PLAN_ORDER_REFUSED selector={PLAN_ORDER_SELECTOR} reason={reason:?}"
        );
    }
    anyhow::anyhow!(reason)
}

/// ATLAS_QWEN4_PREFILL_MOE_PLAN_ORDER (+ _MAX_ROWS): absent/0/v7 -> None (V7's planner, unchanged). `inter` loads
/// the V8 planner; it requires the W4A8 path (the only dispatch that consults it). Malformed values or `inter`
/// without W4A8 refuse the load (fail closed). One ENGAGED/REFUSED line per process.
fn load_plan_order(gpu: &dyn GpuBackend, w4a8: bool) -> Result<Option<(KernelHandle, PlanOrder)>> {
    let order = env_value(PLAN_ORDER_SELECTOR)?;
    let max_rows = env_value(PLAN_ORDER_MAX_ROWS_SELECTOR)?;
    let parsed = parse_plan_order(order.as_deref(), max_rows.as_deref());
    let Some(order) = parsed.map_err(plan_order_refused)? else {
        return Ok(None);
    };
    if !w4a8 {
        return Err(plan_order_refused(format!(
            "{PLAN_ORDER_SELECTOR}=inter requires {W4A8_SELECTOR}=1"
        )));
    }
    let (handle, planner_name) = if w4a8_wide_level()? == 7 {
        match gpu.kernel(
            "qwen4_moe_compact_t_k32_v28",
            "qwen4_moe_compact_t_plan_k32_v28_inter",
        ) {
            Ok(h) => (h, "qwen4_moe_compact_t_plan_k32_v28_inter"),
            Err(_) => (gpu.kernel(PLAN_V8_MODULE, PLAN_V8_KERNEL)?, PLAN_V8_KERNEL),
        }
    } else {
        (gpu.kernel(PLAN_V8_MODULE, PLAN_V8_KERNEL)?, PLAN_V8_KERNEL)
    };
    let trigger = env_value(PLAN_ORDER_TRIGGER_SELECTOR)?.unwrap_or_else(|| "none".to_string());
    if !PLAN_ORDER_RECEIPT_LOGGED.swap(true, std::sync::atomic::Ordering::Relaxed) {
        tracing::info!(
            "MOE_PREFILL_PLAN_ORDER_ENGAGED selector={PLAN_ORDER_SELECTOR} order=inter planner={planner_name} max_rows={} trigger={trigger}",
            order.max_rows_label()
        );
    }
    Ok(Some((handle, order)))
}

fn load_wide(gpu: &dyn GpuBackend) -> Result<WideKernels> {
    let module = "qwen4_moe_compact_t_k32_v7";
    let (w4a8_module, v, block, smem, smem_gemm) = match w4a8_wide_level()? {
        7 => (
            "qwen4_moe_compact_t_k32_v28",
            "v28",
            512,
            V28_SMEM_BYTES,
            V28_SMEM_BYTES,
        ),
        2 => (
            "qwen4_moe_compact_t_k32_v16",
            "v16",
            512,
            V16_SMEM_BYTES,
            V16_SMEM_BYTES,
        ),
        1 => (
            "qwen4_moe_compact_t_k32_v14",
            "v14",
            512,
            V14_SMEM_BYTES,
            V14_SMEM_BYTES,
        ),
        _ if flag(W4A8_K64_SELECTOR)? => ("qwen4_moe_compact_t_k32_v12", "v12", 256, 0, 0),
        _ => ("qwen4_moe_compact_t_k32_v10", "v10", 256, 0, 0),
    };
    let v28q = v28q_level()?;
    if v28q > 0 {
        ensure!(
            w4a8_selected()? && v == "v28",
            "{V28Q_SELECTOR} requires {W4A8_SELECTOR}=1 and {W4A8_WIDE_SELECTOR}=7"
        );
    }
    let v28_m64 = v28_m64_selected()?;
    if v28_m64 {
        ensure!(
            w4a8_selected()? && v == "v28",
            "{V28_M64_SELECTOR} requires {W4A8_SELECTOR}=1 and {W4A8_WIDE_SELECTOR}=7"
        );
    }
    let gemm_suffix = if v28_m64 { "v28_m64" } else { v };
    let (gateup_name, gateup_smem) = match v28q {
        1 | 3 if !v28_m64 => (
            "qwen4_moe_compact_t_gemm_gateup_k32_v28q4".to_string(),
            smem,
        ),
        2 if !v28_m64 => (
            "qwen4_moe_compact_t_gemm_gateup_k32_v28_twarp".to_string(),
            smem,
        ),
        _ if v28_m64 => (
            format!("qwen4_moe_compact_t_gemm_gateup_k32_{gemm_suffix}"),
            smem,
        ),
        _ => (format!("qwen4_moe_compact_t_gemm_gateup_k32_{v}"), smem),
    };
    let w4a8_gateup = gpu.kernel(w4a8_module, &gateup_name)?;
    if gateup_smem > 0 {
        gpu.set_kernel_max_dynamic_shared_memory(w4a8_gateup, gateup_smem)?;
    }
    let (w4a8_gateup_nopf, w4a8_smem_nopf) = if v28q == 2 {
        (Some(w4a8_gateup), gateup_smem)
    } else {
        (None, smem)
    };
    let w4a8_gemm = gpu.kernel(
        w4a8_module,
        &format!("qwen4_moe_compact_t_gemm_k32_{gemm_suffix}"),
    )?;
    if smem > 0 {
        gpu.set_kernel_max_dynamic_shared_memory(w4a8_gemm, smem_gemm)?;
    }
    let slot_storage = slot_storage_selected()?;
    let (
        w4a8_down_mxfp8,
        w4a8_smem_down_mxfp8,
        w4a8_down_mxfp8_nopf,
        w4a8_smem_down_mxfp8_nopf,
        unpermute_mxfp8,
        unpermute_blend_mxfp8,
        down_mxfp8_label,
    ) = if slot_storage.is_mxfp8() {
        ensure!(
            w4a8_selected()?,
            "{SLOT_MXFP8_SELECTOR} requires {W4A8_SELECTOR}=1"
        );
        ensure!(
            matches!(v, "v16" | "v28"),
            "{SLOT_MXFP8_SELECTOR} requires {W4A8_WIDE_SELECTOR}=2 or 7 (and {W4A8_SELECTOR}=1)"
        );
        let unpermute_module = "qwen4_moe_compact_t_k32_v17";
        let (down_module, down_name, unpermute_name, down_smem) = match (slot_storage, v28q) {
            (SlotStorage::Mxfp8, 1 | 4) if !v28_m64 => (
                "qwen4_moe_compact_t_k32_v28",
                "qwen4_moe_compact_t_gemm_k32_v28q",
                "moe_unpermute_reduce_mxfp8",
                V16_SMEM_BYTES,
            ),
            (SlotStorage::Mxfp8, 2) if !v28_m64 => (
                "qwen4_moe_compact_t_k32_v28",
                "qwen4_moe_compact_t_gemm_k32_v28q_twarp",
                "moe_unpermute_reduce_mxfp8",
                V16_SMEM_BYTES,
            ),
            (SlotStorage::Mxfp8, _) if v28_m64 => (
                w4a8_module,
                "qwen4_moe_compact_t_gemm_k32_v28q_m64",
                "moe_unpermute_reduce_mxfp8",
                V16_SMEM_BYTES,
            ),
            (SlotStorage::Mxfp8, _) => (
                "qwen4_moe_compact_t_k32_v17",
                "qwen4_moe_compact_t_gemm_k32_v17q",
                "moe_unpermute_reduce_mxfp8",
                V16_SMEM_BYTES,
            ),
            (SlotStorage::Mxfp8R2, _) => (
                "qwen4_moe_compact_t_k32_v17",
                "qwen4_moe_compact_t_gemm_k32_v17qr2",
                "moe_unpermute_reduce_mxfp8_r2",
                V16_SMEM_BYTES,
            ),
            (SlotStorage::Bf16, _) => unreachable!("guarded MXFP8 storage"),
        };
        let down = gpu.kernel(down_module, down_name)?;
        gpu.set_kernel_max_dynamic_shared_memory(down, down_smem)?;
        let (down_nopf, down_nopf_smem) = if matches!((slot_storage, v28q), (SlotStorage::Mxfp8, 2))
        {
            (Some(down), down_smem)
        } else {
            (None, down_smem)
        };
        let unpermute_blend = if slot_storage == SlotStorage::Mxfp8 {
            gpu.kernel(unpermute_module, "moe_unpermute_reduce_blend_mxfp8_bf16")
                .ok()
        } else {
            None
        };
        (
            Some(down),
            down_smem,
            down_nopf,
            down_nopf_smem,
            Some(gpu.kernel(unpermute_module, unpermute_name)?),
            unpermute_blend,
            down_name,
        )
    } else {
        (None, 0, None, 0, None, None, "none")
    };
    if v28q > 0 && !V28Q_RECEIPT_LOGGED.swap(true, std::sync::atomic::Ordering::Relaxed) {
        tracing::info!(
            "MOE_PREFILL_V28Q_ENGAGED selector={V28Q_SELECTOR} level={v28q} gateup={gateup_name} down_mxfp8={down_mxfp8_label}"
        );
    }
    if v28_m64 && !V28_M64_RECEIPT_LOGGED.swap(true, std::sync::atomic::Ordering::Relaxed) {
        tracing::info!(
            "MOE_TILE64_ENGAGED MOE_PREFILL_V28_M64_ENGAGED selector={V28_M64_SELECTOR} gateup={gateup_name} down_mxfp8={down_mxfp8_label}"
        );
    }
    // Off: the set's own `qwen4_mxfp8_quant_rows_{v}` (v10/v12/v14/v16/v28). On
    // (ATLAS_QWEN4_PREFILL_MOE_SLOT_BF16_FUSED): the quad-lane twin, which the V16 module defines
    // and the V28 module carries through its `#include` of the V16 source.
    let xq_fused = slot_bf16_fused_selected()?;
    let xq_quant_kernel = match xq_quant_kernel(xq_fused, v) {
        Ok(name) => name,
        Err(reason) => {
            tracing::warn!(
                "MOE_PREFILL_SLOT_BF16_FUSED_REFUSED selector={SLOT_BF16_FUSED_SELECTOR} reason={reason:?} loaded={v}"
            );
            bail!("{SLOT_BF16_FUSED_SELECTOR} {reason}");
        }
    };
    if xq_fused {
        ensure!(
            w4a8_selected()?,
            "{SLOT_BF16_FUSED_SELECTOR} requires {W4A8_SELECTOR}=1"
        );
    }
    let w4a8_quant_gate = if matches!(v, "v16" | "v28") {
        gpu.kernel(
            w4a8_module,
            super::qwen4_xq_quant::XQ_QUANT_KERNEL_V16Q4_GATE,
        )
        .ok()
    } else {
        None
    };
    let plan = if v == "v28" {
        gpu.kernel(w4a8_module, "qwen4_moe_compact_t_plan_k32_v28")
            .or_else(|_| gpu.kernel(module, "qwen4_moe_compact_t_plan_k32_v7"))?
    } else {
        gpu.kernel(module, "qwen4_moe_compact_t_plan_k32_v7")?
    };
    let v29_policy = v29_m256_policy()?;
    if v29_policy == V29M256Policy::Force {
        ensure!(
            w4a8_selected()? && v == "v28",
            "{V29_M256_SELECTOR}=1 requires {W4A8_SELECTOR}=1 and {W4A8_WIDE_SELECTOR}=7"
        );
        ensure!(
            !v28_m64,
            "{V29_M256_SELECTOR}=1 conflicts with {V28_M64_SELECTOR}=1"
        );
    }
    let v29_m256 = if v == "v28" && v29_policy != V29M256Policy::Disabled {
        match super::qwen4_compact_ops::load_v29_m256_kernels(gpu) {
            Ok(kernels) => Some(kernels),
            Err(err) if v29_policy == V29M256Policy::Force => return Err(err),
            Err(err) => {
                tracing::warn!(
                    "MOE_PREFILL_V29_M256_UNAVAILABLE selector={V29_M256_SELECTOR} error={err:#}"
                );
                None
            }
        }
    } else {
        None
    };
    let v30_policy = v30_2cta_policy()?;
    if v30_policy == V302CtaPolicy::Force {
        ensure!(
            w4a8_selected()? && v == "v28",
            "{V30_2CTA_SELECTOR}=1 requires {W4A8_SELECTOR}=1 and {W4A8_WIDE_SELECTOR}=7"
        );
        ensure!(
            !v28_m64,
            "{V30_2CTA_SELECTOR}=1 conflicts with {V28_M64_SELECTOR}=1"
        );
        ensure!(
            v29_policy != V29M256Policy::Force,
            "{V30_2CTA_SELECTOR}=1 conflicts with {V29_M256_SELECTOR}=1"
        );
    }
    let v30_2cta = if v == "v28" && v30_policy != V302CtaPolicy::Disabled {
        match super::qwen4_compact_ops::load_v30_2cta_kernels(gpu) {
            Ok(kernels) => Some(kernels),
            Err(err) if v30_policy == V302CtaPolicy::Force => return Err(err),
            Err(err) => {
                tracing::warn!(
                    "MOE_PREFILL_V30_2CTA_UNAVAILABLE selector={V30_2CTA_SELECTOR} error={err:#}"
                );
                None
            }
        }
    } else {
        None
    };
    let v31_policy = v31_ws_policy()?;
    if v31_policy == V31WsPolicy::Force {
        ensure!(
            w4a8_selected()? && v == "v28",
            "{V31_WS_SELECTOR}=1 requires {W4A8_SELECTOR}=1 and {W4A8_WIDE_SELECTOR}=7"
        );
        ensure!(
            !v28_m64,
            "{V31_WS_SELECTOR}=1 conflicts with {V28_M64_SELECTOR}=1"
        );
        ensure!(
            v29_policy != V29M256Policy::Force,
            "{V31_WS_SELECTOR}=1 conflicts with {V29_M256_SELECTOR}=1"
        );
        ensure!(
            v30_policy != V302CtaPolicy::Force,
            "{V31_WS_SELECTOR}=1 conflicts with {V30_2CTA_SELECTOR}=1"
        );
    }
    let v31_ws = if v == "v28" && v31_policy != V31WsPolicy::Disabled {
        match super::qwen4_compact_ops::load_v31_ws_kernels(gpu) {
            Ok(kernels) => Some(kernels),
            Err(err) if v31_policy == V31WsPolicy::Force => return Err(err),
            Err(err) => {
                tracing::warn!(
                    "MOE_PREFILL_V31_WS_UNAVAILABLE selector={V31_WS_SELECTOR} error={err:#}"
                );
                None
            }
        }
    } else {
        None
    };
    let w4a4 = w4a4_mode()?;
    let w4a4_v32 = if w4a4.is_enabled() {
        ensure!(
            w4a8_selected()? && v == "v28",
            "{W4A4_SELECTOR}={} requires {W4A8_SELECTOR}=1 and {W4A8_WIDE_SELECTOR}=7",
            w4a4.as_str(),
        );
        ensure!(
            !v28_m64,
            "{W4A4_SELECTOR} conflicts with {V28_M64_SELECTOR}=1"
        );
        ensure!(
            v29_policy != V29M256Policy::Force,
            "{W4A4_SELECTOR} conflicts with {V29_M256_SELECTOR}=1"
        );
        ensure!(
            v30_policy != V302CtaPolicy::Force,
            "{W4A4_SELECTOR} conflicts with {V30_2CTA_SELECTOR}=1"
        );
        ensure!(
            !slot_storage.has_corrections(),
            "{W4A4_SELECTOR} conflicts with {SLOT_MXFP8_SELECTOR}=2 residual corrections"
        );
        validate_w4a4_v31_compat(w4a4, v31_ws.is_some())?;
        Some(load_w4a4_v32_kernels(gpu)?)
    } else {
        None
    };
    Ok(WideKernels {
        plan,
        gemm: gpu.kernel(module, "qwen4_moe_compact_t_gemm_k32_v7")?,
        gateup: gpu.kernel(module, "qwen4_moe_compact_t_gemm_gateup_k32_v7")?,
        gemm_b12xw: gpu.kernel(module, "qwen4_moe_compact_t_gemm_k32_v7_b12xw")?,
        gateup_b12xw: gpu.kernel(module, "qwen4_moe_compact_t_gemm_gateup_k32_v7_b12xw")?,
        w4a8_quant: gpu.kernel(w4a8_module, &xq_quant_kernel)?,
        xq_fused,
        w4a8_quant_gate,
        w4a8_gateup,
        w4a8_gemm,
        w4a8_tag: v,
        v28q_level: v28q,
        v28_m64,
        w4a8_block: block,
        w4a8_smem: gateup_smem,
        w4a8_smem_nopf,
        w4a8_smem_gemm: smem_gemm,
        slot_storage,
        w4a8_down_mxfp8,
        w4a8_smem_down_mxfp8,
        unpermute_mxfp8,
        unpermute_blend_mxfp8,
        w4a8_gateup_nopf,
        w4a8_down_mxfp8_nopf,
        w4a8_smem_down_mxfp8_nopf,
        plan_order: load_plan_order(gpu, w4a8_selected()?)?,
        v29_m256,
        v30_2cta,
        v31_ws,
        w4a4_v32,
    })
}

fn validate_w4a4_v31_compat(w4a4: qwen4_moe_w4a4::W4a4Mode, v31_ws_loaded: bool) -> Result<()> {
    ensure!(
        w4a4 != qwen4_moe_w4a4::W4a4Mode::GateUp || v31_ws_loaded,
        "{W4A4_SELECTOR}=gateup requires V31_WS DOWN kernels (do not set {V31_WS_SELECTOR}=0)"
    );
    Ok(())
}

fn validate_w4a4_arenas(
    rows: usize,
    w4a4: qwen4_moe_w4a4::W4a4Mode,
    expert_up_out_bytes: usize,
    expert_down_out_bytes: usize,
    expert_gate_out_bytes: usize,
) -> Result<()> {
    let _ = W4a4ScratchLayout::for_rows(
        rows,
        w4a4,
        expert_up_out_bytes.min(expert_down_out_bytes),
        expert_gate_out_bytes,
    )?;
    Ok(())
}

pub(super) fn validate_handles(kernels: Option<Kernels>) -> Result<()> {
    let step_k = selected_step_k()?;
    let transposed = transposed_selected()?;
    let streamed = stream_selected()?;
    let slot_storage = slot_storage_selected()?;
    let v28q = v28q_level()?;
    let v28_m64 = v28_m64_selected()?;
    let v29_policy = v29_m256_policy()?;
    ensure!(
        v29_policy != V29M256Policy::Force
            || kernels
                .and_then(|kernels| kernels.wide)
                .is_some_and(|wide| wide.v29_m256.is_some()),
        "{V29_M256_SELECTOR}=1 requires loaded v29_m256 kernel handles"
    );
    let v30_policy = v30_2cta_policy()?;
    ensure!(
        v30_policy != V302CtaPolicy::Force
            || kernels
                .and_then(|kernels| kernels.wide)
                .is_some_and(|wide| wide.v30_2cta.is_some()),
        "{V30_2CTA_SELECTOR}=1 requires loaded v30_2cta kernel handles"
    );
    let v31_policy = v31_ws_policy()?;
    ensure!(
        v31_policy != V31WsPolicy::Force
            || kernels
                .and_then(|kernels| kernels.wide)
                .is_some_and(|wide| wide.v31_ws.is_some()),
        "{V31_WS_SELECTOR}=1 requires loaded v31_ws kernel handles"
    );
    let w4a4 = w4a4_mode()?;
    ensure!(
        !w4a4.is_enabled()
            || kernels
                .and_then(|kernels| kernels.wide)
                .is_some_and(|wide| wide.w4a4_v32.is_some()),
        "{W4A4_SELECTOR}={} requires loaded v32_w4a4 kernel handles",
        w4a4.as_str(),
    );
    ensure!(
        w4a4 != qwen4_moe_w4a4::W4a4Mode::GateUp
            || kernels
                .and_then(|kernels| kernels.wide)
                .is_some_and(|wide| wide.v31_ws.is_some()),
        "{W4A4_SELECTOR}=gateup requires loaded v31_ws DOWN kernel handles",
    );
    ensure!(
        abi::bundle_matches(selected()?, kernels.map(|k| (k.plan.0, k.gemm.0))),
        "{SELECTOR} requires an unchanged selector and both private kernels"
    );
    ensure!(
        kernels.is_none_or(|kernels| kernels.step_k == step_k),
        "compact STEP_K selector changed after kernel load"
    );
    ensure!(
        kernels.is_none_or(|kernels| kernels.transposed == transposed),
        "compact layout selector changed after kernel load"
    );
    ensure!(
        kernels.is_none_or(|kernels| kernels.streamed == streamed),
        "compact stream selector changed after kernel load"
    );
    ensure!(
        kernels
            .and_then(|kernels| kernels.wide)
            .is_none_or(|wide| wide.slot_storage == slot_storage),
        "compact slot-storage selector changed after kernel load"
    );
    ensure!(
        kernels
            .and_then(|kernels| kernels.wide)
            .is_none_or(|wide| wide.v28q_level == v28q),
        "{V28Q_SELECTOR} changed after kernel load"
    );
    ensure!(
        kernels
            .and_then(|kernels| kernels.wide)
            .is_none_or(|wide| wide.v28_m64 == v28_m64),
        "{V28_M64_SELECTOR} changed after kernel load"
    );
    let xq_fused = slot_bf16_fused_selected()?;
    ensure!(
        kernels
            .and_then(|kernels| kernels.wide)
            .is_none_or(|wide| wide.xq_fused == xq_fused),
        "{SLOT_BF16_FUSED_SELECTOR} changed after kernel load"
    );
    ensure!(
        !slot_storage.is_mxfp8()
            || kernels
                .and_then(|kernels| kernels.wide)
                .is_some_and(|wide| {
                    wide.slot_storage == slot_storage
                        && wide.w4a8_down_mxfp8.is_some()
                        && wide.unpermute_mxfp8.is_some()
                }),
        "{SLOT_MXFP8_SELECTOR} requires its loaded producer and consumer handles"
    );
    Ok(())
}

/// The native Qwen4 MTP head is a one-layer copy of the trunk config
/// (`weight_loader::qwen4_mtp_config`). It only ever decodes one token at a
/// time (drafting and prompt replay), never enters compact prefill, and so
/// must not be held to the trunk's canonical 48-layer shape.
pub(crate) fn is_qwen4_mtp_head(config: &ModelConfig) -> bool {
    config.is_qwen4_exp() && config.weight_prefix == "mtp" && config.num_hidden_layers == 1
}

pub(super) fn load(gpu: &dyn GpuBackend, config: &ModelConfig) -> Result<Option<Kernels>> {
    // No compact kernels for the MTP head. Its MoE reaching compact prefill
    // anyway would fail closed in `validate_handles` (selected, no bundle).
    if is_qwen4_mtp_head(config) {
        return Ok(None);
    }
    admit_request(config, 2, 0)?;
    if !selected()? {
        return Ok(None);
    }
    let step_k = selected_step_k()?;
    let transposed = transposed_selected()?;
    let streamed = stream_selected()?;
    let (module, plan, gemm) = match (transposed, step_k) {
        (false, 16) => (
            "qwen4_moe_compact",
            "qwen4_moe_compact_plan",
            "qwen4_moe_compact_gemm",
        ),
        (false, 32) => (
            "qwen4_moe_compact_k32",
            "qwen4_moe_compact_plan_k32",
            "qwen4_moe_compact_gemm_k32",
        ),
        (true, 32) => (
            "qwen4_moe_compact_t_k32",
            "qwen4_moe_compact_t_plan_k32",
            "qwen4_moe_compact_t_gemm_k32",
        ),
        _ => unreachable!("validated compact layout and STEP_K"),
    };
    // Default-off V2 GEMM (same ABI/plan/workspace, pipelined kernel body).
    let v2 = transposed && step_k == 32 && v2_level()? > 0;
    let (plan_handle, gemm_handle, block, gateup, gateup_block, wide_set) = if v2 {
        let level = v2_level()?;
        let wide = matches!(level, 7 | 8);
        let module = if wide {
            "qwen4_moe_compact_t_k32_v7"
        } else {
            "qwen4_moe_compact_t_k32_v2"
        };
        let name = match level {
            7 | 8 => "qwen4_moe_compact_t_gemm_k32_v7",
            6 => "qwen4_moe_compact_t_gemm_k32_v6",
            5 => "qwen4_moe_compact_t_gemm_k32_v5",
            4 => "qwen4_moe_compact_t_gemm_k32_v4",
            2 | 3 => "qwen4_moe_compact_t_gemm_k32_v3",
            _ => "qwen4_moe_compact_t_gemm_k32_v2",
        };
        // 256-thread CTAs for the 64x128 (v3/v4/v6) and 128x128 (v7) tiles; v2 and v5 are 64x64/128.
        let block = if matches!(level, 2 | 3 | 4 | 6 | 7 | 8) {
            256
        } else {
            128
        };
        let gemm = gpu.kernel(module, name)?;
        let plan_name = if wide {
            "qwen4_moe_compact_t_plan_k32_v7"
        } else {
            "qwen4_moe_compact_t_plan_k32_v2"
        };
        let plan = gpu.kernel(module, plan_name)?;
        let gateup = if level == 8 {
            Some(gpu.kernel(module, "qwen4_moe_compact_t_gemm_gateup_k32_v7")?)
        } else if level == 3 {
            Some(gpu.kernel(
                "qwen4_moe_compact_t_k32_v2",
                "qwen4_moe_compact_t_gemm_gateup_k32",
            )?)
        } else {
            None
        };
        tracing::info!(
            "MOE_PREFILL_COMPACT_V2_SELECTED selector={V2_SELECTOR} kernel={name} fused_gateup={}",
            gateup.is_some()
        );
        let gateup_block = if level == 8 { 256 } else { 128 };
        (
            plan,
            gemm,
            block,
            gateup,
            gateup_block,
            Some(load_wide(gpu)?),
        )
    } else {
        (
            gpu.kernel(module, plan)?,
            gpu.kernel(module, gemm)?,
            128,
            None,
            128,
            None,
        )
    };
    let layer_idx = {
        static NEXT_LAYER_ORDINAL: std::sync::atomic::AtomicUsize =
            std::sync::atomic::AtomicUsize::new(0);
        let total = config.num_hidden_layers.max(1);
        NEXT_LAYER_ORDINAL.fetch_add(1, std::sync::atomic::Ordering::Relaxed) % total
    };
    let kernels = Some(Kernels {
        plan: plan_handle,
        gemm: gemm_handle,
        block,
        gateup,
        gateup_block,
        wide: wide_set,
        step_k,
        transposed,
        streamed,
        layer_idx,
    });
    validate_handles(kernels)?;
    // Pre-allocate capture-safe verify constants, identity indices, and v28 m32/m64 kernel handles
    // once at model load time so CUDA graph capture executes zero allocations or H2D uploads.
    if let Some(wide) = wide_set
        && w4a8_selected()?
    {
        if super::qwen4_moe_status::selected() {
            super::qwen4_moe_status::prewarm_verify_constants(gpu)?;
            let _ = super::qwen4_moe_status::prewarm_kernels(gpu);
        }
        if super::qwen4_shared_v7::selected() {
            super::qwen4_shared_v7::prewarm_identity(gpu)?;
        }
        if wide.w4a8_tag == "v28" {
            super::qwen4_compact_ops::prewarm_v28_verify_kernels(gpu);
        }
    }
    Ok(kernels)
}

/// Called by F8 before serial core effects, and defensively at the GEMM boundary.
pub(crate) fn validate_context(ctx: &ForwardContext, rows: usize) -> Result<()> {
    // Join any side-stream shared-expert W4A8 launch forked earlier in this layer
    // before routed GEMMs, async status flushes, or combine can access `attn_output`
    // or `expert_up_out`.
    super::qwen4_shared_v7::join_pending_shared_overlap(ctx.gpu)?;
    admit_request(ctx.config, rows, 0)?;
    if !selected()? {
        return Ok(());
    }
    ensure!(
        ctx.config.ep_world_size <= 4 && ctx.config.tp_world_size <= 4,
        "{SELECTOR} requires TP<=4 and EP<=4 execution"
    );
    if rows == 1 {
        return Ok(());
    }
    abi::contract(rows).map_err(anyhow::Error::msg)?;
    let sizes = ctx.buffers.sizes();
    let expanded = rows * 10;
    let w4a8_only =
        rows > 32 && w4a8_selected().unwrap_or(false) && !check_selected().unwrap_or(false);
    let slot_mxfp8_only = w4a8_only
        && slot_storage_selected()
            .map(|s| s == SlotStorage::Mxfp8)
            .unwrap_or(false)
        && !oracle_dump_present();
    let up_bytes = if w4a8_only {
        rows * 2640
    } else {
        expanded * 640 * 2
    };
    let down_bytes = if slot_mxfp8_only {
        expanded * 2640
    } else {
        expanded * 2560 * 2
    };
    ensure!(
        ctx.buffers.max_batch_tokens() >= rows
            && sizes.moe_worklist >= abi::ARENA_BYTES
            && sizes.moe_worklist_total >= 4
            && sizes.expert_gate_out >= expanded * 640 * 2
            && sizes.expert_up_out >= up_bytes
            && sizes.expert_down_out >= down_bytes
            && sizes.norm_output >= rows * 2560 * 2
            && sizes.attn_output >= rows * 2560 * 2
            && sizes.gate_logits >= (expanded * 3 + 513) * 4
            && sizes.scratch >= expanded * 8,
        "{SELECTOR} persistent workspace/status/expert arenas are undersized"
    );
    let w4a4 = w4a4_mode()?;
    if w4a4.is_enabled() {
        validate_w4a4_arenas(
            rows,
            w4a4,
            sizes.expert_up_out,
            sizes.expert_down_out,
            sizes.expert_gate_out,
        )?;
    }
    // ATLAS_QWEN4_PREFILL_MHC_GATEPRE: the shared quant writes one f32 gate
    // scalar per row into the arena; admit the rows before any launch.
    if crate::layers::qwen4_hyper::prefill_inject::selection()?.gatepre {
        ensure!(
            !ctx.buffers.moe_gate_scalars().is_null() && sizes.moe_gate_scalars >= rows * 4,
            "{SELECTOR} gatepre gate-scalar arena is undersized"
        );
    }
    let buffers = [
        (ctx.buffers.moe_worklist(), abi::ARENA_BYTES, 16),
        (ctx.buffers.moe_worklist_total(), 4, 4),
        (ctx.buffers.expert_gate_out(), expanded * 640 * 2, 16),
        (ctx.buffers.expert_up_out(), up_bytes, 16),
        (ctx.buffers.expert_down_out(), down_bytes, 16),
        (ctx.buffers.norm_output(), sizes.norm_output, 16),
        (ctx.buffers.attn_output(), sizes.attn_output, 16),
        (ctx.buffers.gate_logits(), sizes.gate_logits, 16),
        (ctx.buffers.scratch(), sizes.scratch, 16),
    ];
    let regions = buffers.map(|(ptr, bytes, align)| abi::region(ptr.0, bytes, align));
    for (i, region) in regions.iter().enumerate() {
        let region = (*region).map_err(anyhow::Error::msg)?;
        for other in &regions[..i] {
            ensure!(
                abi::disjoint(region, (*other).map_err(anyhow::Error::msg)?),
                "{SELECTOR} live arena alias"
            );
        }
    }
    Ok(())
}

#[cfg(test)]
mod mtp_head_tests {
    use super::is_qwen4_mtp_head;
    use atlas_core::config::{LayerType, ModelConfig};

    fn flash_next_trunk() -> ModelConfig {
        let mut config = ModelConfig::qwen3_next_80b_nvfp4();
        config.model_type = "qwen4_exp".into();
        config.hidden_size = 2_560;
        config.num_hidden_layers = 48;
        config.layer_types = (0..48)
            .map(|i| {
                if i % 4 == 3 {
                    LayerType::FullAttention
                } else {
                    LayerType::LinearAttention
                }
            })
            .collect();
        config
    }

    #[test]
    fn mtp_head_config_is_recognised() {
        let trunk = flash_next_trunk();
        let head = crate::weight_loader::qwen4_mtp_config(&trunk);
        assert!(is_qwen4_mtp_head(&head));
    }

    #[test]
    fn trunk_and_foreign_configs_are_not_mtp_heads() {
        let trunk = flash_next_trunk();
        assert!(!is_qwen4_mtp_head(&trunk));
        // A one-layer trunk that is not the MTP prefix keeps the canonical check.
        let mut one_layer = trunk.clone();
        one_layer.num_hidden_layers = 1;
        assert!(!is_qwen4_mtp_head(&one_layer));
        // An "mtp" prefix on a non-Qwen4 model is not exempt either.
        let mut other = crate::weight_loader::qwen4_mtp_config(&trunk);
        other.model_type = "qwen3_next".into();
        assert!(!is_qwen4_mtp_head(&other));
    }
}

#[cfg(test)]
mod tests {
    use super::{
        BatchedVerifyTcMoeEligibility, BatchedVerifyTcMoePolicy,
        DEFAULT_BATCHED_VERIFY_TC_MOE_MIN_ROWS, DEFAULT_PREFILL_GRID, SMALL_BATCH_GRID,
        SMALL_BATCH_THRESHOLD, SlotStorage, V29_M256_AUTO_MIN_ROWS, V29_M256_SELECTOR,
        V29M256Eligibility, V29M256Policy, V30_2CTA_AUTO_MIN_ROWS, V30_2CTA_SELECTOR,
        V31_WS_AUTO_MIN_ROWS, V31_WS_SELECTOR, V31WsEligibility, V31WsPolicy, V302CtaEligibility,
        V302CtaPolicy, dispatch_slot_storage, get_optimal_grid_dim, parse_batched_verify_tc_moe,
        parse_batched_verify_tc_moe_min_rows, parse_batched_verify_tc_moe_policy,
        parse_slot_storage, parse_v28_m64, parse_v28q_level, parse_v29_m256_policy,
        parse_v30_2cta_policy, parse_v31_ws_policy, request_active_from, slot_layout,
    };
    use crate::layers::moe::qwen4_plan_order_rule::{EXPERTS, tiles, v7_items, v8_items};

    #[test]
    fn batched_verify_tc_moe_selector_defaults_on_and_rejects_invalid_values() {
        assert_eq!(parse_batched_verify_tc_moe(None), Ok(true));
        assert_eq!(parse_batched_verify_tc_moe(Some("1")), Ok(true));
        assert_eq!(parse_batched_verify_tc_moe(Some("auto")), Ok(true));
        assert_eq!(parse_batched_verify_tc_moe(Some("0")), Ok(false));
        for bad in ["", "true", "false", "2", "AUTO", "on", "off"] {
            assert!(
                parse_batched_verify_tc_moe(Some(bad)).is_err(),
                "accepted invalid selector {bad:?}"
            );
        }
    }

    #[test]
    fn batched_verify_tc_moe_min_rows_defaults_to_4_and_enforces_lower_bound() {
        assert_eq!(DEFAULT_BATCHED_VERIFY_TC_MOE_MIN_ROWS, 4);
        assert_eq!(parse_batched_verify_tc_moe_min_rows(None), Ok(4));
        assert_eq!(parse_batched_verify_tc_moe_min_rows(Some("2")), Ok(2));
        assert_eq!(parse_batched_verify_tc_moe_min_rows(Some("4")), Ok(4));
        assert_eq!(parse_batched_verify_tc_moe_min_rows(Some("6")), Ok(6));
        assert_eq!(parse_batched_verify_tc_moe_min_rows(Some("9")), Ok(9));
        assert_eq!(parse_batched_verify_tc_moe_min_rows(Some("16")), Ok(16));
        assert_eq!(parse_batched_verify_tc_moe_min_rows(Some("20")), Ok(20));
        assert_eq!(parse_batched_verify_tc_moe_min_rows(Some("128")), Ok(128));
        for bad in ["", "0", "1", "-1", "16x", "auto"] {
            assert!(
                parse_batched_verify_tc_moe_min_rows(Some(bad)).is_err(),
                "accepted invalid min_rows {bad:?}"
            );
        }
        assert_eq!(
            parse_batched_verify_tc_moe_policy(None, None).unwrap(),
            BatchedVerifyTcMoePolicy {
                enabled: true,
                min_rows: 4,
            }
        );
        assert_eq!(
            parse_batched_verify_tc_moe_policy(Some("0"), Some("20")).unwrap(),
            BatchedVerifyTcMoePolicy {
                enabled: false,
                min_rows: 20,
            }
        );
        assert!(parse_batched_verify_tc_moe_policy(Some("bad"), None).is_err());
        assert!(parse_batched_verify_tc_moe_policy(Some("auto"), Some("1")).is_err());
    }

    #[test]
    fn batched_verify_tc_moe_admission_enforces_threshold_and_graph_capture_readiness() {
        let default_policy = BatchedVerifyTcMoePolicy {
            enabled: true,
            min_rows: DEFAULT_BATCHED_VERIFY_TC_MOE_MIN_ROWS,
        };
        let ready = |rows, graph_capture, capture_scratch_ready| BatchedVerifyTcMoeEligibility {
            rows,
            graph_capture,
            capture_scratch_ready,
            compact_active: true,
            w4a8_ready: true,
            shared_ready: true,
        };
        // rows=2..=3 remain on the fused 0-ULP exact worklist.
        for rows in [2, 3] {
            assert!(
                !default_policy.admits(ready(rows, false, true)),
                "rows={rows} must stay on exact worklist"
            );
            assert!(
                !default_policy.admits(ready(rows, true, true)),
                "rows={rows} must stay on exact worklist under graph capture"
            );
        }
        // rows >= 4 (C=2..16 batched verify) engage when eager OR during graph
        // capture once scratch is pre-allocated, and fail closed ONLY during
        // cold capture before scratch is pre-allocated.
        for rows in [4, 6, 8, 9, 10, 12, 14, 16, 20, 40, 64, 80, 128] {
            assert!(
                default_policy.admits(ready(rows, false, false)),
                "rows={rows} should admit eager TC MoE"
            );
            assert!(
                default_policy.admits(ready(rows, true, true)),
                "rows={rows} should admit warm graph-capture TC MoE"
            );
            assert!(
                !default_policy.admits(ready(rows, true, false)),
                "rows={rows} must refuse cold graph capture without pre-allocated scratch"
            );
        }
        // Selector disabled forces exact t8 for every batch size.
        let disabled = BatchedVerifyTcMoePolicy {
            enabled: false,
            min_rows: 6,
        };
        assert!(!disabled.admits(ready(80, false, true)));
        // Missing compact bundle, W4A8 kernels, or shared weights fail closed.
        assert!(!default_policy.admits(BatchedVerifyTcMoeEligibility {
            compact_active: false,
            ..ready(80, false, true)
        }));
        assert!(!default_policy.admits(BatchedVerifyTcMoeEligibility {
            w4a8_ready: false,
            ..ready(80, false, true)
        }));
        assert!(!default_policy.admits(BatchedVerifyTcMoeEligibility {
            shared_ready: false,
            ..ready(80, false, true)
        }));
    }

    #[test]
    fn v28_m64_selector_is_explicit_and_default_off() {
        assert!(!parse_v28_m64(None, None).unwrap());
        assert!(!parse_v28_m64(Some("0"), None).unwrap());
        assert!(!parse_v28_m64(None, Some("0")).unwrap());
        assert!(parse_v28_m64(Some("1"), None).unwrap());
        assert!(parse_v28_m64(None, Some("1")).unwrap());
        assert!(parse_v28_m64(Some("1"), Some("0")).unwrap());
        assert!(!parse_v28_m64(Some("0"), Some("1")).unwrap());
        for bad in ["", "true", "2", "m64"] {
            assert!(parse_v28_m64(Some(bad), None).is_err(), "accepted {bad:?}");
            assert!(parse_v28_m64(None, Some(bad)).is_err(), "accepted {bad:?}");
        }
    }

    #[test]
    fn v28q_selector_is_explicit_and_default_off() {
        assert_eq!(parse_v28q_level(None).unwrap(), 0);
        assert_eq!(parse_v28q_level(Some("0")).unwrap(), 0);
        assert_eq!(parse_v28q_level(Some("1")).unwrap(), 1);
        assert_eq!(parse_v28q_level(Some("2")).unwrap(), 2);
        assert_eq!(parse_v28q_level(Some("3")).unwrap(), 3);
        assert_eq!(parse_v28q_level(Some("4")).unwrap(), 4);
        for bad in ["", "true", "5", "v28q"] {
            assert!(parse_v28q_level(Some(bad)).is_err(), "accepted {bad:?}");
        }
    }

    #[test]
    fn slot_storage_selector_is_explicit_and_default_off() {
        assert_eq!(parse_slot_storage(None).unwrap(), SlotStorage::Bf16);
        assert_eq!(parse_slot_storage(Some("0")).unwrap(), SlotStorage::Bf16);
        assert_eq!(parse_slot_storage(Some("1")).unwrap(), SlotStorage::Mxfp8);
        assert_eq!(parse_slot_storage(Some("2")).unwrap(), SlotStorage::Mxfp8R2);
        for bad in ["", "true", "3", "r2"] {
            assert!(parse_slot_storage(Some(bad)).is_err(), "accepted {bad:?}");
        }
    }

    #[test]
    fn residual_two_layout_accounts_for_every_mx_block() {
        let pairs = 20usize;
        let plain = slot_layout(pairs, SlotStorage::Mxfp8).unwrap();
        assert_eq!(plain.scales_offset, Some(pairs * 2560));
        assert_eq!(plain.corrections_offset, None);
        assert_eq!(plain.used_bytes, pairs * (2560 + 80));

        let corrected = slot_layout(pairs, SlotStorage::Mxfp8R2).unwrap();
        assert_eq!(corrected.scales_offset, Some(pairs * 2560));
        assert_eq!(corrected.corrections_offset, Some(pairs * (2560 + 80)));
        assert_eq!(corrected.used_bytes, pairs * (2560 + 80 + 80 * 4));
        assert_eq!(corrected.used_bytes / pairs, 2960);
        assert!(corrected.used_bytes <= pairs * 2560 * 2);
    }

    #[test]
    fn residual_two_is_prefill_scoped_and_verify_falls_back_to_bf16() {
        assert_eq!(
            dispatch_slot_storage(SlotStorage::Mxfp8R2, false),
            SlotStorage::Bf16
        );
        assert_eq!(
            dispatch_slot_storage(SlotStorage::Mxfp8R2, true),
            SlotStorage::Mxfp8R2
        );
        assert_eq!(
            dispatch_slot_storage(SlotStorage::Mxfp8, false),
            SlotStorage::Mxfp8
        );
        assert_eq!(
            dispatch_slot_storage(SlotStorage::Bf16, false),
            SlotStorage::Bf16
        );
    }

    #[test]
    fn request_suppression_disables_loaded_compact_kernels() {
        assert!(request_active_from(true, false));
        assert!(!request_active_from(true, true));
        assert!(!request_active_from(false, false));
        assert!(!request_active_from(false, true));
    }

    #[test]
    fn optimal_grid_dim_matches_small_batch_threshold() {
        assert_eq!(SMALL_BATCH_THRESHOLD, 16);
        for m in [0, 1, 2, 5, 16] {
            assert_eq!(get_optimal_grid_dim(m), SMALL_BATCH_GRID, "m={m}");
        }
        for m in [17, 32, 64, 256, 8192] {
            assert_eq!(get_optimal_grid_dim(m), DEFAULT_PREFILL_GRID, "m={m}");
        }
        for sm in [48u32, 110, 170, 188] {
            assert_eq!(super::get_optimal_grid_dim_for_sm(1, sm), sm);
            assert_eq!(super::get_optimal_grid_dim_for_sm(16, sm), sm);
            assert_eq!(
                super::get_optimal_grid_dim_for_sm(17, sm),
                DEFAULT_PREFILL_GRID
            );
        }
        assert_eq!(
            super::get_optimal_grid_dim_for_sm(8, 8192),
            DEFAULT_PREFILL_GRID
        );
    }

    /// Models `v28_plan_impl<INTER>`'s 3-level 16-warp hierarchical prefix scan
    /// and verifies bit-exact equivalence against both `v7_items` and `v8_items` (`inter`).
    #[test]
    fn v28_hierarchical_16_warp_prefix_scan_matches_v7_and_v8_inter() {
        fn run_v28_scan(
            expert_rows: &[u32; EXPERTS],
            inter: bool,
        ) -> (u32, u32, u32, Vec<(u32, u32)>) {
            let s_tiles = tiles(expert_rows);
            let mut incl_tiles = [0u32; EXPERTS];
            let mut incl_big = [0u32; EXPERTS];
            let mut incl_single = [0u32; EXPERTS];
            let mut s_warp_tiles = [0u32; 16];
            let mut s_warp_big = [0u32; 16];
            let mut s_warp_single = [0u32; 16];
            let mut s_warp_active = [0u32; 16];
            let mut s_warp_max_rows = [0u32; 16];

            for w in 0..16 {
                let (mut acc_t, mut acc_b, mut acc_s, mut acc_a, mut max_r) = (0, 0, 0, 0, 0);
                for lane in 0..32 {
                    let e = w * 32 + lane;
                    let r = expert_rows[e];
                    let t = s_tiles[e];
                    acc_t += t;
                    acc_b += if t > 1 { t } else { 0 };
                    acc_s += u32::from(t == 1);
                    acc_a += u32::from(r > 0);
                    max_r = max_r.max(r);
                    incl_tiles[e] = acc_t;
                    incl_big[e] = acc_b;
                    incl_single[e] = acc_s;
                }
                s_warp_tiles[w] = acc_t;
                s_warp_big[w] = acc_b;
                s_warp_single[w] = acc_s;
                s_warp_active[w] = acc_a;
                s_warp_max_rows[w] = max_r;
            }

            let mut s_warp_base_tiles = [0u32; 16];
            let mut s_warp_base_big = [0u32; 16];
            let mut s_warp_base_single = [0u32; 16];
            let (mut total_t, mut total_b, mut total_s, mut total_a, mut global_max) =
                (0, 0, 0, 0, 0);
            for w in 0..16 {
                s_warp_base_tiles[w] = total_t;
                s_warp_base_big[w] = total_b;
                s_warp_base_single[w] = total_s;
                total_t += s_warp_tiles[w];
                total_b += s_warp_big[w];
                total_s += s_warp_single[w];
                total_a += s_warp_active[w];
                global_max = global_max.max(s_warp_max_rows[w]);
            }

            let mut out = vec![(u32::MAX, u32::MAX); total_t as usize];
            for e in 0..EXPERTS {
                let w = e >> 5;
                let t = s_tiles[e];
                let big = if t > 1 { t } else { 0 };
                let single = u32::from(t == 1);
                let prefix = s_warp_base_tiles[w] + (incl_tiles[e] - t);
                let class = if t > 1 {
                    s_warp_base_big[w] + (incl_big[e] - big)
                } else {
                    s_warp_base_single[w] + (incl_single[e] - single)
                };
                if inter {
                    if total_b == 0 || total_s == 0 {
                        for ti in 0..t {
                            out[(prefix + ti) as usize] = (e as u32, ti);
                        }
                    } else if t > 1 {
                        for ti in 0..t {
                            let k = class + ti;
                            out[(k + (k * total_s) / total_b) as usize] = (e as u32, ti);
                        }
                    } else if t == 1 {
                        let j = class;
                        out[(((j + 1) * total_b).div_ceil(total_s) + j) as usize] = (e as u32, 0);
                    }
                } else {
                    for ti in 0..t {
                        out[(prefix + ti) as usize] = (e as u32, ti);
                    }
                }
            }
            (total_t, total_a, global_max, out)
        }

        for seed in 0..32u64 {
            let mut rng = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
            let mut next = || {
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                rng
            };
            let mut rows = [0u32; EXPERTS];
            for _ in 0..2560 {
                let u = (next() % 1_000_003) as f64 / 1_000_003.0;
                let e = ((u * u * u) * EXPERTS as f64) as usize % EXPERTS;
                rows[(e + seed as usize * 37) % EXPERTS] += 1;
            }
            let ref_tiles = tiles(&rows);
            let expected_active = rows.iter().filter(|&&r| r > 0).count() as u32;
            let expected_max = rows.iter().copied().max().unwrap_or(0);

            let (wt, wa, wm, items_v7) = run_v28_scan(&rows, false);
            assert_eq!(items_v7, v7_items(&ref_tiles), "v7 mismatch on seed {seed}");
            assert_eq!(wt as usize, items_v7.len());
            assert_eq!(wa, expected_active);
            assert_eq!(wm, expected_max);

            let (_, _, _, items_inter) = run_v28_scan(&rows, true);
            assert_eq!(
                items_inter,
                v8_items(&ref_tiles),
                "inter mismatch on seed {seed}"
            );
        }
    }

    #[test]
    fn v29_m256_selector_defaults_to_disabled_and_rejects_invalid_values() {
        assert_eq!(V29_M256_SELECTOR, "ATLAS_QWEN4_PREFILL_MOE_V29_M256");
        assert_eq!(V29_M256_AUTO_MIN_ROWS, 2048);
        assert_eq!(parse_v29_m256_policy(None), Ok(V29M256Policy::Disabled));
        assert_eq!(parse_v29_m256_policy(Some("auto")), Ok(V29M256Policy::Auto));
        assert_eq!(
            parse_v29_m256_policy(Some("0")),
            Ok(V29M256Policy::Disabled)
        );
        assert_eq!(parse_v29_m256_policy(Some("1")), Ok(V29M256Policy::Force));
        for bad in ["", "true", "false", "2", "AUTO", "on", "off", "m256"] {
            assert!(
                parse_v29_m256_policy(Some(bad)).is_err(),
                "accepted invalid selector {bad:?}"
            );
        }
    }

    #[test]
    fn v29_m256_admission_enforces_row_gate_and_prerequisites() {
        let ready = |rows| V29M256Eligibility {
            rows,
            w4a8_is_v28: true,
            v28_m64_forced: false,
            has_corrections: false,
            l2_windowed: false,
            kernels_loaded: true,
        };
        // Auto engages only for rows >= 2048.
        for rows in [2, 16, 64, 128, 512, 1024, 2047] {
            assert!(
                !V29M256Policy::Auto.admits(ready(rows)),
                "auto must not engage for rows={rows}"
            );
            assert!(
                V29M256Policy::Force.admits(ready(rows)),
                "force should engage for rows={rows}"
            );
            assert!(!V29M256Policy::Disabled.admits(ready(rows)));
        }
        for rows in [2048, 4096, 8192, 16384] {
            assert!(
                V29M256Policy::Auto.admits(ready(rows)),
                "auto should engage for rows={rows}"
            );
            assert!(V29M256Policy::Force.admits(ready(rows)));
            assert!(!V29M256Policy::Disabled.admits(ready(rows)));
        }
        // Any missing prerequisite blocks both Auto and Force.
        for policy in [V29M256Policy::Auto, V29M256Policy::Force] {
            assert!(!policy.admits(V29M256Eligibility {
                w4a8_is_v28: false,
                ..ready(4096)
            }));
            assert!(!policy.admits(V29M256Eligibility {
                v28_m64_forced: true,
                ..ready(4096)
            }));
            assert!(!policy.admits(V29M256Eligibility {
                has_corrections: true,
                ..ready(4096)
            }));
            assert!(!policy.admits(V29M256Eligibility {
                l2_windowed: true,
                ..ready(4096)
            }));
            assert!(!policy.admits(V29M256Eligibility {
                kernels_loaded: false,
                ..ready(4096)
            }));
        }
    }

    #[test]
    fn v30_2cta_selector_defaults_to_disabled_and_rejects_invalid_values() {
        assert_eq!(V30_2CTA_SELECTOR, "ATLAS_QWEN4_PREFILL_MOE_V30_2CTA");
        assert_eq!(V30_2CTA_AUTO_MIN_ROWS, 2048);
        assert_eq!(parse_v30_2cta_policy(None), Ok(V302CtaPolicy::Disabled));
        assert_eq!(parse_v30_2cta_policy(Some("auto")), Ok(V302CtaPolicy::Auto));
        assert_eq!(
            parse_v30_2cta_policy(Some("0")),
            Ok(V302CtaPolicy::Disabled)
        );
        assert_eq!(parse_v30_2cta_policy(Some("1")), Ok(V302CtaPolicy::Force));
        for bad in ["", "true", "false", "2", "AUTO", "on", "off", "v30"] {
            assert!(
                parse_v30_2cta_policy(Some(bad)).is_err(),
                "accepted invalid selector {bad:?}"
            );
        }
    }

    #[test]
    fn v30_2cta_admission_enforces_row_gate_and_prerequisites() {
        let ready = |rows| V302CtaEligibility {
            rows,
            w4a8_is_v28: true,
            v28_m64_forced: false,
            has_corrections: false,
            l2_windowed: false,
            v29_m256_active: false,
            v28q_level: 2,
            kernels_loaded: true,
        };
        // Auto engages only for rows >= 2048; Force engages for any rows; Disabled never engages.
        for rows in [2, 16, 64, 128, 512, 1024, 2047] {
            assert!(
                !V302CtaPolicy::Auto.admits(ready(rows)),
                "auto must not engage for rows={rows}"
            );
            assert!(
                V302CtaPolicy::Force.admits(ready(rows)),
                "force should engage for rows={rows}"
            );
            assert!(!V302CtaPolicy::Disabled.admits(ready(rows)));
        }
        for rows in [2048, 4096, 8192, 16384] {
            assert!(
                V302CtaPolicy::Auto.admits(ready(rows)),
                "auto should engage for rows={rows}"
            );
            assert!(V302CtaPolicy::Force.admits(ready(rows)));
            assert!(!V302CtaPolicy::Disabled.admits(ready(rows)));
        }
        // Auto admits v28q_level=0 and 2, while preserving explicit v28q_level=1/3/4 ablations.
        assert!(V302CtaPolicy::Auto.admits(V302CtaEligibility {
            v28q_level: 0,
            ..ready(4096)
        }));
        for ablation in [1u32, 3, 4] {
            assert!(!V302CtaPolicy::Auto.admits(V302CtaEligibility {
                v28q_level: ablation,
                ..ready(4096)
            }));
            assert!(V302CtaPolicy::Force.admits(V302CtaEligibility {
                v28q_level: ablation,
                ..ready(4096)
            }));
        }
        // Any missing prerequisite blocks both Auto and Force.
        for policy in [V302CtaPolicy::Auto, V302CtaPolicy::Force] {
            assert!(!policy.admits(V302CtaEligibility {
                w4a8_is_v28: false,
                ..ready(4096)
            }));
            assert!(!policy.admits(V302CtaEligibility {
                v28_m64_forced: true,
                ..ready(4096)
            }));
            assert!(!policy.admits(V302CtaEligibility {
                has_corrections: true,
                ..ready(4096)
            }));
            assert!(!policy.admits(V302CtaEligibility {
                l2_windowed: true,
                ..ready(4096)
            }));
            assert!(!policy.admits(V302CtaEligibility {
                v29_m256_active: true,
                ..ready(4096)
            }));
            assert!(!policy.admits(V302CtaEligibility {
                kernels_loaded: false,
                ..ready(4096)
            }));
        }
    }

    #[test]
    fn v31_ws_selector_defaults_to_auto_and_rejects_invalid_values() {
        assert_eq!(V31_WS_SELECTOR, "ATLAS_QWEN4_PREFILL_MOE_V31_WS");
        assert_eq!(V31_WS_AUTO_MIN_ROWS, 2048);
        assert_eq!(parse_v31_ws_policy(None), Ok(V31WsPolicy::Auto));
        assert_eq!(parse_v31_ws_policy(Some("auto")), Ok(V31WsPolicy::Auto));
        assert_eq!(parse_v31_ws_policy(Some("0")), Ok(V31WsPolicy::Disabled));
        assert_eq!(parse_v31_ws_policy(Some("1")), Ok(V31WsPolicy::Force));
        for bad in ["", "true", "false", "2", "AUTO", "on", "off", "v31"] {
            assert!(
                parse_v31_ws_policy(Some(bad)).is_err(),
                "accepted invalid selector {bad:?}"
            );
        }
    }

    #[test]
    fn v31_ws_admission_enforces_row_gate_and_prerequisites() {
        let ready = |rows| V31WsEligibility {
            rows,
            w4a8_is_v28: true,
            v28_m64_forced: false,
            has_corrections: false,
            l2_windowed: false,
            v29_m256_active: false,
            v30_2cta_active: false,
            v28q_level: 2,
            kernels_loaded: true,
        };
        // Auto engages only for rows >= 2048; Force engages for any rows; Disabled never engages.
        for rows in [2, 16, 64, 128, 512, 1024, 2047] {
            assert!(
                !V31WsPolicy::Auto.admits(ready(rows)),
                "auto must not engage for rows={rows}"
            );
            assert!(
                V31WsPolicy::Force.admits(ready(rows)),
                "force should engage for rows={rows}"
            );
            assert!(!V31WsPolicy::Disabled.admits(ready(rows)));
        }
        for rows in [2048, 4096, 8192, 16384] {
            assert!(
                V31WsPolicy::Auto.admits(ready(rows)),
                "auto should engage for rows={rows}"
            );
            assert!(V31WsPolicy::Force.admits(ready(rows)));
            assert!(!V31WsPolicy::Disabled.admits(ready(rows)));
        }
        // Auto admits v28q_level=0 and 2, while preserving explicit v28q_level=1/3/4 ablations.
        assert!(V31WsPolicy::Auto.admits(V31WsEligibility {
            v28q_level: 0,
            ..ready(4096)
        }));
        for ablation in [1u32, 3, 4] {
            assert!(!V31WsPolicy::Auto.admits(V31WsEligibility {
                v28q_level: ablation,
                ..ready(4096)
            }));
            assert!(V31WsPolicy::Force.admits(V31WsEligibility {
                v28q_level: ablation,
                ..ready(4096)
            }));
        }
        // Any missing prerequisite blocks both Auto and Force.
        for policy in [V31WsPolicy::Auto, V31WsPolicy::Force] {
            assert!(!policy.admits(V31WsEligibility {
                w4a8_is_v28: false,
                ..ready(4096)
            }));
            assert!(!policy.admits(V31WsEligibility {
                v28_m64_forced: true,
                ..ready(4096)
            }));
            assert!(!policy.admits(V31WsEligibility {
                has_corrections: true,
                ..ready(4096)
            }));
            assert!(!policy.admits(V31WsEligibility {
                l2_windowed: true,
                ..ready(4096)
            }));
            assert!(!policy.admits(V31WsEligibility {
                v29_m256_active: true,
                ..ready(4096)
            }));
            assert!(!policy.admits(V31WsEligibility {
                v30_2cta_active: true,
                ..ready(4096)
            }));
            assert!(!policy.admits(V31WsEligibility {
                kernels_loaded: false,
                ..ready(4096)
            }));
        }
    }

    #[test]
    fn v31_ws_auto_lowers_row_threshold_on_high_sm_gpus() {
        assert_eq!(super::v31_ws_auto_min_rows_for_sm(48), V31_WS_AUTO_MIN_ROWS);
        assert_eq!(super::v31_ws_auto_min_rows_for_sm(64), V31_WS_AUTO_MIN_ROWS);
        assert_eq!(
            super::v31_ws_auto_min_rows_for_sm(65),
            super::V31_WS_AUTO_MIN_ROWS_HIGH_SM
        );
        assert_eq!(super::v31_ws_auto_min_rows_for_sm(188), 33);

        let ready = |rows| V31WsEligibility {
            rows,
            w4a8_is_v28: true,
            v28_m64_forced: false,
            has_corrections: false,
            l2_windowed: false,
            v29_m256_active: false,
            v30_2cta_active: false,
            v28q_level: 2,
            kernels_loaded: true,
        };

        // On SM120 (188 SMs), Auto stays off below 33 rows (leaving C=1..4 verify
        // frames on v33_thin_sk) and engages at >= 33 rows (C=8..64 verify and prefill),
        // including v28q_level == 2 (flux-nextgen-max-prefill-decode.env).
        for rows in [2, 16, 32] {
            assert!(
                !V31WsPolicy::Auto.admits_for_sm(ready(rows), 188),
                "sm=188 rows={rows} must stay below threshold"
            );
        }
        for rows in [33, 48, 64, 96, 120, 128, 512, 1024, 2047, 2048, 5800, 16384] {
            assert!(
                V31WsPolicy::Auto.admits_for_sm(ready(rows), 188),
                "sm=188 rows={rows} should auto-engage v31_ws with v28q_level=2"
            );
            assert!(
                V31WsPolicy::Auto.admits_for_sm(
                    V31WsEligibility {
                        v28q_level: 0,
                        ..ready(rows)
                    },
                    188
                ),
                "sm=188 rows={rows} should auto-engage v31_ws with v28q_level=0"
            );
        }
        // GB10 (48 SMs) retains the 2048-row gate unchanged.
        for rows in [512, 1024, 2047] {
            assert!(
                !V31WsPolicy::Auto.admits_for_sm(ready(rows), 48),
                "sm=48 rows={rows} must keep 2048-row gate"
            );
        }
        assert!(V31WsPolicy::Auto.admits_for_sm(ready(2048), 48));
        // Disabled stays off and explicit v28q ablations (1/3/4) remain respected on SM120.
        assert!(!V31WsPolicy::Disabled.admits_for_sm(ready(5800), 188));
        for ablation in [1u32, 3, 4] {
            assert!(!V31WsPolicy::Auto.admits_for_sm(
                V31WsEligibility {
                    v28q_level: ablation,
                    ..ready(5800)
                },
                188
            ));
        }
    }

    #[test]
    fn w4a4_v32_defaults_off_and_preserves_gb10_baseline() {
        use super::qwen4_moe_w4a4::W4a4Mode;
        assert_eq!(super::W4A4_SELECTOR, "ATLAS_QWEN4_PREFILL_MOE_W4A4");
        assert_eq!(
            super::qwen4_moe_w4a4::parse_w4a4_mode(None),
            Ok(W4a4Mode::Off)
        );
        assert!(!W4a4Mode::Off.is_enabled());
        assert!(W4a4Mode::GateUp.is_enabled());
        assert!(!W4a4Mode::GateUp.uses_w4a4_down());
        assert!(W4a4Mode::All.is_enabled());
        assert!(W4a4Mode::All.uses_w4a4_down());

        // Fail-closed: GateUp requires loaded V31_WS DOWN kernels.
        assert!(super::validate_w4a4_v31_compat(W4a4Mode::Off, false).is_ok());
        assert!(super::validate_w4a4_v31_compat(W4a4Mode::All, false).is_ok());
        assert!(super::validate_w4a4_v31_compat(W4a4Mode::GateUp, true).is_ok());
        assert!(super::validate_w4a4_v31_compat(W4a4Mode::GateUp, false).is_err());

        // Scratch validation checks the smaller of expert_up_out and expert_down_out
        // (since shared-expert overlap redirects NVFP4 input into expert_down_out).
        let rows = 512usize;
        let expanded = rows * 10;
        assert!(
            super::validate_w4a4_arenas(
                rows,
                W4a4Mode::All,
                rows * 2_640,
                expanded * 2_640,
                expanded * 1_280,
            )
            .is_ok()
        );
        assert!(
            super::validate_w4a4_arenas(
                rows,
                W4a4Mode::All,
                rows * 2_640,
                rows * 1_443,
                expanded * 1_280,
            )
            .is_err()
        );
        assert!(
            super::validate_w4a4_arenas(
                rows,
                W4a4Mode::All,
                rows * 1_443,
                expanded * 2_640,
                expanded * 1_280,
            )
            .is_err()
        );
    }
}
