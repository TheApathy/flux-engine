// SPDX-License-Identifier: AGPL-3.0-only

//! Private original-layout gate/up/down dispatch; checked status precedes reduction.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{Context, Result, bail, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend, HostToDeviceCopy, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

#[path = "qwen4_moe_thin_sk.rs"]
pub(super) mod qwen4_moe_thin_sk;

use super::{
    ExpertPtrTable, MoeLayer, RoutedScaleLayout, ops, qwen4_compact_contract as abi,
    qwen4_prefill_compact, qwen4_xq_quant,
};
use crate::layer::ForwardContext;

static K32_RECEIPT_LOGGED: AtomicBool = AtomicBool::new(false);
static W4A8_RECEIPT_LOGGED: AtomicBool = AtomicBool::new(false);
static SLOT_MXFP8_RECEIPT_LOGGED: AtomicBool = AtomicBool::new(false);
static SLOT_MXFP8_R2_RECEIPT_LOGGED: AtomicBool = AtomicBool::new(false);
static XQ_REUSE_RECEIPT_LOGGED: AtomicBool = AtomicBool::new(false);
static SLOT_BF16_FUSED_RECEIPT_LOGGED: AtomicBool = AtomicBool::new(false);
static XQ_GATEPRE_RECEIPT_LOGGED: AtomicBool = AtomicBool::new(false);
static L2_WINDOW_RECEIPT_LOGGED: AtomicBool = AtomicBool::new(false);
static TS4E_PREFILL_RECEIPT_LOGGED: AtomicBool = AtomicBool::new(false);
static V28_M16_RECEIPT_LOGGED: AtomicBool = AtomicBool::new(false);
static V28_M32_RECEIPT_LOGGED: AtomicBool = AtomicBool::new(false);
static V29_M256_RECEIPT_LOGGED: AtomicBool = AtomicBool::new(false);
static V30_2CTA_RECEIPT_LOGGED: AtomicBool = AtomicBool::new(false);
static V31_WS_RECEIPT_LOGGED: AtomicBool = AtomicBool::new(false);
static UNPERMUTE_BLEND_W4_RECEIPT_LOGGED: AtomicBool = AtomicBool::new(false);

/// Standard dynamic SMEM footprint for flat-scale `v28` / `v28_m64` kernels (`89,088 B`).
const V28_FLAT_SMEM_BYTES: u32 = 89_088;
/// Dynamic SMEM footprint for Atlas-TS4E-128 `v28_twarp` / `v28_twarp_m64` gate/up kernels
/// (`89,088 B` base + `256 B` `sBase` + `768 B` `sExc` + `640 B` `sBktBeg` + `320 B` `sBktCnt`
/// = `91,072 B`, within GB10's `101,376 B` dynamic SMEM cap).
const V28_TS4E_GATEUP_SMEM_BYTES: u32 = 91_072;
/// Dynamic SMEM footprint for Atlas-TS4E-128 `v28_twarp` / `v28_twarp_m64` down kernels
/// (`89,088 B` base + `256 B` `sBase` + `768 B` `sExc` + `160 B` `sBktBeg` + `80 B` `sBktCnt`
/// = `90,352 B`, within GB10's `101,376 B` dynamic SMEM cap).
const V28_TS4E_DOWN_SMEM_BYTES: u32 = 90_352;
/// Thread-block width of the 4-CTA/SM `v28_m16` (`TILE_M=16, BC=128`) kernels (4 warps).
pub(super) const V28_M16_BLOCK: u32 = 128;
/// Dynamic SMEM footprint for flat-scale `v28_m16` gate/up (`18.875 KiB <= 24 KiB` -> 4 CTAs/SM on GB10).
pub(super) const V28_M16_FLAT_GATEUP_SMEM_BYTES: u32 = 19_328;
/// Dynamic SMEM footprint for flat-scale `v28_m16` down (`18.625 KiB <= 24 KiB` -> 4 CTAs/SM on GB10).
pub(super) const V28_M16_FLAT_DOWN_SMEM_BYTES: u32 = 19_072;
/// Dynamic SMEM footprint for Atlas-TS4E-128 `v28_m16_ts4e` gate/up (`20.6875 KiB <= 24 KiB` -> 4 CTAs/SM on GB10).
pub(super) const V28_M16_TS4E_GATEUP_SMEM_BYTES: u32 = 21_184;
/// Dynamic SMEM footprint for Atlas-TS4E-128 `v28_m16_ts4e` down (`19.25 KiB <= 24 KiB` -> 4 CTAs/SM on GB10).
pub(super) const V28_M16_TS4E_DOWN_SMEM_BYTES: u32 = 19_712;
/// Maximum batch rows for automatic 4-CTA/SM `v28_m16` batched-decode dispatch (`rows <= 96`).
/// Above 96 rows (`rows = 97..=128`), 8-warp 2-CTA/SM `v28_m32` outperforms 4-warp `v28_m16`
/// on both GB10 and 188-SM SM120.
pub(super) const V28_M16_DECODE_AUTO_MAX_ROWS: usize = 96;
/// SM count above which 4-CTA/SM `v28_m16` auto-enables for `rows <= 96` (`C=1..12` DFlash2, `C=1..48` MTP).
const V28_M16_HIGH_SM_THRESHOLD: u32 = 64;
/// Env toggle for 4-CTA/SM `v28_m16` batched-decode dispatch (`1` / `auto` = enable;
/// `0` = disable; unset = auto-ON for `active_sm_count() > 64`, OFF on `<= 64` SMs
/// where 8-warp 2-CTA/SM `v28_m32` handles `rows <= 256`).
pub(crate) const V28_M16_DECODE_SELECTOR: &str = "ATLAS_QWEN4_MOE_DECODE_M16";
/// 1-CTA/SM static wave floor (clamped to `abi::GRID` = 4096).
/// Returns 48 on GB10 and scales dynamically to 110–188 CTAs on SM120 / SM103.
#[inline]
#[must_use]
fn sm_grid_1x(sm_count: u32) -> usize {
    (sm_count.max(1) as usize).min(abi::GRID as usize)
}

/// 2-CTA/SM static wave floor (clamped to `abi::GRID` = 4096).
/// Returns 96 on GB10 and scales dynamically to 220–376 CTAs on SM120 / SM103.
#[inline]
#[must_use]
fn sm_grid_2x(sm_count: u32) -> usize {
    (sm_count.max(1) as usize)
        .saturating_mul(2)
        .min(abi::GRID as usize)
}

/// 4-CTA/SM static wave floor (clamped to `abi::GRID` = 4096).
/// Returns 192 on GB10 and scales dynamically to 440–752 CTAs on SM120 / SM103.
#[inline]
#[must_use]
fn sm_grid_4x(sm_count: u32) -> usize {
    (sm_count.max(1) as usize)
        .saturating_mul(4)
        .min(abi::GRID as usize)
}

#[inline]
#[must_use]
fn v28_m16_min_grid_for_sm(sm_count: u32) -> usize {
    sm_grid_4x(sm_count)
}

#[inline]
#[must_use]
fn v28_m32_min_grid_for_sm(sm_count: u32) -> usize {
    sm_grid_2x(sm_count)
}

#[inline]
#[must_use]
fn v30_2cta_min_grid_for_sm(sm_count: u32) -> usize {
    sm_grid_2x(sm_count)
}

#[inline]
#[must_use]
fn v31_ws_min_grid_for_sm(sm_count: u32) -> usize {
    sm_grid_1x(sm_count)
}

#[inline]
#[must_use]
fn v31_ws_default_max_grid_for_sm(sm_count: u32) -> usize {
    v31_ws_min_grid_for_sm(sm_count)
}

#[inline]
#[must_use]
fn active_sm_grid_1x() -> usize {
    sm_grid_1x(atlas_core::device::active_sm_count())
}

/// Thread-block width of the 2-CTA/SM `v28_m32` (`TILE_M=32, BC=128`) kernels (8 warps).
pub(super) const V28_M32_BLOCK: u32 = 256;
/// Dynamic SMEM footprint for flat-scale `v28_m32` (`38.75 KiB <= 48 KiB` -> 2 CTAs/SM on GB10;
/// covers `39,680 B` GATEUP with SMEM `lut[256]` + `sMax[32][4]` and `39,168 B` DOWN).
const V28_M32_FLAT_SMEM_BYTES: u32 = 39_680;
/// Dynamic SMEM footprint for Atlas-TS4E-128 `v28_m32_ts4e` gate/up (`40.56 KiB <= 48 KiB` -> 2 CTAs/SM).
const V28_M32_TS4E_GATEUP_SMEM_BYTES: u32 = 41_536;
/// Dynamic SMEM footprint for Atlas-TS4E-128 `v28_m32_ts4e` down (`38.88 KiB <= 48 KiB` -> 2 CTAs/SM).
const V28_M32_TS4E_DOWN_SMEM_BYTES: u32 = 39_808;
/// Maximum batch rows for automatic `v28_m32` (`BC=128`, 2 CTAs/SM) gate/up dispatch.
/// Covers batched verify frames up to `rows <= 256` (e.g. `C=16..32` DFlash2 / `C=32..64` MTP),
/// keeping 2 CTAs/SM and avoiding 1-CTA/SM `v28_m64`/`Twarp` occupancy loss and padded MMA work.
pub(super) const V28_M32_GATEUP_AUTO_MAX_ROWS: usize = 256;
/// Maximum batch rows for automatic `v28_m32` (`BC=128`, 2 CTAs/SM) down dispatch (`KS=10`),
/// where 2 CTAs/SM hide Warp-0 TS4E decode stalls across verify batches `rows = 6..=256`.
pub(super) const V28_M32_DOWN_AUTO_MAX_ROWS: usize = 256;
/// Optional override to disable 2-CTA/SM `v28_m32` dispatch (`0` = fall back to `m64`/`twarp`).
pub(crate) const V28_M32_SELECTOR: &str = "ATLAS_QWEN4_MOE_V28_M32";
/// Optional opt-out for 2-CTA/SM `v28_m32_ts4e` on short prefill (`rows <= 256`, TS4E active);
/// default ON unless `"0"`.
pub(crate) const V28_M32_SHORT_PREFILL_SELECTOR: &str = "ATLAS_QWEN4_MOE_SHORT_PREFILL_M32";
/// Optional cap for static `v28_m32` grid width (default `4096` = W18 dynamic static grid
/// `(max_items * 10|20).clamp(96, 4096)`, eliminating wave-tail imbalance across `C=2..16`).
pub(crate) const V28_M32_MAX_GRID_SELECTOR: &str = "ATLAS_QWEN4_MOE_M32_MAX_GRID";
/// Default static grid cap for 2-CTA/SM `v28_m32` (full `abi::GRID = 4096` W18 width).
const V28_M32_DEFAULT_MAX_GRID: usize = abi::GRID as usize;
/// Maximum batch rows for automatic `v28_m64` (2x8 warp N-split for `valid_rows <= 64`)
/// dispatch on small batches (batched verify `rows=6..=256` and small prefill chunks).
pub(super) const V28_M64_AUTO_MAX_ROWS: usize = 256;
/// Thread-block width of the Wave-17 Path A `v29_m256` (`TILE_M=256, BC=128`) kernels (16 warps, 8x2).
pub(super) const V29_M256_BLOCK: u32 = 512;
/// Dynamic SMEM footprint for flat-scale `v29_m256` kernels (`85,056 B <= 101,376 B`).
pub(super) const V29_M256_FLAT_SMEM_BYTES: u32 = 85_056;
/// Dynamic SMEM footprint for Atlas-TS4E-128 `v29_m256` gate/up kernel (`86,912 B <= 101,376 B`).
pub(super) const V29_M256_TS4E_GATEUP_SMEM_BYTES: u32 = 86_912;
/// Dynamic SMEM footprint for Atlas-TS4E-128 `v29_m256` down kernels (`85,696 B <= 101,376 B`).
pub(super) const V29_M256_TS4E_DOWN_SMEM_BYTES: u32 = 85_696;
/// Rows covered by one `v29_m256` super-item (`TILE_M = 256`).
const V29_M256_TILE_M: usize = 256;
/// Thread-block width of the Wave-21 2-CTA/SM `v30_2cta` (`TILE_M=128, BC=128`) kernels (8 warps, 4x2).
pub(super) const V30_2CTA_BLOCK: u32 = 256;
/// Dynamic SMEM footprint for flat-scale `v30_2cta` gate/up (`46,144 B <= 48,128 B <= 49,152 B` -> 2 CTAs/SM).
pub(super) const V30_2CTA_FLAT_GATEUP_SMEM_BYTES: u32 = 46_144;
/// Dynamic SMEM footprint for flat-scale `v30_2cta` down (`45,632 B <= 48,128 B <= 49,152 B` -> 2 CTAs/SM).
pub(super) const V30_2CTA_FLAT_DOWN_SMEM_BYTES: u32 = 45_632;
/// Dynamic SMEM footprint for Atlas-TS4E-128 `v30_2cta_ts4e` gate/up (`47,744 B <= 48,128 B <= 49,152 B` -> 2 CTAs/SM).
pub(super) const V30_2CTA_TS4E_GATEUP_SMEM_BYTES: u32 = 47_744;
/// Dynamic SMEM footprint for Atlas-TS4E-128 `v30_2cta_ts4e` down (`46,144 B <= 48,128 B <= 49,152 B` -> 2 CTAs/SM).
pub(super) const V30_2CTA_TS4E_DOWN_SMEM_BYTES: u32 = 46_144;
/// Thread-block width of the Wave-22 Warp-Specialized `v31_ws` (`TILE_M=128, BC=256`) kernels
/// (16 warps = 4 WGs: WG0+WG1 256 Producer/Transcode threads @ 72 regs + WG2+WG3 256 Consumer MMA threads @ 184 regs).
pub(super) const V31_WS_BLOCK: u32 = 512;
/// Dynamic SMEM footprint for flat-scale `v31_ws` gate/up (`87,616 B <= 101,376 B`).
pub(super) const V31_WS_FLAT_GATEUP_SMEM_BYTES: u32 = 87_616;
/// Dynamic SMEM footprint for flat-scale `v31_ws` down (`87,104 B <= 101,376 B`).
pub(super) const V31_WS_FLAT_DOWN_SMEM_BYTES: u32 = 87_104;
/// Dynamic SMEM footprint for Atlas-TS4E-128 `v31_ws_ts4e` gate/up (`91,904 B <= 101,376 B`).
pub(super) const V31_WS_TS4E_GATEUP_SMEM_BYTES: u32 = 91_904;
/// Dynamic SMEM footprint for Atlas-TS4E-128 `v31_ws_ts4e` down (`90,672 B <= 101,376 B`).
pub(super) const V31_WS_TS4E_DOWN_SMEM_BYTES: u32 = 90_672;
/// GB10 grid cap for `qwen4_mxfp8_quant_rows_v16q4_gate` (1024 CTAs).
const XQ_QUANT_GATE_GRID_CAP_GB10: u32 = 1024;
/// Target concurrent CTAs per SM for `qwen4_mxfp8_quant_rows_v16q4_gate` on high-SM GPUs
/// (256 threads/CTA × 6 = 1536 threads/SM; 6 × 188 = 1128 eliminates the 5.45 tail wave at 16K).
const XQ_QUANT_GATE_CTAS_PER_SM: u32 = 6;
/// SM count above which `qwen4_mxfp8_quant_rows_v16q4_gate` scales its grid cap to `6 * sm_count`.
const XQ_QUANT_GATE_HIGH_SM_THRESHOLD: u32 = 64;

#[inline]
#[must_use]
fn xq_quant_gate_grid_for_sm(rows: usize, sm_count: u32) -> u32 {
    let cap = if sm_count > XQ_QUANT_GATE_HIGH_SM_THRESHOLD {
        sm_count.saturating_mul(XQ_QUANT_GATE_CTAS_PER_SM)
    } else {
        XQ_QUANT_GATE_GRID_CAP_GB10
    };
    (rows.min(u32::MAX as usize) as u32).clamp(1, cap.max(1))
}
/// Optional cap for static `v31_ws` persistent grid width (default `active_sm_count()` = 1 CTA/SM:
/// 48 on GB10, 110–188 on SM120 / SM103, preserving Producer-next-prologue / Consumer-epilogue overlap,
/// `PF_DIST` cross-task prefetch, and `b_reuse`/`keep_oq` L2 locality; set e.g. `4096` for A/B sweeps).
pub(crate) const V31_WS_MAX_GRID_SELECTOR: &str = "ATLAS_QWEN4_PREFILL_MOE_V31_MAX_GRID";

#[must_use]
pub(super) fn v29_m256_max_items(pairs: usize) -> usize {
    pairs.min(pairs.div_ceil(V29_M256_TILE_M) + 512)
}

#[must_use]
fn v29_m256_grid_gateup_for_sm(max_items: usize, sm_count: u32) -> u32 {
    max_items
        .saturating_mul(10)
        .clamp(sm_grid_1x(sm_count), abi::GRID as usize) as u32
}

#[must_use]
fn v29_m256_grid_down_for_sm(max_items: usize, sm_count: u32) -> u32 {
    max_items
        .saturating_mul(20)
        .clamp(sm_grid_1x(sm_count), abi::GRID as usize) as u32
}

#[must_use]
pub(super) fn v29_m256_grid_gateup(max_items: usize) -> u32 {
    v29_m256_grid_gateup_for_sm(max_items, atlas_core::device::active_sm_count())
}

#[must_use]
pub(super) fn v29_m256_grid_down(max_items: usize) -> u32 {
    v29_m256_grid_down_for_sm(max_items, atlas_core::device::active_sm_count())
}

#[must_use]
fn v30_2cta_grid_gateup_for_sm(max_items: usize, sm_count: u32) -> u32 {
    max_items
        .saturating_mul(10)
        .clamp(v30_2cta_min_grid_for_sm(sm_count), abi::GRID as usize) as u32
}

#[must_use]
fn v30_2cta_grid_down_for_sm(max_items: usize, sm_count: u32) -> u32 {
    max_items
        .saturating_mul(20)
        .clamp(v30_2cta_min_grid_for_sm(sm_count), abi::GRID as usize) as u32
}

#[must_use]
pub(super) fn v30_2cta_grid_gateup(max_items: usize) -> u32 {
    v30_2cta_grid_gateup_for_sm(max_items, atlas_core::device::active_sm_count())
}

#[must_use]
pub(super) fn v30_2cta_grid_down(max_items: usize) -> u32 {
    v30_2cta_grid_down_for_sm(max_items, atlas_core::device::active_sm_count())
}

#[must_use]
fn parse_v31_ws_max_grid_for_sm(raw: Option<&str>, sm_count: u32) -> usize {
    let min_grid = v31_ws_min_grid_for_sm(sm_count);
    let parsed = raw
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .and_then(|s| s.parse::<usize>().ok());
    match parsed {
        Some(v) if v > 0 => v.clamp(min_grid, abi::GRID as usize),
        _ => v31_ws_default_max_grid_for_sm(sm_count),
    }
}

#[cfg(test)]
#[must_use]
fn parse_v31_ws_max_grid(raw: Option<&str>) -> usize {
    parse_v31_ws_max_grid_for_sm(raw, atlas_core::device::active_sm_count())
}

#[must_use]
fn v31_ws_max_grid() -> usize {
    static RAW: OnceLock<Option<String>> = OnceLock::new();
    let raw = RAW.get_or_init(|| std::env::var(V31_WS_MAX_GRID_SELECTOR).ok());
    parse_v31_ws_max_grid_for_sm(raw.as_deref(), atlas_core::device::active_sm_count())
}

#[must_use]
fn v31_ws_grid_gateup_capped_for_sm(max_items: usize, max_grid: usize, sm_count: u32) -> u32 {
    let min_grid = v31_ws_min_grid_for_sm(sm_count);
    let cap = max_grid.clamp(min_grid, abi::GRID as usize);
    max_items.saturating_mul(5).clamp(min_grid, cap) as u32
}

#[must_use]
fn v31_ws_grid_down_capped_for_sm(max_items: usize, max_grid: usize, sm_count: u32) -> u32 {
    let min_grid = v31_ws_min_grid_for_sm(sm_count);
    let cap = max_grid.clamp(min_grid, abi::GRID as usize);
    max_items.saturating_mul(10).clamp(min_grid, cap) as u32
}

#[must_use]
fn v31_ws_grid_gateup_capped(max_items: usize, max_grid: usize) -> u32 {
    v31_ws_grid_gateup_capped_for_sm(max_items, max_grid, atlas_core::device::active_sm_count())
}

#[must_use]
fn v31_ws_grid_down_capped(max_items: usize, max_grid: usize) -> u32 {
    v31_ws_grid_down_capped_for_sm(max_items, max_grid, atlas_core::device::active_sm_count())
}

#[must_use]
pub(super) fn v31_ws_grid_gateup(max_items: usize) -> u32 {
    v31_ws_grid_gateup_capped(max_items, v31_ws_max_grid())
}

#[must_use]
pub(super) fn v31_ws_grid_down(max_items: usize) -> u32 {
    v31_ws_grid_down_capped(max_items, v31_ws_max_grid())
}

#[must_use]
pub(super) fn v28_m32_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var(V28_M32_SELECTOR).ok().as_deref() != Some("0"))
}

/// True when `ATLAS_QWEN4_MOE_V28_M32=1` is explicitly set to force `v28_m32`
/// even outside batched verify (e.g. single-sequence short prefill). Default
/// unset scopes `v28_m32` to `BatchedVerifyTcDenseScope` (`C >= 2`) so `C=1`
/// short prefill (`rows <= 64`) preserves strict 0-ULP reference parity while
/// batched verify retains the full 2-CTA/SM speedup.
#[must_use]
fn v28_m32_forced_all() -> bool {
    static FORCED: OnceLock<bool> = OnceLock::new();
    *FORCED.get_or_init(|| std::env::var(V28_M32_SELECTOR).ok().as_deref() == Some("1"))
}

#[must_use]
pub(super) fn v28_m32_active_for_scope() -> bool {
    v28_m32_enabled()
        && (v28_m32_forced_all()
            || crate::layers::qwen4_hyper::batched_verify_tc_dense_scope_active())
}

#[must_use]
fn parse_v28_m32_short_prefill_enabled(raw: Option<&str>) -> bool {
    raw.map(str::trim) != Some("0")
}

#[must_use]
fn v28_m32_short_prefill_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        parse_v28_m32_short_prefill_enabled(
            std::env::var(V28_M32_SHORT_PREFILL_SELECTOR)
                .ok()
                .as_deref(),
        )
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct V28M32RoutedContext {
    m32_enabled: bool,
    scope_active: bool,
    short_prefill_enabled: bool,
    use_ts4e: bool,
    rows: usize,
    has_corrections: bool,
    dump_or_oracle: bool,
}

#[must_use]
const fn v28_m32_active_for_routed(ctx: V28M32RoutedContext) -> bool {
    if !ctx.m32_enabled || ctx.has_corrections {
        return false;
    }
    if ctx.scope_active {
        return true;
    }
    ctx.short_prefill_enabled
        && ctx.use_ts4e
        && ctx.rows <= V28_M32_GATEUP_AUTO_MAX_ROWS
        && !ctx.dump_or_oracle
}

#[must_use]
fn parse_v28_m32_max_grid_for_sm(raw: Option<&str>, sm_count: u32) -> usize {
    let min_grid = v28_m32_min_grid_for_sm(sm_count);
    let parsed = raw
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .and_then(|s| s.parse::<usize>().ok());
    match parsed {
        Some(v) if v > 0 => v.clamp(min_grid, abi::GRID as usize),
        _ => V28_M32_DEFAULT_MAX_GRID,
    }
}

#[cfg(test)]
#[must_use]
fn parse_v28_m32_max_grid(raw: Option<&str>) -> usize {
    parse_v28_m32_max_grid_for_sm(raw, atlas_core::device::active_sm_count())
}

#[must_use]
fn v28_m32_max_grid() -> usize {
    static RAW: OnceLock<Option<String>> = OnceLock::new();
    let raw = RAW.get_or_init(|| std::env::var(V28_M32_MAX_GRID_SELECTOR).ok());
    parse_v28_m32_max_grid_for_sm(raw.as_deref(), atlas_core::device::active_sm_count())
}

#[must_use]
fn v28_m32_grid_gateup_for_sm(max_items: usize, max_grid: usize, sm_count: u32) -> u32 {
    let min_grid = v28_m32_min_grid_for_sm(sm_count);
    let cap = max_grid.clamp(min_grid, abi::GRID as usize);
    max_items.saturating_mul(10).clamp(min_grid, cap) as u32
}

#[must_use]
fn v28_m32_grid_down_for_sm(max_items: usize, max_grid: usize, sm_count: u32) -> u32 {
    let min_grid = v28_m32_min_grid_for_sm(sm_count);
    let cap = max_grid.clamp(min_grid, abi::GRID as usize);
    max_items.saturating_mul(20).clamp(min_grid, cap) as u32
}

#[must_use]
fn v28_m32_grid_gateup(max_items: usize, max_grid: usize) -> u32 {
    v28_m32_grid_gateup_for_sm(max_items, max_grid, atlas_core::device::active_sm_count())
}

#[must_use]
fn v28_m32_grid_down(max_items: usize, max_grid: usize) -> u32 {
    v28_m32_grid_down_for_sm(max_items, max_grid, atlas_core::device::active_sm_count())
}

#[must_use]
fn parse_v28_m16_decode_enabled_for_sm(raw: Option<&str>, sm_count: u32) -> bool {
    match raw.map(str::trim) {
        Some("1" | "auto") => true,
        Some("0") => false,
        Some("") | None => sm_count > V28_M16_HIGH_SM_THRESHOLD,
        _ => false,
    }
}

#[cfg(test)]
#[must_use]
fn parse_v28_m16_decode_enabled(raw: Option<&str>) -> bool {
    parse_v28_m16_decode_enabled_for_sm(raw, 48)
}

#[must_use]
fn v28_m16_decode_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        parse_v28_m16_decode_enabled_for_sm(
            std::env::var(V28_M16_DECODE_SELECTOR).ok().as_deref(),
            atlas_core::device::active_sm_count(),
        )
    })
}

#[must_use]
const fn v28_m16_active_for_routed(
    m16_enabled: bool,
    m32_active: bool,
    rows: usize,
    has_corrections: bool,
) -> bool {
    m16_enabled && m32_active && !has_corrections && rows <= V28_M16_DECODE_AUTO_MAX_ROWS
}

#[must_use]
fn v28_m16_grid_gateup_for_sm(max_items: usize, max_grid: usize, sm_count: u32) -> u32 {
    let min_grid = v28_m16_min_grid_for_sm(sm_count);
    let cap = max_grid.clamp(min_grid, abi::GRID as usize);
    max_items.saturating_mul(10).clamp(min_grid, cap) as u32
}

#[must_use]
fn v28_m16_grid_down_for_sm(max_items: usize, max_grid: usize, sm_count: u32) -> u32 {
    let min_grid = v28_m16_min_grid_for_sm(sm_count);
    let cap = max_grid.clamp(min_grid, abi::GRID as usize);
    max_items.saturating_mul(20).clamp(min_grid, cap) as u32
}

#[must_use]
fn v28_m16_grid_gateup(max_items: usize, max_grid: usize) -> u32 {
    v28_m16_grid_gateup_for_sm(max_items, max_grid, atlas_core::device::active_sm_count())
}

#[must_use]
fn v28_m16_grid_down(max_items: usize, max_grid: usize) -> u32 {
    v28_m16_grid_down_for_sm(max_items, max_grid, atlas_core::device::active_sm_count())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum V28M64Dispatch {
    Twarp,
    M64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum V28OpTile {
    Twarp,
    M64,
    M32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct V28DispatchPlan {
    pub gateup: V28OpTile,
    pub down: V28OpTile,
}

#[must_use]
pub(super) const fn v28_ts4e_supported(
    w4a8_tag: &str,
    v28q_level: u32,
    v28_m64: bool,
    has_corrections: bool,
) -> bool {
    matches!(w4a8_tag.as_bytes(), b"v28") && !has_corrections && (v28q_level == 2 || v28_m64)
}

#[must_use]
pub(super) const fn select_v28_m64_dispatch(
    w4a8_tag: &str,
    v28_m64_flag: bool,
    rows: usize,
    has_corrections: bool,
) -> V28M64Dispatch {
    if matches!(w4a8_tag.as_bytes(), b"v28")
        && !has_corrections
        && (v28_m64_flag || rows <= V28_M64_AUTO_MAX_ROWS)
    {
        V28M64Dispatch::M64
    } else {
        V28M64Dispatch::Twarp
    }
}

#[must_use]
pub(super) const fn select_v28_dispatch(
    w4a8_tag: &str,
    v28_m64_flag: bool,
    m32_enabled: bool,
    rows: usize,
    has_corrections: bool,
) -> V28DispatchPlan {
    if !matches!(w4a8_tag.as_bytes(), b"v28") || has_corrections {
        return V28DispatchPlan {
            gateup: V28OpTile::Twarp,
            down: V28OpTile::Twarp,
        };
    }
    if m32_enabled {
        let gateup_m32 = rows <= V28_M32_GATEUP_AUTO_MAX_ROWS;
        let down_m32 = rows <= V28_M32_DOWN_AUTO_MAX_ROWS;
        if gateup_m32 || down_m32 {
            return V28DispatchPlan {
                gateup: if gateup_m32 {
                    V28OpTile::M32
                } else {
                    V28OpTile::M64
                },
                down: if down_m32 {
                    V28OpTile::M32
                } else {
                    V28OpTile::M64
                },
            };
        }
    }
    let fallback = match select_v28_m64_dispatch(w4a8_tag, v28_m64_flag, rows, has_corrections) {
        V28M64Dispatch::M64 => V28OpTile::M64,
        V28M64Dispatch::Twarp => V28OpTile::Twarp,
    };
    V28DispatchPlan {
        gateup: fallback,
        down: fallback,
    }
}

/// Grid width for Twarp-tier W4A8 gate/up: dynamic clamping on `v28`, fixed `abi::GRID` on
/// legacy sets (`v10`/`v12`/`v14`/`v16`) whose kernels enforce `gridDim.x == FIXED_GRID`.
#[must_use]
fn w4a8_twarp_grid_gateup_for_sm(w4a8_tag: &str, max_items: usize, sm_count: u32) -> u32 {
    if w4a8_tag == "v28" {
        max_items
            .saturating_mul(5)
            .clamp(sm_grid_1x(sm_count), abi::GRID as usize) as u32
    } else {
        abi::GRID
    }
}

/// Grid width for Twarp/M64-tier W4A8 down: dynamic clamping on `v28` and `v17q`/`v17qr2` MXFP8
/// down (`1..=FIXED_GRID`), fixed `abi::GRID` on legacy BF16 down (`v10`/`v12`/`v14`/`v16`).
#[must_use]
fn w4a8_twarp_grid_down_for_sm(
    w4a8_tag: &str,
    mxfp8_slot_down: bool,
    max_items: usize,
    sm_count: u32,
) -> u32 {
    if w4a8_tag == "v28" || mxfp8_slot_down {
        max_items
            .saturating_mul(10)
            .clamp(sm_grid_1x(sm_count), abi::GRID as usize) as u32
    } else {
        abi::GRID
    }
}

#[must_use]
const fn ts4e_v28_receipt_smem(plan: V28DispatchPlan, m16_active: bool) -> (u32, u32) {
    let gu = match plan.gateup {
        V28OpTile::M32 if m16_active => V28_M16_TS4E_GATEUP_SMEM_BYTES,
        V28OpTile::M32 => V28_M32_TS4E_GATEUP_SMEM_BYTES,
        V28OpTile::M64 | V28OpTile::Twarp => V28_TS4E_GATEUP_SMEM_BYTES,
    };
    let dn = match plan.down {
        V28OpTile::M32 if m16_active => V28_M16_TS4E_DOWN_SMEM_BYTES,
        V28OpTile::M32 => V28_M32_TS4E_DOWN_SMEM_BYTES,
        V28OpTile::M64 | V28OpTile::Twarp => V28_TS4E_DOWN_SMEM_BYTES,
    };
    (gu, dn)
}

/// Controls host-side L2-windowed `gate_up(wi_b..wi_e)` + `down(wi_b..wi_e)` dispatch on the
/// V28 `twarp` path: unset / `0` = disabled (default OFF, single launch); `1` = auto
/// (when `expanded_rows * 660 B > 24 MiB`, targeting `<= 18 MiB` per window inside GB10's
/// 32 MiB L2); `N >= 2` = explicit max work items per window; `<M>m` / `<M>MiB` = target
/// window MiB.
pub(crate) const L2_WINDOW_SELECTOR: &str = "ATLAS_QWEN4_MOE_L2_WINDOW";
/// MXFP8 intermediate footprint per expanded row: 640 E4M3 payload + 20 UE8M0 scale bytes.
const OQ_BYTES_PER_EXPANDED_ROW: usize = 640 + 20;
/// Full-chunk `oq + os` threshold above which auto L2 windowing engages (`rows >= 3814`, e.g. 4096+).
const L2_WINDOW_TRIGGER_BYTES: usize = 24 * 1024 * 1024;
/// Target per-window `oq + os` working set inside GB10's 32 MiB L2 cache.
const L2_WINDOW_TARGET_BYTES: usize = 18 * 1024 * 1024;
/// Rows covered by one V28 compact work item (`V10_TILE_M = 128`).
const L2_WORK_ITEM_TILE_M: usize = 128;
/// Worst-case intermediate bytes per work item (`128 * 660 = 84,480 B`).
const L2_BYTES_PER_WORK_ITEM: usize = L2_WORK_ITEM_TILE_M * OQ_BYTES_PER_EXPANDED_ROW;
/// Default max work items per L2 window (`18 MiB / 84,480 B = 223` items = `17.97 MiB` worst case).
const DEFAULT_L2_WINDOW_ITEMS: usize = L2_WINDOW_TARGET_BYTES / L2_BYTES_PER_WORK_ITEM;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum L2WindowPolicy {
    Disabled,
    Auto { max_window_items: usize },
    Fixed { max_window_items: usize },
}

fn parse_l2_window_policy(raw: Option<&str>) -> Result<L2WindowPolicy> {
    let Some(raw) = raw.map(str::trim).filter(|s| !s.is_empty()) else {
        // Live GB10 A/B showed multi-window splitting adds 576-864 wave
        // tail-drain stalls across 48 layers (-268 tok/s at 8K); keep opt-in.
        return Ok(L2WindowPolicy::Disabled);
    };
    if raw == "0" {
        return Ok(L2WindowPolicy::Disabled);
    }
    if raw == "1" {
        return Ok(L2WindowPolicy::Auto {
            max_window_items: DEFAULT_L2_WINDOW_ITEMS,
        });
    }
    let lower = raw.to_ascii_lowercase();
    if let Some(prefix) = lower
        .strip_suffix("mib")
        .or_else(|| lower.strip_suffix('m'))
    {
        let mib: usize = prefix
            .trim()
            .parse()
            .with_context(|| format!("{L2_WINDOW_SELECTOR}: invalid MiB window size {raw:?}"))?;
        if mib == 0 {
            return Ok(L2WindowPolicy::Disabled);
        }
        let target_bytes = mib
            .checked_mul(1024 * 1024)
            .context("L2 window byte size overflow")?;
        let max_window_items = (target_bytes / L2_BYTES_PER_WORK_ITEM).max(1);
        return Ok(L2WindowPolicy::Auto { max_window_items });
    }
    let items: usize = raw.parse().with_context(|| {
        format!("{L2_WINDOW_SELECTOR}: expected 0, 1, item count, or <N>m; got {raw:?}")
    })?;
    if items == 0 {
        bail!("{L2_WINDOW_SELECTOR}: window size must be positive or 0 to disable");
    }
    Ok(L2WindowPolicy::Fixed {
        max_window_items: items,
    })
}

fn l2_window_policy() -> Result<L2WindowPolicy> {
    static POLICY: OnceLock<std::result::Result<L2WindowPolicy, String>> = OnceLock::new();
    POLICY
        .get_or_init(|| {
            let raw = std::env::var(L2_WINDOW_SELECTOR).ok();
            parse_l2_window_policy(raw.as_deref()).map_err(|e| format!("{e:#}"))
        })
        .clone()
        .map_err(anyhow::Error::msg)
}

#[must_use]
const fn resolve_l2_window_items(policy: L2WindowPolicy, expanded_rows: usize) -> Option<usize> {
    match policy {
        L2WindowPolicy::Disabled => None,
        L2WindowPolicy::Auto { max_window_items } => {
            if expanded_rows.saturating_mul(OQ_BYTES_PER_EXPANDED_ROW) > L2_WINDOW_TRIGGER_BYTES {
                Some(max_window_items)
            } else {
                None
            }
        }
        L2WindowPolicy::Fixed { max_window_items } => Some(max_window_items),
    }
}

/// Partition `[0..work_count)` into contiguous non-overlapping `[wi_begin..wi_end)` windows
/// of at most `max_window_items` items each. `max_window_items == 0` disables partitioning
/// and yields a single `(0, work_count)` window when `work_count > 0`.
#[must_use]
pub(super) fn plan_l2_windows(work_count: usize, max_window_items: usize) -> Vec<(usize, usize)> {
    if work_count == 0 {
        return Vec::new();
    }
    if max_window_items == 0 || work_count <= max_window_items {
        return vec![(0, work_count)];
    }
    let mut windows = Vec::with_capacity(work_count.div_ceil(max_window_items));
    let mut begin = 0;
    while begin < work_count {
        let end = (begin + max_window_items).min(work_count);
        windows.push((begin, end));
        begin = end;
    }
    windows
}

#[derive(Clone, Copy)]
struct WindowedTwarpKernels {
    gateup: KernelHandle,
    down_mxfp8: KernelHandle,
    down_bf16: KernelHandle,
}

fn windowed_twarp_kernels(
    gpu: &dyn GpuBackend,
    smem_gateup: u32,
    smem_down: u32,
) -> Result<WindowedTwarpKernels> {
    static KERNELS: OnceLock<std::result::Result<WindowedTwarpKernels, String>> = OnceLock::new();
    KERNELS
        .get_or_init(|| {
            let module = "qwen4_moe_compact_t_k32_v28";
            let load = |name: &str| {
                gpu.kernel(module, name)
                    .map_err(|e| format!("{module}::{name}: {e}"))
            };
            let gateup = load("qwen4_moe_compact_t_gemm_gateup_k32_v28_twarp_win")?;
            let down_mxfp8 = load("qwen4_moe_compact_t_gemm_k32_v28q_twarp_win")?;
            let down_bf16 = load("qwen4_moe_compact_t_gemm_k32_v28_twarp_win")?;
            if smem_gateup > 0 {
                gpu.set_kernel_max_dynamic_shared_memory(gateup, smem_gateup)
                    .map_err(|e| e.to_string())?;
            }
            if smem_down > 0 {
                gpu.set_kernel_max_dynamic_shared_memory(down_mxfp8, smem_down)
                    .and_then(|()| gpu.set_kernel_max_dynamic_shared_memory(down_bf16, smem_down))
                    .map_err(|e| e.to_string())?;
            }
            Ok(WindowedTwarpKernels {
                gateup,
                down_mxfp8,
                down_bf16,
            })
        })
        .clone()
        .map_err(|e| anyhow::anyhow!("{L2_WINDOW_SELECTOR}: {e}"))
}

fn load_v28_twarp_bundle(
    gpu: &dyn GpuBackend,
    gateup_name: &str,
    down_mxfp8_name: &str,
    down_bf16_name: &str,
    gateup_smem_bytes: u32,
    down_smem_bytes: u32,
) -> std::result::Result<WindowedTwarpKernels, String> {
    let module = "qwen4_moe_compact_t_k32_v28";
    let load = |name: &str| {
        gpu.kernel(module, name)
            .map_err(|e| format!("{module}::{name}: {e}"))
    };
    let gateup = load(gateup_name)?;
    let down_mxfp8 = load(down_mxfp8_name)?;
    let down_bf16 = load(down_bf16_name)?;
    if gateup_smem_bytes > 0 {
        gpu.set_kernel_max_dynamic_shared_memory(gateup, gateup_smem_bytes)
            .map_err(|e| e.to_string())?;
    }
    if down_smem_bytes > 0 {
        gpu.set_kernel_max_dynamic_shared_memory(down_mxfp8, down_smem_bytes)
            .and_then(|()| gpu.set_kernel_max_dynamic_shared_memory(down_bf16, down_smem_bytes))
            .map_err(|e| e.to_string())?;
    }
    Ok(WindowedTwarpKernels {
        gateup,
        down_mxfp8,
        down_bf16,
    })
}

fn windowed_twarp_ts4e_kernels(gpu: &dyn GpuBackend) -> Result<WindowedTwarpKernels> {
    static KERNELS: OnceLock<std::result::Result<WindowedTwarpKernels, String>> = OnceLock::new();
    KERNELS
        .get_or_init(|| {
            load_v28_twarp_bundle(
                gpu,
                "qwen4_moe_compact_t_gemm_gateup_k32_v28_twarp_win_ts4e",
                "qwen4_moe_compact_t_gemm_k32_v28q_twarp_win_ts4e",
                "qwen4_moe_compact_t_gemm_k32_v28_twarp_win_ts4e",
                V28_TS4E_GATEUP_SMEM_BYTES,
                V28_TS4E_DOWN_SMEM_BYTES,
            )
        })
        .clone()
        .map_err(|e| anyhow::anyhow!("ATLAS_MOE_SCALE_TS4E: {e}"))
}

fn twarp_ts4e_kernels(gpu: &dyn GpuBackend) -> Result<WindowedTwarpKernels> {
    static KERNELS: OnceLock<std::result::Result<WindowedTwarpKernels, String>> = OnceLock::new();
    KERNELS
        .get_or_init(|| {
            load_v28_twarp_bundle(
                gpu,
                "qwen4_moe_compact_t_gemm_gateup_k32_v28_twarp_ts4e",
                "qwen4_moe_compact_t_gemm_k32_v28q_twarp_ts4e",
                "qwen4_moe_compact_t_gemm_k32_v28_twarp_ts4e",
                V28_TS4E_GATEUP_SMEM_BYTES,
                V28_TS4E_DOWN_SMEM_BYTES,
            )
        })
        .clone()
        .map_err(|e| anyhow::anyhow!("ATLAS_MOE_SCALE_TS4E: {e}"))
}

fn twarp_m64_ts4e_kernels(gpu: &dyn GpuBackend) -> Result<WindowedTwarpKernels> {
    static KERNELS: OnceLock<std::result::Result<WindowedTwarpKernels, String>> = OnceLock::new();
    KERNELS
        .get_or_init(|| {
            load_v28_twarp_bundle(
                gpu,
                "qwen4_moe_compact_t_gemm_gateup_k32_v28_twarp_m64_ts4e",
                "qwen4_moe_compact_t_gemm_k32_v28q_twarp_m64_ts4e",
                "qwen4_moe_compact_t_gemm_k32_v28_twarp_m64_ts4e",
                V28_TS4E_GATEUP_SMEM_BYTES,
                V28_TS4E_DOWN_SMEM_BYTES,
            )
        })
        .clone()
        .map_err(|e| anyhow::anyhow!("ATLAS_MOE_SCALE_TS4E (m64): {e}"))
}

fn v28_m64_flat_kernels(gpu: &dyn GpuBackend) -> Result<WindowedTwarpKernels> {
    static KERNELS: OnceLock<std::result::Result<WindowedTwarpKernels, String>> = OnceLock::new();
    KERNELS
        .get_or_init(|| {
            load_v28_twarp_bundle(
                gpu,
                "qwen4_moe_compact_t_gemm_gateup_k32_v28_m64",
                "qwen4_moe_compact_t_gemm_k32_v28q_m64",
                "qwen4_moe_compact_t_gemm_k32_v28_m64",
                V28_FLAT_SMEM_BYTES,
                V28_FLAT_SMEM_BYTES,
            )
        })
        .clone()
        .map_err(|e| anyhow::anyhow!("v28_m64 flat: {e}"))
}

pub(super) fn v28_m64_shared_kernels(
    gpu: &dyn GpuBackend,
) -> Result<(KernelHandle, KernelHandle, u32)> {
    let k = v28_m64_flat_kernels(gpu)?;
    Ok((k.gateup, k.down_bf16, V28_FLAT_SMEM_BYTES))
}

fn v28_m32_ts4e_kernels(gpu: &dyn GpuBackend) -> Result<WindowedTwarpKernels> {
    static KERNELS: OnceLock<std::result::Result<WindowedTwarpKernels, String>> = OnceLock::new();
    KERNELS
        .get_or_init(|| {
            load_v28_twarp_bundle(
                gpu,
                "qwen4_moe_compact_t_gemm_gateup_k32_v28_m32_ts4e",
                "qwen4_moe_compact_t_gemm_k32_v28q_m32_ts4e",
                "qwen4_moe_compact_t_gemm_k32_v28_m32_ts4e",
                V28_M32_TS4E_GATEUP_SMEM_BYTES,
                V28_M32_TS4E_DOWN_SMEM_BYTES,
            )
        })
        .clone()
        .map_err(|e| anyhow::anyhow!("ATLAS_MOE_SCALE_TS4E (m32): {e}"))
}

fn v28_m32_flat_kernels(gpu: &dyn GpuBackend) -> Result<WindowedTwarpKernels> {
    static KERNELS: OnceLock<std::result::Result<WindowedTwarpKernels, String>> = OnceLock::new();
    KERNELS
        .get_or_init(|| {
            load_v28_twarp_bundle(
                gpu,
                "qwen4_moe_compact_t_gemm_gateup_k32_v28_m32",
                "qwen4_moe_compact_t_gemm_k32_v28q_m32",
                "qwen4_moe_compact_t_gemm_k32_v28_m32",
                V28_M32_FLAT_SMEM_BYTES,
                V28_M32_FLAT_SMEM_BYTES,
            )
        })
        .clone()
        .map_err(|e| anyhow::anyhow!("v28_m32 flat: {e}"))
}

pub(super) fn v28_m32_shared_kernels(
    gpu: &dyn GpuBackend,
) -> Result<(KernelHandle, KernelHandle, u32)> {
    let k = v28_m32_flat_kernels(gpu)?;
    Ok((k.gateup, k.down_bf16, V28_M32_FLAT_SMEM_BYTES))
}

fn v28_m16_ts4e_kernels(gpu: &dyn GpuBackend) -> Result<WindowedTwarpKernels> {
    static KERNELS: OnceLock<std::result::Result<WindowedTwarpKernels, String>> = OnceLock::new();
    KERNELS
        .get_or_init(|| {
            load_v28_twarp_bundle(
                gpu,
                "qwen4_moe_compact_t_gemm_gateup_k32_v28_m16_ts4e",
                "qwen4_moe_compact_t_gemm_k32_v28q_m16_ts4e",
                "qwen4_moe_compact_t_gemm_k32_v28_m16_ts4e",
                V28_M16_TS4E_GATEUP_SMEM_BYTES,
                V28_M16_TS4E_DOWN_SMEM_BYTES,
            )
        })
        .clone()
        .map_err(|e| anyhow::anyhow!("ATLAS_MOE_SCALE_TS4E (m16): {e}"))
}

fn v28_m16_flat_kernels(gpu: &dyn GpuBackend) -> Result<WindowedTwarpKernels> {
    static KERNELS: OnceLock<std::result::Result<WindowedTwarpKernels, String>> = OnceLock::new();
    KERNELS
        .get_or_init(|| {
            load_v28_twarp_bundle(
                gpu,
                "qwen4_moe_compact_t_gemm_gateup_k32_v28_m16",
                "qwen4_moe_compact_t_gemm_k32_v28q_m16",
                "qwen4_moe_compact_t_gemm_k32_v28_m16",
                V28_M16_FLAT_GATEUP_SMEM_BYTES,
                V28_M16_FLAT_DOWN_SMEM_BYTES,
            )
        })
        .clone()
        .map_err(|e| anyhow::anyhow!("v28_m16 flat: {e}"))
}

/// Kernel module hosting both legacy and 4-warp coalesced MXFP8 unpermute+blend kernels.
pub(crate) const UNPERMUTE_BLEND_MXFP8_MODULE: &str = "qwen4_moe_compact_t_k32_v17";
/// 4-warp/CTA (`block = 128`) coalesced MXFP8 unpermute+blend entrypoint (0-ULP identical).
pub(crate) const UNPERMUTE_BLEND_MXFP8_W4_KERNEL: &str = "moe_unpermute_reduce_blend_mxfp8_bf16_w4";
/// SM count threshold above which the 4-warp unpermute+blend kernel engages (GB10 `<= 64` stays legacy).
const UNPERMUTE_BLEND_W4_HIGH_SM_THRESHOLD: u32 = 64;

/// Selected MXFP8 unpermute+blend kernel variant.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum UnpermuteBlendVariant {
    /// Legacy 1-warp/CTA (`grid = [rows, 10, 1], block = 32`), preserved on GB10 (`sm_count <= 64`).
    Legacy1Warp,
    /// 4-warp/CTA SMEM-metadata (`block = 128`) variant for SM120/SM103 (`sm_count > 64`).
    Coalesced4Warp,
}

/// Launch geometry for `moe_unpermute_reduce_blend_mxfp8_bf16[_w4]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct UnpermuteBlendLaunch {
    pub variant: UnpermuteBlendVariant,
    pub grid_y: u32,
    pub block_x: u32,
}

/// Choose unpermute+blend variant and `(grid_y, block_x)` for the GPU's SM count.
///
/// - GB10 (`sm_count <= 64`): preserves the legacy 1-warp `(10, 32)` / `(4, 64)` dispatch.
/// - SM120/SM103 (`sm_count > 64`): engages 4-warp `moe_unpermute_reduce_blend_mxfp8_bf16_w4`
///   (`block = 128`), selecting `grid_y = 10` for `rows < 16` (doubles SM fill on tiny verify
///   frames) and `grid_y = 5` for `rows >= 16` (512 cols/CTA, 32-bit coalesced `slot_q` loads,
///   64-bit BF16 stores, and 2x less metadata traffic per token).
#[must_use]
pub(crate) const fn select_unpermute_blend_launch_for_sm(
    rows: usize,
    hidden: u32,
    top_k: u32,
    sm_count: u32,
) -> UnpermuteBlendLaunch {
    if sm_count <= UNPERMUTE_BLEND_W4_HIGH_SM_THRESHOLD {
        let (grid_y, block_x) = if hidden == 2560 && top_k == 10 {
            (10u32, 32u32)
        } else {
            (4u32, 64u32)
        };
        return UnpermuteBlendLaunch {
            variant: UnpermuteBlendVariant::Legacy1Warp,
            grid_y,
            block_x,
        };
    }
    let grid_y = if hidden == 2560 && top_k == 10 && rows < 16 {
        10u32
    } else {
        5u32
    };
    UnpermuteBlendLaunch {
        variant: UnpermuteBlendVariant::Coalesced4Warp,
        grid_y,
        block_x: 128u32,
    }
}

/// Cached handle loader for `moe_unpermute_reduce_blend_mxfp8_bf16_w4`.
pub(super) fn unpermute_reduce_blend_mxfp8_w4_kernel(gpu: &dyn GpuBackend) -> Result<KernelHandle> {
    static KERNEL: OnceLock<std::result::Result<KernelHandle, String>> = OnceLock::new();
    KERNEL
        .get_or_init(|| {
            gpu.kernel(
                UNPERMUTE_BLEND_MXFP8_MODULE,
                UNPERMUTE_BLEND_MXFP8_W4_KERNEL,
            )
            .map_err(|e| {
                format!("{UNPERMUTE_BLEND_MXFP8_MODULE}::{UNPERMUTE_BLEND_MXFP8_W4_KERNEL}: {e}")
            })
        })
        .clone()
        .map_err(|e| anyhow::anyhow!("{e}"))
}

/// Bundled parameters for [`launch_unpermute_reduce_blend_mxfp8_bf16`].
pub(crate) struct UnpermuteBlendArgs<'a> {
    pub gpu: &'a dyn GpuBackend,
    pub legacy_kernel: KernelHandle,
    pub slot_q: DevicePtr,
    pub slot_s: DevicePtr,
    pub shared_out: DevicePtr,
    pub ss_ptr: DevicePtr,
    pub output: DevicePtr,
    pub token_to_perm: DevicePtr,
    pub topk_weights: DevicePtr,
    pub hidden: u32,
    pub rows: usize,
    pub top_k: u32,
    pub routed_status: DevicePtr,
    pub shared_status: DevicePtr,
    pub sticky_err: DevicePtr,
    pub stream: u64,
}

/// Launch fused MXFP8 unpermute+blend, selecting the 4-warp coalesced `_w4` kernel on
/// `active_sm_count() > 64` (SM120/SM103) and keeping the legacy 1-warp kernel on GB10.
pub(crate) fn launch_unpermute_reduce_blend_mxfp8_bf16(args: UnpermuteBlendArgs<'_>) -> Result<()> {
    let sm_count = atlas_core::device::active_sm_count();
    let launch = select_unpermute_blend_launch_for_sm(args.rows, args.hidden, args.top_k, sm_count);
    let (kernel, grid_y, block_x) = match launch.variant {
        UnpermuteBlendVariant::Coalesced4Warp => {
            match unpermute_reduce_blend_mxfp8_w4_kernel(args.gpu) {
                Ok(w4) => {
                    if !UNPERMUTE_BLEND_W4_RECEIPT_LOGGED.swap(true, Ordering::Relaxed) {
                        tracing::info!(
                            "MOE_UNPERMUTE_BLEND_MXFP8_W4_ENGAGED kernel={UNPERMUTE_BLEND_MXFP8_W4_KERNEL} rows={} grid_y={} block={} sm_count={sm_count}",
                            args.rows,
                            launch.grid_y,
                            launch.block_x,
                        );
                    }
                    (w4, launch.grid_y, launch.block_x)
                }
                Err(_) => {
                    let fallback = select_unpermute_blend_launch_for_sm(
                        args.rows,
                        args.hidden,
                        args.top_k,
                        48,
                    );
                    (args.legacy_kernel, fallback.grid_y, fallback.block_x)
                }
            }
        }
        UnpermuteBlendVariant::Legacy1Warp => (args.legacy_kernel, launch.grid_y, launch.block_x),
    };
    let rows_u32 = u32::try_from(args.rows).context("unpermute_blend rows overflow u32")?;
    qwen4_moe_thin_sk::launch_unpermute_blend_kernel(
        args.gpu,
        kernel,
        rows_u32,
        grid_y,
        block_x,
        args.slot_q,
        args.slot_s,
        args.shared_out,
        args.ss_ptr,
        args.output,
        args.token_to_perm,
        args.topk_weights,
        args.hidden,
        args.top_k,
        args.routed_status,
        args.shared_status,
        args.sticky_err,
        args.stream,
    )
}

/// Eagerly load and configure dynamic SMEM attributes for all batched-verify `v28` `m16`/`m32`/`m64`,
/// `v33_thin_sk`, and 4-warp unpermute/blend kernels at model load time so CUDA graph capture never
/// runs lazy kernel setup.
pub(super) fn prewarm_v28_verify_kernels(gpu: &dyn GpuBackend) {
    let warn_if_err = |bundle: &str, err: anyhow::Error| {
        tracing::warn!(bundle, "v28 batched-verify kernel prewarm failed: {err:#}");
    };
    if let Err(err) = v28_m16_ts4e_kernels(gpu) {
        warn_if_err("v28_m16_ts4e", err);
    }
    if let Err(err) = v28_m16_flat_kernels(gpu) {
        warn_if_err("v28_m16_flat", err);
    }
    if let Err(err) = v28_m32_ts4e_kernels(gpu) {
        warn_if_err("v28_m32_ts4e", err);
    }
    if let Err(err) = v28_m32_flat_kernels(gpu) {
        warn_if_err("v28_m32_flat", err);
    }
    if let Err(err) = twarp_m64_ts4e_kernels(gpu) {
        warn_if_err("twarp_m64_ts4e", err);
    }
    if let Err(err) = v28_m64_flat_kernels(gpu) {
        warn_if_err("v28_m64_flat", err);
    }
    qwen4_moe_thin_sk::prewarm_thin_sk_kernels(gpu);
    if let Err(err) = qwen4_moe_thin_sk::thin_sk_kernels(gpu) {
        warn_if_err("v33_thin_sk", err);
    }
    if let Err(err) = unpermute_reduce_blend_mxfp8_w4_kernel(gpu) {
        warn_if_err(UNPERMUTE_BLEND_MXFP8_W4_KERNEL, err);
    }
}

/// Loaded Wave-17 Path A (`TILE_M=256 x BC=128`, `0-ULP`) GEMM and M256 planner kernels.
#[derive(Clone, Copy)]
pub(super) struct V29M256Kernels {
    pub plan: KernelHandle,
    pub plan_inter: KernelHandle,
    pub fused_sort_plan: Option<KernelHandle>,
    pub gateup_flat: KernelHandle,
    pub down_bf16_flat: KernelHandle,
    pub down_mxfp8_flat: KernelHandle,
    pub gateup_ts4e: KernelHandle,
    pub down_bf16_ts4e: KernelHandle,
    pub down_mxfp8_ts4e: KernelHandle,
}

/// Load all 6 `v29_m256` GEMM entrypoints (flat + TS4E-128), configure their dynamic SMEM caps,
/// and load the 2 non-fused `M=256` planners (+ optional fused `qwen4_moe_sort_plan_t_v8` handle).
pub(super) fn load_v29_m256_kernels(gpu: &dyn GpuBackend) -> Result<V29M256Kernels> {
    let module = "qwen4_moe_compact_t_k32_v28";
    let load = |name: &str| {
        gpu.kernel(module, name)
            .with_context(|| format!("{module}::{name}"))
    };
    let plan = load("qwen4_moe_compact_t_plan_k32_v29_m256")?;
    let plan_inter = load("qwen4_moe_compact_t_plan_k32_v29_m256_inter")?;
    let fused_sort_plan = gpu
        .kernel(
            super::qwen4_plan_order_rule::PLAN_V8_MODULE,
            super::qwen4_router_pipe::FUSED_SORT_PLAN_KERNEL,
        )
        .ok();
    let gateup_flat = load("qwen4_moe_compact_t_gemm_gateup_k32_v29_m256")?;
    let down_bf16_flat = load("qwen4_moe_compact_t_gemm_k32_v29_m256")?;
    let down_mxfp8_flat = load("qwen4_moe_compact_t_gemm_k32_v29q_m256")?;
    let gateup_ts4e = load("qwen4_moe_compact_t_gemm_gateup_k32_v29_m256_ts4e")?;
    let down_bf16_ts4e = load("qwen4_moe_compact_t_gemm_k32_v29_m256_ts4e")?;
    let down_mxfp8_ts4e = load("qwen4_moe_compact_t_gemm_k32_v29q_m256_ts4e")?;
    gpu.set_kernel_max_dynamic_shared_memory(gateup_flat, V29_M256_FLAT_SMEM_BYTES)?;
    gpu.set_kernel_max_dynamic_shared_memory(down_bf16_flat, V29_M256_FLAT_SMEM_BYTES)?;
    gpu.set_kernel_max_dynamic_shared_memory(down_mxfp8_flat, V29_M256_FLAT_SMEM_BYTES)?;
    gpu.set_kernel_max_dynamic_shared_memory(gateup_ts4e, V29_M256_TS4E_GATEUP_SMEM_BYTES)?;
    gpu.set_kernel_max_dynamic_shared_memory(down_bf16_ts4e, V29_M256_TS4E_DOWN_SMEM_BYTES)?;
    gpu.set_kernel_max_dynamic_shared_memory(down_mxfp8_ts4e, V29_M256_TS4E_DOWN_SMEM_BYTES)?;
    Ok(V29M256Kernels {
        plan,
        plan_inter,
        fused_sort_plan,
        gateup_flat,
        down_bf16_flat,
        down_mxfp8_flat,
        gateup_ts4e,
        down_bf16_ts4e,
        down_mxfp8_ts4e,
    })
}

/// Loaded Wave-21 2-CTA/SM (`TILE_M=128 x BC=128`, `0-ULP`, `<= 48 KiB` SMEM) V30 GEMM kernels.
#[derive(Clone, Copy)]
pub(super) struct V302CtaKernels {
    pub gateup_flat: KernelHandle,
    pub down_bf16_flat: KernelHandle,
    pub down_mxfp8_flat: KernelHandle,
    pub gateup_ts4e: KernelHandle,
    pub down_bf16_ts4e: KernelHandle,
    pub down_mxfp8_ts4e: KernelHandle,
}

/// Load all 6 `v30_2cta` GEMM entrypoints (flat + TS4E-128) and configure their dynamic SMEM caps.
pub(super) fn load_v30_2cta_kernels(gpu: &dyn GpuBackend) -> Result<V302CtaKernels> {
    let module = "qwen4_moe_compact_t_k32_v28";
    let load = |name: &str| {
        gpu.kernel(module, name)
            .with_context(|| format!("{module}::{name}"))
    };
    let gateup_flat = load("qwen4_moe_compact_t_gemm_gateup_k32_v30_2cta")?;
    let down_bf16_flat = load("qwen4_moe_compact_t_gemm_k32_v30_2cta")?;
    let down_mxfp8_flat = load("qwen4_moe_compact_t_gemm_k32_v30q_2cta")?;
    let gateup_ts4e = load("qwen4_moe_compact_t_gemm_gateup_k32_v30_2cta_ts4e")?;
    let down_bf16_ts4e = load("qwen4_moe_compact_t_gemm_k32_v30_2cta_ts4e")?;
    let down_mxfp8_ts4e = load("qwen4_moe_compact_t_gemm_k32_v30q_2cta_ts4e")?;
    gpu.set_kernel_max_dynamic_shared_memory(gateup_flat, V30_2CTA_FLAT_GATEUP_SMEM_BYTES)?;
    gpu.set_kernel_max_dynamic_shared_memory(down_bf16_flat, V30_2CTA_FLAT_DOWN_SMEM_BYTES)?;
    gpu.set_kernel_max_dynamic_shared_memory(down_mxfp8_flat, V30_2CTA_FLAT_DOWN_SMEM_BYTES)?;
    gpu.set_kernel_max_dynamic_shared_memory(gateup_ts4e, V30_2CTA_TS4E_GATEUP_SMEM_BYTES)?;
    gpu.set_kernel_max_dynamic_shared_memory(down_bf16_ts4e, V30_2CTA_TS4E_DOWN_SMEM_BYTES)?;
    gpu.set_kernel_max_dynamic_shared_memory(down_mxfp8_ts4e, V30_2CTA_TS4E_DOWN_SMEM_BYTES)?;
    Ok(V302CtaKernels {
        gateup_flat,
        down_bf16_flat,
        down_mxfp8_flat,
        gateup_ts4e,
        down_bf16_ts4e,
        down_mxfp8_ts4e,
    })
}

/// Loaded Wave-22 Warp-Specialized (`TILE_M=128 x BC=256`, 512 threads, `0-ULP`, `<= 99 KiB` SMEM) V31 GEMM kernels.
#[derive(Clone, Copy)]
pub(super) struct V31WsKernels {
    pub gateup_flat: KernelHandle,
    pub down_bf16_flat: KernelHandle,
    pub down_mxfp8_flat: KernelHandle,
    pub gateup_ts4e: KernelHandle,
    pub down_bf16_ts4e: KernelHandle,
    pub down_mxfp8_ts4e: KernelHandle,
}

/// Load all 6 `v31_ws` GEMM entrypoints (flat + TS4E-128) and configure their dynamic SMEM caps.
pub(super) fn load_v31_ws_kernels(gpu: &dyn GpuBackend) -> Result<V31WsKernels> {
    let module = "qwen4_moe_compact_t_k32_v28";
    let load = |name: &str| {
        gpu.kernel(module, name)
            .with_context(|| format!("{module}::{name}"))
    };
    let gateup_flat = load("qwen4_moe_compact_t_gemm_gateup_k32_v31_ws")?;
    let down_bf16_flat = load("qwen4_moe_compact_t_gemm_k32_v31_ws")?;
    let down_mxfp8_flat = load("qwen4_moe_compact_t_gemm_k32_v31q_ws")?;
    let gateup_ts4e = load("qwen4_moe_compact_t_gemm_gateup_k32_v31_ws_ts4e")?;
    let down_bf16_ts4e = load("qwen4_moe_compact_t_gemm_k32_v31_ws_ts4e")?;
    let down_mxfp8_ts4e = load("qwen4_moe_compact_t_gemm_k32_v31q_ws_ts4e")?;
    gpu.set_kernel_max_dynamic_shared_memory(gateup_flat, V31_WS_FLAT_GATEUP_SMEM_BYTES)?;
    gpu.set_kernel_max_dynamic_shared_memory(down_bf16_flat, V31_WS_FLAT_DOWN_SMEM_BYTES)?;
    gpu.set_kernel_max_dynamic_shared_memory(down_mxfp8_flat, V31_WS_FLAT_DOWN_SMEM_BYTES)?;
    gpu.set_kernel_max_dynamic_shared_memory(gateup_ts4e, V31_WS_TS4E_GATEUP_SMEM_BYTES)?;
    gpu.set_kernel_max_dynamic_shared_memory(down_bf16_ts4e, V31_WS_TS4E_DOWN_SMEM_BYTES)?;
    gpu.set_kernel_max_dynamic_shared_memory(down_mxfp8_ts4e, V31_WS_TS4E_DOWN_SMEM_BYTES)?;
    Ok(V31WsKernels {
        gateup_flat,
        down_bf16_flat,
        down_mxfp8_flat,
        gateup_ts4e,
        down_bf16_ts4e,
        down_mxfp8_ts4e,
    })
}

/// The gatepre `ss[rows]` f32 buffer: the arena's preallocated
/// `moe_gate_scalars` region (one f32 per batch token, sized at load), never a
/// mid-request allocation. `validate_context` admits `rows` against it.
fn gate_scalars_buf(
    buffers: &spark_runtime::buffers::BufferArena,
    rows: usize,
) -> Result<DevicePtr> {
    let ptr = buffers.moe_gate_scalars();
    ensure!(
        !ptr.is_null() && buffers.sizes().moe_gate_scalars >= rows * 4,
        "gatepre gate scalars: the arena holds {} B, {rows} rows need {} B",
        buffers.sizes().moe_gate_scalars,
        rows * 4
    );
    Ok(ptr)
}

static PLAN_ORDER_RAN_LOGGED: AtomicBool = AtomicBool::new(false);
static PLAN_ORDER_SKIP_LOGGED: AtomicBool = AtomicBool::new(false);

static PLAN_ORDER_TRIGGER: std::sync::OnceLock<Option<std::path::PathBuf>> =
    std::sync::OnceLock::new();
/// Last trigger state logged: 0 off, 1 on, 2 none yet.
static PLAN_ORDER_TRIGGER_STATE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(2);

/// ATLAS_QWEN4_PREFILL_MOE_PLAN_ORDER_TRIGGER (test-only): unset -> true; set -> whether the file exists now.
/// Every state change logs one MOE_PREFILL_PLAN_ORDER_TRIGGER line (with the dispatch's rows).
fn plan_order_trigger_on(rows: usize) -> bool {
    let trigger = PLAN_ORDER_TRIGGER.get_or_init(|| {
        std::env::var_os(super::qwen4_plan_order_rule::PLAN_ORDER_TRIGGER_SELECTOR)
            .map(std::path::PathBuf::from)
    });
    let Some(path) = trigger else {
        return true;
    };
    let on = path.exists();
    let state = u8::from(on);
    if PLAN_ORDER_TRIGGER_STATE.swap(state, Ordering::Relaxed) != state {
        tracing::info!(
            "MOE_PREFILL_PLAN_ORDER_TRIGGER state={} rows={rows} file={}",
            if on { "on" } else { "off" },
            path.display()
        );
    }
    on
}

/// Zero top-k weights whose expert falls outside `[local_start, local_end)` so
/// EP unpermute kernels skip remote experts without reading uncomputed rows.
pub(super) fn launch_ep_mask_remote_weights(
    gpu: &dyn spark_runtime::gpu::GpuBackend,
    indices_dev: DevicePtr,
    weights_dev: DevicePtr,
    local_start: usize,
    local_end: usize,
    total_expanded: u32,
    stream: u64,
) -> Result<()> {
    if total_expanded == 0 {
        return Ok(());
    }
    let k = gpu.kernel("moe", "moe_ep_mask_remote_weights")?;
    let grid = total_expanded.div_ceil(256);
    KernelLaunch::new(gpu, k)
        .grid([grid, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(indices_dev)
        .arg_ptr(weights_dev)
        .arg_u32(local_start as u32)
        .arg_u32(local_end as u32)
        .arg_u32(total_expanded)
        .launch(stream)
}

/// The routed W4A8 dispatch's planner: V8 (ATLAS_QWEN4_PREFILL_MOE_PLAN_ORDER=inter) when its row gate admits
/// `rows` (and the test-only trigger, if set, is on), else V7. Same items and header either way; only the item
/// order differs (byte-identical outputs).
fn plan_for_rows(wide: &qwen4_prefill_compact::WideKernels, rows: usize) -> KernelHandle {
    let Some((v8, order)) = wide.plan_order else {
        return wide.plan;
    };
    if !order.engages(rows) {
        if !PLAN_ORDER_SKIP_LOGGED.swap(true, Ordering::Relaxed) {
            tracing::info!(
                "MOE_PREFILL_PLAN_ORDER_SKIPPED order=inter rows={rows} max_rows={}",
                order.max_rows_label()
            );
        }
        return wide.plan;
    }
    if !plan_order_trigger_on(rows) {
        return wide.plan;
    }
    if !PLAN_ORDER_RAN_LOGGED.swap(true, Ordering::Relaxed) {
        tracing::info!("MOE_PREFILL_PLAN_ORDER_RAN order=inter planner=v8 rows={rows}");
    }
    v8
}

/// The bf16 -> MXFP8 x quant of the W4A8 path (payload `xq` [rows][2560], UE8M0 `xs` [rows][80]),
/// launched by the shared dispatch (and by the routed one when XQ_REUSE does not apply). The loaded
/// entry is the W4A8 set's own `qwen4_mxfp8_quant_rows_<tag>` or, under
/// ATLAS_QWEN4_PREFILL_MOE_SLOT_BF16_FUSED=1, the quad-lane twin v16q4 (V16 or V28 set); all
/// grid-stride, so the launch geometry is shared.
#[allow(clippy::too_many_arguments)]
pub(super) fn launch_xq_quant(
    gpu: &dyn spark_runtime::gpu::GpuBackend,
    buffers: &spark_runtime::buffers::BufferArena,
    wide: &qwen4_prefill_compact::WideKernels,
    input: DevicePtr,
    shared_gate: DevicePtr,
    xq: DevicePtr,
    xs: DevicePtr,
    rows: usize,
    stream: u64,
    allow_router_fp8: bool,
) -> Result<()> {
    let want_gate_scalars = crate::layers::qwen4_hyper::prefill_inject::selection()?.gatepre
        || super::forward_k5::batched_verify_tc_moe_fused_blend_active();
    if want_gate_scalars && let Some(k_gate) = wide.w4a8_quant_gate {
        static FUSED_FP8_ENV_OK: OnceLock<bool> = OnceLock::new();
        let env_ok = *FUSED_FP8_ENV_OK.get_or_init(|| {
            std::env::var_os("ATLAS_QWEN4_ORACLE_DUMP").is_none()
                && std::env::var("ATLAS_QWEN4_PREFILL_NATIVE").ok().as_deref() != Some("1")
        });
        let ss_ptr = gate_scalars_buf(buffers, rows)?;
        let allow_fused_fp8 = allow_router_fp8
            && env_ok
            && !super::dump::enabled()
            && !super::qwen4_router_bf16::selected().is_ok_and(|v| v)
            && !crate::model::qwen4_prefill_engagement::slot_r2_active_for(rows);
        let mut fused_fp8 = None;
        if allow_fused_fp8 {
            fused_fp8 = crate::layers::qwen4_fast_proj::prepare_fused_fp8_act(
                gpu, input, rows, 2560, stream,
            )?;
        } else {
            crate::layers::qwen4_fast_proj::clear_prequant_fp8_act();
        }
        if let Some(pre) = crate::layers::qwen4_hyper::prefill_fused::take_prequant_mxfp8(
            input, xq, xs, rows, stream,
        ) && pre.has_gate_scalars(shared_gate, ss_ptr)
        {
            if let Some((a_amax, a8_out, a_scale_out)) = fused_fp8 {
                crate::layers::qwen4_hyper::prefill_fused::launch_fp8_act_from_bf16(
                    gpu,
                    input,
                    a_amax,
                    a8_out,
                    a_scale_out,
                    rows,
                    stream,
                )?;
                crate::layers::qwen4_fast_proj::note_prequant_fp8_act(input, rows, 2560, stream);
            }
            GATE_SCALARS_READY.with(|c| c.set(Some((input, shared_gate, rows, ss_ptr))));
            if wide.xq_fused && !SLOT_BF16_FUSED_RECEIPT_LOGGED.swap(true, Ordering::Relaxed) {
                tracing::info!(
                    "MOE_PREFILL_SLOT_BF16_FUSED_ENGAGED selector={} kernel={} set={} rows={rows}",
                    qwen4_prefill_compact::SLOT_BF16_FUSED_SELECTOR,
                    qwen4_xq_quant::XQ_QUANT_KERNEL_V16Q4_GATE,
                    wide.w4a8_tag
                );
            }
            if !XQ_GATEPRE_RECEIPT_LOGGED.swap(true, Ordering::Relaxed) {
                tracing::info!(
                    "MOE_PREFILL_XQ_GATEPRE_ENGAGED kernel={} set={} rows={rows}",
                    qwen4_xq_quant::XQ_QUANT_KERNEL_V16Q4_GATE,
                    wide.w4a8_tag
                );
            }
            return Ok(());
        }
        let (a_amax, a8_out, a_scale_out) =
            fused_fp8.unwrap_or((DevicePtr::NULL, DevicePtr::NULL, DevicePtr::NULL));
        let grid_x = xq_quant_gate_grid_for_sm(rows, atlas_core::device::active_sm_count());
        KernelLaunch::new(gpu, k_gate)
            .grid([grid_x, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(input)
            .arg_ptr(xq)
            .arg_ptr(xs)
            .arg_ptr(shared_gate)
            .arg_ptr(ss_ptr)
            .arg_u32(rows as u32)
            .arg_ptr(a_amax)
            .arg_ptr(a8_out)
            .arg_ptr(a_scale_out)
            .launch(stream)?;
        if fused_fp8.is_some() {
            crate::layers::qwen4_fast_proj::note_prequant_fp8_act(input, rows, 2560, stream);
        }
        GATE_SCALARS_READY.with(|c| c.set(Some((input, shared_gate, rows, ss_ptr))));
        if wide.xq_fused && !SLOT_BF16_FUSED_RECEIPT_LOGGED.swap(true, Ordering::Relaxed) {
            tracing::info!(
                "MOE_PREFILL_SLOT_BF16_FUSED_ENGAGED selector={} kernel={} set={} rows={rows}",
                qwen4_prefill_compact::SLOT_BF16_FUSED_SELECTOR,
                qwen4_xq_quant::XQ_QUANT_KERNEL_V16Q4_GATE,
                wide.w4a8_tag
            );
        }
        if !XQ_GATEPRE_RECEIPT_LOGGED.swap(true, Ordering::Relaxed) {
            tracing::info!(
                "MOE_PREFILL_XQ_GATEPRE_ENGAGED kernel={} set={} rows={rows}",
                qwen4_xq_quant::XQ_QUANT_KERNEL_V16Q4_GATE,
                wide.w4a8_tag
            );
        }
        return Ok(());
    }
    GATE_SCALARS_READY.with(|c| c.set(None));
    if crate::layers::qwen4_hyper::prefill_fused::take_prequant_mxfp8(input, xq, xs, rows, stream)
        .is_some()
    {
        if wide.xq_fused && !SLOT_BF16_FUSED_RECEIPT_LOGGED.swap(true, Ordering::Relaxed) {
            tracing::info!(
                "MOE_PREFILL_SLOT_BF16_FUSED_ENGAGED selector={} kernel={} set={} rows={rows}",
                qwen4_prefill_compact::SLOT_BF16_FUSED_SELECTOR,
                qwen4_xq_quant::XQ_QUANT_KERNEL_V16Q4,
                wide.w4a8_tag
            );
        }
        return Ok(());
    }
    KernelLaunch::new(gpu, wide.w4a8_quant)
        .grid([1024, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(xq)
        .arg_ptr(xs)
        .arg_u32(rows as u32)
        .arg_u32(2560)
        .launch(stream)?;
    if wide.xq_fused && !SLOT_BF16_FUSED_RECEIPT_LOGGED.swap(true, Ordering::Relaxed) {
        tracing::info!(
            "MOE_PREFILL_SLOT_BF16_FUSED_ENGAGED selector={} kernel={} set={} rows={rows}",
            qwen4_prefill_compact::SLOT_BF16_FUSED_SELECTOR,
            qwen4_xq_quant::XQ_QUANT_KERNEL_V16Q4,
            wide.w4a8_tag
        );
    }
    Ok(())
}

thread_local! {
    /// Set by the W4A8 dispatch to describe the slot-row ABI; taken by the unpermute step.
    static SLOT_STORAGE: std::cell::Cell<qwen4_prefill_compact::SlotStorage> =
        const { std::cell::Cell::new(qwen4_prefill_compact::SlotStorage::Bf16) };
    /// Set by the shared W4A8 dispatch: the (input, rows) whose MXFP8 x sits in expert_up_out.
    static XQ_READY: std::cell::Cell<Option<(DevicePtr, usize)>> = const { std::cell::Cell::new(None) };
    /// Set by `launch_xq_quant_with_gate`: (input, shared_gate, rows, ss_ptr).
    static GATE_SCALARS_READY: std::cell::Cell<Option<(DevicePtr, DevicePtr, usize, DevicePtr)>> =
        const { std::cell::Cell::new(None) };
}

/// The shared W4A8 dispatch just wrote `input`'s MXFP8 x (payload [rows][2560], UE8M0
/// [rows][80] at +rows*2560) into expert_up_out.
pub(super) fn note_xq(input: DevicePtr, rows: usize) {
    XQ_READY.with(|c| c.set(Some((input, rows))));
}

/// Something may have written expert_up_out since the shared expert quantised x.
pub(super) fn forget_xq() {
    XQ_READY.with(|c| c.set(None));
    GATE_SCALARS_READY.with(|c| c.set(None));
    crate::layers::qwen4_fast_proj::clear_prequant_fp8_act();
}

/// True when `GATE_SCALARS_READY` already holds precomputed `ss[rows]` for `(input, shared_gate, rows)`.
fn has_gate_scalars(input: DevicePtr, shared_gate: DevicePtr, rows: usize) -> bool {
    matches!(
        GATE_SCALARS_READY.with(std::cell::Cell::get),
        Some((inp, sg, r, ptr)) if inp == input && sg == shared_gate && r == rows && !ptr.is_null()
    )
}

/// Record precomputed `ss[rows]` f32 scalars for consumption by fused unpermute+blend.
fn note_gate_scalars(input: DevicePtr, shared_gate: DevicePtr, rows: usize, ss_ptr: DevicePtr) {
    GATE_SCALARS_READY.with(|c| c.set(Some((input, shared_gate, rows, ss_ptr))));
}

/// Take the precomputed `ss[rows]` f32 buffer if it matches `(input, shared_gate, rows)`.
pub(super) fn take_gate_scalars(
    input: DevicePtr,
    shared_gate: DevicePtr,
    rows: usize,
) -> DevicePtr {
    match GATE_SCALARS_READY.with(std::cell::Cell::take) {
        Some((inp, sg, r, ptr)) if inp == input && sg == shared_gate && r == rows => ptr,
        _ => DevicePtr::NULL,
    }
}

/// True (once) when expert_up_out still holds the MXFP8 x of exactly this input and row count.
fn take_xq(input: DevicePtr, rows: usize) -> bool {
    XQ_READY.with(|c| c.take()) == Some((input, rows))
}

/// Return (once) the slot-row ABI written by the routed dispatch on this thread.
pub(super) fn take_slot_storage() -> qwen4_prefill_compact::SlotStorage {
    SLOT_STORAGE.with(|c| c.replace(qwen4_prefill_compact::SlotStorage::Bf16))
}

fn current_slot_storage() -> qwen4_prefill_compact::SlotStorage {
    SLOT_STORAGE.with(std::cell::Cell::get)
}

/// Drop a producer ABI left behind when a prior prefill failed before its consumer.
pub(crate) fn discard_stale_slot_storage() -> bool {
    take_slot_storage().is_mxfp8()
}

impl MoeLayer {
    fn compact_tables(
        &self,
        transposed: bool,
    ) -> Result<(&ExpertPtrTable, &ExpertPtrTable, &ExpertPtrTable)> {
        if transposed {
            return Ok((
                self.gate_ptrs_t
                    .as_ref()
                    .context("transposed compact gate table absent")?,
                self.up_ptrs_t
                    .as_ref()
                    .context("transposed compact up table absent")?,
                self.down_ptrs_t
                    .as_ref()
                    .context("transposed compact down table absent")?,
            ));
        }
        Ok((&self.gate_ptrs, &self.up_ptrs, &self.down_ptrs))
    }

    pub(super) fn run_qwen4_compact(
        &self,
        input: DevicePtr,
        offsets: DevicePtr,
        sorted_ids: DevicePtr,
        rows: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        qwen4_prefill_compact::validate_context(ctx, rows)?;
        self.validate_qwen4_prefill_moe()?;
        ensure!(
            qwen4_prefill_compact::request_active(self.qwen4_compact.is_some()),
            "request-suppressed Qwen4 compact dispatch"
        );
        let check = qwen4_prefill_compact::check_selected()?;
        let kernels = self
            .qwen4_compact
            .context("compact kernel bundle is absent")?;
        let dispatched_slot_storage =
            kernels
                .wide
                .map_or(qwen4_prefill_compact::SlotStorage::Bf16, |wide| {
                    qwen4_prefill_compact::dispatch_slot_storage(
                        wide.slot_storage,
                        crate::model::qwen4_prefill_engagement::slot_r2_active_for(rows),
                    )
                });
        if crate::layers::qwen4_oracle::root().is_some() {
            ensure!(
                !dispatched_slot_storage.is_mxfp8(),
                "compact oracle does not label MXFP8 slot payloads as BF16; use the dedicated R2 gate"
            );
            ensure!(
                self.routed_scale_layout == RoutedScaleLayout::FlatE4m3,
                "compact oracle capture requires FlatE4m3 scales; set ATLAS_MOE_SCALE_TS4E=0"
            );
        }
        if kernels.streamed {
            ensure!(
                self.routed_scale_layout == RoutedScaleLayout::FlatE4m3,
                "unported streamed compact MoE path requires FlatE4m3; set ATLAS_MOE_SCALE_TS4E=0"
            );
            self.populate_qwen4_stream_t(rows, ctx, stream)?;
        }
        if self.routed_scale_layout == RoutedScaleLayout::Ts4e128 {
            ensure!(
                kernels.transposed,
                "Atlas-TS4E-128 compact prefill requires transposed expert pointer tables; set ATLAS_MOE_SCALE_TS4E=0"
            );
        }
        let (gate_table, up_table, down_table) = self.compact_tables(kernels.transposed)?;
        ensure!(
            input == ctx.buffers.norm_output()
                && sorted_ids == ctx.buffers.gate_logits()
                && offsets == ctx.buffers.gate_logits().offset(rows * 10 * 8),
            "compact dispatch requires the exact F8 input/sort layout"
        );
        let contract_bytes = abi::contract(rows).map_err(anyhow::Error::msg)?;
        let workspace = ctx.buffers.moe_worklist();
        let contract = workspace.offset(abi::WORKSPACE_BYTES);
        let status = ctx.buffers.moe_worklist_total();
        // The normal sort stores all metadata in gate_logits. Require exact
        // extents here; neither inputs nor metadata may alias writable scratch.
        let input_region = abi::region(input.0, rows * 2560 * 2, 16).map_err(anyhow::Error::msg)?;
        let metadata = abi::region(
            ctx.buffers.gate_logits().0,
            ctx.buffers.sizes().gate_logits,
            16,
        )
        .map_err(anyhow::Error::msg)?;
        for (ptr, bytes) in [(offsets, 513 * 4), (sorted_ids, rows * 10 * 4)] {
            let region = abi::region(ptr.0, bytes, 4).map_err(anyhow::Error::msg)?;
            ensure!(
                region.0 >= metadata.0 && region.1 <= metadata.1,
                "compact routing metadata is outside its admitted arena"
            );
        }
        let b_sizes = ctx.buffers.sizes();
        for (ptr, bytes) in [
            (workspace, abi::ARENA_BYTES),
            (status, 4),
            (ctx.buffers.expert_gate_out(), rows * 10 * 640 * 2),
            (
                ctx.buffers.expert_up_out(),
                (rows * 10 * 640 * 2).min(b_sizes.expert_up_out),
            ),
            (
                ctx.buffers.expert_down_out(),
                (rows * 10 * 2560 * 2).min(b_sizes.expert_down_out),
            ),
        ] {
            ensure!(
                abi::disjoint(
                    input_region,
                    abi::region(ptr.0, bytes, 4).map_err(anyhow::Error::msg)?
                ),
                "compact input aliases writable scratch"
            );
        }
        spark_runtime::cublaslt::stage_timer::mark("compact_pre", stream);
        let pending = (-1i32).to_le_bytes();
        // Default-off (ATLAS_QWEN4_PREFILL_MOE_STATUS_ASYNC=1): no host sync per layer; the
        // constant bytes come from a device copy and the status is checked at the chunk/round end.
        let status_async = super::qwen4_moe_status::selected()
            && !check
            && (crate::model::qwen4_prefill_engagement::active_for(rows)
                || super::forward_k5::batched_verify_tc_moe_status_async_active());
        if super::qwen4_moe_status::is_capturing(ctx.gpu, stream) {
            ensure!(
                status_async && !check && !kernels.streamed,
                "compact MoE during CUDA graph capture requires async MoE status without check/streamed mode"
            );
        }
        let is_ep = ctx.config.ep_world_size > 1 && ctx.comm.is_some_and(|c| c.world_size() > 1);
        let fused_sort_plan = !is_ep
            && super::qwen4_router_pipe::fused_sort_plan_active(true, self.qwen4_compact, rows)?;
        if !fused_sort_plan {
            // Byte offset of `Workspace::next_task_dn` (zeroed here so legacy V7/EP planners,
            // which only zero `Workspace::reserved` at +32, leave both work-steal counters clean).
            const WORKSPACE_NEXT_TASK_DN_OFFSET: usize = 36;
            // `Contract::reserved0` at offset 56 is guaranteed 0u32 by `abi::contract`.
            const CONTRACT_ZERO_U32_OFFSET: usize = 56;
            let zero_u32 = 0u32.to_le_bytes();
            if status_async {
                let mut bytes = [0u8; abi::CONTRACT_BYTES + 4];
                bytes[..abi::CONTRACT_BYTES].copy_from_slice(&contract_bytes);
                bytes[abi::CONTRACT_BYTES..].copy_from_slice(&pending);
                let constants =
                    super::qwen4_moe_status::constant(ctx.gpu, "routed", rows, &bytes, stream)?;
                ctx.gpu
                    .copy_d2d_async(constants, contract, contract_bytes.len(), stream)?;
                ctx.gpu.copy_d2d_async(
                    constants.offset(contract_bytes.len()),
                    status,
                    4,
                    stream,
                )?;
                ctx.gpu.copy_d2d_async(
                    constants.offset(CONTRACT_ZERO_U32_OFFSET),
                    workspace.offset(WORKSPACE_NEXT_TASK_DN_OFFSET),
                    4,
                    stream,
                )?;
            } else {
                ctx.gpu.copy_h2d_group_on_stream(
                    &[
                        HostToDeviceCopy::new(&contract_bytes, contract),
                        HostToDeviceCopy::new(&pending, status),
                        HostToDeviceCopy::new(
                            &zero_u32,
                            workspace.offset(WORKSPACE_NEXT_TASK_DN_OFFSET),
                        ),
                    ],
                    stream,
                )?;
            }
        }
        // Default-off quality-gate selection (ATLAS_QWEN4_PREFILL_FAKEQUANT=v7|b12x): the V7
        // kernels (and their b12x weight-rounding twins) replace the loaded set per dispatch.
        let spec = crate::layers::qwen4_fakequant::current()?;
        // Default-off W4A8 (ATLAS_QWEN4_PREFILL_MOE_W4A8=1, or FAKEQUANT=w4a8 per dispatch):
        // the V10 kernels quantise their own activations to MXFP8.
        let w4a8 = if qwen4_prefill_compact::w4a8_selected()? || spec.moe_kernel_w4a8 {
            ensure!(!check, "W4A8 compact MoE rejects the compact CHECK path");
            ensure!(
                spec.moe_in.is_none()
                    && spec.moe_down.is_none()
                    && !spec.moe_weights_b12x
                    && !spec.moe_kernel_v7,
                "W4A8 compact MoE quantises its own activations; it conflicts with FAKEQUANT \
                 moe formats, v7 and b12x"
            );
            Some(kernels.wide.context(
                "ATLAS_QWEN4_PREFILL_MOE_W4A8 needs the V2 compact path (ATLAS_QWEN4_PREFILL_MOE_COMPACT_V2>0)",
            )?)
        } else {
            None
        };
        let (mut plan_kernel, gemm_kernel, gemm_block, gateup_kernel, gateup_block) = if let Some(
            wide,
        ) = w4a8
        {
            (
                plan_for_rows(&wide, rows),
                wide.gemm,
                256,
                Some(wide.gateup),
                256,
            )
        } else if spec.moe_kernel_v7 || spec.moe_weights_b12x {
            let wide = kernels
                    .wide
                    .context("FAKEQUANT v7/b12x needs the V2 compact path (ATLAS_QWEN4_PREFILL_MOE_COMPACT_V2>0)")?;
            if spec.moe_weights_b12x {
                (
                    wide.plan,
                    wide.gemm_b12xw,
                    256,
                    Some(wide.gateup_b12xw),
                    256,
                )
            } else {
                (wide.plan, wide.gemm, 256, Some(wide.gateup), 256)
            }
        } else {
            (
                kernels.plan,
                kernels.gemm,
                kernels.block,
                kernels.gateup,
                kernels.gateup_block,
            )
        };
        let use_v8_inter = w4a8.is_some_and(|wide| plan_kernel.0 != wide.plan.0);
        let pairs = rows * 10;
        let v29_m256_active = if !is_ep && let Some(wide) = w4a8 {
            let twarp_windowable = wide.w4a8_tag == "v28"
                && wide.v28q_level == 2
                && !wide.v28_m64
                && !dispatched_slot_storage.has_corrections();
            let l2_windowed = if twarp_windowable {
                let max_items_128 = pairs.min(pairs.div_ceil(128) + 512);
                resolve_l2_window_items(l2_window_policy()?, pairs)
                    .map(|cap| plan_l2_windows(max_items_128, cap))
                    .is_some_and(|windows| windows.len() > 1)
            } else {
                false
            };
            let policy = qwen4_prefill_compact::v29_m256_policy()?;
            let eligible = qwen4_prefill_compact::V29M256Eligibility {
                rows,
                w4a8_is_v28: wide.w4a8_tag == "v28",
                v28_m64_forced: wide.v28_m64,
                has_corrections: dispatched_slot_storage.has_corrections(),
                l2_windowed,
                kernels_loaded: wide.v29_m256.is_some(),
            };
            if policy.admits(eligible) {
                wide.v29_m256
            } else {
                None
            }
        } else {
            None
        };
        if let Some(v29) = v29_m256_active {
            plan_kernel = if use_v8_inter {
                v29.plan_inter
            } else {
                v29.plan
            };
        }
        spark_runtime::cublaslt::stage_timer::mark("compact_contract", stream);
        if fused_sort_plan {
            let base_perm_off = rows * 10 * 8 + 513 * 4;
            let gatepre_perm_off = base_perm_off.max(rows * 640);
            let perm_off = if crate::layers::qwen4_hyper::prefill_inject::selection()?.gatepre
                && gatepre_perm_off + rows * 10 * 4 <= rows * 512 * 2
            {
                gatepre_perm_off
            } else {
                base_perm_off
            };
            let (flags, single_cta_handle) = if let Some(v29) = v29_m256_active {
                (u32::from(use_v8_inter) | 2u32, v29.fused_sort_plan)
            } else {
                (u32::from(use_v8_inter), None)
            };
            // `expert_gate_out` is dead before the gate/up GEMM overwrites it and strictly
            // disjoint from input, metadata, and any side-stream shared-expert scratch.
            let bufs = super::qwen4_router_pipe::FusedSortPlanBuffers {
                topk_ids: ctx.buffers.scratch(),
                sorted_token_ids: sorted_ids,
                expert_offsets: offsets,
                token_to_perm: ctx.buffers.gate_logits().offset(perm_off),
                contract,
                workspace,
                status,
                hist_scratch: Some((
                    ctx.buffers.expert_gate_out(),
                    ctx.buffers.sizes().expert_gate_out,
                )),
            };
            super::qwen4_router_pipe::launch_fused_sort_plan(
                ctx.gpu,
                bufs,
                rows as u32,
                flags,
                single_cta_handle,
                stream,
            )?;
        } else if is_ep {
            let (ep_start, ep_end) = ctx.config.local_expert_range();
            let ep_plan = ctx.gpu.kernel(
                "qwen4_moe_compact_t_k32_v7",
                "qwen4_moe_compact_t_plan_k32_ep",
            )?;
            KernelLaunch::new(ctx.gpu, ep_plan)
                .grid([1, 1, 1])
                .block([512, 1, 1])
                .arg_ptr(offsets)
                .arg_ptr(sorted_ids)
                .arg_ptr(contract)
                .arg_ptr(workspace)
                .arg_ptr(status)
                .arg_u32(ep_start as u32)
                .arg_u32(ep_end as u32)
                .launch(stream)?;
        } else {
            KernelLaunch::new(ctx.gpu, plan_kernel)
                .grid([1, 1, 1])
                .block([512, 1, 1])
                .arg_ptr(offsets)
                .arg_ptr(sorted_ids)
                .arg_ptr(contract)
                .arg_ptr(workspace)
                .arg_ptr(status)
                .launch(stream)?;
        }
        let gemm = |a: DevicePtr, table: &ExpertPtrTable, c: DevicePtr, n: u32, k: u32| {
            KernelLaunch::new(ctx.gpu, gemm_kernel)
                .grid([abi::GRID, 1, 1])
                .block([gemm_block, 1, 1])
                .arg_ptr(a)
                .arg_ptr(table.packed_ptrs)
                .arg_ptr(table.scale_ptrs)
                .arg_ptr(table.scale2_vals)
                .arg_ptr(c)
                .arg_ptr(offsets)
                .arg_ptr(sorted_ids)
                .arg_ptr(contract)
                .arg_ptr(workspace)
                .arg_ptr(status)
                .arg_u32(n)
                .arg_u32(k)
                .launch(stream)
        };
        spark_runtime::cublaslt::stage_timer::mark("compact_plan", stream);
        let gate = ctx.buffers.expert_gate_out();
        let up = ctx.buffers.expert_up_out();
        if let Some(wide) = w4a8 {
            self.run_qwen4_w4a8(
                &wide,
                v29_m256_active,
                kernels.layer_idx,
                input,
                offsets,
                sorted_ids,
                rows,
                (gate_table, up_table, down_table),
                (contract, workspace, status),
                ctx,
                stream,
            )?;
        } else {
            ensure!(
                self.routed_scale_layout == RoutedScaleLayout::FlatE4m3,
                "unported non-W4A8 compact MoE path requires FlatE4m3; set ATLAS_MOE_SCALE_TS4E=0"
            );
            // Default-off quality-gate fake quantization (ATLAS_QWEN4_PREFILL_FAKEQUANT).
            let fq_restore =
                crate::layers::qwen4_fakequant::moe_in_begin(ctx.gpu, input, rows, 2560, stream)?;
            if let Some(gateup) = gateup_kernel {
                ensure!(!check, "fused gate/up rejects the compact CHECK path");
                // One A tile feeds both accumulators; the epilogue applies the
                // shipping silu*up on the BF16-rounded values and writes the
                // activated result into the gate buffer (what silu_mul produced).
                KernelLaunch::new(ctx.gpu, gateup)
                    .grid([abi::GRID, 1, 1])
                    .block([gateup_block, 1, 1])
                    .arg_ptr(input)
                    .arg_ptr(gate_table.packed_ptrs)
                    .arg_ptr(gate_table.scale_ptrs)
                    .arg_ptr(gate_table.scale2_vals)
                    .arg_ptr(gate)
                    .arg_ptr(up_table.packed_ptrs)
                    .arg_ptr(up_table.scale_ptrs)
                    .arg_ptr(up_table.scale2_vals)
                    .arg_ptr(offsets)
                    .arg_ptr(sorted_ids)
                    .arg_ptr(contract)
                    .arg_ptr(workspace)
                    .arg_ptr(status)
                    .arg_u32(640)
                    .arg_u32(2560)
                    .launch(stream)?;
                crate::layers::qwen4_fakequant::moe_in_end(ctx.gpu, fq_restore, stream)?;
            } else {
                gemm(input, gate_table, gate, 640, 2560)?;
                gemm(input, up_table, up, 640, 2560)?;
                crate::layers::qwen4_fakequant::moe_in_end(ctx.gpu, fq_restore, stream)?;
                if check {
                    super::qwen4_compact_check::check_planned_after_stream(ctx, status, stream)?;
                    self.check_compact_gate_up(input, offsets, sorted_ids, rows, ctx, stream)?;
                }
                // Exactly the shipping activation and in-place BF16 boundary.
                ops::silu_mul(
                    ctx.gpu,
                    self.moe_act_mul,
                    gate,
                    up,
                    gate,
                    (rows * 10 * 640) as u32,
                    stream,
                )?;
            }
            crate::layers::qwen4_fakequant::moe_down(ctx.gpu, gate, rows * 10, 640, stream)?;
            gemm(gate, down_table, ctx.buffers.expert_down_out(), 2560, 640)?;
        }
        spark_runtime::cublaslt::stage_timer::mark("compact_gemms", stream);
        // Env-gated oracle capture (ATLAS_QWEN4_ORACLE_DUMP). Records the exact
        // arguments and results of ONE compact-MoE dispatch so the GEMM can be
        // replayed standalone. Off by default; costs nothing when unset.
        self.oracle_capture_compact(
            input, offsets, sorted_ids, rows, gate_table, up_table, down_table, ctx, stream,
        )?;
        // PLANNED=1 is not a completion marker: the ordered D2H drains every
        // GEMM first. Only then can unchanged status admit unpermute/blend/HC.
        // Every planner/GEMM failure (including still PENDING) propagates.
        // In batched-verify sticky mode, `forward_k5` folds `routed_status` + `shared_status`
        // directly into `sticky_err` via `atomicOr` (zero per-layer D2D copies, capture safe).
        if status_async {
            if !super::forward_k5::batched_verify_tc_moe_status_async_active() {
                super::qwen4_moe_status::defer(ctx.gpu, status, "routed", stream)?;
            }
        } else {
            let mut result = [0u8; 4];
            ctx.gpu.copy_d2h_on_stream(status, &mut result, stream)?;
            let result = i32::from_le_bytes(result);
            abi::check_status(result)
                .map_err(|error| anyhow::anyhow!("{error}: status={result}"))?;
        }
        crate::layers::qwen4_fakequant::moe_hist_dump(ctx.gpu, offsets, 512, rows, stream)?;
        if kernels.step_k == 32 && !K32_RECEIPT_LOGGED.swap(true, Ordering::Relaxed) {
            tracing::info!(
                "MOE_PREFILL_COMPACT_K32_ENGAGED selector={} rows={rows} transposed={}",
                if kernels.transposed {
                    if kernels.streamed {
                        qwen4_prefill_compact::STREAM_SELECTOR
                    } else {
                        qwen4_prefill_compact::TRANSPOSED_SELECTOR
                    }
                } else {
                    qwen4_prefill_compact::K32_SELECTOR
                },
                kernels.transposed,
            );
        }
        if check {
            self.check_compact_down(offsets, rows, ctx, stream)?;
        }
        Ok(())
    }

    /// V10 W4A8 gate/up/down over the planned work list. Scratch reuse: the MXFP8 input
    /// (rows x 2560 payload + rows x 80 UE8M0) lives in the up buffer, which the fused path
    /// never writes; the MXFP8 activated intermediate (pairs x 640 + pairs x 20) lives in the
    /// gate buffer. Down writes BF16 pairs exactly where the BF16 path does.
    #[allow(clippy::too_many_arguments)]
    fn run_qwen4_w4a8(
        &self,
        wide: &qwen4_prefill_compact::WideKernels,
        v29_m256: Option<V29M256Kernels>,
        layer_idx: usize,
        input: DevicePtr,
        offsets: DevicePtr,
        sorted_ids: DevicePtr,
        rows: usize,
        (gate_table, up_table, down_table): (&ExpertPtrTable, &ExpertPtrTable, &ExpertPtrTable),
        (contract, workspace, status): (DevicePtr, DevicePtr, DevicePtr),
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let pairs = rows * 10;
        let up = ctx.buffers.expert_up_out();
        let gate = ctx.buffers.expert_gate_out();
        let slot_storage = qwen4_prefill_compact::dispatch_slot_storage(
            wide.slot_storage,
            crate::model::qwen4_prefill_engagement::slot_r2_active_for(rows),
        );
        let use_ts4e = self.routed_scale_layout == RoutedScaleLayout::Ts4e128;
        let thin_sk_policy = qwen4_moe_thin_sk::thin_sk_policy()?;
        if thin_sk_policy != qwen4_moe_thin_sk::ThinSkPolicy::Off {
            let thin_sk_loaded = qwen4_moe_thin_sk::thin_sk_kernels(ctx.gpu).ok();
            let thin_sk_elig = qwen4_moe_thin_sk::ThinSkEligibility {
                rows,
                w4a8_is_v28: wide.w4a8_tag == "v28",
                has_corrections: slot_storage.has_corrections(),
                v29_m256_active: v29_m256.is_some(),
                oracle_capture: crate::layers::qwen4_oracle::root().is_some(),
                kernels_loaded: thin_sk_loaded.is_some(),
            };
            if thin_sk_policy.admits(thin_sk_elig)
                && let Some(thin_sk_kernels) = thin_sk_loaded
            {
                // Consume any staged XQ marker without touching `expert_up_out` (so concurrent
                // shared-expert overlap streams reading `xq`/`xs` stay race-free).
                let _ = take_xq(input, rows);
                let want_gate_scalars = crate::layers::qwen4_hyper::prefill_inject::selection()?
                    .gatepre
                    || super::forward_k5::batched_verify_tc_moe_fused_blend_active();
                let shared_gate_precompute = if want_gate_scalars {
                    let sg_weight = self.weights.shared_expert_gate.weight;
                    if has_gate_scalars(input, sg_weight, rows) {
                        None
                    } else {
                        let ss_ptr = gate_scalars_buf(ctx.buffers, rows)?;
                        note_gate_scalars(input, sg_weight, rows, ss_ptr);
                        Some(qwen4_moe_thin_sk::SharedGatePrecompute {
                            shared_gate_weight: sg_weight,
                            ss_out: ss_ptr,
                        })
                    }
                } else {
                    GATE_SCALARS_READY.with(|c| c.set(None));
                    None
                };
                let b_sizes = ctx.buffers.sizes();
                let written_storage = qwen4_moe_thin_sk::launch_thin_sk_routed(
                    qwen4_moe_thin_sk::ThinSkDispatchArgs {
                        gpu: ctx.gpu,
                        kernels: thin_sk_kernels,
                        scale_layout: self.routed_scale_layout,
                        slot_storage,
                        input,
                        offsets,
                        sorted_ids,
                        rows,
                        gate_table,
                        up_table,
                        down_table,
                        contract,
                        workspace,
                        status,
                        gate_buf: gate,
                        gate_buf_bytes: b_sizes.expert_gate_out,
                        down_buf: ctx.buffers.expert_down_out(),
                        down_buf_bytes: b_sizes.expert_down_out,
                        shared_gate_precompute,
                        stream,
                    },
                )?;
                SLOT_STORAGE.with(|c| c.set(written_storage));
                if written_storage.is_mxfp8()
                    && !SLOT_MXFP8_RECEIPT_LOGGED.swap(true, Ordering::Relaxed)
                {
                    let tag = if use_ts4e {
                        "v33_thin_sk_ts4e"
                    } else {
                        "v33_thin_sk"
                    };
                    tracing::info!("MOE_PREFILL_SLOT_MXFP8_ENGAGED kernels={tag} rows={rows}");
                }
                if use_ts4e && !TS4E_PREFILL_RECEIPT_LOGGED.swap(true, Ordering::Relaxed) {
                    tracing::info!(
                        "MOE_PREFILL_TS4E128_ENGAGED smem_gateup={} smem_down={} gateup=V33_THIN_SK down=V33_THIN_SK rows={rows}",
                        qwen4_moe_thin_sk::THIN_SK_TS4E_SMEM_BYTES,
                        qwen4_moe_thin_sk::THIN_SK_TS4E_SMEM_BYTES,
                    );
                }
                if !W4A8_RECEIPT_LOGGED.swap(true, Ordering::Relaxed) {
                    tracing::info!(
                        "MOE_PREFILL_W4A8_ENGAGED kernels={} rows={rows}",
                        wide.w4a8_tag
                    );
                }
                return Ok(());
            }
        }
        let max_items = pairs.min(pairs.div_ceil(128) + 512);
        let twarp_windowable = wide.w4a8_tag == "v28"
            && wide.v28q_level == 2
            && !wide.v28_m64
            && !slot_storage.has_corrections();
        let l2_window_items = if twarp_windowable {
            resolve_l2_window_items(l2_window_policy()?, pairs)
        } else {
            None
        };
        let l2_windows = l2_window_items
            .map(|cap| plan_l2_windows(max_items, cap))
            .filter(|windows| windows.len() > 1);
        let v30_policy = qwen4_prefill_compact::v30_2cta_policy()?;
        let v30_eligible = qwen4_prefill_compact::V302CtaEligibility {
            rows,
            w4a8_is_v28: wide.w4a8_tag == "v28",
            v28_m64_forced: wide.v28_m64,
            has_corrections: slot_storage.has_corrections(),
            l2_windowed: l2_windows.is_some(),
            v29_m256_active: v29_m256.is_some(),
            v28q_level: wide.v28q_level,
            kernels_loaded: wide.v30_2cta.is_some(),
        };
        let v30_2cta_active = v30_policy.admits(v30_eligible);
        let w4a4_mode = qwen4_prefill_compact::qwen4_moe_w4a4::w4a4_mode()?;
        if w4a4_mode.is_enabled()
            && let Some(w4a4_kernels) = wide.w4a4_v32
        {
            use qwen4_prefill_compact::qwen4_moe_w4a4 as w4a4;
            let verify_scope_active =
                crate::layers::qwen4_hyper::batched_verify_tc_dense_scope_active()
                    || super::forward_k5::batched_verify_tc_moe_status_async_active()
                    || super::forward_k5::batched_verify_tc_moe_xq_prequant_active()
                    || super::forward_k5::batched_verify_tc_moe_fused_blend_active()
                    || ctx.graph_capture;
            let w4a4_boundary = w4a4::w4a4_protected_boundary()?;
            let w4a4_elig = w4a4::W4a4Eligibility {
                rows,
                w4a8_is_v28: wide.w4a8_tag == "v28",
                v28_m64_forced: wide.v28_m64,
                has_corrections: slot_storage.has_corrections(),
                l2_windowed: l2_windows.is_some(),
                v29_m256_active: v29_m256.is_some(),
                v30_2cta_active,
                verify_scope_active,
                kernels_loaded: true,
                layer_idx: Some(layer_idx),
                num_layers: Some(ctx.config.num_hidden_layers),
            };
            if w4a4_mode.admits_with_boundary(w4a4_elig, w4a4_boundary)
                && crate::layers::qwen4_oracle::root().is_none()
            {
                // If shared-expert overlap pre-quantized MXFP8 x into `expert_up_out`,
                // place NVFP4 `xq`/`xs`/`x_scale2` (`rows * 1444 B` = 23.7 MiB @ 16K) in
                // `expert_down_out` (800 MiB @ 16K), which is 100% dead during `w4a4_quant`
                // and `w4a4_gateup` and only written later by `w4a4_down` / `V31_WS` DOWN.
                // This keeps `expert_up_out` untouched for any concurrent shared-expert
                // stream while fitting 16K/32K/64K with >770 MiB to spare and 0 allocs.
                let shared_overlap_active = take_xq(input, rows);
                let b_sizes = ctx.buffers.sizes();
                let (w4_in_base, w4_in_bytes) = if shared_overlap_active {
                    (ctx.buffers.expert_down_out(), b_sizes.expert_down_out)
                } else {
                    (up, b_sizes.expert_up_out)
                };
                let w4a4_layout = w4a4::W4a4ScratchLayout::for_rows(
                    rows,
                    w4a4_mode,
                    w4_in_bytes,
                    b_sizes.expert_gate_out,
                )?;
                let (w4_xq, w4_xs, w4_xscale2) = w4a4_layout.input_ptrs(w4_in_base);
                let (w4_oq, w4_os, w4_oq_scale2) = w4a4_layout.intermediate_ptrs(gate);
                let want_gate_scalars = crate::layers::qwen4_hyper::prefill_inject::selection()?
                    .gatepre
                    || super::forward_k5::batched_verify_tc_moe_fused_blend_active();
                let (sg_weight, ss_out) = if want_gate_scalars {
                    let sg = self.weights.shared_expert_gate.weight;
                    if has_gate_scalars(input, sg, rows) {
                        (DevicePtr::NULL, DevicePtr::NULL)
                    } else {
                        let ss_ptr = gate_scalars_buf(ctx.buffers, rows)?;
                        note_gate_scalars(input, sg, rows, ss_ptr);
                        (sg, ss_ptr)
                    }
                } else {
                    GATE_SCALARS_READY.with(|c| c.set(None));
                    (DevicePtr::NULL, DevicePtr::NULL)
                };
                w4a4::launch_w4a4_quant(
                    ctx.gpu,
                    &w4a4_kernels,
                    input,
                    w4_xq,
                    w4_xs,
                    w4_xscale2,
                    sg_weight,
                    ss_out,
                    rows,
                    stream,
                )?;
                spark_runtime::cublaslt::stage_timer::mark("compact_quantx", stream);

                let scale_layout = if use_ts4e {
                    w4a4::W4a4ScaleLayout::Ts4e128
                } else {
                    w4a4::W4a4ScaleLayout::Flat
                };
                let work = w4a4::W4a4WorkBuffers {
                    expert_offsets: offsets,
                    sorted_token_ids: sorted_ids,
                    contract,
                    workspace,
                    status,
                };
                let (grid_gu, smem_gu) = w4a4::launch_w4a4_gateup(
                    ctx.gpu,
                    &w4a4_kernels,
                    w4a4_mode,
                    scale_layout,
                    w4_xq,
                    w4_xs,
                    w4_xscale2,
                    gate_table,
                    up_table,
                    w4_oq,
                    w4_os,
                    w4_oq_scale2,
                    work,
                    max_items,
                    w4a4::DEFAULT_OQ_SCALE2,
                    stream,
                )?;
                spark_runtime::cublaslt::stage_timer::mark("compact_gateup", stream);

                let (grid_dn, smem_dn) = if w4a4_mode.uses_w4a4_down() {
                    let down_out = if slot_storage.is_mxfp8() {
                        let slots = ctx.buffers.expert_down_out();
                        let layout = qwen4_prefill_compact::slot_layout(pairs, slot_storage)?;
                        ensure!(
                            layout.used_bytes <= pairs * 2560 * 2,
                            "MXFP8 slot rows do not fit expert_down_out"
                        );
                        let slot_scales = slots.offset(
                            layout
                                .scales_offset
                                .context("MXFP8 slot layout omitted scales")?,
                        );
                        SLOT_STORAGE.with(|c| c.set(slot_storage));
                        w4a4::W4a4DownOutput::Mxfp8Slots {
                            slot_q: slots,
                            slot_s: slot_scales,
                        }
                    } else {
                        SLOT_STORAGE.with(|c| c.set(qwen4_prefill_compact::SlotStorage::Bf16));
                        w4a4::W4a4DownOutput::Bf16 {
                            c_out: ctx.buffers.expert_down_out(),
                        }
                    };
                    w4a4::launch_w4a4_down(
                        ctx.gpu,
                        &w4a4_kernels,
                        scale_layout,
                        w4_oq,
                        w4_os,
                        w4_oq_scale2,
                        down_table,
                        down_out,
                        work,
                        max_items,
                        w4a4::DEFAULT_OQ_SCALE2,
                        stream,
                    )?
                } else {
                    // Hybrid mode (`gateup`): W4A4 GATEUP emitted exact MXFP8 `oq`/`os`
                    // into `(w4_oq, w4_os)`; feed `V31_WS` DOWN (preferred) or `V28` DOWN.
                    let v31 = wide.v31_ws.context(
                        "ATLAS_QWEN4_PREFILL_MOE_W4A4=gateup requires V31_WS DOWN kernels",
                    )?;
                    let grid_down = v31_ws_grid_down(max_items);
                    let down_smem = if use_ts4e {
                        V31_WS_TS4E_DOWN_SMEM_BYTES
                    } else {
                        V31_WS_FLAT_DOWN_SMEM_BYTES
                    };
                    if slot_storage.is_mxfp8() {
                        let slots = ctx.buffers.expert_down_out();
                        let layout = qwen4_prefill_compact::slot_layout(pairs, slot_storage)?;
                        ensure!(
                            layout.used_bytes <= pairs * 2560 * 2,
                            "MXFP8 slot rows do not fit expert_down_out"
                        );
                        let slot_scales = slots.offset(
                            layout
                                .scales_offset
                                .context("MXFP8 slot layout omitted scales")?,
                        );
                        let down_handle = if use_ts4e {
                            v31.down_mxfp8_ts4e
                        } else {
                            v31.down_mxfp8_flat
                        };
                        KernelLaunch::new(ctx.gpu, down_handle)
                            .grid([grid_down, 1, 1])
                            .block([V31_WS_BLOCK, 1, 1])
                            .shared_mem(down_smem)
                            .arg_ptr(w4_oq)
                            .arg_ptr(w4_os)
                            .arg_ptr(down_table.packed_ptrs)
                            .arg_ptr(down_table.scale_ptrs)
                            .arg_ptr(down_table.scale2_vals)
                            .arg_ptr(slots)
                            .arg_ptr(slot_scales)
                            .arg_ptr(offsets)
                            .arg_ptr(sorted_ids)
                            .arg_ptr(contract)
                            .arg_ptr(workspace)
                            .arg_ptr(status)
                            .arg_u32(2560)
                            .arg_u32(640)
                            .launch(stream)?;
                        SLOT_STORAGE.with(|c| c.set(slot_storage));
                    } else {
                        let down_handle = if use_ts4e {
                            v31.down_bf16_ts4e
                        } else {
                            v31.down_bf16_flat
                        };
                        KernelLaunch::new(ctx.gpu, down_handle)
                            .grid([grid_down, 1, 1])
                            .block([V31_WS_BLOCK, 1, 1])
                            .shared_mem(down_smem)
                            .arg_ptr(w4_oq)
                            .arg_ptr(w4_os)
                            .arg_ptr(down_table.packed_ptrs)
                            .arg_ptr(down_table.scale_ptrs)
                            .arg_ptr(down_table.scale2_vals)
                            .arg_ptr(ctx.buffers.expert_down_out())
                            .arg_ptr(offsets)
                            .arg_ptr(sorted_ids)
                            .arg_ptr(contract)
                            .arg_ptr(workspace)
                            .arg_ptr(status)
                            .arg_u32(2560)
                            .arg_u32(640)
                            .launch(stream)?;
                        SLOT_STORAGE.with(|c| c.set(qwen4_prefill_compact::SlotStorage::Bf16));
                    }
                    (grid_down, down_smem)
                };
                spark_runtime::cublaslt::stage_timer::mark("compact_down", stream);
                w4a4::log_w4a4_receipt(
                    w4a4_mode,
                    rows,
                    grid_gu,
                    grid_dn,
                    smem_gu,
                    smem_dn,
                    scale_layout,
                );
                if slot_storage.is_mxfp8()
                    && !SLOT_MXFP8_RECEIPT_LOGGED.swap(true, Ordering::Relaxed)
                {
                    let tag = if use_ts4e {
                        "v32_w4a4_ts4e"
                    } else {
                        "v32_w4a4"
                    };
                    tracing::info!("MOE_PREFILL_SLOT_MXFP8_ENGAGED kernels={tag} rows={rows}");
                }
                if use_ts4e && !TS4E_PREFILL_RECEIPT_LOGGED.swap(true, Ordering::Relaxed) {
                    tracing::info!(
                        "MOE_PREFILL_TS4E128_ENGAGED smem_gateup={smem_gu} smem_down={smem_dn} gateup=V32_W4A4 down={} rows={rows}",
                        if w4a4_mode.uses_w4a4_down() {
                            "V32_W4A4"
                        } else {
                            "V31_WS"
                        },
                    );
                }
                if !W4A8_RECEIPT_LOGGED.swap(true, Ordering::Relaxed) {
                    tracing::info!(
                        "MOE_PREFILL_W4A8_ENGAGED kernels={} rows={rows}",
                        wide.w4a8_tag
                    );
                }
                return Ok(());
            }
        }
        ensure!(
            rows * (2560 + 80) <= pairs * 640 * 2 && pairs * (640 + 20) <= pairs * 640 * 2,
            "W4A8 MXFP8 operands do not fit the compact scratch"
        );
        let (xq, xs) = (up, up.offset(rows * 2560));
        let (aq, as_) = (gate, gate.offset(pairs * 640));
        // ATLAS_QWEN4_PREFILL_MOE_XQ_REUSE (or batched-verify shared overlap, which pre-quants
        // x on the primary stream before forking the shared expert onto the secondary stream):
        // the same MXFP8 x sits in expert_up_out; nothing has written expert_up_out since.
        if take_xq(input, rows)
            && (super::forward_k5::batched_verify_tc_moe_xq_prequant_active()
                || qwen4_prefill_compact::xq_reuse_selected()?)
        {
            if !XQ_REUSE_RECEIPT_LOGGED.swap(true, Ordering::Relaxed) {
                tracing::info!("MOE_PREFILL_XQ_REUSE_ENGAGED rows={rows}");
            }
        } else {
            #[rustfmt::skip]
            launch_xq_quant(ctx.gpu, ctx.buffers, wide, input, self.weights.shared_expert_gate.weight, xq, xs, rows, stream, false)?;
        }
        spark_runtime::cublaslt::stage_timer::mark("compact_quantx", stream);
        if let Some(v29) = v29_m256 {
            debug_assert!(!slot_storage.has_corrections());
            let max_items_m256 = v29_m256_max_items(pairs);
            let grid_gateup = v29_m256_grid_gateup(max_items_m256);
            let grid_down = v29_m256_grid_down(max_items_m256);
            let (gateup_handle, gateup_smem) = if use_ts4e {
                (v29.gateup_ts4e, V29_M256_TS4E_GATEUP_SMEM_BYTES)
            } else {
                (v29.gateup_flat, V29_M256_FLAT_SMEM_BYTES)
            };
            KernelLaunch::new(ctx.gpu, gateup_handle)
                .grid([grid_gateup, 1, 1])
                .block([V29_M256_BLOCK, 1, 1])
                .shared_mem(gateup_smem)
                .arg_ptr(xq)
                .arg_ptr(xs)
                .arg_ptr(gate_table.packed_ptrs)
                .arg_ptr(gate_table.scale_ptrs)
                .arg_ptr(gate_table.scale2_vals)
                .arg_ptr(aq)
                .arg_ptr(as_)
                .arg_ptr(up_table.packed_ptrs)
                .arg_ptr(up_table.scale_ptrs)
                .arg_ptr(up_table.scale2_vals)
                .arg_ptr(offsets)
                .arg_ptr(sorted_ids)
                .arg_ptr(contract)
                .arg_ptr(workspace)
                .arg_ptr(status)
                .arg_u32(640)
                .arg_u32(2560)
                .launch(stream)?;
            spark_runtime::cublaslt::stage_timer::mark("compact_gateup", stream);
            let down_smem = if use_ts4e {
                V29_M256_TS4E_DOWN_SMEM_BYTES
            } else {
                V29_M256_FLAT_SMEM_BYTES
            };
            if slot_storage.is_mxfp8() {
                let slots = ctx.buffers.expert_down_out();
                let layout = qwen4_prefill_compact::slot_layout(pairs, slot_storage)?;
                ensure!(
                    layout.used_bytes <= pairs * 2560 * 2,
                    "MXFP8 slot rows do not fit expert_down_out"
                );
                let slot_scales = slots.offset(
                    layout
                        .scales_offset
                        .context("MXFP8 slot layout omitted scales")?,
                );
                let down_handle = if use_ts4e {
                    v29.down_mxfp8_ts4e
                } else {
                    v29.down_mxfp8_flat
                };
                KernelLaunch::new(ctx.gpu, down_handle)
                    .grid([grid_down, 1, 1])
                    .block([V29_M256_BLOCK, 1, 1])
                    .shared_mem(down_smem)
                    .arg_ptr(aq)
                    .arg_ptr(as_)
                    .arg_ptr(down_table.packed_ptrs)
                    .arg_ptr(down_table.scale_ptrs)
                    .arg_ptr(down_table.scale2_vals)
                    .arg_ptr(slots)
                    .arg_ptr(slot_scales)
                    .arg_ptr(offsets)
                    .arg_ptr(sorted_ids)
                    .arg_ptr(contract)
                    .arg_ptr(workspace)
                    .arg_ptr(status)
                    .arg_u32(2560)
                    .arg_u32(640)
                    .launch(stream)?;
                SLOT_STORAGE.with(|c| c.set(slot_storage));
                if !SLOT_MXFP8_RECEIPT_LOGGED.swap(true, Ordering::Relaxed) {
                    let tag = if use_ts4e {
                        "v29q_m256_ts4e"
                    } else {
                        "v29q_m256"
                    };
                    tracing::info!("MOE_PREFILL_SLOT_MXFP8_ENGAGED kernels={tag} rows={rows}");
                }
            } else {
                let down_handle = if use_ts4e {
                    v29.down_bf16_ts4e
                } else {
                    v29.down_bf16_flat
                };
                KernelLaunch::new(ctx.gpu, down_handle)
                    .grid([grid_down, 1, 1])
                    .block([V29_M256_BLOCK, 1, 1])
                    .shared_mem(down_smem)
                    .arg_ptr(aq)
                    .arg_ptr(as_)
                    .arg_ptr(down_table.packed_ptrs)
                    .arg_ptr(down_table.scale_ptrs)
                    .arg_ptr(down_table.scale2_vals)
                    .arg_ptr(ctx.buffers.expert_down_out())
                    .arg_ptr(offsets)
                    .arg_ptr(sorted_ids)
                    .arg_ptr(contract)
                    .arg_ptr(workspace)
                    .arg_ptr(status)
                    .arg_u32(2560)
                    .arg_u32(640)
                    .launch(stream)?;
                SLOT_STORAGE.with(|c| c.set(qwen4_prefill_compact::SlotStorage::Bf16));
            }
            spark_runtime::cublaslt::stage_timer::mark("compact_down", stream);
            if !V29_M256_RECEIPT_LOGGED.swap(true, Ordering::Relaxed) {
                tracing::info!(
                    "MOE_PREFILL_V29_M256_ENGAGED selector={} rows={rows} max_items={max_items_m256} grid_gu={grid_gateup} grid_dn={grid_down} block={V29_M256_BLOCK} ts4e={use_ts4e} slot_mxfp8={} smem_gu={gateup_smem} smem_dn={down_smem}",
                    qwen4_prefill_compact::V29_M256_SELECTOR,
                    slot_storage.is_mxfp8(),
                );
            }
            if use_ts4e && !TS4E_PREFILL_RECEIPT_LOGGED.swap(true, Ordering::Relaxed) {
                tracing::info!(
                    "MOE_PREFILL_TS4E128_ENGAGED smem_gateup={V29_M256_TS4E_GATEUP_SMEM_BYTES} smem_down={V29_M256_TS4E_DOWN_SMEM_BYTES} gateup=M256 down=M256 rows={rows}"
                );
            }
            if !W4A8_RECEIPT_LOGGED.swap(true, Ordering::Relaxed) {
                tracing::info!(
                    "MOE_PREFILL_W4A8_ENGAGED kernels={} rows={rows}",
                    wide.w4a8_tag
                );
            }
            return Ok(());
        }
        if v30_2cta_active && let Some(v30) = wide.v30_2cta {
            debug_assert!(!slot_storage.has_corrections());
            let grid_gateup = v30_2cta_grid_gateup(max_items);
            let grid_down = v30_2cta_grid_down(max_items);
            let (gateup_handle, gateup_smem) = if use_ts4e {
                (v30.gateup_ts4e, V30_2CTA_TS4E_GATEUP_SMEM_BYTES)
            } else {
                (v30.gateup_flat, V30_2CTA_FLAT_GATEUP_SMEM_BYTES)
            };
            KernelLaunch::new(ctx.gpu, gateup_handle)
                .grid([grid_gateup, 1, 1])
                .block([V30_2CTA_BLOCK, 1, 1])
                .shared_mem(gateup_smem)
                .arg_ptr(xq)
                .arg_ptr(xs)
                .arg_ptr(gate_table.packed_ptrs)
                .arg_ptr(gate_table.scale_ptrs)
                .arg_ptr(gate_table.scale2_vals)
                .arg_ptr(aq)
                .arg_ptr(as_)
                .arg_ptr(up_table.packed_ptrs)
                .arg_ptr(up_table.scale_ptrs)
                .arg_ptr(up_table.scale2_vals)
                .arg_ptr(offsets)
                .arg_ptr(sorted_ids)
                .arg_ptr(contract)
                .arg_ptr(workspace)
                .arg_ptr(status)
                .arg_u32(640)
                .arg_u32(2560)
                .launch(stream)?;
            spark_runtime::cublaslt::stage_timer::mark("compact_gateup", stream);
            let down_smem = if use_ts4e {
                V30_2CTA_TS4E_DOWN_SMEM_BYTES
            } else {
                V30_2CTA_FLAT_DOWN_SMEM_BYTES
            };
            if slot_storage.is_mxfp8() {
                let slots = ctx.buffers.expert_down_out();
                let layout = qwen4_prefill_compact::slot_layout(pairs, slot_storage)?;
                ensure!(
                    layout.used_bytes <= pairs * 2560 * 2,
                    "MXFP8 slot rows do not fit expert_down_out"
                );
                let slot_scales = slots.offset(
                    layout
                        .scales_offset
                        .context("MXFP8 slot layout omitted scales")?,
                );
                let down_handle = if use_ts4e {
                    v30.down_mxfp8_ts4e
                } else {
                    v30.down_mxfp8_flat
                };
                KernelLaunch::new(ctx.gpu, down_handle)
                    .grid([grid_down, 1, 1])
                    .block([V30_2CTA_BLOCK, 1, 1])
                    .shared_mem(down_smem)
                    .arg_ptr(aq)
                    .arg_ptr(as_)
                    .arg_ptr(down_table.packed_ptrs)
                    .arg_ptr(down_table.scale_ptrs)
                    .arg_ptr(down_table.scale2_vals)
                    .arg_ptr(slots)
                    .arg_ptr(slot_scales)
                    .arg_ptr(offsets)
                    .arg_ptr(sorted_ids)
                    .arg_ptr(contract)
                    .arg_ptr(workspace)
                    .arg_ptr(status)
                    .arg_u32(2560)
                    .arg_u32(640)
                    .launch(stream)?;
                SLOT_STORAGE.with(|c| c.set(slot_storage));
                if !SLOT_MXFP8_RECEIPT_LOGGED.swap(true, Ordering::Relaxed) {
                    let tag = if use_ts4e {
                        "v30q_2cta_ts4e"
                    } else {
                        "v30q_2cta"
                    };
                    tracing::info!("MOE_PREFILL_SLOT_MXFP8_ENGAGED kernels={tag} rows={rows}");
                }
            } else {
                let down_handle = if use_ts4e {
                    v30.down_bf16_ts4e
                } else {
                    v30.down_bf16_flat
                };
                KernelLaunch::new(ctx.gpu, down_handle)
                    .grid([grid_down, 1, 1])
                    .block([V30_2CTA_BLOCK, 1, 1])
                    .shared_mem(down_smem)
                    .arg_ptr(aq)
                    .arg_ptr(as_)
                    .arg_ptr(down_table.packed_ptrs)
                    .arg_ptr(down_table.scale_ptrs)
                    .arg_ptr(down_table.scale2_vals)
                    .arg_ptr(ctx.buffers.expert_down_out())
                    .arg_ptr(offsets)
                    .arg_ptr(sorted_ids)
                    .arg_ptr(contract)
                    .arg_ptr(workspace)
                    .arg_ptr(status)
                    .arg_u32(2560)
                    .arg_u32(640)
                    .launch(stream)?;
                SLOT_STORAGE.with(|c| c.set(qwen4_prefill_compact::SlotStorage::Bf16));
            }
            spark_runtime::cublaslt::stage_timer::mark("compact_down", stream);
            if !V30_2CTA_RECEIPT_LOGGED.swap(true, Ordering::Relaxed) {
                tracing::info!(
                    "MOE_PREFILL_V30_2CTA_ENGAGED selector={} rows={rows} max_items={max_items} grid_gu={grid_gateup} grid_dn={grid_down} block={V30_2CTA_BLOCK} ts4e={use_ts4e} slot_mxfp8={} smem_gu={gateup_smem} smem_dn={down_smem}",
                    qwen4_prefill_compact::V30_2CTA_SELECTOR,
                    slot_storage.is_mxfp8(),
                );
            }
            if use_ts4e && !TS4E_PREFILL_RECEIPT_LOGGED.swap(true, Ordering::Relaxed) {
                tracing::info!(
                    "MOE_PREFILL_TS4E128_ENGAGED smem_gateup={V30_2CTA_TS4E_GATEUP_SMEM_BYTES} smem_down={V30_2CTA_TS4E_DOWN_SMEM_BYTES} gateup=V30_2CTA down=V30_2CTA rows={rows}"
                );
            }
            if !W4A8_RECEIPT_LOGGED.swap(true, Ordering::Relaxed) {
                tracing::info!(
                    "MOE_PREFILL_W4A8_ENGAGED kernels={} rows={rows}",
                    wide.w4a8_tag
                );
            }
            return Ok(());
        }
        let v31_policy = qwen4_prefill_compact::v31_ws_policy()?;
        let v31_eligible = qwen4_prefill_compact::V31WsEligibility {
            rows,
            w4a8_is_v28: wide.w4a8_tag == "v28",
            v28_m64_forced: wide.v28_m64,
            has_corrections: slot_storage.has_corrections(),
            l2_windowed: l2_windows.is_some(),
            v29_m256_active: false,
            v30_2cta_active: false,
            v28q_level: wide.v28q_level,
            kernels_loaded: wide.v31_ws.is_some(),
        };
        if v31_policy.admits(v31_eligible)
            && let Some(v31) = wide.v31_ws
        {
            debug_assert!(!slot_storage.has_corrections());
            let grid_gateup = v31_ws_grid_gateup(max_items);
            let grid_down = v31_ws_grid_down(max_items);
            let (gateup_handle, gateup_smem) = if use_ts4e {
                (v31.gateup_ts4e, V31_WS_TS4E_GATEUP_SMEM_BYTES)
            } else {
                (v31.gateup_flat, V31_WS_FLAT_GATEUP_SMEM_BYTES)
            };
            KernelLaunch::new(ctx.gpu, gateup_handle)
                .grid([grid_gateup, 1, 1])
                .block([V31_WS_BLOCK, 1, 1])
                .shared_mem(gateup_smem)
                .arg_ptr(xq)
                .arg_ptr(xs)
                .arg_ptr(gate_table.packed_ptrs)
                .arg_ptr(gate_table.scale_ptrs)
                .arg_ptr(gate_table.scale2_vals)
                .arg_ptr(aq)
                .arg_ptr(as_)
                .arg_ptr(up_table.packed_ptrs)
                .arg_ptr(up_table.scale_ptrs)
                .arg_ptr(up_table.scale2_vals)
                .arg_ptr(offsets)
                .arg_ptr(sorted_ids)
                .arg_ptr(contract)
                .arg_ptr(workspace)
                .arg_ptr(status)
                .arg_u32(640)
                .arg_u32(2560)
                .launch(stream)?;
            spark_runtime::cublaslt::stage_timer::mark("compact_gateup", stream);
            let down_smem = if use_ts4e {
                V31_WS_TS4E_DOWN_SMEM_BYTES
            } else {
                V31_WS_FLAT_DOWN_SMEM_BYTES
            };
            if slot_storage.is_mxfp8() {
                let slots = ctx.buffers.expert_down_out();
                let layout = qwen4_prefill_compact::slot_layout(pairs, slot_storage)?;
                ensure!(
                    layout.used_bytes <= pairs * 2560 * 2,
                    "MXFP8 slot rows do not fit expert_down_out"
                );
                let slot_scales = slots.offset(
                    layout
                        .scales_offset
                        .context("MXFP8 slot layout omitted scales")?,
                );
                let down_handle = if use_ts4e {
                    v31.down_mxfp8_ts4e
                } else {
                    v31.down_mxfp8_flat
                };
                KernelLaunch::new(ctx.gpu, down_handle)
                    .grid([grid_down, 1, 1])
                    .block([V31_WS_BLOCK, 1, 1])
                    .shared_mem(down_smem)
                    .arg_ptr(aq)
                    .arg_ptr(as_)
                    .arg_ptr(down_table.packed_ptrs)
                    .arg_ptr(down_table.scale_ptrs)
                    .arg_ptr(down_table.scale2_vals)
                    .arg_ptr(slots)
                    .arg_ptr(slot_scales)
                    .arg_ptr(offsets)
                    .arg_ptr(sorted_ids)
                    .arg_ptr(contract)
                    .arg_ptr(workspace)
                    .arg_ptr(status)
                    .arg_u32(2560)
                    .arg_u32(640)
                    .launch(stream)?;
                SLOT_STORAGE.with(|c| c.set(slot_storage));
                if !SLOT_MXFP8_RECEIPT_LOGGED.swap(true, Ordering::Relaxed) {
                    let tag = if use_ts4e { "v31q_ws_ts4e" } else { "v31q_ws" };
                    tracing::info!("MOE_PREFILL_SLOT_MXFP8_ENGAGED kernels={tag} rows={rows}");
                }
            } else {
                let down_handle = if use_ts4e {
                    v31.down_bf16_ts4e
                } else {
                    v31.down_bf16_flat
                };
                KernelLaunch::new(ctx.gpu, down_handle)
                    .grid([grid_down, 1, 1])
                    .block([V31_WS_BLOCK, 1, 1])
                    .shared_mem(down_smem)
                    .arg_ptr(aq)
                    .arg_ptr(as_)
                    .arg_ptr(down_table.packed_ptrs)
                    .arg_ptr(down_table.scale_ptrs)
                    .arg_ptr(down_table.scale2_vals)
                    .arg_ptr(ctx.buffers.expert_down_out())
                    .arg_ptr(offsets)
                    .arg_ptr(sorted_ids)
                    .arg_ptr(contract)
                    .arg_ptr(workspace)
                    .arg_ptr(status)
                    .arg_u32(2560)
                    .arg_u32(640)
                    .launch(stream)?;
                SLOT_STORAGE.with(|c| c.set(qwen4_prefill_compact::SlotStorage::Bf16));
            }
            spark_runtime::cublaslt::stage_timer::mark("compact_down", stream);
            if !V31_WS_RECEIPT_LOGGED.swap(true, Ordering::Relaxed) {
                tracing::info!(
                    "MOE_PREFILL_V31_WS_ENGAGED selector={} rows={rows} max_items={max_items} grid_gu={grid_gateup} grid_dn={grid_down} block={V31_WS_BLOCK} ts4e={use_ts4e} slot_mxfp8={} smem_gu={gateup_smem} smem_dn={down_smem}",
                    qwen4_prefill_compact::V31_WS_SELECTOR,
                    slot_storage.is_mxfp8(),
                );
            }
            if use_ts4e && !TS4E_PREFILL_RECEIPT_LOGGED.swap(true, Ordering::Relaxed) {
                tracing::info!(
                    "MOE_PREFILL_TS4E128_ENGAGED smem_gateup={V31_WS_TS4E_GATEUP_SMEM_BYTES} smem_down={V31_WS_TS4E_DOWN_SMEM_BYTES} gateup=V31_WS down=V31_WS rows={rows}"
                );
            }
            if !W4A8_RECEIPT_LOGGED.swap(true, Ordering::Relaxed) {
                tracing::info!(
                    "MOE_PREFILL_W4A8_ENGAGED kernels={} rows={rows}",
                    wide.w4a8_tag
                );
            }
            return Ok(());
        }
        let ts4e_supported = v28_ts4e_supported(
            wide.w4a8_tag,
            wide.v28q_level,
            wide.v28_m64,
            slot_storage.has_corrections(),
        );
        if use_ts4e {
            ensure!(
                ts4e_supported,
                "Atlas-TS4E-128 compact prefill requires v28 W4A8 (twarp or m64) without R2 corrections (tag={}, v28q_level={}, m64={}, r2={}); set ATLAS_MOE_SCALE_TS4E=0",
                wide.w4a8_tag,
                wide.v28q_level,
                wide.v28_m64,
                slot_storage.has_corrections()
            );
        }
        let m32_active = v28_m32_active_for_routed(V28M32RoutedContext {
            m32_enabled: v28_m32_enabled(),
            scope_active: v28_m32_active_for_scope(),
            short_prefill_enabled: v28_m32_short_prefill_enabled(),
            use_ts4e,
            rows,
            has_corrections: slot_storage.has_corrections(),
            dump_or_oracle: super::dump::enabled() || crate::layers::qwen4_oracle::root().is_some(),
        });
        let v28_plan = if l2_windows.is_none() {
            select_v28_dispatch(
                wide.w4a8_tag,
                wide.v28_m64,
                m32_active,
                rows,
                slot_storage.has_corrections(),
            )
        } else {
            V28DispatchPlan {
                gateup: V28OpTile::Twarp,
                down: V28OpTile::Twarp,
            }
        };
        let m16_active = l2_windows.is_none()
            && v28_m16_active_for_routed(
                v28_m16_decode_enabled(),
                m32_active,
                rows,
                slot_storage.has_corrections(),
            );
        if let Some(windows) = l2_windows {
            let (win_k, gateup_smem, down_smem) = if use_ts4e {
                (
                    windowed_twarp_ts4e_kernels(ctx.gpu)?,
                    V28_TS4E_GATEUP_SMEM_BYTES,
                    V28_TS4E_DOWN_SMEM_BYTES,
                )
            } else {
                let down_smem = wide.w4a8_smem_gemm.max(wide.w4a8_smem_down_mxfp8);
                (
                    windowed_twarp_kernels(ctx.gpu, wide.w4a8_smem, down_smem)?,
                    wide.w4a8_smem,
                    down_smem,
                )
            };
            let slots = ctx.buffers.expert_down_out();
            // `v28_m64_impl` down writes every valid expanded row via direct disjoint scatter
            // (no atomicAdd across windows), so `slots` requires no per-window zeroing.
            let mxfp8_scales = if slot_storage.is_mxfp8() {
                let layout = qwen4_prefill_compact::slot_layout(pairs, slot_storage)?;
                ensure!(
                    layout.used_bytes <= pairs * 2560 * 2,
                    "MXFP8 slot rows do not fit expert_down_out"
                );
                Some(
                    slots.offset(
                        layout
                            .scales_offset
                            .context("MXFP8 slot layout omitted scales")?,
                    ),
                )
            } else {
                None
            };
            let min_grid = active_sm_grid_1x();
            for &(wi_b, wi_e) in &windows {
                let win_items = wi_e - wi_b;
                let win_grid_gateup = (win_items * 5).clamp(min_grid, abi::GRID as usize) as u32;
                let win_grid_down = (win_items * 10).clamp(min_grid, abi::GRID as usize) as u32;
                let wi_b_u32 = u32::try_from(wi_b).context("L2 window wi_begin exceeds u32")?;
                let wi_e_u32 = u32::try_from(wi_e).context("L2 window wi_end exceeds u32")?;
                KernelLaunch::new(ctx.gpu, win_k.gateup)
                    .grid([win_grid_gateup, 1, 1])
                    .block([wide.w4a8_block, 1, 1])
                    .shared_mem(gateup_smem)
                    .arg_ptr(xq)
                    .arg_ptr(xs)
                    .arg_ptr(gate_table.packed_ptrs)
                    .arg_ptr(gate_table.scale_ptrs)
                    .arg_ptr(gate_table.scale2_vals)
                    .arg_ptr(aq)
                    .arg_ptr(as_)
                    .arg_ptr(up_table.packed_ptrs)
                    .arg_ptr(up_table.scale_ptrs)
                    .arg_ptr(up_table.scale2_vals)
                    .arg_ptr(offsets)
                    .arg_ptr(sorted_ids)
                    .arg_ptr(contract)
                    .arg_ptr(workspace)
                    .arg_ptr(status)
                    .arg_u32(640)
                    .arg_u32(2560)
                    .arg_u32(wi_b_u32)
                    .arg_u32(wi_e_u32)
                    .launch(stream)?;
                spark_runtime::cublaslt::stage_timer::mark("compact_gateup", stream);
                if let Some(slot_scales) = mxfp8_scales {
                    KernelLaunch::new(ctx.gpu, win_k.down_mxfp8)
                        .grid([win_grid_down, 1, 1])
                        .block([wide.w4a8_block, 1, 1])
                        .shared_mem(down_smem)
                        .arg_ptr(aq)
                        .arg_ptr(as_)
                        .arg_ptr(down_table.packed_ptrs)
                        .arg_ptr(down_table.scale_ptrs)
                        .arg_ptr(down_table.scale2_vals)
                        .arg_ptr(slots)
                        .arg_ptr(slot_scales)
                        .arg_ptr(offsets)
                        .arg_ptr(sorted_ids)
                        .arg_ptr(contract)
                        .arg_ptr(workspace)
                        .arg_ptr(status)
                        .arg_u32(2560)
                        .arg_u32(640)
                        .arg_u32(wi_b_u32)
                        .arg_u32(wi_e_u32)
                        .launch(stream)?;
                } else {
                    KernelLaunch::new(ctx.gpu, win_k.down_bf16)
                        .grid([win_grid_down, 1, 1])
                        .block([wide.w4a8_block, 1, 1])
                        .shared_mem(down_smem)
                        .arg_ptr(aq)
                        .arg_ptr(as_)
                        .arg_ptr(down_table.packed_ptrs)
                        .arg_ptr(down_table.scale_ptrs)
                        .arg_ptr(down_table.scale2_vals)
                        .arg_ptr(slots)
                        .arg_ptr(offsets)
                        .arg_ptr(sorted_ids)
                        .arg_ptr(contract)
                        .arg_ptr(workspace)
                        .arg_ptr(status)
                        .arg_u32(2560)
                        .arg_u32(640)
                        .arg_u32(wi_b_u32)
                        .arg_u32(wi_e_u32)
                        .launch(stream)?;
                }
                spark_runtime::cublaslt::stage_timer::mark("compact_down", stream);
            }
            if mxfp8_scales.is_some() {
                SLOT_STORAGE.with(|c| c.set(slot_storage));
                if !SLOT_MXFP8_RECEIPT_LOGGED.swap(true, Ordering::Relaxed) {
                    let tag = if rows <= 4096 { "v28q" } else { "v28q_pf" };
                    tracing::info!("MOE_PREFILL_SLOT_MXFP8_ENGAGED kernels={tag} rows={rows}");
                }
            } else {
                SLOT_STORAGE.with(|c| c.set(qwen4_prefill_compact::SlotStorage::Bf16));
            }
            if !L2_WINDOW_RECEIPT_LOGGED.swap(true, Ordering::Relaxed) {
                tracing::info!(
                    "MOE_PREFILL_L2_WINDOW_ENGAGED selector={L2_WINDOW_SELECTOR} rows={rows} max_items={max_items} window_items={} windows={}",
                    l2_window_items.map_or(0, |v| v),
                    windows.len()
                );
            }
        } else {
            let use_v28q_nopf = wide.v28q_level == 2 && rows <= 4096;
            let (grid_gateup, block_gateup, gateup_handle, gateup_smem) = match v28_plan.gateup {
                V28OpTile::M32 => {
                    if m16_active {
                        let (handle, smem) = if use_ts4e {
                            (
                                v28_m16_ts4e_kernels(ctx.gpu)?.gateup,
                                V28_M16_TS4E_GATEUP_SMEM_BYTES,
                            )
                        } else {
                            (
                                v28_m16_flat_kernels(ctx.gpu)?.gateup,
                                V28_M16_FLAT_GATEUP_SMEM_BYTES,
                            )
                        };
                        (
                            v28_m16_grid_gateup(max_items, v28_m32_max_grid()),
                            V28_M16_BLOCK,
                            handle,
                            smem,
                        )
                    } else {
                        let (handle, smem) = if use_ts4e {
                            (
                                v28_m32_ts4e_kernels(ctx.gpu)?.gateup,
                                V28_M32_TS4E_GATEUP_SMEM_BYTES,
                            )
                        } else {
                            (
                                v28_m32_flat_kernels(ctx.gpu)?.gateup,
                                V28_M32_FLAT_SMEM_BYTES,
                            )
                        };
                        (
                            v28_m32_grid_gateup(max_items, v28_m32_max_grid()),
                            V28_M32_BLOCK,
                            handle,
                            smem,
                        )
                    }
                }
                V28OpTile::M64 => {
                    let (handle, smem) = if use_ts4e {
                        (
                            twarp_m64_ts4e_kernels(ctx.gpu)?.gateup,
                            V28_TS4E_GATEUP_SMEM_BYTES,
                        )
                    } else if !wide.v28_m64 {
                        (v28_m64_flat_kernels(ctx.gpu)?.gateup, V28_FLAT_SMEM_BYTES)
                    } else {
                        (wide.w4a8_gateup, wide.w4a8_smem)
                    };
                    (
                        (max_items * 5).clamp(active_sm_grid_1x(), abi::GRID as usize) as u32,
                        wide.w4a8_block,
                        handle,
                        smem,
                    )
                }
                V28OpTile::Twarp => {
                    let (handle, smem) = if use_ts4e {
                        (
                            twarp_ts4e_kernels(ctx.gpu)?.gateup,
                            V28_TS4E_GATEUP_SMEM_BYTES,
                        )
                    } else if use_v28q_nopf && let Some(nopf) = wide.w4a8_gateup_nopf {
                        (nopf, wide.w4a8_smem_nopf)
                    } else {
                        (wide.w4a8_gateup, wide.w4a8_smem)
                    };
                    (
                        w4a8_twarp_grid_gateup_for_sm(
                            wide.w4a8_tag,
                            max_items,
                            atlas_core::device::active_sm_count(),
                        ),
                        wide.w4a8_block,
                        handle,
                        smem,
                    )
                }
            };
            KernelLaunch::new(ctx.gpu, gateup_handle)
                .grid([grid_gateup, 1, 1])
                .block([block_gateup, 1, 1])
                .shared_mem(gateup_smem)
                .arg_ptr(xq)
                .arg_ptr(xs)
                .arg_ptr(gate_table.packed_ptrs)
                .arg_ptr(gate_table.scale_ptrs)
                .arg_ptr(gate_table.scale2_vals)
                .arg_ptr(aq)
                .arg_ptr(as_)
                .arg_ptr(up_table.packed_ptrs)
                .arg_ptr(up_table.scale_ptrs)
                .arg_ptr(up_table.scale2_vals)
                .arg_ptr(offsets)
                .arg_ptr(sorted_ids)
                .arg_ptr(contract)
                .arg_ptr(workspace)
                .arg_ptr(status)
                .arg_u32(640)
                .arg_u32(2560)
                .launch(stream)?;
            spark_runtime::cublaslt::stage_timer::mark("compact_gateup", stream);
            let slot_down = if slot_storage.is_mxfp8() {
                match v28_plan.down {
                    V28OpTile::M32 if m16_active => Some(if use_ts4e {
                        (
                            v28_m16_ts4e_kernels(ctx.gpu)?.down_mxfp8,
                            V28_M16_TS4E_DOWN_SMEM_BYTES,
                        )
                    } else {
                        (
                            v28_m16_flat_kernels(ctx.gpu)?.down_mxfp8,
                            V28_M16_FLAT_DOWN_SMEM_BYTES,
                        )
                    }),
                    V28OpTile::M32 => Some(if use_ts4e {
                        (
                            v28_m32_ts4e_kernels(ctx.gpu)?.down_mxfp8,
                            V28_M32_TS4E_DOWN_SMEM_BYTES,
                        )
                    } else {
                        (
                            v28_m32_flat_kernels(ctx.gpu)?.down_mxfp8,
                            V28_M32_FLAT_SMEM_BYTES,
                        )
                    }),
                    V28OpTile::M64 => {
                        if use_ts4e {
                            Some((
                                twarp_m64_ts4e_kernels(ctx.gpu)?.down_mxfp8,
                                V28_TS4E_DOWN_SMEM_BYTES,
                            ))
                        } else if !wide.v28_m64 {
                            Some((
                                v28_m64_flat_kernels(ctx.gpu)?.down_mxfp8,
                                V28_FLAT_SMEM_BYTES,
                            ))
                        } else {
                            wide.w4a8_down_mxfp8.map(|k| (k, wide.w4a8_smem_down_mxfp8))
                        }
                    }
                    V28OpTile::Twarp => {
                        if use_ts4e {
                            Some((
                                twarp_ts4e_kernels(ctx.gpu)?.down_mxfp8,
                                V28_TS4E_DOWN_SMEM_BYTES,
                            ))
                        } else if use_v28q_nopf {
                            wide.w4a8_down_mxfp8_nopf
                                .map(|k| (k, wide.w4a8_smem_down_mxfp8_nopf))
                                .or_else(|| {
                                    wide.w4a8_down_mxfp8.map(|k| (k, wide.w4a8_smem_down_mxfp8))
                                })
                        } else {
                            wide.w4a8_down_mxfp8.map(|k| (k, wide.w4a8_smem_down_mxfp8))
                        }
                    }
                }
            } else {
                None
            };
            let (grid_down, block_down) = match v28_plan.down {
                V28OpTile::M32 if m16_active => (
                    v28_m16_grid_down(max_items, v28_m32_max_grid()),
                    V28_M16_BLOCK,
                ),
                V28OpTile::M32 => (
                    v28_m32_grid_down(max_items, v28_m32_max_grid()),
                    V28_M32_BLOCK,
                ),
                V28OpTile::M64 | V28OpTile::Twarp => (
                    w4a8_twarp_grid_down_for_sm(
                        wide.w4a8_tag,
                        slot_down.is_some(),
                        max_items,
                        atlas_core::device::active_sm_count(),
                    ),
                    wide.w4a8_block,
                ),
            };
            if let Some((down, down_smem)) = slot_down {
                let slots = ctx.buffers.expert_down_out();
                let layout = qwen4_prefill_compact::slot_layout(pairs, slot_storage)?;
                ensure!(
                    layout.used_bytes <= pairs * 2560 * 2,
                    "MXFP8 slot rows do not fit expert_down_out"
                );
                let slot_scales = slots.offset(
                    layout
                        .scales_offset
                        .context("MXFP8 slot layout omitted scales")?,
                );
                let slot_corrections = layout
                    .corrections_offset
                    .map_or(DevicePtr::NULL, |offset| slots.offset(offset));
                if slot_storage.has_corrections() {
                    crate::model::qwen4_prefill_engagement::preflight_slot_r2_producer(rows)?;
                }
                let launch = KernelLaunch::new(ctx.gpu, down)
                    .grid([grid_down, 1, 1])
                    .block([block_down, 1, 1])
                    .shared_mem(down_smem)
                    .arg_ptr(aq)
                    .arg_ptr(as_)
                    .arg_ptr(down_table.packed_ptrs)
                    .arg_ptr(down_table.scale_ptrs)
                    .arg_ptr(down_table.scale2_vals)
                    .arg_ptr(slots)
                    .arg_ptr(slot_scales);
                let launch = if slot_storage.has_corrections() {
                    launch.arg_ptr(slot_corrections)
                } else {
                    launch
                };
                launch
                    .arg_ptr(offsets)
                    .arg_ptr(sorted_ids)
                    .arg_ptr(contract)
                    .arg_ptr(workspace)
                    .arg_ptr(status)
                    .arg_u32(2560)
                    .arg_u32(640)
                    .launch(stream)?;
                if slot_storage.has_corrections() {
                    crate::model::qwen4_prefill_engagement::commit_slot_r2_producer();
                }
                SLOT_STORAGE.with(|c| c.set(slot_storage));
                let (receipt, kernel) = if slot_storage.has_corrections() {
                    (&SLOT_MXFP8_R2_RECEIPT_LOGGED, "v17qr2")
                } else {
                    let tag = match (v28_plan.down, wide.v28q_level) {
                        (V28OpTile::M32, _) if m16_active => "v28q_m16",
                        (V28OpTile::M32, _) => "v28q_m32",
                        (V28OpTile::M64, _) => "v28q_m64",
                        (V28OpTile::Twarp, 1 | 4) => "v28q",
                        (V28OpTile::Twarp, 2) if use_v28q_nopf => "v28q",
                        (V28OpTile::Twarp, 2) => "v28q_pf",
                        (V28OpTile::Twarp, _) => "v17q",
                    };
                    (&SLOT_MXFP8_RECEIPT_LOGGED, tag)
                };
                if !receipt.swap(true, Ordering::Relaxed) {
                    tracing::info!("MOE_PREFILL_SLOT_MXFP8_ENGAGED kernels={kernel} rows={rows}");
                }
            } else {
                let (down_bf16_k, down_bf16_smem) = match v28_plan.down {
                    V28OpTile::M32 if m16_active => {
                        if use_ts4e {
                            (
                                v28_m16_ts4e_kernels(ctx.gpu)?.down_bf16,
                                V28_M16_TS4E_DOWN_SMEM_BYTES,
                            )
                        } else {
                            (
                                v28_m16_flat_kernels(ctx.gpu)?.down_bf16,
                                V28_M16_FLAT_DOWN_SMEM_BYTES,
                            )
                        }
                    }
                    V28OpTile::M32 => {
                        if use_ts4e {
                            (
                                v28_m32_ts4e_kernels(ctx.gpu)?.down_bf16,
                                V28_M32_TS4E_DOWN_SMEM_BYTES,
                            )
                        } else {
                            (
                                v28_m32_flat_kernels(ctx.gpu)?.down_bf16,
                                V28_M32_FLAT_SMEM_BYTES,
                            )
                        }
                    }
                    V28OpTile::M64 => {
                        if use_ts4e {
                            (
                                twarp_m64_ts4e_kernels(ctx.gpu)?.down_bf16,
                                V28_TS4E_DOWN_SMEM_BYTES,
                            )
                        } else if !wide.v28_m64 {
                            (
                                v28_m64_flat_kernels(ctx.gpu)?.down_bf16,
                                V28_FLAT_SMEM_BYTES,
                            )
                        } else {
                            (wide.w4a8_gemm, wide.w4a8_smem_gemm)
                        }
                    }
                    V28OpTile::Twarp => {
                        if use_ts4e {
                            (
                                twarp_ts4e_kernels(ctx.gpu)?.down_bf16,
                                V28_TS4E_DOWN_SMEM_BYTES,
                            )
                        } else {
                            (wide.w4a8_gemm, wide.w4a8_smem_gemm)
                        }
                    }
                };
                KernelLaunch::new(ctx.gpu, down_bf16_k)
                    .grid([grid_down, 1, 1])
                    .block([block_down, 1, 1])
                    .shared_mem(down_bf16_smem)
                    .arg_ptr(aq)
                    .arg_ptr(as_)
                    .arg_ptr(down_table.packed_ptrs)
                    .arg_ptr(down_table.scale_ptrs)
                    .arg_ptr(down_table.scale2_vals)
                    .arg_ptr(ctx.buffers.expert_down_out())
                    .arg_ptr(offsets)
                    .arg_ptr(sorted_ids)
                    .arg_ptr(contract)
                    .arg_ptr(workspace)
                    .arg_ptr(status)
                    .arg_u32(2560)
                    .arg_u32(640)
                    .launch(stream)?;
                SLOT_STORAGE.with(|c| c.set(qwen4_prefill_compact::SlotStorage::Bf16));
            }
            spark_runtime::cublaslt::stage_timer::mark("compact_down", stream);
        }
        let m32_or_m16_dispatched =
            v28_plan.gateup == V28OpTile::M32 || v28_plan.down == V28OpTile::M32;
        if m32_or_m16_dispatched
            && m16_active
            && !V28_M16_RECEIPT_LOGGED.swap(true, Ordering::Relaxed)
        {
            tracing::info!(
                "MOE_PREFILL_V28_M16_ENGAGED selector={V28_M16_DECODE_SELECTOR} ts4e={use_ts4e} rows={rows} block={V28_M16_BLOCK} smem_gu={} smem_dn={}",
                if use_ts4e {
                    V28_M16_TS4E_GATEUP_SMEM_BYTES
                } else {
                    V28_M16_FLAT_GATEUP_SMEM_BYTES
                },
                if use_ts4e {
                    V28_M16_TS4E_DOWN_SMEM_BYTES
                } else {
                    V28_M16_FLAT_DOWN_SMEM_BYTES
                }
            );
        }
        if m32_or_m16_dispatched
            && !m16_active
            && !V28_M32_RECEIPT_LOGGED.swap(true, Ordering::Relaxed)
        {
            tracing::info!(
                "MOE_PREFILL_V28_M32_ENGAGED gateup={:?} down={:?} ts4e={use_ts4e} rows={rows}",
                v28_plan.gateup,
                v28_plan.down
            );
        }
        if use_ts4e && !TS4E_PREFILL_RECEIPT_LOGGED.swap(true, Ordering::Relaxed) {
            let (ts_gu_smem, ts_dn_smem) = ts4e_v28_receipt_smem(v28_plan, m16_active);
            tracing::info!(
                "MOE_PREFILL_TS4E128_ENGAGED smem_gateup={ts_gu_smem} smem_down={ts_dn_smem} gateup={:?} down={:?} m16={m16_active} rows={rows}",
                v28_plan.gateup,
                v28_plan.down
            );
        }
        if !W4A8_RECEIPT_LOGGED.swap(true, Ordering::Relaxed) {
            tracing::info!(
                "MOE_PREFILL_W4A8_ENGAGED kernels={} rows={rows}",
                wide.w4a8_tag
            );
        }
        Ok(())
    }

    /// Record one compact-MoE dispatch to `ATLAS_QWEN4_ORACLE_DUMP` (no-op when
    /// the variable is unset or this tag was already captured).
    ///
    /// Everything a standalone replay needs is written: the gathered BF16
    /// activation tile, the routing metadata and planner arena, the three expert
    /// pointer tables WITH the per-expert NVFP4 weight blobs they address, and
    /// the three result buffers.
    #[allow(clippy::too_many_arguments)]
    fn oracle_capture_compact(
        &self,
        input: DevicePtr,
        offsets: DevicePtr,
        sorted_ids: DevicePtr,
        rows: usize,
        gate_table: &ExpertPtrTable,
        up_table: &ExpertPtrTable,
        down_table: &ExpertPtrTable,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        use crate::layers::qwen4_oracle::Sink;
        if crate::layers::qwen4_oracle::root().is_some() {
            ensure!(
                !current_slot_storage().is_mxfp8(),
                "compact oracle does not label MXFP8 slot payloads as BF16; use the dedicated R2 gate"
            );
            ensure!(
                self.routed_scale_layout == RoutedScaleLayout::FlatE4m3,
                "compact oracle capture requires FlatE4m3 scales; set ATLAS_MOE_SCALE_TS4E=0"
            );
        }
        let Some(mut sink) = Sink::open("moe_compact") else {
            return Ok(());
        };
        ctx.gpu.synchronize(stream)?;
        let experts = ctx.config.num_experts;
        let expanded = rows * 10;
        sink.scalar("rows", rows);
        sink.scalar("top_k", 10);
        sink.scalar("num_experts", experts);
        sink.scalar("hidden", 2560);
        sink.scalar("inter", 640);
        sink.scalar("grid", abi::GRID);
        // Inputs: activations and the routing/plan metadata.
        sink.buf(ctx.gpu, "input_bf16", input, rows * 2560 * 2)?;
        sink.buf(ctx.gpu, "offsets_u32", offsets, 513 * 4)?;
        sink.buf(ctx.gpu, "sorted_ids_u32", sorted_ids, expanded * 4)?;
        sink.buf(
            ctx.gpu,
            "contract",
            ctx.buffers.moe_worklist().offset(abi::WORKSPACE_BYTES),
            abi::ARENA_BYTES - abi::WORKSPACE_BYTES,
        )?;
        sink.buf(
            ctx.gpu,
            "workspace",
            ctx.buffers.moe_worklist(),
            abi::WORKSPACE_BYTES,
        )?;
        // Expert weights, followed through each pointer table. gate/up are
        // [640 x 2560] and down is [2560 x 640]; both pack to K*N/2 bytes with
        // one e4m3 scale byte per 16 elements.
        for (name, table, n, k) in [
            ("gate", gate_table, 640usize, 2560usize),
            ("up", up_table, 640, 2560),
            ("down", down_table, 2560, 640),
        ] {
            let packed = sink.read_ptr_table(ctx.gpu, table.packed_ptrs, experts)?;
            let scales = sink.read_ptr_table(ctx.gpu, table.scale_ptrs, experts)?;
            sink.blobs(ctx.gpu, &format!("{name}_packed"), &packed, n * k / 2)?;
            sink.blobs(ctx.gpu, &format!("{name}_scale"), &scales, n * k / 16)?;
            sink.buf(
                ctx.gpu,
                &format!("{name}_scale2_f32"),
                table.scale2_vals,
                experts * 4,
            )?;
        }
        // Results, after all three GEMMs have completed.
        sink.buf(
            ctx.gpu,
            "out_gate_activated_bf16",
            ctx.buffers.expert_gate_out(),
            expanded * 640 * 2,
        )?;
        sink.buf(
            ctx.gpu,
            "out_up_bf16",
            ctx.buffers.expert_up_out(),
            (expanded * 640 * 2).min(ctx.buffers.sizes().expert_up_out),
        )?;
        sink.buf(
            ctx.gpu,
            "out_down_bf16",
            ctx.buffers.expert_down_out(),
            expanded * 2560 * 2,
        )?;
        sink.finish();
        Ok(())
    }
}

#[cfg(test)]
mod slot_state_tests {
    use super::*;

    #[test]
    fn stale_slot_storage_is_detected_and_cleared_before_reuse() {
        SLOT_STORAGE.with(|cell| cell.set(qwen4_prefill_compact::SlotStorage::Mxfp8R2));
        assert!(discard_stale_slot_storage());
        assert_eq!(
            take_slot_storage(),
            qwen4_prefill_compact::SlotStorage::Bf16
        );
        assert!(!discard_stale_slot_storage());
    }

    #[test]
    fn oracle_rejects_mxfp8_before_opening_the_one_shot_sink() {
        let source = include_str!("qwen4_compact_ops.rs");
        let start = source.find("pub(super) fn run_qwen4_compact(").unwrap();
        let body = &source[start..];
        let preflight = body.find("!dispatched_slot_storage.is_mxfp8()").unwrap();
        let first_launch = body.find(".launch(stream)?;").unwrap();
        let capture = body.find("self.oracle_capture_compact(").unwrap();
        assert!(preflight < first_launch && first_launch < capture);

        let capture_body = &source[source.find("fn oracle_capture_compact(").unwrap()..];
        let repeat_guard = capture_body
            .find("current_slot_storage().is_mxfp8()")
            .unwrap();
        let open = capture_body.find("Sink::open(\"moe_compact\")").unwrap();
        assert!(repeat_guard < open);
    }

    #[test]
    fn plan_l2_windows_covers_work_count_contiguously_with_zero_gaps_or_overlaps() {
        assert!(plan_l2_windows(0, 223).is_empty());
        assert!(plan_l2_windows(0, 0).is_empty());
        assert_eq!(plan_l2_windows(1, 223), vec![(0, 1)]);
        assert_eq!(plan_l2_windows(1, 1), vec![(0, 1)]);
        assert_eq!(plan_l2_windows(223, 223), vec![(0, 223)]);
        assert_eq!(plan_l2_windows(446, 223), vec![(0, 223), (223, 446)]);
        assert_eq!(
            plan_l2_windows(500, 223),
            vec![(0, 223), (223, 446), (446, 500)]
        );
        assert_eq!(plan_l2_windows(1792, 0), vec![(0, 1792)]);

        for (work_count, max_window_items) in [
            (1, 1),
            (7, 3),
            (12, 4),
            (832, 223),
            (1152, 223),
            (1792, 223),
            (1792, 256),
            (5632, 223),
        ] {
            let windows = plan_l2_windows(work_count, max_window_items);
            assert!(
                !windows.is_empty(),
                "work_count={work_count}, max={max_window_items}"
            );
            assert_eq!(windows.first().map(|&(b, _)| b), Some(0));
            assert_eq!(windows.last().map(|&(_, e)| e), Some(work_count));
            let mut cursor = 0;
            for &(begin, end) in &windows {
                assert_eq!(begin, cursor, "gap or overlap at {begin} != {cursor}");
                assert!(begin < end, "empty window ({begin}, {end})");
                assert!(
                    end - begin <= max_window_items,
                    "window ({begin}, {end}) exceeds cap {max_window_items}"
                );
                cursor = end;
            }
            assert_eq!(cursor, work_count);
        }
    }

    #[test]
    fn l2_window_policy_defaults_off_and_explicit_one_engages_for_large_chunks() {
        let default_policy = parse_l2_window_policy(None).expect("unset policy");
        assert_eq!(default_policy, L2WindowPolicy::Disabled);
        assert_eq!(resolve_l2_window_items(default_policy, 16384 * 10), None);
        // `0` also disables completely.
        let disabled = parse_l2_window_policy(Some("0")).expect("disable policy");
        assert_eq!(disabled, L2WindowPolicy::Disabled);
        assert_eq!(resolve_l2_window_items(disabled, 16384 * 10), None);
        // `1` opts in to auto policy for large chunks (> 24 MiB expanded `oq`).
        let auto_policy = parse_l2_window_policy(Some("1")).expect("auto policy");
        assert_eq!(
            auto_policy,
            L2WindowPolicy::Auto {
                max_window_items: DEFAULT_L2_WINDOW_ITEMS
            }
        );
        const {
            assert!(DEFAULT_L2_WINDOW_ITEMS * L2_BYTES_PER_WORK_ITEM <= L2_WINDOW_TARGET_BYTES);
        }
        assert_eq!(resolve_l2_window_items(auto_policy, 2048 * 10), None);
        assert_eq!(resolve_l2_window_items(auto_policy, 3072 * 10), None);
        assert_eq!(
            resolve_l2_window_items(auto_policy, 4096 * 10),
            Some(DEFAULT_L2_WINDOW_ITEMS)
        );
        assert_eq!(
            resolve_l2_window_items(auto_policy, 8192 * 10),
            Some(DEFAULT_L2_WINDOW_ITEMS)
        );
        assert_eq!(
            resolve_l2_window_items(auto_policy, 16384 * 10),
            Some(DEFAULT_L2_WINDOW_ITEMS)
        );
        // Custom item count and MiB suffix.
        let fixed = parse_l2_window_policy(Some("192")).expect("fixed items");
        assert_eq!(
            fixed,
            L2WindowPolicy::Fixed {
                max_window_items: 192
            }
        );
        assert_eq!(resolve_l2_window_items(fixed, 1024 * 10), Some(192));
        let mib = parse_l2_window_policy(Some("16m")).expect("MiB target");
        assert_eq!(
            mib,
            L2WindowPolicy::Auto {
                max_window_items: (16 * 1024 * 1024) / L2_BYTES_PER_WORK_ITEM
            }
        );
        assert!(parse_l2_window_policy(Some("bad")).is_err());
    }

    #[test]
    fn v28_m64_ts4e_dispatch_selects_m64_for_small_batches_and_explicit_flag() {
        // Small batches (C=4..32 verify: rows=20..256) engage m64 automatically on v28.
        for rows in [16, 20, 40, 64, 80, 128, 160, 256] {
            assert_eq!(
                select_v28_m64_dispatch("v28", false, rows, false),
                V28M64Dispatch::M64,
                "rows={rows} should engage m64 on v28"
            );
        }
        // Large prefill chunks stay on twarp unless v28_m64 is explicitly selected.
        for rows in [257, 512, 2048, 4096, 8192] {
            assert_eq!(
                select_v28_m64_dispatch("v28", false, rows, false),
                V28M64Dispatch::Twarp,
                "rows={rows} without explicit flag should keep twarp"
            );
            assert_eq!(
                select_v28_m64_dispatch("v28", true, rows, false),
                V28M64Dispatch::M64,
                "rows={rows} with explicit flag should select m64"
            );
        }
        // Non-v28 sets and R2 corrections never select v28_m64.
        for tag in ["v10", "v12", "v14", "v16"] {
            assert_eq!(
                select_v28_m64_dispatch(tag, true, 20, false),
                V28M64Dispatch::Twarp,
                "non-v28 tag={tag}"
            );
        }
        assert_eq!(
            select_v28_m64_dispatch("v28", true, 20, true),
            V28M64Dispatch::Twarp
        );
    }

    #[test]
    fn v28_ts4e_support_admits_twarp_and_m64_and_rejects_r2_or_non_v28() {
        assert!(v28_ts4e_supported("v28", 2, false, false));
        assert!(v28_ts4e_supported("v28", 2, true, false));
        assert!(v28_ts4e_supported("v28", 0, true, false));
        assert!(!v28_ts4e_supported("v28", 0, false, false));
        assert!(!v28_ts4e_supported("v28", 1, false, false));
        assert!(!v28_ts4e_supported("v28", 2, true, true));
        assert!(!v28_ts4e_supported("v16", 2, true, false));
        assert_eq!(V28_TS4E_GATEUP_SMEM_BYTES, 91_072);
        assert_eq!(V28_TS4E_DOWN_SMEM_BYTES, 90_352);
        assert_eq!(V28_TS4E_GATEUP_SMEM_BYTES - V28_FLAT_SMEM_BYTES, 1_984);
        assert_eq!(V28_TS4E_DOWN_SMEM_BYTES - V28_FLAT_SMEM_BYTES, 1_264);
        const {
            assert!(V28_TS4E_GATEUP_SMEM_BYTES <= 101_376);
            assert!(V28_TS4E_DOWN_SMEM_BYTES <= 101_376);
        }
    }

    #[test]
    fn v28_m32_dispatch_covers_small_batches_and_ks10_down_within_2cta_smem_cap() {
        // 2 CTAs/SM on GB10 (99 KiB SMEM => <= 48 KiB = 49,152 B per CTA).
        assert_eq!(V28_M32_BLOCK, 256);
        assert_eq!(V28_M32_FLAT_SMEM_BYTES, 39_680);
        assert_eq!(V28_M32_TS4E_GATEUP_SMEM_BYTES, 41_536);
        assert_eq!(V28_M32_TS4E_DOWN_SMEM_BYTES, 39_808);
        const {
            assert!(V28_M32_FLAT_SMEM_BYTES <= 49_152);
            assert!(V28_M32_TS4E_GATEUP_SMEM_BYTES <= 49_152);
            assert!(V28_M32_TS4E_DOWN_SMEM_BYTES <= 49_152);
        }

        assert_eq!(V28_M32_GATEUP_AUTO_MAX_ROWS, 256);
        assert_eq!(V28_M32_DOWN_AUTO_MAX_ROWS, 256);

        // rows <= 256: both gateup and down engage 2-CTA/SM M32 BC=128.
        for rows in [
            2, 6, 8, 9, 10, 14, 16, 20, 32, 33, 40, 64, 80, 128, 160, 256,
        ] {
            assert_eq!(
                select_v28_dispatch("v28", false, true, rows, false),
                V28DispatchPlan {
                    gateup: V28OpTile::M32,
                    down: V28OpTile::M32,
                },
                "rows={rows}"
            );
        }
        // rows > 256: prefill keeps Twarp (or explicit M64).
        for rows in [257, 512, 2048, 4096, 8192] {
            assert_eq!(
                select_v28_dispatch("v28", false, true, rows, false),
                V28DispatchPlan {
                    gateup: V28OpTile::Twarp,
                    down: V28OpTile::Twarp,
                },
                "rows={rows}"
            );
            assert_eq!(
                select_v28_dispatch("v28", true, true, rows, false),
                V28DispatchPlan {
                    gateup: V28OpTile::M64,
                    down: V28OpTile::M64,
                },
                "rows={rows} explicit m64"
            );
        }
        // M32 explicitly disabled falls back cleanly to M64 for rows <= 256.
        assert_eq!(
            select_v28_dispatch("v28", false, false, 10, false),
            V28DispatchPlan {
                gateup: V28OpTile::M64,
                down: V28OpTile::M64,
            }
        );
        // Non-v28 sets and R2 corrections stay on Twarp.
        assert_eq!(
            select_v28_dispatch("v16", false, true, 10, false),
            V28DispatchPlan {
                gateup: V28OpTile::Twarp,
                down: V28OpTile::Twarp,
            }
        );
        assert_eq!(
            select_v28_dispatch("v28", false, true, 10, true),
            V28DispatchPlan {
                gateup: V28OpTile::Twarp,
                down: V28OpTile::Twarp,
            }
        );
    }

    #[test]
    fn v28_m32_max_grid_defaults_to_4096_and_honors_override() {
        let default_cap = abi::GRID as usize;
        assert_eq!(default_cap, 4096);
        assert_eq!(V28_M32_MAX_GRID_SELECTOR, "ATLAS_QWEN4_MOE_M32_MAX_GRID");
        assert_eq!(parse_v28_m32_max_grid(None), default_cap);
        assert_eq!(parse_v28_m32_max_grid(Some("")), default_cap);
        assert_eq!(parse_v28_m32_max_grid(Some("0")), default_cap);
        assert_eq!(parse_v28_m32_max_grid(Some("bad")), default_cap);
        assert_eq!(parse_v28_m32_max_grid(Some("48")), 96);
        assert_eq!(parse_v28_m32_max_grid(Some("192")), 192);
        assert_eq!(parse_v28_m32_max_grid(Some(" 384 ")), 384);
        assert_eq!(parse_v28_m32_max_grid(Some("768")), 768);
        assert_eq!(parse_v28_m32_max_grid(Some("99999")), default_cap);

        // Default cap (4096 = W18 dynamic static grid): scales cleanly with max_items.
        assert_eq!(v28_m32_grid_gateup(0, default_cap), 96);
        assert_eq!(v28_m32_grid_down(0, default_cap), 96);
        assert_eq!(v28_m32_grid_gateup(20, default_cap), 200);
        assert_eq!(v28_m32_grid_down(20, default_cap), 400);
        assert_eq!(v28_m32_grid_gateup(100, default_cap), 1_000);
        assert_eq!(v28_m32_grid_down(100, default_cap), 2_000);
        assert_eq!(v28_m32_grid_gateup(512, default_cap), abi::GRID);
        assert_eq!(v28_m32_grid_down(512, default_cap), abi::GRID);

        // Explicit override (e.g. 384 or 768) caps the static grid cleanly for sweeps.
        assert_eq!(v28_m32_grid_gateup(512, 384), 384);
        assert_eq!(v28_m32_grid_down(512, 384), 384);
        assert_eq!(v28_m32_grid_gateup(512, 768), 768);
        assert_eq!(v28_m32_grid_down(512, 768), 768);
    }

    #[test]
    fn v28_m32_short_prefill_engages_for_ts4e_small_rows_and_honors_guards() {
        assert_eq!(
            V28_M32_SHORT_PREFILL_SELECTOR,
            "ATLAS_QWEN4_MOE_SHORT_PREFILL_M32"
        );
        assert!(parse_v28_m32_short_prefill_enabled(None));
        assert!(parse_v28_m32_short_prefill_enabled(Some("1")));
        assert!(!parse_v28_m32_short_prefill_enabled(Some("0")));
        assert!(!parse_v28_m32_short_prefill_enabled(Some(" 0 ")));

        let base = V28M32RoutedContext {
            m32_enabled: true,
            scope_active: false,
            short_prefill_enabled: true,
            use_ts4e: true,
            rows: 20,
            has_corrections: false,
            dump_or_oracle: false,
        };
        // Short prefill (rows <= 256) with TS4E engages v28_m32_ts4e by default.
        for rows in [2, 16, 20, 64, 128, 256] {
            assert!(v28_m32_active_for_routed(V28M32RoutedContext {
                rows,
                ..base
            }));
        }
        // Opt-out, non-TS4E, rows > 256, R2 corrections, dump/oracle, or global M32=0 stay off.
        assert!(!v28_m32_active_for_routed(V28M32RoutedContext {
            short_prefill_enabled: false,
            ..base
        }));
        assert!(!v28_m32_active_for_routed(V28M32RoutedContext {
            use_ts4e: false,
            ..base
        }));
        assert!(!v28_m32_active_for_routed(V28M32RoutedContext {
            rows: 257,
            ..base
        }));
        assert!(!v28_m32_active_for_routed(V28M32RoutedContext {
            has_corrections: true,
            ..base
        }));
        assert!(!v28_m32_active_for_routed(V28M32RoutedContext {
            dump_or_oracle: true,
            ..base
        }));
        assert!(!v28_m32_active_for_routed(V28M32RoutedContext {
            m32_enabled: false,
            scope_active: true,
            ..base
        }));
        // Batched verify scope still engages M32 regardless of short-prefill toggle or TS4E.
        assert!(v28_m32_active_for_routed(V28M32RoutedContext {
            scope_active: true,
            short_prefill_enabled: false,
            use_ts4e: false,
            ..base
        }));
    }

    #[test]
    fn v29_m256_smem_constants_and_grid_bounds_match_gb10_limits() {
        // Must fit GB10's 101,376 B dynamic SMEM ceiling and match V29M256Smem<GATEUP, TS4E>.
        assert_eq!(V29_M256_BLOCK, 512);
        assert_eq!(V29_M256_FLAT_SMEM_BYTES, 85_056);
        assert_eq!(V29_M256_TS4E_GATEUP_SMEM_BYTES, 86_912);
        assert_eq!(V29_M256_TS4E_DOWN_SMEM_BYTES, 85_696);
        const {
            assert!(V29_M256_FLAT_SMEM_BYTES <= 101_376);
            assert!(V29_M256_TS4E_GATEUP_SMEM_BYTES <= 101_376);
            assert!(V29_M256_TS4E_DOWN_SMEM_BYTES <= 101_376);
        }
        assert_eq!(
            V29_M256_TS4E_GATEUP_SMEM_BYTES - V29_M256_FLAT_SMEM_BYTES,
            1_856
        );
        assert_eq!(
            V29_M256_TS4E_DOWN_SMEM_BYTES - V29_M256_FLAT_SMEM_BYTES,
            640
        );

        // max_items = min(pairs, ceil(pairs / 256) + 512), grid_gu = (max_items * 10).clamp(48, 4096),
        // grid_dn = (max_items * 20).clamp(48, 4096).
        assert_eq!(v29_m256_max_items(0), 0);
        assert_eq!(v29_m256_grid_gateup(0), 48);
        assert_eq!(v29_m256_grid_down(0), 48);

        // Small rows (forced M256): pairs = 20 -> max_items = 20 -> gu = 200, dn = 400.
        assert_eq!(v29_m256_max_items(20), 20);
        assert_eq!(v29_m256_grid_gateup(20), 200);
        assert_eq!(v29_m256_grid_down(20), 400);

        // rows = 2048 -> pairs = 20,480 -> ceil(20480/256) + 512 = 80 + 512 = 592 -> clamped to 4096.
        assert_eq!(v29_m256_max_items(20_480), 592);
        assert_eq!(v29_m256_grid_gateup(592), abi::GRID);
        assert_eq!(v29_m256_grid_down(592), abi::GRID);

        // rows = 16384 -> pairs = 163,840 -> ceil(163840/256) + 512 = 640 + 512 = 1152 -> clamped to 4096.
        assert_eq!(v29_m256_max_items(163_840), 1_152);
        assert_eq!(v29_m256_grid_gateup(1_152), abi::GRID);
        assert_eq!(v29_m256_grid_down(1_152), abi::GRID);

        // Boundary below upper clamp: max_items = 100 -> gu = 1000, dn = 2000.
        assert_eq!(v29_m256_grid_gateup(100), 1_000);
        assert_eq!(v29_m256_grid_down(100), 2_000);
    }

    #[test]
    fn v28_m16_decode_constants_policy_and_grid_bounds_fit_4cta_smem_cap() {
        // 4 CTAs/SM on GB10 (99 KiB SMEM => <= 24 KiB = 24,576 B per CTA).
        assert_eq!(V28_M16_DECODE_SELECTOR, "ATLAS_QWEN4_MOE_DECODE_M16");
        assert_eq!(V28_M16_BLOCK, 128);
        assert_eq!(V28_M16_FLAT_GATEUP_SMEM_BYTES, 19_328);
        assert_eq!(V28_M16_FLAT_DOWN_SMEM_BYTES, 19_072);
        assert_eq!(V28_M16_TS4E_GATEUP_SMEM_BYTES, 21_184);
        assert_eq!(V28_M16_TS4E_DOWN_SMEM_BYTES, 19_712);
        const {
            assert!(V28_M16_FLAT_GATEUP_SMEM_BYTES <= 24_576);
            assert!(V28_M16_FLAT_DOWN_SMEM_BYTES <= 24_576);
            assert!(V28_M16_TS4E_GATEUP_SMEM_BYTES <= 24_576);
            assert!(V28_M16_TS4E_DOWN_SMEM_BYTES <= 24_576);
        }
        assert_eq!(V28_M16_DECODE_AUTO_MAX_ROWS, 96);

        // Env toggle: default OFF on 48-SM GB10 unless explicitly "1" or "auto".
        assert!(!parse_v28_m16_decode_enabled(None));
        assert!(parse_v28_m16_decode_enabled(Some("1")));
        assert!(parse_v28_m16_decode_enabled(Some(" 1 ")));
        assert!(parse_v28_m16_decode_enabled(Some("auto")));
        assert!(!parse_v28_m16_decode_enabled(Some("0")));
        assert!(!parse_v28_m16_decode_enabled(Some(" 0 ")));
        assert!(!parse_v28_m16_decode_enabled(Some("")));

        // Admitted for rows <= 96 when m32_active && m16_enabled && !has_corrections.
        for rows in [4, 10, 16, 20, 40, 64, 80, 96] {
            assert!(
                v28_m16_active_for_routed(true, true, rows, false),
                "rows={rows} should engage 4-CTA/SM v28_m16"
            );
        }
        // rows = 97..=256 stay on v28_m32; opt-out, !m32_active, or R2 corrections decline m16.
        for rows in [97, 100, 128, 256] {
            assert!(!v28_m16_active_for_routed(true, true, rows, false));
        }
        assert!(!v28_m16_active_for_routed(false, true, 80, false));
        assert!(!v28_m16_active_for_routed(true, false, 80, false));
        assert!(!v28_m16_active_for_routed(true, true, 80, true));

        // Grid scales cleanly from 1 full 4-CTA/SM wave (192 CTAs) up to abi::GRID (4096).
        let default_cap = abi::GRID as usize;
        assert_eq!(v28_m16_grid_gateup(0, default_cap), 192);
        assert_eq!(v28_m16_grid_down(0, default_cap), 192);
        assert_eq!(v28_m16_grid_gateup(10, default_cap), 192);
        assert_eq!(v28_m16_grid_down(10, default_cap), 200);
        assert_eq!(v28_m16_grid_gateup(100, default_cap), 1_000);
        assert_eq!(v28_m16_grid_down(100, default_cap), 2_000);
        assert_eq!(v28_m16_grid_gateup(512, default_cap), abi::GRID);
        assert_eq!(v28_m16_grid_down(512, default_cap), abi::GRID);
    }

    #[test]
    fn v30_2cta_smem_constants_and_grid_bounds_fit_2cta_smem_cap() {
        // 2 CTAs/SM on GB10 (99 KiB SMEM => <= 48,128 B <= 48 KiB = 49,152 B per CTA).
        assert_eq!(V30_2CTA_BLOCK, 256);
        assert_eq!(V30_2CTA_FLAT_GATEUP_SMEM_BYTES, 46_144);
        assert_eq!(V30_2CTA_FLAT_DOWN_SMEM_BYTES, 45_632);
        assert_eq!(V30_2CTA_TS4E_GATEUP_SMEM_BYTES, 47_744);
        assert_eq!(V30_2CTA_TS4E_DOWN_SMEM_BYTES, 46_144);
        const {
            assert!(V30_2CTA_FLAT_GATEUP_SMEM_BYTES <= 48_128);
            assert!(V30_2CTA_FLAT_DOWN_SMEM_BYTES <= 48_128);
            assert!(V30_2CTA_TS4E_GATEUP_SMEM_BYTES <= 48_128);
            assert!(V30_2CTA_TS4E_DOWN_SMEM_BYTES <= 48_128);
            assert!(V30_2CTA_TS4E_GATEUP_SMEM_BYTES <= 49_152);
        }

        // Grid scales cleanly from 1 full 2-CTA/SM wave (96 CTAs) up to abi::GRID (4096).
        assert_eq!(v30_2cta_grid_gateup(0), 96);
        assert_eq!(v30_2cta_grid_down(0), 96);
        assert_eq!(v30_2cta_grid_gateup(5), 96);
        assert_eq!(v30_2cta_grid_down(5), 100);
        assert_eq!(v30_2cta_grid_gateup(100), 1_000);
        assert_eq!(v30_2cta_grid_down(100), 2_000);
        assert_eq!(v30_2cta_grid_gateup(832), abi::GRID);
        assert_eq!(v30_2cta_grid_down(832), abi::GRID);
    }

    #[test]
    fn v31_ws_smem_constants_and_grid_bounds_fit_gb10_smem_cap() {
        // 1 CTA/SM on GB10 (99 KiB = 101,376 B dynamic SMEM ceiling; matches V31WsSmem).
        assert_eq!(V31_WS_BLOCK, 512);
        assert_eq!(V31_WS_FLAT_GATEUP_SMEM_BYTES, 87_616);
        assert_eq!(V31_WS_FLAT_DOWN_SMEM_BYTES, 87_104);
        assert_eq!(V31_WS_TS4E_GATEUP_SMEM_BYTES, 91_904);
        assert_eq!(V31_WS_TS4E_DOWN_SMEM_BYTES, 90_672);
        const {
            assert!(V31_WS_FLAT_GATEUP_SMEM_BYTES <= 101_376);
            assert!(V31_WS_FLAT_DOWN_SMEM_BYTES <= 101_376);
            assert!(V31_WS_TS4E_GATEUP_SMEM_BYTES <= 101_376);
            assert!(V31_WS_TS4E_DOWN_SMEM_BYTES <= 101_376);
        }

        // Default persistent 1-CTA/SM wave (48 CTAs across GB10's 48 SMs).
        let default_gb10 = v31_ws_default_max_grid_for_sm(48);
        assert_eq!(default_gb10, 48);
        for max_items in [0usize, 5, 100, 832] {
            assert_eq!(
                v31_ws_grid_gateup_capped(max_items, default_gb10),
                48,
                "default gateup grid must stay persistent at 48 for max_items={max_items}"
            );
            assert_eq!(
                v31_ws_grid_down_capped(max_items, default_gb10),
                48,
                "default down grid must stay persistent at 48 for max_items={max_items}"
            );
        }
    }

    #[test]
    fn compact_moe_grids_scale_dynamically_across_48_110_170_188_sms_and_clamp_to_abi_grid() {
        for sm in [48u32, 110, 170, 188] {
            let sm_u = sm as usize;
            // v31_ws persistent 1-CTA/SM wave launches `sm` CTAs (188 on RTX PRO 6000).
            assert_eq!(v31_ws_min_grid_for_sm(sm), sm_u);
            assert_eq!(v31_ws_default_max_grid_for_sm(sm), sm_u);
            assert_eq!(parse_v31_ws_max_grid_for_sm(None, sm), sm_u);
            assert_eq!(parse_v31_ws_max_grid_for_sm(Some("24"), sm), sm_u);
            assert_eq!(
                parse_v31_ws_max_grid_for_sm(Some("4096"), sm),
                abi::GRID as usize
            );
            assert_eq!(
                v31_ws_grid_gateup_capped_for_sm(832, sm_u, sm),
                sm,
                "v31_ws gateup persistent grid for {sm} SMs"
            );
            assert_eq!(
                v31_ws_grid_down_capped_for_sm(832, sm_u, sm),
                sm,
                "v31_ws down persistent grid for {sm} SMs"
            );

            // v29_m256 1x SM wave floor.
            assert_eq!(v29_m256_grid_gateup_for_sm(0, sm), sm);
            assert_eq!(v29_m256_grid_down_for_sm(0, sm), sm);

            // v28_m32 and v30_2cta 2x SM wave floor.
            assert_eq!(v28_m32_min_grid_for_sm(sm), sm_u * 2);
            assert_eq!(v30_2cta_min_grid_for_sm(sm), sm_u * 2);
            assert_eq!(v30_2cta_grid_gateup_for_sm(0, sm), sm * 2);
            assert_eq!(v30_2cta_grid_down_for_sm(0, sm), sm * 2);
            assert_eq!(parse_v28_m32_max_grid_for_sm(Some("48"), sm), sm_u * 2);
            assert_eq!(
                v28_m32_grid_gateup_for_sm(0, abi::GRID as usize, sm),
                sm * 2
            );
            assert_eq!(v28_m32_grid_down_for_sm(0, abi::GRID as usize, sm), sm * 2);

            // v28_m16 4x SM wave floor.
            assert_eq!(v28_m16_min_grid_for_sm(sm), sm_u * 4);
            assert_eq!(
                v28_m16_grid_gateup_for_sm(0, abi::GRID as usize, sm),
                sm * 4
            );
            assert_eq!(v28_m16_grid_down_for_sm(0, abi::GRID as usize, sm), sm * 4);
        }

        // All helpers clamp safely to abi::GRID (4096) even for extreme SM counts.
        assert_eq!(v31_ws_min_grid_for_sm(8192), abi::GRID as usize);
        assert_eq!(v28_m32_min_grid_for_sm(4096), abi::GRID as usize);
        assert_eq!(v28_m16_min_grid_for_sm(2048), abi::GRID as usize);
    }

    #[test]
    fn v31_ws_max_grid_defaults_to_48_and_honors_override() {
        assert_eq!(
            V31_WS_MAX_GRID_SELECTOR,
            "ATLAS_QWEN4_PREFILL_MOE_V31_MAX_GRID"
        );
        assert_eq!(parse_v31_ws_max_grid(None), 48);
        assert_eq!(parse_v31_ws_max_grid(Some("")), 48);
        assert_eq!(parse_v31_ws_max_grid(Some("0")), 48);
        assert_eq!(parse_v31_ws_max_grid(Some("bad")), 48);
        assert_eq!(parse_v31_ws_max_grid(Some("24")), 48);
        assert_eq!(parse_v31_ws_max_grid(Some("48")), 48);
        assert_eq!(parse_v31_ws_max_grid(Some(" 96 ")), 96);
        assert_eq!(parse_v31_ws_max_grid(Some("4096")), abi::GRID as usize);
        assert_eq!(parse_v31_ws_max_grid(Some("99999")), abi::GRID as usize);

        // Explicit 4096 override reproduces full TN=128 (5 GU) / TN=256 (10 DN) grid scaling.
        let full_cap = abi::GRID as usize;
        assert_eq!(v31_ws_grid_gateup_capped(0, full_cap), 48);
        assert_eq!(v31_ws_grid_down_capped(0, full_cap), 48);
        assert_eq!(v31_ws_grid_gateup_capped(5, full_cap), 48);
        assert_eq!(v31_ws_grid_down_capped(5, full_cap), 50);
        assert_eq!(v31_ws_grid_gateup_capped(100, full_cap), 500);
        assert_eq!(v31_ws_grid_down_capped(100, full_cap), 1_000);
        assert_eq!(v31_ws_grid_gateup_capped(832, full_cap), abi::GRID);
        assert_eq!(v31_ws_grid_down_capped(832, full_cap), abi::GRID);

        // Intermediate cap (e.g. 96) clamps cleanly for A/B sweeps.
        assert_eq!(v31_ws_grid_gateup_capped(100, 96), 96);
        assert_eq!(v31_ws_grid_down_capped(100, 96), 96);
    }

    #[test]
    fn v28_m16_wide_sm_auto_enables_for_96_row_decode_frames() {
        // <= 64 SMs (GB10): default OFF, opt-in via "1"/"auto".
        assert!(!parse_v28_m16_decode_enabled_for_sm(None, 48));
        assert!(!parse_v28_m16_decode_enabled_for_sm(None, 64));

        // > 64 SMs (SM120/SM103): auto-ON when unset, opt-out via "0", capped at 96 rows
        // (rows 97..=128 stay on 8-warp 2-CTA/SM v28_m32, which beats v28_m16 at 128 rows).
        assert!(parse_v28_m16_decode_enabled_for_sm(None, 170));
        assert!(parse_v28_m16_decode_enabled_for_sm(None, 188));
        assert!(parse_v28_m16_decode_enabled_for_sm(Some(""), 188));
        assert!(!parse_v28_m16_decode_enabled_for_sm(Some("0"), 188));
        assert!(v28_m16_active_for_routed(true, true, 96, false));
        assert!(!v28_m16_active_for_routed(true, true, 112, false));
        assert!(!v28_m16_active_for_routed(true, true, 128, false));
    }

    #[test]
    fn xq_quant_gate_grid_eliminates_tail_wave_on_high_sm_and_preserves_gb10() {
        for sm in [0u32, 48, 64] {
            assert_eq!(xq_quant_gate_grid_for_sm(0, sm), 1);
            assert_eq!(xq_quant_gate_grid_for_sm(1, sm), 1);
            assert_eq!(xq_quant_gate_grid_for_sm(512, sm), 512);
            assert_eq!(xq_quant_gate_grid_for_sm(1024, sm), 1024);
            assert_eq!(xq_quant_gate_grid_for_sm(16_384, sm), 1024);
        }
        assert_eq!(xq_quant_gate_grid_for_sm(512, 188), 512);
        assert_eq!(xq_quant_gate_grid_for_sm(1024, 188), 1024);
        assert_eq!(xq_quant_gate_grid_for_sm(1128, 188), 1128);
        assert_eq!(xq_quant_gate_grid_for_sm(16_384, 188), 1128);
        assert_eq!(xq_quant_gate_grid_for_sm(16_384, 148), 888);
    }

    #[test]
    fn unpermute_blend_w4_engages_only_above_64_sms_and_keeps_gb10_legacy() {
        for sm in [0u32, 48, 64] {
            for rows in [4usize, 8, 16, 32, 64, 128, 256] {
                assert_eq!(
                    select_unpermute_blend_launch_for_sm(rows, 2560, 10, sm),
                    UnpermuteBlendLaunch {
                        variant: UnpermuteBlendVariant::Legacy1Warp,
                        grid_y: 10,
                        block_x: 32,
                    }
                );
                assert_eq!(
                    select_unpermute_blend_launch_for_sm(rows, 2048, 8, sm),
                    UnpermuteBlendLaunch {
                        variant: UnpermuteBlendVariant::Legacy1Warp,
                        grid_y: 4,
                        block_x: 64,
                    }
                );
            }
        }
        for sm in [65u32, 110, 188] {
            for rows in [4usize, 8, 15] {
                assert_eq!(
                    select_unpermute_blend_launch_for_sm(rows, 2560, 10, sm),
                    UnpermuteBlendLaunch {
                        variant: UnpermuteBlendVariant::Coalesced4Warp,
                        grid_y: 10,
                        block_x: 128,
                    }
                );
            }
            for rows in [16usize, 32, 64, 128, 256] {
                assert_eq!(
                    select_unpermute_blend_launch_for_sm(rows, 2560, 10, sm),
                    UnpermuteBlendLaunch {
                        variant: UnpermuteBlendVariant::Coalesced4Warp,
                        grid_y: 5,
                        block_x: 128,
                    }
                );
            }
            assert_eq!(
                select_unpermute_blend_launch_for_sm(32, 2048, 8, sm),
                UnpermuteBlendLaunch {
                    variant: UnpermuteBlendVariant::Coalesced4Warp,
                    grid_y: 5,
                    block_x: 128,
                }
            );
        }
    }

    #[test]
    fn gate_scalars_state_round_trips_and_clears_once_taken() {
        let inp = DevicePtr(0x1000);
        let sg = DevicePtr(0x2000);
        let ss = DevicePtr(0x3000);
        forget_xq();
        assert!(!has_gate_scalars(inp, sg, 16));
        note_gate_scalars(inp, sg, 16, ss);
        assert!(has_gate_scalars(inp, sg, 16));
        assert!(!has_gate_scalars(inp, sg, 32));
        assert_eq!(take_gate_scalars(inp, sg, 16), ss);
        assert!(!has_gate_scalars(inp, sg, 16));
    }

    #[test]
    fn legacy_w4a8_sets_keep_fixed_grid_while_v28_and_mxfp8_down_clamp_dynamically() {
        for tag in ["v10", "v12", "v14", "v16"] {
            for max_items in [1usize, 10, 40, 672, 1000] {
                assert_eq!(
                    w4a8_twarp_grid_gateup_for_sm(tag, max_items, 48),
                    abi::GRID,
                    "legacy gateup {tag} must keep FIXED_GRID for max_items={max_items}"
                );
                assert_eq!(
                    w4a8_twarp_grid_down_for_sm(tag, false, max_items, 48),
                    abi::GRID,
                    "legacy BF16 down {tag} must keep FIXED_GRID for max_items={max_items}"
                );
            }
            // v17q/v17qr2 MXFP8 down accepts 1..=FIXED_GRID.
            assert_eq!(w4a8_twarp_grid_down_for_sm(tag, true, 10, 48), 100);
        }
        assert_eq!(w4a8_twarp_grid_gateup_for_sm("v28", 10, 48), 50);
        assert_eq!(w4a8_twarp_grid_gateup_for_sm("v28", 672, 48), 3_360);
        assert_eq!(w4a8_twarp_grid_down_for_sm("v28", false, 10, 48), 100);
        assert_eq!(w4a8_twarp_grid_down_for_sm("v28", true, 10, 48), 100);
    }

    #[test]
    fn ts4e_v28_receipt_smem_matches_dispatched_tile_variant() {
        let m32_plan = V28DispatchPlan {
            gateup: V28OpTile::M32,
            down: V28OpTile::M32,
        };
        assert_eq!(
            ts4e_v28_receipt_smem(m32_plan, true),
            (V28_M16_TS4E_GATEUP_SMEM_BYTES, V28_M16_TS4E_DOWN_SMEM_BYTES)
        );
        assert_eq!(
            ts4e_v28_receipt_smem(m32_plan, false),
            (V28_M32_TS4E_GATEUP_SMEM_BYTES, V28_M32_TS4E_DOWN_SMEM_BYTES)
        );
        for tile in [V28OpTile::M64, V28OpTile::Twarp] {
            let plan = V28DispatchPlan {
                gateup: tile,
                down: tile,
            };
            assert_eq!(
                ts4e_v28_receipt_smem(plan, false),
                (V28_TS4E_GATEUP_SMEM_BYTES, V28_TS4E_DOWN_SMEM_BYTES)
            );
        }
    }
}
