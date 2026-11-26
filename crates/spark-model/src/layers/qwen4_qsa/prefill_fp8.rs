// SPDX-License-Identifier: AGPL-3.0-only

//! E4M3 operands for the QSA-sparse prefill core (`ATLAS_QWEN4_PREFILL_QSA_FP8`).
//!
//! - `qk`: Q and K in E4M3 (m16n8k32 QK^T, fp32 accumulate); P*V stays BF16
//!   on the BF16 V cache.
//! - `qkpv`: Q, K, V and P in E4M3 (both MMAs E4M3).
//!
//! The fused attention prep (`ATLAS_QWEN4_PREFILL_ATTN_PREP_FUSED=1`) writes
//! the E4M3 copies once, next to the BF16 values it already writes, with static
//! power-of-two prescales (Q per GQA group, K and V per KV head; see
//! [`LayerScales`]): Q8 rows into a per-chunk scratch, and K8/V8 into a shadow of
//! the paged BF16 cache (same block layout, one byte per element). The core then
//! only loads bytes and issues E4M3 MMAs. The BF16 cache stays the decode cache,
//! so decode is unchanged.
//!
//! `qkpv` takes the row sum l from the same E4M3 P its P*V uses (an MMA against
//! ones, so an underflowed P drops out of numerator and denominator alike) and
//! reads its tiles in the order of `qwen4_qsa_prefill_plan_sort_desc`: newest
//! blocks first, the sink block last. O accumulates in the MMA (C = O): on GB10
//! a two-level fold measured the same error and cost 6% of the core.
//!
//! Not bit-exact with the BF16 core (E4M3 rounding of the operands); the
//! products are exact and accumulate in fp32 in a fixed order, so it is
//! deterministic. Intended for opt-in FP8 prefill profiles.
//!
//! Shadow validity: the prep writes the shadow for every row it preps. A chunk
//! that starts where this layer's previous chunk ended reads only rows the prep
//! wrote. Any other start (a sequence that reuses cached blocks, or a
//! continuation after decode, whose tokens only the BF16 cache holds) first
//! refills the shadow for [0, start) from the BF16 pages. That is exact with
//! one sequence at a time, which the sparse core's C1 metadata already
//! requires.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use anyhow::{Context, Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;
use spark_runtime::kv_cache::{KvCacheDtype, PagedKvCache};
use spark_runtime::startup_ledger;

use super::QUERY_HEADS;
use super::prefill_sparse::{ATTN_CAP, HEAD_DIM, KV_HEADS};

pub const SELECTOR: &str = "ATLAS_QWEN4_PREFILL_QSA_FP8";
/// `q,k,v`: one power-of-two prescale per tensor kind for every layer and head,
/// instead of the per-layer ones derived from the norm weights (Q, K) and
/// [`V_PRESCALE`] (V).
pub const PRESCALE_SELECTOR: &str = "ATLAS_QWEN4_PREFILL_QSA_FP8_PRESCALE";
/// Decoupled-warp `qkpv` E4M3 core (`1`/`4k`: 2x2 KiB/warp = 53,296 B dynamic SMEM;
/// `2`/`8k`: 2x4 KiB/warp = 86,064 B dynamic SMEM). Bit-identical to `qkpv`.
pub const WARP_DECOUPLE_SELECTOR: &str = "ATLAS_QWEN4_PREFILL_QSA_WARP_DECOUPLE";
/// Split-sub-CTA `qkpv` E4M3 core (`1`/`half`: 4 warps = 128 threads/CTA; `2`/`qtr`: 2 warps = 64
/// threads/CTA; 32,768 B dynamic (+ 64..80 B static) SMEM, 2 CTAs/SM). Bit-identical to `qkpv`.
pub const SUB_CTA_SELECTOR: &str = "ATLAS_QWEN4_PREFILL_QSA_SUB_CTA";
/// Fused 1-kernel QSA plan + descending need-bin counting sort (`1` enables).
pub const PLAN_FUSED_SELECTOR: &str = "ATLAS_QWEN4_PREFILL_QSA_PLAN_FUSED";
pub const UNION_PRUNE_ALIAS: &str = "ATLAS_QWEN4_QSA_UNION_PRUNE";
/// Intra-warp software-pipelined E4M3 core (`1` enables for both `qk` and `qkpv`).
/// Bit-identical to the non-pipelined `qk` / `qkpv` cores.
pub const PIPE_SELECTOR: &str = "ATLAS_QWEN4_PREFILL_QSA_PIPE";
/// Share one `V8` shadow plane across all attention layers when `Mode::QkPv` is
/// selected (default enabled; `0` / `off` / `false` reverts to legacy per-layer
/// `V8` shadow planes). On `start == 0` chunks each layer writes and immediately
/// consumes the shared `V8` plane on the same stream; continuation chunks
/// (`start > 0`) fall back to `Mode::Qk` (per-layer `K8` + lossless BF16 `V`).
pub const V8_SHARED_SELECTOR: &str = "ATLAS_QWEN4_PREFILL_QSA_V8_SHARED";
/// Direct static-prescaled E4M3 output from `qwen4_qsa_prefill_attn_fp8_qkpv_e4m3_pipe`
/// into `qwen4_fast_proj`'s FP8 activation buffer (`default` ON when `mode == qkpv`,
/// `pipe == true`, FP8 tensor-scaled `o_proj`, `rows >= 256`, and no exact-check/oracle/dump
/// diagnostic is active; `0` / `off` / `false` reverts cleanly to BF16 + `amax_pipe`).
pub const E4M3_OUT_SELECTOR: &str = "ATLAS_QWEN4_PREFILL_QSA_E4M3_OUT";
/// 2-CTA/SM intra-CTA D-split + Half-CTA variant of `qkpv_e4m3_pipe`
/// (`qwen4_qsa_prefill_attn_fp8_qkpv_e4m3_pipe_2cta`: `<= 128` regs, `43,584 B <= 49,152 B`
/// dynamic SMEM -> 2 CTAs/SM = 16 warps/SM on GB10; bit-exact 0-ULP identical to
/// `qkpv_e4m3_pipe`). Default OFF (1-CTA/SM `qkpv_e4m3_pipe`); opt in with
/// `ATLAS_QWEN4_PREFILL_QSA_E4M3_2CTA=1|on|true`.
pub const E4M3_2CTA_SELECTOR: &str = "ATLAS_QWEN4_PREFILL_QSA_E4M3_2CTA";
/// Grid Y multiplier for `qwen4_qsa_prefill_attn_fp8_qkpv_e4m3_pipe_2cta` (2 Half-CTAs/KV head).
pub const E4M3_2CTA_SUB_FACTOR: u32 = 2;
/// Dynamic shared memory of `qwen4_qsa_prefill_attn_fp8_qkpv_e4m3_pipe_2cta`:
/// `2 * 16,384` B (2-stage K8+V8) + `8,192` B (`Q_hi` cols 128..255) + `512` B (`s_mx`) +
/// `2,048` B (`s_a`) + `64` B (`s_ent`) = `43,584` B (`<= 49,152` B for 2 CTAs/SM on `sm_121a`).
pub const E4M3_2CTA_SMEM: usize = 2 * 2 * TILE8 + 64 * 128 + 512 + 2048 + 64;
/// Minimum chunk row count for direct static-prescaled E4M3 QSA output.
pub const E4M3_OUT_MIN_ROWS: usize = 256;
/// Contiguous output width (`24 * 256 = 6144`) expected by `qkpv_e4m3_pipe` and `o_proj`.
pub const E4M3_OUT_COLS: usize = QUERY_HEADS * HEAD_DIM;
/// Maximum finite representable magnitude in OCP FP8 E4M3FN.
pub const E4M3_MAX: f32 = 448.0;
/// V has no norm to bound it; |v| <= 9.4 on layers 0/5/11 leaves 6x headroom at x8.
pub const V_PRESCALE: f32 = 8.0;
const ATTN_MODULE: &str = "qwen4_qsa_prefill_attn_fp8";
const PREP_MODULE: &str = "qwen4_attn_prep_fused_fp8";
/// One 32-token tile: E4M3 K (8 KiB) plus E4M3 V (8 KiB) or BF16 V (16 KiB).
const TILE8: usize = 32 * HEAD_DIM;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Mode {
    /// E4M3 QK^T, BF16 P*V.
    Qk,
    /// E4M3 QK^T and P*V.
    QkPv,
}

/// Layout of the E4M3 `V8` shadow planes when [`Mode::QkPv`] is active.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum V8Layout {
    /// One `V8` shadow plane shared across all attention layers of a cache.
    /// Used on `start == 0` chunks; continuation chunks (`start > 0`) switch
    /// to [`Mode::Qk`] via [`effective_chunk_mode`].
    Shared,
    /// Legacy per-layer `V8` shadow plane (`2 * num_layers` total planes).
    PerLayer,
}

impl Mode {
    pub fn name(self) -> &'static str {
        match self {
            Mode::Qk => "qk",
            Mode::QkPv => "qkpv",
        }
    }

    /// Dynamic shared memory of the core: two stages plus the tile plan.
    pub(super) fn smem(self) -> usize {
        let stage = TILE8 + if self == Mode::QkPv { TILE8 } else { 2 * TILE8 };
        2 * stage + ATTN_CAP * 4
    }

    /// Number of E4M3 shadow KV planes required per layer under legacy per-layer
    /// `V8` allocation (`1` = K8-only for `Mode::Qk`, which reads lossless BF16
    /// V directly; `2` = K8 + V8 for `Mode::QkPv`). See
    /// [`checked_total_shadow_planes`] for total planes under [`V8Layout::Shared`].
    pub const fn shadow_planes(self) -> usize {
        match self {
            Mode::Qk => 1,
            Mode::QkPv => 2,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WarpDecouple {
    /// 2 x 2 KiB per warp (32 KiB ring + tile plan = 53,296 B).
    Wdec4k,
    /// 2 x 4 KiB per warp (64 KiB ring + tile plan = 86,064 B).
    Wdec8k,
}

impl WarpDecouple {
    pub fn name(self) -> &'static str {
        match self {
            WarpDecouple::Wdec4k => "4k",
            WarpDecouple::Wdec8k => "8k",
        }
    }

    pub(super) fn smem(self) -> usize {
        let ring = match self {
            WarpDecouple::Wdec4k => 8 * 4096,
            WarpDecouple::Wdec8k => 8 * 8192,
        };
        ring + ATTN_CAP * 4
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SubCta {
    /// 4 warps (128 threads) per sub-CTA, grid.y = 2 * KV_HEADS.
    Half,
    /// 2 warps (64 threads) per sub-CTA, grid.y = 4 * KV_HEADS.
    Quarter,
}

impl SubCta {
    pub fn name(self) -> &'static str {
        match self {
            SubCta::Half => "half",
            SubCta::Quarter => "qtr",
        }
    }

    pub fn sub_factor(self) -> u32 {
        match self {
            SubCta::Half => 2,
            SubCta::Quarter => 4,
        }
    }

    pub fn threads(self) -> u32 {
        match self {
            SubCta::Half => 128,
            SubCta::Quarter => 64,
        }
    }

    pub(super) fn smem(self) -> usize {
        2 * 16384
    }
}

fn parse(value: Option<&str>) -> Result<Option<Mode>> {
    match value {
        None | Some("0") => Ok(None),
        Some("qk") => Ok(Some(Mode::Qk)),
        Some("qkpv") => Ok(Some(Mode::QkPv)),
        Some(other) => anyhow::bail!("{SELECTOR} must be absent, 0, qk, or qkpv; got {other:?}"),
    }
}

fn parse_warp_decouple(value: Option<&str>) -> Result<Option<WarpDecouple>> {
    match value {
        None | Some("0") => Ok(None),
        Some("1") | Some("4k") => Ok(Some(WarpDecouple::Wdec4k)),
        Some("2") | Some("8k") => Ok(Some(WarpDecouple::Wdec8k)),
        Some(other) => anyhow::bail!(
            "{WARP_DECOUPLE_SELECTOR} must be absent, 0, 1, 4k, 2, or 8k; got {other:?}"
        ),
    }
}

fn parse_sub_cta(value: Option<&str>) -> Result<Option<SubCta>> {
    match value {
        None | Some("0") => Ok(None),
        Some("1") | Some("half") => Ok(Some(SubCta::Half)),
        Some("2") | Some("qtr") | Some("quarter") => Ok(Some(SubCta::Quarter)),
        Some(other) => anyhow::bail!(
            "{SUB_CTA_SELECTOR} must be absent, 0, 1, half, 2, qtr, or quarter; got {other:?}"
        ),
    }
}

/// The selected E4M3 mode, if any (read once per process).
pub fn mode() -> Result<Option<Mode>> {
    static S: OnceLock<Result<Option<Mode>, String>> = OnceLock::new();
    S.get_or_init(|| parse(std::env::var(SELECTOR).ok().as_deref()).map_err(|e| e.to_string()))
        .clone()
        .map_err(|e| anyhow::anyhow!(e))
}

/// The selected decoupled-warp mode, if any (read once per process).
pub fn warp_decouple() -> Result<Option<WarpDecouple>> {
    static S: OnceLock<Result<Option<WarpDecouple>, String>> = OnceLock::new();
    S.get_or_init(|| {
        parse_warp_decouple(std::env::var(WARP_DECOUPLE_SELECTOR).ok().as_deref())
            .map_err(|e| e.to_string())
    })
    .clone()
    .map_err(|e| anyhow::anyhow!(e))
}

/// The selected split-sub-CTA mode, if any (read once per process).
pub fn sub_cta() -> Result<Option<SubCta>> {
    static S: OnceLock<Result<Option<SubCta>, String>> = OnceLock::new();
    S.get_or_init(|| {
        parse_sub_cta(std::env::var(SUB_CTA_SELECTOR).ok().as_deref()).map_err(|e| e.to_string())
    })
    .clone()
    .map_err(|e| anyhow::anyhow!(e))
}

fn parse_plan_fused_val(v: Option<&str>, sel: &str) -> Result<bool, String> {
    match v {
        None | Some("0") => Ok(false),
        Some("1") => Ok(true),
        Some(other) => Err(format!("{sel} must be absent, 0, or 1; got {other:?}")),
    }
}

/// Whether `ATLAS_QWEN4_PREFILL_QSA_PLAN_FUSED=1` or `ATLAS_QWEN4_QSA_UNION_PRUNE=1` is selected (read once per process).
pub fn plan_fused() -> Result<bool> {
    static S: OnceLock<Result<bool, String>> = OnceLock::new();
    S.get_or_init(|| {
        let alias = if let Ok(v) = std::env::var(UNION_PRUNE_ALIAS) {
            parse_plan_fused_val(Some(v.as_str()), UNION_PRUNE_ALIAS)?
        } else {
            false
        };
        let primary = parse_plan_fused_val(
            std::env::var(PLAN_FUSED_SELECTOR).ok().as_deref(),
            PLAN_FUSED_SELECTOR,
        )?;
        let active_selector = if alias {
            Some(UNION_PRUNE_ALIAS)
        } else if primary {
            Some(PLAN_FUSED_SELECTOR)
        } else {
            None
        };
        if let Some(sel) = active_selector {
            tracing::info!("QSA_UNION_PRUNE_ENGAGED selector={sel}");
        }
        Ok(active_selector.is_some())
    })
    .clone()
    .map_err(|e| anyhow::anyhow!(e))
}

fn parse_pipe(value: Option<&str>) -> Result<bool> {
    match value {
        None | Some("") | Some("0") => Ok(false),
        Some("1") => Ok(true),
        Some(other) => anyhow::bail!("{PIPE_SELECTOR} must be absent, 0, or 1; got {other:?}"),
    }
}

/// Whether `ATLAS_QWEN4_PREFILL_QSA_PIPE=1` is selected (read once per process).
pub fn pipe() -> Result<bool> {
    static S: OnceLock<Result<bool, String>> = OnceLock::new();
    S.get_or_init(|| {
        parse_pipe(std::env::var(PIPE_SELECTOR).ok().as_deref()).map_err(|e| e.to_string())
    })
    .clone()
    .map_err(|e| anyhow::anyhow!(e))
}

fn parse_v8_shared(value: Option<&str>) -> Result<bool> {
    let Some(raw) = value else {
        return Ok(true);
    };
    match raw.trim().to_ascii_lowercase().as_str() {
        "" | "1" | "on" | "true" => Ok(true),
        "0" | "off" | "false" => Ok(false),
        _ => anyhow::bail!(
            "{V8_SHARED_SELECTOR} must be absent, 1, on, true, 0, off, or false; got {raw:?}"
        ),
    }
}

/// Whether shared 1-layer `V8` shadow allocation is enabled (read once per process;
/// defaults to `true`, opt out with `ATLAS_QWEN4_PREFILL_QSA_V8_SHARED=0|off|false`).
pub fn v8_shared() -> Result<bool> {
    static S: OnceLock<Result<bool, String>> = OnceLock::new();
    S.get_or_init(|| {
        parse_v8_shared(std::env::var(V8_SHARED_SELECTOR).ok().as_deref())
            .map_err(|e| e.to_string())
    })
    .clone()
    .map_err(|e| anyhow::anyhow!(e))
}

/// Convenience boolean for [`v8_shared`] (defaults to `true` if unset).
#[must_use]
pub fn v8_shared_enabled() -> bool {
    v8_shared().unwrap_or(true)
}

/// Active [`V8Layout`] derived from [`v8_shared`].
pub fn v8_layout() -> Result<V8Layout> {
    if v8_shared()? {
        Ok(V8Layout::Shared)
    } else {
        Ok(V8Layout::PerLayer)
    }
}

/// Pure layout-parameterized form of [`effective_chunk_mode`].
#[must_use]
pub const fn effective_chunk_mode_for_layout(mode: Mode, layout: V8Layout, start: usize) -> Mode {
    match (mode, layout) {
        (Mode::QkPv, V8Layout::Shared) if start > 0 => Mode::Qk,
        _ => mode,
    }
}

/// Effective E4M3 attention mode for a chunk starting at `start`.
///
/// When `mode == Mode::QkPv` and [`v8_shared_enabled`] is active, single-chunk
/// prefills and chunk 0 (`start == 0`) run `Mode::QkPv` against the cache-shared
/// `V8` shadow plane written immediately before by this layer's prep on the same
/// stream, while continuation chunks (`start > 0`) fall back to `Mode::Qk`
/// (per-layer `K8` shadow + lossless BF16 `V` cache).
#[must_use]
pub fn effective_chunk_mode(mode: Mode, start: usize) -> Mode {
    let layout = if v8_shared_enabled() {
        V8Layout::Shared
    } else {
        V8Layout::PerLayer
    };
    effective_chunk_mode_for_layout(mode, layout, start)
}

fn parse_e4m3_out(value: Option<&str>) -> Result<bool> {
    let Some(raw) = value else {
        return Ok(true);
    };
    match raw.trim().to_ascii_lowercase().as_str() {
        "" | "1" | "on" | "true" => Ok(true),
        "0" | "off" | "false" => Ok(false),
        _ => anyhow::bail!(
            "{E4M3_OUT_SELECTOR} must be absent, 1, on, true, 0, off, or false; got {raw:?}"
        ),
    }
}

/// Whether direct static-prescaled E4M3 output from `qkpv_pipe` is enabled
/// (read once per process; defaults to `true`, opt out with
/// `ATLAS_QWEN4_PREFILL_QSA_E4M3_OUT=0|off|false`).
pub fn e4m3_out() -> Result<bool> {
    static S: OnceLock<Result<bool, String>> = OnceLock::new();
    S.get_or_init(|| {
        parse_e4m3_out(std::env::var(E4M3_OUT_SELECTOR).ok().as_deref()).map_err(|e| e.to_string())
    })
    .clone()
    .map_err(|e| anyhow::anyhow!(e))
}

fn parse_e4m3_2cta(value: Option<&str>) -> Result<bool> {
    let Some(raw) = value else {
        return Ok(false);
    };
    match raw.trim().to_ascii_lowercase().as_str() {
        "" | "0" | "off" | "false" => Ok(false),
        "1" | "on" | "true" => Ok(true),
        _ => anyhow::bail!(
            "{E4M3_2CTA_SELECTOR} must be absent, 1, on, true, 0, off, or false; got {raw:?}"
        ),
    }
}

/// Whether the 2-CTA/SM `qkpv_e4m3_pipe_2cta` kernel is enabled for direct E4M3
/// output (read once per process; defaults to `false`, opt in with
/// `ATLAS_QWEN4_PREFILL_QSA_E4M3_2CTA=1|on|true`).
pub fn e4m3_2cta() -> Result<bool> {
    static S: OnceLock<Result<bool, String>> = OnceLock::new();
    S.get_or_init(|| {
        parse_e4m3_2cta(std::env::var(E4M3_2CTA_SELECTOR).ok().as_deref())
            .map_err(|e| e.to_string())
    })
    .clone()
    .map_err(|e| anyhow::anyhow!(e))
}

thread_local! {
    static E4M3_OUT_ARMED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Scoped, one-shot handoff arm declaring that the attention core feeds an FP8
/// tensor-scale `o_proj` directly, so `qkpv_pipe` may emit static-prescaled
/// E4M3 rows straight into `qwen4_fast_proj`'s activation scratch.
pub(crate) struct E4m3OutArm {
    prev: bool,
    armed: bool,
}

impl E4m3OutArm {
    /// Arm direct E4M3 output for the enclosed attention core + `o_proj` pair.
    #[must_use]
    pub(crate) fn arm() -> Self {
        let prev = E4M3_OUT_ARMED.with(|c| c.replace(true));
        Self { prev, armed: true }
    }

    /// Explicitly disarm direct E4M3 output for the enclosed scope.
    #[must_use]
    pub(crate) fn disarm() -> Self {
        let prev = E4M3_OUT_ARMED.with(|c| c.replace(false));
        Self { prev, armed: false }
    }
}

impl Drop for E4m3OutArm {
    fn drop(&mut self) {
        E4M3_OUT_ARMED.with(|c| c.set(self.prev));
        if self.armed {
            crate::layers::qwen4_fast_proj::clear_prequant_fp8_act();
        }
    }
}

/// Consume the one-shot direct-E4M3-output arm inside `launch_e4m3_core`.
pub(crate) fn take_e4m3_out_armed() -> bool {
    E4M3_OUT_ARMED.with(|c| c.replace(false))
}

/// Static activation bound and scale contract for direct E4M3 output from
/// `qwen4_qsa_prefill_attn_fp8_qkpv_e4m3_pipe`.
///
/// Every `V8` shadow element is a finite E4M3 value (`|v8| <= 448`), `o / l`
/// is a convex combination of `V8` elements (`|o / l| <= 448`), `0 < sigmoid(g) < 1`,
/// and KV head `h` rescales by `1 / s_v[h]`. Thus every gated output element
/// satisfies `|y| <= 448 / min(s_v[0], s_v[1]) =: bound`, giving exact
/// power-of-two `inv_scale = min(s_v)` and `a_scale = 1 / min(s_v)` matching
/// `cublaslt::fp8_gemm_device_scaled` (`a8 * a_scale == y`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct E4m3OutScale {
    /// Guaranteed upper bound on `|gated_out|`: `448.0 / min(s_v[0], s_v[1])`.
    pub bound: f32,
    /// Quantization multiplier (`448.0 / bound == min(s_v[0], s_v[1])`).
    pub inv_scale: f32,
    /// cuBLASLt device activation tensor scale (`bound / 448.0 == 1.0 / inv_scale`).
    pub a_scale: f32,
    /// Per-KV-head epilogue scale folded into `ia = head_out_scale[h] / l`,
    /// equal to `inv_scale / s_v[h]` (`1.0` whenever `s_v[0] == s_v[1]`).
    pub head_out_scale: [f32; KV_HEADS],
}

/// Derive the static E4M3 output scale contract from the layer's `V8` prescales.
///
/// # Errors
/// Returns an error if any `V8` prescale is non-finite or non-positive.
pub fn e4m3_out_scale_for_v_prescales(v_prescales: [f32; KV_HEADS]) -> Result<E4m3OutScale> {
    ensure!(
        v_prescales.iter().all(|&s| s.is_finite() && s > 0.0),
        "{E4M3_OUT_SELECTOR}: invalid V prescales {v_prescales:?}"
    );
    let v_min = v_prescales[0].min(v_prescales[1]);
    let bound = E4M3_MAX / v_min;
    let inv_scale = v_min;
    let a_scale = 1.0 / v_min;
    ensure!(
        bound.is_finite() && bound > 0.0 && a_scale.is_finite() && a_scale > 0.0,
        "{E4M3_OUT_SELECTOR}: non-finite E4M3 output scale for V prescales {v_prescales:?}"
    );
    Ok(E4m3OutScale {
        bound,
        inv_scale,
        a_scale,
        head_out_scale: [inv_scale / v_prescales[0], inv_scale / v_prescales[1]],
    })
}

/// Inputs governing whether `qkpv_pipe` may emit direct static-prescaled E4M3 rows.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct E4m3OutEligibility {
    pub selector_enabled: bool,
    pub armed_for_fp8_oproj: bool,
    pub mode: Mode,
    pub pipe: bool,
    pub wdec_active: bool,
    pub sub_cta_active: bool,
    pub rows: usize,
    pub hidden_size: usize,
    pub diagnostics_active: bool,
}

/// Pure eligibility predicate for `qwen4_qsa_prefill_attn_fp8_qkpv_e4m3_pipe`.
#[must_use]
pub const fn e4m3_out_eligible(input: E4m3OutEligibility) -> bool {
    input.selector_enabled
        && input.armed_for_fp8_oproj
        && matches!(input.mode, Mode::QkPv)
        && input.pipe
        && !input.wdec_active
        && !input.sub_cta_active
        && input.rows >= E4M3_OUT_MIN_ROWS
        && input.hidden_size > 0
        && input.hidden_size.is_multiple_of(16)
        && !input.diagnostics_active
}

/// True when any exact-check, oracle capture, or diagnostic dump hook is active
/// that requires BF16 `raw_outputs` or exact unfused `amax_pipe` parity.
pub(crate) fn e4m3_out_diagnostics_active() -> Result<bool> {
    let dump_env_set = |name: &str| std::env::var_os(name).is_some_and(|v| !v.is_empty());
    let check_env_on = |name: &str| std::env::var(name).ok().as_deref() == Some("1");
    Ok(crate::layers::qwen4_oracle::root().is_some()
        || super::prefill_sparse_dump::dump_enabled()
        || crate::layers::qwen4_prefill_moe::hyper_check_selected()?
        || check_env_on("ATLAS_QWEN4_PREFILL_MOE_COMPACT_CHECK")
        || check_env_on("ATLAS_QWEN4_PREFILL_SSM_CHECK")
        || dump_env_set("ATLAS_QWEN4_FP8_PROJ_DUMP_DIR")
        || dump_env_set(crate::layers::qwen4_fp8_act_amax::OPROJ_TRIGGER))
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Prescale {
    pub q: f32,
    pub k: f32,
    pub v: f32,
}

fn parse_prescale(value: Option<&str>) -> Result<Option<Prescale>> {
    let Some(text) = value else {
        return Ok(None);
    };
    let parts: Vec<&str> = text.split(',').map(str::trim).collect();
    ensure!(
        parts.len() == 3,
        "{PRESCALE_SELECTOR} must be q,k,v; got {text:?}"
    );
    let mut s = [0.0f32; 3];
    for (dst, part) in s.iter_mut().zip(&parts) {
        let x: u32 = part
            .parse()
            .with_context(|| format!("{PRESCALE_SELECTOR}: {part:?} is not an integer"))?;
        // Powers of two keep the scaling exact; the core folds 1/(s_q s_k) and 1/s_v.
        ensure!(
            x.is_power_of_two() && x <= 256,
            "{PRESCALE_SELECTOR}: {x} is not a power of two in 1..=256"
        );
        *dst = x as f32;
    }
    Ok(Some(Prescale {
        q: s[0],
        k: s[1],
        v: s[2],
    }))
}

/// The global prescale override, if set (read once per process).
pub fn prescale_override() -> Result<Option<Prescale>> {
    static S: OnceLock<Result<Option<Prescale>, String>> = OnceLock::new();
    S.get_or_init(|| {
        parse_prescale(std::env::var(PRESCALE_SELECTOR).ok().as_deref()).map_err(|e| e.to_string())
    })
    .clone()
    .map_err(|e| anyhow::anyhow!(e))
}

pub(crate) struct Kernels {
    pub prep: KernelHandle,
    pub shadow_fill: KernelHandle,
    pub attn_qk: KernelHandle,
    pub attn_qkpv: KernelHandle,
    /// The V2 plan sort with blocks descending inside each need group (sink last).
    pub plan_sort_desc: KernelHandle,
}

/// Loaded only when the selector is on, so the default path never needs the modules.
pub(crate) fn kernels(gpu: &dyn GpuBackend) -> Result<&'static Kernels> {
    static K: OnceLock<Result<Kernels, String>> = OnceLock::new();
    K.get_or_init(|| {
        let load = |m: &str, f: &str| gpu.kernel(m, f).map_err(|e| format!("{m}::{f}: {e}"));
        let k = Kernels {
            prep: load(PREP_MODULE, "qwen4_attn_prep_fused_fp8")?,
            shadow_fill: load(PREP_MODULE, "qwen4_kv_shadow_e4m3")?,
            attn_qk: load(ATTN_MODULE, "qwen4_qsa_prefill_attn_fp8_qk")?,
            attn_qkpv: load(ATTN_MODULE, "qwen4_qsa_prefill_attn_fp8_qkpv")?,
            plan_sort_desc: load(ATTN_MODULE, "qwen4_qsa_prefill_plan_sort_desc")?,
        };
        gpu.set_kernel_max_dynamic_shared_memory(k.attn_qk, Mode::Qk.smem() as u32)
            .and_then(|()| {
                gpu.set_kernel_max_dynamic_shared_memory(k.attn_qkpv, Mode::QkPv.smem() as u32)
            })
            .map_err(|e| e.to_string())?;
        Ok(k)
    })
    .as_ref()
    .map_err(|e| anyhow::anyhow!("{SELECTOR}: kernels unavailable: {e}"))
}

/// `ATLAS_QWEN4_PREFILL_OPROJ_AMAX_FUSED`: the same cores with the o_proj absmax reduced in their
/// epilogue (`..._amax` entries: one more argument, the zeroed u32 slot). Loaded only when that
/// selector is on, so the default path never needs them.
pub(crate) struct AmaxKernels {
    pub attn_qk: KernelHandle,
    pub attn_qkpv: KernelHandle,
}

pub(crate) fn amax_kernels(gpu: &dyn GpuBackend) -> Result<&'static AmaxKernels> {
    static K: OnceLock<Result<AmaxKernels, String>> = OnceLock::new();
    K.get_or_init(|| {
        let load = |f: &str| {
            gpu.kernel(ATTN_MODULE, f)
                .map_err(|e| format!("{ATTN_MODULE}::{f}: {e}"))
        };
        let k = AmaxKernels {
            attn_qk: load("qwen4_qsa_prefill_attn_fp8_qk_amax")?,
            attn_qkpv: load("qwen4_qsa_prefill_attn_fp8_qkpv_amax")?,
        };
        gpu.set_kernel_max_dynamic_shared_memory(k.attn_qk, Mode::Qk.smem() as u32)
            .and_then(|()| {
                gpu.set_kernel_max_dynamic_shared_memory(k.attn_qkpv, Mode::QkPv.smem() as u32)
            })
            .map_err(|e| e.to_string())?;
        Ok(k)
    })
    .as_ref()
    .map_err(|e| {
        anyhow::anyhow!(
            "{}: o_proj absmax kernels unavailable: {e}",
            crate::layers::qwen4_fp8_act_amax::OPROJ_SELECTOR
        )
    })
}

pub(crate) struct PrepFusedKernels {
    pub prep_vec: KernelHandle,
    pub prep_v2: KernelHandle,
    pub prep_vec_k8: KernelHandle,
    pub prep_v2_k8: KernelHandle,
}

pub(crate) fn prep_fused_kernels(
    gpu: &dyn GpuBackend,
) -> Result<Option<&'static PrepFusedKernels>> {
    if !super::prefill_sparse::prep_fused_selected()? {
        return Ok(None);
    }
    static K: OnceLock<Result<PrepFusedKernels, String>> = OnceLock::new();
    let k = K
        .get_or_init(|| {
            let load = |f: &str| {
                gpu.kernel(PREP_MODULE, f)
                    .map_err(|e| format!("{PREP_MODULE}::{f}: {e}"))
            };
            Ok(PrepFusedKernels {
                prep_vec: load("qwen4_attn_prep_fused_fp8_vec")?,
                prep_v2: load("qwen4_attn_prep_fused_fp8_v2")?,
                prep_vec_k8: load("qwen4_attn_prep_fused_fp8_vec_k8")?,
                prep_v2_k8: load("qwen4_attn_prep_fused_fp8_v2_k8")?,
            })
        })
        .as_ref()
        .map_err(|e| {
            anyhow::anyhow!(
                "{}: FP8 prep v2 kernels unavailable: {e}",
                super::prefill_sparse::PREP_FUSED_SELECTOR
            )
        })?;
    Ok(Some(k))
}

pub(crate) struct WdecKernels {
    pub qkpv_4k: KernelHandle,
    pub qkpv_amax_4k: KernelHandle,
    pub qkpv_8k: KernelHandle,
    pub qkpv_amax_8k: KernelHandle,
}

pub(crate) fn wdec_kernels(gpu: &dyn GpuBackend) -> Result<&'static WdecKernels> {
    static K: OnceLock<Result<WdecKernels, String>> = OnceLock::new();
    K.get_or_init(|| {
        let load = |f: &str| {
            gpu.kernel(ATTN_MODULE, f)
                .map_err(|e| format!("{ATTN_MODULE}::{f}: {e}"))
        };
        let k = WdecKernels {
            qkpv_4k: load("qwen4_qsa_prefill_attn_fp8_qkpv_wdec")?,
            qkpv_amax_4k: load("qwen4_qsa_prefill_attn_fp8_qkpv_amax_wdec")?,
            qkpv_8k: load("qwen4_qsa_prefill_attn_fp8_qkpv_wdec8k")?,
            qkpv_amax_8k: load("qwen4_qsa_prefill_attn_fp8_qkpv_amax_wdec8k")?,
        };
        let smem4 = WarpDecouple::Wdec4k.smem() as u32;
        let smem8 = WarpDecouple::Wdec8k.smem() as u32;
        gpu.set_kernel_max_dynamic_shared_memory(k.qkpv_4k, smem4)
            .and_then(|()| gpu.set_kernel_max_dynamic_shared_memory(k.qkpv_amax_4k, smem4))
            .and_then(|()| gpu.set_kernel_max_dynamic_shared_memory(k.qkpv_8k, smem8))
            .and_then(|()| gpu.set_kernel_max_dynamic_shared_memory(k.qkpv_amax_8k, smem8))
            .map_err(|e| e.to_string())?;
        Ok(k)
    })
    .as_ref()
    .map_err(|e| anyhow::anyhow!("{WARP_DECOUPLE_SELECTOR}: kernels unavailable: {e}"))
}

pub(crate) struct SubKernels {
    pub qkpv_half: KernelHandle,
    pub qkpv_amax_half: KernelHandle,
    pub qkpv_qtr: KernelHandle,
    pub qkpv_amax_qtr: KernelHandle,
}

pub(crate) fn sub_kernels(gpu: &dyn GpuBackend) -> Result<&'static SubKernels> {
    static K: OnceLock<Result<SubKernels, String>> = OnceLock::new();
    K.get_or_init(|| {
        let load = |f: &str| {
            gpu.kernel(ATTN_MODULE, f)
                .map_err(|e| format!("{ATTN_MODULE}::{f}: {e}"))
        };
        Ok(SubKernels {
            qkpv_half: load("qwen4_qsa_prefill_attn_fp8_qkpv_half")?,
            qkpv_amax_half: load("qwen4_qsa_prefill_attn_fp8_qkpv_amax_half")?,
            qkpv_qtr: load("qwen4_qsa_prefill_attn_fp8_qkpv_qtr")?,
            qkpv_amax_qtr: load("qwen4_qsa_prefill_attn_fp8_qkpv_amax_qtr")?,
        })
    })
    .as_ref()
    .map_err(|e| anyhow::anyhow!("{SUB_CTA_SELECTOR}: kernels unavailable: {e}"))
}

pub(crate) struct PipeKernels {
    pub qk: KernelHandle,
    pub qk_amax: KernelHandle,
    pub qkpv: KernelHandle,
    pub qkpv_amax: KernelHandle,
}

pub(crate) fn pipe_kernels(gpu: &dyn GpuBackend) -> Result<&'static PipeKernels> {
    static K: OnceLock<Result<PipeKernels, String>> = OnceLock::new();
    K.get_or_init(|| {
        let load = |f: &str| {
            gpu.kernel(ATTN_MODULE, f)
                .map_err(|e| format!("{ATTN_MODULE}::{f}: {e}"))
        };
        let k = PipeKernels {
            qk: load("qwen4_qsa_prefill_attn_fp8_qk_pipe")?,
            qk_amax: load("qwen4_qsa_prefill_attn_fp8_qk_amax_pipe")?,
            qkpv: load("qwen4_qsa_prefill_attn_fp8_qkpv_pipe")?,
            qkpv_amax: load("qwen4_qsa_prefill_attn_fp8_qkpv_amax_pipe")?,
        };
        let qk_smem = Mode::Qk.smem() as u32;
        let qkpv_smem = Mode::QkPv.smem() as u32;
        gpu.set_kernel_max_dynamic_shared_memory(k.qk, qk_smem)
            .and_then(|()| gpu.set_kernel_max_dynamic_shared_memory(k.qk_amax, qk_smem))
            .and_then(|()| gpu.set_kernel_max_dynamic_shared_memory(k.qkpv, qkpv_smem))
            .and_then(|()| gpu.set_kernel_max_dynamic_shared_memory(k.qkpv_amax, qkpv_smem))
            .map_err(|e| e.to_string())?;
        Ok(k)
    })
    .as_ref()
    .map_err(|e| anyhow::anyhow!("{PIPE_SELECTOR}: kernels unavailable: {e}"))
}

pub(crate) fn pipe_e4m3_kernel(gpu: &dyn GpuBackend) -> Result<KernelHandle> {
    static K: OnceLock<Result<KernelHandle, String>> = OnceLock::new();
    K.get_or_init(|| {
        let name = "qwen4_qsa_prefill_attn_fp8_qkpv_e4m3_pipe";
        let k = gpu
            .kernel(ATTN_MODULE, name)
            .map_err(|e| format!("{ATTN_MODULE}::{name}: {e}"))?;
        gpu.set_kernel_max_dynamic_shared_memory(k, Mode::QkPv.smem() as u32)
            .map_err(|e| e.to_string())?;
        Ok(k)
    })
    .clone()
    .map_err(|e| anyhow::anyhow!("{E4M3_OUT_SELECTOR}: kernel unavailable: {e}"))
}

pub(crate) fn pipe_e4m3_2cta_kernel(gpu: &dyn GpuBackend) -> Result<KernelHandle> {
    static K: OnceLock<Result<KernelHandle, String>> = OnceLock::new();
    K.get_or_init(|| {
        let name = "qwen4_qsa_prefill_attn_fp8_qkpv_e4m3_pipe_2cta";
        let k = gpu
            .kernel(ATTN_MODULE, name)
            .map_err(|e| format!("{ATTN_MODULE}::{name}: {e}"))?;
        gpu.set_kernel_max_dynamic_shared_memory(k, E4M3_2CTA_SMEM as u32)
            .map_err(|e| e.to_string())?;
        Ok(k)
    })
    .clone()
    .map_err(|e| anyhow::anyhow!("{E4M3_2CTA_SELECTOR}: kernel unavailable: {e}"))
}

pub(crate) fn plan_fused_kernel(gpu: &dyn GpuBackend) -> Result<KernelHandle> {
    static K: OnceLock<Result<KernelHandle, String>> = OnceLock::new();
    K.get_or_init(|| {
        gpu.kernel(ATTN_MODULE, "qwen4_qsa_prefill_plan_fused_sort_desc")
            .map_err(|e| format!("{ATTN_MODULE}::qwen4_qsa_prefill_plan_fused_sort_desc: {e}"))
    })
    .clone()
    .map_err(|e| anyhow::anyhow!("{PLAN_FUSED_SELECTOR}: kernel unavailable: {e}"))
}

/// Static power-of-two prescales of one attention layer: Q per GQA group (query
/// heads `12g .. 12g+11`), K and V per KV head. Folded exactly into the core's
/// softmax scale (1/(s_q s_k)) and output scale (1/s_v).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LayerScales {
    pub q: [f32; KV_HEADS],
    pub k: [f32; KV_HEADS],
    pub v: [f32; KV_HEADS],
}

/// A Q or K head row is RMS-normed (sum (x_i r)^2 <= 256, so |x_i r| <= 16),
/// scaled by (1 + w) and rotated in pairs, so every element is bounded by
/// 16 * max|1 + w|. The prescale is the largest power of two that keeps 1.0125x
/// that bound (bf16 rounding margin) at or below 448: no element can saturate.
pub fn norm_bound_prescale(weight_bf16: &[u16]) -> f32 {
    let max_gain = weight_bf16
        .iter()
        .map(|&b| (1.0 + f32::from_bits(u32::from(b) << 16)).abs())
        .fold(0.0f32, f32::max);
    let bound = 16.0 * 1.0125 * max_gain.max(f32::MIN_POSITIVE);
    2f32.powi((448.0 / bound).log2().floor().clamp(0.0, 8.0) as i32)
}

fn scale_table() -> &'static Mutex<HashMap<usize, LayerScales>> {
    static S: OnceLock<Mutex<HashMap<usize, LayerScales>>> = OnceLock::new();
    S.get_or_init(|| Mutex::new(HashMap::new()))
}

/// The layer's prescales, derived once from its Q/K norm weights (device bf16
/// `[256]`, read back to the host on first use) unless the global override is set.
pub fn layer_scales(
    gpu: &dyn GpuBackend,
    layer: usize,
    q_norm_weight: DevicePtr,
    k_norm_weight: DevicePtr,
) -> Result<LayerScales> {
    let mut map = scale_table()
        .lock()
        .map_err(|_| anyhow::anyhow!("{SELECTOR}: scale lock poisoned"))?;
    if let Some(sc) = map.get(&layer) {
        return Ok(*sc);
    }
    let sc = if let Some(p) = prescale_override()? {
        LayerScales {
            q: [p.q; KV_HEADS],
            k: [p.k; KV_HEADS],
            v: [p.v; KV_HEADS],
        }
    } else {
        let read = |ptr: DevicePtr| -> Result<Vec<u16>> {
            let mut buf = vec![0u8; HEAD_DIM * 2];
            gpu.copy_d2h(ptr, &mut buf)
                .with_context(|| format!("{SELECTOR}: read norm weight (layer {layer})"))?;
            Ok(buf
                .chunks_exact(2)
                .map(|c| u16::from_le_bytes([c[0], c[1]]))
                .collect())
        };
        let (q, k) = (
            norm_bound_prescale(&read(q_norm_weight)?),
            norm_bound_prescale(&read(k_norm_weight)?),
        );
        LayerScales {
            q: [q; KV_HEADS],
            k: [k; KV_HEADS],
            v: [V_PRESCALE; KV_HEADS],
        }
    };
    tracing::info!(
        "QWEN4_PREFILL_QSA_FP8 prescales layer={layer} q={:?} k={:?} v={:?}",
        sc.q,
        sc.k,
        sc.v
    );
    map.insert(layer, sc);
    Ok(sc)
}

/// The prescales the prep derived for `layer` (the core runs after its prep).
pub fn scales(layer: usize) -> Result<LayerScales> {
    scale_table()
        .lock()
        .map_err(|_| anyhow::anyhow!("{SELECTOR}: scale lock poisoned"))?
        .get(&layer)
        .copied()
        .ok_or_else(|| anyhow::anyhow!("{SELECTOR}: layer {layer} has no prescales (prep not run)"))
}

/// Bytes of one chunk's E4M3 queries `[rows][24*256]`.
pub fn q8_bytes(rows: usize) -> usize {
    rows * QUERY_HEADS * HEAD_DIM
}

/// Grow-only E4M3 query scratch shared by every attention layer: the prep of a
/// layer writes it and the same layer's core reads it, all on the one prefill
/// stream. Never freed: a free here would run mid-request (the GB10
/// free-under-traffic class), so growing retires the old block (kept, not
/// freed), as the QSA sparse scratch does. [`reserve_q8`] sizes it at load for
/// the largest chunk the engine admits, so a request never grows it.
pub fn q8_scratch(gpu: &dyn GpuBackend, rows: usize) -> Result<DevicePtr> {
    static S: Mutex<Option<(DevicePtr, usize)>> = Mutex::new(None);
    let bytes = q8_bytes(rows);
    let mut guard = S
        .lock()
        .map_err(|_| anyhow::anyhow!("{SELECTOR}: q8 scratch lock poisoned"))?;
    let retired = match *guard {
        Some((ptr, cap)) if cap >= bytes => return Ok(ptr),
        Some((_, cap)) => cap,
        None => 0,
    };
    let ptr = gpu
        .alloc(bytes)
        .with_context(|| format!("{SELECTOR}: alloc {bytes} B q8 scratch"))?;
    spark_runtime::scratch_init::init_grown(gpu, ptr, bytes, "qsa_fp8_q8")?;
    if retired > 0 {
        tracing::warn!(
            "QWEN4_PREFILL_QSA_FP8 q8 scratch grown bytes={bytes} retired={retired} freed=0 (past the load-time reserve)"
        );
    } else {
        tracing::info!("QWEN4_PREFILL_QSA_FP8 q8 scratch allocated bytes={bytes}");
    }
    *guard = Some((ptr, bytes));
    Ok(ptr)
}

/// Load-time sizing of the E4M3 query scratch for chunks of up to `max_rows`
/// rows (no-op unless the E4M3 core is selected), so no request allocates,
/// grows or frees it.
pub fn reserve_q8(gpu: &dyn GpuBackend, max_rows: usize) -> Result<()> {
    if mode()?.is_none() || max_rows == 0 {
        return Ok(());
    }
    q8_scratch(gpu, max_rows)?;
    tracing::info!(
        "QWEN4_PREFILL_SCRATCH_RESERVED site=qsa_fp8_q8 rows={max_rows} bytes={} never_freed=1",
        q8_bytes(max_rows)
    );
    Ok(())
}

/// Per attention layer: the K8/V8 shadow pools and the end of the last chunk
/// the prep wrote (rows [0, end) of the shadow are valid).
struct LayerShadow {
    k8: DevicePtr,
    v8: DevicePtr,
    end: Option<usize>,
    num_blocks: usize,
    block_bytes: usize,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct ShadowKey {
    cache_id: u64,
    layer: usize,
}

fn shadows() -> &'static Mutex<HashMap<ShadowKey, LayerShadow>> {
    // Entries intentionally follow the process-lifetime allocation policy of
    // the backing GPU pointers. A hot swap gives its new PagedKvCache a fresh
    // monotonic identity, so stale metadata can never alias the new model; the
    // old, small host records remain until process exit just like other global
    // prefill caches.
    static S: OnceLock<Mutex<HashMap<ShadowKey, LayerShadow>>> = OnceLock::new();
    S.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Shadow block stride in bytes: the BF16 page layout at one byte per element.
pub fn shadow_block_bytes(kv_cache: &PagedKvCache) -> Result<usize> {
    kv_cache
        .block_size()
        .checked_mul(KV_HEADS)
        .and_then(|bytes| bytes.checked_mul(HEAD_DIM))
        .ok_or_else(|| anyhow::anyhow!("{SELECTOR}: shadow block-byte geometry overflow"))
}

/// Forget every layer's shadow end for `cache_id`, so the next chunk refills
/// [0, start) from the BF16 pages. A `SeqStateBlob` import writes BF16 pages
/// the shadow never saw; without this a chunk starting where another
/// sequence's last chunk ended would read stale shadow rows.
pub(crate) fn invalidate_shadow_ends(cache_id: u64) -> Result<()> {
    let mut map = shadows()
        .lock()
        .map_err(|_| anyhow::anyhow!("{SELECTOR}: shadow lock poisoned"))?;
    for (key, shadow) in map.iter_mut() {
        if key.cache_id == cache_id {
            shadow.end = None;
        }
    }
    Ok(())
}

/// Invalidate per-layer E4M3 shadow continuity for `kv_cache` so the next
/// chunk starting at `start > 0` always refills `[0, start)` from BF16 pages.
///
/// Must be called on a Marconi prefix-cache hit before resuming an uncached
/// suffix on the fast prefill path: if an intervening request ended at the
/// same offset `start`, `s.end == Some(start)` would otherwise skip
/// [`refill_shadow`] and read stale E4M3 shadow rows.
///
/// # Errors
/// Returns an error if the shadow registry mutex is poisoned.
pub fn invalidate_shadow_continuity(kv_cache: &PagedKvCache) -> Result<()> {
    invalidate_shadow_ends(kv_cache.cache_id())
}

/// Total E4M3 shadow planes across `num_layers` for `(mode, layout)`:
/// - [`Mode::Qk`]: `num_layers` (`K8`-only).
/// - [`Mode::QkPv`] + [`V8Layout::Shared`]: `num_layers + 1` when `num_layers > 0`
///   (`num_layers` per-layer `K8` planes + `1` cache-shared `V8` plane), else `0`.
/// - [`Mode::QkPv`] + [`V8Layout::PerLayer`]: `2 * num_layers` (`K8 + V8` per layer).
pub fn checked_total_shadow_planes(
    mode: Mode,
    layout: V8Layout,
    num_layers: usize,
) -> Result<usize> {
    if num_layers == 0 {
        return Ok(0);
    }
    match (mode, layout) {
        (Mode::Qk, _) => Ok(num_layers),
        (Mode::QkPv, V8Layout::Shared) => num_layers.checked_add(1).ok_or_else(|| {
            anyhow::anyhow!(
                "{SELECTOR}: shadow plane count overflow for mode={} layers={num_layers}",
                mode.name()
            )
        }),
        (Mode::QkPv, V8Layout::PerLayer) => num_layers.checked_mul(2).ok_or_else(|| {
            anyhow::anyhow!(
                "{SELECTOR}: shadow plane count overflow for mode={} layers={num_layers}",
                mode.name()
            )
        }),
    }
}

/// Total bytes for the E4M3 shadow pools across `num_layers` for an explicit
/// `(mode, layout)`.
pub fn checked_shadow_reserve_bytes_for_layout(
    mode: Mode,
    layout: V8Layout,
    block_size: usize,
    num_blocks: usize,
    num_layers: usize,
) -> Result<usize> {
    let total_planes = checked_total_shadow_planes(mode, layout, num_layers)?;
    block_size
        .checked_mul(KV_HEADS)
        .and_then(|bytes| bytes.checked_mul(HEAD_DIM))
        .and_then(|bytes| bytes.checked_mul(num_blocks))
        .and_then(|bytes| bytes.checked_mul(total_planes))
        .ok_or_else(|| {
            anyhow::anyhow!(
                "{SELECTOR}: shadow reserve overflow for mode={} layout={layout:?} block_size={block_size} blocks={num_blocks} layers={num_layers}",
                mode.name()
            )
        })
}

/// Total bytes for the E4M3 shadow pools across `num_layers` for `mode` under
/// the active [`v8_layout`] (`num_layers` planes for [`Mode::Qk`]; `num_layers + 1`
/// planes for [`Mode::QkPv`] with shared `V8`, or `2 * num_layers` when opted out).
///
/// This is the single accounting formula shared by server preflight, factory
/// pool sizing, and the eager allocator. `num_blocks` is the target cache's
/// physical pool size, including its dummy/spare blocks, rather than a request's
/// logical length.
pub fn checked_shadow_reserve_bytes_for_mode(
    mode: Mode,
    block_size: usize,
    num_blocks: usize,
    num_layers: usize,
) -> Result<usize> {
    let layout = v8_layout()?;
    checked_shadow_reserve_bytes_for_layout(mode, layout, block_size, num_blocks, num_layers)
}

/// Total bytes for the E4M3 K and V shadow pools across `num_layers` under
/// [`Mode::QkPv`] and the active [`v8_layout`].
pub fn checked_shadow_reserve_bytes(
    block_size: usize,
    num_blocks: usize,
    num_layers: usize,
) -> Result<usize> {
    checked_shadow_reserve_bytes_for_mode(Mode::QkPv, block_size, num_blocks, num_layers)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SharedV8State {
    Absent,
    Allocated,
}

fn subtract_existing_shared_v8_bytes(
    pending_bytes: usize,
    mode: Mode,
    layout: V8Layout,
    missing_layers: usize,
    shared_v8: SharedV8State,
    num_blocks: usize,
    block_bytes: usize,
) -> Result<usize> {
    if missing_layers > 0
        && shared_v8 == SharedV8State::Allocated
        && mode == Mode::QkPv
        && layout == V8Layout::Shared
    {
        let plane_bytes = num_blocks
            .checked_mul(block_bytes)
            .ok_or_else(|| anyhow::anyhow!("{SELECTOR}: shared V8 plane-byte overflow"))?;
        pending_bytes
            .checked_sub(plane_bytes)
            .ok_or_else(|| anyhow::anyhow!("{SELECTOR}: pending shadow byte underflow"))
    } else {
        Ok(pending_bytes)
    }
}

fn shadow_key(kv_cache: &PagedKvCache, layer: usize) -> Result<ShadowKey> {
    ensure!(
        layer < kv_cache.num_layers(),
        "{SELECTOR}: shadow layer {layer} is outside cache {} ({} layers)",
        kv_cache.cache_id(),
        kv_cache.num_layers()
    );
    Ok(ShadowKey {
        cache_id: kv_cache.cache_id(),
        layer,
    })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ShadowFloorAdmission {
    Enforced,
    Overridden,
}

fn shadow_floor_admission(
    mem_available: Option<usize>,
    shadow_bytes: usize,
    temporary_bytes: usize,
    floor: usize,
    override_selected: bool,
) -> Result<ShadowFloorAdmission> {
    ensure!(
        floor >= startup_ledger::QUALIFICATION_FLOOR_BYTES || override_selected,
        "{SELECTOR}: shadow floor is below the qualification floor without an explicit override"
    );
    let required = shadow_bytes
        .checked_add(temporary_bytes)
        .and_then(|bytes| bytes.checked_add(floor))
        .ok_or_else(|| anyhow::anyhow!("{SELECTOR}: shadow floor arithmetic overflow"))?;
    // An explicit override is never an ordinary admission, even with headroom.
    if override_selected {
        return Ok(ShadowFloorAdmission::Overridden);
    }
    let available = mem_available.ok_or_else(|| {
        anyhow::anyhow!(
            "{SELECTOR}: refusing shadow allocation: /proc/meminfo has no readable MemAvailable; \
             set {}=1 only for an explicitly NONQUALIFYING run",
            startup_ledger::FLOOR_OVERRIDE_SELECTOR
        )
    })?;
    ensure!(
        available >= required,
        "{SELECTOR}: refusing shadow allocation: MemAvailable={available} B, \
         shadow={shadow_bytes} B, temporary={temporary_bytes} B, floor={floor} B, \
         required={required} B; selected QSA FP8 mode is unchanged"
    );
    Ok(ShadowFloorAdmission::Enforced)
}

fn ensure_shadow_allocation_above_floor(shadow_bytes: usize) -> Result<()> {
    let floor = startup_ledger::memavailable_floor_bytes()?;
    let available = startup_ledger::mem_available_bytes();
    // alloc + synchronous device memset uses no tensor-sized host staging or
    // second device tensor. Driver/registry overhead is not a measured bound.
    let temporary_bytes = 0;
    let admission = shadow_floor_admission(
        available,
        shadow_bytes,
        temporary_bytes,
        floor,
        startup_ledger::floor_override_selected(),
    )?;
    if admission == ShadowFloorAdmission::Overridden {
        tracing::warn!(
            "{SELECTOR}: shadow allocation floor OVERRIDDEN, NONQUALIFYING \
             available_bytes={available:?} shadow_bytes={shadow_bytes} \
             temporary_bytes={temporary_bytes} floor_bytes={floor}"
        );
    }
    Ok(())
}

/// The layer's K8/(optional V8) shadow pools, allocated on first use (kept for
/// the process). In [`Mode::Qk`], `V8` is not allocated and is returned as a
/// null [`DevicePtr`].
pub fn shadow(
    gpu: &dyn GpuBackend,
    layer: usize,
    kv_cache: &PagedKvCache,
) -> Result<(DevicePtr, DevicePtr)> {
    let active_mode = mode()?.unwrap_or(Mode::QkPv);
    shadow_with_mode_admission(
        gpu,
        layer,
        kv_cache,
        active_mode,
        &mut ensure_shadow_allocation_above_floor,
    )
}

#[cfg(test)]
fn shadow_with_admission(
    gpu: &dyn GpuBackend,
    layer: usize,
    kv_cache: &PagedKvCache,
    admit: &mut impl FnMut(usize) -> Result<()>,
) -> Result<(DevicePtr, DevicePtr)> {
    shadow_with_mode_admission(gpu, layer, kv_cache, Mode::QkPv, admit)
}

fn shadow_with_mode_admission(
    gpu: &dyn GpuBackend,
    layer: usize,
    kv_cache: &PagedKvCache,
    mode: Mode,
    admit: &mut impl FnMut(usize) -> Result<()>,
) -> Result<(DevicePtr, DevicePtr)> {
    shadow_with_mode_layout_admission(gpu, layer, kv_cache, mode, v8_layout()?, admit)
}

fn shadow_with_mode_layout_admission(
    gpu: &dyn GpuBackend,
    layer: usize,
    kv_cache: &PagedKvCache,
    mode: Mode,
    layout: V8Layout,
    admit: &mut impl FnMut(usize) -> Result<()>,
) -> Result<(DevicePtr, DevicePtr)> {
    let key = shadow_key(kv_cache, layer)?;
    let block_bytes = shadow_block_bytes(kv_cache)?;
    let bytes = kv_cache
        .num_blocks()
        .checked_mul(block_bytes)
        .ok_or_else(|| anyhow::anyhow!("{SELECTOR}: layer {layer} shadow-byte overflow"))?;
    let mut map = shadows()
        .lock()
        .map_err(|_| anyhow::anyhow!("{SELECTOR}: shadow lock poisoned"))?;
    if let Some(s) = map.get(&key) {
        ensure!(
            s.num_blocks == kv_cache.num_blocks() && s.block_bytes == block_bytes,
            "{SELECTOR}: cache {} layer {layer} shadow geometry changed",
            kv_cache.cache_id()
        );
        ensure!(
            mode == Mode::Qk || !s.v8.is_null(),
            "{SELECTOR}: cache {} layer {layer} has K8-only shadow but mode=qkpv requested",
            kv_cache.cache_id()
        );
        return Ok((s.k8, s.v8));
    }
    let existing_shared_v8 = if mode == Mode::QkPv && layout == V8Layout::Shared {
        let found = map
            .iter()
            .find(|(k, s)| k.cache_id == key.cache_id && !s.v8.is_null())
            .map(|(_, s)| (s.v8, s.num_blocks, s.block_bytes));
        if let Some((v8, nb, bb)) = found {
            ensure!(
                nb == kv_cache.num_blocks() && bb == block_bytes,
                "{SELECTOR}: cache {} shared V8 shadow geometry changed",
                kv_cache.cache_id()
            );
            Some(v8)
        } else {
            None
        }
    } else {
        None
    };
    let planes = match (mode, layout, existing_shared_v8) {
        (Mode::Qk, _, _) => 1,
        (Mode::QkPv, V8Layout::PerLayer, _) => 2,
        (Mode::QkPv, V8Layout::Shared, Some(_)) => 1,
        (Mode::QkPv, V8Layout::Shared, None) => 2,
    };
    let layer_bytes = bytes
        .checked_mul(planes)
        .ok_or_else(|| anyhow::anyhow!("{SELECTOR}: layer {layer} K/V shadow-byte overflow"))?;
    // Check while holding the registry lock, after the idempotent-hit return.
    // MemAvailable is a sample, not an atomic reservation against other users.
    admit(layer_bytes)?;
    // Keep K (and optional V) in one allocation so a second allocation cannot fail after
    // leaving a half-created layer behind. The synchronous memset is not just
    // data initialisation: GB10 device allocations are backed by unified memory
    // and may not debit MemAvailable until first touch. Materialising every
    // page here makes the later fast-projection weight-cache admission observe
    // the real post-shadow memory floor before the server becomes ready.
    let k8 = gpu.alloc(layer_bytes).with_context(|| {
        format!(
            "{SELECTOR}: alloc {layer_bytes} B materialized shadow (layer {layer}, mode={})",
            mode.name()
        )
    })?;
    let v8 = if planes == 2 {
        k8.offset(bytes)
    } else {
        existing_shared_v8.unwrap_or(DevicePtr(0))
    };
    gpu.memset(k8, 0, layer_bytes).with_context(|| {
        format!(
            "{SELECTOR}: materialize {layer_bytes} B shadow (layer {layer}, mode={})",
            mode.name()
        )
    })?;
    tracing::info!(
        "QWEN4_PREFILL_QSA_FP8 shadow allocated+materialized layer={layer} mode={} layout={layout:?} planes={planes} shared_v8={} bytes={} blocks={}",
        mode.name(),
        existing_shared_v8.is_some(),
        layer_bytes,
        kv_cache.num_blocks()
    );
    map.insert(
        key,
        LayerShadow {
            k8,
            v8,
            end: None,
            num_blocks: kv_cache.num_blocks(),
            block_bytes,
        },
    );
    Ok((k8, v8))
}

fn validate_target_shadow_geometry(kv_cache: &PagedKvCache, expected_layers: usize) -> Result<()> {
    ensure!(
        kv_cache.block_size() >= 4 && kv_cache.block_size().is_multiple_of(4),
        "{SELECTOR}: target shadow block size {} must be positive and divisible by 4",
        kv_cache.block_size()
    );
    ensure!(
        u32::try_from(kv_cache.block_size()).is_ok(),
        "{SELECTOR}: target shadow block size {} exceeds the kernel ABI",
        kv_cache.block_size()
    );
    ensure!(
        kv_cache.num_blocks() > 0,
        "{SELECTOR}: target shadow cache must contain at least one physical block"
    );
    ensure!(
        u32::try_from(kv_cache.num_blocks()).is_ok(),
        "{SELECTOR}: target shadow cache has {} blocks, exceeding the paged-KV block-id ABI",
        kv_cache.num_blocks()
    );
    ensure!(
        expected_layers > 0 && kv_cache.num_layers() == expected_layers,
        "{SELECTOR}: target shadow reserve expected {expected_layers} attention layers, cache {} has {}",
        kv_cache.cache_id(),
        kv_cache.num_layers()
    );
    for layer in 0..expected_layers {
        ensure!(
            kv_cache.dtype_for_layer(layer) == KvCacheDtype::Bf16,
            "{SELECTOR}: target layer {layer} shadow requires BF16 KV"
        );
        ensure!(
            kv_cache.config().dims_for_layer(layer) == (KV_HEADS, HEAD_DIM),
            "{SELECTOR}: target layer {layer} shadow requires {KV_HEADS}x{HEAD_DIM} KV geometry"
        );
        let expected_bf16_stride = shadow_block_bytes(kv_cache)?
            .checked_mul(2)
            .ok_or_else(|| anyhow::anyhow!("{SELECTOR}: BF16 stride overflow"))?;
        ensure!(
            kv_cache.k_block_stride_bytes_for_layer(layer) == expected_bf16_stride
                && kv_cache.v_block_stride_bytes_for_layer(layer) == expected_bf16_stride,
            "{SELECTOR}: target layer {layer} BF16 K/V strides must both equal {expected_bf16_stride} bytes"
        );
    }
    Ok(())
}

fn reserve_shadows_for_cache(
    gpu: &dyn GpuBackend,
    kv_cache: &PagedKvCache,
    mode: Mode,
) -> Result<usize> {
    reserve_shadows_with_mode_admission(
        gpu,
        kv_cache,
        mode,
        &mut ensure_shadow_allocation_above_floor,
    )
}

#[cfg(test)]
fn reserve_shadows_with_admission(
    gpu: &dyn GpuBackend,
    kv_cache: &PagedKvCache,
    admit: &mut impl FnMut(usize) -> Result<()>,
) -> Result<usize> {
    reserve_shadows_with_mode_admission(gpu, kv_cache, Mode::QkPv, admit)
}

fn reserve_shadows_with_mode_admission(
    gpu: &dyn GpuBackend,
    kv_cache: &PagedKvCache,
    mode: Mode,
    admit: &mut impl FnMut(usize) -> Result<()>,
) -> Result<usize> {
    let layout = v8_layout()?;
    let bytes = checked_shadow_reserve_bytes_for_mode(
        mode,
        kv_cache.block_size(),
        kv_cache.num_blocks(),
        kv_cache.num_layers(),
    )?;
    let block_bytes = shadow_block_bytes(kv_cache)?;
    let (missing_layers, shared_v8) = {
        let map = shadows()
            .lock()
            .map_err(|_| anyhow::anyhow!("{SELECTOR}: shadow lock poisoned"))?;
        let mut missing = 0usize;
        let mut shared_v8 = SharedV8State::Absent;
        for layer in 0..kv_cache.num_layers() {
            let key = shadow_key(kv_cache, layer)?;
            if let Some(shadow) = map.get(&key) {
                ensure!(
                    shadow.num_blocks == kv_cache.num_blocks() && shadow.block_bytes == block_bytes,
                    "{SELECTOR}: cache {} layer {layer} shadow geometry changed",
                    kv_cache.cache_id()
                );
                if !shadow.v8.is_null() {
                    shared_v8 = SharedV8State::Allocated;
                }
            } else {
                missing = missing.checked_add(1).ok_or_else(|| {
                    anyhow::anyhow!("{SELECTOR}: missing shadow layer count overflow")
                })?;
            }
        }
        (missing, shared_v8)
    };
    let pending_bytes = checked_shadow_reserve_bytes_for_mode(
        mode,
        kv_cache.block_size(),
        kv_cache.num_blocks(),
        missing_layers,
    )?;
    let pending_bytes = subtract_existing_shared_v8_bytes(
        pending_bytes,
        mode,
        layout,
        missing_layers,
        shared_v8,
        kv_cache.num_blocks(),
        block_bytes,
    )?;
    if pending_bytes > 0 {
        admit(pending_bytes)?;
    }
    for layer in 0..kv_cache.num_layers() {
        shadow_with_mode_admission(gpu, layer, kv_cache, mode, admit)?;
    }
    Ok(bytes)
}

/// Eagerly allocate the selected target model's QSA FP8 shadows.
///
/// The caller supplies the target attention-layer count. This rejects the
/// native-MTP cache (one layer, separate [`PagedKvCache`] identity) instead of
/// silently populating layer zero in the process-global shadow registry.
pub fn reserve_target_shadows(
    gpu: &dyn GpuBackend,
    kv_cache: &PagedKvCache,
    expected_layers: usize,
) -> Result<usize> {
    let Some(active_mode) = mode()? else {
        return Ok(0);
    };
    let layout = v8_layout()?;
    validate_target_shadow_geometry(kv_cache, expected_layers)?;
    let bytes = reserve_shadows_for_cache(gpu, kv_cache, active_mode)?;
    let total_planes = checked_total_shadow_planes(active_mode, layout, expected_layers)?;
    tracing::info!(
        "QWEN4_PREFILL_SCRATCH_RESERVED site=qsa_fp8_target_shadows mode={} v8_layout={layout:?} planes={total_planes} cache_id={} layers={} blocks={} bytes={} materialized=1 never_freed=1",
        active_mode.name(),
        kv_cache.cache_id(),
        expected_layers,
        kv_cache.num_blocks(),
        bytes
    );
    Ok(bytes)
}

/// Record that the prep wrote rows [start, start + rows) of `layer`'s shadow and
/// return the prefix [0, start) that must be refilled from the BF16 pages first
/// (empty when this chunk continues the layer's previous one).
pub fn note_prepped(
    kv_cache: &PagedKvCache,
    layer: usize,
    start: usize,
    rows: usize,
) -> Result<usize> {
    let key = shadow_key(kv_cache, layer)?;
    let mut map = shadows()
        .lock()
        .map_err(|_| anyhow::anyhow!("{SELECTOR}: shadow lock poisoned"))?;
    let s = map.get_mut(&key).ok_or_else(|| {
        anyhow::anyhow!(
            "{SELECTOR}: cache {} layer {layer} has no shadow",
            kv_cache.cache_id()
        )
    })?;
    let refill = if start == 0 || s.end == Some(start) {
        0
    } else {
        start
    };
    s.end = Some(
        start
            .checked_add(rows)
            .ok_or_else(|| anyhow::anyhow!("{SELECTOR}: shadow row extent overflow"))?,
    );
    Ok(refill)
}

/// Refill the shadow for tokens [0, end) of the sequence from its BF16 pages.
#[allow(clippy::too_many_arguments)]
pub fn refill_shadow(
    gpu: &dyn GpuBackend,
    kv_cache: &PagedKvCache,
    layer: usize,
    block_table: DevicePtr,
    end: usize,
    stream: u64,
) -> Result<()> {
    if end == 0 {
        return Ok(());
    }
    let (k8, v8) = shadow(gpu, layer, kv_cache)?;
    let active_mode = mode()?.unwrap_or(Mode::QkPv);
    let _ = v8_shared()?;
    // Refill only executes for `start = end > 0`. Under shared-V8, continuation
    // chunks run `Mode::Qk` (reading BF16 V directly), so pass a null `v8` to
    // refill only `k8` and avoid overwriting or reading V8 traffic unnecessarily.
    let refill_v8 = match effective_chunk_mode(active_mode, end) {
        Mode::QkPv => v8,
        Mode::Qk => DevicePtr::NULL,
    };
    let sc = scales(layer)?;
    KernelLaunch::new(gpu, kernels(gpu)?.shadow_fill)
        .grid([u32::try_from(end.div_ceil(4))?, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(kv_cache.k_pool_ptr(layer))
        .arg_ptr(kv_cache.v_pool_ptr(layer))
        .arg_ptr(k8)
        .arg_ptr(refill_v8)
        .arg_ptr(block_table)
        .arg_u32(0)
        .arg_u32(u32::try_from(end)?)
        .arg_u32(kv_cache.block_size() as u32)
        .arg_f32(sc.k[0])
        .arg_f32(sc.k[1])
        .arg_f32(sc.v[0])
        .arg_f32(sc.v[1])
        .launch(stream)?;
    static LOGGED: std::sync::Once = std::sync::Once::new();
    LOGGED.call_once(|| {
        tracing::warn!("QWEN4_PREFILL_QSA_FP8 shadow refill layer={layer} tokens={end}")
    });
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use spark_runtime::gpu::mock::MockGpuBackend;
    use spark_runtime::kv_cache::KvCacheConfig;

    fn qsa_cache(gpu: &MockGpuBackend, layers: usize, blocks: usize) -> PagedKvCache {
        PagedKvCache::new(
            KvCacheConfig {
                block_size: 4,
                num_kv_heads: KV_HEADS,
                head_dim: HEAD_DIM,
                num_layers: layers,
                dtype: KvCacheDtype::Bf16,
                layer_dtypes: vec![],
                layer_dims: vec![],
                cache_blocks_per_seq: None,
            },
            blocks,
            gpu,
        )
        .unwrap()
    }

    fn ample_shadow_admission(bytes: usize) -> Result<()> {
        assert_eq!(
            shadow_floor_admission(
                Some(usize::MAX),
                bytes,
                0,
                startup_ledger::QUALIFICATION_FLOOR_BYTES,
                false,
            )?,
            ShadowFloorAdmission::Enforced
        );
        Ok(())
    }

    #[test]
    fn shadow_floor_checks_exact_boundary_and_temporary_cost() {
        let floor = startup_ledger::QUALIFICATION_FLOOR_BYTES;
        assert_eq!(
            shadow_floor_admission(Some(floor + 3), 2, 1, floor, false).unwrap(),
            ShadowFloorAdmission::Enforced
        );
        assert!(shadow_floor_admission(Some(floor + 2), 2, 1, floor, false).is_err());
        assert!(shadow_floor_admission(Some(floor - 1), 0, 0, floor, false).is_err());
        assert!(shadow_floor_admission(Some(usize::MAX), 0, 0, floor - 1, false).is_err());
    }

    #[test]
    fn shadow_floor_missing_telemetry_refuses_and_override_is_always_nonqualifying() {
        let floor = startup_ledger::QUALIFICATION_FLOOR_BYTES;
        assert!(shadow_floor_admission(None, 1, 0, floor, false).is_err());
        for available in [None, Some(0), Some(usize::MAX)] {
            assert_eq!(
                shadow_floor_admission(available, 1, 0, floor, true).unwrap(),
                ShadowFloorAdmission::Overridden
            );
        }
        assert_eq!(
            shadow_floor_admission(None, 1, 0, 0, true).unwrap(),
            ShadowFloorAdmission::Overridden
        );
    }

    #[test]
    fn shadow_floor_overflow_refuses_even_when_overridden() {
        let floor = startup_ledger::QUALIFICATION_FLOOR_BYTES;
        for overridden in [false, true] {
            assert!(
                shadow_floor_admission(Some(usize::MAX), usize::MAX, 1, floor, overridden).is_err()
            );
            assert!(
                shadow_floor_admission(Some(usize::MAX), usize::MAX, 0, floor, overridden).is_err()
            );
            assert!(
                shadow_floor_admission(Some(usize::MAX), 0, usize::MAX, floor, overridden).is_err()
            );
        }
    }

    fn reserve_shadows_with_mode_layout_admission(
        gpu: &dyn GpuBackend,
        kv_cache: &PagedKvCache,
        mode: Mode,
        layout: V8Layout,
        admit: &mut impl FnMut(usize) -> Result<()>,
    ) -> Result<usize> {
        let bytes = checked_shadow_reserve_bytes_for_layout(
            mode,
            layout,
            kv_cache.block_size(),
            kv_cache.num_blocks(),
            kv_cache.num_layers(),
        )?;
        let block_bytes = shadow_block_bytes(kv_cache)?;
        let (missing_layers, shared_v8) = {
            let map = shadows()
                .lock()
                .map_err(|_| anyhow::anyhow!("{SELECTOR}: shadow lock poisoned"))?;
            let mut missing = 0usize;
            let mut shared_v8 = SharedV8State::Absent;
            for layer in 0..kv_cache.num_layers() {
                let key = shadow_key(kv_cache, layer)?;
                if let Some(shadow) = map.get(&key) {
                    ensure!(
                        shadow.num_blocks == kv_cache.num_blocks()
                            && shadow.block_bytes == block_bytes,
                        "{SELECTOR}: cache {} layer {layer} shadow geometry changed",
                        kv_cache.cache_id()
                    );
                    if !shadow.v8.is_null() {
                        shared_v8 = SharedV8State::Allocated;
                    }
                } else {
                    missing = missing.checked_add(1).ok_or_else(|| {
                        anyhow::anyhow!("{SELECTOR}: missing shadow layer count overflow")
                    })?;
                }
            }
            (missing, shared_v8)
        };
        let pending_bytes = checked_shadow_reserve_bytes_for_layout(
            mode,
            layout,
            kv_cache.block_size(),
            kv_cache.num_blocks(),
            missing_layers,
        )?;
        let pending_bytes = subtract_existing_shared_v8_bytes(
            pending_bytes,
            mode,
            layout,
            missing_layers,
            shared_v8,
            kv_cache.num_blocks(),
            block_bytes,
        )?;
        if pending_bytes > 0 {
            admit(pending_bytes)?;
        }
        for layer in 0..kv_cache.num_layers() {
            shadow_with_mode_layout_admission(gpu, layer, kv_cache, mode, layout, admit)?;
        }
        Ok(bytes)
    }

    #[test]
    fn aggregate_shadow_refusal_allocates_and_materializes_nothing() {
        let gpu = MockGpuBackend::new();
        let cache = qsa_cache(&gpu, 2, 3);
        let allocations = gpu.alloc_count();
        let memsets = gpu.memset_count();
        let mut offered = Vec::new();
        let mut refuse = |bytes| {
            offered.push(bytes);
            anyhow::bail!("synthetic aggregate refusal")
        };
        assert!(reserve_shadows_with_admission(&gpu, &cache, &mut refuse).is_err());
        assert_eq!(offered, vec![18_432]);
        assert_eq!(gpu.alloc_count(), allocations);
        assert_eq!(gpu.memset_count(), memsets);

        let mut offered_per_layer = Vec::new();
        let mut refuse_per_layer = |bytes| {
            offered_per_layer.push(bytes);
            anyhow::bail!("synthetic per-layer aggregate refusal")
        };
        assert!(
            reserve_shadows_with_mode_layout_admission(
                &gpu,
                &cache,
                Mode::QkPv,
                V8Layout::PerLayer,
                &mut refuse_per_layer,
            )
            .is_err()
        );
        assert_eq!(offered_per_layer, vec![24_576]);
        assert_eq!(gpu.alloc_count(), allocations);
        assert_eq!(gpu.memset_count(), memsets);
    }

    #[test]
    fn shadow_rechecks_each_layer_and_retry_charges_only_missing_layers() {
        let gpu = MockGpuBackend::new();
        let cache = qsa_cache(&gpu, 2, 3);
        let allocations = gpu.alloc_count();
        let memsets = gpu.memset_count();
        let mut offered = Vec::new();
        let mut changing_headroom = |bytes| {
            offered.push(bytes);
            if offered.len() == 3 {
                anyhow::bail!("synthetic headroom loss before second layer");
            }
            ample_shadow_admission(bytes)
        };
        assert!(
            reserve_shadows_with_mode_layout_admission(
                &gpu,
                &cache,
                Mode::QkPv,
                V8Layout::PerLayer,
                &mut changing_headroom,
            )
            .is_err()
        );
        assert_eq!(offered, vec![24_576, 12_288, 12_288]);
        assert_eq!(gpu.alloc_count() - allocations, 1);
        assert_eq!(gpu.memset_count() - memsets, 1);

        let mut retry_offered = Vec::new();
        let mut retry = |bytes| {
            retry_offered.push(bytes);
            ample_shadow_admission(bytes)
        };
        assert_eq!(
            reserve_shadows_with_mode_layout_admission(
                &gpu,
                &cache,
                Mode::QkPv,
                V8Layout::PerLayer,
                &mut retry,
            )
            .unwrap(),
            24_576
        );
        assert_eq!(retry_offered, vec![12_288, 12_288]);
        let mut no_admission_needed =
            |_| anyhow::bail!("existing shadows must not be charged twice");
        assert_eq!(
            reserve_shadows_with_mode_layout_admission(
                &gpu,
                &cache,
                Mode::QkPv,
                V8Layout::PerLayer,
                &mut no_admission_needed,
            )
            .unwrap(),
            24_576
        );
        assert!(
            shadow_with_mode_layout_admission(
                &gpu,
                0,
                &cache,
                Mode::QkPv,
                V8Layout::PerLayer,
                &mut no_admission_needed,
            )
            .is_ok()
        );
        assert_eq!(gpu.alloc_count() - allocations, 2);
        assert_eq!(gpu.memset_count() - memsets, 2);

        // Default Shared-V8 layout charges 2 planes on layer 0 (K8 + shared V8)
        // and 1 plane on layer 1 (K8 only); retry after layer 0 charges only K8.
        let shared_cache = qsa_cache(&gpu, 2, 3);
        let shared_allocs = gpu.alloc_count();
        let shared_memsets = gpu.memset_count();
        let mut shared_offered = Vec::new();
        let mut shared_changing = |bytes| {
            shared_offered.push(bytes);
            if shared_offered.len() == 3 {
                anyhow::bail!("synthetic headroom loss before second shared-V8 layer");
            }
            ample_shadow_admission(bytes)
        };
        assert!(reserve_shadows_with_admission(&gpu, &shared_cache, &mut shared_changing).is_err());
        assert_eq!(shared_offered, vec![18_432, 12_288, 6_144]);
        assert_eq!(gpu.alloc_count() - shared_allocs, 1);
        assert_eq!(gpu.memset_count() - shared_memsets, 1);

        let mut shared_retry_offered = Vec::new();
        let mut shared_retry = |bytes| {
            shared_retry_offered.push(bytes);
            ample_shadow_admission(bytes)
        };
        assert_eq!(
            reserve_shadows_with_admission(&gpu, &shared_cache, &mut shared_retry).unwrap(),
            18_432
        );
        assert_eq!(shared_retry_offered, vec![6_144, 6_144]);
        assert_eq!(
            reserve_shadows_with_admission(&gpu, &shared_cache, &mut no_admission_needed).unwrap(),
            18_432
        );
        assert!(shadow_with_admission(&gpu, 0, &shared_cache, &mut no_admission_needed).is_ok());
        assert_eq!(gpu.alloc_count() - shared_allocs, 2);
        assert_eq!(gpu.memset_count() - shared_memsets, 2);
    }

    #[test]
    fn lazy_shadow_refusal_precedes_device_allocation() {
        let gpu = MockGpuBackend::new();
        let cache = qsa_cache(&gpu, 2, 3);
        let allocations = gpu.alloc_count();
        let memsets = gpu.memset_count();
        let mut refuse = |_| anyhow::bail!("synthetic lazy refusal");
        assert!(shadow_with_admission(&gpu, 0, &cache, &mut refuse).is_err());
        assert_eq!(gpu.alloc_count(), allocations);
        assert_eq!(gpu.memset_count(), memsets);
    }

    #[test]
    fn selector_values() {
        assert_eq!(parse(None).unwrap(), None);
        assert_eq!(parse(Some("0")).unwrap(), None);
        assert_eq!(parse(Some("qk")).unwrap(), Some(Mode::Qk));
        assert_eq!(parse(Some("qkpv")).unwrap(), Some(Mode::QkPv));
        assert!(parse(Some("1")).is_err());
    }

    #[test]
    fn checked_shadow_reserve_matches_configured_physical_geometry() {
        // Default Shared-V8: 12 K8 planes + 1 shared V8 plane = 13 planes.
        assert_eq!(
            checked_shadow_reserve_bytes(16, 2_050, 12).unwrap(),
            218_316_800
        );
        assert_eq!(
            checked_shadow_reserve_bytes(16, 8_450, 12).unwrap(),
            899_891_200
        );
        assert_eq!(
            checked_shadow_reserve_bytes(16, 16_384, 12).unwrap(),
            1_744_830_464
        );
        assert_eq!(checked_shadow_reserve_bytes(16, 16_384, 0).unwrap(), 0);
        // Legacy PerLayer-V8: 12 K8 + 12 V8 = 24 planes (3,072 MiB at 262K vs 1,664 MiB shared).
        assert_eq!(
            checked_shadow_reserve_bytes_for_layout(Mode::QkPv, V8Layout::PerLayer, 16, 2_050, 12)
                .unwrap(),
            403_046_400
        );
        assert_eq!(
            checked_shadow_reserve_bytes_for_layout(Mode::QkPv, V8Layout::PerLayer, 16, 8_450, 12)
                .unwrap(),
            1_661_337_600
        );
        assert_eq!(
            checked_shadow_reserve_bytes_for_layout(Mode::QkPv, V8Layout::PerLayer, 16, 16_384, 12)
                .unwrap(),
            3_221_225_472
        );
        assert_eq!(
            checked_shadow_reserve_bytes_for_layout(Mode::QkPv, V8Layout::PerLayer, 16, 16_384, 0)
                .unwrap(),
            0
        );
        // Shared-V8 saves exactly 11 planes = 1,408 MiB at 262K (16,384 blocks x 16 tok).
        assert_eq!(
            checked_shadow_reserve_bytes_for_layout(Mode::QkPv, V8Layout::PerLayer, 16, 16_384, 12)
                .unwrap()
                - checked_shadow_reserve_bytes_for_layout(
                    Mode::QkPv,
                    V8Layout::Shared,
                    16,
                    16_384,
                    12
                )
                .unwrap(),
            1_408 * 1024 * 1024
        );
        assert!(checked_shadow_reserve_bytes(usize::MAX, 2, 12).is_err());
        assert!(checked_shadow_reserve_bytes(16, usize::MAX, 12).is_err());
        assert!(checked_shadow_reserve_bytes(16, 2, usize::MAX).is_err());
        // Mode::Qk allocates K8-only (12 planes: 1 plane less than Shared Mode::QkPv, half of PerLayer).
        assert_eq!(
            checked_shadow_reserve_bytes_for_mode(Mode::Qk, 16, 16_401, 12).unwrap()
                + 16_401 * 16 * KV_HEADS * HEAD_DIM,
            checked_shadow_reserve_bytes_for_mode(Mode::QkPv, 16, 16_401, 12).unwrap()
        );
        assert_eq!(
            checked_shadow_reserve_bytes_for_layout(Mode::Qk, V8Layout::PerLayer, 16, 16_401, 12)
                .unwrap()
                * 2,
            checked_shadow_reserve_bytes_for_layout(Mode::QkPv, V8Layout::PerLayer, 16, 16_401, 12)
                .unwrap()
        );
    }

    #[test]
    fn mode_qk_allocates_k8_only_with_null_v8() {
        let gpu = MockGpuBackend::new();
        let cache = qsa_cache(&gpu, 2, 3);
        let mut offered = Vec::new();
        let mut record_admit = |bytes| {
            offered.push(bytes);
            ample_shadow_admission(bytes)
        };
        let total =
            reserve_shadows_with_mode_admission(&gpu, &cache, Mode::Qk, &mut record_admit).unwrap();
        assert_eq!(total, 12_288);
        assert_eq!(offered, vec![12_288, 6_144, 6_144]);
        let (k8, v8) =
            shadow_with_mode_admission(&gpu, 0, &cache, Mode::Qk, &mut ample_shadow_admission)
                .unwrap();
        assert!(!k8.is_null());
        assert!(v8.is_null(), "Mode::Qk must not allocate V8 shadow");
        // Upgrading a K8-only cache entry to Mode::QkPv must fail closed.
        assert!(
            shadow_with_mode_admission(&gpu, 0, &cache, Mode::QkPv, &mut ample_shadow_admission)
                .is_err()
        );
    }

    #[test]
    fn eager_shadows_are_materialized_idempotent_and_cache_scoped() {
        let gpu = MockGpuBackend::new();
        let first = qsa_cache(&gpu, 2, 3);
        validate_target_shadow_geometry(&first, 2).unwrap();

        let allocs_before = gpu.alloc_count();
        let memsets_before = gpu.memset_count();
        assert_eq!(
            reserve_shadows_with_admission(&gpu, &first, &mut ample_shadow_admission).unwrap(),
            18_432
        );
        assert_eq!(gpu.alloc_count() - allocs_before, 2);
        assert_eq!(gpu.memset_count() - memsets_before, 2);
        let first_ptrs =
            shadow_with_admission(&gpu, 0, &first, &mut ample_shadow_admission).unwrap();
        let first_layer1 =
            shadow_with_admission(&gpu, 1, &first, &mut ample_shadow_admission).unwrap();
        assert_eq!(first_ptrs.1.0 - first_ptrs.0.0, 6_144);
        assert_ne!(first_ptrs.0, first_layer1.0);
        assert_eq!(first_ptrs.1, first_layer1.1);

        let allocs_after_first = gpu.alloc_count();
        let memsets_after_first = gpu.memset_count();
        assert_eq!(
            reserve_shadows_with_admission(&gpu, &first, &mut ample_shadow_admission).unwrap(),
            18_432
        );
        assert_eq!(gpu.alloc_count(), allocs_after_first);
        assert_eq!(gpu.memset_count(), memsets_after_first);

        let second = qsa_cache(&gpu, 2, 3);
        let memsets_before_second_shadow = gpu.memset_count();
        let second_ptrs =
            shadow_with_admission(&gpu, 0, &second, &mut ample_shadow_admission).unwrap();
        assert_ne!(first.cache_id(), second.cache_id());
        assert_ne!(first_ptrs.0, second_ptrs.0);
        assert_ne!(first_ptrs.1, second_ptrs.1);
        assert_eq!(gpu.memset_count(), memsets_before_second_shadow + 1);
    }

    #[test]
    fn target_geometry_rejects_native_mtp_and_incompatible_cache_shapes() {
        let gpu = MockGpuBackend::new();
        let mtp = qsa_cache(&gpu, 1, 2);
        assert!(validate_target_shadow_geometry(&mtp, 12).is_err());

        let zero_blocks = qsa_cache(&gpu, 12, 0);
        assert!(validate_target_shadow_geometry(&zero_blocks, 12).is_err());

        let fp8 = PagedKvCache::new(
            KvCacheConfig {
                block_size: 4,
                num_kv_heads: KV_HEADS,
                head_dim: HEAD_DIM,
                num_layers: 12,
                dtype: KvCacheDtype::Fp8,
                layer_dtypes: vec![],
                layer_dims: vec![],
                cache_blocks_per_seq: None,
            },
            2,
            &gpu,
        )
        .unwrap();
        assert!(validate_target_shadow_geometry(&fp8, 12).is_err());

        for block_size in [2, 6] {
            let misaligned = PagedKvCache::new(
                KvCacheConfig {
                    block_size,
                    num_kv_heads: KV_HEADS,
                    head_dim: HEAD_DIM,
                    num_layers: 12,
                    dtype: KvCacheDtype::Bf16,
                    layer_dtypes: vec![],
                    layer_dims: vec![],
                    cache_blocks_per_seq: None,
                },
                1,
                &gpu,
            )
            .unwrap();
            let error = validate_target_shadow_geometry(&misaligned, 12).unwrap_err();
            assert!(error.to_string().contains("divisible by 4"));
        }

        let mut mixed_dims = vec![(KV_HEADS, HEAD_DIM); 12];
        mixed_dims[5] = (KV_HEADS - 1, HEAD_DIM);
        let wrong_dims = PagedKvCache::new(
            KvCacheConfig {
                block_size: 4,
                num_kv_heads: KV_HEADS,
                head_dim: HEAD_DIM,
                num_layers: 12,
                dtype: KvCacheDtype::Bf16,
                layer_dtypes: vec![],
                layer_dims: mixed_dims,
                cache_blocks_per_seq: None,
            },
            1,
            &gpu,
        )
        .unwrap();
        let error = validate_target_shadow_geometry(&wrong_dims, 12).unwrap_err();
        assert!(error.to_string().contains("2x256 KV geometry"));
    }

    #[test]
    fn prescale_values() {
        assert_eq!(parse_prescale(None).unwrap(), None);
        assert_eq!(
            parse_prescale(Some("8, 4,16")).unwrap(),
            Some(Prescale {
                q: 8.0,
                k: 4.0,
                v: 16.0
            })
        );
        assert!(parse_prescale(Some("8,6,16")).is_err());
        assert!(parse_prescale(Some("8,8")).is_err());
        assert!(parse_prescale(Some("8,8,512")).is_err());
    }

    #[test]
    fn norm_bound_prescales() {
        let bf16 = |x: f32| (x.to_bits() >> 16) as u16;
        // max|1 + w| = 1.594 (layer 3's q_norm) -> bound 25.8 -> 16; 2.375 -> 38.5 -> 8.
        assert_eq!(norm_bound_prescale(&[bf16(0.594), bf16(-0.2)]), 16.0);
        assert_eq!(norm_bound_prescale(&[bf16(1.375)]), 8.0);
        // A large gain clamps to 1, a zero gain to 256.
        assert_eq!(norm_bound_prescale(&[bf16(40.0)]), 1.0);
        assert_eq!(norm_bound_prescale(&[bf16(-1.0)]), 256.0);
    }

    #[test]
    fn smem_fits_the_optin_limit() {
        assert_eq!(Mode::QkPv.smem(), 2 * 16384 + ATTN_CAP * 4);
        assert_eq!(Mode::Qk.smem(), 2 * 24576 + ATTN_CAP * 4);
        assert_eq!(WarpDecouple::Wdec4k.smem(), 32768 + ATTN_CAP * 4);
        assert_eq!(WarpDecouple::Wdec8k.smem(), 65536 + ATTN_CAP * 4);
        assert!(Mode::Qk.smem() <= 101_376 && Mode::QkPv.smem() <= 101_376);
        assert!(WarpDecouple::Wdec4k.smem() <= 101_376 && WarpDecouple::Wdec8k.smem() <= 101_376);
    }

    #[test]
    fn warp_decouple_values() {
        assert_eq!(parse_warp_decouple(None).unwrap(), None);
        assert_eq!(parse_warp_decouple(Some("0")).unwrap(), None);
        assert_eq!(
            parse_warp_decouple(Some("1")).unwrap(),
            Some(WarpDecouple::Wdec4k)
        );
        assert_eq!(
            parse_warp_decouple(Some("4k")).unwrap(),
            Some(WarpDecouple::Wdec4k)
        );
        assert_eq!(
            parse_warp_decouple(Some("2")).unwrap(),
            Some(WarpDecouple::Wdec8k)
        );
        assert_eq!(
            parse_warp_decouple(Some("8k")).unwrap(),
            Some(WarpDecouple::Wdec8k)
        );
        assert!(parse_warp_decouple(Some("3")).is_err());
    }

    #[test]
    fn v8_shared_selector_values() {
        assert!(parse_v8_shared(None).unwrap());
        assert!(parse_v8_shared(Some("")).unwrap());
        assert!(parse_v8_shared(Some("1")).unwrap());
        assert!(parse_v8_shared(Some("on")).unwrap());
        assert!(parse_v8_shared(Some("true")).unwrap());
        assert!(parse_v8_shared(Some(" TRUE ")).unwrap());
        assert!(!parse_v8_shared(Some("0")).unwrap());
        assert!(!parse_v8_shared(Some("off")).unwrap());
        assert!(!parse_v8_shared(Some("false")).unwrap());
        assert!(!parse_v8_shared(Some(" OFF ")).unwrap());
        assert!(parse_v8_shared(Some("2")).is_err());
        assert!(parse_v8_shared(Some("yes")).is_err());
    }

    #[test]
    fn effective_chunk_mode_falls_back_to_qk_only_for_shared_v8_continuations() {
        assert_eq!(
            effective_chunk_mode_for_layout(Mode::QkPv, V8Layout::Shared, 0),
            Mode::QkPv
        );
        assert_eq!(
            effective_chunk_mode_for_layout(Mode::QkPv, V8Layout::Shared, 1),
            Mode::Qk
        );
        assert_eq!(
            effective_chunk_mode_for_layout(Mode::QkPv, V8Layout::Shared, 16_384),
            Mode::Qk
        );
        assert_eq!(
            effective_chunk_mode_for_layout(Mode::QkPv, V8Layout::PerLayer, 0),
            Mode::QkPv
        );
        assert_eq!(
            effective_chunk_mode_for_layout(Mode::QkPv, V8Layout::PerLayer, 16_384),
            Mode::QkPv
        );
        assert_eq!(
            effective_chunk_mode_for_layout(Mode::Qk, V8Layout::Shared, 0),
            Mode::Qk
        );
        assert_eq!(
            effective_chunk_mode_for_layout(Mode::Qk, V8Layout::Shared, 16_384),
            Mode::Qk
        );
        assert_eq!(
            effective_chunk_mode_for_layout(Mode::Qk, V8Layout::PerLayer, 0),
            Mode::Qk
        );
        assert_eq!(
            effective_chunk_mode_for_layout(Mode::Qk, V8Layout::PerLayer, 16_384),
            Mode::Qk
        );
    }

    #[test]
    fn shared_v8_shares_single_plane_across_12_layers_and_per_layer_keeps_distinct_v8() {
        let gpu = MockGpuBackend::new();
        let shared = qsa_cache(&gpu, 12, 3);
        let plane_bytes = 3 * 4 * KV_HEADS * HEAD_DIM;
        let total_shared = reserve_shadows_with_mode_layout_admission(
            &gpu,
            &shared,
            Mode::QkPv,
            V8Layout::Shared,
            &mut ample_shadow_admission,
        )
        .unwrap();
        assert_eq!(total_shared, 13 * plane_bytes);

        let (k8_0, v8_shared) = shadow_with_mode_layout_admission(
            &gpu,
            0,
            &shared,
            Mode::QkPv,
            V8Layout::Shared,
            &mut ample_shadow_admission,
        )
        .unwrap();
        assert!(!k8_0.is_null() && !v8_shared.is_null());
        for layer in 1..12 {
            let (k8_l, v8_l) = shadow_with_mode_layout_admission(
                &gpu,
                layer,
                &shared,
                Mode::QkPv,
                V8Layout::Shared,
                &mut ample_shadow_admission,
            )
            .unwrap();
            assert!(!k8_l.is_null());
            assert_ne!(k8_l, k8_0, "layer {layer} must own a distinct K8 plane");
            assert_eq!(
                v8_l, v8_shared,
                "layer {layer} must share layer 0's single V8 plane"
            );
        }

        let per_layer = qsa_cache(&gpu, 12, 3);
        let total_per_layer = reserve_shadows_with_mode_layout_admission(
            &gpu,
            &per_layer,
            Mode::QkPv,
            V8Layout::PerLayer,
            &mut ample_shadow_admission,
        )
        .unwrap();
        assert_eq!(total_per_layer, 24 * plane_bytes);

        let (_, v8_pl_0) = shadow_with_mode_layout_admission(
            &gpu,
            0,
            &per_layer,
            Mode::QkPv,
            V8Layout::PerLayer,
            &mut ample_shadow_admission,
        )
        .unwrap();
        for layer in 1..12 {
            let (_, v8_pl_l) = shadow_with_mode_layout_admission(
                &gpu,
                layer,
                &per_layer,
                Mode::QkPv,
                V8Layout::PerLayer,
                &mut ample_shadow_admission,
            )
            .unwrap();
            assert_ne!(
                v8_pl_l, v8_pl_0,
                "PerLayer layer {layer} must own a distinct V8 plane"
            );
        }
    }

    #[test]
    fn e4m3_out_selector_values() {
        assert!(parse_e4m3_out(None).unwrap());
        assert!(parse_e4m3_out(Some("")).unwrap());
        assert!(parse_e4m3_out(Some("1")).unwrap());
        assert!(parse_e4m3_out(Some("on")).unwrap());
        assert!(parse_e4m3_out(Some("true")).unwrap());
        assert!(parse_e4m3_out(Some(" TRUE ")).unwrap());
        assert!(!parse_e4m3_out(Some("0")).unwrap());
        assert!(!parse_e4m3_out(Some("off")).unwrap());
        assert!(!parse_e4m3_out(Some("false")).unwrap());
        assert!(!parse_e4m3_out(Some(" OFF ")).unwrap());
        assert!(parse_e4m3_out(Some("2")).is_err());
        assert!(parse_e4m3_out(Some("yes")).is_err());
    }

    #[test]
    fn e4m3_2cta_selector_and_smem_fit_two_ctas_per_sm() {
        assert!(!parse_e4m3_2cta(None).unwrap());
        assert!(!parse_e4m3_2cta(Some("")).unwrap());
        assert!(parse_e4m3_2cta(Some("1")).unwrap());
        assert!(parse_e4m3_2cta(Some("on")).unwrap());
        assert!(parse_e4m3_2cta(Some("true")).unwrap());
        assert!(parse_e4m3_2cta(Some(" TRUE ")).unwrap());
        assert!(!parse_e4m3_2cta(Some("0")).unwrap());
        assert!(!parse_e4m3_2cta(Some("off")).unwrap());
        assert!(!parse_e4m3_2cta(Some("false")).unwrap());
        assert!(!parse_e4m3_2cta(Some(" OFF ")).unwrap());
        assert!(parse_e4m3_2cta(Some("2")).is_err());
        assert!(parse_e4m3_2cta(Some("yes")).is_err());
        assert_eq!(E4M3_2CTA_SMEM, 43_584);
        const {
            assert!(E4M3_2CTA_SMEM <= 49_152);
            assert!(E4M3_2CTA_SMEM * 2 <= 102_400);
        }
        assert_eq!(E4M3_2CTA_SUB_FACTOR, 2);
    }

    #[test]
    fn e4m3_out_scale_matches_v_prescale_bound_and_cublaslt_contract() {
        let default_scale = e4m3_out_scale_for_v_prescales([V_PRESCALE, V_PRESCALE]).unwrap();
        assert_eq!(default_scale.bound, 56.0);
        assert_eq!(default_scale.inv_scale, 8.0);
        assert_eq!(default_scale.a_scale, 0.125);
        assert_eq!(default_scale.head_out_scale, [1.0, 1.0]);
        assert_eq!(E4M3_MAX / default_scale.bound, default_scale.inv_scale);
        assert_eq!(default_scale.bound / E4M3_MAX, default_scale.a_scale);
        assert_eq!(default_scale.inv_scale * default_scale.a_scale, 1.0);

        // Asymmetric per-KV-head V prescales keep both heads inside [-448, +448]
        // and reconstruct the exact true magnitude via `a8 * a_scale`.
        let asym = e4m3_out_scale_for_v_prescales([8.0, 16.0]).unwrap();
        assert_eq!(asym.bound, 56.0);
        assert_eq!(asym.inv_scale, 8.0);
        assert_eq!(asym.a_scale, 0.125);
        assert_eq!(asym.head_out_scale, [1.0, 0.5]);
        for (h, s_v) in [8.0f32, 16.0f32].into_iter().enumerate() {
            let v_true = 2.0f32;
            let v8 = v_true * s_v;
            let q = v8 * asym.head_out_scale[h];
            assert!(q.abs() <= E4M3_MAX);
            assert_eq!(q * asym.a_scale, v_true);
        }

        // Every legal power-of-two V prescale override (1..=256) is exact.
        for exp in 0..=8 {
            let sv = 2f32.powi(exp);
            let sc = e4m3_out_scale_for_v_prescales([sv, sv]).unwrap();
            assert_eq!(sc.bound, E4M3_MAX / sv);
            assert_eq!(sc.inv_scale, sv);
            assert_eq!(sc.a_scale, 1.0 / sv);
            assert_eq!(sc.head_out_scale, [1.0, 1.0]);
            assert_eq!(E4M3_MAX / sc.bound, sc.inv_scale);
            assert_eq!(sc.bound / E4M3_MAX, sc.a_scale);
        }

        // Non-positive or non-finite prescales fail closed.
        assert!(e4m3_out_scale_for_v_prescales([0.0, 8.0]).is_err());
        assert!(e4m3_out_scale_for_v_prescales([8.0, -8.0]).is_err());
        assert!(e4m3_out_scale_for_v_prescales([f32::NAN, 8.0]).is_err());
        assert!(e4m3_out_scale_for_v_prescales([f32::INFINITY, 8.0]).is_err());
    }

    #[test]
    fn e4m3_out_eligibility_gates_all_conditions() {
        let base = E4m3OutEligibility {
            selector_enabled: true,
            armed_for_fp8_oproj: true,
            mode: Mode::QkPv,
            pipe: true,
            wdec_active: false,
            sub_cta_active: false,
            rows: 16_384,
            hidden_size: 2560,
            diagnostics_active: false,
        };
        assert!(e4m3_out_eligible(base));
        assert!(e4m3_out_eligible(E4m3OutEligibility {
            rows: E4M3_OUT_MIN_ROWS,
            ..base
        }));
        assert!(!e4m3_out_eligible(E4m3OutEligibility {
            selector_enabled: false,
            ..base
        }));
        assert!(!e4m3_out_eligible(E4m3OutEligibility {
            armed_for_fp8_oproj: false,
            ..base
        }));
        assert!(!e4m3_out_eligible(E4m3OutEligibility {
            mode: Mode::Qk,
            ..base
        }));
        assert!(!e4m3_out_eligible(E4m3OutEligibility {
            pipe: false,
            ..base
        }));
        assert!(!e4m3_out_eligible(E4m3OutEligibility {
            wdec_active: true,
            ..base
        }));
        assert!(!e4m3_out_eligible(E4m3OutEligibility {
            sub_cta_active: true,
            ..base
        }));
        assert!(!e4m3_out_eligible(E4m3OutEligibility {
            rows: E4M3_OUT_MIN_ROWS - 1,
            ..base
        }));
        assert!(!e4m3_out_eligible(E4m3OutEligibility {
            hidden_size: 0,
            ..base
        }));
        assert!(!e4m3_out_eligible(E4m3OutEligibility {
            hidden_size: 2564,
            ..base
        }));
        assert!(!e4m3_out_eligible(E4m3OutEligibility {
            diagnostics_active: true,
            ..base
        }));
    }

    #[test]
    fn e4m3_out_arm_is_one_shot_and_scoped() {
        assert!(!take_e4m3_out_armed());
        {
            let _arm = E4m3OutArm::arm();
            assert!(take_e4m3_out_armed());
            assert!(!take_e4m3_out_armed());
            {
                let _nested_disarm = E4m3OutArm::disarm();
                assert!(!take_e4m3_out_armed());
            }
        }
        assert!(!take_e4m3_out_armed());
        {
            let _outer = E4m3OutArm::arm();
            {
                let _inner = E4m3OutArm::disarm();
                assert!(!take_e4m3_out_armed());
            }
            assert!(take_e4m3_out_armed());
        }
        assert!(!take_e4m3_out_armed());
    }

    #[test]
    fn invalidate_shadow_continuity_forces_refill_even_at_matching_offset() {
        let gpu = MockGpuBackend::new();
        let cache_a = qsa_cache(&gpu, 2, 8);
        let cache_b = qsa_cache(&gpu, 2, 8);
        reserve_shadows_with_admission(&gpu, &cache_a, &mut ample_shadow_admission).unwrap();
        reserve_shadows_with_admission(&gpu, &cache_b, &mut ample_shadow_admission).unwrap();

        // Simulate a prior request writing [0, 16) on cache_a and cache_b.
        assert_eq!(note_prepped(&cache_a, 0, 0, 16).unwrap(), 0);
        assert_eq!(note_prepped(&cache_a, 1, 0, 16).unwrap(), 0);
        assert_eq!(note_prepped(&cache_b, 0, 0, 16).unwrap(), 0);

        // Without invalidation, a continuation at start=16 skips refill (s.end == Some(16)).
        assert_eq!(note_prepped(&cache_a, 0, 16, 8).unwrap(), 0);

        // Invalidate continuity on cache_a (e.g. Marconi prefix hit resuming at 24 / 16):
        // note_prepped MUST force refill == start even though s.end matched start.
        invalidate_shadow_continuity(&cache_a).unwrap();
        assert_eq!(note_prepped(&cache_a, 0, 24, 8).unwrap(), 24);
        assert_eq!(note_prepped(&cache_a, 1, 16, 8).unwrap(), 16);

        // Unrelated cache_id must remain continuous.
        assert_eq!(note_prepped(&cache_b, 0, 16, 8).unwrap(), 0);
    }
}
