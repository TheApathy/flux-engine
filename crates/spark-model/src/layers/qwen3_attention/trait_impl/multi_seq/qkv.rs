// SPDX-License-Identifier: AGPL-3.0-only

//! Phase 2: multi-sequence Q/K/V projection and Q/K RMS-norm dispatch.
//!
//! Dispatch order in [`Qwen3AttentionLayer::ms_phase_qkv`]:
//! - Staged exact M17 (`ATLAS_ATTN_QKV_EXACT_M17_ASTAGE=1`) for eligible gated NVFP4
//! - SM120/SM103 fast NVFP4 path (`try_ms_qkv_sm120_fast`: coalesced BF16 MMA,
//!   cuBLASLt NVFP4, or transposed W4A16 TC) when enabled for `n` and `sm_count`
//! - Exact M4/M17/M32 (or wide chunked exact for `33..=QWEN4_NATIVE_MAX_ROWS`) / serial K1
//! - `n=3` / `n=2` NVFP4 batch GEMV paths
//! - Batched M16 (`ATLAS_TC_NVFP4_M16=1` + `ATLAS_TC_NVFP4_M16_MS_ATTN=1`) or
//!   batched plain (`ATLAS_ATTN_QKV_BATCHED=1`)
//! - Sequential per-token GEMV fallback (supports FP8, NVFP4, and BF16)

use anyhow::{Result, bail};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

use super::ctx::MultiSeqCtx;
use crate::layers::ops;
use crate::layers::qwen3_attention::Qwen3AttentionLayer;
use crate::weight_map::QuantizedWeight;

/// Cached `ATLAS_ATTN_QKV_BATCHED` env-var lookup. When `1` the n∈(3,32]
/// multi_seq QKV projection runs THREE plain `w4a16_gemm` calls at M=n
/// (non-transposed NVFP4 weights, M_TILE=64 single tile — the same kernel
/// family validated token-exact for the K=γ FFN at M=17) instead of the
/// per-token GEMV loop that re-reads q/k/v weights n times per layer
/// (~53ms/step of the K=17 DFlash verify at n=17 across 16 attn layers).
/// Output goes to scratch then scatters into the interleaved qkv_buf with
/// the same per-token `deinterleave_qg` fixup as the batched-M16 path.
///
/// Enabled by `ATLAS_ATTN_QKV_BATCHED=1` (the serve-script default) OR the
/// alias `ATLAS_ATTN_QKV_M16=1`. Both select the same batched path, which
/// internally prefers the M-efficient transposed `w4a16_gemm_t_m32_n64`
/// kernel (single B read, 272 CTAs) over the plain M_TILE=64 `w4a16_gemm`
/// when the transposed QKV weights are installed — see `use_m32` in
/// `ms_qkv_batched_plain`. The path is not lossless on gated Qwen3.8
/// attention: a locked K=5 oracle first diverged at the first attention
/// layer and caused deterministic 17-token coding stops. Gated layers
/// therefore require the explicit unsafe profiling opt-in below. Default off.
///
/// REACHABILITY: on gated ordinary-NVFP4 attention this branch is unreachable
/// for `n in 4..=32`: `ms_phase_qkv` first checks `try_ms_qkv_sm120_fast`
/// (engaging at `n >= 4` on `sm_count > 64`, or `n >= 16` on `<= 64` SMs when
/// explicitly enabled via `ATLAS_ATTN_PROJ_TC=1` / `ATLAS_SSM_PROJ_TC=1`) and
/// then `ops::exact_attention_qkv_route` (returning `Some(..)` for every
/// `4..=32` width, selecting its exact M4/M17/M32 tier or serial-K1 fallback).
/// On GB10 (`<= 64` SMs, default env), `4..=32` therefore belongs to
/// `ms_qkv_exact`; only `n > 32` (or non-NVFP4 encodings) can fall past it.
/// Pinned by `gated_nvfp4_never_falls_through_to_the_batched_route`.
fn attn_qkv_batched_plain_enabled() -> bool {
    static CACHE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHE.get_or_init(|| {
        let on = |k: &str| {
            std::env::var(k)
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                .unwrap_or(false)
        };
        on("ATLAS_ATTN_QKV_BATCHED") || on("ATLAS_ATTN_QKV_M16")
    })
}

/// Escape hatch for profiling the known-lossy batched QKV path on gated
/// attention. Never enable this in a correctness or serving profile.
fn attn_qkv_batched_gated_unsafe_enabled() -> bool {
    static CACHE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHE.get_or_init(|| {
        std::env::var("ATLAS_ATTN_QKV_BATCHED_GATED_UNSAFE")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
    })
}

/// Cached `ATLAS_ATTN_QKV_FUSED` env-var lookup. When `1`/`true` the
/// batch3 QKV projection writes directly into the interleaved
/// `qkv_buf` layout and uses a single batched RMS norm launch instead
/// of 6 separate ones. Default off for A/B safety.
fn attn_qkv_fused_enabled() -> bool {
    static CACHE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHE.get_or_init(|| {
        std::env::var("ATLAS_ATTN_QKV_FUSED")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
    })
}

/// Separate gate for the n>3 batched-M16 QKV path in multi_seq decode.
///
/// Distinct from `ATLAS_TC_NVFP4_M16` (which enables the same kernel for
/// SSM in_proj / SSM out_proj / prefill QKV — all verified). The multi_seq
/// attention QKV path produces numerically different output from the
/// per-token sequential GEMV path on gated Qwen3.6-27B layers when
/// `w4a16_gemm_n128_m16` is used with NVFP4-T weights + `deinterleave_qg`
/// fixup; verify accept rate collapses to 0%. Root cause not yet
/// identified — the dispatch wiring is correct (set_prefill_weights now
/// populates q_nvfp4_t / k_nvfp4_t / v_nvfp4_t for the qwen35_dense
/// loader; m16 + deinterleave_qg matches the FP8/dense gated decode
/// shape; SSM uses the same kernel at the same M=17 without issue).
/// Keeping the dispatch behind its own flag so production sets
/// `ATLAS_TC_NVFP4_M16=1` (for SSM/prefill gains) without enabling the
/// broken attention path. Set `ATLAS_TC_NVFP4_M16_MS_ATTN=1` to opt in.
pub(super) fn tc_nvfp4_m16_ms_attn_enabled() -> bool {
    static GATE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *GATE.get_or_init(|| std::env::var("ATLAS_TC_NVFP4_M16_MS_ATTN").ok().as_deref() == Some("1"))
}

/// Cached `ATLAS_ATTN_QKV_EXACT_STRIDED` env-var lookup. When `1` (or auto-enabled
/// on wide-SM GPUs with `sm_count > 64`) the K1-order exact M4/M17/M32 attention
/// QKV route writes Q/G and K/V directly into the per-token-strided `qkv_buf`
/// layout instead of staging in contiguous scratch and scattering with per-row
/// d2d copies. Bit-exact (same bytes to the same addresses; the exact kernels
/// already deinterleave QG at the store), it only removes the 3 copy launches
/// per row plus their bandwidth.
fn parse_attn_qkv_exact_strided(value: Option<&str>) -> std::result::Result<bool, &'static str> {
    match value {
        None | Some("0") => Ok(false),
        Some("1") => Ok(true),
        Some(_) => Err("ATLAS_ATTN_QKV_EXACT_STRIDED must be exactly 0 or 1"),
    }
}

const fn resolve_sm120_bool_opt(explicit: Option<bool>, sm_count: u32) -> bool {
    match explicit {
        Some(v) => v,
        None => sm_count > 64,
    }
}

fn attn_qkv_exact_strided_mode() -> Result<Option<bool>> {
    static CACHE: std::sync::OnceLock<std::result::Result<Option<bool>, &'static str>> =
        std::sync::OnceLock::new();
    (*CACHE.get_or_init(|| match std::env::var("ATLAS_ATTN_QKV_EXACT_STRIDED") {
        Ok(value) => parse_attn_qkv_exact_strided(Some(value.as_str())).map(Some),
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(std::env::VarError::NotUnicode(_)) => {
            Err("ATLAS_ATTN_QKV_EXACT_STRIDED must be valid UTF-8 and exactly 0 or 1")
        }
    }))
    .map_err(anyhow::Error::msg)
}

fn attn_qkv_exact_strided_requested() -> Result<bool> {
    Ok(resolve_sm120_bool_opt(
        attn_qkv_exact_strided_mode()?,
        atlas_core::device::active_sm_count(),
    ))
}

fn log_exact_strided_engagement_once(rows: usize) {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        tracing::info!("ENGAGED ATLAS_ATTN_QKV_EXACT_STRIDED: multi_seq M={rows}");
    });
}

/// Default-off M17 exact-QKV candidate that cooperatively stages each
/// 512-column BF16 activation tile once per CTA. The flag is intentionally
/// separate from the proven exact route: when requested for an eligible
/// n=5..=17 gated ordinary-NVFP4 phase, both staged QG and dual-KV symbols
/// must be present or the request fails before launching a projection.
fn parse_attn_qkv_exact_m17_astage(value: Option<&str>) -> std::result::Result<bool, &'static str> {
    match value {
        None | Some("0") => Ok(false),
        Some("1") => Ok(true),
        Some(_) => Err("ATLAS_ATTN_QKV_EXACT_M17_ASTAGE must be exactly 0 or 1"),
    }
}

fn attn_qkv_exact_m17_astage_enabled() -> Result<bool> {
    static CACHE: std::sync::OnceLock<std::result::Result<bool, &'static str>> =
        std::sync::OnceLock::new();
    (*CACHE.get_or_init(|| match std::env::var("ATLAS_ATTN_QKV_EXACT_M17_ASTAGE") {
        Ok(value) => parse_attn_qkv_exact_m17_astage(Some(value.as_str())),
        Err(std::env::VarError::NotPresent) => parse_attn_qkv_exact_m17_astage(None),
        Err(std::env::VarError::NotUnicode(_)) => {
            Err("ATLAS_ATTN_QKV_EXACT_M17_ASTAGE must be valid UTF-8 and exactly 0 or 1")
        }
    }))
    .map_err(anyhow::Error::msg)
}

fn log_m17_astage_engagement_once() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        tracing::info!("ENGAGED ATLAS_ATTN_QKV_EXACT_M17_ASTAGE: atomic staged QG/dual-KV pair");
    });
}

/// Cached `ATLAS_ATTN_QK_NORM_BATCHED` env-var lookup. When `1` the
/// exact M4/M17/M32 attention QKV routes replace their per-row `q_norm`/`k_norm`
/// pair with a single strided `rms_norm_qk_batch3` launch covering all `n`
/// tokens.
///
/// Bit-exact, not approximately: `rms_norm_qk_batch3` indexes the token via
/// `blockIdx.y` with no hardcoded count, and its per-head body is the
/// `rms_norm` body verbatim — same 2-wide BF16 unpack, same `warp_reduce_sum`
/// plus 32-entry `warp_sums` tree, same `(1 + weight)` scale, same
/// `blockDim.x`. `ms_qk_norm` launches `rms_norm` with `num_tokens = nq` and
/// `hidden_size = head_dim`, i.e. one block per head at `blockDim.x =
/// head_dim`; the batched kernel launches one block per (head, token, q|k) at
/// the same `blockDim.x`. Identical reduction shape ⇒ identical FP32
/// association ⇒ identical BF16 bytes.
///
/// What it buys: at n=17 the exact route issues `2 * n = 34` norm launches per
/// full-attention layer. Batching them leaves 1. Default off on `<= 64` SMs;
/// auto-enabled on wide-SM GPUs (`sm_count > 64`) via `resolve_sm120_bool_opt`.
fn parse_attn_qk_norm_batched(value: Option<&str>) -> std::result::Result<bool, &'static str> {
    match value {
        None | Some("0") => Ok(false),
        Some("1") => Ok(true),
        Some(_) => Err("ATLAS_ATTN_QK_NORM_BATCHED must be exactly 0 or 1"),
    }
}

fn attn_qk_norm_batched_mode() -> Result<Option<bool>> {
    static CACHE: std::sync::OnceLock<std::result::Result<Option<bool>, &'static str>> =
        std::sync::OnceLock::new();
    (*CACHE.get_or_init(|| match std::env::var("ATLAS_ATTN_QK_NORM_BATCHED") {
        Ok(value) => parse_attn_qk_norm_batched(Some(value.as_str())).map(Some),
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(std::env::VarError::NotUnicode(_)) => {
            Err("ATLAS_ATTN_QK_NORM_BATCHED must be valid UTF-8 and exactly 0 or 1")
        }
    }))
    .map_err(anyhow::Error::msg)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum AttnProjTcOverride {
    Disabled,
    Enabled,
    Auto,
}

pub(super) fn parse_attn_proj_tc_override(raw: Option<&str>) -> AttnProjTcOverride {
    match raw.map(str::trim) {
        Some("0") | Some("false") | Some("FALSE") | Some("off") | Some("OFF") => {
            AttnProjTcOverride::Disabled
        }
        Some("1") | Some("true") | Some("TRUE") | Some("on") | Some("ON") => {
            AttnProjTcOverride::Enabled
        }
        _ => AttnProjTcOverride::Auto,
    }
}

/// Shared multi-seq attention QKV/O-projection TC override. Checks
/// `ATLAS_ATTN_PROJ_TC` first, falling back to `ATLAS_SSM_PROJ_TC` for
/// backward compatibility with existing deployment profiles.
pub(super) fn raw_attn_proj_tc_override() -> AttnProjTcOverride {
    static RAW: std::sync::OnceLock<AttnProjTcOverride> = std::sync::OnceLock::new();
    *RAW.get_or_init(|| {
        let raw = std::env::var("ATLAS_ATTN_PROJ_TC")
            .ok()
            .or_else(|| std::env::var("ATLAS_SSM_PROJ_TC").ok());
        parse_attn_proj_tc_override(raw.as_deref())
    })
}

pub(super) const fn resolve_attn_proj_tc_for_sm_with_policy(
    env_override: AttnProjTcOverride,
    sm_count: u32,
    rows: usize,
    det_batch: crate::layers::deterministic_batch::DeterministicBatchPolicy,
) -> bool {
    if det_batch.is_enabled() {
        return false;
    }
    resolve_attn_proj_tc_for_sm(env_override, sm_count, rows)
}

pub(super) const fn resolve_attn_proj_tc_for_sm(
    env_override: AttnProjTcOverride,
    sm_count: u32,
    rows: usize,
) -> bool {
    let min_rows = if sm_count > 64 { 4 } else { 16 };
    match env_override {
        AttnProjTcOverride::Disabled => false,
        AttnProjTcOverride::Enabled => rows >= min_rows,
        AttnProjTcOverride::Auto => sm_count > 64 && rows >= 4,
    }
}

fn cached_scatter_qkv_multi_seq_kernel(gpu: &dyn GpuBackend) -> KernelHandle {
    static KERNELS: std::sync::OnceLock<
        std::sync::Mutex<std::collections::HashMap<usize, KernelHandle>>,
    > = std::sync::OnceLock::new();
    let map = KERNELS.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()));
    let key = (gpu as *const dyn GpuBackend).cast::<()>() as usize;
    let Ok(mut guard) = map.lock() else {
        return crate::layers::try_kernel(gpu, "strided_copy_rows", "scatter_qkv_multi_seq");
    };
    *guard.entry(key).or_insert_with(|| {
        crate::layers::try_kernel(gpu, "strided_copy_rows", "scatter_qkv_multi_seq")
    })
}

fn log_qk_norm_batched_engagement_once(rows: usize) {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        tracing::info!("ENGAGED ATLAS_ATTN_QK_NORM_BATCHED: multi_seq M={rows}");
    });
}

/// True when every <=32-row chunk of `ms_qkv_exact` for `n > 32` has both its
/// exact QG and dual-KV kernel handles loaded.
const fn exact_attention_wide_qkv_ready(
    n: usize,
    kernels: ops::W4a16ExactAttentionKernels,
) -> bool {
    let mut row_off = 0usize;
    while row_off < n {
        let rem = n - row_off;
        let chunk = if rem == 33 {
            17
        } else if rem < 32 {
            rem
        } else {
            32
        };
        if kernels.qg_for_rows(chunk).0 == 0 || kernels.dual_kv_for_rows(chunk).0 == 0 {
            return false;
        }
        row_off += chunk;
    }
    n > 32
}

impl Qwen3AttentionLayer {
    /// Validate the complete exact-QKV route without launching or mutating any
    /// device buffer.  The native Qwen4 orchestrator calls this before its HC
    /// preparation because that preparation overwrites row state.
    pub(super) fn validate_qwen4_native_exact_qkv(&self, rows: usize) -> Result<()> {
        anyhow::ensure!(
            (2..=crate::layers::QWEN4_NATIVE_MAX_ROWS).contains(&rows),
            "native Qwen4 attention QKV admits only C2-C{}, got C{}",
            crate::layers::QWEN4_NATIVE_MAX_ROWS,
            rows
        );
        anyhow::ensure!(
            self.gated,
            "native Qwen4 attention requires gated Q projection"
        );
        anyhow::ensure!(
            self.q_weight.as_ref().and_then(|w| w.as_nvfp4()).is_some()
                && self.k_weight.as_ref().and_then(|w| w.as_nvfp4()).is_some()
                && self.v_weight.as_ref().and_then(|w| w.as_nvfp4()).is_some(),
            "native Qwen4 attention requires ordinary NVFP4 Q/K/V weights"
        );
        let mut check_rem = rows;
        while check_rem > 0 {
            let chunk = if check_rem == 33 {
                17
            } else {
                check_rem.min(32)
            };
            anyhow::ensure!(
                self.w4a16_exact_qkv_kernels.qg_for_rows(chunk).0 != 0
                    && self.w4a16_exact_qkv_kernels.dual_kv_for_rows(chunk).0 != 0,
                "native Qwen4 attention requires the complete selected exact QG/dual-KV tier"
            );
            check_rem -= chunk;
        }
        let _ = attn_qkv_exact_strided_requested()?;
        if attn_qk_norm_batched_mode()? == Some(true) {
            anyhow::ensure!(
                self.rms_norm_qk_batch3_k.0 != 0
                    && !self.attn.q_norm.weight.is_null()
                    && !self.attn.k_norm.weight.is_null(),
                "ATLAS_ATTN_QK_NORM_BATCHED=1 requires rms_norm_qk_batch3 and both Q/K norm weights for native Qwen4 C{rows}"
            );
        }
        Ok(())
    }

    /// Exact Qwen4 target-decode projection for native-concurrency widths
    /// (`2..=QWEN4_NATIVE_MAX_ROWS`). This deliberately bypasses the generic
    /// C2/C3 batched routes: those kernel families have a different reduction
    /// order, while the M4/M17/M32 pairs below preserve the proven K1 lane
    /// ownership and BF16 stores for every C2-C256 width.
    pub(super) fn ms_phase_qkv_qwen4_native_exact(&self, c: &MultiSeqCtx<'_>) -> Result<()> {
        self.validate_qwen4_native_exact_qkv(c.n)?;
        self.ms_qkv_exact(c, false)
    }

    pub(super) fn ms_phase_qkv(&self, c: &MultiSeqCtx<'_>) -> Result<()> {
        let n = c.n;

        let ordinary_nvfp4 = self.q_weight.as_ref().and_then(|w| w.as_nvfp4()).is_some()
            && self.k_weight.as_ref().and_then(|w| w.as_nvfp4()).is_some()
            && self.v_weight.as_ref().and_then(|w| w.as_nvfp4()).is_some();
        match ops::exact_attention_m17_astage_route(
            attn_qkv_exact_m17_astage_enabled()?,
            n,
            self.gated,
            ordinary_nvfp4,
            self.w4a16_exact_qkv_m17_astage_kernels,
        ) {
            ops::ExactAttentionM17AStageRoute::Complete => {
                self.ms_qkv_exact(c, true)?;
                log_m17_astage_engagement_once();
                return Ok(());
            }
            ops::ExactAttentionM17AStageRoute::Missing => {
                bail!(
                    "ATLAS_ATTN_QKV_EXACT_M17_ASTAGE requested but the staged QG/dual-KV kernel pair is incomplete"
                );
            }
            ops::ExactAttentionM17AStageRoute::Disabled
            | ops::ExactAttentionM17AStageRoute::Ineligible => {}
        }
        let exact_route = ops::exact_attention_qkv_route(
            n,
            self.gated,
            ordinary_nvfp4,
            self.w4a16_exact_qkv_kernels,
        );

        // On SM120/SM103 (`sm_count > 64`) at `n >= 4` (or on `<= 64` SMs at
        // `n >= 16` with explicit `ATLAS_ATTN_PROJ_TC=1` / `ATLAS_SSM_PROJ_TC=1`),
        // bypass scalar `w4a16_gemv_qg_exact_*` / `dual_kv_exact_*` and route
        // Q/K/V through coalesced BF16 MMA (`n <= sm120_coalesced_max_m()`),
        // native block-scaled cuBLASLt NVFP4 (`n >= CUBLASLT_NVFP4_MIN_M` when
        // above coalesced max M), or transposed TC `w4a16_gemm_t_m32_n64[_splitk]`.
        // GB10 (`sm_count <= 64`) returns `Ok(false)` when unset.
        if ordinary_nvfp4 && self.try_ms_qkv_sm120_fast(c)? {
            return Ok(());
        }

        let det_batch =
            crate::layers::deterministic_batch::deterministic_batch_policy()?.is_enabled();
        if matches!(
            exact_route,
            Some(
                ops::ExactAttentionQkvRoute::ExactM4
                    | ops::ExactAttentionQkvRoute::ExactM17
                    | ops::ExactAttentionQkvRoute::ExactM32
            )
        ) {
            self.ms_qkv_exact(c, false)?;
        } else if matches!(
            exact_route,
            Some(
                ops::ExactAttentionQkvRoute::SerialK1M4
                    | ops::ExactAttentionQkvRoute::SerialK1M17
                    | ops::ExactAttentionQkvRoute::SerialK1M32
            )
        ) {
            self.ms_qkv_serial(c)?;
        } else if !det_batch
            && n == 3
            && self.q_weight.as_ref().and_then(|w| w.as_nvfp4()).is_some()
            && self.k_weight.as_ref().and_then(|w| w.as_nvfp4()).is_some()
            && self.v_weight.as_ref().and_then(|w| w.as_nvfp4()).is_some()
        {
            self.ms_qkv_batch3(c)?;
        } else if !det_batch
            && n == 2
            && self.q_weight.as_ref().and_then(|w| w.as_nvfp4()).is_some()
            && self.k_weight.as_ref().and_then(|w| w.as_nvfp4()).is_some()
            && self.v_weight.as_ref().and_then(|w| w.as_nvfp4()).is_some()
        {
            self.ms_qkv_batch2(c)?;
        } else if !det_batch
            && n > 3
            && n <= 32
            && self.w4a16_gemm_t_m16_k.0 != 0
            && crate::layers::tc_nvfp4_m16_enabled()
            && tc_nvfp4_m16_ms_attn_enabled()
            && self.q_nvfp4_t.is_some()
            && self.k_nvfp4_t.is_some()
            && self.v_nvfp4_t.is_some()
        {
            self.ms_qkv_batched_m16(c)?;
        } else if !det_batch
            && n > 3
            && n <= 32
            && attn_qkv_batched_plain_enabled()
            && (!self.gated || attn_qkv_batched_gated_unsafe_enabled())
            && self.q_weight.as_ref().and_then(|w| w.as_nvfp4()).is_some()
            && self.k_weight.as_ref().and_then(|w| w.as_nvfp4()).is_some()
            && self.v_weight.as_ref().and_then(|w| w.as_nvfp4()).is_some()
        {
            self.ms_qkv_batched_plain(c)?;
        } else if n > 32
            && n <= crate::layers::QWEN4_NATIVE_MAX_ROWS
            && self.gated
            && ordinary_nvfp4
            && exact_attention_wide_qkv_ready(n, self.w4a16_exact_qkv_kernels)
        {
            // Wide gated multi-seq decode (e.g. C=48/64 on Qwen3.8-27B):
            // chunk into <=32-row K1-exact QG + dual-KV launches (2 sweeps at
            // C=64 instead of 64 serial GEMVs).
            self.ms_qkv_exact(c, false)?;
        } else {
            self.ms_qkv_serial(c)?;
        }
        Ok(())
    }

    /// SM120/SM103 fast Q/K/V multi-seq projection (`n >= 4` on `sm_count > 64`,
    /// or `n >= 16` on `<= 64` SMs when explicitly enabled).
    ///
    /// - When `n >= CUBLASLT_NVFP4_MIN_M` and `n > sm120_coalesced_max_m()`,
    ///   prefers native block-scaled cuBLASLt NVFP4 writing directly into
    ///   strided `qkv_buf` (`ldc = stride_bf16`, zero D2D scatter) + single
    ///   batched `deinterleave_qg` (if gated) + batched QK norm.
    /// - Otherwise uses SM120 coalesced BF16 MMA (`n <= sm120_coalesced_max_m()`)
    ///   or transposed TC `w4a16_gemm_t_m32_n64[_splitk]` + single-launch
    ///   `scatter_qkv_multi_seq` (with fused Q/Gate deinterleave) + batched QK norm.
    fn try_ms_qkv_sm120_fast(&self, c: &MultiSeqCtx<'_>) -> Result<bool> {
        let MultiSeqCtx {
            fwd,
            n,
            stream,
            h,
            nq,
            nkv,
            hd,
            eps,
            bf16,
            q_proj_dim,
            q_proj_bytes,
            per_seq_qkv,
            normed,
            qkv_buf,
            ..
        } = *c;
        let sm_count = atlas_core::device::active_sm_count();
        let tc_override = raw_attn_proj_tc_override();
        if !resolve_attn_proj_tc_for_sm_with_policy(
            tc_override,
            sm_count,
            n,
            crate::layers::deterministic_batch::deterministic_batch_policy_or_disabled(),
        ) {
            return Ok(false);
        }
        if !per_seq_qkv.is_multiple_of(bf16) {
            return Ok(false);
        }
        let kv_dim = nkv * hd;
        let kv_bytes = kv_dim as usize * bf16;
        let stride_bf16 = (per_seq_qkv / bf16) as u32;
        let Some(q_weight) = self.q_weight.as_ref().and_then(|w| w.as_nvfp4()) else {
            return Ok(false);
        };
        let Some(k_weight) = self.k_weight.as_ref().and_then(|w| w.as_nvfp4()) else {
            return Ok(false);
        };
        let Some(v_weight) = self.v_weight.as_ref().and_then(|w| w.as_nvfp4()) else {
            return Ok(false);
        };

        // Path 1: M >= 24 native cuBLASLt NVFP4 direct-strided write into `qkv_buf`
        // (used when `n > sm120_coalesced_max_m()` or coalesced wide-M is disabled).
        let sm120_max_m = crate::layers::dense_ffn::sm120_coalesced_max_m();
        if n >= crate::layers::dense_ffn::CUBLASLT_NVFP4_MIN_M as usize
            && n > sm120_max_m
            && per_seq_qkv.is_multiple_of(16)
            && q_proj_bytes.is_multiple_of(16)
            && kv_bytes.is_multiple_of(16)
            && (!self.gated || self.deinterleave_qg_k.0 != 0)
        {
            let k_out = qkv_buf.offset(q_proj_bytes);
            let v_out = k_out.offset(kv_bytes);
            if ops::try_cublaslt_nvfp4_multiseq_qkv(
                fwd.gpu,
                normed,
                q_weight,
                k_weight,
                v_weight,
                qkv_buf,
                k_out,
                v_out,
                n as u32,
                q_proj_dim,
                kv_dim,
                h as u32,
                stride_bf16,
                fwd.graph_capture,
                stream,
            )? {
                if self.gated {
                    ops::deinterleave_qg(
                        fwd.gpu,
                        self.deinterleave_qg_k,
                        qkv_buf,
                        n as u32,
                        nq,
                        hd,
                        stride_bf16,
                        stream,
                    )?;
                }
                if self.qk_norm_batched_available()? {
                    self.ms_qk_norm_rows(c)?;
                    log_qk_norm_batched_engagement_once(n);
                } else {
                    for row in 0..n {
                        let q_row = qkv_buf.offset(row * per_seq_qkv);
                        let k_row = q_row.offset(q_proj_bytes);
                        self.ms_qk_norm(fwd, q_row, k_row, nq, nkv, hd, eps, stream)?;
                    }
                }
                return Ok(true);
            }
        }

        // Path 2: M <= sm120_max_m (up to 64) BF16xBF16 MMA coalesced / warp-split-K
        // on SM120 (`q_proj`: v2_n16_sk4_m{16,32,64}; `k_proj`/`v_proj`: coalesced
        // fast-proj, zero partial workspace, zero activation quantization), or
        // M >= 4 transposed W4A16 TC (`w4a16_gemm_t_m32_n64[_splitk]`) +
        // single-launch scatter/deinterleave into `qkv_buf`.
        let q_scratch = fwd.buffers.ssm_qkvz();
        let k_scratch = fwd.buffers.attn_output();
        let v_scratch = k_scratch.offset(n * kv_bytes);

        let coalesced_qkv_done = if n <= sm120_max_m
            && crate::layers::dense_ffn::try_coalesced_nvfp4_sm120_small_m(
                fwd.gpu,
                normed,
                q_weight,
                q_scratch,
                n as u32,
                q_proj_dim,
                h as u32,
                sm120_max_m,
                stream,
            )? {
            let k_ok = crate::layers::dense_ffn::try_coalesced_nvfp4_sm120_small_m(
                fwd.gpu,
                normed,
                k_weight,
                k_scratch,
                n as u32,
                kv_dim,
                h as u32,
                sm120_max_m,
                stream,
            )?;
            k_ok && crate::layers::dense_ffn::try_coalesced_nvfp4_sm120_small_m(
                fwd.gpu,
                normed,
                v_weight,
                v_scratch,
                n as u32,
                kv_dim,
                h as u32,
                sm120_max_m,
                stream,
            )?
        } else {
            false
        };

        if !coalesced_qkv_done {
            let (Some(q_t), Some(k_t), Some(v_t)) = (
                self.q_nvfp4_t.as_ref(),
                self.k_nvfp4_t.as_ref(),
                self.v_nvfp4_t.as_ref(),
            ) else {
                return Ok(false);
            };
            if self.w4a16_gemm_t_m32_n64_k.0 == 0 {
                return Ok(false);
            }

            ops::w4a16_gemm_n64_m32(
                fwd.gpu,
                self.w4a16_gemm_t_m32_n64_k,
                normed,
                q_t,
                q_scratch,
                n as u32,
                q_proj_dim,
                h as u32,
                stream,
            )?;
            let qkv_splits = crate::layers::attn_qkv_splitk();
            let ws = if n <= 32 && qkv_splits > 0 {
                *self
                    .qkv_splitk_workspace
                    .lock()
                    .map_err(|_| anyhow::anyhow!("qkv_splitk_workspace lock poisoned"))?
            } else {
                None
            };
            if let Some(ws_ptr) = ws
                && n <= 32
                && qkv_splits > 0
                && self.w4a16_gemm_t_m32_n64_splitk_k.0 != 0
                && self.reduce_splitk_k.0 != 0
            {
                ops::w4a16_gemm_n64_m32_splitk(
                    fwd.gpu,
                    self.w4a16_gemm_t_m32_n64_splitk_k,
                    self.reduce_splitk_k,
                    normed,
                    k_t,
                    k_scratch,
                    ws_ptr,
                    n as u32,
                    kv_dim,
                    h as u32,
                    kv_dim,
                    qkv_splits,
                    stream,
                )?;
                ops::w4a16_gemm_n64_m32_splitk(
                    fwd.gpu,
                    self.w4a16_gemm_t_m32_n64_splitk_k,
                    self.reduce_splitk_k,
                    normed,
                    v_t,
                    v_scratch,
                    ws_ptr,
                    n as u32,
                    kv_dim,
                    h as u32,
                    kv_dim,
                    qkv_splits,
                    stream,
                )?;
            } else {
                ops::w4a16_gemm_n64_m32(
                    fwd.gpu,
                    self.w4a16_gemm_t_m32_n64_k,
                    normed,
                    k_t,
                    k_scratch,
                    n as u32,
                    kv_dim,
                    h as u32,
                    stream,
                )?;
                ops::w4a16_gemm_n64_m32(
                    fwd.gpu,
                    self.w4a16_gemm_t_m32_n64_k,
                    normed,
                    v_t,
                    v_scratch,
                    n as u32,
                    kv_dim,
                    h as u32,
                    stream,
                )?;
            }
        }

        let scatter_k = cached_scatter_qkv_multi_seq_kernel(fwd.gpu);
        if scatter_k.0 != 0 {
            ops::scatter_qkv_multi_seq(
                fwd.gpu,
                scatter_k,
                q_scratch,
                k_scratch,
                v_scratch,
                qkv_buf,
                n as u32,
                q_proj_dim,
                kv_dim,
                stride_bf16,
                nq,
                hd,
                self.gated,
                stream,
            )?;
        } else {
            for row in 0..n {
                let q_out = qkv_buf.offset(row * per_seq_qkv);
                let k_out = q_out.offset(q_proj_bytes);
                let v_out = k_out.offset(kv_bytes);
                fwd.gpu.copy_d2d_async(
                    q_scratch.offset(row * q_proj_bytes),
                    q_out,
                    q_proj_bytes,
                    stream,
                )?;
                fwd.gpu.copy_d2d_async(
                    k_scratch.offset(row * kv_bytes),
                    k_out,
                    kv_bytes,
                    stream,
                )?;
                fwd.gpu.copy_d2d_async(
                    v_scratch.offset(row * kv_bytes),
                    v_out,
                    kv_bytes,
                    stream,
                )?;
            }
            if self.gated {
                ops::deinterleave_qg(
                    fwd.gpu,
                    self.deinterleave_qg_k,
                    qkv_buf,
                    n as u32,
                    nq,
                    hd,
                    stride_bf16,
                    stream,
                )?;
            }
        }

        if self.qk_norm_batched_available()? {
            self.ms_qk_norm_rows(c)?;
            log_qk_norm_batched_engagement_once(n);
        } else {
            for row in 0..n {
                let q_row = qkv_buf.offset(row * per_seq_qkv);
                let k_row = q_row.offset(q_proj_bytes);
                self.ms_qk_norm(fwd, q_row, k_row, nq, nkv, hd, eps, stream)?;
            }
        }
        Ok(true)
    }

    /// K1-order exact M4/M17/M32 route for gated ordinary-NVFP4 attention. QG is
    /// deinterleaved at the exact BF16 output store; dual K/V remains row-major.
    fn ms_qkv_exact(&self, c: &MultiSeqCtx<'_>, m17_astage: bool) -> Result<()> {
        if attn_qkv_exact_strided_requested()? {
            return self.ms_qkv_exact_strided(c, m17_astage);
        }

        let MultiSeqCtx {
            fwd,
            n,
            stream,
            h,
            nq,
            nkv,
            hd,
            eps,
            bf16,
            q_proj_dim,
            q_proj_bytes,
            per_seq_qkv,
            normed,
            qkv_buf,
            ..
        } = *c;
        let q_weight = self.q_weight.as_ref().and_then(|w| w.as_nvfp4()).unwrap();
        let k_weight = self.k_weight.as_ref().and_then(|w| w.as_nvfp4()).unwrap();
        let v_weight = self.v_weight.as_ref().and_then(|w| w.as_nvfp4()).unwrap();
        let batched_norm = self.qk_norm_batched_available()?;
        let kv_dim = nkv * hd;
        let kv_bytes = kv_dim as usize * bf16;
        let q_scratch = fwd.buffers.ssm_qkvz();
        let k_scratch = fwd.buffers.attn_output();
        let v_scratch = k_scratch.offset(n * kv_bytes);

        let mut row_off = 0usize;
        while row_off < n {
            let rem = n - row_off;
            let chunk = if m17_astage {
                rem
            } else if rem == 33 {
                17
            } else {
                rem.min(32)
            };
            let in_ptr = normed.offset(row_off * h * bf16);
            ops::w4a16_gemv_qg_exact(
                fwd.gpu,
                if m17_astage {
                    self.w4a16_exact_qkv_m17_astage_kernels.qg()
                } else {
                    self.w4a16_exact_qkv_kernels.qg_for_rows(chunk)
                },
                in_ptr,
                q_weight,
                q_scratch.offset(row_off * q_proj_bytes),
                chunk as u32,
                q_proj_dim,
                h as u32,
                nq,
                hd,
                q_proj_dim,
                stream,
            )?;
            ops::w4a16_gemv_dual_kv_exact(
                fwd.gpu,
                if m17_astage {
                    self.w4a16_exact_qkv_m17_astage_kernels.dual_kv()
                } else {
                    self.w4a16_exact_qkv_kernels.dual_kv_for_rows(chunk)
                },
                in_ptr,
                k_weight,
                k_scratch.offset(row_off * kv_bytes),
                v_weight,
                v_scratch.offset(row_off * kv_bytes),
                chunk as u32,
                kv_dim,
                h as u32,
                kv_dim,
                stream,
            )?;
            row_off += chunk;
        }

        for row in 0..n {
            let q_out = qkv_buf.offset(row * per_seq_qkv);
            let k_out = q_out.offset(q_proj_bytes);
            let v_out = k_out.offset(kv_bytes);
            fwd.gpu.copy_d2d_async(
                q_scratch.offset(row * q_proj_bytes),
                q_out,
                q_proj_bytes,
                stream,
            )?;
            fwd.gpu
                .copy_d2d_async(k_scratch.offset(row * kv_bytes), k_out, kv_bytes, stream)?;
            fwd.gpu
                .copy_d2d_async(v_scratch.offset(row * kv_bytes), v_out, kv_bytes, stream)?;
            if !batched_norm {
                self.ms_qk_norm(fwd, q_out, k_out, nq, nkv, hd, eps, stream)?;
            }
        }
        // The batched norm must follow the scatter: it reads and rewrites the
        // Q/K slots the copies above fill.
        if batched_norm {
            self.ms_qk_norm_rows(c)?;
            log_qk_norm_batched_engagement_once(n);
        }
        Ok(())
    }

    /// Strided variant of [`Self::ms_qkv_exact`]: the exact kernels write Q/G
    /// and K/V directly into the per-token-strided `qkv_buf`, removing the
    /// contiguous-scratch + per-row d2d scatter. Arithmetic and byte placement
    /// are identical to the contiguous path (the exact kernels already
    /// deinterleave QG at the BF16 output store), so this is bit-exact.
    fn ms_qkv_exact_strided(&self, c: &MultiSeqCtx<'_>, m17_astage: bool) -> Result<()> {
        let MultiSeqCtx {
            fwd,
            n,
            stream,
            h,
            nq,
            nkv,
            hd,
            eps,
            bf16,
            q_proj_dim,
            q_proj_bytes,
            per_seq_qkv,
            normed,
            qkv_buf,
            ..
        } = *c;
        let q_weight = self.q_weight.as_ref().and_then(|w| w.as_nvfp4()).unwrap();
        let k_weight = self.k_weight.as_ref().and_then(|w| w.as_nvfp4()).unwrap();
        let v_weight = self.v_weight.as_ref().and_then(|w| w.as_nvfp4()).unwrap();
        let batched_norm = self.qk_norm_batched_available()?;
        let kv_dim = nkv * hd;
        let kv_bytes = kv_dim as usize * bf16;
        debug_assert!(per_seq_qkv % bf16 == 0, "qkv stride must be BF16-aligned");
        let stride_bf16 = (per_seq_qkv / bf16) as u32;

        let mut row_off = 0usize;
        while row_off < n {
            let rem = n - row_off;
            let chunk = if m17_astage {
                rem
            } else if rem == 33 {
                17
            } else {
                rem.min(32)
            };
            let in_ptr = normed.offset(row_off * h * bf16);
            let out_base = qkv_buf.offset(row_off * per_seq_qkv);
            // Q writes to qkv_buf[row].Q (offset 0 of each token slot), with Gate
            // deinterleaved at the store. Out row stride is the full token stride.
            ops::w4a16_gemv_qg_exact(
                fwd.gpu,
                if m17_astage {
                    self.w4a16_exact_qkv_m17_astage_kernels.qg()
                } else {
                    self.w4a16_exact_qkv_kernels.qg_for_rows(chunk)
                },
                in_ptr,
                q_weight,
                out_base,
                chunk as u32,
                q_proj_dim,
                h as u32,
                nq,
                hd,
                stride_bf16,
                stream,
            )?;

            // K writes to qkv_buf[row].K, V to qkv_buf[row].V; the base pointers
            // are pre-offset, the row stride is the full token stride.
            let k_base = out_base.offset(q_proj_bytes);
            let v_base = k_base.offset(kv_bytes);
            ops::w4a16_gemv_dual_kv_exact(
                fwd.gpu,
                if m17_astage {
                    self.w4a16_exact_qkv_m17_astage_kernels.dual_kv()
                } else {
                    self.w4a16_exact_qkv_kernels.dual_kv_for_rows(chunk)
                },
                in_ptr,
                k_weight,
                k_base,
                v_weight,
                v_base,
                chunk as u32,
                kv_dim,
                h as u32,
                stride_bf16,
                stream,
            )?;
            row_off += chunk;
        }
        log_exact_strided_engagement_once(n);

        if batched_norm {
            self.ms_qk_norm_rows(c)?;
            log_qk_norm_batched_engagement_once(n);
            return Ok(());
        }
        for row in 0..n {
            let q_out = qkv_buf.offset(row * per_seq_qkv);
            let k_out = q_out.offset(q_proj_bytes);
            self.ms_qk_norm(fwd, q_out, k_out, nq, nkv, hd, eps, stream)?;
        }
        Ok(())
    }

    /// Complete ordinary K1 fallback. Keeping QG and dual-KV together avoids a
    /// partially exact phase when one selected M17 symbol is unavailable.
    fn ms_qkv_serial(&self, c: &MultiSeqCtx<'_>) -> Result<()> {
        let MultiSeqCtx {
            fwd,
            n,
            stream,
            h,
            nq,
            nkv,
            hd,
            eps,
            bf16,
            q_dim,
            q_proj_dim,
            q_proj_bytes,
            per_seq_qkv,
            normed,
            qkv_buf,
            ..
        } = *c;
        for row in 0..n {
            let normed_row = normed.offset(row * h * bf16);
            let q_out = qkv_buf.offset(row * per_seq_qkv);
            let k_out = q_out.offset(q_proj_bytes);
            let v_out = k_out.offset((nkv * hd) as usize * bf16);

            self.ms_qkv_seq_q(fwd, normed_row, q_out, q_proj_dim, q_dim, nq, hd, h, stream)?;
            self.ms_qkv_seq_kv(fwd, normed_row, k_out, v_out, nkv, hd, h, stream)?;
            self.ms_qk_norm(fwd, q_out, k_out, nq, nkv, hd, eps, stream)?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn ms_qk_norm(
        &self,
        fwd: &crate::layer::ForwardContext<'_>,
        q_out: DevicePtr,
        k_out: DevicePtr,
        nq: u32,
        nkv: u32,
        hd: u32,
        eps: f32,
        stream: u64,
    ) -> Result<()> {
        if !self.attn.q_norm.weight.is_null() {
            ops::rms_norm(
                fwd.gpu,
                self.rms_norm_k,
                q_out,
                &self.attn.q_norm,
                q_out,
                nq,
                hd,
                eps,
                stream,
            )?;
        }
        if !self.attn.k_norm.weight.is_null() {
            ops::rms_norm(
                fwd.gpu,
                self.rms_norm_k,
                k_out,
                &self.attn.k_norm,
                k_out,
                nkv,
                hd,
                eps,
                stream,
            )?;
        }
        Ok(())
    }

    /// Per-head `q_norm` + `k_norm` over all `n` tokens of `qkv_buf`.
    ///
    /// Callers gate on [`Self::qk_norm_batched_available`] and keep their
    /// per-row `ms_qk_norm` loop when it is false. Both norm weights must be
    /// present: the batched kernel does q and k in one grid and has no way to
    /// skip one half.
    fn qk_norm_batched_available(&self) -> Result<bool> {
        let mode = attn_qk_norm_batched_mode()?;
        let enabled = resolve_sm120_bool_opt(mode, atlas_core::device::active_sm_count());
        if !enabled {
            return Ok(false);
        }
        let ready = self.rms_norm_qk_batch3_k.0 != 0
            && !self.attn.q_norm.weight.is_null()
            && !self.attn.k_norm.weight.is_null();
        if mode == Some(true) {
            anyhow::ensure!(
                ready,
                "ATLAS_ATTN_QK_NORM_BATCHED=1 requires rms_norm_qk_batch3 and both Q/K norm weights"
            );
        }
        Ok(ready)
    }

    fn ms_qk_norm_rows(&self, c: &MultiSeqCtx<'_>) -> Result<()> {
        let MultiSeqCtx {
            fwd,
            n,
            stream,
            nq,
            nkv,
            hd,
            eps,
            bf16,
            q_proj_dim,
            per_seq_qkv,
            qkv_buf,
            ..
        } = *c;
        debug_assert!(
            self.rms_norm_qk_batch3_k.0 != 0
                && !self.attn.q_norm.weight.is_null()
                && !self.attn.k_norm.weight.is_null()
        );
        debug_assert!(per_seq_qkv % bf16 == 0, "qkv stride must be BF16-aligned");
        ops::rms_norm_qk_batched(
            fwd.gpu,
            self.rms_norm_qk_batch3_k,
            qkv_buf,
            &self.attn.q_norm,
            &self.attn.k_norm,
            (per_seq_qkv / bf16) as u32,
            nq * hd,
            nkv * hd,
            q_proj_dim,
            hd,
            n as u32,
            eps,
            stream,
        )
    }

    /// n=3 NVFP4 batched path.
    fn ms_qkv_batch3(&self, c: &MultiSeqCtx<'_>) -> Result<()> {
        // Fused path: GEMVs write directly into the interleaved qkv_buf
        // layout and q_norm/k_norm fuse into a single 6-block kernel.
        if attn_qkv_fused_enabled()
            && self.gated
            && self.w4a16_gemv_qg_batch3_strided_k.0 != 0
            && self.w4a16_gemv_dual_batch3_strided_k.0 != 0
            && self.rms_norm_qk_batch3_k.0 != 0
            && !self.attn.q_norm.weight.is_null()
            && !self.attn.k_norm.weight.is_null()
        {
            return self.ms_qkv_batch3_fused(c);
        }

        let MultiSeqCtx {
            fwd,
            stream,
            h,
            nq,
            nkv,
            hd,
            eps,
            bf16,
            q_proj_dim,
            q_proj_bytes,
            per_seq_qkv,
            normed,
            qkv_buf,
            ..
        } = *c;
        let q_nvfp4 = self.q_weight.as_ref().and_then(|w| w.as_nvfp4()).unwrap();
        let k_nvfp4 = self.k_weight.as_ref().and_then(|w| w.as_nvfp4()).unwrap();
        let v_nvfp4 = self.v_weight.as_ref().and_then(|w| w.as_nvfp4()).unwrap();

        let q_scratch = fwd.buffers.ssm_qkvz();
        if self.gated {
            ops::w4a16_gemv_qg_batch3(
                fwd.gpu,
                self.w4a16_gemv_qg_batch3_k,
                normed,
                q_nvfp4,
                q_scratch,
                q_proj_dim,
                h as u32,
                nq,
                hd,
                stream,
            )?;
        } else {
            ops::w4a16_gemv_batch3(
                fwd.gpu,
                self.w4a16_gemv_batch3_k,
                normed,
                q_nvfp4,
                q_scratch,
                q_proj_dim,
                h as u32,
                stream,
            )?;
        }

        let kv_dim = nkv * hd;
        let kv_bytes = kv_dim as usize * bf16;
        let k_scratch = fwd.buffers.attn_output();
        let v_scratch = k_scratch.offset(3 * kv_bytes);
        ops::w4a16_gemv_dual_batch3(
            fwd.gpu,
            self.w4a16_gemv_dual_batch3_k,
            normed,
            k_nvfp4,
            k_scratch,
            v_nvfp4,
            v_scratch,
            kv_dim,
            h as u32,
            stream,
        )?;

        for i in 0..3usize {
            let q_out_i = qkv_buf.offset(i * per_seq_qkv);
            let k_out_i = q_out_i.offset(q_proj_bytes);
            let v_out_i = k_out_i.offset(kv_bytes);
            fwd.gpu.copy_d2d_async(
                q_scratch.offset(i * q_proj_bytes),
                q_out_i,
                q_proj_bytes,
                stream,
            )?;
            fwd.gpu
                .copy_d2d_async(k_scratch.offset(i * kv_bytes), k_out_i, kv_bytes, stream)?;
            fwd.gpu
                .copy_d2d_async(v_scratch.offset(i * kv_bytes), v_out_i, kv_bytes, stream)?;
        }

        for i in 0..3usize {
            let q_out_i = qkv_buf.offset(i * per_seq_qkv);
            let k_out_i = q_out_i.offset(q_proj_bytes);
            if !self.attn.q_norm.weight.is_null() {
                ops::rms_norm(
                    fwd.gpu,
                    self.rms_norm_k,
                    q_out_i,
                    &self.attn.q_norm,
                    q_out_i,
                    nq,
                    hd,
                    eps,
                    stream,
                )?;
            }
            if !self.attn.k_norm.weight.is_null() {
                ops::rms_norm(
                    fwd.gpu,
                    self.rms_norm_k,
                    k_out_i,
                    &self.attn.k_norm,
                    k_out_i,
                    nkv,
                    hd,
                    eps,
                    stream,
                )?;
            }
        }
        Ok(())
    }

    /// Fused n=3 NVFP4 batched QKV path (ATLAS_ATTN_QKV_FUSED=1).
    ///
    /// Differences vs `ms_qkv_batch3`:
    ///   - GEMVs use the `_strided` variants so they write directly into
    ///     the per-seq-strided `qkv_buf` (no scratch + 9× d2d copies).
    ///   - The 6 q_norm/k_norm launches collapse into one
    ///     `rms_norm_qk_batch3` launch (`grid = (max(nq, nkv), 3, 2)`).
    fn ms_qkv_batch3_fused(&self, c: &MultiSeqCtx<'_>) -> Result<()> {
        let MultiSeqCtx {
            fwd,
            stream,
            h,
            nq,
            nkv,
            hd,
            eps,
            bf16,
            q_proj_dim,
            q_proj_bytes,
            per_seq_qkv,
            normed,
            qkv_buf,
            ..
        } = *c;
        let q_nvfp4 = self.q_weight.as_ref().and_then(|w| w.as_nvfp4()).unwrap();
        let k_nvfp4 = self.k_weight.as_ref().and_then(|w| w.as_nvfp4()).unwrap();
        let v_nvfp4 = self.v_weight.as_ref().and_then(|w| w.as_nvfp4()).unwrap();

        // Per-token stride within qkv_buf in BF16 elements.
        // qkv_buf layout for token i: [Q_i (q_proj_bytes)] [K_i (kv_bytes)] [V_i (kv_bytes)]
        debug_assert!(per_seq_qkv % bf16 == 0, "qkv stride must be BF16-aligned");
        let stride_bf16 = (per_seq_qkv / bf16) as u32;
        let kv_dim = nkv * hd;
        let kv_bytes = kv_dim as usize * bf16;

        // Q + gate writes [Q_all|Gate_all] for each token at qkv_buf + i*stride.
        // Q region is the first `q_proj_bytes` (= q_proj_dim BF16) of every token's slot.
        ops::w4a16_gemv_qg_batch3_strided(
            fwd.gpu,
            self.w4a16_gemv_qg_batch3_strided_k,
            normed,
            q_nvfp4,
            qkv_buf,
            q_proj_dim,
            h as u32,
            nq,
            hd,
            stride_bf16,
            stream,
        )?;

        // K writes at qkv_buf + i*stride + q_proj_bytes
        // V writes at qkv_buf + i*stride + q_proj_bytes + kv_bytes
        let k_base = qkv_buf.offset(q_proj_bytes);
        let v_base = k_base.offset(kv_bytes);
        ops::w4a16_gemv_dual_batch3_strided(
            fwd.gpu,
            self.w4a16_gemv_dual_batch3_strided_k,
            normed,
            k_nvfp4,
            k_base,
            v_nvfp4,
            v_base,
            kv_dim,
            h as u32,
            stride_bf16,
            stream,
        )?;

        // Batched per-head q_norm + k_norm for all 3 tokens.
        // `q_dim` for the norm is `nq * hd` (the actual Q region — gate is
        // not normed). `k_offset` is `q_proj_dim` BF16 elements: for gated
        // layers q_proj_dim == 2*nq*hd because of the Gate slab between Q
        // and K. For ungated, q_proj_dim == nq*hd == q_dim.
        ops::rms_norm_qk_batch3(
            fwd.gpu,
            self.rms_norm_qk_batch3_k,
            qkv_buf,
            &self.attn.q_norm,
            &self.attn.k_norm,
            stride_bf16,
            nq * hd,
            nkv * hd,
            q_proj_dim,
            hd,
            eps,
            stream,
        )?;
        Ok(())
    }

    /// n=2 NVFP4 batched path.
    fn ms_qkv_batch2(&self, c: &MultiSeqCtx<'_>) -> Result<()> {
        let MultiSeqCtx {
            fwd,
            stream,
            h,
            nq,
            nkv,
            hd,
            eps,
            bf16,
            q_proj_dim,
            q_proj_bytes,
            per_seq_qkv,
            normed,
            qkv_buf,
            ..
        } = *c;
        let q_nvfp4 = self.q_weight.as_ref().and_then(|w| w.as_nvfp4()).unwrap();
        let k_nvfp4 = self.k_weight.as_ref().and_then(|w| w.as_nvfp4()).unwrap();
        let v_nvfp4 = self.v_weight.as_ref().and_then(|w| w.as_nvfp4()).unwrap();

        let q_scratch = fwd.buffers.ssm_qkvz();
        if self.gated {
            ops::w4a16_gemv_qg_batch2(
                fwd.gpu,
                self.w4a16_gemv_qg_batch2_k,
                normed,
                q_nvfp4,
                q_scratch,
                q_proj_dim,
                h as u32,
                nq,
                hd,
                stream,
            )?;
        } else {
            ops::w4a16_gemv_batch2(
                fwd.gpu,
                self.w4a16_gemv_batch2_k,
                normed,
                q_nvfp4,
                q_scratch,
                q_proj_dim,
                h as u32,
                stream,
            )?;
        }

        let kv_dim = nkv * hd;
        let kv_bytes = kv_dim as usize * bf16;
        let k_scratch = fwd.buffers.attn_output();
        let v_scratch = k_scratch.offset(2 * kv_bytes);
        ops::w4a16_gemv_dual_batch2(
            fwd.gpu,
            self.w4a16_gemv_dual_batch2_k,
            normed,
            k_nvfp4,
            k_scratch,
            v_nvfp4,
            v_scratch,
            kv_dim,
            h as u32,
            stream,
        )?;

        for i in 0..2usize {
            let q_out_i = qkv_buf.offset(i * per_seq_qkv);
            let k_out_i = q_out_i.offset(q_proj_bytes);
            let v_out_i = k_out_i.offset(kv_bytes);
            fwd.gpu.copy_d2d_async(
                q_scratch.offset(i * q_proj_bytes),
                q_out_i,
                q_proj_bytes,
                stream,
            )?;
            fwd.gpu
                .copy_d2d_async(k_scratch.offset(i * kv_bytes), k_out_i, kv_bytes, stream)?;
            fwd.gpu
                .copy_d2d_async(v_scratch.offset(i * kv_bytes), v_out_i, kv_bytes, stream)?;
        }

        for i in 0..2usize {
            let q_out_i = qkv_buf.offset(i * per_seq_qkv);
            let k_out_i = q_out_i.offset(q_proj_bytes);
            if !self.attn.q_norm.weight.is_null() {
                ops::rms_norm(
                    fwd.gpu,
                    self.rms_norm_k,
                    q_out_i,
                    &self.attn.q_norm,
                    q_out_i,
                    nq,
                    hd,
                    eps,
                    stream,
                )?;
            }
            if !self.attn.k_norm.weight.is_null() {
                ops::rms_norm(
                    fwd.gpu,
                    self.rms_norm_k,
                    k_out_i,
                    &self.attn.k_norm,
                    k_out_i,
                    nkv,
                    hd,
                    eps,
                    stream,
                )?;
            }
        }
        Ok(())
    }

    /// `n in (3, 32]` batched QKV via transposed `w4a16_gemm_t_{m32_n64,m128}`
    /// (preferred when installed) or plain `w4a16_gemm` (M_TILE=64, single tile
    /// at these M) with the regular non-transposed NVFP4 weights. Same
    /// scratch/scatter/deinterleave/norm structure as `ms_qkv_batched_m16`,
    /// but avoids both the per-token weight re-reads of the sequential loop
    /// and the `w4a16_gemm_t_m16` + NVFP4-T path that corrupts on gated
    /// Qwen3.6-27B layers (see the ATLAS_TC_NVFP4_M16_MS_ATTN warning above).
    fn ms_qkv_batched_plain(&self, c: &MultiSeqCtx<'_>) -> Result<()> {
        let MultiSeqCtx {
            fwd,
            n,
            stream,
            h,
            nq,
            nkv,
            hd,
            eps,
            bf16,
            q_proj_dim,
            q_proj_bytes,
            per_seq_qkv,
            normed,
            qkv_buf,
            ..
        } = *c;

        let q_nvfp4 = self.q_weight.as_ref().and_then(|w| w.as_nvfp4()).unwrap();
        let k_nvfp4 = self.k_weight.as_ref().and_then(|w| w.as_nvfp4()).unwrap();
        let v_nvfp4 = self.v_weight.as_ref().and_then(|w| w.as_nvfp4()).unwrap();

        let n_u32 = n as u32;
        let kv_dim = nkv * hd;
        let kv_bytes = kv_dim as usize * bf16;

        // Prefer the transposed M_TILE=128 kernel when the prefill weights
        // are installed: nsys ground truth (2026-06-11) shows the plain
        // non-transposed `w4a16_gemm` runs ~5× off the bandwidth floor at
        // M=17 (median 695μs for the Q projection vs ~140μs floor) due to
        // strided B access; the transposed m128 kernel covers M≤128 in one
        // tile with coalesced loads. Same scratch/scatter layout either way.
        let m128_t = self.w4a16_gemm_t_m128_k.0 != 0
            && self.q_nvfp4_t.is_some()
            && self.k_nvfp4_t.is_some()
            && self.v_nvfp4_t.is_some();
        // Prefer the m32_n64 kernel when present: same single-B-read
        // property as m128 at n ≤ 32 but 2× the CTA count (no SM
        // starvation). Falls back to m128 on older PTX caches.
        let use_m32 = m128_t && n <= 32 && self.w4a16_gemm_t_m32_n64_k.0 != 0;
        let (t_kernel, t_launch): (
            KernelHandle,
            fn(
                &dyn GpuBackend,
                KernelHandle,
                DevicePtr,
                &QuantizedWeight,
                DevicePtr,
                u32,
                u32,
                u32,
                u64,
            ) -> Result<()>,
        ) = if use_m32 {
            (self.w4a16_gemm_t_m32_n64_k, ops::w4a16_gemm_n64_m32)
        } else {
            (self.w4a16_gemm_t_m128_k, ops::w4a16_gemm_n128_m128)
        };

        // Q scratch: [n, q_proj_dim] BF16 contiguous in ssm_qkvz.
        let q_scratch = fwd.buffers.ssm_qkvz();
        // K scratch: [n, kv_dim] BF16 contiguous in attn_output.
        // V scratch: [n, kv_dim] BF16 contiguous at attn_output + n*kv_bytes.
        let k_scratch = fwd.buffers.attn_output();
        let v_scratch = k_scratch.offset(n * kv_bytes);

        if m128_t {
            let q_t = self.q_nvfp4_t.as_ref().unwrap();
            let k_t = self.k_nvfp4_t.as_ref().unwrap();
            let v_t = self.v_nvfp4_t.as_ref().unwrap();
            // Q projection: N=q_proj_dim (12288 gated) → ~192 CTAs on the
            // single-slice m32_n64, already well-provisioned; split-K is a
            // no-op here (like FFN gate/up) so Q always uses `t_launch`.
            t_launch(
                fwd.gpu, t_kernel, normed, q_t, q_scratch, n_u32, q_proj_dim, h as u32, stream,
            )?;

            // ── Target 1: K/V split-K (ATLAS_ATTN_QKV_SPLITK=n) ──
            // K and V are N=kv_dim (1024) → only ~16 CTAs on the single
            // slice: occupancy-starved on GB10's 48 SMs. Route through the
            // proven ffn_down split-K path (partial → FP32 workspace →
            // reduce → BF16) when the gate + kernels + use_m32 + workspace
            // are all present. Token-exact (FP32 partials). Falls through
            // to the single-slice `t_launch` otherwise.
            let qkv_splits = crate::layers::attn_qkv_splitk();
            let ws = if qkv_splits > 0 {
                *self
                    .qkv_splitk_workspace
                    .lock()
                    .map_err(|_| anyhow::anyhow!("qkv_splitk_workspace lock poisoned"))?
            } else {
                None
            };
            let kv_splitk_ok = use_m32
                && qkv_splits > 0
                && self.w4a16_gemm_t_m32_n64_splitk_k.0 != 0
                && self.reduce_splitk_k.0 != 0
                && ws.is_some();
            if let (true, Some(ws)) = (kv_splitk_ok, ws) {
                ops::w4a16_gemm_n64_m32_splitk(
                    fwd.gpu,
                    self.w4a16_gemm_t_m32_n64_splitk_k,
                    self.reduce_splitk_k,
                    normed,
                    k_t,
                    k_scratch,
                    ws,
                    n_u32,
                    kv_dim,
                    h as u32,
                    kv_dim, // ldb == N for tightly-packed T-weight
                    qkv_splits,
                    stream,
                )?;
                ops::w4a16_gemm_n64_m32_splitk(
                    fwd.gpu,
                    self.w4a16_gemm_t_m32_n64_splitk_k,
                    self.reduce_splitk_k,
                    normed,
                    v_t,
                    v_scratch,
                    ws,
                    n_u32,
                    kv_dim,
                    h as u32,
                    kv_dim,
                    qkv_splits,
                    stream,
                )?;
            } else {
                t_launch(
                    fwd.gpu, t_kernel, normed, k_t, k_scratch, n_u32, kv_dim, h as u32, stream,
                )?;
                t_launch(
                    fwd.gpu, t_kernel, normed, v_t, v_scratch, n_u32, kv_dim, h as u32, stream,
                )?;
            }
        } else {
            ops::w4a16_gemm(
                fwd.gpu,
                self.w4a16_gemm_k,
                normed,
                q_nvfp4,
                q_scratch,
                n_u32,
                q_proj_dim,
                h as u32,
                stream,
            )?;
            ops::w4a16_gemm(
                fwd.gpu,
                self.w4a16_gemm_k,
                normed,
                k_nvfp4,
                k_scratch,
                n_u32,
                kv_dim,
                h as u32,
                stream,
            )?;
            ops::w4a16_gemm(
                fwd.gpu,
                self.w4a16_gemm_k,
                normed,
                v_nvfp4,
                v_scratch,
                n_u32,
                kv_dim,
                h as u32,
                stream,
            )?;
        }

        // Scatter + gated deinterleave + per-token q/k norms: identical
        // semantics to ms_qkv_batched_m16 (and to one iteration of the
        // sequential loop after its fused w4a16_gemv_qg).
        for i in 0..n {
            let q_out_i = qkv_buf.offset(i * per_seq_qkv);
            let k_out_i = q_out_i.offset(q_proj_bytes);
            let v_out_i = k_out_i.offset(kv_bytes);
            fwd.gpu.copy_d2d_async(
                q_scratch.offset(i * q_proj_bytes),
                q_out_i,
                q_proj_bytes,
                stream,
            )?;
            fwd.gpu
                .copy_d2d_async(k_scratch.offset(i * kv_bytes), k_out_i, kv_bytes, stream)?;
            fwd.gpu
                .copy_d2d_async(v_scratch.offset(i * kv_bytes), v_out_i, kv_bytes, stream)?;
            if self.gated {
                ops::deinterleave_qg(
                    fwd.gpu,
                    self.deinterleave_qg_k,
                    q_out_i,
                    1,
                    nq,
                    hd,
                    q_proj_dim,
                    stream,
                )?;
            }
            if !self.attn.q_norm.weight.is_null() {
                ops::rms_norm(
                    fwd.gpu,
                    self.rms_norm_k,
                    q_out_i,
                    &self.attn.q_norm,
                    q_out_i,
                    nq,
                    hd,
                    eps,
                    stream,
                )?;
            }
            if !self.attn.k_norm.weight.is_null() {
                ops::rms_norm(
                    fwd.gpu,
                    self.rms_norm_k,
                    k_out_i,
                    &self.attn.k_norm,
                    k_out_i,
                    nkv,
                    hd,
                    eps,
                    stream,
                )?;
            }
        }
        Ok(())
    }

    /// Batched M=16 NVFP4 QKV path for K=γ verify (DFlash γ>3, n=4..=32).
    ///
    /// Replaces the per-token sequential GEMV loop (3*n launches per layer)
    /// with three batched GEMMs (3 launches per layer) using the small-M
    /// specialization `w4a16_gemm_t_m16` (M_TILE=16, redesigned warp
    /// partitioning so all 4 warps share the 16 rows). Requires transposed
    /// NVFP4 weights (`q_nvfp4_t`/`k_nvfp4_t`/`v_nvfp4_t`).
    ///
    /// Output staging: Q goes to `ssm_qkvz` scratch [n, q_proj_dim], K/V go
    /// to `attn_output` scratch [2n, kv_dim] (K then V). Per-token d2d copies
    /// scatter into the strided `qkv_buf` layout. Per-token rms_norm calls
    /// follow (same as the n=3/n=2 batched paths — rms_norm is cheap, the
    /// QKV GEMVs are the bottleneck at n=17).
    fn ms_qkv_batched_m16(&self, c: &MultiSeqCtx<'_>) -> Result<()> {
        let MultiSeqCtx {
            fwd,
            n,
            stream,
            h,
            nq,
            nkv,
            hd,
            eps,
            bf16,
            q_proj_dim,
            q_proj_bytes,
            per_seq_qkv,
            normed,
            qkv_buf,
            ..
        } = *c;

        let q_nvfp4_t = self.q_nvfp4_t.as_ref().unwrap();
        let k_nvfp4_t = self.k_nvfp4_t.as_ref().unwrap();
        let v_nvfp4_t = self.v_nvfp4_t.as_ref().unwrap();

        let n_u32 = n as u32;
        let kv_dim = nkv * hd;
        let kv_bytes = kv_dim as usize * bf16;

        // Q scratch: [n, q_proj_dim] BF16 contiguous in ssm_qkvz.
        let q_scratch = fwd.buffers.ssm_qkvz();
        ops::w4a16_gemm_n128_m16(
            fwd.gpu,
            self.w4a16_gemm_t_m16_k,
            normed,
            q_nvfp4_t,
            q_scratch,
            n_u32,
            q_proj_dim,
            h as u32,
            stream,
        )?;

        // K scratch: [n, kv_dim] BF16 contiguous in attn_output.
        // V scratch: [n, kv_dim] BF16 contiguous at attn_output + n*kv_bytes.
        let k_scratch = fwd.buffers.attn_output();
        let v_scratch = k_scratch.offset(n * kv_bytes);
        ops::w4a16_gemm_n128_m16(
            fwd.gpu,
            self.w4a16_gemm_t_m16_k,
            normed,
            k_nvfp4_t,
            k_scratch,
            n_u32,
            kv_dim,
            h as u32,
            stream,
        )?;
        ops::w4a16_gemm_n128_m16(
            fwd.gpu,
            self.w4a16_gemm_t_m16_k,
            normed,
            v_nvfp4_t,
            v_scratch,
            n_u32,
            kv_dim,
            h as u32,
            stream,
        )?;

        // Scatter each token's (Q, K, V) into the per-seq-strided qkv_buf.
        // qkv_buf layout per token i: [Q (q_proj_bytes) | K (kv_bytes) | V (kv_bytes)].
        //
        // For gated layers the GEMM produces Q+Gate interleaved per head
        // ([Q_h0, G_h0, Q_h1, G_h1, ...]). The downstream multi_seq attention
        // path expects deinterleaved layout ([Q_h0..Q_hN-1, G_h0..G_hN-1])
        // since `ms_phase_o_proj` reads Gate from `q_dim*bf16` offset. We
        // apply `deinterleave_qg` per token after the scatter — same fixup
        // as `ms_qkv_seq_q` does for FP8/Dense gated.
        for i in 0..n {
            let q_out_i = qkv_buf.offset(i * per_seq_qkv);
            let k_out_i = q_out_i.offset(q_proj_bytes);
            let v_out_i = k_out_i.offset(kv_bytes);
            fwd.gpu.copy_d2d_async(
                q_scratch.offset(i * q_proj_bytes),
                q_out_i,
                q_proj_bytes,
                stream,
            )?;
            fwd.gpu
                .copy_d2d_async(k_scratch.offset(i * kv_bytes), k_out_i, kv_bytes, stream)?;
            fwd.gpu
                .copy_d2d_async(v_scratch.offset(i * kv_bytes), v_out_i, kv_bytes, stream)?;
            if self.gated {
                ops::deinterleave_qg(
                    fwd.gpu,
                    self.deinterleave_qg_k,
                    q_out_i,
                    1,
                    nq,
                    hd,
                    q_proj_dim,
                    stream,
                )?;
            }
        }

        // Per-token RMS norm (q_norm/k_norm). Each is a tiny per-head
        // single-block kernel — not the bottleneck. Keeping per-token here
        // preserves correctness for gated layers (Q + Gate share q_proj_dim)
        // without needing a strided rms_norm variant.
        for i in 0..n {
            let q_out_i = qkv_buf.offset(i * per_seq_qkv);
            let k_out_i = q_out_i.offset(q_proj_bytes);
            if !self.attn.q_norm.weight.is_null() {
                ops::rms_norm(
                    fwd.gpu,
                    self.rms_norm_k,
                    q_out_i,
                    &self.attn.q_norm,
                    q_out_i,
                    nq,
                    hd,
                    eps,
                    stream,
                )?;
            }
            if !self.attn.k_norm.weight.is_null() {
                ops::rms_norm(
                    fwd.gpu,
                    self.rms_norm_k,
                    k_out_i,
                    &self.attn.k_norm,
                    k_out_i,
                    nkv,
                    hd,
                    eps,
                    stream,
                )?;
            }
        }
        Ok(())
    }

    /// Sequential per-token Q projection (handles gated and ungated).
    #[allow(clippy::too_many_arguments)]
    fn ms_qkv_seq_q(
        &self,
        fwd: &crate::layer::ForwardContext<'_>,
        normed_i: spark_runtime::gpu::DevicePtr,
        q_out_i: spark_runtime::gpu::DevicePtr,
        q_proj_dim: u32,
        q_dim: u32,
        nq: u32,
        hd: u32,
        h: usize,
        stream: u64,
    ) -> Result<()> {
        if self.gated {
            if let Some(fp8) = self.q_weight.as_ref().and_then(|w| w.as_fp8()) {
                ops::w8a16_gemv(
                    fwd.gpu,
                    self.w8a16_gemv_k,
                    normed_i,
                    fp8.weight,
                    fp8.row_scale,
                    q_out_i,
                    q_proj_dim,
                    h as u32,
                    stream,
                )?;
                ops::deinterleave_qg(
                    fwd.gpu,
                    self.deinterleave_qg_k,
                    q_out_i,
                    1,
                    nq,
                    hd,
                    q_proj_dim,
                    stream,
                )?;
            } else if let Some(nvfp4) = self.q_weight.as_ref().and_then(|w| w.as_nvfp4()) {
                ops::w4a16_gemv_qg(
                    fwd.gpu,
                    self.w4a16_gemv_qg_k,
                    normed_i,
                    nvfp4,
                    q_out_i,
                    q_proj_dim,
                    h as u32,
                    nq,
                    hd,
                    stream,
                )?;
            } else {
                anyhow::ensure!(
                    !self.attn.q_proj.weight.is_null(),
                    "attention BF16 fallback weight was released at load (ATLAS_QWEN4_DENSE_DEDUP)"
                );
                ops::dense_gemv(
                    fwd.gpu,
                    self.dense_gemv_k,
                    normed_i,
                    &self.attn.q_proj,
                    q_out_i,
                    q_proj_dim,
                    h as u32,
                    stream,
                )?;
                ops::deinterleave_qg(
                    fwd.gpu,
                    self.deinterleave_qg_k,
                    q_out_i,
                    1,
                    nq,
                    hd,
                    q_proj_dim,
                    stream,
                )?;
            }
        } else if let Some(fp8) = self.q_weight.as_ref().and_then(|w| w.as_fp8()) {
            ops::w8a16_gemv(
                fwd.gpu,
                self.w8a16_gemv_k,
                normed_i,
                fp8.weight,
                fp8.row_scale,
                q_out_i,
                q_dim,
                h as u32,
                stream,
            )?;
        } else if let Some(nvfp4) = self.q_weight.as_ref().and_then(|w| w.as_nvfp4()) {
            ops::w4a16_gemv(
                fwd.gpu,
                self.w4a16_gemv_k,
                normed_i,
                nvfp4,
                q_out_i,
                q_dim,
                h as u32,
                stream,
            )?;
        } else {
            anyhow::ensure!(
                !self.attn.q_proj.weight.is_null(),
                "attention BF16 fallback weight was released at load (ATLAS_QWEN4_DENSE_DEDUP)"
            );
            ops::dense_gemv(
                fwd.gpu,
                self.dense_gemv_k,
                normed_i,
                &self.attn.q_proj,
                q_out_i,
                q_dim,
                h as u32,
                stream,
            )?;
        }
        Ok(())
    }

    /// Sequential per-token K + V projections.
    #[allow(clippy::too_many_arguments)]
    fn ms_qkv_seq_kv(
        &self,
        fwd: &crate::layer::ForwardContext<'_>,
        normed_i: spark_runtime::gpu::DevicePtr,
        k_out_i: spark_runtime::gpu::DevicePtr,
        v_out_i: spark_runtime::gpu::DevicePtr,
        nkv: u32,
        hd: u32,
        h: usize,
        stream: u64,
    ) -> Result<()> {
        if let (Some(k_fp8), Some(v_fp8)) = (
            self.k_weight.as_ref().and_then(|w| w.as_fp8()),
            self.v_weight.as_ref().and_then(|w| w.as_fp8()),
        ) {
            ops::w8a16_gemv(
                fwd.gpu,
                self.w8a16_gemv_k,
                normed_i,
                k_fp8.weight,
                k_fp8.row_scale,
                k_out_i,
                nkv * hd,
                h as u32,
                stream,
            )?;
            ops::w8a16_gemv(
                fwd.gpu,
                self.w8a16_gemv_k,
                normed_i,
                v_fp8.weight,
                v_fp8.row_scale,
                v_out_i,
                nkv * hd,
                h as u32,
                stream,
            )?;
        } else if let (Some(k_fp4), Some(v_fp4)) = (
            self.k_weight.as_ref().and_then(|w| w.as_nvfp4()),
            self.v_weight.as_ref().and_then(|w| w.as_nvfp4()),
        ) {
            ops::w4a16_gemv_dual(
                fwd.gpu,
                self.w4a16_gemv_dual_k,
                normed_i,
                k_fp4,
                k_out_i,
                v_fp4,
                v_out_i,
                nkv * hd,
                h as u32,
                stream,
            )?;
        } else {
            if let Some(nvfp4) = self.k_weight.as_ref().and_then(|w| w.as_nvfp4()) {
                ops::w4a16_gemv(
                    fwd.gpu,
                    self.w4a16_gemv_k,
                    normed_i,
                    nvfp4,
                    k_out_i,
                    nkv * hd,
                    h as u32,
                    stream,
                )?;
            } else {
                anyhow::ensure!(
                    !self.attn.k_proj.weight.is_null(),
                    "attention BF16 fallback weight was released at load (ATLAS_QWEN4_DENSE_DEDUP)"
                );
                ops::dense_gemv(
                    fwd.gpu,
                    self.dense_gemv_k,
                    normed_i,
                    &self.attn.k_proj,
                    k_out_i,
                    nkv * hd,
                    h as u32,
                    stream,
                )?;
            }
            if let Some(nvfp4) = self.v_weight.as_ref().and_then(|w| w.as_nvfp4()) {
                ops::w4a16_gemv(
                    fwd.gpu,
                    self.w4a16_gemv_k,
                    normed_i,
                    nvfp4,
                    v_out_i,
                    nkv * hd,
                    h as u32,
                    stream,
                )?;
            } else {
                anyhow::ensure!(
                    !self.attn.v_proj.weight.is_null(),
                    "attention BF16 fallback weight was released at load (ATLAS_QWEN4_DENSE_DEDUP)"
                );
                ops::dense_gemv(
                    fwd.gpu,
                    self.dense_gemv_k,
                    normed_i,
                    &self.attn.v_proj,
                    v_out_i,
                    nkv * hd,
                    h as u32,
                    stream,
                )?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod selector_flag_tests {
    use super::{
        AttnProjTcOverride, parse_attn_proj_tc_override, parse_attn_qk_norm_batched,
        parse_attn_qkv_exact_m17_astage, parse_attn_qkv_exact_strided, resolve_attn_proj_tc_for_sm,
        resolve_sm120_bool_opt,
    };

    #[test]
    fn sm120_qkv_auto_defaults_preserve_gb10() {
        assert!(!resolve_sm120_bool_opt(None, 48));
        assert!(!resolve_sm120_bool_opt(None, 64));
        assert!(resolve_sm120_bool_opt(None, 188));
        assert!(!resolve_sm120_bool_opt(Some(false), 188));
        assert!(resolve_sm120_bool_opt(Some(true), 48));

        assert!(!resolve_attn_proj_tc_for_sm(
            AttnProjTcOverride::Auto,
            48,
            32
        ));
        assert!(!resolve_attn_proj_tc_for_sm(
            AttnProjTcOverride::Auto,
            188,
            3
        ));
        assert!(resolve_attn_proj_tc_for_sm(
            AttnProjTcOverride::Auto,
            188,
            4
        ));
        assert!(resolve_attn_proj_tc_for_sm(
            AttnProjTcOverride::Auto,
            188,
            8
        ));
        assert!(resolve_attn_proj_tc_for_sm(
            AttnProjTcOverride::Auto,
            188,
            16
        ));
        assert!(resolve_attn_proj_tc_for_sm(
            AttnProjTcOverride::Auto,
            188,
            64
        ));
        assert!(!resolve_attn_proj_tc_for_sm(
            AttnProjTcOverride::Disabled,
            188,
            64
        ));
        assert!(!resolve_attn_proj_tc_for_sm(
            AttnProjTcOverride::Enabled,
            48,
            8
        ));
        assert!(resolve_attn_proj_tc_for_sm(
            AttnProjTcOverride::Enabled,
            48,
            16
        ));
        assert!(resolve_attn_proj_tc_for_sm(
            AttnProjTcOverride::Enabled,
            188,
            8
        ));
        assert_eq!(
            parse_attn_proj_tc_override(Some("0")),
            AttnProjTcOverride::Disabled
        );
        assert_eq!(
            parse_attn_proj_tc_override(Some("1")),
            AttnProjTcOverride::Enabled
        );
        assert_eq!(parse_attn_proj_tc_override(None), AttnProjTcOverride::Auto);
    }

    #[test]
    fn m17_astage_accepts_only_absent_zero_or_one() {
        assert_eq!(parse_attn_qkv_exact_m17_astage(None), Ok(false));
        assert_eq!(parse_attn_qkv_exact_m17_astage(Some("0")), Ok(false));
        assert_eq!(parse_attn_qkv_exact_m17_astage(Some("1")), Ok(true));
        for invalid in ["", "true", "false", "2", "01", " 1"] {
            assert!(
                parse_attn_qkv_exact_m17_astage(Some(invalid)).is_err(),
                "{invalid:?}"
            );
        }
    }

    #[test]
    fn exact_strided_accepts_only_absent_zero_or_one() {
        assert_eq!(parse_attn_qkv_exact_strided(None), Ok(false));
        assert_eq!(parse_attn_qkv_exact_strided(Some("0")), Ok(false));
        assert_eq!(parse_attn_qkv_exact_strided(Some("1")), Ok(true));
        for invalid in ["", "true", "false", "2", "01", " 1", "1 "] {
            assert!(
                parse_attn_qkv_exact_strided(Some(invalid)).is_err(),
                "{invalid:?}"
            );
        }
    }

    #[test]
    fn qk_norm_batched_accepts_only_absent_zero_or_one() {
        assert_eq!(parse_attn_qk_norm_batched(None), Ok(false));
        assert_eq!(parse_attn_qk_norm_batched(Some("0")), Ok(false));
        assert_eq!(parse_attn_qk_norm_batched(Some("1")), Ok(true));
        for invalid in ["", "true", "false", "2", "01", " 1", "1 "] {
            assert!(
                parse_attn_qk_norm_batched(Some(invalid)).is_err(),
                "{invalid:?}"
            );
        }
    }

    #[test]
    fn deterministic_batch_disables_attn_proj_tc_across_1_to_128() {
        use super::resolve_attn_proj_tc_for_sm_with_policy;
        use crate::layers::deterministic_batch::DeterministicBatchPolicy;

        for sm in [48u32, 188u32] {
            for m in [1usize, 2, 4, 8, 16, 32, 64, 128] {
                for ov in [
                    AttnProjTcOverride::Auto,
                    AttnProjTcOverride::Enabled,
                    AttnProjTcOverride::Disabled,
                ] {
                    assert!(
                        !resolve_attn_proj_tc_for_sm_with_policy(
                            ov,
                            sm,
                            m,
                            DeterministicBatchPolicy::Enabled,
                        ),
                        "attn proj TC must be disabled at M={m} sm={sm} ov={ov:?}"
                    );
                }
            }
        }
        assert!(resolve_attn_proj_tc_for_sm_with_policy(
            AttnProjTcOverride::Auto,
            188,
            8,
            DeterministicBatchPolicy::Disabled,
        ));
    }
}
