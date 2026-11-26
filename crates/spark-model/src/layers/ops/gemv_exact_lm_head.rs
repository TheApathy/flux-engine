// SPDX-License-Identifier: AGPL-3.0-only

//! Launcher and fail-closed routing policy for the exact multi-row NVFP4
//! LM-head GEMV family.

use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use crate::weight_map::QuantizedWeight;

#[path = "gemv_exact_lm_head/route.rs"]
mod route;
pub use route::*;
#[path = "gemv_exact_lm_head/grid32.rs"]
mod grid32;
pub use grid32::*;

pub const W4A16_EXACT_LM_HEAD_OUTS_PER_BLOCK: u32 = 4;

/// Output rows per lane group in the register-tiled family. The grid shrinks
/// by this factor because each lane group now covers T adjacent outputs.
pub const W4A16_EXACT_LM_HEAD_RT2_TILE: u32 = 2;

/// `ATLAS_W4A16_GEMV_RT2=1` opts into the register-tiled exact GEMV family.
///
/// Default OFF. Unset, or any value other than `1`, leaves every launch on the
/// kernel it uses today — the register-tiled twin is numerics-preserving but
/// this stays opt-in so a benchmark campaign cannot pick it up implicitly.
pub fn w4a16_gemv_rt2_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("ATLAS_W4A16_GEMV_RT2").as_deref() == Ok("1"))
}

/// Fewest rows at which the env-driven exact launch takes the register-tiled
/// twin. fn-mtp gemv_bench-2 (FN shapes, rt2 vs plain exact tier): rt2 is
/// slower at M=2..6 (M=5: qkvz +0.12, ssm_out +0.45, lm_head +0.11 ms per
/// step) and faster at M=7..8, so speculative verify (K2..K6) keeps the plain
/// tiers. Both families are byte-identical per row; this only picks the
/// faster one. Prefill callers pass their own choice through
/// [`w4a16_gemv_batch_logits_exact_with`] and keep RT2 (the grid32 prefill
/// route requires it).
pub const W4A16_EXACT_RT2_MIN_ROWS: u32 = 7;

/// `ATLAS_W4A16_EXACT_RT2_MIN_ROWS` (read once) overrides
/// [`W4A16_EXACT_RT2_MIN_ROWS`] for A/B runs; `2` restores rt2 at every width.
pub fn w4a16_exact_rt2_min_rows() -> u32 {
    static MIN_ROWS: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *MIN_ROWS.get_or_init(|| {
        std::env::var("ATLAS_W4A16_EXACT_RT2_MIN_ROWS")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
            .unwrap_or(W4A16_EXACT_RT2_MIN_ROWS)
    })
}

/// Whether an env-driven exact launch of `rows` rows takes the rt2 twin.
pub const fn w4a16_exact_rt2_for_rows(enabled: bool, min_rows: u32, rows: u32) -> bool {
    enabled && rows >= min_rows
}

/// Output-count ceiling for the row-parallel substitution. fn-mtp microbench
/// (FN hyper mixers, 96 per step): N=4 (inject) 1.48 -> 0.59 ms at M=5, but
/// N=320 (down) is slower row-parallel at M>=5 (2.09 vs 1.77 ms), so only
/// tiny-N projections qualify.
pub const W4A16_EXACT_ROWPAR_MAX_N: u32 = 64;

/// `ATLAS_W4A16_EXACT_ROWPAR=1` and a small output count select the
/// row-parallel K1 kernel for exact multi-row GEMVs.
pub fn w4a16_exact_rowpar_selected(kernel: KernelHandle, n: u32) -> bool {
    w4a16_exact_rowpar_policy(w4a16_exact_rowpar_flag(), kernel.0 != 0, n)
}

/// `ATLAS_W4A16_EXACT_ROWPAR=1` (read once). Also replaces per-row K1 launch
/// loops (the Qwen4 exact-rows MoE router) with one `w4a16_gemv_rows` launch.
pub fn w4a16_exact_rowpar_flag() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var("ATLAS_W4A16_EXACT_ROWPAR").as_deref() == Ok("1");
        if on {
            tracing::info!("ENGAGED ATLAS_W4A16_EXACT_ROWPAR: row-parallel exact GEMV for N <= {W4A16_EXACT_ROWPAR_MAX_N} and per-row K1 loops");
        }
        on
    })
}

/// One launch of the K1 `w4a16_gemv` arithmetic over `rows` contiguous
/// activation rows (`w4a16_gemv_rows`, grid.y = row); byte-identical per row
/// to `rows` separate `w4a16_gemv` launches.
#[allow(clippy::too_many_arguments)]
pub fn w4a16_gemv_rows(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: &QuantizedWeight,
    output: DevicePtr,
    rows: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    ensure!(
        rows > 0 && n > 0 && k.is_multiple_of(16),
        "w4a16_gemv_rows shape {rows}x{n}x{k}"
    );
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, W4A16_EXACT_LM_HEAD_OUTS_PER_BLOCK), rows, 1])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight.weight)
        .arg_ptr(weight.weight_scale)
        .arg_f32(weight.weight_scale_2)
        .arg_ptr(output)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

pub const fn w4a16_exact_rowpar_policy(enabled: bool, kernel_present: bool, n: u32) -> bool {
    enabled && kernel_present && n > 0 && n <= W4A16_EXACT_ROWPAR_MAX_N
}

/// Warps per block of the small-K exact twins.
pub const W4A16_EXACT_SWK_WARPS: u32 = 8;
/// K ceiling for the small-K exact twins (Qwen4 hyper up: K=320, K16=20 < 64 lanes).
pub const W4A16_EXACT_SWK_MAX_K: u32 = 512;

/// `ATLAS_W4A16_EXACT_SWK=1`, K <= 512 and an M4/M8 tier with its twin attached
/// select (kernel, outputs per warp).
pub fn w4a16_exact_swk_selected(
    kernels: W4a16ExactLmHeadKernels,
    tier: ExactLmHeadTier,
    k: u32,
) -> Option<(KernelHandle, u32)> {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let on = *ON.get_or_init(|| {
        let on = std::env::var("ATLAS_W4A16_EXACT_SWK").as_deref() == Ok("1");
        if on {
            tracing::info!("ENGAGED ATLAS_W4A16_EXACT_SWK: one-warp-per-output exact GEMV for K <= {W4A16_EXACT_SWK_MAX_K}");
        }
        on
    });
    w4a16_exact_swk_policy(on, kernels.sw_m4, kernels.sw_m8, tier, k)
}

pub const fn w4a16_exact_swk_policy(
    enabled: bool,
    sw_m4: KernelHandle,
    sw_m8: KernelHandle,
    tier: ExactLmHeadTier,
    k: u32,
) -> Option<(KernelHandle, u32)> {
    if !enabled || k == 0 || k > W4A16_EXACT_SWK_MAX_K {
        return None;
    }
    match tier {
        ExactLmHeadTier::M4 if sw_m4.0 != 0 => Some((sw_m4, 2)),
        ExactLmHeadTier::M8 if sw_m8.0 != 0 => Some((sw_m8, 1)),
        _ => None,
    }
}

/// Outputs per warp and warps per block of the wide-N exact twin
/// (`w4a16_gemv_batch_logits_exact_sw_m8_o4w4`).
pub const W4A16_EXACT_SW4_OUTS_PER_WARP: u32 = 4;
pub const W4A16_EXACT_SW4_WARPS: u32 = 4;
/// Core wide-N shape window where `sw4_m8` wins on single-row and wide-N
/// projections (e.g. `qkvz 16384x2560`, LM head `248320x2560`); small K idles
/// lanes (`K16 < 128`).
pub const W4A16_EXACT_SW4_MIN_N: u32 = 8192;
pub const W4A16_EXACT_SW4_MIN_K: u32 = 2048;
pub const W4A16_EXACT_SW4_MAX_K: u32 = 4096;

/// Widened multi-row chunked shape window admitted in
/// [`w4a16_gemv_batch_logits_exact_with`] alongside the core wide-N window
/// (covering mid-width projections such as `ssm_out 2560x6144` up to
/// `N = 32_768`, `K = 6144`).
pub const W4A16_EXACT_SW4_CHUNKED_MIN_N: u32 = 2560;
pub const W4A16_EXACT_SW4_CHUNKED_MAX_N: u32 = 32_768;
pub const W4A16_EXACT_SW4_CHUNKED_MIN_K: u32 = 2048;
pub const W4A16_EXACT_SW4_CHUNKED_MAX_K: u32 = 6144;

/// `ATLAS_W4A16_EXACT_SW4=1|2|all` or `ATLAS_QWEN4_VERIFY_FAST_HOST=1` (read once).
pub fn w4a16_exact_sw4_flag() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let on = matches!(
            std::env::var("ATLAS_W4A16_EXACT_SW4").as_deref(),
            Ok("1" | "2" | "all")
        ) || crate::layers::qwen4_verify_fast_host_selected();
        if on {
            tracing::info!(
                "ATLAS_W4A16_EXACT_SW4 enabled (via ATLAS_W4A16_EXACT_SW4=1|2|all or ATLAS_QWEN4_VERIFY_FAST_HOST=1): four-outputs-per-warp exact GEMV for N >= {W4A16_EXACT_SW4_MIN_N}, K in {W4A16_EXACT_SW4_MIN_K}..={W4A16_EXACT_SW4_MAX_K} (plus multi-row chunked N in {W4A16_EXACT_SW4_CHUNKED_MIN_N}..={W4A16_EXACT_SW4_CHUNKED_MAX_N}, K in {W4A16_EXACT_SW4_CHUNKED_MIN_K}..={W4A16_EXACT_SW4_CHUNKED_MAX_K})"
            );
        }
        on
    })
}

/// `ATLAS_W4A16_EXACT_SW4=2|all` or `ATLAS_QWEN4_VERIFY_FAST_HOST=1`: also
/// route the M4 exact tier (`rows` in `2..=4`) through `sw4_m8` on wide-N
/// bounded-K shapes (`N >= 8192`, `K in 2048..=4096`), matching `rows=5..=8`.
pub fn w4a16_exact_sw4_m4_flag() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        matches!(
            std::env::var("ATLAS_W4A16_EXACT_SW4").as_deref(),
            Ok("2" | "all")
        ) || crate::layers::qwen4_verify_fast_host_selected()
    })
}

/// Log the first wide-N launch of each (N, K) shape (the env read alone
/// proves nothing).
fn sw4_launch_receipt(n: u32, k: u32, rows: u32) {
    static SEEN: std::sync::Mutex<Vec<(u32, u32)>> = std::sync::Mutex::new(Vec::new());
    let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    if !seen.contains(&(n, k)) {
        seen.push((n, k));
        tracing::info!(
            "ENGAGED ATLAS_W4A16_EXACT_SW4: wide-N exact GEMV launched (N={n} K={k} rows={rows})"
        );
    }
}

/// Core wide-N policy for `sw4_m8` (`N >= 8192`, `K in 2048..=4096`), used by
/// `M = 1` [`w4a16_gemv_or_sw4`] and as the primary wide-N clause of
/// [`w4a16_exact_sw4_chunked_policy`].
pub const fn w4a16_exact_sw4_policy(
    enabled: bool,
    kernel_present: bool,
    tier: ExactLmHeadTier,
    n: u32,
    k: u32,
) -> bool {
    enabled
        && kernel_present
        && matches!(tier, ExactLmHeadTier::M8)
        && n >= W4A16_EXACT_SW4_MIN_N
        && k >= W4A16_EXACT_SW4_MIN_K
        && k <= W4A16_EXACT_SW4_MAX_K
}

/// Multi-row chunked `sw4` dispatch policy: admits [`w4a16_exact_sw4_policy`]
/// plus mid-width projections (`N in 2560..=32_768`, `K in 2048..=6144`).
pub const fn w4a16_exact_sw4_chunked_policy(
    enabled: bool,
    kernel_present: bool,
    tier: ExactLmHeadTier,
    n: u32,
    k: u32,
) -> bool {
    w4a16_exact_sw4_policy(enabled, kernel_present, tier, n, k)
        || (enabled
            && kernel_present
            && matches!(tier, ExactLmHeadTier::M8)
            && n >= W4A16_EXACT_SW4_CHUNKED_MIN_N
            && n <= W4A16_EXACT_SW4_CHUNKED_MAX_N
            && k >= W4A16_EXACT_SW4_CHUNKED_MIN_K
            && k <= W4A16_EXACT_SW4_CHUNKED_MAX_K)
}

/// Resolved handles for the four register-sized exact kernels.
///
/// A missing handle selects the separately qualified serial-K1 fallback in
/// production rather than a numerically different batch/MMA kernel.
#[derive(Debug, Clone, Copy)]
pub struct W4a16ExactLmHeadKernels {
    m4: KernelHandle,
    m8: KernelHandle,
    m17: KernelHandle,
    m32: KernelHandle,
    rt2_m4: KernelHandle,
    rt2_m8: KernelHandle,
    rt2_m17: KernelHandle,
    rt2_m32: KernelHandle,
    /// ABI-separated, default-unrouted M17 activation-staging candidate.
    m17_astage: KernelHandle,
    rt2_m32_grid: KernelHandle,
    /// Row-parallel K1 kernel (`w4a16_gemv_rows`); zero = unavailable.
    rowpar: KernelHandle,
    /// One-warp-per-output exact twins for small K: (M4 tier, 2 outputs/warp),
    /// (M8 tier, 1 output/warp), (M10 chunk, 1 output/warp), (M25 chunk, 1 output/warp); zero = unavailable.
    sw_m4: KernelHandle,
    sw_m8: KernelHandle,
    sw_m10: KernelHandle,
    sw_m25: KernelHandle,
    /// Wide-N four-outputs-per-warp M8 twin; zero = unavailable.
    sw4_m8: KernelHandle,
    /// Wide-N four-outputs-per-warp M10 twin (`9..=10` rows); zero = unavailable.
    sw4_m10: KernelHandle,
    /// Two-outputs-per-warp M20 twin (`11..=20` rows, legacy field name `sw4_m15`,
    /// 8 warps); zero = unavailable.
    sw4_m15: KernelHandle,
    /// Two-outputs-per-warp M25 twin (`21..=25` rows, 8 warps); zero = unavailable.
    sw4_m25: KernelHandle,
    /// Single-sweep 16-row-slab wide-row twins (`26..=64`, `65..=80`, `81..=128` rows);
    /// zero = lazily resolved from `w4a16_gemv` at launch time.
    sw_m64: KernelHandle,
    sw_m80: KernelHandle,
    sw_m128: KernelHandle,
}

impl W4a16ExactLmHeadKernels {
    pub const fn new(
        m4: KernelHandle,
        m8: KernelHandle,
        m17: KernelHandle,
        m32: KernelHandle,
    ) -> Self {
        Self {
            m4,
            m8,
            m17,
            m32,
            rt2_m4: KernelHandle(0),
            rt2_m8: KernelHandle(0),
            rt2_m17: KernelHandle(0),
            rt2_m32: KernelHandle(0),
            m17_astage: KernelHandle(0),
            rt2_m32_grid: KernelHandle(0),
            rowpar: KernelHandle(0),
            sw_m4: KernelHandle(0),
            sw_m8: KernelHandle(0),
            sw_m10: KernelHandle(0),
            sw_m25: KernelHandle(0),
            sw4_m8: KernelHandle(0),
            sw4_m10: KernelHandle(0),
            sw4_m15: KernelHandle(0),
            sw4_m25: KernelHandle(0),
            sw_m64: KernelHandle(0),
            sw_m80: KernelHandle(0),
            sw_m128: KernelHandle(0),
        }
    }

    /// Attach the single-sweep 16-row-slab wide-row exact twins
    /// (`w4a16_gemv_batch_logits_exact_sw_{m64,m80,m128}_o1w8`).
    pub const fn with_single_sweep_m64_m80_m128(
        mut self,
        m64: KernelHandle,
        m80: KernelHandle,
        m128: KernelHandle,
    ) -> Self {
        self.sw_m64 = m64;
        self.sw_m80 = m80;
        self.sw_m128 = m128;
        self
    }

    pub const fn single_sweep_for_tier(self, tier: ExactSingleSweepTier) -> KernelHandle {
        match tier {
            ExactSingleSweepTier::M64 => self.sw_m64,
            ExactSingleSweepTier::M80 => self.sw_m80,
            ExactSingleSweepTier::M128 => self.sw_m128,
        }
    }

    /// Attach the wide-N four-outputs-per-warp M8 twin (`ATLAS_W4A16_EXACT_SW4=1`).
    pub const fn with_wide_n_sw4(mut self, m8: KernelHandle) -> Self {
        self.sw4_m8 = m8;
        self
    }

    /// Attach the wide-N four-outputs-per-warp M10 twin (`9..=10` rows).
    pub const fn with_wide_n_sw4_m10(mut self, m10: KernelHandle) -> Self {
        self.sw4_m10 = m10;
        self
    }

    /// Attach the two-outputs-per-warp M20 twin (`11..=20` rows, legacy parameter
    /// name `m15`) and two-outputs-per-warp M25 twin (`21..=25` rows, 8 warps).
    pub const fn with_wide_n_sw4_m15_m25(mut self, m15: KernelHandle, m25: KernelHandle) -> Self {
        self.sw4_m15 = m15;
        self.sw4_m25 = m25;
        self
    }

    pub const fn sw4_m8(self) -> KernelHandle {
        self.sw4_m8
    }

    /// Attach the one-warp-per-output small-K exact twins (`ATLAS_W4A16_EXACT_SWK=1`).
    pub const fn with_small_k_sw(mut self, m4: KernelHandle, m8: KernelHandle) -> Self {
        self.sw_m4 = m4;
        self.sw_m8 = m8;
        self
    }

    /// Attach the one-warp-per-output small-K M10 exact twin (`9..=10` rows).
    pub const fn with_small_k_sw_m10(mut self, m10: KernelHandle) -> Self {
        self.sw_m10 = m10;
        self
    }

    /// Attach the one-warp-per-output small-K M25 exact twin (`11..=25` rows).
    pub const fn with_small_k_sw_m25(mut self, m25: KernelHandle) -> Self {
        self.sw_m25 = m25;
        self
    }

    /// Attach the row-parallel K1 kernel. Used only under
    /// `ATLAS_W4A16_EXACT_ROWPAR=1` and only for small N (see
    /// [`w4a16_exact_rowpar_selected`]).
    pub const fn with_rowpar(mut self, kernel: KernelHandle) -> Self {
        self.rowpar = kernel;
        self
    }

    /// Attach the register-tiled twins. Callers that never do this keep the
    /// zero handles, which the launcher treats as "rt2 unavailable" and falls
    /// through to the shipping kernel regardless of the env gate.
    pub const fn with_rt2(
        mut self,
        m4: KernelHandle,
        m8: KernelHandle,
        m17: KernelHandle,
        m32: KernelHandle,
    ) -> Self {
        self.rt2_m4 = m4;
        self.rt2_m8 = m8;
        self.rt2_m17 = m17;
        self.rt2_m32 = m32;
        self
    }

    /// Attach the ABI-separated M17 activation-staging candidate.  Merely
    /// resolving this symbol never changes normal exact/RT2 routing.
    pub const fn with_m17_astage(mut self, kernel: KernelHandle) -> Self {
        self.m17_astage = kernel;
        self
    }

    pub const fn m17_astage(self) -> KernelHandle {
        self.m17_astage
    }

    pub const fn m17_astage_present(self) -> bool {
        self.m17_astage.0 != 0
    }

    pub const fn rt2_for_tier(self, tier: ExactLmHeadTier) -> KernelHandle {
        match tier {
            ExactLmHeadTier::M4 => self.rt2_m4,
            ExactLmHeadTier::M8 => self.rt2_m8,
            ExactLmHeadTier::M17 => self.rt2_m17,
            ExactLmHeadTier::M32 => self.rt2_m32,
        }
    }

    pub const fn with_rt2_m32_grid(mut self, kernel: KernelHandle) -> Self {
        self.rt2_m32_grid = kernel;
        self
    }

    pub const fn rt2_m32_grid(self) -> KernelHandle {
        self.rt2_m32_grid
    }

    pub const fn for_tier(self, tier: ExactLmHeadTier) -> KernelHandle {
        match tier {
            ExactLmHeadTier::M4 => self.m4,
            ExactLmHeadTier::M8 => self.m8,
            ExactLmHeadTier::M17 => self.m17,
            ExactLmHeadTier::M32 => self.m32,
        }
    }

    pub const fn is_present(self, tier: ExactLmHeadTier) -> bool {
        self.for_tier(tier).0 != 0
    }

    pub const fn route_for_rows(self, rows: u32) -> Option<ExactLmHeadRoute> {
        if let Some(sw_tier) = exact_single_sweep_tier_for_rows(rows)
            && self.single_sweep_for_tier(sw_tier).0 != 0
        {
            return Some(ExactLmHeadRoute::Exact(ExactLmHeadTier::M32));
        }
        if rows >= 33 && rows <= 256 {
            let mut rem = rows;
            while rem > 0 {
                let chunk = if rem == 33 {
                    17
                } else if rem > 32 {
                    32
                } else {
                    rem
                };
                let Some(tier) = exact_lm_head_tier_for_rows(chunk) else {
                    return None;
                };
                if !self.is_present(tier) {
                    return Some(ExactLmHeadRoute::SerialK1(tier));
                }
                rem -= chunk;
            }
            return Some(ExactLmHeadRoute::Exact(ExactLmHeadTier::M32));
        }
        let Some(tier) = exact_lm_head_tier_for_rows(rows) else {
            return None;
        };
        exact_lm_head_route_for_rows(rows, self.is_present(tier))
    }
}

fn resolve_single_sweep_kernel(
    gpu: &dyn GpuBackend,
    kernels: W4a16ExactLmHeadKernels,
    tier: ExactSingleSweepTier,
) -> KernelHandle {
    let explicit = kernels.single_sweep_for_tier(tier);
    if explicit.0 != 0 {
        return explicit;
    }
    static CACHE: std::sync::OnceLock<
        std::sync::Mutex<std::collections::HashMap<(usize, u8), KernelHandle>>,
    > = std::sync::OnceLock::new();
    let gpu_id = std::ptr::from_ref(gpu) as *const () as usize;
    let tier_key = match tier {
        ExactSingleSweepTier::M64 => 0u8,
        ExactSingleSweepTier::M80 => 1u8,
        ExactSingleSweepTier::M128 => 2u8,
    };
    let key = (gpu_id, tier_key);
    let cache = CACHE.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()));
    if let Ok(guard) = cache.lock()
        && let Some(&handle) = guard.get(&key)
    {
        return handle;
    }
    let handle = gpu
        .kernel("w4a16_gemv", tier.symbol())
        .unwrap_or(KernelHandle(0));
    if handle.0 != 0
        && let Ok(mut guard) = cache.lock()
    {
        guard.insert(key, handle);
    }
    handle
}

#[allow(clippy::too_many_arguments)]
fn launch_single_sweep_exact(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    tier: ExactSingleSweepTier,
    input: DevicePtr,
    weight: &QuantizedWeight,
    output: DevicePtr,
    rows: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    ensure!(n > 0, "exact NVFP4 LM-head vocab must be non-zero");
    ensure!(k > 0, "exact NVFP4 LM-head hidden width must be non-zero");
    ensure!(
        k.is_multiple_of(16),
        "exact NVFP4 LM-head hidden width must be divisible by 16, got {k}"
    );
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, tier.outs_per_block()), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight.weight)
        .arg_ptr(weight.weight_scale)
        .arg_f32(weight.weight_scale_2)
        .arg_ptr(output)
        .arg_u32(rows)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

/// Launch the exact row-major NVFP4 LM-head GEMV selected by `rows`.
///
/// ABI: `(A, B_packed, B_scale, scale2, C, M, N, K)`.
#[allow(clippy::too_many_arguments)]
pub fn w4a16_gemv_batch_logits_exact(
    gpu: &dyn GpuBackend,
    kernels: W4a16ExactLmHeadKernels,
    input: DevicePtr,
    weight: &QuantizedWeight,
    output: DevicePtr,
    rows: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    w4a16_gemv_batch_logits_exact_with(
        gpu,
        kernels,
        input,
        weight,
        output,
        rows,
        n,
        k,
        stream,
        w4a16_exact_rt2_for_rows(w4a16_gemv_rt2_enabled(), w4a16_exact_rt2_min_rows(), rows),
    )
}

/// Launch an `M = 1` NVFP4 GEMV, routing wide-N bounded-K shapes (`N >= 8192`,
/// `K in 2048..=4096`) through `w4a16_gemv_batch_logits_exact_sw_m8_o4w4` when
/// `w4a16_exact_sw4_m4_flag()` (`ATLAS_QWEN4_VERIFY_FAST_HOST=1` or
/// `ATLAS_W4A16_EXACT_SW4=2|all`) is enabled. At `M = 1`, `sw4_m8` executes the
/// exact same per-lane dot product, warp reduction tree, and BF16 conversion as
/// `w4a16_gemv`, while computing 4 outputs per warp in registers with zero
/// shared memory and zero `__syncthreads()` barriers.
#[allow(clippy::too_many_arguments)]
pub fn w4a16_gemv_or_sw4(
    gpu: &dyn GpuBackend,
    gemv_kernel: KernelHandle,
    sw4_m8: KernelHandle,
    input: DevicePtr,
    weight: &QuantizedWeight,
    output: DevicePtr,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    if w4a16_exact_sw4_m4_flag()
        && w4a16_exact_sw4_policy(
            w4a16_exact_sw4_flag(),
            sw4_m8.0 != 0,
            ExactLmHeadTier::M8,
            n,
            k,
        )
        && k.is_multiple_of(16)
    {
        sw4_launch_receipt(n, k, 1);
        return KernelLaunch::new(gpu, sw4_m8)
            .grid([
                div_ceil(n, W4A16_EXACT_SW4_OUTS_PER_WARP * W4A16_EXACT_SW4_WARPS),
                1,
                1,
            ])
            .block([32 * W4A16_EXACT_SW4_WARPS, 1, 1])
            .arg_ptr(input)
            .arg_ptr(weight.weight)
            .arg_ptr(weight.weight_scale)
            .arg_f32(weight.weight_scale_2)
            .arg_ptr(output)
            .arg_u32(1)
            .arg_u32(n)
            .arg_u32(k)
            .launch(stream);
    }
    super::w4a16_gemv(gpu, gemv_kernel, input, weight, output, n, k, stream)
}

/// Same launch with the register-tiled choice passed in rather than read from
/// the environment, so a parity harness can drive both families in one process.
#[allow(clippy::too_many_arguments)]
pub fn w4a16_gemv_batch_logits_exact_with(
    gpu: &dyn GpuBackend,
    kernels: W4a16ExactLmHeadKernels,
    input: DevicePtr,
    weight: &QuantizedWeight,
    output: DevicePtr,
    rows: u32,
    n: u32,
    k: u32,
    stream: u64,
    use_rt2: bool,
) -> Result<()> {
    if (33..=256).contains(&rows) {
        if w4a16_exact_rowpar_selected(kernels.rowpar, n) {
            ensure!(
                n > 0 && k > 0 && k.is_multiple_of(16),
                "exact NVFP4 row-parallel GEMV shape {n}x{k}"
            );
            return KernelLaunch::new(gpu, kernels.rowpar)
                .grid([div_ceil(n, W4A16_EXACT_LM_HEAD_OUTS_PER_BLOCK), rows, 1])
                .block([256, 1, 1])
                .arg_ptr(input)
                .arg_ptr(weight.weight)
                .arg_ptr(weight.weight_scale)
                .arg_f32(weight.weight_scale_2)
                .arg_ptr(output)
                .arg_u32(n)
                .arg_u32(k)
                .launch(stream);
        }
        if rows > 128 {
            // Split 129..=256 into two balanced halves (each in 64..=128) so both
            // halves can use the single-sweep M64/M80/M128 kernels without ever
            // leaving a 1-row remainder.
            let first = rows / 2;
            let second = rows - first;
            let in_row_bytes = (k as usize) * 2;
            let out_row_bytes = (n as usize) * 2;
            w4a16_gemv_batch_logits_exact_with(
                gpu, kernels, input, weight, output, first, n, k, stream, use_rt2,
            )?;
            w4a16_gemv_batch_logits_exact_with(
                gpu,
                kernels,
                input.offset(first as usize * in_row_bytes),
                weight,
                output.offset(first as usize * out_row_bytes),
                second,
                n,
                k,
                stream,
                use_rt2,
            )?;
            return Ok(());
        }
        if let Some(sw_tier) = exact_single_sweep_tier_for_rows(rows)
            && (kernels.single_sweep_for_tier(sw_tier).0 != 0 || kernels.m32.0 != 0)
        {
            let sw_k = resolve_single_sweep_kernel(gpu, kernels, sw_tier);
            if sw_k.0 != 0 {
                return launch_single_sweep_exact(
                    gpu, sw_k, sw_tier, input, weight, output, rows, n, k, stream,
                );
            }
        }
        let mut check_rem = rows;
        while check_rem > 0 {
            let chunk = if check_rem == 33 {
                17
            } else {
                check_rem.min(32)
            };
            let tier = exact_lm_head_tier_for_rows(chunk).ok_or_else(|| {
                anyhow::anyhow!("missing exact NVFP4 LM-head tier for chunk M={chunk}")
            })?;
            ensure!(
                kernels.for_tier(tier).0 != 0,
                "missing exact LM-head kernel {}",
                tier.symbol()
            );
            check_rem -= chunk;
        }
        let in_row_bytes = (k as usize) * 2;
        let out_row_bytes = (n as usize) * 2;
        let mut row_off = 0u32;
        while row_off < rows {
            let rem = rows - row_off;
            let chunk = if rem == 33 { 17 } else { rem.min(32) };
            w4a16_gemv_batch_logits_exact_with(
                gpu,
                kernels,
                input.offset(row_off as usize * in_row_bytes),
                weight,
                output.offset(row_off as usize * out_row_bytes),
                chunk,
                n,
                k,
                stream,
                use_rt2,
            )?;
            row_off += chunk;
        }
        return Ok(());
    }
    let tier = exact_lm_head_tier_for_rows(rows).ok_or_else(|| {
        anyhow::anyhow!(
            "exact NVFP4 LM-head rows must be in 2..=256 (M=1 uses ordinary GEMV), got {rows}"
        )
    })?;
    let kernel = kernels.for_tier(tier);
    let explicit_sw = exact_single_sweep_tier_for_rows(rows).map_or(KernelHandle(0), |sw_tier| {
        kernels.single_sweep_for_tier(sw_tier)
    });

    if explicit_sw.0 == 0 {
        ensure!(
            kernel.0 != 0,
            "missing exact LM-head kernel {}",
            tier.symbol()
        );
    }

    // Row-parallel substitution for small N: the K1 kernel per row (blockIdx.y),
    // byte-identical to every exact tier, but the rows run in parallel blocks.
    if w4a16_exact_rowpar_selected(kernels.rowpar, n) {
        ensure!(
            n > 0 && k > 0 && k.is_multiple_of(16),
            "exact NVFP4 row-parallel GEMV shape {n}x{k}"
        );
        return KernelLaunch::new(gpu, kernels.rowpar)
            .grid([div_ceil(n, W4A16_EXACT_LM_HEAD_OUTS_PER_BLOCK), rows, 1])
            .block([256, 1, 1])
            .arg_ptr(input)
            .arg_ptr(weight.weight)
            .arg_ptr(weight.weight_scale)
            .arg_f32(weight.weight_scale_2)
            .arg_ptr(output)
            .arg_u32(n)
            .arg_u32(k)
            .launch(stream);
    }

    // Single-sweep 16-row-slab wide-row exact GEMV (`rows = 26..=32` -> `sw_m64_o1w8`,
    // eliminating the unstaged `m32` drop above `sw4_m25`).
    if let Some(sw_tier) = exact_single_sweep_tier_for_rows(rows) {
        let sw_k = resolve_single_sweep_kernel(gpu, kernels, sw_tier);
        if sw_k.0 != 0 {
            return launch_single_sweep_exact(
                gpu, sw_k, sw_tier, input, weight, output, rows, n, k, stream,
            );
        }
    }

    // Small-K substitution: one warp per output reproduces both 32-lane halves
    // of the base kernel (byte-identical); the exact tiers idle most lanes when
    // K16 < 64. For M17/M32 frames (rows in 9..=32), slice into <=8-row chunks.
    // `w4a16_gemv_batch_logits_exact_sw_m4_o2` is templated for M in 1..=5, so
    // rows == 5 (which routes to ExactLmHeadTier::M8 on full-K) uses the 2-output
    // M<=5 small-K kernel directly.
    let swk_tier = if matches!(tier, ExactLmHeadTier::M8)
        && rows <= 5
        && k <= W4A16_EXACT_SWK_MAX_K
        && kernels.sw_m4.0 != 0
    {
        ExactLmHeadTier::M4
    } else {
        tier
    };
    if let Some((kernel, outs_per_warp)) = w4a16_exact_swk_selected(kernels, swk_tier, k) {
        ensure!(
            n > 0 && k.is_multiple_of(16),
            "exact NVFP4 small-K GEMV shape {n}x{k}"
        );
        return KernelLaunch::new(gpu, kernel)
            .grid([div_ceil(n, outs_per_warp * W4A16_EXACT_SWK_WARPS), 1, 1])
            .block([32 * W4A16_EXACT_SWK_WARPS, 1, 1])
            .arg_ptr(input)
            .arg_ptr(weight.weight)
            .arg_ptr(weight.weight_scale)
            .arg_f32(weight.weight_scale_2)
            .arg_ptr(output)
            .arg_u32(rows)
            .arg_u32(n)
            .arg_u32(k)
            .launch(stream);
    } else if matches!(tier, ExactLmHeadTier::M17 | ExactLmHeadTier::M32)
        && w4a16_exact_swk_selected(kernels, ExactLmHeadTier::M8, k).is_some()
    {
        ensure!(
            n > 0 && k.is_multiple_of(16),
            "exact NVFP4 small-K GEMV shape {n}x{k}"
        );
        let in_row_bytes = (k as usize) * 2;
        let out_row_bytes = (n as usize) * 2;
        let max_chunk = if kernels.sw_m25.0 != 0 {
            25u32
        } else if kernels.sw_m10.0 != 0 {
            10u32
        } else {
            8u32
        };
        let mut row_off = 0u32;
        while row_off < rows {
            let chunk = (rows - row_off).min(max_chunk);
            let (kernel, outs_per_warp) = if chunk > 10 && kernels.sw_m25.0 != 0 {
                (kernels.sw_m25, 1)
            } else if chunk > 8 && kernels.sw_m10.0 != 0 {
                (kernels.sw_m10, 1)
            } else {
                let chunk_tier = if chunk <= 5 {
                    ExactLmHeadTier::M4
                } else {
                    ExactLmHeadTier::M8
                };
                w4a16_exact_swk_selected(kernels, chunk_tier, k)
                    .or_else(|| w4a16_exact_swk_selected(kernels, ExactLmHeadTier::M8, k))
                    .expect("M8 swk checked above")
            };
            KernelLaunch::new(gpu, kernel)
                .grid([div_ceil(n, outs_per_warp * W4A16_EXACT_SWK_WARPS), 1, 1])
                .block([32 * W4A16_EXACT_SWK_WARPS, 1, 1])
                .arg_ptr(input.offset(row_off as usize * in_row_bytes))
                .arg_ptr(weight.weight)
                .arg_ptr(weight.weight_scale)
                .arg_f32(weight.weight_scale_2)
                .arg_ptr(output.offset(row_off as usize * out_row_bytes))
                .arg_u32(chunk)
                .arg_u32(n)
                .arg_u32(k)
                .launch(stream)?;
            row_off += chunk;
        }
        return Ok(());
    }

    // Wide-N four-outputs-per-warp twin: the per-output arithmetic of every
    // exact tier on the shapes it wins at; it takes precedence over rt2.
    let max_sw4_chunk = if kernels.sw4_m25.0 != 0 {
        25
    } else if kernels.sw4_m15.0 != 0 {
        20
    } else if kernels.sw4_m10.0 != 0 {
        10
    } else {
        8
    };
    let sw4_tier = if matches!(tier, ExactLmHeadTier::M4) && w4a16_exact_sw4_m4_flag() {
        ExactLmHeadTier::M8
    } else if matches!(tier, ExactLmHeadTier::M17 | ExactLmHeadTier::M32) && rows <= max_sw4_chunk {
        ExactLmHeadTier::M8
    } else {
        tier
    };
    let use_sw4_chunked = w4a16_exact_sw4_chunked_policy(
        w4a16_exact_sw4_flag(),
        kernels.sw4_m8.0 != 0,
        sw4_tier,
        n,
        k,
    );
    if use_sw4_chunked {
        ensure!(
            k.is_multiple_of(16),
            "exact NVFP4 wide-N GEMV shape {n}x{k}"
        );
        sw4_launch_receipt(n, k, rows);
        let in_row_bytes = (k as usize) * 2;
        let out_row_bytes = (n as usize) * 2;
        let mut row_off = 0u32;
        while row_off < rows {
            let chunk = (rows - row_off).min(max_sw4_chunk);
            let (sw4_k, outs_per_warp, warps) = if chunk > 20 && kernels.sw4_m25.0 != 0 {
                (kernels.sw4_m25, 2, 8)
            } else if chunk > 10 && kernels.sw4_m15.0 != 0 {
                (kernels.sw4_m15, 2, 8)
            } else if chunk > 8 && kernels.sw4_m10.0 != 0 {
                (
                    kernels.sw4_m10,
                    W4A16_EXACT_SW4_OUTS_PER_WARP,
                    W4A16_EXACT_SW4_WARPS,
                )
            } else {
                (
                    kernels.sw4_m8,
                    W4A16_EXACT_SW4_OUTS_PER_WARP,
                    W4A16_EXACT_SW4_WARPS,
                )
            };
            KernelLaunch::new(gpu, sw4_k)
                .grid([div_ceil(n, outs_per_warp * warps), 1, 1])
                .block([32 * warps, 1, 1])
                .arg_ptr(input.offset(row_off as usize * in_row_bytes))
                .arg_ptr(weight.weight)
                .arg_ptr(weight.weight_scale)
                .arg_f32(weight.weight_scale_2)
                .arg_ptr(output.offset(row_off as usize * out_row_bytes))
                .arg_u32(chunk)
                .arg_u32(n)
                .arg_u32(k)
                .launch(stream)?;
            row_off += chunk;
        }
        return Ok(());
    }

    // Register-tiled substitution. Only the kernel handle and the grid divisor
    // change; every argument, the block shape, and the per-output arithmetic
    // are the same, so this is a bandwidth swap rather than a routing decision.
    let rt2 = kernels.rt2_for_tier(tier);
    let (kernel, outs_per_block) = if use_rt2 && rt2.0 != 0 {
        (
            rt2,
            W4A16_EXACT_LM_HEAD_OUTS_PER_BLOCK * W4A16_EXACT_LM_HEAD_RT2_TILE,
        )
    } else {
        (kernel, W4A16_EXACT_LM_HEAD_OUTS_PER_BLOCK)
    };
    ensure!(n > 0, "exact NVFP4 LM-head vocab must be non-zero");
    ensure!(k > 0, "exact NVFP4 LM-head hidden width must be non-zero");
    ensure!(
        k.is_multiple_of(16),
        "exact NVFP4 LM-head hidden width must be divisible by 16, got {k}"
    );

    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, outs_per_block), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight.weight)
        .arg_ptr(weight.weight_scale)
        .arg_f32(weight.weight_scale_2)
        .arg_ptr(output)
        .arg_u32(rows)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

/// Launch the ABI-separated activation-staged M17 exact GEMV.
///
/// This helper contains no environment policy: production first resolves an
/// atomic Disabled/Ineligible/Complete/Missing decision, while raw gates can
/// drive this symbol and the parent independently in one process.
/// ABI: `(A, B_packed, B_scale, scale2, C, M, N, K)`.
#[allow(clippy::too_many_arguments)]
pub fn w4a16_gemv_batch_logits_exact_m17_astage(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: &QuantizedWeight,
    output: DevicePtr,
    rows: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    ensure!(
        (9..=17).contains(&rows),
        "activation-staged exact M17 rows must be in 9..=17, got {rows}"
    );
    ensure!(
        kernel.0 != 0,
        "missing exact M17 activation-staging kernel w4a16_gemv_batch_logits_exact_m17_astage"
    );
    ensure!(
        n > 0,
        "activation-staged exact M17 output width must be non-zero"
    );
    ensure!(
        k > 0,
        "activation-staged exact M17 hidden width must be non-zero"
    );
    ensure!(
        k.is_multiple_of(16),
        "activation-staged exact M17 hidden width must be divisible by 16, got {k}"
    );

    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, W4A16_EXACT_LM_HEAD_OUTS_PER_BLOCK), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight.weight)
        .arg_ptr(weight.weight_scale)
        .arg_f32(weight.weight_scale_2)
        .arg_ptr(output)
        .arg_u32(rows)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

#[cfg(test)]
mod swk_policy_tests {
    use super::{ExactLmHeadTier, w4a16_exact_swk_policy as p};
    use spark_runtime::gpu::KernelHandle;

    #[test]
    fn swk_needs_flag_small_k_and_a_twin_for_the_tier() {
        let (m4, m8) = (KernelHandle(4), KernelHandle(8));
        let q = |on, a, b, t, k| p(on, a, b, t, k).map(|(h, o): (KernelHandle, u32)| (h.0, o));
        assert_eq!(q(true, m4, m8, ExactLmHeadTier::M4, 320), Some((4, 2)));
        assert_eq!(q(true, m4, m8, ExactLmHeadTier::M8, 320), Some((8, 1)));
        assert_eq!(q(true, m4, m8, ExactLmHeadTier::M17, 320), None);
        assert_eq!(q(true, m4, m8, ExactLmHeadTier::M8, 2560), None);
        assert_eq!(q(false, m4, m8, ExactLmHeadTier::M8, 320), None);
        assert_eq!(q(true, KernelHandle(0), m8, ExactLmHeadTier::M4, 320), None);
    }
}

#[cfg(test)]
mod sw4_policy_tests {
    use super::{
        ExactLmHeadTier, w4a16_exact_sw4_chunked_policy as chunked, w4a16_exact_sw4_policy as p,
    };

    #[test]
    fn sw4_needs_flag_kernel_m8_tier_wide_n_and_bounded_k() {
        // Core wide-N policy (used by M=1 `w4a16_gemv_or_sw4` and wide-N LM head):
        // admits qkvz 16384x2560 and LM head 248320x2560, excludes narrow N
        // (ssm_out 2560x6144, hyper down 320x10240) and non-M8 tiers.
        assert!(p(true, true, ExactLmHeadTier::M8, 16_384, 2560));
        assert!(p(true, true, ExactLmHeadTier::M8, 248_320, 2560));
        assert!(!p(true, true, ExactLmHeadTier::M8, 2560, 6144));
        assert!(!p(true, true, ExactLmHeadTier::M8, 320, 10_240));
        assert!(!p(true, true, ExactLmHeadTier::M4, 16_384, 2560));
        assert!(!p(true, true, ExactLmHeadTier::M17, 16_384, 2560));
        assert!(!p(true, true, ExactLmHeadTier::M8, 16_384, 8192));
        assert!(!p(true, true, ExactLmHeadTier::M8, 10_240, 320));
        assert!(p(true, true, ExactLmHeadTier::M8, 8192, 2048));
        assert!(p(true, true, ExactLmHeadTier::M8, 8192, 4096));
        assert!(!p(false, true, ExactLmHeadTier::M8, 16_384, 2560));
        assert!(!p(true, false, ExactLmHeadTier::M8, 16_384, 2560));

        // Multi-row chunked policy additionally admits mid-width projections
        // (N in 2560..=32_768, K in 2048..=6144), such as ssm_out 2560x6144,
        // while still excluding hyper down (N=320) and out-of-window K.
        assert!(chunked(true, true, ExactLmHeadTier::M8, 2560, 6144));
        assert!(chunked(true, true, ExactLmHeadTier::M8, 248_320, 2560));
        assert!(!chunked(true, true, ExactLmHeadTier::M8, 320, 10_240));
        assert!(!chunked(true, true, ExactLmHeadTier::M8, 16_384, 8192));
        assert!(!chunked(true, true, ExactLmHeadTier::M4, 2560, 6144));
    }
}

#[cfg(test)]
mod rt2_rows_policy_tests {
    use super::{W4A16_EXACT_RT2_MIN_ROWS, w4a16_exact_rt2_for_rows as p};

    #[test]
    fn rt2_needs_flag_and_enough_rows() {
        // K5 verify keeps the plain exact tier; K7/K8 and the M17 tier keep rt2.
        assert!(!p(true, W4A16_EXACT_RT2_MIN_ROWS, 5));
        assert!(!p(true, W4A16_EXACT_RT2_MIN_ROWS, 6));
        assert!(p(true, W4A16_EXACT_RT2_MIN_ROWS, 7));
        assert!(p(true, W4A16_EXACT_RT2_MIN_ROWS, 16));
        assert!(!p(false, W4A16_EXACT_RT2_MIN_ROWS, 16));
        // ATLAS_W4A16_EXACT_RT2_MIN_ROWS=2 restores rt2 at every exact width.
        assert!(p(true, 2, 2));
    }
}

#[cfg(test)]
mod rowpar_policy_tests {
    use super::{W4A16_EXACT_ROWPAR_MAX_N, w4a16_exact_rowpar_policy as p};

    #[test]
    fn rowpar_needs_flag_kernel_and_small_n() {
        // Qwen4 hyper mixers: inject N=4 qualifies; down N=320 and up N=10240 do not.
        assert!(p(true, true, 4));
        assert!(!p(true, true, 320));
        assert!(p(true, true, W4A16_EXACT_ROWPAR_MAX_N));
        assert!(!p(true, true, W4A16_EXACT_ROWPAR_MAX_N + 1));
        assert!(!p(true, true, 10_240));
        assert!(!p(true, true, 0));
        assert!(!p(false, true, 320));
        assert!(!p(true, false, 320));
    }
}

#[cfg(test)]
mod single_sweep_route_tests {
    use super::{
        ExactLmHeadRoute, ExactLmHeadTier, ExactSingleSweepTier, W4a16ExactLmHeadKernels,
        exact_single_sweep_route_for_rows, exact_single_sweep_tier_for_rows,
    };
    use spark_runtime::gpu::KernelHandle;

    #[test]
    fn single_sweep_tiers_cover_26_to_128_and_exclude_outside_rows() {
        for rows in [0, 1, 2, 16, 25, 129, u32::MAX] {
            assert_eq!(exact_single_sweep_tier_for_rows(rows), None, "rows={rows}");
        }
        for (rows, tier, symbol) in [
            (
                26,
                ExactSingleSweepTier::M64,
                "w4a16_gemv_batch_logits_exact_sw_m64_o1w8",
            ),
            (
                32,
                ExactSingleSweepTier::M64,
                "w4a16_gemv_batch_logits_exact_sw_m64_o1w8",
            ),
            (
                33,
                ExactSingleSweepTier::M64,
                "w4a16_gemv_batch_logits_exact_sw_m64_o1w8",
            ),
            (
                40,
                ExactSingleSweepTier::M64,
                "w4a16_gemv_batch_logits_exact_sw_m64_o1w8",
            ),
            (
                64,
                ExactSingleSweepTier::M64,
                "w4a16_gemv_batch_logits_exact_sw_m64_o1w8",
            ),
            (
                65,
                ExactSingleSweepTier::M80,
                "w4a16_gemv_batch_logits_exact_sw_m80_o1w8",
            ),
            (
                80,
                ExactSingleSweepTier::M80,
                "w4a16_gemv_batch_logits_exact_sw_m80_o1w8",
            ),
            (
                81,
                ExactSingleSweepTier::M128,
                "w4a16_gemv_batch_logits_exact_sw_m128_o1w8",
            ),
            (
                128,
                ExactSingleSweepTier::M128,
                "w4a16_gemv_batch_logits_exact_sw_m128_o1w8",
            ),
        ] {
            assert_eq!(
                exact_single_sweep_tier_for_rows(rows),
                Some(tier),
                "rows={rows}"
            );
            assert!(tier.max_rows() >= rows, "rows={rows}");
            assert_eq!(tier.symbol(), symbol, "rows={rows}");
            assert_eq!(tier.outs_per_block(), 8, "rows={rows}");
            assert_eq!(
                exact_single_sweep_route_for_rows(rows, true, true, true),
                Some(tier),
                "rows={rows}"
            );
            assert_eq!(
                exact_single_sweep_route_for_rows(rows, false, false, false),
                None,
                "rows={rows}"
            );
        }
    }

    #[test]
    fn single_sweep_handles_route_exact_without_breaking_legacy_chunk_fallback() {
        let sw_only = W4a16ExactLmHeadKernels::new(
            KernelHandle(0),
            KernelHandle(0),
            KernelHandle(0),
            KernelHandle(0),
        )
        .with_single_sweep_m64_m80_m128(
            KernelHandle(64),
            KernelHandle(80),
            KernelHandle(128),
        );
        for rows in [26, 32, 33, 40, 64, 65, 80, 81, 128] {
            assert_eq!(
                sw_only.route_for_rows(rows),
                Some(ExactLmHeadRoute::Exact(ExactLmHeadTier::M32)),
                "rows={rows}"
            );
        }
        assert_eq!(
            sw_only.route_for_rows(25),
            Some(ExactLmHeadRoute::SerialK1(ExactLmHeadTier::M32))
        );
        assert_eq!(
            sw_only.route_for_rows(129),
            Some(ExactLmHeadRoute::SerialK1(ExactLmHeadTier::M32))
        );
        assert_eq!(sw_only.route_for_rows(257), None);

        let full = W4a16ExactLmHeadKernels::new(
            KernelHandle(4),
            KernelHandle(8),
            KernelHandle(17),
            KernelHandle(32),
        );
        for rows in [129, 160, 192, 224, 256] {
            assert_eq!(
                full.route_for_rows(rows),
                Some(ExactLmHeadRoute::Exact(ExactLmHeadTier::M32)),
                "rows={rows}"
            );
        }
        assert_eq!(full.route_for_rows(257), None);

        // Missing selected single-sweep tier never promotes to another single-sweep tier.
        assert_eq!(
            exact_single_sweep_route_for_rows(64, false, true, true),
            None
        );
        assert_eq!(
            exact_single_sweep_route_for_rows(65, true, false, true),
            None
        );
        assert_eq!(
            exact_single_sweep_route_for_rows(81, true, true, false),
            None
        );
    }
}
