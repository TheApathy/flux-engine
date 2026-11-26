// SPDX-License-Identifier: AGPL-3.0-only

//! Native CUDA 13.0 cuBLASLt NVFP4 (`CUDA_R_4F_E2M1` + `VEC16_UE4M3`) fast
//! projection backend for Flash-Next dense projections (`qkvz`, `ssm.out`,
//! `q_proj`, `k_proj`, `v_proj`, `o_proj`).
//!
//! Selected by `ATLAS_QWEN4_FAST_PROJ_NVFP4`:
//! - absent / `""` / `"auto"`: auto-selects `"all"` on high-SM Blackwell GPUs
//!   (`active_sm_count() > 64` AND `active_compute_capability() >= 10.0`, e.g.
//!   SM120/SM103) and stays `"off"` on GB10 (`active_sm_count() <= 64`) and
//!   pre-Blackwell GPUs (`CC < 10.0`, preserving exact existing FP8/W4A16 behavior).
//! - `"0"` / `"off"` / `"false"`: explicitly disabled everywhere.
//! - `"wide"`: wide dense projections (`N >= 2560`: `qkvz`, `ssm.out`,
//!   `q_proj`, `o_proj`), keeping narrow `k_proj`/`v_proj` (`N = 512`) on FP8.
//! - `"1"` / `"all"` / `"on"` / `"true"`: all eligible dense projections
//!   (`N >= 512`, excluding MoE router `qg == Some((0, 0))`, hyperconnections,
//!   and `lm_head`).
//!
//! Zero host synchronization per call: activation `amax`, `128x4` UE4M3 block
//! scales, packed E2M1 nibbles, and device FP32 `alpha = act_scale2 * weight_scale_2`
//! are produced on stream and passed via [`BlockScaleScalars::device_alpha`],
//! making the steady-state path 100% CUDA-graph capture safe.

use std::collections::{HashMap, HashSet};
use std::sync::{Mutex, OnceLock};

use anyhow::{Context, Result, bail};
use spark_runtime::cublaslt::{
    self, BlockScaleScalars, GemmDtype, NVFP4_BLOCK_SIZE, block_scale_padded_bytes,
};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use crate::weight_map::QuantizedWeight;

/// Primary selector gating native cuBLASLt NVFP4 dense projections.
pub(crate) const NVFP4_SELECTOR: &str = "ATLAS_QWEN4_FAST_PROJ_NVFP4";

/// Optional override for the minimum batched-verify row count (`scope_active`)
/// at which cuBLASLt NVFP4 supersedes single-launch coalesced W4A16.
pub(crate) const NVFP4_MIN_M_ENV: &str = "ATLAS_QWEN4_FAST_PROJ_NVFP4_MIN_M";

/// Minimum output width `N` admitted in [`Nvfp4FastProjMode::Wide`] mode
/// (`ssm.qkvz` 16384, `attn.q_proj` 12288, `ssm.out` 2560, `attn.o_proj` 2560).
pub(crate) const NVFP4_WIDE_MIN_N: usize = 2_560;

/// Minimum output width `N` admitted in [`Nvfp4FastProjMode::All`] mode
/// (adds `attn.k_proj` / `attn.v_proj` at `N = 512`).
pub(crate) const NVFP4_ALL_MIN_N: usize = 512;

/// Maximum output width `N` admitted for dense projections (excludes `lm_head`
/// vocabulary projections at `N >= 65536` from 4-bit activation quantization).
pub(crate) const NVFP4_MAX_N: usize = 32_768;

/// Minimum inner width `K` admitted (excludes short-K `hc.up` at `K = 320`).
pub(crate) const NVFP4_MIN_K: usize = 1_024;

/// Default minimum batch rows inside batched verify (`scope_active == true`)
/// on GB10 (`sm_count <= 64`) where cuBLASLt NVFP4 supersedes single-launch
/// coalesced W4A16.
pub(crate) const DEFAULT_NVFP4_VERIFY_MIN_M: usize = 64;

/// Default minimum batch rows inside batched verify (`scope_active == true`)
/// on high-SM Blackwell GPUs (`sm_count > 64`), keeping `C=4/8` (`rows=32..64`)
/// and ragged `C=16` on cuBLASLt NVFP4 Tensor Cores.
pub(crate) const DEFAULT_NVFP4_VERIFY_MIN_M_HIGH_SM: usize = 32;

/// Minimum batched-verify row capacity pre-allocated for CUDA-graph safety.
const VERIFY_SCRATCH_MIN_M: usize = 256;

/// Minimum inner-width capacity pre-allocated for Flash-Next dense projections
/// (`ssm.out` / `attn.o_proj` use `K = 6144`).
const SCRATCH_MIN_K: usize = 6_144;

/// Maximum `grid.y` for `qwen4_nvfp4_act_quant` kernels (grid-stride loops
/// cover larger row counts cleanly without exceeding CUDA's 65535 Y limit).
const MAX_QUANT_GRID_Y: u32 = 32_768;

/// Kernel module name built from `kernels/gb10/qwen3.8-flash-next/nvfp4/qwen4_nvfp4_act_quant.cu`.
const QUANT_MODULE: &str = "qwen4_nvfp4_act_quant";

/// Selection mode for `ATLAS_QWEN4_FAST_PROJ_NVFP4`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub(crate) enum Nvfp4FastProjMode {
    /// Disabled (default): preserve exact existing FP8 and coalesced W4A16 paths.
    #[default]
    Off,
    /// Wide dense projections (`N >= 2560`: `qkvz`, `ssm.out`, `q_proj`, `o_proj`);
    /// narrow `k_proj`/`v_proj` (`N = 512`) remain on FP8 for KV-precision A/B.
    Wide,
    /// All eligible Flash-Next dense projections (`N >= 512`: `qkvz`, `ssm.out`,
    /// `q_proj`, `k_proj`, `v_proj`, `o_proj`; excludes MoE router, hyperconnections,
    /// and `lm_head`).
    All,
}

impl Nvfp4FastProjMode {
    /// Returns `true` when any native NVFP4 projection mode is enabled.
    #[inline]
    #[must_use]
    pub(crate) const fn is_enabled(self) -> bool {
        !matches!(self, Self::Off)
    }

    /// Minimum output width `N` admitted by this mode.
    #[inline]
    #[must_use]
    pub(crate) const fn min_n(self) -> usize {
        match self {
            Self::Off => usize::MAX,
            Self::Wide => NVFP4_WIDE_MIN_N,
            Self::All => NVFP4_ALL_MIN_N,
        }
    }

    /// Short telemetry label for one-time engagement logs.
    #[inline]
    #[must_use]
    pub(crate) const fn label(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Wide => "wide",
            Self::All => "all",
        }
    }

    /// Whether a projection weight `(n, k, qg)` is eligible for native cuBLASLt NVFP4.
    ///
    /// Excludes the MoE router sentinel `qg == Some((0, 0))`, hyperconnections
    /// (`N = 320` or `K = 320`), `lm_head` (`N > 32768`), and shapes violating
    /// cuBLASLt NVFP4 alignment (`n % 16 == 0`, `k % 32 == 0`).
    #[inline]
    #[must_use]
    pub(crate) const fn admits_weight(
        self,
        n: usize,
        k: usize,
        qg: Option<(usize, usize)>,
    ) -> bool {
        if !self.is_enabled() {
            return false;
        }
        if let Some((heads, head_dim)) = qg
            && (heads == 0
                || head_dim == 0
                || heads.saturating_mul(head_dim).saturating_mul(2) != n)
        {
            return false;
        }
        n >= self.min_n()
            && n <= NVFP4_MAX_N
            && k >= NVFP4_MIN_K
            && n.is_multiple_of(16)
            && k.is_multiple_of(32)
    }

    /// Whether a full projection call `(m, n, k, ldc, qg)` should execute on
    /// native cuBLASLt NVFP4 under `scope_active` (batched verify vs prefill).
    ///
    /// In prefill (`!scope_active`), every `m >= 1` is admitted so short warmup
    /// prompts never fall back to FP8 and materialize the 2.55 GiB FP8 weight cache.
    /// In batched verify (`scope_active`), `m >= verify_min_m` hands compute-bound
    /// wide cohorts (`M = 64..=256`) to cuBLASLt NVFP4 while keeping small `M` on
    /// single-launch coalesced W4A16.
    #[allow(clippy::too_many_arguments)]
    #[inline]
    #[must_use]
    pub(crate) const fn admits_call(
        self,
        m: usize,
        n: usize,
        k: usize,
        ldc: usize,
        qg: Option<(usize, usize)>,
        scope_active: bool,
        verify_min_m: usize,
    ) -> bool {
        if !self.admits_weight(n, k, qg) || m == 0 || ldc < n || !ldc.is_multiple_of(8) {
            return false;
        }
        if scope_active {
            m >= verify_min_m
        } else {
            true
        }
    }

    /// Whether a `try_coalesced_nvfp4_w4a16_gemm` call inside a batched-verify
    /// dense TC scope should upgrade to native cuBLASLt NVFP4.
    ///
    /// Restricts to wide dense projections (`N >= 2560`: `ssm.qkvz`, `ssm.out`,
    /// `attn.q_proj`, `attn.o_proj`) so MoE router (`N = 512`), hyperconnections
    /// (`N = 320` or `K = 320`), and `lm_head` (`N >= 65536`) are never diverted.
    #[inline]
    #[must_use]
    pub(crate) const fn admits_coalesced_verify(
        self,
        scope_active: bool,
        m: usize,
        n: usize,
        k: usize,
        verify_min_m: usize,
    ) -> bool {
        scope_active
            && self.is_enabled()
            && m >= verify_min_m
            && n >= NVFP4_WIDE_MIN_N
            && n <= NVFP4_MAX_N
            && k >= NVFP4_MIN_K
            && n.is_multiple_of(16)
            && k.is_multiple_of(32)
    }
}

/// SM count threshold above which unset/`"auto"` [`NVFP4_SELECTOR`] auto-enables
/// on Blackwell GPUs (keeping GB10 `<= 64` SMs [`Nvfp4FastProjMode::Off`]).
pub(crate) const NVFP4_HIGH_SM_THRESHOLD: u32 = 64;

/// Minimum CUDA compute capability major version required for auto-enabling
/// native cuBLASLt NVFP4 (Blackwell `SM100+` / `SM120+`; prevents Hopper/Ada/Ampere
/// `> 64`-SM GPUs from attempting unsupported FP4 Tensor Core GEMMs).
pub(crate) const NVFP4_MIN_COMPUTE_MAJOR: u32 = 10;

/// Parse `ATLAS_QWEN4_FAST_PROJ_NVFP4` into an explicit override (`Some(mode)`)
/// or device-aware auto selection (`None` for unset / `""` / `"auto"`).
fn parse_mode_override(raw: Option<&str>) -> Result<Option<Nvfp4FastProjMode>> {
    let Some(raw) = raw else {
        return Ok(None);
    };
    match raw.trim().to_ascii_lowercase().as_str() {
        "" | "auto" => Ok(None),
        "0" | "off" | "false" => Ok(Some(Nvfp4FastProjMode::Off)),
        "wide" => Ok(Some(Nvfp4FastProjMode::Wide)),
        "1" | "all" | "on" | "true" => Ok(Some(Nvfp4FastProjMode::All)),
        other => bail!(
            "{NVFP4_SELECTOR}: unknown mode {other:?} (expected 0/off, auto, wide, 1, or all)"
        ),
    }
}

/// Resolve an optional explicit override against `(sm_count, compute_capability)`:
/// unset/`"auto"` (`None`) defaults to [`Nvfp4FastProjMode::All`] only when
/// `sm_count > 64` AND `compute_major >= 10` (SM120/SM103 Blackwell), staying
/// [`Nvfp4FastProjMode::Off`] on GB10 (`<= 64`) and all pre-Blackwell GPUs.
#[inline]
#[must_use]
pub(crate) const fn resolve_mode_for_device(
    override_mode: Option<Nvfp4FastProjMode>,
    sm_count: u32,
    cc: Option<(u32, u32)>,
) -> Nvfp4FastProjMode {
    match override_mode {
        Some(mode) => mode,
        None if sm_count > NVFP4_HIGH_SM_THRESHOLD
            && matches!(cc, Some((maj, _)) if maj >= NVFP4_MIN_COMPUTE_MAJOR) =>
        {
            Nvfp4FastProjMode::All
        }
        None => Nvfp4FastProjMode::Off,
    }
}

/// Parse `ATLAS_QWEN4_FAST_PROJ_NVFP4` for a GPU with `(sm_count, cc)`:
/// - `None` / `""` / `"auto"`: `All` when `sm_count > 64` and `cc >= 10.0`,
///   `Off` on GB10 (`<= 64`) and non-Blackwell GPUs (`cc < 10.0` or unknown).
/// - `"0"` / `"off"` / `"false"` -> `Off`
/// - `"wide"` -> `Wide`
/// - `"1"` / `"all"` / `"on"` / `"true"` -> `All`
#[cfg(test)]
pub(crate) fn parse_mode_for_device(
    raw: Option<&str>,
    sm_count: u32,
    cc: Option<(u32, u32)>,
) -> Result<Nvfp4FastProjMode> {
    Ok(resolve_mode_for_device(
        parse_mode_override(raw)?,
        sm_count,
        cc,
    ))
}

/// Process-wide parsed [`Nvfp4FastProjMode`]; malformed values fail the first call.
///
/// Caches only the parsed env override so unset/`"auto"` resolves against live
/// `active_sm_count()` and `active_compute_capability()` (preventing pre-CUDA
/// probes from latching uninitialized device properties).
pub(crate) fn mode() -> Result<Nvfp4FastProjMode> {
    static OVERRIDE: OnceLock<std::result::Result<Option<Nvfp4FastProjMode>, String>> =
        OnceLock::new();
    let parsed = OVERRIDE
        .get_or_init(|| {
            parse_mode_override(std::env::var(NVFP4_SELECTOR).ok().as_deref())
                .map_err(|e| e.to_string())
        })
        .clone()
        .map_err(anyhow::Error::msg)?;
    Ok(resolve_mode_for_device(
        parsed,
        atlas_core::device::active_sm_count(),
        atlas_core::device::active_compute_capability(),
    ))
}

/// Non-fallible view of [`mode`] for pure query sites (e.g. cache-readiness probes).
#[inline]
#[must_use]
pub(crate) fn mode_or_off() -> Nvfp4FastProjMode {
    mode().unwrap_or(Nvfp4FastProjMode::Off)
}

/// Default batched-verify row floor for native cuBLASLt NVFP4 on a GPU with
/// `sm_count` SMs (`32` on `sm_count > 64`, `64` on GB10 `<= 64`).
#[inline]
#[must_use]
pub(crate) const fn nvfp4_verify_min_m_for_sm(sm_count: u32) -> usize {
    if sm_count > NVFP4_HIGH_SM_THRESHOLD {
        DEFAULT_NVFP4_VERIFY_MIN_M_HIGH_SM
    } else {
        DEFAULT_NVFP4_VERIFY_MIN_M
    }
}

/// Parse an explicit positive integer override from `ATLAS_QWEN4_FAST_PROJ_NVFP4_MIN_M`.
#[inline]
#[must_use]
fn parse_verify_min_m_override(raw: Option<&str>) -> Option<usize> {
    raw.and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|&v| v > 0)
}

/// Parse `ATLAS_QWEN4_FAST_PROJ_NVFP4_MIN_M` for a GPU with `sm_count` SMs.
#[cfg(test)]
#[inline]
#[must_use]
pub(crate) fn parse_verify_min_m_for_sm(raw: Option<&str>, sm_count: u32) -> usize {
    parse_verify_min_m_override(raw).unwrap_or_else(|| nvfp4_verify_min_m_for_sm(sm_count))
}

/// Parse `ATLAS_QWEN4_FAST_PROJ_NVFP4_MIN_M` against the active device SM count.
#[cfg(test)]
#[inline]
#[must_use]
pub(crate) fn parse_verify_min_m(raw: Option<&str>) -> usize {
    parse_verify_min_m_for_sm(raw, atlas_core::device::active_sm_count())
}

/// Minimum batched-verify row count (`scope_active == true`) for native cuBLASLt NVFP4.
///
/// Caches only the parsed env override so unset resolves against live
/// `active_sm_count()` (`32` on `sm_count > 64`, `64` on GB10 `<= 64`).
#[inline]
#[must_use]
pub(crate) fn verify_min_m() -> usize {
    static OVERRIDE: OnceLock<Option<usize>> = OnceLock::new();
    let parsed = *OVERRIDE.get_or_init(|| {
        parse_verify_min_m_override(std::env::var(NVFP4_MIN_M_ENV).ok().as_deref())
    });
    parsed.unwrap_or_else(|| nvfp4_verify_min_m_for_sm(atlas_core::device::active_sm_count()))
}

/// Outcome of routing [`super::fp8_gemm_shared_act`] through native NVFP4.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SharedActNvfp4Outcome {
    /// Mode is off, no output is eligible, or kernels/capture state fell back to FP8.
    NotHandled,
    /// All outputs were computed via native NVFP4 with a single activation quantization.
    HandledAll,
}

#[derive(Clone, Copy, Debug)]
struct Nvfp4QuantKernels {
    act_amax: KernelHandle,
    quant_bf16: KernelHandle,
    quant_e4m3: KernelHandle,
    swizzle_weight: KernelHandle,
    compute_alphas4: KernelHandle,
}

impl Nvfp4QuantKernels {
    #[inline]
    const fn is_complete(self) -> bool {
        self.act_amax.0 != 0
            && self.quant_bf16.0 != 0
            && self.quant_e4m3.0 != 0
            && self.swizzle_weight.0 != 0
            && self.compute_alphas4.0 != 0
    }
}

struct Nvfp4QuantKernelCache {
    primary: OnceLock<(usize, Nvfp4QuantKernels)>,
    fallback: OnceLock<parking_lot::Mutex<HashMap<usize, Nvfp4QuantKernels>>>,
}

impl Nvfp4QuantKernelCache {
    const fn new() -> Self {
        Self {
            primary: OnceLock::new(),
            fallback: OnceLock::new(),
        }
    }

    fn get_or_load(&self, gpu: &dyn GpuBackend) -> Nvfp4QuantKernels {
        let key = gpu_backend_key(gpu);
        if let Some(&(cached_key, cached_val)) = self.primary.get() {
            if cached_key == key {
                return cached_val;
            }
        } else {
            let val = probe_nvfp4_quant_kernels(gpu);
            let &(stored_key, stored_val) = self.primary.get_or_init(|| (key, val));
            if stored_key == key {
                return stored_val;
            }
            let map = self
                .fallback
                .get_or_init(|| parking_lot::Mutex::new(HashMap::new()));
            return *map.lock().entry(key).or_insert(val);
        }
        let map = self
            .fallback
            .get_or_init(|| parking_lot::Mutex::new(HashMap::new()));
        if let Some(val) = map.lock().get(&key).copied() {
            return val;
        }
        let val = probe_nvfp4_quant_kernels(gpu);
        *map.lock().entry(key).or_insert(val)
    }
}

#[inline]
fn gpu_backend_key(gpu: &dyn GpuBackend) -> usize {
    std::ptr::from_ref(gpu) as *const () as usize
}

fn probe_nvfp4_quant_kernels(gpu: &dyn GpuBackend) -> Nvfp4QuantKernels {
    let act_amax = {
        let self_contained =
            crate::layers::try_kernel(gpu, QUANT_MODULE, "qwen4_nvfp4_act_amax_bf16");
        if self_contained.0 != 0 {
            self_contained
        } else {
            crate::layers::try_kernel(gpu, "qwen4_fp8_proj", "qwen4_fp8_act_amax")
        }
    };
    Nvfp4QuantKernels {
        act_amax,
        quant_bf16: crate::layers::try_kernel(
            gpu,
            QUANT_MODULE,
            "qwen4_nvfp4_quant_bf16_128x4_from_amax",
        ),
        quant_e4m3: crate::layers::try_kernel(gpu, QUANT_MODULE, "qwen4_nvfp4_quant_e4m3_128x4"),
        swizzle_weight: crate::layers::try_kernel(
            gpu,
            QUANT_MODULE,
            "qwen4_nvfp4_swizzle_weight_scales_128x4",
        ),
        compute_alphas4: crate::layers::try_kernel(
            gpu,
            QUANT_MODULE,
            "qwen4_nvfp4_compute_alphas4",
        ),
    }
}

fn quant_kernel_cache() -> &'static Nvfp4QuantKernelCache {
    static CACHE: Nvfp4QuantKernelCache = Nvfp4QuantKernelCache::new();
    &CACHE
}

/// Returns `true` when every `qwen4_nvfp4_act_quant` kernel handle required by
/// native cuBLASLt NVFP4 dense projections is loaded on `gpu`.
#[must_use]
pub(crate) fn quant_kernels_ready(gpu: &dyn GpuBackend) -> bool {
    quant_kernel_cache().get_or_load(gpu).is_complete()
}

fn quant_kernels(gpu: &dyn GpuBackend) -> Option<Nvfp4QuantKernels> {
    let ks = quant_kernel_cache().get_or_load(gpu);
    if ks.is_complete() {
        Some(ks)
    } else {
        static MISSING_WARNED: std::sync::Once = std::sync::Once::new();
        crate::layers::warn_kernel_fallback(
            &MISSING_WARNED,
            NVFP4_SELECTOR,
            "qwen4_nvfp4_act_quant::*",
            "falling back to FP8 / coalesced W4A16 projections",
        );
        None
    }
}

/// Grow-only stream-ordered scratch for quantized NVFP4 activations, `128x4`
/// block scales, and device-resident scalar metadata (`[amax, scale2, alphas[0..4]]`).
#[derive(Clone, Copy, Debug)]
struct Nvfp4ActScratch {
    a_packed: (DevicePtr, usize),
    a_scale: (DevicePtr, usize),
    meta: DevicePtr,
    ba_f32: (DevicePtr, usize),
}

impl Nvfp4ActScratch {
    const fn empty() -> Self {
        Self {
            a_packed: (DevicePtr::NULL, 0),
            a_scale: (DevicePtr::NULL, 0),
            meta: DevicePtr::NULL,
            ba_f32: (DevicePtr::NULL, 0),
        }
    }

    #[inline]
    const fn a_amax(self) -> DevicePtr {
        self.meta
    }

    #[inline]
    fn a_scale2(self) -> DevicePtr {
        self.meta.offset(16)
    }

    #[inline]
    fn alpha(self, idx: usize) -> DevicePtr {
        self.meta.offset(32 + idx * 4)
    }

    #[inline]
    fn covers(self, m_cap: usize, k_cap: usize) -> bool {
        let need_packed = m_cap * (k_cap / 2);
        let need_scale = block_scale_padded_bytes(m_cap as u32, k_cap as u32, NVFP4_BLOCK_SIZE);
        !self.meta.is_null() && self.a_packed.1 >= need_packed && self.a_scale.1 >= need_scale
    }
}

static NVFP4_SCRATCH: Mutex<Option<HashMap<usize, Nvfp4ActScratch>>> = Mutex::new(None);

#[inline]
#[must_use]
fn nvfp4_scratch_caps(m: usize, k: usize, scope_active: bool) -> (usize, usize) {
    let verify_floor = if scope_active {
        VERIFY_SCRATCH_MIN_M
    } else {
        0
    };
    let chunk_floor = crate::layers::qwen4_prefill_moe::max_chunk_rows();
    let m_cap = m.max(verify_floor).max(chunk_floor).next_multiple_of(256);
    let k_cap = k.max(SCRATCH_MIN_K);
    (m_cap, k_cap)
}

/// Whether any initialized process-wide NVFP4 activation scratch already covers
/// `(m, k)` without allocating GPU memory (CUDA-graph capture safe).
#[must_use]
pub(crate) fn scratch_ready_for_capture(m: usize, k: usize) -> bool {
    let Ok(guard) = NVFP4_SCRATCH.lock() else {
        return false;
    };
    let Some(map) = guard.as_ref() else {
        return false;
    };
    let scope_active = crate::layers::qwen4_hyper::batched_verify_tc_dense_scope_active();
    let (m_cap, k_cap) = nvfp4_scratch_caps(m, k, scope_active);
    map.values().any(|s| s.covers(m_cap, k_cap))
}

/// Whether `gpu`'s initialized NVFP4 activation scratch already covers `(m, k)`
/// without allocating GPU memory (CUDA-graph capture safe).
#[must_use]
pub(crate) fn scratch_ready_for_capture_for_gpu(gpu: &dyn GpuBackend, m: usize, k: usize) -> bool {
    let Ok(guard) = NVFP4_SCRATCH.lock() else {
        return false;
    };
    let Some(map) = guard.as_ref() else {
        return false;
    };
    let scope_active = crate::layers::qwen4_hyper::batched_verify_tc_dense_scope_active();
    let (m_cap, k_cap) = nvfp4_scratch_caps(m, k, scope_active);
    map.get(&gpu_backend_key(gpu))
        .is_some_and(|s| s.covers(m_cap, k_cap))
}

fn ensure_act_scratch(
    gpu: &dyn GpuBackend,
    m: usize,
    k: usize,
    scope_active: bool,
) -> Result<Option<Nvfp4ActScratch>> {
    let key = gpu_backend_key(gpu);
    let (m_cap, k_cap) = nvfp4_scratch_caps(m, k, scope_active);
    let need_packed = m_cap * (k_cap / 2);
    let need_scale = block_scale_padded_bytes(m_cap as u32, k_cap as u32, NVFP4_BLOCK_SIZE);

    let mut guard = NVFP4_SCRATCH
        .lock()
        .expect("qwen4_fast_proj_nvfp4 scratch lock poisoned");
    let map = guard.get_or_insert_with(HashMap::new);
    let mut cur = map
        .get(&key)
        .copied()
        .unwrap_or_else(Nvfp4ActScratch::empty);
    if cur.covers(m_cap, k_cap) {
        return Ok(Some(cur));
    }
    if spark_runtime::gpu_capture_region::active_capture_region().is_some() {
        return Ok(None);
    }
    if cur.meta.is_null() {
        cur.meta = gpu
            .alloc(256)
            .with_context(|| format!("{NVFP4_SELECTOR}: alloc 256 B scalar metadata"))?;
        spark_runtime::scratch_init::init_grown(gpu, cur.meta, 256, "nvfp4_proj_meta")?;
    }
    if cur.a_packed.1 < need_packed {
        let cap = need_packed.next_multiple_of(1 << 20);
        let ptr = gpu.alloc(cap).with_context(|| {
            format!("{NVFP4_SELECTOR}: alloc {cap} B activation packed scratch")
        })?;
        spark_runtime::scratch_init::init_grown(gpu, ptr, cap, "nvfp4_proj_act")?;
        cur.a_packed = (ptr, cap);
    }
    if cur.a_scale.1 < need_scale {
        let cap = need_scale.next_multiple_of(1 << 20);
        let ptr = gpu
            .alloc(cap)
            .with_context(|| format!("{NVFP4_SELECTOR}: alloc {cap} B activation scale scratch"))?;
        spark_runtime::scratch_init::init_grown(gpu, ptr, cap, "nvfp4_proj_act_scale")?;
        cur.a_scale = (ptr, cap);
    }
    map.insert(key, cur);
    Ok(Some(cur))
}

fn ensure_ba_f32_scratch(
    gpu: &dyn GpuBackend,
    rows: usize,
    gates_dim: usize,
) -> Result<Option<DevicePtr>> {
    let key = gpu_backend_key(gpu);
    let need = rows * gates_dim * std::mem::size_of::<f32>();
    let mut guard = NVFP4_SCRATCH
        .lock()
        .expect("qwen4_fast_proj_nvfp4 scratch lock poisoned");
    let map = guard.get_or_insert_with(HashMap::new);
    let cur = map.entry(key).or_insert_with(Nvfp4ActScratch::empty);
    if cur.ba_f32.1 >= need {
        return Ok(Some(cur.ba_f32.0));
    }
    if spark_runtime::gpu_capture_region::active_capture_region().is_some() {
        return Ok(None);
    }
    let cap = need.next_multiple_of(1 << 20);
    let ptr = gpu
        .alloc(cap)
        .with_context(|| format!("{NVFP4_SELECTOR}: alloc {cap} B fused BA scratch"))?;
    spark_runtime::scratch_init::init_grown(gpu, ptr, cap, "nvfp4_proj_ba_f32")?;
    cur.ba_f32 = (ptr, cap);
    Ok(Some(ptr))
}

/// Pre-load NVFP4 activation quantization kernels and pre-allocate the
/// batched-verify activation scratch (`m_cap = 256, k_cap = 6144`) before
/// CUDA graph capture starts.
pub(crate) fn warm_kernels_and_scratch(gpu: &dyn GpuBackend) {
    if !mode_or_off().is_enabled() {
        return;
    }
    if quant_kernels(gpu).is_some() {
        let _ = ensure_act_scratch(gpu, VERIFY_SCRATCH_MIN_M, SCRATCH_MIN_K, true);
    }
}

/// Pre-allocate batched-verify activation scratch covering `(m, k)` and
/// pre-swizzle `weight` into `NVFP4_WEIGHT_CACHE` strictly outside CUDA graph
/// capture so subsequent captured replays execute with zero GPU allocations.
pub(crate) fn prewarm_weight_and_scratch(
    gpu: &dyn GpuBackend,
    weight: &QuantizedWeight,
    m: usize,
    n: usize,
    k: usize,
    qg: Option<(usize, usize)>,
    stream: u64,
) -> Result<bool> {
    let active_mode = mode()?;
    if !active_mode.admits_weight(n, k, qg)
        || m == 0
        || weight.weight.is_null()
        || weight.weight_scale.is_null()
        || !weight.weight_scale_2.is_finite()
        || !weight.weight.0.is_multiple_of(16)
    {
        return Ok(false);
    }
    let Some(ks) = quant_kernels(gpu) else {
        return Ok(false);
    };
    let Some(_scratch) = ensure_act_scratch(gpu, m, k, true)? else {
        return Ok(false);
    };
    let Some(_cached_w) = get_or_prepare_weight(gpu, ks, weight, n, k, qg, stream)? else {
        return Ok(false);
    };
    Ok(true)
}

/// Cached one-time `128x4`-swizzled NVFP4 weight representation.
///
/// For unpermuted weights (`qg == None`), `packed` borrows `weight.weight` directly
/// (zero weight bytes duplicated; only the `128x4` UE4M3 scale buffer is allocated).
/// For gated-Q weights (`qg == Some((heads, head_dim))`), `packed` holds a one-time
/// row-permuted `[Q_all | G_all]` packed E2M1 copy co-allocated with its swizzled scales.
#[derive(Clone, Copy, Debug)]
struct CachedNvfp4Weight {
    packed: DevicePtr,
    scale_swizzled: DevicePtr,
    #[allow(dead_code)]
    bytes: usize,
}

type WeightIdentityKey = (u64, u64, u32, usize, usize, usize, usize);
type ScopedWeightKey = (usize, WeightIdentityKey);

struct Nvfp4WeightCache {
    entries: HashMap<ScopedWeightKey, CachedNvfp4Weight>,
    present_weights: HashSet<WeightIdentityKey>,
    bytes: usize,
    cached: usize,
}

static NVFP4_WEIGHT_CACHE: Mutex<Option<Nvfp4WeightCache>> = Mutex::new(None);

#[inline]
fn weight_identity_key(
    weight: &QuantizedWeight,
    n: usize,
    k: usize,
    qg: Option<(usize, usize)>,
) -> WeightIdentityKey {
    (
        weight.weight.0,
        weight.weight_scale.0,
        weight.weight_scale_2.to_bits(),
        n,
        k,
        qg.map_or(0, |(h, _)| h),
        qg.map_or(0, |(_, d)| d),
    )
}

/// Returns `true` when `weight` already has cached `128x4`-swizzled NVFP4 scales
/// (and permuted packed weights when `qg.is_some()`).
#[cfg(test)]
#[must_use]
pub(crate) fn has_cached_weight(
    weight: &QuantizedWeight,
    n: usize,
    k: usize,
    qg: Option<(usize, usize)>,
) -> bool {
    let Ok(guard) = NVFP4_WEIGHT_CACHE.lock() else {
        return false;
    };
    let Some(cache) = guard.as_ref() else {
        return false;
    };
    let id = weight_identity_key(weight, n, k, qg);
    cache.present_weights.contains(&id)
}

/// Returns `true` when `weight` is already cached in `NVFP4_WEIGHT_CACHE` for
/// `gpu` under `(n, k, qg)` (exact match with [`get_or_prepare_weight`], so
/// CUDA graph capture is guaranteed zero-allocation and zero-swizzle).
#[must_use]
pub(crate) fn has_cached_weight_for_gpu(
    gpu: &dyn GpuBackend,
    weight: &QuantizedWeight,
    n: usize,
    k: usize,
    qg: Option<(usize, usize)>,
) -> bool {
    let Ok(guard) = NVFP4_WEIGHT_CACHE.lock() else {
        return false;
    };
    let Some(cache) = guard.as_ref() else {
        return false;
    };
    let scoped = (gpu_backend_key(gpu), weight_identity_key(weight, n, k, qg));
    cache.entries.contains_key(&scoped)
}

#[allow(clippy::too_many_arguments)]
fn get_or_prepare_weight(
    gpu: &dyn GpuBackend,
    ks: Nvfp4QuantKernels,
    weight: &QuantizedWeight,
    n: usize,
    k: usize,
    qg: Option<(usize, usize)>,
    stream: u64,
) -> Result<Option<CachedNvfp4Weight>> {
    let id = weight_identity_key(weight, n, k, qg);
    let scoped = (gpu_backend_key(gpu), id);

    {
        let guard = NVFP4_WEIGHT_CACHE
            .lock()
            .expect("qwen4_fast_proj_nvfp4 weight cache lock poisoned");
        if let Some(cache) = guard.as_ref()
            && let Some(&hit) = cache.entries.get(&scoped)
        {
            return Ok(Some(hit));
        }
    }

    if spark_runtime::gpu_capture_region::active_capture_region().is_some() {
        return Ok(None);
    }

    let (qg_heads, qg_head_dim) = qg.unwrap_or((0, 0));
    let scale_bytes = block_scale_padded_bytes(n as u32, k as u32, NVFP4_BLOCK_SIZE);
    let need_packed_copy = qg_heads > 0;
    let packed_padded_bytes = if need_packed_copy {
        (n * (k / 2)).next_multiple_of(256)
    } else {
        0
    };
    let alloc_bytes = packed_padded_bytes + scale_bytes.next_multiple_of(256);

    let base = gpu
        .alloc(alloc_bytes)
        .with_context(|| format!("{NVFP4_SELECTOR}: alloc {alloc_bytes} B swizzled weight"))?;
    let (packed_dst, scale_dst, packed_arg) = if need_packed_copy {
        (base, base.offset(packed_padded_bytes), base)
    } else {
        (weight.weight, base, DevicePtr::NULL)
    };

    let groups = (k / 16) as u32;
    let padded_groups = groups.next_multiple_of(4);
    let padded_n = (n as u32).next_multiple_of(128);
    if let Err(err) = KernelLaunch::new(gpu, ks.swizzle_weight)
        .grid([
            div_ceil(padded_groups, 128),
            padded_n.min(MAX_QUANT_GRID_Y),
            1,
        ])
        .block([128, 1, 1])
        .arg_ptr(if need_packed_copy {
            weight.weight
        } else {
            DevicePtr::NULL
        })
        .arg_ptr(weight.weight_scale)
        .arg_ptr(packed_arg)
        .arg_ptr(scale_dst)
        .arg_u32(n as u32)
        .arg_u32(k as u32)
        .arg_u32(qg_heads as u32)
        .arg_u32(qg_head_dim as u32)
        .launch(stream)
    {
        let _ = gpu.free(base);
        return Err(err);
    }

    let entry = CachedNvfp4Weight {
        packed: packed_dst,
        scale_swizzled: scale_dst,
        bytes: alloc_bytes,
    };

    let mut guard = NVFP4_WEIGHT_CACHE
        .lock()
        .expect("qwen4_fast_proj_nvfp4 weight cache lock poisoned");
    let cache = guard.get_or_insert_with(|| Nvfp4WeightCache {
        entries: HashMap::new(),
        present_weights: HashSet::new(),
        bytes: 0,
        cached: 0,
    });
    if let Some(&existing) = cache.entries.get(&scoped) {
        drop(guard);
        gpu.synchronize(stream)?;
        let _ = gpu.free(base);
        return Ok(Some(existing));
    }
    cache.entries.insert(scoped, entry);
    cache.present_weights.insert(id);
    cache.bytes += alloc_bytes;
    cache.cached += 1;
    let count = cache.cached;
    if count == 1 || count.is_multiple_of(32) {
        tracing::info!(
            "FAST_PROJ_NVFP4_WEIGHT_CACHE selector={NVFP4_SELECTOR} weights={count} bytes={} resident_mib={}",
            cache.bytes,
            cache.bytes >> 20,
        );
    }
    Ok(Some(entry))
}

#[inline]
fn nvfp4_ptrs_valid(act: DevicePtr, weight: &QuantizedWeight, out: DevicePtr) -> bool {
    !act.is_null()
        && !out.is_null()
        && !weight.weight.is_null()
        && !weight.weight_scale.is_null()
        && weight.weight_scale_2.is_finite()
        && act.0.is_multiple_of(16)
        && weight.weight.0.is_multiple_of(16)
        && out.0.is_multiple_of(16)
}

fn launch_act_amax_bf16(
    gpu: &dyn GpuBackend,
    ks: Nvfp4QuantKernels,
    act: DevicePtr,
    a_amax: DevicePtr,
    m: usize,
    k: usize,
    stream: u64,
) -> Result<()> {
    gpu.memset_async(a_amax, 0, 4, stream)?;
    let count = (m * k) as u64;
    let blocks = div_ceil((count / 8) as u32, 256).clamp(1, 256);
    KernelLaunch::new(gpu, ks.act_amax)
        .grid([blocks, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(act)
        .arg_u64(count)
        .arg_ptr(a_amax)
        .launch(stream)
}

#[allow(clippy::too_many_arguments)]
fn launch_quant_bf16_128x4(
    gpu: &dyn GpuBackend,
    ks: Nvfp4QuantKernels,
    act: DevicePtr,
    a_amax: DevicePtr,
    scratch: Nvfp4ActScratch,
    alpha_out: DevicePtr,
    weight_scale_2: f32,
    m: usize,
    k: usize,
    stream: u64,
) -> Result<()> {
    let groups = (k / 16) as u32;
    let padded_groups = groups.next_multiple_of(4);
    let padded_rows = (m as u32).next_multiple_of(128);
    KernelLaunch::new(gpu, ks.quant_bf16)
        .grid([
            div_ceil(padded_groups, 128),
            padded_rows.min(MAX_QUANT_GRID_Y),
            1,
        ])
        .block([128, 1, 1])
        .arg_ptr(act)
        .arg_ptr(a_amax)
        .arg_ptr(scratch.a_packed.0)
        .arg_ptr(scratch.a_scale.0)
        .arg_ptr(scratch.a_scale2())
        .arg_ptr(alpha_out)
        .arg_f32(weight_scale_2)
        .arg_u32(m as u32)
        .arg_u32(k as u32)
        .launch(stream)
}

#[allow(clippy::too_many_arguments)]
fn launch_quant_e4m3_128x4(
    gpu: &dyn GpuBackend,
    ks: Nvfp4QuantKernels,
    a8: DevicePtr,
    a_scale: DevicePtr,
    scratch: Nvfp4ActScratch,
    alpha_out: DevicePtr,
    weight_scale_2: f32,
    m: usize,
    k: usize,
    stream: u64,
) -> Result<()> {
    let groups = (k / 16) as u32;
    let padded_groups = groups.next_multiple_of(4);
    let padded_rows = (m as u32).next_multiple_of(128);
    KernelLaunch::new(gpu, ks.quant_e4m3)
        .grid([
            div_ceil(padded_groups, 128),
            padded_rows.min(MAX_QUANT_GRID_Y),
            1,
        ])
        .block([128, 1, 1])
        .arg_ptr(a8)
        .arg_ptr(a_scale)
        .arg_ptr(scratch.a_packed.0)
        .arg_ptr(scratch.a_scale.0)
        .arg_ptr(scratch.a_scale2())
        .arg_ptr(alpha_out)
        .arg_f32(weight_scale_2)
        .arg_u32(m as u32)
        .arg_u32(k as u32)
        .launch(stream)
}

/// Core BF16 -> NVFP4 activation quantization + native cuBLASLt NVFP4 GEMM
/// bypassing the process-wide env selector (used by both production dispatch
/// and live GPU parity tests).
#[allow(clippy::too_many_arguments)]
fn run_bf16_gemm_unchecked(
    gpu: &dyn GpuBackend,
    act: DevicePtr,
    precomputed_amax: Option<DevicePtr>,
    weight: &QuantizedWeight,
    out: DevicePtr,
    m: usize,
    n: usize,
    k: usize,
    ldc: usize,
    qg: Option<(usize, usize)>,
    scope_active: bool,
    stream: u64,
) -> Result<bool> {
    if !nvfp4_ptrs_valid(act, weight, out)
        || m == 0
        || n == 0
        || k == 0
        || ldc < n
        || !k.is_multiple_of(32)
        || !n.is_multiple_of(8)
        || !ldc.is_multiple_of(8)
    {
        return Ok(false);
    }
    let Some(ks) = quant_kernels(gpu) else {
        return Ok(false);
    };
    let Some(scratch) = ensure_act_scratch(gpu, m, k, scope_active)? else {
        return Ok(false);
    };
    let Some(cached_w) = get_or_prepare_weight(gpu, ks, weight, n, k, qg, stream)? else {
        return Ok(false);
    };

    let a_amax = match precomputed_amax {
        Some(ptr) => ptr,
        None if crate::layers::qwen4_fp8_act_amax::take_matching(act, m, k, stream) => {
            crate::layers::qwen4_fp8_act_amax::slot(gpu)?
        }
        None => {
            let slot = scratch.a_amax();
            launch_act_amax_bf16(gpu, ks, act, slot, m, k, stream)?;
            slot
        }
    };

    let d_alpha = scratch.alpha(0);
    launch_quant_bf16_128x4(
        gpu,
        ks,
        act,
        a_amax,
        scratch,
        d_alpha,
        weight.weight_scale_2,
        m,
        k,
        stream,
    )?;

    cublaslt::nvfp4_gemm_block_scaled(
        scratch.a_packed.0.0,
        scratch.a_scale.0.0,
        cached_w.packed.0,
        cached_w.scale_swizzled.0,
        out.0,
        u32::try_from(m)?,
        u32::try_from(n)?,
        u32::try_from(k)?,
        u32::try_from(ldc)?,
        BlockScaleScalars::device_alpha(d_alpha.0),
        stream,
    )?;
    Ok(true)
}

/// Core FP8 E4M3 -> NVFP4 requantization + native cuBLASLt NVFP4 GEMM
/// bypassing the process-wide env selector.
#[allow(clippy::too_many_arguments)]
fn run_e4m3_gemm_unchecked(
    gpu: &dyn GpuBackend,
    a8: DevicePtr,
    a_scale: DevicePtr,
    weight: &QuantizedWeight,
    out: DevicePtr,
    m: usize,
    n: usize,
    k: usize,
    ldc: usize,
    qg: Option<(usize, usize)>,
    scope_active: bool,
    stream: u64,
) -> Result<bool> {
    if !nvfp4_ptrs_valid(a8, weight, out)
        || a_scale.is_null()
        || !a_scale.0.is_multiple_of(4)
        || m == 0
        || n == 0
        || k == 0
        || ldc < n
        || !k.is_multiple_of(32)
        || !n.is_multiple_of(8)
        || !ldc.is_multiple_of(8)
    {
        return Ok(false);
    }
    let Some(ks) = quant_kernels(gpu) else {
        return Ok(false);
    };
    let Some(scratch) = ensure_act_scratch(gpu, m, k, scope_active)? else {
        return Ok(false);
    };
    let Some(cached_w) = get_or_prepare_weight(gpu, ks, weight, n, k, qg, stream)? else {
        return Ok(false);
    };

    let d_alpha = scratch.alpha(0);
    launch_quant_e4m3_128x4(
        gpu,
        ks,
        a8,
        a_scale,
        scratch,
        d_alpha,
        weight.weight_scale_2,
        m,
        k,
        stream,
    )?;

    cublaslt::nvfp4_gemm_block_scaled(
        scratch.a_packed.0.0,
        scratch.a_scale.0.0,
        cached_w.packed.0,
        cached_w.scale_swizzled.0,
        out.0,
        u32::try_from(m)?,
        u32::try_from(n)?,
        u32::try_from(k)?,
        u32::try_from(ldc)?,
        BlockScaleScalars::device_alpha(d_alpha.0),
        stream,
    )?;
    Ok(true)
}

/// Try dispatching one BF16-activation projection via native cuBLASLt NVFP4.
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_gemm_bf16(
    gpu: &dyn GpuBackend,
    act: DevicePtr,
    weight: &QuantizedWeight,
    out: DevicePtr,
    m: usize,
    n: usize,
    k: usize,
    ldc: usize,
    qg: Option<(usize, usize)>,
    scope_active: bool,
    stream: u64,
) -> Result<bool> {
    let active_mode = mode()?;
    if !active_mode.admits_call(m, n, k, ldc, qg, scope_active, verify_min_m()) {
        return Ok(false);
    }
    let ran = run_bf16_gemm_unchecked(
        gpu,
        act,
        None,
        weight,
        out,
        m,
        n,
        k,
        ldc,
        qg,
        scope_active,
        stream,
    )?;
    if ran {
        static LOGGED: std::sync::Once = std::sync::Once::new();
        LOGGED.call_once(|| {
            tracing::info!(
                selector = NVFP4_SELECTOR,
                mode = active_mode.label(),
                m,
                n,
                k,
                ldc,
                scope_active,
                "FAST_PROJ_NVFP4_ENGAGED"
            );
        });
    }
    Ok(ran)
}

/// Try dispatching one BF16-activation projection using an already-computed device
/// `a_amax` slot (used for mixed `wide` shared-act Q/K/V where FP8 already reduced
/// `act` absmax).
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_gemm_bf16_with_amax(
    gpu: &dyn GpuBackend,
    act: DevicePtr,
    a_amax: DevicePtr,
    weight: &QuantizedWeight,
    out: DevicePtr,
    m: usize,
    n: usize,
    k: usize,
    ldc: usize,
    qg: Option<(usize, usize)>,
    scope_active: bool,
    stream: u64,
) -> Result<bool> {
    let active_mode = mode()?;
    if !active_mode.admits_call(m, n, k, ldc, qg, scope_active, verify_min_m()) {
        return Ok(false);
    }
    let ran = run_bf16_gemm_unchecked(
        gpu,
        act,
        Some(a_amax),
        weight,
        out,
        m,
        n,
        k,
        ldc,
        qg,
        scope_active,
        stream,
    )?;
    if ran {
        static LOGGED: std::sync::Once = std::sync::Once::new();
        LOGGED.call_once(|| {
            tracing::info!(
                selector = NVFP4_SELECTOR,
                mode = active_mode.label(),
                m,
                n,
                k,
                ldc,
                scope_active,
                "FAST_PROJ_NVFP4_ENGAGED"
            );
        });
    }
    Ok(ran)
}

/// Try upgrading a batched-verify `try_coalesced_nvfp4_w4a16_gemm` call (`m >= verify_min_m`,
/// wide dense projections `ssm.qkvz`, `ssm.out`, `attn.q_proj`, `attn.o_proj`) to native
/// cuBLASLt NVFP4.
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_gemm_coalesced_verify(
    gpu: &dyn GpuBackend,
    act: DevicePtr,
    weight: &QuantizedWeight,
    out: DevicePtr,
    m: usize,
    n: usize,
    k: usize,
    stream: u64,
) -> Result<bool> {
    let active_mode = mode()?;
    let scope_active = crate::layers::qwen4_hyper::batched_verify_tc_dense_scope_active();
    if !active_mode.admits_coalesced_verify(scope_active, m, n, k, verify_min_m()) {
        return Ok(false);
    }
    let ran = run_bf16_gemm_unchecked(gpu, act, None, weight, out, m, n, k, n, None, true, stream)?;
    if ran {
        static LOGGED: std::sync::Once = std::sync::Once::new();
        LOGGED.call_once(|| {
            tracing::info!(
                selector = NVFP4_SELECTOR,
                mode = active_mode.label(),
                m,
                n,
                k,
                "FAST_PROJ_NVFP4_VERIFY_ENGAGED"
            );
        });
    }
    Ok(ran)
}

/// Try dispatching one pre-quantized FP8 E4M3 activation (`a8` + device `a_scale`)
/// via single-pass E4M3 -> NVFP4 requantization + native cuBLASLt NVFP4 GEMM.
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_gemm_from_e4m3(
    gpu: &dyn GpuBackend,
    a8: DevicePtr,
    a_scale: DevicePtr,
    weight: &QuantizedWeight,
    out: DevicePtr,
    m: usize,
    n: usize,
    k: usize,
    ldc: usize,
    qg: Option<(usize, usize)>,
    scope_active: bool,
    stream: u64,
) -> Result<bool> {
    let active_mode = mode()?;
    if !active_mode.admits_call(m, n, k, ldc, qg, scope_active, verify_min_m()) {
        return Ok(false);
    }
    let ran = run_e4m3_gemm_unchecked(
        gpu,
        a8,
        a_scale,
        weight,
        out,
        m,
        n,
        k,
        ldc,
        qg,
        scope_active,
        stream,
    )?;
    if ran {
        static LOGGED: std::sync::Once = std::sync::Once::new();
        LOGGED.call_once(|| {
            tracing::info!(
                selector = NVFP4_SELECTOR,
                mode = active_mode.label(),
                m,
                n,
                k,
                ldc,
                scope_active,
                "FAST_PROJ_NVFP4_E4M3_ENGAGED"
            );
        });
    }
    Ok(ran)
}

fn run_shared_act_all_unchecked(
    gpu: &dyn GpuBackend,
    act: DevicePtr,
    outs: &[super::SharedActOut<'_>],
    m: usize,
    k: usize,
    ldc: usize,
    scope_active: bool,
    stream: u64,
) -> Result<SharedActNvfp4Outcome> {
    if outs.is_empty()
        || outs.len() > 4
        || m == 0
        || k == 0
        || !k.is_multiple_of(32)
        || !ldc.is_multiple_of(8)
        || !outs.iter().all(|o| {
            o.n > 0 && o.n.is_multiple_of(8) && ldc >= o.n && nvfp4_ptrs_valid(act, o.weight, o.out)
        })
    {
        return Ok(SharedActNvfp4Outcome::NotHandled);
    }
    let Some(ks) = quant_kernels(gpu) else {
        return Ok(SharedActNvfp4Outcome::NotHandled);
    };
    let Some(scratch) = ensure_act_scratch(gpu, m, k, scope_active)? else {
        return Ok(SharedActNvfp4Outcome::NotHandled);
    };

    // Prepare every weight BEFORE consuming any fused `act_amax` hand-off so
    // a capture-time refusal never drops the producer absmax record.
    let mut cached_weights = [CachedNvfp4Weight {
        packed: DevicePtr::NULL,
        scale_swizzled: DevicePtr::NULL,
        bytes: 0,
    }; 4];
    let mut ws2 = [0.0f32; 4];
    for (i, o) in outs.iter().enumerate() {
        let Some(cw) = get_or_prepare_weight(gpu, ks, o.weight, o.n, k, o.qg, stream)? else {
            return Ok(SharedActNvfp4Outcome::NotHandled);
        };
        cached_weights[i] = cw;
        ws2[i] = o.weight.weight_scale_2;
    }

    let a_amax = if crate::layers::qwen4_fp8_act_amax::take_matching(act, m, k, stream) {
        crate::layers::qwen4_fp8_act_amax::slot(gpu)?
    } else {
        let slot = scratch.a_amax();
        launch_act_amax_bf16(gpu, ks, act, slot, m, k, stream)?;
        slot
    };

    launch_quant_bf16_128x4(
        gpu,
        ks,
        act,
        a_amax,
        scratch,
        DevicePtr::NULL,
        1.0,
        m,
        k,
        stream,
    )?;

    KernelLaunch::new(gpu, ks.compute_alphas4)
        .grid([1, 1, 1])
        .block([32, 1, 1])
        .arg_ptr(scratch.a_scale2())
        .arg_f32(ws2[0])
        .arg_f32(ws2[1])
        .arg_f32(ws2[2])
        .arg_f32(ws2[3])
        .arg_u32(outs.len() as u32)
        .arg_ptr(scratch.alpha(0))
        .launch(stream)?;

    let m_u32 = u32::try_from(m)?;
    let k_u32 = u32::try_from(k)?;
    let ldc_u32 = u32::try_from(ldc)?;
    for (i, o) in outs.iter().enumerate() {
        cublaslt::nvfp4_gemm_block_scaled(
            scratch.a_packed.0.0,
            scratch.a_scale.0.0,
            cached_weights[i].packed.0,
            cached_weights[i].scale_swizzled.0,
            o.out.0,
            m_u32,
            u32::try_from(o.n)?,
            k_u32,
            ldc_u32,
            BlockScaleScalars::device_alpha(scratch.alpha(i).0),
            stream,
        )?;
    }

    Ok(SharedActNvfp4Outcome::HandledAll)
}

/// Try dispatching all outputs of [`super::fp8_gemm_shared_act`] via native cuBLASLt
/// NVFP4 with a single BF16 -> NVFP4 activation quantization.
///
/// Returns [`SharedActNvfp4Outcome::HandledAll`] only when EVERY output in `outs`
/// is admitted by the active mode (e.g. `mode == All` for `[K, V, Q]` or `[K, V]`).
/// Mixed lists (e.g. `mode == Wide` for `[K, V, Q]`) return `NotHandled` so the
/// caller runs one shared `a_amax` pass and dispatches wide outputs via
/// [`try_gemm_bf16_with_amax`].
pub(crate) fn try_gemm_shared_act_all(
    gpu: &dyn GpuBackend,
    act: DevicePtr,
    outs: &[super::SharedActOut<'_>],
    m: usize,
    k: usize,
    ldc: usize,
    stream: u64,
) -> Result<SharedActNvfp4Outcome> {
    let active_mode = mode()?;
    if !active_mode.is_enabled() || outs.is_empty() || outs.len() > 4 {
        return Ok(SharedActNvfp4Outcome::NotHandled);
    }
    let scope_active = crate::layers::qwen4_hyper::batched_verify_tc_dense_scope_active();
    let min_m = verify_min_m();
    if !outs.iter().all(|o| {
        active_mode.admits_call(m, o.n, k, ldc, o.qg, scope_active, min_m)
            && nvfp4_ptrs_valid(act, o.weight, o.out)
    }) {
        return Ok(SharedActNvfp4Outcome::NotHandled);
    }
    let outcome = run_shared_act_all_unchecked(gpu, act, outs, m, k, ldc, scope_active, stream)?;
    if outcome == SharedActNvfp4Outcome::HandledAll {
        static LOGGED: std::sync::Once = std::sync::Once::new();
        LOGGED.call_once(|| {
            tracing::info!(
                selector = NVFP4_SELECTOR,
                mode = active_mode.label(),
                m,
                k,
                ldc,
                outputs = outs.len(),
                "FAST_PROJ_NVFP4_SHARED_ACT_ENGAGED"
            );
        });
    }
    Ok(outcome)
}

fn ba_gates_kernel(gpu: &dyn GpuBackend, planar: bool) -> Option<KernelHandle> {
    if planar {
        let h = crate::layers::try_kernel(gpu, "qwen4_wide_glue", "qwen4_ba_gates_from_f32_planar");
        (h.0 != 0).then_some(h)
    } else {
        let v2 = crate::layers::try_kernel(gpu, "qwen4_wide_glue", "qwen4_ba_gates_from_f32_v2");
        if v2.0 != 0 {
            Some(v2)
        } else {
            let v1 = crate::layers::try_kernel(gpu, "qwen4_wide_glue", "qwen4_ba_gates_from_f32");
            (v1.0 != 0).then_some(v1)
        }
    }
}

/// Try dispatching [`super::fp8_qkvz_and_ba_gates_fused`] via native cuBLASLt NVFP4
/// for `qkvz_weight` (`16384 x 2560`, saving `1,440 MiB` of FP8 weight cache across
/// 36 SSM layers) paired with cuBLASLt BF16 -> FP32 BA projection + GDN gate transform.
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_qkvz_and_ba_gates_fused(
    gpu: &dyn GpuBackend,
    act: DevicePtr,
    qkvz_weight: &QuantizedWeight,
    qkvz_out: DevicePtr,
    ba_weight: DevicePtr,
    a_log: DevicePtr,
    dt_bias: DevicePtr,
    gates_out: DevicePtr,
    m: usize,
    n_qkvz: usize,
    k: usize,
    planar: bool,
    stream: u64,
) -> Result<bool> {
    let active_mode = mode()?;
    if !active_mode.admits_call(m, n_qkvz, k, n_qkvz, None, false, verify_min_m())
        || !nvfp4_ptrs_valid(act, qkvz_weight, qkvz_out)
        || ba_weight.is_null()
        || !ba_weight.0.is_multiple_of(16)
    {
        return Ok(false);
    }
    let Some(ba_k) = ba_gates_kernel(gpu, planar) else {
        return Ok(false);
    };
    const BA_GATES_DIM: usize = 96;
    const VHEADS_PER_GROUP: u32 = 3;
    let Some(ba_buf) = ensure_ba_f32_scratch(gpu, m, BA_GATES_DIM)? else {
        return Ok(false);
    };

    if !run_bf16_gemm_unchecked(
        gpu,
        act,
        None,
        qkvz_weight,
        qkvz_out,
        m,
        n_qkvz,
        k,
        n_qkvz,
        None,
        false,
        stream,
    )? {
        return Ok(false);
    }

    let m_u32 = u32::try_from(m)?;
    let k_u32 = u32::try_from(k)?;
    cublaslt::gemm_act_weight_t_typed_pinned(
        act.0,
        k_u32,
        ba_weight.0,
        ba_buf.0,
        BA_GATES_DIM as u32,
        m_u32,
        BA_GATES_DIM as u32,
        k_u32,
        GemmDtype::Bf16,
        GemmDtype::F32,
        false,
        m_u32,
        stream,
    )?;
    let last_arg = if planar { m_u32 } else { BA_GATES_DIM as u32 };
    KernelLaunch::new(gpu, ba_k)
        .grid([m_u32, 1, 1])
        .block([BA_GATES_DIM as u32, 1, 1])
        .arg_ptr(ba_buf)
        .arg_ptr(a_log)
        .arg_ptr(dt_bias)
        .arg_ptr(gates_out)
        .arg_u32(BA_GATES_DIM as u32)
        .arg_u32(VHEADS_PER_GROUP)
        .arg_u32(last_arg)
        .launch(stream)?;

    static LOGGED: std::sync::Once = std::sync::Once::new();
    LOGGED.call_once(|| {
        tracing::info!(
            selector = NVFP4_SELECTOR,
            mode = active_mode.label(),
            m,
            n_qkvz,
            k,
            planar,
            "FAST_PROJ_NVFP4_QKVZ_BA_ENGAGED"
        );
    });
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_selector_modes_and_min_m_cleanly() {
        // GB10 (<= 64 SMs): unset / "" / "auto" stay Off even on CC 12.1 (100% unchanged).
        for gb10_sms in [0u32, 48, 64] {
            for cc in [None, Some((12, 1)), Some((12, 0))] {
                assert_eq!(
                    parse_mode_for_device(None, gb10_sms, cc).unwrap(),
                    Nvfp4FastProjMode::Off,
                    "gb10_sms={gb10_sms} cc={cc:?}"
                );
                assert_eq!(
                    parse_mode_for_device(Some(""), gb10_sms, cc).unwrap(),
                    Nvfp4FastProjMode::Off
                );
                assert_eq!(
                    parse_mode_for_device(Some("auto"), gb10_sms, cc).unwrap(),
                    Nvfp4FastProjMode::Off
                );
            }
            assert_eq!(
                parse_mode_for_device(Some("0"), gb10_sms, Some((12, 1))).unwrap(),
                Nvfp4FastProjMode::Off
            );
            assert_eq!(
                parse_mode_for_device(Some("wide"), gb10_sms, Some((12, 1))).unwrap(),
                Nvfp4FastProjMode::Wide
            );
            assert_eq!(
                parse_mode_for_device(Some("1"), gb10_sms, Some((12, 1))).unwrap(),
                Nvfp4FastProjMode::All
            );
        }

        // Pre-Blackwell high-SM GPUs (Hopper CC 9.0, Ada CC 8.9, Ampere CC 8.0) and
        // uninitialized CC (None) MUST stay Off under auto to avoid unsupported FP4 TC crash.
        for (non_blackwell_sms, cc) in [
            (108u32, Some((8, 0))),
            (128, Some((8, 9))),
            (132, Some((9, 0))),
            (188, None),
        ] {
            assert_eq!(
                parse_mode_for_device(None, non_blackwell_sms, cc).unwrap(),
                Nvfp4FastProjMode::Off,
                "non-Blackwell sm={non_blackwell_sms} cc={cc:?} must stay Off"
            );
            assert_eq!(
                parse_mode_for_device(Some("auto"), non_blackwell_sms, cc).unwrap(),
                Nvfp4FastProjMode::Off,
                "non-Blackwell auto sm={non_blackwell_sms} cc={cc:?} must stay Off"
            );
        }

        // SM120 / SM103 Blackwell (> 64 SMs AND CC >= 10.0): unset / "" / "auto" default
        // to All; explicit overrides ("0"/"off"/"false", "wide", "1"/"all"/"on"/"true") win.
        for (high_sms, cc) in [
            (65u32, Some((10, 0))),
            (110, Some((10, 3))),
            (148, Some((10, 3))),
            (170, Some((12, 0))),
            (188, Some((12, 0))),
        ] {
            assert_eq!(
                parse_mode_for_device(None, high_sms, cc).unwrap(),
                Nvfp4FastProjMode::All,
                "high_sms={high_sms} cc={cc:?}"
            );
            assert_eq!(
                parse_mode_for_device(Some(""), high_sms, cc).unwrap(),
                Nvfp4FastProjMode::All
            );
            assert_eq!(
                parse_mode_for_device(Some(" AUTO "), high_sms, cc).unwrap(),
                Nvfp4FastProjMode::All
            );
            assert_eq!(
                parse_mode_for_device(Some("0"), high_sms, cc).unwrap(),
                Nvfp4FastProjMode::Off
            );
            assert_eq!(
                parse_mode_for_device(Some(" off "), high_sms, cc).unwrap(),
                Nvfp4FastProjMode::Off
            );
            assert_eq!(
                parse_mode_for_device(Some("false"), high_sms, cc).unwrap(),
                Nvfp4FastProjMode::Off
            );
            assert_eq!(
                parse_mode_for_device(Some(" WIDE "), high_sms, cc).unwrap(),
                Nvfp4FastProjMode::Wide
            );
            assert_eq!(
                parse_mode_for_device(Some("1"), high_sms, cc).unwrap(),
                Nvfp4FastProjMode::All
            );
            assert_eq!(
                parse_mode_for_device(Some("all"), high_sms, cc).unwrap(),
                Nvfp4FastProjMode::All
            );
            assert_eq!(
                parse_mode_for_device(Some("ON"), high_sms, cc).unwrap(),
                Nvfp4FastProjMode::All
            );
            assert_eq!(
                parse_mode_for_device(Some("true"), high_sms, cc).unwrap(),
                Nvfp4FastProjMode::All
            );
        }
        assert!(parse_mode_for_device(Some("2"), 188, Some((12, 0))).is_err());
        assert!(parse_mode_for_device(Some("fp4"), 48, Some((12, 1))).is_err());

        // SM-aware verify_min_m: 64 on GB10 (<=64 SMs), 32 on SM120/SM103 (>64 SMs).
        assert_eq!(nvfp4_verify_min_m_for_sm(48), DEFAULT_NVFP4_VERIFY_MIN_M);
        assert_eq!(nvfp4_verify_min_m_for_sm(64), DEFAULT_NVFP4_VERIFY_MIN_M);
        assert_eq!(
            nvfp4_verify_min_m_for_sm(65),
            DEFAULT_NVFP4_VERIFY_MIN_M_HIGH_SM
        );
        assert_eq!(
            nvfp4_verify_min_m_for_sm(188),
            DEFAULT_NVFP4_VERIFY_MIN_M_HIGH_SM
        );
        assert_eq!(parse_verify_min_m_for_sm(None, 48), 64);
        assert_eq!(parse_verify_min_m_for_sm(None, 188), 32);
        assert_eq!(parse_verify_min_m_for_sm(Some("0"), 188), 32);
        assert_eq!(parse_verify_min_m_for_sm(Some("64"), 188), 64);
        assert_eq!(parse_verify_min_m(None), DEFAULT_NVFP4_VERIFY_MIN_M);
        assert_eq!(parse_verify_min_m(Some("")), DEFAULT_NVFP4_VERIFY_MIN_M);
        assert_eq!(parse_verify_min_m(Some("0")), DEFAULT_NVFP4_VERIFY_MIN_M);
        assert_eq!(parse_verify_min_m(Some("bad")), DEFAULT_NVFP4_VERIFY_MIN_M);
        assert_eq!(parse_verify_min_m(Some("32")), 32);
        assert_eq!(parse_verify_min_m(Some(" 128 ")), 128);
    }

    #[test]
    fn mode_admits_exact_flash_next_dense_projections_and_excludes_sensitive_shapes() {
        // Off never admits anything.
        assert!(!Nvfp4FastProjMode::Off.admits_weight(16_384, 2_560, None));

        // Wide dense projections: admitted by both Wide and All.
        for (n, k, qg) in [
            (16_384usize, 2_560usize, None),  // ssm.qkvz
            (2_560, 6_144, None),             // ssm.out / attn.o_proj
            (12_288, 2_560, Some((24, 256))), // attn.q_proj (gated-Q permuted)
            (12_288, 2_560, None),            // attn.q_proj (verify coalesced)
            (8_192, 2_560, None),             // TP2 ssm.qkvz
            (2_560, 3_072, None),             // TP2 ssm.out / o_proj
            (6_144, 2_560, Some((12, 256))),  // TP2 attn.q_proj
        ] {
            assert!(
                Nvfp4FastProjMode::Wide.admits_weight(n, k, qg),
                "wide should admit n={n} k={k} qg={qg:?}"
            );
            assert!(
                Nvfp4FastProjMode::All.admits_weight(n, k, qg),
                "all should admit n={n} k={k} qg={qg:?}"
            );
        }

        // Narrow K/V projections (N=512, K=2560, qg=None): All only, excluded in Wide.
        assert!(!Nvfp4FastProjMode::Wide.admits_weight(512, 2_560, None));
        assert!(Nvfp4FastProjMode::All.admits_weight(512, 2_560, None));

        // Excluded in both Wide and All: MoE router sentinel, hyperconnections, lm_head.
        for (n, k, qg) in [
            (512usize, 2_560usize, Some((0, 0))), // MoE router sentinel
            (320, 10_240, None),                  // hc.down
            (10_240, 320, None),                  // hc.up (short K)
            (96_000, 2_560, None),                // mtp_lm_head
            (248_077, 2_560, None),               // lm_head
            (2_560, 6_128, None),                 // k % 32 != 0
            (12_288, 2_560, Some((24, 128))),     // mismatched qg heads*head_dim*2 != n
        ] {
            assert!(
                !Nvfp4FastProjMode::Wide.admits_weight(n, k, qg),
                "wide must reject n={n} k={k} qg={qg:?}"
            );
            assert!(
                !Nvfp4FastProjMode::All.admits_weight(n, k, qg),
                "all must reject n={n} k={k} qg={qg:?}"
            );
        }

        // Prefill admits m >= 1; batched verify enforces verify_min_m and rejects strided < n.
        assert!(Nvfp4FastProjMode::Wide.admits_call(1, 16_384, 2_560, 16_384, None, false, 64));
        assert!(!Nvfp4FastProjMode::Wide.admits_call(32, 16_384, 2_560, 16_384, None, true, 64));
        assert!(Nvfp4FastProjMode::Wide.admits_call(64, 16_384, 2_560, 16_384, None, true, 64));
        assert!(Nvfp4FastProjMode::Wide.admits_call(160, 16_384, 2_560, 16_384, None, true, 64));
        assert!(Nvfp4FastProjMode::Wide.admits_call(256, 2_560, 6_144, 2_560, None, true, 64));
        assert!(Nvfp4FastProjMode::All.admits_call(
            2048,
            12_288,
            2_560,
            13_312,
            Some((24, 256)),
            false,
            64
        ));
        assert!(!Nvfp4FastProjMode::All.admits_call(
            2048,
            12_288,
            2_560,
            12_280,
            Some((24, 256)),
            false,
            64
        ));

        // Coalesced-verify upgrade hits wide dense shapes only (never router/hc/lm_head).
        assert!(Nvfp4FastProjMode::All.admits_coalesced_verify(true, 128, 16_384, 2_560, 64));
        assert!(Nvfp4FastProjMode::All.admits_coalesced_verify(true, 256, 2_560, 6_144, 64));
        assert!(Nvfp4FastProjMode::All.admits_coalesced_verify(true, 160, 12_288, 2_560, 64));
        assert!(!Nvfp4FastProjMode::All.admits_coalesced_verify(false, 128, 16_384, 2_560, 64));
        assert!(!Nvfp4FastProjMode::All.admits_coalesced_verify(true, 40, 16_384, 2_560, 64));
        assert!(!Nvfp4FastProjMode::All.admits_coalesced_verify(true, 128, 512, 2_560, 64));
        assert!(!Nvfp4FastProjMode::All.admits_coalesced_verify(true, 128, 320, 10_240, 64));
        assert!(!Nvfp4FastProjMode::All.admits_coalesced_verify(true, 128, 10_240, 320, 64));
        assert!(!Nvfp4FastProjMode::All.admits_coalesced_verify(true, 128, 96_000, 2_560, 64));
    }

    #[test]
    fn vram_savings_estimate_matches_flash_next_geometry() {
        // Flash-Next 48 layers: 36 SSM (qkvz 16384x2560, out 2560x6144) +
        // 12 Attn (q 12288x2560 permuted, k 512x2560, v 512x2560, o 2560x6144).
        let fp8_cache_bytes = 36 * (16_384 * 2_560 + 2_560 * 6_144)
            + 12 * (12_288 * 2_560 + 512 * 2_560 * 2 + 2_560 * 6_144);
        let nvfp4_scales_bytes = 36
            * (block_scale_padded_bytes(16_384, 2_560, NVFP4_BLOCK_SIZE)
                + block_scale_padded_bytes(2_560, 6_144, NVFP4_BLOCK_SIZE))
            + 12 * (block_scale_padded_bytes(12_288, 2_560, NVFP4_BLOCK_SIZE)
                + 2 * block_scale_padded_bytes(512, 2_560, NVFP4_BLOCK_SIZE)
                + block_scale_padded_bytes(2_560, 6_144, NVFP4_BLOCK_SIZE));
        let q_proj_perm_bytes = 12 * (12_288 * (2_560 / 2));
        let saved_bytes = fp8_cache_bytes - (nvfp4_scales_bytes + q_proj_perm_bytes);
        assert_eq!(fp8_cache_bytes >> 20, 2_550);
        assert_eq!(nvfp4_scales_bytes >> 20, 159);
        assert_eq!(q_proj_perm_bytes >> 20, 180);
        assert_eq!(saved_bytes >> 20, 2_210);
    }

    #[test]
    fn has_cached_weight_and_gpu_scoped_probe_isolate_qg_permutation_keys() {
        let gpu_a = spark_runtime::gpu::mock::MockGpuBackend::new();
        let gpu_b = spark_runtime::gpu::mock::MockGpuBackend::new();
        // Use distinct non-colliding sentinel pointers unique to this unit test.
        let q_weight = QuantizedWeight {
            weight: DevicePtr(0x0000_7f00_0010_0000),
            weight_scale: DevicePtr(0x0000_7f00_0020_0000),
            weight_scale_2: 1.25,
            input_scale: DevicePtr::NULL,
        };
        let qg = Some((24usize, 256usize));

        assert!(!has_cached_weight(&q_weight, 12_288, 2_560, qg));
        assert!(!has_cached_weight_for_gpu(
            &gpu_a, &q_weight, 12_288, 2_560, qg
        ));

        // Insert a gated-Q cached entry on gpu_a (simulating warmup under FAST_PROJ_NVFP4=wide).
        let id = weight_identity_key(&q_weight, 12_288, 2_560, qg);
        let scoped_a = (gpu_backend_key(&gpu_a), id);
        {
            let mut guard = NVFP4_WEIGHT_CACHE.lock().expect("weight cache lock");
            let cache = guard.get_or_insert_with(|| Nvfp4WeightCache {
                entries: HashMap::new(),
                present_weights: HashSet::new(),
                bytes: 0,
                cached: 0,
            });
            cache.entries.insert(
                scoped_a,
                CachedNvfp4Weight {
                    packed: DevicePtr(0x0000_7f00_0030_0000),
                    scale_swizzled: DevicePtr(0x0000_7f00_0040_0000),
                    bytes: 16 * 1024 * 1024,
                },
            );
            cache.present_weights.insert(id);
        }

        // Gated-Q key hits on gpu_a, misses on gpu_b, and misses under `qg = None`.
        assert!(has_cached_weight(&q_weight, 12_288, 2_560, qg));
        assert!(has_cached_weight_for_gpu(
            &gpu_a, &q_weight, 12_288, 2_560, qg
        ));
        assert!(!has_cached_weight_for_gpu(
            &gpu_b, &q_weight, 12_288, 2_560, qg
        ));
        assert!(!has_cached_weight(&q_weight, 12_288, 2_560, None));
        assert!(!has_cached_weight_for_gpu(
            &gpu_a, &q_weight, 12_288, 2_560, None
        ));

        // Clean up global test entry so other tests see a pristine cache.
        if let Some(cache) = NVFP4_WEIGHT_CACHE
            .lock()
            .expect("weight cache lock")
            .as_mut()
        {
            cache.entries.remove(&scoped_a);
            cache.present_weights.remove(&id);
        }
    }

    #[test]
    #[ignore = "requires live CUDA 13.0 Blackwell GPU (GB10 SM121 / SM120)"]
    fn real_gpu_nvfp4_fast_proj_matches_bf16_reference_across_flash_next_shapes() {
        use std::process::Command;

        use spark_runtime::cuda_backend::AtlasCudaBackend;
        use spark_runtime::gpu::GpuBackend;

        // Compile `qwen4_nvfp4_act_quant.cu` to PTX directly so the test works
        // even under `ATLAS_SKIP_BUILD` or multi-target cargo test invocations.
        let manifest_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let cu_path = manifest_dir
            .join("../../kernels/gb10/qwen3.8-flash-next/nvfp4/qwen4_nvfp4_act_quant.cu");
        assert!(cu_path.is_file(), "missing {}", cu_path.display());

        let tmp_ptx = std::env::temp_dir().join(format!(
            "qwen4_nvfp4_act_quant_test_{}.ptx",
            std::process::id()
        ));
        let nvcc = std::path::Path::new("/usr/local/cuda/bin/nvcc");
        let nvcc_bin = if nvcc.is_file() {
            nvcc.as_os_str()
        } else {
            std::ffi::OsStr::new("nvcc")
        };
        let status = Command::new(nvcc_bin)
            .args([
                "-O3",
                "--fmad=false",
                "-arch=sm_121a",
                "-ptx",
                "-o",
                tmp_ptx.to_str().expect("utf8 tmp ptx"),
                cu_path.to_str().expect("utf8 cu path"),
            ])
            .status()
            .expect("invoke nvcc");
        assert!(
            status.success(),
            "nvcc failed to compile {}",
            cu_path.display()
        );
        let ptx_src = std::fs::read_to_string(&tmp_ptx).expect("read compiled ptx");
        let _ = std::fs::remove_file(&tmp_ptx);

        let ptx_static: &'static str = Box::leak(ptx_src.into_boxed_str());
        let gpu = AtlasCudaBackend::new(0, &[(QUANT_MODULE, ptx_static)])
            .expect("init AtlasCudaBackend with qwen4_nvfp4_act_quant");
        let stream = 0u64;

        let decode_e2m1 = |nib: u8| -> f32 {
            const LUT: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
            let mag = LUT[(nib & 0x7) as usize];
            if (nib & 0x8) != 0 { -mag } else { mag }
        };
        let decode_e4m3 = |byte: u8| -> f32 {
            let sign = if (byte & 0x80) != 0 { -1.0f32 } else { 1.0f32 };
            let exp = ((byte >> 3) & 0x0f) as i32;
            let man = (byte & 0x07) as u32;
            if exp == 0 {
                sign * (man as f32) * (1.0f32 / 512.0f32)
            } else if exp == 15 && man == 7 {
                0.0
            } else {
                let f32_bits = (((exp + 120) as u32) << 23) | (man << 20);
                sign * f32::from_bits(f32_bits)
            }
        };

        // Positive UE4M3 scales spanning 0.25..2.0.
        const W_SCALES: [u8; 8] = [0x28, 0x2c, 0x30, 0x34, 0x38, 0x3a, 0x3c, 0x40];

        // Keep weight device allocations alive until test teardown so `cuMemAlloc`
        // never reuses a freed pointer address and hits a stale entry in the
        // process-wide static `NVFP4_WEIGHT_CACHE`.
        let kept_weights = std::cell::RefCell::new(Vec::<DevicePtr>::new());

        let run_bf16_shape = |label: &str,
                              m: usize,
                              n: usize,
                              k: usize,
                              ldc: usize,
                              qg: Option<(usize, usize)>,
                              act_amp: f32,
                              weight_scale_2: f32| {
            let mut state: u64 = 0x517c_c1b7_2722_0a95
                ^ ((m as u64) << 32)
                ^ ((n as u64) << 16)
                ^ (k as u64)
                ^ u64::from(act_amp.to_bits());
            let mut next_u32 = || {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                (state >> 32) as u32
            };

            // Generate smooth BF16 activations (Gaussian-like sum of uniforms).
            let mut act_bf16 = vec![0u16; m * k];
            let mut act_f32 = vec![0.0f32; m * k];
            for idx in 0..(m * k) {
                let u0 = ((next_u32() >> 8) as f32) * (1.0 / 16_777_216.0) - 0.5;
                let u1 = ((next_u32() >> 8) as f32) * (1.0 / 16_777_216.0) - 0.5;
                let v = (u0 + u1) * act_amp;
                let b = half::bf16::from_f32(v);
                act_bf16[idx] = b.to_bits();
                act_f32[idx] = b.to_f32();
            }

            // Generate NVFP4 weights (packed [n, k/2] + flat [n, k/16] UE4M3 scales).
            let mut w_packed = vec![0u8; n * (k / 2)];
            let mut w_scales = vec![0u8; n * (k / 16)];
            for b in &mut w_packed {
                let lo = (next_u32() & 0x0f) as u8;
                let hi = (next_u32() & 0x0f) as u8;
                *b = (hi << 4) | lo;
            }
            for (idx, s) in w_scales.iter_mut().enumerate() {
                *s = W_SCALES[((next_u32() as usize) ^ idx) % W_SCALES.len()];
            }

            // Dequantize weights to BF16 (applying QG permutation when set) so cuBLASLt BF16
            // computes the exact reference in <1 ms even for 2048x16384x2560.
            let mut w_ref_bf16 = vec![0u16; n * k];
            for src_row in 0..n {
                let dst_row = if let Some((heads, head_dim)) = qg {
                    let group_dim = 2 * head_dim;
                    let h = src_row / group_dim;
                    let idx = src_row - h * group_dim;
                    if idx < head_dim {
                        h * head_dim + idx
                    } else {
                        heads * head_dim + h * head_dim + (idx - head_dim)
                    }
                } else {
                    src_row
                };
                for g in 0..(k / 16) {
                    let sc = decode_e4m3(w_scales[src_row * (k / 16) + g]) * weight_scale_2;
                    for j in 0..16 {
                        let byte = w_packed[src_row * (k / 2) + g * 8 + (j / 2)];
                        let nib = if (j & 1) == 0 { byte & 0x0f } else { byte >> 4 };
                        w_ref_bf16[dst_row * k + g * 16 + j] =
                            half::bf16::from_f32(decode_e2m1(nib) * sc).to_bits();
                    }
                }
            }

            let d_act = gpu.alloc(act_bf16.len() * 2).expect("alloc act");
            let d_w_packed = gpu.alloc(w_packed.len()).expect("alloc w_packed");
            let d_w_scales = gpu.alloc(w_scales.len()).expect("alloc w_scales");
            kept_weights.borrow_mut().push(d_w_packed);
            kept_weights.borrow_mut().push(d_w_scales);
            let d_w_ref = gpu.alloc(w_ref_bf16.len() * 2).expect("alloc w_ref");
            let out_elems = m * ldc;
            let d_out_nvfp4 = gpu.alloc(out_elems * 2).expect("alloc out_nvfp4");
            let d_out_ref = gpu.alloc(out_elems * 2).expect("alloc out_ref");

            let act_bytes: Vec<u8> = act_bf16.iter().flat_map(|v| v.to_le_bytes()).collect();
            let w_ref_bytes: Vec<u8> = w_ref_bf16.iter().flat_map(|v| v.to_le_bytes()).collect();
            gpu.copy_h2d(&act_bytes, d_act).expect("h2d act");
            gpu.copy_h2d(&w_packed, d_w_packed).expect("h2d w_packed");
            gpu.copy_h2d(&w_scales, d_w_scales).expect("h2d w_scales");
            gpu.copy_h2d(&w_ref_bytes, d_w_ref).expect("h2d w_ref");

            // Fill strided padding columns with sentinel to verify ldc > n is untouched.
            let sentinel_bits = half::bf16::from_f32(3.5).to_bits();
            let sentinel_buf: Vec<u8> = (0..out_elems)
                .flat_map(|_| sentinel_bits.to_le_bytes())
                .collect();
            gpu.copy_h2d(&sentinel_buf, d_out_nvfp4)
                .expect("h2d sentinel");

            let qw = QuantizedWeight {
                weight: d_w_packed,
                weight_scale: d_w_scales,
                weight_scale_2,
                input_scale: DevicePtr::NULL,
            };

            // Run native zero-sync cuBLASLt NVFP4 path.
            let ran = run_bf16_gemm_unchecked(
                &gpu,
                d_act,
                None,
                &qw,
                d_out_nvfp4,
                m,
                n,
                k,
                ldc,
                qg,
                false,
                stream,
            )
            .expect("nvfp4 gemm");
            assert!(ran, "case `{label}` failed to dispatch");

            // Run BF16 cuBLASLt reference on the exact dequantized weight.
            cublaslt::bf16_gemm_act_weight_t_ldc(
                d_act.0,
                d_w_ref.0,
                d_out_ref.0,
                m as u32,
                n as u32,
                k as u32,
                ldc as u32,
                1.0,
                stream,
            )
            .expect("bf16 ref gemm");
            gpu.synchronize(stream).expect("sync");

            let mut got_bytes = vec![0u8; out_elems * 2];
            let mut ref_bytes = vec![0u8; out_elems * 2];
            gpu.copy_d2h(d_out_nvfp4, &mut got_bytes).expect("d2h got");
            gpu.copy_d2h(d_out_ref, &mut ref_bytes).expect("d2h ref");

            let mut dot = 0.0f64;
            let mut norm_g = 0.0f64;
            let mut norm_r = 0.0f64;
            let mut sq_err = 0.0f64;
            for row in 0..m {
                for col in 0..ldc {
                    let idx = row * ldc + col;
                    let g_bits = u16::from_le_bytes([got_bytes[idx * 2], got_bytes[idx * 2 + 1]]);
                    if col >= n {
                        assert_eq!(
                            g_bits, sentinel_bits,
                            "case `{label}` clobbered ldc padding at ({row}, {col})"
                        );
                        continue;
                    }
                    let r_bits = u16::from_le_bytes([ref_bytes[idx * 2], ref_bytes[idx * 2 + 1]]);
                    let g = f64::from(half::bf16::from_bits(g_bits).to_f32());
                    let r = f64::from(half::bf16::from_bits(r_bits).to_f32());
                    assert!(g.is_finite(), "non-finite NVFP4 output at ({row}, {col})");
                    dot += g * r;
                    norm_g += g * g;
                    norm_r += r * r;
                    let d = g - r;
                    sq_err += d * d;
                }
            }
            let cosine = dot / (norm_g.sqrt() * norm_r.sqrt());
            let rel_rmse = sq_err.sqrt() / norm_r.sqrt();
            println!(
                "NVFP4_FAST_PROJ [{label}] {m}x{n}x{k} ldc={ldc} amp={act_amp}: cosine={cosine:.6} rel_rmse={rel_rmse:.6}"
            );
            assert!(
                cosine > 0.995,
                "case `{label}` cosine {cosine:.6} <= 0.995 (rel_rmse={rel_rmse:.6})"
            );
            assert!(
                rel_rmse < 0.10,
                "case `{label}` rel_rmse {rel_rmse:.6} >= 0.10 (cosine={cosine:.6})"
            );

            // Also verify back-to-back ZERO-SYNC 100x scale jump on the SAME cached weight
            // (no `gpu.synchronize` between call 1 and call 2) to prove zero stale `d_alpha`.
            if label == "verify_m128_ssm_out" {
                let act100_bytes: Vec<u8> = act_f32
                    .iter()
                    .flat_map(|&v| half::bf16::from_f32(v * 100.0).to_bits().to_le_bytes())
                    .collect();
                let d_act100 = gpu.alloc(act100_bytes.len()).expect("alloc act100");
                let d_out1 = gpu.alloc(out_elems * 2).expect("alloc out1");
                let d_out2 = gpu.alloc(out_elems * 2).expect("alloc out2");
                let d_ref2 = gpu.alloc(out_elems * 2).expect("alloc ref2");
                gpu.copy_h2d(&act100_bytes, d_act100).expect("h2d act100");

                // Queue 1x followed IMMEDIATELY by 100x on the same stream (zero host sync).
                assert!(
                    run_bf16_gemm_unchecked(
                        &gpu, d_act, None, &qw, d_out1, m, n, k, ldc, qg, true, stream
                    )
                    .expect("async call1")
                );
                assert!(
                    run_bf16_gemm_unchecked(
                        &gpu, d_act100, None, &qw, d_out2, m, n, k, ldc, qg, true, stream
                    )
                    .expect("async call2")
                );
                cublaslt::bf16_gemm_act_weight_t_ldc(
                    d_act100.0, d_w_ref.0, d_ref2.0, m as u32, n as u32, k as u32, ldc as u32, 1.0,
                    stream,
                )
                .expect("bf16 ref2");
                gpu.synchronize(stream).expect("sync back_to_back");

                let mut out2_bytes = vec![0u8; out_elems * 2];
                let mut ref2_bytes = vec![0u8; out_elems * 2];
                gpu.copy_d2h(d_out2, &mut out2_bytes).expect("d2h out2");
                gpu.copy_d2h(d_ref2, &mut ref2_bytes).expect("d2h ref2");
                let (mut s_dot, mut s_ng, mut s_nr, mut s_sq) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
                for idx in 0..(m * n) {
                    let g = f64::from(
                        half::bf16::from_bits(u16::from_le_bytes([
                            out2_bytes[idx * 2],
                            out2_bytes[idx * 2 + 1],
                        ]))
                        .to_f32(),
                    );
                    let r = f64::from(
                        half::bf16::from_bits(u16::from_le_bytes([
                            ref2_bytes[idx * 2],
                            ref2_bytes[idx * 2 + 1],
                        ]))
                        .to_f32(),
                    );
                    s_dot += g * r;
                    s_ng += g * g;
                    s_nr += r * r;
                    let d = g - r;
                    s_sq += d * d;
                }
                let s_cos = s_dot / (s_ng.sqrt() * s_nr.sqrt());
                let s_rmse = s_sq.sqrt() / s_nr.sqrt();
                println!(
                    "NVFP4_FAST_PROJ [back_to_back_zero_sync_1x_to_100x] {m}x{n}x{k}: cosine={s_cos:.6} rel_rmse={s_rmse:.6}"
                );
                assert!(
                    s_cos > 0.995 && s_rmse < 0.10,
                    "back-to-back zero-sync 100x jump failed: cosine={s_cos:.6} rel_rmse={s_rmse:.6}"
                );
                gpu.free(d_act100).expect("free act100");
                gpu.free(d_out1).expect("free out1");
                gpu.free(d_out2).expect("free out2");
                gpu.free(d_ref2).expect("free ref2");
            }

            // Also verify E4M3 -> NVFP4 requant path on the same shape (zero amax pass).
            if qg.is_none() && m <= 256 {
                let amax = act_f32
                    .iter()
                    .fold(0.0f32, |acc, &v| acc.max(v.abs()))
                    .max(1e-30);
                let a_scale_val = amax / 448.0;
                let inv = 448.0 / amax;
                let mut a8_host = vec![0u8; m * k];
                for (i, &v) in act_f32.iter().enumerate() {
                    // Nearest finite E4M3 encoding via bf16-precision scaled magnitude.
                    let scaled = (v * inv).clamp(-448.0, 448.0);
                    let sign = if scaled < 0.0 { 0x80u8 } else { 0u8 };
                    let av = scaled.abs();
                    let bits = av.to_bits();
                    let exp = ((bits >> 23) & 0xff) as i32 - 127;
                    let man = bits & 0x7f_ffff;
                    let mag_byte = if av == 0.0 || exp < -9 {
                        0u8
                    } else if exp < -6 {
                        ((av * 512.0).round() as u8).min(7)
                    } else {
                        let mut e = exp + 7;
                        let mut m3 = ((man + (1 << 19)) >> 20) as i32;
                        if m3 > 7 {
                            m3 = 0;
                            e += 1;
                        }
                        if e > 15 || (e == 15 && m3 > 6) {
                            (15u8 << 3) | 6
                        } else {
                            ((e as u8) << 3) | (m3 as u8)
                        }
                    };
                    a8_host[i] = sign | mag_byte;
                }
                let d_a8 = gpu.alloc(a8_host.len()).expect("alloc a8");
                let d_ascale = gpu.alloc(4).expect("alloc a_scale");
                let a8_dequant_bytes: Vec<u8> = a8_host
                    .iter()
                    .flat_map(|&b| {
                        half::bf16::from_f32(decode_e4m3(b) * a_scale_val)
                            .to_bits()
                            .to_le_bytes()
                    })
                    .collect();
                let d_a8_bf16 = gpu.alloc(a8_dequant_bytes.len()).expect("alloc a8_bf16");
                let d_e4m3_ref = gpu.alloc(out_elems * 2).expect("alloc e4m3_ref");
                gpu.copy_h2d(&a8_host, d_a8).expect("h2d a8");
                gpu.copy_h2d(&a_scale_val.to_le_bytes(), d_ascale)
                    .expect("h2d a_scale");
                gpu.copy_h2d(&a8_dequant_bytes, d_a8_bf16)
                    .expect("h2d a8_bf16");
                gpu.copy_h2d(&sentinel_buf, d_out_nvfp4)
                    .expect("reset sentinel");
                let ran_e4m3 = run_e4m3_gemm_unchecked(
                    &gpu,
                    d_a8,
                    d_ascale,
                    &qw,
                    d_out_nvfp4,
                    m,
                    n,
                    k,
                    ldc,
                    qg,
                    false,
                    stream,
                )
                .expect("nvfp4 e4m3 gemm");
                assert!(ran_e4m3, "e4m3 case `{label}` failed to dispatch");
                cublaslt::bf16_gemm_act_weight_t_ldc(
                    d_a8_bf16.0,
                    d_w_ref.0,
                    d_e4m3_ref.0,
                    m as u32,
                    n as u32,
                    k as u32,
                    ldc as u32,
                    1.0,
                    stream,
                )
                .expect("e4m3 dequant ref gemm");
                gpu.synchronize(stream).expect("sync e4m3");
                gpu.copy_d2h(d_out_nvfp4, &mut got_bytes)
                    .expect("d2h e4m3 got");
                let mut e4m3_ref_bytes = vec![0u8; out_elems * 2];
                gpu.copy_d2h(d_e4m3_ref, &mut e4m3_ref_bytes)
                    .expect("d2h e4m3 ref");
                let (mut e_dot, mut e_ng, mut e_nr, mut e_sq) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
                let (mut b_dot, mut b_nr, mut b_sq) = (0.0f64, 0.0f64, 0.0f64);
                for row in 0..m {
                    for col in 0..n {
                        let idx = row * ldc + col;
                        let g = f64::from(
                            half::bf16::from_bits(u16::from_le_bytes([
                                got_bytes[idx * 2],
                                got_bytes[idx * 2 + 1],
                            ]))
                            .to_f32(),
                        );
                        let r_e4m3 = f64::from(
                            half::bf16::from_bits(u16::from_le_bytes([
                                e4m3_ref_bytes[idx * 2],
                                e4m3_ref_bytes[idx * 2 + 1],
                            ]))
                            .to_f32(),
                        );
                        let r_bf16 = f64::from(
                            half::bf16::from_bits(u16::from_le_bytes([
                                ref_bytes[idx * 2],
                                ref_bytes[idx * 2 + 1],
                            ]))
                            .to_f32(),
                        );
                        e_dot += g * r_e4m3;
                        e_ng += g * g;
                        e_nr += r_e4m3 * r_e4m3;
                        let d_e = g - r_e4m3;
                        e_sq += d_e * d_e;
                        b_dot += g * r_bf16;
                        b_nr += r_bf16 * r_bf16;
                        let d_b = g - r_bf16;
                        b_sq += d_b * d_b;
                    }
                }
                let cos_vs_e4m3 = e_dot / (e_ng.sqrt() * e_nr.sqrt());
                let rmse_vs_e4m3 = e_sq.sqrt() / e_nr.sqrt();
                let cos_vs_bf16 = b_dot / (e_ng.sqrt() * b_nr.sqrt());
                let rmse_vs_bf16 = b_sq.sqrt() / b_nr.sqrt();
                println!(
                    "NVFP4_FAST_PROJ_E4M3 [{label}] {m}x{n}x{k}: vs_e4m3=(cos={cos_vs_e4m3:.6}, rmse={rmse_vs_e4m3:.6}) vs_bf16=(cos={cos_vs_bf16:.6}, rmse={rmse_vs_bf16:.6})"
                );
                assert!(
                    cos_vs_e4m3 > 0.995 && rmse_vs_e4m3 < 0.10,
                    "e4m3 case `{label}` vs E4M3 ref failed: cos={cos_vs_e4m3:.6} rmse={rmse_vs_e4m3:.6}"
                );
                assert!(
                    cos_vs_bf16 > 0.994 && rmse_vs_bf16 < 0.11,
                    "e4m3 case `{label}` vs BF16 ref failed: cos={cos_vs_bf16:.6} rmse={rmse_vs_bf16:.6}"
                );
                gpu.free(d_a8).expect("free a8");
                gpu.free(d_ascale).expect("free a_scale");
                gpu.free(d_a8_bf16).expect("free a8_bf16");
                gpu.free(d_e4m3_ref).expect("free e4m3_ref");
            }

            gpu.free(d_act).expect("free act");
            gpu.free(d_w_ref).expect("free w_ref");
            gpu.free(d_out_nvfp4).expect("free out_nvfp4");
            gpu.free(d_out_ref).expect("free out_ref");
        };

        // Cover Flash-Next shapes across verify (M=128, 160, 256) and prefill (M=2048),
        // including strided gated-Q permutation and 100x zero-sync back-to-back scale jump.
        run_bf16_shape(
            "verify_m128_ssm_out",
            128,
            2_560,
            6_144,
            2_560,
            None,
            1.0,
            0.25,
        );
        run_bf16_shape(
            "verify_m160_qkvz",
            160,
            16_384,
            2_560,
            16_384,
            None,
            2.0,
            0.125,
        );
        run_bf16_shape(
            "verify_m256_ssm_out",
            256,
            2_560,
            6_144,
            2_560,
            None,
            0.5,
            0.5,
        );
        run_bf16_shape(
            "verify_m256_attn_q_strided_qg",
            256,
            12_288,
            2_560,
            13_312,
            Some((24, 256)),
            1.5,
            0.375,
        );
        run_bf16_shape(
            "verify_m128_kv_strided",
            128,
            512,
            2_560,
            13_312,
            None,
            1.0,
            0.25,
        );
        run_bf16_shape(
            "prefill_m2048_ssm_out",
            2_048,
            2_560,
            6_144,
            2_560,
            None,
            1.0,
            0.25,
        );
        run_bf16_shape(
            "prefill_m2048_attn_q_qg",
            2_048,
            12_288,
            2_560,
            13_312,
            Some((24, 256)),
            1.0,
            0.25,
        );
        run_bf16_shape(
            "prefill_m2048_qkvz",
            2_048,
            16_384,
            2_560,
            16_384,
            None,
            1.0,
            0.125,
        );
        run_bf16_shape(
            "prefill_m2048_kv",
            2_048,
            512,
            2_560,
            13_312,
            None,
            1.0,
            0.25,
        );

        // Verify shared-activation [Q, K, V] single-quant path (`qwen4_nvfp4_compute_alphas4`)
        // writing packed `[Q_all(6144) | G_all(6144) | K(512) | V(512)]` at ldc=13312.
        {
            let (m, k, ldc) = (256usize, 2_560usize, 13_312usize);
            let mut state = 0x1234_5678_9abc_def0u64;
            let mut next_u32 = || {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                (state >> 32) as u32
            };
            let act_bytes: Vec<u8> = (0..(m * k))
                .flat_map(|_| {
                    let u0 = ((next_u32() >> 8) as f32) * (1.0 / 16_777_216.0) - 0.5;
                    let u1 = ((next_u32() >> 8) as f32) * (1.0 / 16_777_216.0) - 0.5;
                    half::bf16::from_f32((u0 + u1) * 1.25)
                        .to_bits()
                        .to_le_bytes()
                })
                .collect();
            let d_act = gpu.alloc(act_bytes.len()).expect("alloc shared act");
            gpu.copy_h2d(&act_bytes, d_act).expect("h2d shared act");

            let specs: [(&str, usize, usize, Option<(usize, usize)>, f32); 3] = [
                ("q", 12_288, 0, Some((24, 256)), 0.125),
                ("k", 512, 12_288, None, 0.25),
                ("v", 512, 12_800, None, 0.5),
            ];
            let mut qws = Vec::new();
            let mut refs = Vec::new();
            for &(_, n, _, qg, ws2) in &specs {
                let mut wp = vec![0u8; n * (k / 2)];
                let mut ws = vec![0u8; n * (k / 16)];
                for b in &mut wp {
                    *b = (((next_u32() & 0x0f) << 4) | (next_u32() & 0x0f)) as u8;
                }
                for (i, s) in ws.iter_mut().enumerate() {
                    *s = W_SCALES[((next_u32() as usize) ^ i) % W_SCALES.len()];
                }
                let mut w_ref = vec![0u16; n * k];
                for src_row in 0..n {
                    let dst_row = if let Some((heads, head_dim)) = qg {
                        let group_dim = 2 * head_dim;
                        let h = src_row / group_dim;
                        let idx = src_row - h * group_dim;
                        if idx < head_dim {
                            h * head_dim + idx
                        } else {
                            heads * head_dim + h * head_dim + (idx - head_dim)
                        }
                    } else {
                        src_row
                    };
                    for g in 0..(k / 16) {
                        let sc = decode_e4m3(ws[src_row * (k / 16) + g]) * ws2;
                        for j in 0..16 {
                            let byte = wp[src_row * (k / 2) + g * 8 + (j / 2)];
                            let nib = if (j & 1) == 0 { byte & 0x0f } else { byte >> 4 };
                            w_ref[dst_row * k + g * 16 + j] =
                                half::bf16::from_f32(decode_e2m1(nib) * sc).to_bits();
                        }
                    }
                }
                let d_wp = gpu.alloc(wp.len()).expect("alloc shared wp");
                let d_ws = gpu.alloc(ws.len()).expect("alloc shared ws");
                kept_weights.borrow_mut().push(d_wp);
                kept_weights.borrow_mut().push(d_ws);
                let d_wr = gpu.alloc(w_ref.len() * 2).expect("alloc shared wr");
                gpu.copy_h2d(&wp, d_wp).expect("h2d shared wp");
                gpu.copy_h2d(&ws, d_ws).expect("h2d shared ws");
                let wr_bytes: Vec<u8> = w_ref.iter().flat_map(|v| v.to_le_bytes()).collect();
                gpu.copy_h2d(&wr_bytes, d_wr).expect("h2d shared wr");
                qws.push(QuantizedWeight {
                    weight: d_wp,
                    weight_scale: d_ws,
                    weight_scale_2: ws2,
                    input_scale: DevicePtr::NULL,
                });
                refs.push(d_wr);
            }

            let d_out = gpu.alloc(m * ldc * 2).expect("alloc shared out");
            let d_ref = gpu.alloc(m * ldc * 2).expect("alloc shared ref");
            let outs = [
                super::super::SharedActOut {
                    weight: &qws[0],
                    out: d_out,
                    n: 12_288,
                    qg: Some((24, 256)),
                },
                super::super::SharedActOut {
                    weight: &qws[1],
                    out: d_out.offset(12_288 * 2),
                    n: 512,
                    qg: None,
                },
                super::super::SharedActOut {
                    weight: &qws[2],
                    out: d_out.offset(12_800 * 2),
                    n: 512,
                    qg: None,
                },
            ];
            let outcome =
                run_shared_act_all_unchecked(&gpu, d_act, &outs, m, k, ldc, false, stream)
                    .expect("shared_act nvfp4");
            assert_eq!(outcome, SharedActNvfp4Outcome::HandledAll);

            for (i, &(_, n, col_off, _, _)) in specs.iter().enumerate() {
                cublaslt::bf16_gemm_act_weight_t_ldc(
                    d_act.0,
                    refs[i].0,
                    d_ref.offset(col_off * 2).0,
                    m as u32,
                    n as u32,
                    k as u32,
                    ldc as u32,
                    1.0,
                    stream,
                )
                .expect("shared ref gemm");
            }
            gpu.synchronize(stream).expect("sync shared");

            let mut got_bytes = vec![0u8; m * ldc * 2];
            let mut ref_bytes = vec![0u8; m * ldc * 2];
            gpu.copy_d2h(d_out, &mut got_bytes).expect("d2h shared got");
            gpu.copy_d2h(d_ref, &mut ref_bytes).expect("d2h shared ref");
            for &(name, n, col_off, _, _) in &specs {
                let (mut dot, mut ng, mut nr, mut sq) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
                for row in 0..m {
                    for col in col_off..(col_off + n) {
                        let idx = row * ldc + col;
                        let g = f64::from(
                            half::bf16::from_bits(u16::from_le_bytes([
                                got_bytes[idx * 2],
                                got_bytes[idx * 2 + 1],
                            ]))
                            .to_f32(),
                        );
                        let r = f64::from(
                            half::bf16::from_bits(u16::from_le_bytes([
                                ref_bytes[idx * 2],
                                ref_bytes[idx * 2 + 1],
                            ]))
                            .to_f32(),
                        );
                        dot += g * r;
                        ng += g * g;
                        nr += r * r;
                        let d = g - r;
                        sq += d * d;
                    }
                }
                let cos = dot / (ng.sqrt() * nr.sqrt());
                let rmse = sq.sqrt() / nr.sqrt();
                println!(
                    "NVFP4_FAST_PROJ_SHARED_ACT [{name}] {m}x{n}x{k} ldc={ldc}: cosine={cos:.6} rel_rmse={rmse:.6}"
                );
                assert!(
                    cos > 0.995 && rmse < 0.10,
                    "shared_act `{name}` parity failed: cosine={cos:.6} rel_rmse={rmse:.6}"
                );
            }
            gpu.free(d_act).expect("free shared act");
            gpu.free(d_out).expect("free shared out");
            gpu.free(d_ref).expect("free shared ref");
            for ptr in refs {
                gpu.free(ptr).expect("free shared ref weight");
            }
        }

        for ptr in kept_weights.into_inner() {
            let _ = gpu.free(ptr);
        }
    }
}
