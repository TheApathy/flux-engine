// SPDX-License-Identifier: AGPL-3.0-only

//! Persistent on-disk cache for `cuBLASLt` autotuned GEMM plans.
//!
//! `tuned_plan_dt` and `fp8_gemm_device_scaled` benchmark up to 16 heuristic
//! candidates on the GPU (tens of warmup + timed bursts per candidate) the
//! first time a shape is seen in a process. Persisting the winning candidate's
//! 9 `cublasLtMatmulAlgoConfigAttributes_t` values (`0..=8`, including
//! Blackwell `INNER_SHAPE_ID` and `CLUSTER_SHAPE_ID`) keyed by `(cuBLASLt
//! version, GPU compute capability, SM count, device name, workspace size,
//! GEMM family & shape)` lets warm server restarts match and validate the
//! winner directly against the CPU-only `cublasLtMatmulAlgoGetHeuristic` +
//! `cublasLtMatmulAlgoCheck` calls in microseconds with zero GPU memory
//! allocations, zero dummy launches, and zero stream synchronizations.
//!
//! All filesystem errors (read-only container root, missing `$HOME`, corrupt
//! JSON) degrade gracefully to the live in-process autotuner and never fail a
//! GEMM call.

use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::ffi::{CStr, c_char, c_void};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

use super::{AlignedAlgo, HeurResult};

// SAFETY: Symbols and C ABI signatures match `cublasLt.h` and `cuda.h` in the linked CUDA libraries.
unsafe extern "C" {
    fn cublasLtGetVersion() -> usize;
    fn cublasLtMatmulAlgoConfigGetAttribute(
        algo: *const c_void,
        attr: u32,
        value: *mut c_void,
        bytes: usize,
        written: *mut usize,
    ) -> i32;
    fn cuCtxGetDevice(device: *mut i32) -> i32;
    fn cuDeviceGetAttribute(pi: *mut i32, attrib: u32, dev: i32) -> i32;
    fn cuDeviceGetName(name: *mut c_char, len: i32, dev: i32) -> i32;
}

const SCHEMA_VERSION: u32 = 2;
/// `library_types.h` value for `CUDA_R_8F_E4M3`.
const CUDA_R_8F_E4M3: i32 = 28;
/// SM121a (Blackwell GB10) practical per-SM dense FP8 E4M3 -> BF16 tensor
/// throughput in TFLOPS (~204 TFLOPS across 48 SMs; golden prefill shapes hit
/// 193–209 TFLOPS).
const SM121A_FP8_TFLOPS_PER_SM: f64 = 4.25;
/// Conservative effective LPDDR5X streaming bandwidth on GB10 in TB/s (250 GB/s).
const SM121A_EFFECTIVE_BW_TB_S: f64 = 0.250;
/// SM120 (discrete Blackwell: RTX PRO 5000 / RTX 5090 / RTX PRO 6000) practical
/// per-SM dense FP8 E4M3 -> BF16 tensor throughput in TFLOPS (~512–874 TFLOPS
/// across 110–188 SMs).
const SM120_FP8_TFLOPS_PER_SM: f64 = 4.65;
/// Conservative effective GDDR7 streaming bandwidth on SM120 in TB/s (1.40 TB/s).
const SM120_EFFECTIVE_BW_TB_S: f64 = 1.40;
/// SM103 (Blackwell Ultra: B300 / GB300) practical per-SM dense FP8 E4M3 -> BF16
/// tensor throughput in TFLOPS (~2.24 PFLOPS across 160 SMs).
const SM103_FP8_TFLOPS_PER_SM: f64 = 14.0;
/// Conservative effective HBM3e streaming bandwidth on SM103 in TB/s (6.0 TB/s).
const SM103_EFFECTIVE_BW_TB_S: f64 = 6.0;
/// Fixed kernel launch overhead floor in milliseconds (8 us).
const FP8_LAUNCH_OVERHEAD_MS: f64 = 0.008;

/// Architecture-calibrated dense FP8 E4M3 -> BF16 roofline parameters.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Fp8RooflineParams {
    tflops_per_sm: f64,
    effective_bw_tb_s: f64,
    launch_overhead_ms: f64,
}

const fn fp8_roofline_params(cc_major: i32, cc_minor: i32) -> Option<Fp8RooflineParams> {
    match (cc_major, cc_minor) {
        (12, 1) => Some(Fp8RooflineParams {
            tflops_per_sm: SM121A_FP8_TFLOPS_PER_SM,
            effective_bw_tb_s: SM121A_EFFECTIVE_BW_TB_S,
            launch_overhead_ms: FP8_LAUNCH_OVERHEAD_MS,
        }),
        (12, 0) => Some(Fp8RooflineParams {
            tflops_per_sm: SM120_FP8_TFLOPS_PER_SM,
            effective_bw_tb_s: SM120_EFFECTIVE_BW_TB_S,
            launch_overhead_ms: FP8_LAUNCH_OVERHEAD_MS,
        }),
        (10, 3) => Some(Fp8RooflineParams {
            tflops_per_sm: SM103_FP8_TFLOPS_PER_SM,
            effective_bw_tb_s: SM103_EFFECTIVE_BW_TB_S,
            launch_overhead_ms: FP8_LAUNCH_OVERHEAD_MS,
        }),
        _ => None,
    }
}
/// Minimum SM121a FP8 roofline efficiency required to accept a canonical
/// (`req`-tagged) entry or persist a newly live-tuned winner to disk. Rejects
/// contended cold-tune poisoning (~13–115 TFLOPS under NVMe/LPDDR5X pressure)
/// while leaving ~35% headroom for thermal throttling.
const CANONICAL_ROOFLINE_FLOOR: f64 = 0.65;
/// Stricter SM121a FP8 roofline efficiency required to reuse a legacy `v2`
/// entry (lacking `req`/`det` tags, e.g. from bundled seed caches or older
/// narrow-search runs). Preserves 0ms hits for already-fast shapes while
/// upgrading poisoned or suboptimal narrow-search shapes via `req=16` re-tune.
const LEGACY_ROOFLINE_FLOOR: f64 = 0.72;
/// Minimum `M` (for both target and donor) eligible for zero-GPU-launch
/// cross-`M` `AlgoConfig` reuse. Wide prefill shapes (`M >= 1024`) share
/// cuBLASLt tiling and staging families smoothly across nearby `M`.
const CROSS_M_MIN_M: u32 = 1024;
/// Index of `CUBLASLT_ALGO_CONFIG_STAGES_ID` in [`AlgoConfig`].
pub(super) const ALGO_CONFIG_IDX_STAGES_ID: usize = 6;
/// Documented `cublasLtMatmulStages_t` 64-wide K-stage IDs (`cublasLt.h`):
/// `CUBLASLT_MATMUL_STAGES_64x1` (13) through `CUBLASLT_MATMUL_STAGES_64x6` (18)
/// and `CUBLASLT_MATMUL_STAGES_64xAUTO` (35). On SM121a dense FP8 GEMMs these
/// correspond to Ada-generation `sm89_xmma` kernels, whereas native Blackwell
/// `nvjet_sm121` kernels use `CUBLASLT_MATMUL_STAGES_128xAUTO` (36) or wider.
const STAGES_64X_FIRST: u32 = 13;
const STAGES_64X_LAST: u32 = 18;
const STAGES_64X_AUTO: u32 = 35;
/// Minimum `M` for the SM121a large dense FP8 anti-`sm89_xmma` guard.
const SM121A_LARGE_FP8_MIN_M: u32 = 4096;
/// Minimum `N` and `K` for the SM121a large dense FP8 anti-`sm89_xmma` guard.
const SM121A_LARGE_FP8_MIN_NK: u32 = 2560;
/// 4-byte `cublasLtMatmulAlgoConfigAttributes_t`: ALGO_ID, TILE_ID, SPLITK_NUM,
/// REDUCTION_SCHEME, CTA_SWIZZLING, CUSTOM_OPTION, STAGES_ID.
const CONFIG_ATTRS_U32: [u32; 7] = [0, 1, 2, 3, 4, 5, 6];
/// 2-byte Blackwell `cublasLtMatmulAlgoConfigAttributes_t`: INNER_SHAPE_ID,
/// CLUSTER_SHAPE_ID (`uint16_t` in `cublasLt.h`).
const CONFIG_ATTRS_U16: [u32; 2] = [7, 8];

const CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT: u32 = 16;
const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR: u32 = 75;
const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR: u32 = 76;

pub(super) type AlgoConfig = [u32; 9];

/// Which autotuned `cuBLASLt` GEMM family produced the plan.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(super) enum PlanKind {
    TunedActWeightT,
    Fp8DeviceScaled,
}

impl PlanKind {
    const fn tag(self) -> &'static str {
        match self {
            Self::TunedActWeightT => "tuned",
            Self::Fp8DeviceScaled => "fp8_scaled",
        }
    }
}

/// GEMM shape and tuning-policy parameters that identify a cached plan on a
/// given device.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct PlanSpec {
    pub ws_size: usize,
    pub kind: PlanKind,
    pub m: u32,
    pub n: u32,
    pub k: u32,
    pub at: i32,
    pub ldc: u32,
    pub outer: bool,
    pub tie_eps: f32,
    pub requested: u32,
    pub deterministic: bool,
}

/// Hardware + library + shape identity for one autotuned GEMM plan.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct PlanCacheKey {
    pub lt_version: usize,
    pub cc_major: i32,
    pub cc_minor: i32,
    pub sm_count: i32,
    pub device_name: String,
    pub ws_size: usize,
    pub kind: PlanKind,
    pub m: u32,
    pub n: u32,
    pub k: u32,
    pub at: i32,
    pub ldc: u32,
    pub outer: bool,
    pub tie_eps_bits: u32,
    pub requested: u32,
    pub deterministic: bool,
}

impl PlanCacheKey {
    pub(super) fn for_current_device(spec: PlanSpec) -> Option<Self> {
        let hw = current_hw_fingerprint()?;
        Some(Self {
            lt_version: hw.lt_version,
            cc_major: hw.cc_major,
            cc_minor: hw.cc_minor,
            sm_count: hw.sm_count,
            device_name: hw.device_name,
            ws_size: spec.ws_size,
            kind: spec.kind,
            m: spec.m,
            n: spec.n,
            k: spec.k,
            at: spec.at,
            ldc: spec.ldc,
            outer: spec.outer,
            tie_eps_bits: spec.tie_eps.to_bits(),
            requested: spec.requested,
            deterministic: spec.deterministic,
        })
    }

    /// Prefix of a serialized v2 key immediately preceding `{m}x{n}x{k}`.
    fn key_prefix_before_m(&self) -> String {
        format!(
            "v{SCHEMA_VERSION}|lt={}|sm={}.{}:{}:{}|ws={}|{}|",
            self.lt_version,
            self.cc_major,
            self.cc_minor,
            self.sm_count,
            self.device_name,
            self.ws_size,
            self.kind.tag(),
        )
    }

    /// Suffix of a legacy v2 key immediately following `{m}`.
    fn legacy_key_suffix_after_m(&self) -> String {
        format!(
            "x{}x{}|at={}|ldc={}|outer={}|eps={:#010x}",
            self.n,
            self.k,
            self.at,
            self.ldc,
            u8::from(self.outer),
            self.tie_eps_bits,
        )
    }

    /// Suffix of a canonical `req`/`det`-tagged v2 key immediately following `{m}`.
    fn canonical_key_suffix_after_m(&self) -> String {
        format!(
            "{}|req={}|det={}",
            self.legacy_key_suffix_after_m(),
            self.requested,
            u8::from(self.deterministic),
        )
    }

    /// Canonical key including heuristic candidate search width (`req`) and
    /// deterministic split-K filtering (`det`).
    fn encoded(&self) -> String {
        format!(
            "{}{}{}",
            self.key_prefix_before_m(),
            self.m,
            self.canonical_key_suffix_after_m(),
        )
    }

    /// Pre-`req`/`det` v2 key format used by bundled seed caches and older
    /// user cache files.
    fn legacy_v2_encoded(&self) -> String {
        format!(
            "{}{}{}",
            self.key_prefix_before_m(),
            self.m,
            self.legacy_key_suffix_after_m(),
        )
    }
}

/// Returns `true` when `stages_id` is a documented `CUBLASLT_MATMUL_STAGES_64x*`
/// value (`64x1..=64x6` or `64xAUTO` from `cublasLt.h`).
pub(super) const fn is_legacy_64x_stages(stages_id: u32) -> bool {
    matches!(
        stages_id,
        STAGES_64X_FIRST..=STAGES_64X_LAST | STAGES_64X_AUTO
    )
}

/// Returns `true` when `key` describes a large dense FP8 E4M3 GEMM on SM120 or
/// SM121a (`cc_major == 12 && (cc_minor == 0 || cc_minor == 1)`,
/// `m >= 4096 && n >= 2560 && k >= 2560`, `!outer`) where Ada-generation
/// `sm89_xmma` kernels can win transiently at cold Layer 0 and collapse in
/// steady state.
pub(super) fn is_sm121a_large_fp8_key(key: &PlanCacheKey) -> bool {
    key.cc_major == 12
        && matches!(key.cc_minor, 0 | 1)
        && key.kind == PlanKind::Fp8DeviceScaled
        && key.at == CUDA_R_8F_E4M3
        && !key.outer
        && key.m >= SM121A_LARGE_FP8_MIN_M
        && key.n >= SM121A_LARGE_FP8_MIN_NK
        && key.k >= SM121A_LARGE_FP8_MIN_NK
}

/// Returns `true` when `measured_ms` passes the canonical Blackwell
/// (`SM103`/`SM120`/`SM121a`) roofline health floor (and finite positive sanity
/// check) for `key`.
pub(super) fn is_canonical_healthy(key: &PlanCacheKey, measured_ms: f32) -> bool {
    evaluate_entry_health(key, measured_ms, CANONICAL_ROOFLINE_FLOOR) == HealthCheck::Healthy
}

/// Returns `true` when a cached 9-attribute [`AlgoConfig`] uses no split-K
/// (`SPLITK_NUM <= 1`) and no cross-CTA reduction (`REDUCTION_SCHEME == 0`).
pub(super) const fn is_deterministic_config(config: &AlgoConfig) -> bool {
    (config[2] as i32) <= 1 && config[3] == 0
}

/// Result of checking a cached or newly measured GEMM plan timing against
/// sanity and hardware roofline bounds.
#[derive(Clone, Copy, Debug, PartialEq)]
enum HealthCheck {
    /// Entry passed validation (either no roofline floor applies for this
    /// hardware/dtype, or measured throughput meets the floor).
    Healthy,
    /// `measured_ms` is non-finite or non-positive.
    InvalidTiming,
    /// Measured throughput fell below the hardware roofline floor, indicating
    /// transient memory/I/O contention during cold tuning or a suboptimal plan.
    Degraded {
        achieved_tflops: f64,
        floor_tflops: f64,
        roofline_eff: f64,
    },
}

fn evaluate_entry_health(
    key: &PlanCacheKey,
    measured_ms: f32,
    min_roofline_eff: f64,
) -> HealthCheck {
    if !measured_ms.is_finite() || measured_ms <= 0.0 {
        return HealthCheck::InvalidTiming;
    }
    let Some(roofline) = fp8_roofline_params(key.cc_major, key.cc_minor) else {
        return HealthCheck::Healthy;
    };
    let is_fp8_tensor = key.sm_count > 0
        && key.kind == PlanKind::Fp8DeviceScaled
        && key.at == CUDA_R_8F_E4M3
        && !key.outer;
    if !is_fp8_tensor || key.m == 0 || key.n == 0 || key.k == 0 {
        return HealthCheck::Healthy;
    }

    let m = f64::from(key.m);
    let n = f64::from(key.n);
    let k = f64::from(key.k);
    let flops = 2.0 * m * n * k;
    // Minimum memory traffic: FP8 A[m,k] + FP8 W[n,k] + BF16 D[m,n].
    let bytes = (m * k + n * k) + 2.0 * m * n;
    let peak_tflops = f64::from(key.sm_count) * roofline.tflops_per_sm;
    let t_compute_ms = flops / (peak_tflops * 1e9);
    let t_bw_ms = bytes / (roofline.effective_bw_tb_s * 1e9);
    let t_roof_ms = t_compute_ms.max(t_bw_ms) + roofline.launch_overhead_ms;
    let ms = f64::from(measured_ms);
    let roofline_eff = t_roof_ms / ms;
    if roofline_eff < min_roofline_eff {
        let achieved_tflops = flops / (ms * 1e9);
        let floor_ms = t_roof_ms / min_roofline_eff;
        let floor_tflops = flops / (floor_ms * 1e9);
        return HealthCheck::Degraded {
            achieved_tflops,
            floor_tflops,
            roofline_eff,
        };
    }
    HealthCheck::Healthy
}

#[derive(Clone, Debug)]
struct HwFingerprint {
    lt_version: usize,
    cc_major: i32,
    cc_minor: i32,
    sm_count: i32,
    device_name: String,
}

fn sanitize_device_name(raw: &str) -> String {
    raw.trim()
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '.' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

fn current_hw_fingerprint() -> Option<HwFingerprint> {
    static HW_BY_DEV: OnceLock<Mutex<BTreeMap<i32, HwFingerprint>>> = OnceLock::new();
    let mut dev: i32 = 0;
    // SAFETY: `cuCtxGetDevice` queries the thread-active CUDA context without side effects.
    if unsafe { cuCtxGetDevice(&mut dev) } != 0 {
        return None;
    }
    let cache = HW_BY_DEV.get_or_init(|| Mutex::new(BTreeMap::new()));
    if let Some(existing) = cache.lock().get(&dev).cloned() {
        return Some(existing);
    }
    // SAFETY: pure query FFI calls into CUDA driver and cuBLASLt for valid `dev`.
    let queried = unsafe {
        let lt_version = cublasLtGetVersion();
        let (mut major, mut minor, mut sms) = (0i32, 0i32, 0i32);
        if cuDeviceGetAttribute(
            &mut major,
            CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
            dev,
        ) != 0
            || cuDeviceGetAttribute(
                &mut minor,
                CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                dev,
            ) != 0
            || cuDeviceGetAttribute(&mut sms, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev) != 0
        {
            return None;
        }
        let mut name_buf = [0 as c_char; 128];
        let device_name = if cuDeviceGetName(name_buf.as_mut_ptr(), 128, dev) == 0 {
            name_buf[127] = 0;
            let raw = CStr::from_ptr(name_buf.as_ptr()).to_string_lossy();
            let clean = sanitize_device_name(&raw);
            if clean.is_empty() {
                format!("dev{dev}")
            } else {
                clean
            }
        } else {
            format!("dev{dev}")
        };
        HwFingerprint {
            lt_version,
            cc_major: major,
            cc_minor: minor,
            sm_count: sms,
            device_name,
        }
    };
    cache.lock().insert(dev, queried.clone());
    Some(queried)
}

/// Cached `(cc_major, cc_minor)` of the thread-active CUDA device, or `None`
/// when no CUDA context is active.
pub(super) fn current_device_cc() -> Option<(i32, i32)> {
    let hw = current_hw_fingerprint()?;
    Some((hw.cc_major, hw.cc_minor))
}

/// Extract all 9 `cublasLtMatmulAlgoConfigAttributes_t` integers (`0..=6` as
/// `u32`, `7..=8` as `u16` widened to `u32`) from an initialized 8-byte-aligned
/// `cublasLtMatmulAlgo_t`.
pub(super) fn extract_algo_config(algo: &AlignedAlgo) -> Option<AlgoConfig> {
    let mut config = [0u32; 9];
    for (slot, attr) in config[..7].iter_mut().zip(CONFIG_ATTRS_U32) {
        let mut written = 0usize;
        // SAFETY: `algo` is an 8-byte-aligned 64-byte `cublasLtMatmulAlgo_t`; `slot` is 4-byte `u32`.
        let st = unsafe {
            cublasLtMatmulAlgoConfigGetAttribute(
                algo.as_ptr().cast(),
                attr,
                (slot as *mut u32).cast(),
                std::mem::size_of::<u32>(),
                &mut written,
            )
        };
        if st != 0 || written != std::mem::size_of::<u32>() {
            return None;
        }
    }
    for (slot, attr) in config[7..9].iter_mut().zip(CONFIG_ATTRS_U16) {
        let mut val16: u16 = 0;
        let mut written = 0usize;
        // SAFETY: `algo` is 8-byte-aligned; `INNER_SHAPE_ID` (7) and `CLUSTER_SHAPE_ID` (8) use `uint16_t` storage.
        let st = unsafe {
            cublasLtMatmulAlgoConfigGetAttribute(
                algo.as_ptr().cast(),
                attr,
                (&mut val16 as *mut u16).cast(),
                std::mem::size_of::<u16>(),
                &mut written,
            )
        };
        if st != 0 || written != std::mem::size_of::<u16>() {
            return None;
        }
        *slot = u32::from(val16);
    }
    Some(config)
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub(super) struct CachedEntry {
    pub config: AlgoConfig,
    pub heur_index: u32,
    pub measured_ms: f32,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
struct DiskCacheFile {
    schema: u32,
    entries: BTreeMap<String, CachedEntry>,
}

#[derive(Serialize)]
struct DiskCacheFileRef<'a> {
    schema: u32,
    entries: &'a BTreeMap<String, CachedEntry>,
}

/// Healthy cached plan selected as a cross-`M` donor for a nearby wide prefill
/// shape. Must still be validated against the target's exact `(m, n, k, ldc)`
/// matmul descriptor via CPU-only `cublasLtMatmulAlgoCheck`.
#[derive(Clone, Debug, PartialEq)]
pub(super) struct CrossMDonor {
    pub donor_m: u32,
    pub config: AlgoConfig,
    pub measured_ms: f32,
    pub from_legacy_key: bool,
}

/// Strict total-order ranking for healthy cross-`M` donors (smaller is better):
/// 1. Smallest `|donor_m - target_m|` (exact match `0` wins first),
/// 2. Tie-break favoring `donor_m >= target_m` (`false < true`),
/// 3. Canonical (`req`/`det`-tagged) entries before legacy v2 entries,
/// 4. Faster measured latency.
fn donor_rank_cmp(target_m: u32, a: &CrossMDonor, b: &CrossMDonor) -> std::cmp::Ordering {
    let a_key = (
        a.donor_m.abs_diff(target_m),
        a.donor_m < target_m,
        a.from_legacy_key,
    );
    let b_key = (
        b.donor_m.abs_diff(target_m),
        b.donor_m < target_m,
        b.from_legacy_key,
    );
    a_key
        .cmp(&b_key)
        .then_with(|| a.measured_ms.total_cmp(&b.measured_ms))
}

/// In-memory plan cache supporting exact lookups and zero-GPU-launch cross-`M`
/// donor search across persisted `cublaslt_plans_v2.json` entries.
#[derive(Clone, Debug, Default)]
pub(super) struct PlanCache {
    entries: BTreeMap<String, CachedEntry>,
}

impl PlanCache {
    pub(super) fn from_entries(entries: BTreeMap<String, CachedEntry>) -> Self {
        Self { entries }
    }

    /// Search cached entries for the nearest healthy donor sharing identical
    /// `(version, lt_version, sm, workspace_bytes, op_kind, n, k, a_type, ldc,
    /// outer, tie_eps_bits)` where only `m` may differ.
    ///
    /// Eligible only for wide prefill shapes (`target.m >= 1024`,
    /// `donor.m >= 1024`) within a `2x` `M` ratio
    /// (`max(m, donor.m) <= 2 * min(m, donor.m)`). Ranks exact match first,
    /// then smallest `|m - donor.m|` with tie-break favoring `donor.m >= m`.
    /// Rejects entries failing the Blackwell (`SM103`/`SM120`/`SM121a`) roofline
    /// health floor and legacy `STAGES_64x*` (`sm89_xmma`) donors on large
    /// SM120/SM121a FP8 shapes.
    pub(super) fn find_nearest_m_donor(&self, target: &PlanCacheKey) -> Option<CrossMDonor> {
        if target.m < CROSS_M_MIN_M {
            return None;
        }
        let prefix = target.key_prefix_before_m();
        let canonical_suffix = target.canonical_key_suffix_after_m();
        let legacy_suffix = target.legacy_key_suffix_after_m();
        let guard_sm121a_large = is_sm121a_large_fp8_key(target);
        let mut donor_key = target.clone();
        let mut best: Option<CrossMDonor> = None;

        for (raw_key, entry) in &self.entries {
            let Some(rest) = raw_key.strip_prefix(&prefix) else {
                continue;
            };
            let (m_str, from_legacy_key) = if let Some(m) = rest.strip_suffix(&canonical_suffix) {
                (m, false)
            } else if let Some(m) = rest.strip_suffix(&legacy_suffix) {
                (m, true)
            } else {
                continue;
            };
            let Ok(donor_m) = m_str.parse::<u32>() else {
                continue;
            };
            if donor_m < CROSS_M_MIN_M {
                continue;
            }
            let lo = u64::from(target.m.min(donor_m));
            let hi = u64::from(target.m.max(donor_m));
            if hi > 2 * lo {
                continue;
            }
            if target.deterministic && !is_deterministic_config(&entry.config) {
                continue;
            }
            if guard_sm121a_large && is_legacy_64x_stages(entry.config[ALGO_CONFIG_IDX_STAGES_ID]) {
                continue;
            }
            donor_key.m = donor_m;
            let floor = if from_legacy_key {
                LEGACY_ROOFLINE_FLOOR
            } else {
                CANONICAL_ROOFLINE_FLOOR
            };
            if evaluate_entry_health(&donor_key, entry.measured_ms, floor) != HealthCheck::Healthy {
                continue;
            }
            let candidate = CrossMDonor {
                donor_m,
                config: entry.config,
                measured_ms: entry.measured_ms,
                from_legacy_key,
            };
            if best
                .as_ref()
                .is_none_or(|current| donor_rank_cmp(target.m, &candidate, current).is_lt())
            {
                best = Some(candidate);
            }
        }
        best
    }
}

struct PersistentStore {
    path: Option<PathBuf>,
    cache: PlanCache,
}

static STORE: OnceLock<Mutex<PersistentStore>> = OnceLock::new();
static TMP_SEQ: AtomicU64 = AtomicU64::new(0);

/// Parse the `ATLAS_CUBLASLT_CROSS_M_REUSE` opt-out selector.
/// Enabled by default (`true`); `"0"` disables cross-`M` donor reuse.
fn parse_cross_m_reuse(raw: Option<&str>) -> bool {
    !matches!(raw.map(str::trim), Some("0"))
}

fn cross_m_reuse_enabled() -> bool {
    parse_cross_m_reuse(
        std::env::var("ATLAS_CUBLASLT_CROSS_M_REUSE")
            .ok()
            .as_deref(),
    )
}

fn resolve_cache_path() -> Option<PathBuf> {
    if std::env::var("ATLAS_CUBLASLT_CACHE").ok().as_deref() == Some("0") {
        return None;
    }
    if let Ok(explicit) = std::env::var("ATLAS_CUBLASLT_CACHE_PATH") {
        let trimmed = explicit.trim();
        if !trimmed.is_empty() {
            return Some(PathBuf::from(trimmed));
        }
    }
    let home = std::env::var_os("HOME")?;
    if home.is_empty() {
        return None;
    }
    Some(
        PathBuf::from(home)
            .join(".cache")
            .join("atlas")
            .join("cublaslt_plans_v2.json"),
    )
}

fn load_entries_from_path(path: &Path) -> BTreeMap<String, CachedEntry> {
    let bytes = match fs::read(path) {
        Ok(b) => b,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return BTreeMap::new(),
        Err(e) => {
            tracing::warn!(
                path = %path.display(),
                error = %e,
                "cuBLASLt plan cache read failed; falling back to live autotune"
            );
            return BTreeMap::new();
        }
    };
    match serde_json::from_slice::<DiskCacheFile>(&bytes) {
        Ok(parsed) if parsed.schema == SCHEMA_VERSION => parsed.entries,
        Ok(parsed) => {
            tracing::info!(
                path = %path.display(),
                found_schema = parsed.schema,
                expected_schema = SCHEMA_VERSION,
                "cuBLASLt plan cache schema changed; ignoring old entries"
            );
            BTreeMap::new()
        }
        Err(e) => {
            tracing::warn!(
                path = %path.display(),
                error = %e,
                "cuBLASLt plan cache corrupt; ignoring and regenerating on next tune"
            );
            BTreeMap::new()
        }
    }
}

#[cfg(unix)]
struct AdvisoryFileLock {
    file: fs::File,
}

#[cfg(unix)]
impl AdvisoryFileLock {
    fn acquire(lock_path: &Path) -> Option<Self> {
        use std::os::unix::io::AsRawFd;
        let file = fs::OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(false)
            .open(lock_path)
            .ok()?;
        // SAFETY: `file.as_raw_fd()` is a valid open file descriptor owned by `file`.
        if unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX) } != 0 {
            return None;
        }
        Some(Self { file })
    }
}

#[cfg(unix)]
impl Drop for AdvisoryFileLock {
    fn drop(&mut self) {
        use std::os::unix::io::AsRawFd;
        // SAFETY: `self.file` remains open until after `drop` returns.
        unsafe {
            let _ = libc::flock(self.file.as_raw_fd(), libc::LOCK_UN);
        }
    }
}

fn should_replace_cached_entry(existing: &CachedEntry, incoming: &CachedEntry) -> bool {
    if !incoming.measured_ms.is_finite() || incoming.measured_ms <= 0.0 {
        return false;
    }
    if !existing.measured_ms.is_finite() || existing.measured_ms <= 0.0 {
        return true;
    }
    incoming.measured_ms <= existing.measured_ms
}

fn merge_entry(entries: &mut BTreeMap<String, CachedEntry>, key: String, incoming: CachedEntry) {
    if !incoming.measured_ms.is_finite() || incoming.measured_ms <= 0.0 {
        return;
    }
    match entries.get(&key) {
        Some(existing) if !should_replace_cached_entry(existing, &incoming) => {}
        _ => {
            entries.insert(key, incoming);
        }
    }
}

fn write_disk_cache_atomic(path: &Path, file_name: &str, entries: &BTreeMap<String, CachedEntry>) {
    let payload = DiskCacheFileRef {
        schema: SCHEMA_VERSION,
        entries,
    };
    let Ok(serialized) = serde_json::to_vec_pretty(&payload) else {
        return;
    };
    let seq = TMP_SEQ.fetch_add(1, Ordering::Relaxed);
    let tmp_name = format!(".{file_name}.tmp.{}.{seq}", std::process::id());
    let tmp_path = path.with_file_name(tmp_name);
    let write_res = fs::File::create(&tmp_path).and_then(|mut f| {
        f.write_all(&serialized)?;
        f.flush()?;
        f.sync_all()
    });
    if let Err(e) = write_res {
        let _ = fs::remove_file(&tmp_path);
        tracing::debug!(
            path = %tmp_path.display(),
            error = %e,
            "cuBLASLt plan cache temp write failed; keeping in-memory plan"
        );
        return;
    }
    if let Err(e) = fs::rename(&tmp_path, path) {
        let _ = fs::remove_file(&tmp_path);
        tracing::debug!(
            path = %path.display(),
            error = %e,
            "cuBLASLt plan cache atomic rename failed; keeping in-memory plan"
        );
        return;
    }
    // Best-effort parent directory sync for rename durability across power loss.
    #[cfg(unix)]
    {
        let sync_dir = path
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        if let Ok(dir) = fs::File::open(sync_dir) {
            let _ = dir.sync_all();
        }
    }
}

fn save_entries_atomic(path: &Path, entries: &mut BTreeMap<String, CachedEntry>) {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
        && let Err(e) = fs::create_dir_all(parent)
    {
        tracing::debug!(
            path = %parent.display(),
            error = %e,
            "cuBLASLt plan cache directory unavailable; skipping disk persist"
        );
        return;
    }

    let file_name = path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("cublaslt_plans_v2.json");
    #[cfg(unix)]
    let _file_lock = AdvisoryFileLock::acquire(&path.with_file_name(format!(".{file_name}.lock")));

    // Merge concurrent updates from sibling processes under advisory file lock
    // so multi-rank or parallel runs never clobber each other's newly tuned shapes
    // and slower/contended timings never overwrite faster verified timings.
    let mut merged = load_entries_from_path(path);
    for (k, v) in entries.iter() {
        merge_entry(&mut merged, k.clone(), v.clone());
    }
    *entries = merged;

    write_disk_cache_atomic(path, file_name, entries);
}

const BUNDLED_SEED_CACHE_PATH: &str = "/opt/flux/cache/cublaslt_plans_v2.json";

fn load_entries_with_seed(
    user_path: Option<&Path>,
    seed_path: Option<&Path>,
) -> BTreeMap<String, CachedEntry> {
    let mut entries = seed_path.map(load_entries_from_path).unwrap_or_default();
    if let Some(p) = user_path {
        for (k, v) in load_entries_from_path(p) {
            entries.insert(k, v);
        }
    }
    entries
}

fn store() -> &'static Mutex<PersistentStore> {
    STORE.get_or_init(|| {
        let path = resolve_cache_path();
        let explicit_override = std::env::var("ATLAS_CUBLASLT_CACHE_PATH")
            .ok()
            .is_some_and(|v| !v.trim().is_empty());
        let seed_path = if path.is_some() && !explicit_override {
            Some(Path::new(BUNDLED_SEED_CACHE_PATH))
        } else {
            None
        };
        let entries = load_entries_with_seed(path.as_deref(), seed_path);
        if !entries.is_empty()
            && let Some(ref p) = path
        {
            tracing::info!(
                path = %p.display(),
                cached_shapes = entries.len(),
                "loaded persistent cuBLASLt autotune plan cache"
            );
        }
        Mutex::new(PersistentStore {
            path,
            cache: PlanCache::from_entries(entries),
        })
    })
}

/// Resolved persistent-cache hit pointing to a validated index in the current
/// `cublasLtMatmulAlgoGetHeuristic` candidate array.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct LookupHit {
    pub candidate_idx: usize,
    pub measured_ms: f32,
    pub from_legacy_key: bool,
}

/// Match a cached 9-attribute `AlgoConfig` against heuristic candidates using
/// `extract` and verify runtime eligibility via `algo_ok(idx, cand)`.
fn find_matching_candidate<F, V>(
    target_config: &AlgoConfig,
    candidates: &[HeurResult],
    ws_size: usize,
    mut extract: F,
    mut algo_ok: V,
) -> Option<usize>
where
    F: FnMut(&AlignedAlgo) -> Option<AlgoConfig>,
    V: FnMut(usize, &HeurResult) -> bool,
{
    for (idx, cand) in candidates.iter().enumerate() {
        if cand.state != 0 || cand.workspace_size > ws_size {
            continue;
        }
        if extract(&cand.algo).as_ref() == Some(target_config) && algo_ok(idx, cand) {
            return Some(idx);
        }
    }
    None
}

fn lookup_candidate_from_entries<F, V>(
    canonical_entry: Option<&CachedEntry>,
    legacy_entry: Option<&CachedEntry>,
    key: &PlanCacheKey,
    candidates: &[HeurResult],
    ws_size: usize,
    mut extract: F,
    mut algo_ok: V,
) -> Option<LookupHit>
where
    F: FnMut(&AlignedAlgo) -> Option<AlgoConfig>,
    V: FnMut(usize, &HeurResult) -> bool,
{
    let guard_sm121a_large = is_sm121a_large_fp8_key(key);
    if let Some(entry) = canonical_entry {
        if guard_sm121a_large && is_legacy_64x_stages(entry.config[ALGO_CONFIG_IDX_STAGES_ID]) {
            tracing::warn!(
                key = %key.encoded(),
                stages_id = entry.config[ALGO_CONFIG_IDX_STAGES_ID],
                "cuBLASLt plan cache entry uses legacy 64x stages on SM{}{} large FP8 shape; self-healing via live re-tune",
                key.cc_major,
                key.cc_minor,
            );
        } else {
            match evaluate_entry_health(key, entry.measured_ms, CANONICAL_ROOFLINE_FLOOR) {
                HealthCheck::Healthy => {
                    if let Some(idx) = find_matching_candidate(
                        &entry.config,
                        candidates,
                        ws_size,
                        &mut extract,
                        &mut algo_ok,
                    ) {
                        return Some(LookupHit {
                            candidate_idx: idx,
                            measured_ms: entry.measured_ms,
                            from_legacy_key: false,
                        });
                    }
                }
                HealthCheck::InvalidTiming => {
                    tracing::warn!(
                        key = %key.encoded(),
                        measured_ms = f64::from(entry.measured_ms),
                        "cuBLASLt plan cache entry has invalid timing; self-healing via live re-tune"
                    );
                }
                HealthCheck::Degraded {
                    achieved_tflops,
                    floor_tflops,
                    roofline_eff,
                } => {
                    tracing::warn!(
                        key = %key.encoded(),
                        measured_ms = f64::from(entry.measured_ms),
                        "cuBLASLt plan cache entry degraded ({achieved_tflops:.1} TFLOPS < floor {floor_tflops:.1} TFLOPS, eff={roofline_eff:.2}); self-healing via live re-tune"
                    );
                }
            }
        }
    }

    // Backwards-compatible fallback to pre-`req`/`det` v2 entries (bundled seed
    // cache or pre-upgrade user cache). Reused only when passing the stricter
    // roofline health floor, rejecting legacy `STAGES_64x*` (`sm89_xmma`) entries
    // on large SM120/SM121a FP8 shapes, and passing the runtime `algo_ok` check.
    if let Some(entry) = legacy_entry {
        if guard_sm121a_large && is_legacy_64x_stages(entry.config[ALGO_CONFIG_IDX_STAGES_ID]) {
            tracing::warn!(
                key = %key.legacy_v2_encoded(),
                stages_id = entry.config[ALGO_CONFIG_IDX_STAGES_ID],
                "legacy cuBLASLt plan cache entry uses legacy 64x stages on SM{}{} large FP8 shape; upgrading via live re-tune",
                key.cc_major,
                key.cc_minor,
            );
        } else {
            match evaluate_entry_health(key, entry.measured_ms, LEGACY_ROOFLINE_FLOOR) {
                HealthCheck::Healthy => {
                    if let Some(idx) = find_matching_candidate(
                        &entry.config,
                        candidates,
                        ws_size,
                        &mut extract,
                        &mut algo_ok,
                    ) {
                        return Some(LookupHit {
                            candidate_idx: idx,
                            measured_ms: entry.measured_ms,
                            from_legacy_key: true,
                        });
                    }
                }
                HealthCheck::InvalidTiming => {
                    tracing::warn!(
                        key = %key.legacy_v2_encoded(),
                        measured_ms = f64::from(entry.measured_ms),
                        "legacy cuBLASLt plan cache entry has invalid timing; upgrading via live re-tune"
                    );
                }
                HealthCheck::Degraded {
                    achieved_tflops,
                    floor_tflops,
                    roofline_eff,
                } => {
                    tracing::warn!(
                        key = %key.legacy_v2_encoded(),
                        measured_ms = f64::from(entry.measured_ms),
                        requested = key.requested,
                        "legacy cuBLASLt plan cache entry below health floor ({achieved_tflops:.1} TFLOPS < floor {floor_tflops:.1} TFLOPS, eff={roofline_eff:.2}); upgrading via live re-tune"
                    );
                }
            }
        }
    }

    None
}

/// Look up a previously autotuned winner on disk/in-memory and resolve its
/// index inside `candidates` (`0..returned`) without launching GPU kernels.
pub(super) fn lookup_candidate<V>(
    key: &PlanCacheKey,
    candidates: &[HeurResult],
    ws_size: usize,
    algo_ok: V,
) -> Option<LookupHit>
where
    V: FnMut(usize, &HeurResult) -> bool,
{
    let canonical_key = key.encoded();
    let legacy_key = key.legacy_v2_encoded();
    let (canonical_entry, legacy_entry) = {
        let guard = store().lock();
        let canonical = guard.cache.entries.get(&canonical_key).cloned();
        let legacy = if legacy_key != canonical_key {
            guard.cache.entries.get(&legacy_key).cloned()
        } else {
            None
        };
        (canonical, legacy)
    };
    lookup_candidate_from_entries(
        canonical_entry.as_ref(),
        legacy_entry.as_ref(),
        key,
        candidates,
        ws_size,
        extract_algo_config,
        algo_ok,
    )
}

/// Look up the nearest healthy cross-`M` donor for `key` from the persistent
/// in-memory cache without launching GPU kernels (respects
/// `ATLAS_CUBLASLT_CROSS_M_REUSE`, enabled by default and disabled by `"0"`).
pub(super) fn lookup_nearest_m_donor(key: &PlanCacheKey) -> Option<CrossMDonor> {
    if !cross_m_reuse_enabled() || key.m < CROSS_M_MIN_M {
        return None;
    }
    let guard = store().lock();
    guard.cache.find_nearest_m_donor(key)
}

/// Record a newly autotuned winner both in memory and atomically on disk.
///
/// Refuses to persist timings that fail the hardware roofline health floor
/// (e.g. cold tuning conducted under heavy NVMe/LPDDR5X memory contention),
/// preventing poisoned measurements from polluting the persistent cache.
pub(super) fn record_winner(
    key: &PlanCacheKey,
    algo: &AlignedAlgo,
    heur_index: usize,
    measured_ms: f32,
) {
    match evaluate_entry_health(key, measured_ms, CANONICAL_ROOFLINE_FLOOR) {
        HealthCheck::Healthy => {}
        HealthCheck::InvalidTiming => {
            tracing::warn!(
                key = %key.encoded(),
                measured_ms = f64::from(measured_ms),
                "skipping cuBLASLt plan cache persist for invalid timing"
            );
            return;
        }
        HealthCheck::Degraded {
            achieved_tflops,
            floor_tflops,
            roofline_eff,
        } => {
            tracing::warn!(
                key = %key.encoded(),
                measured_ms = f64::from(measured_ms),
                "skipping cuBLASLt plan cache persist for degraded cold-tune timing ({achieved_tflops:.1} TFLOPS < floor {floor_tflops:.1} TFLOPS, eff={roofline_eff:.2}); keeping in-memory plan only"
            );
            return;
        }
    }

    let Some(config) = extract_algo_config(algo) else {
        return;
    };
    let encoded = key.encoded();
    let entry = CachedEntry {
        config,
        heur_index: heur_index as u32,
        measured_ms,
    };
    let mut guard = store().lock();
    merge_entry(&mut guard.cache.entries, encoded.clone(), entry.clone());
    if let Some(path) = guard.path.clone() {
        let mut delta = BTreeMap::new();
        delta.insert(encoded, entry);
        save_entries_atomic(&path, &mut delta);
        // Fold any newer/faster entries written by sibling processes into memory
        // without discarding bundled seed entries.
        for (k, v) in delta {
            merge_entry(&mut guard.cache.entries, k, v);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_key(m: u32, lt_version: usize) -> PlanCacheKey {
        PlanCacheKey {
            lt_version,
            cc_major: 12,
            cc_minor: 1,
            sm_count: 48,
            device_name: "NVIDIA_GB10".to_string(),
            ws_size: 64 * 1024 * 1024,
            kind: PlanKind::TunedActWeightT,
            m,
            n: 4096,
            k: 2560,
            at: 28,
            ldc: 4096,
            outer: false,
            tie_eps_bits: 0.20f32.to_bits(),
            requested: 16,
            deterministic: false,
        }
    }

    fn sample_sm121a_fp8_key(m: u32, n: u32, k: u32, ldc: u32) -> PlanCacheKey {
        PlanCacheKey {
            lt_version: 130101,
            cc_major: 12,
            cc_minor: 1,
            sm_count: 48,
            device_name: "NVIDIA_GB10".to_string(),
            ws_size: 64 * 1024 * 1024,
            kind: PlanKind::Fp8DeviceScaled,
            m,
            n,
            k,
            at: CUDA_R_8F_E4M3,
            ldc,
            outer: false,
            tie_eps_bits: 0.0f32.to_bits(),
            requested: 16,
            deterministic: true,
        }
    }

    #[test]
    fn key_encoding_isolates_library_version_device_shape_and_tie_band() {
        let base = sample_key(4096, 130002);
        let diff_ver = sample_key(4096, 130100);
        let diff_m = sample_key(8192, 130002);
        let mut diff_outer = base.clone();
        diff_outer.outer = true;
        let mut diff_dev = base.clone();
        diff_dev.device_name = "NVIDIA_B200".to_string();
        let mut diff_req = base.clone();
        diff_req.requested = 8;
        let mut diff_det = base.clone();
        diff_det.deterministic = true;
        assert_ne!(base.encoded(), diff_ver.encoded());
        assert_ne!(base.encoded(), diff_m.encoded());
        assert_ne!(base.encoded(), diff_outer.encoded());
        assert_ne!(base.encoded(), diff_dev.encoded());
        assert_ne!(base.encoded(), diff_req.encoded());
        assert_ne!(base.encoded(), diff_det.encoded());
        assert!(base.encoded().ends_with("|req=16|det=0"));
        assert_eq!(
            sample_sm121a_fp8_key(8192, 16384, 2560, 16384).legacy_v2_encoded(),
            "v2|lt=130101|sm=12.1:48:NVIDIA_GB10|ws=67108864|fp8_scaled|8192x16384x2560|at=28|ldc=16384|outer=0|eps=0x00000000"
        );
    }

    #[test]
    fn atomic_save_merges_concurrent_writers_and_recovers_from_corruption() {
        let dir = PathBuf::from(format!(
            "/tmp/atlas-cublaslt-cache-test-{}-{}",
            std::process::id(),
            TMP_SEQ.fetch_add(1, Ordering::Relaxed)
        ));
        let path = dir.join("cublaslt_plans_v2.json");

        std::thread::scope(|s| {
            for t in 0..8u32 {
                let p = path.clone();
                s.spawn(move || {
                    let k = sample_key(512 * (t + 1), 130002).encoded();
                    let mut local = BTreeMap::new();
                    local.insert(
                        k,
                        CachedEntry {
                            config: [23, 15 + t, 1, 0, 0, 0, 18, 1, 0],
                            heur_index: t,
                            measured_ms: 0.25 * (t as f32 + 1.0),
                        },
                    );
                    save_entries_atomic(&p, &mut local);
                });
            }
        });

        let reloaded = load_entries_from_path(&path);
        assert_eq!(
            reloaded.len(),
            8,
            "all 8 concurrent writers must be preserved under advisory flock"
        );
        for t in 0..8u32 {
            let k = sample_key(512 * (t + 1), 130002).encoded();
            assert_eq!(reloaded[&k].config, [23, 15 + t, 1, 0, 0, 0, 18, 1, 0]);
        }

        // Corrupt JSON must degrade cleanly to empty map without panicking.
        fs::write(&path, b"{truncated_json").expect("write corrupt test file");
        assert!(load_entries_from_path(&path).is_empty());

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn candidate_matcher_disambiguates_blackwell_cluster_shape_and_algo_check() {
        let mut cands = [HeurResult {
            algo: AlignedAlgo::ZERO,
            workspace_size: 32 * 1024 * 1024,
            state: 0,
            waves_count: 0.0,
            reserved: [0; 4],
        }; 4];
        // Candidate 0: state != 0
        cands[0].algo.0[0] = 10;
        cands[0].algo.0[8] = 2;
        cands[0].state = 7;
        // Candidate 1: same [0..=6], but cluster_shape_id = 3 (not target 2)
        cands[1].algo.0[0] = 10;
        cands[1].algo.0[8] = 3;
        // Candidate 2: exact 9-attr match, but rejected by cublasLtMatmulAlgoCheck predicate
        cands[2].algo.0[0] = 10;
        cands[2].algo.0[8] = 2;
        cands[2].algo.0[63] = 1; // mark rejected by algo_ok
        // Candidate 3: exact 9-attr match AND accepted by algo_ok
        cands[3].algo.0[0] = 10;
        cands[3].algo.0[8] = 2;

        let idx = find_matching_candidate(
            &[10, 1, 1, 0, 0, 0, 1, 4, 2],
            &cands,
            64 * 1024 * 1024,
            |a| Some([u32::from(a.0[0]), 1, 1, 0, 0, 0, 1, 4, u32::from(a.0[8])]),
            |_idx, cand| cand.algo.0[63] == 0,
        );
        assert_eq!(idx, Some(3));
    }

    #[test]
    #[ignore = "requires exclusive GB10 GPU lock"]
    fn real_gpu_cold_vs_warm_plan_cache_roundtrip() {
        use crate::gpu::GpuBackend;
        use std::process::Command;
        use std::time::Instant;

        if let Ok(phase) = std::env::var("ATLAS_CUBLASLT_TEST_PHASE") {
            let gpu =
                crate::cuda_backend::AtlasCudaBackend::new(0, &[]).expect("init AtlasCudaBackend");
            // Warm up cublasLt context + one-time libcublasLt heuristic table load outside t0.
            let c = super::super::ctx().expect("init cublasLt ctx");
            let ws = c
                .workspace_for_stream(0)
                .expect("cublasLt stream 0 workspace");
            super::super::bf16_gemm_act_weight_t(ws, ws, ws, 16, 16, 16, 0)
                .expect("warmup cublasLt heuristic table");
            gpu.synchronize(0).expect("sync warmup");
            let t0 = Instant::now();
            let p_bf16 =
                super::super::tuned_plan_dt(1536, 2560, 2560, super::super::CUDA_R_16BF, 2, 2560)
                    .expect("tuned_plan_dt bf16");
            let p_fp8 = super::super::tuned_plan_dt(
                1536,
                4096,
                2560,
                super::super::CUDA_R_8F_E4M3,
                1,
                4096,
            )
            .expect("tuned_plan_dt fp8");
            let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
            let cfg_bf16 = extract_algo_config(&p_bf16.algo).expect("extract bf16 config");
            let cfg_fp8 = extract_algo_config(&p_fp8.algo).expect("extract fp8 config");
            println!(
                "CUBLASLT_PHASE={phase} elapsed_ms={elapsed_ms:.3} bf16={cfg_bf16:?} fp8={cfg_fp8:?}"
            );
            return;
        }

        let dir = PathBuf::from(format!(
            "/tmp/atlas-cublaslt-gpu-roundtrip-{}-{}",
            std::process::id(),
            TMP_SEQ.fetch_add(1, Ordering::Relaxed)
        ));
        let cache_file = dir.join("cublaslt_plans_v2.json");
        let exe = std::env::current_exe().expect("current test binary path");

        let run_phase = |phase: &str| -> String {
            let out = Command::new(&exe)
                .args([
                    "--exact",
                    "cublaslt::plan_cache::tests::real_gpu_cold_vs_warm_plan_cache_roundtrip",
                    "--ignored",
                    "--nocapture",
                ])
                .env("ATLAS_CUBLASLT_TEST_PHASE", phase)
                .env("ATLAS_CUBLASLT_CACHE", "1")
                .env("ATLAS_CUBLASLT_CACHE_PATH", &cache_file)
                .env("ATLAS_CUBLASLT_AUTOTUNE", "1")
                .output()
                .expect("spawn child test phase");
            assert!(
                out.status.success(),
                "child phase {phase} failed:\nstdout:\n{}\nstderr:\n{}",
                String::from_utf8_lossy(&out.stdout),
                String::from_utf8_lossy(&out.stderr)
            );
            String::from_utf8_lossy(&out.stdout).into_owned()
        };

        let cold_out = run_phase("cold");
        assert!(cache_file.exists(), "cold phase must persist cache file");
        let entries = load_entries_from_path(&cache_file);
        assert_eq!(entries.len(), 2, "expected 2 persisted GEMM plans");

        let warm_out = run_phase("warm");
        println!("COLD:\n{cold_out}\nWARM:\n{warm_out}");

        let parse_line = |s: &str, prefix: &str| -> (f64, String) {
            let line = s
                .lines()
                .find(|l| l.starts_with(prefix))
                .expect("find CUBLASLT_PHASE line");
            let ms_part = line
                .split_whitespace()
                .nth(1)
                .and_then(|kv| kv.strip_prefix("elapsed_ms="))
                .expect("parse elapsed_ms")
                .parse::<f64>()
                .expect("f64 ms");
            let cfg_part = line.split_once("bf16=").expect("bf16=").1.to_string();
            (ms_part, cfg_part)
        };

        let (cold_ms, cold_cfgs) = parse_line(&cold_out, "CUBLASLT_PHASE=cold");
        let (warm_ms, warm_cfgs) = parse_line(&warm_out, "CUBLASLT_PHASE=warm");
        assert_eq!(
            cold_cfgs, warm_cfgs,
            "warm reload must restore exact [u32; 9] algorithm configs"
        );
        assert!(
            warm_ms < 5.0 && warm_ms < cold_ms * 0.25,
            "expected warm cache lookup ({warm_ms:.3} ms) << cold autotune ({cold_ms:.3} ms)"
        );

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn seed_cache_populates_missing_entries_and_user_cache_wins_on_conflict() {
        let dir = std::env::temp_dir().join(format!(
            "atlas_cublaslt_seed_test_{}_{}",
            std::process::id(),
            TMP_SEQ.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = fs::create_dir_all(&dir);
        let seed_path = dir.join("seed.json");
        let user_path = dir.join("user.json");

        let mut seed_entries = BTreeMap::new();
        seed_entries.insert(
            "k_seed_only".to_string(),
            CachedEntry {
                config: [1; 9],
                heur_index: 0,
                measured_ms: 1.5,
            },
        );
        seed_entries.insert(
            "k_shared".to_string(),
            CachedEntry {
                config: [2; 9],
                heur_index: 1,
                measured_ms: 2.5,
            },
        );
        save_entries_atomic(&seed_path, &mut seed_entries);

        let mut user_entries = BTreeMap::new();
        user_entries.insert(
            "k_shared".to_string(),
            CachedEntry {
                config: [9; 9],
                heur_index: 3,
                measured_ms: 0.9,
            },
        );
        user_entries.insert(
            "k_user_only".to_string(),
            CachedEntry {
                config: [7; 9],
                heur_index: 2,
                measured_ms: 1.1,
            },
        );
        save_entries_atomic(&user_path, &mut user_entries);

        let merged = load_entries_with_seed(Some(&user_path), Some(&seed_path));
        assert_eq!(merged.len(), 3);
        assert_eq!(merged["k_seed_only"].config, [1; 9]);
        assert_eq!(merged["k_shared"].config, [9; 9]);
        assert_eq!(merged["k_user_only"].config, [7; 9]);

        let seed_only = load_entries_with_seed(Some(&dir.join("missing.json")), Some(&seed_path));
        assert_eq!(seed_only.len(), 2);
        assert_eq!(seed_only["k_shared"].config, [2; 9]);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn shipped_repro_kit_seed_cache_parses_with_host_and_container_shapes() {
        let seed_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../repro-kit/cache/cublaslt_plans_v2.json");
        assert!(
            seed_path.exists(),
            "expected shipped seed cache at {}",
            seed_path.display()
        );
        let entries = load_entries_from_path(&seed_path);
        assert!(
            entries.len() >= 60,
            "expected shipped seed cache to contain >= 60 tuned shapes (got {}); check top-level `schema: 2` key",
            entries.len()
        );
        assert!(
            entries.keys().any(|k| k.contains("|lt=130100|")),
            "expected host CUDA 13.0 (lt=130100) entries in shipped seed cache"
        );
        assert!(
            entries.keys().any(|k| k.contains("|lt=130101|")),
            "expected container CUDA 13.0 (lt=130101) entries in shipped seed cache"
        );
    }

    #[test]
    fn serialization_roundtrip_atomic_persistence_and_monotonic_merge() {
        let dir = std::env::temp_dir().join(format!(
            "atlas_cublaslt_atomic_merge_{}_{}",
            std::process::id(),
            TMP_SEQ.fetch_add(1, Ordering::Relaxed)
        ));
        let path = dir.join("cublaslt_plans_v2.json");
        let key = sample_sm121a_fp8_key(8192, 16384, 2560, 16384).encoded();

        // Persist an initially slower entry, then verify a faster entry replaces it,
        // while a slower contended entry cannot overwrite the faster verified one.
        let slow = CachedEntry {
            config: [67, 467, 1, 0, 0, 30, 36, 0, 0],
            heur_index: 0,
            measured_ms: 4.50,
        };
        let fast = CachedEntry {
            config: [67, 23, 1, 0, 0, 31, 36, 0, 0],
            heur_index: 9,
            measured_ms: 3.45,
        };
        let poisoned = CachedEntry {
            config: [35, 20, 1, 0, 1, 0, 15, 0, 0],
            heur_index: 2,
            measured_ms: 13.294,
        };

        let mut initial = BTreeMap::new();
        initial.insert(key.clone(), slow);
        save_entries_atomic(&path, &mut initial);

        let mut faster_delta = BTreeMap::new();
        faster_delta.insert(key.clone(), fast.clone());
        save_entries_atomic(&path, &mut faster_delta);

        let mut slower_delta = BTreeMap::new();
        slower_delta.insert(key.clone(), poisoned);
        save_entries_atomic(&path, &mut slower_delta);

        let raw = fs::read(&path).expect("read serialized cache file");
        let parsed: DiskCacheFile =
            serde_json::from_slice(&raw).expect("parse serialized DiskCacheFile");
        assert_eq!(parsed.schema, SCHEMA_VERSION);
        let stored = parsed.entries.get(&key).expect("key present after merge");
        assert_eq!(stored.config, fast.config);
        assert_eq!(stored.heur_index, fast.heur_index);
        assert!(
            (stored.measured_ms - fast.measured_ms).abs() < 1e-6,
            "faster timing must win over slower contended timing, got {}",
            stored.measured_ms
        );

        // Atomic write-to-temp + rename must leave zero orphaned `.tmp.` files.
        let leftover_temps: Vec<_> = fs::read_dir(&dir)
            .expect("list temp cache dir")
            .filter_map(|entry| entry.ok())
            .filter(|entry| entry.file_name().to_string_lossy().contains(".tmp."))
            .collect();
        assert!(
            leftover_temps.is_empty(),
            "found leftover temp files after atomic rename: {leftover_temps:?}"
        );

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn self_healing_rejects_degraded_cold_tune_and_reuses_healthy_legacy_plans() {
        let mut cands = [HeurResult {
            algo: AlignedAlgo::ZERO,
            workspace_size: 16 * 1024 * 1024,
            state: 0,
            waves_count: 0.0,
            reserved: [0; 4],
        }; 2];
        cands[0].algo.0[0] = 67;
        cands[1].algo.0[0] = 35;
        let extract = |a: &AlignedAlgo| Some([u32::from(a.0[0]), 467, 1, 0, 0, 30, 36, 0, 0]);

        // 1. Degraded 16384x16384x2560 entry (~103 TFLOPS at 13.294ms under NVMe contention)
        // must be rejected on both canonical and legacy floors so live re-tuning heals it.
        let poisoned_key = sample_sm121a_fp8_key(16384, 16384, 2560, 16384);
        assert!(matches!(
            evaluate_entry_health(&poisoned_key, 13.294, CANONICAL_ROOFLINE_FLOOR),
            HealthCheck::Degraded { .. }
        ));
        let poisoned_entry = CachedEntry {
            config: [67, 467, 1, 0, 0, 30, 36, 0, 0],
            heur_index: 0,
            measured_ms: 13.294,
        };
        assert!(
            lookup_candidate_from_entries(
                Some(&poisoned_entry),
                Some(&poisoned_entry),
                &poisoned_key,
                &cands,
                64 * 1024 * 1024,
                extract,
                |_idx, _cand| true,
            )
            .is_none(),
            "degraded poisoned entry must self-heal via live re-tune"
        );

        // 2. Healthy golden prefill shapes pass both canonical and legacy floors.
        let golden_cases = [
            (8192, 16384, 2560, 16384, 3.514f32),
            (8192, 2560, 6144, 2560, 1.322f32),
            (8192, 12288, 2560, 13312, 2.627f32),
            (16384, 2560, 6144, 2560, 2.658f32),
            (8192, 512, 2560, 512, 0.138f32),
        ];
        for (m, n, k, ldc, ms) in golden_cases {
            let gkey = sample_sm121a_fp8_key(m, n, k, ldc);
            assert_eq!(
                evaluate_entry_health(&gkey, ms, LEGACY_ROOFLINE_FLOOR),
                HealthCheck::Healthy,
                "healthy golden shape {m}x{n}x{k} @ {ms}ms must pass floor"
            );
        }

        // 3. Healthy legacy v2 entry is reused with 0ms GPU tune; canonical req=16 wins when present.
        let golden_key = sample_sm121a_fp8_key(8192, 16384, 2560, 16384);
        let legacy_fast = CachedEntry {
            config: [67, 467, 1, 0, 0, 30, 36, 0, 0],
            heur_index: 0,
            measured_ms: 3.514,
        };
        let canonical_faster = CachedEntry {
            config: [35, 467, 1, 0, 0, 30, 36, 0, 0],
            heur_index: 1,
            measured_ms: 3.310,
        };
        let legacy_hit = lookup_candidate_from_entries(
            None,
            Some(&legacy_fast),
            &golden_key,
            &cands,
            64 * 1024 * 1024,
            extract,
            |_idx, _cand| true,
        )
        .expect("healthy legacy entry must hit");
        assert_eq!(legacy_hit.candidate_idx, 0);
        assert!(legacy_hit.from_legacy_key);

        let canonical_hit = lookup_candidate_from_entries(
            Some(&canonical_faster),
            Some(&legacy_fast),
            &golden_key,
            &cands,
            64 * 1024 * 1024,
            extract,
            |_idx, _cand| true,
        )
        .expect("canonical req=16 entry must hit");
        assert_eq!(canonical_hit.candidate_idx, 1);
        assert!(!canonical_hit.from_legacy_key);

        // 4. Suboptimal legacy narrow-search entry (~142.5 TFLOPS on 15104x16384x2560)
        // misses the stricter legacy floor so it upgrades via req=16 search.
        let subopt_key = sample_sm121a_fp8_key(15104, 16384, 2560, 16384);
        assert!(matches!(
            evaluate_entry_health(&subopt_key, 8.889, LEGACY_ROOFLINE_FLOOR),
            HealthCheck::Degraded { .. }
        ));

        // 5. Non-SM121a GPUs pass all finite positive timings; invalid timings always fail.
        let mut h100_key = poisoned_key;
        h100_key.cc_major = 9;
        h100_key.cc_minor = 0;
        assert_eq!(
            evaluate_entry_health(&h100_key, 13.294, CANONICAL_ROOFLINE_FLOOR),
            HealthCheck::Healthy
        );
        for bad_ms in [0.0f32, -1.0, f32::NAN, f32::INFINITY] {
            assert_eq!(
                evaluate_entry_health(&h100_key, bad_ms, CANONICAL_ROOFLINE_FLOOR),
                HealthCheck::InvalidTiming,
                "invalid timing {bad_ms} must be rejected"
            );
        }
    }

    #[test]
    fn lookup_passes_candidate_index_for_deterministic_filtering() {
        let mut cands = [HeurResult {
            algo: AlignedAlgo::ZERO,
            workspace_size: 8 * 1024 * 1024,
            state: 0,
            waves_count: 0.0,
            reserved: [0; 4],
        }; 2];
        cands[0].algo.0[0] = 67;
        cands[1].algo.0[0] = 35;
        let extract = |a: &AlignedAlgo| Some([u32::from(a.0[0]), 20, 2, 1, 0, 0, 15, 0, 0]);

        // Index 0 (`heur0`) allowed even if non-deterministic; index 1 rejected when predicate forbids idx > 0.
        let hit0 = find_matching_candidate(
            &[67, 20, 2, 1, 0, 0, 15, 0, 0],
            &cands,
            64 * 1024 * 1024,
            extract,
            |idx, _cand| idx == 0,
        );
        assert_eq!(hit0, Some(0));

        let hit1 = find_matching_candidate(
            &[35, 20, 2, 1, 0, 0, 15, 0, 0],
            &cands,
            64 * 1024 * 1024,
            extract,
            |idx, _cand| idx == 0,
        );
        assert_eq!(hit1, None);
    }

    #[test]
    fn find_nearest_m_donor_exact_priority_nearest_and_tie_break() {
        let nvjet_8k: AlgoConfig = [67, 467, 1, 0, 0, 30, 36, 0, 0];
        let nvjet_8k_legacy: AlgoConfig = [67, 416, 1, 0, 0, 30, 36, 0, 0];
        let nvjet_8448: AlgoConfig = [67, 23, 1, 0, 0, 31, 36, 0, 0];
        let nvjet_16k: AlgoConfig = [67, 406, 1, 0, 0, 31, 36, 0, 0];

        let mut entries = BTreeMap::new();
        let k8 = sample_sm121a_fp8_key(8192, 16384, 2560, 16384);
        let k8448 = sample_sm121a_fp8_key(8448, 16384, 2560, 16384);
        let k16 = sample_sm121a_fp8_key(16384, 16384, 2560, 16384);

        entries.insert(
            k8.legacy_v2_encoded(),
            CachedEntry {
                config: nvjet_8k_legacy,
                heur_index: 1,
                measured_ms: 3.550,
            },
        );
        entries.insert(
            k8.encoded(),
            CachedEntry {
                config: nvjet_8k,
                heur_index: 0,
                measured_ms: 3.514,
            },
        );
        entries.insert(
            k8448.encoded(),
            CachedEntry {
                config: nvjet_8448,
                heur_index: 0,
                measured_ms: 3.700,
            },
        );
        entries.insert(
            k16.encoded(),
            CachedEntry {
                config: nvjet_16k,
                heur_index: 0,
                measured_ms: 7.050,
            },
        );
        let cache = PlanCache::from_entries(entries);

        // Exact match takes priority over nearby donors, and canonical beats legacy at equal M.
        let exact = cache
            .find_nearest_m_donor(&k8)
            .expect("exact M=8192 must hit");
        assert_eq!(exact.donor_m, 8192);
        assert_eq!(exact.config, nvjet_8k);
        assert!(!exact.from_legacy_key);

        // Nearest M wins when distances differ.
        let near_8k = cache
            .find_nearest_m_donor(&sample_sm121a_fp8_key(10240, 16384, 2560, 16384))
            .expect("M=10240 must hit M=8448");
        assert_eq!(near_8k.donor_m, 8448);

        let near_16k = cache
            .find_nearest_m_donor(&sample_sm121a_fp8_key(14336, 16384, 2560, 16384))
            .expect("M=14336 must hit M=16384");
        assert_eq!(near_16k.donor_m, 16384);

        // Equidistant tie-break favors `donor.m >= m`: M=12288 is 4096 away from both 8192 and 16384.
        let mut tie_entries = BTreeMap::new();
        tie_entries.insert(
            k8.encoded(),
            CachedEntry {
                config: nvjet_8k,
                heur_index: 0,
                measured_ms: 3.514,
            },
        );
        tie_entries.insert(
            k16.encoded(),
            CachedEntry {
                config: nvjet_16k,
                heur_index: 0,
                measured_ms: 7.050,
            },
        );
        let tie_cache = PlanCache::from_entries(tie_entries);
        let tie = tie_cache
            .find_nearest_m_donor(&sample_sm121a_fp8_key(12288, 16384, 2560, 16384))
            .expect("equidistant M=12288 must hit");
        assert_eq!(
            tie.donor_m, 16384,
            "equidistant tie-break must favor donor.m >= target.m"
        );
    }

    #[test]
    fn find_nearest_m_donor_enforces_2x_ratio_and_min_m_bounds() {
        let nvjet: AlgoConfig = [67, 467, 1, 0, 0, 30, 36, 0, 0];

        // Donor at M=8192 matches M=7168, M=15360, and exact 2x bound M=16384, but rejects M=16385.
        let mut entries_8k = BTreeMap::new();
        entries_8k.insert(
            sample_sm121a_fp8_key(8192, 16384, 2560, 16384).encoded(),
            CachedEntry {
                config: nvjet,
                heur_index: 0,
                measured_ms: 3.514,
            },
        );
        let cache_8k = PlanCache::from_entries(entries_8k);
        for target_m in [7168u32, 15360, 16384] {
            let hit = cache_8k
                .find_nearest_m_donor(&sample_sm121a_fp8_key(target_m, 16384, 2560, 16384))
                .unwrap_or_else(|| panic!("donor 8192 must match target M={target_m}"));
            assert_eq!(hit.donor_m, 8192);
        }
        assert!(
            cache_8k
                .find_nearest_m_donor(&sample_sm121a_fp8_key(16385, 16384, 2560, 16384))
                .is_none(),
            "M=16385 exceeds 2x ratio against donor 8192"
        );

        // Donor at M=16384 matches M=14885, but rejects M=4096 because 4x > 2x.
        let mut entries_16k = BTreeMap::new();
        entries_16k.insert(
            sample_sm121a_fp8_key(16384, 16384, 2560, 16384).encoded(),
            CachedEntry {
                config: nvjet,
                heur_index: 0,
                measured_ms: 7.050,
            },
        );
        let cache_16k = PlanCache::from_entries(entries_16k);
        let hit_14885 = cache_16k
            .find_nearest_m_donor(&sample_sm121a_fp8_key(14885, 16384, 2560, 16384))
            .expect("donor 16384 must match target M=14885");
        assert_eq!(hit_14885.donor_m, 16384);
        assert!(
            cache_16k
                .find_nearest_m_donor(&sample_sm121a_fp8_key(4096, 16384, 2560, 16384))
                .is_none(),
            "donor 16384 must reject target M=4096 (4x > 2x)"
        );

        // Symmetric check: donor at M=4096 rejects target M=16384 (4x > 2x).
        let mut entries_4k = BTreeMap::new();
        entries_4k.insert(
            sample_sm121a_fp8_key(4096, 16384, 2560, 16384).encoded(),
            CachedEntry {
                config: nvjet,
                heur_index: 0,
                measured_ms: 1.800,
            },
        );
        let cache_4k = PlanCache::from_entries(entries_4k);
        assert!(
            cache_4k
                .find_nearest_m_donor(&sample_sm121a_fp8_key(16384, 16384, 2560, 16384))
                .is_none(),
            "donor 4096 must reject target M=16384 (4x > 2x)"
        );

        // Minimum M >= 1024 bound applies to both target and donor.
        let mut entries_small = BTreeMap::new();
        entries_small.insert(
            sample_sm121a_fp8_key(512, 16384, 2560, 16384).encoded(),
            CachedEntry {
                config: nvjet,
                heur_index: 0,
                measured_ms: 0.260,
            },
        );
        let cache_small = PlanCache::from_entries(entries_small);
        assert!(
            cache_small
                .find_nearest_m_donor(&sample_sm121a_fp8_key(1024, 16384, 2560, 16384))
                .is_none(),
            "donor M=512 (< 1024) must not donate to M=1024"
        );
        let mut entries_1k = BTreeMap::new();
        entries_1k.insert(
            sample_sm121a_fp8_key(1024, 16384, 2560, 16384).encoded(),
            CachedEntry {
                config: nvjet,
                heur_index: 0,
                measured_ms: 0.460,
            },
        );
        let cache_1k = PlanCache::from_entries(entries_1k);
        assert!(
            cache_1k
                .find_nearest_m_donor(&sample_sm121a_fp8_key(512, 16384, 2560, 16384))
                .is_none(),
            "target M=512 (< 1024) must not use cross-M donor"
        );
    }

    #[test]
    fn find_nearest_m_donor_rejects_any_non_m_field_mismatch() {
        let nvjet: AlgoConfig = [67, 467, 1, 0, 0, 30, 36, 0, 0];
        let donor_key = sample_sm121a_fp8_key(8192, 16384, 2560, 16384);
        let mut entries = BTreeMap::new();
        entries.insert(
            donor_key.encoded(),
            CachedEntry {
                config: nvjet,
                heur_index: 0,
                measured_ms: 3.514,
            },
        );
        // Also add a wrong-schema-version entry (v1) to ensure version mismatch is rejected.
        entries.insert(
            donor_key.encoded().replacen("v2|", "v1|", 1),
            CachedEntry {
                config: nvjet,
                heur_index: 0,
                measured_ms: 3.514,
            },
        );
        let cache = PlanCache::from_entries(entries);
        let base_target = sample_sm121a_fp8_key(15360, 16384, 2560, 16384);
        assert!(
            cache.find_nearest_m_donor(&base_target).is_some(),
            "baseline matching target must hit"
        );

        let mut mismatches: Vec<(&str, PlanCacheKey)> = Vec::new();
        let mut push = |label: &'static str, mutate: fn(&mut PlanCacheKey)| {
            let mut k = base_target.clone();
            mutate(&mut k);
            mismatches.push((label, k));
        };
        push("n", |k| k.n = 12288);
        push("k", |k| k.k = 6144);
        push("ldc", |k| k.ldc = 17408);
        push("outer", |k| k.outer = true);
        push("op_kind", |k| k.kind = PlanKind::TunedActWeightT);
        push("cc_major", |k| k.cc_major = 9);
        push("cc_minor", |k| k.cc_minor = 0);
        push("sm_count", |k| k.sm_count = 64);
        push("device_name", |k| k.device_name = "NVIDIA_B200".to_string());
        push("lt_version", |k| k.lt_version = 130100);
        push("ws_size", |k| k.ws_size = 32 * 1024 * 1024);
        push("a_type", |k| k.at = 14);
        push("tie_eps_bits", |k| k.tie_eps_bits = 0.20f32.to_bits());
        push("requested", |k| k.requested = 8);
        push("deterministic", |k| k.deterministic = false);

        for (label, target) in mismatches {
            assert!(
                cache.find_nearest_m_donor(&target).is_none(),
                "mismatch on `{label}` must reject cross-M donor"
            );
        }

        // Schema v1-only cache must never match v2 target.
        let mut v1_only = BTreeMap::new();
        v1_only.insert(
            donor_key.encoded().replacen("v2|", "v1|", 1),
            CachedEntry {
                config: nvjet,
                heur_index: 0,
                measured_ms: 3.514,
            },
        );
        assert!(
            PlanCache::from_entries(v1_only)
                .find_nearest_m_donor(&base_target)
                .is_none(),
            "schema v1 entry must be rejected for v2 target"
        );
    }

    #[test]
    fn find_nearest_m_donor_rejects_unhealthy_and_sm89_xmma_donors() {
        let nvjet: AlgoConfig = [67, 467, 1, 0, 0, 30, 36, 0, 0];
        let sm89_xmma: AlgoConfig = [35, 20, 1, 0, 1, 0, 15, 0, 0];
        let split_k_cfg: AlgoConfig = [67, 467, 2, 2, 0, 30, 36, 0, 0];
        let target = sample_sm121a_fp8_key(15360, 16384, 2560, 16384);

        // 1. Degraded canonical donor + below-floor legacy donor + invalid timings + sm89_xmma
        //    + non-deterministic split-K legacy donor must ALL be rejected when no healthy
        //    native Blackwell donor exists.
        let mut bad_entries = BTreeMap::new();
        bad_entries.insert(
            sample_sm121a_fp8_key(16384, 16384, 2560, 16384).encoded(),
            CachedEntry {
                config: nvjet,
                heur_index: 0,
                measured_ms: 13.294, // ~103 TFLOPS, below canonical roofline floor
            },
        );
        bad_entries.insert(
            sample_sm121a_fp8_key(15104, 16384, 2560, 16384).legacy_v2_encoded(),
            CachedEntry {
                config: nvjet,
                heur_index: 2,
                measured_ms: 8.889, // ~142.5 TFLOPS, below legacy roofline floor
            },
        );
        bad_entries.insert(
            sample_sm121a_fp8_key(15616, 16384, 2560, 16384).legacy_v2_encoded(),
            CachedEntry {
                config: sm89_xmma,
                heur_index: 3,
                measured_ms: 7.337, // STAGES_64x3 Ada sm89_xmma blocked on SM121a large FP8
            },
        );
        bad_entries.insert(
            sample_sm121a_fp8_key(14848, 16384, 2560, 16384).legacy_v2_encoded(),
            CachedEntry {
                config: split_k_cfg,
                heur_index: 1,
                measured_ms: 6.400, // non-deterministic split_k=2 rejected for det=1 target
            },
        );
        bad_entries.insert(
            sample_sm121a_fp8_key(14080, 16384, 2560, 16384).encoded(),
            CachedEntry {
                config: nvjet,
                heur_index: 0,
                measured_ms: 0.0, // invalid timing
            },
        );
        assert!(
            PlanCache::from_entries(bad_entries.clone())
                .find_nearest_m_donor(&target)
                .is_none(),
            "degraded, invalid, sm89_xmma, and split-K donors must all be rejected"
        );

        // 2. Adding a healthy native Blackwell M=8192 donor rescues M=15360 despite closer bad entries.
        bad_entries.insert(
            sample_sm121a_fp8_key(8192, 16384, 2560, 16384).legacy_v2_encoded(),
            CachedEntry {
                config: nvjet,
                heur_index: 0,
                measured_ms: 3.557,
            },
        );
        let hit = PlanCache::from_entries(bad_entries)
            .find_nearest_m_donor(&target)
            .expect("healthy M=8192 nvjet donor must rescue M=15360");
        assert_eq!(hit.donor_m, 8192);
        assert_eq!(hit.config, nvjet);
        assert!(hit.from_legacy_key);
    }

    #[test]
    fn find_nearest_m_donor_reuses_shipped_seed_cache_without_sm89_xmma() {
        let seed_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../repro-kit/cache/cublaslt_plans_v2.json");
        let cache = PlanCache::from_entries(load_entries_from_path(&seed_path));

        let nvjet_256x128_c31: AlgoConfig = [67, 24, 1, 0, 0, 31, 36, 0, 0];
        let nvjet_256x128_c144: AlgoConfig = [67, 24, 1, 0, 0, 144, 36, 0, 0];

        // Container (lt=130101) and host (lt=130100) CUDA 13.0: non-pow2 and exact prefill M
        // values adopt healthy native SM121 256x128 (tile=24, stages=36) plans directly.
        for lt_version in [130101usize, 130100] {
            for (target_m, expected_donor_m) in [
                (7168u32, 8192u32),
                (8192, 8192),
                (14885, 15360),
                (15360, 15360),
                (16128, 16384),
                (16384, 16384),
            ] {
                let mut target = sample_sm121a_fp8_key(target_m, 16384, 2560, 16384);
                target.lt_version = lt_version;
                let hit = cache.find_nearest_m_donor(&target).unwrap_or_else(|| {
                    panic!("lt={lt_version} M={target_m} must hit M={expected_donor_m} seed donor")
                });
                assert_eq!(hit.donor_m, expected_donor_m);
                assert!(!is_legacy_64x_stages(hit.config[ALGO_CONFIG_IDX_STAGES_ID]));
                assert_eq!(hit.config, nvjet_256x128_c31);
            }

            // All 3 critical 8K/16K shapes resolve to exact native SM121 plans (no remote donor).
            for (m, n, k, ldc, expected_cfg) in [
                (16384u32, 16384u32, 2560u32, 16384u32, nvjet_256x128_c31),
                (16384, 2560, 6144, 2560, nvjet_256x128_c144),
                (16384, 12288, 2560, 13312, nvjet_256x128_c31),
                (8192, 16384, 2560, 16384, nvjet_256x128_c31),
                (8192, 2560, 6144, 2560, nvjet_256x128_c144),
                (8192, 12288, 2560, 13312, nvjet_256x128_c31),
            ] {
                let mut target = sample_sm121a_fp8_key(m, n, k, ldc);
                target.lt_version = lt_version;
                let hit = cache.find_nearest_m_donor(&target).unwrap_or_else(|| {
                    panic!("lt={lt_version} {m}x{n}x{k} ldc={ldc} must hit exact native SM121 seed")
                });
                assert_eq!(hit.donor_m, m);
                assert!(!is_legacy_64x_stages(hit.config[ALGO_CONFIG_IDX_STAGES_ID]));
                assert_eq!(hit.config, expected_cfg);
            }
        }

        // Host CUDA 13.0 (lt=130100): M=14336 resolves to nearest healthy M=14080 seed donor.
        let mut host_target = sample_sm121a_fp8_key(14336, 16384, 2560, 16384);
        host_target.lt_version = 130100;
        let host_hit = cache
            .find_nearest_m_donor(&host_target)
            .expect("lt=130100 M=14336 must hit healthy M=14080 seed donor");
        assert_eq!(host_hit.donor_m, 14080);
        assert!(!is_legacy_64x_stages(
            host_hit.config[ALGO_CONFIG_IDX_STAGES_ID]
        ));
        assert_eq!(host_hit.config, [67, 467, 1, 0, 0, 30, 36, 0, 0]);
    }

    #[test]
    fn lookup_rejects_legacy_64x_stages_on_sm121a_large_fp8_shapes() {
        let sm89_xmma: AlgoConfig = [35, 20, 1, 0, 1, 0, 15, 0, 0];
        let sm89_auto: AlgoConfig = [35, 20, 1, 0, 1, 0, STAGES_64X_AUTO, 0, 0];
        let nvjet: AlgoConfig = [67, 467, 1, 0, 0, 30, 36, 0, 0];

        let mut cands = [HeurResult {
            algo: AlignedAlgo::ZERO,
            workspace_size: 16 * 1024 * 1024,
            state: 0,
            waves_count: 0.0,
            reserved: [0; 4],
        }; 3];
        cands[0].algo.0[0] = 0;
        cands[1].algo.0[0] = 1;
        cands[2].algo.0[0] = 2;
        let extract = |a: &AlignedAlgo| match a.0[0] {
            0 => Some(sm89_xmma),
            1 => Some(sm89_auto),
            2 => Some(nvjet),
            _ => None,
        };

        let key = sample_sm121a_fp8_key(16384, 2560, 6144, 2560);
        assert!(is_sm121a_large_fp8_key(&key));
        // 2.658 ms passes the roofline timing floor on its own, so only the
        // SM121a large FP8 legacy-64x stage guard catches this poisoned entry.
        let poisoned_15 = CachedEntry {
            config: sm89_xmma,
            heur_index: 3,
            measured_ms: 2.658,
        };
        let poisoned_auto = CachedEntry {
            config: sm89_auto,
            heur_index: 3,
            measured_ms: 2.658,
        };
        let native_sm121 = CachedEntry {
            config: nvjet,
            heur_index: 0,
            measured_ms: 2.600,
        };
        assert_eq!(
            evaluate_entry_health(&key, poisoned_15.measured_ms, LEGACY_ROOFLINE_FLOOR),
            HealthCheck::Healthy
        );

        assert!(
            lookup_candidate_from_entries(
                None,
                Some(&poisoned_15),
                &key,
                &cands,
                64 * 1024 * 1024,
                extract,
                |_idx, _cand| true,
            )
            .is_none(),
            "legacy stages=15 entry for 16384x2560x6144 must be rejected"
        );
        assert!(
            lookup_candidate_from_entries(
                Some(&poisoned_15),
                None,
                &key,
                &cands,
                64 * 1024 * 1024,
                extract,
                |_idx, _cand| true,
            )
            .is_none(),
            "canonical stages=15 entry for 16384x2560x6144 must be rejected"
        );
        assert!(
            lookup_candidate_from_entries(
                Some(&poisoned_auto),
                Some(&poisoned_15),
                &key,
                &cands,
                64 * 1024 * 1024,
                extract,
                |_idx, _cand| true,
            )
            .is_none(),
            "STAGES_64xAUTO and STAGES_64x3 must both be rejected on SM121a large FP8"
        );

        // Rejecting a poisoned canonical entry still allows a healthy native Blackwell legacy entry to hit.
        let rescued = lookup_candidate_from_entries(
            Some(&poisoned_15),
            Some(&native_sm121),
            &key,
            &cands,
            64 * 1024 * 1024,
            extract,
            |_idx, _cand| true,
        )
        .expect("healthy native SM121 legacy entry must rescue poisoned canonical entry");
        assert_eq!(rescued.candidate_idx, 2);
        assert!(rescued.from_legacy_key);

        // Non-large shapes (M < 4096) are not blocked by the SM121a large FP8 guard.
        let small_key = sample_sm121a_fp8_key(1024, 12288, 2560, 13312);
        assert!(!is_sm121a_large_fp8_key(&small_key));
        let small_entry = CachedEntry {
            config: sm89_xmma,
            heur_index: 1,
            measured_ms: 0.345,
        };
        let small_hit = lookup_candidate_from_entries(
            None,
            Some(&small_entry),
            &small_key,
            &cands,
            64 * 1024 * 1024,
            extract,
            |_idx, _cand| true,
        )
        .expect("small M=1024 legacy entry must remain eligible");
        assert_eq!(small_hit.candidate_idx, 0);
        assert!(small_hit.from_legacy_key);

        // Shipped repro-kit seed cache must contain zero poisoned legacy 64x entries for M >= 4096.
        let seed_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../repro-kit/cache/cublaslt_plans_v2.json");
        let seed_entries = load_entries_from_path(&seed_path);
        for (raw_key, entry) in &seed_entries {
            if let Some((_, after_kind)) = raw_key.split_once("|fp8_scaled|")
                && let Some(shape) = after_kind.split('|').next()
                && let Some(m_str) = shape.split('x').next()
                && let Ok(m) = m_str.parse::<u32>()
                && m >= SM121A_LARGE_FP8_MIN_M
            {
                assert!(
                    !is_legacy_64x_stages(entry.config[ALGO_CONFIG_IDX_STAGES_ID]),
                    "shipped seed cache still contains legacy 64x stages={} entry for M={m}: {raw_key}",
                    entry.config[ALGO_CONFIG_IDX_STAGES_ID]
                );
            }
        }
    }

    #[test]
    fn cross_m_reuse_env_parser_defaults_enabled_and_zero_disables() {
        assert!(parse_cross_m_reuse(None));
        assert!(parse_cross_m_reuse(Some("")));
        assert!(parse_cross_m_reuse(Some("1")));
        assert!(parse_cross_m_reuse(Some("true")));
        assert!(!parse_cross_m_reuse(Some("0")));
        assert!(!parse_cross_m_reuse(Some(" 0 ")));
    }

    #[test]
    fn fp8_roofline_health_guard_covers_sm121_sm120_and_sm103() {
        assert_eq!(
            fp8_roofline_params(12, 1),
            Some(Fp8RooflineParams {
                tflops_per_sm: 4.25,
                effective_bw_tb_s: 0.250,
                launch_overhead_ms: 0.008,
            })
        );
        assert!(fp8_roofline_params(12, 0).is_some());
        assert!(fp8_roofline_params(10, 3).is_some());
        assert_eq!(fp8_roofline_params(9, 0), None);

        // SM120 discrete GPUs (110 / 170 / 188 SMs): healthy timings pass, degraded tactics fail.
        for (sm_count, device_name, healthy_ms, degraded_ms) in [
            (110i32, "NVIDIA_RTX_PRO_5000", 2.95f32, 6.0f32),
            (170i32, "NVIDIA_RTX_5090", 1.90f32, 4.5f32),
            (188i32, "NVIDIA_RTX_PRO_6000", 1.75f32, 4.0f32),
        ] {
            let mut key = sample_sm121a_fp8_key(16384, 16384, 2560, 16384);
            key.cc_major = 12;
            key.cc_minor = 0;
            key.sm_count = sm_count;
            key.device_name = device_name.to_string();
            assert_eq!(
                evaluate_entry_health(&key, healthy_ms, CANONICAL_ROOFLINE_FLOOR),
                HealthCheck::Healthy,
                "{device_name} healthy timing {healthy_ms}ms must pass"
            );
            assert!(
                matches!(
                    evaluate_entry_health(&key, degraded_ms, CANONICAL_ROOFLINE_FLOOR),
                    HealthCheck::Degraded { .. }
                ),
                "{device_name} degraded timing {degraded_ms}ms must be rejected"
            );
        }

        // SM103 Blackwell Ultra (B300 / GB300, 160 SMs, HBM3e ~6.0 TB/s).
        let mut sm103_key = sample_sm121a_fp8_key(16384, 16384, 2560, 16384);
        sm103_key.cc_major = 10;
        sm103_key.cc_minor = 3;
        sm103_key.sm_count = 160;
        sm103_key.device_name = "NVIDIA_B300".to_string();
        assert_eq!(
            evaluate_entry_health(&sm103_key, 0.72, CANONICAL_ROOFLINE_FLOOR),
            HealthCheck::Healthy,
            "SM103 healthy timing (~1.9 PFLOPS) must pass"
        );
        assert!(
            matches!(
                evaluate_entry_health(&sm103_key, 2.0, CANONICAL_ROOFLINE_FLOOR),
                HealthCheck::Degraded { .. }
            ),
            "SM103 degraded timing (~687 TFLOPS) must be rejected"
        );
    }

    #[test]
    fn sm120_large_fp8_rejects_legacy_64x_entries() {
        let sm89_xmma: AlgoConfig = [35, 20, 1, 0, 1, 0, 15, 0, 0];
        let sm89_auto: AlgoConfig = [35, 20, 1, 0, 1, 0, STAGES_64X_AUTO, 0, 0];
        let nvjet: AlgoConfig = [67, 467, 1, 0, 0, 30, 36, 0, 0];

        let mut cands = [HeurResult {
            algo: AlignedAlgo::ZERO,
            workspace_size: 16 * 1024 * 1024,
            state: 0,
            waves_count: 0.0,
            reserved: [0; 4],
        }; 3];
        cands[0].algo.0[0] = 0;
        cands[1].algo.0[0] = 1;
        cands[2].algo.0[0] = 2;
        let extract = |a: &AlignedAlgo| match a.0[0] {
            0 => Some(sm89_xmma),
            1 => Some(sm89_auto),
            2 => Some(nvjet),
            _ => None,
        };

        let mut sm120_key = sample_sm121a_fp8_key(16384, 2560, 6144, 2560);
        sm120_key.cc_major = 12;
        sm120_key.cc_minor = 0;
        sm120_key.sm_count = 188;
        sm120_key.device_name = "NVIDIA_RTX_PRO_6000".to_string();
        assert!(
            is_sm121a_large_fp8_key(&sm120_key),
            "SM120 (12, 0) large FP8 must activate legacy 64x sm89_xmma guard"
        );

        let poisoned_15 = CachedEntry {
            config: sm89_xmma,
            heur_index: 3,
            measured_ms: 0.65,
        };
        let poisoned_auto = CachedEntry {
            config: sm89_auto,
            heur_index: 3,
            measured_ms: 0.65,
        };
        let native_sm120 = CachedEntry {
            config: nvjet,
            heur_index: 0,
            measured_ms: 0.62,
        };

        assert!(
            lookup_candidate_from_entries(
                Some(&poisoned_15),
                Some(&poisoned_auto),
                &sm120_key,
                &cands,
                64 * 1024 * 1024,
                extract,
                |_idx, _cand| true,
            )
            .is_none(),
            "SM120 large FP8 must reject both STAGES_64x3 and STAGES_64xAUTO sm89_xmma entries"
        );

        let rescued = lookup_candidate_from_entries(
            Some(&poisoned_15),
            Some(&native_sm120),
            &sm120_key,
            &cands,
            64 * 1024 * 1024,
            extract,
            |_idx, _cand| true,
        )
        .expect("healthy native Blackwell entry must rescue poisoned SM120 entry");
        assert_eq!(rescued.candidate_idx, 2);
        assert!(rescued.from_legacy_key);

        let mut small_sm120 = sm120_key.clone();
        small_sm120.m = 1024;
        assert!(!is_sm121a_large_fp8_key(&small_sm120));

        let mut hopper_key = sm120_key.clone();
        hopper_key.cc_major = 9;
        hopper_key.cc_minor = 0;
        assert!(!is_sm121a_large_fp8_key(&hopper_key));

        let mut sm103_key = sm120_key;
        sm103_key.cc_major = 10;
        sm103_key.cc_minor = 3;
        assert!(!is_sm121a_large_fp8_key(&sm103_key));
    }
}
