// SPDX-License-Identifier: AGPL-3.0-only

//! Compute-capability detection, route selection policies, env gates, and one-shot route loggers for dense FFN.

use std::sync::Mutex;

use spark_runtime::gpu::{GpuBackend, KernelHandle};

use super::e2m1_flashinfer::{E2m1Scope, E2m1SelectedKernel};
use super::layer::FfnActivation;

/// True on Blackwell datacenter GPUs (`SM100` B200/GB200 or `SM103` B300/GB300,
/// compute capabilities 10.0 and 10.3).
///
/// SM100/SM103 do not implement SM120/SM121 warp-level
/// `mma.sync.aligned.kind::mxf4nvf4.block_scale` PTX instructions (compiled as
/// `__trap()` stubs in `cutlass_nvfp4_gemm.cu`); native block-scaled NVFP4
/// GEMMs execute through cuBLASLt (`VEC16_UE4M3`, `tcgen05` tensor cores).
pub(super) const fn is_blackwell_dc_compute_capability(cc: Option<(u32, u32)>) -> bool {
    matches!(cc, Some((10, 0) | (10, 3)))
}

/// True when the active GPU is Blackwell desktop/workstation (`SM120`, e.g. RTX PRO 6000).
pub(super) const fn is_sm120_compute_capability(cc: Option<(u32, u32)>) -> bool {
    matches!(cc, Some((12, 0)))
}

/// True when the active GPU is Blackwell GB10 (`SM121`, DGX Spark).
pub(super) const fn is_sm121_compute_capability(cc: Option<(u32, u32)>) -> bool {
    matches!(cc, Some((12, 1)))
}

/// True when the active GPU supports warp-level SM120/SM121 `kind::mxf4nvf4.block_scale` MMA.
///
/// `None` (unit-test mock backends without a CUDA device) returns `true` so existing
/// mock-backend tests retain their contracts; real devices with `major < 12` (including
/// SM100/SM103) return `false` so `cutlass_nvfp4_gemm.cu` is never loaded or dispatched.
pub(super) const fn supports_warp_nvfp4_mma(cc: Option<(u32, u32)>) -> bool {
    match cc {
        Some((major, _)) => major >= 12,
        None => true,
    }
}

/// Whether the optional FlashInfer SM121 dense FFN prefill runtime is prepared
/// on this layer (`libatlas_fi_fp4_sm121.so` loaded and admitted).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum FlashInferFfnAvailability {
    Absent,
    Prepared,
}

/// Minimum token count `M` for dispatching the cuBLASLt block-scaled NVFP4
/// (`VEC16_UE4M3`) dense FFN and multi-seq projection routes.
/// `e2m1_scratch_layout` and `quantize_bf16_to_nvfp4_atlas_128x4` pad activation
/// scales to 128-row blocks while `cublaslt::nvfp4_gemm_block_scaled` accepts
/// any `M > 0`.
/// Lowered to `24` so SM120 multi-seq `C=24` (`padded_batch_n(17..=24) = 24`)
/// through `C=64+` dispatches through native NVFP4 tensor cores immediately
/// above the `M <= 16` coalesced BF16 MMA tier, eliminating the `M=17..=31`
/// dead zone.
pub(crate) const CUBLASLT_NVFP4_MIN_M: u32 = 24;

/// Dispatch policy for the native cuBLASLt NVFP4 (`VEC16_UE4M3`) dense FFN prefill route.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum CublasLtNvfp4Policy {
    /// Explicit `ATLAS_CUBLASLT_NVFP4=0` / `false`: disabled even on SM100/SM103/SM120.
    Disabled,
    /// Unset: auto-enabled on SM100/SM103 (`cc == Some((10, 0) | (10, 3))`) and on SM120
    /// (`cc == Some((12, 0))`) when FlashInfer FFN is absent; off on SM121.
    Auto,
    /// Explicit `ATLAS_CUBLASLT_NVFP4=1` / `true`: enabled on SM100/SM103 and SM120/SM121.
    Enabled,
}

pub(super) fn parse_cublaslt_nvfp4_policy(raw: Option<&str>) -> CublasLtNvfp4Policy {
    match raw.map(str::trim) {
        Some(v) if v == "0" || v.eq_ignore_ascii_case("false") => CublasLtNvfp4Policy::Disabled,
        Some(v) if v == "1" || v.eq_ignore_ascii_case("true") => CublasLtNvfp4Policy::Enabled,
        _ => CublasLtNvfp4Policy::Auto,
    }
}

pub(super) fn prefill_ffn_cublaslt_nvfp4_policy() -> CublasLtNvfp4Policy {
    static POLICY: std::sync::OnceLock<CublasLtNvfp4Policy> = std::sync::OnceLock::new();
    *POLICY.get_or_init(|| {
        let raw = std::env::var("ATLAS_CUBLASLT_NVFP4").ok();
        parse_cublaslt_nvfp4_policy(raw.as_deref())
    })
}

/// Pure eligibility check for the cuBLASLt block-scaled NVFP4 (`VEC16_UE4M3`) prefill route.
///
/// Requires `M >= CUBLASLT_NVFP4_MIN_M` (`24`), `H` and `Inter` positive multiples of 64 (so `K % 64 == 0` ->
/// `k_blocks % 4 == 0` for `quantize_bf16_to_nvfp4_atlas_128x4` and `N % 8 == 0` for
/// `cublaslt::nvfp4_gemm_block_scaled`), the `128x4` activation quantizer kernel, and either
/// static checkpoint scales or the dynamic absmax kernel.
#[cfg(test)]
#[expect(
    clippy::too_many_arguments,
    reason = "pure const policy predicate mirroring every eligibility dimension"
)]
pub(super) const fn cublaslt_nvfp4_route_enabled(
    policy: CublasLtNvfp4Policy,
    cc: Option<(u32, u32)>,
    flashinfer_ffn: FlashInferFfnAvailability,
    m: u32,
    h: u32,
    inter: u32,
    has_quantizer_128x4: bool,
    has_scale_source: bool,
) -> bool {
    cublaslt_nvfp4_route_enabled_with_gb10_fast_prefill(
        policy,
        cc,
        flashinfer_ffn,
        m,
        h,
        inter,
        has_quantizer_128x4,
        has_scale_source,
        false,
    )
}

/// Same eligibility check as [`cublaslt_nvfp4_route_enabled`], additionally
/// allowing `CublasLtNvfp4Policy::Auto` to engage on SM12.1 (GB10) when the
/// opt-in `ATLAS_GB10_FAST_PREFILL=1` umbrella is active and FlashInfer FFN is
/// absent. Explicit `ATLAS_CUBLASLT_NVFP4=0` (`Disabled`) still disables.
#[expect(
    clippy::too_many_arguments,
    reason = "pure const policy predicate mirroring every eligibility dimension including GB10 umbrella"
)]
pub(super) const fn cublaslt_nvfp4_route_enabled_with_gb10_fast_prefill(
    policy: CublasLtNvfp4Policy,
    cc: Option<(u32, u32)>,
    flashinfer_ffn: FlashInferFfnAvailability,
    m: u32,
    h: u32,
    inter: u32,
    has_quantizer_128x4: bool,
    has_scale_source: bool,
    gb10_fast_prefill: bool,
) -> bool {
    let policy_allows = match policy {
        CublasLtNvfp4Policy::Disabled => false,
        CublasLtNvfp4Policy::Auto => {
            is_blackwell_dc_compute_capability(cc)
                || ((is_sm120_compute_capability(cc)
                    || (gb10_fast_prefill && is_sm121_compute_capability(cc)))
                    && matches!(flashinfer_ffn, FlashInferFfnAvailability::Absent))
        }
        CublasLtNvfp4Policy::Enabled => true,
    };
    policy_allows
        && m >= CUBLASLT_NVFP4_MIN_M
        && h > 0
        && inter > 0
        && h.is_multiple_of(64)
        && inter.is_multiple_of(64)
        && has_quantizer_128x4
        && has_scale_source
}

/// Select the cuBLASLt `128x4` activation quantizer kernel, preferring the
/// single-pass vectorized Blackwell PTX kernel when present in the bundle.
pub(super) const fn select_cublaslt_quantize_128x4_kernel(
    ptx_k: KernelHandle,
    legacy_k: KernelHandle,
) -> KernelHandle {
    if ptx_k.0 != 0 { ptx_k } else { legacy_k }
}

/// Return the static activation `scale2` when cuBLASLt can fuse SwiGLU and
/// `128x4` activation quantization into `silu_mul_quantize_nvfp4_atlas_128x4`.
/// Returns `None` for non-SiLU activations, dynamic-absmax runs, or bundles
/// without the fused kernel so the caller falls back cleanly to `silu_mul` +
/// `prepare_e2m1_input_128x4`.
pub(super) const fn cublaslt_fused_silu_quantize_scale(
    activation: FfnActivation,
    down_scale: Option<f32>,
    fused_k: KernelHandle,
) -> Option<f32> {
    if matches!(activation, FfnActivation::SiLU) && fused_k.0 != 0 {
        down_scale
    } else {
        None
    }
}

/// Policy for merging `[gate_proj; up_proj]` into a single `N = 2 * inter`
/// cuBLASLt NVFP4 GEMM (`ATLAS_CUBLASLT_NVFP4_MERGED_GATEUP`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum CublasLtMergedGateUpPolicy {
    /// Explicit `"0"` / `"false"`: always run separate gate and up GEMMs.
    Disabled,
    /// Unset: on `sm_count <= 64`, merge only when zero weight duplication is
    /// required (gate and up are already contiguous in VRAM, or FlashInfer
    /// already built a merged buffer); on `sm_count > 64`, also allow lazy
    /// materialization when `gpu.free_memory()` has `>= merged_bytes + 4 GiB`.
    Auto,
    /// Explicit `"1"` / `"true"`: also allow lazily copying `[gate; up]` when
    /// `gpu.free_memory()` meets the SM-tier headroom reserve (`>= merged_bytes
    /// + 4 GiB` on `sm_count > 64`, `>= merged_bytes + 12 GiB` on `<= 64`).
    Enabled,
}

/// Minimum free VRAM reserve (12 GiB) required on UMA / compact GPUs
/// (`sm_count <= 64`) before lazily materializing a separate merged `[gate; up]`
/// weight copy.
#[cfg(any(test, feature = "cuda"))]
pub(super) const CUBLASLT_MERGED_GATEUP_VRAM_RESERVE_BYTES: usize = 12 * 1024 * 1024 * 1024;

/// Minimum free VRAM reserve (4 GiB) required on wide discrete GPUs
/// (`sm_count > 64`) before lazily materializing a separate merged `[gate; up]`
/// weight copy (static model-lifetime allocation, no per-request growth).
#[cfg(any(test, feature = "cuda"))]
pub(super) const CUBLASLT_MERGED_GATEUP_VRAM_RESERVE_DISCRETE_BYTES: usize = 4 * 1024 * 1024 * 1024;

/// Default strip-mining height (rows) for L2-resident cuBLASLt NVFP4 dense FFN
/// prefill on wide discrete GPUs (`sm_count > 64`, 96 MiB L2).
/// At `inter = 17408`, `1024` rows of merged BF16 `[1024, 34816]` occupy
/// `71.3 MB < 96 MiB L2` instead of thrashing DRAM with `570 MB` at `M = 8192`.
#[cfg(any(test, feature = "cuda"))]
pub(super) const CUBLASLT_FFN_PREFILL_L2_STRIP_DEFAULT: u32 = 1024;

pub(super) fn parse_cublaslt_merged_gate_up_policy(
    raw: Option<&str>,
) -> CublasLtMergedGateUpPolicy {
    match raw.map(str::trim) {
        Some(v) if v == "0" || v.eq_ignore_ascii_case("false") => {
            CublasLtMergedGateUpPolicy::Disabled
        }
        Some(v) if v == "1" || v.eq_ignore_ascii_case("true") => {
            CublasLtMergedGateUpPolicy::Enabled
        }
        _ => CublasLtMergedGateUpPolicy::Auto,
    }
}

#[cfg(feature = "cuda")]
pub(super) fn cublaslt_merged_gate_up_policy() -> CublasLtMergedGateUpPolicy {
    static POLICY: std::sync::OnceLock<CublasLtMergedGateUpPolicy> = std::sync::OnceLock::new();
    *POLICY.get_or_init(|| {
        let raw = std::env::var("ATLAS_CUBLASLT_NVFP4_MERGED_GATEUP").ok();
        parse_cublaslt_merged_gate_up_policy(raw.as_deref())
    })
}

#[cfg(any(test, feature = "cuda"))]
pub(super) const fn cublaslt_merged_gate_up_vram_ok_for_sm(
    free_bytes: Option<usize>,
    needed_bytes: usize,
    sm_count: u32,
) -> bool {
    let reserve = if sm_count > PROJ_27B_CUBLASLT_NVFP4_SM_THRESHOLD {
        CUBLASLT_MERGED_GATEUP_VRAM_RESERVE_DISCRETE_BYTES
    } else {
        CUBLASLT_MERGED_GATEUP_VRAM_RESERVE_BYTES
    };
    match free_bytes {
        Some(free) => match needed_bytes.checked_add(reserve) {
            Some(required) => free >= required,
            None => false,
        },
        None => false,
    }
}

#[cfg(test)]
pub(super) const fn cublaslt_merged_gate_up_vram_ok(
    free_bytes: Option<usize>,
    needed_bytes: usize,
) -> bool {
    cublaslt_merged_gate_up_vram_ok_for_sm(free_bytes, needed_bytes, 48)
}

#[cfg(any(test, feature = "cuda"))]
pub(super) const fn cublaslt_merged_gate_up_allow_materialize(
    policy: CublasLtMergedGateUpPolicy,
    sm_count: u32,
) -> bool {
    match policy {
        CublasLtMergedGateUpPolicy::Disabled => false,
        CublasLtMergedGateUpPolicy::Enabled => true,
        CublasLtMergedGateUpPolicy::Auto => sm_count > PROJ_27B_CUBLASLT_NVFP4_SM_THRESHOLD,
    }
}

#[cfg(any(test, feature = "cuda"))]
pub(super) fn gpu_sm_count(gpu: &dyn GpuBackend) -> u32 {
    gpu.sm_count()
        .filter(|&s| s > 0)
        .unwrap_or_else(atlas_core::device::active_sm_count)
}

#[cfg(any(test, feature = "cuda"))]
pub(super) const fn select_merged_silu_quant_kernel(
    ptx_k: KernelHandle,
    legacy_k: KernelHandle,
    sm_count: u32,
) -> KernelHandle {
    if sm_count > PROJ_27B_CUBLASLT_NVFP4_SM_THRESHOLD && ptx_k.0 != 0 {
        ptx_k
    } else if legacy_k.0 != 0 {
        legacy_k
    } else {
        ptx_k
    }
}

/// Parse `ATLAS_FFN_PREFILL_L2_STRIP`:
/// - `"0"` / `"false"` -> `None` (disable strip-mining)
/// - `"1"` / `"true"` -> `Some(1024)`
/// - positive integer multiple of `128` -> `Some(n)`
/// - unset -> `Some(1024)` on `sm_count > 64`, `None` on GB10 (`<= 64`).
#[cfg(any(test, feature = "cuda"))]
pub(super) fn parse_ffn_prefill_l2_strip(raw: Option<&str>, sm_count: u32) -> Option<u32> {
    match raw.map(str::trim) {
        Some(v) if v == "0" || v.eq_ignore_ascii_case("false") => None,
        Some(v) if v == "1" || v.eq_ignore_ascii_case("true") => {
            Some(CUBLASLT_FFN_PREFILL_L2_STRIP_DEFAULT)
        }
        Some(v) if !v.is_empty() => match v.parse::<u32>() {
            Ok(rows) if rows >= 128 && rows.is_multiple_of(128) => Some(rows),
            _ if sm_count > PROJ_27B_CUBLASLT_NVFP4_SM_THRESHOLD => {
                Some(CUBLASLT_FFN_PREFILL_L2_STRIP_DEFAULT)
            }
            _ => None,
        },
        _ => {
            if sm_count > PROJ_27B_CUBLASLT_NVFP4_SM_THRESHOLD {
                Some(CUBLASLT_FFN_PREFILL_L2_STRIP_DEFAULT)
            } else {
                None
            }
        }
    }
}

#[cfg(feature = "cuda")]
pub(super) fn ffn_prefill_l2_strip_rows(gpu: &dyn GpuBackend) -> Option<u32> {
    static RAW: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    let raw = RAW.get_or_init(|| std::env::var("ATLAS_FFN_PREFILL_L2_STRIP").ok());
    parse_ffn_prefill_l2_strip(raw.as_deref(), gpu_sm_count(gpu))
}

#[must_use]
pub(super) fn resolve_sm_auto_opt_in(raw: Option<&str>, sm_count: u32) -> bool {
    match raw.map(str::trim) {
        Some(v) if v == "0" || v.eq_ignore_ascii_case("false") => false,
        Some(v) if v == "1" || v.eq_ignore_ascii_case("true") => true,
        _ => sm_count > PROJ_27B_CUBLASLT_NVFP4_SM_THRESHOLD,
    }
}

/// Whether fused prefill RMSNorm/gated-norm/sigmoid-gate + `nvfp4_global_absmax`
/// is active (`ATLAS_PREFILL_FUSE_ABSMAX`, auto-ON on `sm_count > 64`, OFF on GB10).
#[cfg(feature = "cuda")]
#[must_use]
pub(crate) fn prefill_fuse_absmax_enabled(gpu: &dyn GpuBackend) -> bool {
    static RAW: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    let raw = RAW.get_or_init(|| std::env::var("ATLAS_PREFILL_FUSE_ABSMAX").ok());
    resolve_sm_auto_opt_in(raw.as_deref(), gpu_sm_count(gpu))
}

#[cfg(not(feature = "cuda"))]
#[must_use]
pub(crate) fn prefill_fuse_absmax_enabled(_gpu: &dyn GpuBackend) -> bool {
    false
}

/// Whether fused static-scale `residual_add_rms_norm -> nvfp4_128x4` + L2-stripped
/// FFN + `beta = 1.0` down-projection residual add is active
/// (`ATLAS_FFN_PREFILL_FUSED_NORM`, auto-ON on `sm_count > 64`, OFF on GB10).
#[cfg(feature = "cuda")]
#[must_use]
pub(super) fn ffn_prefill_fused_norm_enabled(gpu: &dyn GpuBackend) -> bool {
    static RAW: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    let raw = RAW.get_or_init(|| std::env::var("ATLAS_FFN_PREFILL_FUSED_NORM").ok());
    resolve_sm_auto_opt_in(raw.as_deref(), gpu_sm_count(gpu))
}

/// Minimum SM count for auto-enabling `ATLAS_27B_PROJ_CUBLASLT_NVFP4` on
/// discrete Blackwell GPUs while keeping GB10 (`<= 64` SMs) default-OFF.
pub(super) const PROJ_27B_CUBLASLT_NVFP4_SM_THRESHOLD: u32 = 64;

/// Minimum prefill token count `M` for routing SSM/Attention projections through
/// cuBLASLt block-scaled NVFP4 (`VEC16_UE4M3`).
pub(super) const PROJ_27B_CUBLASLT_NVFP4_MIN_M: u32 = 64;

/// Policy for `ATLAS_27B_PROJ_CUBLASLT_NVFP4` (routing SSM `qkvz`/`out_proj`
/// and Attention `q/k/v`/`o_proj` prefill GEMMs through `cublaslt::nvfp4_gemm_block_scaled`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Proj27bCublasLtNvfp4Policy {
    Disabled,
    Auto,
    Enabled,
}

#[must_use]
pub(super) fn parse_proj_27b_cublaslt_nvfp4_policy(
    raw: Option<&str>,
) -> Proj27bCublasLtNvfp4Policy {
    match raw.map(str::trim) {
        Some(v) if v == "0" || v.eq_ignore_ascii_case("false") => {
            Proj27bCublasLtNvfp4Policy::Disabled
        }
        Some(v) if v == "1" || v.eq_ignore_ascii_case("true") => {
            Proj27bCublasLtNvfp4Policy::Enabled
        }
        _ => Proj27bCublasLtNvfp4Policy::Auto,
    }
}

#[cfg(test)]
#[must_use]
pub(super) const fn resolve_proj_27b_cublaslt_nvfp4(
    proj_policy: Proj27bCublasLtNvfp4Policy,
    ffn_policy: CublasLtNvfp4Policy,
    sm_count: u32,
    cc: Option<(u32, u32)>,
) -> bool {
    resolve_proj_27b_cublaslt_nvfp4_with_gb10_fast_prefill(
        proj_policy,
        ffn_policy,
        sm_count,
        cc,
        false,
    )
}

/// Same resolver as [`resolve_proj_27b_cublaslt_nvfp4`], additionally allowing
/// `Auto` to engage on SM12.1 (GB10, `sm_count <= 64`) when
/// `ATLAS_GB10_FAST_PREFILL=1` is active and neither `ATLAS_27B_PROJ_CUBLASLT_NVFP4=0`
/// nor `ATLAS_CUBLASLT_NVFP4=0` opted out.
#[must_use]
pub(super) const fn resolve_proj_27b_cublaslt_nvfp4_with_gb10_fast_prefill(
    proj_policy: Proj27bCublasLtNvfp4Policy,
    ffn_policy: CublasLtNvfp4Policy,
    sm_count: u32,
    cc: Option<(u32, u32)>,
    gb10_fast_prefill: bool,
) -> bool {
    match proj_policy {
        Proj27bCublasLtNvfp4Policy::Disabled => false,
        Proj27bCublasLtNvfp4Policy::Enabled => true,
        Proj27bCublasLtNvfp4Policy::Auto => {
            if matches!(ffn_policy, CublasLtNvfp4Policy::Disabled) {
                return false;
            }
            if sm_count <= PROJ_27B_CUBLASLT_NVFP4_SM_THRESHOLD {
                return gb10_fast_prefill && is_sm121_compute_capability(cc);
            }
            match ffn_policy {
                CublasLtNvfp4Policy::Disabled => false,
                CublasLtNvfp4Policy::Enabled => true,
                CublasLtNvfp4Policy::Auto => {
                    is_sm120_compute_capability(cc) || is_blackwell_dc_compute_capability(cc)
                }
            }
        }
    }
}

#[must_use]
pub(super) const fn proj_27b_cublaslt_nvfp4_shape_ok(m: u32, n: u32, k: u32) -> bool {
    m >= PROJ_27B_CUBLASLT_NVFP4_MIN_M
        && n > 0
        && k > 0
        && n.is_multiple_of(8)
        && k.is_multiple_of(64)
}

#[must_use]
pub(super) fn proj_27b_cublaslt_nvfp4_enabled(cc: Option<(u32, u32)>) -> bool {
    static POLICY: std::sync::OnceLock<Proj27bCublasLtNvfp4Policy> = std::sync::OnceLock::new();
    let proj_policy = *POLICY.get_or_init(|| {
        let raw = std::env::var("ATLAS_27B_PROJ_CUBLASLT_NVFP4").ok();
        parse_proj_27b_cublaslt_nvfp4_policy(raw.as_deref())
    });
    resolve_proj_27b_cublaslt_nvfp4_with_gb10_fast_prefill(
        proj_policy,
        prefill_ffn_cublaslt_nvfp4_policy(),
        atlas_core::device::active_sm_count(),
        cc,
        crate::layers::ops::gb10_fast_prefill_enabled(),
    )
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(super) struct E2m1RouteProof {
    pub(super) scope: E2m1Scope,
    pub(super) kernel: E2m1SelectedKernel,
    pub(super) checkpoint_static: bool,
    pub(super) fused_silu_input: bool,
    pub(super) activation: FfnActivation,
    pub(super) m: u32,
    pub(super) h: u32,
    pub(super) inter: u32,
}

pub(super) fn log_e2m1_route_once(proof: E2m1RouteProof) {
    static SEEN: std::sync::OnceLock<Mutex<std::collections::HashSet<E2m1RouteProof>>> =
        std::sync::OnceLock::new();
    let seen = SEEN.get_or_init(|| Mutex::new(std::collections::HashSet::new()));
    let mut seen = seen.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    if !seen.insert(proof) {
        return;
    }
    let scope = match proof.scope {
        E2m1Scope::Full => "full",
        E2m1Scope::DownOnly => "down-only",
    };
    let gate_up = match proof.scope {
        E2m1Scope::Full => "e2m1",
        E2m1Scope::DownOnly => "w4a16-m128",
    };
    let kernel = match proof.kernel {
        E2m1SelectedKernel::RowMajorM64 => "row-major-m64",
        E2m1SelectedKernel::KmajorM128 => "kmajor-m128",
        E2m1SelectedKernel::KmajorM256 => "kmajor-m256",
    };
    let activation_scale = if proof.checkpoint_static {
        "checkpoint-static"
    } else {
        "dynamic-absmax"
    };
    let down_input = if proof.fused_silu_input {
        "fused-silu-nvfp4"
    } else {
        "separate-bf16-silu"
    };
    let activation = match proof.activation {
        FfnActivation::SiLU => "silu",
        FfnActivation::GeLU => "gelu",
    };
    tracing::info!(
        "ENGAGED ATLAS_E2M1_PREFILL: scope={scope} gate_up={gate_up} \
         e2m1_kernel={kernel} activation_scale={activation_scale} \
         down_input={down_input} activation={activation} M={} H={} intermediate={}",
        proof.m,
        proof.h,
        proof.inter,
    );
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum PrefillFfnFastRoute {
    Disabled,
    Ineligible,
    Complete,
    Missing,
}

pub(super) const fn prefill_ffn_fast_route(
    requested: bool,
    shape_eligible: bool,
    weights_ready: bool,
    m16_kernel_ready: bool,
    m128_kernel_ready: bool,
) -> PrefillFfnFastRoute {
    if !requested {
        return PrefillFfnFastRoute::Disabled;
    }
    if !shape_eligible {
        return PrefillFfnFastRoute::Ineligible;
    }
    if weights_ready && m16_kernel_ready && m128_kernel_ready {
        PrefillFfnFastRoute::Complete
    } else {
        PrefillFfnFastRoute::Missing
    }
}

pub(super) fn log_prefill_ffn_fast_route_once(m: u32, h: u32, inter: u32) {
    static FAST: std::sync::Once = std::sync::Once::new();
    FAST.call_once(|| {
        tracing::info!(
            "ENGAGED ATLAS_PREFILL_FFN_FAST: approximate transposed M128 route complete M={m} H={h} intermediate={inter}"
        );
    });
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum PrefillFusedEpilogueRoute {
    Disabled,
    Ineligible,
    Complete,
    Missing,
}

pub(super) fn prefill_fused_epilogue_route(
    requested: bool,
    parent_pipe_ready: bool,
    has_kernel: bool,
    activation: FfnActivation,
    m: u32,
    h: u32,
    inter: u32,
) -> PrefillFusedEpilogueRoute {
    if !requested {
        return PrefillFusedEpilogueRoute::Disabled;
    }
    if activation != FfnActivation::SiLU
        || m <= 32
        || !h.is_multiple_of(64)
        || !inter.is_multiple_of(64)
    {
        return PrefillFusedEpilogueRoute::Ineligible;
    }
    if parent_pipe_ready && has_kernel {
        PrefillFusedEpilogueRoute::Complete
    } else {
        PrefillFusedEpilogueRoute::Missing
    }
}

pub(super) fn log_prefill_pipe_route_once() {
    static PIPE: std::sync::Once = std::sync::Once::new();
    PIPE.call_once(|| {
        tracing::info!("ENGAGED ATLAS_PREFILL_FFN_PIPE: ordinary exact pipe route");
    });
}

pub(super) fn log_prefill_up_fused_route_once() {
    static UP_FUSED: std::sync::Once = std::sync::Once::new();
    UP_FUSED.call_once(|| {
        tracing::info!("ENGAGED ATLAS_PREFILL_FFN_FUSED_EPILOGUE: exact up-only fusion");
    });
}

pub(super) fn log_prefill_dual_fused_route_once() {
    static DUAL_FUSED: std::sync::Once = std::sync::Once::new();
    DUAL_FUSED.call_once(|| {
        tracing::info!("ENGAGED ATLAS_PREFILL_FFN_DUAL_FUSED: exact gate/up fusion");
    });
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ExactFfnDispatch {
    Existing,
    Batched,
    PerRowK1,
}

pub(super) const fn exact_ffn_dispatch(
    rows: u32,
    exact_w4_silu_eligible: bool,
    selected_tier_present: bool,
) -> ExactFfnDispatch {
    if !exact_w4_silu_eligible || rows <= 1 || rows > 32 {
        ExactFfnDispatch::Existing
    } else if selected_tier_present {
        ExactFfnDispatch::Batched
    } else {
        ExactFfnDispatch::PerRowK1
    }
}

pub(super) const fn exact_ffn_auto_kgamma_applicable(
    rows: u32,
    exact_w4_silu_eligible: bool,
) -> bool {
    matches!(
        exact_ffn_dispatch(rows, exact_w4_silu_eligible, false),
        ExactFfnDispatch::PerRowK1
    )
}

pub(super) const fn w3_kgamma_applicable(rows: u32, activation_silu: bool, prepared: bool) -> bool {
    rows > 1 && rows <= 32 && activation_silu && prepared
}

#[derive(Debug, Clone, Copy)]
pub(super) struct ExactFfnMaterializedAvailability {
    pub(super) m8_dual_silu: bool,
    pub(super) m8_f32_down: bool,
    pub(super) m17_dual_silu: bool,
    pub(super) m17_f32_down: bool,
    pub(super) m17_fused_dual_silu: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ExactFfnMaterializedRoute {
    Inline,
    Split,
    FusedM17,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ExactFfnPhysicalRoute {
    Native,
    SplitM8x2,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct ExactFfnRowOffsets {
    pub(super) input: usize,
    pub(super) gate_up: usize,
    pub(super) preactivation: usize,
    pub(super) output: usize,
}

pub(super) fn exact_ffn_row_offsets(
    first_row: u32,
    hidden: u32,
    intermediate: u32,
) -> Option<ExactFfnRowOffsets> {
    let row = first_row as usize;
    let hidden = hidden as usize;
    let intermediate = intermediate as usize;
    Some(ExactFfnRowOffsets {
        input: row.checked_mul(hidden)?.checked_mul(2)?,
        gate_up: row.checked_mul(intermediate)?.checked_mul(2)?,
        preactivation: row.checked_mul(intermediate)?.checked_mul(4)?,
        output: row.checked_mul(hidden)?.checked_mul(2)?,
    })
}

pub(super) fn exact_ffn_physical_route_with_policy(
    rows: u32,
    intermediate: u32,
    split_m8_enabled: bool,
    m8_exact_complete: bool,
    m8_materialized_complete: bool,
    scratch_bytes: usize,
    det_batch: crate::layers::deterministic_batch::DeterministicBatchPolicy,
) -> ExactFfnPhysicalRoute {
    if det_batch.is_enabled() {
        return ExactFfnPhysicalRoute::Native;
    }
    exact_ffn_physical_route(
        rows,
        intermediate,
        split_m8_enabled,
        m8_exact_complete,
        m8_materialized_complete,
        scratch_bytes,
    )
}

pub(super) fn exact_ffn_physical_route(
    rows: u32,
    intermediate: u32,
    split_m8_enabled: bool,
    m8_exact_complete: bool,
    m8_materialized_complete: bool,
    scratch_bytes: usize,
) -> ExactFfnPhysicalRoute {
    let required_bytes = (rows as usize)
        .checked_mul(intermediate as usize)
        .and_then(|elements| elements.checked_mul(std::mem::size_of::<f32>()));
    if rows == 16
        && split_m8_enabled
        && m8_exact_complete
        && m8_materialized_complete
        && required_bytes.is_some_and(|required| required <= scratch_bytes)
    {
        ExactFfnPhysicalRoute::SplitM8x2
    } else {
        ExactFfnPhysicalRoute::Native
    }
}

pub(super) fn exact_ffn_split_m8_enabled() -> bool {
    static GATE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *GATE.get_or_init(|| std::env::var("ATLAS_EXACT_FFN_SPLIT_M8").ok().as_deref() == Some("1"))
}

pub(super) fn exact_ffn_lowreg_gate_up_m16_enabled() -> bool {
    static GATE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *GATE.get_or_init(|| {
        std::env::var("ATLAS_EXACT_FFN_LOWREG_GATE_UP_M16")
            .ok()
            .as_deref()
            == Some("1")
    })
}

/// Minimum SM count above which wide discrete Blackwell GPUs (e.g. SM120 / SM103)
/// auto-enable batched tensor-core `forward_kgamma` for `M >= 4` while keeping
/// GB10 (`active_sm_count() <= 64`) default-exact (0-ULP scalar GEMV) unchanged.
pub(super) const FFN_KGAMMA_AUTO_TC_SM_THRESHOLD: u32 = 64;

/// Maximum batched row count served by wide `forward_kgamma` (`w4a16_gemm_t_m128`
/// or cuBLASLt NVFP4).
pub(super) const FFN_KGAMMA_WIDE_MAX_ROWS: u32 = 256;

/// Policy for `ATLAS_FFN_TC`:
/// - `Disabled` (`"0"` / `"false"`): force bit-exact scalar-GEMV FFN even on `> 64`-SM GPUs.
/// - `Auto` (unset): off on `<= 64`-SM GPUs (GB10 100% unchanged); auto-enables batched
///   tensor-core `forward_kgamma` for `rows >= TC_VERIFY_MIN_ROWS` (`4..=256`) on `> 64`-SM GPUs.
/// - `Enabled` (`"1"` / `"true"`): force tensor-core FFN on any GPU (refreeze mode).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum FfnTcPolicy {
    Disabled,
    Auto,
    Enabled,
}

#[must_use]
pub(super) fn parse_ffn_tc_policy(raw: Option<&str>) -> FfnTcPolicy {
    match raw.map(str::trim) {
        Some(v) if v == "0" || v.eq_ignore_ascii_case("false") => FfnTcPolicy::Disabled,
        Some(v) if v == "1" || v.eq_ignore_ascii_case("true") => FfnTcPolicy::Enabled,
        _ => FfnTcPolicy::Auto,
    }
}

pub(super) fn ffn_tc_policy() -> FfnTcPolicy {
    static POLICY: std::sync::OnceLock<FfnTcPolicy> = std::sync::OnceLock::new();
    *POLICY.get_or_init(|| {
        let raw = std::env::var("ATLAS_FFN_TC").ok();
        parse_ffn_tc_policy(raw.as_deref())
    })
}

/// Policy-aware SM- and row-aware resolver for tensor-core `forward_kgamma`.
#[must_use]
pub(super) const fn resolve_ffn_tc_for_rows_with_policy(
    policy: FfnTcPolicy,
    sm_count: u32,
    rows: u32,
    det_batch: crate::layers::deterministic_batch::DeterministicBatchPolicy,
) -> bool {
    if det_batch.is_enabled() {
        return false;
    }
    resolve_ffn_tc_for_rows(policy, sm_count, rows)
}

/// Pure SM- and row-aware resolver for tensor-core `forward_kgamma`.
#[must_use]
pub(super) const fn resolve_ffn_tc_for_rows(policy: FfnTcPolicy, sm_count: u32, rows: u32) -> bool {
    match policy {
        FfnTcPolicy::Disabled => false,
        FfnTcPolicy::Enabled => true,
        FfnTcPolicy::Auto => {
            sm_count > FFN_KGAMMA_AUTO_TC_SM_THRESHOLD
                && rows >= crate::layers::TC_VERIFY_MIN_ROWS as u32
        }
    }
}

/// Pure resolver for env knobs that remain opt-in (`"1"`) on `<= 64`-SM GPUs
/// (GB10 unchanged) and auto-enable on `> 64`-SM GPUs unless explicitly `"0"` / `"false"`.
#[must_use]
pub(super) fn resolve_kgamma_sm_auto_flag(raw: Option<&str>, sm_count: u32) -> bool {
    match raw.map(str::trim) {
        Some(v) if v == "0" || v.eq_ignore_ascii_case("false") => false,
        Some(v) if v == "1" || v.eq_ignore_ascii_case("true") => true,
        _ => sm_count > FFN_KGAMMA_AUTO_TC_SM_THRESHOLD,
    }
}

/// Explicit `ATLAS_FFN_TC=1` REFREEZE switch (used by single-token `decode_tc_parity_route`).
pub(super) fn exact_ffn_tc_override() -> bool {
    !crate::layers::deterministic_batch::deterministic_batch_enabled()
        && matches!(ffn_tc_policy(), FfnTcPolicy::Enabled)
}

#[must_use]
pub(super) fn ffn_tc_active_for_rows(rows: u32) -> bool {
    resolve_ffn_tc_for_rows_with_policy(
        ffn_tc_policy(),
        atlas_core::device::active_sm_count(),
        rows,
        crate::layers::deterministic_batch::deterministic_batch_policy_or_disabled(),
    )
}

#[must_use]
pub(super) fn kgamma_wide_active() -> bool {
    if crate::layers::deterministic_batch::deterministic_batch_enabled() {
        return false;
    }
    static RAW: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    let raw = RAW.get_or_init(|| std::env::var("ATLAS_FFN_KGAMMA_WIDE").ok());
    resolve_kgamma_sm_auto_flag(raw.as_deref(), atlas_core::device::active_sm_count())
}

#[must_use]
pub(super) fn kgamma_m128_active() -> bool {
    if crate::layers::deterministic_batch::deterministic_batch_enabled() {
        return false;
    }
    static RAW: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    let raw = RAW.get_or_init(|| std::env::var("ATLAS_FFN_KGAMMA_M128").ok());
    resolve_kgamma_sm_auto_flag(raw.as_deref(), atlas_core::device::active_sm_count())
}

#[must_use]
pub(super) fn kgamma_m32_allowed() -> bool {
    static ALLOWED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ALLOWED.get_or_init(|| std::env::var("ATLAS_FFN_KGAMMA_M32").ok().as_deref() != Some("0"))
}

#[must_use]
pub(super) fn kgamma_fused_gateup_active() -> bool {
    if crate::layers::deterministic_batch::deterministic_batch_enabled() {
        return false;
    }
    static RAW: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    let raw = RAW.get_or_init(|| std::env::var("ATLAS_FFN_FUSED_GATEUP").ok());
    resolve_kgamma_sm_auto_flag(raw.as_deref(), atlas_core::device::active_sm_count())
}

#[must_use]
pub(super) fn resolve_kgamma_cublaslt_opt_in(raw: Option<&str>, sm_count: u32) -> bool {
    resolve_sm_auto_opt_in(raw, sm_count)
}

#[cfg(feature = "cuda")]
#[must_use]
pub(super) fn kgamma_cublaslt_opt_in(gpu: &dyn GpuBackend) -> bool {
    if crate::layers::deterministic_batch::deterministic_batch_enabled() {
        return false;
    }
    static RAW: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    let raw = RAW.get_or_init(|| std::env::var("ATLAS_FFN_KGAMMA_CUBLASLT").ok());
    resolve_kgamma_cublaslt_opt_in(raw.as_deref(), gpu_sm_count(gpu))
}

pub(super) fn exact_ffn_materialized_route(
    rows: u32,
    intermediate: u32,
    handles: ExactFfnMaterializedAvailability,
    scratch_bytes: usize,
) -> ExactFfnMaterializedRoute {
    let (dual_silu_handle_present, f32_down_handle_present) = match rows {
        5..=8 => (handles.m8_dual_silu, handles.m8_f32_down),
        9..=17 => (handles.m17_dual_silu, handles.m17_f32_down),
        _ => return ExactFfnMaterializedRoute::Inline,
    };
    let required_bytes = (rows as usize)
        .checked_mul(intermediate as usize)
        .and_then(|elements| elements.checked_mul(std::mem::size_of::<f32>()));
    if required_bytes.is_none_or(|required| required > scratch_bytes) || !f32_down_handle_present {
        return ExactFfnMaterializedRoute::Inline;
    }
    if matches!(rows, 9..=17) && handles.m17_fused_dual_silu {
        ExactFfnMaterializedRoute::FusedM17
    } else if dual_silu_handle_present {
        ExactFfnMaterializedRoute::Split
    } else {
        ExactFfnMaterializedRoute::Inline
    }
}

#[cfg(test)]
mod deterministic_batch_tests {
    use super::{
        ExactFfnPhysicalRoute, FfnTcPolicy, exact_ffn_physical_route_with_policy,
        resolve_ffn_tc_for_rows_with_policy,
    };
    use crate::layers::deterministic_batch::DeterministicBatchPolicy;

    #[test]
    fn deterministic_batch_disables_dense_ffn_tc_and_m16_split_across_1_to_128() {
        for sm in [48u32, 188u32] {
            for m in [1u32, 2, 4, 8, 16, 32, 64, 128] {
                for tc in [
                    FfnTcPolicy::Auto,
                    FfnTcPolicy::Enabled,
                    FfnTcPolicy::Disabled,
                ] {
                    assert!(
                        !resolve_ffn_tc_for_rows_with_policy(
                            tc,
                            sm,
                            m,
                            DeterministicBatchPolicy::Enabled,
                        ),
                        "dense FFN TC must be disabled at M={m} sm={sm}"
                    );
                }
                assert_eq!(
                    exact_ffn_physical_route_with_policy(
                        m,
                        17_408,
                        true,
                        true,
                        true,
                        16 * 1024 * 1024,
                        DeterministicBatchPolicy::Enabled,
                    ),
                    ExactFfnPhysicalRoute::Native,
                    "dense FFN physical route must stay Native at M={m}"
                );
            }
        }
        assert!(resolve_ffn_tc_for_rows_with_policy(
            FfnTcPolicy::Auto,
            188,
            8,
            DeterministicBatchPolicy::Disabled,
        ));
        assert_eq!(
            exact_ffn_physical_route_with_policy(
                16,
                17_408,
                true,
                true,
                true,
                16 * 1024 * 1024,
                DeterministicBatchPolicy::Disabled,
            ),
            ExactFfnPhysicalRoute::SplitM8x2
        );
    }
}
