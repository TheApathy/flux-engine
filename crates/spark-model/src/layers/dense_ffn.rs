// SPDX-License-Identifier: AGPL-3.0-only

//! Dense SwiGLU/GeGLU FFN component for non-MoE models.
//!
//! Forward: gate = gate_proj(x), up = up_proj(x), out = down_proj(act(gate) * up)
//! Single-token NVFP4 decode uses 2 fused launches for SiLU (dual GEMV +
//! SiLU-fused down GEMV) or 3 launches for GeLU (dual GEMV + GeLU-mul + down
//! GEMV); BF16 weights use 4 launches (separate gate/up GEMVs + activation + down).

use anyhow::{Context, Result, bail, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
#[cfg(all(feature = "cuda", target_os = "linux"))]
use std::sync::{Arc, Mutex, MutexGuard};

use crate::layer::ForwardContext;
use crate::layers::ffn_dual_tuned_enabled;
use crate::layers::ops;
use crate::weight_map::{DenseWeight, QuantizedWeight};

/// Scratch buffers for the inline BF16 → NVFP4 activation prequant
/// (W4A4 `nvfp4_nvfp4_gemm` fast path). The exact FlashInfer route
/// preallocates its maximum admitted shape during model construction; other
/// routes allocate on first prefill and resize in-place if M or K grows.
///
/// Sizes (per row M, per col K):
///   - `a_packed`: M × K/2 bytes (E2M1 nibbles)
///   - `a_scale`:  ceil(M/128)×128 × K/16 bytes (FP8 E4M3 scales; the
///     default layout uses the logical prefix while the optional
///     CUTLASS route uses the full padded 128x4 layout)
///   - `a_max`:    4 bytes (FP32 per-tensor absmax scratch)
///
/// One arena per `DenseFfnLayer` (= per transformer layer). Reused across
/// gate / up / down GEMMs within a layer: gate/up share K=H, down has
/// K=Inter, so the arena is sized for `max(H, Inter)` × `max_M`.
struct E2m1Scratch {
    a_packed: DevicePtr,
    a_scale: DevicePtr,
    a_max: DevicePtr,
    /// Current row capacity (M) the buffers can hold.
    cap_m: usize,
    /// Current column capacity (K) the buffers can hold (full K, not K/2).
    cap_k: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct E2m1ScratchLayout {
    cap_m: usize,
    cap_k: usize,
    packed_bytes: usize,
    scale_bytes: usize,
    max_bytes: usize,
}

fn e2m1_scratch_layout(m: usize, k: usize) -> Result<E2m1ScratchLayout> {
    ensure!(m > 0, "W4A4 activation scratch requires M > 0");
    ensure!(
        k >= 16 && k.is_multiple_of(16),
        "W4A4 activation scratch requires K >= 16 and K % 16 == 0; got {k}"
    );
    let cap_m = m.max(128);
    let cap_k = k;
    let elements = cap_m
        .checked_mul(cap_k)
        .context("W4A4 activation scratch size overflow")?;
    let packed_bytes = elements / 2;
    let scale_rows = cap_m
        .div_ceil(128)
        .checked_mul(128)
        .context("W4A4 activation scratch row-padding overflow")?;
    let scale_bytes = scale_rows
        .checked_mul(cap_k)
        .context("W4A4 padded activation-scale scratch size overflow")?
        / 16;
    Ok(E2m1ScratchLayout {
        cap_m,
        cap_k,
        packed_bytes,
        scale_bytes,
        max_bytes: std::mem::size_of::<f32>(),
    })
}

#[derive(Debug, Clone, Copy)]
struct E2m1CheckpointScales {
    /// SGLang's merged gate/up linear takes the maximum of the checkpoint's
    /// logical-partition scales before quantizing their shared input.
    gate_up: f32,
    down: f32,
}

#[cfg(all(feature = "cuda", target_os = "linux"))]
struct FlashinferFfnProjection {
    weight: DevicePtr,
    weight_scales_128x4: DevicePtr,
    weight_scales_hash: u64,
    alpha_f32: DevicePtr,
    input_scale_bits: u32,
    alpha_bits: u32,
}

#[cfg(all(feature = "cuda", target_os = "linux"))]
struct FlashinferMergedGateUp {
    layer: usize,
    weight: DevicePtr,
    weight_scales_128x4: DevicePtr,
    weight_hash: u64,
    weight_scales_hash: u64,
    alpha_f32: DevicePtr,
    input_scale_bits: u32,
    alpha_bits: u32,
    source_gate_weight: DevicePtr,
    source_up_weight: DevicePtr,
}

#[cfg(all(feature = "cuda", target_os = "linux"))]
struct FlashinferFfnPrefill {
    layer: usize,
    library: ops::flashinfer_sm121::FlashInferSm121,
    merged_gate_up: FlashinferMergedGateUp,
    down: FlashinferFfnProjection,
    quantize_atlas_128x4_k: KernelHandle,
    quantize_merged_silu_atlas_128x4_k: KernelHandle,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FlashinferFfnPrefillRoute {
    Disabled,
    Ineligible,
    Missing,
    Complete,
}

/// True on Blackwell datacenter GPUs (`SM100` B200/GB200 or `SM103` B300/GB300,
/// compute capabilities 10.0 and 10.3).
///
/// SM100/SM103 do not implement SM120/SM121 warp-level
/// `mma.sync.aligned.kind::mxf4nvf4.block_scale` PTX instructions (compiled as
/// `__trap()` stubs in `cutlass_nvfp4_gemm.cu`); native block-scaled NVFP4
/// GEMMs execute through cuBLASLt (`VEC16_UE4M3`, `tcgen05` tensor cores).
const fn is_blackwell_dc_compute_capability(cc: Option<(u32, u32)>) -> bool {
    matches!(cc, Some((10, 0) | (10, 3)))
}

/// True when the active GPU is Blackwell desktop/workstation (`SM120`, e.g. RTX PRO 6000).
const fn is_sm120_compute_capability(cc: Option<(u32, u32)>) -> bool {
    matches!(cc, Some((12, 0)))
}

/// True when the active GPU supports warp-level SM120/SM121 `kind::mxf4nvf4.block_scale` MMA.
///
/// `None` (unit-test mock backends without a CUDA device) returns `true` so existing
/// mock-backend tests retain their contracts; real devices with `major < 12` (including
/// SM100/SM103) return `false` so `cutlass_nvfp4_gemm.cu` is never loaded or dispatched.
const fn supports_warp_nvfp4_mma(cc: Option<(u32, u32)>) -> bool {
    match cc {
        Some((major, _)) => major >= 12,
        None => true,
    }
}

/// Whether the optional FlashInfer SM121 dense FFN prefill runtime is prepared
/// on this layer (`libatlas_fi_fp4_sm121.so` loaded and admitted).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FlashInferFfnAvailability {
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
enum CublasLtNvfp4Policy {
    /// Explicit `ATLAS_CUBLASLT_NVFP4=0` / `false`: disabled even on SM100/SM103/SM120.
    Disabled,
    /// Unset: auto-enabled on SM100/SM103 (`cc == Some((10, 0) | (10, 3))`) and on SM120
    /// (`cc == Some((12, 0))`) when FlashInfer FFN is absent; off on SM121.
    Auto,
    /// Explicit `ATLAS_CUBLASLT_NVFP4=1` / `true`: enabled on SM100/SM103 and SM120/SM121.
    Enabled,
}

fn parse_cublaslt_nvfp4_policy(raw: Option<&str>) -> CublasLtNvfp4Policy {
    match raw.map(str::trim) {
        Some(v) if v == "0" || v.eq_ignore_ascii_case("false") => CublasLtNvfp4Policy::Disabled,
        Some(v) if v == "1" || v.eq_ignore_ascii_case("true") => CublasLtNvfp4Policy::Enabled,
        _ => CublasLtNvfp4Policy::Auto,
    }
}

fn prefill_ffn_cublaslt_nvfp4_policy() -> CublasLtNvfp4Policy {
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
#[expect(
    clippy::too_many_arguments,
    reason = "pure const policy predicate mirroring every eligibility dimension"
)]
const fn cublaslt_nvfp4_route_enabled(
    policy: CublasLtNvfp4Policy,
    cc: Option<(u32, u32)>,
    flashinfer_ffn: FlashInferFfnAvailability,
    m: u32,
    h: u32,
    inter: u32,
    has_quantizer_128x4: bool,
    has_scale_source: bool,
) -> bool {
    let policy_allows = match policy {
        CublasLtNvfp4Policy::Disabled => false,
        CublasLtNvfp4Policy::Auto => {
            is_blackwell_dc_compute_capability(cc)
                || (is_sm120_compute_capability(cc)
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
const fn select_cublaslt_quantize_128x4_kernel(
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
const fn cublaslt_fused_silu_quantize_scale(
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
enum CublasLtMergedGateUpPolicy {
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
const CUBLASLT_MERGED_GATEUP_VRAM_RESERVE_BYTES: usize = 12 * 1024 * 1024 * 1024;

/// Minimum free VRAM reserve (4 GiB) required on wide discrete GPUs
/// (`sm_count > 64`) before lazily materializing a separate merged `[gate; up]`
/// weight copy (static model-lifetime allocation, no per-request growth).
#[cfg(any(test, feature = "cuda"))]
const CUBLASLT_MERGED_GATEUP_VRAM_RESERVE_DISCRETE_BYTES: usize = 4 * 1024 * 1024 * 1024;

/// Default strip-mining height (rows) for L2-resident cuBLASLt NVFP4 dense FFN
/// prefill on wide discrete GPUs (`sm_count > 64`, 96 MiB L2).
/// At `inter = 17408`, `1024` rows of merged BF16 `[1024, 34816]` occupy
/// `71.3 MB < 96 MiB L2` instead of thrashing DRAM with `570 MB` at `M = 8192`.
#[cfg(any(test, feature = "cuda"))]
const CUBLASLT_FFN_PREFILL_L2_STRIP_DEFAULT: u32 = 1024;

fn parse_cublaslt_merged_gate_up_policy(raw: Option<&str>) -> CublasLtMergedGateUpPolicy {
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
fn cublaslt_merged_gate_up_policy() -> CublasLtMergedGateUpPolicy {
    static POLICY: std::sync::OnceLock<CublasLtMergedGateUpPolicy> = std::sync::OnceLock::new();
    *POLICY.get_or_init(|| {
        let raw = std::env::var("ATLAS_CUBLASLT_NVFP4_MERGED_GATEUP").ok();
        parse_cublaslt_merged_gate_up_policy(raw.as_deref())
    })
}

#[cfg(any(test, feature = "cuda"))]
const fn cublaslt_merged_gate_up_vram_ok_for_sm(
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
const fn cublaslt_merged_gate_up_vram_ok(free_bytes: Option<usize>, needed_bytes: usize) -> bool {
    cublaslt_merged_gate_up_vram_ok_for_sm(free_bytes, needed_bytes, 48)
}

#[cfg(any(test, feature = "cuda"))]
const fn cublaslt_merged_gate_up_allow_materialize(
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
fn gpu_sm_count(gpu: &dyn GpuBackend) -> u32 {
    gpu.sm_count()
        .filter(|&s| s > 0)
        .unwrap_or_else(atlas_core::device::active_sm_count)
}

#[cfg(any(test, feature = "cuda"))]
const fn select_merged_silu_quant_kernel(
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
fn parse_ffn_prefill_l2_strip(raw: Option<&str>, sm_count: u32) -> Option<u32> {
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
fn ffn_prefill_l2_strip_rows(gpu: &dyn GpuBackend) -> Option<u32> {
    static RAW: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    let raw = RAW.get_or_init(|| std::env::var("ATLAS_FFN_PREFILL_L2_STRIP").ok());
    parse_ffn_prefill_l2_strip(raw.as_deref(), gpu_sm_count(gpu))
}

#[must_use]
fn resolve_sm_auto_opt_in(raw: Option<&str>, sm_count: u32) -> bool {
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
fn ffn_prefill_fused_norm_enabled(gpu: &dyn GpuBackend) -> bool {
    static RAW: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    let raw = RAW.get_or_init(|| std::env::var("ATLAS_FFN_PREFILL_FUSED_NORM").ok());
    resolve_sm_auto_opt_in(raw.as_deref(), gpu_sm_count(gpu))
}

/// Minimum SM count for auto-enabling `ATLAS_27B_PROJ_CUBLASLT_NVFP4` on
/// discrete Blackwell GPUs while keeping GB10 (`<= 64` SMs) default-OFF.
const PROJ_27B_CUBLASLT_NVFP4_SM_THRESHOLD: u32 = 64;

/// Minimum prefill token count `M` for routing SSM/Attention projections through
/// cuBLASLt block-scaled NVFP4 (`VEC16_UE4M3`).
const PROJ_27B_CUBLASLT_NVFP4_MIN_M: u32 = 64;

/// Policy for `ATLAS_27B_PROJ_CUBLASLT_NVFP4` (routing SSM `qkvz`/`out_proj`
/// and Attention `q/k/v`/`o_proj` prefill GEMMs through `cublaslt::nvfp4_gemm_block_scaled`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Proj27bCublasLtNvfp4Policy {
    Disabled,
    Auto,
    Enabled,
}

#[must_use]
fn parse_proj_27b_cublaslt_nvfp4_policy(raw: Option<&str>) -> Proj27bCublasLtNvfp4Policy {
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

#[must_use]
const fn resolve_proj_27b_cublaslt_nvfp4(
    proj_policy: Proj27bCublasLtNvfp4Policy,
    ffn_policy: CublasLtNvfp4Policy,
    sm_count: u32,
    cc: Option<(u32, u32)>,
) -> bool {
    match proj_policy {
        Proj27bCublasLtNvfp4Policy::Disabled => false,
        Proj27bCublasLtNvfp4Policy::Enabled => true,
        Proj27bCublasLtNvfp4Policy::Auto => {
            if sm_count <= PROJ_27B_CUBLASLT_NVFP4_SM_THRESHOLD {
                return false;
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
const fn proj_27b_cublaslt_nvfp4_shape_ok(m: u32, n: u32, k: u32) -> bool {
    m >= PROJ_27B_CUBLASLT_NVFP4_MIN_M
        && n > 0
        && k > 0
        && n.is_multiple_of(8)
        && k.is_multiple_of(64)
}

#[must_use]
fn proj_27b_cublaslt_nvfp4_enabled(cc: Option<(u32, u32)>) -> bool {
    static POLICY: std::sync::OnceLock<Proj27bCublasLtNvfp4Policy> = std::sync::OnceLock::new();
    let proj_policy = *POLICY.get_or_init(|| {
        let raw = std::env::var("ATLAS_27B_PROJ_CUBLASLT_NVFP4").ok();
        parse_proj_27b_cublaslt_nvfp4_policy(raw.as_deref())
    });
    resolve_proj_27b_cublaslt_nvfp4(
        proj_policy,
        prefill_ffn_cublaslt_nvfp4_policy(),
        atlas_core::device::active_sm_count(),
        cc,
    )
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Sm120CoalescedSmallMRoute {
    /// Disabled (e.g. GB10 `sm_count <= 64`, out-of-range `M`, or explicit opt-out).
    Disabled,
    /// Medium/wide-N, deep-K (`4097 <= N <= 32768, K >= 4096`, e.g. `N = 5120`,
    /// `12288`, `16384`): launch intra-CTA 4-way warp-split-K (`CO_N = 16,
    /// TILE_M = 16`, 320–1024 CTAs with `L2::evict_first`), preferring v2
    /// 2-stage K=256 prefetch when `K % 64 == 0`.
    MediumNWarpSplitK { prefer_v2: bool },
    /// Very-wide-N, deep-K (`N > 32768, K >= 4096`, e.g. `lm_head` `N = 248320`):
    /// launch intra-CTA 4-way warp-split-K (`CO_N = 32, TILE_M = 16`, 7753 CTAs
    /// with `L2::evict_first` + 2-stage K=256 prefetch).
    WideNWarpSplitK,
    /// Narrow (`N <= 4096`) or shallow-K projections: delegate to the
    /// SM-aware coalesced selector (`n8_sk8_m16`, `v2_n32_m16`, `v2_n64_m16`).
    DelegateFastProj,
}

#[must_use]
const fn select_sm120_coalesced_small_m_route(
    env_enabled: bool,
    wide_sk4_enabled: bool,
    sm_count: u32,
    m: usize,
    n: usize,
    k: usize,
    max_m: usize,
    scale_aligned4: bool,
) -> Sm120CoalescedSmallMRoute {
    if !env_enabled
        || sm_count <= PROJ_27B_CUBLASLT_NVFP4_SM_THRESHOLD
        || m == 0
        || m > max_m
        || n == 0
        || k < 32
        || !k.is_multiple_of(32)
    {
        return Sm120CoalescedSmallMRoute::Disabled;
    }
    if n > 4_096 && k >= 4_096 {
        let can_v2 = k.is_multiple_of(64) && scale_aligned4;
        if n <= 8_192 {
            return Sm120CoalescedSmallMRoute::MediumNWarpSplitK { prefer_v2: can_v2 };
        }
        if wide_sk4_enabled && can_v2 {
            if n <= 32_768 {
                return Sm120CoalescedSmallMRoute::MediumNWarpSplitK { prefer_v2: true };
            }
            return Sm120CoalescedSmallMRoute::WideNWarpSplitK;
        }
    }
    Sm120CoalescedSmallMRoute::DelegateFastProj
}

#[must_use]
fn sm120_coalesced_m16_env_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        !matches!(
            std::env::var("ATLAS_SM120_COALESCED_M16")
                .ok()
                .as_deref()
                .map(str::trim),
            Some("0" | "false" | "FALSE" | "off" | "OFF")
        )
    })
}

#[must_use]
fn sm120_wide_sk4_env_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        !matches!(
            std::env::var("ATLAS_SM120_WIDE_SK4")
                .ok()
                .as_deref()
                .map(str::trim),
            Some("0" | "false" | "FALSE" | "off" | "OFF")
        )
    })
}

#[must_use]
fn sm120_gateup_n32_env_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        matches!(
            std::env::var("ATLAS_SM120_GATEUP_N32")
                .ok()
                .as_deref()
                .map(str::trim),
            Some("1" | "true" | "TRUE" | "on" | "ON")
        )
    })
}

/// Controls whether wide multi-seq batches (`M = 17..=64`, e.g. Qwen3.8-27B
/// `C=32/64`) route through SM120 BF16xBF16 MMA coalesced warp-split-K
/// (`TILE_M = 32 / 64`) ahead of cuBLASLt NVFP4 FP4xFP4 tensor cores.
///
/// Default OFF (`sm120_coalesced_max_m() = 16`): live 600W SM120 A/B proved
/// coalesced BF16 MMA dominates `M <= 16` (`C=8: 561.5`, `C=16: 932.5`,
/// `C=1 DFlash2: 383.5`), while native cuBLASLt NVFP4 FP4xFP4 dominates
/// `M >= 32` (`C=32: 1,332.0`, `C=64: 1,964.9`, +632 tok/s over BF16 MMA at
/// C=64, 96.8% of raw GDDR7 bandwidth). Set `ATLAS_SM120_COALESCED_WIDE_M=1`
/// to force `M = 17..=64` coalesced BF16 MMA for A/B testing.
#[must_use]
pub(crate) fn sm120_coalesced_wide_m_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        matches!(
            std::env::var("ATLAS_SM120_COALESCED_WIDE_M")
                .ok()
                .as_deref()
                .map(str::trim),
            Some("1" | "true" | "TRUE" | "on" | "ON")
        )
    })
}

#[must_use]
pub(crate) fn sm120_coalesced_max_m() -> usize {
    if sm120_coalesced_wide_m_enabled() {
        64
    } else {
        16
    }
}

#[derive(Clone, Copy)]
struct Sm120WarpSplitKKernels {
    v2_n16_sk4_m16: KernelHandle,
    v1_n16_sk4_m16: KernelHandle,
    v2_n32_sk4_m16: KernelHandle,
    v2_n16_sk4_m32: KernelHandle,
    v2_n32_sk4_m32: KernelHandle,
    v2_n16_sk4_m64: KernelHandle,
    v2_n32_sk2_m64: KernelHandle,
    v2_n16_sk4_m16_gateup_silu: KernelHandle,
    v2_n32_sk4_m16_gateup_silu: KernelHandle,
    v2_n16_sk4_m32_gateup_silu: KernelHandle,
    v2_n16_sk4_m64_gateup_silu: KernelHandle,
    v2_n32_sk2_m64_gateup_silu: KernelHandle,
}

fn try_w4a16_coalesced_symbol(gpu: &dyn GpuBackend, name: &str) -> KernelHandle {
    let h = super::try_kernel(gpu, "w4a16", name);
    if h.0 != 0 {
        h
    } else {
        super::try_kernel(gpu, "w4a16_gemm", name)
    }
}

fn cached_sm120_warp_splitk_kernels(gpu: &dyn GpuBackend) -> Sm120WarpSplitKKernels {
    static CACHE: std::sync::OnceLock<
        std::sync::Mutex<std::collections::HashMap<usize, Sm120WarpSplitKKernels>>,
    > = std::sync::OnceLock::new();
    let key = std::ptr::from_ref(gpu) as *const () as usize;
    let load = || Sm120WarpSplitKKernels {
        v2_n16_sk4_m16: try_w4a16_coalesced_symbol(gpu, "w4a16_gemm_coalesced_v2_n16_sk4_m16"),
        v1_n16_sk4_m16: try_w4a16_coalesced_symbol(gpu, "w4a16_gemm_coalesced_n16_sk4_m16"),
        v2_n32_sk4_m16: try_w4a16_coalesced_symbol(gpu, "w4a16_gemm_coalesced_v2_n32_sk4_m16"),
        v2_n16_sk4_m32: try_w4a16_coalesced_symbol(gpu, "w4a16_gemm_coalesced_v2_n16_sk4_m32"),
        v2_n32_sk4_m32: try_w4a16_coalesced_symbol(gpu, "w4a16_gemm_coalesced_v2_n32_sk4_m32"),
        v2_n16_sk4_m64: try_w4a16_coalesced_symbol(gpu, "w4a16_gemm_coalesced_v2_n16_sk4_m64"),
        v2_n32_sk2_m64: try_w4a16_coalesced_symbol(gpu, "w4a16_gemm_coalesced_v2_n32_sk2_m64"),
        v2_n16_sk4_m16_gateup_silu: try_w4a16_coalesced_symbol(
            gpu,
            "w4a16_gemm_coalesced_v2_n16_sk4_m16_gateup_silu",
        ),
        v2_n32_sk4_m16_gateup_silu: try_w4a16_coalesced_symbol(
            gpu,
            "w4a16_gemm_coalesced_v2_n32_sk4_m16_gateup_silu",
        ),
        v2_n16_sk4_m32_gateup_silu: try_w4a16_coalesced_symbol(
            gpu,
            "w4a16_gemm_coalesced_v2_n16_sk4_m32_gateup_silu",
        ),
        v2_n16_sk4_m64_gateup_silu: try_w4a16_coalesced_symbol(
            gpu,
            "w4a16_gemm_coalesced_v2_n16_sk4_m64_gateup_silu",
        ),
        v2_n32_sk2_m64_gateup_silu: try_w4a16_coalesced_symbol(
            gpu,
            "w4a16_gemm_coalesced_v2_n32_sk2_m64_gateup_silu",
        ),
    };
    let map = CACHE.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()));
    let Ok(mut guard) = map.lock() else {
        return load();
    };
    *guard.entry(key).or_insert_with(load)
}

/// Pre-load SM120 coalesced NVFP4 W4A16 BF16xBF16 MMA kernel handles before
/// CUDA graph capture begins so first-time lookups never occur mid-capture.
pub(crate) fn warm_coalesced_nvfp4_sm120_kernels(gpu: &dyn GpuBackend) {
    let _ = cached_sm120_warp_splitk_kernels(gpu);
    crate::layers::qwen4_fast_proj::warm_coalesced_nvfp4_kernels(gpu);
}

/// Dispatch fused small/medium-M (`m <= max_m`, up to `64`) NVFP4 W4A16
/// `silu(A @ Bg^T) * (A @ Bu^T)` through BF16xBF16 MMA intra-CTA warp-split-K
/// on wide discrete GPUs (`sm_count > 64`). Always returns `Ok(false)` on GB10.
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_coalesced_nvfp4_sm120_small_m_gateup_silu(
    gpu: &dyn GpuBackend,
    act: DevicePtr,
    gate: &QuantizedWeight,
    up: &QuantizedWeight,
    out: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    max_m: usize,
    stream: u64,
) -> Result<bool> {
    if !sm120_coalesced_m16_env_enabled()
        || atlas_core::device::active_sm_count() <= PROJ_27B_CUBLASLT_NVFP4_SM_THRESHOLD
        || m == 0
        || (m as usize) > max_m
        || m > 64
        || n == 0
        || k < 64
        || !k.is_multiple_of(64)
        || act.is_null()
        || out.is_null()
        || gate.weight.is_null()
        || gate.weight_scale.is_null()
        || up.weight.is_null()
        || up.weight_scale.is_null()
        || !gate.weight_scale_2.is_finite()
        || !up.weight_scale_2.is_finite()
        || !act.0.is_multiple_of(16)
        || !gate.weight.0.is_multiple_of(16)
        || !up.weight.0.is_multiple_of(16)
        || !gate.weight_scale.0.is_multiple_of(4)
        || !up.weight_scale.0.is_multiple_of(4)
        || !out.0.is_multiple_of(2)
    {
        return Ok(false);
    }
    let kernels = cached_sm120_warp_splitk_kernels(gpu);
    let prefer_n32 = sm120_gateup_n32_env_enabled();
    let (kernel, tile_m, co_n, block_threads) = if m <= 16 {
        if prefer_n32 && kernels.v2_n32_sk4_m16_gateup_silu.0 != 0 {
            (kernels.v2_n32_sk4_m16_gateup_silu, 16u32, 32u32, 512u32)
        } else if kernels.v2_n16_sk4_m16_gateup_silu.0 != 0 {
            (kernels.v2_n16_sk4_m16_gateup_silu, 16u32, 16u32, 256u32)
        } else {
            return Ok(false);
        }
    } else if m <= 32 {
        if kernels.v2_n16_sk4_m32_gateup_silu.0 != 0 {
            (kernels.v2_n16_sk4_m32_gateup_silu, 32u32, 16u32, 256u32)
        } else {
            return Ok(false);
        }
    } else if prefer_n32 && kernels.v2_n32_sk2_m64_gateup_silu.0 != 0 {
        (kernels.v2_n32_sk2_m64_gateup_silu, 64u32, 32u32, 256u32)
    } else if kernels.v2_n16_sk4_m64_gateup_silu.0 != 0 {
        (kernels.v2_n16_sk4_m64_gateup_silu, 64u32, 16u32, 256u32)
    } else if kernels.v2_n32_sk2_m64_gateup_silu.0 != 0 {
        (kernels.v2_n32_sk2_m64_gateup_silu, 64u32, 32u32, 256u32)
    } else {
        return Ok(false);
    };
    spark_runtime::kernel_args::KernelLaunch::new(gpu, kernel)
        .grid([n.div_ceil(co_n), m.div_ceil(tile_m), 1])
        .block([block_threads, 1, 1])
        .arg_ptr(act)
        .arg_ptr(gate.weight)
        .arg_ptr(gate.weight_scale)
        .arg_f32(gate.weight_scale_2)
        .arg_ptr(up.weight)
        .arg_ptr(up.weight_scale)
        .arg_f32(up.weight_scale_2)
        .arg_ptr(out)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)?;
    Ok(true)
}

/// Dispatch small/medium-M (`m <= max_m`, up to `64`) NVFP4 W4A16 projection
/// through BF16xBF16 MMA coalesced / intra-CTA warp-split-K kernels on wide
/// discrete GPUs (`sm_count > 64`), returning `Ok(true)` if dispatched or
/// `Ok(false)` to let callers fall back cleanly.
///
/// Always returns `Ok(false)` on GB10 (`sm_count <= 64`) for zero regression.
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_coalesced_nvfp4_sm120_small_m(
    gpu: &dyn GpuBackend,
    act: DevicePtr,
    weight: &QuantizedWeight,
    out: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    max_m: usize,
    stream: u64,
) -> Result<bool> {
    if act.is_null()
        || out.is_null()
        || weight.weight.is_null()
        || weight.weight_scale.is_null()
        || !weight.weight_scale_2.is_finite()
        || !act.0.is_multiple_of(16)
        || !weight.weight.0.is_multiple_of(16)
        || !weight.weight_scale.0.is_multiple_of(2)
        || !out.0.is_multiple_of(2)
    {
        return Ok(false);
    }
    let sm_count = atlas_core::device::active_sm_count();
    let route = select_sm120_coalesced_small_m_route(
        sm120_coalesced_m16_env_enabled(),
        sm120_wide_sk4_env_enabled(),
        sm_count,
        m as usize,
        n as usize,
        k as usize,
        max_m,
        weight.weight_scale.0.is_multiple_of(4),
    );
    match route {
        Sm120CoalescedSmallMRoute::Disabled => Ok(false),
        Sm120CoalescedSmallMRoute::MediumNWarpSplitK { prefer_v2 } => {
            let kernels = cached_sm120_warp_splitk_kernels(gpu);
            let (kernel, tile_m, co_n, block_threads) = if m <= 16 {
                if prefer_v2 && kernels.v2_n16_sk4_m16.0 != 0 {
                    (kernels.v2_n16_sk4_m16, 16u32, 16u32, 256u32)
                } else if kernels.v1_n16_sk4_m16.0 != 0 {
                    (kernels.v1_n16_sk4_m16, 16u32, 16u32, 256u32)
                } else {
                    (KernelHandle(0), 16u32, 16u32, 256u32)
                }
            } else if m <= 32 {
                if prefer_v2 && kernels.v2_n16_sk4_m32.0 != 0 {
                    (kernels.v2_n16_sk4_m32, 32u32, 16u32, 256u32)
                } else if prefer_v2 && kernels.v2_n32_sk4_m32.0 != 0 {
                    (kernels.v2_n32_sk4_m32, 32u32, 32u32, 512u32)
                } else {
                    (KernelHandle(0), 32u32, 16u32, 256u32)
                }
            } else if m <= 64 {
                if prefer_v2 && kernels.v2_n16_sk4_m64.0 != 0 {
                    (kernels.v2_n16_sk4_m64, 64u32, 16u32, 256u32)
                } else if prefer_v2 && kernels.v2_n32_sk2_m64.0 != 0 {
                    (kernels.v2_n32_sk2_m64, 64u32, 32u32, 256u32)
                } else {
                    (KernelHandle(0), 64u32, 16u32, 256u32)
                }
            } else {
                (KernelHandle(0), 64u32, 16u32, 256u32)
            };
            if kernel.0 != 0 {
                spark_runtime::kernel_args::KernelLaunch::new(gpu, kernel)
                    .grid([n.div_ceil(co_n), m.div_ceil(tile_m), 1])
                    .block([block_threads, 1, 1])
                    .arg_ptr(act)
                    .arg_ptr(weight.weight)
                    .arg_ptr(weight.weight_scale)
                    .arg_f32(weight.weight_scale_2)
                    .arg_ptr(out)
                    .arg_u32(m)
                    .arg_u32(n)
                    .arg_u32(k)
                    .launch(stream)?;
                return Ok(true);
            }
            crate::layers::qwen4_fast_proj::try_coalesced_nvfp4_w4a16_gemm(
                gpu, act, weight, out, m as usize, n as usize, k as usize, stream,
            )
        }
        Sm120CoalescedSmallMRoute::WideNWarpSplitK => {
            let kernels = cached_sm120_warp_splitk_kernels(gpu);
            let (kernel, tile_m, co_n, block_threads) = if m <= 16 {
                (kernels.v2_n32_sk4_m16, 16u32, 32u32, 512u32)
            } else if m <= 32 {
                (kernels.v2_n32_sk4_m32, 32u32, 32u32, 512u32)
            } else if m <= 64 {
                (kernels.v2_n32_sk2_m64, 64u32, 32u32, 256u32)
            } else {
                (KernelHandle(0), 64u32, 32u32, 256u32)
            };
            if kernel.0 != 0 {
                spark_runtime::kernel_args::KernelLaunch::new(gpu, kernel)
                    .grid([n.div_ceil(co_n), m.div_ceil(tile_m), 1])
                    .block([block_threads, 1, 1])
                    .arg_ptr(act)
                    .arg_ptr(weight.weight)
                    .arg_ptr(weight.weight_scale)
                    .arg_f32(weight.weight_scale_2)
                    .arg_ptr(out)
                    .arg_u32(m)
                    .arg_u32(n)
                    .arg_u32(k)
                    .launch(stream)?;
                return Ok(true);
            }
            crate::layers::qwen4_fast_proj::try_coalesced_nvfp4_w4a16_gemm(
                gpu, act, weight, out, m as usize, n as usize, k as usize, stream,
            )
        }
        Sm120CoalescedSmallMRoute::DelegateFastProj => {
            crate::layers::qwen4_fast_proj::try_coalesced_nvfp4_w4a16_gemm(
                gpu, act, weight, out, m as usize, n as usize, k as usize, stream,
            )
        }
    }
}

/// Device scalar scratch layout for zero-sync dynamic NVFP4 quantization:
/// `[0..4] global_max_f32 | [4..8] scale2_a_f32 | [8..12] status_u32 |
///  [12..16] alpha_0 | [16..20] alpha_1 | [20..24] alpha_2` (32 B, 16-B aligned).
#[cfg(feature = "cuda")]
const CUBLASLT_DYNAMIC_SCALARS_BYTES: usize = 32;
#[cfg(feature = "cuda")]
const CUBLASLT_DYNAMIC_ZERO_PREFIX_BYTES: usize = 12;
#[cfg(feature = "cuda")]
const CUBLASLT_DYNAMIC_GLOBAL_MAX_OFF: usize = 0;
#[cfg(feature = "cuda")]
const CUBLASLT_DYNAMIC_SCALE2_OFF: usize = 4;
#[cfg(feature = "cuda")]
const CUBLASLT_DYNAMIC_STATUS_OFF: usize = 8;
#[cfg(feature = "cuda")]
const CUBLASLT_DYNAMIC_ALPHA0_OFF: usize = 12;
#[cfg(feature = "cuda")]
const CUBLASLT_DYNAMIC_ALPHA1_OFF: usize = 16;
#[cfg(feature = "cuda")]
const CUBLASLT_DYNAMIC_ALPHA2_OFF: usize = 20;

/// Independent device `alpha` slot for projections sharing one dynamic activation
/// quantization on the same CUDA stream (e.g. gate/up or Q/K/V).
#[cfg(feature = "cuda")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CublasLtAlphaSlot {
    Primary,
    Secondary,
    Tertiary,
}

/// Activation second-level scale source for a cuBLASLt NVFP4 GEMM.
#[cfg(feature = "cuda")]
#[derive(Debug, Clone, Copy)]
enum CublasLtInputScale {
    /// Static ModelOpt checkpoint `input_scale` (or legacy sync fallback): zero
    /// device scalar indirection, uses `BlockScaleScalars::host`.
    Static(f32),
    /// Zero-sync device-resident `scale2_a` + `status` produced asynchronously on
    /// `stream` by `nvfp4_global_absmax` + `quantize_bf16_to_nvfp4_atlas_128x4_from_absmax`.
    DynamicDevice {
        scale2_a_f32: DevicePtr,
        status_u32: DevicePtr,
        alpha_slots: [DevicePtr; 3],
    },
}

/// Cached `128x4`-swizzled `UE4M3` block-scale device copies for `gate_proj`, `up_proj`,
/// and `down_proj`, plus a 32-byte zero-sync dynamic scalar scratch and optional merged
/// `[gate; up]` operand, prepared lazily on first cuBLASLt NVFP4 prefill dispatch.
#[cfg(feature = "cuda")]
struct CublasLtNvfp4Scales {
    gate: DevicePtr,
    up: DevicePtr,
    down: DevicePtr,
    dynamic_scalars: DevicePtr,
    merged_weight: Option<DevicePtr>,
    owns_merged_weight: bool,
    merged_scales: Option<DevicePtr>,
    owns_merged_scales: bool,
    gate_src: DevicePtr,
    up_src: DevicePtr,
    down_src: DevicePtr,
}

#[cfg(feature = "cuda")]
fn swizzle_nvfp4_scales_128x4_host(logical: &[u8], n: u32, k: u32) -> Result<Vec<u8>> {
    use spark_runtime::cublaslt::{
        NVFP4_BLOCK_SIZE, block_scale_padded_bytes, block_scale_swizzle_offset,
    };

    ensure!(
        n > 0 && k > 0 && k.is_multiple_of(NVFP4_BLOCK_SIZE),
        "cuBLASLt NVFP4 scale swizzle requires positive N and K % {NVFP4_BLOCK_SIZE} == 0; got N={n} K={k}"
    );
    let k_blocks = k / NVFP4_BLOCK_SIZE;
    let expected_len = (n as usize)
        .checked_mul(k_blocks as usize)
        .context("cuBLASLt NVFP4 logical scale length overflow")?;
    ensure!(
        logical.len() == expected_len,
        "cuBLASLt NVFP4 logical scale length mismatch: expected {expected_len}, got {}",
        logical.len()
    );
    let padded_len = block_scale_padded_bytes(n, k, NVFP4_BLOCK_SIZE);
    let mut swizzled = vec![0u8; padded_len];
    for row in 0..n {
        let row_base = (row as usize) * (k_blocks as usize);
        for g in 0..k_blocks {
            let dst = block_scale_swizzle_offset(row, g, k_blocks);
            swizzled[dst] = logical[row_base + (g as usize)];
        }
    }
    Ok(swizzled)
}

#[cfg(feature = "cuda")]
fn upload_swizzled_nvfp4_scales_128x4(
    gpu: &dyn GpuBackend,
    weight: &QuantizedWeight,
    n: u32,
    k: u32,
) -> Result<DevicePtr> {
    let k_blocks = k / spark_runtime::cublaslt::NVFP4_BLOCK_SIZE;
    let logical_len = (n as usize)
        .checked_mul(k_blocks as usize)
        .context("cuBLASLt NVFP4 weight scale length overflow")?;
    let mut logical = vec![0u8; logical_len];
    gpu.copy_d2h(weight.weight_scale, &mut logical)
        .context("read logical NVFP4 weight scales for cuBLASLt 128x4 swizzle")?;
    let swizzled = swizzle_nvfp4_scales_128x4_host(&logical, n, k)?;
    let d_swizzled = gpu
        .alloc(swizzled.len())
        .context("alloc cuBLASLt NVFP4 128x4 weight scales")?;
    if let Err(err) = gpu.copy_h2d(&swizzled, d_swizzled) {
        let _ = gpu.free(d_swizzled);
        return Err(err).context("upload cuBLASLt NVFP4 128x4 weight scales");
    }
    Ok(d_swizzled)
}

#[cfg(feature = "cuda")]
fn upload_swizzled_merged_gate_up_scales_128x4(
    gpu: &dyn GpuBackend,
    gate: &QuantizedWeight,
    up: &QuantizedWeight,
    inter: u32,
    h: u32,
) -> Result<DevicePtr> {
    let k_blocks = h / spark_runtime::cublaslt::NVFP4_BLOCK_SIZE;
    let half_len = (inter as usize)
        .checked_mul(k_blocks as usize)
        .context("cuBLASLt NVFP4 merged scale half-length overflow")?;
    let full_len = half_len
        .checked_mul(2)
        .context("cuBLASLt NVFP4 merged scale total-length overflow")?;
    let mut logical = vec![0u8; full_len];
    gpu.copy_d2h(gate.weight_scale, &mut logical[..half_len])
        .context("read logical gate NVFP4 weight scales for merged cuBLASLt 128x4 swizzle")?;
    gpu.copy_d2h(up.weight_scale, &mut logical[half_len..])
        .context("read logical up NVFP4 weight scales for merged cuBLASLt 128x4 swizzle")?;
    let merged_n = inter
        .checked_mul(2)
        .context("cuBLASLt NVFP4 merged N overflow")?;
    let swizzled = swizzle_nvfp4_scales_128x4_host(&logical, merged_n, h)?;
    let d_swizzled = gpu
        .alloc(swizzled.len())
        .context("alloc cuBLASLt NVFP4 merged 128x4 weight scales")?;
    if let Err(err) = gpu.copy_h2d(&swizzled, d_swizzled) {
        let _ = gpu.free(d_swizzled);
        return Err(err).context("upload cuBLASLt NVFP4 merged 128x4 weight scales");
    }
    Ok(d_swizzled)
}

/// Whether `dynamic_scalars[global_max]` was already populated on-stream by a
/// preceding fused norm/gate kernel or still needs `memset_async` + `nvfp4_global_absmax`.
#[cfg(feature = "cuda")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DynamicAbsmaxState {
    NeedsScan,
    PrecomputedInDynamicScalars,
}

#[cfg(feature = "cuda")]
#[expect(
    clippy::too_many_arguments,
    reason = "CUDA kernel launcher forwarding explicit quantization handles and device buffers"
)]
fn launch_quantize_bf16_to_nvfp4_128x4_zero_sync(
    gpu: &dyn GpuBackend,
    quant_static_k: KernelHandle,
    absmax_k: KernelHandle,
    from_absmax_k: KernelHandle,
    combined_alpha_k: KernelHandle,
    input: DevicePtr,
    a_packed: DevicePtr,
    a_scale: DevicePtr,
    dynamic_scalars: DevicePtr,
    legacy_a_max: DevicePtr,
    m: u32,
    k: u32,
    checkpoint_scale2: Option<f32>,
    absmax_state: DynamicAbsmaxState,
    stream: u64,
) -> Result<CublasLtInputScale> {
    if let Some(scale2) = checkpoint_scale2 {
        ops::quantize_bf16_to_nvfp4_atlas_128x4(
            gpu,
            quant_static_k,
            input,
            a_packed,
            a_scale,
            scale2,
            m,
            k,
            stream,
        )?;
        return Ok(CublasLtInputScale::Static(scale2));
    }

    let has_absmax_source = matches!(
        absmax_state,
        DynamicAbsmaxState::PrecomputedInDynamicScalars
    ) || absmax_k.0 != 0;
    if from_absmax_k.0 != 0
        && combined_alpha_k.0 != 0
        && has_absmax_source
        && !dynamic_scalars.is_null()
    {
        let total_elements = m
            .checked_mul(k)
            .context("cuBLASLt dynamic NVFP4 activation element-count overflow")?;
        let padded_rows = m
            .checked_add(127)
            .context("cuBLASLt dynamic NVFP4 row-padding overflow")?
            / 128
            * 128;
        let cta_cap = gpu_sm_count(gpu).saturating_mul(2).max(96);
        let global_max = dynamic_scalars.offset(CUBLASLT_DYNAMIC_GLOBAL_MAX_OFF);
        let scale2_a = dynamic_scalars.offset(CUBLASLT_DYNAMIC_SCALE2_OFF);
        let status = dynamic_scalars.offset(CUBLASLT_DYNAMIC_STATUS_OFF);

        if matches!(absmax_state, DynamicAbsmaxState::NeedsScan) {
            gpu.memset_async(
                dynamic_scalars,
                0,
                CUBLASLT_DYNAMIC_ZERO_PREFIX_BYTES,
                stream,
            )?;
            ops::nvfp4_global_absmax(gpu, absmax_k, input, global_max, total_elements, stream)?;
        }
        spark_runtime::kernel_args::KernelLaunch::new(gpu, from_absmax_k)
            .grid([padded_rows.min(cta_cap).max(1), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(input)
            .arg_ptr(a_packed)
            .arg_ptr(a_scale)
            .arg_ptr(global_max)
            .arg_ptr(scale2_a)
            .arg_ptr(status)
            .arg_u32(m)
            .arg_u32(k)
            .launch(stream)?;

        return Ok(CublasLtInputScale::DynamicDevice {
            scale2_a_f32: scale2_a,
            status_u32: status,
            alpha_slots: [
                dynamic_scalars.offset(CUBLASLT_DYNAMIC_ALPHA0_OFF),
                dynamic_scalars.offset(CUBLASLT_DYNAMIC_ALPHA1_OFF),
                dynamic_scalars.offset(CUBLASLT_DYNAMIC_ALPHA2_OFF),
            ],
        });
    }

    // Legacy fallback only when `quantize_bf16_to_nvfp4_atlas_128x4_from_absmax`
    // or `nvfp4_dynamic_combined_alpha` is absent from an older kernel bundle.
    ensure!(
        absmax_k.0 != 0 && !legacy_a_max.is_null(),
        "cuBLASLt NVFP4 dynamic activation quantization requires nvfp4_global_absmax"
    );
    let total_elements = m
        .checked_mul(k)
        .context("cuBLASLt NVFP4 legacy absmax element-count overflow")?;
    gpu.memset_async(legacy_a_max, 0, 4, stream)?;
    ops::nvfp4_global_absmax(gpu, absmax_k, input, legacy_a_max, total_elements, stream)?;
    gpu.synchronize(stream)?;
    let mut bytes = [0u8; 4];
    gpu.copy_d2h(legacy_a_max, &mut bytes)?;
    let global_max = f32::from_le_bytes(bytes);
    let a_scale2 = if global_max > 0.0 {
        global_max / (6.0 * 448.0)
    } else {
        1.0
    };
    ops::quantize_bf16_to_nvfp4_atlas_128x4(
        gpu,
        quant_static_k,
        input,
        a_packed,
        a_scale,
        a_scale2,
        m,
        k,
        stream,
    )?;
    Ok(CublasLtInputScale::Static(a_scale2))
}

#[cfg(feature = "cuda")]
fn resolve_cublaslt_scalars(
    gpu: &dyn GpuBackend,
    combined_alpha_k: KernelHandle,
    input_scale: CublasLtInputScale,
    weight_scale_2: f32,
    slot: CublasLtAlphaSlot,
    stream: u64,
) -> Result<spark_runtime::cublaslt::BlockScaleScalars> {
    match input_scale {
        CublasLtInputScale::Static(a_scale2) => Ok(
            spark_runtime::cublaslt::BlockScaleScalars::host(a_scale2 * weight_scale_2, 0.0),
        ),
        CublasLtInputScale::DynamicDevice {
            scale2_a_f32,
            status_u32,
            alpha_slots,
        } => {
            let alpha_dst = match slot {
                CublasLtAlphaSlot::Primary => alpha_slots[0],
                CublasLtAlphaSlot::Secondary => alpha_slots[1],
                CublasLtAlphaSlot::Tertiary => alpha_slots[2],
            };
            spark_runtime::kernel_args::KernelLaunch::new(gpu, combined_alpha_k)
                .grid([1, 1, 1])
                .block([1, 1, 1])
                .arg_ptr(scale2_a_f32)
                .arg_f32(weight_scale_2)
                .arg_ptr(status_u32)
                .arg_ptr(alpha_dst)
                .launch(stream)?;
            Ok(spark_runtime::cublaslt::BlockScaleScalars::device_alpha(
                alpha_dst.0,
            ))
        }
    }
}

#[cfg(feature = "cuda")]
struct SharedMergedGateUpOut {
    ptr: DevicePtr,
    cap_bytes: usize,
}

#[cfg(feature = "cuda")]
fn ensure_shared_merged_gate_up_out(
    gpu: &dyn GpuBackend,
    bytes: usize,
) -> Result<Option<DevicePtr>> {
    use std::collections::HashMap;

    static BUF: std::sync::OnceLock<Mutex<HashMap<usize, SharedMergedGateUpOut>>> =
        std::sync::OnceLock::new();
    let gpu_id = std::ptr::from_ref(gpu) as *const () as usize;
    let lock = BUF.get_or_init(|| Mutex::new(HashMap::new()));
    let mut guard = lock
        .lock()
        .map_err(|_| anyhow::anyhow!("shared merged gate/up output scratch lock poisoned"))?;
    if let Some(existing) = guard.get(&gpu_id)
        && existing.cap_bytes >= bytes
    {
        return Ok(Some(existing.ptr));
    }
    if !cublaslt_merged_gate_up_vram_ok_for_sm(gpu.free_memory().ok(), bytes, gpu_sm_count(gpu)) {
        return Ok(None);
    }
    if let Some(old) = guard.remove(&gpu_id) {
        gpu.synchronize_device()?;
        let _ = gpu.free(old.ptr);
    }
    let ptr = gpu
        .alloc(bytes)
        .context("alloc shared cuBLASLt merged gate/up BF16 scratch")?;
    guard.insert(
        gpu_id,
        SharedMergedGateUpOut {
            ptr,
            cap_bytes: bytes,
        },
    );
    Ok(Some(ptr))
}

#[cfg(feature = "cuda")]
#[derive(Clone, Copy)]
struct CachedProjNvfp4Scale {
    scales_128x4: DevicePtr,
    input_scale: Option<f32>,
}

#[cfg(feature = "cuda")]
fn cached_proj_nvfp4_scale(
    gpu: &dyn GpuBackend,
    weight: &QuantizedWeight,
    n: u32,
    k: u32,
) -> Result<CachedProjNvfp4Scale> {
    use std::collections::HashMap;

    static CACHE: std::sync::OnceLock<
        Mutex<HashMap<(usize, u64, u32, u32), CachedProjNvfp4Scale>>,
    > = std::sync::OnceLock::new();
    let gpu_id = std::ptr::from_ref(gpu) as *const () as usize;
    let key = (gpu_id, weight.weight_scale.0, n, k);
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(&entry) = cache
        .lock()
        .map_err(|_| anyhow::anyhow!("proj cuBLASLt NVFP4 scale cache lock poisoned"))?
        .get(&key)
    {
        return Ok(entry);
    }

    let scales_128x4 = upload_swizzled_nvfp4_scales_128x4(gpu, weight, n, k)?;
    let input_scale = if !weight.input_scale.is_null() {
        let mut bytes = [0u8; 4];
        gpu.copy_d2h(weight.input_scale, &mut bytes)
            .ok()
            .map(move |()| f32::from_le_bytes(bytes))
            .filter(|v| v.is_finite() && *v > 0.0)
    } else {
        None
    };
    let entry = CachedProjNvfp4Scale {
        scales_128x4,
        input_scale,
    };
    let mut guard = cache
        .lock()
        .map_err(|_| anyhow::anyhow!("proj cuBLASLt NVFP4 scale cache lock poisoned"))?;
    if let Some(&existing) = guard.get(&key) {
        let _ = gpu.free(scales_128x4);
        return Ok(existing);
    }
    guard.insert(key, entry);
    Ok(entry)
}

#[cfg(feature = "cuda")]
struct SharedProjNvfp4Scratch {
    a_packed: DevicePtr,
    a_scale: DevicePtr,
    cap_m: usize,
    cap_k: usize,
    bound_stream: Option<u64>,
}

/// Process-lifetime permanent 32-byte `dynamic_scalars` buffer per GPU so
/// fused norm/gate absmax producers can write `global_max` without any
/// dangling-pointer risk when `a_packed`/`a_scale` resize across K widths.
#[cfg(feature = "cuda")]
fn ensure_proj_dynamic_scalars(gpu: &dyn GpuBackend) -> Result<DevicePtr> {
    use std::collections::HashMap;

    static SCALARS: std::sync::OnceLock<Mutex<HashMap<usize, DevicePtr>>> =
        std::sync::OnceLock::new();
    let gpu_id = std::ptr::from_ref(gpu) as *const () as usize;
    let map = SCALARS.get_or_init(|| Mutex::new(HashMap::new()));
    let mut guard = map
        .lock()
        .map_err(|_| anyhow::anyhow!("shared proj dynamic_scalars lock poisoned"))?;
    if let Some(&ptr) = guard.get(&gpu_id) {
        return Ok(ptr);
    }
    let ptr = gpu
        .alloc(CUBLASLT_DYNAMIC_SCALARS_BYTES)
        .context("alloc permanent shared proj NVFP4 dynamic scalars")?;
    guard.insert(gpu_id, ptr);
    Ok(ptr)
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PendingProjAbsmax {
    gpu_id: usize,
    act_ptr: u64,
    total_elements: u64,
    stream: u64,
}

#[cfg(feature = "cuda")]
static HAS_PENDING_PROJ_ABSMAX: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

#[cfg(feature = "cuda")]
fn pending_proj_absmax_lock() -> &'static Mutex<Option<PendingProjAbsmax>> {
    static SLOT: std::sync::OnceLock<Mutex<Option<PendingProjAbsmax>>> = std::sync::OnceLock::new();
    SLOT.get_or_init(|| Mutex::new(None))
}

/// Prepare the permanent `dynamic_scalars` buffer (`global_max` + `scale2` +
/// `status` zeroed on `stream`) so a fused prefill norm/gate kernel can write
/// `global_max` inline and the immediately following dynamic NVFP4 projection
/// can skip `nvfp4_global_absmax`.
#[cfg(feature = "cuda")]
pub(crate) fn prepare_precomputed_proj_absmax(
    gpu: &dyn GpuBackend,
    act_ptr: DevicePtr,
    m: u32,
    k: u32,
    stream: u64,
) -> Result<Option<DevicePtr>> {
    if !prefill_fuse_absmax_enabled(gpu)
        || act_ptr.is_null()
        || m < PROJ_27B_CUBLASLT_NVFP4_MIN_M
        || k < 64
        || !k.is_multiple_of(64)
        || gpu.stream_is_capturing(stream)
    {
        invalidate_precomputed_proj_absmax(gpu, act_ptr);
        return Ok(None);
    }
    let Some(total_elements) = u64::from(m).checked_mul(u64::from(k)) else {
        invalidate_precomputed_proj_absmax(gpu, act_ptr);
        return Ok(None);
    };
    let dynamic_scalars = ensure_proj_dynamic_scalars(gpu)?;
    gpu.memset_async(
        dynamic_scalars,
        0,
        CUBLASLT_DYNAMIC_ZERO_PREFIX_BYTES,
        stream,
    )?;
    let gpu_id = std::ptr::from_ref(gpu) as *const () as usize;
    let mut guard = pending_proj_absmax_lock()
        .lock()
        .map_err(|_| anyhow::anyhow!("pending proj absmax lock poisoned"))?;
    *guard = Some(PendingProjAbsmax {
        gpu_id,
        act_ptr: act_ptr.0,
        total_elements,
        stream,
    });
    HAS_PENDING_PROJ_ABSMAX.store(true, std::sync::atomic::Ordering::Release);
    Ok(Some(
        dynamic_scalars.offset(CUBLASLT_DYNAMIC_GLOBAL_MAX_OFF),
    ))
}

#[cfg(not(feature = "cuda"))]
pub(crate) fn prepare_precomputed_proj_absmax(
    _gpu: &dyn GpuBackend,
    _act_ptr: DevicePtr,
    _m: u32,
    _k: u32,
    _stream: u64,
) -> Result<Option<DevicePtr>> {
    Ok(None)
}

/// Invalidate any pending precomputed absmax targeting `act_ptr` whenever a
/// non-absmax kernel overwrites `act_ptr`. Fast-path is one relaxed atomic load.
#[cfg(feature = "cuda")]
pub(crate) fn invalidate_precomputed_proj_absmax(gpu: &dyn GpuBackend, act_ptr: DevicePtr) {
    if !HAS_PENDING_PROJ_ABSMAX.load(std::sync::atomic::Ordering::Acquire) {
        return;
    }
    let gpu_id = std::ptr::from_ref(gpu) as *const () as usize;
    if let Ok(mut guard) = pending_proj_absmax_lock().lock()
        && guard.is_some_and(|p| p.gpu_id == gpu_id && p.act_ptr == act_ptr.0)
    {
        *guard = None;
        HAS_PENDING_PROJ_ABSMAX.store(false, std::sync::atomic::Ordering::Release);
    }
}

#[cfg(not(feature = "cuda"))]
pub(crate) fn invalidate_precomputed_proj_absmax(_gpu: &dyn GpuBackend, _act_ptr: DevicePtr) {}

#[cfg(feature = "cuda")]
fn clear_precomputed_proj_absmax() {
    if !HAS_PENDING_PROJ_ABSMAX.load(std::sync::atomic::Ordering::Acquire) {
        return;
    }
    if let Ok(mut guard) = pending_proj_absmax_lock().lock() {
        *guard = None;
        HAS_PENDING_PROJ_ABSMAX.store(false, std::sync::atomic::Ordering::Release);
    }
}

#[cfg(feature = "cuda")]
fn take_precomputed_proj_absmax(
    gpu: &dyn GpuBackend,
    act_ptr: DevicePtr,
    m: u32,
    k: u32,
    stream: u64,
) -> DynamicAbsmaxState {
    if !HAS_PENDING_PROJ_ABSMAX.load(std::sync::atomic::Ordering::Acquire) {
        return DynamicAbsmaxState::NeedsScan;
    }
    let gpu_id = std::ptr::from_ref(gpu) as *const () as usize;
    let Some(total_elements) = u64::from(m).checked_mul(u64::from(k)) else {
        clear_precomputed_proj_absmax();
        return DynamicAbsmaxState::NeedsScan;
    };
    let Ok(mut guard) = pending_proj_absmax_lock().lock() else {
        return DynamicAbsmaxState::NeedsScan;
    };
    let matched = guard.take().is_some_and(|pending| {
        pending.gpu_id == gpu_id
            && pending.act_ptr == act_ptr.0
            && pending.total_elements == total_elements
            && pending.stream == stream
    });
    HAS_PENDING_PROJ_ABSMAX.store(false, std::sync::atomic::Ordering::Release);
    if matched {
        DynamicAbsmaxState::PrecomputedInDynamicScalars
    } else {
        DynamicAbsmaxState::NeedsScan
    }
}

#[cfg(feature = "cuda")]
fn proj_nvfp4_scratch_lock()
-> &'static Mutex<std::collections::HashMap<usize, SharedProjNvfp4Scratch>> {
    static SCRATCH: std::sync::OnceLock<
        Mutex<std::collections::HashMap<usize, SharedProjNvfp4Scratch>>,
    > = std::sync::OnceLock::new();
    SCRATCH.get_or_init(|| Mutex::new(std::collections::HashMap::new()))
}

#[cfg(feature = "cuda")]
fn ensure_proj_nvfp4_scratch(
    guard: &mut std::collections::HashMap<usize, SharedProjNvfp4Scratch>,
    gpu: &dyn GpuBackend,
    m: usize,
    k: usize,
    stream: u64,
) -> Result<(DevicePtr, DevicePtr, DevicePtr)> {
    let gpu_id = std::ptr::from_ref(gpu) as *const () as usize;
    let dynamic_scalars = ensure_proj_dynamic_scalars(gpu)?;
    let layout = e2m1_scratch_layout(m, k)?;
    let needs_realloc = match guard.get(&gpu_id) {
        Some(existing) => existing.cap_m < layout.cap_m || existing.cap_k < layout.cap_k,
        None => true,
    };
    if let Some(existing) = guard.get_mut(&gpu_id)
        && let Some(prev_stream) = existing.bound_stream
        && prev_stream != stream
    {
        gpu.synchronize(prev_stream)?;
        existing.bound_stream = Some(stream);
    }
    if needs_realloc {
        if let Some(prev) = guard.remove(&gpu_id) {
            gpu.synchronize_device()?;
            let _ = gpu.free(prev.a_packed);
            let _ = gpu.free(prev.a_scale);
        }
        let a_packed = gpu
            .alloc(layout.packed_bytes)
            .context("alloc shared proj NVFP4 a_packed scratch")?;
        let a_scale = match gpu.alloc(layout.scale_bytes) {
            Ok(ptr) => ptr,
            Err(err) => {
                let _ = gpu.free(a_packed);
                return Err(err).context("alloc shared proj NVFP4 a_scale scratch");
            }
        };
        guard.insert(
            gpu_id,
            SharedProjNvfp4Scratch {
                a_packed,
                a_scale,
                cap_m: layout.cap_m,
                cap_k: layout.cap_k,
                bound_stream: Some(stream),
            },
        );
        return Ok((a_packed, a_scale, dynamic_scalars));
    }
    let s = guard
        .get(&gpu_id)
        .ok_or_else(|| anyhow::anyhow!("proj NVFP4 scratch missing after capacity check"))?;
    Ok((s.a_packed, s.a_scale, dynamic_scalars))
}

#[cfg(feature = "cuda")]
struct DecodeFfnE2m1Scratch {
    a_packed: DevicePtr,
    a_scale: DevicePtr,
    merged_out: DevicePtr,
    cap_m: usize,
    cap_k: usize,
    merged_cap_bytes: usize,
}

#[cfg(feature = "cuda")]
fn decode_ffn_e2m1_scratch_map()
-> &'static Mutex<std::collections::HashMap<(usize, u64), DecodeFfnE2m1Scratch>> {
    static MAP: std::sync::OnceLock<
        Mutex<std::collections::HashMap<(usize, u64), DecodeFfnE2m1Scratch>>,
    > = std::sync::OnceLock::new();
    MAP.get_or_init(|| Mutex::new(std::collections::HashMap::new()))
}

/// Per-`(gpu, stream)` small W4A4 activation + merged gate/up output scratch for
/// multi-seq decode FFN (`forward_kgamma`), completely independent of prefill's
/// stream-bound `DenseFfnLayer::e2m1_scratch`. Pre-sizes to `max(m, 256)` rows
/// for E2M1 (~2.5 MiB) and `max(m, 128)` rows for merged gate/up BF16 (~8.9 MiB
/// at `inter = 17408`) so subsequent `C=32..128` steps and CUDA graph captures
/// never reallocate or race with the prefill stream.
#[cfg(feature = "cuda")]
fn ensure_decode_ffn_e2m1_scratch(
    gpu: &dyn GpuBackend,
    stream: u64,
    m: usize,
    k: usize,
    merged_bytes: usize,
    allow_alloc: bool,
) -> Result<Option<(DevicePtr, DevicePtr, DevicePtr, usize)>> {
    let layout = e2m1_scratch_layout(m.max(256), k)?;
    let alloc_merged_bytes = if merged_bytes > 0 {
        let row_bytes = merged_bytes / m.max(1);
        merged_bytes.max(row_bytes.saturating_mul(128))
    } else {
        0
    };
    let key = (std::ptr::from_ref(gpu) as *const () as usize, stream);
    let mut map = decode_ffn_e2m1_scratch_map()
        .lock()
        .map_err(|_| anyhow::anyhow!("decode FFN E2M1 scratch lock poisoned"))?;
    if let Some(existing) = map.get(&key)
        && existing.cap_m >= m
        && existing.cap_k >= k
        && (merged_bytes == 0 || existing.merged_cap_bytes >= merged_bytes)
    {
        return Ok(Some((
            existing.a_packed,
            existing.a_scale,
            existing.merged_out,
            existing.merged_cap_bytes,
        )));
    }
    if !allow_alloc {
        if let Some(existing) = map.get(&key)
            && existing.cap_m >= m
            && existing.cap_k >= k
        {
            return Ok(Some((
                existing.a_packed,
                existing.a_scale,
                existing.merged_out,
                existing.merged_cap_bytes,
            )));
        }
        return Ok(None);
    }
    if let Some(prev) = map.remove(&key) {
        gpu.synchronize(stream)?;
        let _ = gpu.free(prev.a_packed);
        let _ = gpu.free(prev.a_scale);
        if !prev.merged_out.is_null() {
            let _ = gpu.free(prev.merged_out);
        }
    }
    let a_packed = gpu
        .alloc(layout.packed_bytes)
        .context("alloc decode FFN NVFP4 a_packed scratch")?;
    let a_scale = match gpu.alloc(layout.scale_bytes) {
        Ok(ptr) => ptr,
        Err(err) => {
            let _ = gpu.free(a_packed);
            return Err(err).context("alloc decode FFN NVFP4 a_scale scratch");
        }
    };
    let (merged_out, merged_cap_bytes) = if alloc_merged_bytes > 0 {
        match gpu.alloc(alloc_merged_bytes) {
            Ok(ptr) => (ptr, alloc_merged_bytes),
            Err(_) => (DevicePtr::NULL, 0),
        }
    } else {
        (DevicePtr::NULL, 0)
    };
    map.insert(
        key,
        DecodeFfnE2m1Scratch {
            a_packed,
            a_scale,
            merged_out,
            cap_m: layout.cap_m,
            cap_k: layout.cap_k,
            merged_cap_bytes,
        },
    );
    Ok(Some((a_packed, a_scale, merged_out, merged_cap_bytes)))
}

#[cfg(feature = "cuda")]
#[derive(Clone, Copy)]
struct ProjNvfp4Kernels {
    quant_static: KernelHandle,
    absmax: KernelHandle,
    from_absmax: KernelHandle,
    combined_alpha: KernelHandle,
}

#[cfg(feature = "cuda")]
fn resolve_proj_nvfp4_kernels(gpu: &dyn GpuBackend) -> ProjNvfp4Kernels {
    let ptx_k = super::try_kernel(
        gpu,
        "quantize_bf16_to_nvfp4_cutlass",
        "quantize_bf16_to_nvfp4_atlas_128x4_ptx",
    );
    let legacy_k = super::try_kernel(
        gpu,
        "quantize_bf16_to_nvfp4_cutlass",
        "quantize_bf16_to_nvfp4_atlas_128x4",
    );
    let from_absmax_ptx = if prefill_fuse_absmax_enabled(gpu) {
        super::try_kernel(
            gpu,
            "quantize_nvfp4",
            "quantize_bf16_to_nvfp4_atlas_128x4_from_absmax_ptx",
        )
    } else {
        KernelHandle(0)
    };
    let from_absmax_legacy = super::try_kernel(
        gpu,
        "quantize_bf16_to_nvfp4_cutlass",
        "quantize_bf16_to_nvfp4_atlas_128x4_from_absmax",
    );
    ProjNvfp4Kernels {
        quant_static: select_cublaslt_quantize_128x4_kernel(ptx_k, legacy_k),
        absmax: super::try_kernel(gpu, "quantize_nvfp4", "nvfp4_global_absmax"),
        from_absmax: select_cublaslt_quantize_128x4_kernel(from_absmax_ptx, from_absmax_legacy),
        combined_alpha: super::try_kernel(
            gpu,
            "quantize_bf16_to_nvfp4_cutlass",
            "nvfp4_dynamic_combined_alpha",
        ),
    }
}

#[cfg(feature = "cuda")]
fn nvfp4_weight_ready(w: &QuantizedWeight) -> bool {
    !w.weight.is_null()
        && !w.weight_scale.is_null()
        && w.weight_scale_2.is_finite()
        && w.weight_scale_2 > 0.0
}

/// True when `ATLAS_27B_PROJ_CUBLASLT_NVFP4` is active and has every runtime
/// weight and kernel prerequisite needed to serve this `[n, k]` projection on
/// cuBLASLt NVFP4, allowing callers to skip eager FP8 weight materialization.
#[cfg(feature = "cuda")]
#[must_use]
pub(crate) fn proj_27b_cublaslt_nvfp4_can_replace_fp8(
    gpu: &dyn GpuBackend,
    weight: Option<&QuantizedWeight>,
    n: u32,
    k: u32,
) -> bool {
    if !proj_27b_cublaslt_nvfp4_enabled(gpu.compute_capability()) {
        return false;
    }
    if n == 0 || k == 0 || !n.is_multiple_of(8) || !k.is_multiple_of(64) {
        return false;
    }
    let Some(w) = weight else {
        return false;
    };
    if !nvfp4_weight_ready(w) {
        return false;
    }
    let ks = resolve_proj_nvfp4_kernels(gpu);
    let has_dynamic = ks.absmax.0 != 0 && ks.from_absmax.0 != 0 && ks.combined_alpha.0 != 0;
    let has_static = !w.input_scale.is_null() && ks.quant_static.0 != 0;
    has_dynamic || has_static
}

#[cfg(not(feature = "cuda"))]
#[must_use]
pub(crate) fn proj_27b_cublaslt_nvfp4_can_replace_fp8(
    _gpu: &dyn GpuBackend,
    _weight: Option<&QuantizedWeight>,
    _n: u32,
    _k: u32,
) -> bool {
    false
}

/// Route a single NVFP4 prefill projection (`ssm_qkvz`, `ssm_out_proj`, or
/// `attn_o_proj`) through `cublaslt::nvfp4_gemm_block_scaled` with cached
/// `128x4`-swizzled UE4M3 scales and zero-sync activation quantization.
#[cfg(feature = "cuda")]
#[expect(
    clippy::too_many_arguments,
    reason = "projection GEMM dispatcher forwarding activation, weight, output, shape, label, and stream"
)]
pub(crate) fn try_cublaslt_nvfp4_proj_single(
    gpu: &dyn GpuBackend,
    act_bf16: DevicePtr,
    weight: &QuantizedWeight,
    out_bf16: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    label: &'static str,
    stream: u64,
) -> Result<bool> {
    if !proj_27b_cublaslt_nvfp4_enabled(gpu.compute_capability())
        || !proj_27b_cublaslt_nvfp4_shape_ok(m, n, k)
        || !nvfp4_weight_ready(weight)
    {
        return Ok(false);
    }
    let ks = resolve_proj_nvfp4_kernels(gpu);
    let cached = cached_proj_nvfp4_scale(gpu, weight, n, k)?;
    let can_static = cached.input_scale.is_some() && ks.quant_static.0 != 0;
    let can_dynamic = ks.absmax.0 != 0 && ks.from_absmax.0 != 0 && ks.combined_alpha.0 != 0;
    if !can_static && !can_dynamic {
        return Ok(false);
    }

    let mut scratch_guard = proj_nvfp4_scratch_lock()
        .lock()
        .map_err(|_| anyhow::anyhow!("shared proj NVFP4 scratch lock poisoned"))?;
    let (a_packed, a_scale, dynamic_scalars) =
        ensure_proj_nvfp4_scratch(&mut scratch_guard, gpu, m as usize, k as usize, stream)?;
    let absmax_state = if can_static {
        clear_precomputed_proj_absmax();
        DynamicAbsmaxState::NeedsScan
    } else {
        take_precomputed_proj_absmax(gpu, act_bf16, m, k, stream)
    };
    let input_scale = launch_quantize_bf16_to_nvfp4_128x4_zero_sync(
        gpu,
        ks.quant_static,
        ks.absmax,
        ks.from_absmax,
        ks.combined_alpha,
        act_bf16,
        a_packed,
        a_scale,
        dynamic_scalars,
        DevicePtr::NULL,
        m,
        k,
        if can_static { cached.input_scale } else { None },
        absmax_state,
        stream,
    )?;
    let scalars = resolve_cublaslt_scalars(
        gpu,
        ks.combined_alpha,
        input_scale,
        weight.weight_scale_2,
        CublasLtAlphaSlot::Primary,
        stream,
    )?;
    spark_runtime::cublaslt::nvfp4_gemm_block_scaled(
        a_packed.0,
        a_scale.0,
        weight.weight.0,
        cached.scales_128x4.0,
        out_bf16.0,
        m,
        n,
        k,
        n,
        scalars,
        stream,
    )?;

    static LOGGED: std::sync::Once = std::sync::Once::new();
    LOGGED.call_once(|| {
        tracing::info!(
            label,
            m,
            n,
            k,
            static_scale = can_static,
            "ENGAGED ATLAS_27B_PROJ_CUBLASLT_NVFP4: cuBLASLt NVFP4 prefill projection"
        );
    });
    Ok(true)
}

#[cfg(not(feature = "cuda"))]
#[expect(
    clippy::too_many_arguments,
    reason = "non-CUDA stub matching CUDA dispatcher signature"
)]
pub(crate) fn try_cublaslt_nvfp4_proj_single(
    _gpu: &dyn GpuBackend,
    _act_bf16: DevicePtr,
    _weight: &QuantizedWeight,
    _out_bf16: DevicePtr,
    _m: u32,
    _n: u32,
    _k: u32,
    _label: &'static str,
    _stream: u64,
) -> Result<bool> {
    Ok(false)
}

/// Route Attention `q_proj`, `k_proj`, and `v_proj` prefill GEMMs through
/// `cublaslt::nvfp4_gemm_block_scaled` with a **single shared activation
/// quantization** and cached `128x4`-swizzled UE4M3 weight scales.
#[cfg(feature = "cuda")]
#[expect(
    clippy::too_many_arguments,
    reason = "shared-activation Q/K/V projection dispatcher forwarding three weight/output pairs"
)]
pub(crate) fn try_cublaslt_nvfp4_proj_qkv(
    gpu: &dyn GpuBackend,
    act_bf16: DevicePtr,
    q_weight: &QuantizedWeight,
    k_weight: &QuantizedWeight,
    v_weight: &QuantizedWeight,
    q_out: DevicePtr,
    k_out: DevicePtr,
    v_out: DevicePtr,
    m: u32,
    q_n: u32,
    kv_n: u32,
    k: u32,
    stream: u64,
) -> Result<bool> {
    if !proj_27b_cublaslt_nvfp4_enabled(gpu.compute_capability())
        || !proj_27b_cublaslt_nvfp4_shape_ok(m, q_n, k)
        || !proj_27b_cublaslt_nvfp4_shape_ok(m, kv_n, k)
        || !nvfp4_weight_ready(q_weight)
        || !nvfp4_weight_ready(k_weight)
        || !nvfp4_weight_ready(v_weight)
    {
        return Ok(false);
    }
    let ks = resolve_proj_nvfp4_kernels(gpu);
    let q_cached = cached_proj_nvfp4_scale(gpu, q_weight, q_n, k)?;
    let k_cached = cached_proj_nvfp4_scale(gpu, k_weight, kv_n, k)?;
    let v_cached = cached_proj_nvfp4_scale(gpu, v_weight, kv_n, k)?;
    let shared_static_scale = match (
        q_cached.input_scale,
        k_cached.input_scale,
        v_cached.input_scale,
    ) {
        (Some(qs), Some(ks_s), Some(vs)) if ks.quant_static.0 != 0 => Some(qs.max(ks_s).max(vs)),
        _ => None,
    };
    let can_dynamic = ks.absmax.0 != 0 && ks.from_absmax.0 != 0 && ks.combined_alpha.0 != 0;
    if shared_static_scale.is_none() && !can_dynamic {
        return Ok(false);
    }

    let mut scratch_guard = proj_nvfp4_scratch_lock()
        .lock()
        .map_err(|_| anyhow::anyhow!("shared proj NVFP4 scratch lock poisoned"))?;
    let (a_packed, a_scale, dynamic_scalars) =
        ensure_proj_nvfp4_scratch(&mut scratch_guard, gpu, m as usize, k as usize, stream)?;
    let absmax_state = if shared_static_scale.is_some() {
        clear_precomputed_proj_absmax();
        DynamicAbsmaxState::NeedsScan
    } else {
        take_precomputed_proj_absmax(gpu, act_bf16, m, k, stream)
    };
    let input_scale = launch_quantize_bf16_to_nvfp4_128x4_zero_sync(
        gpu,
        ks.quant_static,
        ks.absmax,
        ks.from_absmax,
        ks.combined_alpha,
        act_bf16,
        a_packed,
        a_scale,
        dynamic_scalars,
        DevicePtr::NULL,
        m,
        k,
        shared_static_scale,
        absmax_state,
        stream,
    )?;

    for (weight, scales_128x4, out_ptr, n_dim, slot) in [
        (
            q_weight,
            q_cached.scales_128x4,
            q_out,
            q_n,
            CublasLtAlphaSlot::Primary,
        ),
        (
            k_weight,
            k_cached.scales_128x4,
            k_out,
            kv_n,
            CublasLtAlphaSlot::Secondary,
        ),
        (
            v_weight,
            v_cached.scales_128x4,
            v_out,
            kv_n,
            CublasLtAlphaSlot::Tertiary,
        ),
    ] {
        let scalars = resolve_cublaslt_scalars(
            gpu,
            ks.combined_alpha,
            input_scale,
            weight.weight_scale_2,
            slot,
            stream,
        )?;
        spark_runtime::cublaslt::nvfp4_gemm_block_scaled(
            a_packed.0,
            a_scale.0,
            weight.weight.0,
            scales_128x4.0,
            out_ptr.0,
            m,
            n_dim,
            k,
            n_dim,
            scalars,
            stream,
        )?;
    }

    static LOGGED_QKV: std::sync::Once = std::sync::Once::new();
    LOGGED_QKV.call_once(|| {
        tracing::info!(
            m,
            q_n,
            kv_n,
            k,
            static_scale = shared_static_scale.is_some(),
            "ENGAGED ATLAS_27B_PROJ_CUBLASLT_NVFP4: shared-quant Attention Q/K/V prefill"
        );
    });
    Ok(true)
}

#[cfg(not(feature = "cuda"))]
#[expect(
    clippy::too_many_arguments,
    reason = "non-CUDA stub matching CUDA shared Q/K/V dispatcher signature"
)]
pub(crate) fn try_cublaslt_nvfp4_proj_qkv(
    _gpu: &dyn GpuBackend,
    _act_bf16: DevicePtr,
    _q_weight: &QuantizedWeight,
    _k_weight: &QuantizedWeight,
    _v_weight: &QuantizedWeight,
    _q_out: DevicePtr,
    _k_out: DevicePtr,
    _v_out: DevicePtr,
    _m: u32,
    _q_n: u32,
    _kv_n: u32,
    _k: u32,
    _stream: u64,
) -> Result<bool> {
    Ok(false)
}

/// One W4A4/FlashInfer activation arena and stream binding shared by every
/// dense FFN layer of a model (`ATLAS_MEMLEAN_SHARED_SCRATCH`). Layers run
/// one after another on the one stream the binding admits, and each launch
/// group fully rewrites the rows it reads, so one arena serves all of them;
/// the per-layer copies were 64 x 80 MB on Qwen3.8-27B.
#[derive(Clone, Default)]
pub struct SharedE2m1Scratch {
    scratch: Arc<Mutex<Option<E2m1Scratch>>>,
    stream: Arc<Mutex<Option<u64>>>,
}

/// Packed-weight views of gate and up inside the merged gate-then-up
/// FlashInfer operand. `build_flashinfer_merged_gate_up` uploads the gate
/// bytes followed by the up bytes, so the two views read exactly the bytes
/// the separate checkpoint allocations held.
#[cfg(all(feature = "cuda", target_os = "linux"))]
fn merged_gate_up_views(
    merged: DevicePtr,
    gate_packed_bytes: usize,
) -> Result<(DevicePtr, DevicePtr)> {
    ensure!(!merged.is_null(), "merged gate/up operand is null");
    ensure!(
        gate_packed_bytes > 0 && gate_packed_bytes.is_multiple_of(256),
        "gate packed extent {gate_packed_bytes} must be a positive multiple of 256 bytes"
    );
    Ok((merged, merged.offset(gate_packed_bytes)))
}

/// `m_qualified` is supplied by the caller rather than matched inline so the
/// extended ladder (`ATLAS_FLASHINFER_FFN_EXTRA_M=1`) shares one source of truth
/// with `qwen38_ffn_launch_candidate`; the frozen pair stays the default.
const fn flashinfer_ffn_prefill_route(
    requested: bool,
    exact_qwen38_dense: bool,
    m_qualified: bool,
    prepared: bool,
) -> FlashinferFfnPrefillRoute {
    if !requested {
        FlashinferFfnPrefillRoute::Disabled
    } else if !exact_qwen38_dense || !m_qualified {
        FlashinferFfnPrefillRoute::Ineligible
    } else if !prepared {
        FlashinferFfnPrefillRoute::Missing
    } else {
        FlashinferFfnPrefillRoute::Complete
    }
}

const fn e2m1_kmajor_projection_shape(m: u32, n: u32, k: u32) -> bool {
    m >= 128 && n.is_multiple_of(128) && k.is_multiple_of(64)
}

const E2M1_KMAJOR_M256_MIN_M: u32 = 2048;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum E2m1KmajorKernel {
    M128,
    M256,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum E2m1Scope {
    Full,
    DownOnly,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum E2m1PrefillRoute {
    Disabled,
    Ineligible,
    Complete(E2m1Scope),
    Missing(E2m1Scope),
}

const fn e2m1_prefill_route(
    full_requested: bool,
    down_requested: bool,
    shape_eligible: bool,
    full_ready: bool,
    down_ready: bool,
) -> E2m1PrefillRoute {
    let scope = if full_requested {
        E2m1Scope::Full
    } else if down_requested {
        E2m1Scope::DownOnly
    } else {
        return E2m1PrefillRoute::Disabled;
    };
    if !shape_eligible {
        return E2m1PrefillRoute::Ineligible;
    }
    match scope {
        E2m1Scope::Full if full_ready => E2m1PrefillRoute::Complete(scope),
        E2m1Scope::DownOnly if down_ready => E2m1PrefillRoute::Complete(scope),
        E2m1Scope::Full | E2m1Scope::DownOnly => E2m1PrefillRoute::Missing(scope),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum E2m1SelectedKernel {
    RowMajorM64,
    KmajorM128,
    KmajorM256,
}

const fn e2m1_selected_kernel(kmajor: bool, m256_requested: bool, m: u32) -> E2m1SelectedKernel {
    if !kmajor {
        E2m1SelectedKernel::RowMajorM64
    } else if matches!(
        e2m1_kmajor_kernel(m, m256_requested),
        E2m1KmajorKernel::M256
    ) {
        E2m1SelectedKernel::KmajorM256
    } else {
        E2m1SelectedKernel::KmajorM128
    }
}

const fn e2m1_kmajor_kernel(m: u32, m256_requested: bool) -> E2m1KmajorKernel {
    if m256_requested && m >= E2M1_KMAJOR_M256_MIN_M {
        E2m1KmajorKernel::M256
    } else {
        E2m1KmajorKernel::M128
    }
}

fn validated_e2m1_checkpoint_scales(gate: f32, up: f32, down: f32) -> Result<E2m1CheckpointScales> {
    for (name, value) in [("gate", gate), ("up", up), ("down", down)] {
        ensure!(
            value.is_finite() && value > 0.0,
            "ATLAS_E2M1_STATIC_SCALE requires a finite positive {name} input_scale; got {value}"
        );
    }
    Ok(E2m1CheckpointScales {
        gate_up: gate.max(up),
        down,
    })
}

fn read_e2m1_input_scale(gpu: &dyn GpuBackend, ptr: DevicePtr, projection: &str) -> Result<f32> {
    ensure!(
        ptr != DevicePtr::NULL,
        "ATLAS_E2M1_STATIC_SCALE requires {projection}.input_scale in the checkpoint"
    );
    let mut bytes = [0u8; 4];
    gpu.copy_d2h(ptr, &mut bytes)
        .with_context(|| format!("read {projection}.input_scale for ATLAS_E2M1_STATIC_SCALE"))?;
    Ok(f32::from_le_bytes(bytes))
}

#[cfg(all(feature = "cuda", target_os = "linux"))]
fn flashinfer_scale_fingerprint(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

#[cfg(all(feature = "cuda", target_os = "linux"))]
fn ensure_flashinfer_ffn_disjoint(
    left: DevicePtr,
    left_len: usize,
    right: DevicePtr,
    right_len: usize,
) -> Result<()> {
    let left_end = left
        .0
        .checked_add(u64::try_from(left_len).context("left FFN pointer extent exceeds u64")?)
        .context("left FFN pointer range overflow")?;
    let right_end = right
        .0
        .checked_add(u64::try_from(right_len).context("right FFN pointer extent exceeds u64")?)
        .context("right FFN pointer range overflow")?;
    ensure!(
        left_end <= right.0 || right_end <= left.0,
        "FlashInfer FFN buffers overlap"
    );
    Ok(())
}

#[cfg(all(feature = "cuda", target_os = "linux"))]
fn flashinfer_ffn_extent(rows: usize, cols: usize, element_bytes: usize) -> Result<usize> {
    rows.checked_mul(cols)
        .and_then(|elements| elements.checked_mul(element_bytes))
        .context("FlashInfer FFN buffer extent overflow")
}

#[cfg(all(feature = "cuda", target_os = "linux"))]
fn flashinfer_ffn_activation_scale_extent(rows: usize, cols: usize) -> Result<usize> {
    use crate::weight_map::cutlass_scale_layout::{CUTLASS_SCALE_ROW_TILE, NVFP4_GROUP_SIZE};

    ensure!(
        rows > 0,
        "FlashInfer FFN activation-scale extent requires rows > 0"
    );
    ensure!(
        cols >= NVFP4_GROUP_SIZE && cols.is_multiple_of(NVFP4_GROUP_SIZE),
        "FlashInfer FFN activation-scale extent requires cols divisible by {NVFP4_GROUP_SIZE}"
    );
    let padded_rows = rows
        .checked_add(CUTLASS_SCALE_ROW_TILE - 1)
        .context("FlashInfer FFN activation-scale row padding overflow")?
        / CUTLASS_SCALE_ROW_TILE;
    let padded_rows = padded_rows
        .checked_mul(CUTLASS_SCALE_ROW_TILE)
        .context("FlashInfer FFN padded activation-scale rows overflow")?;
    flashinfer_ffn_extent(padded_rows, cols / NVFP4_GROUP_SIZE, 1)
}

#[cfg(all(feature = "cuda", target_os = "linux"))]
fn read_flashinfer_ffn_checkpoint_projection(
    gpu: &dyn GpuBackend,
    layer: usize,
    projection: crate::weight_map::flashinfer_ffn_admission::Qwen38FfnProjection,
    weight: &QuantizedWeight,
    n: usize,
    k: usize,
) -> Result<(
    Vec<u8>,
    Vec<u8>,
    crate::weight_map::modelopt_scale_admission::AdmittedModeloptScale,
)> {
    use crate::weight_map::cutlass_scale_layout::NVFP4_GROUP_SIZE;
    use crate::weight_map::modelopt_scale_admission::{
        ModeloptScaleProjection, ModeloptScaleSource, admit_modelopt_checkpoint_scale,
    };
    use spark_runtime::weights::WeightDtype;

    ensure!(
        !weight.weight.is_null() && !weight.weight_scale.is_null() && !weight.input_scale.is_null(),
        "FlashInfer FFN {projection:?} requires original non-NULL ModelOpt operands"
    );
    ensure!(
        n > 0 && k > 0 && k.is_multiple_of(NVFP4_GROUP_SIZE),
        "FlashInfer FFN {projection:?} has invalid N/K geometry"
    );
    let packed_len = n
        .checked_mul(k / 2)
        .context("FlashInfer FFN packed-weight length overflow")?;
    let scale_len = n
        .checked_mul(k / NVFP4_GROUP_SIZE)
        .context("FlashInfer FFN weight-scale length overflow")?;
    let mut packed = vec![0_u8; packed_len];
    let mut logical_scales = vec![0_u8; scale_len];
    let mut input_scale_bytes = [0_u8; 4];
    gpu.copy_d2h(weight.weight, &mut packed)
        .with_context(|| format!("read ModelOpt FFN {projection:?} packed weight"))?;
    gpu.copy_d2h(weight.weight_scale, &mut logical_scales)
        .with_context(|| format!("read ModelOpt FFN {projection:?} block scales"))?;
    gpu.copy_d2h(weight.input_scale, &mut input_scale_bytes)
        .with_context(|| format!("read ModelOpt FFN {projection:?} input scale"))?;
    let modelopt_projection = match projection {
        crate::weight_map::flashinfer_ffn_admission::Qwen38FfnProjection::Gate => {
            ModeloptScaleProjection::FfnGate
        }
        crate::weight_map::flashinfer_ffn_admission::Qwen38FfnProjection::Up => {
            ModeloptScaleProjection::FfnUp
        }
        crate::weight_map::flashinfer_ffn_admission::Qwen38FfnProjection::Down => {
            ModeloptScaleProjection::FfnDown
        }
    };
    let input_scale = admit_modelopt_checkpoint_scale(
        ModeloptScaleSource {
            layer,
            projection: modelopt_projection,
        },
        WeightDtype::FP32,
        &[],
        weight.input_scale,
        input_scale_bytes,
    )?;
    Ok((packed, logical_scales, input_scale))
}

#[cfg(all(feature = "cuda", target_os = "linux"))]
fn build_flashinfer_merged_gate_up(
    gpu: &dyn GpuBackend,
    layer: usize,
    gate: &QuantizedWeight,
    up: &QuantizedWeight,
) -> Result<FlashinferMergedGateUp> {
    use crate::weight_map::flashinfer_ffn_admission::{
        QWEN38_HIDDEN, QWEN38_INTERMEDIATE, Qwen38FfnCheckpointProjection, Qwen38FfnProjection,
        Qwen38FfnSource, admit_qwen38_merged_gate_up,
    };

    let (gate_packed, gate_scales, gate_input_scale) = read_flashinfer_ffn_checkpoint_projection(
        gpu,
        layer,
        Qwen38FfnProjection::Gate,
        gate,
        QWEN38_INTERMEDIATE,
        QWEN38_HIDDEN,
    )?;
    let (up_packed, up_scales, up_input_scale) = read_flashinfer_ffn_checkpoint_projection(
        gpu,
        layer,
        Qwen38FfnProjection::Up,
        up,
        QWEN38_INTERMEDIATE,
        QWEN38_HIDDEN,
    )?;
    let admitted = admit_qwen38_merged_gate_up(
        layer,
        Qwen38FfnCheckpointProjection {
            source: Qwen38FfnSource {
                layer,
                projection: Qwen38FfnProjection::Gate,
            },
            packed_weight: &gate_packed,
            logical_weight_scales: &gate_scales,
            input_scale: gate_input_scale,
            weight_scale_2_le_bytes: gate.weight_scale_2.to_le_bytes(),
        },
        Qwen38FfnCheckpointProjection {
            source: Qwen38FfnSource {
                layer,
                projection: Qwen38FfnProjection::Up,
            },
            packed_weight: &up_packed,
            logical_weight_scales: &up_scales,
            input_scale: up_input_scale,
            weight_scale_2_le_bytes: up.weight_scale_2.to_le_bytes(),
        },
    )?;

    let merged_weight = gpu.alloc(admitted.packed_weight().len())?;
    gpu.copy_h2d(admitted.packed_weight(), merged_weight)
        .context("upload immutable merged FFN gate/up packed weight")?;
    let merged_scales = gpu.alloc(admitted.physical_weight_scales().len())?;
    gpu.copy_h2d(admitted.physical_weight_scales(), merged_scales)
        .context("upload immutable merged FFN gate/up physical scales")?;
    let alpha_f32 = gpu.alloc(std::mem::size_of::<f32>())?;
    gpu.copy_h2d(&admitted.shared_alpha_bits().to_le_bytes(), alpha_f32)
        .context("upload immutable merged FFN gate/up alpha")?;

    Ok(FlashinferMergedGateUp {
        layer,
        weight: merged_weight,
        weight_scales_128x4: merged_scales,
        weight_hash: flashinfer_scale_fingerprint(admitted.packed_weight()),
        weight_scales_hash: flashinfer_scale_fingerprint(admitted.physical_weight_scales()),
        alpha_f32,
        input_scale_bits: admitted.input_scales()[0].value_bits(),
        alpha_bits: admitted.shared_alpha_bits(),
        source_gate_weight: gate.weight,
        source_up_weight: up.weight,
    })
}

#[cfg(all(feature = "cuda", target_os = "linux"))]
fn build_flashinfer_ffn_projection(
    gpu: &dyn GpuBackend,
    layer: usize,
    projection: crate::weight_map::flashinfer_ffn_admission::Qwen38FfnProjection,
    weight: &QuantizedWeight,
    n: usize,
    k: usize,
) -> Result<FlashinferFfnProjection> {
    use crate::weight_map::cutlass_scale_layout::{
        NVFP4_GROUP_SIZE, deinterleave_nvfp4_scales_128x4, interleave_nvfp4_scales_128x4,
    };
    use crate::weight_map::modelopt_scale_admission::{
        ModeloptScaleProjection, ModeloptScaleSource, admit_modelopt_checkpoint_scale,
    };
    use spark_runtime::weights::WeightDtype;

    ensure!(
        !weight.weight.is_null(),
        "FlashInfer FFN packed weight is null"
    );
    ensure!(
        !weight.weight_scale.is_null(),
        "FlashInfer FFN block-scale pointer is null"
    );
    ensure!(
        !weight.input_scale.is_null(),
        "FlashInfer FFN input-scale pointer is null"
    );
    ensure!(
        n > 0 && k > 0 && k.is_multiple_of(NVFP4_GROUP_SIZE),
        "FlashInfer FFN projection has invalid N/K geometry"
    );

    let groups = k / NVFP4_GROUP_SIZE;
    let scale_len = n
        .checked_mul(groups)
        .context("FlashInfer FFN weight-scale length overflow")?;
    let mut logical_scales = vec![0_u8; scale_len];
    gpu.copy_d2h(weight.weight_scale, &mut logical_scales)
        .context("read ModelOpt FFN block scales for CUTLASS interleave")?;
    let physical_scales =
        interleave_nvfp4_scales_128x4(&logical_scales, &[n, groups], NVFP4_GROUP_SIZE)?;
    ensure!(
        deinterleave_nvfp4_scales_128x4(&physical_scales, &[n, groups], NVFP4_GROUP_SIZE,)?
            == logical_scales,
        "FlashInfer FFN physical block scales failed exact round trip"
    );

    let mut input_scale_bytes = [0_u8; 4];
    gpu.copy_d2h(weight.input_scale, &mut input_scale_bytes)
        .context("read ModelOpt FFN input scale")?;
    let modelopt_projection = match projection {
        crate::weight_map::flashinfer_ffn_admission::Qwen38FfnProjection::Gate => {
            ModeloptScaleProjection::FfnGate
        }
        crate::weight_map::flashinfer_ffn_admission::Qwen38FfnProjection::Up => {
            ModeloptScaleProjection::FfnUp
        }
        crate::weight_map::flashinfer_ffn_admission::Qwen38FfnProjection::Down => {
            ModeloptScaleProjection::FfnDown
        }
    };
    let input_scale = admit_modelopt_checkpoint_scale(
        ModeloptScaleSource {
            layer,
            projection: modelopt_projection,
        },
        WeightDtype::FP32,
        &[],
        weight.input_scale,
        input_scale_bytes,
    )?;
    ensure!(
        weight.weight_scale_2.is_finite() && weight.weight_scale_2 > 0.0,
        "FlashInfer FFN weight_scale_2 must be finite and positive"
    );
    let alpha = input_scale.value() * weight.weight_scale_2;
    ensure!(
        alpha.is_finite() && alpha > 0.0,
        "FlashInfer FFN alpha must be finite and positive"
    );

    let weight_scales_128x4 = gpu.alloc(physical_scales.len())?;
    gpu.copy_h2d(&physical_scales, weight_scales_128x4)
        .context("upload FlashInfer FFN physical block scales")?;
    let alpha_f32 = gpu.alloc(std::mem::size_of::<f32>())?;
    gpu.copy_h2d(&alpha.to_le_bytes(), alpha_f32)
        .context("upload FlashInfer FFN alpha")?;

    Ok(FlashinferFfnProjection {
        weight: weight.weight,
        weight_scales_128x4,
        weight_scales_hash: flashinfer_scale_fingerprint(&physical_scales),
        alpha_f32,
        input_scale_bits: input_scale.value_bits(),
        alpha_bits: alpha.to_bits(),
    })
}

#[cfg(all(feature = "cuda", target_os = "linux"))]
fn log_flashinfer_ffn_success(
    layer: usize,
    operation: crate::weight_map::flashinfer_ffn_admission::Qwen38FfnOperation,
    m: usize,
    tactic: usize,
) {
    use crate::weight_map::flashinfer_ffn_admission::Qwen38FfnOperation;
    use std::collections::HashSet;

    static RECEIPTS: std::sync::OnceLock<Mutex<HashSet<(u8, usize, usize)>>> =
        std::sync::OnceLock::new();
    let (family_id, family) = match operation {
        Qwen38FfnOperation::MergedGateUp => (0u8, "ffn_merged_gate_up"),
        Qwen38FfnOperation::Down => (1u8, "ffn_down"),
        Qwen38FfnOperation::Gate | Qwen38FfnOperation::Up => {
            unreachable!("production FFN route does not launch separate gate/up")
        }
    };
    let receipts = RECEIPTS.get_or_init(|| Mutex::new(HashSet::new()));
    let newly_logged = receipts
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .insert((family_id, m, tactic));
    if !newly_logged {
        return;
    }
    tracing::info!(
        layer,
        m,
        tactic,
        family,
        "routed Qwen3.8 dense FFN prefill through FlashInfer SM121"
    );
    eprintln!(
        "ATLAS_PREFILL_FFN_FLASHINFER ENGAGED family={family} layer={layer} M={m} tactic={tactic}"
    );
}

fn load_e2m1_checkpoint_scales(
    weights: &DenseFfnWeights,
    gpu: &dyn GpuBackend,
) -> Result<Option<E2m1CheckpointScales>> {
    let flashinfer_requested = crate::layers::prefill_ffn_flashinfer_enabled()?;
    let requested = flashinfer_requested
        || std::env::var("ATLAS_E2M1_STATIC_SCALE").ok().as_deref() == Some("1");
    if !requested {
        return Ok(None);
    }
    let full_w4a4 = std::env::var("ATLAS_E2M1_GEMM").ok().as_deref() == Some("1");
    let down_w4a4 = std::env::var("ATLAS_E2M1_GEMM_DOWN_ONLY").ok().as_deref() == Some("1");
    ensure!(
        full_w4a4 || down_w4a4 || flashinfer_requested,
        "ATLAS_E2M1_STATIC_SCALE=1 requires an active W4A4 or FlashInfer FFN route"
    );

    let scales = validated_e2m1_checkpoint_scales(
        read_e2m1_input_scale(gpu, weights.gate_proj.input_scale, "gate_proj")?,
        read_e2m1_input_scale(gpu, weights.up_proj.input_scale, "up_proj")?,
        read_e2m1_input_scale(gpu, weights.down_proj.input_scale, "down_proj")?,
    )?;
    tracing::info!(
        gate_up_input_scale = scales.gate_up,
        down_input_scale = scales.down,
        "enabled checkpoint-static W4A4 activation scales"
    );
    Ok(Some(scales))
}

/// Opportunistically load ModelOpt checkpoint `input_scale` scalars for the
/// cuBLASLt NVFP4 prefill route on SM103/SM120 (or explicit `ATLAS_CUBLASLT_NVFP4=1`),
/// eliminating per-layer `nvfp4_global_absmax` + GPU stream syncs when present.
/// Falls back cleanly to `None` (dynamic absmax) if any pointer is null or invalid.
#[cfg(feature = "cuda")]
fn try_load_cublaslt_checkpoint_scales(
    weights: &DenseFfnWeights,
    gpu: &dyn GpuBackend,
    existing: Option<E2m1CheckpointScales>,
) -> Option<E2m1CheckpointScales> {
    if let Some(scales) = existing {
        return Some(scales);
    }
    let policy = prefill_ffn_cublaslt_nvfp4_policy();
    let cc = gpu.compute_capability();
    let may_engage = match policy {
        CublasLtNvfp4Policy::Disabled => false,
        CublasLtNvfp4Policy::Auto => {
            is_blackwell_dc_compute_capability(cc) || is_sm120_compute_capability(cc)
        }
        CublasLtNvfp4Policy::Enabled => cc.is_some(),
    };
    if !may_engage
        || weights.gate_proj.input_scale.is_null()
        || weights.up_proj.input_scale.is_null()
        || weights.down_proj.input_scale.is_null()
    {
        return None;
    }
    let gate = read_e2m1_input_scale(gpu, weights.gate_proj.input_scale, "gate_proj").ok()?;
    let up = read_e2m1_input_scale(gpu, weights.up_proj.input_scale, "up_proj").ok()?;
    let down = read_e2m1_input_scale(gpu, weights.down_proj.input_scale, "down_proj").ok()?;
    validated_e2m1_checkpoint_scales(gate, up, down).ok()
}

pub struct DenseFfnWeights {
    pub gate_proj: QuantizedWeight,
    pub up_proj: QuantizedWeight,
    pub down_proj: QuantizedWeight,
}

/// BF16 dense MLP weights — alternative to NVFP4 for precision-sensitive
/// models (Gemma-4-31B). Each is `[N, K]` row-major BF16. When installed
/// on a `DenseFfnLayer` via `set_bf16_weights`, the forward paths
/// dispatch to `dense_gemv_bf16` / `dense_gemm_bf16` instead of the
/// w4a16 NVFP4 kernels. Costs ~3.4 GB extra GPU memory on Gemma-4-31B
/// (3 × hidden×intermediate × 2 bytes) vs NVFP4's 0.5 bytes/weight.
pub struct DenseFfnWeightsBf16 {
    pub gate_proj: DenseWeight,
    pub up_proj: DenseWeight,
    pub down_proj: DenseWeight,
}

/// Activation function for gated FFN (SiLU for Qwen/Llama, GELU for Gemma-4).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FfnActivation {
    SiLU,
    GeLU,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct E2m1RouteProof {
    scope: E2m1Scope,
    kernel: E2m1SelectedKernel,
    checkpoint_static: bool,
    fused_silu_input: bool,
    activation: FfnActivation,
    m: u32,
    h: u32,
    inter: u32,
}

fn log_e2m1_route_once(proof: E2m1RouteProof) {
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
enum PrefillFfnFastRoute {
    Disabled,
    Ineligible,
    Complete,
    Missing,
}

const fn prefill_ffn_fast_route(
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

fn log_prefill_ffn_fast_route_once(m: u32, h: u32, inter: u32) {
    static FAST: std::sync::Once = std::sync::Once::new();
    FAST.call_once(|| {
        tracing::info!(
            "ENGAGED ATLAS_PREFILL_FFN_FAST: approximate transposed M128 route complete M={m} H={h} intermediate={inter}"
        );
    });
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PrefillFusedEpilogueRoute {
    Disabled,
    Ineligible,
    Complete,
    Missing,
}

fn prefill_fused_epilogue_route(
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

fn log_prefill_pipe_route_once() {
    static PIPE: std::sync::Once = std::sync::Once::new();
    PIPE.call_once(|| {
        tracing::info!("ENGAGED ATLAS_PREFILL_FFN_PIPE: ordinary exact pipe route");
    });
}

fn log_prefill_up_fused_route_once() {
    static UP_FUSED: std::sync::Once = std::sync::Once::new();
    UP_FUSED.call_once(|| {
        tracing::info!("ENGAGED ATLAS_PREFILL_FFN_FUSED_EPILOGUE: exact up-only fusion");
    });
}

fn log_prefill_dual_fused_route_once() {
    static DUAL_FUSED: std::sync::Once = std::sync::Once::new();
    DUAL_FUSED.call_once(|| {
        tracing::info!("ENGAGED ATLAS_PREFILL_FFN_DUAL_FUSED: exact gate/up fusion");
    });
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ExactFfnDispatch {
    Existing,
    Batched,
    PerRowK1,
}

const fn exact_ffn_dispatch(
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

const fn exact_ffn_auto_kgamma_applicable(rows: u32, exact_w4_silu_eligible: bool) -> bool {
    matches!(
        exact_ffn_dispatch(rows, exact_w4_silu_eligible, false),
        ExactFfnDispatch::PerRowK1
    )
}

const fn w3_kgamma_applicable(rows: u32, activation_silu: bool, prepared: bool) -> bool {
    rows > 1 && rows <= 32 && activation_silu && prepared
}

#[derive(Debug, Clone, Copy)]
struct ExactFfnMaterializedAvailability {
    m8_dual_silu: bool,
    m8_f32_down: bool,
    m17_dual_silu: bool,
    m17_f32_down: bool,
    m17_fused_dual_silu: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ExactFfnMaterializedRoute {
    Inline,
    Split,
    FusedM17,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ExactFfnPhysicalRoute {
    Native,
    SplitM8x2,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ExactFfnRowOffsets {
    input: usize,
    gate_up: usize,
    preactivation: usize,
    output: usize,
}

fn exact_ffn_row_offsets(
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

fn exact_ffn_physical_route(
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

fn exact_ffn_split_m8_enabled() -> bool {
    static GATE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *GATE.get_or_init(|| std::env::var("ATLAS_EXACT_FFN_SPLIT_M8").ok().as_deref() == Some("1"))
}

fn exact_ffn_lowreg_gate_up_m16_enabled() -> bool {
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
const FFN_KGAMMA_AUTO_TC_SM_THRESHOLD: u32 = 64;

/// Maximum batched row count served by wide `forward_kgamma` (`w4a16_gemm_t_m128`
/// or cuBLASLt NVFP4).
const FFN_KGAMMA_WIDE_MAX_ROWS: u32 = 256;

/// Policy for `ATLAS_FFN_TC`:
/// - `Disabled` (`"0"` / `"false"`): force bit-exact scalar-GEMV FFN even on `> 64`-SM GPUs.
/// - `Auto` (unset): off on `<= 64`-SM GPUs (GB10 100% unchanged); auto-enables batched
///   tensor-core `forward_kgamma` for `rows >= TC_VERIFY_MIN_ROWS` (`4..=256`) on `> 64`-SM GPUs.
/// - `Enabled` (`"1"` / `"true"`): force tensor-core FFN on any GPU (refreeze mode).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FfnTcPolicy {
    Disabled,
    Auto,
    Enabled,
}

#[must_use]
fn parse_ffn_tc_policy(raw: Option<&str>) -> FfnTcPolicy {
    match raw.map(str::trim) {
        Some(v) if v == "0" || v.eq_ignore_ascii_case("false") => FfnTcPolicy::Disabled,
        Some(v) if v == "1" || v.eq_ignore_ascii_case("true") => FfnTcPolicy::Enabled,
        _ => FfnTcPolicy::Auto,
    }
}

fn ffn_tc_policy() -> FfnTcPolicy {
    static POLICY: std::sync::OnceLock<FfnTcPolicy> = std::sync::OnceLock::new();
    *POLICY.get_or_init(|| {
        let raw = std::env::var("ATLAS_FFN_TC").ok();
        parse_ffn_tc_policy(raw.as_deref())
    })
}

/// Pure SM- and row-aware resolver for tensor-core `forward_kgamma`.
#[must_use]
const fn resolve_ffn_tc_for_rows(policy: FfnTcPolicy, sm_count: u32, rows: u32) -> bool {
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
fn resolve_kgamma_sm_auto_flag(raw: Option<&str>, sm_count: u32) -> bool {
    match raw.map(str::trim) {
        Some(v) if v == "0" || v.eq_ignore_ascii_case("false") => false,
        Some(v) if v == "1" || v.eq_ignore_ascii_case("true") => true,
        _ => sm_count > FFN_KGAMMA_AUTO_TC_SM_THRESHOLD,
    }
}

/// Explicit `ATLAS_FFN_TC=1` REFREEZE switch (used by single-token `decode_tc_parity_route`).
fn exact_ffn_tc_override() -> bool {
    matches!(ffn_tc_policy(), FfnTcPolicy::Enabled)
}

#[must_use]
fn ffn_tc_active_for_rows(rows: u32) -> bool {
    resolve_ffn_tc_for_rows(ffn_tc_policy(), atlas_core::device::active_sm_count(), rows)
}

#[must_use]
fn kgamma_wide_active() -> bool {
    static RAW: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    let raw = RAW.get_or_init(|| std::env::var("ATLAS_FFN_KGAMMA_WIDE").ok());
    resolve_kgamma_sm_auto_flag(raw.as_deref(), atlas_core::device::active_sm_count())
}

#[must_use]
fn kgamma_m128_active() -> bool {
    static RAW: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    let raw = RAW.get_or_init(|| std::env::var("ATLAS_FFN_KGAMMA_M128").ok());
    resolve_kgamma_sm_auto_flag(raw.as_deref(), atlas_core::device::active_sm_count())
}

#[must_use]
fn kgamma_m32_allowed() -> bool {
    static ALLOWED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ALLOWED.get_or_init(|| std::env::var("ATLAS_FFN_KGAMMA_M32").ok().as_deref() != Some("0"))
}

#[must_use]
fn kgamma_fused_gateup_active() -> bool {
    static RAW: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    let raw = RAW.get_or_init(|| std::env::var("ATLAS_FFN_FUSED_GATEUP").ok());
    resolve_kgamma_sm_auto_flag(raw.as_deref(), atlas_core::device::active_sm_count())
}

#[must_use]
fn resolve_kgamma_cublaslt_opt_in(raw: Option<&str>, sm_count: u32) -> bool {
    resolve_sm_auto_opt_in(raw, sm_count)
}

#[cfg(feature = "cuda")]
#[must_use]
fn kgamma_cublaslt_opt_in(gpu: &dyn GpuBackend) -> bool {
    static RAW: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    let raw = RAW.get_or_init(|| std::env::var("ATLAS_FFN_KGAMMA_CUBLASLT").ok());
    resolve_kgamma_cublaslt_opt_in(raw.as_deref(), gpu_sm_count(gpu))
}

fn exact_ffn_materialized_route(
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

pub struct DenseFfnLayer {
    pub weights: DenseFfnWeights,
    activation: FfnActivation,
    w4a16_gemv: KernelHandle,
    w4a16_gemv_dual: KernelHandle,
    w4a16_gemv_silu_input: KernelHandle,
    /// Exact K1-arithmetic-order W4 FFN kernels for dynamic M=2..=32.
    /// Missing selected-tier handles fail closed to independent K1 rows.
    exact_ffn_kernels: ops::W4a16ExactFfnKernels,
    /// Optional full-M16 gate/up projections. The diagnostic route is atomic;
    /// any missing handle retains the current split-M8 implementation.
    exact_ffn_lowreg_m16: ops::W4a16ExactFfnLowregM16Kernels,
    /// Single-warp-per-output M=1 decode GEMVs — lossless, 8 outputs per
    /// 256-thread block instead of 4, no cross-warp smem round-trip. Loaded
    /// via `try_kernel` so a kernel cache built before these symbols existed
    /// still links; `KernelHandle(0)` on a miss falls back to the 64-thread
    /// base kernels, which are bit-identical.
    w4a16_gemv_sw: KernelHandle,
    w4a16_gemv_dual_sw: KernelHandle,
    w4a16_gemv_silu_input_sw: KernelHandle,
    /// `ATLAS_NO_GEMV_SW != "1"`, cached at construction.
    gemv_sw: bool,
    w4a16_gemv_dual_batch2: KernelHandle,
    w4a16_gemv_dual_batch3: KernelHandle,
    /// Tuned dual-batch3 variant: fuses gate+up into the SAME CTA so the
    /// 3-token activation vector is loaded once per CTA. Gated behind the
    /// `ATLAS_FFN_DUAL_TUNED=1` env var; falls back to the baseline kernel
    /// when off. Loaded via `try_kernel` so older built kernel caches that
    /// pre-date this symbol still link.
    w4a16_gemv_dual_batch3_tuned: KernelHandle,
    w4a16_gemv_batch2: KernelHandle,
    w4a16_gemv_batch3: KernelHandle,
    w4a16_gemm: KernelHandle,
    /// cp.async double-buffered byte-exact shadow of `w4a16_gemm` for the
    /// prefill FFN path. Same dequant arithmetic + MMA order; only the load
    /// pipeline differs. Loaded via `try_kernel`; handle 0 causes an eligible
    /// explicit `ATLAS_PREFILL_FFN_PIPE=1` request to fail closed with `bail!`.
    w4a16_gemm_pipe: KernelHandle,
    /// Exact up-projection + SiLU(gate) epilogue variant used only by the
    /// opt-in large-M prefill pipe route. It preserves the BF16 up round trip
    /// while avoiding the full up activation buffer and standalone SiLU
    /// launch. Missing symbol fails closed on eligible explicit
    /// `ATLAS_PREFILL_FFN_FUSED_EPILOGUE=1` requests.
    w4a16_gemm_pipe_silu_mul: KernelHandle,
    /// Exact large-M gate+up+SiLU fusion. Both projection accumulators retain
    /// their BF16 materialization boundary, but A is loaded once and neither
    /// intermediate activation is written to global memory.
    w4a16_gemm_pipe_dual: KernelHandle,
    /// SiLU(gate)*up or GELU(gate)*up depending on activation.
    act_mul: KernelHandle,
    /// BF16 dense MLP weights — when `Some`, all forward paths use the
    /// `dense_gemv_bf16` / `dense_gemm_bf16` kernels instead of w4a16
    /// NVFP4. Falls back to the NVFP4 weights when `None`. Set via
    /// `set_bf16_weights`. Used by Gemma-4 dense to avoid the structural
    /// NVFP4 attention drift on greedy code generation (the fib test's
    /// broken-indentation pattern).
    bf16_weights: Option<DenseFfnWeightsBf16>,
    dense_gemv_bf16_k: KernelHandle,
    dense_gemm_bf16_k: KernelHandle,
    /// Transposed (`nvfp4_t` layout) FFN projections — populated when
    /// `ATLAS_FFN_M16_TRANSPOSED` is active (default ON unless `=0`) and the
    /// loader successfully built the transposed copies via
    /// `QuantizedWeight::transpose_for_gemm`. When `Some`, `forward_kgamma`
    /// routes gate/up/down through the `w4a16_gemm_t_m16` (M_TILE=16) kernel
    /// which has near-zero MMA accumulator waste at M=γ+1 (typically 17).
    /// Falls back to the non-transposed `w4a16_gemm` (M_TILE=64) when `None`
    /// OR when the `w4a16_gemm_t_m16` kernel symbol is missing.
    gate_proj_t: Option<QuantizedWeight>,
    up_proj_t: Option<QuantizedWeight>,
    down_proj_t: Option<QuantizedWeight>,
    /// The transposed copies were installed for the M>=128 prefill route only
    /// (`set_prefill_transposed_weights`), so that route does not also need
    /// `w4a16_gemm_t_m16`. `has_transposed_ffn()` still does, which keeps every
    /// decode / verify consumer of the copies off on targets without it.
    prefill_t_m128_only: bool,
    /// `w4a16_gemm_t_m16` — small-M (M_TILE=16) transposed-weight GEMM.
    /// Loaded via `try_kernel`; handle is 0 when the symbol is missing
    /// (older PTX caches), in which case `forward_kgamma` always uses the
    /// non-transposed fallback regardless of the transposed weights.
    w4a16_gemm_t_m16: KernelHandle,
    /// `w4a16_gemm_t_m16_n64` — small-M (M_TILE=16), small-N (N_TILE=64)
    /// transposed-weight GEMM, tuned for the K=3 MTP verify path on dense
    /// Qwen3.6-27B (M=3 padded to MMA-16). At intermediate=17408 the N=128
    /// parent only fields ~136 CTAs/projection (~2.8 CTAs/SM on GB10's 48 SMs),
    /// limiting occupancy; N=64 doubles the grid to ~272 CTAs/projection
    /// at half the per-CTA work. Loaded via `try_kernel`; handle 0 falls
    /// back to the GEMV path silently.
    w4a16_gemm_t_m16_n64: KernelHandle,
    /// `w4a16_gemm_t_m128` — large-M (M_TILE=128, N_TILE=128)
    /// transposed-weight GEMM. Loaded via `try_kernel`; handle is 0 when
    /// the symbol is missing. Used by `forward_prefill` when
    /// `ATLAS_PREFILL_FFN_FAST` is enabled (default ON unless `=0`; explicit
    /// `=1` fails closed when absent) AND transposed weights are installed
    /// AND M >= 128. Mirrors the attention `w4a16_gemm_t_m128_k`
    /// dispatch in `qwen3_attention/prefill_weights.rs`. Designed for
    /// large-M prefill: kernel comment claims ~2x speedup over
    /// `w4a16_gemm_t` at ISL>128. For Qwen3.6-27B prefill at M=3575,
    /// N=17408 (gate/up dual): grid=(136, 28)=3808 CTAs vs the default
    /// (272, 56)=15232 CTAs — 4x fewer CTAs but 4x more work per CTA,
    /// and ~2x less weight DRAM traffic.
    w4a16_gemm_t_m128: KernelHandle,
    /// `w4a16_gemm_t_m32_n64` — DFlash K=17 verify specialization:
    /// single B read (one 32-row M-tile) × 272 CTAs (N_TILE=64).
    /// Loaded via `try_kernel`; 0 falls back to m128/m16.
    w4a16_gemm_t_m32_n64: KernelHandle,
    /// `w4a16_gemm_t_m32_n64_gateup_silu` — FUSED gate_proj + up_proj +
    /// SiLU·mul for the K=γ verify path. Loads the shared [M,K] input
    /// tile once, streams BOTH transposed weights (gate + up), and writes
    /// only the fused silu(gate)*up [M,N] activation in one launch (vs the
    /// baseline's two m32_n64 GEMMs + standalone `moe_silu_mul`). Gated by
    /// `ATLAS_FFN_FUSED_GATEUP`. Loaded via `try_kernel`; handle 0 keeps
    /// the split gate/up path.
    w4a16_gemm_t_m32_n64_gateup_silu: KernelHandle,
    /// `w4a16_gemm_t_m32_n64_gateup_silu_pipe` — DEQUANT-IN-REGISTERS fork of
    /// the fused gate+up+SiLU kernel (`ATLAS_DEQUANT_PIPE=1`). Byte-exact
    /// with `w4a16_gemm_t_m32_n64_gateup_silu` (same shape, accumulation
    /// order, and BF16 round-trips) but the NVFP4→FP8 dequant runs in
    /// registers immediately before each MMA instead of via a `smem_B_fp8`
    /// staging array — dropping the 2nd per-K-step `__syncthreads`, shrinking
    /// SMEM ~27% (10.9 KB vs 15.0 KB → higher occupancy), and using
    /// `cp.async.wait_group<1>` so the next tile's load overlaps the current
    /// dequant+MMA. Loaded via `try_kernel`; handle 0 keeps the staged fused
    /// kernel as silent fallback.
    w4a16_gemm_t_m32_n64_gateup_silu_pipe: KernelHandle,
    /// `w4a16_gemm_t_m32_n64_gateup_silu_pipe_k64` — K_STEP=64 fork of the
    /// `_pipe` register-dequant kernel (`ATLAS_GATEUP_K64=1`). Doubles the
    /// K-tile from 32 to 64 elements: 80 K-loop iterations vs 160, halving
    /// sync count and loop overhead. Each step issues two m16n8k32 MMAs per
    /// accumulator (K[0..31] then K[32..63]) and 2× the cp.async volume
    /// (6 KB vs 3 KB per stage), so the background load overlaps more of the
    /// inline dequant+compute work. Takes priority over ATLAS_DEQUANT_PIPE.
    /// Requires K divisible by 64 (hidden_size 5120 qualifies). Handle 0
    /// silently falls back through _pipe → staged fused kernel.
    w4a16_gemm_t_m32_n64_gateup_silu_pipe_k64: KernelHandle,
    /// `w4a16_gemm_t_m32_n32_gateup_silu_pipe_k64` — `N_TILE=32, K_STEP=64`
    /// wide-SM fork (`ATLAS_GATEUP_N32`, auto on `> 128`-SM GPUs). Doubles CTA
    /// count (272 -> 544 at `N=17408`), halves registers (48 vs 96), and is
    /// 0-ULP byte-exact with the N64 fused kernels.
    w4a16_gemm_t_m32_n32_gateup_silu_pipe_k64: KernelHandle,
    /// `w4a16_gemm_t_m32_n64_splitk` — split-K variant of the above for
    /// the DFlash K=17 verify `down_proj` ([M=17,N=5120,K=16384]). The
    /// single-slice kernel fields only 80 CTAs (N=5120/64) and is
    /// occupancy-starved on the long K=16384 loop (~91 GB/s vs gate/up's
    /// ~163). Split-K multiplies CTAs by k_splits into an FP32 workspace,
    /// then `reduce_splitk_f32_to_bf16` sums + downcasts. Gated by
    /// `ATLAS_FFN_DOWN_SPLITK` (default 0 on `<= 64` SMs, 4 on `65..=128` SMs,
    /// 8 on `> 128` SMs; 0/1 disables). Loaded via
    /// `try_kernel` — handle 0 keeps the single-slice m32_n64 path.
    w4a16_gemm_t_m32_n64_splitk: KernelHandle,
    /// `reduce_splitk_f32_to_bf16` — companion reduce kernel for the
    /// split-K down_proj. Sums the k_splits FP32 partial bands → BF16.
    reduce_splitk_k: KernelHandle,
    /// Lazily-allocated FP32 split-K workspace [k_splits, M, N].
    /// `Mutex` because `forward_kgamma` takes `&self`.
    splitk_workspace: Mutex<Option<DevicePtr>>,
    /// `w4a16_gemm_t_m128_v2` — 8-warp (blockDim 256) shadow of
    /// `w4a16_gemm_t_m128`. Same 2-stage cp.async pipeline + same SMEM
    /// footprint (~29.8KB → 3 CTAs/SM), but parallelizes chunk 0 and
    /// chunk 1 MMA computation across warps {0-3} and {4-7} instead of
    /// serializing both chunks on 4 warps. Yields 2× more warps/SM (768
    /// vs 384) → more MMA pipeline slots in flight. Originally a
    /// MiniMax-only kernel (kernels/gb10/minimax-m2-229b/nvfp4/
    /// w4a16_gemm_v2.cu) — copied verbatim into the qwen3.6-27b target
    /// so the FFN prefill GEMM can route through it. Gated by
    /// `ATLAS_FFN_M128_V2=1`. Loaded via `try_kernel`; handle 0 keeps
    /// the v1 path as silent fallback.
    w4a16_gemm_t_m128_v2: KernelHandle,
    /// `fp8_gemm_t_m128` — large-M FP8×FP8 GEMM kernel for pre-dequanted
    /// FFN weights. Loaded via `try_kernel`. When set together with
    /// `gate_fp8`/`up_fp8`/`down_fp8` (installed by
    /// `predequant_for_prefill`), the FFN prefill bypasses NVFP4 dequant
    /// entirely — saving 1 __syncthreads + the 16-iteration dequant
    /// phase per K-step inside `w4a16_gemm_t_m128`. Mirrors the
    /// attention `predequant_for_prefill` + `fp8_gemm_n128_m128`
    /// pattern (`qwen3_attention/prefill_weights.rs`).
    fp8_gemm_t_m128_k: KernelHandle,
    /// Pre-dequanted FP8 [N, K] gate weight. `Some` only when
    /// `ATLAS_FFN_PREDEQUANT_FP8=1` is set at startup and the
    /// `predequant_nvfp4_to_fp8` + `fp8_gemm_t_m128` kernels are present.
    /// Memory cost: N×K bytes per projection (e.g. 17408×5120 = 89 MB
    /// for gate/up, 5120×17408 = 89 MB for down → ~17 GB per Qwen3.6-27B
    /// run, ~270 MB per layer × 64 layers). Roughly DOUBLES the FFN
    /// weight footprint vs NVFP4-only — gate by intent.
    gate_fp8: Option<DevicePtr>,
    up_fp8: Option<DevicePtr>,
    down_fp8: Option<DevicePtr>,
    /// W4A4 NVFP4×NVFP4 native tensor-core GEMM
    /// (`nvfp4_cutlass::nvfp4_nvfp4_gemm_t_m64`). Loaded via `try_kernel`
    /// — handle 0 causes an eligible explicit `ATLAS_E2M1_GEMM=1` request to
    /// fail closed with `bail!`.
    nvfp4_gemm_k: KernelHandle,
    /// K-major M=128 W4A4 shadow consuming `transpose_for_gemm` buffers.
    /// Handle 0 makes an eligible explicit `ATLAS_E2M1_KMAJOR=1` request
    /// fail closed; the ordinary row-major W4A4 route is otherwise unchanged.
    nvfp4_gemm_kmajor_m128_k: KernelHandle,
    /// Sixteen-warp M=256 K-major long-prefill shadow. An eligible explicit
    /// `ATLAS_E2M1_KMAJOR_M256=1` request fails closed when this is absent;
    /// M128 remains the K-major fallback below the M256 saturation threshold.
    nvfp4_gemm_kmajor_m256_k: KernelHandle,
    /// `quantize_nvfp4::nvfp4_global_absmax` — per-tensor absmax scan
    /// used to derive the activation `scale2` for the W4A4 path.
    nvfp4_absmax_k: KernelHandle,
    /// `quantize_nvfp4::quantize_bf16_to_nvfp4` — per-row E2M1 quantizer
    /// used to convert BF16 activations to the W4A4 GEMM input layout.
    nvfp4_quantize_k: KernelHandle,
    /// Fused SiLU(gate)*up -> NVFP4 down-input quantizer. Handle 0 makes an
    /// eligible explicit `ATLAS_E2M1_SILU_QUANT=1` request fail closed.
    nvfp4_silu_quantize_k: KernelHandle,
    /// `quantize_bf16_to_nvfp4_cutlass::quantize_bf16_to_nvfp4_atlas_128x4` —
    /// BF16 → NVFP4 activation quantizer emitting `128x4`-swizzled `UE4M3`
    /// scales for native cuBLASLt (`VEC16_UE4M3`) GEMMs on SM103/SM120.
    nvfp4_quantize_128x4_k: KernelHandle,
    /// Single-pass vectorized Blackwell PTX sibling (`quantize_bf16_to_nvfp4_atlas_128x4_ptx`)
    /// using `uint4` loads, `__nv_fp8_e4m3`, `cvt.rn.satfinite.e2m1x2.f32`, and 64-bit stores.
    nvfp4_quantize_128x4_ptx_k: KernelHandle,
    /// Fused SwiGLU + vectorized Blackwell PTX NVFP4 quantizer
    /// (`silu_mul_quantize_nvfp4_atlas_128x4`) for cuBLASLt `down_proj` prefill.
    nvfp4_silu_quantize_128x4_k: KernelHandle,
    /// Zero-sync device-dynamic `128x4` activation quantizer
    /// (`quantize_bf16_to_nvfp4_atlas_128x4_from_absmax`) consuming `global_max`.
    nvfp4_quantize_128x4_from_absmax_k: KernelHandle,
    /// Device-resident combined alpha builder (`nvfp4_dynamic_combined_alpha`)
    /// producing `scale2_a * weight_scale_2` on stream without host sync.
    nvfp4_dynamic_combined_alpha_k: KernelHandle,
    /// Fused merged `[gate | up]` SwiGLU + `128x4` NVFP4 quantizer
    /// (`quantize_merged_silu_mul_bf16_to_nvfp4_atlas_128x4`) for merged `[gate; up]` cuBLASLt.
    nvfp4_merged_silu_quantize_128x4_k: KernelHandle,
    /// Vectorized Blackwell PTX merged `[gate | up]` SwiGLU + `128x4` NVFP4 quantizer
    /// (`quantize_merged_silu_mul_bf16_to_nvfp4_atlas_128x4_ptx`), preferred on `sm_count > 64`.
    nvfp4_merged_silu_quantize_128x4_ptx_k: KernelHandle,
    /// Fused `residual_add_rms_norm_quantize_nvfp4_128x4` kernel for static-scale
    /// prefill FFN on `sm_count > 64`.
    nvfp4_residual_add_rms_norm_quant_128x4_k: KernelHandle,
    /// Lazily-swizzled `128x4` `UE4M3` weight block scales for the cuBLASLt
    /// NVFP4 prefill path (`SM103`/`SM120` auto or `ATLAS_CUBLASLT_NVFP4=1`).
    #[cfg(feature = "cuda")]
    cublaslt_nvfp4_scales: Mutex<Option<CublasLtNvfp4Scales>>,
    /// Opportunistically loaded ModelOpt checkpoint `input_scale` scalars used
    /// solely by the cuBLASLt NVFP4 prefill path so SM120/SM103 avoid per-layer
    /// dynamic absmax stream syncs when checkpoint scales are available.
    #[cfg(feature = "cuda")]
    cublaslt_checkpoint_scales: Option<E2m1CheckpointScales>,
    /// Scratch arena for the W4A4 fast path. Holds packed activation nibbles,
    /// per-group FP8 scales, and absmax scratch. FlashInfer preparation
    /// allocates its maximum admitted shape at construction; other routes
    /// allocate lazily and resize if M or K grows. `Mutex` because
    /// `forward_prefill` takes `&self`. `Arc` so the layers of one model can
    /// share a single arena (`ATLAS_MEMLEAN_SHARED_SCRATCH`).
    e2m1_scratch: Arc<Mutex<Option<E2m1Scratch>>>,
    /// Serializes every full launch group that uses `e2m1_scratch` and binds
    /// it to one CUDA stream. The kernels are asynchronous, so allocation-only
    /// locking is insufficient: a later caller on another stream could
    /// overwrite or free operands still in flight. Reuse on the same stream is
    /// safe because CUDA preserves enqueue order. Shared together with
    /// `e2m1_scratch`, so every layer using one arena is bound to one stream.
    e2m1_use_stream: Arc<Mutex<Option<u64>>>,
    /// The arena above is shared with other layers
    /// ([`Self::adopt_shared_e2m1_scratch`]).
    e2m1_scratch_shared: bool,
    /// Optional ModelOpt calibration scales read once during layer
    /// construction. `ATLAS_E2M1_STATIC_SCALE=1` uses these with the existing
    /// on-device quantizer, eliminating runtime absmax scans and host syncs.
    /// This is a different activation-quantization contract from dynamic
    /// absmax and remains default-off pending output-quality qualification.
    e2m1_checkpoint_scales: Option<E2m1CheckpointScales>,
    /// Exact Atlas-quantization / FlashInfer-GEMM prefill route. Constructed
    /// only by the Qwen3.8 loader after every original checkpoint operand,
    /// scalar, physical 128x4 scale copy, native symbol and zero-workspace
    /// tactic has passed admission. Default-unrouted on every other build.
    #[cfg(all(feature = "cuda", target_os = "linux"))]
    flashinfer_ffn_prefill: Option<FlashinferFfnPrefill>,
    /// `ffn_sparsity_measure` — TEAL-style activation-sparsity observer for
    /// the sparsity-drafted self-speculation feasibility gate
    /// (`ATLAS_MEASURE_FFN_SPARSITY=1`). Loaded via `try_kernel`; handle 0
    /// disables the measurement silently (older kernel caches).
    ffn_sparsity_measure_k: KernelHandle,
    /// Lazily-allocated device counter buffers for the sparsity measurement.
    /// `Mutex` because `forward` takes `&self`. Allocated on first measured
    /// `forward` call (gpu.alloc is illegal during graph capture, but the
    /// self-spec draft + measurement run EAGER — see the gate docs). None
    /// until the first measured forward.
    sparsity_meas: Mutex<Option<SparsityMeas>>,
    /// W3 (3-bit weight) FFN projections — mixed-precision byte-reduction
    /// lane. `Some` only when `ATLAS_FFN_W3_LAYERS` names this layer AND
    /// the repacked sidecar (`ATLAS_FFN_W3_SIDECAR`, loaded via
    /// `weight_map::w3_sidecar`) contained its tensors; installed by the
    /// loader via `set_w3_weights`. GEMV layout `[N, 3K/8]` — used by the
    /// single-token `forward` SiLU path (dual gate/up + fused SiLU down).
    /// Cuts packed FFN weight bytes 25% vs NVFP4 on a weight-bandwidth-
    /// bound decode. NOT md5-gated (weights differ from W4 by
    /// construction) — quality is protected by the ABBA eval gate; the
    /// default path (gate off / no sidecar) stays byte-identical.
    w3_weights: Option<DenseFfnWeights>,
    /// Transposed W3 copies (`[3K/8, N_pad64]`) for the K=γ verify GEMM
    /// path (`w3a16_gemm_t_m32_n64`). Built host-side by the sidecar
    /// loader. `forward_kgamma` routes gate/up/down through the W3 GEMM
    /// when set (superseding the W4 m32/fused/split-K variants on this
    /// layer). Other paths (prefill, K=2/3 batched GEMV) intentionally
    /// stay on the retained W4 weights — they are not weight-bandwidth-
    /// bound the same way and keep their higher-precision copies.
    w3_weights_t: Option<DenseFfnWeights>,
    /// `w3a16_gemv_dual` — W3 gate+up dual GEMV. `try_kernel`; handle 0
    /// (older PTX caches) disables the W3 GEMV path silently.
    w3a16_gemv_dual_k: KernelHandle,
    /// `w3a16_gemv_silu_input` — W3 fused SiLU-input down GEMV.
    w3a16_gemv_silu_input_k: KernelHandle,
    /// `w3a16_gemm_t_m32_n64` — W3 clone of the m32_n64 verify GEMM.
    w3a16_gemm_t_m32_n64_k: KernelHandle,
    /// `ffn_build_keep_chunks` — on-device keep-chunk selector for the SPARSE
    /// self-spec DRAFT path (`ATLAS_SELF_SPEC_SPARSE=1`). Handle 0 disables.
    ffn_build_keep_chunks_k: KernelHandle,
    /// `w4a16_gemv_sparse_cols` — column-sparse GEMV for the SPARSE draft
    /// path. Handle 0 disables the sparse draft (falls back to dense GEMV).
    w4a16_gemv_sparse_cols_k: KernelHandle,
    /// Lazily-allocated per-layer keep_idx / keep_len device scratch for the
    /// sparse draft path. `Mutex` because `forward_draft_sparse` takes `&self`.
    sparse_draft_scratch: Mutex<Option<SparseDraftScratch>>,
}

/// Per-layer device counter buffers for the FFN activation-sparsity
/// measurement. Two sites (gate/up input + down input), each with a
/// `NUM_THRESH`-entry u32 histogram (below-threshold counts, atomically
/// accumulated) and a 2-entry u32 `count` ([0]=rows seen, [1]=elements seen).
struct SparsityMeas {
    /// Histogram for site 0 (gate/up input, K=hidden). `NUM_THRESH` u32s.
    hist_gateup: DevicePtr,
    /// [rows, elements] u32 counter for site 0.
    count_gateup: DevicePtr,
    /// Histogram for site 1 (down input, K=intermediate). `NUM_THRESH` u32s.
    hist_down: DevicePtr,
    /// [rows, elements] u32 counter for site 1.
    count_down: DevicePtr,
    /// Dedicated BF16 scratch [1, intermediate] into which the observer
    /// recomputes `silu(gate)*up` for the DOWN-input site. This is a SEPARATE
    /// buffer from the token-stream's `gate_out`/`up_out` — the fused down
    /// GEMV (`w4a16_gemv_silu_input`) applies SiLU internally and never
    /// materialises a standalone silu'd vector, so the observer computes its
    /// own copy here WITHOUT touching the buffers the fused kernel reads.
    /// Keeps the measurement a pure reader (token stream byte-identical).
    meas_silu: DevicePtr,
    /// Number of measured `forward` calls since process start on this layer.
    /// Drives the periodic D2H dump cadence.
    steps: u64,
}

/// Per-layer device scratch for the SPARSE self-spec draft: `keep_idx`
/// (surviving k8-chunk indices, capacity `K_max/8`) + `keep_len` (1 u32).
struct SparseDraftScratch {
    /// Surviving k8-chunk index list, sized for the largest K this layer
    /// might sparsify (down input K=intermediate). `keep_idx.len == K/8`.
    keep_idx: DevicePtr,
    /// Single u32: number of surviving chunks written by
    /// `ffn_build_keep_chunks`.
    keep_len: DevicePtr,
    /// Capacity in k8 chunks (= K_max/8) the `keep_idx` buffer can hold.
    cap_chunks: usize,
}

impl DenseFfnLayer {
    pub fn new(weights: DenseFfnWeights, gpu: &dyn GpuBackend) -> Result<Self> {
        Self::new_with_activation(weights, FfnActivation::SiLU, gpu)
    }

    pub fn new_with_activation(
        weights: DenseFfnWeights,
        activation: FfnActivation,
        gpu: &dyn GpuBackend,
    ) -> Result<Self> {
        let act_mul = match activation {
            FfnActivation::SiLU => gpu.kernel("moe_silu_mul", "moe_silu_mul")?,
            FfnActivation::GeLU => gpu.kernel("gelu", "gelu_mul")?,
        };
        // BF16 path kernels — optional (only loaded if available; gemma4
        // is the only consumer today). `try_kernel` returns
        // `KernelHandle(0)` on miss so we don't break NVFP4-only models
        // that were built without these kernels. Module names per
        // `kernels/gb10/{target}/nvfp4/KERNEL.toml`:
        //   `dense_gemv_bf16 = "gemv"`, `dense_gemm_bf16 = "gemm"`.
        let dense_gemv_bf16_k = super::try_kernel(gpu, "gemv", "dense_gemv_bf16");
        let dense_gemm_bf16_k = super::try_kernel(gpu, "gemm", "dense_gemm_bf16");
        let e2m1_checkpoint_scales = load_e2m1_checkpoint_scales(&weights, gpu)?;
        #[cfg(feature = "cuda")]
        let cublaslt_checkpoint_scales =
            try_load_cublaslt_checkpoint_scales(&weights, gpu, e2m1_checkpoint_scales);
        let warp_nvfp4_mma_supported = supports_warp_nvfp4_mma(gpu.compute_capability());
        if gpu_sm_count(gpu) > PROJ_27B_CUBLASLT_NVFP4_SM_THRESHOLD {
            warm_coalesced_nvfp4_sm120_kernels(gpu);
        }

        Ok(Self {
            weights,
            activation,
            w4a16_gemv: gpu.kernel("w4a16_gemv", "w4a16_gemv")?,
            w4a16_gemv_dual: gpu.kernel("w4a16_gemv_fused", "w4a16_gemv_dual")?,
            w4a16_gemv_silu_input: gpu.kernel("w4a16_gemv_fused", "w4a16_gemv_silu_input")?,
            exact_ffn_kernels: ops::W4a16ExactFfnKernels::new(
                super::try_kernel(gpu, "w4a16_gemv_fused", "w4a16_gemv_dual_exact_m4"),
                super::try_kernel(gpu, "w4a16_gemv_fused", "w4a16_gemv_dual_exact_m8"),
                super::try_kernel(gpu, "w4a16_gemv_fused", "w4a16_gemv_dual_exact_m17"),
                super::try_kernel(gpu, "w4a16_gemv_fused", "w4a16_gemv_dual_exact_m32"),
                super::try_kernel(gpu, "w4a16_gemv_fused", "w4a16_gemv_silu_input_exact_m4"),
                super::try_kernel(gpu, "w4a16_gemv_fused", "w4a16_gemv_silu_input_exact_m8"),
                super::try_kernel(gpu, "w4a16_gemv_fused", "w4a16_gemv_silu_input_exact_m17"),
                super::try_kernel(gpu, "w4a16_gemv_fused", "w4a16_gemv_silu_input_exact_m32"),
            )
            .with_materialized_m8(
                super::try_kernel(gpu, "w4a16_gemv_fused", "w4a16_gemv_dual_silu_f32_exact_m8"),
                super::try_kernel(gpu, "w4a16_gemv_fused", "w4a16_gemv_f32_input_exact_m8"),
            )
            .with_materialized_m17(
                super::try_kernel(
                    gpu,
                    "w4a16_gemv_fused",
                    "w4a16_gemv_dual_silu_f32_exact_m17",
                ),
                super::try_kernel(gpu, "w4a16_gemv_fused", "w4a16_gemv_f32_input_exact_m17"),
            )
            .with_fused_materialized_m17(super::try_kernel(
                gpu,
                "w4a16_gemv_fused",
                "w4a16_gemv_dual_exact_materialize_f32_m17",
            ))
            .with_rt2(
                super::try_kernel(
                    gpu,
                    "w4a16_gemv_fused_rt",
                    "w4a16_gemv_dual_exact_materialize_f32_rt2_m17",
                ),
                super::try_kernel(
                    gpu,
                    "w4a16_gemv_fused_rt",
                    "w4a16_gemv_f32_input_exact_rt2_m8",
                ),
                super::try_kernel(
                    gpu,
                    "w4a16_gemv_fused_rt",
                    "w4a16_gemv_f32_input_exact_rt2_m17",
                ),
            ),
            exact_ffn_lowreg_m16: ops::W4a16ExactFfnLowregM16Kernels::new(
                super::try_kernel(
                    gpu,
                    "w4a16_gemv_exact_ffn_lowreg_m16",
                    "w4a16_gemv_gate_exact_m16_lowreg",
                ),
                super::try_kernel(
                    gpu,
                    "w4a16_gemv_exact_ffn_lowreg_m16",
                    "w4a16_gemv_up_exact_m16_lowreg",
                ),
                super::try_kernel(
                    gpu,
                    "w4a16_gemv_exact_ffn_lowreg_m16",
                    "w4a16_gate_up_materialize_f32_m16",
                ),
            ),
            w4a16_gemv_sw: super::try_kernel(gpu, "w4a16_gemv", "w4a16_gemv_sw"),
            w4a16_gemv_dual_sw: super::try_kernel(gpu, "w4a16_gemv_fused", "w4a16_gemv_dual_sw"),
            w4a16_gemv_silu_input_sw: super::try_kernel(
                gpu,
                "w4a16_gemv_fused",
                "w4a16_gemv_silu_input_sw",
            ),
            gemv_sw: ops::gemv_sw_enabled(),
            w4a16_gemv_dual_batch2: gpu.kernel("w4a16_gemv", "w4a16_gemv_dual_batch2")?,
            w4a16_gemv_dual_batch3: gpu.kernel("w4a16_gemv", "w4a16_gemv_dual_batch3")?,
            w4a16_gemv_dual_batch3_tuned: super::try_kernel(
                gpu,
                "w4a16_gemv",
                "w4a16_gemv_dual_batch3_tuned",
            ),
            w4a16_gemv_batch2: gpu.kernel("w4a16_gemv", "w4a16_gemv_batch2")?,
            w4a16_gemv_batch3: gpu.kernel("w4a16_gemv", "w4a16_gemv_batch3")?,
            w4a16_gemm: gpu.kernel("w4a16", "w4a16_gemm")?,
            w4a16_gemm_pipe: super::try_kernel(gpu, "w4a16", "w4a16_gemm_pipe"),
            w4a16_gemm_pipe_silu_mul: super::try_kernel(gpu, "w4a16", "w4a16_gemm_pipe_silu_mul"),
            w4a16_gemm_pipe_dual: super::try_kernel(gpu, "w4a16", "w4a16_gemm_pipe_dual"),
            act_mul,
            bf16_weights: None,
            dense_gemv_bf16_k,
            dense_gemm_bf16_k,
            gate_proj_t: None,
            up_proj_t: None,
            down_proj_t: None,
            prefill_t_m128_only: false,
            // Optional small-M (M_TILE=16) transposed-weight GEMM.
            // Missing on older kernel caches — handle 0 disables routing
            // through this kernel regardless of the transposed weights.
            w4a16_gemm_t_m16: super::try_kernel(gpu, "w4a16", "w4a16_gemm_t_m16"),
            // Optional small-M (M_TILE=16), small-N (N_TILE=64) variant for
            // K=3 MTP verify on dense Qwen3.6-27B. Missing on non-qwen3.6-27b
            // kernel caches — handle 0 falls back to GEMV silently.
            w4a16_gemm_t_m16_n64: super::try_kernel(gpu, "w4a16", "w4a16_gemm_t_m16_n64"),
            // Optional large-M (M_TILE=128, N_TILE=128) transposed-weight
            // GEMM for prefill. Handle 0 disables the
            // `ATLAS_PREFILL_FFN_FAST` fast path silently.
            w4a16_gemm_t_m128: super::try_kernel(gpu, "w4a16", "w4a16_gemm_t_m128"),
            w4a16_gemm_t_m32_n64: super::try_kernel(gpu, "w4a16", "w4a16_gemm_t_m32_n64"),
            // Optional fused gate+up+silu kernel. Handle 0 keeps the split
            // gate/up path (ATLAS_FFN_FUSED_GATEUP).
            w4a16_gemm_t_m32_n64_gateup_silu: super::try_kernel(
                gpu,
                "w4a16",
                "w4a16_gemm_t_m32_n64_gateup_silu",
            ),
            // Optional dequant-in-registers fork of the fused kernel. Handle
            // 0 keeps the staged fused kernel (ATLAS_DEQUANT_PIPE).
            w4a16_gemm_t_m32_n64_gateup_silu_pipe: super::try_kernel(
                gpu,
                "w4a16",
                "w4a16_gemm_t_m32_n64_gateup_silu_pipe",
            ),
            // Optional K_STEP=64 register-dequant fork (ATLAS_GATEUP_K64).
            // Handle 0 falls back through _pipe → staged fused kernel.
            w4a16_gemm_t_m32_n64_gateup_silu_pipe_k64: super::try_kernel(
                gpu,
                "w4a16",
                "w4a16_gemm_t_m32_n64_gateup_silu_pipe_k64",
            ),
            w4a16_gemm_t_m32_n32_gateup_silu_pipe_k64: super::try_kernel(
                gpu,
                "w4a16",
                "w4a16_gemm_t_m32_n32_gateup_silu_pipe_k64",
            ),
            // Optional split-K down_proj variant + reduce. Handle 0 keeps
            // the single-slice m32_n64 path (ATLAS_FFN_DOWN_SPLITK).
            w4a16_gemm_t_m32_n64_splitk: super::try_kernel(
                gpu,
                "w4a16",
                "w4a16_gemm_t_m32_n64_splitk",
            ),
            reduce_splitk_k: super::try_kernel(gpu, "w4a16", "reduce_splitk_f32_to_bf16"),
            splitk_workspace: Mutex::new(None),
            // Optional 8-warp shadow of the M=128 kernel. Handle 0
            // disables `ATLAS_FFN_M128_V2` silently.
            w4a16_gemm_t_m128_v2: super::try_kernel(gpu, "w4a16_v2", "w4a16_gemm_t_m128_v2"),
            // Optional FP8×FP8 M=128 GEMM for the predequant fast path.
            // Handle 0 disables `ATLAS_FFN_PREDEQUANT_FP8` silently.
            fp8_gemm_t_m128_k: super::try_kernel(gpu, "w4a16", "fp8_gemm_t_m128"),
            gate_fp8: None,
            up_fp8: None,
            down_fp8: None,
            // Optional W4A4 (NVFP4×NVFP4) native tensor-core GEMM. Loaded
            // via `try_kernel` — handle 0 disables the `ATLAS_E2M1_GEMM`
            // fast path silently. On SM103 (`cc == (10, 3)`), warp-level
            // SM120 `kind::mxf4nvf4.block_scale` PTX is a `__trap()` stub
            // (cuBLASLt `VEC16_UE4M3` serves native NVFP4 instead), so these
            // handles stay 0 and can never be dispatched.
            nvfp4_gemm_k: if warp_nvfp4_mma_supported {
                super::try_kernel(gpu, "nvfp4_cutlass", "nvfp4_nvfp4_gemm_t_m64")
            } else {
                KernelHandle(0)
            },
            nvfp4_gemm_kmajor_m128_k: if warp_nvfp4_mma_supported {
                super::try_kernel(gpu, "nvfp4_cutlass", "nvfp4_nvfp4_gemm_kmajor_m128")
            } else {
                KernelHandle(0)
            },
            nvfp4_gemm_kmajor_m256_k: if warp_nvfp4_mma_supported {
                super::try_kernel(gpu, "nvfp4_cutlass", "nvfp4_nvfp4_gemm_kmajor_m256")
            } else {
                KernelHandle(0)
            },
            nvfp4_absmax_k: super::try_kernel(gpu, "quantize_nvfp4", "nvfp4_global_absmax"),
            nvfp4_quantize_k: super::try_kernel(gpu, "quantize_nvfp4", "quantize_bf16_to_nvfp4"),
            nvfp4_silu_quantize_k: super::try_kernel(
                gpu,
                "quantize_nvfp4",
                "quantize_silu_mul_bf16_to_nvfp4",
            ),
            nvfp4_quantize_128x4_k: super::try_kernel(
                gpu,
                "quantize_bf16_to_nvfp4_cutlass",
                "quantize_bf16_to_nvfp4_atlas_128x4",
            ),
            nvfp4_quantize_128x4_ptx_k: super::try_kernel(
                gpu,
                "quantize_bf16_to_nvfp4_cutlass",
                "quantize_bf16_to_nvfp4_atlas_128x4_ptx",
            ),
            nvfp4_silu_quantize_128x4_k: super::try_kernel(
                gpu,
                "quantize_bf16_to_nvfp4_cutlass",
                "silu_mul_quantize_nvfp4_atlas_128x4",
            ),
            nvfp4_quantize_128x4_from_absmax_k: super::try_kernel(
                gpu,
                "quantize_bf16_to_nvfp4_cutlass",
                "quantize_bf16_to_nvfp4_atlas_128x4_from_absmax",
            ),
            nvfp4_dynamic_combined_alpha_k: super::try_kernel(
                gpu,
                "quantize_bf16_to_nvfp4_cutlass",
                "nvfp4_dynamic_combined_alpha",
            ),
            nvfp4_merged_silu_quantize_128x4_k: super::try_kernel(
                gpu,
                "quantize_nvfp4",
                "quantize_merged_silu_mul_bf16_to_nvfp4_atlas_128x4",
            ),
            nvfp4_merged_silu_quantize_128x4_ptx_k: super::try_kernel(
                gpu,
                "quantize_nvfp4",
                "quantize_merged_silu_mul_bf16_to_nvfp4_atlas_128x4_ptx",
            ),
            nvfp4_residual_add_rms_norm_quant_128x4_k: super::try_kernel(
                gpu,
                "quantize_nvfp4",
                "residual_add_rms_norm_quantize_nvfp4_128x4",
            ),
            #[cfg(feature = "cuda")]
            cublaslt_nvfp4_scales: Mutex::new(None),
            #[cfg(feature = "cuda")]
            cublaslt_checkpoint_scales,
            e2m1_scratch: Arc::new(Mutex::new(None)),
            e2m1_use_stream: Arc::new(Mutex::new(None)),
            e2m1_scratch_shared: false,
            e2m1_checkpoint_scales,
            #[cfg(all(feature = "cuda", target_os = "linux"))]
            flashinfer_ffn_prefill: None,
            // Sparsity-drafted self-speculation kernels (default-off features).
            // The .cu files live in kernels/gb10/common/ and register under
            // their file-stem module names. try_kernel → handle 0 disables the
            // feature silently on caches built before these kernels existed.
            // W3 (3-bit) FFN lane — weights installed later by the loader
            // (set_w3_weights) iff ATLAS_FFN_W3_LAYERS + sidecar match.
            // Kernels live in kernels/gb10/common/w3a16_gemv.cu /
            // w3a16_gemm.cu; try_kernel → handle 0 on caches built before
            // they existed (W3 then stays fully disabled).
            w3_weights: None,
            w3_weights_t: None,
            w3a16_gemv_dual_k: super::try_kernel(gpu, "w3a16_gemv", "w3a16_gemv_dual"),
            w3a16_gemv_silu_input_k: super::try_kernel(gpu, "w3a16_gemv", "w3a16_gemv_silu_input"),
            w3a16_gemm_t_m32_n64_k: super::try_kernel(gpu, "w3a16_gemm", "w3a16_gemm_t_m32_n64"),
            ffn_sparsity_measure_k: super::try_kernel(
                gpu,
                "ffn_sparsity_measure",
                "ffn_sparsity_measure",
            ),
            sparsity_meas: Mutex::new(None),
            ffn_build_keep_chunks_k: super::try_kernel(
                gpu,
                "w4a16_gemv_sparse_cols",
                "ffn_build_keep_chunks",
            ),
            w4a16_gemv_sparse_cols_k: super::try_kernel(
                gpu,
                "w4a16_gemv_sparse_cols",
                "w4a16_gemv_sparse_cols",
            ),
            sparse_draft_scratch: Mutex::new(None),
        })
    }

    /// Whether the W4A4 (NVFP4×NVFP4) native tensor-core FFN prefill path
    /// is wired up. Returns true when all three required kernel symbols
    /// are present in the loaded module. Used by `forward_prefill` to
    /// choose between the W4A4 fast path and the existing fp8/v2/m128
    /// fallbacks.
    pub fn has_e2m1_ffn(&self) -> bool {
        self.nvfp4_gemm_k.0 != 0
            && self.nvfp4_quantize_k.0 != 0
            && (self.e2m1_checkpoint_scales.is_some() || self.nvfp4_absmax_k.0 != 0)
    }

    /// Whether the K-major W4A4 implementation has every runtime component
    /// other than the per-layer transformed weights (installed after `new`).
    fn has_e2m1_kmajor_runtime(&self) -> bool {
        self.nvfp4_gemm_kmajor_m128_k.0 != 0
            && self.nvfp4_quantize_k.0 != 0
            && (self.e2m1_checkpoint_scales.is_some() || self.nvfp4_absmax_k.0 != 0)
    }

    fn has_e2m1_kmajor_m256_runtime(&self) -> bool {
        self.nvfp4_gemm_kmajor_m256_k.0 != 0 && self.has_e2m1_kmajor_runtime()
    }

    /// Ensure the W4A4 activation scratch arena has capacity for `m` rows
    /// and `k` columns. Reallocates in-place if either dimension grew.
    ///
    /// Returns `(a_packed, a_scale, a_max)`.
    fn ensure_e2m1_scratch(
        &self,
        gpu: &dyn GpuBackend,
        m: usize,
        k: usize,
    ) -> Result<(DevicePtr, DevicePtr, DevicePtr)> {
        let layout = e2m1_scratch_layout(m, k)?;
        let mut slot = self.e2m1_scratch.lock().unwrap();
        let needs_realloc = match slot.as_ref() {
            Some(s) => s.cap_m < layout.cap_m || s.cap_k < layout.cap_k,
            None => true,
        };
        if needs_realloc {
            // Prior launches (either this layer's or another layer's on a
            // shared arena) may still read the previous scratch buffers on the
            // active stream; drain before freeing.
            if slot.is_some() {
                gpu.synchronize_device()?;
            }
            // Free previous (if any) so we don't leak when M or K grows.
            if let Some(prev) = slot.take() {
                let _ = gpu.free(prev.a_packed);
                let _ = gpu.free(prev.a_scale);
                let _ = gpu.free(prev.a_max);
            }
            let a_packed = gpu.alloc(layout.packed_bytes)?;
            let a_scale = match gpu.alloc(layout.scale_bytes) {
                Ok(ptr) => ptr,
                Err(err) => {
                    let _ = gpu.free(a_packed);
                    return Err(err);
                }
            };
            let a_max = match gpu.alloc(layout.max_bytes) {
                Ok(ptr) => ptr,
                Err(err) => {
                    let _ = gpu.free(a_packed);
                    let _ = gpu.free(a_scale);
                    return Err(err);
                }
            };
            *slot = Some(E2m1Scratch {
                a_packed,
                a_scale,
                a_max,
                cap_m: layout.cap_m,
                cap_k: layout.cap_k,
            });
        }
        let s = slot.as_ref().unwrap();
        Ok((s.a_packed, s.a_scale, s.a_max))
    }

    /// Use `shared`'s activation arena and stream binding instead of this
    /// layer's own (`ATLAS_MEMLEAN_SHARED_SCRATCH`). Must run before the
    /// layer allocates any scratch, i.e. before `prepare_flashinfer_prefill`.
    pub fn adopt_shared_e2m1_scratch(&mut self, shared: &SharedE2m1Scratch) -> Result<()> {
        ensure!(
            self.e2m1_scratch
                .lock()
                .map_err(|_| anyhow::anyhow!("W4A4 activation scratch lock poisoned"))?
                .is_none(),
            "a shared W4A4 activation scratch must be adopted before the layer allocates its own"
        );
        self.e2m1_scratch = Arc::clone(&shared.scratch);
        self.e2m1_use_stream = Arc::clone(&shared.stream);
        self.e2m1_scratch_shared = true;
        Ok(())
    }

    fn lock_e2m1_use(&self, stream: u64) -> Result<MutexGuard<'_, Option<u64>>> {
        let mut admitted_stream = self
            .e2m1_use_stream
            .lock()
            .map_err(|_| anyhow::anyhow!("W4A4 activation scratch stream-binding lock poisoned"))?;
        match *admitted_stream {
            Some(existing) => ensure!(
                existing == stream,
                "W4A4 activation scratch is bound to CUDA stream {existing:#x}; rejected concurrent/reordered use on {stream:#x}"
            ),
            None => *admitted_stream = Some(stream),
        }
        Ok(admitted_stream)
    }

    /// Quantize one BF16 activation matrix for the W4A4 path.
    ///
    /// The returned packed values and scales remain owned by this layer's
    /// scratch arena. Callers may issue multiple same-stream GEMMs from them
    /// before preparing a different input. Gate and up share the same BF16
    /// input, so preparing once avoids a duplicate absmax scan, host readback,
    /// stream synchronization, and quantization launch without changing any
    /// quantized byte or GEMM arithmetic.
    fn prepare_e2m1_input(
        &self,
        ctx: &ForwardContext,
        input: DevicePtr,
        m: u32,
        k: u32,
        checkpoint_scale2: Option<f32>,
        stream: u64,
    ) -> Result<(DevicePtr, DevicePtr, f32)> {
        let (a_packed, a_scale, a_max) =
            self.ensure_e2m1_scratch(ctx.gpu, m as usize, k as usize)?;

        let a_scale2 = if let Some(scale2) = checkpoint_scale2 {
            // ModelOpt calibrated this scalar offline and stored it as
            // `.input_scale`. It was read once during layer construction, so
            // the unchanged on-device quantizer can launch immediately.
            scale2
        } else {
            // Dynamic fallback: scan BF16 [M,K], drain the stream, and derive
            // a per-request scale. This remains the default W4A4 contract.
            ctx.gpu.memset_async(a_max, 0, 4, stream)?;
            ops::nvfp4_global_absmax(ctx.gpu, self.nvfp4_absmax_k, input, a_max, m * k, stream)?;
            ctx.gpu.synchronize(stream)?;
            let mut bytes = [0u8; 4];
            ctx.gpu.copy_d2h(a_max, &mut bytes)?;
            let global_max = f32::from_le_bytes(bytes);
            if global_max > 0.0 {
                global_max / (6.0 * 448.0)
            } else {
                1.0
            }
        };

        // Phase 3: per-row E2M1 quantization of the activation. Writes
        // `a_packed` [M, K/2] + `a_scale` [M, K/16] in place.
        ops::quantize_bf16_to_nvfp4(
            ctx.gpu,
            self.nvfp4_quantize_k,
            input,
            a_packed,
            a_scale,
            a_scale2,
            m,
            k,
            stream,
        )?;

        Ok((a_packed, a_scale, a_scale2))
    }

    /// Materialize the SwiGLU down input directly in NVFP4 scratch.
    ///
    /// The CUDA kernel rounds each SiLU(gate)*up value through BF16 before
    /// deriving group scales and nibbles, matching the standalone activation
    /// kernel's numerical boundary without writing the temporary BF16 matrix.
    fn prepare_e2m1_silu_input(
        &self,
        ctx: &ForwardContext,
        gate: DevicePtr,
        up: DevicePtr,
        m: u32,
        k: u32,
        checkpoint_scale2: f32,
        stream: u64,
    ) -> Result<(DevicePtr, DevicePtr, f32)> {
        let (a_packed, a_scale, _) = self.ensure_e2m1_scratch(ctx.gpu, m as usize, k as usize)?;
        ops::quantize_silu_mul_bf16_to_nvfp4(
            ctx.gpu,
            self.nvfp4_silu_quantize_k,
            gate,
            up,
            a_packed,
            a_scale,
            checkpoint_scale2,
            m,
            k,
            stream,
        )?;
        Ok((a_packed, a_scale, checkpoint_scale2))
    }

    /// Dispatch one native W4A4 GEMM from an already prepared activation.
    ///
    /// `weight` is either the standard `[N, K/2]` row-major NVFP4 weight or,
    /// when `kmajor_m128` is true, the existing `[K/2,N]` / `[K/16,N]`
    /// transformed pair. Both kernels retain the same native block-scaled
    /// OMMA and BF16 epilogue order.
    #[allow(clippy::too_many_arguments)]
    fn forward_e2m1_prepared(
        &self,
        ctx: &ForwardContext,
        a_packed: DevicePtr,
        a_scale: DevicePtr,
        a_scale2: f32,
        weight: &QuantizedWeight,
        output: DevicePtr,
        m: u32,
        n: u32,
        k: u32,
        kmajor_m128: bool,
        stream: u64,
    ) -> Result<()> {
        ensure!(
            supports_warp_nvfp4_mma(ctx.gpu.compute_capability()),
            "warp-level SM120 NVFP4 MMA (cutlass_nvfp4_gemm.cu) is unsupported on {:?}; use cuBLASLt NVFP4 or w4a16",
            ctx.gpu.compute_capability()
        );
        let scale2_ab = a_scale2 * weight.weight_scale_2;
        if kmajor_m128 {
            if e2m1_kmajor_kernel(m, crate::layers::prefill_ffn_e2m1_kmajor_m256_enabled())
                == E2m1KmajorKernel::M256
            {
                ops::nvfp4_nvfp4_gemm_kmajor_m256(
                    ctx.gpu,
                    self.nvfp4_gemm_kmajor_m256_k,
                    a_packed,
                    a_scale,
                    weight.weight,
                    weight.weight_scale,
                    scale2_ab,
                    output,
                    m,
                    n,
                    k,
                    stream,
                )?;
            } else {
                ops::nvfp4_nvfp4_gemm_kmajor_m128(
                    ctx.gpu,
                    self.nvfp4_gemm_kmajor_m128_k,
                    a_packed,
                    a_scale,
                    weight.weight,
                    weight.weight_scale,
                    scale2_ab,
                    output,
                    m,
                    n,
                    k,
                    stream,
                )?;
            }
        } else {
            ops::nvfp4_nvfp4_gemm(
                ctx.gpu,
                self.nvfp4_gemm_k,
                a_packed,
                a_scale,
                weight.weight,
                weight.weight_scale,
                scale2_ab,
                output,
                m,
                n,
                k,
                stream,
            )?;
        }
        Ok(())
    }

    /// Run the W4A4 fast path for one FFN projection: prequant BF16 input
    /// to NVFP4, then dispatch `nvfp4_nvfp4_gemm`.
    #[allow(clippy::too_many_arguments)]
    fn forward_e2m1_proj(
        &self,
        ctx: &ForwardContext,
        input: DevicePtr,
        weight: &QuantizedWeight,
        output: DevicePtr,
        m: u32,
        n: u32,
        k: u32,
        checkpoint_scale2: Option<f32>,
        kmajor_m128: bool,
        stream: u64,
    ) -> Result<()> {
        let (a_packed, a_scale, a_scale2) =
            self.prepare_e2m1_input(ctx, input, m, k, checkpoint_scale2, stream)?;
        self.forward_e2m1_prepared(
            ctx,
            a_packed,
            a_scale,
            a_scale2,
            weight,
            output,
            m,
            n,
            k,
            kmajor_m128,
            stream,
        )
    }

    /// Install transposed (`nvfp4_t` layout) FFN projection weights for
    /// the `forward_kgamma` M_TILE=16 fast path. Called by the loader
    /// after `DenseFfnLayer::new` when `ATLAS_FFN_M16_TRANSPOSED=1`.
    /// Takes ownership of `gate_proj_t` / `up_proj_t` / `down_proj_t`
    /// (additional allocations alongside the standard `weights.*_proj`;
    /// the originals are kept for decode-side GEMV paths that target the
    /// HuggingFace `[N, K/2]` layout).
    pub fn set_transposed_weights(
        &mut self,
        gate_proj_t: QuantizedWeight,
        up_proj_t: QuantizedWeight,
        down_proj_t: QuantizedWeight,
    ) {
        self.gate_proj_t = Some(gate_proj_t);
        self.up_proj_t = Some(up_proj_t);
        self.down_proj_t = Some(down_proj_t);
    }

    /// Install transposed FFN weights for the M>=128 `w4a16_gemm_t_m128`
    /// prefill route alone (Gemma-4 dense, whose target ships no
    /// `w4a16_gemm_t_m16`). Decode and verify routing is unchanged: they gate
    /// on `has_transposed_ffn()`, which still requires the m16 kernel.
    pub fn set_prefill_transposed_weights(
        &mut self,
        gate_proj_t: QuantizedWeight,
        up_proj_t: QuantizedWeight,
        down_proj_t: QuantizedWeight,
    ) {
        self.set_transposed_weights(gate_proj_t, up_proj_t, down_proj_t);
        self.prefill_t_m128_only = true;
    }

    /// Build the exact, default-off FlashInfer FFN prefill operands for one
    /// Qwen3.8 layer. Gate and up are copied once into an immutable merged
    /// gate-then-up operand; down retains the original checkpoint allocation.
    #[cfg(all(feature = "cuda", target_os = "linux"))]
    pub fn prepare_flashinfer_prefill(
        &mut self,
        gpu: &dyn GpuBackend,
        layer: usize,
        hidden: usize,
        intermediate: usize,
    ) -> Result<()> {
        use crate::weight_map::flashinfer_ffn_admission::{
            QWEN38_HIDDEN, QWEN38_INTERMEDIATE, Qwen38FfnOperation, Qwen38FfnProjection,
            qwen38_ffn_launch_candidate, qwen38_merged_gate_up_retained_bytes,
        };
        use ops::flashinfer_sm121::{FlashInferSm121, FlashInferSm121Shape};

        ensure!(
            crate::layers::prefill_ffn_flashinfer_enabled()?,
            "prepare_flashinfer_prefill called while its route is disabled"
        );
        ensure!(
            self.flashinfer_ffn_prefill.is_none(),
            "FlashInfer FFN operands are already prepared"
        );
        ensure!(
            hidden == QWEN38_HIDDEN && intermediate == QWEN38_INTERMEDIATE,
            "FlashInfer FFN route requires Qwen3.8 H={QWEN38_HIDDEN} I={QWEN38_INTERMEDIATE}; got H={hidden} I={intermediate}"
        );
        ensure!(
            self.activation == FfnActivation::SiLU,
            "FlashInfer FFN route requires a SiLU-gated dense FFN"
        );
        ensure!(
            self.e2m1_checkpoint_scales.is_some(),
            "FlashInfer FFN route requires admitted checkpoint-static activation scales"
        );

        let library_path = std::env::var_os("ATLAS_FLASHINFER_SM121_LIB").ok_or_else(|| {
            anyhow::anyhow!("ATLAS_PREFILL_FFN_FLASHINFER=1 requires ATLAS_FLASHINFER_SM121_LIB")
        })?;
        let library = FlashInferSm121::open_qualified(std::path::Path::new(&library_path))?;
        let quantize_atlas_128x4_k = gpu.kernel(
            "quantize_bf16_to_nvfp4_cutlass",
            "quantize_bf16_to_nvfp4_atlas_128x4",
        )?;
        let quantize_merged_silu_atlas_128x4_k = gpu.kernel(
            "quantize_nvfp4",
            "quantize_merged_silu_mul_bf16_to_nvfp4_atlas_128x4",
        )?;
        ensure!(
            quantize_atlas_128x4_k.0 != 0 && quantize_merged_silu_atlas_128x4_k.0 != 0,
            "FlashInfer FFN preparation resolved a NULL required kernel"
        );

        // Freeze the exact zero-workspace tactics before retaining any route.
        // Querying at construction catches a mismatched native library rather
        // than discovering it after some layer projections have run.
        let stream = gpu.default_stream();
        for (operation, m) in [
            (Qwen38FfnOperation::MergedGateUp, 2_079),
            (Qwen38FfnOperation::Down, 2_079),
            (Qwen38FfnOperation::MergedGateUp, 8_192),
            (Qwen38FfnOperation::Down, 8_192),
        ] {
            let plan = qwen38_ffn_launch_candidate(operation, m)?;
            ensure!(
                plan.performance_qualified && plan.workspace_bytes == 0,
                "FlashInfer FFN launch is not fully qualified: {operation:?} M={m}"
            );
            let shape = FlashInferSm121Shape::new(
                u8::try_from(plan.tactic).context("FlashInfer tactic exceeds u8")?,
                plan.m,
                plan.n,
                plan.k,
                1,
            )?;
            let prepared = library.prepare_borrowed_zero_workspace(gpu, shape, stream)?;
            ensure!(
                prepared.shape() == shape,
                "FlashInfer FFN preparation changed its frozen shape"
            );
        }

        // This exact allocation is already charged by the server's model-load
        // preflight. Materialize it now so the first measured M=2079 request
        // cannot pay 64 lazy allocations (or discover an OOM after serving has
        // started). The runtime route reuses this fixed-address maximum shape.
        let scratch_layout = e2m1_scratch_layout(8_192, intermediate)?;
        let (scratch_packed, scratch_scales, scratch_max) =
            self.ensure_e2m1_scratch(gpu, scratch_layout.cap_m, scratch_layout.cap_k)?;
        for pointer in [scratch_packed, scratch_scales, scratch_max] {
            ensure!(
                !pointer.is_null() && pointer.0.is_multiple_of(16),
                "FlashInfer FFN construction scratch is NULL or misaligned"
            );
        }
        {
            let retained = self.e2m1_scratch.lock().unwrap();
            let retained = retained
                .as_ref()
                .context("FlashInfer FFN construction scratch was not retained")?;
            ensure!(
                retained.cap_m == scratch_layout.cap_m && retained.cap_k == scratch_layout.cap_k,
                "FlashInfer FFN construction scratch retained the wrong capacity"
            );
        }

        let merged_gate_up = build_flashinfer_merged_gate_up(
            gpu,
            layer,
            &self.weights.gate_proj,
            &self.weights.up_proj,
        )?;
        let down = build_flashinfer_ffn_projection(
            gpu,
            layer,
            Qwen38FfnProjection::Down,
            &self.weights.down_proj,
            hidden,
            intermediate,
        )?;

        let (library_device, library_inode) = library.file_identity();
        tracing::info!(
            layer,
            library = %library.path().display(),
            library_sha256 = %library.sha256_hex(),
            library_device,
            library_inode,
            merged_gate_up_weight = format_args!("{:#x}", merged_gate_up.weight.0),
            down_weight = format_args!("{:#x}", down.weight.0),
            merged_gate_up_weight_hash = format_args!("{:016x}", merged_gate_up.weight_hash),
            merged_gate_up_scale_hash = format_args!("{:016x}", merged_gate_up.weight_scales_hash),
            down_scale_hash = format_args!("{:016x}", down.weight_scales_hash),
            merged_gate_up_alpha_bits = format_args!("{:08x}", merged_gate_up.alpha_bits),
            down_alpha_bits = format_args!("{:08x}", down.alpha_bits),
            merged_gate_up_retained_bytes = qwen38_merged_gate_up_retained_bytes(),
            scratch_cap_m = scratch_layout.cap_m,
            scratch_cap_k = scratch_layout.cap_k,
            scratch_packed_bytes = scratch_layout.packed_bytes,
            scratch_scale_bytes = scratch_layout.scale_bytes,
            "admitted FlashInfer SM121 FFN prefill operands"
        );
        self.flashinfer_ffn_prefill = Some(FlashinferFfnPrefill {
            layer,
            library,
            merged_gate_up,
            down,
            quantize_atlas_128x4_k,
            quantize_merged_silu_atlas_128x4_k,
        });
        Ok(())
    }

    /// `ATLAS_MEMLEAN_FFN_GATE_UP_ALIAS=1`: point `gate_proj` / `up_proj` at
    /// their bytes inside the prepared merged FlashInfer operand, so the
    /// separate checkpoint allocations can be released. Every route that
    /// reads the original-layout packed weights (the pipe and baseline
    /// prefill GEMMs, the W4A4 routes, and the transposes built after this)
    /// then reads identical bytes from the merged buffer; block scales,
    /// scalars and down are untouched. Returns the replaced gate and up
    /// packed-weight pointers for the caller to release. The caller must have
    /// synchronized the construction stream.
    #[cfg(all(feature = "cuda", target_os = "linux"))]
    pub fn alias_gate_up_to_flashinfer_merged(&mut self) -> Result<(DevicePtr, DevicePtr)> {
        use crate::weight_map::flashinfer_ffn_admission::{QWEN38_HIDDEN, QWEN38_INTERMEDIATE};

        let route = self
            .flashinfer_ffn_prefill
            .as_mut()
            .context("gate/up alias requires prepared FlashInfer FFN operands")?;
        let replaced = (self.weights.gate_proj.weight, self.weights.up_proj.weight);
        ensure!(
            route.merged_gate_up.source_gate_weight == replaced.0
                && route.merged_gate_up.source_up_weight == replaced.1,
            "merged gate/up operand was not built from this layer's gate/up weights"
        );
        ensure!(
            route.merged_gate_up.weight != replaced.0 && route.merged_gate_up.weight != replaced.1,
            "gate/up are already aliased to the merged operand"
        );
        let (gate, up) = merged_gate_up_views(
            route.merged_gate_up.weight,
            QWEN38_INTERMEDIATE * QWEN38_HIDDEN / 2,
        )?;
        self.weights.gate_proj.weight = gate;
        self.weights.up_proj.weight = up;
        route.merged_gate_up.source_gate_weight = gate;
        route.merged_gate_up.source_up_weight = up;
        Ok(replaced)
    }

    /// Try the fully-admitted FlashInfer FFN route for an exact qualified
    /// chunk size. Other rows (notably a final short prefill tail) retain the
    /// existing path unchanged.
    #[cfg(all(feature = "cuda", target_os = "linux"))]
    fn try_forward_flashinfer_prefill(
        &self,
        input: DevicePtr,
        m: u32,
        h: u32,
        inter: u32,
        gate_out: DevicePtr,
        up_out: DevicePtr,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<bool> {
        use crate::weight_map::flashinfer_ffn_admission::{
            QWEN38_HIDDEN, QWEN38_INTERMEDIATE, Qwen38FfnOperation, select_qwen38_ffn_launch,
        };
        use ops::flashinfer_sm121::{FlashInferSm121Buffers, FlashInferSm121Shape};

        let requested = crate::layers::prefill_ffn_flashinfer_enabled()?;
        let exact_qwen38_dense = ctx.config.model_type == "qwen3_5"
            && ctx.config.num_experts == 0
            && h as usize == QWEN38_HIDDEN
            && inter as usize == QWEN38_INTERMEDIATE
            && ctx.config.tp_world_size.max(1) == 1;
        let m_qualified = matches!(m, 2_079 | 8_192)
            || crate::weight_map::flashinfer_ffn_admission::qwen38_ffn_launch_candidate(
                Qwen38FfnOperation::MergedGateUp,
                m as usize,
            )
            .is_ok();
        match flashinfer_ffn_prefill_route(
            requested,
            exact_qwen38_dense,
            m_qualified,
            self.flashinfer_ffn_prefill.is_some(),
        ) {
            FlashinferFfnPrefillRoute::Disabled | FlashinferFfnPrefillRoute::Ineligible => {
                return Ok(false);
            }
            FlashinferFfnPrefillRoute::Missing => anyhow::bail!(
                "ATLAS_PREFILL_FFN_FLASHINFER=1 requires prepared merged FFN operands for exact M={m}"
            ),
            FlashinferFfnPrefillRoute::Complete => {}
        }
        let route = self.flashinfer_ffn_prefill.as_ref().unwrap();
        ensure!(
            route.merged_gate_up.layer == route.layer
                && route.merged_gate_up.source_gate_weight == self.weights.gate_proj.weight
                && route.merged_gate_up.source_up_weight == self.weights.up_proj.weight
                && route.down.weight == self.weights.down_proj.weight,
            "FlashInfer FFN checkpoint operand identity changed after preparation"
        );

        let mut merged_plan =
            select_qwen38_ffn_launch(Qwen38FfnOperation::MergedGateUp, m as usize)?;
        let mut down_plan = select_qwen38_ffn_launch(Qwen38FfnOperation::Down, m as usize)?;
        if let Some((gu, dn)) = crate::layers::flashinfer_ffn_tactic_override() {
            static SEEN: std::sync::Once = std::sync::Once::new();
            SEEN.call_once(|| tracing::warn!("ATLAS_FLASHINFER_FFN_TACTIC override: merged_gate_up={gu} down={dn} (diagnostic, not exact-qualified)"));
            merged_plan.tactic = gu;
            down_plan.tactic = dn;
        }
        ensure!(
            merged_plan.workspace_bytes == 0 && down_plan.workspace_bytes == 0,
            "FlashInfer FFN runtime selected a nonzero-workspace plan"
        );

        let rows = m as usize;
        let input_bytes = flashinfer_ffn_extent(rows, QWEN38_HIDDEN, 2)?;
        let projection_bytes = flashinfer_ffn_extent(rows, QWEN38_INTERMEDIATE, 2)?;
        let merged_bytes = flashinfer_ffn_extent(rows, 2 * QWEN38_INTERMEDIATE, 2)?;
        let output_bytes = flashinfer_ffn_extent(rows, QWEN38_HIDDEN, 2)?;
        let activation_fp4_bytes = flashinfer_ffn_extent(rows, QWEN38_INTERMEDIATE, 1)? / 2;
        let activation_scale_bytes =
            flashinfer_ffn_activation_scale_extent(rows, QWEN38_INTERMEDIATE)?;
        ensure!(
            ctx.buffers.sizes().ffn_gate_up_bf16 >= merged_bytes,
            "merged FlashInfer FFN output arena is undersized"
        );
        ensure!(
            ctx.buffers.sizes().expert_gate_out >= projection_bytes
                && ctx.buffers.sizes().expert_up_out >= projection_bytes
                && ctx.buffers.sizes().moe_output >= output_bytes,
            "FlashInfer FFN destination arena is undersized"
        );

        let merged_out = ctx.buffers.ffn_gate_up_bf16();
        let output = ctx.buffers.moe_output();
        for pointer in [input, merged_out, gate_out, up_out, output] {
            ensure!(
                !pointer.is_null() && pointer.0.is_multiple_of(16),
                "FlashInfer FFN input/output pointer is NULL or misaligned"
            );
        }
        ensure_flashinfer_ffn_disjoint(input, input_bytes, merged_out, merged_bytes)?;
        ensure_flashinfer_ffn_disjoint(input, input_bytes, gate_out, projection_bytes)?;
        ensure_flashinfer_ffn_disjoint(input, input_bytes, up_out, projection_bytes)?;
        ensure_flashinfer_ffn_disjoint(input, input_bytes, output, output_bytes)?;
        ensure_flashinfer_ffn_disjoint(merged_out, merged_bytes, gate_out, projection_bytes)?;
        ensure_flashinfer_ffn_disjoint(merged_out, merged_bytes, up_out, projection_bytes)?;
        ensure_flashinfer_ffn_disjoint(merged_out, merged_bytes, output, output_bytes)?;
        ensure_flashinfer_ffn_disjoint(gate_out, projection_bytes, up_out, projection_bytes)?;
        ensure_flashinfer_ffn_disjoint(gate_out, projection_bytes, output, output_bytes)?;
        ensure_flashinfer_ffn_disjoint(up_out, projection_bytes, output, output_bytes)?;

        let merged_shape = FlashInferSm121Shape::new(
            u8::try_from(merged_plan.tactic).context("FlashInfer tactic exceeds u8")?,
            merged_plan.m,
            merged_plan.n,
            merged_plan.k,
            1,
        )?;
        let down_shape = FlashInferSm121Shape::new(
            u8::try_from(down_plan.tactic).context("FlashInfer tactic exceeds u8")?,
            down_plan.m,
            down_plan.n,
            down_plan.k,
            1,
        )?;
        let mut merged_launch =
            route
                .library
                .prepare_borrowed_zero_workspace(ctx.gpu, merged_shape, stream)?;
        let mut down_launch = route
            .library
            .prepare_borrowed_zero_workspace(ctx.gpu, down_shape, stream)?;

        // Allocate once at the larger down-projection K so no asynchronous
        // activation operand can be freed while it is still in flight. All
        // route shapes, symbols, destinations and native plans are already
        // preflighted before the first quantization/device output effect.
        let _scratch_use = self.lock_e2m1_use(stream)?;
        let (activation_fp4, activation_scales, _) =
            self.ensure_e2m1_scratch(ctx.gpu, m as usize, inter as usize)?;
        for pointer in [activation_fp4, activation_scales] {
            ensure!(
                !pointer.is_null() && pointer.0.is_multiple_of(16),
                "FlashInfer FFN activation scratch is NULL or misaligned"
            );
        }
        ensure_flashinfer_ffn_disjoint(
            activation_fp4,
            activation_fp4_bytes,
            activation_scales,
            activation_scale_bytes,
        )?;
        for (pointer, bytes) in [
            (input, input_bytes),
            (merged_out, merged_bytes),
            (gate_out, projection_bytes),
            (up_out, projection_bytes),
            (output, output_bytes),
        ] {
            ensure_flashinfer_ffn_disjoint(activation_fp4, activation_fp4_bytes, pointer, bytes)?;
            ensure_flashinfer_ffn_disjoint(
                activation_scales,
                activation_scale_bytes,
                pointer,
                bytes,
            )?;
        }

        ops::quantize_bf16_to_nvfp4_atlas_128x4(
            ctx.gpu,
            route.quantize_atlas_128x4_k,
            input,
            activation_fp4,
            activation_scales,
            f32::from_bits(route.merged_gate_up.input_scale_bits),
            m,
            h,
            stream,
        )?;
        merged_launch.launch_eager(
            FlashInferSm121Buffers {
                output_bf16: merged_out,
                activation_fp4,
                weight_fp4: route.merged_gate_up.weight,
                activation_scales,
                weight_scales: route.merged_gate_up.weight_scales_128x4,
                global_scale_f32: route.merged_gate_up.alpha_f32,
            },
            stream,
        )?;
        ops::quantize_merged_silu_mul_bf16_to_nvfp4_atlas_128x4(
            ctx.gpu,
            route.quantize_merged_silu_atlas_128x4_k,
            merged_out,
            activation_fp4,
            activation_scales,
            f32::from_bits(route.down.input_scale_bits),
            m,
            inter,
            stream,
        )?;

        down_launch.launch_eager(
            FlashInferSm121Buffers {
                output_bf16: output,
                activation_fp4,
                weight_fp4: route.down.weight,
                activation_scales,
                weight_scales: route.down.weight_scales_128x4,
                global_scale_f32: route.down.alpha_f32,
            },
            stream,
        )?;

        log_flashinfer_ffn_success(
            route.layer,
            Qwen38FfnOperation::MergedGateUp,
            merged_plan.m,
            merged_plan.tactic,
        );
        log_flashinfer_ffn_success(
            route.layer,
            Qwen38FfnOperation::Down,
            down_plan.m,
            down_plan.tactic,
        );
        Ok(true)
    }

    /// Install W3 (3-bit) FFN weights for this layer — GEMV-layout copies
    /// (used by the single-token `forward` SiLU path) and transposed
    /// GEMM-layout copies (used by `forward_kgamma`). Called by the loader
    /// when `ATLAS_FFN_W3_LAYERS` names this layer and the sidecar tensors
    /// loaded cleanly (see `weight_map::w3_sidecar`). The original W4
    /// weights are RETAINED for the paths W3 does not cover (prefill,
    /// K=2/3 batched GEMV, GELU models, sparse draft).
    pub fn set_w3_weights(&mut self, gemv: DenseFfnWeights, gemm_t: DenseFfnWeights) {
        self.w3_weights = Some(gemv);
        self.w3_weights_t = Some(gemm_t);
    }

    /// Whether the W3 single-token GEMV path is fully wired: weights
    /// installed + both kernel symbols present + SiLU activation (the W3
    /// GEMV set has no GELU-fused down kernel; GELU models stay on W4).
    fn has_w3_gemv(&self) -> bool {
        self.w3_weights.is_some()
            && self.w3a16_gemv_dual_k.0 != 0
            && self.w3a16_gemv_silu_input_k.0 != 0
            && self.activation == FfnActivation::SiLU
    }

    /// Whether the W3 K=γ verify GEMM path is fully wired.
    fn has_w3_gemm(&self) -> bool {
        self.w3_weights_t.is_some()
            && self.w3a16_gemm_t_m32_n64_k.0 != 0
            && self.activation == FfnActivation::SiLU
    }

    /// Whether ANY W3 routing is active on this layer (loader log helper).
    pub fn has_w3(&self) -> bool {
        self.has_w3_gemv() || self.has_w3_gemm()
    }

    /// Quantize a BF16 activation matrix `[m, k]` into row-major packed NVFP4
    /// (`[m, k/2]`) plus `128x4`-swizzled `UE4M3` block scales for cuBLASLt.
    ///
    /// Uses static checkpoint `input_scale` when present; otherwise runs the
    /// zero-sync device dynamic-absmax chain (`nvfp4_global_absmax` +
    /// `quantize_bf16_to_nvfp4_atlas_128x4_from_absmax`) paired with
    /// `BlockScaleScalars::device_alpha`, eliminating all host syncs even
    /// without static scales.
    #[cfg(feature = "cuda")]
    #[expect(
        clippy::too_many_arguments,
        reason = "activation quantizer forwarding scratch, dynamic scalars, shape, scale, and stream"
    )]
    fn prepare_e2m1_input_128x4(
        &self,
        ctx: &ForwardContext,
        input: DevicePtr,
        dynamic_scalars: DevicePtr,
        m: u32,
        k: u32,
        checkpoint_scale2: Option<f32>,
        stream: u64,
    ) -> Result<(DevicePtr, DevicePtr, CublasLtInputScale)> {
        let (a_packed, a_scale, a_max) =
            self.ensure_e2m1_scratch(ctx.gpu, m as usize, k as usize)?;
        let quant_k = select_cublaslt_quantize_128x4_kernel(
            self.nvfp4_quantize_128x4_ptx_k,
            self.nvfp4_quantize_128x4_k,
        );
        let input_scale = launch_quantize_bf16_to_nvfp4_128x4_zero_sync(
            ctx.gpu,
            quant_k,
            self.nvfp4_absmax_k,
            self.nvfp4_quantize_128x4_from_absmax_k,
            self.nvfp4_dynamic_combined_alpha_k,
            input,
            a_packed,
            a_scale,
            dynamic_scalars,
            a_max,
            m,
            k,
            checkpoint_scale2,
            DynamicAbsmaxState::NeedsScan,
            stream,
        )?;
        Ok((a_packed, a_scale, input_scale))
    }

    /// Lazily swizzle and cache `128x4` `UE4M3` block scales for `gate_proj`,
    /// `up_proj`, and `down_proj`, plus a 32-byte zero-sync dynamic scalar
    /// buffer and (when permitted without unsafe VRAM duplication) a merged
    /// `[gate; up]` operand.
    #[cfg(feature = "cuda")]
    fn ensure_cublaslt_nvfp4_scales(
        &self,
        gpu: &dyn GpuBackend,
        h: u32,
        inter: u32,
        want_merged_gate_up: bool,
        stream: u64,
    ) -> Result<(
        DevicePtr,
        DevicePtr,
        DevicePtr,
        DevicePtr,
        Option<(DevicePtr, DevicePtr)>,
    )> {
        let mut guard = self
            .cublaslt_nvfp4_scales
            .lock()
            .expect("cublaslt_nvfp4_scales lock poisoned");
        let gate_src = self.weights.gate_proj.weight_scale;
        let up_src = self.weights.up_proj.weight_scale;
        let down_src = self.weights.down_proj.weight_scale;
        if let Some(ref cached) = *guard
            && cached.gate_src == gate_src
            && cached.up_src == up_src
            && cached.down_src == down_src
        {
            let merged = cached.merged_weight.zip(cached.merged_scales);
            return Ok((
                cached.gate,
                cached.up,
                cached.down,
                cached.dynamic_scalars,
                merged,
            ));
        }

        let gate_packed_bytes = (inter as usize)
            .checked_mul((h as usize) / 2)
            .context("cuBLASLt NVFP4 gate packed byte length overflow")?;
        let merged_packed_bytes = gate_packed_bytes
            .checked_mul(2)
            .context("cuBLASLt NVFP4 merged gate/up byte length overflow")?;
        let policy = cublaslt_merged_gate_up_policy();
        let can_try_merged = want_merged_gate_up && policy != CublasLtMergedGateUpPolicy::Disabled;

        // 1. Resolve zero-copy merged [gate; up] weight first (FlashInfer merged
        // operand or already-contiguous gate+up allocations: 0 extra VRAM).
        let mut merged_weight: Option<DevicePtr> = None;
        let mut owns_merged_weight = false;
        let mut borrowed_merged_scales: Option<DevicePtr> = None;
        if can_try_merged {
            #[cfg(target_os = "linux")]
            if let Some(ref fi) = self.flashinfer_ffn_prefill
                && fi.merged_gate_up.source_gate_weight == self.weights.gate_proj.weight
                && fi.merged_gate_up.source_up_weight == self.weights.up_proj.weight
            {
                merged_weight = Some(fi.merged_gate_up.weight);
                borrowed_merged_scales = Some(fi.merged_gate_up.weight_scales_128x4);
            }
            if merged_weight.is_none()
                && self
                    .weights
                    .gate_proj
                    .weight
                    .0
                    .checked_add(gate_packed_bytes as u64)
                    == Some(self.weights.up_proj.weight.0)
            {
                merged_weight = Some(self.weights.gate_proj.weight);
            }
            // Materialize a separate merged weight copy when policy allows
            // (`Auto` on discrete `sm_count > 64` or explicit `Enabled`) AND
            // VRAM headroom meets the platform reserve (`4 GiB` discrete / `12 GiB` UMA).
            let sm_count = gpu_sm_count(gpu);
            if merged_weight.is_none()
                && cublaslt_merged_gate_up_allow_materialize(policy, sm_count)
                && cublaslt_merged_gate_up_vram_ok_for_sm(
                    gpu.free_memory().ok(),
                    merged_packed_bytes,
                    sm_count,
                )
            {
                let d_merged = gpu
                    .alloc(merged_packed_bytes)
                    .context("alloc cuBLASLt NVFP4 merged gate/up weight")?;
                let copy_res = (|| -> Result<()> {
                    gpu.copy_d2d_async(
                        self.weights.gate_proj.weight,
                        d_merged,
                        gate_packed_bytes,
                        stream,
                    )?;
                    gpu.copy_d2d_async(
                        self.weights.up_proj.weight,
                        d_merged.offset(gate_packed_bytes),
                        gate_packed_bytes,
                        stream,
                    )?;
                    gpu.synchronize(stream)?;
                    Ok(())
                })();
                if let Err(err) = copy_res {
                    let _ = gpu.free(d_merged);
                    return Err(err);
                }
                merged_weight = Some(d_merged);
                owns_merged_weight = true;
            }
        }

        let (gate, up, merged_scales, owns_merged_scales) = if let Some(borrowed_ms) =
            borrowed_merged_scales
        {
            let gate = upload_swizzled_nvfp4_scales_128x4(gpu, &self.weights.gate_proj, inter, h)?;
            let up = match upload_swizzled_nvfp4_scales_128x4(gpu, &self.weights.up_proj, inter, h)
            {
                Ok(ptr) => ptr,
                Err(err) => {
                    let _ = gpu.free(gate);
                    return Err(err);
                }
            };
            (gate, up, Some(borrowed_ms), false)
        } else if merged_weight.is_some() && inter.is_multiple_of(128) {
            // When inter % 128 == 0 (e.g. Qwen3.8-27B inter=17408), the 128x4
            // swizzled scales for [gate; up] are the exact concatenation of
            // gate and up 128x4 tiles: ONE allocation serves both merged and
            // split views with ZERO extra scale VRAM.
            let merged_s = match upload_swizzled_merged_gate_up_scales_128x4(
                gpu,
                &self.weights.gate_proj,
                &self.weights.up_proj,
                inter,
                h,
            ) {
                Ok(ptr) => ptr,
                Err(err) => {
                    if owns_merged_weight && let Some(mw) = merged_weight {
                        let _ = gpu.free(mw);
                    }
                    return Err(err);
                }
            };
            let gate_scale_bytes = spark_runtime::cublaslt::block_scale_padded_bytes(
                inter,
                h,
                spark_runtime::cublaslt::NVFP4_BLOCK_SIZE,
            );
            (
                merged_s,
                merged_s.offset(gate_scale_bytes),
                Some(merged_s),
                true,
            )
        } else {
            let gate =
                match upload_swizzled_nvfp4_scales_128x4(gpu, &self.weights.gate_proj, inter, h) {
                    Ok(ptr) => ptr,
                    Err(err) => {
                        if owns_merged_weight && let Some(mw) = merged_weight {
                            let _ = gpu.free(mw);
                        }
                        return Err(err);
                    }
                };
            let up = match upload_swizzled_nvfp4_scales_128x4(gpu, &self.weights.up_proj, inter, h)
            {
                Ok(ptr) => ptr,
                Err(err) => {
                    let _ = gpu.free(gate);
                    if owns_merged_weight && let Some(mw) = merged_weight {
                        let _ = gpu.free(mw);
                    }
                    return Err(err);
                }
            };
            let merged_s = if merged_weight.is_some() {
                match upload_swizzled_merged_gate_up_scales_128x4(
                    gpu,
                    &self.weights.gate_proj,
                    &self.weights.up_proj,
                    inter,
                    h,
                ) {
                    Ok(ptr) => Some(ptr),
                    Err(err) => {
                        let _ = gpu.free(gate);
                        let _ = gpu.free(up);
                        if owns_merged_weight && let Some(mw) = merged_weight {
                            let _ = gpu.free(mw);
                        }
                        return Err(err);
                    }
                }
            } else {
                None
            };
            let owns_ms = merged_s.is_some();
            (gate, up, merged_s, owns_ms)
        };

        let cleanup_gate_up = |gpu: &dyn GpuBackend| {
            if owns_merged_scales && merged_scales == Some(gate) {
                let _ = gpu.free(gate);
            } else {
                let _ = gpu.free(gate);
                let _ = gpu.free(up);
                if owns_merged_scales && let Some(ms) = merged_scales {
                    let _ = gpu.free(ms);
                }
            }
            if owns_merged_weight && let Some(mw) = merged_weight {
                let _ = gpu.free(mw);
            }
        };

        let down = match upload_swizzled_nvfp4_scales_128x4(gpu, &self.weights.down_proj, h, inter)
        {
            Ok(ptr) => ptr,
            Err(err) => {
                cleanup_gate_up(gpu);
                return Err(err);
            }
        };
        let dynamic_scalars = match gpu.alloc(CUBLASLT_DYNAMIC_SCALARS_BYTES) {
            Ok(ptr) => ptr,
            Err(err) => {
                cleanup_gate_up(gpu);
                let _ = gpu.free(down);
                return Err(err).context("alloc cuBLASLt NVFP4 dynamic scalar scratch");
            }
        };

        if let Some(old) = guard.replace(CublasLtNvfp4Scales {
            gate,
            up,
            down,
            dynamic_scalars,
            merged_weight,
            owns_merged_weight,
            merged_scales,
            owns_merged_scales,
            gate_src,
            up_src,
            down_src,
        }) {
            if old.owns_merged_scales && old.merged_scales == Some(old.gate) {
                let _ = gpu.free(old.gate);
            } else {
                let _ = gpu.free(old.gate);
                let _ = gpu.free(old.up);
                if old.owns_merged_scales
                    && let Some(ms) = old.merged_scales
                {
                    let _ = gpu.free(ms);
                }
            }
            if old.owns_merged_weight
                && let Some(mw) = old.merged_weight
            {
                let _ = gpu.free(mw);
            }
            let _ = gpu.free(old.down);
            let _ = gpu.free(old.dynamic_scalars);
        }
        Ok((
            gate,
            up,
            down,
            dynamic_scalars,
            merged_weight.zip(merged_scales),
        ))
    }

    /// Native cuBLASLt block-scaled NVFP4 (`VEC16_UE4M3`) prefill route.
    ///
    /// Auto-enabled on Blackwell datacenter (`SM100`/`SM103`,
    /// `compute_capability == Some((10, 0) | (10, 3))`)
    /// and Blackwell desktop/workstation (`SM120`) when FlashInfer FFN is absent, or via
    /// `ATLAS_CUBLASLT_NVFP4=1`. Returns `Ok(false)` when disabled, geometrically
    /// ineligible, or missing the `128x4` activation quantizer so prefill falls back
    /// cleanly to `w4a16_gemm.cu`.
    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    fn try_forward_cublaslt_nvfp4_prefill(
        &self,
        input: DevicePtr,
        m: u32,
        h: u32,
        inter: u32,
        gate_out: DevicePtr,
        up_out: DevicePtr,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<bool> {
        let cc = ctx.gpu.compute_capability();
        #[cfg(target_os = "linux")]
        let flashinfer_ffn = if self.flashinfer_ffn_prefill.is_some() {
            FlashInferFfnAvailability::Prepared
        } else {
            FlashInferFfnAvailability::Absent
        };
        #[cfg(not(target_os = "linux"))]
        let flashinfer_ffn = FlashInferFfnAvailability::Absent;
        let checkpoint_scales = self
            .cublaslt_checkpoint_scales
            .or(self.e2m1_checkpoint_scales);
        let has_scale_source = checkpoint_scales.is_some() || self.nvfp4_absmax_k.0 != 0;
        let has_quantizer_128x4 = select_cublaslt_quantize_128x4_kernel(
            self.nvfp4_quantize_128x4_ptx_k,
            self.nvfp4_quantize_128x4_k,
        )
        .0 != 0;
        if !cublaslt_nvfp4_route_enabled(
            prefill_ffn_cublaslt_nvfp4_policy(),
            cc,
            flashinfer_ffn,
            m,
            h,
            inter,
            has_quantizer_128x4,
            has_scale_source,
        ) {
            return Ok(false);
        }

        let weights_ready = [
            &self.weights.gate_proj,
            &self.weights.up_proj,
            &self.weights.down_proj,
        ]
        .into_iter()
        .all(nvfp4_weight_ready);
        if !weights_ready {
            return Ok(false);
        }

        clear_precomputed_proj_absmax();
        let sm_count = gpu_sm_count(ctx.gpu);
        let merged_silu_k = select_merged_silu_quant_kernel(
            self.nvfp4_merged_silu_quantize_128x4_ptx_k,
            self.nvfp4_merged_silu_quantize_128x4_k,
            sm_count,
        );
        let down_scale = checkpoint_scales.map(|s| s.down);
        let merged_eligible = self.activation == FfnActivation::SiLU
            && self.weights.gate_proj.weight_scale_2.to_bits()
                == self.weights.up_proj.weight_scale_2.to_bits()
            && down_scale.is_some()
            && merged_silu_k.0 != 0;

        let l2_strip_opt = ffn_prefill_l2_strip_rows(ctx.gpu);
        let _scratch_use = self.lock_e2m1_use(stream)?;
        let (gate_s128x4, up_s128x4, down_s128x4, dynamic_scalars, merged_gate_up) =
            self.ensure_cublaslt_nvfp4_scales(ctx.gpu, h, inter, merged_eligible, stream)?;

        let l2_strip_active = l2_strip_opt.filter(|&strip| {
            m > strip && merged_eligible && merged_gate_up.is_some() && checkpoint_scales.is_some()
        });
        let scratch_rows = l2_strip_active.unwrap_or(m);
        // Pre-size activation scratch for `scratch_rows` × `max(h, inter)` so
        // down_proj never reallocates while gate/up remain in flight on `stream`.
        let _ = self.ensure_e2m1_scratch(ctx.gpu, scratch_rows as usize, h.max(inter) as usize)?;

        let merged_out_opt =
            if let (true, Some((merged_w, merged_s))) = (merged_eligible, merged_gate_up) {
                let proj_bytes = (scratch_rows as usize)
                    .checked_mul(inter as usize)
                    .and_then(|e| e.checked_mul(2))
                    .context("cuBLASLt merged gate/up projection byte overflow")?;
                let merged_bytes = proj_bytes
                    .checked_mul(2)
                    .context("cuBLASLt merged gate/up output byte overflow")?;
                let buf = if ctx.buffers.sizes().ffn_gate_up_bf16 >= merged_bytes {
                    Some(ctx.buffers.ffn_gate_up_bf16())
                } else if gate_out.0.checked_add(proj_bytes as u64) == Some(up_out.0)
                    && l2_strip_active.is_none()
                {
                    Some(gate_out)
                } else {
                    ensure_shared_merged_gate_up_out(ctx.gpu, merged_bytes)?
                };
                buf.map(|out_buf| (merged_w, merged_s, out_buf))
            } else {
                None
            };

        let zero_sync_dynamic = checkpoint_scales.is_none()
            && self.nvfp4_quantize_128x4_from_absmax_k.0 != 0
            && self.nvfp4_dynamic_combined_alpha_k.0 != 0;
        static LOGGED: std::sync::Once = std::sync::Once::new();
        LOGGED.call_once(|| {
            tracing::info!(
                cc = ?cc,
                m,
                h,
                inter,
                static_scale = checkpoint_scales.is_some(),
                zero_sync_dynamic,
                merged_gate_up = merged_out_opt.is_some(),
                l2_strip_rows = ?l2_strip_active,
                fused_silu_quant = cublaslt_fused_silu_quantize_scale(
                    self.activation,
                    down_scale,
                    self.nvfp4_silu_quantize_128x4_k,
                )
                .is_some(),
                "ENGAGED cuBLASLt NVFP4 (VEC16_UE4M3) dense FFN prefill route"
            );
        });

        let output = ctx.buffers.moe_output();
        // L2-resident strip-mined FFN pass for SM120 (`M > strip`, static scales +
        // merged `[gate; up]`): keeps `merged_out` (`[strip, 2*inter]` BF16 = 71.3 MB
        // at `strip = 1024, inter = 17408`) hot inside SM120's 96 MiB L2 cache instead
        // of writing/reading 570 MB through DRAM. Handles non-1024 remainders cleanly
        // because each strip pads its `128x4` scales independently.
        if let (Some(strip), Some((merged_w, merged_s128x4, merged_out)), Some(scales)) =
            (l2_strip_active, merged_out_opt, checkpoint_scales)
        {
            let merged_n = inter
                .checked_mul(2)
                .context("cuBLASLt merged gate/up N overflow")?;
            let quant_k = select_cublaslt_quantize_128x4_kernel(
                self.nvfp4_quantize_128x4_ptx_k,
                self.nvfp4_quantize_128x4_k,
            );
            let merged_scalars = resolve_cublaslt_scalars(
                ctx.gpu,
                self.nvfp4_dynamic_combined_alpha_k,
                CublasLtInputScale::Static(scales.gate_up),
                self.weights.gate_proj.weight_scale_2,
                CublasLtAlphaSlot::Primary,
                stream,
            )?;
            let down_scalars = resolve_cublaslt_scalars(
                ctx.gpu,
                self.nvfp4_dynamic_combined_alpha_k,
                CublasLtInputScale::Static(scales.down),
                self.weights.down_proj.weight_scale_2,
                CublasLtAlphaSlot::Primary,
                stream,
            )?;
            let (a_packed, a_scale, _) =
                self.ensure_e2m1_scratch(ctx.gpu, strip as usize, h.max(inter) as usize)?;
            let row_bytes_bf16 = (h as usize) * 2;
            let mut row0 = 0u32;
            while row0 < m {
                let m_s = (m - row0).min(strip);
                let in_s = input.offset((row0 as usize) * row_bytes_bf16);
                let out_s = output.offset((row0 as usize) * row_bytes_bf16);
                ops::quantize_bf16_to_nvfp4_atlas_128x4(
                    ctx.gpu,
                    quant_k,
                    in_s,
                    a_packed,
                    a_scale,
                    scales.gate_up,
                    m_s,
                    h,
                    stream,
                )?;
                spark_runtime::cublaslt::nvfp4_gemm_block_scaled(
                    a_packed.0,
                    a_scale.0,
                    merged_w.0,
                    merged_s128x4.0,
                    merged_out.0,
                    m_s,
                    merged_n,
                    h,
                    merged_n,
                    merged_scalars,
                    stream,
                )?;
                ops::quantize_merged_silu_mul_bf16_to_nvfp4_atlas_128x4(
                    ctx.gpu,
                    merged_silu_k,
                    merged_out,
                    a_packed,
                    a_scale,
                    scales.down,
                    m_s,
                    inter,
                    stream,
                )?;
                spark_runtime::cublaslt::nvfp4_gemm_block_scaled(
                    a_packed.0,
                    a_scale.0,
                    self.weights.down_proj.weight.0,
                    down_s128x4.0,
                    out_s.0,
                    m_s,
                    h,
                    inter,
                    h,
                    down_scalars,
                    stream,
                )?;
                row0 += m_s;
            }
            return Ok(true);
        }

        // If L2 strip-mining was initially planned but fell back (e.g. VRAM-tight
        // merged buffer unavailable), grow scratch to full `m` × `max(h, inter)`
        // before launching gate/up so `down_proj` never reallocates mid-stream.
        if l2_strip_active.is_some() {
            let _ = self.ensure_e2m1_scratch(ctx.gpu, m as usize, h.max(inter) as usize)?;
        }

        let gate_up_scale = checkpoint_scales.map(|s| s.gate_up);
        let (a_packed, a_scale, gate_up_input_scale) = self.prepare_e2m1_input_128x4(
            ctx,
            input,
            dynamic_scalars,
            m,
            h,
            gate_up_scale,
            stream,
        )?;

        let (down_packed, down_scale_buf, down_input_scale) =
            if let (Some((merged_w, merged_s128x4, merged_out)), Some(scale2)) =
                (merged_out_opt, down_scale)
            {
                let merged_n = inter
                    .checked_mul(2)
                    .context("cuBLASLt merged gate/up N overflow")?;
                let merged_scalars = resolve_cublaslt_scalars(
                    ctx.gpu,
                    self.nvfp4_dynamic_combined_alpha_k,
                    gate_up_input_scale,
                    self.weights.gate_proj.weight_scale_2,
                    CublasLtAlphaSlot::Primary,
                    stream,
                )?;
                spark_runtime::cublaslt::nvfp4_gemm_block_scaled(
                    a_packed.0,
                    a_scale.0,
                    merged_w.0,
                    merged_s128x4.0,
                    merged_out.0,
                    m,
                    merged_n,
                    h,
                    merged_n,
                    merged_scalars,
                    stream,
                )?;
                let (down_a_packed, down_a_scale, _) =
                    self.ensure_e2m1_scratch(ctx.gpu, m as usize, inter as usize)?;
                ops::quantize_merged_silu_mul_bf16_to_nvfp4_atlas_128x4(
                    ctx.gpu,
                    merged_silu_k,
                    merged_out,
                    down_a_packed,
                    down_a_scale,
                    scale2,
                    m,
                    inter,
                    stream,
                )?;
                (
                    down_a_packed,
                    down_a_scale,
                    CublasLtInputScale::Static(scale2),
                )
            } else {
                let gate_scalars = resolve_cublaslt_scalars(
                    ctx.gpu,
                    self.nvfp4_dynamic_combined_alpha_k,
                    gate_up_input_scale,
                    self.weights.gate_proj.weight_scale_2,
                    CublasLtAlphaSlot::Primary,
                    stream,
                )?;
                let up_scalars = resolve_cublaslt_scalars(
                    ctx.gpu,
                    self.nvfp4_dynamic_combined_alpha_k,
                    gate_up_input_scale,
                    self.weights.up_proj.weight_scale_2,
                    CublasLtAlphaSlot::Secondary,
                    stream,
                )?;
                spark_runtime::cublaslt::nvfp4_gemm_block_scaled(
                    a_packed.0,
                    a_scale.0,
                    self.weights.gate_proj.weight.0,
                    gate_s128x4.0,
                    gate_out.0,
                    m,
                    inter,
                    h,
                    inter,
                    gate_scalars,
                    stream,
                )?;
                spark_runtime::cublaslt::nvfp4_gemm_block_scaled(
                    a_packed.0,
                    a_scale.0,
                    self.weights.up_proj.weight.0,
                    up_s128x4.0,
                    up_out.0,
                    m,
                    inter,
                    h,
                    inter,
                    up_scalars,
                    stream,
                )?;

                if let Some(scale2) = cublaslt_fused_silu_quantize_scale(
                    self.activation,
                    down_scale,
                    self.nvfp4_silu_quantize_128x4_k,
                ) {
                    let (down_a_packed, down_a_scale, _) =
                        self.ensure_e2m1_scratch(ctx.gpu, m as usize, inter as usize)?;
                    ops::quantize_silu_mul_bf16_to_nvfp4_atlas_128x4(
                        ctx.gpu,
                        self.nvfp4_silu_quantize_128x4_k,
                        gate_out,
                        up_out,
                        down_a_packed,
                        down_a_scale,
                        scale2,
                        m,
                        inter,
                        stream,
                    )?;
                    (
                        down_a_packed,
                        down_a_scale,
                        CublasLtInputScale::Static(scale2),
                    )
                } else {
                    ops::silu_mul(
                        ctx.gpu,
                        self.act_mul,
                        gate_out,
                        up_out,
                        gate_out,
                        m * inter,
                        stream,
                    )?;
                    self.prepare_e2m1_input_128x4(
                        ctx,
                        gate_out,
                        dynamic_scalars,
                        m,
                        inter,
                        down_scale,
                        stream,
                    )?
                }
            };

        let down_scalars = resolve_cublaslt_scalars(
            ctx.gpu,
            self.nvfp4_dynamic_combined_alpha_k,
            down_input_scale,
            self.weights.down_proj.weight_scale_2,
            CublasLtAlphaSlot::Primary,
            stream,
        )?;
        spark_runtime::cublaslt::nvfp4_gemm_block_scaled(
            down_packed.0,
            down_scale_buf.0,
            self.weights.down_proj.weight.0,
            down_s128x4.0,
            output.0,
            m,
            h,
            inter,
            h,
            down_scalars,
            stream,
        )?;

        Ok(true)
    }

    #[cfg(feature = "cuda")]
    #[expect(
        clippy::type_complexity,
        reason = "internal tuple returning cached cuBLASLt 128x4 scale pointers and optional merged gate/up buffers"
    )]
    fn cached_cublaslt_nvfp4_scales(
        &self,
    ) -> Option<(
        DevicePtr,
        DevicePtr,
        DevicePtr,
        Option<(DevicePtr, DevicePtr)>,
    )> {
        let guard = self
            .cublaslt_nvfp4_scales
            .lock()
            .expect("cublaslt_nvfp4_scales lock poisoned");
        let gate_src = self.weights.gate_proj.weight_scale;
        let up_src = self.weights.up_proj.weight_scale;
        let down_src = self.weights.down_proj.weight_scale;
        let cached = guard.as_ref()?;
        if cached.gate_src == gate_src && cached.up_src == up_src && cached.down_src == down_src {
            Some((
                cached.gate,
                cached.up,
                cached.down,
                cached.merged_weight.zip(cached.merged_scales),
            ))
        } else {
            None
        }
    }

    /// Multi-seq decode (`forward_kgamma`) cuBLASLt NVFP4 (`VEC16_UE4M3`) FFN
    /// route for `M >= 32` using a dedicated per-`(gpu, stream)` decode scratch
    /// (`ensure_decode_ffn_e2m1_scratch`) so it never collides with prefill's
    /// stream-bound `e2m1_scratch` and stays 100% CUDA-graph capture safe once
    /// warmed on step 1.
    #[cfg(feature = "cuda")]
    #[expect(
        clippy::too_many_arguments,
        reason = "mirrors forward_kgamma FFN entry parameters plus scratch outputs and CUDA stream"
    )]
    fn try_forward_cublaslt_nvfp4_kgamma(
        &self,
        input: DevicePtr,
        m: u32,
        h: u32,
        inter: u32,
        gate_out: DevicePtr,
        up_out: DevicePtr,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<bool> {
        let Some(scales) = self.cublaslt_checkpoint_scales else {
            return Ok(false);
        };
        let cc = ctx.gpu.compute_capability();
        #[cfg(target_os = "linux")]
        let flashinfer_ffn = if self.flashinfer_ffn_prefill.is_some() {
            FlashInferFfnAvailability::Prepared
        } else {
            FlashInferFfnAvailability::Absent
        };
        #[cfg(not(target_os = "linux"))]
        let flashinfer_ffn = FlashInferFfnAvailability::Absent;
        let has_quantizer_128x4 = select_cublaslt_quantize_128x4_kernel(
            self.nvfp4_quantize_128x4_ptx_k,
            self.nvfp4_quantize_128x4_k,
        )
        .0 != 0;
        if !cublaslt_nvfp4_route_enabled(
            prefill_ffn_cublaslt_nvfp4_policy(),
            cc,
            flashinfer_ffn,
            m,
            h,
            inter,
            has_quantizer_128x4,
            true,
        ) {
            return Ok(false);
        }
        let weights_ready = [
            &self.weights.gate_proj,
            &self.weights.up_proj,
            &self.weights.down_proj,
        ]
        .into_iter()
        .all(nvfp4_weight_ready);
        if !weights_ready {
            return Ok(false);
        }

        let sm_count = gpu_sm_count(ctx.gpu);
        let merged_silu_k = select_merged_silu_quant_kernel(
            self.nvfp4_merged_silu_quantize_128x4_ptx_k,
            self.nvfp4_merged_silu_quantize_128x4_k,
            sm_count,
        );
        let merged_eligible = self.activation == FfnActivation::SiLU
            && self.weights.gate_proj.weight_scale_2.to_bits()
                == self.weights.up_proj.weight_scale_2.to_bits()
            && merged_silu_k.0 != 0;

        let (gate_s128x4, up_s128x4, down_s128x4, merged_gate_up) = if ctx.graph_capture {
            let Some(cached) = self.cached_cublaslt_nvfp4_scales() else {
                return Ok(false);
            };
            cached
        } else {
            let (g, u, d, _, merged) =
                self.ensure_cublaslt_nvfp4_scales(ctx.gpu, h, inter, merged_eligible, stream)?;
            (g, u, d, merged)
        };

        let want_merged_bytes = if merged_eligible && merged_gate_up.is_some() {
            (m as usize)
                .checked_mul(inter as usize)
                .and_then(|e| e.checked_mul(4))
                .context("cuBLASLt kgamma merged gate/up output byte overflow")?
        } else {
            0
        };
        let Some((a_packed, a_scale, decode_merged_out, decode_merged_cap)) =
            ensure_decode_ffn_e2m1_scratch(
                ctx.gpu,
                stream,
                m as usize,
                h.max(inter) as usize,
                want_merged_bytes,
                !ctx.graph_capture,
            )?
        else {
            return Ok(false);
        };

        let merged_out_opt =
            if let (true, Some((merged_w, merged_s))) = (merged_eligible, merged_gate_up) {
                let proj_bytes = (m as usize)
                    .checked_mul(inter as usize)
                    .and_then(|e| e.checked_mul(2))
                    .context("cuBLASLt kgamma merged gate/up projection byte overflow")?;
                let merged_bytes = proj_bytes
                    .checked_mul(2)
                    .context("cuBLASLt kgamma merged gate/up output byte overflow")?;
                if ctx.buffers.sizes().ffn_gate_up_bf16 >= merged_bytes {
                    Some((merged_w, merged_s, ctx.buffers.ffn_gate_up_bf16()))
                } else if !decode_merged_out.is_null() && decode_merged_cap >= merged_bytes {
                    Some((merged_w, merged_s, decode_merged_out))
                } else if gate_out.0.checked_add(proj_bytes as u64) == Some(up_out.0) {
                    Some((merged_w, merged_s, gate_out))
                } else {
                    None
                }
            } else {
                None
            };

        let quant_k = select_cublaslt_quantize_128x4_kernel(
            self.nvfp4_quantize_128x4_ptx_k,
            self.nvfp4_quantize_128x4_k,
        );
        ops::quantize_bf16_to_nvfp4_atlas_128x4(
            ctx.gpu,
            quant_k,
            input,
            a_packed,
            a_scale,
            scales.gate_up,
            m,
            h,
            stream,
        )?;

        if let Some((merged_w, merged_s128x4, merged_out)) = merged_out_opt {
            let merged_n = inter
                .checked_mul(2)
                .context("cuBLASLt kgamma merged gate/up N overflow")?;
            let merged_scalars = resolve_cublaslt_scalars(
                ctx.gpu,
                self.nvfp4_dynamic_combined_alpha_k,
                CublasLtInputScale::Static(scales.gate_up),
                self.weights.gate_proj.weight_scale_2,
                CublasLtAlphaSlot::Primary,
                stream,
            )?;
            spark_runtime::cublaslt::nvfp4_gemm_block_scaled(
                a_packed.0,
                a_scale.0,
                merged_w.0,
                merged_s128x4.0,
                merged_out.0,
                m,
                merged_n,
                h,
                merged_n,
                merged_scalars,
                stream,
            )?;
            ops::quantize_merged_silu_mul_bf16_to_nvfp4_atlas_128x4(
                ctx.gpu,
                merged_silu_k,
                merged_out,
                a_packed,
                a_scale,
                scales.down,
                m,
                inter,
                stream,
            )?;
        } else {
            let gate_scalars = resolve_cublaslt_scalars(
                ctx.gpu,
                self.nvfp4_dynamic_combined_alpha_k,
                CublasLtInputScale::Static(scales.gate_up),
                self.weights.gate_proj.weight_scale_2,
                CublasLtAlphaSlot::Primary,
                stream,
            )?;
            let up_scalars = resolve_cublaslt_scalars(
                ctx.gpu,
                self.nvfp4_dynamic_combined_alpha_k,
                CublasLtInputScale::Static(scales.gate_up),
                self.weights.up_proj.weight_scale_2,
                CublasLtAlphaSlot::Secondary,
                stream,
            )?;
            spark_runtime::cublaslt::nvfp4_gemm_block_scaled(
                a_packed.0,
                a_scale.0,
                self.weights.gate_proj.weight.0,
                gate_s128x4.0,
                gate_out.0,
                m,
                inter,
                h,
                inter,
                gate_scalars,
                stream,
            )?;
            spark_runtime::cublaslt::nvfp4_gemm_block_scaled(
                a_packed.0,
                a_scale.0,
                self.weights.up_proj.weight.0,
                up_s128x4.0,
                up_out.0,
                m,
                inter,
                h,
                inter,
                up_scalars,
                stream,
            )?;
            if let Some(scale2) = cublaslt_fused_silu_quantize_scale(
                self.activation,
                Some(scales.down),
                self.nvfp4_silu_quantize_128x4_k,
            ) {
                ops::quantize_silu_mul_bf16_to_nvfp4_atlas_128x4(
                    ctx.gpu,
                    self.nvfp4_silu_quantize_128x4_k,
                    gate_out,
                    up_out,
                    a_packed,
                    a_scale,
                    scale2,
                    m,
                    inter,
                    stream,
                )?;
            } else {
                ops::silu_mul(
                    ctx.gpu,
                    self.act_mul,
                    gate_out,
                    up_out,
                    gate_out,
                    m * inter,
                    stream,
                )?;
                ops::quantize_bf16_to_nvfp4_atlas_128x4(
                    ctx.gpu,
                    quant_k,
                    gate_out,
                    a_packed,
                    a_scale,
                    scales.down,
                    m,
                    inter,
                    stream,
                )?;
            }
        }

        let output = ctx.buffers.moe_output();
        let down_scalars = resolve_cublaslt_scalars(
            ctx.gpu,
            self.nvfp4_dynamic_combined_alpha_k,
            CublasLtInputScale::Static(scales.down),
            self.weights.down_proj.weight_scale_2,
            CublasLtAlphaSlot::Primary,
            stream,
        )?;
        spark_runtime::cublaslt::nvfp4_gemm_block_scaled(
            a_packed.0,
            a_scale.0,
            self.weights.down_proj.weight.0,
            down_s128x4.0,
            output.0,
            m,
            h,
            inter,
            h,
            down_scalars,
            stream,
        )?;

        static LOGGED: std::sync::Once = std::sync::Once::new();
        LOGGED.call_once(|| {
            tracing::info!(
                m,
                h,
                inter,
                merged_gate_up = merged_out_opt.is_some(),
                graph_capture = ctx.graph_capture,
                "ENGAGED cuBLASLt NVFP4 (VEC16_UE4M3) multi-seq decode FFN route"
            );
        });
        Ok(true)
    }

    /// Fused prefill post-mixer block for single-GPU static-scale NVFP4 FFN
    /// (`sm_count > 64` auto-ON, `ATLAS_FFN_PREFILL_FUSED_NORM`):
    ///
    /// Replaces:
    /// 1. `residual_add_rms_norm(hidden += src -> norm_output, residual)`
    /// 2. `quantize_bf16_to_nvfp4_atlas_128x4(norm_output -> a_packed)`
    /// 3. `gate_up` + `silu_mul_quant` + `down_proj(-> moe_output)`
    /// 4. `bf16_residual_add(hidden += moe_output)`
    ///
    /// with a strip-mined L2-resident sequence per `strip = 1024` rows:
    /// - `residual_add_rms_norm_quantize_nvfp4_128x4(hidden_s += src_s, residual=NULL -> a_packed)`
    /// - `merged_gate_up(a_packed -> merged_out)` (`71.3 MB < 96 MiB L2`)
    /// - `quantize_merged_silu_mul_bf16_to_nvfp4_atlas_128x4_ptx(merged_out -> a_packed)`
    /// - `down_proj(a_packed -> hidden_s, beta = 1.0)`
    ///
    /// Returns `Ok(true)` when executed, or `Ok(false)` to let the caller run
    /// the standard unfused sequence cleanly.
    #[cfg(feature = "cuda")]
    #[expect(
        clippy::too_many_arguments,
        reason = "fused layer-tail prefill block accepting hidden, residual src, norm weight, shape, context, and stream"
    )]
    pub(crate) fn try_forward_prefill_fused_residual_norm(
        &self,
        hidden: DevicePtr,
        src: DevicePtr,
        norm_weight: &DenseWeight,
        num_tokens: usize,
        eps: f32,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<bool> {
        let Ok(m) = u32::try_from(num_tokens) else {
            return Ok(false);
        };
        let h = ctx.config.hidden_size as u32;
        let inter = ctx.config.intermediate_size as u32;
        if !ffn_prefill_fused_norm_enabled(ctx.gpu)
            || ctx.graph_capture
            || ctx.gpu.stream_is_capturing(stream)
            || ctx.config.use_fp32_residual()
            || ctx.config.model_type == "gemma4"
            || ctx.config.tp_world_size > 1
            || self.bf16_weights.is_some()
            || self.activation != FfnActivation::SiLU
            || self.nvfp4_residual_add_rms_norm_quant_128x4_k.0 == 0
            || hidden.is_null()
            || src.is_null()
            || norm_weight.weight.is_null()
            || !hidden.0.is_multiple_of(16)
            || !src.0.is_multiple_of(16)
            || !norm_weight.weight.0.is_multiple_of(16)
        {
            return Ok(false);
        }

        let cc = ctx.gpu.compute_capability();
        #[cfg(target_os = "linux")]
        let flashinfer_ffn = if self.flashinfer_ffn_prefill.is_some() {
            FlashInferFfnAvailability::Prepared
        } else {
            FlashInferFfnAvailability::Absent
        };
        #[cfg(not(target_os = "linux"))]
        let flashinfer_ffn = FlashInferFfnAvailability::Absent;
        let Some(scales) = self
            .cublaslt_checkpoint_scales
            .or(self.e2m1_checkpoint_scales)
        else {
            return Ok(false);
        };
        let sm_count = gpu_sm_count(ctx.gpu);
        let merged_silu_k = select_merged_silu_quant_kernel(
            self.nvfp4_merged_silu_quantize_128x4_ptx_k,
            self.nvfp4_merged_silu_quantize_128x4_k,
            sm_count,
        );
        if merged_silu_k.0 == 0
            || self.weights.gate_proj.weight_scale_2.to_bits()
                != self.weights.up_proj.weight_scale_2.to_bits()
        {
            return Ok(false);
        }
        if !cublaslt_nvfp4_route_enabled(
            prefill_ffn_cublaslt_nvfp4_policy(),
            cc,
            flashinfer_ffn,
            m,
            h,
            inter,
            true,
            true,
        ) {
            return Ok(false);
        }
        let weights_ready = [
            &self.weights.gate_proj,
            &self.weights.up_proj,
            &self.weights.down_proj,
        ]
        .into_iter()
        .all(nvfp4_weight_ready);
        if !weights_ready {
            return Ok(false);
        }

        let _scratch_use = self.lock_e2m1_use(stream)?;
        let (_gate_s, _up_s, down_s128x4, _dyn, merged_gate_up) =
            self.ensure_cublaslt_nvfp4_scales(ctx.gpu, h, inter, true, stream)?;
        let Some((merged_w, merged_s128x4)) = merged_gate_up else {
            return Ok(false);
        };

        let strip = ffn_prefill_l2_strip_rows(ctx.gpu).unwrap_or(m).min(m);
        let scratch_rows = strip.max(1);
        let proj_bytes = (scratch_rows as usize)
            .checked_mul(inter as usize)
            .and_then(|e| e.checked_mul(2))
            .context("fused prefill FFN merged projection byte overflow")?;
        let merged_bytes = proj_bytes
            .checked_mul(2)
            .context("fused prefill FFN merged output byte overflow")?;
        let merged_out = if ctx.buffers.sizes().ffn_gate_up_bf16 >= merged_bytes {
            ctx.buffers.ffn_gate_up_bf16()
        } else {
            let Some(ptr) = ensure_shared_merged_gate_up_out(ctx.gpu, merged_bytes)? else {
                return Ok(false);
            };
            ptr
        };

        let (a_packed, a_scale, _) =
            self.ensure_e2m1_scratch(ctx.gpu, scratch_rows as usize, h.max(inter) as usize)?;
        let merged_n = inter
            .checked_mul(2)
            .context("fused prefill FFN merged gate/up N overflow")?;
        let merged_alpha = scales.gate_up * self.weights.gate_proj.weight_scale_2;
        let down_alpha = scales.down * self.weights.down_proj.weight_scale_2;
        ensure!(
            merged_alpha.is_finite() && merged_alpha > 0.0,
            "fused prefill FFN invalid merged alpha {merged_alpha}"
        );
        ensure!(
            down_alpha.is_finite() && down_alpha > 0.0,
            "fused prefill FFN invalid down alpha {down_alpha}"
        );
        let merged_scalars = spark_runtime::cublaslt::BlockScaleScalars::host(merged_alpha, 0.0);
        let down_scalars = spark_runtime::cublaslt::BlockScaleScalars::host(down_alpha, 1.0);

        clear_precomputed_proj_absmax();
        static LOGGED: std::sync::Once = std::sync::Once::new();
        LOGGED.call_once(|| {
            tracing::info!(
                cc = ?cc,
                m,
                h,
                inter,
                strip_rows = strip,
                "ENGAGED fused residual_add_rms_norm_quantize_nvfp4_128x4 + L2-strip + beta=1.0 FFN prefill"
            );
        });

        let row_bytes_bf16 = (h as usize) * 2;
        let mut row0 = 0u32;
        while row0 < m {
            let m_s = (m - row0).min(strip);
            let h_s = hidden.offset((row0 as usize) * row_bytes_bf16);
            let s_s = src.offset((row0 as usize) * row_bytes_bf16);
            ops::residual_add_rms_norm_quantize_nvfp4_128x4(
                ctx.gpu,
                self.nvfp4_residual_add_rms_norm_quant_128x4_k,
                h_s,
                s_s,
                norm_weight,
                DevicePtr::NULL,
                a_packed,
                a_scale,
                scales.gate_up,
                eps,
                m_s,
                h,
                stream,
            )?;
            spark_runtime::cublaslt::nvfp4_gemm_block_scaled(
                a_packed.0,
                a_scale.0,
                merged_w.0,
                merged_s128x4.0,
                merged_out.0,
                m_s,
                merged_n,
                h,
                merged_n,
                merged_scalars,
                stream,
            )?;
            ops::quantize_merged_silu_mul_bf16_to_nvfp4_atlas_128x4(
                ctx.gpu,
                merged_silu_k,
                merged_out,
                a_packed,
                a_scale,
                scales.down,
                m_s,
                inter,
                stream,
            )?;
            spark_runtime::cublaslt::nvfp4_gemm_block_scaled(
                a_packed.0,
                a_scale.0,
                self.weights.down_proj.weight.0,
                down_s128x4.0,
                h_s.0,
                m_s,
                h,
                inter,
                h,
                down_scalars,
                stream,
            )?;
            row0 += m_s;
        }
        Ok(true)
    }

    #[cfg(not(feature = "cuda"))]
    #[expect(
        clippy::too_many_arguments,
        reason = "non-CUDA stub matching fused prefill FFN signature"
    )]
    pub(crate) fn try_forward_prefill_fused_residual_norm(
        &self,
        _hidden: DevicePtr,
        _src: DevicePtr,
        _norm_weight: &DenseWeight,
        _num_tokens: usize,
        _eps: f32,
        _ctx: &ForwardContext,
        _stream: u64,
    ) -> Result<bool> {
        Ok(false)
    }

    /// Eagerly allocate the FP32 split-K workspace for the down_proj
    /// (`[k_splits, 32, n]` where n = hidden, M padded to the M_TILE=32
    /// of the split-K kernel). Called at load time (pre-graph-capture)
    /// because `gpu.alloc` is illegal during CUDA graph capture. No-op
    /// when split-K is disabled or the kernel symbols are missing.
    pub fn alloc_splitk_workspace(
        &self,
        gpu: &dyn GpuBackend,
        n: u32,
        shared: Option<&crate::layers::shared_splitk::SharedSplitkWorkspace>,
    ) -> Result<()> {
        // `n` is the largest split-K output dim this layer might use. The
        // down_proj path needs N=hidden; the gate/up path (when
        // `ATLAS_FFN_GATEUP_SPLITK` is on) needs N=intermediate. Callers pass
        // `max(hidden, intermediate)` so ONE FP32 workspace serves both.
        // gate and up run back-to-back on the same stream and each fully
        // reduces into its own output before the next partial phase, so they
        // can safely share this scratch.
        let down_splits = crate::layers::ffn_down_splitk();
        let gateup_splits = crate::layers::ffn_gateup_splitk();
        if (down_splits == 0 && gateup_splits == 0)
            || self.w4a16_gemm_t_m32_n64_splitk.0 == 0
            || self.reduce_splitk_k.0 == 0
        {
            return Ok(());
        }
        let mut slot = self.splitk_workspace.lock().unwrap();
        if slot.is_none() {
            // 8 = max split clamp; 32 = M_TILE of the split-K kernel.
            let bytes = 8usize * 32 * n as usize * 4;
            *slot = Some(match shared {
                Some(shared) => shared.lease(gpu, bytes)?,
                None => gpu.alloc(bytes)?,
            });
        }
        Ok(())
    }

    /// Whether the M_TILE=16 transposed-weight path is wired up.
    /// Used by `forward_kgamma` for dispatch and by the loader log.
    pub fn has_transposed_ffn(&self) -> bool {
        self.gate_proj_t.is_some()
            && self.up_proj_t.is_some()
            && self.down_proj_t.is_some()
            && self.w4a16_gemm_t_m16.0 != 0
    }

    /// Whether the predequanted-FP8 FFN prefill path is wired up.
    /// Returns true when all three projections have FP8 buffers AND the
    /// `fp8_gemm_t_m128` kernel symbol is present. Used by
    /// `forward_prefill` to choose between the NVFP4 + dequant path and
    /// the predequant fast path.
    pub fn has_fp8_ffn(&self) -> bool {
        self.gate_fp8.is_some()
            && self.up_fp8.is_some()
            && self.down_fp8.is_some()
            && self.fp8_gemm_t_m128_k.0 != 0
    }

    /// Pre-dequant NVFP4 FFN weights to FP8 [N, K] for the predequant
    /// fast prefill path. Mirrors
    /// `Qwen3AttentionLayer::predequant_for_prefill` — uses the
    /// NON-transposed weights in `self.weights.gate_proj` etc. (the
    /// `predequant_nvfp4_to_fp8` kernel reads the original [N, K/2]
    /// NVFP4 layout). Allocates 3 × N×K bytes of GPU memory per layer.
    ///
    /// Called by the loader after `DenseFfnLayer::new` when the
    /// `ATLAS_FFN_PREDEQUANT_FP8` env var is set. Silently no-ops
    /// when the kernel symbol is missing.
    ///
    /// `inter` = intermediate_size (gate/up output dim; down input dim).
    /// `hidden` = hidden_size (gate/up input dim; down output dim).
    pub fn predequant_for_prefill(
        &mut self,
        gpu: &dyn GpuBackend,
        hidden: usize,
        inter: usize,
        stream: u64,
    ) -> Result<()> {
        if self.fp8_gemm_t_m128_k.0 == 0 {
            return Ok(()); // FP8 kernel not available — silently skip
        }
        let predequant_k = gpu.kernel("w4a16", "predequant_nvfp4_to_fp8")?;
        // gate_proj: [inter, hidden] NVFP4 → [inter, hidden] FP8
        self.gate_fp8 = Some(self.weights.gate_proj.predequant_to_fp8(
            gpu,
            predequant_k,
            inter,
            hidden,
            stream,
        )?);
        // up_proj: same shape as gate
        self.up_fp8 = Some(self.weights.up_proj.predequant_to_fp8(
            gpu,
            predequant_k,
            inter,
            hidden,
            stream,
        )?);
        // down_proj: [hidden, inter]
        self.down_fp8 = Some(self.weights.down_proj.predequant_to_fp8(
            gpu,
            predequant_k,
            hidden,
            inter,
            stream,
        )?);
        Ok(())
    }

    /// Install BF16 dense MLP weights. After this call, the forward paths
    /// dispatch to the BF16 GEMV/GEMM kernels instead of w4a16. The
    /// caller must ensure the BF16 kernels are loaded (see
    /// `dense_gemv_bf16_k` / `dense_gemm_bf16_k` checks). Spec-decode
    /// batched paths (`forward_k2`, `forward_k3`) are NOT supported on
    /// the BF16 path — Gemma-4 dense has no MTP so they're never called.
    pub fn set_bf16_weights(&mut self, gate: DenseWeight, up: DenseWeight, down: DenseWeight) {
        self.bf16_weights = Some(DenseFfnWeightsBf16 {
            gate_proj: gate,
            up_proj: up,
            down_proj: down,
        });
    }

    /// Whether this layer can execute batched `forward_kgamma` (quantized W4/W3
    /// weights; BF16 dense MLP weights use per-row GEMV/GEMM).
    pub fn can_forward_kgamma(&self) -> bool {
        self.bf16_weights.is_none()
    }

    /// Whether `forward_kgamma(n)` will take the wide (`32 < n <= 256`)
    /// transposed `w4a16_gemm_t_m128` tensor-core path rather than falling
    /// back to legacy `ops::w4a16_gemm` or breaking default exactness.
    pub fn wide_kgamma_ready(&self, n: u32) -> bool {
        self.bf16_weights.is_none()
            && ffn_tc_active_for_rows(n)
            && n as usize > crate::layers::TC_VERIFY_MAX_ROWS
            && n <= FFN_KGAMMA_WIDE_MAX_ROWS
            && kgamma_wide_active()
            && kgamma_m128_active()
            && self.w4a16_gemm_t_m128.0 != 0
            && (crate::layers::ffn_m16_transposed_enabled()
                || crate::layers::tc_nvfp4_m16_enabled())
            && self.has_transposed_ffn()
    }

    /// Whether the FFN activation-sparsity MEASUREMENT harness is wired up:
    /// the env gate is on AND the measure kernel symbol is present.
    fn sparsity_measure_active(&self) -> bool {
        crate::layers::measure_ffn_sparsity_enabled() && self.ffn_sparsity_measure_k.0 != 0
    }

    /// Ensure the per-layer sparsity-measurement counter buffers exist and are
    /// zeroed on first allocation. Returns the five device pointers. Allocated
    /// lazily on the first measured `forward` (never during graph capture —
    /// the measured path runs eager).
    fn ensure_sparsity_meas(
        &self,
        gpu: &dyn GpuBackend,
        inter: usize,
    ) -> Result<(DevicePtr, DevicePtr, DevicePtr, DevicePtr, DevicePtr)> {
        let n_thresh = ops::SPARSITY_NUM_THRESH;
        let mut slot = self.sparsity_meas.lock().unwrap();
        if slot.is_none() {
            let hist_gateup = gpu.alloc(n_thresh * 4)?;
            let count_gateup = gpu.alloc(2 * 4)?;
            let hist_down = gpu.alloc(n_thresh * 4)?;
            let count_down = gpu.alloc(2 * 4)?;
            let meas_silu = gpu.alloc(inter * 2)?; // BF16 [1, intermediate]
            // Zero the accumulators up front (kernel uses atomicAdd).
            gpu.memset(hist_gateup, 0, n_thresh * 4)?;
            gpu.memset(count_gateup, 0, 2 * 4)?;
            gpu.memset(hist_down, 0, n_thresh * 4)?;
            gpu.memset(count_down, 0, 2 * 4)?;
            *slot = Some(SparsityMeas {
                hist_gateup,
                count_gateup,
                hist_down,
                count_down,
                meas_silu,
                steps: 0,
            });
        }
        let s = slot.as_ref().unwrap();
        Ok((
            s.hist_gateup,
            s.count_gateup,
            s.hist_down,
            s.count_down,
            s.meas_silu,
        ))
    }

    /// Observer: launch `ffn_sparsity_measure` on `input` at the given site.
    /// PURE READER — never mutates `input` or any token-stream buffer; writes
    /// only into the dedicated `hist`/`count` accumulators. Called from
    /// `forward` at the two FFN sites when the measurement gate is on.
    fn measure_sparsity_site(
        &self,
        ctx: &ForwardContext,
        input: DevicePtr,
        hist: DevicePtr,
        count: DevicePtr,
        k: u32,
        stream: u64,
    ) -> Result<()> {
        ops::ffn_sparsity_measure(
            ctx.gpu,
            self.ffn_sparsity_measure_k,
            input,
            hist,
            count,
            k,
            stream,
        )
    }

    /// Periodic D2H dump of the accumulated per-site sparsity histograms,
    /// averaged over all rows measured since process start. Emits a
    /// `tracing::info` line every `measure_ffn_sparsity_dump_every` measured
    /// forwards. Bumps the per-layer step counter each call. No-op when the
    /// dump cadence has not been reached.
    ///
    /// The reported fraction for threshold t at a site is
    /// `hist[t] / elements_seen` — the mean below-threshold activation
    /// fraction, i.e. the UPPER BOUND on the column-skip weight-byte savings
    /// for that projection at that threshold. The go/no-go number is the
    /// down-input (K=intermediate) fraction at the 1% threshold.
    fn maybe_dump_sparsity(
        &self,
        ctx: &ForwardContext,
        layer_tag: &str,
        stream: u64,
    ) -> Result<()> {
        let every = crate::layers::measure_ffn_sparsity_dump_every();
        let (hist_gateup, count_gateup, hist_down, count_down, steps) = {
            let mut slot = self.sparsity_meas.lock().unwrap();
            let Some(s) = slot.as_mut() else {
                return Ok(());
            };
            s.steps += 1;
            if !s.steps.is_multiple_of(every) {
                return Ok(());
            }
            (
                s.hist_gateup,
                s.count_gateup,
                s.hist_down,
                s.count_down,
                s.steps,
            )
        };

        // Sync so the accumulators reflect all launched measurements, then
        // read the histograms + counts back to the host.
        ctx.gpu.synchronize(stream)?;
        let n_thresh = ops::SPARSITY_NUM_THRESH;
        let read_hist = |hist: DevicePtr| -> Result<Vec<u32>> {
            let mut bytes = vec![0u8; n_thresh * 4];
            ctx.gpu.copy_d2h(hist, &mut bytes)?;
            Ok(bytes
                .chunks_exact(4)
                .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect())
        };
        let read_count = |count: DevicePtr| -> Result<(u64, u64)> {
            let mut bytes = vec![0u8; 2 * 4];
            ctx.gpu.copy_d2h(count, &mut bytes)?;
            let rows = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as u64;
            let elems = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]) as u64;
            Ok((rows, elems))
        };

        let fmt_site = |hist: &[u32], elems: u64| -> String {
            if elems == 0 {
                return "n/a".to_string();
            }
            ops::SPARSITY_TAU
                .iter()
                .zip(hist.iter())
                .map(|(tau, &cnt)| {
                    let frac = cnt as f64 / elems as f64;
                    format!("{:.1}%tau={:.1}%", tau * 100.0, frac * 100.0)
                })
                .collect::<Vec<_>>()
                .join(" ")
        };

        let hg = read_hist(hist_gateup)?;
        let hd = read_hist(hist_down)?;
        let (rows_g, elems_g) = read_count(count_gateup)?;
        let (rows_d, elems_d) = read_count(count_down)?;

        tracing::info!(
            "FFN_SPARSITY[{layer_tag}] steps={steps} \
             gateup_in(K=hidden rows={rows_g}): {} | \
             down_in(K=inter rows={rows_d}): {}",
            fmt_site(&hg, elems_g),
            fmt_site(&hd, elems_d),
        );
        Ok(())
    }

    /// Whether the column-sparse self-spec DRAFT FFN path is wired up: both
    /// kernel symbols present. The env gate (`ATLAS_SELF_SPEC_SPARSE`) is
    /// checked by the caller (`step_self_spec`) so this only reports capability.
    pub fn has_sparse_draft(&self) -> bool {
        self.ffn_build_keep_chunks_k.0 != 0 && self.w4a16_gemv_sparse_cols_k.0 != 0
    }

    /// Ensure the per-layer sparse-draft scratch (`keep_idx` + `keep_len`)
    /// exists with capacity for `k` (the largest K this layer will sparsify —
    /// the down input K=intermediate). Allocated lazily on the first sparse
    /// draft forward (eager path, no graph capture).
    fn ensure_sparse_draft_scratch(
        &self,
        gpu: &dyn GpuBackend,
        k: usize,
    ) -> Result<(DevicePtr, DevicePtr)> {
        let need_chunks = k / 8;
        let mut slot = self.sparse_draft_scratch.lock().unwrap();
        let realloc = match slot.as_ref() {
            Some(s) => s.cap_chunks < need_chunks,
            None => true,
        };
        if realloc {
            if slot.is_some() {
                gpu.synchronize_device()?;
            }
            if let Some(prev) = slot.take() {
                let _ = gpu.free(prev.keep_idx);
                let _ = gpu.free(prev.keep_len);
            }
            let keep_idx = gpu.alloc(need_chunks * 4)?; // u32 per chunk
            let keep_len = match gpu.alloc(4) {
                Ok(ptr) => ptr,
                Err(err) => {
                    let _ = gpu.free(keep_idx);
                    return Err(err);
                }
            }; // single u32
            *slot = Some(SparseDraftScratch {
                keep_idx,
                keep_len,
                cap_chunks: need_chunks,
            });
        }
        let s = slot.as_ref().unwrap();
        Ok((s.keep_idx, s.keep_len))
    }

    /// SPARSE self-spec DRAFT single-token FFN forward.
    ///
    /// Same gate/up GEMV shape as `forward` (gate/up input is the dense
    /// residual stream — low activation sparsity, per the TEAL analysis, so
    /// it stays a dense dual GEMV), then swaps the down_proj GEMV for the
    /// column-sparse path: `ffn_build_keep_chunks` thresholds the silu(gate)*up
    /// activation into a surviving-chunk list, then `w4a16_gemv_sparse_cols`
    /// reads only those weight columns. APPROXIMATE by design — the dense
    /// verify is the lossless oracle, so this only proposes.
    ///
    /// `thresh_frac` is the keep threshold as a fraction of per-row max-abs
    /// (e.g. 0.01 for 1%). Falls back to the exact `forward` dense path when
    /// the sparse kernels are missing (`has_sparse_draft` false), so callers
    /// can always invoke it safely.
    ///
    /// EAGER only (the self-spec draft never captures a CUDA graph): the
    /// `keep_len` scalar is read back D2H before the sparse GEMV launch.
    pub fn forward_draft_sparse(
        &self,
        input: DevicePtr,
        ctx: &ForwardContext,
        thresh_frac: f32,
        stream: u64,
    ) -> Result<DevicePtr> {
        // Capability + BF16-weight guard: the sparse kernels operate on the
        // NVFP4 `QuantizedWeight` layout only. Fall back to the exact dense
        // forward when sparse kernels are missing or BF16 weights are active.
        if !self.has_sparse_draft() || self.bf16_weights.is_some() {
            return self.forward(input, ctx, stream);
        }

        let h = ctx.config.hidden_size as u32;
        let inter = ctx.config.intermediate_size as u32;
        let gate_out = ctx.buffers.expert_gate_out();
        let up_out = ctx.buffers.expert_up_out();

        // gate/up stay DENSE (residual-stream input, low sparsity).
        ops::w4a16_decode_gemv_dual(
            ctx.gpu,
            self.w4a16_gemv_dual,
            self.w4a16_gemv_dual_sw,
            self.gemv_sw,
            input,
            &self.weights.gate_proj,
            gate_out,
            &self.weights.up_proj,
            up_out,
            inter,
            h,
            stream,
        )?;

        // silu(gate)*up → gate_out (the down-proj input we sparsify).
        ops::silu_mul(
            ctx.gpu,
            self.act_mul,
            gate_out,
            up_out,
            gate_out,
            inter,
            stream,
        )?;

        // Threshold the down-input into a surviving k8-chunk list.
        let (keep_idx, keep_len) = self.ensure_sparse_draft_scratch(ctx.gpu, inter as usize)?;
        ops::ffn_build_keep_chunks(
            ctx.gpu,
            self.ffn_build_keep_chunks_k,
            gate_out,
            thresh_frac,
            keep_idx,
            keep_len,
            inter,
            stream,
        )?;

        // Read back keep_len (scalar-by-value kernel arg). Sync is acceptable
        // on the eager draft path; it also bounds the sparse GEMV's loop.
        ctx.gpu.synchronize(stream)?;
        let mut kl_bytes = [0u8; 4];
        ctx.gpu.copy_d2h(keep_len, &mut kl_bytes)?;
        let keep_len_val = u32::from_le_bytes(kl_bytes);

        // Column-sparse down_proj GEMV over the surviving chunks only.
        let output = ctx.buffers.moe_output();
        ops::w4a16_gemv_sparse_cols(
            ctx.gpu,
            self.w4a16_gemv_sparse_cols_k,
            gate_out,
            &self.weights.down_proj,
            keep_idx,
            keep_len_val,
            output,
            h,
            inter,
            stream,
        )?;
        Ok(output)
    }

    /// Single-token decode: 2-3 kernel launches depending on activation.
    /// SiLU: dual GEMV + SiLU-fused down GEMV (2 launches).
    /// GELU: dual GEMV + gelu_mul + down GEMV (3 launches, no fused GELU down kernel).
    pub fn forward(
        &self,
        input: DevicePtr,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<DevicePtr> {
        let h = ctx.config.hidden_size as u32;
        let inter = ctx.config.intermediate_size as u32;

        let gate_out = ctx.buffers.expert_gate_out();
        let up_out = ctx.buffers.expert_up_out();

        // BF16 dispatch: per-projection GEMV via `dense_gemv_bf16`. We
        // don't have a fused dual-BF16-GEMV kernel today; two sequential
        // launches are still BF16-precision-correct and only ~10% slower
        // than the fused w4a16 path on Gemma-4-31B (the cost is dominated
        // by the bigger BF16 weight reads, not launch overhead).
        if let Some(ref bf16w) = self.bf16_weights {
            ops::dense_gemv(
                ctx.gpu,
                self.dense_gemv_bf16_k,
                input,
                &bf16w.gate_proj,
                gate_out,
                inter,
                h,
                stream,
            )?;
            ops::dense_gemv(
                ctx.gpu,
                self.dense_gemv_bf16_k,
                input,
                &bf16w.up_proj,
                up_out,
                inter,
                h,
                stream,
            )?;
            ops::silu_mul(
                ctx.gpu,
                self.act_mul,
                gate_out,
                up_out,
                gate_out,
                inter,
                stream,
            )?;
            let output = ctx.buffers.moe_output();
            ops::dense_gemv(
                ctx.gpu,
                self.dense_gemv_bf16_k,
                gate_out,
                &bf16w.down_proj,
                output,
                h,
                inter,
                stream,
            )?;
            return Ok(output);
        }

        // W3 (3-bit) single-token decode GEMV path when W3 weights and kernels
        // are installed on this SiLU layer (`ATLAS_FFN_W3_LAYERS`).
        if self.has_w3_gemv()
            && let Some(ref w3) = self.w3_weights
        {
            crate::kprof!(ctx.gpu, stream, "ffn_gate_up_dual_w3_m1", {
                ops::w3a16_gemv_dual(
                    ctx.gpu,
                    self.w3a16_gemv_dual_k,
                    input,
                    &w3.gate_proj,
                    gate_out,
                    &w3.up_proj,
                    up_out,
                    inter,
                    h,
                    stream,
                )?;
                anyhow::Result::<()>::Ok(())
            })?;
            if self.sparsity_measure_active() && !ctx.graph_capture {
                let (hist_g, count_g, hist_d, count_d, meas_silu) =
                    self.ensure_sparsity_meas(ctx.gpu, inter as usize)?;
                self.measure_sparsity_site(ctx, input, hist_g, count_g, h, stream)?;
                ops::silu_mul(
                    ctx.gpu,
                    self.act_mul,
                    gate_out,
                    up_out,
                    meas_silu,
                    inter,
                    stream,
                )?;
                self.measure_sparsity_site(ctx, meas_silu, hist_d, count_d, inter, stream)?;
                self.maybe_dump_sparsity(ctx, "dense_ffn", stream)?;
            }
            let output = ctx.buffers.moe_output();
            crate::kprof!(ctx.gpu, stream, "ffn_down_silu_w3_m1", {
                ops::w3a16_gemv_silu_input(
                    ctx.gpu,
                    self.w3a16_gemv_silu_input_k,
                    gate_out,
                    up_out,
                    &w3.down_proj,
                    output,
                    h,
                    inter,
                    stream,
                )?;
                anyhow::Result::<()>::Ok(())
            })?;
            return Ok(output);
        }

        // Fused gate_proj + up_proj: [1, H] → [1, inter] × 2
        crate::kprof!(ctx.gpu, stream, "ffn_gate_up_dual_m1", {
            ops::w4a16_decode_gemv_dual(
                ctx.gpu,
                self.w4a16_gemv_dual,
                self.w4a16_gemv_dual_sw,
                self.gemv_sw,
                input,
                &self.weights.gate_proj,
                gate_out,
                &self.weights.up_proj,
                up_out,
                inter,
                h,
                stream,
            )?;
            anyhow::Result::<()>::Ok(())
        })?;

        // ── FFN activation-sparsity MEASUREMENT (observer, default-off) ──
        // Runs ONLY when ATLAS_MEASURE_FFN_SPARSITY=1 and the kernel symbol is
        // present. Pure reader: measures `input` (gate/up in, K=hidden) and a
        // freshly-recomputed `silu(gate)*up` copy (down in, K=inter) into
        // dedicated counter buffers. It never touches `input`, `gate_out`,
        // `up_out`, or `output`, so the token stream stays byte-identical
        // whether the gate is on or off (deterministic greedy token parity preserved).
        //
        // Skipped under CUDA graph capture: the observer lazily `gpu.alloc`s its
        // counter buffers on first use, which is illegal mid-capture. The
        // measurement run is intended for eager decode (the operator sets the
        // gate for a measurement window); skipping graphed steps only omits a
        // subset of rows from the average and never perturbs the token stream.
        if self.sparsity_measure_active() && !ctx.graph_capture {
            let (hist_g, count_g, hist_d, count_d, meas_silu) =
                self.ensure_sparsity_meas(ctx.gpu, inter as usize)?;
            // Site 0: gate/up input (residual stream, K=hidden).
            self.measure_sparsity_site(ctx, input, hist_g, count_g, h, stream)?;
            // Site 1: down input = silu(gate)*up (K=intermediate). Recompute
            // into the observer's OWN scratch so gate_out/up_out (read by the
            // fused down GEMV below) are untouched.
            ops::silu_mul(
                ctx.gpu,
                self.act_mul,
                gate_out,
                up_out,
                meas_silu,
                inter,
                stream,
            )?;
            self.measure_sparsity_site(ctx, meas_silu, hist_d, count_d, inter, stream)?;
            self.maybe_dump_sparsity(ctx, "dense_ffn", stream)?;
        }

        let output = ctx.buffers.moe_output();
        match self.activation {
            FfnActivation::SiLU => {
                // Fused SiLU(gate)*up + down_proj: [1, inter] → [1, H]
                crate::kprof!(ctx.gpu, stream, "ffn_down_silu_m1", {
                    ops::w4a16_decode_gemv_silu_input(
                        ctx.gpu,
                        self.w4a16_gemv_silu_input,
                        self.w4a16_gemv_silu_input_sw,
                        self.gemv_sw,
                        gate_out,
                        up_out,
                        &self.weights.down_proj,
                        output,
                        h,
                        inter,
                        stream,
                    )?;
                    anyhow::Result::<()>::Ok(())
                })?;
            }
            FfnActivation::GeLU => {
                // GELU(gate)*up → gate_out, then down_proj GEMV
                crate::kprof!(ctx.gpu, stream, "ffn_gelu_mul_m1", {
                    ops::silu_mul(
                        ctx.gpu,
                        self.act_mul,
                        gate_out,
                        up_out,
                        gate_out,
                        inter,
                        stream,
                    )?;
                    anyhow::Result::<()>::Ok(())
                })?;
                crate::kprof!(ctx.gpu, stream, "ffn_down_m1", {
                    ops::w4a16_decode_gemv(
                        ctx.gpu,
                        self.w4a16_gemv,
                        self.w4a16_gemv_sw,
                        self.gemv_sw,
                        gate_out,
                        &self.weights.down_proj,
                        output,
                        h,
                        inter,
                        stream,
                    )?;
                    anyhow::Result::<()>::Ok(())
                })?;
            }
        }

        Ok(output)
    }

    /// K=2 speculative: batched GEMV for 2 tokens.
    /// 3 launches: dual batch2 (gate+up) + silu_mul + batch2 (down).
    pub fn forward_k2(&self, input: DevicePtr, ctx: &ForwardContext, stream: u64) -> Result<()> {
        let h = ctx.config.hidden_size as u32;
        let inter = ctx.config.intermediate_size as u32;

        let gate_out = ctx.buffers.expert_gate_out();
        let up_out = ctx.buffers.expert_up_out();

        // Fused gate+up for 2 tokens
        ops::w4a16_gemv_dual_batch2(
            ctx.gpu,
            self.w4a16_gemv_dual_batch2,
            input,
            &self.weights.gate_proj,
            gate_out,
            &self.weights.up_proj,
            up_out,
            inter,
            h,
            stream,
        )?;
        ops::silu_mul(
            ctx.gpu,
            self.act_mul,
            gate_out,
            up_out,
            gate_out,
            2 * inter,
            stream,
        )?;
        let output = ctx.buffers.moe_output();
        ops::w4a16_gemv_batch2(
            ctx.gpu,
            self.w4a16_gemv_batch2,
            gate_out,
            &self.weights.down_proj,
            output,
            h,
            inter,
            stream,
        )?;

        Ok(())
    }

    /// K=3 speculative: batched GEMV for 3 tokens.
    /// 3 launches: dual batch3 (gate+up) + silu_mul + batch3 (down).
    pub fn forward_k3(&self, input: DevicePtr, ctx: &ForwardContext, stream: u64) -> Result<()> {
        // Tensor-core M=3 fast path (ATLAS_TC_NVFP4_K3=1):
        //
        // Routes through `w4a16_gemm_t_m16_n64` (small-M, N_TILE=64) for
        // gate / up / down. Dispatches 3 GEMM launches (gate, up, down)
        // instead of the GEMV path's 3 launches (dual gate+up, silu, down)
        // but each GEMM runs on tensor cores via m16n8k32 e4m3 MMA.
        //
        // The first attempt (routing to `forward_kgamma(n=3)` which used
        // `w4a16_gemm_t_m16` with N_TILE=128) measured −23% mean tok/s
        // on Qwen3.6-27B because the 136-CTA grid under-saturated GB10's
        // 48 SMs (~2.8 CTAs/SM).
        // The N_TILE=64 variant fields 272 CTAs/projection (~5.7 CTAs/SM)
        // at half the per-CTA work — designed to keep the tensor-core
        // pipeline fed at M=3.
        //
        // Bounds: requires transposed weights + the n64 kernel symbol
        // loaded; falls through to the GEMV path otherwise.
        if crate::layers::tc_nvfp4_k3_enabled()
            && self.has_transposed_ffn()
            && self.w4a16_gemm_t_m16_n64.0 != 0
        {
            let h = ctx.config.hidden_size as u32;
            let inter = ctx.config.intermediate_size as u32;
            let gate_out_buf = ctx.buffers.expert_gate_out();
            let up_out_buf = ctx.buffers.expert_up_out();
            let gt = self.gate_proj_t.as_ref().unwrap();
            let ut = self.up_proj_t.as_ref().unwrap();
            let dt = self.down_proj_t.as_ref().unwrap();

            crate::kprof!(ctx.gpu, stream, "ffn_gate_up_dual_batch3", {
                ops::w4a16_gemm_n64_m16(
                    ctx.gpu,
                    self.w4a16_gemm_t_m16_n64,
                    input,
                    gt,
                    gate_out_buf,
                    3,
                    inter,
                    h,
                    stream,
                )?;
                ops::w4a16_gemm_n64_m16(
                    ctx.gpu,
                    self.w4a16_gemm_t_m16_n64,
                    input,
                    ut,
                    up_out_buf,
                    3,
                    inter,
                    h,
                    stream,
                )?;
                anyhow::Result::<()>::Ok(())
            })?;
            crate::kprof!(ctx.gpu, stream, "ffn_silu_mul", {
                ops::silu_mul(
                    ctx.gpu,
                    self.act_mul,
                    gate_out_buf,
                    up_out_buf,
                    gate_out_buf,
                    3 * inter,
                    stream,
                )?;
                anyhow::Result::<()>::Ok(())
            })?;
            let output = ctx.buffers.moe_output();
            crate::kprof!(ctx.gpu, stream, "ffn_down_batch3", {
                ops::w4a16_gemm_n64_m16(
                    ctx.gpu,
                    self.w4a16_gemm_t_m16_n64,
                    gate_out_buf,
                    dt,
                    output,
                    3,
                    h,
                    inter,
                    stream,
                )?;
                anyhow::Result::<()>::Ok(())
            })?;
            return Ok(());
        }

        let h = ctx.config.hidden_size as u32;
        let inter = ctx.config.intermediate_size as u32;

        let gate_out = ctx.buffers.expert_gate_out();
        let up_out = ctx.buffers.expert_up_out();

        // Fused gate+up for 3 tokens. Tuned variant (gated by
        // `ATLAS_FFN_DUAL_TUNED=1`) fuses both projections into the SAME CTA
        // so the 3-token activation vector is loaded once per CTA instead of
        // twice. Falls back to the baseline kernel when the env var is unset
        // OR the tuned kernel symbol was not present in the loaded cache.
        let use_tuned = ffn_dual_tuned_enabled() && self.w4a16_gemv_dual_batch3_tuned.0 != 0;
        let dual_kernel = if use_tuned {
            self.w4a16_gemv_dual_batch3_tuned
        } else {
            self.w4a16_gemv_dual_batch3
        };
        crate::kprof!(ctx.gpu, stream, "ffn_gate_up_dual_batch3", {
            if use_tuned {
                ops::w4a16_gemv_dual_batch3_tuned(
                    ctx.gpu,
                    dual_kernel,
                    input,
                    &self.weights.gate_proj,
                    gate_out,
                    &self.weights.up_proj,
                    up_out,
                    inter,
                    h,
                    stream,
                )?;
            } else {
                ops::w4a16_gemv_dual_batch3(
                    ctx.gpu,
                    dual_kernel,
                    input,
                    &self.weights.gate_proj,
                    gate_out,
                    &self.weights.up_proj,
                    up_out,
                    inter,
                    h,
                    stream,
                )?;
            }
            anyhow::Result::<()>::Ok(())
        })?;
        crate::kprof!(ctx.gpu, stream, "ffn_silu_mul", {
            ops::silu_mul(
                ctx.gpu,
                self.act_mul,
                gate_out,
                up_out,
                gate_out,
                3 * inter,
                stream,
            )?;
            anyhow::Result::<()>::Ok(())
        })?;
        let output = ctx.buffers.moe_output();
        crate::kprof!(ctx.gpu, stream, "ffn_down_batch3", {
            ops::w4a16_gemv_batch3(
                ctx.gpu,
                self.w4a16_gemv_batch3,
                gate_out,
                &self.weights.down_proj,
                output,
                h,
                inter,
                stream,
            )?;
            anyhow::Result::<()>::Ok(())
        })?;

        Ok(())
    }

    fn measure_exact_kgamma_inputs(
        &self,
        input: DevicePtr,
        gate_out: DevicePtr,
        up_out: DevicePtr,
        ctx: &ForwardContext,
        hidden: u32,
        intermediate: u32,
        stream: u64,
    ) -> Result<()> {
        if !self.sparsity_measure_active() || ctx.graph_capture {
            return Ok(());
        }

        let (hist_g, count_g, hist_d, count_d, meas_silu) =
            self.ensure_sparsity_meas(ctx.gpu, intermediate as usize)?;
        self.measure_sparsity_site(ctx, input, hist_g, count_g, hidden, stream)?;
        ops::silu_mul(
            ctx.gpu,
            self.act_mul,
            gate_out,
            up_out,
            meas_silu,
            intermediate,
            stream,
        )?;
        self.measure_sparsity_site(ctx, meas_silu, hist_d, count_d, intermediate, stream)?;
        self.maybe_dump_sparsity(ctx, "dense_ffn_kgamma_exact", stream)
    }

    fn forward_kgamma_exact_split_m8_down(
        &self,
        ctx: &ForwardContext,
        hidden: u32,
        intermediate: u32,
        stream: u64,
    ) -> Result<()> {
        const GROUP_ROWS: u32 = 8;
        let preactivation = ctx.buffers.ssm_conv_out_f32();
        let output = ctx.buffers.moe_output();

        for first_row in [0, GROUP_ROWS] {
            let offsets = exact_ffn_row_offsets(first_row, hidden, intermediate)
                .ok_or_else(|| anyhow::anyhow!("exact FFN M8 down row-offset overflow"))?;
            ops::w4a16_gemv_f32_input_exact(
                ctx.gpu,
                self.exact_ffn_kernels,
                preactivation.offset(offsets.preactivation),
                &self.weights.down_proj,
                output.offset(offsets.output),
                GROUP_ROWS,
                hidden,
                intermediate,
                stream,
            )?;
        }
        Ok(())
    }

    fn forward_kgamma_exact_lowreg_gate_up_m16(
        &self,
        input: DevicePtr,
        ctx: &ForwardContext,
        hidden: u32,
        intermediate: u32,
        stream: u64,
    ) -> Result<()> {
        const ROWS: u32 = 16;
        let gate_out = ctx.buffers.expert_gate_out();
        let up_out = ctx.buffers.expert_up_out();
        let preactivation = ctx.buffers.ssm_conv_out_f32();

        ops::w4a16_gemv_gate_exact_m16_lowreg(
            ctx.gpu,
            self.exact_ffn_lowreg_m16,
            input,
            &self.weights.gate_proj,
            gate_out,
            ROWS,
            intermediate,
            hidden,
            stream,
        )?;
        ops::w4a16_gemv_up_exact_m16_lowreg(
            ctx.gpu,
            self.exact_ffn_lowreg_m16,
            input,
            &self.weights.up_proj,
            up_out,
            ROWS,
            intermediate,
            hidden,
            stream,
        )?;
        self.measure_exact_kgamma_inputs(
            input,
            gate_out,
            up_out,
            ctx,
            hidden,
            intermediate,
            stream,
        )?;
        ops::w4a16_gate_up_materialize_f32_m16(
            ctx.gpu,
            self.exact_ffn_lowreg_m16,
            gate_out,
            up_out,
            preactivation,
            ROWS,
            intermediate,
            stream,
        )?;
        self.forward_kgamma_exact_split_m8_down(ctx, hidden, intermediate, stream)
    }

    fn forward_kgamma_exact_split_m8(
        &self,
        input: DevicePtr,
        ctx: &ForwardContext,
        hidden: u32,
        intermediate: u32,
        stream: u64,
    ) -> Result<()> {
        const GROUP_ROWS: u32 = 8;
        if ops::exact_ffn_lowreg_m16_route(
            2 * GROUP_ROWS,
            exact_ffn_lowreg_gate_up_m16_enabled(),
            self.exact_ffn_lowreg_m16,
        ) == Some(ops::ExactFfnLowregM16Route::Lowreg)
        {
            return self.forward_kgamma_exact_lowreg_gate_up_m16(
                input,
                ctx,
                hidden,
                intermediate,
                stream,
            );
        }
        let gate_out = ctx.buffers.expert_gate_out();
        let up_out = ctx.buffers.expert_up_out();
        let preactivation = ctx.buffers.ssm_conv_out_f32();
        let output = ctx.buffers.moe_output();

        for first_row in [0, GROUP_ROWS] {
            let offsets = exact_ffn_row_offsets(first_row, hidden, intermediate)
                .ok_or_else(|| anyhow::anyhow!("exact FFN M8 row-offset overflow"))?;
            let input_group = input.offset(offsets.input);
            let gate_group = gate_out.offset(offsets.gate_up);
            let up_group = up_out.offset(offsets.gate_up);
            let preactivation_group = preactivation.offset(offsets.preactivation);
            let output_group = output.offset(offsets.output);

            ops::w4a16_gemv_dual_exact(
                ctx.gpu,
                self.exact_ffn_kernels,
                input_group,
                &self.weights.gate_proj,
                gate_group,
                &self.weights.up_proj,
                up_group,
                GROUP_ROWS,
                intermediate,
                hidden,
                stream,
            )?;
            self.measure_exact_kgamma_inputs(
                input_group,
                gate_group,
                up_group,
                ctx,
                hidden,
                intermediate,
                stream,
            )?;
            ops::w4a16_gemv_dual_silu_f32_exact(
                ctx.gpu,
                self.exact_ffn_kernels,
                gate_group,
                up_group,
                preactivation_group,
                GROUP_ROWS,
                intermediate,
                stream,
            )?;
            ops::w4a16_gemv_f32_input_exact(
                ctx.gpu,
                self.exact_ffn_kernels,
                preactivation_group,
                &self.weights.down_proj,
                output_group,
                GROUP_ROWS,
                hidden,
                intermediate,
                stream,
            )?;
        }
        Ok(())
    }

    fn forward_kgamma_exact(
        &self,
        input: DevicePtr,
        ctx: &ForwardContext,
        rows: u32,
        stream: u64,
    ) -> Result<()> {
        let hidden = ctx.config.hidden_size as u32;
        let intermediate = ctx.config.intermediate_size as u32;
        let gate_out = ctx.buffers.expert_gate_out();
        let up_out = ctx.buffers.expert_up_out();

        if exact_ffn_physical_route(
            rows,
            intermediate,
            exact_ffn_split_m8_enabled(),
            self.exact_ffn_kernels
                .tier_is_complete(ops::ExactFfnTier::M8),
            self.exact_ffn_kernels.materialized_m8_is_complete(),
            ctx.buffers.sizes().ssm_conv_out_f32,
        ) == ExactFfnPhysicalRoute::SplitM8x2
        {
            return self.forward_kgamma_exact_split_m8(input, ctx, hidden, intermediate, stream);
        }

        // Attention/SSM has completed before FFN on this stream, so its FP32
        // recurrence workspace is dead until the next layer. Reuse it only
        // when the arena's authoritative byte capacity covers every row.
        let route = exact_ffn_materialized_route(
            rows,
            intermediate,
            ExactFfnMaterializedAvailability {
                m8_dual_silu: self.exact_ffn_kernels.dual_silu_f32_m8_is_present(),
                m8_f32_down: self.exact_ffn_kernels.f32_input_m8_is_present(),
                m17_dual_silu: self.exact_ffn_kernels.dual_silu_f32_m17_is_present(),
                m17_f32_down: self.exact_ffn_kernels.f32_input_m17_is_present(),
                // The fused stage does not expose gate/up tensors required by
                // sparsity diagnostics; retain the split exact route then.
                m17_fused_dual_silu: self.exact_ffn_kernels.fused_materialized_m17_is_present()
                    && (!self.sparsity_measure_active() || ctx.graph_capture),
            },
            ctx.buffers.sizes().ssm_conv_out_f32,
        );

        let preactivation_f32 = ctx.buffers.ssm_conv_out_f32();
        match route {
            ExactFfnMaterializedRoute::FusedM17 => {
                ops::w4a16_gemv_dual_materialize_f32_exact_m17(
                    ctx.gpu,
                    self.exact_ffn_kernels,
                    input,
                    &self.weights.gate_proj,
                    &self.weights.up_proj,
                    preactivation_f32,
                    rows,
                    intermediate,
                    hidden,
                    stream,
                )?;
            }
            ExactFfnMaterializedRoute::Split | ExactFfnMaterializedRoute::Inline => {
                ops::w4a16_gemv_dual_exact(
                    ctx.gpu,
                    self.exact_ffn_kernels,
                    input,
                    &self.weights.gate_proj,
                    gate_out,
                    &self.weights.up_proj,
                    up_out,
                    rows,
                    intermediate,
                    hidden,
                    stream,
                )?;
                self.measure_exact_kgamma_inputs(
                    input,
                    gate_out,
                    up_out,
                    ctx,
                    hidden,
                    intermediate,
                    stream,
                )?;
            }
        }

        match route {
            ExactFfnMaterializedRoute::Split => ops::w4a16_gemv_dual_silu_f32_exact(
                ctx.gpu,
                self.exact_ffn_kernels,
                gate_out,
                up_out,
                preactivation_f32,
                rows,
                intermediate,
                stream,
            )?,
            ExactFfnMaterializedRoute::Inline => {
                return ops::w4a16_gemv_silu_input_exact(
                    ctx.gpu,
                    self.exact_ffn_kernels,
                    gate_out,
                    up_out,
                    &self.weights.down_proj,
                    ctx.buffers.moe_output(),
                    rows,
                    hidden,
                    intermediate,
                    stream,
                );
            }
            ExactFfnMaterializedRoute::FusedM17 => {}
        }
        ops::w4a16_gemv_f32_input_exact(
            ctx.gpu,
            self.exact_ffn_kernels,
            preactivation_f32,
            &self.weights.down_proj,
            ctx.buffers.moe_output(),
            rows,
            hidden,
            intermediate,
            stream,
        )
    }

    fn forward_kgamma_k1_rows(
        &self,
        input: DevicePtr,
        ctx: &ForwardContext,
        rows: u32,
        stream: u64,
    ) -> Result<()> {
        let hidden = ctx.config.hidden_size as u32;
        let intermediate = ctx.config.intermediate_size as u32;
        let gate_out = ctx.buffers.expert_gate_out();
        let up_out = ctx.buffers.expert_up_out();
        let output = ctx.buffers.moe_output();

        for row in 0..rows as usize {
            let input_row = input.offset(row * hidden as usize * 2);
            let gate_row = gate_out.offset(row * intermediate as usize * 2);
            let up_row = up_out.offset(row * intermediate as usize * 2);
            let output_row = output.offset(row * hidden as usize * 2);
            ops::w4a16_decode_gemv_dual(
                ctx.gpu,
                self.w4a16_gemv_dual,
                self.w4a16_gemv_dual_sw,
                self.gemv_sw,
                input_row,
                &self.weights.gate_proj,
                gate_row,
                &self.weights.up_proj,
                up_row,
                intermediate,
                hidden,
                stream,
            )?;
            ops::w4a16_decode_gemv_silu_input(
                ctx.gpu,
                self.w4a16_gemv_silu_input,
                self.w4a16_gemv_silu_input_sw,
                self.gemv_sw,
                gate_row,
                up_row,
                &self.weights.down_proj,
                output_row,
                hidden,
                intermediate,
                stream,
            )?;
        }

        self.measure_exact_kgamma_inputs(input, gate_out, up_out, ctx, hidden, intermediate, stream)
    }

    /// K=γ verify batch (DFlash γ ≥ 16, typical n=17 with γ=16).
    ///
    /// Replaces the per-token loop that calls `forward()` n times (n=γ+1).
    /// Each `forward()` call runs 2 M=1 GEMVs that re-read 134 MB of NVFP4
    /// FFN weights from LPDDR5X; per-step this costs `64 layers × 17 tokens
    /// × 134 MB = 145 GB` of redundant weight bandwidth.
    ///
    /// This path issues 3 `w4a16_gemm` calls with M=n. The standard
    /// (non-transposed) NVFP4 GEMM has M_TILE=64; M=17 fits inside a single
    /// CTA-row with some accumulator waste but loads the weight tile once
    /// per layer instead of n times. Expected: ~64 × 134 MB = 8.6 GB per
    /// step, an ~18× reduction in FFN-loop bandwidth.
    ///
    /// Sequence: gate_proj (GEMM M=n) → up_proj (GEMM M=n) → silu_mul
    /// (n × intermediate) → down_proj (GEMM M=n).
    /// W4+SiLU rows 2..=32 instead use exact dynamic-M dual and fused-down
    /// GEMVs that preserve K1 accumulation order. If either selected-tier
    /// symbol is missing, every row runs through the ordinary K1 kernels;
    /// BF16, W3, GELU, and rows above 32 retain their existing routes.
    ///
    /// Reuses `ctx.buffers.expert_gate_out` / `expert_up_out` /
    /// `moe_output`, which are sized for `max_batch_tokens × intermediate`
    /// (always ≥ n, see `buffers/sizes.rs`).
    ///
    /// Output is written to `ctx.buffers.moe_output()`; callers downstream
    /// already consume from there (see `ms_phase_ffn` for the n==3 branch
    /// which is the contract this matches at higher n).
    ///
    /// When `ATLAS_FFN_M16_TRANSPOSED` is active (default ON unless `=0`) and
    /// the loader installed transposed (`nvfp4_t`) FFN weights via
    /// `set_transposed_weights`, this path dispatches gate/up/down through
    /// `w4a16_gemm_t_m16` (M_TILE=16, near-zero MMA accumulator waste at
    /// M ≤ 32). Otherwise the
    /// non-transposed `w4a16_gemm` (M_TILE=64) is used as the fallback —
    /// layout-compatible with the standard HuggingFace `[N, K/2]` weights
    /// but discards ~73% of accumulator writes at M=17. The transposed
    /// fast path is the K=γ verify analogue of the existing SSM `qkvz` /
    /// `out_proj` and DFlash drafter routings.
    /// BF16-weight fallback is not supported on this path (Gemma-4 dense
    /// has no MTP / DFlash, so n is always 1 there).
    pub fn forward_kgamma(
        &self,
        input: DevicePtr,
        ctx: &ForwardContext,
        n: u32,
        stream: u64,
    ) -> Result<()> {
        debug_assert!(
            n > 1 || self.decode_tc_parity_route(),
            "forward_kgamma is for batched verify; use forward() at n=1"
        );
        let h = ctx.config.hidden_size as u32;
        let inter = ctx.config.intermediate_size as u32;

        let gate_out = ctx.buffers.expert_gate_out();
        let up_out = ctx.buffers.expert_up_out();

        let exact_w4_silu_eligible = self.exact_kgamma_internal_eligible(n);
        let exact_tier_present = matches!(
            self.exact_ffn_kernels.route_for_rows(n),
            Some(ops::ExactFfnRoute::Exact(_))
        );
        match exact_ffn_dispatch(n, exact_w4_silu_eligible, exact_tier_present) {
            ExactFfnDispatch::Batched => {
                return self.forward_kgamma_exact(input, ctx, n, stream);
            }
            ExactFfnDispatch::PerRowK1 => {
                return self.forward_kgamma_k1_rows(input, ctx, n, stream);
            }
            ExactFfnDispatch::Existing => {}
        }

        if w3_kgamma_applicable(
            n,
            self.activation == FfnActivation::SiLU,
            self.has_w3_gemm(),
        ) {
            let w3 = self
                .w3_weights_t
                .as_ref()
                .context("W3 K-gamma route selected without transposed weights")?;
            crate::kprof!(ctx.gpu, stream, "ffn_gate_w3_kgamma", {
                ops::w3a16_gemm_n64_m32(
                    ctx.gpu,
                    self.w3a16_gemm_t_m32_n64_k,
                    input,
                    &w3.gate_proj,
                    gate_out,
                    n,
                    inter,
                    h,
                    stream,
                )?;
                anyhow::Result::<()>::Ok(())
            })?;
            crate::kprof!(ctx.gpu, stream, "ffn_up_w3_kgamma", {
                ops::w3a16_gemm_n64_m32(
                    ctx.gpu,
                    self.w3a16_gemm_t_m32_n64_k,
                    input,
                    &w3.up_proj,
                    up_out,
                    n,
                    inter,
                    h,
                    stream,
                )?;
                anyhow::Result::<()>::Ok(())
            })?;
            crate::kprof!(ctx.gpu, stream, "ffn_silu_mul_w3_kgamma", {
                ops::silu_mul(
                    ctx.gpu,
                    self.act_mul,
                    gate_out,
                    up_out,
                    gate_out,
                    n * inter,
                    stream,
                )?;
                anyhow::Result::<()>::Ok(())
            })?;
            let output = ctx.buffers.moe_output();
            crate::kprof!(ctx.gpu, stream, "ffn_down_w3_kgamma", {
                ops::w3a16_gemm_n64_m32(
                    ctx.gpu,
                    self.w3a16_gemm_t_m32_n64_k,
                    gate_out,
                    &w3.down_proj,
                    output,
                    n,
                    h,
                    inter,
                    stream,
                )?;
                anyhow::Result::<()>::Ok(())
            })?;
            static LOGGED: std::sync::Once = std::sync::Once::new();
            LOGGED.call_once(|| {
                tracing::info!(
                    "ENGAGED W3 FFN K-gamma: approximate ghost/quality-gated target route"
                );
            });
            return Ok(());
        }

        // Route through M_TILE=16 + transposed weights when:
        //   - ATLAS_FFN_M16_TRANSPOSED=1 OR ATLAS_TC_NVFP4_M16=1
        //   - the loader installed transposed copies of all 3 FFN projections
        //   - the `w4a16_gemm_t_m16` kernel symbol is present (try_kernel)
        //   - n ≤ 32 (the kernel's intended small-M window)
        // The combined gate matches the SSM `qkvz` / drafter pattern.
        // WIDE window (SASS audit 2026-07-08): at c>=2 batched verify,
        // M = 17c exceeds 32 and this whole transposed family used to
        // disengage — silent fallback to legacy `w4a16_gemm` (no cp.async,
        // scalar LDG.U8, ~4x sector overfetch, issue-capped ~47% of DRAM
        // BW): the measured cause of the concurrency ceiling. Route
        // 32 < n <= 256 through `w4a16_gemm_t_m128` (y-tiled; at M=136 two
        // weight reads still beat legacy's 3 sweeps x overfetch). Requires
        // the m128 kernel — the m16 kernel's small-M window is NOT widened.
        let kgamma_m128_on = kgamma_m128_active();
        let kgamma_wide_on = kgamma_wide_active();
        let wide_m128 = n as usize > crate::layers::TC_VERIFY_MAX_ROWS
            && n <= FFN_KGAMMA_WIDE_MAX_ROWS
            && kgamma_wide_on
            && kgamma_m128_on
            && self.w4a16_gemm_t_m128.0 != 0
            && (crate::layers::ffn_m16_transposed_enabled()
                || crate::layers::tc_nvfp4_m16_enabled())
            && self.has_transposed_ffn();
        // On wide discrete Blackwell GPUs (`sm_count > 64`) for small/medium-M
        // (`n <= sm120_coalesced_max_m()`, up to 64 rows: `C=1..64` multi-seq
        // and `C=1..8` DFlash2 verify), prefer single-launch BF16xBF16 MMA
        // intra-CTA warp-split-K fused `gate+up+silu` + `down_proj` directly on
        // row-major NVFP4 weights (zero activation quantize, zero absmax/alpha
        // launches, zero transposed-weight dependency, full 188-SM 600W saturation).
        // Always skipped on GB10 (`sm_count <= 64`).
        let sm120_max_m = sm120_coalesced_max_m();
        if (n as usize) <= sm120_max_m
            && self.activation == FfnActivation::SiLU
            && !self.weights.gate_proj.is_null()
            && !self.weights.up_proj.is_null()
            && !self.weights.down_proj.is_null()
            && crate::kprof!(ctx.gpu, stream, "ffn_gateup_fused_kgamma", {
                try_coalesced_nvfp4_sm120_small_m_gateup_silu(
                    ctx.gpu,
                    input,
                    &self.weights.gate_proj,
                    &self.weights.up_proj,
                    gate_out,
                    n,
                    inter,
                    h,
                    sm120_max_m,
                    stream,
                )
            })?
        {
            let output = ctx.buffers.moe_output();
            let down_ok = crate::kprof!(ctx.gpu, stream, "ffn_down_kgamma", {
                try_coalesced_nvfp4_sm120_small_m(
                    ctx.gpu,
                    gate_out,
                    &self.weights.down_proj,
                    output,
                    n,
                    h,
                    inter,
                    sm120_max_m,
                    stream,
                )
            })?;
            if down_ok {
                return Ok(());
            }
            // Fallback if down_proj kernel symbol was absent (rare).
            ops::w4a16_gemm(
                ctx.gpu,
                self.w4a16_gemm,
                gate_out,
                &self.weights.down_proj,
                output,
                n,
                h,
                inter,
                stream,
            )?;
            return Ok(());
        }

        // Native cuBLASLt NVFP4 (`VEC16_UE4M3`) for wide M >= 32 (both eager
        // and inside captured multi-seq CUDA graphs once warmed on step 1) when
        // enabled (`ATLAS_FFN_KGAMMA_CUBLASLT=1`, auto-ON on `sm_count > 64`,
        // default-OFF on GB10) or when transposed `w4a16_gemm_t_m128` is
        // unavailable (e.g. SM103). Uses a per-(gpu, stream) decode scratch so
        // it never collides with prefill's stream-bound `e2m1_scratch`.
        #[cfg(feature = "cuda")]
        if n >= CUBLASLT_NVFP4_MIN_M
            && (kgamma_cublaslt_opt_in(ctx.gpu) || !wide_m128)
            && self.try_forward_cublaslt_nvfp4_kgamma(
                input, n, h, inter, gate_out, up_out, ctx, stream,
            )?
        {
            return Ok(());
        }
        let m16_path = (n as usize <= crate::layers::TC_VERIFY_MAX_ROWS
            && (crate::layers::ffn_m16_transposed_enabled()
                || crate::layers::tc_nvfp4_m16_enabled())
            && self.has_transposed_ffn())
            || wide_m128;
        // m128 upgrade of the m16 path: ONE M-tile at n ≤ 128 → single
        // weight read (m16 re-reads B per 16-row tile: 2× traffic at
        // n=17 on a memory-bound GEMM). See ffn_kgamma_m128_enabled.
        //
        // FAIL-CLOSED GUARD (2026-08-19): ATLAS_FFN_KGAMMA_M128=0 selects
        // the M_TILE=16-only route (`w4a16_gemm_t_m16`, 2 M-tile rows at
        // n=17). That route is NON-DETERMINISTIC in this binary — three
        // identical 1,500-token MinHeap requests returned three different
        // completions (513/1500/1500 tokens, three distinct SHAs). The
        // m128/m32 route is deterministic (verified: 3/3 identical,
        // 41.7 tok/s). Fail closed: when the transposed m16 family would
        // otherwise be the only route, force the deterministic m128/m32
        // path regardless of the flag and warn once.
        let m128_forced = m16_path && !kgamma_m128_on && self.w4a16_gemm_t_m128.0 != 0;
        if m128_forced {
            static FORCED: std::sync::Once = std::sync::Once::new();
            FORCED.call_once(|| {
                tracing::warn!(
                    "ATLAS_FFN_KGAMMA_M128=0 requests the M_TILE=16 verify FFN route, \
                     which is NON-DETERMINISTIC in this binary — forcing the \
                     deterministic m128/m32 route instead (fail-closed)."
                );
            });
        }
        let m128_path =
            (m16_path && kgamma_m128_on && self.w4a16_gemm_t_m128.0 != 0) || m128_forced;
        if crate::layers::ffn_kgamma_m128_enabled() && self.w4a16_gemm_t_m128.0 == 0 {
            static WARNED: std::sync::Once = std::sync::Once::new();
            crate::layers::warn_kernel_fallback(
                &WARNED,
                "ATLAS_FFN_KGAMMA_M128=1",
                "w4a16_gemm_t_m128",
                "the K=γ verify FFN stays on the M_TILE=64 path, and every knob \
                 downstream of m128_path (ATLAS_FFN_KGAMMA_M32, \
                 ATLAS_FFN_GATEUP_SPLITK, ATLAS_FFN_FUSED_GATEUP) is disabled with it",
            );
        }
        // m32_n64: single B read AND full SM occupancy — strictly better
        // than m128 at n ≤ 32 when the kernel symbol is present. The n<=32
        // bound keeps the WIDE window off m32/fused/split-K variants (their
        // M_TILE=32 shapes don't cover wide M).
        // ATLAS_FFN_KGAMMA_M32=0 opts out for bisection.
        let m32_path = m128_path
            && n as usize <= crate::layers::TC_VERIFY_MAX_ROWS
            && self.w4a16_gemm_t_m32_n64.0 != 0
            && kgamma_m32_allowed();

        // gate_proj GEMM: [n, H] → [n, inter]
        // Split-K [M=n, N=inter, K=h] when ATLAS_FFN_GATEUP_SPLITK is set —
        // slices K across gridDim.z into the shared FP32 workspace (lossless,
        // token-exact). Falls through to the single-slice m32_n64 path below.
        let gateup_splitk = crate::layers::ffn_gateup_splitk();
        let gateup_ws = if gateup_splitk > 0 {
            *self.splitk_workspace.lock().unwrap()
        } else {
            None
        };
        let gateup_splitk_ok = m32_path
            && gateup_splitk > 0
            && self.w4a16_gemm_t_m32_n64_splitk.0 != 0
            && self.reduce_splitk_k.0 != 0
            && gateup_ws.is_some();

        // FUSED gate+up+silu (ATLAS_FFN_FUSED_GATEUP=1 or auto on >64-SM GPUs):
        // one launch reads the shared [n,H] input once, streams both transposed
        // weights, and writes silu(gate)*up into `gate_out` (the same buffer
        // moe_silu_mul targets) — replacing the gate GEMM + up GEMM + silu_mul
        // below. Superseded by explicit ATLAS_FFN_GATEUP_SPLITK when active. Requires the m32 transposed path,
        // SiLU activation, and a fused kernel symbol; byte-exact with unfused.
        let fused_gateup_on = kgamma_fused_gateup_active();
        let fused_gateup = m32_path
            && !gateup_splitk_ok
            && self.activation == FfnActivation::SiLU
            && fused_gateup_on
            && self.w4a16_gemm_t_m32_n64_gateup_silu.0 != 0;
        if fused_gateup_on
            && crate::layers::ffn_fused_gateup_enabled()
            && self.w4a16_gemm_t_m32_n64_gateup_silu.0 == 0
        {
            static WARNED: std::sync::Once = std::sync::Once::new();
            crate::layers::warn_kernel_fallback(
                &WARNED,
                "ATLAS_FFN_FUSED_GATEUP=1",
                "w4a16_gemm_t_m32_n64_gateup_silu",
                "gate/up stay on two separate GEMMs plus a silu_mul launch",
            );
        }
        if matches!(ffn_tc_policy(), FfnTcPolicy::Auto)
            && atlas_core::device::active_sm_count() > FFN_KGAMMA_AUTO_TC_SM_THRESHOLD
            && (m32_path || wide_m128)
        {
            static SM120_TC_LOGGED: std::sync::Once = std::sync::Once::new();
            SM120_TC_LOGGED.call_once(|| {
                tracing::info!(
                    rows = n,
                    m32_path,
                    fused_gateup,
                    wide_m128,
                    graph_capture = ctx.graph_capture,
                    sm_count = atlas_core::device::active_sm_count(),
                    "FFN_KGAMMA_SM120_TC_ENGAGED"
                );
            });
        }
        if fused_gateup {
            // On wide discrete GPUs (`sm_count > 64`) at small `M <= 16`, prefer the
            // coalesced 4-way warp-split-K BF16xBF16 MMA fused gate+up+silu kernel
            // (`w4a16_gemm_coalesced_v2_n16_sk4_m16_gateup_silu`, 1,088 CTAs at
            // `N=17408`): reads A once, streams row-major 128-bit NVFP4 weights with
            // `L2::evict_first`, eliminates M_TILE=32 waste and FP8 activation clipping.
            // Always skipped on GB10 (`sm_count <= 64`).
            let coalesced_gateup_done = n <= 16
                && !self.weights.gate_proj.is_null()
                && !self.weights.up_proj.is_null()
                && crate::kprof!(ctx.gpu, stream, "ffn_gateup_fused_kgamma", {
                    try_coalesced_nvfp4_sm120_small_m_gateup_silu(
                        ctx.gpu,
                        input,
                        &self.weights.gate_proj,
                        &self.weights.up_proj,
                        gate_out,
                        n,
                        inter,
                        h,
                        16,
                        stream,
                    )
                })?;
            if coalesced_gateup_done {
                // Dispatched via SM120 BF16xBF16 MMA coalesced warp-split-K fused gate+up+silu.
            } else {
                let gt = self.gate_proj_t.as_ref().unwrap();
                let ut = self.up_proj_t.as_ref().unwrap();
                // Kernel selection priority (highest first, all 0-ULP byte-exact):
                //   ATLAS_GATEUP_N32    → n32_pipe_k64 (544 CTAs on N=17408, auto >128 SMs)
                //   ATLAS_GATEUP_K64    → n64_pipe_k64 (K_STEP=64, reg-dequant, auto >64 SMs)
                //   ATLAS_DEQUANT_PIPE  → n64_pipe     (K_STEP=32, reg-dequant, auto >64 SMs)
                //   default             → n64 staged   (smem_B_fp8 staging)
                if crate::layers::gateup_n32_enabled()
                    && self.w4a16_gemm_t_m32_n32_gateup_silu_pipe_k64.0 != 0
                    && h.is_multiple_of(64)
                    && inter.is_multiple_of(32)
                {
                    crate::kprof!(ctx.gpu, stream, "ffn_gateup_fused_kgamma", {
                        ops::w4a16_gemm_n32_m32_gateup_silu(
                            ctx.gpu,
                            self.w4a16_gemm_t_m32_n32_gateup_silu_pipe_k64,
                            input,
                            gt,
                            ut,
                            gate_out,
                            n,
                            inter,
                            h,
                            stream,
                        )?;
                        anyhow::Result::<()>::Ok(())
                    })?;
                } else {
                    let fused_kernel = if crate::layers::gateup_k64_enabled()
                        && self.w4a16_gemm_t_m32_n64_gateup_silu_pipe_k64.0 != 0
                        && h.is_multiple_of(64)
                    // K must be divisible by 64
                    {
                        self.w4a16_gemm_t_m32_n64_gateup_silu_pipe_k64
                    } else if crate::layers::dequant_pipe_enabled()
                        && self.w4a16_gemm_t_m32_n64_gateup_silu_pipe.0 != 0
                    {
                        self.w4a16_gemm_t_m32_n64_gateup_silu_pipe
                    } else {
                        self.w4a16_gemm_t_m32_n64_gateup_silu
                    };
                    crate::kprof!(ctx.gpu, stream, "ffn_gateup_fused_kgamma", {
                        ops::w4a16_gemm_n64_m32_gateup_silu(
                            ctx.gpu,
                            fused_kernel,
                            input,
                            gt,
                            ut,
                            gate_out,
                            n,
                            inter,
                            h,
                            stream,
                        )?;
                        anyhow::Result::<()>::Ok(())
                    })?;
                }
            }
        } else {
            crate::kprof!(ctx.gpu, stream, "ffn_gate_kgamma", {
                if gateup_splitk_ok {
                    let gt = self.gate_proj_t.as_ref().unwrap();
                    ops::w4a16_gemm_n64_m32_splitk(
                        ctx.gpu,
                        self.w4a16_gemm_t_m32_n64_splitk,
                        self.reduce_splitk_k,
                        input,
                        gt,
                        gate_out,
                        gateup_ws.unwrap(),
                        n,
                        inter,
                        h,
                        inter, // ldb == N for tightly-packed T-weight
                        gateup_splitk,
                        stream,
                    )?;
                } else if m32_path {
                    let gt = self.gate_proj_t.as_ref().unwrap();
                    ops::w4a16_gemm_n64_m32(
                        ctx.gpu,
                        self.w4a16_gemm_t_m32_n64,
                        input,
                        gt,
                        gate_out,
                        n,
                        inter,
                        h,
                        stream,
                    )?;
                } else if m128_path {
                    let gt = self.gate_proj_t.as_ref().unwrap();
                    ops::w4a16_gemm_n128_m128(
                        ctx.gpu,
                        self.w4a16_gemm_t_m128,
                        input,
                        gt,
                        gate_out,
                        n,
                        inter,
                        h,
                        stream,
                    )?;
                } else if m16_path {
                    let gt = self.gate_proj_t.as_ref().unwrap();
                    ops::w4a16_gemm_n128_m16(
                        ctx.gpu,
                        self.w4a16_gemm_t_m16,
                        input,
                        gt,
                        gate_out,
                        n,
                        inter,
                        h,
                        stream,
                    )?;
                } else {
                    ops::w4a16_gemm(
                        ctx.gpu,
                        self.w4a16_gemm,
                        input,
                        &self.weights.gate_proj,
                        gate_out,
                        n,
                        inter,
                        h,
                        stream,
                    )?;
                }
                anyhow::Result::<()>::Ok(())
            })?;

            // up_proj GEMM: [n, H] → [n, inter]. Same split-K treatment as gate;
            // reuses the shared workspace (gate's reduce already consumed it).
            crate::kprof!(ctx.gpu, stream, "ffn_up_kgamma", {
                if gateup_splitk_ok {
                    let ut = self.up_proj_t.as_ref().unwrap();
                    ops::w4a16_gemm_n64_m32_splitk(
                        ctx.gpu,
                        self.w4a16_gemm_t_m32_n64_splitk,
                        self.reduce_splitk_k,
                        input,
                        ut,
                        up_out,
                        gateup_ws.unwrap(),
                        n,
                        inter,
                        h,
                        inter, // ldb == N for tightly-packed T-weight
                        gateup_splitk,
                        stream,
                    )?;
                } else if m32_path {
                    let ut = self.up_proj_t.as_ref().unwrap();
                    ops::w4a16_gemm_n64_m32(
                        ctx.gpu,
                        self.w4a16_gemm_t_m32_n64,
                        input,
                        ut,
                        up_out,
                        n,
                        inter,
                        h,
                        stream,
                    )?;
                } else if m128_path {
                    let ut = self.up_proj_t.as_ref().unwrap();
                    ops::w4a16_gemm_n128_m128(
                        ctx.gpu,
                        self.w4a16_gemm_t_m128,
                        input,
                        ut,
                        up_out,
                        n,
                        inter,
                        h,
                        stream,
                    )?;
                } else if m16_path {
                    let ut = self.up_proj_t.as_ref().unwrap();
                    ops::w4a16_gemm_n128_m16(
                        ctx.gpu,
                        self.w4a16_gemm_t_m16,
                        input,
                        ut,
                        up_out,
                        n,
                        inter,
                        h,
                        stream,
                    )?;
                } else {
                    ops::w4a16_gemm(
                        ctx.gpu,
                        self.w4a16_gemm,
                        input,
                        &self.weights.up_proj,
                        up_out,
                        n,
                        inter,
                        h,
                        stream,
                    )?;
                }
                anyhow::Result::<()>::Ok(())
            })?;

            // activation(gate) * up for n tokens
            crate::kprof!(ctx.gpu, stream, "ffn_silu_mul_kgamma", {
                ops::silu_mul(
                    ctx.gpu,
                    self.act_mul,
                    gate_out,
                    up_out,
                    gate_out,
                    n * inter,
                    stream,
                )?;
                anyhow::Result::<()>::Ok(())
            })?;
        } // end !fused_gateup

        // FFN activation-sparsity MEASUREMENT (ATLAS_MEASURE_FFN_SPARSITY).
        // This is the K=γ verify FFN — the path DFlash decode ACTUALLY uses
        // (forward() M=1 is not hit per decode step). Row 0 of the [n, *]
        // buffers is a representative decode activation. `input` is the
        // gate/up input (K=hidden); `gate_out` now holds silu(gate)*up, the
        // down_proj input (K=intermediate) — no recompute needed here (unlike
        // forward(), which consumes gate_out in its fused path). PURE READER:
        // measures row 0 into dedicated accumulators, never mutates the token
        // stream. Skipped under graph capture (lazy alloc illegal mid-capture)
        // → run with ATLAS_DFLASH_DEBUG_NO_GRAPH=1 to observe eager decode.
        if self.sparsity_measure_active() && !ctx.graph_capture {
            let (hist_g, count_g, hist_d, count_d, _meas) =
                self.ensure_sparsity_meas(ctx.gpu, inter as usize)?;
            self.measure_sparsity_site(ctx, input, hist_g, count_g, h, stream)?;
            self.measure_sparsity_site(ctx, gate_out, hist_d, count_d, inter, stream)?;
            self.maybe_dump_sparsity(ctx, "dense_ffn_kgamma", stream)?;
        }

        // down_proj GEMM: [n, inter] → [n, H]
        let output = ctx.buffers.moe_output();
        crate::kprof!(ctx.gpu, stream, "ffn_down_kgamma", {
            // On wide discrete GPUs (`sm_count > 64`) at small `M <= 16`, route
            // `down_proj` directly through intra-CTA warp-split-K BF16xBF16 MMA
            // (`w4a16_gemm_coalesced_{v2_}n16_sk4_m16`, 320 CTAs at `H=5120`):
            // eliminates global FP32 Split-K partial traffic, eliminates the
            // `reduce_splitk_f32_to_bf16` launch across all 64 layers, removes
            // unscaled FP8 E4M3 activation clipping, and halves wasted M-tile
            // rows vs M32. Always skipped on GB10 (`<= 64` SMs).
            let coalesced_done = m32_path
                && n <= 16
                && !self.weights.down_proj.is_null()
                && try_coalesced_nvfp4_sm120_small_m(
                    ctx.gpu,
                    gate_out,
                    &self.weights.down_proj,
                    output,
                    n,
                    h,
                    inter,
                    16,
                    stream,
                )?;
            if !coalesced_done {
                // Split-K down_proj: [M=n, N=h, K=inter]. The single-slice
                // m32_n64 kernel is occupancy-starved here (N=h=5120 → 80 CTAs
                // vs gate/up's 256 at N=inter=16384) and grinds a long K-loop.
                // Split-K multiplies CTAs by `splits` into an FP32 workspace,
                // then reduces → BF16. Gated by ATLAS_FFN_DOWN_SPLITK; falls
                // through to the single-slice path when disabled or unallocated.
                let splitk = crate::layers::ffn_down_splitk();
                let ws = if splitk > 0 {
                    *self.splitk_workspace.lock().unwrap()
                } else {
                    None
                };
                if let (true, Some(ws)) = (
                    m32_path
                        && splitk > 0
                        && self.w4a16_gemm_t_m32_n64_splitk.0 != 0
                        && self.reduce_splitk_k.0 != 0,
                    ws,
                ) {
                    let dt = self.down_proj_t.as_ref().unwrap();
                    ops::w4a16_gemm_n64_m32_splitk(
                        ctx.gpu,
                        self.w4a16_gemm_t_m32_n64_splitk,
                        self.reduce_splitk_k,
                        gate_out,
                        dt,
                        output,
                        ws,
                        n,
                        h,
                        inter,
                        h, // ldb == N for tightly-packed T-weight
                        splitk,
                        stream,
                    )?;
                } else if m32_path {
                    let dt = self.down_proj_t.as_ref().unwrap();
                    ops::w4a16_gemm_n64_m32(
                        ctx.gpu,
                        self.w4a16_gemm_t_m32_n64,
                        gate_out,
                        dt,
                        output,
                        n,
                        h,
                        inter,
                        stream,
                    )?;
                } else if m128_path {
                    let dt = self.down_proj_t.as_ref().unwrap();
                    ops::w4a16_gemm_n128_m128(
                        ctx.gpu,
                        self.w4a16_gemm_t_m128,
                        gate_out,
                        dt,
                        output,
                        n,
                        h,
                        inter,
                        stream,
                    )?;
                } else if m16_path {
                    let dt = self.down_proj_t.as_ref().unwrap();
                    ops::w4a16_gemm_n128_m16(
                        ctx.gpu,
                        self.w4a16_gemm_t_m16,
                        gate_out,
                        dt,
                        output,
                        n,
                        h,
                        inter,
                        stream,
                    )?;
                } else {
                    ops::w4a16_gemm(
                        ctx.gpu,
                        self.w4a16_gemm,
                        gate_out,
                        &self.weights.down_proj,
                        output,
                        n,
                        h,
                        inter,
                        stream,
                    )?;
                }
            }
            anyhow::Result::<()>::Ok(())
        })?;

        Ok(())
    }

    /// Internal exact-scalar eligibility inside `forward_kgamma`: stays true
    /// for `rows in 2..=32` on GB10 (`<= 64` SMs) or when `ATLAS_FFN_TC=0`, and
    /// yields to batched tensor-core kernels for `rows >= 4` on `> 64`-SM GPUs.
    fn exact_kgamma_internal_eligible(&self, rows: u32) -> bool {
        !ffn_tc_active_for_rows(rows)
            && exact_ffn_auto_kgamma_applicable(
                rows,
                self.bf16_weights.is_none()
                    && !self.has_w3_gemm()
                    && self.activation == FfnActivation::SiLU,
            )
    }

    /// True when a `> 64`-SM GPU should auto-route `rows in 4..=256` into
    /// batched tensor-core `forward_kgamma` (even when `ATLAS_FFN_KGAMMA_M16`
    /// is unset), preventing attention layers from falling back to a per-token
    /// `ffn.forward()` loop at `C = 16..64`. Always `false` on GB10 (`<= 64` SMs)
    /// and whenever `ATLAS_FFN_TC=0`.
    fn sm120_auto_tc_kgamma_ready(&self, rows: u32) -> bool {
        if atlas_core::device::active_sm_count() <= FFN_KGAMMA_AUTO_TC_SM_THRESHOLD
            || !ffn_tc_active_for_rows(rows)
            || rows < crate::layers::TC_VERIFY_MIN_ROWS as u32
            || rows > FFN_KGAMMA_WIDE_MAX_ROWS
            || self.bf16_weights.is_some()
            || self.has_w3_gemm()
            || self.activation != FfnActivation::SiLU
        {
            return false;
        }
        if (rows as usize) <= sm120_coalesced_max_m()
            && sm120_coalesced_m16_env_enabled()
            && !self.weights.gate_proj.is_null()
            && !self.weights.up_proj.is_null()
            && !self.weights.down_proj.is_null()
        {
            return true;
        }
        if rows as usize <= crate::layers::TC_VERIFY_MAX_ROWS {
            self.has_transposed_ffn() && self.w4a16_gemm_t_m128.0 != 0
        } else {
            self.wide_kgamma_ready(rows)
        }
    }

    /// Whether `forward_kgamma` should service `rows` without requiring an
    /// explicit `ATLAS_FFN_KGAMMA_M16=1` opt-in.
    ///
    /// On GB10 (`<= 64` SMs) this is strictly the exact W4+SiLU dispatcher for
    /// `rows in 2..=32`. On `> 64`-SM GPUs (unless `ATLAS_FFN_TC=0`) this also
    /// admits `rows in 4..=256` when the transposed tensor-core FFN kernels are
    /// ready so `M = 4, 8, 16, 32, 64` never fall back to per-token loops.
    pub fn exact_kgamma_applicable(&self, rows: u32) -> bool {
        self.exact_kgamma_internal_eligible(rows) || self.sm120_auto_tc_kgamma_ready(rows)
    }

    /// Whether single-token decode should run `forward_kgamma(n = 1)` so it
    /// shares the K=γ verify's tensor-core FFN route
    /// (`ATLAS_DECODE_TC_PARITY=1`). Engages only when verify itself is on
    /// that route: `ATLAS_FFN_TC=1` bypasses the exact dispatcher and
    /// `ATLAS_FFN_KGAMMA_M16=1` sends verify rows through `forward_kgamma`.
    pub fn decode_tc_parity_route(&self) -> bool {
        crate::layers::decode_tc_parity_enabled()
            && exact_ffn_tc_override()
            && crate::layers::ffn_kgamma_m16_enabled()
            && self.bf16_weights.is_none()
            && !self.has_w3_gemm()
            && self.activation == FfnActivation::SiLU
            && self.has_transposed_ffn()
    }

    /// N-token prefill: GEMM for all projections.
    pub fn forward_prefill(
        &self,
        input: DevicePtr,
        num_tokens: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let h = ctx.config.hidden_size as u32;
        let inter = ctx.config.intermediate_size as u32;
        let m = num_tokens as u32;

        let gate_out = ctx.buffers.expert_gate_out();
        let up_out = ctx.buffers.expert_up_out();

        // BF16 prefill dispatch: dense_gemm_bf16 for all three projections.
        // (Gemma-4 dense path — bypasses NVFP4 entirely; not affected by
        // ATLAS_PREFILL_FFN_FAST.)
        if let Some(ref bf16w) = self.bf16_weights {
            ops::dense_gemm(
                ctx.gpu,
                self.dense_gemm_bf16_k,
                input,
                &bf16w.gate_proj,
                gate_out,
                m,
                inter,
                h,
                stream,
            )?;
            ops::dense_gemm(
                ctx.gpu,
                self.dense_gemm_bf16_k,
                input,
                &bf16w.up_proj,
                up_out,
                m,
                inter,
                h,
                stream,
            )?;
            ops::silu_mul(
                ctx.gpu,
                self.act_mul,
                gate_out,
                up_out,
                gate_out,
                m * inter,
                stream,
            )?;
            let output = ctx.buffers.moe_output();
            ops::dense_gemm(
                ctx.gpu,
                self.dense_gemm_bf16_k,
                gate_out,
                &bf16w.down_proj,
                output,
                m,
                h,
                inter,
                stream,
            )?;
            return Ok(());
        }

        #[cfg(all(feature = "cuda", target_os = "linux"))]
        if self.try_forward_flashinfer_prefill(input, m, h, inter, gate_out, up_out, ctx, stream)? {
            return Ok(());
        }

        #[cfg(feature = "cuda")]
        if self
            .try_forward_cublaslt_nvfp4_prefill(input, m, h, inter, gate_out, up_out, ctx, stream)?
        {
            return Ok(());
        }

        // Large-M W4A4 native NVFP4×NVFP4 fast path: route through the
        // CUTLASS-style `nvfp4_nvfp4_gemm_t_m64` kernel (native E2M1
        // tensor-core MMA) when:
        //   - ATLAS_E2M1_GEMM=1 was set at startup
        //   - all three required kernel symbols loaded (nvfp4_cutlass +
        //     quantize_nvfp4)
        //   - M >= 128 (kernel's intended window — matches the other
        //     prefill fast paths)
        // Each GEMM prequantizes BF16 activations to NVFP4 inline
        // (absmax + per-group E2M1) before dispatching the native MMA.
        // Theoretical 2× MFU lift over `w4a16_gemm_t_m128` (BF16×NVFP4)
        // since we eliminate the inner-loop dequant AND halve activation
        // DRAM traffic (0.5 B/elt vs 2 B/elt BF16). Takes precedence
        // over the fp8/v2/m128 paths when active. Disabled or geometrically
        // ineligible requests retain those fallbacks; eligible incomplete
        // requests fail before projection.
        let e2m1_full_requested = crate::layers::prefill_ffn_e2m1_enabled();
        let e2m1_down_requested = crate::layers::prefill_ffn_e2m1_down_only_enabled();
        let e2m1_kmajor_requested = crate::layers::prefill_ffn_e2m1_kmajor_enabled();
        let e2m1_kmajor_m256_requested = crate::layers::prefill_ffn_e2m1_kmajor_m256_enabled();
        let e2m1_silu_quant_requested = crate::layers::prefill_ffn_e2m1_silu_quant_enabled();
        if m >= 128 && e2m1_silu_quant_requested {
            ensure!(
                e2m1_full_requested || e2m1_down_requested,
                "ATLAS_E2M1_SILU_QUANT=1 requires ATLAS_E2M1_GEMM=1 or ATLAS_E2M1_GEMM_DOWN_ONLY=1"
            );
            ensure!(
                self.e2m1_checkpoint_scales.is_some(),
                "ATLAS_E2M1_SILU_QUANT=1 requires ATLAS_E2M1_STATIC_SCALE=1 and valid checkpoint input scales"
            );
            ensure!(
                self.activation == FfnActivation::SiLU,
                "ATLAS_E2M1_SILU_QUANT=1 is valid only for SiLU-gated FFNs"
            );
            ensure!(
                self.nvfp4_silu_quantize_k.0 != 0,
                "ATLAS_E2M1_SILU_QUANT=1 requires quantize_silu_mul_bf16_to_nvfp4 in the quantize_nvfp4 module"
            );
        }
        if m >= 128 && e2m1_kmajor_requested {
            ensure!(
                e2m1_full_requested || e2m1_down_requested,
                "ATLAS_E2M1_KMAJOR=1 requires ATLAS_E2M1_GEMM=1 or ATLAS_E2M1_GEMM_DOWN_ONLY=1"
            );
            ensure!(
                self.has_e2m1_kmajor_runtime(),
                "ATLAS_E2M1_KMAJOR=1 requires nvfp4_nvfp4_gemm_kmajor_m128 plus the activation quantizer"
            );
            ensure!(
                self.has_transposed_ffn(),
                "ATLAS_E2M1_KMAJOR=1 requires transformed gate/up/down weights (ATLAS_FFN_M16_TRANSPOSED=1)"
            );
            ensure!(
                e2m1_kmajor_projection_shape(m, inter, h)
                    && e2m1_kmajor_projection_shape(m, h, inter),
                "ATLAS_E2M1_KMAJOR=1 requires M>=128, N divisible by 128, and K divisible by 64 for every FFN projection; got M={m} hidden={h} intermediate={inter}"
            );
        }
        if e2m1_kmajor_m256_requested {
            ensure!(
                e2m1_kmajor_requested,
                "ATLAS_E2M1_KMAJOR_M256=1 requires ATLAS_E2M1_KMAJOR=1"
            );
            if m >= E2M1_KMAJOR_M256_MIN_M {
                ensure!(
                    self.has_e2m1_kmajor_m256_runtime(),
                    "ATLAS_E2M1_KMAJOR_M256=1 requires nvfp4_nvfp4_gemm_kmajor_m256 plus the M128 K-major runtime"
                );
            }
        }
        let kmajor_geometry_ready =
            e2m1_kmajor_projection_shape(m, inter, h) && e2m1_kmajor_projection_shape(m, h, inter);
        let selected_kernel =
            e2m1_selected_kernel(e2m1_kmajor_requested, e2m1_kmajor_m256_requested, m);
        let e2m1_runtime_ready = if e2m1_kmajor_requested {
            self.has_e2m1_kmajor_runtime()
                && self.has_transposed_ffn()
                && kmajor_geometry_ready
                && (selected_kernel != E2m1SelectedKernel::KmajorM256
                    || self.has_e2m1_kmajor_m256_runtime())
        } else {
            self.has_e2m1_ffn()
        };
        let silu_quant_ready = !e2m1_silu_quant_requested
            || (self.e2m1_checkpoint_scales.is_some()
                && self.activation == FfnActivation::SiLU
                && self.nvfp4_silu_quantize_k.0 != 0);

        // Per-shape E2M1 dispatch: gate/up stay on `w4a16_gemm_t_m128`,
        // down_proj routes through native E2M1 hardware MMA. Per the
        // shape table at the `prefill_ffn_e2m1_down_only_enabled` doc
        // site: gate (K=5120,N=17408) + up (same) are faster on the
        // BF16×NVFP4 w4a16 m128 path, while down (K=17408,N=5120) is
        // 1.31× faster via E2M1 MMA — net ~30% down savings with no
        // gate/up regression. Mutually exclusive with the all-three
        // `e2m1_fast_path`.
        let e2m1_route = e2m1_prefill_route(
            e2m1_full_requested,
            e2m1_down_requested,
            m >= 128,
            e2m1_runtime_ready && silu_quant_ready,
            e2m1_runtime_ready
                && silu_quant_ready
                && self.has_transposed_ffn()
                && self.w4a16_gemm_t_m128.0 != 0,
        );
        let (e2m1_fast_path, e2m1_down_only_path) = match e2m1_route {
            E2m1PrefillRoute::Complete(E2m1Scope::Full) => (true, false),
            E2m1PrefillRoute::Complete(E2m1Scope::DownOnly) => (false, true),
            E2m1PrefillRoute::Missing(E2m1Scope::Full) => {
                bail!(
                    "ATLAS_E2M1_GEMM=1 requires the complete selected W4A4 runtime before gate projection"
                );
            }
            E2m1PrefillRoute::Missing(E2m1Scope::DownOnly) => {
                bail!(
                    "ATLAS_E2M1_GEMM_DOWN_ONLY=1 requires the complete selected W4A4 runtime plus transformed W4A16 M128 gate/up before gate projection"
                );
            }
            E2m1PrefillRoute::Disabled | E2m1PrefillRoute::Ineligible => (false, false),
        };
        if let E2m1PrefillRoute::Complete(scope) = e2m1_route {
            log_e2m1_route_once(E2m1RouteProof {
                scope,
                kernel: selected_kernel,
                checkpoint_static: self.e2m1_checkpoint_scales.is_some(),
                fused_silu_input: e2m1_silu_quant_requested,
                activation: self.activation,
                m,
                h,
                inter,
            });
        }

        // Large-M FP8 predequant fast path: route through the
        // `fp8_gemm_t_m128` kernel (BF16 A × pre-dequanted FP8 B) when:
        //   - ATLAS_FFN_PREDEQUANT_FP8=1 was set at startup
        //   - `predequant_for_prefill` ran successfully (gate/up/down
        //     FP8 buffers installed)
        //   - the `fp8_gemm_t_m128` kernel symbol is present
        //   - M >= 128
        // Saves the entire DEQUANT phase + one __syncthreads per
        // K-step inside w4a16_gemm_t_m128. Empirically a REGRESSION
        // on Qwen3.6-27B (5.68s → 8.11s TTFT at ISL=3603) because the
        // 2× B DRAM traffic (FP8 1 byte vs NVFP4 0.5 byte/elt) at
        // K=5120 + N=17408 exceeds the dequant savings — the FFN GEMM
        // is memory-bound, not compute-bound. Falls back to the w4a16
        // m128 path when not active. Mirrors the attention
        // `predequant_for_prefill` + `fp8_gemm_n128_m128` pattern.
        let fp8_fast_path = !e2m1_fast_path
            && !e2m1_down_only_path
            && m >= 128
            && crate::layers::prefill_ffn_fp8_enabled()
            && self.has_fp8_ffn();

        // Large-M v2 fast path: route through the 8-warp shadow
        // `w4a16_gemm_t_m128_v2` kernel when:
        //   - ATLAS_FFN_M128_V2=1
        //   - transposed (`nvfp4_t`) FFN weights installed
        //   - the v2 kernel symbol is present
        //   - M >= 128
        // Same SMEM footprint + 2-stage pipeline as v1 but doubles the
        // active warp count (256 threads/CTA, 4 warps per chunk run
        // chunk 0 and chunk 1 MMAs in parallel instead of serial). Net
        // result on compute-bound GEMMs: ~10-20% kernel-time win. On
        // memory-bound GEMMs: roughly neutral. Falls back to v1 when
        // not active. Original kernel:
        // kernels/gb10/minimax-m2-229b/nvfp4/w4a16_gemm_v2.cu —
        // copied verbatim into qwen3.6-27b/.
        let v2_fast_path = !e2m1_fast_path
            && !e2m1_down_only_path
            && !fp8_fast_path
            && m >= 128
            && crate::layers::prefill_ffn_m128_v2_enabled()
            && self.has_transposed_ffn()
            && self.w4a16_gemm_t_m128_v2.0 != 0;

        // Large-M fast path: route through the transposed-weight
        // M_TILE=128 kernel (`w4a16_gemm_t_m128`) when:
        //   - ATLAS_PREFILL_FFN_FAST is enabled (default ON unless `=0`)
        //   - transposed (`nvfp4_t`) FFN weights installed at load
        //     (controlled by ATLAS_FFN_M16_TRANSPOSED, default ON unless `=0`)
        //   - the kernel symbol is present in the loaded module
        //   - M >= 128 (kernel's intended window — small-M would waste
        //     128-row CTAs the same way the M_TILE=64 path wastes 64-row
        //     CTAs at M < 64)
        // Mirrors the attention `w4a16_gemm_m128_dispatch` pattern in
        // `qwen3_attention/prefill_weights.rs`. Disabled or ineligible
        // requests retain the lower-priority route. An eligible explicit
        // request fails closed when any required weight/kernel is absent so
        // a stale bundle cannot be mislabeled as a FAST benchmark.
        let fast_shape_eligible =
            !e2m1_fast_path && !e2m1_down_only_path && !fp8_fast_path && !v2_fast_path && m >= 128;
        let fast_route = prefill_ffn_fast_route(
            crate::layers::prefill_ffn_fast_enabled(),
            fast_shape_eligible,
            self.gate_proj_t.is_some() && self.up_proj_t.is_some() && self.down_proj_t.is_some(),
            self.w4a16_gemm_t_m16.0 != 0 || self.prefill_t_m128_only,
            self.w4a16_gemm_t_m128.0 != 0,
        );
        let fast_path = match fast_route {
            PrefillFfnFastRoute::Complete => true,
            PrefillFfnFastRoute::Missing if !crate::layers::prefill_ffn_fast_explicit() => {
                static DEFAULT_MISSING: std::sync::Once = std::sync::Once::new();
                DEFAULT_MISSING.call_once(|| {
                    tracing::warn!(
                        "default-on ATLAS_PREFILL_FFN_FAST skipped: transposed FFN weights or \
                         w4a16_gemm_t_m16/m128 are absent for this checkpoint; using the \
                         M_TILE=64 route (set ATLAS_PREFILL_FFN_FAST=1 to fail closed instead)"
                    );
                });
                false
            }
            PrefillFfnFastRoute::Missing => {
                bail!(
                    "ATLAS_PREFILL_FFN_FAST=1 requires transformed gate/up/down weights plus w4a16_gemm_t_m16 and w4a16_gemm_t_m128 before gate projection"
                );
            }
            PrefillFfnFastRoute::Disabled | PrefillFfnFastRoute::Ineligible => false,
        };

        // One-shot info log on first prefill so the bench harness can
        // verify which path is firing. The atomic ensures we log once
        // across all 64 layers per process. Uses both tracing::info! and
        // eprintln! — the latter bypasses tracing's BufWriter when the
        // server is launched under `nohup ... > file 2>&1` (stderr is
        // line-buffered to a regular file, so a single `\n`-terminated
        // write hits disk immediately, even before process teardown).
        static LOGGED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        if !LOGGED.swap(true, std::sync::atomic::Ordering::Relaxed) {
            let has_t = self.has_transposed_ffn();
            let has_fp8 = self.has_fp8_ffn();
            let has_e2m1 = self.has_e2m1_ffn();
            let has_e2m1_kmajor = self.has_e2m1_kmajor_runtime();
            let has_e2m1_kmajor_m256 = self.has_e2m1_kmajor_m256_runtime();
            let m128_ok = self.w4a16_gemm_t_m128.0 != 0;
            let m128_v2_ok = self.w4a16_gemm_t_m128_v2.0 != 0;
            let fp8_m128_ok = self.fp8_gemm_t_m128_k.0 != 0;
            let e2m1_ok = self.nvfp4_gemm_k.0 != 0;
            let gate_on = crate::layers::prefill_ffn_fast_enabled();
            let fp8_gate_on =
                std::env::var("ATLAS_FFN_PREDEQUANT_FP8").ok().as_deref() == Some("1");
            let v2_gate_on = std::env::var("ATLAS_FFN_M128_V2").ok().as_deref() == Some("1");
            let e2m1_gate_on = std::env::var("ATLAS_E2M1_GEMM").ok().as_deref() == Some("1");
            let e2m1_down_only_gate_on =
                std::env::var("ATLAS_E2M1_GEMM_DOWN_ONLY").ok().as_deref() == Some("1");
            let has_e2m1_checkpoint_scales = self.e2m1_checkpoint_scales.is_some();
            let e2m1_static_scale_env =
                std::env::var("ATLAS_E2M1_STATIC_SCALE").ok().as_deref() == Some("1");
            tracing::info!(
                m,
                inter,
                h,
                e2m1_fast_path,
                e2m1_down_only_path,
                fp8_fast_path,
                v2_fast_path,
                fast_path,
                has_e2m1_ffn = has_e2m1,
                has_fp8_ffn = has_fp8,
                has_transposed = has_t,
                e2m1_kernel = e2m1_ok,
                fp8_m128_kernel = fp8_m128_ok,
                m128_v2_kernel = m128_v2_ok,
                m128_kernel = m128_ok,
                gate = gate_on,
                fp8_gate = fp8_gate_on,
                v2_gate = v2_gate_on,
                e2m1_gate = e2m1_gate_on,
                e2m1_down_only_gate = e2m1_down_only_gate_on,
                has_e2m1_checkpoint_scales,
                e2m1_static_scale_env,
                e2m1_kmajor_requested,
                e2m1_kmajor_m256_requested,
                e2m1_silu_quant_requested,
                has_e2m1_kmajor,
                has_e2m1_kmajor_m256,
                "dense_ffn forward_prefill dispatch (one-shot)"
            );
            eprintln!(
                "[atlas-prefill-ffn] dispatch: M={m} inter={inter} h={h} \
                 e2m1_fast_path={e2m1_fast_path} \
                 e2m1_down_only_path={e2m1_down_only_path} \
                 fp8_fast_path={fp8_fast_path} v2_fast_path={v2_fast_path} \
                 fast_path={fast_path} has_e2m1={has_e2m1} has_fp8={has_fp8} \
                 has_transposed={has_t} e2m1_kernel={e2m1_ok} \
                 fp8_m128_kernel={fp8_m128_ok} \
                 m128_v2_kernel={m128_v2_ok} m128_kernel={m128_ok} \
                 e2m1_gate=ATLAS_E2M1_GEMM={e2m1_gate_on} \
                 e2m1_down_only_gate=ATLAS_E2M1_GEMM_DOWN_ONLY={e2m1_down_only_gate_on} \
                 has_e2m1_checkpoint_scales={has_e2m1_checkpoint_scales} \
                 e2m1_static_scale=ATLAS_E2M1_STATIC_SCALE={e2m1_static_scale_env} \
                 e2m1_kmajor=ATLAS_E2M1_KMAJOR={e2m1_kmajor_requested} \
                 e2m1_kmajor_m256=ATLAS_E2M1_KMAJOR_M256={e2m1_kmajor_m256_requested} \
                 e2m1_silu_quant=ATLAS_E2M1_SILU_QUANT={e2m1_silu_quant_requested} \
                 e2m1_kmajor_kernel={has_e2m1_kmajor} \
                 e2m1_kmajor_m256_kernel={has_e2m1_kmajor_m256} \
                 gate=ATLAS_PREFILL_FFN_FAST={gate_on} \
                 fp8_gate=ATLAS_FFN_PREDEQUANT_FP8={fp8_gate_on} \
                 v2_gate=ATLAS_FFN_M128_V2={v2_gate_on}"
            );
        }

        if e2m1_fast_path {
            let _scratch_use = self.lock_e2m1_use(stream)?;
            // Native W4A4 NVFP4×NVFP4 dispatch: prequant activations to
            // NVFP4 in-place, then issue the CUTLASS-style E2M1×E2M1 MMA
            // GEMM. Uses the standard `[N, K/2]` HuggingFace weight
            // layout (NOT the `nvfp4_t` transposed layout) — matches the
            // kernel's coalesced gmem read pattern.
            //
            // gate_proj and up_proj share the exact same [M, H] BF16 input.
            // Prepare it once: the old path repeated an identical global
            // absmax, D2H synchronization/readback, and quantization before
            // each projection even though both consumed identical bytes.
            let gate_up_scale = self.e2m1_checkpoint_scales.map(|scales| scales.gate_up);
            let gate_weight = if e2m1_kmajor_requested {
                self.gate_proj_t.as_ref().unwrap()
            } else {
                &self.weights.gate_proj
            };
            let up_weight = if e2m1_kmajor_requested {
                self.up_proj_t.as_ref().unwrap()
            } else {
                &self.weights.up_proj
            };
            let down_weight = if e2m1_kmajor_requested {
                self.down_proj_t.as_ref().unwrap()
            } else {
                &self.weights.down_proj
            };
            let (a_packed, a_scale, a_scale2) =
                self.prepare_e2m1_input(ctx, input, m, h, gate_up_scale, stream)?;
            self.forward_e2m1_prepared(
                ctx,
                a_packed,
                a_scale,
                a_scale2,
                gate_weight,
                gate_out,
                m,
                inter,
                h,
                e2m1_kmajor_requested,
                stream,
            )?;
            self.forward_e2m1_prepared(
                ctx,
                a_packed,
                a_scale,
                a_scale2,
                up_weight,
                up_out,
                m,
                inter,
                h,
                e2m1_kmajor_requested,
                stream,
            )?;
            // down_proj: [M, inter] BF16 → [M, H] BF16
            let output = ctx.buffers.moe_output();
            if e2m1_silu_quant_requested {
                let down_scale = self.e2m1_checkpoint_scales.unwrap().down;
                let (a_packed, a_scale, a_scale2) = self
                    .prepare_e2m1_silu_input(ctx, gate_out, up_out, m, inter, down_scale, stream)?;
                self.forward_e2m1_prepared(
                    ctx,
                    a_packed,
                    a_scale,
                    a_scale2,
                    down_weight,
                    output,
                    m,
                    h,
                    inter,
                    e2m1_kmajor_requested,
                    stream,
                )?;
            } else {
                // SiLU/GELU(gate) * up for all M tokens
                ops::silu_mul(
                    ctx.gpu,
                    self.act_mul,
                    gate_out,
                    up_out,
                    gate_out,
                    m * inter,
                    stream,
                )?;
                self.forward_e2m1_proj(
                    ctx,
                    gate_out,
                    down_weight,
                    output,
                    m,
                    h,
                    inter,
                    self.e2m1_checkpoint_scales.map(|scales| scales.down),
                    e2m1_kmajor_requested,
                    stream,
                )?;
            }
            return Ok(());
        }

        if e2m1_down_only_path {
            let _scratch_use = self.lock_e2m1_use(stream)?;
            // gate/up stay on the w4a16 m128 fast path; only down_proj
            // routes through E2M1 hardware MMA. Matches the shape table
            // documented at `prefill_ffn_e2m1_down_only_enabled` — net
            // ~30% down_proj savings, no gate/up regression vs the
            // standard `fast_path`.
            let gt = self.gate_proj_t.as_ref().unwrap();
            let ut = self.up_proj_t.as_ref().unwrap();
            ops::w4a16_gemm_n128_m128(
                ctx.gpu,
                self.w4a16_gemm_t_m128,
                input,
                gt,
                gate_out,
                m,
                inter,
                h,
                stream,
            )?;
            ops::w4a16_gemm_n128_m128(
                ctx.gpu,
                self.w4a16_gemm_t_m128,
                input,
                ut,
                up_out,
                m,
                inter,
                h,
                stream,
            )?;
            let output = ctx.buffers.moe_output();
            let down_weight = if e2m1_kmajor_requested {
                self.down_proj_t.as_ref().unwrap()
            } else {
                &self.weights.down_proj
            };
            if e2m1_silu_quant_requested {
                let down_scale = self.e2m1_checkpoint_scales.unwrap().down;
                let (a_packed, a_scale, a_scale2) = self
                    .prepare_e2m1_silu_input(ctx, gate_out, up_out, m, inter, down_scale, stream)?;
                self.forward_e2m1_prepared(
                    ctx,
                    a_packed,
                    a_scale,
                    a_scale2,
                    down_weight,
                    output,
                    m,
                    h,
                    inter,
                    e2m1_kmajor_requested,
                    stream,
                )?;
            } else {
                ops::silu_mul(
                    ctx.gpu,
                    self.act_mul,
                    gate_out,
                    up_out,
                    gate_out,
                    m * inter,
                    stream,
                )?;
                self.forward_e2m1_proj(
                    ctx,
                    gate_out,
                    down_weight,
                    output,
                    m,
                    h,
                    inter,
                    self.e2m1_checkpoint_scales.map(|scales| scales.down),
                    e2m1_kmajor_requested,
                    stream,
                )?;
            }
            return Ok(());
        }

        if v2_fast_path {
            let gt = self.gate_proj_t.as_ref().unwrap();
            let ut = self.up_proj_t.as_ref().unwrap();
            let dt = self.down_proj_t.as_ref().unwrap();
            // gate_proj GEMM via 8-warp v2 kernel
            ops::w4a16_gemm_n128_m128_v2(
                ctx.gpu,
                self.w4a16_gemm_t_m128_v2,
                input,
                gt,
                gate_out,
                m,
                inter,
                h,
                stream,
            )?;
            ops::w4a16_gemm_n128_m128_v2(
                ctx.gpu,
                self.w4a16_gemm_t_m128_v2,
                input,
                ut,
                up_out,
                m,
                inter,
                h,
                stream,
            )?;
            ops::silu_mul(
                ctx.gpu,
                self.act_mul,
                gate_out,
                up_out,
                gate_out,
                m * inter,
                stream,
            )?;
            let output = ctx.buffers.moe_output();
            ops::w4a16_gemm_n128_m128_v2(
                ctx.gpu,
                self.w4a16_gemm_t_m128_v2,
                gate_out,
                dt,
                output,
                m,
                h,
                inter,
                stream,
            )?;
            return Ok(());
        }

        if fp8_fast_path {
            // Pre-dequanted FP8 weights — single sync per K-step in
            // the kernel, no DEQUANT phase. Same A×B GEMM math, just
            // bypassing the inner-loop NVFP4 → FP8 dequant.
            let gfp8 = self.gate_fp8.unwrap();
            let ufp8 = self.up_fp8.unwrap();
            let dfp8 = self.down_fp8.unwrap();

            // gate_proj GEMM: [M, H] BF16 × [inter, H] FP8 → [M, inter] BF16
            ops::fp8_gemm_n128_m128(
                ctx.gpu,
                self.fp8_gemm_t_m128_k,
                input,
                gfp8,
                gate_out,
                m,
                inter,
                h,
                stream,
            )?;
            ops::fp8_gemm_n128_m128(
                ctx.gpu,
                self.fp8_gemm_t_m128_k,
                input,
                ufp8,
                up_out,
                m,
                inter,
                h,
                stream,
            )?;
            ops::silu_mul(
                ctx.gpu,
                self.act_mul,
                gate_out,
                up_out,
                gate_out,
                m * inter,
                stream,
            )?;
            let output = ctx.buffers.moe_output();
            ops::fp8_gemm_n128_m128(
                ctx.gpu,
                self.fp8_gemm_t_m128_k,
                gate_out,
                dfp8,
                output,
                m,
                h,
                inter,
                stream,
            )?;
            return Ok(());
        }

        if fast_path {
            let gt = self.gate_proj_t.as_ref().unwrap();
            let ut = self.up_proj_t.as_ref().unwrap();
            let dt = self.down_proj_t.as_ref().unwrap();

            // gate_proj GEMM: [M, H] → [M, inter]
            ops::w4a16_gemm_n128_m128(
                ctx.gpu,
                self.w4a16_gemm_t_m128,
                input,
                gt,
                gate_out,
                m,
                inter,
                h,
                stream,
            )?;
            // up_proj GEMM: [M, H] → [M, inter]
            ops::w4a16_gemm_n128_m128(
                ctx.gpu,
                self.w4a16_gemm_t_m128,
                input,
                ut,
                up_out,
                m,
                inter,
                h,
                stream,
            )?;
            // activation(gate) * up for all M tokens (SiLU or GELU)
            ops::silu_mul(
                ctx.gpu,
                self.act_mul,
                gate_out,
                up_out,
                gate_out,
                m * inter,
                stream,
            )?;
            // down_proj GEMM: [M, inter] → [M, H]
            let output = ctx.buffers.moe_output();
            ops::w4a16_gemm_n128_m128(
                ctx.gpu,
                self.w4a16_gemm_t_m128,
                gate_out,
                dt,
                output,
                m,
                h,
                inter,
                stream,
            )?;
            log_prefill_ffn_fast_route_once(m, h, inter);
            return Ok(());
        }

        // Byte-exact cp.async pipelined path (ATLAS_PREFILL_FFN_PIPE=1):
        // `w4a16_gemm_pipe` is a load-pipelined byte-exact shadow of the
        // baseline M_TILE=64 kernel below (same B dequant arithmetic, same
        // m16n8k16 MMA sequence/accumulation order) — only the weight-load
        // pipeline differs. Small-M prefills pay a latency-bound fixed cost
        // per layer on the baseline kernel (~8ms/layer at M=18); the pipe
        // kernel overlaps the stream and should land near the ~190 GB/s the
        // verify-path kernels achieve. Default off; an eligible explicit
        // request with handle 0 fails before launching the gate projection.
        // `w4a16_gemm_pipe` is byte-exact ONLY when the K reduction is a
        // multiple of its 64-row stage; a partial final stage would load past
        // the row's packed-weight boundary and silently corrupt output. The
        // FFN shapes (h=5120, inter=17408) satisfy this, but guard explicitly
        // so any future topology change falls back to the exact baseline.
        let pipe_requested = crate::layers::prefill_ffn_pipe_enabled();
        let pipe_shape_eligible = h.is_multiple_of(64) && inter.is_multiple_of(64);
        if pipe_requested && pipe_shape_eligible && self.w4a16_gemm_pipe.0 == 0 {
            bail!(
                "ATLAS_PREFILL_FFN_PIPE requested for an eligible shape but w4a16_gemm_pipe is missing"
            );
        }
        let pipe_ready = pipe_requested && pipe_shape_eligible && self.w4a16_gemm_pipe.0 != 0;
        let dual_route = prefill_fused_epilogue_route(
            crate::layers::prefill_ffn_dual_fused_enabled()?,
            pipe_ready,
            self.w4a16_gemm_pipe_dual.0 != 0,
            self.activation,
            m,
            h,
            inter,
        );
        if dual_route == PrefillFusedEpilogueRoute::Missing {
            bail!(
                "ATLAS_PREFILL_FFN_DUAL_FUSED requested for an eligible shape but its parent pipe or dual kernel is missing"
            );
        }
        let dual_fused = dual_route == PrefillFusedEpilogueRoute::Complete;
        let fused_route = if dual_fused {
            // Dual is authoritative when both explicit candidate flags are set.
            PrefillFusedEpilogueRoute::Disabled
        } else {
            prefill_fused_epilogue_route(
                crate::layers::prefill_ffn_fused_epilogue_enabled(),
                pipe_ready,
                self.w4a16_gemm_pipe_silu_mul.0 != 0,
                self.activation,
                m,
                h,
                inter,
            )
        };
        if fused_route == PrefillFusedEpilogueRoute::Missing {
            bail!(
                "ATLAS_PREFILL_FFN_FUSED_EPILOGUE requested for an eligible shape but its parent pipe or fused kernel is missing"
            );
        }
        let fused_epilogue = fused_route == PrefillFusedEpilogueRoute::Complete;

        if pipe_ready {
            if !dual_fused && fused_epilogue {
                log_prefill_up_fused_route_once();
            } else if !dual_fused {
                log_prefill_pipe_route_once();
            }
            if dual_fused {
                ops::w4a16_gemm_pipe_dual(
                    ctx.gpu,
                    self.w4a16_gemm_pipe_dual,
                    input,
                    &self.weights.gate_proj,
                    &self.weights.up_proj,
                    gate_out,
                    DevicePtr::NULL,
                    true,
                    m,
                    inter,
                    h,
                    stream,
                )?;
                log_prefill_dual_fused_route_once();
            } else {
                ops::w4a16_gemm_pipe(
                    ctx.gpu,
                    self.w4a16_gemm_pipe,
                    input,
                    &self.weights.gate_proj,
                    gate_out,
                    m,
                    inter,
                    h,
                    stream,
                )?;
            }
            if !dual_fused && fused_epilogue {
                ops::w4a16_gemm_pipe_silu_mul(
                    ctx.gpu,
                    self.w4a16_gemm_pipe_silu_mul,
                    input,
                    &self.weights.up_proj,
                    gate_out,
                    m,
                    inter,
                    h,
                    stream,
                )?;
            } else if !dual_fused {
                ops::w4a16_gemm_pipe(
                    ctx.gpu,
                    self.w4a16_gemm_pipe,
                    input,
                    &self.weights.up_proj,
                    up_out,
                    m,
                    inter,
                    h,
                    stream,
                )?;
                ops::silu_mul(
                    ctx.gpu,
                    self.act_mul,
                    gate_out,
                    up_out,
                    gate_out,
                    m * inter,
                    stream,
                )?;
            }
            let output = ctx.buffers.moe_output();
            ops::w4a16_gemm_pipe(
                ctx.gpu,
                self.w4a16_gemm_pipe,
                gate_out,
                &self.weights.down_proj,
                output,
                m,
                h,
                inter,
                stream,
            )?;
            return Ok(());
        }

        // Baseline M_TILE=64 path (unchanged).
        // gate_proj GEMM: [M, H] → [M, inter]
        ops::w4a16_gemm(
            ctx.gpu,
            self.w4a16_gemm,
            input,
            &self.weights.gate_proj,
            gate_out,
            m,
            inter,
            h,
            stream,
        )?;

        // up_proj GEMM: [M, H] → [M, inter]
        ops::w4a16_gemm(
            ctx.gpu,
            self.w4a16_gemm,
            input,
            &self.weights.up_proj,
            up_out,
            m,
            inter,
            h,
            stream,
        )?;

        // activation(gate) * up for all M tokens (SiLU or GELU)
        ops::silu_mul(
            ctx.gpu,
            self.act_mul,
            gate_out,
            up_out,
            gate_out,
            m * inter,
            stream,
        )?;

        // down_proj GEMM: [M, inter] → [M, H]
        let output = ctx.buffers.moe_output();
        ops::w4a16_gemm(
            ctx.gpu,
            self.w4a16_gemm,
            gate_out,
            &self.weights.down_proj,
            output,
            m,
            h,
            inter,
            stream,
        )?;

        Ok(())
    }

    /// Batched forward (delegates to `forward_prefill`). Used by `forward_batched` in model loop.
    pub fn forward_batched(
        &self,
        input: DevicePtr,
        num_tokens: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        self.forward_prefill(input, num_tokens, ctx, stream)
    }
}

#[cfg(test)]
#[path = "dense_ffn/exact_route_tests.rs"]
mod exact_route_tests;

#[cfg(test)]
#[path = "dense_ffn/w3_route_tests.rs"]
mod w3_route_tests;

#[cfg(test)]
#[path = "dense_ffn/memlean_tests.rs"]
mod memlean_tests;

#[cfg(test)]
mod flashinfer_merged_ffn_tests {
    use super::{
        FlashinferFfnPrefillRoute as Route, e2m1_scratch_layout, flashinfer_ffn_prefill_route,
    };
    #[cfg(all(feature = "cuda", target_os = "linux"))]
    use super::{
        ensure_flashinfer_ffn_disjoint, flashinfer_ffn_activation_scale_extent,
        flashinfer_ffn_extent,
    };
    #[cfg(all(feature = "cuda", target_os = "linux"))]
    use spark_runtime::gpu::{DevicePtr, KernelHandle, mock::MockGpuBackend};

    #[test]
    fn explicit_route_is_exact_and_missing_is_fail_closed() {
        // `m_qualified` is now supplied by the caller (see the fn doc): the
        // frozen pair 2079/8192 qualifies, 2080 does not.
        let q = |m: u32| matches!(m, 2_079 | 8_192);
        assert_eq!(
            flashinfer_ffn_prefill_route(false, true, q(2_079), false),
            Route::Disabled
        );
        assert_eq!(
            flashinfer_ffn_prefill_route(true, false, q(2_079), true),
            Route::Ineligible
        );
        assert_eq!(
            flashinfer_ffn_prefill_route(true, true, q(2_080), false),
            Route::Ineligible
        );
        for m in [2_079u32, 8_192] {
            assert_eq!(
                flashinfer_ffn_prefill_route(true, true, q(m), false),
                Route::Missing
            );
            assert_eq!(
                flashinfer_ffn_prefill_route(true, true, q(m), true),
                Route::Complete
            );
        }
    }

    /// The extended ladder must stay OFF by default: with the env unset, an M
    /// that is only in `QWEN38_FFN_EXTRA_M` must still be rejected, so default
    /// behaviour is bit-for-bit the frozen table.
    #[test]
    fn extended_m_ladder_is_off_unless_requested() {
        use crate::weight_map::flashinfer_ffn_admission::{
            Qwen38FfnOperation, qwen38_ffn_extra_m_enabled, qwen38_ffn_launch_candidate,
        };
        if qwen38_ffn_extra_m_enabled() {
            return; // caller opted in; the negative assertion does not apply
        }
        for m in [1_024usize, 2_048, 4_096, 6_144, 7_168, 8_187] {
            assert!(
                qwen38_ffn_launch_candidate(Qwen38FfnOperation::MergedGateUp, m).is_err(),
                "M={m} must be unqualified while ATLAS_FLASHINFER_FFN_EXTRA_M is unset"
            );
        }
    }

    #[test]
    fn construction_scratch_layout_matches_the_charged_maximum() {
        let layout = e2m1_scratch_layout(8_192, 17_408).unwrap();
        assert_eq!(layout.cap_m, 8_192);
        assert_eq!(layout.cap_k, 17_408);
        assert_eq!(layout.packed_bytes, 71_303_168);
        assert_eq!(layout.scale_bytes, 8_912_896);
        assert_eq!(layout.max_bytes, 4);
        assert_eq!(
            layout.packed_bytes + layout.scale_bytes + layout.max_bytes,
            80_216_068
        );
        assert!(e2m1_scratch_layout(usize::MAX, 17_408).is_err());
        assert!(e2m1_scratch_layout(8_192, usize::MAX - 15).is_err());
    }

    #[test]
    fn production_order_preflights_then_merges_directly_quantizes_and_completes_down() {
        let source = include_str!("dense_ffn.rs");
        let start = source
            .find("fn try_forward_flashinfer_prefill(")
            .expect("FlashInfer FFN runtime route");
        let end = source[start..]
            .find("/// Install W3")
            .map(|offset| start + offset)
            .expect("FlashInfer FFN runtime boundary");
        let body = &source[start..end];

        let missing = body.find("FlashinferFfnPrefillRoute::Missing").unwrap();
        let down_preflight = body.find("let mut down_launch").unwrap();
        let scratch_guard = body.find("self.lock_e2m1_use(stream)").unwrap();
        let physical_scale_extent = body
            .find("flashinfer_ffn_activation_scale_extent(rows, QWEN38_INTERMEDIATE)")
            .unwrap();
        let quantize = body
            .find("ops::quantize_bf16_to_nvfp4_atlas_128x4(")
            .unwrap();
        let merged = body.find("merged_launch.launch_eager(").unwrap();
        let direct_silu_quantize = body
            .find("ops::quantize_merged_silu_mul_bf16_to_nvfp4_atlas_128x4(")
            .unwrap();
        let down = body.find("down_launch.launch_eager(").unwrap();
        let receipt = body.find("log_flashinfer_ffn_success(").unwrap();
        assert!(
            missing < physical_scale_extent
                && physical_scale_extent < down_preflight
                && down_preflight < scratch_guard
                && scratch_guard < quantize
        );
        assert!(quantize < merged && merged < direct_silu_quantize && direct_silu_quantize < down);
        assert!(down < receipt);
        assert!(!body.contains("ops::silu_mul("));
        assert!(!body.contains("KernelLaunch::new(ctx.gpu, route.split_gate_up_k)"));
        assert!(!body.contains("ops::quantize_silu_mul_bf16_to_nvfp4_atlas_128x4("));
        assert!(body.contains("f32::from_bits(route.down.input_scale_bits)"));
        assert!(body.contains("ctx.buffers.sizes().ffn_gate_up_bf16 >= merged_bytes"));
        assert!(body.contains("ctx.buffers.ffn_gate_up_bf16()"));
        assert!(!body.contains("Qwen38FfnOperation::Gate, m as usize"));
        assert!(!body.contains("Qwen38FfnOperation::Up, m as usize"));
        assert!(!body.contains("route.stream"));
        assert!(body.contains("prepare_borrowed_zero_workspace(ctx.gpu, merged_shape, stream)"));
        assert!(body.contains("prepare_borrowed_zero_workspace(ctx.gpu, down_shape, stream)"));
    }

    #[test]
    fn construction_stream_is_preflight_only_and_runtime_owns_launch_stream() {
        let source = include_str!("dense_ffn.rs");
        let struct_start = source.find("struct FlashinferFfnPrefill {").unwrap();
        let struct_end = source[struct_start..]
            .find("\n}")
            .map(|offset| struct_start + offset)
            .unwrap();
        assert!(!source[struct_start..struct_end].contains("stream: u64"));

        let prepare_start = source.find("pub fn prepare_flashinfer_prefill(").unwrap();
        let runtime_start = source.find("fn try_forward_flashinfer_prefill(").unwrap();
        let prepare = &source[prepare_start..runtime_start];
        assert!(prepare.contains("let stream = gpu.default_stream();"));
        assert!(!prepare.contains("stream,\n        });"));

        let runtime_end = source[runtime_start..]
            .find("/// Install W3")
            .map(|offset| runtime_start + offset)
            .unwrap();
        let runtime = &source[runtime_start..runtime_end];
        assert!(!runtime.contains("stream == route.stream"));
        assert!(runtime.contains("self.lock_e2m1_use(stream)"));

        let lock_start = source.find("fn lock_e2m1_use(").unwrap();
        let lock_end = source[lock_start..]
            .find("/// Quantize one BF16 activation matrix")
            .map(|offset| lock_start + offset)
            .unwrap();
        let lock = &source[lock_start..lock_end];
        assert!(lock.contains(".e2m1_use_stream"));
        assert!(lock.contains(".lock()"));
        assert!(lock.contains(".map_err("));
        assert!(!lock.contains(".unwrap()"));
    }

    #[test]
    fn split_kernel_is_exact_row_major_gate_then_up() {
        let cuda = include_str!("../../../../kernels/gb10/common/flashinfer_projection_split.cu");
        let start = cuda
            .find("void flashinfer_projection_split_ffn_gate_up(")
            .expect("merged FFN split kernel");
        let body = &cuda[start..];
        assert!(body.contains("constexpr uint32_t INTERMEDIATE = 17408"));
        assert!(body.contains("constexpr uint32_t TOTAL = 2 * INTERMEDIATE"));
        assert!(body.contains("merged + row * TOTAL"));
        assert!(body.contains("gate_dst[vector] = src[vector]"));
        assert!(body.contains("up_dst[vector] = src[VECTORS + vector]"));
    }

    #[cfg(all(feature = "cuda", target_os = "linux"))]
    #[test]
    fn physical_activation_scale_extent_is_exact_for_qualified_rows() {
        assert_eq!(
            flashinfer_ffn_activation_scale_extent(2_079, 17_408).unwrap(),
            2_367_488
        );
        assert_eq!(
            flashinfer_ffn_activation_scale_extent(8_192, 17_408).unwrap(),
            8_912_896
        );
    }

    #[cfg(all(feature = "cuda", target_os = "linux"))]
    #[test]
    fn physical_activation_scale_extent_rejects_padding_overflow() {
        assert!(flashinfer_ffn_activation_scale_extent(usize::MAX, 17_408).is_err());
    }

    #[cfg(all(feature = "cuda", target_os = "linux"))]
    #[test]
    fn physical_tail_overlap_is_rejected_before_launch() {
        let rows = 2_079;
        let cols = 17_408;
        let logical_bytes = flashinfer_ffn_extent(rows, cols / 16, 1).unwrap();
        let physical_bytes = flashinfer_ffn_activation_scale_extent(rows, cols).unwrap();
        assert_eq!(physical_bytes - logical_bytes, 105_536);

        let scales = DevicePtr(0x1_0000);
        let destination = DevicePtr(scales.0 + u64::try_from(logical_bytes).unwrap());
        assert!(
            ensure_flashinfer_ffn_disjoint(scales, logical_bytes, destination, 16).is_ok(),
            "the hostile destination begins exactly after the logical prefix"
        );
        assert!(
            ensure_flashinfer_ffn_disjoint(scales, physical_bytes, destination, 16).is_err(),
            "the padded physical tail must reject the same destination before launch"
        );
    }

    #[cfg(all(feature = "cuda", target_os = "linux"))]
    #[test]
    fn fused_physical_wrapper_accepts_exact_rows_and_rejects_hostile_inputs_before_launch() {
        let gpu = MockGpuBackend::new();
        let gate = DevicePtr(0x0010_0000);
        let up = DevicePtr(0x1000_0000);
        let packed = DevicePtr(0x2000_0000);
        let scales = DevicePtr(0x3000_0000);
        for rows in [2_079, 8_192] {
            super::ops::quantize_silu_mul_bf16_to_nvfp4_atlas_128x4(
                &gpu,
                KernelHandle(0x55),
                gate,
                up,
                packed,
                scales,
                0.25,
                rows,
                17_408,
                0x77,
            )
            .unwrap();
        }
        assert_eq!(gpu.launch_count(), 2);

        for (kernel, gate, up, packed, scales, scale2, rows, cols) in [
            (
                KernelHandle(0),
                gate,
                up,
                packed,
                scales,
                0.25,
                2_079,
                17_408,
            ),
            (
                KernelHandle(1),
                DevicePtr::NULL,
                up,
                packed,
                scales,
                0.25,
                2_079,
                17_408,
            ),
            (
                KernelHandle(1),
                gate,
                DevicePtr::NULL,
                packed,
                scales,
                0.25,
                2_079,
                17_408,
            ),
            (
                KernelHandle(1),
                gate,
                up,
                DevicePtr::NULL,
                scales,
                0.25,
                2_079,
                17_408,
            ),
            (
                KernelHandle(1),
                gate,
                up,
                packed,
                DevicePtr::NULL,
                0.25,
                2_079,
                17_408,
            ),
            (
                KernelHandle(1),
                gate,
                up,
                packed,
                scales,
                f32::NAN,
                2_079,
                17_408,
            ),
            (KernelHandle(1), gate, up, packed, scales, 0.25, 0, 17_408),
            (KernelHandle(1), gate, up, packed, scales, 0.25, 2_079, 16),
            (
                KernelHandle(1),
                gate,
                up,
                packed,
                scales,
                0.25,
                u32::MAX,
                17_408,
            ),
        ] {
            assert!(
                super::ops::quantize_silu_mul_bf16_to_nvfp4_atlas_128x4(
                    &gpu, kernel, gate, up, packed, scales, scale2, rows, cols, 0x77,
                )
                .is_err()
            );
        }
        assert_eq!(gpu.launch_count(), 2);
    }

    #[cfg(all(feature = "cuda", target_os = "linux"))]
    #[test]
    fn direct_merged_wrapper_is_abi_separate_and_fails_before_launch() {
        let gpu = MockGpuBackend::new();
        let merged = DevicePtr(0x0010_0000);
        let packed = DevicePtr(0x2000_0000);
        let scales = DevicePtr(0x3000_0000);
        for rows in [2_079, 8_192] {
            super::ops::quantize_merged_silu_mul_bf16_to_nvfp4_atlas_128x4(
                &gpu,
                KernelHandle(0x66),
                merged,
                packed,
                scales,
                0.25,
                rows,
                17_408,
                0x77,
            )
            .unwrap();
        }
        assert_eq!(gpu.launch_count(), 2);

        for (kernel, merged, packed, scales, scale2, rows, cols) in [
            (KernelHandle(0), merged, packed, scales, 0.25, 2_079, 17_408),
            (
                KernelHandle(1),
                DevicePtr::NULL,
                packed,
                scales,
                0.25,
                2_079,
                17_408,
            ),
            (
                KernelHandle(1),
                merged,
                DevicePtr::NULL,
                scales,
                0.25,
                2_079,
                17_408,
            ),
            (
                KernelHandle(1),
                merged,
                packed,
                DevicePtr::NULL,
                0.25,
                2_079,
                17_408,
            ),
            (
                KernelHandle(1),
                merged,
                packed,
                scales,
                f32::NAN,
                2_079,
                17_408,
            ),
            (KernelHandle(1), merged, packed, scales, 0.25, 0, 17_408),
            (KernelHandle(1), merged, packed, scales, 0.25, 2_079, 16),
            (
                KernelHandle(1),
                merged,
                packed,
                scales,
                0.25,
                u32::MAX,
                17_408,
            ),
        ] {
            assert!(
                super::ops::quantize_merged_silu_mul_bf16_to_nvfp4_atlas_128x4(
                    &gpu, kernel, merged, packed, scales, scale2, rows, cols, 0x77,
                )
                .is_err()
            );
        }
        assert_eq!(gpu.launch_count(), 2);
    }

    #[test]
    fn fused_physical_symbol_layout_abi_handle_and_receipt_are_source_bound() {
        let cuda = include_str!("../../../../kernels/gb10/common/quantize_bf16_to_nvfp4.cu");
        let logical = cuda
            .find("void quantize_silu_mul_bf16_to_nvfp4(")
            .expect("existing logical W4A4 fused symbol");
        let physical = cuda
            .find("void quantize_silu_mul_bf16_to_nvfp4_atlas_128x4(")
            .expect("ABI-separated physical fused symbol");
        assert!(logical < physical);
        let physical_body = &cuda[physical..];
        assert!(physical_body.contains("((N + 127u) / 128u) * 128u"));
        assert!(physical_body.contains("if (row >= N)"));
        assert!(physical_body.contains("silu_scale_offset_128x4(row, group, num_groups)] = 0"));
        assert!(physical_body.contains("silu_mul_bf16_round(gate, up"));
        assert!(physical_body.contains("float_to_fp8_e4m3(fp8_float)"));
        assert!(physical_body.contains("quantize_e2m1(v0)"));

        let direct = cuda
            .find("void quantize_merged_silu_mul_bf16_to_nvfp4_atlas_128x4(")
            .expect("ABI-separated direct merged physical symbol");
        assert!(physical < direct);
        let direct_body = &cuda[direct..];
        assert!(direct_body.contains("static_cast<unsigned long long>(row) * (2ull * K)"));
        assert!(direct_body.contains("const __nv_bfloat16* up = gate + K"));
        assert!(direct_body.contains("float rounded[GROUP_SIZE]"));
        assert_eq!(
            direct_body.matches("silu_mul_bf16_round(gate, up").count(),
            1
        );
        assert!(direct_body.contains("rounded[i] = value"));
        assert!(direct_body.contains("quantize_e2m1(rounded[i] * inv_eff)"));
        assert!(direct_body.contains("silu_scale_offset_128x4(row, group, num_groups)] = 0"));

        let wrapper_source = include_str!("ops/gemm_dense.rs");
        let wrapper_start = wrapper_source
            .find("pub fn quantize_merged_silu_mul_bf16_to_nvfp4_atlas_128x4(")
            .expect("direct merged physical Rust wrapper");
        let wrapper_end = wrapper_source[wrapper_start..]
            .find("/// Native NVFP4")
            .map(|offset| wrapper_start + offset)
            .expect("physical fused Rust wrapper boundary");
        let wrapper = &wrapper_source[wrapper_start..wrapper_end];
        let merged = wrapper.find(".arg_ptr(merged_gate_up)").unwrap();
        let packed = wrapper.find(".arg_ptr(packed_out)").unwrap();
        let scales = wrapper.find(".arg_ptr(scale_out)").unwrap();
        let scale2 = wrapper.find(".arg_f32(scale2)").unwrap();
        let rows = wrapper.find(".arg_u32(rows)").unwrap();
        let cols = wrapper.find(".arg_u32(cols)").unwrap();
        assert!(merged < packed && packed < scales && scales < scale2);
        assert!(scale2 < rows && rows < cols);
        assert!(wrapper.contains("cols >= 64 && cols.is_multiple_of(64)"));
        assert!(wrapper.contains(".checked_add(127)"));
        assert!(wrapper.contains(".grid([quantize_128x4_grid_x(padded_rows), 1, 1])"));

        let source = include_str!("dense_ffn.rs");
        let prepare_start = source.find("pub fn prepare_flashinfer_prefill(").unwrap();
        let prepare_end = source[prepare_start..]
            .find("fn try_forward_flashinfer_prefill(")
            .map(|offset| prepare_start + offset)
            .unwrap();
        let prepare = &source[prepare_start..prepare_end];
        let symbol = prepare
            .find("\"quantize_merged_silu_mul_bf16_to_nvfp4_atlas_128x4\"")
            .unwrap();
        let nonzero = prepare
            .find("quantize_merged_silu_atlas_128x4_k.0 != 0")
            .unwrap();
        let plan_freeze = prepare.find("for (operation, m) in [").unwrap();
        let scratch_preallocation = prepare
            .find("e2m1_scratch_layout(8_192, intermediate)")
            .unwrap();
        let scratch_capacity_check = prepare
            .find("retained.cap_m == scratch_layout.cap_m")
            .unwrap();
        let retained_copy = prepare.find("build_flashinfer_merged_gate_up(").unwrap();
        let route_retention = prepare.find("self.flashinfer_ffn_prefill = Some(").unwrap();
        assert!(
            symbol < nonzero
                && nonzero < plan_freeze
                && plan_freeze < scratch_preallocation
                && scratch_preallocation < scratch_capacity_check
                && scratch_capacity_check < retained_copy
                && retained_copy < route_retention
        );
        assert!(prepare.contains("\"quantize_nvfp4\""));
        assert!(prepare.contains("quantize_merged_silu_atlas_128x4_k,"));
        assert!(!prepare.contains("\"flashinfer_projection_split\""));
    }
}

#[cfg(test)]
mod prefill_ffn_fast_route_tests {
    use super::{PrefillFfnFastRoute as Route, prefill_ffn_fast_route};

    #[test]
    fn distinguishes_disabled_ineligible_complete_and_every_missing_dependency() {
        assert_eq!(
            prefill_ffn_fast_route(false, true, true, true, true),
            Route::Disabled
        );
        assert_eq!(
            prefill_ffn_fast_route(true, false, false, false, false),
            Route::Ineligible
        );
        assert_eq!(
            prefill_ffn_fast_route(true, true, true, true, true),
            Route::Complete
        );
        for readiness in [
            (false, true, true),
            (true, false, true),
            (true, true, false),
        ] {
            assert_eq!(
                prefill_ffn_fast_route(true, true, readiness.0, readiness.1, readiness.2),
                Route::Missing
            );
        }
    }

    #[test]
    fn missing_fails_before_projection_and_marker_follows_completed_down_projection() {
        let source = include_str!("dense_ffn.rs");
        let select = source.find("let fast_route =").expect("FAST selector");
        let branch = source[select..]
            .find("if fast_path {")
            .map(|offset| select + offset)
            .expect("FAST projection branch");
        let selection = &source[select..branch];
        assert!(selection.contains("PrefillFfnFastRoute::Missing"));
        assert!(selection.contains("before gate projection"));

        let end = source[branch..]
            .find("// Byte-exact cp.async pipelined path")
            .map(|offset| branch + offset)
            .expect("FAST projection boundary");
        let body = &source[branch..end];
        let down = body
            .rfind("ops::w4a16_gemm_n128_m128(")
            .expect("FAST down projection");
        let marker = body
            .find("log_prefill_ffn_fast_route_once(m, h, inter);")
            .expect("FAST completion marker");
        let returned = body.rfind("return Ok(());").expect("FAST return");
        assert!(down < marker && marker < returned);
        assert!(source.contains(
            "ENGAGED ATLAS_PREFILL_FFN_FAST: approximate transposed M128 route complete"
        ));
    }

    #[test]
    fn prefill_only_copies_waive_m16_for_fast_route_but_not_has_transposed() {
        let source = include_str!("dense_ffn.rs");
        let select = source.find("let fast_route =").expect("FAST selector");
        let selection = &source[select..select + 400];
        assert!(selection.contains("self.w4a16_gemm_t_m16.0 != 0 || self.prefill_t_m128_only,"));

        let has_t = source
            .find("pub fn has_transposed_ffn(&self) -> bool {")
            .expect("has_transposed_ffn");
        let body = &source[has_t..has_t + 300];
        assert!(body.contains("&& self.w4a16_gemm_t_m16.0 != 0"));
        assert!(!body.contains("prefill_t_m128_only"));
        // Leading indent + newline so this test's own literal does not match.
        let setter = "        self.prefill_t_m128_only = true;\n";
        assert_eq!(source.matches(setter).count(), 1);
    }
}

#[cfg(test)]
mod prefill_fused_epilogue_tests {
    use super::{FfnActivation, PrefillFusedEpilogueRoute, prefill_fused_epilogue_route};

    #[test]
    fn route_distinguishes_disabled_ineligible_complete_and_missing() {
        assert_eq!(
            prefill_fused_epilogue_route(false, true, true, FfnActivation::SiLU, 8192, 5120, 17408,),
            PrefillFusedEpilogueRoute::Disabled
        );
        for (activation, m, h, inter) in [
            (FfnActivation::GeLU, 8192, 5120, 17408),
            (FfnActivation::SiLU, 32, 5120, 17408),
            (FfnActivation::SiLU, 8192, 5119, 17408),
            (FfnActivation::SiLU, 8192, 5120, 17407),
        ] {
            assert_eq!(
                prefill_fused_epilogue_route(true, true, true, activation, m, h, inter),
                PrefillFusedEpilogueRoute::Ineligible
            );
        }
        assert_eq!(
            prefill_fused_epilogue_route(true, true, true, FfnActivation::SiLU, 33, 5120, 17408,),
            PrefillFusedEpilogueRoute::Complete
        );
        assert_eq!(
            prefill_fused_epilogue_route(true, false, true, FfnActivation::SiLU, 8192, 5120, 17408,),
            PrefillFusedEpilogueRoute::Missing
        );
        assert_eq!(
            prefill_fused_epilogue_route(true, true, false, FfnActivation::SiLU, 8192, 5120, 17408,),
            PrefillFusedEpilogueRoute::Missing
        );
    }

    #[test]
    fn source_fails_before_projection_and_logs_each_engaged_route() {
        let source = include_str!("dense_ffn.rs");
        let dispatch_start = source.find("let pipe_requested =").unwrap();
        let dispatch_end = source[dispatch_start..]
            .find("// Baseline M_TILE=64 path")
            .map(|offset| dispatch_start + offset)
            .unwrap();
        let dispatch = &source[dispatch_start..dispatch_end];
        assert!(dispatch.contains(
            "ATLAS_PREFILL_FFN_PIPE requested for an eligible shape but w4a16_gemm_pipe is missing"
        ));
        assert!(dispatch.contains(
            "ATLAS_PREFILL_FFN_DUAL_FUSED requested for an eligible shape but its parent pipe or dual kernel is missing"
        ));
        assert!(dispatch.contains(
            "ATLAS_PREFILL_FFN_FUSED_EPILOGUE requested for an eligible shape but its parent pipe or fused kernel is missing"
        ));
        assert!(
            dispatch.contains("Dual is authoritative when both explicit candidate flags are set")
        );
        let log_start = source.find("fn log_prefill_pipe_route_once").unwrap();
        let log_end = source[log_start..]
            .find("enum ExactFfnDispatch")
            .map(|offset| log_start + offset)
            .unwrap();
        let loggers = &source[log_start..log_end];
        assert!(loggers.contains("ENGAGED ATLAS_PREFILL_FFN_PIPE"));
        assert!(loggers.contains("ENGAGED ATLAS_PREFILL_FFN_FUSED_EPILOGUE"));
        assert!(loggers.contains("ENGAGED ATLAS_PREFILL_FFN_DUAL_FUSED"));

        let dual_launch = dispatch.find("ops::w4a16_gemm_pipe_dual(").unwrap();
        let dual_marker = dispatch
            .find("log_prefill_dual_fused_route_once();")
            .unwrap();
        assert!(dual_launch < dual_marker);
    }

    #[test]
    fn legacy_boolean_silent_fallback_is_absent() {
        let source = include_str!("dense_ffn.rs");
        let legacy_name = ["fn use_prefill", "_fused_epilogue("].concat();
        assert!(!source.contains(&legacy_name));
    }

    #[test]
    fn cuda_source_retains_bf16_round_trip_and_in_place_abi() {
        let source = include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../kernels/gb10/qwen3.8-27b/nvfp4/w4a16_gemm.cu"
        ));
        let start = source
            .find("extern \"C\" __global__ void w4a16_gemm_pipe_silu_mul")
            .expect("fused prefill epilogue kernel is missing");
        let body = &source[start..];
        assert!(body.contains("__nv_bfloat16 up_bf16 = __float2bfloat16(value)"));
        assert!(body.contains("float u = __bfloat162float(up_bf16)"));
        assert!(body.contains("float g = __bfloat162float(gate_in_out[out_idx])"));
        assert!(body.contains("gate_in_out[out_idx] = __float2bfloat16(g * sigmoid_g * u)"));
    }

    #[test]
    fn dual_cuda_source_retains_both_bf16_round_trips_and_independent_weights() {
        let source = include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../kernels/gb10/qwen3.8-27b/nvfp4/w4a16_gemm.cu"
        ));
        let start = source
            .find("extern \"C\" __global__ void w4a16_gemm_pipe_dual")
            .expect("dual fused prefill kernel is missing");
        let body = &source[start..];
        assert!(body.contains("const unsigned char* __restrict__ gate_packed"));
        assert!(body.contains("const unsigned char* __restrict__ up_packed"));
        assert!(body.contains("__nv_bfloat16* C1"));
        assert!(body.contains("if (fuse_silu != 0)"));
        assert!(body.contains("__bfloat162float(__float2bfloat16(gate_value))"));
        assert!(body.contains("__bfloat162float(__float2bfloat16(up_value))"));
        assert!(body.contains("__float2bfloat16(g * sigmoid_g * u)"));
        assert!(body.contains("C0[out_idx] = __float2bfloat16(g)"));
        assert!(body.contains("C1[out_idx] = __float2bfloat16(u)"));
    }

    #[test]
    fn dual_rust_abi_matches_cuda_and_fused_call_uses_null_second_output() {
        let ops = include_str!("ops/gemm_dense.rs");
        let start = ops
            .find("pub fn w4a16_gemm_pipe_dual(")
            .expect("dual Rust launcher is missing");
        let body = &ops[start..];
        let ordered = [
            ".arg_ptr(input)",
            ".arg_ptr(gate.weight)",
            ".arg_ptr(gate.weight_scale)",
            ".arg_f32(gate.weight_scale_2)",
            ".arg_ptr(up.weight)",
            ".arg_ptr(up.weight_scale)",
            ".arg_f32(up.weight_scale_2)",
            ".arg_ptr(output_first)",
            ".arg_ptr(output_second)",
            ".arg_u32(u32::from(fuse_silu))",
            ".arg_u32(m)",
            ".arg_u32(n)",
            ".arg_u32(k)",
        ];
        let mut cursor = 0;
        for argument in ordered {
            let relative = body[cursor..]
                .find(argument)
                .unwrap_or_else(|| panic!("missing or misordered dual argument {argument}"));
            cursor += relative + argument.len();
        }

        let host = include_str!("dense_ffn.rs");
        let dispatch_start = host.find("let pipe_requested =").unwrap();
        let dispatch = &host[dispatch_start..];
        let call = dispatch.find("ops::w4a16_gemm_pipe_dual(").unwrap();
        let call_body = &dispatch[call..];
        let null_output = call_body.find("DevicePtr::NULL").unwrap();
        let fused = call_body.find("true,").unwrap();
        let launch_end = call_body.find(")?;").unwrap();
        assert!(null_output < fused && fused < launch_end);
    }
}

#[cfg(test)]
mod e2m1_prefill_reuse_tests {
    #[cfg(feature = "cuda")]
    use super::swizzle_nvfp4_scales_128x4_host;
    use super::{
        CublasLtNvfp4Policy, E2M1_KMAJOR_M256_MIN_M, E2m1KmajorKernel, E2m1PrefillRoute, E2m1Scope,
        E2m1SelectedKernel, FlashInferFfnAvailability, Sm120CoalescedSmallMRoute,
        cublaslt_nvfp4_route_enabled, e2m1_kmajor_kernel, e2m1_kmajor_projection_shape,
        e2m1_prefill_route, e2m1_selected_kernel, is_blackwell_dc_compute_capability,
        is_sm120_compute_capability, parse_cublaslt_nvfp4_policy,
        select_sm120_coalesced_small_m_route, supports_warp_nvfp4_mma,
        validated_e2m1_checkpoint_scales,
    };

    #[test]
    fn base_w4a4_selector_is_full_authoritative_and_fail_closed() {
        assert_eq!(
            e2m1_prefill_route(false, false, true, false, false),
            E2m1PrefillRoute::Disabled
        );
        assert_eq!(
            e2m1_prefill_route(true, false, false, false, false),
            E2m1PrefillRoute::Ineligible
        );
        assert_eq!(
            e2m1_prefill_route(true, false, true, true, false),
            E2m1PrefillRoute::Complete(E2m1Scope::Full)
        );
        assert_eq!(
            e2m1_prefill_route(false, true, true, false, true),
            E2m1PrefillRoute::Complete(E2m1Scope::DownOnly)
        );
        assert_eq!(
            e2m1_prefill_route(true, false, true, false, true),
            E2m1PrefillRoute::Missing(E2m1Scope::Full)
        );
        assert_eq!(
            e2m1_prefill_route(false, true, true, true, false),
            E2m1PrefillRoute::Missing(E2m1Scope::DownOnly)
        );
        assert_eq!(
            e2m1_prefill_route(true, true, true, false, true),
            E2m1PrefillRoute::Missing(E2m1Scope::Full),
            "full must never downgrade to a ready down-only route"
        );
    }

    #[test]
    fn selected_w4a4_kernel_names_the_actual_m2048_boundary() {
        assert_eq!(
            e2m1_selected_kernel(false, true, 8192),
            E2m1SelectedKernel::RowMajorM64
        );
        assert_eq!(
            e2m1_selected_kernel(true, true, 2047),
            E2m1SelectedKernel::KmajorM128
        );
        assert_eq!(
            e2m1_selected_kernel(true, true, 2048),
            E2m1SelectedKernel::KmajorM256
        );
        assert_eq!(
            e2m1_selected_kernel(true, false, 8192),
            E2m1SelectedKernel::KmajorM128
        );
    }

    #[test]
    fn base_w4a4_missing_route_precedes_projection_and_marker_is_actual_only() {
        let source = include_str!("dense_ffn.rs");
        let select = source.find("let e2m1_route =").expect("W4A4 selector");
        let projection = source[select..]
            .find("if e2m1_fast_path {")
            .map(|offset| select + offset)
            .expect("W4A4 projection");
        let selection = &source[select..projection];
        assert!(selection.contains("E2m1PrefillRoute::Missing(E2m1Scope::Full)"));
        assert!(selection.contains("E2m1PrefillRoute::Missing(E2m1Scope::DownOnly)"));
        assert!(selection.contains("before gate projection"));

        let log_start = source.find("fn log_e2m1_route_once").expect("W4A4 logger");
        // Inspect this logger, not unrelated helpers inserted after it.
        // Inner blocks are indented; this is its top-level closing brace.
        let log_end = source[log_start..]
            .find("\n}\n")
            .map(|offset| log_start + offset + 3)
            .expect("W4A4 logger boundary");
        let logger = &source[log_start..log_end];
        assert!(logger.contains("ENGAGED ATLAS_E2M1_PREFILL"));
        for field in [
            "scope={scope}",
            "gate_up={gate_up}",
            "e2m1_kernel={kernel}",
            "activation_scale={activation_scale}",
            "down_input={down_input}",
            "activation={activation}",
            "M={}",
            "H={}",
            "intermediate={}",
        ] {
            assert!(logger.contains(field), "missing actual field {field}");
        }
        assert!(!logger.contains("requested"));
        assert!(!logger.contains("has_e2m1"));
    }

    #[test]
    fn kmajor_m128_accepts_only_native_ffn_tile_geometry() {
        assert!(e2m1_kmajor_projection_shape(8192, 17408, 5120));
        assert!(e2m1_kmajor_projection_shape(8192, 5120, 17408));
        assert!(!e2m1_kmajor_projection_shape(127, 17408, 5120));
        assert!(!e2m1_kmajor_projection_shape(128, 17409, 5120));
        assert!(!e2m1_kmajor_projection_shape(128, 17408, 5119));
    }

    #[test]
    fn kmajor_kernel_stages_transformed_weights_for_eight_row_warps() {
        let source =
            include_str!("../../../../kernels/gb10/qwen3.8-27b/nvfp4/cutlass_nvfp4_gemm.cu");
        let start = source
            .find("void nvfp4_nvfp4_gemm_kmajor_m128(")
            .expect("K-major W4A4 kernel is missing");
        let end = source[start..]
            .find("// K-major M256 shadow.")
            .map(|offset| start + offset)
            .expect("K-major kernel boundary is missing");
        let body = &source[start..end];

        assert!(source.contains("__launch_bounds__(256)\nvoid nvfp4_nvfp4_gemm_kmajor_m128"));
        assert!(body.contains("const unsigned char* __restrict__ B_packed_t"));
        assert!(body.contains("const unsigned char* __restrict__ B_scale_t"));
        assert!(body.contains("B_packed_t[(unsigned long long)gk_byte * N + gn]"));
        assert!(body.contains("B_scale_t[(unsigned long long)gg * N + gn]"));
        assert!(body.contains("smem_Bp_km[2][N_TILE_LG][K_STEP / 2]"));
        assert!(body.contains("smem_Bs_km[2][N_TILE_LG][K_STEP / GROUP_SIZE]"));
        assert!(body.contains("warp_m_offset = warp_id * 16"));
        assert!(body.contains("mma.sync.aligned.kind::mxf4nvf4.block_scale"));
        assert!(body.contains("acc[nt][0] * scale2_ab"));
        assert_eq!(body.matches("__syncthreads();").count(), 2);
    }

    #[test]
    fn kmajor_m256_routes_only_at_the_saturated_long_prefill_threshold() {
        assert_eq!(E2M1_KMAJOR_M256_MIN_M, 2048);
        for m in [0, 127, 128, 256, 1024, 2047] {
            assert_eq!(e2m1_kmajor_kernel(m, true), E2m1KmajorKernel::M128);
        }
        for m in [2048, 2049, 8192, u32::MAX] {
            assert_eq!(e2m1_kmajor_kernel(m, true), E2m1KmajorKernel::M256);
            assert_eq!(e2m1_kmajor_kernel(m, false), E2m1KmajorKernel::M128);
        }
    }

    #[test]
    fn kmajor_m256_reuses_one_weight_tile_across_sixteen_row_warps() {
        let source =
            include_str!("../../../../kernels/gb10/qwen3.8-27b/nvfp4/cutlass_nvfp4_gemm.cu");
        let start = source
            .find("void nvfp4_nvfp4_gemm_kmajor_m256(")
            .expect("K-major M256 W4A4 kernel is missing");
        let end = source[start..]
            .find("// FIX PATH 3 LANDED")
            .map(|offset| start + offset)
            .expect("K-major M256 kernel boundary is missing");
        let body = &source[start..end];

        assert!(source.contains("__launch_bounds__(512, 1)\nvoid nvfp4_nvfp4_gemm_kmajor_m256"));
        assert!(body.contains("cta_m = blockIdx.y * 256u"));
        assert!(body.contains("smem_Ap_km256[2][256][K_STEP / 2]"));
        assert!(body.contains("smem_Bp_km256[2][N_TILE_LG][K_STEP / 2]"));
        assert!(body.contains("smem_Bs_km256[2][N_TILE_LG][K_STEP / GROUP_SIZE]"));
        assert!(body.contains("if (threadIdx.x < 256)"));
        assert!(body.contains("warp_m_offset = warp_id * 16"));
        assert!(body.contains("B_packed_t[(unsigned long long)gk_byte * N + gn]"));
        assert!(body.contains("B_scale_t[(unsigned long long)gg * N + gn]"));
        assert!(body.contains("mma.sync.aligned.kind::mxf4nvf4.block_scale"));
        assert!(body.contains("acc[nt][0] * scale2_ab"));
        assert_eq!(body.matches("__syncthreads();").count(), 2);
    }

    #[test]
    fn kmajor_dispatch_fails_closed_and_uses_only_transformed_weights() {
        let source = include_str!("dense_ffn.rs");
        let start = source
            .find("let e2m1_kmajor_requested")
            .expect("K-major request gate is missing");
        let end = source[start..]
            .find("if v2_fast_path {")
            .map(|offset| start + offset)
            .expect("K-major dispatch boundary is missing");
        let body = &source[start..end];

        assert!(body.contains("self.has_e2m1_kmajor_runtime()"));
        assert!(body.contains("self.has_e2m1_kmajor_m256_runtime()"));
        assert!(body.contains("self.has_transposed_ffn()"));
        assert!(body.contains("e2m1_kmajor_projection_shape(m, inter, h)"));
        assert!(body.contains("self.gate_proj_t.as_ref().unwrap()"));
        assert!(body.contains("self.up_proj_t.as_ref().unwrap()"));
        assert!(body.contains("self.down_proj_t.as_ref().unwrap()"));
        assert!(source.contains("ops::nvfp4_nvfp4_gemm_kmajor_m128("));
        assert!(source.contains("ops::nvfp4_nvfp4_gemm_kmajor_m256("));

        let wrapper = include_str!("ops/gemm_dense.rs");
        assert!(wrapper.contains("pub fn nvfp4_nvfp4_gemm_kmajor_m256("));
        assert!(wrapper.contains(".grid([n / 128, div_ceil(m, 256), 1])"));
        assert!(wrapper.contains(".block([512, 1, 1])"));
    }

    #[test]
    fn static_scale_matches_merged_gate_up_contract_and_rejects_bad_metadata() {
        let scales = validated_e2m1_checkpoint_scales(0.001, 0.00125, 0.002).unwrap();
        assert_eq!(scales.gate_up.to_bits(), 0.00125_f32.to_bits());
        assert_eq!(scales.down.to_bits(), 0.002_f32.to_bits());

        for bad in [0.0, -0.1, f32::INFINITY, f32::NAN] {
            assert!(validated_e2m1_checkpoint_scales(bad, 0.001, 0.002).is_err());
            assert!(validated_e2m1_checkpoint_scales(0.001, bad, 0.002).is_err());
            assert!(validated_e2m1_checkpoint_scales(0.001, 0.002, bad).is_err());
        }
    }

    #[test]
    fn static_scale_bypasses_runtime_absmax_and_sync_before_quantization() {
        let source = include_str!("dense_ffn.rs");
        let start = source
            .find("fn prepare_e2m1_input(")
            .expect("W4A4 activation preparation is missing");
        let end = source[start..]
            .find("fn forward_e2m1_prepared(")
            .map(|offset| start + offset)
            .expect("prepared W4A4 GEMM boundary is missing");
        let body = &source[start..end];

        let fixed = body
            .find("if let Some(scale2) = checkpoint_scale2")
            .expect("checkpoint-static branch is missing");
        let dynamic = body
            .find("ctx.gpu.synchronize(stream)?")
            .expect("dynamic absmax fallback is missing");
        let quantize = body
            .find("ops::quantize_bf16_to_nvfp4(")
            .expect("shared on-device quantizer is missing");
        assert!(fixed < dynamic && dynamic < quantize);
    }

    #[test]
    fn full_w4a4_gate_and_up_share_one_prepared_activation() {
        let source = include_str!("dense_ffn.rs");
        let start = source
            .find("if e2m1_fast_path {")
            .expect("full W4A4 dispatch is missing");
        let end = source[start..]
            .find("if e2m1_down_only_path {")
            .map(|offset| start + offset)
            .expect("down-only dispatch boundary is missing");
        let branch = &source[start..end];

        assert_eq!(branch.matches("self.prepare_e2m1_input(").count(), 1);
        assert_eq!(branch.matches("self.forward_e2m1_prepared(").count(), 3);
        assert_eq!(branch.matches("self.forward_e2m1_proj(").count(), 1);
        assert_eq!(branch.matches(".prepare_e2m1_silu_input(").count(), 1);
        assert!(branch.contains("&self.weights.gate_proj"));
        assert!(branch.contains("&self.weights.up_proj"));
        assert!(branch.contains("&self.weights.down_proj"));
    }

    #[test]
    fn fused_silu_quant_preserves_bf16_boundary_and_fails_closed() {
        let cuda = include_str!("../../../../kernels/gb10/common/quantize_bf16_to_nvfp4.cu");
        let start = cuda
            .find("void quantize_silu_mul_bf16_to_nvfp4(")
            .expect("fused SwiGLU quantizer is missing");
        let body = &cuda[start..];
        assert!(cuda.contains("return __bfloat162float(__float2bfloat16(g * sigmoid_g * u));"));
        assert!(body.contains("float_to_fp8_e4m3(fp8_float)"));
        assert!(body.contains("quantize_e2m1(v0)"));
        assert!(body.contains("quantize_e2m1(v1)"));

        let rust = include_str!("dense_ffn.rs");
        assert!(rust.contains("ATLAS_E2M1_SILU_QUANT=1 requires ATLAS_E2M1_STATIC_SCALE=1"));
        assert!(rust.contains("ATLAS_E2M1_SILU_QUANT=1 is valid only for SiLU-gated FFNs"));
        assert!(rust.contains("ATLAS_E2M1_SILU_QUANT=1 requires quantize_silu_mul_bf16_to_nvfp4"));
    }

    #[test]
    fn kmajor_transform_and_tile_scatter_reconstruct_parent_layout() {
        const N: usize = 256;
        const K: usize = 192;
        let packed_k = K / 2;
        let scale_k = K / 16;

        let packed_parent: Vec<u32> = (0..N * packed_k).map(|index| index as u32).collect();
        let scale_parent: Vec<u32> = (0..N * scale_k)
            .map(|index| 1_000_000 + index as u32)
            .collect();
        let mut packed_t = vec![0_u32; packed_parent.len()];
        let mut scale_t = vec![0_u32; scale_parent.len()];
        for column in 0..N {
            for k_byte in 0..packed_k {
                packed_t[k_byte * N + column] = packed_parent[column * packed_k + k_byte];
            }
            for group in 0..scale_k {
                scale_t[group * N + column] = scale_parent[column * scale_k + group];
            }
        }

        for n_base in [0, 128] {
            for k_base in [0, 64, 128] {
                for local_n in 0..128 {
                    let column = n_base + local_n;
                    for local_byte in 0..32 {
                        let k_byte = k_base / 2 + local_byte;
                        let staged = packed_t[k_byte * N + column];
                        assert_eq!(staged, packed_parent[column * packed_k + k_byte]);
                    }
                    for local_group in 0..4 {
                        let group = k_base / 16 + local_group;
                        let staged = scale_t[group * N + column];
                        assert_eq!(staged, scale_parent[column * scale_k + group]);
                    }
                }
            }
        }
    }

    #[test]
    fn sm103_disables_warp_nvfp4_mma_and_enables_cublaslt_nvfp4_route() {
        assert!(is_blackwell_dc_compute_capability(Some((10, 0))));
        assert!(is_blackwell_dc_compute_capability(Some((10, 3))));
        assert!(!is_blackwell_dc_compute_capability(Some((12, 0))));
        assert!(!is_blackwell_dc_compute_capability(Some((12, 1))));
        assert!(!is_blackwell_dc_compute_capability(None));

        assert!(is_sm120_compute_capability(Some((12, 0))));
        assert!(!is_sm120_compute_capability(Some((12, 1))));
        assert!(!is_sm120_compute_capability(Some((10, 0))));
        assert!(!is_sm120_compute_capability(Some((10, 3))));
        assert!(!is_sm120_compute_capability(None));

        assert!(!supports_warp_nvfp4_mma(Some((10, 3))));
        assert!(!supports_warp_nvfp4_mma(Some((10, 0))));
        assert!(supports_warp_nvfp4_mma(Some((12, 0))));
        assert!(supports_warp_nvfp4_mma(Some((12, 1))));
        assert!(supports_warp_nvfp4_mma(None));

        assert_eq!(parse_cublaslt_nvfp4_policy(None), CublasLtNvfp4Policy::Auto);
        assert_eq!(
            parse_cublaslt_nvfp4_policy(Some("1")),
            CublasLtNvfp4Policy::Enabled
        );
        assert_eq!(
            parse_cublaslt_nvfp4_policy(Some("true")),
            CublasLtNvfp4Policy::Enabled
        );
        assert_eq!(
            parse_cublaslt_nvfp4_policy(Some("0")),
            CublasLtNvfp4Policy::Disabled
        );
        assert_eq!(
            parse_cublaslt_nvfp4_policy(Some("false")),
            CublasLtNvfp4Policy::Disabled
        );

        // Auto mode engages on SM100 (B200/GB200) and SM103 (B300/GB300).
        for dc_cc in [Some((10, 0)), Some((10, 3))] {
            assert!(cublaslt_nvfp4_route_enabled(
                CublasLtNvfp4Policy::Auto,
                dc_cc,
                FlashInferFfnAvailability::Absent,
                64,
                5120,
                17408,
                true,
                true,
            ));
        }
        assert!(cublaslt_nvfp4_route_enabled(
            CublasLtNvfp4Policy::Auto,
            Some((10, 3)),
            FlashInferFfnAvailability::Absent,
            128,
            5120,
            17408,
            true,
            true,
        ));
        // Auto mode engages on SM120 (RTX PRO 6000) when FlashInfer FFN is absent
        // (e.g. Qwen3.8-27B H=5120, I=17408 for M>=64), and defers when FlashInfer
        // is prepared.
        assert!(cublaslt_nvfp4_route_enabled(
            CublasLtNvfp4Policy::Auto,
            Some((12, 0)),
            FlashInferFfnAvailability::Absent,
            64,
            5120,
            17408,
            true,
            true,
        ));
        assert!(cublaslt_nvfp4_route_enabled(
            CublasLtNvfp4Policy::Auto,
            Some((12, 0)),
            FlashInferFfnAvailability::Absent,
            512,
            5120,
            17408,
            true,
            true,
        ));
        assert!(!cublaslt_nvfp4_route_enabled(
            CublasLtNvfp4Policy::Auto,
            Some((12, 0)),
            FlashInferFfnAvailability::Prepared,
            512,
            5120,
            17408,
            true,
            true,
        ));
        // Auto mode stays off on SM121 (GB10) and mock/None.
        assert!(!cublaslt_nvfp4_route_enabled(
            CublasLtNvfp4Policy::Auto,
            Some((12, 1)),
            FlashInferFfnAvailability::Absent,
            512,
            5120,
            17408,
            true,
            true,
        ));
        assert!(!cublaslt_nvfp4_route_enabled(
            CublasLtNvfp4Policy::Auto,
            None,
            FlashInferFfnAvailability::Absent,
            512,
            5120,
            17408,
            true,
            true,
        ));
        // Explicit opt-out disables even on SM103/SM120; explicit opt-in allows SM121 testing.
        assert!(!cublaslt_nvfp4_route_enabled(
            CublasLtNvfp4Policy::Disabled,
            Some((10, 3)),
            FlashInferFfnAvailability::Absent,
            512,
            5120,
            17408,
            true,
            true,
        ));
        assert!(!cublaslt_nvfp4_route_enabled(
            CublasLtNvfp4Policy::Disabled,
            Some((12, 0)),
            FlashInferFfnAvailability::Absent,
            512,
            5120,
            17408,
            true,
            true,
        ));
        assert!(cublaslt_nvfp4_route_enabled(
            CublasLtNvfp4Policy::Enabled,
            Some((12, 1)),
            FlashInferFfnAvailability::Absent,
            512,
            5120,
            17408,
            true,
            true,
        ));
        // Missing 128x4 quantizer, M < 24, or non-64-aligned shapes fall back cleanly to w4a16.
        assert!(!cublaslt_nvfp4_route_enabled(
            CublasLtNvfp4Policy::Auto,
            Some((12, 0)),
            FlashInferFfnAvailability::Absent,
            23,
            5120,
            17408,
            true,
            true,
        ));
        assert!(cublaslt_nvfp4_route_enabled(
            CublasLtNvfp4Policy::Auto,
            Some((12, 0)),
            FlashInferFfnAvailability::Absent,
            24,
            5120,
            17408,
            true,
            true,
        ));
        assert!(cublaslt_nvfp4_route_enabled(
            CublasLtNvfp4Policy::Auto,
            Some((12, 0)),
            FlashInferFfnAvailability::Absent,
            32,
            5120,
            17408,
            true,
            true,
        ));
        assert!(!cublaslt_nvfp4_route_enabled(
            CublasLtNvfp4Policy::Auto,
            Some((10, 3)),
            FlashInferFfnAvailability::Absent,
            512,
            5120,
            17408,
            false,
            true,
        ));
        assert!(!cublaslt_nvfp4_route_enabled(
            CublasLtNvfp4Policy::Auto,
            Some((10, 3)),
            FlashInferFfnAvailability::Absent,
            512,
            5100,
            17408,
            true,
            true,
        ));
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cublaslt_nvfp4_scale_swizzle_matches_cutlass_128x4_layout() {
        use crate::weight_map::cutlass_scale_layout::{
            NVFP4_GROUP_SIZE, interleave_nvfp4_scales_128x4,
        };

        let n = 256u32;
        let k = 128u32;
        let groups = (k as usize) / NVFP4_GROUP_SIZE;
        let logical: Vec<u8> = (0..(n as usize) * groups)
            .map(|i| ((i.wrapping_mul(37) + 11) & 0xff) as u8)
            .collect();
        let via_cublaslt = swizzle_nvfp4_scales_128x4_host(&logical, n, k).unwrap();
        let via_cutlass =
            interleave_nvfp4_scales_128x4(&logical, &[n as usize, groups], NVFP4_GROUP_SIZE)
                .unwrap();
        assert_eq!(via_cublaslt, via_cutlass);

        // Non-128-multiple N pads cleanly without error.
        let unaligned_n = 192u32;
        let logical_unaligned = vec![0x38u8; (unaligned_n as usize) * groups];
        let padded = swizzle_nvfp4_scales_128x4_host(&logical_unaligned, unaligned_n, k).unwrap();
        assert_eq!(padded.len(), 256 * groups);
    }

    #[test]
    fn cublaslt_prefers_ptx_quantizer_and_gates_fused_silu_quantizer() {
        use super::{
            FfnActivation, KernelHandle, cublaslt_fused_silu_quantize_scale,
            select_cublaslt_quantize_128x4_kernel,
        };

        let legacy = KernelHandle(11);
        let ptx = KernelHandle(22);
        let fused = KernelHandle(33);
        let missing = KernelHandle(0);

        assert_eq!(select_cublaslt_quantize_128x4_kernel(ptx, legacy).0, ptx.0);
        assert_eq!(
            select_cublaslt_quantize_128x4_kernel(missing, legacy).0,
            legacy.0
        );
        assert_eq!(select_cublaslt_quantize_128x4_kernel(ptx, missing).0, ptx.0);
        assert_eq!(
            select_cublaslt_quantize_128x4_kernel(missing, missing).0,
            missing.0
        );

        assert_eq!(
            cublaslt_fused_silu_quantize_scale(FfnActivation::SiLU, Some(0.125), fused),
            Some(0.125)
        );
        assert_eq!(
            cublaslt_fused_silu_quantize_scale(FfnActivation::SiLU, None, fused),
            None
        );
        assert_eq!(
            cublaslt_fused_silu_quantize_scale(FfnActivation::SiLU, Some(0.125), missing),
            None
        );
        assert_eq!(
            cublaslt_fused_silu_quantize_scale(FfnActivation::GeLU, Some(0.125), fused),
            None
        );

        let cuda = include_str!(
            "../../../../kernels/gb10/qwen3.8-27b/nvfp4/quantize_bf16_to_nvfp4_cutlass.cu"
        );
        assert!(cuda.contains("quantize_bf16_to_nvfp4_atlas_128x4_ptx("));
        assert!(cuda.contains("silu_mul_quantize_nvfp4_atlas_128x4("));
        assert!(cuda.contains("cvt.rn.satfinite.e2m1x2.f32"));
    }

    #[test]
    fn cublaslt_merged_gate_up_and_27b_proj_policies_protect_gb10_and_vram() {
        use super::{
            CUBLASLT_MERGED_GATEUP_VRAM_RESERVE_BYTES, CublasLtMergedGateUpPolicy,
            CublasLtNvfp4Policy, Proj27bCublasLtNvfp4Policy, cublaslt_merged_gate_up_vram_ok,
            parse_cublaslt_merged_gate_up_policy, parse_proj_27b_cublaslt_nvfp4_policy,
            proj_27b_cublaslt_nvfp4_shape_ok, resolve_proj_27b_cublaslt_nvfp4,
        };

        assert_eq!(
            parse_cublaslt_merged_gate_up_policy(None),
            CublasLtMergedGateUpPolicy::Auto
        );
        assert_eq!(
            parse_cublaslt_merged_gate_up_policy(Some("0")),
            CublasLtMergedGateUpPolicy::Disabled
        );
        assert_eq!(
            parse_cublaslt_merged_gate_up_policy(Some("false")),
            CublasLtMergedGateUpPolicy::Disabled
        );
        assert_eq!(
            parse_cublaslt_merged_gate_up_policy(Some("1")),
            CublasLtMergedGateUpPolicy::Enabled
        );
        assert_eq!(
            parse_cublaslt_merged_gate_up_policy(Some("true")),
            CublasLtMergedGateUpPolicy::Enabled
        );

        let needed = 89 * 1024 * 1024;
        assert!(!cublaslt_merged_gate_up_vram_ok(None, needed));
        assert!(!cublaslt_merged_gate_up_vram_ok(
            Some(CUBLASLT_MERGED_GATEUP_VRAM_RESERVE_BYTES + needed - 1),
            needed
        ));
        assert!(cublaslt_merged_gate_up_vram_ok(
            Some(CUBLASLT_MERGED_GATEUP_VRAM_RESERVE_BYTES + needed),
            needed
        ));

        assert_eq!(
            parse_proj_27b_cublaslt_nvfp4_policy(None),
            Proj27bCublasLtNvfp4Policy::Auto
        );
        assert_eq!(
            parse_proj_27b_cublaslt_nvfp4_policy(Some("0")),
            Proj27bCublasLtNvfp4Policy::Disabled
        );
        assert_eq!(
            parse_proj_27b_cublaslt_nvfp4_policy(Some("1")),
            Proj27bCublasLtNvfp4Policy::Enabled
        );

        // GB10 (<= 64 SMs) stays default-OFF under Auto; explicit "1" enables.
        assert!(!resolve_proj_27b_cublaslt_nvfp4(
            Proj27bCublasLtNvfp4Policy::Auto,
            CublasLtNvfp4Policy::Auto,
            48,
            Some((12, 1))
        ));
        assert!(!resolve_proj_27b_cublaslt_nvfp4(
            Proj27bCublasLtNvfp4Policy::Auto,
            CublasLtNvfp4Policy::Enabled,
            64,
            Some((12, 1))
        ));
        assert!(resolve_proj_27b_cublaslt_nvfp4(
            Proj27bCublasLtNvfp4Policy::Enabled,
            CublasLtNvfp4Policy::Disabled,
            48,
            Some((12, 1))
        ));

        // SM120 / SM103 (> 64 SMs): auto-ON unless either gate opts out.
        for cc in [Some((12, 0)), Some((10, 3))] {
            assert!(resolve_proj_27b_cublaslt_nvfp4(
                Proj27bCublasLtNvfp4Policy::Auto,
                CublasLtNvfp4Policy::Auto,
                188,
                cc
            ));
            assert!(!resolve_proj_27b_cublaslt_nvfp4(
                Proj27bCublasLtNvfp4Policy::Disabled,
                CublasLtNvfp4Policy::Auto,
                188,
                cc
            ));
            assert!(!resolve_proj_27b_cublaslt_nvfp4(
                Proj27bCublasLtNvfp4Policy::Auto,
                CublasLtNvfp4Policy::Disabled,
                188,
                cc
            ));
        }

        // Shape eligibility: Qwen3.8-27B SSM/Attn projections require M >= 64, N % 8 == 0, K % 64 == 0.
        assert!(proj_27b_cublaslt_nvfp4_shape_ok(64, 16_384, 5_120));
        assert!(proj_27b_cublaslt_nvfp4_shape_ok(7_942, 12_288, 5_120));
        assert!(proj_27b_cublaslt_nvfp4_shape_ok(8_192, 1_024, 5_120));
        assert!(proj_27b_cublaslt_nvfp4_shape_ok(8_192, 5_120, 6_144));
        assert!(!proj_27b_cublaslt_nvfp4_shape_ok(63, 16_384, 5_120));
        assert!(!proj_27b_cublaslt_nvfp4_shape_ok(8_192, 1_020, 5_120));
        assert!(!proj_27b_cublaslt_nvfp4_shape_ok(8_192, 5_120, 5_100));
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn merged_128x4_scales_equal_concatenated_halves_for_128_aligned_inter() {
        let inter = 256u32;
        let h = 128u32;
        let groups = (h / spark_runtime::cublaslt::NVFP4_BLOCK_SIZE) as usize;
        let gate_logical: Vec<u8> = (0..(inter as usize) * groups)
            .map(|i| ((i * 19 + 3) & 0xff) as u8)
            .collect();
        let up_logical: Vec<u8> = (0..(inter as usize) * groups)
            .map(|i| ((i * 41 + 7) & 0xff) as u8)
            .collect();
        let mut combined_logical = Vec::with_capacity(gate_logical.len() + up_logical.len());
        combined_logical.extend_from_slice(&gate_logical);
        combined_logical.extend_from_slice(&up_logical);

        let gate_s = swizzle_nvfp4_scales_128x4_host(&gate_logical, inter, h).unwrap();
        let up_s = swizzle_nvfp4_scales_128x4_host(&up_logical, inter, h).unwrap();
        let merged_s = swizzle_nvfp4_scales_128x4_host(&combined_logical, 2 * inter, h).unwrap();

        assert_eq!(&merged_s[..gate_s.len()], gate_s.as_slice());
        assert_eq!(&merged_s[gate_s.len()..], up_s.as_slice());
    }

    #[test]
    fn sm120_kgamma_auto_tc_unlocks_m4_to_256_and_preserves_gb10_defaults() {
        use super::{
            FfnTcPolicy, parse_ffn_tc_policy, resolve_ffn_tc_for_rows, resolve_kgamma_sm_auto_flag,
        };

        assert_eq!(parse_ffn_tc_policy(None), FfnTcPolicy::Auto);
        assert_eq!(parse_ffn_tc_policy(Some("0")), FfnTcPolicy::Disabled);
        assert_eq!(parse_ffn_tc_policy(Some("false")), FfnTcPolicy::Disabled);
        assert_eq!(parse_ffn_tc_policy(Some("1")), FfnTcPolicy::Enabled);
        assert_eq!(parse_ffn_tc_policy(Some("true")), FfnTcPolicy::Enabled);

        // GB10 (<= 64 SMs): Auto keeps exact scalar GEMV and all opt-in flags OFF.
        for sm in [48u32, 64u32] {
            for rows in [1u32, 2, 3, 4, 8, 16, 32, 33, 64, 128, 256] {
                assert!(
                    !resolve_ffn_tc_for_rows(FfnTcPolicy::Auto, sm, rows),
                    "GB10 sm={sm} rows={rows} must stay exact under Auto"
                );
                assert!(
                    resolve_ffn_tc_for_rows(FfnTcPolicy::Enabled, sm, rows),
                    "explicit ATLAS_FFN_TC=1 must override on sm={sm}"
                );
            }
            assert!(!resolve_kgamma_sm_auto_flag(None, sm));
            assert!(resolve_kgamma_sm_auto_flag(Some("1"), sm));
            assert!(!resolve_kgamma_sm_auto_flag(Some("0"), sm));
        }

        // SM120 / SM103 (> 64 SMs): Auto enables TC batched kgamma for rows >= 4
        // (including M=4, 8, 16, 32, 64, 128, 256) while keeping rows < 4 on exact GEMV,
        // and honors explicit "0"/"false" opt-out for strict 0-ULP parity.
        for sm in [128u32, 188u32] {
            for small in [1u32, 2, 3] {
                assert!(
                    !resolve_ffn_tc_for_rows(FfnTcPolicy::Auto, sm, small),
                    "sm={sm} rows={small} should keep batched GEMV"
                );
            }
            for batched in [4u32, 8, 16, 32, 33, 64, 128, 256] {
                assert!(
                    resolve_ffn_tc_for_rows(FfnTcPolicy::Auto, sm, batched),
                    "sm={sm} rows={batched} must auto-enable TC batched kgamma"
                );
                assert!(
                    !resolve_ffn_tc_for_rows(FfnTcPolicy::Disabled, sm, batched),
                    "sm={sm} rows={batched} must honor ATLAS_FFN_TC=0 opt-out"
                );
            }
            assert!(resolve_kgamma_sm_auto_flag(None, sm));
            assert!(!resolve_kgamma_sm_auto_flag(Some("0"), sm));
            assert!(!resolve_kgamma_sm_auto_flag(Some("false"), sm));
            assert!(resolve_kgamma_sm_auto_flag(Some("1"), sm));
        }
    }

    #[test]
    fn sm120_ffn_strip_reserve_and_kgamma_cublaslt_policies_keep_gb10_zero_regression() {
        use super::{
            CUBLASLT_FFN_PREFILL_L2_STRIP_DEFAULT, CUBLASLT_MERGED_GATEUP_VRAM_RESERVE_BYTES,
            CUBLASLT_MERGED_GATEUP_VRAM_RESERVE_DISCRETE_BYTES, CublasLtMergedGateUpPolicy,
            cublaslt_merged_gate_up_allow_materialize, cublaslt_merged_gate_up_vram_ok_for_sm,
            parse_ffn_prefill_l2_strip, resolve_kgamma_cublaslt_opt_in, resolve_sm_auto_opt_in,
        };

        let gib = 1024 * 1024 * 1024usize;
        let needed = 89 * 1024 * 1024usize; // 27B single-layer merged gate+up

        // GB10 (<= 64 SMs): 12 GiB reserve, Auto stays zero-copy, L2 strip OFF,
        // kgamma cuBLASLt OFF, fused absmax/norm OFF.
        for sm in [48u32, 64u32] {
            assert!(!cublaslt_merged_gate_up_allow_materialize(
                CublasLtMergedGateUpPolicy::Auto,
                sm
            ));
            assert!(cublaslt_merged_gate_up_allow_materialize(
                CublasLtMergedGateUpPolicy::Enabled,
                sm
            ));
            assert!(!cublaslt_merged_gate_up_vram_ok_for_sm(
                Some(11 * gib + 600 * 1024 * 1024),
                needed,
                sm
            ));
            assert!(cublaslt_merged_gate_up_vram_ok_for_sm(
                Some(CUBLASLT_MERGED_GATEUP_VRAM_RESERVE_BYTES + needed),
                needed,
                sm
            ));
            assert_eq!(parse_ffn_prefill_l2_strip(None, sm), None);
            assert_eq!(
                parse_ffn_prefill_l2_strip(Some("1"), sm),
                Some(CUBLASLT_FFN_PREFILL_L2_STRIP_DEFAULT)
            );
            assert_eq!(parse_ffn_prefill_l2_strip(Some("512"), sm), Some(512));
            assert!(!resolve_kgamma_cublaslt_opt_in(None, sm));
            assert!(resolve_kgamma_cublaslt_opt_in(Some("1"), sm));
            assert!(!resolve_sm_auto_opt_in(None, sm));
        }

        // SM120 / SM103 (> 64 SMs): 4 GiB reserve (C=64 with 11.6 GiB free fits!),
        // Auto allows materialization, L2 strip=1024, kgamma cuBLASLt ON, fused absmax/norm ON.
        for sm in [65u32, 188u32] {
            assert!(cublaslt_merged_gate_up_allow_materialize(
                CublasLtMergedGateUpPolicy::Auto,
                sm
            ));
            assert!(!cublaslt_merged_gate_up_allow_materialize(
                CublasLtMergedGateUpPolicy::Disabled,
                sm
            ));
            // 11.6 GiB free (C=64 with --kv-pool-tokens 131072) passes!
            assert!(cublaslt_merged_gate_up_vram_ok_for_sm(
                Some(11 * gib + 600 * 1024 * 1024),
                needed,
                sm
            ));
            assert!(!cublaslt_merged_gate_up_vram_ok_for_sm(
                Some(CUBLASLT_MERGED_GATEUP_VRAM_RESERVE_DISCRETE_BYTES + needed - 1),
                needed,
                sm
            ));
            assert!(cublaslt_merged_gate_up_vram_ok_for_sm(
                Some(CUBLASLT_MERGED_GATEUP_VRAM_RESERVE_DISCRETE_BYTES + needed),
                needed,
                sm
            ));
            assert_eq!(
                parse_ffn_prefill_l2_strip(None, sm),
                Some(CUBLASLT_FFN_PREFILL_L2_STRIP_DEFAULT)
            );
            assert_eq!(parse_ffn_prefill_l2_strip(Some("0"), sm), None);
            assert_eq!(parse_ffn_prefill_l2_strip(Some("false"), sm), None);
            assert_eq!(parse_ffn_prefill_l2_strip(Some("2048"), sm), Some(2048));
            assert!(resolve_kgamma_cublaslt_opt_in(None, sm));
            assert!(!resolve_kgamma_cublaslt_opt_in(Some("0"), sm));
            assert!(resolve_sm_auto_opt_in(None, sm));
            assert!(!resolve_sm_auto_opt_in(Some("0"), sm));
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn gpu_parity_l2_strip_m8058_ptx_quantizers_and_fused_residual_norm() -> anyhow::Result<()> {
        use super::{
            CUBLASLT_DYNAMIC_GLOBAL_MAX_OFF, CUBLASLT_DYNAMIC_SCALARS_BYTES,
            CUBLASLT_DYNAMIC_SCALE2_OFF, CUBLASLT_DYNAMIC_STATUS_OFF,
            CUBLASLT_DYNAMIC_ZERO_PREFIX_BYTES, swizzle_nvfp4_scales_128x4_host,
        };
        use crate::layers::{ops, try_kernel};
        use crate::weight_map::{DenseWeight, QuantizedWeight, quantize_to_nvfp4};
        use half::bf16;
        use spark_runtime::cublaslt::BlockScaleScalars;
        use spark_runtime::gpu::{DevicePtr, GpuBackend};

        let Some(ptx) = atlas_kernels::ptx_for_model("qwen3.8-27b")
            .or_else(|| atlas_kernels::all_ptx_sets().into_iter().next())
        else {
            return Ok(());
        };
        let Ok(gpu) = spark_runtime::cuda_backend::AtlasCudaBackend::new(0, &ptx.modules) else {
            return Ok(());
        };
        let stream = gpu.default_stream();

        let quant_static_k = try_kernel(
            &gpu,
            "quantize_bf16_to_nvfp4_cutlass",
            "quantize_bf16_to_nvfp4_atlas_128x4_ptx",
        );
        let quant_static_legacy_k = try_kernel(
            &gpu,
            "quantize_bf16_to_nvfp4_cutlass",
            "quantize_bf16_to_nvfp4_atlas_128x4",
        );
        let quant_k = if quant_static_k.0 != 0 {
            quant_static_k
        } else {
            quant_static_legacy_k
        };
        let merged_silu_legacy_k = try_kernel(
            &gpu,
            "quantize_nvfp4",
            "quantize_merged_silu_mul_bf16_to_nvfp4_atlas_128x4",
        );
        let merged_silu_ptx_k = try_kernel(
            &gpu,
            "quantize_nvfp4",
            "quantize_merged_silu_mul_bf16_to_nvfp4_atlas_128x4_ptx",
        );
        let from_absmax_legacy_k = try_kernel(
            &gpu,
            "quantize_bf16_to_nvfp4_cutlass",
            "quantize_bf16_to_nvfp4_atlas_128x4_from_absmax",
        );
        let from_absmax_ptx_k = try_kernel(
            &gpu,
            "quantize_nvfp4",
            "quantize_bf16_to_nvfp4_atlas_128x4_from_absmax_ptx",
        );
        let fused_norm_quant_k = try_kernel(
            &gpu,
            "quantize_nvfp4",
            "residual_add_rms_norm_quantize_nvfp4_128x4",
        );
        let absmax_k = try_kernel(&gpu, "quantize_nvfp4", "nvfp4_global_absmax");
        let quant_w_k = try_kernel(&gpu, "quantize_nvfp4", "quantize_bf16_to_nvfp4");
        let resid_norm_k = try_kernel(&gpu, "norm", "residual_add_rms_norm");
        let rms_norm_res_k = try_kernel(&gpu, "norm", "rms_norm_residual");
        let rms_norm_res_absmax_k = try_kernel(&gpu, "norm", "rms_norm_residual_absmax");
        let rms_norm_absmax_k = try_kernel(&gpu, "norm", "rms_norm_absmax");
        let gated_prefill_k = try_kernel(&gpu, "norm", "gated_rms_norm_prefill");
        let gated_prefill_absmax_k = try_kernel(&gpu, "norm", "gated_rms_norm_prefill_absmax");
        let sig_gate_k = try_kernel(&gpu, "residual_add", "sigmoid_gate_mul_batched");
        let sig_gate_absmax_k = try_kernel(&gpu, "residual_add", "sigmoid_gate_mul_batched_absmax");
        let resid_add_k = try_kernel(&gpu, "residual_add", "bf16_residual_add");

        if quant_k.0 == 0
            || merged_silu_legacy_k.0 == 0
            || merged_silu_ptx_k.0 == 0
            || from_absmax_legacy_k.0 == 0
            || from_absmax_ptx_k.0 == 0
            || fused_norm_quant_k.0 == 0
            || absmax_k.0 == 0
            || quant_w_k.0 == 0
            || resid_norm_k.0 == 0
            || rms_norm_res_k.0 == 0
            || rms_norm_res_absmax_k.0 == 0
            || rms_norm_absmax_k.0 == 0
            || gated_prefill_k.0 == 0
            || gated_prefill_absmax_k.0 == 0
            || sig_gate_k.0 == 0
            || sig_gate_absmax_k.0 == 0
            || resid_add_k.0 == 0
        {
            return Ok(());
        }

        // 1. Test M = 8058 (= 7 * 1024 + 890) with H = 512, Inter = 1024:
        //    verifies non-1024 remainder strip + PTX merged SiLU quantizer +
        //    fused `residual_add_rms_norm_quantize_nvfp4_128x4` + `beta = 1.0`
        //    against the unfused reference path!
        let m = 8058u32;
        let strip = 1024u32;
        let h = 512u32;
        let inter = 1024u32;
        let eps = 1e-6f32;
        let scale2_in = 0.05f32;
        let scale2_down = 0.05f32;

        let make_bf16 = |len: usize, seed: usize, amp: f32| -> Vec<u8> {
            let mut out = Vec::with_capacity(len * 2);
            for i in 0..len {
                let x = i
                    .wrapping_mul(1315423911)
                    .wrapping_add(seed.wrapping_mul(2654435761));
                let v = (((x >> 16) & 0x3FF) as f32 / 512.0 - 1.0) * amp;
                out.extend_from_slice(&bf16::from_f32(v).to_le_bytes());
            }
            out
        };

        let hidden_bytes = make_bf16((m * h) as usize, 1, 0.8);
        let src_bytes = make_bf16((m * h) as usize, 2, 0.4);
        let norm_w_bytes = make_bf16(h as usize, 3, 0.25);
        let gate_up_w_bytes = make_bf16((2 * inter * h) as usize, 4, 0.15);
        let down_w_bytes = make_bf16((h * inter) as usize, 5, 0.15);

        let hidden_ref = gpu.alloc(hidden_bytes.len())?;
        let hidden_strip = gpu.alloc(hidden_bytes.len())?;
        let hidden_fused = gpu.alloc(hidden_bytes.len())?;
        let src_dev = gpu.alloc(src_bytes.len())?;
        let norm_w_dev = gpu.alloc(norm_w_bytes.len())?;
        let resid_dev = gpu.alloc(hidden_bytes.len())?;
        let norm_out_dev = gpu.alloc(hidden_bytes.len())?;
        let ffn_out_ref = gpu.alloc(hidden_bytes.len())?;
        let ffn_out_strip = gpu.alloc(hidden_bytes.len())?;

        gpu.copy_h2d(&hidden_bytes, hidden_ref)?;
        gpu.copy_h2d(&hidden_bytes, hidden_strip)?;
        gpu.copy_h2d(&hidden_bytes, hidden_fused)?;
        gpu.copy_h2d(&src_bytes, src_dev)?;
        gpu.copy_h2d(&norm_w_bytes, norm_w_dev)?;

        let merged_w_bf16 = gpu.alloc(gate_up_w_bytes.len())?;
        let down_w_bf16 = gpu.alloc(down_w_bytes.len())?;
        gpu.copy_h2d(&gate_up_w_bytes, merged_w_bf16)?;
        gpu.copy_h2d(&down_w_bytes, down_w_bf16)?;

        let merged_q = quantize_to_nvfp4(
            &DenseWeight {
                weight: merged_w_bf16,
            },
            (2 * inter) as usize,
            h as usize,
            &gpu,
            absmax_k,
            quant_w_k,
            stream,
        )?;
        let down_q = quantize_to_nvfp4(
            &DenseWeight {
                weight: down_w_bf16,
            },
            h as usize,
            inter as usize,
            &gpu,
            absmax_k,
            quant_w_k,
            stream,
        )?;
        gpu.synchronize(stream)?;

        let swizzle_weight_scales =
            |w: &QuantizedWeight, n: u32, k: u32| -> anyhow::Result<DevicePtr> {
                let mut raw = vec![0u8; (n as usize) * (k as usize / 16)];
                gpu.copy_d2h(w.weight_scale, &mut raw)?;
                let swizzled = swizzle_nvfp4_scales_128x4_host(&raw, n, k)?;
                let dev = gpu.alloc(swizzled.len())?;
                gpu.copy_h2d(&swizzled, dev)?;
                Ok(dev)
            };
        let merged_s128 = swizzle_weight_scales(&merged_q, 2 * inter, h)?;
        let down_s128 = swizzle_weight_scales(&down_q, h, inter)?;

        let full_layout = super::e2m1_scratch_layout(m as usize, inter as usize)?;
        let a_packed = gpu.alloc(full_layout.packed_bytes)?;
        let a_scale = gpu.alloc(full_layout.scale_bytes)?;
        let merged_out_full = gpu.alloc((m as usize) * (2 * inter as usize) * 2)?;
        let merged_out_strip = gpu.alloc((strip as usize) * (2 * inter as usize) * 2)?;

        let norm_w = DenseWeight { weight: norm_w_dev };
        let merged_scalars = BlockScaleScalars::host(scale2_in * merged_q.weight_scale_2, 0.0);
        let down_scalars_b0 = BlockScaleScalars::host(scale2_down * down_q.weight_scale_2, 0.0);
        let down_scalars_b1 = BlockScaleScalars::host(scale2_down * down_q.weight_scale_2, 1.0);

        // (A) Unfused full-M=8058 reference path (using `merged_silu_ptx_k` so
        //     we isolate M_strip=1024 + remainder M=890 parity cleanly; direct
        //     `merged_silu_legacy_k` vs `merged_silu_ptx_k` nibble/scale parity
        //     is checked separately below).
        ops::residual_add_rms_norm(
            &gpu,
            resid_norm_k,
            hidden_ref,
            src_dev,
            &norm_w,
            norm_out_dev,
            resid_dev,
            m,
            h,
            eps,
            stream,
        )?;
        ops::quantize_bf16_to_nvfp4_atlas_128x4(
            &gpu,
            quant_k,
            norm_out_dev,
            a_packed,
            a_scale,
            scale2_in,
            m,
            h,
            stream,
        )?;
        spark_runtime::cublaslt::nvfp4_gemm_block_scaled(
            a_packed.0,
            a_scale.0,
            merged_q.weight.0,
            merged_s128.0,
            merged_out_full.0,
            m,
            2 * inter,
            h,
            2 * inter,
            merged_scalars,
            stream,
        )?;
        ops::quantize_merged_silu_mul_bf16_to_nvfp4_atlas_128x4(
            &gpu,
            merged_silu_ptx_k,
            merged_out_full,
            a_packed,
            a_scale,
            scale2_down,
            m,
            inter,
            stream,
        )?;
        spark_runtime::cublaslt::nvfp4_gemm_block_scaled(
            a_packed.0,
            a_scale.0,
            down_q.weight.0,
            down_s128.0,
            ffn_out_ref.0,
            m,
            h,
            inter,
            h,
            down_scalars_b0,
            stream,
        )?;
        ops::residual_add(&gpu, resid_add_k, hidden_ref, ffn_out_ref, m * h, stream)?;

        // (B) L2-stripped M_strip=1024 path using PTX merged SiLU quantizer (reading `norm_out_dev`):
        let mut row0 = 0u32;
        while row0 < m {
            let m_s = (m - row0).min(strip);
            let off = (row0 as usize) * (h as usize) * 2;
            ops::quantize_bf16_to_nvfp4_atlas_128x4(
                &gpu,
                quant_k,
                norm_out_dev.offset(off),
                a_packed,
                a_scale,
                scale2_in,
                m_s,
                h,
                stream,
            )?;
            spark_runtime::cublaslt::nvfp4_gemm_block_scaled(
                a_packed.0,
                a_scale.0,
                merged_q.weight.0,
                merged_s128.0,
                merged_out_strip.0,
                m_s,
                2 * inter,
                h,
                2 * inter,
                merged_scalars,
                stream,
            )?;
            ops::quantize_merged_silu_mul_bf16_to_nvfp4_atlas_128x4(
                &gpu,
                merged_silu_ptx_k,
                merged_out_strip,
                a_packed,
                a_scale,
                scale2_down,
                m_s,
                inter,
                stream,
            )?;
            spark_runtime::cublaslt::nvfp4_gemm_block_scaled(
                a_packed.0,
                a_scale.0,
                down_q.weight.0,
                down_s128.0,
                ffn_out_strip.offset(off).0,
                m_s,
                h,
                inter,
                h,
                down_scalars_b0,
                stream,
            )?;
            row0 += m_s;
        }
        ops::residual_add(&gpu, resid_add_k, hidden_strip, src_dev, m * h, stream)?;
        ops::residual_add(
            &gpu,
            resid_add_k,
            hidden_strip,
            ffn_out_strip,
            m * h,
            stream,
        )?;

        // (C) Fully fused `residual_add_rms_norm_quantize_nvfp4_128x4` + L2-strip + `beta = 1.0`:
        row0 = 0;
        while row0 < m {
            let m_s = (m - row0).min(strip);
            let off = (row0 as usize) * (h as usize) * 2;
            ops::residual_add_rms_norm_quantize_nvfp4_128x4(
                &gpu,
                fused_norm_quant_k,
                hidden_fused.offset(off),
                src_dev.offset(off),
                &norm_w,
                DevicePtr::NULL,
                a_packed,
                a_scale,
                scale2_in,
                eps,
                m_s,
                h,
                stream,
            )?;
            spark_runtime::cublaslt::nvfp4_gemm_block_scaled(
                a_packed.0,
                a_scale.0,
                merged_q.weight.0,
                merged_s128.0,
                merged_out_strip.0,
                m_s,
                2 * inter,
                h,
                2 * inter,
                merged_scalars,
                stream,
            )?;
            ops::quantize_merged_silu_mul_bf16_to_nvfp4_atlas_128x4(
                &gpu,
                merged_silu_ptx_k,
                merged_out_strip,
                a_packed,
                a_scale,
                scale2_down,
                m_s,
                inter,
                stream,
            )?;
            spark_runtime::cublaslt::nvfp4_gemm_block_scaled(
                a_packed.0,
                a_scale.0,
                down_q.weight.0,
                down_s128.0,
                hidden_fused.offset(off).0,
                m_s,
                h,
                inter,
                h,
                down_scalars_b1,
                stream,
            )?;
            row0 += m_s;
        }
        gpu.synchronize(stream)?;

        let read_bf16 = |ptr: DevicePtr, bytes: usize| -> anyhow::Result<Vec<f32>> {
            let mut raw = vec![0u8; bytes];
            gpu.copy_d2h(ptr, &mut raw)?;
            Ok(raw
                .chunks_exact(2)
                .map(|c| bf16::from_le_bytes([c[0], c[1]]).to_f32())
                .collect())
        };
        let ref_ffn = read_bf16(ffn_out_ref, hidden_bytes.len())?;
        let strip_ffn = read_bf16(ffn_out_strip, hidden_bytes.len())?;
        let ref_h = read_bf16(hidden_ref, hidden_bytes.len())?;
        let fused_h = read_bf16(hidden_fused, hidden_bytes.len())?;

        let mut max_strip_diff = 0.0f32;
        let mut max_fused_diff = 0.0f32;
        let mut max_ref_abs = 0.0f32;
        for i in 0..ref_ffn.len() {
            assert!(strip_ffn[i].is_finite() && fused_h[i].is_finite());
            max_ref_abs = max_ref_abs.max(ref_h[i].abs());
            max_strip_diff = max_strip_diff.max((ref_ffn[i] - strip_ffn[i]).abs());
            max_fused_diff = max_fused_diff.max((ref_h[i] - fused_h[i]).abs());
        }
        // Strip-mined M_strip=1024 (7x1024 + 890) and fused `residual_add_rms_norm_quantize_nvfp4_128x4`
        // + `beta = 1.0` must match unstripped full M=8058 within 2% of peak magnitude (BF16 ULP).
        let tol = (max_ref_abs * 0.02).max(0.02);
        assert!(
            max_strip_diff <= tol,
            "L2 M_strip=1024 at M=8058 diverged: max_strip_diff={max_strip_diff}, tol={tol}"
        );
        assert!(
            max_fused_diff <= tol,
            "Fused residual_add_rms_norm_quantize + beta=1.0 diverged: max_fused_diff={max_fused_diff}, tol={tol}"
        );

        // Direct A/B of `merged_silu_legacy_k` vs `merged_silu_ptx_k` on identical `merged_out_full`:
        // verifies 128x4 scale swizzle + E2M1 nibble packing match (only <= 1-step RNE vs half-away ties).
        let dbg_p_leg = gpu.alloc((strip as usize) * (inter as usize / 2))?;
        let dbg_s_leg = gpu.alloc((strip as usize) * (inter as usize / 16))?;
        let dbg_p_ptx = gpu.alloc((strip as usize) * (inter as usize / 2))?;
        let dbg_s_ptx = gpu.alloc((strip as usize) * (inter as usize / 16))?;
        ops::quantize_merged_silu_mul_bf16_to_nvfp4_atlas_128x4(
            &gpu,
            merged_silu_legacy_k,
            merged_out_full,
            dbg_p_leg,
            dbg_s_leg,
            scale2_down,
            strip,
            inter,
            stream,
        )?;
        ops::quantize_merged_silu_mul_bf16_to_nvfp4_atlas_128x4(
            &gpu,
            merged_silu_ptx_k,
            merged_out_full,
            dbg_p_ptx,
            dbg_s_ptx,
            scale2_down,
            strip,
            inter,
            stream,
        )?;
        gpu.synchronize(stream)?;
        let mut pl = vec![0u8; (strip as usize) * (inter as usize / 2)];
        let mut pp = vec![0u8; pl.len()];
        let mut sl = vec![0u8; (strip as usize) * (inter as usize / 16)];
        let mut sp = vec![0u8; sl.len()];
        gpu.copy_d2h(dbg_p_leg, &mut pl)?;
        gpu.copy_d2h(dbg_p_ptx, &mut pp)?;
        gpu.copy_d2h(dbg_s_leg, &mut sl)?;
        gpu.copy_d2h(dbg_s_ptx, &mut sp)?;
        let scale_diffs = sl.iter().zip(&sp).filter(|(a, b)| a != b).count();
        let mut nibble_diffs = 0usize;
        for (&a, &b) in pl.iter().zip(&pp) {
            for shift in [0u8, 4u8] {
                let na = (a >> shift) & 0xF;
                let nb = (b >> shift) & 0xF;
                if na != nb {
                    nibble_diffs += 1;
                    // Adjacent E2M1 code (same sign, magnitude diff <= 1) or +/-0 tie.
                    let mag_a = (na & 0x7) as i8;
                    let mag_b = (nb & 0x7) as i8;
                    let same_sign = (na & 0x8) == (nb & 0x8);
                    assert!(
                        (same_sign && (mag_a - mag_b).abs() <= 1) || (mag_a == 0 && mag_b == 0),
                        "merged_silu_ptx nibble diverged by >1 E2M1 step: leg=0x{na:x} ptx=0x{nb:x}"
                    );
                }
            }
        }
        assert!(
            scale_diffs * 1000 <= sl.len(),
            "merged_silu_ptx scale_diffs too high: {scale_diffs}/{}",
            sl.len()
        );
        assert!(
            nibble_diffs * 200 <= pl.len() * 2,
            "merged_silu_ptx nibble_diffs too high: {nibble_diffs}/{}",
            pl.len() * 2
        );
        let _ = gpu.free(dbg_p_leg);
        let _ = gpu.free(dbg_s_leg);
        let _ = gpu.free(dbg_p_ptx);
        let _ = gpu.free(dbg_s_ptx);

        // 2. Verify 0-ULP parity of `rms_norm_absmax`, `rms_norm_residual_absmax`,
        //    `gated_rms_norm_prefill_absmax`, `sigmoid_gate_mul_batched_absmax`, and
        //    `quantize_bf16_to_nvfp4_atlas_128x4_from_absmax_ptx`.
        let n_tok = 128u32;
        let dim = 512u32;
        let elem_bytes = (n_tok * dim * 2) as usize;
        let out_a = gpu.alloc(elem_bytes)?;
        let out_b = gpu.alloc(elem_bytes)?;
        let res_a = gpu.alloc(elem_bytes)?;
        let res_b = gpu.alloc(elem_bytes)?;
        let gmax_ref = gpu.alloc(4)?;
        let gmax_fused = gpu.alloc(4)?;

        // 2a. rms_norm_residual vs rms_norm_residual_absmax & rms_norm_absmax
        gpu.memset_async(gmax_ref, 0, 4, stream)?;
        gpu.memset_async(gmax_fused, 0, 4, stream)?;
        ops::rms_norm_residual(
            &gpu,
            rms_norm_res_k,
            src_dev,
            &norm_w,
            out_a,
            res_a,
            n_tok,
            dim,
            eps,
            stream,
        )?;
        ops::nvfp4_global_absmax(&gpu, absmax_k, out_a, gmax_ref, n_tok * dim, stream)?;
        ops::rms_norm_residual_absmax(
            &gpu,
            rms_norm_res_absmax_k,
            src_dev,
            &norm_w,
            out_b,
            res_b,
            gmax_fused,
            n_tok,
            dim,
            eps,
            stream,
        )?;
        gpu.synchronize(stream)?;
        let mut raw_a = vec![0u8; elem_bytes];
        let mut raw_b = vec![0u8; elem_bytes];
        let mut gm_a = [0u8; 4];
        let mut gm_b = [0u8; 4];
        gpu.copy_d2h(out_a, &mut raw_a)?;
        gpu.copy_d2h(out_b, &mut raw_b)?;
        gpu.copy_d2h(gmax_ref, &mut gm_a)?;
        gpu.copy_d2h(gmax_fused, &mut gm_b)?;
        assert_eq!(
            raw_a, raw_b,
            "rms_norm_residual_absmax must be 0-ULP bit-identical"
        );
        assert_eq!(
            gm_a, gm_b,
            "rms_norm_residual_absmax global_max must be bit-identical"
        );

        gpu.memset_async(gmax_fused, 0, 4, stream)?;
        ops::rms_norm_absmax(
            &gpu,
            rms_norm_absmax_k,
            src_dev,
            &norm_w,
            out_b,
            gmax_fused,
            n_tok,
            dim,
            eps,
            stream,
        )?;
        gpu.synchronize(stream)?;
        gpu.copy_d2h(out_b, &mut raw_b)?;
        gpu.copy_d2h(gmax_fused, &mut gm_b)?;
        assert_eq!(raw_a, raw_b, "rms_norm_absmax must be 0-ULP bit-identical");
        assert_eq!(
            gm_a, gm_b,
            "rms_norm_absmax global_max must be bit-identical"
        );

        // 2b. gated_rms_norm_prefill vs gated_rms_norm_prefill_absmax (head_dim=128, 4 heads)
        gpu.memset_async(gmax_ref, 0, 4, stream)?;
        gpu.memset_async(gmax_fused, 0, 4, stream)?;
        ops::gated_rms_norm_prefill(
            &gpu,
            gated_prefill_k,
            src_dev,
            hidden_ref,
            &norm_w,
            out_a,
            4,
            128,
            eps,
            n_tok,
            dim,
            dim,
            stream,
        )?;
        ops::nvfp4_global_absmax(&gpu, absmax_k, out_a, gmax_ref, n_tok * dim, stream)?;
        ops::gated_rms_norm_prefill_absmax(
            &gpu,
            gated_prefill_absmax_k,
            src_dev,
            hidden_ref,
            &norm_w,
            out_b,
            gmax_fused,
            4,
            128,
            eps,
            n_tok,
            dim,
            dim,
            stream,
        )?;
        gpu.synchronize(stream)?;
        gpu.copy_d2h(out_a, &mut raw_a)?;
        gpu.copy_d2h(out_b, &mut raw_b)?;
        gpu.copy_d2h(gmax_ref, &mut gm_a)?;
        gpu.copy_d2h(gmax_fused, &mut gm_b)?;
        assert_eq!(
            raw_a, raw_b,
            "gated_rms_norm_prefill_absmax must be 0-ULP bit-identical"
        );
        assert_eq!(
            gm_a, gm_b,
            "gated_rms_norm_prefill_absmax global_max must be bit-identical"
        );

        // 2c. sigmoid_gate_mul_batched vs sigmoid_gate_mul_batched_absmax
        gpu.memset_async(gmax_ref, 0, 4, stream)?;
        gpu.memset_async(gmax_fused, 0, 4, stream)?;
        ops::sigmoid_gate_mul_batched(
            &gpu, sig_gate_k, src_dev, hidden_ref, out_a, dim, dim, n_tok, stream,
        )?;
        ops::nvfp4_global_absmax(&gpu, absmax_k, out_a, gmax_ref, n_tok * dim, stream)?;
        ops::sigmoid_gate_mul_batched_absmax(
            &gpu,
            sig_gate_absmax_k,
            src_dev,
            hidden_ref,
            out_b,
            gmax_fused,
            dim,
            dim,
            n_tok,
            stream,
        )?;
        gpu.synchronize(stream)?;
        gpu.copy_d2h(out_a, &mut raw_a)?;
        gpu.copy_d2h(out_b, &mut raw_b)?;
        gpu.copy_d2h(gmax_ref, &mut gm_a)?;
        gpu.copy_d2h(gmax_fused, &mut gm_b)?;
        assert_eq!(
            raw_a, raw_b,
            "sigmoid_gate_mul_batched_absmax must be 0-ULP bit-identical"
        );
        assert_eq!(
            gm_a, gm_b,
            "sigmoid_gate_mul_batched_absmax global_max must be bit-identical"
        );

        // 2d. `quantize_bf16_to_nvfp4_atlas_128x4_from_absmax` legacy vs PTX
        let dyn_a = gpu.alloc(CUBLASLT_DYNAMIC_SCALARS_BYTES)?;
        let dyn_b = gpu.alloc(CUBLASLT_DYNAMIC_SCALARS_BYTES)?;
        let packed_b = gpu.alloc(full_layout.packed_bytes)?;
        let scale_b = gpu.alloc(full_layout.scale_bytes)?;
        gpu.memset_async(dyn_a, 0, CUBLASLT_DYNAMIC_ZERO_PREFIX_BYTES, stream)?;
        gpu.memset_async(dyn_b, 0, CUBLASLT_DYNAMIC_ZERO_PREFIX_BYTES, stream)?;
        ops::nvfp4_global_absmax(
            &gpu,
            absmax_k,
            out_a,
            dyn_a.offset(CUBLASLT_DYNAMIC_GLOBAL_MAX_OFF),
            n_tok * dim,
            stream,
        )?;
        ops::nvfp4_global_absmax(
            &gpu,
            absmax_k,
            out_a,
            dyn_b.offset(CUBLASLT_DYNAMIC_GLOBAL_MAX_OFF),
            n_tok * dim,
            stream,
        )?;
        for (kern, p_out, s_out, dyn_buf) in [
            (from_absmax_legacy_k, a_packed, a_scale, dyn_a),
            (from_absmax_ptx_k, packed_b, scale_b, dyn_b),
        ] {
            spark_runtime::kernel_args::KernelLaunch::new(&gpu, kern)
                .grid([128, 1, 1])
                .block([256, 1, 1])
                .arg_ptr(out_a)
                .arg_ptr(p_out)
                .arg_ptr(s_out)
                .arg_ptr(dyn_buf.offset(CUBLASLT_DYNAMIC_GLOBAL_MAX_OFF))
                .arg_ptr(dyn_buf.offset(CUBLASLT_DYNAMIC_SCALE2_OFF))
                .arg_ptr(dyn_buf.offset(CUBLASLT_DYNAMIC_STATUS_OFF))
                .arg_u32(n_tok)
                .arg_u32(dim)
                .launch(stream)?;
        }
        gpu.synchronize(stream)?;
        let p_bytes = (n_tok as usize) * (dim as usize / 2);
        let s_bytes = (n_tok as usize) * (dim as usize / 16);
        let mut pa = vec![0u8; p_bytes];
        let mut pb = vec![0u8; p_bytes];
        let mut sa = vec![0u8; s_bytes];
        let mut sb = vec![0u8; s_bytes];
        let mut s2a = [0u8; 4];
        let mut s2b = [0u8; 4];
        let mut sta = [0u8; 4];
        let mut stb = [0u8; 4];
        gpu.copy_d2h(a_packed, &mut pa)?;
        gpu.copy_d2h(packed_b, &mut pb)?;
        gpu.copy_d2h(a_scale, &mut sa)?;
        gpu.copy_d2h(scale_b, &mut sb)?;
        gpu.copy_d2h(dyn_a.offset(CUBLASLT_DYNAMIC_SCALE2_OFF), &mut s2a)?;
        gpu.copy_d2h(dyn_b.offset(CUBLASLT_DYNAMIC_SCALE2_OFF), &mut s2b)?;
        gpu.copy_d2h(dyn_a.offset(CUBLASLT_DYNAMIC_STATUS_OFF), &mut sta)?;
        gpu.copy_d2h(dyn_b.offset(CUBLASLT_DYNAMIC_STATUS_OFF), &mut stb)?;
        assert_eq!(s2a, s2b, "from_absmax_ptx scale2_a mismatch");
        assert_eq!(sta, [0, 0, 0, 0], "from_absmax legacy status non-zero");
        assert_eq!(stb, [0, 0, 0, 0], "from_absmax_ptx status non-zero");
        assert_eq!(
            sa, sb,
            "from_absmax_ptx 128x4 scales must match legacy byte-for-byte"
        );
        // Legacy uses round-half-away-from-zero (`atlas_quantize_e2m1`) while PTX uses
        // hardware IEEE RNE (`cvt.rn.satfinite.e2m1x2.f32`) + `rcp.approx.ftz.f32`:
        // every differing nibble must be an adjacent E2M1 midpoint tie (<= 1 step).
        let mut from_absmax_nibble_diffs = 0usize;
        for (&a, &b) in pa.iter().zip(&pb) {
            for shift in [0u8, 4u8] {
                let na = (a >> shift) & 0xF;
                let nb = (b >> shift) & 0xF;
                if na != nb {
                    from_absmax_nibble_diffs += 1;
                    let mag_a = (na & 0x7) as i8;
                    let mag_b = (nb & 0x7) as i8;
                    let same_sign = (na & 0x8) == (nb & 0x8);
                    assert!(
                        (same_sign && (mag_a - mag_b).abs() <= 1) || (mag_a == 0 && mag_b == 0),
                        "from_absmax_ptx nibble diverged by >1 E2M1 step: leg=0x{na:x} ptx=0x{nb:x}"
                    );
                }
            }
        }
        assert!(
            from_absmax_nibble_diffs * 100 <= pa.len() * 2,
            "from_absmax_ptx nibble_diffs too high: {from_absmax_nibble_diffs}/{}",
            pa.len() * 2
        );
        eprintln!(
            "PARITY OK dense_ffn SM120 (M={m}, strip={strip}, H={h}, Inter={inter}): \
             max_strip_diff={max_strip_diff:.6}, max_fused_diff={max_fused_diff:.6}, tol={tol:.6}, \
             merged_silu_scale_diffs={scale_diffs}/{}, merged_silu_nibble_ties={nibble_diffs}/{}, \
             from_absmax_ptx_scales=byte-exact, from_absmax_ptx_nibble_ties={from_absmax_nibble_diffs}/{}, \
             absmax_kernels=0-ULP-exact",
            sl.len(),
            pl.len() * 2,
            pa.len() * 2,
        );

        for ptr in [
            scale_b,
            packed_b,
            dyn_b,
            dyn_a,
            gmax_fused,
            gmax_ref,
            res_b,
            res_a,
            out_b,
            out_a,
            merged_out_strip,
            merged_out_full,
            a_scale,
            a_packed,
            down_s128,
            merged_s128,
            down_q.weight,
            down_q.weight_scale,
            merged_q.weight,
            merged_q.weight_scale,
            down_w_bf16,
            merged_w_bf16,
            ffn_out_strip,
            ffn_out_ref,
            norm_out_dev,
            resid_dev,
            norm_w_dev,
            src_dev,
            hidden_fused,
            hidden_strip,
            hidden_ref,
        ] {
            let _ = gpu.free(ptr);
        }
        Ok(())
    }

    #[test]
    fn sm120_coalesced_small_m_route_gates_gb10_and_selects_medium_n_warp_splitk() {
        // GB10 (<= 64 SMs) is always Disabled.
        assert_eq!(
            select_sm120_coalesced_small_m_route(true, true, 48, 8, 5120, 17408, 16, true),
            Sm120CoalescedSmallMRoute::Disabled
        );
        assert_eq!(
            select_sm120_coalesced_small_m_route(true, true, 64, 8, 5120, 25600, 32, true),
            Sm120CoalescedSmallMRoute::Disabled
        );
        // Explicit opt-out disables even on SM120.
        assert_eq!(
            select_sm120_coalesced_small_m_route(false, true, 188, 8, 5120, 17408, 16, true),
            Sm120CoalescedSmallMRoute::Disabled
        );
        // SM120 N=5120 deep-K projections (fc, down_proj, ssm.out, attn.o) pick MediumNWarpSplitK.
        assert_eq!(
            select_sm120_coalesced_small_m_route(true, true, 188, 8, 5120, 25600, 32, true),
            Sm120CoalescedSmallMRoute::MediumNWarpSplitK { prefer_v2: true }
        );
        assert_eq!(
            select_sm120_coalesced_small_m_route(true, true, 188, 8, 5120, 17408, 16, true),
            Sm120CoalescedSmallMRoute::MediumNWarpSplitK { prefer_v2: true }
        );
        assert_eq!(
            select_sm120_coalesced_small_m_route(true, true, 188, 8, 5120, 6144, 16, false),
            Sm120CoalescedSmallMRoute::MediumNWarpSplitK { prefer_v2: false }
        );
        // Wide deep-K projections (ssm.in_proj_qkvz N=16384, attn.q_proj N=12288) use
        // MediumNWarpSplitK (v2_n16_sk4_m16) when wide_sk4 is enabled, and lm_head
        // (N=248320) uses WideNWarpSplitK (v2_n32_sk4_m16). Narrow (N=1024) delegates.
        assert_eq!(
            select_sm120_coalesced_small_m_route(true, true, 188, 8, 16384, 5120, 16, true),
            Sm120CoalescedSmallMRoute::MediumNWarpSplitK { prefer_v2: true }
        );
        assert_eq!(
            select_sm120_coalesced_small_m_route(true, false, 188, 8, 16384, 5120, 16, true),
            Sm120CoalescedSmallMRoute::DelegateFastProj
        );
        assert_eq!(
            select_sm120_coalesced_small_m_route(true, true, 188, 8, 1024, 5120, 16, true),
            Sm120CoalescedSmallMRoute::DelegateFastProj
        );
        assert_eq!(
            select_sm120_coalesced_small_m_route(true, true, 188, 7, 248320, 5120, 16, true),
            Sm120CoalescedSmallMRoute::WideNWarpSplitK
        );
        assert_eq!(
            select_sm120_coalesced_small_m_route(true, false, 188, 7, 248320, 5120, 16, true),
            Sm120CoalescedSmallMRoute::DelegateFastProj
        );
        // Out-of-range M or unaligned K fail closed.
        assert_eq!(
            select_sm120_coalesced_small_m_route(true, true, 188, 0, 5120, 17408, 16, true),
            Sm120CoalescedSmallMRoute::Disabled
        );
        assert_eq!(
            select_sm120_coalesced_small_m_route(true, true, 188, 17, 5120, 17408, 16, true),
            Sm120CoalescedSmallMRoute::Disabled
        );
        assert_eq!(
            select_sm120_coalesced_small_m_route(true, true, 188, 8, 5120, 5136, 16, true),
            Sm120CoalescedSmallMRoute::Disabled
        );
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn qwen38_27b_ported_coalesced_bf16_kernels_match_reference_and_preserve_outliers()
    -> anyhow::Result<()> {
        use crate::weight_map::{DenseWeight, quantize_to_nvfp4};
        use half::bf16;
        use spark_runtime::gpu::GpuBackend;
        use spark_runtime::kernel_args::KernelLaunch;

        let Some(ptx) = atlas_kernels::ptx_for_model("qwen3.8-27b") else {
            return Ok(());
        };
        let Ok(gpu) = spark_runtime::cuda_backend::AtlasCudaBackend::new(0, &ptx.modules) else {
            return Ok(());
        };
        let stream = gpu.default_stream();

        let absmax_k = crate::layers::try_kernel(&gpu, "quantize_nvfp4", "nvfp4_global_absmax");
        let quantize_k =
            crate::layers::try_kernel(&gpu, "quantize_nvfp4", "quantize_bf16_to_nvfp4");
        let ref_k = crate::layers::try_kernel(&gpu, "w4a16", "w4a16_gemm");
        if absmax_k.0 == 0 || quantize_k.0 == 0 || ref_k.0 == 0 {
            return Ok(());
        }

        const E2M1_LUT: [f32; 16] = [
            0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
        ];
        fn fp8_e4m3_to_f32(bits: u8) -> f32 {
            let sign = if (bits & 0x80) != 0 { -1.0f32 } else { 1.0f32 };
            let exp = ((bits >> 3) & 0x0f) as i32;
            let mant = (bits & 0x07) as f32;
            if exp == 0 {
                sign * (mant / 8.0) * 2.0f32.powi(-6)
            } else if exp == 15 && (bits & 0x07) == 7 {
                f32::NAN
            } else {
                sign * (1.0 + mant / 8.0) * 2.0f32.powi(exp - 7)
            }
        }

        // Test shapes covering fc / down_proj / ssm.out (N=5120), ssm.qkvz (N=16384),
        // attn.k/v (N=1024), and un-normalized outliers > +-448 (which FP8 E4M3 clips).
        let cases: &[(&str, u32, u32, u32, u32, u32, u32)] = &[
            (
                "w4a16_gemm_coalesced_v2_n16_sk4_m16",
                8,
                512,
                1024,
                16,
                16,
                256,
            ),
            (
                "w4a16_gemm_coalesced_v2_n32_sk4_m16",
                8,
                512,
                1024,
                32,
                16,
                512,
            ),
            (
                "w4a16_gemm_coalesced_v2_n16_sk4_m32",
                32,
                512,
                1024,
                16,
                32,
                256,
            ),
            (
                "w4a16_gemm_coalesced_v2_n16_sk4_m32",
                27,
                512,
                1024,
                16,
                32,
                256,
            ),
            (
                "w4a16_gemm_coalesced_v2_n32_sk4_m32",
                32,
                512,
                1024,
                32,
                32,
                512,
            ),
            (
                "w4a16_gemm_coalesced_v2_n16_sk4_m64",
                64,
                512,
                1024,
                16,
                64,
                256,
            ),
            (
                "w4a16_gemm_coalesced_v2_n16_sk4_m64",
                53,
                512,
                1024,
                16,
                64,
                256,
            ),
            (
                "w4a16_gemm_coalesced_v2_n32_sk2_m64",
                64,
                512,
                1024,
                32,
                64,
                256,
            ),
            (
                "w4a16_gemm_coalesced_n16_sk4_m16",
                8,
                512,
                1024,
                16,
                16,
                256,
            ),
            ("w4a16_gemm_coalesced_n8_sk8_m16", 8, 512, 1024, 8, 16, 256),
            ("w4a16_gemm_coalesced_v2_n32_m16", 8, 512, 1024, 32, 16, 128),
            ("w4a16_gemm_coalesced_v2_n64_m16", 7, 512, 1024, 64, 16, 256),
        ];

        for &(sym, m, n, k, co_n, tile_m, threads) in cases {
            let kern = crate::layers::try_kernel(&gpu, "w4a16", sym);
            assert_ne!(
                kern.0, 0,
                "ported coalesced kernel `{sym}` missing from qwen3.8-27b PTX bundle"
            );

            // Deterministic BF16 activations including +-1200.0 outliers (> E4M3 max 448).
            let mut a_host = Vec::with_capacity((m * k) as usize);
            for r in 0..m {
                for c in 0..k {
                    let v = if c % 97 == 0 {
                        if (r + c) % 2 == 0 {
                            1200.0f32
                        } else {
                            -1200.0f32
                        }
                    } else {
                        ((((r * 31 + c * 17) % 101) as f32) - 50.0) * 0.02
                    };
                    a_host.push(bf16::from_f32(v));
                }
            }
            let mut w_host = Vec::with_capacity((n * k) as usize);
            for r in 0..n {
                for c in 0..k {
                    let v = ((((r * 13 + c * 29) % 89) as f32) - 44.0) * 0.008;
                    w_host.push(bf16::from_f32(v));
                }
            }

            let a_bytes: Vec<u8> = a_host.iter().flat_map(|v| v.to_le_bytes()).collect();
            let w_bytes: Vec<u8> = w_host.iter().flat_map(|v| v.to_le_bytes()).collect();
            let out_bytes = (m * n) as usize * 2;

            let a_dev = gpu.alloc(a_bytes.len())?;
            let w_dev = gpu.alloc(w_bytes.len())?;
            let out_ref_dev = gpu.alloc(out_bytes)?;
            let out_test_dev = gpu.alloc(out_bytes)?;
            gpu.copy_h2d(&a_bytes, a_dev)?;
            gpu.copy_h2d(&w_bytes, w_dev)?;

            let qw = quantize_to_nvfp4(
                &DenseWeight { weight: w_dev },
                n as usize,
                k as usize,
                &gpu,
                absmax_k,
                quantize_k,
                stream,
            )?;

            crate::layers::ops::w4a16_gemm(&gpu, ref_k, a_dev, &qw, out_ref_dev, m, n, k, stream)?;
            KernelLaunch::new(&gpu, kern)
                .grid([n.div_ceil(co_n), m.div_ceil(tile_m), 1])
                .block([threads, 1, 1])
                .arg_ptr(a_dev)
                .arg_ptr(qw.weight)
                .arg_ptr(qw.weight_scale)
                .arg_f32(qw.weight_scale_2)
                .arg_ptr(out_test_dev)
                .arg_u32(m)
                .arg_u32(n)
                .arg_u32(k)
                .launch(stream)?;
            gpu.synchronize(stream)?;

            let mut ref_raw = vec![0u8; out_bytes];
            let mut test_raw = vec![0u8; out_bytes];
            let mut packed_raw = vec![0u8; (n * (k / 2)) as usize];
            let mut scale_raw = vec![0u8; (n * (k / 16)) as usize];
            gpu.copy_d2h(out_ref_dev, &mut ref_raw)?;
            gpu.copy_d2h(out_test_dev, &mut test_raw)?;
            gpu.copy_d2h(qw.weight, &mut packed_raw)?;
            gpu.copy_d2h(qw.weight_scale, &mut scale_raw)?;

            // Exact FP32 reference matching the coalesced kernel's BF16 weight
            // dequant (`bf16(e2m1 * fp8)`) followed by epilogue `* scale2`.
            let mut max_abs_cpu = 0.0f32;
            let mut max_bf16_ulp = 0u32;
            let mut max_abs_vs_ref = 0.0f32;
            let mut max_abs_clipped = 0.0f32;
            for r in 0..m as usize {
                for col in 0..n as usize {
                    let mut sum_exact = 0.0f32;
                    let mut sum_clipped = 0.0f32;
                    for kk in 0..k as usize {
                        let byte = packed_raw[col * (k as usize / 2) + (kk / 2)];
                        let nib = if (kk & 1) == 1 {
                            (byte >> 4) & 0x0f
                        } else {
                            byte & 0x0f
                        };
                        let sc = fp8_e4m3_to_f32(scale_raw[col * (k as usize / 16) + (kk / 16)]);
                        let w_noscale2 = bf16::from_f32(E2M1_LUT[nib as usize] * sc).to_f32();
                        let a_val = a_host[r * k as usize + kk].to_f32();
                        sum_exact += a_val * w_noscale2;
                        sum_clipped += a_val.clamp(-448.0, 448.0) * w_noscale2;
                    }
                    let cpu_bf16_val = bf16::from_f32(sum_exact * qw.weight_scale_2);
                    let cpu_bf16 = cpu_bf16_val.to_f32();
                    let clipped_bf16 = bf16::from_f32(sum_clipped * qw.weight_scale_2).to_f32();
                    let idx = r * n as usize + col;
                    let ref_v =
                        bf16::from_le_bytes([ref_raw[2 * idx], ref_raw[2 * idx + 1]]).to_f32();
                    let test_bf16_val =
                        bf16::from_le_bytes([test_raw[2 * idx], test_raw[2 * idx + 1]]);
                    let test_v = test_bf16_val.to_f32();
                    assert!(
                        test_v.is_finite(),
                        "{sym} emitted non-finite output at {idx}"
                    );
                    let ulp = if cpu_bf16_val.to_bits() == test_bf16_val.to_bits()
                        || (cpu_bf16 == 0.0 && test_v == 0.0)
                    {
                        0u32
                    } else if (cpu_bf16_val.to_bits() & 0x8000)
                        == (test_bf16_val.to_bits() & 0x8000)
                    {
                        u32::from(cpu_bf16_val.to_bits().abs_diff(test_bf16_val.to_bits()))
                    } else {
                        u32::from(cpu_bf16_val.to_bits() & 0x7fff)
                            + u32::from(test_bf16_val.to_bits() & 0x7fff)
                    };
                    max_bf16_ulp = max_bf16_ulp.max(ulp);
                    max_abs_cpu = max_abs_cpu.max((cpu_bf16 - test_v).abs());
                    max_abs_vs_ref = max_abs_vs_ref.max((ref_v - test_v).abs());
                    max_abs_clipped = max_abs_clipped.max((clipped_bf16 - test_v).abs());
                }
            }
            eprintln!(
                "PARITY {sym} (M={m}, N={n}, K={k}): max_bf16_ulp={max_bf16_ulp}, max_abs_vs_exact_cpu={max_abs_cpu:.6}, max_abs_vs_w4a16_gemm={max_abs_vs_ref:.6}, max_diff_from_e4m3_clipped={max_abs_clipped:.3}"
            );
            assert!(
                max_bf16_ulp <= 1,
                "{sym} drifted vs exact BF16xBF16 CPU reference (max_bf16_ulp={max_bf16_ulp}, max_abs_cpu={max_abs_cpu:.6}, max_abs_vs_ref={max_abs_vs_ref:.6})"
            );
            assert!(
                max_abs_clipped > 100.0,
                "{sym} unexpectedly matched E4M3-clipped output (max_diff={max_abs_clipped:.3})"
            );

            for ptr in [
                a_dev,
                w_dev,
                out_ref_dev,
                out_test_dev,
                qw.weight,
                qw.weight_scale,
            ] {
                let _ = gpu.free(ptr);
            }
        }

        // Verify fused BF16xBF16 MMA coalesced warp-split-K gate+up+silu kernels
        // (`m16`, `m32`, `m64`) against exact FP32 CPU reference across full and
        // partial-M tail shapes (including K=128, 1024, 2560).
        let fused_cases: &[(&str, u32, u32, u32, u32, u32, u32)] = &[
            (
                "w4a16_gemm_coalesced_v2_n16_sk4_m16_gateup_silu",
                8,
                256,
                1024,
                16,
                16,
                256,
            ),
            (
                "w4a16_gemm_coalesced_v2_n32_sk4_m16_gateup_silu",
                8,
                256,
                1024,
                32,
                16,
                512,
            ),
            (
                "w4a16_gemm_coalesced_v2_n16_sk4_m16_gateup_silu",
                7,
                272,
                2560,
                16,
                16,
                256,
            ),
            (
                "w4a16_gemm_coalesced_v2_n16_sk4_m16_gateup_silu",
                1,
                128,
                128,
                16,
                16,
                256,
            ),
            (
                "w4a16_gemm_coalesced_v2_n16_sk4_m32_gateup_silu",
                32,
                256,
                1024,
                16,
                32,
                256,
            ),
            (
                "w4a16_gemm_coalesced_v2_n16_sk4_m32_gateup_silu",
                25,
                272,
                1024,
                16,
                32,
                256,
            ),
            (
                "w4a16_gemm_coalesced_v2_n16_sk4_m64_gateup_silu",
                64,
                256,
                1024,
                16,
                64,
                256,
            ),
            (
                "w4a16_gemm_coalesced_v2_n16_sk4_m64_gateup_silu",
                53,
                272,
                1024,
                16,
                64,
                256,
            ),
            (
                "w4a16_gemm_coalesced_v2_n32_sk2_m64_gateup_silu",
                64,
                256,
                1024,
                32,
                64,
                256,
            ),
            (
                "w4a16_gemm_coalesced_v2_n32_sk2_m64_gateup_silu",
                49,
                288,
                1024,
                32,
                64,
                256,
            ),
        ];
        for &(sym, m, n, k, co_n, tile_m, threads) in fused_cases {
            let kern = crate::layers::try_kernel(&gpu, "w4a16", sym);
            assert_ne!(kern.0, 0, "fused coalesced kernel `{sym}` missing");

            let mut a_host = Vec::with_capacity((m * k) as usize);
            for r in 0..m {
                for c in 0..k {
                    let v = ((((r * 31 + c * 17) % 101) as f32) - 50.0) * 0.015;
                    a_host.push(bf16::from_f32(v));
                }
            }
            let mut wg_host = Vec::with_capacity((n * k) as usize);
            let mut wu_host = Vec::with_capacity((n * k) as usize);
            for r in 0..n {
                for c in 0..k {
                    wg_host.push(bf16::from_f32(
                        ((((r * 13 + c * 29) % 89) as f32) - 44.0) * 0.006,
                    ));
                    wu_host.push(bf16::from_f32(
                        ((((r * 19 + c * 23) % 83) as f32) - 41.0) * 0.006,
                    ));
                }
            }

            let a_bytes: Vec<u8> = a_host.iter().flat_map(|v| v.to_le_bytes()).collect();
            let wg_bytes: Vec<u8> = wg_host.iter().flat_map(|v| v.to_le_bytes()).collect();
            let wu_bytes: Vec<u8> = wu_host.iter().flat_map(|v| v.to_le_bytes()).collect();
            let out_bytes = (m * n) as usize * 2;

            let a_dev = gpu.alloc(a_bytes.len())?;
            let wg_dev = gpu.alloc(wg_bytes.len())?;
            let wu_dev = gpu.alloc(wu_bytes.len())?;
            let out_dev = gpu.alloc(out_bytes)?;
            gpu.copy_h2d(&a_bytes, a_dev)?;
            gpu.copy_h2d(&wg_bytes, wg_dev)?;
            gpu.copy_h2d(&wu_bytes, wu_dev)?;

            let qg = quantize_to_nvfp4(
                &DenseWeight { weight: wg_dev },
                n as usize,
                k as usize,
                &gpu,
                absmax_k,
                quantize_k,
                stream,
            )?;
            let qu = quantize_to_nvfp4(
                &DenseWeight { weight: wu_dev },
                n as usize,
                k as usize,
                &gpu,
                absmax_k,
                quantize_k,
                stream,
            )?;

            KernelLaunch::new(&gpu, kern)
                .grid([n.div_ceil(co_n), m.div_ceil(tile_m), 1])
                .block([threads, 1, 1])
                .arg_ptr(a_dev)
                .arg_ptr(qg.weight)
                .arg_ptr(qg.weight_scale)
                .arg_f32(qg.weight_scale_2)
                .arg_ptr(qu.weight)
                .arg_ptr(qu.weight_scale)
                .arg_f32(qu.weight_scale_2)
                .arg_ptr(out_dev)
                .arg_u32(m)
                .arg_u32(n)
                .arg_u32(k)
                .launch(stream)?;
            gpu.synchronize(stream)?;

            let mut out_raw = vec![0u8; out_bytes];
            let mut pg_raw = vec![0u8; (n * (k / 2)) as usize];
            let mut sg_raw = vec![0u8; (n * (k / 16)) as usize];
            let mut pu_raw = vec![0u8; (n * (k / 2)) as usize];
            let mut su_raw = vec![0u8; (n * (k / 16)) as usize];
            gpu.copy_d2h(out_dev, &mut out_raw)?;
            gpu.copy_d2h(qg.weight, &mut pg_raw)?;
            gpu.copy_d2h(qg.weight_scale, &mut sg_raw)?;
            gpu.copy_d2h(qu.weight, &mut pu_raw)?;
            gpu.copy_d2h(qu.weight_scale, &mut su_raw)?;

            let mut max_abs = 0.0f32;
            let mut max_rel_den = 1.0f32;
            for r in 0..m as usize {
                for col in 0..n as usize {
                    let mut sum_g = 0.0f32;
                    let mut sum_u = 0.0f32;
                    for kk in 0..k as usize {
                        let a_val = a_host[r * k as usize + kk].to_f32();
                        let bg = pg_raw[col * (k as usize / 2) + (kk / 2)];
                        let ng = if (kk & 1) == 1 {
                            (bg >> 4) & 0x0f
                        } else {
                            bg & 0x0f
                        };
                        let scg = fp8_e4m3_to_f32(sg_raw[col * (k as usize / 16) + (kk / 16)]);
                        let wg_v = bf16::from_f32(E2M1_LUT[ng as usize] * scg).to_f32();
                        sum_g += a_val * wg_v;

                        let bu = pu_raw[col * (k as usize / 2) + (kk / 2)];
                        let nu = if (kk & 1) == 1 {
                            (bu >> 4) & 0x0f
                        } else {
                            bu & 0x0f
                        };
                        let scu = fp8_e4m3_to_f32(su_raw[col * (k as usize / 16) + (kk / 16)]);
                        let wu_v = bf16::from_f32(E2M1_LUT[nu as usize] * scu).to_f32();
                        sum_u += a_val * wu_v;
                    }
                    let gv = sum_g * qg.weight_scale_2;
                    let uv = sum_u * qu.weight_scale_2;
                    let cpu_v = bf16::from_f32((gv / (1.0 + (-gv).exp())) * uv).to_f32();
                    let idx = r * n as usize + col;
                    let got_v =
                        bf16::from_le_bytes([out_raw[2 * idx], out_raw[2 * idx + 1]]).to_f32();
                    assert!(
                        got_v.is_finite(),
                        "{sym} emitted non-finite output at {idx}"
                    );
                    max_abs = max_abs.max((cpu_v - got_v).abs());
                    max_rel_den = max_rel_den.max(cpu_v.abs());
                }
            }
            let rel = max_abs / max_rel_den;
            eprintln!(
                "PARITY FUSED {sym} (M={m}, N={n}, K={k}): max_abs={max_abs:.6}, rel={rel:.6}"
            );
            assert!(
                rel <= 0.015 && max_abs <= 0.05,
                "{sym} drifted vs exact CPU SiLU*mul ref (max_abs={max_abs:.6}, rel={rel:.6})"
            );

            for ptr in [
                a_dev,
                wg_dev,
                wu_dev,
                out_dev,
                qg.weight,
                qg.weight_scale,
                qu.weight,
                qu.weight_scale,
            ] {
                let _ = gpu.free(ptr);
            }
        }
        Ok(())
    }
}
