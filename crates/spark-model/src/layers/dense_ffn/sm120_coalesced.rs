// SPDX-License-Identifier: AGPL-3.0-only

//! SM120 coalesced NVFP4 small-M and warp-split-K GEMM dispatchers.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

use super::policy::PROJ_27B_CUBLASLT_NVFP4_SM_THRESHOLD;
use crate::weight_map::QuantizedWeight;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Sm120CoalescedSmallMRoute {
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
pub(super) const fn select_sm120_coalesced_small_m_route(
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
pub(super) fn sm120_coalesced_m16_env_enabled() -> bool {
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
pub(super) fn sm120_wide_sk4_env_enabled() -> bool {
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
pub(super) fn sm120_gateup_n32_env_enabled() -> bool {
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
pub(super) fn sm120_coalesced_wide_m_enabled() -> bool {
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
    let h = crate::layers::try_kernel(gpu, "w4a16", name);
    if h.0 != 0 {
        h
    } else {
        crate::layers::try_kernel(gpu, "w4a16_gemm", name)
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
