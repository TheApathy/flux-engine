// SPDX-License-Identifier: AGPL-3.0-only

//! Default-off Qwen4 prefill MoE router GEMM pipelining (`ATLAS_QWEN4_PREFILL_ROUTER_GEMM_PIPE=1`)
//! and fused 1-launch sort + contract + V8/V7 planner (`ATLAS_QWEN4_PREFILL_MOE_FUSED_SORT_PLAN=1`).
//! Both paths are 100% bit-exact (0 ULP) with their default-off counterparts.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

use super::qwen4_compact_contract as abi;
use super::qwen4_plan_order_rule::PLAN_V8_MODULE;
use super::qwen4_prefill_compact::{self, Kernels};
use crate::layers::ops;
use crate::weight_map::QuantizedWeight;

pub(super) const ROUTER_PIPE_SELECTOR: &str = "ATLAS_QWEN4_PREFILL_ROUTER_GEMM_PIPE";
pub(super) const ROUTER_PIPE_ALIAS: &str = "ATLAS_QWEN4_PREFILL_ROUTER_PIPE";
pub(super) const FUSED_SORT_PLAN_SELECTOR: &str = "ATLAS_QWEN4_PREFILL_MOE_FUSED_SORT_PLAN";
pub(super) const FUSED_SORT_PLAN_ALIAS: &str = "ATLAS_QWEN4_PREFILL_MOE_SORT_PLAN_FUSED";
pub(super) const FUSED_SORT_PLAN_KERNEL: &str = "qwen4_moe_sort_plan_t_v8";
pub(super) const FUSED_SORT_PLAN_P1_HIST_KERNEL: &str = "qwen4_moe_sort_plan_t_v8_p1_hist";
pub(super) const FUSED_SORT_PLAN_P2_PLAN_KERNEL: &str = "qwen4_moe_sort_plan_t_v8_p2_plan";
pub(super) const FUSED_SORT_PLAN_P3_SCATTER_KERNEL: &str = "qwen4_moe_sort_plan_t_v8_p3_scatter";

/// Minimum prefill rows for 3-phase multi-CTA `sort_plan_v8` on wide-SM GPUs.
/// Smaller batches (`rows < 512`) stay on single-launch `grid=[1, 1, 1]` to avoid
/// 3-kernel launch overhead when the work fits in a few microseconds.
const MULTI_CTA_SORT_PLAN_MIN_ROWS: usize = 512;
/// SM count above which 3-phase multi-CTA `sort_plan_v8` activates (`<= 64` keeps GB10
/// on single-CTA `grid=[1, 1, 1]`).
const MULTI_CTA_SORT_PLAN_HIGH_SM_THRESHOLD: u32 = 64;
/// Upper cap on multi-CTA `sort_plan_v8` grid width (bounds scratch to `<= 512 KiB`,
/// while saturating all 188 SMs on SM120 with `188 * 2 KiB = 376 KiB <= 384 KiB`).
const MULTI_CTA_SORT_PLAN_MAX_GRID: u32 = 256;
/// Thread-block width of Phase-1 histogram and Phase-3 scatter kernels.
const MULTI_CTA_SORT_PLAN_BLOCK: u32 = 256;
/// Expert count for Qwen4 Flash-Next compact MoE (`512` bins per CTA histogram row).
const MULTI_CTA_SORT_PLAN_EXPERTS: usize = 512;

static ROUTER_PIPE_ENGAGED: AtomicBool = AtomicBool::new(false);
static BATCHED_VERIFY_TC_ROUTER_ENGAGED: AtomicBool = AtomicBool::new(false);
static FUSED_SORT_PLAN_ENGAGED: AtomicBool = AtomicBool::new(false);
static MULTI_CTA_SORT_PLAN_ENGAGED: AtomicBool = AtomicBool::new(false);

pub(super) fn router_pipe_selected() -> Result<bool> {
    static SEL: OnceLock<Result<bool, String>> = OnceLock::new();
    SEL.get_or_init(|| {
        let primary = std::env::var(ROUTER_PIPE_SELECTOR).ok();
        let val = primary.or_else(|| std::env::var(ROUTER_PIPE_ALIAS).ok());
        abi::parse(val.as_deref()).map_err(|e| format!("{ROUTER_PIPE_SELECTOR}: {e}"))
    })
    .clone()
    .map_err(anyhow::Error::msg)
}

pub(super) fn fused_sort_plan_selected() -> Result<bool> {
    static SEL: OnceLock<Result<bool, String>> = OnceLock::new();
    SEL.get_or_init(|| {
        let primary = std::env::var(FUSED_SORT_PLAN_SELECTOR).ok();
        let val = primary.or_else(|| std::env::var(FUSED_SORT_PLAN_ALIAS).ok());
        abi::parse(val.as_deref()).map_err(|e| format!("{FUSED_SORT_PLAN_SELECTOR}: {e}"))
    })
    .clone()
    .map_err(anyhow::Error::msg)
}

fn pipe_m128_kernel(gpu: &dyn GpuBackend) -> Result<KernelHandle> {
    static K: OnceLock<Result<KernelHandle, String>> = OnceLock::new();
    K.get_or_init(|| {
        gpu.kernel("w4a16", "w4a16_gemm_pipe_m128n128")
            .or_else(|_| gpu.kernel("w4a16_gemm", "w4a16_gemm_pipe_m128n128"))
            .map_err(|e| e.to_string())
    })
    .clone()
    .map_err(|e| anyhow::anyhow!("{ROUTER_PIPE_SELECTOR}: w4a16_gemm_pipe_m128n128: {e}"))
}

fn pipe_m64_kernel(gpu: &dyn GpuBackend) -> Result<KernelHandle> {
    static K: OnceLock<Result<KernelHandle, String>> = OnceLock::new();
    K.get_or_init(|| {
        gpu.kernel("w4a16", "w4a16_gemm_pipe")
            .or_else(|_| gpu.kernel("w4a16_gemm", "w4a16_gemm_pipe"))
            .map_err(|e| e.to_string())
    })
    .clone()
    .map_err(|e| anyhow::anyhow!("{ROUTER_PIPE_SELECTOR}: w4a16_gemm_pipe: {e}"))
}

/// Launch the single-kernel K-coalesced NVFP4 Tensor Core router GEMM
/// (`w4a16_gemm_coalesced_*` Warp-Split-K / narrow tiers for `num_experts <= 1024`)
/// for batched verify (`rows in 6..=256`), returning
/// `Ok(false)` when disabled (`ATLAS_QWEN4_BATCHED_VERIFY_COALESCED_NVFP4=0`) or
/// prerequisites are absent so the caller falls back cleanly to the exact scalar
/// GEMV path.
#[allow(clippy::too_many_arguments)]
pub(super) fn try_launch_batched_verify_router_gemm(
    gpu: &dyn GpuBackend,
    router_in: DevicePtr,
    nvfp4: &QuantizedWeight,
    gate_logits: DevicePtr,
    rows: u32,
    num_experts: u32,
    hidden: u32,
    stream: u64,
) -> Result<bool> {
    if !crate::layers::qwen4_fast_proj::coalesced_nvfp4_dense_enabled() {
        return Ok(false);
    }
    if !crate::layers::qwen4_fast_proj::try_coalesced_nvfp4_w4a16_gemm(
        gpu,
        router_in,
        nvfp4,
        gate_logits,
        rows as usize,
        num_experts as usize,
        hidden as usize,
        stream,
    )? {
        return Ok(false);
    }
    if !BATCHED_VERIFY_TC_ROUTER_ENGAGED.swap(true, Ordering::Relaxed) {
        let kernel = crate::layers::qwen4_fast_proj::coalesced_nvfp4_dispatch_symbol(
            rows as usize,
            num_experts as usize,
            hidden as usize,
        );
        tracing::info!(
            selector = crate::layers::qwen4_fast_proj::COALESCED_NVFP4_DENSE_ENV,
            kernel,
            rows,
            experts = num_experts,
            hidden,
            "QWEN4_BATCHED_VERIFY_TC_ROUTER_ENGAGED"
        );
    }
    Ok(true)
}

fn fused_sort_plan_kernel(gpu: &dyn GpuBackend) -> Result<KernelHandle> {
    static K: OnceLock<Result<KernelHandle, String>> = OnceLock::new();
    K.get_or_init(|| {
        gpu.kernel(PLAN_V8_MODULE, FUSED_SORT_PLAN_KERNEL)
            .map_err(|e| e.to_string())
    })
    .clone()
    .map_err(|e| anyhow::anyhow!("{FUSED_SORT_PLAN_SELECTOR}: {FUSED_SORT_PLAN_KERNEL}: {e}"))
}

#[derive(Clone, Copy)]
struct MultiCtaSortPlanKernels {
    p1_hist: KernelHandle,
    p2_plan: KernelHandle,
    p3_scatter: KernelHandle,
}

fn resolve_multi_cta_sort_plan_kernels(gpu: &dyn GpuBackend) -> Option<MultiCtaSortPlanKernels> {
    let p1_hist = gpu
        .kernel(PLAN_V8_MODULE, FUSED_SORT_PLAN_P1_HIST_KERNEL)
        .ok()?;
    let p2_plan = gpu
        .kernel(PLAN_V8_MODULE, FUSED_SORT_PLAN_P2_PLAN_KERNEL)
        .ok()?;
    let p3_scatter = gpu
        .kernel(PLAN_V8_MODULE, FUSED_SORT_PLAN_P3_SCATTER_KERNEL)
        .ok()?;
    Some(MultiCtaSortPlanKernels {
        p1_hist,
        p2_plan,
        p3_scatter,
    })
}

fn multi_cta_sort_plan_kernels(gpu: &dyn GpuBackend) -> Option<MultiCtaSortPlanKernels> {
    static K: OnceLock<Option<MultiCtaSortPlanKernels>> = OnceLock::new();
    *K.get_or_init(|| resolve_multi_cta_sort_plan_kernels(gpu))
}

/// Compute the 3-phase multi-CTA `sort_plan_v8` grid width `G` for wide-SM GPUs
/// (`sm_count > 64`) at `rows >= 512`. Returns `None` on GB10 (`<= 64` SMs) or
/// small batches (`rows < 512`) so they remain on single-CTA `grid=[1, 1, 1]`.
#[inline]
#[must_use]
pub(super) fn multi_cta_sort_plan_grid_for_sm(rows: usize, sm_count: u32) -> Option<u32> {
    if sm_count <= MULTI_CTA_SORT_PLAN_HIGH_SM_THRESHOLD || rows < MULTI_CTA_SORT_PLAN_MIN_ROWS {
        return None;
    }
    let pairs = rows.checked_mul(10)?;
    let work_ctas = pairs.div_ceil(MULTI_CTA_SORT_PLAN_BLOCK as usize);
    let grid = (sm_count as usize)
        .min(work_ctas)
        .min(MULTI_CTA_SORT_PLAN_MAX_GRID as usize);
    (grid > 1).then_some(grid as u32)
}

/// Scratch bytes required by the multi-CTA histogram table `hist[G * 512]` (`u32`).
#[inline]
#[must_use]
pub(super) const fn multi_cta_sort_plan_hist_bytes(grid: u32) -> usize {
    (grid as usize) * MULTI_CTA_SORT_PLAN_EXPERTS * std::mem::size_of::<u32>()
}

/// Device buffers bound to fused `qwen4_moe_sort_plan_t_v8` (single-CTA or 3-phase multi-CTA).
#[derive(Clone, Copy, Debug)]
pub(super) struct FusedSortPlanBuffers {
    pub topk_ids: DevicePtr,
    pub sorted_token_ids: DevicePtr,
    pub expert_offsets: DevicePtr,
    pub token_to_perm: DevicePtr,
    pub contract: DevicePtr,
    pub workspace: DevicePtr,
    pub status: DevicePtr,
    /// Optional 16-byte-aligned dead scratch `(ptr, cap_bytes)` (e.g. `expert_gate_out`)
    /// used for the per-CTA `[G, 512]` `u32` histogram table when 3-phase multi-CTA engages.
    pub hist_scratch: Option<(DevicePtr, usize)>,
}

#[allow(clippy::too_many_arguments)]
pub(super) fn try_launch_router_gemm(
    gpu: &dyn GpuBackend,
    router_in: DevicePtr,
    nvfp4: &QuantizedWeight,
    gate_logits: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<bool> {
    if !router_pipe_selected()? || m < 2 || !k.is_multiple_of(32) {
        return Ok(false);
    }
    if m >= 128 && m.is_multiple_of(128) && n.is_multiple_of(128) {
        let handle = pipe_m128_kernel(gpu)?;
        ops::w4a16_gemm_pipe_m128n128(gpu, handle, router_in, nvfp4, gate_logits, m, n, k, stream)?;
        if !ROUTER_PIPE_ENGAGED.swap(true, Ordering::Relaxed) {
            tracing::info!(
                "QWEN4_PREFILL_ROUTER_GEMM_PIPE_ENGAGED selector={ROUTER_PIPE_SELECTOR} kernel=w4a16_gemm_pipe_m128n128 rows={m} experts={n} hidden={k}"
            );
        }
        return Ok(true);
    }
    if n.is_multiple_of(64) {
        let handle = pipe_m64_kernel(gpu)?;
        ops::w4a16_gemm_pipe(gpu, handle, router_in, nvfp4, gate_logits, m, n, k, stream)?;
        if !ROUTER_PIPE_ENGAGED.swap(true, Ordering::Relaxed) {
            tracing::info!(
                "QWEN4_PREFILL_ROUTER_GEMM_PIPE_ENGAGED selector={ROUTER_PIPE_SELECTOR} kernel=w4a16_gemm_pipe rows={m} experts={n} hidden={k}"
            );
        }
        return Ok(true);
    }
    Ok(false)
}

pub(super) fn fused_sort_plan_active(
    compact_active: bool,
    kernels: Option<Kernels>,
    rows: usize,
) -> Result<bool> {
    if !compact_active
        || !fused_sort_plan_selected()?
        || super::dump::enabled()
        || !(2..=abi::MAX_ROWS).contains(&rows)
        || qwen4_prefill_compact::check_selected()?
        || kernels.and_then(|k| k.wide).is_none()
    {
        return Ok(false);
    }
    Ok(qwen4_prefill_compact::w4a8_selected()?
        || matches!(qwen4_prefill_compact::v2_level()?, 7 | 8))
}

fn launch_fused_sort_plan_for_sm(
    gpu: &dyn GpuBackend,
    bufs: FusedSortPlanBuffers,
    rows: u32,
    flags: u32,
    single_cta_handle: Option<KernelHandle>,
    multi_cta_handles: Option<MultiCtaSortPlanKernels>,
    sm_count: u32,
    stream: u64,
) -> Result<()> {
    let use_v8_inter = (flags & 1u32) != 0u32;
    let m256 = (flags & 2u32) != 0u32;
    if let Some(grid) = multi_cta_sort_plan_grid_for_sm(rows as usize, sm_count)
        && let Some((hist_ptr, hist_cap)) = bufs.hist_scratch
        && !hist_ptr.is_null()
        && hist_ptr.0.is_multiple_of(16)
        && hist_cap >= multi_cta_sort_plan_hist_bytes(grid)
        && let Some(kernels) = multi_cta_handles
    {
        KernelLaunch::new(gpu, kernels.p1_hist)
            .grid([grid, 1, 1])
            .block([MULTI_CTA_SORT_PLAN_BLOCK, 1, 1])
            .arg_ptr(bufs.topk_ids)
            .arg_ptr(hist_ptr)
            .arg_u32(rows)
            .launch(stream)?;
        KernelLaunch::new(gpu, kernels.p2_plan)
            .grid([1, 1, 1])
            .block([512, 1, 1])
            .arg_ptr(bufs.topk_ids)
            .arg_ptr(bufs.sorted_token_ids)
            .arg_ptr(bufs.expert_offsets)
            .arg_ptr(bufs.token_to_perm)
            .arg_ptr(hist_ptr)
            .arg_ptr(bufs.contract)
            .arg_ptr(bufs.workspace)
            .arg_ptr(bufs.status)
            .arg_u32(rows)
            .arg_u32(flags)
            .arg_u32(grid)
            .launch(stream)?;
        KernelLaunch::new(gpu, kernels.p3_scatter)
            .grid([grid, 1, 1])
            .block([MULTI_CTA_SORT_PLAN_BLOCK, 1, 1])
            .arg_ptr(bufs.topk_ids)
            .arg_ptr(bufs.expert_offsets)
            .arg_ptr(hist_ptr)
            .arg_ptr(bufs.sorted_token_ids)
            .arg_ptr(bufs.token_to_perm)
            .arg_ptr(bufs.status)
            .arg_u32(rows)
            .launch(stream)?;
        if !MULTI_CTA_SORT_PLAN_ENGAGED.swap(true, Ordering::Relaxed) {
            tracing::info!(
                "MOE_PREFILL_FUSED_SORT_PLAN_MULTI_CTA_ENGAGED selector={FUSED_SORT_PLAN_SELECTOR} \
                 rows={rows} grid={grid} sm_count={sm_count} order={} tile_m={}",
                if use_v8_inter { "inter" } else { "v7" },
                if m256 { 256 } else { 128 },
            );
        }
        return Ok(());
    }

    let handle = match single_cta_handle {
        Some(h) => h,
        None => fused_sort_plan_kernel(gpu)?,
    };
    KernelLaunch::new(gpu, handle)
        .grid([1, 1, 1])
        .block([512, 1, 1])
        .arg_ptr(bufs.topk_ids)
        .arg_ptr(bufs.sorted_token_ids)
        .arg_ptr(bufs.expert_offsets)
        .arg_ptr(bufs.token_to_perm)
        .arg_ptr(bufs.contract)
        .arg_ptr(bufs.workspace)
        .arg_ptr(bufs.status)
        .arg_u32(rows)
        .arg_u32(flags)
        .launch(stream)?;
    if !FUSED_SORT_PLAN_ENGAGED.swap(true, Ordering::Relaxed) {
        tracing::info!(
            "MOE_PREFILL_FUSED_SORT_PLAN_ENGAGED selector={FUSED_SORT_PLAN_SELECTOR} kernel={FUSED_SORT_PLAN_KERNEL} rows={rows} order={} tile_m={}",
            if use_v8_inter { "inter" } else { "v7" },
            if m256 { 256 } else { 128 },
        );
    }
    Ok(())
}

pub(super) fn launch_fused_sort_plan(
    gpu: &dyn GpuBackend,
    bufs: FusedSortPlanBuffers,
    rows: u32,
    flags: u32,
    single_cta_handle: Option<KernelHandle>,
    stream: u64,
) -> Result<()> {
    let sm_count = atlas_core::device::active_sm_count();
    let multi_cta_handles = if multi_cta_sort_plan_grid_for_sm(rows as usize, sm_count).is_some() {
        multi_cta_sort_plan_kernels(gpu)
    } else {
        None
    };
    launch_fused_sort_plan_for_sm(
        gpu,
        bufs,
        rows,
        flags,
        single_cta_handle,
        multi_cta_handles,
        sm_count,
        stream,
    )
}

#[cfg(test)]
mod tests {
    use spark_runtime::gpu::mock::MockGpuBackend;
    use spark_runtime::gpu::{DevicePtr, GpuBackend};

    use super::try_launch_batched_verify_router_gemm;
    use crate::layers::qwen4_fast_proj::{
        CoalescedNvfp4MTile, CoalescedNvfp4NTile, coalesced_nvfp4_dispatch_symbol,
        select_coalesced_nvfp4_tile,
    };
    use crate::weight_map::QuantizedWeight;

    fn sample_weight(weight: DevicePtr, scale2: f32) -> QuantizedWeight {
        QuantizedWeight {
            weight,
            weight_scale: DevicePtr(0x2000),
            weight_scale_2: scale2,
            input_scale: DevicePtr::NULL,
        }
    }

    #[test]
    fn batched_verify_router_selects_n16_coalesced_tiers_for_512_experts() {
        for (rows, m_tile, sym) in [
            (
                6usize,
                CoalescedNvfp4MTile::M32,
                "w4a16_gemm_coalesced_n16_m32",
            ),
            (7, CoalescedNvfp4MTile::M32, "w4a16_gemm_coalesced_n16_m32"),
            (8, CoalescedNvfp4MTile::M32, "w4a16_gemm_coalesced_n16_m32"),
            (10, CoalescedNvfp4MTile::M32, "w4a16_gemm_coalesced_n16_m32"),
            (12, CoalescedNvfp4MTile::M32, "w4a16_gemm_coalesced_n16_m32"),
            (20, CoalescedNvfp4MTile::M32, "w4a16_gemm_coalesced_n16_m32"),
            (40, CoalescedNvfp4MTile::M64, "w4a16_gemm_coalesced_n16_m64"),
            (
                80,
                CoalescedNvfp4MTile::M128,
                "w4a16_gemm_coalesced_n16_m128",
            ),
            (
                128,
                CoalescedNvfp4MTile::M128,
                "w4a16_gemm_coalesced_n16_m128",
            ),
        ] {
            let tile = select_coalesced_nvfp4_tile(rows, 512).expect("router tile");
            assert_eq!(tile.n_tile, CoalescedNvfp4NTile::N16, "rows={rows}");
            assert_eq!(tile.m_tile, m_tile, "rows={rows}");
            assert_eq!(tile.co_n(), 16, "rows={rows}");
            assert_eq!(tile.threads(), 64, "rows={rows}");
            assert_eq!(tile.symbol(), sym, "rows={rows}");
        }

        // Flash-Next router shape (n=512, k=2560) dispatches Warp-Split-K symbols.
        for (rows, expected) in [
            (6usize, "w4a16_gemm_coalesced_n8_sk8_m16"),
            (16, "w4a16_gemm_coalesced_n8_sk8_m16"),
            (20, "w4a16_gemm_coalesced_n16_sk4_m16"),
            (32, "w4a16_gemm_coalesced_n32_sk4_m16"),
            (80, "w4a16_gemm_coalesced_n32_sk4_m16"),
            (128, "w4a16_gemm_coalesced_n32_sk4_m16"),
        ] {
            assert_eq!(
                coalesced_nvfp4_dispatch_symbol(rows, 512, 2560),
                expected,
                "rows={rows} Flash-Next router receipt symbol"
            );
        }
    }

    #[test]
    fn batched_verify_router_launch_is_single_kernel_zero_alloc_and_fail_closed() {
        let gpu = MockGpuBackend::new();
        let w = sample_weight(DevicePtr(0x1000), 1.0);
        for (idx, rows) in [6u32, 7, 8, 10, 12, 20, 40, 80, 128]
            .into_iter()
            .enumerate()
        {
            let launched = try_launch_batched_verify_router_gemm(
                &gpu,
                DevicePtr(0x4000),
                &w,
                DevicePtr(0x8000),
                rows,
                512,
                2560,
                0,
            )
            .expect("coalesced router launch");
            assert!(launched, "rows={rows}");
            assert_eq!(gpu.launch_count(), idx + 1);
        }
        assert_eq!(gpu.alloc_count(), 0);
        assert_eq!(gpu.memset_count(), 0);

        let launches_before = gpu.launch_count();
        // Misaligned router_in, non-multiple-of-32 K, > 128 rows, null pointers, or non-finite scale2 fail closed.
        for (router_in, weight, gate_logits, rows, num_experts, hidden) in [
            (
                DevicePtr(0x4008),
                w,
                DevicePtr(0x8000),
                40u32,
                512u32,
                2560u32,
            ),
            (DevicePtr(0x4000), w, DevicePtr(0x8000), 40, 512, 48),
            (DevicePtr(0x4000), w, DevicePtr(0x8000), 0, 512, 2560),
            (DevicePtr(0x4000), w, DevicePtr(0x8000), 257, 512, 2560),
            (DevicePtr(0x4000), w, DevicePtr(0x8000), 80, 0, 2560),
            (DevicePtr::NULL, w, DevicePtr(0x8000), 80, 512, 2560),
            (DevicePtr(0x4000), w, DevicePtr::NULL, 80, 512, 2560),
            (
                DevicePtr(0x4000),
                sample_weight(DevicePtr::NULL, 1.0),
                DevicePtr(0x8000),
                80,
                512,
                2560,
            ),
            (
                DevicePtr(0x4000),
                sample_weight(DevicePtr(0x1000), f32::NAN),
                DevicePtr(0x8000),
                80,
                512,
                2560,
            ),
            (
                DevicePtr(0x4000),
                sample_weight(DevicePtr(0x1000), f32::INFINITY),
                DevicePtr(0x8000),
                80,
                512,
                2560,
            ),
        ] {
            let ok = try_launch_batched_verify_router_gemm(
                &gpu,
                router_in,
                &weight,
                gate_logits,
                rows,
                num_experts,
                hidden,
                0,
            )
            .expect("invalid router input fail-closed");
            assert!(!ok);
        }
        assert_eq!(gpu.launch_count(), launches_before);
    }

    #[test]
    fn multi_cta_sort_plan_policy_scales_to_188_sms_and_preserves_gb10_and_small_batches() {
        use super::{
            FusedSortPlanBuffers, launch_fused_sort_plan_for_sm, multi_cta_sort_plan_grid_for_sm,
            multi_cta_sort_plan_hist_bytes, resolve_multi_cta_sort_plan_kernels,
        };

        // GB10 (<= 64 SMs) and small batches (< 512 rows) remain single-CTA.
        for sm in [0u32, 48, 64] {
            for rows in [2usize, 128, 256, 512, 2048, 8192, 16_384] {
                assert_eq!(
                    multi_cta_sort_plan_grid_for_sm(rows, sm),
                    None,
                    "sm={sm} rows={rows} must stay single-CTA"
                );
            }
        }
        for rows in [2usize, 64, 128, 256, 511] {
            assert_eq!(multi_cta_sort_plan_grid_for_sm(rows, 188), None);
        }

        // Wide-SM (> 64 SMs), rows >= 512 scales grid = min(sm_count, ceil(rows*10/256), 256).
        assert_eq!(multi_cta_sort_plan_grid_for_sm(512, 188), Some(20));
        assert_eq!(multi_cta_sort_plan_grid_for_sm(2048, 188), Some(80));
        assert_eq!(multi_cta_sort_plan_grid_for_sm(8192, 188), Some(188));
        assert_eq!(multi_cta_sort_plan_grid_for_sm(16_384, 188), Some(188));
        assert_eq!(multi_cta_sort_plan_grid_for_sm(16_384, 170), Some(170));
        assert_eq!(multi_cta_sort_plan_grid_for_sm(16_384, 512), Some(256));
        assert_eq!(multi_cta_sort_plan_hist_bytes(188), 188 * 512 * 4);
        assert!(multi_cta_sort_plan_hist_bytes(188) <= 384 * 1024);

        // Mock launch verification: 3 kernels on 188 SMs for rows >= 512 with sufficient scratch;
        // 1 kernel on GB10 (48 SMs), small rows, or undersized/unaligned scratch.
        let gpu = MockGpuBackend::new();
        let multi_k = resolve_multi_cta_sort_plan_kernels(&gpu);
        let single_k = gpu
            .kernel(super::PLAN_V8_MODULE, super::FUSED_SORT_PLAN_KERNEL)
            .ok();
        let base_bufs = FusedSortPlanBuffers {
            topk_ids: DevicePtr(0x1_0000),
            sorted_token_ids: DevicePtr(0x2_0000),
            expert_offsets: DevicePtr(0x3_0000),
            token_to_perm: DevicePtr(0x4_0000),
            contract: DevicePtr(0x5_0000),
            workspace: DevicePtr(0x6_0000),
            status: DevicePtr(0x7_0000),
            hist_scratch: Some((DevicePtr(0x8_0000), 1024 * 1024)),
        };

        launch_fused_sort_plan_for_sm(&gpu, base_bufs, 16_384, 1, single_k, multi_k, 188, 0)
            .expect("multi-CTA 3-phase launch");
        assert_eq!(gpu.launch_count(), 3);
        assert_eq!(gpu.alloc_count(), 0);

        // GB10 (48 SMs) stays single-CTA (1 launch).
        launch_fused_sort_plan_for_sm(&gpu, base_bufs, 16_384, 1, single_k, multi_k, 48, 0)
            .expect("single-CTA GB10 launch");
        assert_eq!(gpu.launch_count(), 4);

        // Undersized scratch falls back cleanly to single-CTA (1 launch).
        let small_scratch = FusedSortPlanBuffers {
            hist_scratch: Some((DevicePtr(0x8_0000), 1024)),
            ..base_bufs
        };
        launch_fused_sort_plan_for_sm(&gpu, small_scratch, 16_384, 3, single_k, multi_k, 188, 0)
            .expect("fallback single-CTA launch");
        assert_eq!(gpu.launch_count(), 5);
        assert_eq!(gpu.alloc_count(), 0);
    }
}
