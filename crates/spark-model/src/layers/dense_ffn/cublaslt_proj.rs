// SPDX-License-Identifier: AGPL-3.0-only

//! cuBLASLt NVFP4 scale swizzling, zero-sync dynamic quantizer, shared scratch, and 27B projection routes.

#[cfg(feature = "cuda")]
use std::sync::Mutex;

use anyhow::Result;
#[cfg(feature = "cuda")]
use anyhow::{Context, ensure};
#[cfg(feature = "cuda")]
use spark_runtime::gpu::KernelHandle;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

#[cfg(feature = "cuda")]
use super::e2m1_flashinfer::e2m1_scratch_layout;
#[cfg(feature = "cuda")]
use super::policy::{
    PROJ_27B_CUBLASLT_NVFP4_MIN_M, cublaslt_merged_gate_up_vram_ok_for_sm, gpu_sm_count,
    prefill_fuse_absmax_enabled, proj_27b_cublaslt_nvfp4_enabled, proj_27b_cublaslt_nvfp4_shape_ok,
    select_cublaslt_quantize_128x4_kernel,
};
#[cfg(feature = "cuda")]
use crate::layers::ops;
use crate::weight_map::QuantizedWeight;

/// Device scalar scratch layout for zero-sync dynamic NVFP4 quantization:
/// `[0..4] global_max_f32 | [4..8] scale2_a_f32 | [8..12] status_u32 |
///  [12..16] alpha_0 | [16..20] alpha_1 | [20..24] alpha_2` (32 B, 16-B aligned).
#[cfg(feature = "cuda")]
pub(super) const CUBLASLT_DYNAMIC_SCALARS_BYTES: usize = 32;
#[cfg(feature = "cuda")]
pub(super) const CUBLASLT_DYNAMIC_ZERO_PREFIX_BYTES: usize = 12;
#[cfg(feature = "cuda")]
pub(super) const CUBLASLT_DYNAMIC_GLOBAL_MAX_OFF: usize = 0;
#[cfg(feature = "cuda")]
pub(super) const CUBLASLT_DYNAMIC_SCALE2_OFF: usize = 4;
#[cfg(feature = "cuda")]
pub(super) const CUBLASLT_DYNAMIC_STATUS_OFF: usize = 8;
#[cfg(feature = "cuda")]
pub(super) const CUBLASLT_DYNAMIC_ALPHA0_OFF: usize = 12;
#[cfg(feature = "cuda")]
pub(super) const CUBLASLT_DYNAMIC_ALPHA1_OFF: usize = 16;
#[cfg(feature = "cuda")]
pub(super) const CUBLASLT_DYNAMIC_ALPHA2_OFF: usize = 20;

/// Independent device `alpha` slot for projections sharing one dynamic activation
/// quantization on the same CUDA stream (e.g. gate/up or Q/K/V).
#[cfg(feature = "cuda")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum CublasLtAlphaSlot {
    Primary,
    Secondary,
    Tertiary,
}

/// Activation second-level scale source for a cuBLASLt NVFP4 GEMM.
#[cfg(feature = "cuda")]
#[derive(Debug, Clone, Copy)]
pub(super) enum CublasLtInputScale {
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
pub(super) struct CublasLtNvfp4Scales {
    pub(super) gate: DevicePtr,
    pub(super) up: DevicePtr,
    pub(super) down: DevicePtr,
    pub(super) dynamic_scalars: DevicePtr,
    pub(super) merged_weight: Option<DevicePtr>,
    pub(super) owns_merged_weight: bool,
    pub(super) merged_scales: Option<DevicePtr>,
    pub(super) owns_merged_scales: bool,
    pub(super) gate_src: DevicePtr,
    pub(super) up_src: DevicePtr,
    pub(super) down_src: DevicePtr,
}

#[cfg(feature = "cuda")]
pub(super) fn swizzle_nvfp4_scales_128x4_host(logical: &[u8], n: u32, k: u32) -> Result<Vec<u8>> {
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
pub(super) fn upload_swizzled_nvfp4_scales_128x4(
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
pub(super) fn upload_swizzled_merged_gate_up_scales_128x4(
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
pub(super) enum DynamicAbsmaxState {
    NeedsScan,
    PrecomputedInDynamicScalars,
}

#[cfg(feature = "cuda")]
#[expect(
    clippy::too_many_arguments,
    reason = "CUDA kernel launcher forwarding explicit quantization handles and device buffers"
)]
pub(super) fn launch_quantize_bf16_to_nvfp4_128x4_zero_sync(
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
pub(super) fn resolve_cublaslt_scalars(
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
pub(super) struct SharedMergedGateUpOut {
    ptr: DevicePtr,
    cap_bytes: usize,
}

#[cfg(feature = "cuda")]
pub(super) fn ensure_shared_merged_gate_up_out(
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
pub(super) fn ensure_proj_dynamic_scalars(gpu: &dyn GpuBackend) -> Result<DevicePtr> {
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
pub(super) fn clear_precomputed_proj_absmax() {
    if !HAS_PENDING_PROJ_ABSMAX.load(std::sync::atomic::Ordering::Acquire) {
        return;
    }
    if let Ok(mut guard) = pending_proj_absmax_lock().lock() {
        *guard = None;
        HAS_PENDING_PROJ_ABSMAX.store(false, std::sync::atomic::Ordering::Release);
    }
}

#[cfg(feature = "cuda")]
pub(super) fn take_precomputed_proj_absmax(
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
pub(super) fn ensure_decode_ffn_e2m1_scratch(
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
    let ptx_k = crate::layers::try_kernel(
        gpu,
        "quantize_bf16_to_nvfp4_cutlass",
        "quantize_bf16_to_nvfp4_atlas_128x4_ptx",
    );
    let legacy_k = crate::layers::try_kernel(
        gpu,
        "quantize_bf16_to_nvfp4_cutlass",
        "quantize_bf16_to_nvfp4_atlas_128x4",
    );
    let from_absmax_ptx = if prefill_fuse_absmax_enabled(gpu) {
        crate::layers::try_kernel(
            gpu,
            "quantize_nvfp4",
            "quantize_bf16_to_nvfp4_atlas_128x4_from_absmax_ptx",
        )
    } else {
        KernelHandle(0)
    };
    let from_absmax_legacy = crate::layers::try_kernel(
        gpu,
        "quantize_bf16_to_nvfp4_cutlass",
        "quantize_bf16_to_nvfp4_atlas_128x4_from_absmax",
    );
    ProjNvfp4Kernels {
        quant_static: select_cublaslt_quantize_128x4_kernel(ptx_k, legacy_k),
        absmax: crate::layers::try_kernel(gpu, "quantize_nvfp4", "nvfp4_global_absmax"),
        from_absmax: select_cublaslt_quantize_128x4_kernel(from_absmax_ptx, from_absmax_legacy),
        combined_alpha: crate::layers::try_kernel(
            gpu,
            "quantize_bf16_to_nvfp4_cutlass",
            "nvfp4_dynamic_combined_alpha",
        ),
    }
}

#[cfg(feature = "cuda")]
pub(super) fn nvfp4_weight_ready(w: &QuantizedWeight) -> bool {
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
