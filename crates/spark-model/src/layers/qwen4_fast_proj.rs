// SPDX-License-Identifier: AGPL-3.0-only

//! Default-off numerical prefill projections for Flash-Next: NVFP4 weights are
//! dequantized to a BF16 scratch (`w4a16_dequant_bf16`, same per-weight BF16
//! rounding as the W4A16 tensor-core GEMM) and multiplied with cuBLASLt
//! (BF16 x BF16, FP32 accumulation). NOT bit-exact versus the exact GEMV
//! family: accumulation order differs. Selected per family by
//! `ATLAS_QWEN4_PREFILL_FAST_PROJ=<list>` where the list is comma separated
//! from {ssm, hc, attn} or `all`; absent/0 keeps every exact path unchanged.

use std::sync::{Mutex, OnceLock};

use anyhow::{Context, Result, bail};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use crate::weight_map::QuantizedWeight;

#[path = "qwen4_fast_proj_cache_floor.rs"]
pub mod cache_floor;

#[path = "qwen4_fast_proj_nvfp4.rs"]
pub(crate) mod nvfp4;

/// Maximum batched-verify row count supported by `qwen4_fast_proj`'s coalesced
/// W4A16 (`grid.y = div_ceil(m, tile_m)`), transposed shared-expert, and direct
/// NVFP4 paths (`C=32/64` cohorts up to 256 rows).
pub(crate) const COALESCED_NVFP4_MAX_ROWS: usize = 256;
const _: () = assert!(COALESCED_NVFP4_MAX_ROWS >= crate::layers::QWEN4_NATIVE_MAX_ROWS);

pub(crate) const SELECTOR: &str = "ATLAS_QWEN4_PREFILL_FAST_PROJ";

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Selection {
    pub ssm: bool,
    pub hc: bool,
    pub attn: bool,
}

pub(crate) fn parse(value: Option<&str>) -> Result<Selection> {
    let mut sel = Selection::default();
    let Some(value) = value else { return Ok(sel) };
    let value = value.trim();
    if value.is_empty() || value == "0" {
        return Ok(sel);
    }
    for item in value.split(',') {
        match item.trim() {
            "all" => {
                sel = Selection {
                    ssm: true,
                    hc: true,
                    attn: true,
                }
            }
            "ssm" => sel.ssm = true,
            "hc" => sel.hc = true,
            "attn" => sel.attn = true,
            other => bail!("{SELECTOR}: unknown family {other:?} (expected ssm, hc, attn, all)"),
        }
    }
    Ok(sel)
}

/// Parsed once per process; a malformed selector fails the first prefill.
pub(crate) fn selection() -> Result<Selection> {
    static SEL: OnceLock<Result<Selection, String>> = OnceLock::new();
    SEL.get_or_init(|| {
        let raw = std::env::var(SELECTOR).ok();
        parse(raw.as_deref()).map_err(|e| e.to_string())
    })
    .clone()
    .map_err(|e| anyhow::anyhow!(e))
}

fn kernel(gpu: &dyn GpuBackend) -> Result<KernelHandle> {
    static K: OnceLock<Result<KernelHandle, String>> = OnceLock::new();
    K.get_or_init(|| {
        gpu.kernel("w4a16_dequant_bf16", "w4a16_dequant_bf16")
            .map_err(|e| e.to_string())
    })
    .clone()
    .map_err(|e| anyhow::anyhow!("{SELECTOR}: dequant kernel unavailable: {e}"))
}

/// Grow-only BF16 weight scratch shared by every fast projection. All
/// callers run on the single prefill stream, so reuse is stream ordered.
fn scratch(gpu: &dyn GpuBackend, bytes: usize) -> Result<DevicePtr> {
    static S: Mutex<Option<(DevicePtr, usize)>> = Mutex::new(None);
    let mut guard = S.lock().unwrap();
    if let Some((ptr, cap)) = *guard
        && cap >= bytes
    {
        return Ok(ptr);
    }
    // Round up so the SSM QKVZ (16384x2560) and attention QG (12288x2560)
    // shapes share one allocation; the old block is intentionally leaked
    // (at most one grow step in practice) to avoid freeing in-flight memory.
    let cap = bytes.max(16384 * 2560 * 2).next_multiple_of(1 << 20);
    let ptr = gpu
        .alloc(cap)
        .with_context(|| format!("{SELECTOR}: alloc {cap} B dequant scratch"))?;
    spark_runtime::scratch_init::init_grown(gpu, ptr, cap, "fast_proj_dequant")?;
    tracing::info!("QWEN4_FAST_PROJ scratch allocated bytes={cap}");
    *guard = Some((ptr, cap));
    Ok(ptr)
}

fn fold_scale2() -> bool {
    static F: OnceLock<bool> = OnceLock::new();
    *F.get_or_init(|| {
        std::env::var("ATLAS_QWEN4_FAST_PROJ_FOLD_SCALE2")
            .ok()
            .as_deref()
            == Some("1")
    })
}

fn tuned() -> bool {
    static T: OnceLock<bool> = OnceLock::new();
    *T.get_or_init(|| std::env::var("ATLAS_CUBLAS_TUNED").ok().as_deref() == Some("1"))
}

/// `out[m, n] = act[m, k] @ dequant(weight)[n, k]^T`, all BF16 row-major.
#[allow(clippy::too_many_arguments)]
pub(crate) fn dequant_gemm(
    gpu: &dyn GpuBackend,
    act: DevicePtr,
    weight: &QuantizedWeight,
    out: DevicePtr,
    m: usize,
    n: usize,
    k: usize,
    stream: u64,
) -> Result<()> {
    dequant_gemm_ex(gpu, act, weight, out, m, n, k, n, None, stream)
}

fn qk_norm_kernel(gpu: &dyn GpuBackend) -> Result<KernelHandle> {
    static K: OnceLock<Result<KernelHandle, String>> = OnceLock::new();
    K.get_or_init(|| {
        gpu.kernel("qwen4_qk_norm_strided", "qwen4_qk_norm_strided")
            .map_err(|e| e.to_string())
    })
    .clone()
    .map_err(|e| anyhow::anyhow!("{SELECTOR}: qk-norm kernel unavailable: {e}"))
}

/// In-place per-head RMS norm (`x * rsqrt(mean(x^2)+eps) * (1+w)`) over
/// `rows x heads` heads of `head_dim` inside rows of `row_stride` elements.
#[allow(clippy::too_many_arguments)]
pub(crate) fn qk_norm_strided(
    gpu: &dyn GpuBackend,
    x: DevicePtr,
    weight: DevicePtr,
    rows: usize,
    heads: usize,
    head_dim: usize,
    row_stride: usize,
    eps: f32,
    stream: u64,
) -> Result<()> {
    let kh = qk_norm_kernel(gpu)?;
    KernelLaunch::new(gpu, kh)
        .grid([(rows * heads) as u32, 1, 1])
        .block([head_dim.min(1024) as u32, 1, 1])
        .arg_ptr(x)
        .arg_ptr(weight)
        .arg_u32(heads as u32)
        .arg_u32(head_dim as u32)
        .arg_u32(row_stride as u32)
        .arg_f32(eps)
        .launch(stream)
}

pub(crate) const FP8_SELECTOR: &str = "ATLAS_QWEN4_PREFILL_FAST_PROJ_FP8";

/// Families whose fast projections use FP8 e4m3 operands
/// (`ATLAS_QWEN4_PREFILL_FAST_PROJ_FP8=<ssm,attn,hc|all>`, default none).
/// Malformed values fail the first prefill like the BF16 selector.
pub(crate) fn fp8_selection() -> Result<Selection> {
    static SEL: OnceLock<Result<Selection, String>> = OnceLock::new();
    SEL.get_or_init(|| {
        let raw = std::env::var(FP8_SELECTOR).ok();
        parse(raw.as_deref()).map_err(|e| e.to_string().replace(SELECTOR, FP8_SELECTOR))
    })
    .clone()
    .map_err(|e| anyhow::anyhow!(e))
}

/// `ATLAS_QWEN4_PREFILL_FAST_PROJ_FP8_SCALE=tensor`: one scale per weight and
/// per activation tensor (cuBLASLt SCALAR_32F) instead of per-row OUTER_VEC.
pub(crate) fn fp8_tensor_scales() -> bool {
    static T: OnceLock<bool> = OnceLock::new();
    *T.get_or_init(|| {
        std::env::var("ATLAS_QWEN4_PREFILL_FAST_PROJ_FP8_SCALE")
            .ok()
            .as_deref()
            == Some("tensor")
    })
}

#[derive(Clone, Copy)]
pub(crate) struct Fp8TensorKernels {
    pub(crate) weight_amax: KernelHandle,
    pub(crate) act_amax: KernelHandle,
    pub(crate) weight_rows: KernelHandle,
    pub(crate) act_rows: KernelHandle,
}

pub(crate) fn fp8_tensor_kernels(gpu: &dyn GpuBackend) -> Result<Fp8TensorKernels> {
    static K: OnceLock<Result<Fp8TensorKernels, String>> = OnceLock::new();
    K.get_or_init(|| {
        let k = |name: &str| {
            gpu.kernel("qwen4_fp8_proj", name)
                .map_err(|e| e.to_string())
        };
        Ok(Fp8TensorKernels {
            weight_amax: k("qwen4_fp8_weight_amax")?,
            act_amax: k("qwen4_fp8_act_amax")?,
            weight_rows: k("qwen4_fp8_weight_rows_tensor")?,
            act_rows: k("qwen4_fp8_act_rows_tensor")?,
        })
    })
    .clone()
    .map_err(|e| anyhow::anyhow!("{FP8_SELECTOR}: tensor-scale kernel unavailable: {e}"))
}

fn fp8_kernels(gpu: &dyn GpuBackend) -> Result<(KernelHandle, KernelHandle)> {
    static K: OnceLock<Result<(KernelHandle, KernelHandle), String>> = OnceLock::new();
    K.get_or_init(|| {
        let w = gpu
            .kernel("qwen4_fp8_proj", "qwen4_fp8_weight_rows")
            .map_err(|e| e.to_string())?;
        let a = gpu
            .kernel("qwen4_fp8_proj", "qwen4_fp8_act_rows")
            .map_err(|e| e.to_string())?;
        Ok((w, a))
    })
    .clone()
    .map_err(|e| anyhow::anyhow!("{FP8_SELECTOR}: kernel unavailable: {e}"))
}

#[inline]
const fn fp8_scratch_needs_eager_w8(tensor_scales: bool) -> bool {
    !tensor_scales
}

/// Minimum activation row capacity for [`fp8_scratch`]: always covers the
/// active prefill chunk width and, inside a batched-verify dense TC scope,
/// pre-allocates at least [`COALESCED_NVFP4_MAX_ROWS`] (`256`) so widening
/// cohorts (`k=2` -> `k=5`, `C=4` -> `C=64`) never reallocate during CUDA
/// graph capture.
#[inline]
#[must_use]
const fn fp8_scratch_m_cap(m: usize, max_chunk_rows: usize, batched_verify_scope: bool) -> usize {
    let verify_floor = if batched_verify_scope {
        COALESCED_NVFP4_MAX_ROWS
    } else {
        0
    };
    let mut floor = if m > verify_floor { m } else { verify_floor };
    if max_chunk_rows > floor {
        floor = max_chunk_rows;
    }
    if floor > FP8_PAD_GOLDEN_UNSPLIT_MIN_M {
        floor.next_multiple_of(FP8_PAD_GOLDEN_ALIGN)
    } else {
        floor.next_multiple_of(256)
    }
}

/// Grow-only FP8 scratch (weight, weight scales, activation, activation
/// scales), stream-ordered on the single prefill stream.
#[derive(Clone, Copy)]
struct Fp8Scratch {
    w: (DevicePtr, usize),
    ws: (DevicePtr, usize),
    a: (DevicePtr, usize),
    as_: (DevicePtr, usize),
}

static FP8_SCRATCH: Mutex<Option<Fp8Scratch>> = Mutex::new(None);

fn fp8_scratch_a_region() -> Option<(DevicePtr, usize)> {
    let cur = (*FP8_SCRATCH.lock().ok()?)?;
    (!cur.a.0.is_null() && cur.a.1 > 0).then_some(cur.a)
}

/// Whether the process-wide FP8 (or active NVFP4) projection scratch already
/// covers `(n, m, k)` without allocating GPU memory (safe to enter inside CUDA
/// graph capture).
#[must_use]
pub(crate) fn fp8_scratch_ready_for_capture(n: usize, m: usize, k: usize) -> bool {
    let scope_active = crate::layers::qwen4_hyper::batched_verify_tc_dense_scope_active();
    let nvfp4_mode = nvfp4::mode_or_off();
    let fp8_ok = || -> bool {
        let Ok(guard) = FP8_SCRATCH.lock() else {
            return false;
        };
        let Some(cur) = *guard else {
            return false;
        };
        let m_cap = fp8_scratch_m_cap(
            m,
            crate::layers::qwen4_prefill_moe::max_chunk_rows(),
            scope_active,
        );
        let k_cap = k.max(6144);
        let n_cap = n.max(16384);
        let tensor_scales = fp8_tensor_scales();
        let w_ok =
            !fp8_scratch_needs_eager_w8(tensor_scales) || cur.w.1 >= (n * k).max(16384 * 2560);
        let ws_ok = cur.ws.1 >= n_cap * 4;
        let a_ok = cur.a.1 >= m_cap * k_cap;
        let as_ok = tensor_scales || cur.as_.1 >= m_cap * 4;
        w_ok && ws_ok && a_ok && as_ok
    };
    if nvfp4_mode.admits_call(m, n, k, n, None, scope_active, nvfp4::verify_min_m()) {
        let nv_ok = nvfp4::scratch_ready_for_capture(m, k);
        // Wide QKV shared-act (`q_proj_dim = 12288 / 6144`) pairs NVFP4 Q with FP8 K/V
        // whenever K/V falls back to FP8, so require FP8 scratch readiness as well.
        if matches!(n, 12_288 | 6_144) {
            nv_ok && fp8_ok()
        } else {
            nv_ok
        }
    } else {
        fp8_ok()
    }
}

pub(crate) fn fp8_scratch(
    gpu: &dyn GpuBackend,
    n: usize,
    m: usize,
    k: usize,
) -> Result<(DevicePtr, DevicePtr, DevicePtr, DevicePtr)> {
    let mut guard = FP8_SCRATCH
        .lock()
        .map_err(|_| anyhow::anyhow!("{FP8_SELECTOR}: scratch lock poisoned"))?;
    let mut cur = guard.unwrap_or(Fp8Scratch {
        w: (DevicePtr::NULL, 0),
        ws: (DevicePtr::NULL, 0),
        a: (DevicePtr::NULL, 0),
        as_: (DevicePtr::NULL, 0),
    });
    let grow = |slot: &mut (DevicePtr, usize), bytes: usize| -> Result<()> {
        if slot.1 < bytes {
            anyhow::ensure!(
                spark_runtime::gpu_capture_region::active_capture_region().is_none(),
                "{FP8_SELECTOR}: refusing lazy scratch allocation ({bytes} B) during CUDA graph capture"
            );
            // Old blocks are leaked on purpose (in-flight safety, grow-only).
            let cap = bytes.next_multiple_of(1 << 20);
            slot.0 = gpu
                .alloc(cap)
                .with_context(|| format!("{FP8_SELECTOR}: alloc {cap} B"))?;
            slot.1 = cap;
            spark_runtime::scratch_init::init_grown(gpu, slot.0, cap, "fp8_proj")?;
        }
        Ok(())
    };
    let m_cap = fp8_scratch_m_cap(
        m,
        crate::layers::qwen4_prefill_moe::max_chunk_rows(),
        crate::layers::qwen4_hyper::batched_verify_tc_dense_scope_active(),
    );
    let k_cap = k.max(6144);
    let n_cap = n.max(16384);
    let tensor_scales = fp8_tensor_scales();
    // Under tensor scales, every uncached per-call fallback in `fp8_weight_tensor_gemm`
    // lazily allocates via `fp8_w8_scratch`, saving 41.9 MB (40 MiB) UMA when FP8 weights
    // are cached, checkpoint-native (`ATLAS_QWEN4_PREFILL_NATIVE`), or off.
    if fp8_scratch_needs_eager_w8(tensor_scales) {
        grow(&mut cur.w, (n * k).max(16384 * 2560))?;
    }
    grow(&mut cur.ws, n_cap * 4)?;
    grow(&mut cur.a, m_cap * k_cap)?;
    if !tensor_scales {
        grow(&mut cur.as_, m_cap * 4)?;
    }
    *guard = Some(cur);
    Ok((cur.w.0, cur.ws.0, cur.a.0, cur.as_.0))
}

fn fp8_w8_scratch(gpu: &dyn GpuBackend, n: usize, k: usize) -> Result<DevicePtr> {
    static W_SCR: Mutex<(DevicePtr, usize)> = Mutex::new((DevicePtr::NULL, 0));
    let mut cur = W_SCR
        .lock()
        .map_err(|_| anyhow::anyhow!("{FP8_SELECTOR}: w8 scratch lock poisoned"))?;
    let bytes = (n * k).max(16384 * 2560);
    if cur.1 < bytes {
        anyhow::ensure!(
            spark_runtime::gpu_capture_region::active_capture_region().is_none(),
            "{FP8_SELECTOR}: refusing lazy w8 scratch allocation ({bytes} B) during CUDA graph capture"
        );
        let cap = bytes.next_multiple_of(1 << 20);
        cur.0 = gpu
            .alloc(cap)
            .with_context(|| format!("{FP8_SELECTOR}: alloc {cap} B"))?;
        cur.1 = cap;
        spark_runtime::scratch_init::init_grown(gpu, cur.0, cap, "fp8_proj_w8")?;
    }
    Ok(cur.0)
}

thread_local! {
    static PREQUANT_A8_READY: std::cell::Cell<Option<(u64, usize, usize, u64)>> =
        const { std::cell::Cell::new(None) };
    static BYPASSED_FP8_SHARED_ACT: std::cell::Cell<Option<(u64, usize, usize, u64)>> =
        const { std::cell::Cell::new(None) };
    static FP8_OUT_CAP_ROWS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Scoped declaration of the caller's output-arena row capacity so tensor-scale FP8
/// GEMMs (`m >= 512`, `n >= 512`) can pad `M` up to a 256-row (or golden 1024-row for
/// wide `n >= 12288`) multiple (`a8[m..m_pad] = 0`, falling back to 256 then 128 near the
/// arena cap) and stay on cuBLASLt's fastest SM121a TMA kernels without touching rows
/// outside the arena.
pub(crate) struct PadRowsGuard(usize);

impl PadRowsGuard {
    #[must_use]
    pub(crate) fn enter(out_cap_rows: usize) -> Self {
        let prev = FP8_OUT_CAP_ROWS.with(|c| c.replace(out_cap_rows));
        Self(prev)
    }
}

impl Drop for PadRowsGuard {
    fn drop(&mut self) {
        FP8_OUT_CAP_ROWS.with(|c| c.set(self.0));
    }
}

const FP8_PAD_MIN_M: usize = 512;
const FP8_PAD_MIN_N: usize = 512;
/// Wide projections (`N >= 12288`: attn Q = 12288, SSM QKVZ = 16384) exhibit SM121a
/// cuBLASLt throughput cliffs at odd-256 / odd-512 row counts (`4352`, `5632`, `7680`,
/// `8448`). Pad wide chunks above `512` (split remainder) or `3072` (unsplit) to a
/// 1024-row golden multiple whenever it fits within the caller's arena cap.
const FP8_PAD_GOLDEN_WIDE_N: usize = 12_288;
const FP8_PAD_GOLDEN_UNSPLIT_MIN_M: usize = 3_072;
const FP8_PAD_GOLDEN_REM_MIN_M: usize = 512;
const FP8_PAD_GOLDEN_ALIGN: usize = 1_024;

fn fp8_pad128_selected() -> bool {
    static S: OnceLock<bool> = OnceLock::new();
    *S.get_or_init(|| {
        std::env::var("ATLAS_QWEN4_FP8_PAD128").ok().as_deref() != Some("0")
            && std::env::var_os("ATLAS_QWEN4_FP8_PROJ_DUMP_DIR").is_none()
    })
}

#[inline]
#[must_use]
const fn resolve_fp8_pad_m_with_cap(m: usize, n_min: usize, cap: usize, split_cap: usize) -> usize {
    if m < FP8_PAD_MIN_M || n_min < FP8_PAD_MIN_N || cap < m {
        return m;
    }
    let m_golden = if n_min >= FP8_PAD_GOLDEN_WIDE_N {
        let chunk = if split_cap == 0 || m <= FP8_GEMM_M_SPLIT_MIN_M {
            0
        } else {
            let aligned = split_cap & !127;
            if aligned < 128 { 128 } else { aligned }
        };
        if chunk >= FP8_PAD_GOLDEN_ALIGN && chunk.is_multiple_of(FP8_PAD_GOLDEN_ALIGN) && m > chunk
        {
            let rem = m % chunk;
            if rem == 0 {
                m
            } else if rem <= FP8_PAD_GOLDEN_REM_MIN_M {
                (m - rem) + rem.next_multiple_of(256)
            } else {
                let rem_pad = rem.next_multiple_of(FP8_PAD_GOLDEN_ALIGN);
                (m - rem) + if rem_pad > chunk { chunk } else { rem_pad }
            }
        } else if m > FP8_PAD_GOLDEN_UNSPLIT_MIN_M {
            m.next_multiple_of(FP8_PAD_GOLDEN_ALIGN)
        } else {
            m.next_multiple_of(256)
        }
    } else {
        m.next_multiple_of(256)
    };
    if m_golden <= cap {
        return m_golden;
    }
    // SM121a cuBLASLt fp8_device_scaled's fastest 256-row CTA tile requires
    // M % 256 == 0: odd multiples of 128 (e.g. 8,320 or 16,256) fall back to
    // slower tiles (+54-69%, e.g. 12.55 ms vs 7.67 ms at 16K). Prefer 256-row
    // alignment when it fits in the arena, falling back to 128 only near cap.
    let m_pad256 = m.next_multiple_of(256);
    if m_pad256 <= cap {
        return m_pad256;
    }
    let m_pad128 = m.next_multiple_of(128);
    if m_pad128 > m && m_pad128 <= cap {
        m_pad128
    } else {
        m
    }
}

#[inline]
pub(crate) fn resolve_fp8_pad_m(m: usize, n_min: usize) -> usize {
    if m < FP8_PAD_MIN_M || n_min < FP8_PAD_MIN_N || !fp8_pad128_selected() {
        return m;
    }
    let cap = FP8_OUT_CAP_ROWS.with(std::cell::Cell::get);
    resolve_fp8_pad_m_with_cap(m, n_min, cap, fp8_gemm_m_split_cap())
}

/// Prepare `(a_amax, a8, a_scale)` when `act` (`[m, k]` BF16) already has a matching
/// producer absmax in `qwen4_fp8_act_amax` so the caller can fuse tensor-scale E4M3
/// quantization into another pass over `act` (e.g. `qwen4_mxfp8_quant_rows_v16q4_gate`).
pub(crate) fn prepare_fused_fp8_act(
    gpu: &dyn GpuBackend,
    act: DevicePtr,
    m: usize,
    k: usize,
    stream: u64,
) -> Result<Option<(DevicePtr, DevicePtr, DevicePtr)>> {
    PREQUANT_A8_READY.with(|c| c.set(None));
    if m < 1024
        || !fp8_tensor_scales()
        || !crate::layers::qwen4_fp8_act_amax::take_matching(act, m, k, stream)
    {
        return Ok(None);
    }
    let a_amax = crate::layers::qwen4_fp8_act_amax::slot(gpu)?;
    let (_, ws, a8, _) = fp8_scratch(gpu, 512, m, k)?;
    let a_scale = ws.offset(12);
    Ok(Some((a_amax, a8, a_scale)))
}

pub(crate) fn note_prequant_fp8_act(act: DevicePtr, m: usize, k: usize, stream: u64) {
    PREQUANT_A8_READY.with(|c| c.set(Some((act.0, m, k, stream))));
}

pub(crate) fn clear_prequant_fp8_act() {
    PREQUANT_A8_READY.with(|c| c.set(None));
    BYPASSED_FP8_SHARED_ACT.with(|c| c.set(None));
}

fn has_prequant_fp8_act(act: DevicePtr, m: usize, k: usize, stream: u64) -> bool {
    PREQUANT_A8_READY.with(std::cell::Cell::get) == Some((act.0, m, k, stream))
}

fn take_prequant_fp8_act(act: DevicePtr, m: usize, k: usize, stream: u64) -> bool {
    PREQUANT_A8_READY.with(|c| c.take()) == Some((act.0, m, k, stream))
}

fn note_bypassed_fp8_shared_act(act: DevicePtr, m: usize, k: usize, stream: u64) {
    BYPASSED_FP8_SHARED_ACT.with(|c| c.set(Some((act.0, m, k, stream))));
}

fn clear_bypassed_fp8_shared_act() {
    BYPASSED_FP8_SHARED_ACT.with(|c| c.set(None));
}

/// Resolve a BF16 activation slice when the immediately preceding
/// `fp8_gemm_shared_act` ran fully on native NVFP4 without populating
/// `FP8_SCRATCH.a` (supports Layer-47 `Crop::Tail` Step 2 Q-slice projection).
fn take_bypassed_fp8_shared_act_slice(
    a8: DevicePtr,
    m: usize,
    k: usize,
    stream: u64,
) -> Option<DevicePtr> {
    let (orig_act, orig_m, orig_k, orig_stream) = BYPASSED_FP8_SHARED_ACT.with(|c| c.take())?;
    if orig_k != k || orig_stream != stream || m == 0 || k == 0 {
        return None;
    }
    let (a_base, a_cap) = fp8_scratch_a_region()?;
    let byte_off = a8.0.checked_sub(a_base.0)? as usize;
    if byte_off >= a_cap || !byte_off.is_multiple_of(k) {
        return None;
    }
    let skip_rows = byte_off / k;
    if skip_rows.checked_add(m)? > orig_m {
        return None;
    }
    Some(DevicePtr(orig_act).offset(skip_rows * k * 2))
}

pub(crate) const COALESCED_NVFP4_DENSE_ENV: &str = "ATLAS_QWEN4_BATCHED_VERIFY_COALESCED_NVFP4";

#[inline]
#[must_use]
pub(crate) fn parse_coalesced_nvfp4_dense(raw: Option<&str>) -> bool {
    raw.map(str::trim) != Some("0")
}

/// `ATLAS_QWEN4_BATCHED_VERIFY_COALESCED_NVFP4` (default ON; `"0"` disables):
/// execute batched-verify dense projections (`rows <= COALESCED_NVFP4_MAX_ROWS`
/// (256), `k % 32 == 0`, contiguous output `ldc == n`, unpermuted `qg.is_none()`)
/// directly from NVFP4 weights via single-sweep K-coalesced
/// `w4a16_gemm_coalesced*` kernels. Reads half the weight bytes of FP8
/// (`0.5625 B/wt`) in a single launch with zero FP8 weight-cache RAM and full
/// CUDA-graph capture safety while keeping small row counts below the
/// Tensor Core threshold on strict 0-ULP scalar GEMV.
#[must_use]
pub(crate) fn coalesced_nvfp4_dense_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        parse_coalesced_nvfp4_dense(std::env::var(COALESCED_NVFP4_DENSE_ENV).ok().as_deref())
    })
}

pub(crate) const COALESCED_V2_ENV: &str = "ATLAS_QWEN4_COALESCED_V2";

#[inline]
#[must_use]
pub(crate) fn parse_coalesced_v2(raw: Option<&str>) -> bool {
    raw.map(str::trim) != Some("0")
}

/// `ATLAS_QWEN4_COALESCED_V2` (default ON; `"0"` disables): select 2-stage
/// K-prefetch (`K_STEP = 256` per warp + `uint32` 4-byte `B_scale` loads),
/// double-buffered `TILE_M = 80`, `TILE_M = 16`, and 2-CTA/SM shape-matched
/// tiles (`n64`, `n32_sk2`) in `w4a16_gemm_coalesced_v2_*` when `k % 64 == 0`
/// and `weight_scale` is 4-byte aligned, degrading cleanly to v1 otherwise.
#[must_use]
pub(crate) fn coalesced_v2_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| parse_coalesced_v2(std::env::var(COALESCED_V2_ENV).ok().as_deref()))
}

/// Column-tile and intra-CTA Warp-Split-K tiers for K-coalesced NVFP4 W4A16 GEMM.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CoalescedNvfp4NTile {
    /// `CO_N = 8, SPLIT_K = 8` (`8` warps, `256` threads/CTA, barrier-free
    /// K-loop): saturates narrow long-K projections (`HC down N=320 K=10240`,
    /// `v_router N=512 K=2560`) at small row counts.
    N8Sk8,
    /// `CO_N = 16, SPLIT_K = 4` (`8` warps, `256` threads/CTA): saturates
    /// narrow projections (`N <= 1024`) at `m = 17..=31`.
    N16Sk4,
    /// `CO_N = 32, SPLIT_K = 2` (`8` warps, `256` threads/CTA, 2 CTAs/SM):
    /// v2 single-weight-sweep (`gridY = 1`) medium tier saturating all 48 SMs
    /// (`grid = 80`, 0 idle SMs, 2-stage K-prefetch) on `ssm.out` / `attn.oproj`
    /// (`N = 2560, K = 6144`) at `m = 33..=64` (`M64`) and `m = 65..=80` (`M80`).
    N32Sk2,
    /// `CO_N = 32, SPLIT_K = 4` (`16` warps, `512` threads/CTA): saturates
    /// narrow projections (`m16`, `m >= 32`) and medium projections
    /// (`ssm.out`/`attn.oproj N=2560 K=6144`, `m16`/`m32`, `m <= 32`) via v1
    /// split-K.
    N32Sk4,
    /// `CO_N = 64, SPLIT_K = 2` (`16` warps, `512` threads/CTA): 1-wave
    /// (`grid = 40`, half A-matrix DRAM traffic) v1 split-K winner for medium
    /// projections (`N = 2560, K = 6144`) at `m = 33..=64` (`m64`).
    N64Sk2,
    /// `CO_N = 64, SPLIT_K = 4` (`32` warps, `1024` threads/CTA): saturates
    /// narrow long-K projections (`HC down N=320 K=10240`) at `m = 65..=128`
    /// (`m16`) while bounding L2 activation rereads.
    N64Sk4,
    /// `CO_N = 16` (`2` warps, `64` threads/CTA): non-split-K narrow tier
    /// (`K < 1024` and legacy fallback).
    N16,
    /// `CO_N = 32` (`4` warps, `128` threads/CTA): non-split-K medium tier
    /// (also saturates wide `N <= 24576` at `m <= 32`).
    N32,
    /// `CO_N = 64` (`8` warps, `256` threads/CTA, 2 CTAs/SM): balanced wide
    /// tier (`ssm.qkvz 16384`, `attn.q 12288`, `hc.up 10240`, `lm_head`).
    N64,
    /// `CO_N = 128` (`16` warps, `512` threads/CTA): v1 wide fallback.
    N128,
}

/// Row-tile tiers (`TILE_M in {16, 32, 64, 80, 128}`) for K-coalesced NVFP4 W4A16 GEMM.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CoalescedNvfp4MTile {
    M16,
    M32,
    M64,
    M80,
    M128,
}

/// Concrete `(CO_N, TILE_M)` launch configuration for `w4a16_gemm_coalesced*`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct CoalescedNvfp4Tile {
    pub(crate) n_tile: CoalescedNvfp4NTile,
    pub(crate) m_tile: CoalescedNvfp4MTile,
}

impl CoalescedNvfp4Tile {
    #[inline]
    #[must_use]
    pub(crate) const fn co_n(self) -> u32 {
        match self.n_tile {
            CoalescedNvfp4NTile::N8Sk8 => 8,
            CoalescedNvfp4NTile::N16 | CoalescedNvfp4NTile::N16Sk4 => 16,
            CoalescedNvfp4NTile::N32
            | CoalescedNvfp4NTile::N32Sk2
            | CoalescedNvfp4NTile::N32Sk4 => 32,
            CoalescedNvfp4NTile::N64
            | CoalescedNvfp4NTile::N64Sk2
            | CoalescedNvfp4NTile::N64Sk4 => 64,
            CoalescedNvfp4NTile::N128 => 128,
        }
    }

    #[inline]
    #[must_use]
    pub(crate) const fn tile_m(self) -> u32 {
        match self.m_tile {
            CoalescedNvfp4MTile::M16 => 16,
            CoalescedNvfp4MTile::M32 => 32,
            CoalescedNvfp4MTile::M64 => 64,
            CoalescedNvfp4MTile::M80 => 80,
            CoalescedNvfp4MTile::M128 => 128,
        }
    }

    #[inline]
    #[must_use]
    pub(crate) const fn threads(self) -> u32 {
        match self.n_tile {
            CoalescedNvfp4NTile::N16 => 64,
            CoalescedNvfp4NTile::N32 => 128,
            CoalescedNvfp4NTile::N64
            | CoalescedNvfp4NTile::N8Sk8
            | CoalescedNvfp4NTile::N16Sk4
            | CoalescedNvfp4NTile::N32Sk2 => 256,
            CoalescedNvfp4NTile::N128
            | CoalescedNvfp4NTile::N32Sk4
            | CoalescedNvfp4NTile::N64Sk2 => 512,
            CoalescedNvfp4NTile::N64Sk4 => 1024,
        }
    }

    #[inline]
    #[must_use]
    pub(crate) const fn v2_symbol(self) -> Option<&'static str> {
        match (self.n_tile, self.m_tile) {
            (CoalescedNvfp4NTile::N32, CoalescedNvfp4MTile::M16) => {
                Some("w4a16_gemm_coalesced_v2_n32_m16")
            }
            (CoalescedNvfp4NTile::N64, CoalescedNvfp4MTile::M16) => {
                Some("w4a16_gemm_coalesced_v2_n64_m16")
            }
            (CoalescedNvfp4NTile::N32, CoalescedNvfp4MTile::M32) => {
                Some("w4a16_gemm_coalesced_v2_n32_m32")
            }
            (CoalescedNvfp4NTile::N64, CoalescedNvfp4MTile::M32) => {
                Some("w4a16_gemm_coalesced_v2_n64_m32")
            }
            (CoalescedNvfp4NTile::N64, CoalescedNvfp4MTile::M64) => {
                Some("w4a16_gemm_coalesced_v2_n64_m64")
            }
            (CoalescedNvfp4NTile::N64, CoalescedNvfp4MTile::M80) => {
                Some("w4a16_gemm_coalesced_v2_n64_m80")
            }
            (CoalescedNvfp4NTile::N64, CoalescedNvfp4MTile::M128) => {
                Some("w4a16_gemm_coalesced_v2_n64_m128")
            }
            (CoalescedNvfp4NTile::N32Sk2, CoalescedNvfp4MTile::M64) => {
                Some("w4a16_gemm_coalesced_v2_n32_sk2_m64")
            }
            (CoalescedNvfp4NTile::N32Sk2, CoalescedNvfp4MTile::M80) => {
                Some("w4a16_gemm_coalesced_v2_n32_sk2_m80")
            }
            _ => None,
        }
    }

    #[inline]
    #[must_use]
    pub(crate) const fn dispatch_symbol(self, prefer_v2: bool) -> &'static str {
        match (prefer_v2, self.v2_symbol()) {
            (true, Some(sym)) => sym,
            _ => self.symbol(),
        }
    }

    #[inline]
    #[must_use]
    pub(crate) const fn symbol(self) -> &'static str {
        match (self.n_tile, self.m_tile) {
            (CoalescedNvfp4NTile::N8Sk8, _) => "w4a16_gemm_coalesced_n8_sk8_m16",
            (CoalescedNvfp4NTile::N16Sk4, _) => "w4a16_gemm_coalesced_n16_sk4_m16",
            (CoalescedNvfp4NTile::N32Sk2, CoalescedNvfp4MTile::M80) => {
                "w4a16_gemm_coalesced_v2_n32_sk2_m80"
            }
            (CoalescedNvfp4NTile::N32Sk2, _) => "w4a16_gemm_coalesced_v2_n32_sk2_m64",
            (CoalescedNvfp4NTile::N32Sk4, CoalescedNvfp4MTile::M16) => {
                "w4a16_gemm_coalesced_n32_sk4_m16"
            }
            (CoalescedNvfp4NTile::N32Sk4, _) => "w4a16_gemm_coalesced_n32_sk4_m32",
            (CoalescedNvfp4NTile::N64Sk2, _) => "w4a16_gemm_coalesced_n64_sk2_m64",
            (CoalescedNvfp4NTile::N64Sk4, _) => "w4a16_gemm_coalesced_n64_sk4_m16",
            (CoalescedNvfp4NTile::N16, CoalescedNvfp4MTile::M16 | CoalescedNvfp4MTile::M32) => {
                "w4a16_gemm_coalesced_n16_m32"
            }
            (CoalescedNvfp4NTile::N16, CoalescedNvfp4MTile::M64) => "w4a16_gemm_coalesced_n16_m64",
            (CoalescedNvfp4NTile::N16, CoalescedNvfp4MTile::M80 | CoalescedNvfp4MTile::M128) => {
                "w4a16_gemm_coalesced_n16_m128"
            }
            (CoalescedNvfp4NTile::N32, CoalescedNvfp4MTile::M16 | CoalescedNvfp4MTile::M32) => {
                "w4a16_gemm_coalesced_n32_m32"
            }
            (CoalescedNvfp4NTile::N32, CoalescedNvfp4MTile::M64) => "w4a16_gemm_coalesced_n32_m64",
            (CoalescedNvfp4NTile::N32, CoalescedNvfp4MTile::M80 | CoalescedNvfp4MTile::M128) => {
                "w4a16_gemm_coalesced_n32_m128"
            }
            (CoalescedNvfp4NTile::N64, CoalescedNvfp4MTile::M16 | CoalescedNvfp4MTile::M32) => {
                "w4a16_gemm_coalesced_n64_m32"
            }
            (CoalescedNvfp4NTile::N64, CoalescedNvfp4MTile::M64) => "w4a16_gemm_coalesced_n64_m64",
            (CoalescedNvfp4NTile::N64, CoalescedNvfp4MTile::M80) => {
                "w4a16_gemm_coalesced_v2_n64_m80"
            }
            (CoalescedNvfp4NTile::N64, CoalescedNvfp4MTile::M128) => {
                "w4a16_gemm_coalesced_n64_m128"
            }
            (CoalescedNvfp4NTile::N128, CoalescedNvfp4MTile::M16 | CoalescedNvfp4MTile::M32) => {
                "w4a16_gemm_coalesced_m32"
            }
            (CoalescedNvfp4NTile::N128, CoalescedNvfp4MTile::M64) => "w4a16_gemm_coalesced_m64",
            (CoalescedNvfp4NTile::N128, CoalescedNvfp4MTile::M80 | CoalescedNvfp4MTile::M128) => {
                "w4a16_gemm_coalesced_m128"
            }
        }
    }
}

/// Select the baseline non-split-K `(CO_N, TILE_M)` tile for `[m, n]`
/// (`m in 1..=256`, `n >= 1`). Callers with `k` available should prefer
/// [`select_coalesced_nvfp4_tile_for_shape`].
#[inline]
#[must_use]
pub(crate) const fn select_coalesced_nvfp4_tile(m: usize, n: usize) -> Option<CoalescedNvfp4Tile> {
    if m == 0 || m > COALESCED_NVFP4_MAX_ROWS || n == 0 {
        return None;
    }
    let n_tile = if n <= 1024 {
        CoalescedNvfp4NTile::N16
    } else if n <= 4096 {
        CoalescedNvfp4NTile::N32
    } else {
        CoalescedNvfp4NTile::N128
    };
    let m_tile = if m <= 32 {
        CoalescedNvfp4MTile::M32
    } else if m <= 64 {
        CoalescedNvfp4MTile::M64
    } else {
        CoalescedNvfp4MTile::M128
    };
    Some(CoalescedNvfp4Tile { n_tile, m_tile })
}

/// Select the empirical GB10 optimal `(CO_N, TILE_M)` (including intra-CTA
/// Warp-Split-K tiers) for `[m, n, k]` (`m in 1..=256`, `n >= 1`, `k >= 1`).
#[inline]
#[must_use]
pub(crate) const fn select_coalesced_nvfp4_tile_for_shape(
    m: usize,
    n: usize,
    k: usize,
) -> Option<CoalescedNvfp4Tile> {
    if m == 0 || m > COALESCED_NVFP4_MAX_ROWS || n == 0 || k == 0 {
        return None;
    }
    if n <= 1024 {
        if k >= 1024 {
            let n_tile = if m <= 16 || (m <= 20 && k >= 4096) {
                CoalescedNvfp4NTile::N8Sk8
            } else if m < 32 {
                CoalescedNvfp4NTile::N16Sk4
            } else if m <= 64 || k < 4096 {
                CoalescedNvfp4NTile::N32Sk4
            } else {
                CoalescedNvfp4NTile::N64Sk4
            };
            return Some(CoalescedNvfp4Tile {
                n_tile,
                m_tile: CoalescedNvfp4MTile::M16,
            });
        }
        let m_tile = if m <= 32 {
            CoalescedNvfp4MTile::M32
        } else if m <= 64 {
            CoalescedNvfp4MTile::M64
        } else {
            CoalescedNvfp4MTile::M128
        };
        return Some(CoalescedNvfp4Tile {
            n_tile: CoalescedNvfp4NTile::N16,
            m_tile,
        });
    }
    if n <= 4096 {
        if k >= 2048 {
            return Some(if m <= 32 {
                CoalescedNvfp4Tile {
                    n_tile: CoalescedNvfp4NTile::N32Sk4,
                    m_tile: CoalescedNvfp4MTile::M32,
                }
            } else if m <= 64 {
                CoalescedNvfp4Tile {
                    n_tile: CoalescedNvfp4NTile::N64Sk2,
                    m_tile: CoalescedNvfp4MTile::M64,
                }
            } else {
                CoalescedNvfp4Tile {
                    n_tile: CoalescedNvfp4NTile::N64,
                    m_tile: CoalescedNvfp4MTile::M128,
                }
            });
        }
        let m_tile = if m <= 32 {
            CoalescedNvfp4MTile::M32
        } else if m <= 64 {
            CoalescedNvfp4MTile::M64
        } else {
            CoalescedNvfp4MTile::M128
        };
        return Some(CoalescedNvfp4Tile {
            n_tile: CoalescedNvfp4NTile::N32,
            m_tile,
        });
    }
    if m <= 32 {
        let n_tile = if n <= 24_576 {
            CoalescedNvfp4NTile::N32
        } else {
            CoalescedNvfp4NTile::N128
        };
        Some(CoalescedNvfp4Tile {
            n_tile,
            m_tile: CoalescedNvfp4MTile::M32,
        })
    } else if m <= 64 {
        let n_tile = if n <= 12_288 {
            CoalescedNvfp4NTile::N64
        } else {
            CoalescedNvfp4NTile::N128
        };
        Some(CoalescedNvfp4Tile {
            n_tile,
            m_tile: CoalescedNvfp4MTile::M64,
        })
    } else {
        Some(CoalescedNvfp4Tile {
            n_tile: CoalescedNvfp4NTile::N128,
            m_tile: CoalescedNvfp4MTile::M128,
        })
    }
}

/// Select the v2 (`ATLAS_QWEN4_COALESCED_V2 != 0`) shape-matched 2-CTA/SM and
/// 2-stage K-prefetch `(CO_N, TILE_M)` tile for `[m, n, k]` (`m in 1..=256`,
/// `n >= 1`, `k >= 1`).
///
/// Key improvements over [`select_coalesced_nvfp4_tile_for_shape`]:
/// - `m = 1..=16`: selects `TILE_M = 16` (`v2_n32_m16`, `v2_n64_m16`, and v1
///   `n32_sk4_m16` for `N <= 4096, K >= 2048`) to cut 50% of `M32` padding work
///   at `C=2 K=5` and ragged `C=4`.
/// - `N = 2560, K = 6144` (`ssm.out`, `attn.oproj`): routes `m = 1..=32` to
///   v1 split-K (`n32_sk4_m16`, `n32_sk4_m32`), and routes `m = 33..=64` to v2
///   `n32_sk2_m64` and `m = 65..=80` to v2 `n32_sk2_m80` (`grid = 80` CTAs,
///   2 CTAs/SM, full 48-SM saturation, 2-stage K-prefetch).
/// - `N <= 1024` (`router`, DFlash2 Stage-1): routes all split-K tiers
///   (`N8Sk8`, `N16Sk4`, `N32Sk4`, `N64Sk4`) to v1.
/// - `N > 4096` (`ssm.qkvz`, `attn.q`, `hc.up`, `lm_head` `N >= 65536`): uses
///   v2 across all non-split-K tiles (`v2_n32_m16/m32`, `v2_n64_m16/m32/m64/m80/m128`),
///   eliminating `DOUBLE_BUF_A_128=false` at `C=16 K=5`.
#[inline]
#[must_use]
pub(crate) const fn select_coalesced_nvfp4_tile_for_shape_v2(
    m: usize,
    n: usize,
    k: usize,
) -> Option<CoalescedNvfp4Tile> {
    if m == 0 || m > COALESCED_NVFP4_MAX_ROWS || n == 0 || k == 0 {
        return None;
    }
    if n <= 1024 {
        return select_coalesced_nvfp4_tile_for_shape(m, n, k);
    }
    if n <= 4096 {
        if k >= 2048 {
            return Some(if m <= 16 {
                CoalescedNvfp4Tile {
                    n_tile: CoalescedNvfp4NTile::N32Sk4,
                    m_tile: CoalescedNvfp4MTile::M16,
                }
            } else if m <= 32 {
                CoalescedNvfp4Tile {
                    n_tile: CoalescedNvfp4NTile::N32Sk4,
                    m_tile: CoalescedNvfp4MTile::M32,
                }
            } else if m <= 64 {
                CoalescedNvfp4Tile {
                    n_tile: CoalescedNvfp4NTile::N32Sk2,
                    m_tile: CoalescedNvfp4MTile::M64,
                }
            } else if m <= 80 {
                CoalescedNvfp4Tile {
                    n_tile: CoalescedNvfp4NTile::N32Sk2,
                    m_tile: CoalescedNvfp4MTile::M80,
                }
            } else {
                CoalescedNvfp4Tile {
                    n_tile: CoalescedNvfp4NTile::N64,
                    m_tile: CoalescedNvfp4MTile::M128,
                }
            });
        }
        let m_tile = if m <= 16 {
            CoalescedNvfp4MTile::M16
        } else if m <= 32 {
            CoalescedNvfp4MTile::M32
        } else if m <= 64 {
            CoalescedNvfp4MTile::M64
        } else {
            CoalescedNvfp4MTile::M128
        };
        return Some(CoalescedNvfp4Tile {
            n_tile: CoalescedNvfp4NTile::N32,
            m_tile,
        });
    }
    if m <= 16 {
        let n_tile = if n <= 24_576 {
            CoalescedNvfp4NTile::N32
        } else {
            CoalescedNvfp4NTile::N64
        };
        Some(CoalescedNvfp4Tile {
            n_tile,
            m_tile: CoalescedNvfp4MTile::M16,
        })
    } else if m <= 32 {
        let n_tile = if n <= 24_576 {
            CoalescedNvfp4NTile::N32
        } else {
            CoalescedNvfp4NTile::N64
        };
        Some(CoalescedNvfp4Tile {
            n_tile,
            m_tile: CoalescedNvfp4MTile::M32,
        })
    } else if m <= 64 {
        Some(CoalescedNvfp4Tile {
            n_tile: CoalescedNvfp4NTile::N64,
            m_tile: CoalescedNvfp4MTile::M64,
        })
    } else if m <= 80 {
        Some(CoalescedNvfp4Tile {
            n_tile: CoalescedNvfp4NTile::N64,
            m_tile: CoalescedNvfp4MTile::M80,
        })
    } else {
        Some(CoalescedNvfp4Tile {
            n_tile: CoalescedNvfp4NTile::N64,
            m_tile: CoalescedNvfp4MTile::M128,
        })
    }
}

pub(crate) const COALESCED_SM_TILES_ENV: &str = "ATLAS_QWEN4_COALESCED_SM_TILES";
/// SM-count threshold separating GB10-class (`<= 64` SMs) from high-SM
/// Blackwell GPUs (`> 64` SMs, e.g. SM120 188 SMs) in coalesced tile selection.
const COALESCED_HIGH_SM_THRESHOLD: u32 = 64;

#[inline]
#[must_use]
pub(crate) fn parse_coalesced_sm_tiles(raw: Option<&str>) -> bool {
    raw.map(str::trim) != Some("0")
}

/// `ATLAS_QWEN4_COALESCED_SM_TILES` (default ON; `"0"` disables for A/B):
/// scale coalesced NVFP4 tile selection for high-SM Blackwell GPUs
/// (`active_sm_count() > 64`) to increase CTA occupancy on medium/wide shapes.
#[must_use]
pub(crate) fn coalesced_sm_tiles_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        parse_coalesced_sm_tiles(std::env::var(COALESCED_SM_TILES_ENV).ok().as_deref())
    })
}

#[inline]
#[must_use]
fn effective_coalesced_sm_count() -> u32 {
    if coalesced_sm_tiles_enabled() {
        atlas_core::device::active_sm_count()
    } else {
        atlas_core::device::sm121::NUM_SMS
    }
}

/// SM-aware v2 coalesced NVFP4 tile selector for `[m, n, k]`.
///
/// For `sm_count <= 64` (GB10 SM121), matches
/// [`select_coalesced_nvfp4_tile_for_shape_v2`] exactly. For `sm_count > 64`
/// (e.g. SM120 188 SMs with 96 MiB L2), selects higher-CTA tiles on medium
/// (`N in 1025..=4096, K >= 2048`) and wide (`N in 4097..=24576, m in 33..=64`)
/// shapes so all SMs stay saturated while keeping weight reuse inside L2:
/// - `N = 2560, K = 6144`:
///   - `m <= 16`: `N8Sk8` (`M16`) -> `320` CTAs (`1.70x` 188 SMs, barrier-free K-loop)
///     when `n % 8 == 0`, else `N32Sk4` (`M16`);
///   - `m in 17..=32`: `N16Sk4` (`M16`, `grid.y = 2`) -> `320` CTAs (`256` threads/CTA);
///   - `m in 33..=64`: `N32Sk4` (`M32`, `grid.y = 2`) -> `160` CTAs (`512` threads/CTA);
///   - `m in 65..=256`: `N32Sk2` (`M64`, `grid.y = 2..=4`) -> `160..=320` CTAs (v2 2-stage K-prefetch).
/// - `N in 4097..=24576` (`hc.up`, `attn.q`, `ssm.qkvz`):
///   - `m in 33..=64`: `N32` (`M32`, `grid.y = 2`) -> `640..1024` CTAs on
///     `w4a16_gemm_coalesced_v2_n32_m32` (4 CTAs/SM, 2-stage K-prefetch),
///     degrading cleanly to `w4a16_gemm_coalesced_n32_m64` if v2 is absent.
#[inline]
#[must_use]
pub(crate) const fn select_coalesced_nvfp4_tile_for_shape_v2_with_sms(
    m: usize,
    n: usize,
    k: usize,
    sm_count: u32,
) -> Option<CoalescedNvfp4Tile> {
    if m == 0 || m > COALESCED_NVFP4_MAX_ROWS || n == 0 || k == 0 {
        return None;
    }
    if sm_count <= COALESCED_HIGH_SM_THRESHOLD {
        return select_coalesced_nvfp4_tile_for_shape_v2(m, n, k);
    }
    if n <= 1024 {
        return select_coalesced_nvfp4_tile_for_shape_v2(m, n, k);
    }
    if n <= 4096 {
        if k >= 2048 {
            return Some(if m <= 16 {
                let n_tile = if n.is_multiple_of(8) {
                    CoalescedNvfp4NTile::N8Sk8
                } else {
                    CoalescedNvfp4NTile::N32Sk4
                };
                CoalescedNvfp4Tile {
                    n_tile,
                    m_tile: CoalescedNvfp4MTile::M16,
                }
            } else if m <= 32 {
                CoalescedNvfp4Tile {
                    n_tile: CoalescedNvfp4NTile::N16Sk4,
                    m_tile: CoalescedNvfp4MTile::M16,
                }
            } else if m <= 64 {
                CoalescedNvfp4Tile {
                    n_tile: CoalescedNvfp4NTile::N32Sk4,
                    m_tile: CoalescedNvfp4MTile::M32,
                }
            } else {
                CoalescedNvfp4Tile {
                    n_tile: CoalescedNvfp4NTile::N32Sk2,
                    m_tile: CoalescedNvfp4MTile::M64,
                }
            });
        }
        return select_coalesced_nvfp4_tile_for_shape_v2(m, n, k);
    }
    if n <= 24_576 && m > 32 && m <= 64 {
        return Some(CoalescedNvfp4Tile {
            n_tile: CoalescedNvfp4NTile::N32,
            m_tile: CoalescedNvfp4MTile::M32,
        });
    }
    select_coalesced_nvfp4_tile_for_shape_v2(m, n, k)
}

/// Resolve the preferred coalesced kernel symbol for `[m, n, k]` under the
/// active `ATLAS_QWEN4_COALESCED_V2` setting (used for one-time telemetry logs).
#[inline]
#[must_use]
pub(crate) fn coalesced_nvfp4_dispatch_symbol(m: usize, n: usize, k: usize) -> &'static str {
    let use_v2 = coalesced_v2_enabled() && k.is_multiple_of(64);
    let sm_count = effective_coalesced_sm_count();
    let tile = if use_v2 {
        select_coalesced_nvfp4_tile_for_shape_v2_with_sms(m, n, k, sm_count)
    } else {
        select_coalesced_nvfp4_tile_for_shape(m, n, k)
    };
    tile.map_or("w4a16_gemm_coalesced", |t| t.dispatch_symbol(use_v2))
}

/// Maximum output width `n` considered narrow for batched-verify coalesced NVFP4
/// vs cached FP8 cuBLASLt selection (covers `ssm.out` and `attn.oproj` at
/// `N = 2560` while keeping wide `ssm.qkvz` `N = 16384` on coalesced NVFP4).
pub(crate) const COALESCED_NVFP4_NARROW_N_MAX: usize = 4_096;

/// Minimum row count for narrow `n <= 1024` projections (e.g. `moe.router`
/// `N = 512`) to engage coalesced NVFP4 Tensor Core dispatch while keeping
/// `C=1` exact parity probes (`rows <= 9`) and `C=2 K=5` (`rows = 10`) on the
/// exact scalar GEMV path.
pub(crate) const COALESCED_NVFP4_NARROW_M_MIN: usize = 12;

/// Maximum row count where narrow-`N` (`n <= 4096`) projections stay on
/// coalesced/Warp-Split-K NVFP4 before handing off to cached FP8 cuBLASLt.
/// Covers the full `1..=256` batched-verify ceiling (`C=32/64`) via
/// `grid.y = div_ceil(m, tile_m)` with zero FP8 cache RAM (while
/// `ATLAS_QWEN4_FAST_PROJ_NVFP4` auto-enables native cuBLASLt NVFP4 at
/// `m >= 32` on `> 64`-SM Blackwell GPUs, or opt-in at `m >= 64` on `<= 64`-SM).
pub(crate) const COALESCED_NVFP4_NARROW_M_MAX: usize = COALESCED_NVFP4_MAX_ROWS;

/// Whether a batched-verify projection `[m, n]` meets the empirical GB10
/// minimum-row threshold for coalesced/FP8 Tensor Core dispatch rather than
/// exact scalar NVFP4 GEMV (`m >= DEFAULT_BATCHED_VERIFY_TC_FP8_MIN_ROWS` (4)
/// for `n > 1024`, `m >= COALESCED_NVFP4_NARROW_M_MIN` (12) for narrow
/// `n <= 1024`; smaller `m` stays on exact scalar GEMV).
#[inline]
#[must_use]
pub(crate) const fn batched_verify_dense_proj_tc_admits(m: usize, n: usize) -> bool {
    if n <= 1_024 {
        m >= COALESCED_NVFP4_NARROW_M_MIN
    } else {
        m >= crate::layers::qwen4_hyper::DEFAULT_BATCHED_VERIFY_TC_FP8_MIN_ROWS
    }
}

/// Whether a batched-verify projection `[m, n]` prefers cached FP8 cuBLASLt
/// over coalesced NVFP4 on GB10.
#[inline]
#[must_use]
pub(crate) const fn coalesced_nvfp4_prefers_fp8_cached(m: usize, n: usize) -> bool {
    n <= COALESCED_NVFP4_NARROW_N_MAX && m > COALESCED_NVFP4_NARROW_M_MAX
}

#[inline]
#[must_use]
pub(crate) const fn coalesced_nvfp4_batched_verify_shape_eligible(
    scope_active: bool,
    enabled: bool,
    m: usize,
    n: usize,
    k: usize,
    ldc: usize,
    qg: Option<(usize, usize)>,
) -> bool {
    scope_active
        && enabled
        && m >= crate::layers::qwen4_hyper::DEFAULT_BATCHED_VERIFY_TC_DIRECT_MIN_ROWS
        && m <= COALESCED_NVFP4_MAX_ROWS
        && n > 0
        && k > 0
        && k.is_multiple_of(32)
        && ldc == n
        && qg.is_none()
}

#[derive(Clone, Copy, Debug)]
struct W4a16CoalescedV2Kernels {
    n32_m16: KernelHandle,
    n64_m16: KernelHandle,
    n32_m32: KernelHandle,
    n64_m32: KernelHandle,
    n64_m64: KernelHandle,
    n64_m80: KernelHandle,
    n64_m128: KernelHandle,
    n32_sk2_m64: KernelHandle,
    n32_sk2_m80: KernelHandle,
    /// Grid-Split-K `N8Sk8 M32` v2 kernel (`w4a16_gemm_coalesced_n8_sk8_m32_dual_silu_div_v2`),
    /// invoked with `N1 = 0, N2 = n` on high-SM Blackwell GPUs (`sm_count > 64`) to saturate
    /// narrow `N <= 1024, K >= 2048, M <= 16` projections (e.g., `N = 512` Router GEMM:
    /// `grid = [64, 1, 3] -> 192` CTAs instead of `64` CTAs).
    n8_sk8_m32_grid_v2: KernelHandle,
}

impl W4a16CoalescedV2Kernels {
    #[cfg(test)]
    const fn empty() -> Self {
        Self {
            n32_m16: KernelHandle(0),
            n64_m16: KernelHandle(0),
            n32_m32: KernelHandle(0),
            n64_m32: KernelHandle(0),
            n64_m64: KernelHandle(0),
            n64_m80: KernelHandle(0),
            n64_m128: KernelHandle(0),
            n32_sk2_m64: KernelHandle(0),
            n32_sk2_m80: KernelHandle(0),
            n8_sk8_m32_grid_v2: KernelHandle(0),
        }
    }

    #[inline]
    const fn lookup(self, tile: CoalescedNvfp4Tile) -> KernelHandle {
        match (tile.n_tile, tile.m_tile) {
            (CoalescedNvfp4NTile::N32, CoalescedNvfp4MTile::M16) => self.n32_m16,
            (CoalescedNvfp4NTile::N64, CoalescedNvfp4MTile::M16) => self.n64_m16,
            (CoalescedNvfp4NTile::N32, CoalescedNvfp4MTile::M32) => self.n32_m32,
            (CoalescedNvfp4NTile::N64, CoalescedNvfp4MTile::M32) => self.n64_m32,
            (CoalescedNvfp4NTile::N64, CoalescedNvfp4MTile::M64) => self.n64_m64,
            (CoalescedNvfp4NTile::N64, CoalescedNvfp4MTile::M80) => self.n64_m80,
            (CoalescedNvfp4NTile::N64, CoalescedNvfp4MTile::M128) => self.n64_m128,
            (CoalescedNvfp4NTile::N32Sk2, CoalescedNvfp4MTile::M64) => self.n32_sk2_m64,
            (CoalescedNvfp4NTile::N32Sk2, CoalescedNvfp4MTile::M80) => self.n32_sk2_m80,
            _ => KernelHandle(0),
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct W4a16CoalescedKernels {
    n8_sk8_m16: KernelHandle,
    n16_sk4_m16: KernelHandle,
    n32_sk4_m16: KernelHandle,
    n64_sk4_m16: KernelHandle,
    n32_sk4_m32: KernelHandle,
    n64_sk2_m64: KernelHandle,
    n16: [KernelHandle; 3],
    n32: [KernelHandle; 3],
    n64: [KernelHandle; 3],
    n128: [KernelHandle; 3],
    v2: W4a16CoalescedV2Kernels,
}

impl W4a16CoalescedKernels {
    #[inline]
    const fn base_m_tile(m: usize) -> CoalescedNvfp4MTile {
        if m <= 32 {
            CoalescedNvfp4MTile::M32
        } else if m <= 64 {
            CoalescedNvfp4MTile::M64
        } else {
            CoalescedNvfp4MTile::M128
        }
    }

    #[inline]
    const fn base_m_idx(m_tile: CoalescedNvfp4MTile) -> usize {
        match m_tile {
            CoalescedNvfp4MTile::M16 | CoalescedNvfp4MTile::M32 => 0,
            CoalescedNvfp4MTile::M64 => 1,
            CoalescedNvfp4MTile::M80 | CoalescedNvfp4MTile::M128 => 2,
        }
    }

    /// Resolve a preferred v2 tile when `prefer_v2` is set and the exact v2
    /// symbol is available, degrading cleanly to [`Self::resolve`] otherwise so
    /// the returned [`CoalescedNvfp4Tile`] always matches the resolved handle's
    /// `(co_n, tile_m, threads)` launch contract.
    #[inline]
    const fn resolve_with_v2(
        self,
        preferred: CoalescedNvfp4Tile,
        m: usize,
        prefer_v2: bool,
    ) -> Option<(CoalescedNvfp4Tile, KernelHandle)> {
        if prefer_v2 {
            let h = self.v2.lookup(preferred);
            if h.0 != 0 {
                return Some((preferred, h));
            }
        }
        self.resolve(preferred, m)
    }

    /// Resolve the preferred tile handle, degrading cleanly from Warp-Split-K
    /// and `N64` tiers to baseline `n16`/`n32`/`n128` tiers if a module lacks
    /// specialized symbols, while guaranteeing the returned [`CoalescedNvfp4Tile`]
    /// matches the resolved [`KernelHandle`]'s `(co_n, tile_m, threads)` contract.
    #[inline]
    const fn resolve(
        self,
        preferred: CoalescedNvfp4Tile,
        m: usize,
    ) -> Option<(CoalescedNvfp4Tile, KernelHandle)> {
        const fn tile(
            n_tile: CoalescedNvfp4NTile,
            m_tile: CoalescedNvfp4MTile,
        ) -> CoalescedNvfp4Tile {
            CoalescedNvfp4Tile { n_tile, m_tile }
        }
        let base_m = Self::base_m_tile(m);
        let bmi = Self::base_m_idx(base_m);
        let none_slot = (tile(CoalescedNvfp4NTile::N128, base_m), KernelHandle(0));
        let candidates = match (preferred.n_tile, preferred.m_tile) {
            (CoalescedNvfp4NTile::N8Sk8, _) => [
                (
                    tile(CoalescedNvfp4NTile::N8Sk8, CoalescedNvfp4MTile::M16),
                    self.n8_sk8_m16,
                ),
                (tile(CoalescedNvfp4NTile::N16, base_m), self.n16[bmi]),
                (tile(CoalescedNvfp4NTile::N32, base_m), self.n32[bmi]),
                (tile(CoalescedNvfp4NTile::N128, base_m), self.n128[bmi]),
            ],
            (CoalescedNvfp4NTile::N16Sk4, _) => [
                (
                    tile(CoalescedNvfp4NTile::N16Sk4, CoalescedNvfp4MTile::M16),
                    self.n16_sk4_m16,
                ),
                (tile(CoalescedNvfp4NTile::N16, base_m), self.n16[bmi]),
                (tile(CoalescedNvfp4NTile::N32, base_m), self.n32[bmi]),
                (tile(CoalescedNvfp4NTile::N128, base_m), self.n128[bmi]),
            ],
            (CoalescedNvfp4NTile::N32Sk2, _) => {
                if m <= 64 {
                    [
                        (
                            tile(CoalescedNvfp4NTile::N64Sk2, CoalescedNvfp4MTile::M64),
                            self.n64_sk2_m64,
                        ),
                        (tile(CoalescedNvfp4NTile::N64, base_m), self.n64[bmi]),
                        (tile(CoalescedNvfp4NTile::N32, base_m), self.n32[bmi]),
                        (tile(CoalescedNvfp4NTile::N128, base_m), self.n128[bmi]),
                    ]
                } else {
                    [
                        (tile(CoalescedNvfp4NTile::N64, base_m), self.n64[bmi]),
                        (tile(CoalescedNvfp4NTile::N32, base_m), self.n32[bmi]),
                        (tile(CoalescedNvfp4NTile::N128, base_m), self.n128[bmi]),
                        none_slot,
                    ]
                }
            }
            (CoalescedNvfp4NTile::N32Sk4, CoalescedNvfp4MTile::M16) => [
                (
                    tile(CoalescedNvfp4NTile::N32Sk4, CoalescedNvfp4MTile::M16),
                    self.n32_sk4_m16,
                ),
                (tile(CoalescedNvfp4NTile::N16, base_m), self.n16[bmi]),
                (tile(CoalescedNvfp4NTile::N32, base_m), self.n32[bmi]),
                (tile(CoalescedNvfp4NTile::N128, base_m), self.n128[bmi]),
            ],
            (CoalescedNvfp4NTile::N32Sk4, _) => [
                (
                    tile(CoalescedNvfp4NTile::N32Sk4, CoalescedNvfp4MTile::M32),
                    self.n32_sk4_m32,
                ),
                (tile(CoalescedNvfp4NTile::N32, base_m), self.n32[bmi]),
                (tile(CoalescedNvfp4NTile::N64, base_m), self.n64[bmi]),
                (tile(CoalescedNvfp4NTile::N128, base_m), self.n128[bmi]),
            ],
            (CoalescedNvfp4NTile::N64Sk2, _) => [
                (
                    tile(CoalescedNvfp4NTile::N64Sk2, CoalescedNvfp4MTile::M64),
                    self.n64_sk2_m64,
                ),
                (tile(CoalescedNvfp4NTile::N64, base_m), self.n64[bmi]),
                (tile(CoalescedNvfp4NTile::N32, base_m), self.n32[bmi]),
                (tile(CoalescedNvfp4NTile::N128, base_m), self.n128[bmi]),
            ],
            (CoalescedNvfp4NTile::N64Sk4, _) => [
                (
                    tile(CoalescedNvfp4NTile::N64Sk4, CoalescedNvfp4MTile::M16),
                    self.n64_sk4_m16,
                ),
                (
                    tile(CoalescedNvfp4NTile::N32Sk4, CoalescedNvfp4MTile::M16),
                    self.n32_sk4_m16,
                ),
                (tile(CoalescedNvfp4NTile::N16, base_m), self.n16[bmi]),
                (tile(CoalescedNvfp4NTile::N128, base_m), self.n128[bmi]),
            ],
            (CoalescedNvfp4NTile::N16, _) => [
                (tile(CoalescedNvfp4NTile::N16, base_m), self.n16[bmi]),
                (tile(CoalescedNvfp4NTile::N32, base_m), self.n32[bmi]),
                (tile(CoalescedNvfp4NTile::N128, base_m), self.n128[bmi]),
                none_slot,
            ],
            (CoalescedNvfp4NTile::N32, _) => [
                (tile(CoalescedNvfp4NTile::N32, base_m), self.n32[bmi]),
                (tile(CoalescedNvfp4NTile::N64, base_m), self.n64[bmi]),
                (tile(CoalescedNvfp4NTile::N128, base_m), self.n128[bmi]),
                none_slot,
            ],
            (CoalescedNvfp4NTile::N64, _) => [
                (tile(CoalescedNvfp4NTile::N64, base_m), self.n64[bmi]),
                (tile(CoalescedNvfp4NTile::N128, base_m), self.n128[bmi]),
                (tile(CoalescedNvfp4NTile::N32, base_m), self.n32[bmi]),
                none_slot,
            ],
            (CoalescedNvfp4NTile::N128, _) => [
                (tile(CoalescedNvfp4NTile::N128, base_m), self.n128[bmi]),
                (tile(CoalescedNvfp4NTile::N64, base_m), self.n64[bmi]),
                none_slot,
                none_slot,
            ],
        };
        let mut i = 0;
        while i < 4 {
            let (cand_tile, h) = candidates[i];
            if h.0 != 0 {
                return Some((cand_tile, h));
            }
            i += 1;
        }
        None
    }
}

struct W4a16CoalescedKernelCache {
    primary: OnceLock<(usize, W4a16CoalescedKernels)>,
    fallback: OnceLock<parking_lot::Mutex<std::collections::HashMap<usize, W4a16CoalescedKernels>>>,
}

impl W4a16CoalescedKernelCache {
    const fn new() -> Self {
        Self {
            primary: OnceLock::new(),
            fallback: OnceLock::new(),
        }
    }

    fn get_or_load(&self, gpu: &dyn GpuBackend) -> W4a16CoalescedKernels {
        let key = std::ptr::from_ref(gpu) as *const () as usize;
        if let Some(&(cached_key, cached_val)) = self.primary.get() {
            if cached_key == key {
                return cached_val;
            }
        } else {
            let val = probe_w4a16_coalesced_kernels(gpu);
            let &(stored_key, stored_val) = self.primary.get_or_init(|| (key, val));
            if stored_key == key {
                return stored_val;
            }
            let map = self
                .fallback
                .get_or_init(|| parking_lot::Mutex::new(std::collections::HashMap::new()));
            return *map.lock().entry(key).or_insert(val);
        }
        let map = self
            .fallback
            .get_or_init(|| parking_lot::Mutex::new(std::collections::HashMap::new()));
        if let Some(val) = map.lock().get(&key).copied() {
            return val;
        }
        let val = probe_w4a16_coalesced_kernels(gpu);
        *map.lock().entry(key).or_insert(val)
    }
}

fn try_w4a16_kernel(gpu: &dyn GpuBackend, name: &str) -> KernelHandle {
    let h = crate::layers::try_kernel(gpu, "w4a16", name);
    if h.0 != 0 {
        h
    } else {
        crate::layers::try_kernel(gpu, "w4a16_gemm", name)
    }
}

fn probe_w4a16_coalesced_kernels(gpu: &dyn GpuBackend) -> W4a16CoalescedKernels {
    W4a16CoalescedKernels {
        n8_sk8_m16: try_w4a16_kernel(gpu, "w4a16_gemm_coalesced_n8_sk8_m16"),
        n16_sk4_m16: try_w4a16_kernel(gpu, "w4a16_gemm_coalesced_n16_sk4_m16"),
        n32_sk4_m16: try_w4a16_kernel(gpu, "w4a16_gemm_coalesced_n32_sk4_m16"),
        n64_sk4_m16: try_w4a16_kernel(gpu, "w4a16_gemm_coalesced_n64_sk4_m16"),
        n32_sk4_m32: try_w4a16_kernel(gpu, "w4a16_gemm_coalesced_n32_sk4_m32"),
        n64_sk2_m64: try_w4a16_kernel(gpu, "w4a16_gemm_coalesced_n64_sk2_m64"),
        n16: [
            try_w4a16_kernel(gpu, "w4a16_gemm_coalesced_n16_m32"),
            try_w4a16_kernel(gpu, "w4a16_gemm_coalesced_n16_m64"),
            try_w4a16_kernel(gpu, "w4a16_gemm_coalesced_n16_m128"),
        ],
        n32: [
            try_w4a16_kernel(gpu, "w4a16_gemm_coalesced_n32_m32"),
            try_w4a16_kernel(gpu, "w4a16_gemm_coalesced_n32_m64"),
            try_w4a16_kernel(gpu, "w4a16_gemm_coalesced_n32_m128"),
        ],
        n64: [
            try_w4a16_kernel(gpu, "w4a16_gemm_coalesced_n64_m32"),
            try_w4a16_kernel(gpu, "w4a16_gemm_coalesced_n64_m64"),
            try_w4a16_kernel(gpu, "w4a16_gemm_coalesced_n64_m128"),
        ],
        n128: [
            try_w4a16_kernel(gpu, "w4a16_gemm_coalesced_m32"),
            try_w4a16_kernel(gpu, "w4a16_gemm_coalesced_m64"),
            try_w4a16_kernel(gpu, "w4a16_gemm_coalesced_m128"),
        ],
        v2: W4a16CoalescedV2Kernels {
            n32_m16: try_w4a16_kernel(gpu, "w4a16_gemm_coalesced_v2_n32_m16"),
            n64_m16: try_w4a16_kernel(gpu, "w4a16_gemm_coalesced_v2_n64_m16"),
            n32_m32: try_w4a16_kernel(gpu, "w4a16_gemm_coalesced_v2_n32_m32"),
            n64_m32: try_w4a16_kernel(gpu, "w4a16_gemm_coalesced_v2_n64_m32"),
            n64_m64: try_w4a16_kernel(gpu, "w4a16_gemm_coalesced_v2_n64_m64"),
            n64_m80: try_w4a16_kernel(gpu, "w4a16_gemm_coalesced_v2_n64_m80"),
            n64_m128: try_w4a16_kernel(gpu, "w4a16_gemm_coalesced_v2_n64_m128"),
            n32_sk2_m64: try_w4a16_kernel(gpu, "w4a16_gemm_coalesced_v2_n32_sk2_m64"),
            n32_sk2_m80: try_w4a16_kernel(gpu, "w4a16_gemm_coalesced_v2_n32_sk2_m80"),
            n8_sk8_m32_grid_v2: try_w4a16_kernel(
                gpu,
                "w4a16_gemm_coalesced_n8_sk8_m32_dual_silu_div_v2",
            ),
        },
    }
}

/// Bounds for narrow high-SM grid-Split-K (`N <= 1024, M <= 16, K >= 2048, K % 256 == 0`).
const NARROW_SPLITK_MAX_M: usize = 16;
const NARROW_SPLITK_MAX_N: usize = 1_024;
const NARROW_SPLITK_MIN_K: usize = 2_048;
const NARROW_SPLITK_MAX_TILES: usize = NARROW_SPLITK_MAX_N / 8; // 128 tiles (`grid_y = 1`)
const NARROW_SPLITK_MAX_Z: u32 = 4;
const NARROW_SPLITK_TILE_ELEMS: usize = 32 * 8; // TILE_M = 32, CO_N = 8

#[derive(Clone, Copy, Debug)]
struct NarrowGridSplitKScratch {
    partials: DevicePtr,
    counters: DevicePtr,
}

struct NarrowGridSplitKScratchCache {
    primary: OnceLock<(usize, Option<NarrowGridSplitKScratch>)>,
    fallback: OnceLock<
        parking_lot::Mutex<std::collections::HashMap<usize, Option<NarrowGridSplitKScratch>>>,
    >,
}

impl NarrowGridSplitKScratchCache {
    const fn new() -> Self {
        Self {
            primary: OnceLock::new(),
            fallback: OnceLock::new(),
        }
    }

    fn get_if_warmed(&self, gpu: &dyn GpuBackend) -> Option<NarrowGridSplitKScratch> {
        let key = std::ptr::from_ref(gpu) as *const () as usize;
        if let Some(&(cached_key, cached_val)) = self.primary.get()
            && cached_key == key
        {
            return cached_val;
        }
        self.fallback
            .get()
            .and_then(|m| m.lock().get(&key).copied().flatten())
    }

    fn get_or_alloc(&self, gpu: &dyn GpuBackend) -> Option<NarrowGridSplitKScratch> {
        let key = std::ptr::from_ref(gpu) as *const () as usize;
        if let Some(&(cached_key, cached_val)) = self.primary.get() {
            if cached_key == key {
                return cached_val;
            }
        } else {
            let val = alloc_narrow_grid_splitk_scratch(gpu);
            let &(stored_key, stored_val) = self.primary.get_or_init(|| (key, val));
            if stored_key == key {
                return stored_val;
            }
            let map = self
                .fallback
                .get_or_init(|| parking_lot::Mutex::new(std::collections::HashMap::new()));
            return *map.lock().entry(key).or_insert(val);
        }
        let map = self
            .fallback
            .get_or_init(|| parking_lot::Mutex::new(std::collections::HashMap::new()));
        if let Some(&val) = map.lock().get(&key) {
            return val;
        }
        let val = alloc_narrow_grid_splitk_scratch(gpu);
        *map.lock().entry(key).or_insert(val)
    }
}

fn alloc_narrow_grid_splitk_scratch(gpu: &dyn GpuBackend) -> Option<NarrowGridSplitKScratch> {
    let partials_bytes =
        NARROW_SPLITK_MAX_TILES * (NARROW_SPLITK_MAX_Z as usize) * NARROW_SPLITK_TILE_ELEMS * 4;
    let counters_bytes = NARROW_SPLITK_MAX_TILES * 4;
    let partials = gpu.alloc(partials_bytes).ok()?;
    let counters = match gpu.alloc(counters_bytes) {
        Ok(ptr) => ptr,
        Err(_) => {
            let _ = gpu.free(partials);
            return None;
        }
    };
    let stream = gpu.default_stream();
    if gpu
        .memset_async(counters, 0, counters_bytes, stream)
        .is_err()
        || gpu.synchronize(stream).is_err()
    {
        let _ = gpu.free(counters);
        let _ = gpu.free(partials);
        return None;
    }
    Some(NarrowGridSplitKScratch { partials, counters })
}

static NARROW_SPLITK_SCRATCH: NarrowGridSplitKScratchCache = NarrowGridSplitKScratchCache::new();

/// Select inter-CTA `grid_z` (`2..=4`) for narrow `N <= 1024, M <= 16, K >= 2048`
/// NVFP4 W4A16 projections (such as `N = 512, K = 2560` MoE Router GEMM) on
/// high-SM Blackwell GPUs (`sm_count > 64`).
///
/// On SM120 (`188` SMs), `N = 512, K = 2560` at `CO_N = 8` gives `grid_x = 64`
/// and `grid_z = ceil(188 / 64) = 3` (`192` CTAs across all 188 SMs in 1 K-loop
/// iteration instead of 64 CTAs over 2 iterations). GB10 (`<= 64` SMs) always
/// returns `None`.
#[inline]
#[must_use]
pub(crate) const fn select_narrow_grid_split_k_z(
    m: usize,
    n: usize,
    k: usize,
    sm_count: u32,
    use_v2: bool,
    has_kernel: bool,
    has_scratch: bool,
) -> Option<u32> {
    if sm_count <= COALESCED_HIGH_SM_THRESHOLD
        || !use_v2
        || !has_kernel
        || !has_scratch
        || m == 0
        || m > NARROW_SPLITK_MAX_M
        || n == 0
        || n > NARROW_SPLITK_MAX_N
        || k < NARROW_SPLITK_MIN_K
        || !k.is_multiple_of(256)
    {
        return None;
    }
    let grid_x = (n as u32).div_ceil(8);
    if grid_x == 0 || (grid_x as usize) > NARROW_SPLITK_MAX_TILES {
        return None;
    }
    let k_units = (k / 256) as u32;
    let max_z = if k_units < NARROW_SPLITK_MAX_Z {
        k_units
    } else {
        NARROW_SPLITK_MAX_Z
    };
    let target_z = sm_count.div_ceil(grid_x);
    let gz = if target_z < max_z { target_z } else { max_z };
    if gz >= 2 { Some(gz) } else { None }
}

fn w4a16_coalesced_kernels(gpu: &dyn GpuBackend) -> W4a16CoalescedKernels {
    static CACHE: W4a16CoalescedKernelCache = W4a16CoalescedKernelCache::new();
    CACHE.get_or_load(gpu)
}

/// Pre-load the coalesced NVFP4, transposed shared-expert fast-v2, and
/// (when `ATLAS_QWEN4_FAST_PROJ_NVFP4` is active) cuBLASLt NVFP4 quantization
/// kernel tables eagerly before any CUDA graph stream capture begins so
/// first-time module loads or scratch allocations never occur during capture.
pub(crate) fn warm_coalesced_nvfp4_kernels(gpu: &dyn GpuBackend) {
    let kernels = w4a16_coalesced_kernels(gpu);
    if effective_coalesced_sm_count() > COALESCED_HIGH_SM_THRESHOLD
        && kernels.v2.n8_sk8_m32_grid_v2.0 != 0
    {
        let _ = NARROW_SPLITK_SCRATCH.get_or_alloc(gpu);
    }
    let _ = w4a16_shared_fast_v2_kernels(gpu);
    nvfp4::warm_kernels_and_scratch(gpu);
}

/// Launch single-sweep or Warp-Split-K K-coalesced NVFP4 W4A16 GEMM
/// (`w4a16_gemm_coalesced*`) for `m in 1..=256`, `k % 32 == 0`, `n >= 1`
/// (or native cuBLASLt NVFP4 when `ATLAS_QWEN4_FAST_PROJ_NVFP4` is active
/// inside batched verify for `m >= verify_min_m` wide dense projections),
/// returning `Ok(true)` if dispatched or `Ok(false)` if ineligible/unavailable.
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_coalesced_nvfp4_w4a16_gemm(
    gpu: &dyn GpuBackend,
    act: DevicePtr,
    weight: &QuantizedWeight,
    out: DevicePtr,
    m: usize,
    n: usize,
    k: usize,
    stream: u64,
) -> Result<bool> {
    if k == 0
        || !k.is_multiple_of(32)
        || act.is_null()
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
    if nvfp4::try_gemm_coalesced_verify(gpu, act, weight, out, m, n, k, stream)? {
        return Ok(true);
    }
    let (Ok(m_u32), Ok(n_u32), Ok(k_u32)) = (u32::try_from(m), u32::try_from(n), u32::try_from(k))
    else {
        return Ok(false);
    };
    let use_v2 =
        coalesced_v2_enabled() && k.is_multiple_of(64) && weight.weight_scale.0.is_multiple_of(4);
    let sm_count = effective_coalesced_sm_count();
    let kernels = w4a16_coalesced_kernels(gpu);
    if sm_count > COALESCED_HIGH_SM_THRESHOLD && kernels.v2.n8_sk8_m32_grid_v2.0 != 0 {
        let scratch = NARROW_SPLITK_SCRATCH.get_if_warmed(gpu).or_else(|| {
            if spark_runtime::gpu_capture_region::active_capture_region().is_some() {
                None
            } else {
                NARROW_SPLITK_SCRATCH.get_or_alloc(gpu)
            }
        });
        if let Some(scratch) = scratch
            && let Some(grid_z) =
                select_narrow_grid_split_k_z(m, n, k, sm_count, use_v2, true, true)
        {
            KernelLaunch::new(gpu, kernels.v2.n8_sk8_m32_grid_v2)
                .grid([div_ceil(n_u32, 8), 1, grid_z])
                .block([256, 1, 1])
                .pdl(true)
                .arg_ptr(act)
                .arg_ptr(DevicePtr::NULL)
                .arg_ptr(DevicePtr::NULL)
                .arg_f32(0.0)
                .arg_ptr(DevicePtr::NULL)
                .arg_u32(0)
                .arg_ptr(weight.weight)
                .arg_ptr(weight.weight_scale)
                .arg_f32(weight.weight_scale_2)
                .arg_ptr(out)
                .arg_u32(n_u32)
                .arg_u32(m_u32)
                .arg_u32(k_u32)
                .arg_f32(1.0)
                .arg_ptr(scratch.partials)
                .arg_ptr(scratch.counters)
                .launch(stream)?;
            return Ok(true);
        }
    }
    let Some(preferred) = (if use_v2 {
        select_coalesced_nvfp4_tile_for_shape_v2_with_sms(m, n, k, sm_count)
    } else {
        select_coalesced_nvfp4_tile_for_shape(m, n, k)
    }) else {
        return Ok(false);
    };
    let Some((tile, kernel)) = kernels.resolve_with_v2(preferred, m, use_v2) else {
        return Ok(false);
    };
    KernelLaunch::new(gpu, kernel)
        .grid([
            div_ceil(n_u32, tile.co_n()),
            div_ceil(m_u32, tile.tile_m()),
            1,
        ])
        .block([tile.threads(), 1, 1])
        .arg_ptr(act)
        .arg_ptr(weight.weight)
        .arg_ptr(weight.weight_scale)
        .arg_f32(weight.weight_scale_2)
        .arg_ptr(out)
        .arg_u32(m_u32)
        .arg_u32(n_u32)
        .arg_u32(k_u32)
        .launch(stream)?;
    Ok(true)
}

/// High-CTA transposed NVFP4 W4A16 gate+up+SiLU tile for batched-decode shared
/// expert (`rows in 2..=256`, `N = 640`, `K = 2560`, `CO_N = 16` -> 40..80 CTAs).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SharedExpertGateUpTile {
    Sk4M16,
    Sk4M32,
    Sk2M32,
    Sk2M64,
}

impl SharedExpertGateUpTile {
    #[inline]
    #[must_use]
    pub(crate) const fn co_n(self) -> u32 {
        16
    }

    #[inline]
    #[must_use]
    pub(crate) const fn tile_m(self) -> u32 {
        match self {
            Self::Sk4M16 => 16,
            Self::Sk4M32 | Self::Sk2M32 => 32,
            Self::Sk2M64 => 64,
        }
    }

    #[inline]
    #[must_use]
    pub(crate) const fn threads(self) -> u32 {
        match self {
            Self::Sk4M16 | Self::Sk4M32 => 256,
            Self::Sk2M32 | Self::Sk2M64 => 128,
        }
    }

    #[inline]
    #[must_use]
    pub(crate) const fn symbol(self) -> &'static str {
        match self {
            Self::Sk4M16 => "w4a16_shared_gateup_silu_t_n16_sk4_m16",
            Self::Sk4M32 => "w4a16_shared_gateup_silu_t_n16_sk4_m32",
            Self::Sk2M32 => "w4a16_shared_gateup_silu_t_n16_sk2_m32",
            Self::Sk2M64 => "w4a16_shared_gateup_silu_t_n16_sk2_m64",
        }
    }
}

/// High-CTA transposed NVFP4 W4A16 down-projection tile for batched-decode
/// shared expert (`rows in 2..=256`, `N = 2560`, `K = 640`, `CO_N = 32` -> 80 CTAs).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SharedExpertDownTile {
    M16,
    M32,
    M64,
    M80,
    M128,
}

impl SharedExpertDownTile {
    #[inline]
    #[must_use]
    pub(crate) const fn co_n(self) -> u32 {
        32
    }

    #[inline]
    #[must_use]
    pub(crate) const fn tile_m(self) -> u32 {
        match self {
            Self::M16 => 16,
            Self::M32 => 32,
            Self::M64 => 64,
            Self::M80 => 80,
            Self::M128 => 128,
        }
    }

    #[inline]
    #[must_use]
    pub(crate) const fn threads(self) -> u32 {
        128
    }

    #[inline]
    #[must_use]
    pub(crate) const fn symbol(self) -> &'static str {
        match self {
            Self::M16 => "w4a16_shared_down_t_n32_m16",
            Self::M32 => "w4a16_shared_down_t_n32_m32",
            Self::M64 => "w4a16_shared_down_t_n32_m64",
            Self::M80 => "w4a16_shared_down_t_n32_m80",
            Self::M128 => "w4a16_shared_down_t_n32_m128",
        }
    }
}

/// Select the empirically optimal GB10 `(gate+up+SiLU, down)` transposed NVFP4
/// tile pair for batched-decode shared expert (`m in 2..=256`).
#[inline]
#[must_use]
pub(crate) const fn select_shared_expert_nvfp4_t_tiles(
    m: usize,
) -> Option<(SharedExpertGateUpTile, SharedExpertDownTile)> {
    if m < 2 || m > COALESCED_NVFP4_MAX_ROWS {
        return None;
    }
    let gu = if m <= 16 {
        SharedExpertGateUpTile::Sk4M16
    } else if m <= 48 {
        SharedExpertGateUpTile::Sk4M32
    } else {
        SharedExpertGateUpTile::Sk2M64
    };
    let dn = if m <= 16 {
        SharedExpertDownTile::M16
    } else if m <= 32 {
        SharedExpertDownTile::M32
    } else if m <= 64 {
        SharedExpertDownTile::M64
    } else if m <= 80 {
        SharedExpertDownTile::M80
    } else {
        SharedExpertDownTile::M128
    };
    Some((gu, dn))
}

#[derive(Clone, Copy, Debug)]
struct W4a16SharedFastV2Kernels {
    gu_sk4_m16: KernelHandle,
    gu_sk4_m32: KernelHandle,
    gu_sk2_m32: KernelHandle,
    gu_sk2_m64: KernelHandle,
    dn_m16: KernelHandle,
    dn_m32: KernelHandle,
    dn_m64: KernelHandle,
    dn_m80: KernelHandle,
    dn_m128: KernelHandle,
}

#[derive(Clone, Copy, Debug)]
struct ResolvedSharedFastV2 {
    gu_tile: SharedExpertGateUpTile,
    gu_kernel: KernelHandle,
    dn_tile: SharedExpertDownTile,
    dn_kernel: KernelHandle,
}

impl W4a16SharedFastV2Kernels {
    #[inline]
    const fn lookup_gateup(self, tile: SharedExpertGateUpTile) -> KernelHandle {
        match tile {
            SharedExpertGateUpTile::Sk4M16 => self.gu_sk4_m16,
            SharedExpertGateUpTile::Sk4M32 => self.gu_sk4_m32,
            SharedExpertGateUpTile::Sk2M32 => self.gu_sk2_m32,
            SharedExpertGateUpTile::Sk2M64 => self.gu_sk2_m64,
        }
    }

    #[inline]
    const fn lookup_down(self, tile: SharedExpertDownTile) -> KernelHandle {
        match tile {
            SharedExpertDownTile::M16 => self.dn_m16,
            SharedExpertDownTile::M32 => self.dn_m32,
            SharedExpertDownTile::M64 => self.dn_m64,
            SharedExpertDownTile::M80 => self.dn_m80,
            SharedExpertDownTile::M128 => self.dn_m128,
        }
    }

    #[inline]
    const fn resolve(self, m: usize) -> Option<ResolvedSharedFastV2> {
        let Some((pref_gu_tile, dn_tile)) = select_shared_expert_nvfp4_t_tiles(m) else {
            return None;
        };
        let pref_gu_kernel = self.lookup_gateup(pref_gu_tile);
        let (gu_tile, gu_kernel) = if matches!(pref_gu_tile, SharedExpertGateUpTile::Sk2M64)
            && pref_gu_kernel.0 == 0
            && self.gu_sk2_m32.0 != 0
        {
            (SharedExpertGateUpTile::Sk2M32, self.gu_sk2_m32)
        } else {
            (pref_gu_tile, pref_gu_kernel)
        };
        let dn_kernel = self.lookup_down(dn_tile);
        if gu_kernel.0 == 0 || dn_kernel.0 == 0 {
            return None;
        }
        Some(ResolvedSharedFastV2 {
            gu_tile,
            gu_kernel,
            dn_tile,
            dn_kernel,
        })
    }
}

struct W4a16SharedFastV2KernelCache {
    primary: OnceLock<(usize, W4a16SharedFastV2Kernels)>,
    fallback:
        OnceLock<parking_lot::Mutex<std::collections::HashMap<usize, W4a16SharedFastV2Kernels>>>,
}

impl W4a16SharedFastV2KernelCache {
    const fn new() -> Self {
        Self {
            primary: OnceLock::new(),
            fallback: OnceLock::new(),
        }
    }

    fn get_or_load(&self, gpu: &dyn GpuBackend) -> W4a16SharedFastV2Kernels {
        let key = std::ptr::from_ref(gpu) as *const () as usize;
        if let Some(&(cached_key, cached_val)) = self.primary.get() {
            if cached_key == key {
                return cached_val;
            }
        } else {
            let val = probe_w4a16_shared_fast_v2_kernels(gpu);
            let &(stored_key, stored_val) = self.primary.get_or_init(|| (key, val));
            if stored_key == key {
                return stored_val;
            }
            let map = self
                .fallback
                .get_or_init(|| parking_lot::Mutex::new(std::collections::HashMap::new()));
            return *map.lock().entry(key).or_insert(val);
        }
        let map = self
            .fallback
            .get_or_init(|| parking_lot::Mutex::new(std::collections::HashMap::new()));
        if let Some(val) = map.lock().get(&key).copied() {
            return val;
        }
        let val = probe_w4a16_shared_fast_v2_kernels(gpu);
        *map.lock().entry(key).or_insert(val)
    }
}

fn probe_w4a16_shared_fast_v2_kernels(gpu: &dyn GpuBackend) -> W4a16SharedFastV2Kernels {
    W4a16SharedFastV2Kernels {
        gu_sk4_m16: try_w4a16_kernel(gpu, SharedExpertGateUpTile::Sk4M16.symbol()),
        gu_sk4_m32: try_w4a16_kernel(gpu, SharedExpertGateUpTile::Sk4M32.symbol()),
        gu_sk2_m32: try_w4a16_kernel(gpu, SharedExpertGateUpTile::Sk2M32.symbol()),
        gu_sk2_m64: try_w4a16_kernel(gpu, SharedExpertGateUpTile::Sk2M64.symbol()),
        dn_m16: try_w4a16_kernel(gpu, SharedExpertDownTile::M16.symbol()),
        dn_m32: try_w4a16_kernel(gpu, SharedExpertDownTile::M32.symbol()),
        dn_m64: try_w4a16_kernel(gpu, SharedExpertDownTile::M64.symbol()),
        dn_m80: try_w4a16_kernel(gpu, SharedExpertDownTile::M80.symbol()),
        dn_m128: try_w4a16_kernel(gpu, SharedExpertDownTile::M128.symbol()),
    }
}

fn w4a16_shared_fast_v2_kernels(gpu: &dyn GpuBackend) -> W4a16SharedFastV2Kernels {
    static CACHE: W4a16SharedFastV2KernelCache = W4a16SharedFastV2KernelCache::new();
    CACHE.get_or_load(gpu)
}

/// Whether both transposed NVFP4 shared-expert fast-v2 kernels (`gateup_silu` +
/// `down`) for row count `m` (`2..=256`) are loaded in the active `w4a16` module.
#[must_use]
pub(crate) fn shared_expert_nvfp4_t_fast_v2_available(gpu: &dyn GpuBackend, m: usize) -> bool {
    w4a16_shared_fast_v2_kernels(gpu).resolve(m).is_some()
}

/// Execute the 1 shared expert (`gate_proj + up_proj + SiLU*mul` -> `act_scratch`,
/// then `down_proj` -> `out`) directly from BF16 activations and transposed NVFP4
/// weights (`[K/2, N]` packed + `[K/16, N]` flat E4M3 scales) via high-CTA
/// `w4a16_shared_*_t_*` kernels (40..80 CTAs gate+up, 80 CTAs down).
///
/// Zero dynamic allocations (100% CUDA-graph capture safe once warmed). Returns
/// `Ok(true)` if both kernels launched or `Ok(false)` fail-closed on any unmet
/// shape, alignment, weight, or symbol precondition.
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_shared_expert_nvfp4_t_fast_v2(
    gpu: &dyn GpuBackend,
    input: DevicePtr,
    gate_t: &QuantizedWeight,
    up_t: &QuantizedWeight,
    down_t: &QuantizedWeight,
    act_scratch: DevicePtr,
    out: DevicePtr,
    m: usize,
    inter: usize,
    hidden: usize,
    stream: u64,
) -> Result<bool> {
    let qw_ok = |w: &QuantizedWeight| {
        !w.weight.is_null()
            && !w.weight_scale.is_null()
            && w.weight.0.is_multiple_of(16)
            && w.weight_scale.0.is_multiple_of(16)
            && w.weight_scale_2.is_finite()
    };
    if !(2..=COALESCED_NVFP4_MAX_ROWS).contains(&m)
        || inter == 0
        || hidden == 0
        || !inter.is_multiple_of(16)
        || !hidden.is_multiple_of(32)
        || input.is_null()
        || act_scratch.is_null()
        || out.is_null()
        || !input.0.is_multiple_of(16)
        || !act_scratch.0.is_multiple_of(16)
        || !out.0.is_multiple_of(16)
        || !qw_ok(gate_t)
        || !qw_ok(up_t)
        || !qw_ok(down_t)
    {
        return Ok(false);
    }
    let (Ok(m_u32), Ok(inter_u32), Ok(hidden_u32)) = (
        u32::try_from(m),
        u32::try_from(inter),
        u32::try_from(hidden),
    ) else {
        return Ok(false);
    };
    let Some(resolved) = w4a16_shared_fast_v2_kernels(gpu).resolve(m) else {
        return Ok(false);
    };

    KernelLaunch::new(gpu, resolved.gu_kernel)
        .grid([
            div_ceil(inter_u32, resolved.gu_tile.co_n()),
            div_ceil(m_u32, resolved.gu_tile.tile_m()),
            1,
        ])
        .block([resolved.gu_tile.threads(), 1, 1])
        .arg_ptr(input)
        .arg_ptr(gate_t.weight)
        .arg_ptr(gate_t.weight_scale)
        .arg_f32(gate_t.weight_scale_2)
        .arg_ptr(up_t.weight)
        .arg_ptr(up_t.weight_scale)
        .arg_f32(up_t.weight_scale_2)
        .arg_ptr(act_scratch)
        .arg_u32(m_u32)
        .arg_u32(inter_u32)
        .arg_u32(hidden_u32)
        .launch(stream)?;

    KernelLaunch::new(gpu, resolved.dn_kernel)
        .grid([
            div_ceil(hidden_u32, resolved.dn_tile.co_n()),
            div_ceil(m_u32, resolved.dn_tile.tile_m()),
            1,
        ])
        .block([resolved.dn_tile.threads(), 1, 1])
        .arg_ptr(act_scratch)
        .arg_ptr(down_t.weight)
        .arg_ptr(down_t.weight_scale)
        .arg_f32(down_t.weight_scale_2)
        .arg_ptr(out)
        .arg_u32(m_u32)
        .arg_u32(hidden_u32)
        .arg_u32(inter_u32)
        .launch(stream)?;

    Ok(true)
}

pub(crate) const DIRECT_NVFP4_DENSE_ENV: &str = "ATLAS_QWEN4_BATCHED_VERIFY_DIRECT_NVFP4";

#[inline]
#[must_use]
pub(crate) fn parse_direct_nvfp4_dense(raw: Option<&str>) -> bool {
    raw.map(str::trim) == Some("1")
}

/// `ATLAS_QWEN4_BATCHED_VERIFY_DIRECT_NVFP4` (default OFF; `"1"` opts in):
/// execute batched-verify dense projections (`rows <= 256`) directly from
/// NVFP4 weights via `w4a16_gemm_pipe` / `w4a16_gemm_pipe_m128n128` when
/// neither native cuBLASLt NVFP4 (`nvfp4`) nor default-ON coalesced NVFP4
/// handles the call. Precedence inside [`fp8_gemm_ex`] is native cuBLASLt
/// NVFP4 -> default-ON coalesced NVFP4 -> direct NVFP4 -> FP8 cuBLASLt.
#[must_use]
pub(crate) fn direct_nvfp4_dense_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        parse_direct_nvfp4_dense(std::env::var(DIRECT_NVFP4_DENSE_ENV).ok().as_deref())
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DirectNvfp4GemmRoute {
    PipeM128N128,
    Pipe,
    Base,
}

impl DirectNvfp4GemmRoute {
    #[inline]
    #[must_use]
    pub(crate) const fn label(self) -> &'static str {
        match self {
            Self::PipeM128N128 => "w4a16_gemm_pipe_m128n128",
            Self::Pipe => "w4a16_gemm_pipe",
            Self::Base => "w4a16_gemm",
        }
    }
}

/// Select the fastest direct NVFP4 Tensor Core GEMM route for `[m, n, k]`:
/// `w4a16_gemm_pipe_m128n128` for wide `n >= 6144` above 64 rows (single weight
/// sweep with `>= 48` CTAs), `w4a16_gemm_pipe` for `m <= 64` or narrower `n`
/// (higher SM occupancy and L2 weight reuse), falling back to `w4a16_gemm`.
#[allow(clippy::too_many_arguments)]
#[inline]
#[must_use]
pub(crate) const fn select_direct_nvfp4_gemm_route(
    m: usize,
    n: usize,
    k: usize,
    act_aligned_16: bool,
    weight_aligned_16: bool,
    out_aligned_16: bool,
    has_pipe_m128n128: bool,
    has_pipe: bool,
    has_base: bool,
) -> Option<DirectNvfp4GemmRoute> {
    if m == 0 || n == 0 || k == 0 || !k.is_multiple_of(16) {
        return None;
    }
    if has_pipe_m128n128
        && m > 64
        && n >= 6144
        && n.is_multiple_of(128)
        && k.is_multiple_of(32)
        && act_aligned_16
        && weight_aligned_16
        && out_aligned_16
    {
        Some(DirectNvfp4GemmRoute::PipeM128N128)
    } else if has_pipe && k.is_multiple_of(32) && act_aligned_16 && weight_aligned_16 {
        Some(DirectNvfp4GemmRoute::Pipe)
    } else if has_base {
        Some(DirectNvfp4GemmRoute::Base)
    } else {
        None
    }
}

#[inline]
#[must_use]
pub(crate) const fn direct_nvfp4_batched_verify_shape_eligible(
    scope_active: bool,
    enabled: bool,
    m: usize,
    n: usize,
    k: usize,
    ldc: usize,
    qg: Option<(usize, usize)>,
) -> bool {
    scope_active
        && enabled
        && m > 0
        && m <= COALESCED_NVFP4_MAX_ROWS
        && n > 0
        && k > 0
        && k.is_multiple_of(16)
        && ldc == n
        && qg.is_none()
}

#[derive(Clone, Copy, Debug)]
struct W4a16DirectKernels {
    base: KernelHandle,
    pipe: KernelHandle,
    pipe_m128n128: KernelHandle,
}

struct W4a16DirectKernelCache {
    primary: OnceLock<(usize, W4a16DirectKernels)>,
    fallback: OnceLock<parking_lot::Mutex<std::collections::HashMap<usize, W4a16DirectKernels>>>,
}

impl W4a16DirectKernelCache {
    const fn new() -> Self {
        Self {
            primary: OnceLock::new(),
            fallback: OnceLock::new(),
        }
    }

    fn get_or_load(&self, gpu: &dyn GpuBackend) -> W4a16DirectKernels {
        let key = std::ptr::from_ref(gpu) as *const () as usize;
        if let Some(&(cached_key, cached_val)) = self.primary.get() {
            if cached_key == key {
                return cached_val;
            }
        } else {
            let val = probe_w4a16_direct_kernels(gpu);
            let &(stored_key, stored_val) = self.primary.get_or_init(|| (key, val));
            if stored_key == key {
                return stored_val;
            }
            let map = self
                .fallback
                .get_or_init(|| parking_lot::Mutex::new(std::collections::HashMap::new()));
            return *map.lock().entry(key).or_insert(val);
        }
        let map = self
            .fallback
            .get_or_init(|| parking_lot::Mutex::new(std::collections::HashMap::new()));
        if let Some(val) = map.lock().get(&key).copied() {
            return val;
        }
        let val = probe_w4a16_direct_kernels(gpu);
        *map.lock().entry(key).or_insert(val)
    }
}

fn probe_w4a16_direct_kernels(gpu: &dyn GpuBackend) -> W4a16DirectKernels {
    W4a16DirectKernels {
        base: crate::layers::try_kernel(gpu, "w4a16", "w4a16_gemm"),
        pipe: crate::layers::try_kernel(gpu, "w4a16", "w4a16_gemm_pipe"),
        pipe_m128n128: crate::layers::try_kernel(gpu, "w4a16", "w4a16_gemm_pipe_m128n128"),
    }
}

fn w4a16_direct_kernels(gpu: &dyn GpuBackend) -> W4a16DirectKernels {
    static CACHE: W4a16DirectKernelCache = W4a16DirectKernelCache::new();
    CACHE.get_or_load(gpu)
}

#[allow(clippy::too_many_arguments)]
fn try_direct_nvfp4_w4a16_gemm(
    gpu: &dyn GpuBackend,
    act: DevicePtr,
    weight: &QuantizedWeight,
    out: DevicePtr,
    m: usize,
    n: usize,
    k: usize,
    stream: u64,
) -> Result<bool> {
    let ks = w4a16_direct_kernels(gpu);
    let Some(route) = select_direct_nvfp4_gemm_route(
        m,
        n,
        k,
        act.0.is_multiple_of(16),
        weight.weight.0.is_multiple_of(16),
        out.0.is_multiple_of(16),
        ks.pipe_m128n128.0 != 0,
        ks.pipe.0 != 0,
        ks.base.0 != 0,
    ) else {
        return Ok(false);
    };
    match route {
        DirectNvfp4GemmRoute::PipeM128N128 => crate::layers::ops::w4a16_gemm_pipe_m128n128(
            gpu,
            ks.pipe_m128n128,
            act,
            weight,
            out,
            m as u32,
            n as u32,
            k as u32,
            stream,
        )?,
        DirectNvfp4GemmRoute::Pipe => crate::layers::ops::w4a16_gemm_pipe(
            gpu, ks.pipe, act, weight, out, m as u32, n as u32, k as u32, stream,
        )?,
        DirectNvfp4GemmRoute::Base => crate::layers::ops::w4a16_gemm(
            gpu, ks.base, act, weight, out, m as u32, n as u32, k as u32, stream,
        )?,
    }
    static DIRECT_LOGGED: std::sync::Once = std::sync::Once::new();
    DIRECT_LOGGED.call_once(|| {
        tracing::info!(
            selector = DIRECT_NVFP4_DENSE_ENV,
            route = route.label(),
            m,
            n,
            k,
            "QWEN4_BATCHED_VERIFY_DIRECT_NVFP4_ENGAGED"
        );
    });
    Ok(true)
}

/// FP8 twin of [`dequant_gemm_ex`]: `out[m, n] (ldc) = act[m, k] @ W[n, k]^T`
/// with e4m3 operands, per-output-channel weight and per-token activation
/// scales (cuBLASLt OUTER_VEC_32F), scale2 as alpha. Not bit-exact.
///
/// Inside a batched-verify dense TC scope (`rows in 4..=256`, `k % 32 == 0`,
/// contiguous `ldc == n`, unpermuted `qg.is_none()`), dispatches single-sweep
/// and intra-CTA Warp-Split-K K-coalesced NVFP4 `w4a16_gemm_coalesced*` by
/// default (`ATLAS_QWEN4_BATCHED_VERIFY_COALESCED_NVFP4 != 0`, zero FP8
/// weight-cache RAM, 1 launch/proj, CUDA-graph safe). When disabled (`=0`),
/// falls back to opt-in `ATLAS_QWEN4_BATCHED_VERIFY_DIRECT_NVFP4=1`
/// (`w4a16_gemm_pipe*`) or cached/tensor-scale FP8 cuBLASLt.
#[allow(clippy::too_many_arguments)]
pub(crate) fn fp8_gemm_ex(
    gpu: &dyn GpuBackend,
    act: DevicePtr,
    weight: &QuantizedWeight,
    out: DevicePtr,
    m: usize,
    n: usize,
    k: usize,
    ldc: usize,
    qg: Option<(usize, usize)>,
    stream: u64,
) -> Result<()> {
    anyhow::ensure!(
        m > 0 && n > 0 && k > 0 && k.is_multiple_of(16) && n.is_multiple_of(16),
        "{FP8_SELECTOR}: unsupported GEMM shape m={m} n={n} k={k}"
    );
    anyhow::ensure!(
        !weight.weight.is_null()
            && !weight.weight_scale.is_null()
            && weight.weight_scale_2.is_finite(),
        "{FP8_SELECTOR}: invalid NVFP4 weight"
    );
    if let Some((heads, head_dim)) = qg
        && heads > 0
    {
        anyhow::ensure!(
            heads * head_dim * 2 == n,
            "{FP8_SELECTOR}: gated-Q permutation shape mismatch"
        );
    }
    clear_bypassed_fp8_shared_act();
    let scope_active = crate::layers::qwen4_hyper::batched_verify_tc_dense_scope_active();
    let nvfp4_mode = nvfp4::mode()?;
    if nvfp4_mode.is_enabled()
        && (scope_active
            || crate::layers::qwen4_prefill_native::fp8_lookup(weight, n, k, qg)?.is_none())
    {
        if has_prequant_fp8_act(act, m, k, stream) {
            if nvfp4_mode.admits_call(m, n, k, ldc, qg, scope_active, nvfp4::verify_min_m()) {
                let (_, ws, a8, _) = fp8_scratch(gpu, n, m, k)?;
                if nvfp4::try_gemm_from_e4m3(
                    gpu,
                    a8,
                    ws.offset(12),
                    weight,
                    out,
                    m,
                    n,
                    k,
                    ldc,
                    qg,
                    scope_active,
                    stream,
                )? {
                    let _ = take_prequant_fp8_act(act, m, k, stream);
                    return Ok(());
                }
            }
        } else if nvfp4::try_gemm_bf16(
            gpu,
            act,
            weight,
            out,
            m,
            n,
            k,
            ldc,
            qg,
            scope_active,
            stream,
        )? {
            return Ok(());
        }
    }
    let capturing = spark_runtime::gpu_capture_region::active_capture_region().is_some();
    let prefer_fp8_narrow = coalesced_nvfp4_prefers_fp8_cached(m, n)
        && (!capturing
            || (fp8_tensor_scales()
                && has_cached_weight_tensor(weight, n, k, qg)
                && fp8_scratch_ready_for_capture(n, m.max(resolve_fp8_pad_m(m, n)), k)));
    if !prefer_fp8_narrow
        && coalesced_nvfp4_batched_verify_shape_eligible(
            scope_active,
            coalesced_nvfp4_dense_enabled(),
            m,
            n,
            k,
            ldc,
            qg,
        )
        && try_coalesced_nvfp4_w4a16_gemm(gpu, act, weight, out, m, n, k, stream)?
    {
        static COALESCED_LOGGED: std::sync::Once = std::sync::Once::new();
        COALESCED_LOGGED.call_once(|| {
            let kernel = coalesced_nvfp4_dispatch_symbol(m, n, k);
            tracing::info!(
                selector = COALESCED_NVFP4_DENSE_ENV,
                kernel,
                m,
                n,
                k,
                "QWEN4_BATCHED_VERIFY_COALESCED_NVFP4_ENGAGED"
            );
        });
        return Ok(());
    }
    if direct_nvfp4_batched_verify_shape_eligible(
        scope_active,
        direct_nvfp4_dense_enabled(),
        m,
        n,
        k,
        ldc,
        qg,
    ) && try_direct_nvfp4_w4a16_gemm(gpu, act, weight, out, m, n, k, stream)?
    {
        return Ok(());
    }
    let (w8, ws, a8, as_) = fp8_scratch(gpu, n, m, k)?;
    if fp8_tensor_scales() {
        let tk = fp8_tensor_kernels(gpu)?;
        let prequant = take_prequant_fp8_act(act, m, k, stream);
        if !prequant {
            fp8_act_tensor_or_fused(gpu, &tk, act, m, k, ws, a8, stream)?;
        }
        let m_gemm = resolve_fp8_pad_m(m, n);
        if m_gemm > m {
            gpu.memset_async(a8.offset(m * k), 0, (m_gemm - m) * k, stream)?;
        }
        fp8_weight_tensor_gemm(
            gpu, &tk, weight, out, m_gemm, n, k, ldc, qg, w8, ws, a8, !prequant, stream,
        )?;
        static LOGGED_T: std::sync::Once = std::sync::Once::new();
        LOGGED_T.call_once(|| {
            tracing::info!("FAST_PROJ_FP8_ENGAGED selector={FP8_SELECTOR} m={m} m_gemm={m_gemm} n={n} k={k} scales=tensor non_bit_exact")
        });
        return Ok(());
    }
    let (wk, ak) = fp8_kernels(gpu)?;
    KernelLaunch::new(gpu, wk)
        .grid([n as u32, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(weight.weight)
        .arg_ptr(weight.weight_scale)
        .arg_ptr(w8)
        .arg_ptr(ws)
        .arg_u32(k as u32)
        .arg_u32(qg.map_or(0, |(h, _)| h) as u32)
        .arg_u32(qg.map_or(0, |(_, d)| d) as u32)
        .launch(stream)?;
    KernelLaunch::new(gpu, ak)
        .grid([m as u32, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(act)
        .arg_ptr(a8)
        .arg_ptr(as_)
        .arg_u32(k as u32)
        .arg_u32(k as u32)
        .launch(stream)?;
    let split_cap = fp8_gemm_m_split_cap();
    let slices = fp8_gemm_m_split_slices(m, n, k, split_cap);
    let num_slices = slices.len();
    for slice in slices {
        spark_runtime::cublaslt::fp8_gemm_outer_scaled(
            a8.offset(slice.a_byte_offset).0,
            as_.offset(slice.outer_a_scale_byte_offset()).0,
            w8.0,
            ws.0,
            out.offset(slice.out_byte_offset(ldc)).0,
            slice.sub_m as u32,
            n as u32,
            k as u32,
            ldc as u32,
            weight.weight_scale_2,
            stream,
        )?;
    }
    if num_slices > 1 {
        log_fp8_m_split_engaged(m, n, k, ldc, split_cap, num_slices);
    }
    static LOGGED: std::sync::Once = std::sync::Once::new();
    LOGGED.call_once(|| {
        tracing::info!("FAST_PROJ_FP8_ENGAGED selector={FP8_SELECTOR} m={m} n={n} k={k} scales=outer_vec_32f non_bit_exact")
    });
    Ok(())
}

pub(crate) const FP8_ACT_ROWS_V2_SELECTOR: &str = "ATLAS_QWEN4_PREFILL_FP8_ACT_ROWS_V2";

/// `ATLAS_QWEN4_PREFILL_FP8_ACT_ROWS_V2` (default on; `=0` disables): launch the 4-row-per-CTA
/// `qwen4_fp8_act_rows_tensor_v2` kernel so K=2560/4096/6144 run with 0 idle warps and L2 hints.
pub(crate) fn fp8_act_rows_v2_selected() -> bool {
    static S: OnceLock<bool> = OnceLock::new();
    *S.get_or_init(|| std::env::var(FP8_ACT_ROWS_V2_SELECTOR).ok().as_deref() != Some("0"))
}

fn fp8_act_rows_v2_kernel(gpu: &dyn GpuBackend) -> Result<KernelHandle> {
    static K: OnceLock<Result<KernelHandle, String>> = OnceLock::new();
    K.get_or_init(|| {
        gpu.kernel("qwen4_fp8_proj", "qwen4_fp8_act_rows_tensor_v2")
            .map_err(|e| e.to_string())
    })
    .clone()
    .map_err(|e| anyhow::anyhow!("{FP8_ACT_ROWS_V2_SELECTOR}: kernel unavailable: {e}"))
}

fn fp8_act_and_ba_gates_kernel(gpu: &dyn GpuBackend) -> Result<KernelHandle> {
    static K: OnceLock<Result<KernelHandle, String>> = OnceLock::new();
    K.get_or_init(|| {
        gpu.kernel("qwen4_fp8_proj", "qwen4_fp8_act_and_ba_gates_mma")
            .map_err(|e| e.to_string())
    })
    .clone()
    .map_err(|e| anyhow::anyhow!("{FP8_SELECTOR}: qwen4_fp8_act_and_ba_gates_mma unavailable: {e}"))
}

/// Fuses `act [m, 2560]` BF16 -> `a8` E4M3 quantization with the `[m, 2560] x [96, 2560]^T`
/// `ba_gates` MMA + `A_log`/`dt_bias` transform into `gates_out`, then runs the FP8 tensor-scale
/// GEMM for `qkvz_weight` into `qkvz_out`. Reads `act` (`83.89 MB` at `m=16384`) once instead of twice.
#[allow(clippy::too_many_arguments)]
pub(crate) fn fp8_qkvz_and_ba_gates_fused(
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
) -> Result<()> {
    anyhow::ensure!(
        m > 0 && n_qkvz > 0 && k == 2560 && n_qkvz.is_multiple_of(16),
        "{FP8_SELECTOR}: unsupported fused QKVZ+BA shape m={m} n={n_qkvz} k={k}"
    );
    clear_bypassed_fp8_shared_act();
    if crate::layers::qwen4_prefill_native::fp8_lookup(qkvz_weight, n_qkvz, k, None)?.is_none()
        && nvfp4::try_qkvz_and_ba_gates_fused(
            gpu,
            act,
            qkvz_weight,
            qkvz_out,
            ba_weight,
            a_log,
            dt_bias,
            gates_out,
            m,
            n_qkvz,
            k,
            planar,
            stream,
        )?
    {
        return Ok(());
    }
    let tk = fp8_tensor_kernels(gpu)?;
    let fused_kh = fp8_act_and_ba_gates_kernel(gpu)?;
    let (w8, ws, a8, _) = fp8_scratch(gpu, n_qkvz, m, k)?;
    let a_scale = ws.offset(12);
    let a_amax = if crate::layers::qwen4_fp8_act_amax::take_matching(act, m, k, stream) {
        crate::layers::qwen4_fp8_act_amax::slot(gpu)?
    } else {
        let slot = ws.offset(4);
        gpu.memset_async(slot, 0, 4, stream)?;
        KernelLaunch::new(gpu, tk.act_amax)
            .grid([256, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(act)
            .arg_u64((m * k) as u64)
            .arg_ptr(slot)
            .launch(stream)?;
        slot
    };
    let m_u32 = u32::try_from(m)?;
    KernelLaunch::new(gpu, fused_kh)
        .grid([div_ceil(m_u32, 64), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(act)
        .arg_ptr(a_amax)
        .arg_ptr(a8)
        .arg_ptr(a_scale)
        .arg_ptr(ba_weight)
        .arg_ptr(a_log)
        .arg_ptr(dt_bias)
        .arg_ptr(gates_out)
        .arg_u32(m_u32)
        .arg_u32(3)
        .arg_u32(u32::from(planar))
        .launch(stream)?;
    let m_gemm = resolve_fp8_pad_m(m, n_qkvz);
    if m_gemm > m {
        gpu.memset_async(a8.offset(m * k), 0, (m_gemm - m) * k, stream)?;
    }
    fp8_weight_tensor_gemm(
        gpu,
        &tk,
        qkvz_weight,
        qkvz_out,
        m_gemm,
        n_qkvz,
        k,
        n_qkvz,
        None,
        w8,
        ws,
        a8,
        true,
        stream,
    )
}

#[allow(clippy::too_many_arguments)]
fn launch_fp8_act_rows_tensor(
    gpu: &dyn GpuBackend,
    tk: &Fp8TensorKernels,
    act: DevicePtr,
    a_amax: DevicePtr,
    a8: DevicePtr,
    a_scale: DevicePtr,
    m: usize,
    k: usize,
    stream: u64,
) -> Result<()> {
    if fp8_act_rows_v2_selected() {
        let kh = fp8_act_rows_v2_kernel(gpu)?;
        KernelLaunch::new(gpu, kh)
            .grid([div_ceil(m as u32, 4), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(act)
            .arg_ptr(a_amax)
            .arg_ptr(a8)
            .arg_ptr(a_scale)
            .arg_u32(k as u32)
            .arg_u32(k as u32)
            .arg_u32(m as u32)
            .launch(stream)?;
        static LOGGED_V2: std::sync::Once = std::sync::Once::new();
        LOGGED_V2.call_once(|| {
            tracing::info!(
                "FAST_PROJ_FP8_ACT_ROWS_V2_ENGAGED selector={FP8_ACT_ROWS_V2_SELECTOR} m={m} k={k} bit_identical_to_v1"
            );
        });
        return Ok(());
    }
    KernelLaunch::new(gpu, tk.act_rows)
        .grid([m as u32, 1, 1])
        .block([if k == 2560 { 320 } else { 256 }, 1, 1])
        .arg_ptr(act)
        .arg_ptr(a_amax)
        .arg_ptr(a8)
        .arg_ptr(a_scale)
        .arg_u32(k as u32)
        .arg_u32(k as u32)
        .launch(stream)
}

/// Tensor-scale e4m3 activation: absmax into `ws + 4` (zeroed here),
/// rows into `a8`, scale into `ws + 12`.
#[allow(clippy::too_many_arguments)]
fn fp8_act_tensor(
    gpu: &dyn GpuBackend,
    tk: &Fp8TensorKernels,
    act: DevicePtr,
    m: usize,
    k: usize,
    ws: DevicePtr,
    a8: DevicePtr,
    stream: u64,
) -> Result<()> {
    let (a_amax, a_scale) = (ws.offset(4), ws.offset(12));
    gpu.memset_async(a_amax, 0, 4, stream)?;
    KernelLaunch::new(gpu, tk.act_amax)
        .grid([256, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(act)
        .arg_u64((m * k) as u64)
        .arg_ptr(a_amax)
        .launch(stream)?;
    launch_fp8_act_rows_tensor(gpu, tk, act, a_amax, a8, a_scale, m, k, stream)
}

/// [`fp8_act_tensor`], or, when the kernel that wrote `act` already reduced
/// its absmax (`qwen4_fp8_act_amax`), only the quantizing pass reading that
/// absmax: the same e4m3 rows and scale without the absmax read.
#[allow(clippy::too_many_arguments)]
fn fp8_act_tensor_or_fused(
    gpu: &dyn GpuBackend,
    tk: &Fp8TensorKernels,
    act: DevicePtr,
    m: usize,
    k: usize,
    ws: DevicePtr,
    a8: DevicePtr,
    stream: u64,
) -> Result<()> {
    if !crate::layers::qwen4_fp8_act_amax::take_matching(act, m, k, stream) {
        return fp8_act_tensor(gpu, tk, act, m, k, ws, a8, stream);
    }
    let a_amax = crate::layers::qwen4_fp8_act_amax::slot(gpu)?;
    launch_fp8_act_rows_tensor(gpu, tk, act, a_amax, a8, ws.offset(12), m, k, stream)
}

static WEIGHT_AMAX_CACHE: Mutex<Option<std::collections::HashMap<(u64, usize, usize), DevicePtr>>> =
    Mutex::new(None);

fn cached_weight_amax(
    gpu: &dyn GpuBackend,
    tk: &Fp8TensorKernels,
    weight: &QuantizedWeight,
    n: usize,
    k: usize,
    stream: u64,
) -> Result<DevicePtr> {
    let key = (weight.weight_scale.0, n, k);
    let mut guard = WEIGHT_AMAX_CACHE.lock().unwrap();
    let map = guard.get_or_insert_with(std::collections::HashMap::new);
    if let Some(&slot) = map.get(&key) {
        return Ok(slot);
    }
    anyhow::ensure!(
        spark_runtime::gpu_capture_region::active_capture_region().is_none(),
        "{FP8_SELECTOR}: refusing lazy weight_amax allocation during CUDA graph capture"
    );
    let slot = gpu
        .alloc(4)
        .with_context(|| format!("{FP8_SELECTOR}: alloc 4 B for cached weight_amax"))?;
    gpu.memset_async(slot, 0, 4, stream)?;
    KernelLaunch::new(gpu, tk.weight_amax)
        .grid([256, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(weight.weight_scale)
        .arg_u64((n * k / 16) as u64)
        .arg_ptr(slot)
        .launch(stream)?;
    map.insert(key, slot);
    Ok(slot)
}

/// NVFP4 weight -> tensor-scale e4m3 rows in `w8` and the scale in `w_scale`;
/// `w_amax` is cached once per static weight tensor.
#[allow(clippy::too_many_arguments)]
fn weight_tensor_into(
    gpu: &dyn GpuBackend,
    tk: &Fp8TensorKernels,
    weight: &QuantizedWeight,
    n: usize,
    k: usize,
    qg: Option<(usize, usize)>,
    w8: DevicePtr,
    w_scale: DevicePtr,
    stream: u64,
) -> Result<()> {
    let w_amax = cached_weight_amax(gpu, tk, weight, n, k, stream)?;
    KernelLaunch::new(gpu, tk.weight_rows)
        .grid([div_ceil(n as u32, 8), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(weight.weight)
        .arg_ptr(weight.weight_scale)
        .arg_ptr(w_amax)
        .arg_ptr(w8)
        .arg_ptr(w_scale)
        .arg_u32(k as u32)
        .arg_u32(qg.map_or(0, |(h, _)| h) as u32)
        .arg_u32(qg.map_or(0, |(_, d)| d) as u32)
        .launch(stream)?;
    Ok(())
}

pub(crate) const FP8_WEIGHT_CACHE_SELECTOR: &str = "ATLAS_QWEN4_PREFILL_FAST_PROJ_FP8_WEIGHT_CACHE";
pub const WEIGHT_CACHE_ALIAS: &str = "ATLAS_QWEN4_WEIGHT_CACHE";
pub const FAST_PROJ_CACHE_ALIAS: &str = "ATLAS_QWEN4_FAST_PROJ_CACHE";

/// `ATLAS_QWEN4_PREFILL_FAST_PROJ_FP8_WEIGHT_CACHE=1` (default off, tensor
/// scales): convert each NVFP4 weight to tensor-scale e4m3 once, on its first
/// prefill use, and keep it resident (~2.6 GB on Flash-Next: qkvz 1.5, ssm out
/// 0.57, attention q 0.38, o 0.19). Same kernels and inputs as the per-call
/// conversion, so the e4m3 bytes, the scale and the GEMM output are identical;
/// only the per-call re-conversion goes away (-23.8 ms per 8192-row pass,
/// bench-prefill/fn_gdn_fi runs/g8a.out). A weight is left uncached (per-call
/// path, logged once) when caching it would leave less than
/// `..._WEIGHT_CACHE_RESERVE_GB` (default 4) of free memory — or, when that is
/// lower, less than the declared `MemAvailable` floor plus its margin
/// (`cache_floor::CacheFillPolicy`, measured on the host's real `MemAvailable`).
/// Keyed by the weight's device pointers, scale2 and shape: weights are
/// immutable while their model is loaded; a model reload into the same
/// addresses with the same shapes would be served stale rows.
pub(crate) fn fp8_weight_cache_selected() -> bool {
    static S: OnceLock<bool> = OnceLock::new();
    *S.get_or_init(|| {
        std::env::var(FP8_WEIGHT_CACHE_SELECTOR).ok().as_deref() == Some("1")
            || std::env::var(WEIGHT_CACHE_ALIAS).ok().as_deref() == Some("1")
            || std::env::var(FAST_PROJ_CACHE_ALIAS).ok().as_deref() == Some("1")
    })
}

fn fp8_weight_cache_reserve_bytes() -> usize {
    static R: OnceLock<usize> = OnceLock::new();
    *R.get_or_init(|| {
        std::env::var("ATLAS_QWEN4_PREFILL_FAST_PROJ_FP8_WEIGHT_CACHE_RESERVE_GB")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .unwrap_or(4)
            << 30
    })
}

/// The fill policy, resolved once; a malformed floor or margin selector is
/// reported on the first cache attempt instead of silently defaulting.
pub(crate) fn fp8_weight_cache_policy() -> Result<cache_floor::CacheFillPolicy> {
    static P: OnceLock<std::result::Result<cache_floor::CacheFillPolicy, String>> = OnceLock::new();
    P.get_or_init(|| {
        cache_floor::policy_from_env(fp8_weight_cache_reserve_bytes()).map_err(|e| format!("{e:#}"))
    })
    .clone()
    .map_err(|e| anyhow::anyhow!("{FP8_WEIGHT_CACHE_SELECTOR}: {e}"))
}

/// Ledger description of the cache's fill policy; `None` when the cache is off.
pub fn fp8_weight_cache_ledger_summary() -> Option<String> {
    if !fp8_weight_cache_selected() {
        return None;
    }
    if atlas_core::device::active_device_is_integrated() == Some(false) {
        let total_vram =
            atlas_core::device::active_device_props().map_or(0, |props| props.total_mem_bytes);
        return Some(cache_floor::discrete_vram_summary(total_vram));
    }
    Some(match fp8_weight_cache_policy() {
        Ok(policy) => policy.summary(),
        Err(error) => format!("policy error: {error}"),
    })
}

type WeightCacheKey = (u64, u64, u32, usize, usize, usize, usize);
struct WeightCache {
    entries: std::collections::HashMap<WeightCacheKey, Option<(DevicePtr, DevicePtr)>>,
    bytes: usize,
    /// Entries holding a resident weight (`entries` also holds the refused ones).
    cached: usize,
}
static WEIGHT_CACHE: Mutex<Option<WeightCache>> = Mutex::new(None);
static ROUTER_WEIGHT_CACHE: Mutex<Option<WeightCache>> = Mutex::new(None);

fn weight_cache_key(
    weight: &QuantizedWeight,
    n: usize,
    k: usize,
    qg: Option<(usize, usize)>,
) -> WeightCacheKey {
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

/// True when `weight` is already resident in the FP8 tensor-scale weight cache
/// (`WEIGHT_CACHE` / `ROUTER_WEIGHT_CACHE`).
///
/// Checks only the FP8 weight cache (never the NVFP4 swizzled cache), so
/// callers that track FP8 readiness separately or fall back from NVFP4 to FP8
/// never mistake NVFP4 residency for FP8 residency inside CUDA graph capture.
pub(crate) fn has_cached_weight_tensor(
    weight: &QuantizedWeight,
    n: usize,
    k: usize,
    qg: Option<(usize, usize)>,
) -> bool {
    if !fp8_weight_cache_selected() {
        return false;
    }
    let key = weight_cache_key(weight, n, k, qg);
    let cache_mutex = if qg == Some((0, 0)) {
        &ROUTER_WEIGHT_CACHE
    } else {
        &WEIGHT_CACHE
    };
    let guard = cache_mutex.lock().unwrap();
    guard
        .as_ref()
        .and_then(|c| c.entries.get(&key))
        .is_some_and(Option::is_some)
}

/// The cached (e4m3 rows, scale) for `weight`, converting it on first use;
/// `None` when the cache is off or the weight stays uncached.
fn cached_weight_tensor(
    gpu: &dyn GpuBackend,
    tk: &Fp8TensorKernels,
    weight: &QuantizedWeight,
    n: usize,
    k: usize,
    qg: Option<(usize, usize)>,
    stream: u64,
) -> Result<Option<(DevicePtr, DevicePtr)>> {
    if nvfp4::mode_or_off().admits_weight(n, k, qg) {
        return Ok(None);
    }
    if !fp8_weight_cache_selected() {
        return Ok(None);
    }
    let is_router = qg == Some((0, 0));
    let key = weight_cache_key(weight, n, k, qg);
    let cache_mutex = if is_router {
        &ROUTER_WEIGHT_CACHE
    } else {
        &WEIGHT_CACHE
    };
    let mut guard = cache_mutex.lock().unwrap();
    let cache = guard.get_or_insert_with(|| WeightCache {
        entries: std::collections::HashMap::new(),
        bytes: 0,
        cached: 0,
    });
    let is_discrete = !gpu.is_unified_memory()
        || atlas_core::device::active_device_is_integrated() == Some(false);
    let discrete_total_vram = if is_discrete {
        gpu.total_memory()
            .ok()
            .or_else(|| {
                atlas_core::device::active_device_props().map(|props| props.total_mem_bytes)
            })
            .unwrap_or(0)
    } else {
        0
    };
    if let Some(hit) = cache.entries.get(&key) {
        // The first reuse means the first prefill pass has met every projection once:
        // report the fill outcome (the ENGAGED cadence stops at weights=96 of 120 on
        // Flash-Next, so it cannot show a complete fill).
        if !is_router {
            static FILLED: std::sync::Once = std::sync::Once::new();
            FILLED.call_once(|| {
                let uncached = cache.entries.len() - cache.cached;
                let stop_line = if is_discrete {
                    cache_floor::discrete_vram_stop_line_bytes(discrete_total_vram)
                } else {
                    fp8_weight_cache_policy().map_or(0, |p| p.stop_line_bytes())
                };
                tracing::info!(
                    "{}",
                    cache_floor::fill_summary(cache.cached, uncached, cache.bytes, stop_line)
                );
            });
        }
        return Ok(*hit);
    }
    if spark_runtime::gpu_capture_region::active_capture_region().is_some() {
        return Ok(None);
    }
    // Rows, then [scale f32, absmax bits] on a 256-byte boundary.
    let rows_bytes = (n * k).next_multiple_of(256);
    let bytes = rows_bytes + 256;
    let policy = fp8_weight_cache_policy()?;
    let topology = if is_discrete {
        cache_floor::CacheMemoryTopology::Discrete {
            total_vram_bytes: discrete_total_vram,
        }
    } else {
        cache_floor::CacheMemoryTopology::Unified
    };
    match cache_floor::evaluate_cache_admission(
        topology,
        &policy,
        bytes,
        spark_runtime::startup_ledger::mem_available_bytes(),
        cfg!(target_os = "linux"),
        || gpu.free_memory(),
    )? {
        cache_floor::CacheAdmitDecision::Admit => {}
        cache_floor::CacheAdmitDecision::UnmeasurableHost => {
            static UNMEASURABLE: std::sync::Once = std::sync::Once::new();
            UNMEASURABLE.call_once(|| {
                tracing::warn!(
                    "{FP8_WEIGHT_CACHE_SELECTOR}: /proc/meminfo has no readable MemAvailable; \
                     skipping optional FP8 weight-cache allocation"
                );
            });
            return Ok(None);
        }
        cache_floor::CacheAdmitDecision::BelowStopLine {
            available_bytes,
            stop_line_bytes,
            is_discrete_vram,
        } => {
            static SKIPPED: std::sync::Once = std::sync::Once::new();
            SKIPPED.call_once(|| {
                if is_discrete_vram {
                    tracing::warn!(
                        "{FP8_WEIGHT_CACHE_SELECTOR}: free VRAM {} MiB would drop under the {} MiB discrete stop line (min(2048 MiB, total VRAM {} MiB / 8)); weights from here stay per-call (cached {} MiB)",
                        available_bytes >> 20,
                        stop_line_bytes >> 20,
                        discrete_total_vram >> 20,
                        cache.bytes >> 20
                    )
                } else {
                    tracing::warn!(
                        "{FP8_WEIGHT_CACHE_SELECTOR}: MemAvailable {} MiB would drop under the {} MiB stop line (reserve {} MiB, floor {} MiB + margin {} MiB); weights from here stay per-call (cached {} MiB)",
                        available_bytes >> 20,
                        stop_line_bytes >> 20,
                        policy.legacy_reserve_bytes >> 20,
                        policy.floor_bytes >> 20,
                        policy.margin_bytes >> 20,
                        cache.bytes >> 20
                    )
                }
            });
            cache.entries.insert(key, None);
            return Ok(None);
        }
    }
    let base = gpu
        .alloc(bytes)
        .with_context(|| format!("{FP8_WEIGHT_CACHE_SELECTOR}: alloc {bytes} B"))?;
    let (w8, w_scale) = (base, base.offset(rows_bytes));
    weight_tensor_into(gpu, tk, weight, n, k, qg, w8, w_scale, stream)?;
    cache.entries.insert(key, Some((w8, w_scale)));
    cache.bytes += bytes;
    cache.cached += 1;
    if !is_router {
        let count = cache.cached;
        if count == 1 || count.is_multiple_of(32) {
            tracing::info!(
                "FAST_PROJ_FP8_WEIGHT_CACHE_ENGAGED selector={FP8_WEIGHT_CACHE_SELECTOR} weights={count} bytes={} resident_mib={} bit_identical_to_per_call fence=memset",
                cache.bytes,
                cache.bytes >> 20
            );
        }
    }
    Ok(Some((w8, w_scale)))
}

/// `ATLAS_QWEN4_FP8_GEMM_M_SPLIT` (default `0` disabled; set e.g. `8192` to opt in
/// when `m > 8192 && n >= 12288`): splits large-M wide-N FP8 GEMMs along `M` into
/// 128-row-aligned chunks. Kept opt-in on GB10 because unsplit 16K GEMMs are faster
/// and avoid `sm89_xmma` fallback on non-power-of-2 split remainders.
pub(crate) const FP8_GEMM_M_SPLIT_SELECTOR: &str = "ATLAS_QWEN4_FP8_GEMM_M_SPLIT";
/// Default M-split chunk size (`0` = disabled; opt in with e.g. `8192`).
const DEFAULT_FP8_GEMM_M_SPLIT: usize = 0;
/// Minimum `m` (strictly greater) above which FP8 GEMMs split along `M`.
const FP8_GEMM_M_SPLIT_MIN_M: usize = 8_192;
/// Minimum `n` for M-splitting (covers attn Q=12288 and SSM QKVZ=16384; keeps N=2560 out_proj unsplit).
const FP8_GEMM_M_SPLIT_MIN_N: usize = 12_288;

/// One contiguous `[row_start..row_start + sub_m)` slice of an M-split FP8 GEMM.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Fp8GemmMSlice {
    pub row_start: usize,
    pub sub_m: usize,
    /// Byte offset into row-major FP8 E4M3 `a8` (`[m, k]`, 1 B/elem).
    pub a_byte_offset: usize,
}

impl Fp8GemmMSlice {
    /// Byte offset into row-major BF16 `out` (`[m, n]` with row stride `ldc`, 2 B/elem).
    #[inline]
    #[must_use]
    pub(crate) const fn out_byte_offset(self, ldc: usize) -> usize {
        self.row_start * ldc * 2
    }

    /// Byte offset into per-row `f32` activation scales (`4 B/row`) for `OUTER_VEC_32F`.
    /// Per-tensor `SCALAR_32F` callers reuse the single scalar `a_scale` pointer unchanged.
    #[inline]
    #[must_use]
    pub(crate) const fn outer_a_scale_byte_offset(self) -> usize {
        self.row_start * 4
    }
}

fn parse_fp8_gemm_m_split(raw: Option<&str>) -> usize {
    match raw.map(str::trim) {
        None | Some("") => DEFAULT_FP8_GEMM_M_SPLIT,
        Some("0") => 0,
        Some(value) => value.parse::<usize>().unwrap_or(DEFAULT_FP8_GEMM_M_SPLIT),
    }
}

fn fp8_gemm_m_split_cap() -> usize {
    static CAP: OnceLock<usize> = OnceLock::new();
    *CAP.get_or_init(|| {
        parse_fp8_gemm_m_split(std::env::var(FP8_GEMM_M_SPLIT_SELECTOR).ok().as_deref())
    })
}

/// Zero-allocation iterator yielding contiguous M-split slices of an FP8 GEMM.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Fp8GemmMSliceIter {
    m: usize,
    k: usize,
    chunk: usize,
    row_start: usize,
}

impl Fp8GemmMSliceIter {
    #[inline]
    #[must_use]
    pub(crate) const fn len(&self) -> usize {
        if self.row_start >= self.m {
            0
        } else {
            (self.m - self.row_start).div_ceil(self.chunk)
        }
    }

    #[inline]
    #[must_use]
    #[allow(dead_code)]
    pub(crate) const fn is_empty(&self) -> bool {
        self.row_start >= self.m
    }
}

impl Iterator for Fp8GemmMSliceIter {
    type Item = Fp8GemmMSlice;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.row_start >= self.m {
            return None;
        }
        let row_start = self.row_start;
        let sub_m = (self.m - row_start).min(self.chunk);
        self.row_start = row_start + sub_m;
        Some(Fp8GemmMSlice {
            row_start,
            sub_m,
            a_byte_offset: row_start * self.k,
        })
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let rem = self.len();
        (rem, Some(rem))
    }
}

impl ExactSizeIterator for Fp8GemmMSliceIter {}

/// Partition `m` rows into contiguous 128-row-aligned slices of at most `split_cap` rows when
/// `split_cap > 0 && m > 8192 && n >= 12288` (`FP8_GEMM_M_SPLIT_MIN_N`). Preserves output layout
/// bit-for-bit with zero heap allocation.
#[inline]
#[must_use]
pub(crate) const fn fp8_gemm_m_split_slices(
    m: usize,
    n: usize,
    k: usize,
    split_cap: usize,
) -> Fp8GemmMSliceIter {
    if m == 0 {
        return Fp8GemmMSliceIter {
            m: 0,
            k,
            chunk: 1,
            row_start: 0,
        };
    }
    let chunk = if split_cap == 0 || m <= FP8_GEMM_M_SPLIT_MIN_M || n < FP8_GEMM_M_SPLIT_MIN_N {
        m
    } else {
        let aligned = split_cap & !127;
        let aligned = if aligned < 128 { 128 } else { aligned };
        if m <= aligned { m } else { aligned }
    };
    Fp8GemmMSliceIter {
        m,
        k,
        chunk,
        row_start: 0,
    }
}

fn log_fp8_m_split_engaged(
    m: usize,
    n: usize,
    k: usize,
    ldc: usize,
    split_cap: usize,
    slices: usize,
) {
    static LOGGED: std::sync::Once = std::sync::Once::new();
    LOGGED.call_once(|| {
        tracing::info!(
            "FAST_PROJ_FP8_M_SPLIT_ENGAGED selector={FP8_GEMM_M_SPLIT_SELECTOR} m={m} n={n} k={k} ldc={ldc} split_cap={split_cap} slices={slices}"
        );
    });
}

/// Tensor-scale e4m3 weight (cached, or per call: weight absmax from
/// `cached_weight_amax`, rows into `w8`, weight scale into `ws + 8`) and the
/// cuBLASLt GEMM against the e4m3 activation already in `a8` (scale at
/// `ws + 12`; `ws` holds the optional 4-byte `fence_scale` ordering fence).
#[allow(clippy::too_many_arguments)]
pub(crate) fn fp8_weight_tensor_gemm(
    gpu: &dyn GpuBackend,
    tk: &Fp8TensorKernels,
    weight: &QuantizedWeight,
    out: DevicePtr,
    m: usize,
    n: usize,
    k: usize,
    ldc: usize,
    qg: Option<(usize, usize)>,
    w8: DevicePtr,
    ws: DevicePtr,
    a8: DevicePtr,
    fence_scale: bool,
    stream: u64,
) -> Result<()> {
    let split_cap = fp8_gemm_m_split_cap();
    let slices = fp8_gemm_m_split_slices(m, n, k, split_cap);
    let num_slices = slices.len();
    let scope_active = crate::layers::qwen4_hyper::batched_verify_tc_dense_scope_active();
    if let Some(act_slice) = take_bypassed_fp8_shared_act_slice(a8, m, k, stream) {
        if crate::layers::qwen4_prefill_native::fp8_lookup(weight, n, k, qg)?.is_none()
            && nvfp4::try_gemm_bf16(
                gpu,
                act_slice,
                weight,
                out,
                m,
                n,
                k,
                ldc,
                qg,
                scope_active,
                stream,
            )?
        {
            return Ok(());
        }
        // Fallback if NVFP4 declined (e.g. cold capture or native override): materialize
        // valid E4M3 rows into `a8` + `ws + 12` before entering the FP8 GEMM below.
        fp8_act_tensor(gpu, tk, act_slice, m, k, ws, a8, stream)?;
    }
    // ATLAS_QWEN4_PREFILL_NATIVE=1: E4M3 rows built from the checkpoint MXFP8.
    if let Some(native) = crate::layers::qwen4_prefill_native::fp8_lookup(weight, n, k, qg)? {
        if fence_scale {
            // Same ordering fence as a cached weight (see below).
            gpu.memset_async(ws, 0, 4, stream)?;
        }
        if num_slices <= 1 {
            return crate::layers::qwen4_prefill_native::native_fp8_gemm(
                gpu, &native, out, m, n, k, ldc, ws, a8, stream,
            );
        }
        let a_scale = ws.offset(12);
        for slice in slices {
            spark_runtime::cublaslt::fp8_gemm_device_scaled(
                a8.offset(slice.a_byte_offset).0,
                a_scale.0,
                native.w8.0,
                native.scale.0,
                out.offset(slice.out_byte_offset(ldc)).0,
                slice.sub_m as u32,
                n as u32,
                k as u32,
                ldc as u32,
                1.0,
                false,
                stream,
            )?;
        }
        crate::layers::qwen4_fp8_proj_dump::maybe_dump(
            gpu,
            a8,
            a_scale,
            native.w8,
            native.scale,
            1.0,
            out,
            (m, n, k, ldc),
            stream,
        )?;
        crate::layers::qwen4_prefill_native::note_call(
            native.module,
            "fp8_e4m3_mxfp8_pow2_tensor",
            m,
            n,
            k,
            native.scale_value,
        );
        log_fp8_m_split_engaged(m, n, k, ldc, split_cap, num_slices);
        return Ok(());
    }
    let a_scale = ws.offset(12);
    if nvfp4::try_gemm_from_e4m3(
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
    )? {
        return Ok(());
    }
    let (w8, w_scale) = match cached_weight_tensor(gpu, tk, weight, n, k, qg, stream)? {
        Some(cached) => {
            // Ordering fence (2026-10-16, lever-queue 04t–04x). With a cached weight the
            // kernel that writes the activation scale (`ws + 12`) is the GEMM's immediate
            // predecessor, and on GB10 some cuBLASLt CTAs then read a stale device scale
            // (64-row prompts: output columns 192..1151 = right products × the previous
            // call's scale ratio, a8/w8/scales byte-identical). The per-call path always
            // had two conversion kernels in that gap. A 4-byte memset of the weight-absmax
            // slot (unused on this path, rewritten before every per-call conversion) puts a
            // non-kernel dependency between the scale producer and the GEMM. No operand or
            // scale changes: outputs equal the per-call path's.
            if fence_scale {
                gpu.memset_async(ws, 0, 4, stream)?;
            }
            cached
        }
        None => {
            let w8 = if w8.is_null() {
                fp8_w8_scratch(gpu, n, k)?
            } else {
                w8
            };
            weight_tensor_into(gpu, tk, weight, n, k, qg, w8, ws.offset(8), stream)?;
            (w8, ws.offset(8))
        }
    };
    for slice in slices {
        spark_runtime::cublaslt::fp8_gemm_device_scaled(
            a8.offset(slice.a_byte_offset).0,
            a_scale.0,
            w8.0,
            w_scale.0,
            out.offset(slice.out_byte_offset(ldc)).0,
            slice.sub_m as u32,
            n as u32,
            k as u32,
            ldc as u32,
            weight.weight_scale_2,
            false,
            stream,
        )?;
    }
    if num_slices > 1 {
        log_fp8_m_split_engaged(m, n, k, ldc, split_cap, num_slices);
    }
    // ATLAS_QWEN4_FP8_PROJ_DUMP_DIR (diagnostic, default off): G1 operands.
    crate::layers::qwen4_fp8_proj_dump::maybe_dump(
        gpu,
        a8,
        a_scale,
        w8,
        w_scale,
        weight.weight_scale_2,
        out,
        (m, n, k, ldc),
        stream,
    )
}

#[inline]
fn parse_fp8_shared_act(raw: Option<&str>) -> bool {
    raw != Some("0")
}

/// `ATLAS_QWEN4_PREFILL_FAST_PROJ_FP8_SHARED_ACT` (default on; `=0` disables): FP8
/// projections that read the same activation (attention q/k/v) quantize it
/// once. Bit-identical to one [`fp8_gemm_ex`] per output under tensor scales
/// (same tensor, same absmax, same e4m3 rows); only the repeated
/// act_amax/act_rows passes are dropped.
pub(crate) fn fp8_shared_act_selected() -> bool {
    static S: OnceLock<bool> = OnceLock::new();
    *S.get_or_init(|| {
        parse_fp8_shared_act(
            std::env::var("ATLAS_QWEN4_PREFILL_FAST_PROJ_FP8_SHARED_ACT")
                .ok()
                .as_deref(),
        )
    })
}

/// One output of [`fp8_gemm_shared_act`]: `out[m, n]` (row stride `ldc`).
pub(crate) struct SharedActOut<'a> {
    pub weight: &'a QuantizedWeight,
    pub out: DevicePtr,
    pub n: usize,
    pub qg: Option<(usize, usize)>,
}

/// Every `outs[i].out = act[m, k] @ W_i^T` with the activation quantized once
/// (tensor scales only). Callers check [`fp8_shared_act_selected`] and
/// [`fp8_tensor_scales`].
pub(crate) fn fp8_gemm_shared_act(
    gpu: &dyn GpuBackend,
    act: DevicePtr,
    outs: &[SharedActOut<'_>],
    m: usize,
    k: usize,
    ldc: usize,
    stream: u64,
) -> Result<()> {
    anyhow::ensure!(
        fp8_tensor_scales(),
        "{FP8_SELECTOR}: shared activation needs tensor scales"
    );
    let n_max = outs.iter().map(|o| o.n).max().unwrap_or(0);
    for o in outs {
        anyhow::ensure!(
            m > 0 && o.n > 0 && k > 0 && k.is_multiple_of(16) && o.n % 16 == 0,
            "{FP8_SELECTOR}: unsupported GEMM shape m={m} n={} k={k}",
            o.n
        );
        anyhow::ensure!(
            !o.weight.weight.is_null()
                && !o.weight.weight_scale.is_null()
                && o.weight.weight_scale_2.is_finite(),
            "{FP8_SELECTOR}: invalid NVFP4 weight"
        );
        if let Some((heads, head_dim)) = o.qg {
            anyhow::ensure!(
                heads * head_dim * 2 == o.n,
                "{FP8_SELECTOR}: gated-Q permutation shape mismatch"
            );
        }
    }
    clear_bypassed_fp8_shared_act();
    let nvfp4_mode = nvfp4::mode()?;
    let scope_active = crate::layers::qwen4_hyper::batched_verify_tc_dense_scope_active();
    if nvfp4_mode.is_enabled() {
        let mut any_native = false;
        for o in outs {
            if crate::layers::qwen4_prefill_native::fp8_lookup(o.weight, o.n, k, o.qg)?.is_some() {
                any_native = true;
                break;
            }
        }
        if !any_native
            && nvfp4::try_gemm_shared_act_all(gpu, act, outs, m, k, ldc, stream)?
                == nvfp4::SharedActNvfp4Outcome::HandledAll
        {
            note_bypassed_fp8_shared_act(act, m, k, stream);
            return Ok(());
        }
    }
    let tk = fp8_tensor_kernels(gpu)?;
    let (w8, ws, a8, _) = fp8_scratch(gpu, n_max, m, k)?;
    let a_amax = if crate::layers::qwen4_fp8_act_amax::take_matching(act, m, k, stream) {
        crate::layers::qwen4_fp8_act_amax::slot(gpu)?
    } else {
        let slot = ws.offset(4);
        gpu.memset_async(slot, 0, 4, stream)?;
        KernelLaunch::new(gpu, tk.act_amax)
            .grid([256, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(act)
            .arg_u64((m * k) as u64)
            .arg_ptr(slot)
            .launch(stream)?;
        slot
    };
    launch_fp8_act_rows_tensor(gpu, &tk, act, a_amax, a8, ws.offset(12), m, k, stream)?;
    let m_max = outs
        .iter()
        .map(|o| resolve_fp8_pad_m(m, o.n))
        .max()
        .unwrap_or(m);
    if m_max > m {
        gpu.memset_async(a8.offset(m * k), 0, (m_max - m) * k, stream)?;
    }
    let mut first_fp8 = true;
    for o in outs {
        if nvfp4_mode.is_enabled()
            && crate::layers::qwen4_prefill_native::fp8_lookup(o.weight, o.n, k, o.qg)?.is_none()
            && nvfp4::try_gemm_bf16_with_amax(
                gpu,
                act,
                a_amax,
                o.weight,
                o.out,
                m,
                o.n,
                k,
                ldc,
                o.qg,
                scope_active,
                stream,
            )?
        {
            continue;
        }
        let m_out = resolve_fp8_pad_m(m, o.n);
        fp8_weight_tensor_gemm(
            gpu, &tk, o.weight, o.out, m_out, o.n, k, ldc, o.qg, w8, ws, a8, first_fp8, stream,
        )?;
        first_fp8 = false;
    }
    static LOGGED: std::sync::Once = std::sync::Once::new();
    LOGGED.call_once(|| {
        tracing::info!(
            "FAST_PROJ_FP8_SHARED_ACT_ENGAGED m={m} m_max={m_max} k={k} outputs={} scales=tensor bit_identical_to_per_call",
            outs.len()
        )
    });
    Ok(())
}

/// [`dequant_gemm_ex`] or, when `fp8` (the family's FP8 selection), [`fp8_gemm_ex`].
#[allow(clippy::too_many_arguments)]
pub(crate) fn proj_gemm_ex(
    gpu: &dyn GpuBackend,
    fp8: bool,
    act: DevicePtr,
    weight: &QuantizedWeight,
    out: DevicePtr,
    m: usize,
    n: usize,
    k: usize,
    ldc: usize,
    qg: Option<(usize, usize)>,
    stream: u64,
) -> Result<()> {
    if fp8 {
        fp8_gemm_ex(gpu, act, weight, out, m, n, k, ldc, qg, stream)
    } else {
        dequant_gemm_ex(gpu, act, weight, out, m, n, k, ldc, qg, stream)
    }
}

/// Dequantize an NVFP4 weight to exact BF16 `LUT * fp8` values in `out`
/// (`[n, k]` row-major) and return the per-tensor scale2 the caller must apply
/// (GEMM alpha or epilogue scale).
pub(crate) fn dequant_into(
    gpu: &dyn GpuBackend,
    weight: &QuantizedWeight,
    out: DevicePtr,
    n: usize,
    k: usize,
    stream: u64,
) -> Result<f32> {
    anyhow::ensure!(
        n > 0 && k > 0 && k.is_multiple_of(16),
        "{SELECTOR}: unsupported dequant shape n={n} k={k}"
    );
    anyhow::ensure!(
        !weight.weight.is_null()
            && !weight.weight_scale.is_null()
            && weight.weight_scale_2.is_finite(),
        "{SELECTOR}: invalid NVFP4 weight"
    );
    // ATLAS_QWEN4_PREFILL_NATIVE=1: the resident BF16 source, scale 1.
    if let Some((bf16, module)) = crate::layers::qwen4_prefill_native::bf16_lookup(weight, n, k) {
        gpu.copy_d2d_async(bf16, out, n * k * 2, stream)?;
        crate::layers::qwen4_prefill_native::note_call(module, "bf16_copy_into", 0, n, k, 1.0);
        return Ok(1.0);
    }
    let groups = (n * k / 16) as u32;
    KernelLaunch::new(gpu, kernel(gpu)?)
        .grid([div_ceil(groups, 256), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(weight.weight)
        .arg_ptr(weight.weight_scale)
        .arg_f32(1.0)
        .arg_ptr(out)
        .arg_u32(n as u32)
        .arg_u32(k as u32)
        .arg_u32(0)
        .arg_u32(0)
        .launch(stream)?;
    Ok(weight.weight_scale_2)
}

/// Dequantize `weight` (`[n, k]`) into the shared BF16 scratch exactly as
/// [`dequant_gemm`] does and return it with the GEMM alpha that completes it,
/// for fused kernels that consume the dequantized operand themselves. The
/// scratch is valid until the next fast projection on the stream.
pub(crate) fn dequant_to_scratch(
    gpu: &dyn GpuBackend,
    weight: &QuantizedWeight,
    n: usize,
    k: usize,
    stream: u64,
) -> Result<(DevicePtr, f32)> {
    anyhow::ensure!(
        n > 0 && k > 0 && k.is_multiple_of(16),
        "{SELECTOR}: unsupported dequant shape n={n} k={k}"
    );
    anyhow::ensure!(
        !weight.weight.is_null()
            && !weight.weight_scale.is_null()
            && weight.weight_scale_2.is_finite(),
        "{SELECTOR}: invalid NVFP4 weight"
    );
    // ATLAS_QWEN4_PREFILL_NATIVE=1: the resident BF16 source is the operand.
    if let Some((bf16, module)) = crate::layers::qwen4_prefill_native::bf16_lookup(weight, n, k) {
        crate::layers::qwen4_prefill_native::note_call(module, "bf16_operand", 0, n, k, 1.0);
        return Ok((bf16, 1.0));
    }
    let w = scratch(gpu, n * k * 2)?;
    let (dq_scale, alpha) = if fold_scale2() {
        (weight.weight_scale_2, 1.0f32)
    } else {
        (1.0f32, weight.weight_scale_2)
    };
    KernelLaunch::new(gpu, kernel(gpu)?)
        .grid([div_ceil((n * k / 16) as u32, 256), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(weight.weight)
        .arg_ptr(weight.weight_scale)
        .arg_f32(dq_scale)
        .arg_ptr(w)
        .arg_u32(n as u32)
        .arg_u32(k as u32)
        .arg_u32(0)
        .arg_u32(0)
        .launch(stream)?;
    Ok((w, alpha))
}

/// General form: output row stride `ldc` elements and an optional gated-Q
/// weight-row permutation `(heads, head_dim)` applied during dequantization.
#[allow(clippy::too_many_arguments)]
pub(crate) fn dequant_gemm_ex(
    gpu: &dyn GpuBackend,
    act: DevicePtr,
    weight: &QuantizedWeight,
    out: DevicePtr,
    m: usize,
    n: usize,
    k: usize,
    ldc: usize,
    qg: Option<(usize, usize)>,
    stream: u64,
) -> Result<()> {
    anyhow::ensure!(
        m > 0 && n > 0 && k > 0 && k.is_multiple_of(16),
        "{SELECTOR}: unsupported GEMM shape m={m} n={n} k={k}"
    );
    anyhow::ensure!(
        !weight.weight.is_null()
            && !weight.weight_scale.is_null()
            && weight.weight_scale_2.is_finite(),
        "{SELECTOR}: invalid NVFP4 weight"
    );
    // ATLAS_QWEN4_PREFILL_NATIVE=1: BF16-shipped weights multiply as shipped.
    if qg.is_none()
        && let Some((bf16, module)) = crate::layers::qwen4_prefill_native::bf16_lookup(weight, n, k)
    {
        let act = crate::layers::qwen4_fakequant::proj_operand(gpu, act, m, n, k, stream)?;
        return crate::layers::qwen4_prefill_native::bf16_dequant_gemm(
            act,
            bf16,
            module,
            out,
            m,
            n,
            k,
            ldc,
            tuned(),
            stream,
        );
    }
    if let Some((heads, head_dim)) = qg {
        anyhow::ensure!(
            heads.checked_mul(head_dim).and_then(|v| v.checked_mul(2)) == Some(n),
            "{SELECTOR}: gated-Q permutation shape mismatch"
        );
    }
    let kh = kernel(gpu)?;
    let w = scratch(gpu, n * k * 2)?;
    let groups = (n * k / 16) as u32;
    // LUT[nib] * fp8 has <= 7 significant bits, so it is EXACT in BF16; the
    // per-tensor scale2 is applied as the GEMM alpha in FP32 instead of being
    // folded into the rounded weight (default). ATLAS_QWEN4_FAST_PROJ_FOLD_SCALE2=1
    // restores the W4A16-GEMM-style bf16(LUT*fp8*scale2) operand.
    let fold = fold_scale2();
    let (dq_scale, alpha) = if fold {
        (weight.weight_scale_2, 1.0f32)
    } else {
        (1.0f32, weight.weight_scale_2)
    };
    KernelLaunch::new(gpu, kh)
        .grid([div_ceil(groups, 256), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(weight.weight)
        .arg_ptr(weight.weight_scale)
        .arg_f32(dq_scale)
        .arg_ptr(w)
        .arg_u32(n as u32)
        .arg_u32(k as u32)
        .arg_u32(qg.map_or(0, |(h, _)| h) as u32)
        .arg_u32(qg.map_or(0, |(_, d)| d) as u32)
        .launch(stream)?;
    // Default-off quality-gate fake quantization (ATLAS_QWEN4_PREFILL_FAKEQUANT).
    let act = crate::layers::qwen4_fakequant::proj_operand(gpu, act, m, n, k, stream)?;
    if tuned() {
        spark_runtime::cublaslt::bf16_gemm_act_weight_t_tuned_ex(
            act.0, w.0, out.0, m as u32, n as u32, k as u32, ldc as u32, alpha, stream,
        )
    } else {
        spark_runtime::cublaslt::bf16_gemm_act_weight_t_ldc(
            act.0, w.0, out.0, m as u32, n as u32, k as u32, ldc as u32, alpha, stream,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_selector_lists() {
        assert_eq!(parse(None).unwrap(), Selection::default());
        assert_eq!(parse(Some("0")).unwrap(), Selection::default());
        assert_eq!(
            parse(Some("ssm")).unwrap(),
            Selection {
                ssm: true,
                hc: false,
                attn: false
            }
        );
        assert_eq!(
            parse(Some("ssm,hc")).unwrap(),
            Selection {
                ssm: true,
                hc: true,
                attn: false
            }
        );
        assert_eq!(
            parse(Some("all")).unwrap(),
            Selection {
                ssm: true,
                hc: true,
                attn: true
            }
        );
        assert!(parse(Some("1")).is_err());
        assert!(parse(Some("moe")).is_err());
    }

    #[test]
    fn fp8_shared_act_is_default_on_and_zero_disables() {
        assert!(parse_fp8_shared_act(None));
        assert!(parse_fp8_shared_act(Some("1")));
        assert!(!parse_fp8_shared_act(Some("0")));
    }

    #[test]
    fn fp8_scratch_lazy_w8_guard_skips_40mib_under_tensor_scales() {
        // Under tensor scales, `fp8_scratch` skips eager `cur.w` (16384 * 2560 = 40 MiB / 41.9 MB)
        // whether `fp8_weight_cache_selected()` is true or false; `fp8_w8_scratch` allocates lazily.
        assert!(!fp8_scratch_needs_eager_w8(true));
        assert!(fp8_scratch_needs_eager_w8(false));
    }

    #[test]
    fn pad_rows_guard_rounds_wide_unaligned_m_within_arena_cap() {
        // Unguarded (cap = 0): never pads.
        assert_eq!(resolve_fp8_pad_m(7501, 16384), 7501);
        assert_eq!(resolve_fp8_pad_m(8194, 2560), 8194);

        let outer = PadRowsGuard::enter(16384);
        // Wide projections (N >= 12288) above 3072 avoid odd-256 / odd-512 SM121a
        // cuBLASLt cliffs (4352, 5632, 7680, 8448) by promoting to 1024-row golden multiples.
        assert_eq!(resolve_fp8_pad_m(4200, 16384), 5120);
        assert_eq!(resolve_fp8_pad_m(5500, 16384), 6144);
        assert_eq!(resolve_fp8_pad_m(7501, 16384), 8192);
        // Split remainders after 8192 also promote above 512 to 1024-row golden multiples.
        assert_eq!(resolve_fp8_pad_m(12544, 16384), 13312);
        assert_eq!(resolve_fp8_pad_m(13824, 16384), 14336);
        assert_eq!(resolve_fp8_pad_m(14885, 16384), 15360);
        assert_eq!(resolve_fp8_pad_m(15872, 16384), 16384);
        assert_eq!(resolve_fp8_pad_m(16176, 16384), 16384);
        // With M-split disabled by default, 8320 avoids the 8448 cliff by rounding to 9216.
        assert_eq!(resolve_fp8_pad_m(8320, 16384), 9216);
        // With M-split=8192 opt-in, small split remainder (<= 512) aligns to 256/512 and splits into [8192, 256].
        assert_eq!(resolve_fp8_pad_m_with_cap(8320, 16384, 16384, 8192), 8448);
        // Medium/narrow projections (512 <= N < 12288) align to 256-row multiples
        // (covering SSM/attn out 2560, attention K/V 512, and MoE router 512).
        assert_eq!(resolve_fp8_pad_m(8194, 2560), 8448);
        assert_eq!(resolve_fp8_pad_m(8224, 1024), 8448);
        assert_eq!(resolve_fp8_pad_m(7501, 512), 7680);
        // Already golden-aligned: unchanged.
        assert_eq!(resolve_fp8_pad_m(8192, 16384), 8192);
        // Small M (< 512) or very narrow N (< 512): unchanged.
        assert_eq!(resolve_fp8_pad_m(385, 16384), 385);
        assert_eq!(resolve_fp8_pad_m(7501, 256), 7501);

        // Nested tighter cap falls back to 256 then 128 when golden/256 exceed cap, or refuses both.
        {
            let _inner = PadRowsGuard::enter(8320);
            assert_eq!(resolve_fp8_pad_m(8194, 16384), 8320);
            assert_eq!(resolve_fp8_pad_m(7501, 16384), 8192);
        }
        {
            let _inner = PadRowsGuard::enter(7800);
            assert_eq!(resolve_fp8_pad_m(7501, 16384), 7680);
        }
        {
            let _inner = PadRowsGuard::enter(8200);
            assert_eq!(resolve_fp8_pad_m(8194, 16384), 8194);
            assert_eq!(resolve_fp8_pad_m(7501, 16384), 8192);
        }
        assert_eq!(resolve_fp8_pad_m(8194, 16384), 9216);
        assert_eq!(resolve_fp8_pad_m_with_cap(8194, 16384, 16384, 8192), 8448);
        drop(outer);
        assert_eq!(resolve_fp8_pad_m(8194, 16384), 8194);
    }

    #[test]
    fn shared_act_qkv_widths_pad_q_without_suppressing_on_narrow_kv() {
        let _guard = PadRowsGuard::enter(16384);
        let qkv_widths = [12288usize, 512, 512];
        let resolved_7501: Vec<usize> = qkv_widths
            .iter()
            .map(|&n| resolve_fp8_pad_m(7501, n))
            .collect();
        assert_eq!(resolved_7501, vec![8192, 7680, 7680]);
        assert_eq!(resolved_7501.iter().copied().max().unwrap(), 8192);

        let resolved_8194: Vec<usize> = qkv_widths
            .iter()
            .map(|&n| resolve_fp8_pad_m(8194, n))
            .collect();
        assert_eq!(resolved_8194, vec![9216, 8448, 8448]);

        let resolved_14885: Vec<usize> = qkv_widths
            .iter()
            .map(|&n| resolve_fp8_pad_m(14885, n))
            .collect();
        assert_eq!(resolved_14885, vec![15360, 15104, 15104]);
        assert_eq!(resolved_14885.iter().copied().max().unwrap(), 15360);
    }

    #[test]
    fn fp8_gemm_m_split_slices_handles_threshold_width_and_strided_qkv_offsets() {
        // M=8192 does not split; M > 8192 (e.g. 15104 / 15360) splits on 8192 boundaries.
        let s8192 = fp8_gemm_m_split_slices(8192, 16384, 2560, 8192);
        assert_eq!(s8192.len(), 1);
        assert!(!s8192.is_empty());
        assert_eq!(
            s8192.collect::<Vec<_>>(),
            vec![Fp8GemmMSlice {
                row_start: 0,
                sub_m: 8192,
                a_byte_offset: 0,
            }]
        );
        assert_eq!(
            fp8_gemm_m_split_slices(15104, 16384, 2560, 8192).collect::<Vec<_>>(),
            vec![
                Fp8GemmMSlice {
                    row_start: 0,
                    sub_m: 8192,
                    a_byte_offset: 0,
                },
                Fp8GemmMSlice {
                    row_start: 8192,
                    sub_m: 6912,
                    a_byte_offset: 8192 * 2560,
                },
            ]
        );

        // M=16384, N=16384, K=2560 splits cleanly into two 8192-row slices.
        let ssm_qkvz: Vec<_> = fp8_gemm_m_split_slices(16384, 16384, 2560, 8192).collect();
        assert_eq!(
            ssm_qkvz,
            vec![
                Fp8GemmMSlice {
                    row_start: 0,
                    sub_m: 8192,
                    a_byte_offset: 0,
                },
                Fp8GemmMSlice {
                    row_start: 8192,
                    sub_m: 8192,
                    a_byte_offset: 8192 * 2560,
                },
            ]
        );
        assert_eq!(ssm_qkvz[0].out_byte_offset(16384), 0);
        assert_eq!(ssm_qkvz[1].out_byte_offset(16384), 8192 * 16384 * 2);

        // Medium/narrow outputs (N=2560 out_proj, N=512 router / K / V) stay unsplit
        // so SM121a cuBLASLt avoids sm89_xmma fallback on 8192x2560x6144.
        assert_eq!(
            fp8_gemm_m_split_slices(16384, 2560, 6144, 8192).collect::<Vec<_>>(),
            vec![Fp8GemmMSlice {
                row_start: 0,
                sub_m: 16384,
                a_byte_offset: 0,
            }]
        );
        assert_eq!(
            fp8_gemm_m_split_slices(16384, 512, 2560, 8192).collect::<Vec<_>>(),
            vec![Fp8GemmMSlice {
                row_start: 0,
                sub_m: 16384,
                a_byte_offset: 0,
            }]
        );

        // Strided QKV attention output: Q has N=12288, ldc=13312 (> N).
        let attn_q: Vec<_> = fp8_gemm_m_split_slices(16384, 12288, 2560, 8192).collect();
        assert_eq!(attn_q.len(), 2);
        assert_eq!(attn_q[0].a_byte_offset, 0);
        assert_eq!(attn_q[0].out_byte_offset(13312), 0);
        assert_eq!(attn_q[0].outer_a_scale_byte_offset(), 0);
        assert_eq!(attn_q[1].row_start, 8192);
        assert_eq!(attn_q[1].sub_m, 8192);
        assert_eq!(attn_q[1].a_byte_offset, 8192 * 2560);
        assert_eq!(attn_q[1].out_byte_offset(13312), 8192 * 13312 * 2);
        assert_eq!(attn_q[1].outer_a_scale_byte_offset(), 8192 * 4);

        // Disabled (split_cap=0) and zero rows.
        assert_eq!(
            fp8_gemm_m_split_slices(16384, 16384, 2560, 0).collect::<Vec<_>>(),
            vec![Fp8GemmMSlice {
                row_start: 0,
                sub_m: 16384,
                a_byte_offset: 0,
            }]
        );
        let empty = fp8_gemm_m_split_slices(0, 16384, 2560, 8192);
        assert!(empty.is_empty());
        assert_eq!(empty.len(), 0);
        assert!(empty.collect::<Vec<_>>().is_empty());

        // Non-multiple M and PadRowsGuard interaction:
        // 16176 -> 16384 -> 2x8192; 14885 -> 15360 -> (8192, 7168); 8194 -> 8448 -> (8192, 256).
        let _guard = PadRowsGuard::enter(16384);
        let padded_m = resolve_fp8_pad_m(16176, 16384);
        assert_eq!(padded_m, 16384);
        let padded_slices: Vec<_> = fp8_gemm_m_split_slices(padded_m, 16384, 2560, 8192).collect();
        assert_eq!(padded_slices.len(), 2);
        assert_eq!(padded_slices[0].sub_m, 8192);
        assert_eq!(padded_slices[1].sub_m, 8192);

        let padded_14885 = resolve_fp8_pad_m(14885, 16384);
        assert_eq!(padded_14885, 15360);
        let slices_14885: Vec<_> =
            fp8_gemm_m_split_slices(padded_14885, 16384, 2560, 8192).collect();
        assert_eq!(slices_14885.len(), 2);
        assert_eq!(slices_14885[0].sub_m, 8192);
        assert_eq!(slices_14885[1].sub_m, 7168);

        // No split slice for N >= 12288 ever lands on cliff sizes (4352, 5632, 7680, 8448).
        for raw_m in [
            4352usize, 5632, 7680, 8194, 8448, 12544, 13824, 14885, 15872,
        ] {
            let m_pad = resolve_fp8_pad_m(raw_m, 16384);
            assert!(
                m_pad <= fp8_scratch_m_cap(raw_m, 2048, false),
                "raw_m={raw_m}"
            );
            for slice in fp8_gemm_m_split_slices(m_pad, 16384, 2560, 8192) {
                assert!(
                    ![4352, 5632, 7680, 8448].contains(&slice.sub_m),
                    "raw_m={raw_m} landed on cliff sub_m={}",
                    slice.sub_m
                );
            }
        }

        // Non-128-aligned custom cap aligns down to 128-row multiple.
        let custom: Vec<_> = fp8_gemm_m_split_slices(16000, 16384, 2560, 5000).collect();
        assert_eq!(
            custom
                .iter()
                .map(|s| (s.row_start, s.sub_m))
                .collect::<Vec<_>>(),
            vec![(0, 4992), (4992, 4992), (9984, 4992), (14976, 1024)]
        );
        for slice in &custom {
            assert!(slice.row_start.is_multiple_of(128));
            assert_eq!(slice.a_byte_offset, slice.row_start * 2560);
        }

        // Env parser defaults to 0 (disabled), positive integer enables.
        assert_eq!(parse_fp8_gemm_m_split(None), 0);
        assert_eq!(parse_fp8_gemm_m_split(Some("")), 0);
        assert_eq!(parse_fp8_gemm_m_split(Some("0")), 0);
        assert_eq!(parse_fp8_gemm_m_split(Some("4096")), 4096);
        assert_eq!(parse_fp8_gemm_m_split(Some("invalid")), 0);
    }

    #[test]
    fn fp8_scratch_m_cap_preallocates_batched_verify_ceiling_for_graph_safety() {
        // Batched-verify scope guarantees at least QWEN4_NATIVE_MAX_ROWS (256)
        // even if `m` starts small (e.g. k=2 -> m=12) or max_chunk_rows is small, so growing to
        // k=5 (m=80..256) never reallocates `a8`/`as_` during CUDA graph capture.
        assert_eq!(fp8_scratch_m_cap(12, 0, true), 256);
        assert_eq!(fp8_scratch_m_cap(20, 64, true), 256);
        assert_eq!(fp8_scratch_m_cap(80, 0, true), 256);
        assert_eq!(fp8_scratch_m_cap(128, 0, true), 256);
        assert_eq!(fp8_scratch_m_cap(160, 0, true), 256);
        assert_eq!(fp8_scratch_m_cap(256, 0, true), 256);
        assert_eq!(fp8_scratch_m_cap(20, 2048, true), 2048);
        assert_eq!(fp8_scratch_m_cap(80, 2048, false), 2048);
        assert_eq!(fp8_scratch_m_cap(20, 0, false), 256);
        // Wide golden padding above 3072 always stays within `a8` scratch row capacity.
        assert_eq!(fp8_scratch_m_cap(7501, 2048, false), 8192);
        assert_eq!(fp8_scratch_m_cap(14885, 2048, false), 15360);
        assert_eq!(fp8_scratch_m_cap(14885, 16384, false), 16384);
    }

    #[test]
    fn direct_nvfp4_batched_verify_route_covers_c4_to_c16_and_preserves_prefill() {
        // Default OFF: batched-verify dense projections use cached/tensor-scale FP8 cuBLASLt
        // unless explicitly opted in via ATLAS_QWEN4_BATCHED_VERIFY_DIRECT_NVFP4=1.
        assert!(!parse_direct_nvfp4_dense(None));
        assert!(!parse_direct_nvfp4_dense(Some("")));
        assert!(!parse_direct_nvfp4_dense(Some("0")));
        assert!(!parse_direct_nvfp4_dense(Some(" 0 ")));
        assert!(!parse_direct_nvfp4_dense(Some("true")));
        assert!(parse_direct_nvfp4_dense(Some("1")));
        assert!(parse_direct_nvfp4_dense(Some(" 1 ")));

        // Scoped batched verify rows <= 256 with contiguous output and no QG permute admit direct NVFP4.
        for (m, n, k) in [
            (12, 16_384, 2_560),
            (20, 16_384, 2_560),
            (40, 2_560, 6_144),
            (80, 16_384, 2_560),
            (128, 2_560, 6_144),
            (160, 16_384, 2_560),
            (256, 2_560, 6_144),
        ] {
            assert!(direct_nvfp4_batched_verify_shape_eligible(
                true, true, m, n, k, n, None
            ));
            // Prefill (scope_active = false) or disabled stays on FP8 path.
            assert!(!direct_nvfp4_batched_verify_shape_eligible(
                false, true, m, n, k, n, None
            ));
            assert!(!direct_nvfp4_batched_verify_shape_eligible(
                true, false, m, n, k, n, None
            ));
        }
        // Reject > 256 rows, strided ldc != n, or QG weight permutation.
        assert!(!direct_nvfp4_batched_verify_shape_eligible(
            true, true, 257, 16_384, 2_560, 16_384, None
        ));
        assert!(!direct_nvfp4_batched_verify_shape_eligible(
            true, true, 40, 12_288, 2_560, 13_312, None
        ));
        assert!(!direct_nvfp4_batched_verify_shape_eligible(
            true,
            true,
            40,
            12_288,
            2_560,
            12_288,
            Some((24, 256))
        ));

        // M <= 64 always selects Pipe; M > 64 selects PipeM128N128 for wide N >= 6144 and Pipe for N=2560.
        assert_eq!(
            select_direct_nvfp4_gemm_route(40, 16_384, 2_560, true, true, true, true, true, true),
            Some(DirectNvfp4GemmRoute::Pipe)
        );
        assert_eq!(
            select_direct_nvfp4_gemm_route(80, 16_384, 2_560, true, true, true, true, true, true),
            Some(DirectNvfp4GemmRoute::PipeM128N128)
        );
        assert_eq!(
            select_direct_nvfp4_gemm_route(80, 12_288, 2_560, true, true, true, true, true, true),
            Some(DirectNvfp4GemmRoute::PipeM128N128)
        );
        assert_eq!(
            select_direct_nvfp4_gemm_route(80, 2_560, 6_144, true, true, true, true, true, true),
            Some(DirectNvfp4GemmRoute::Pipe)
        );
        // Fallback chain PipeM128N128 -> Pipe -> Base -> None.
        assert_eq!(
            select_direct_nvfp4_gemm_route(80, 16_384, 2_560, true, true, true, false, true, true),
            Some(DirectNvfp4GemmRoute::Pipe)
        );
        assert_eq!(
            select_direct_nvfp4_gemm_route(80, 16_384, 2_560, true, true, true, false, false, true),
            Some(DirectNvfp4GemmRoute::Base)
        );
        assert_eq!(
            select_direct_nvfp4_gemm_route(
                80, 16_384, 2_560, true, true, true, false, false, false
            ),
            None
        );
    }

    #[test]
    fn coalesced_nvfp4_tile_selection_and_fallback_cover_all_flash_next_shapes() {
        // Default ON; `"0"` disables cleanly.
        assert!(parse_coalesced_nvfp4_dense(None));
        assert!(parse_coalesced_nvfp4_dense(Some("")));
        assert!(parse_coalesced_nvfp4_dense(Some("1")));
        assert!(parse_coalesced_nvfp4_dense(Some(" 1 ")));
        assert!(!parse_coalesced_nvfp4_dense(Some("0")));
        assert!(!parse_coalesced_nvfp4_dense(Some(" 0 ")));

        // Narrow N <= 1024 -> N16 (v_router 512, hc.down 320); medium 1025..=4096 -> N32
        // (ssm.out / attn.oproj 2560); wide N > 4096 -> N128 (hc.up 10240, attn.q 12288,
        // ssm.qkvz 16384, mtp_lm_head 96000, odd lm_head 248077).
        for (m, n, n_tile, m_tile, co_n, tile_m, threads, sym) in [
            (
                1,
                320,
                CoalescedNvfp4NTile::N16,
                CoalescedNvfp4MTile::M32,
                16,
                32,
                64,
                "w4a16_gemm_coalesced_n16_m32",
            ),
            (
                20,
                512,
                CoalescedNvfp4NTile::N16,
                CoalescedNvfp4MTile::M32,
                16,
                32,
                64,
                "w4a16_gemm_coalesced_n16_m32",
            ),
            (
                40,
                512,
                CoalescedNvfp4NTile::N16,
                CoalescedNvfp4MTile::M64,
                16,
                64,
                64,
                "w4a16_gemm_coalesced_n16_m64",
            ),
            (
                80,
                320,
                CoalescedNvfp4NTile::N16,
                CoalescedNvfp4MTile::M128,
                16,
                128,
                64,
                "w4a16_gemm_coalesced_n16_m128",
            ),
            (
                32,
                1024,
                CoalescedNvfp4NTile::N16,
                CoalescedNvfp4MTile::M32,
                16,
                32,
                64,
                "w4a16_gemm_coalesced_n16_m32",
            ),
            (
                33,
                1025,
                CoalescedNvfp4NTile::N32,
                CoalescedNvfp4MTile::M64,
                32,
                64,
                128,
                "w4a16_gemm_coalesced_n32_m64",
            ),
            (
                20,
                2560,
                CoalescedNvfp4NTile::N32,
                CoalescedNvfp4MTile::M32,
                32,
                32,
                128,
                "w4a16_gemm_coalesced_n32_m32",
            ),
            (
                64,
                4096,
                CoalescedNvfp4NTile::N32,
                CoalescedNvfp4MTile::M64,
                32,
                64,
                128,
                "w4a16_gemm_coalesced_n32_m64",
            ),
            (
                80,
                2560,
                CoalescedNvfp4NTile::N32,
                CoalescedNvfp4MTile::M128,
                32,
                128,
                128,
                "w4a16_gemm_coalesced_n32_m128",
            ),
            (
                65,
                4097,
                CoalescedNvfp4NTile::N128,
                CoalescedNvfp4MTile::M128,
                128,
                128,
                512,
                "w4a16_gemm_coalesced_m128",
            ),
            (
                20,
                10240,
                CoalescedNvfp4NTile::N128,
                CoalescedNvfp4MTile::M32,
                128,
                32,
                512,
                "w4a16_gemm_coalesced_m32",
            ),
            (
                40,
                12288,
                CoalescedNvfp4NTile::N128,
                CoalescedNvfp4MTile::M64,
                128,
                64,
                512,
                "w4a16_gemm_coalesced_m64",
            ),
            (
                80,
                16384,
                CoalescedNvfp4NTile::N128,
                CoalescedNvfp4MTile::M128,
                128,
                128,
                512,
                "w4a16_gemm_coalesced_m128",
            ),
            (
                128,
                96000,
                CoalescedNvfp4NTile::N128,
                CoalescedNvfp4MTile::M128,
                128,
                128,
                512,
                "w4a16_gemm_coalesced_m128",
            ),
            (
                80,
                248077,
                CoalescedNvfp4NTile::N128,
                CoalescedNvfp4MTile::M128,
                128,
                128,
                512,
                "w4a16_gemm_coalesced_m128",
            ),
        ] {
            let tile = select_coalesced_nvfp4_tile(m, n).expect("valid tile");
            assert_eq!(tile.n_tile, n_tile, "m={m} n={n}");
            assert_eq!(tile.m_tile, m_tile, "m={m} n={n}");
            assert_eq!(tile.co_n(), co_n, "m={m} n={n}");
            assert_eq!(tile.tile_m(), tile_m, "m={m} n={n}");
            assert_eq!(tile.threads(), threads, "m={m} n={n}");
            assert_eq!(tile.symbol(), sym, "m={m} n={n}");
        }
        assert_eq!(select_coalesced_nvfp4_tile(0, 512), None);
        assert!(select_coalesced_nvfp4_tile(129, 512).is_some());
        assert!(select_coalesced_nvfp4_tile(256, 16_384).is_some());
        assert_eq!(select_coalesced_nvfp4_tile(257, 512), None);
        assert_eq!(select_coalesced_nvfp4_tile(80, 0), None);

        // K-aware Warp-Split-K & N64 selection across all Flash-Next shapes.
        for (m, n, k, n_tile, m_tile, co_n, tile_m, threads, sym) in [
            // HC down (N=320, K=10240)
            (
                10,
                320,
                10_240,
                CoalescedNvfp4NTile::N8Sk8,
                CoalescedNvfp4MTile::M16,
                8,
                16,
                256,
                "w4a16_gemm_coalesced_n8_sk8_m16",
            ),
            (
                20,
                320,
                10_240,
                CoalescedNvfp4NTile::N8Sk8,
                CoalescedNvfp4MTile::M16,
                8,
                16,
                256,
                "w4a16_gemm_coalesced_n8_sk8_m16",
            ),
            (
                26,
                320,
                10_240,
                CoalescedNvfp4NTile::N16Sk4,
                CoalescedNvfp4MTile::M16,
                16,
                16,
                256,
                "w4a16_gemm_coalesced_n16_sk4_m16",
            ),
            (
                40,
                320,
                10_240,
                CoalescedNvfp4NTile::N32Sk4,
                CoalescedNvfp4MTile::M16,
                32,
                16,
                512,
                "w4a16_gemm_coalesced_n32_sk4_m16",
            ),
            (
                80,
                320,
                10_240,
                CoalescedNvfp4NTile::N64Sk4,
                CoalescedNvfp4MTile::M16,
                64,
                16,
                1024,
                "w4a16_gemm_coalesced_n64_sk4_m16",
            ),
            (
                128,
                320,
                10_240,
                CoalescedNvfp4NTile::N64Sk4,
                CoalescedNvfp4MTile::M16,
                64,
                16,
                1024,
                "w4a16_gemm_coalesced_n64_sk4_m16",
            ),
            // MoE v_router (N=512, K=2560)
            (
                12,
                512,
                2_560,
                CoalescedNvfp4NTile::N8Sk8,
                CoalescedNvfp4MTile::M16,
                8,
                16,
                256,
                "w4a16_gemm_coalesced_n8_sk8_m16",
            ),
            (
                20,
                512,
                2_560,
                CoalescedNvfp4NTile::N16Sk4,
                CoalescedNvfp4MTile::M16,
                16,
                16,
                256,
                "w4a16_gemm_coalesced_n16_sk4_m16",
            ),
            (
                40,
                512,
                2_560,
                CoalescedNvfp4NTile::N32Sk4,
                CoalescedNvfp4MTile::M16,
                32,
                16,
                512,
                "w4a16_gemm_coalesced_n32_sk4_m16",
            ),
            (
                80,
                512,
                2_560,
                CoalescedNvfp4NTile::N32Sk4,
                CoalescedNvfp4MTile::M16,
                32,
                16,
                512,
                "w4a16_gemm_coalesced_n32_sk4_m16",
            ),
            // Narrow short-K falls back to non-split-K N16
            (
                20,
                512,
                320,
                CoalescedNvfp4NTile::N16,
                CoalescedNvfp4MTile::M32,
                16,
                32,
                64,
                "w4a16_gemm_coalesced_n16_m32",
            ),
            // ssm.out / attn.oproj (N=2560, K=6144)
            (
                10,
                2_560,
                6_144,
                CoalescedNvfp4NTile::N32Sk4,
                CoalescedNvfp4MTile::M32,
                32,
                32,
                512,
                "w4a16_gemm_coalesced_n32_sk4_m32",
            ),
            (
                20,
                2_560,
                6_144,
                CoalescedNvfp4NTile::N32Sk4,
                CoalescedNvfp4MTile::M32,
                32,
                32,
                512,
                "w4a16_gemm_coalesced_n32_sk4_m32",
            ),
            (
                40,
                2_560,
                6_144,
                CoalescedNvfp4NTile::N64Sk2,
                CoalescedNvfp4MTile::M64,
                64,
                64,
                512,
                "w4a16_gemm_coalesced_n64_sk2_m64",
            ),
            (
                64,
                2_560,
                6_144,
                CoalescedNvfp4NTile::N64Sk2,
                CoalescedNvfp4MTile::M64,
                64,
                64,
                512,
                "w4a16_gemm_coalesced_n64_sk2_m64",
            ),
            (
                80,
                2_560,
                6_144,
                CoalescedNvfp4NTile::N64,
                CoalescedNvfp4MTile::M128,
                64,
                128,
                256,
                "w4a16_gemm_coalesced_n64_m128",
            ),
            // Wide HC up (10240x320), attn.q (12288x2560), ssm.qkvz (16384x2560), lm_head
            (
                20,
                10_240,
                320,
                CoalescedNvfp4NTile::N32,
                CoalescedNvfp4MTile::M32,
                32,
                32,
                128,
                "w4a16_gemm_coalesced_n32_m32",
            ),
            (
                40,
                12_288,
                2_560,
                CoalescedNvfp4NTile::N64,
                CoalescedNvfp4MTile::M64,
                64,
                64,
                256,
                "w4a16_gemm_coalesced_n64_m64",
            ),
            (
                20,
                16_384,
                2_560,
                CoalescedNvfp4NTile::N32,
                CoalescedNvfp4MTile::M32,
                32,
                32,
                128,
                "w4a16_gemm_coalesced_n32_m32",
            ),
            (
                40,
                16_384,
                2_560,
                CoalescedNvfp4NTile::N128,
                CoalescedNvfp4MTile::M64,
                128,
                64,
                512,
                "w4a16_gemm_coalesced_m64",
            ),
            (
                80,
                16_384,
                2_560,
                CoalescedNvfp4NTile::N128,
                CoalescedNvfp4MTile::M128,
                128,
                128,
                512,
                "w4a16_gemm_coalesced_m128",
            ),
            (
                20,
                96_000,
                2_560,
                CoalescedNvfp4NTile::N128,
                CoalescedNvfp4MTile::M32,
                128,
                32,
                512,
                "w4a16_gemm_coalesced_m32",
            ),
            (
                80,
                248_077,
                2_560,
                CoalescedNvfp4NTile::N128,
                CoalescedNvfp4MTile::M128,
                128,
                128,
                512,
                "w4a16_gemm_coalesced_m128",
            ),
        ] {
            let tile = select_coalesced_nvfp4_tile_for_shape(m, n, k).expect("valid shape tile");
            assert_eq!(tile.n_tile, n_tile, "m={m} n={n} k={k}");
            assert_eq!(tile.m_tile, m_tile, "m={m} n={n} k={k}");
            assert_eq!(tile.co_n(), co_n, "m={m} n={n} k={k}");
            assert_eq!(tile.tile_m(), tile_m, "m={m} n={n} k={k}");
            assert_eq!(tile.threads(), threads, "m={m} n={n} k={k}");
            assert_eq!(tile.symbol(), sym, "m={m} n={n} k={k}");
        }
        assert_eq!(select_coalesced_nvfp4_tile_for_shape(0, 512, 2_560), None);
        assert!(select_coalesced_nvfp4_tile_for_shape(160, 2_560, 6_144).is_some());
        assert!(select_coalesced_nvfp4_tile_for_shape(256, 16_384, 2_560).is_some());
        assert_eq!(select_coalesced_nvfp4_tile_for_shape(257, 512, 2_560), None);
        assert_eq!(select_coalesced_nvfp4_tile_for_shape(80, 0, 2_560), None);
        assert_eq!(select_coalesced_nvfp4_tile_for_shape(80, 512, 0), None);

        // Handle resolution degrades cleanly from Warp-Split-K / N16 -> N128 -> None,
        // adjusting `m_tile` from M16 to the matching base M-tier on fallback.
        let pref_n16 = select_coalesced_nvfp4_tile(80, 512).expect("n16 m128");
        let pref_sk = select_coalesced_nvfp4_tile_for_shape(20, 320, 10_240).expect("n8_sk8 m16");
        let only_n128 = W4a16CoalescedKernels {
            n8_sk8_m16: KernelHandle(0),
            n16_sk4_m16: KernelHandle(0),
            n32_sk4_m16: KernelHandle(0),
            n64_sk4_m16: KernelHandle(0),
            n32_sk4_m32: KernelHandle(0),
            n64_sk2_m64: KernelHandle(0),
            n16: [KernelHandle(0); 3],
            n32: [KernelHandle(0); 3],
            n64: [KernelHandle(0); 3],
            n128: [KernelHandle(1), KernelHandle(2), KernelHandle(3)],
            v2: W4a16CoalescedV2Kernels::empty(),
        };
        let (resolved_tile, resolved_h) = only_n128.resolve(pref_n16, 80).expect("fallback n128");
        assert_eq!(resolved_tile.n_tile, CoalescedNvfp4NTile::N128);
        assert_eq!(resolved_tile.m_tile, CoalescedNvfp4MTile::M128);
        assert_eq!(resolved_h.0, 3);
        let (sk_fb_tile, sk_fb_h) = only_n128
            .resolve(pref_sk, 20)
            .expect("splitk fallback n128 m32");
        assert_eq!(sk_fb_tile.n_tile, CoalescedNvfp4NTile::N128);
        assert_eq!(sk_fb_tile.m_tile, CoalescedNvfp4MTile::M32);
        assert_eq!(sk_fb_h.0, 1);
        let empty = W4a16CoalescedKernels {
            n8_sk8_m16: KernelHandle(0),
            n16_sk4_m16: KernelHandle(0),
            n32_sk4_m16: KernelHandle(0),
            n64_sk4_m16: KernelHandle(0),
            n32_sk4_m32: KernelHandle(0),
            n64_sk2_m64: KernelHandle(0),
            n16: [KernelHandle(0); 3],
            n32: [KernelHandle(0); 3],
            n64: [KernelHandle(0); 3],
            n128: [KernelHandle(0); 3],
            v2: W4a16CoalescedV2Kernels::empty(),
        };
        assert!(empty.resolve(pref_n16, 80).is_none());
        assert!(empty.resolve(pref_sk, 20).is_none());

        // Scoped batched-verify eligibility requires m in 4..=256, k % 32 == 0, ldc == n, qg == None
        // (preserving C=1 strict 0-ULP scalar GEMV via scope_active=false).
        for m in 0..4 {
            assert!(!coalesced_nvfp4_batched_verify_shape_eligible(
                true, true, m, 16_384, 2_560, 16_384, None
            ));
        }
        for (m, n, k) in [
            (4, 16_384, 2_560),
            (5, 2_560, 6_144),
            (6, 16_384, 2_560),
            (7, 2_560, 6_144),
            (8, 16_384, 2_560),
            (10, 2_560, 6_144),
            (12, 16_384, 2_560),
            (20, 2_560, 6_144),
            (40, 10_240, 320),
            (80, 320, 10_240),
            (128, 16_384, 2_560),
            (160, 16_384, 2_560),
            (256, 2_560, 6_144),
        ] {
            assert!(coalesced_nvfp4_batched_verify_shape_eligible(
                true, true, m, n, k, n, None
            ));
            assert!(!coalesced_nvfp4_batched_verify_shape_eligible(
                false, true, m, n, k, n, None
            ));
            assert!(!coalesced_nvfp4_batched_verify_shape_eligible(
                true, false, m, n, k, n, None
            ));
        }
        assert!(!coalesced_nvfp4_batched_verify_shape_eligible(
            true, true, 257, 16_384, 2_560, 16_384, None
        ));
        assert!(!coalesced_nvfp4_batched_verify_shape_eligible(
            true, true, 80, 16_384, 48, 16_384, None
        ));
        assert!(!coalesced_nvfp4_batched_verify_shape_eligible(
            true, true, 80, 12_288, 2_560, 13_312, None
        ));
        assert!(!coalesced_nvfp4_batched_verify_shape_eligible(
            true,
            true,
            80,
            12_288,
            2_560,
            12_288,
            Some((24, 256))
        ));
        // Empirical GB10 narrow/wide lower crossover: narrow n <= 1024 (moe.router N=512)
        // stays on exact scalar GEMV below 12 rows, while medium/wide n > 1024 (ssm.out /
        // attn.oproj N=2560, plus ssm.qkvz / attn.qkv) engages coalesced/Warp-Split-K NVFP4
        // at m >= 4 (eliminating the C=2 r=4..6 scalar GEMV cliff while C=1 stays unscoped).
        for m in [0usize, 1, 2, 3] {
            assert!(!batched_verify_dense_proj_tc_admits(m, 512));
            assert!(!batched_verify_dense_proj_tc_admits(m, 2_560));
            assert!(!batched_verify_dense_proj_tc_admits(m, 16_384));
        }
        for m in [4usize, 5, 6, 7, 8, 9, 10, 11] {
            assert!(!batched_verify_dense_proj_tc_admits(m, 512));
            assert!(!batched_verify_dense_proj_tc_admits(m, 1_024));
            assert!(batched_verify_dense_proj_tc_admits(m, 2_560));
            assert!(batched_verify_dense_proj_tc_admits(m, 4_096));
            assert!(batched_verify_dense_proj_tc_admits(m, 12_288));
            assert!(batched_verify_dense_proj_tc_admits(m, 16_384));
        }
        for m in [12usize, 20, 40, 80, 128] {
            assert!(batched_verify_dense_proj_tc_admits(m, 512));
            assert!(batched_verify_dense_proj_tc_admits(m, 2_560));
            assert!(batched_verify_dense_proj_tc_admits(m, 16_384));
        }
        // Warp-Split-K + N64 keeps narrow/medium N <= 4096 (ssm.out / attn.oproj N=2560)
        // on zero-VRAM coalesced NVFP4 across the entire 1..=256 batched-verify ceiling,
        // handing off to cached FP8 cuBLASLt only above 256 rows.
        for m in [6usize, 7, 8, 10, 12, 20, 40, 64, 80, 112, 128, 160, 256] {
            assert!(!coalesced_nvfp4_prefers_fp8_cached(m, 2_560));
            assert!(!coalesced_nvfp4_prefers_fp8_cached(m, 4_096));
            assert!(!coalesced_nvfp4_prefers_fp8_cached(m, 16_384));
        }
        for m in [257usize, 512] {
            assert!(coalesced_nvfp4_prefers_fp8_cached(m, 2_560));
            assert!(coalesced_nvfp4_prefers_fp8_cached(m, 4_096));
            assert!(!coalesced_nvfp4_prefers_fp8_cached(m, 16_384));
        }
    }

    #[test]
    fn try_coalesced_nvfp4_w4a16_gemm_launches_zero_alloc_and_fails_closed() {
        use spark_runtime::gpu::mock::MockGpuBackend;

        let gpu = MockGpuBackend::new();
        let w = QuantizedWeight {
            weight: DevicePtr(0x1000),
            weight_scale: DevicePtr(0x2000),
            weight_scale_2: 0.5,
            input_scale: DevicePtr::NULL,
        };
        // Valid shapes (including m=1, odd N=248077, and K=320 non-multiple of 128) launch once with zero alloc/memset.
        for (idx, (m, n, k)) in [
            (1usize, 320usize, 10_240usize),
            (20, 512, 2_560),
            (40, 2_560, 6_144),
            (80, 10_240, 320),
            (80, 16_384, 2_560),
            (128, 248_077, 2_560),
            (160, 2_560, 6_144),
            (256, 16_384, 2_560),
        ]
        .into_iter()
        .enumerate()
        {
            let dispatched = try_coalesced_nvfp4_w4a16_gemm(
                &gpu,
                DevicePtr(0x4000),
                &w,
                DevicePtr(0x8002),
                m,
                n,
                k,
                0,
            )
            .expect("coalesced dispatch");
            assert!(dispatched, "m={m} n={n} k={k}");
            assert_eq!(gpu.launch_count(), idx + 1);
        }
        assert_eq!(gpu.alloc_count(), 0);
        assert_eq!(gpu.memset_count(), 0);

        // Ineligible shapes, misalignments, null pointers, and non-finite scale2 fail closed.
        let launches_before = gpu.launch_count();
        for (act, weight, out, m, n, k) in [
            (DevicePtr(0x4000), w, DevicePtr(0x8000), 0, 512, 2_560),
            (DevicePtr(0x4000), w, DevicePtr(0x8000), 257, 512, 2_560),
            (DevicePtr(0x4000), w, DevicePtr(0x8000), 80, 0, 2_560),
            (DevicePtr(0x4000), w, DevicePtr(0x8000), 80, 512, 0),
            (DevicePtr(0x4000), w, DevicePtr(0x8000), 80, 512, 48),
            (DevicePtr(0x4008), w, DevicePtr(0x8000), 80, 512, 2_560),
            (DevicePtr(0x4000), w, DevicePtr(0x8001), 80, 512, 2_560),
            (DevicePtr::NULL, w, DevicePtr(0x8000), 80, 512, 2_560),
            (DevicePtr(0x4000), w, DevicePtr::NULL, 80, 512, 2_560),
            (
                DevicePtr(0x4000),
                QuantizedWeight {
                    weight: DevicePtr(0x1008),
                    ..w
                },
                DevicePtr(0x8000),
                80,
                512,
                2_560,
            ),
            (
                DevicePtr(0x4000),
                QuantizedWeight {
                    weight_scale: DevicePtr(0x2001),
                    ..w
                },
                DevicePtr(0x8000),
                80,
                512,
                2_560,
            ),
            (
                DevicePtr(0x4000),
                QuantizedWeight {
                    weight_scale_2: f32::NAN,
                    ..w
                },
                DevicePtr(0x8000),
                80,
                512,
                2_560,
            ),
        ] {
            assert!(
                !try_coalesced_nvfp4_w4a16_gemm(&gpu, act, &weight, out, m, n, k, 0)
                    .expect("fail-closed check")
            );
        }
        assert_eq!(gpu.launch_count(), launches_before);

        // Distinct backend instances resolve independently in W4a16CoalescedKernelCache.
        let cache = W4a16CoalescedKernelCache::new();
        let gpu_a = Box::new(MockGpuBackend::new());
        let gpu_b = Box::new(MockGpuBackend::new());
        let loaded_a = cache.get_or_load(gpu_a.as_ref());
        let loaded_b = cache.get_or_load(gpu_b.as_ref());
        assert_ne!(loaded_a.n8_sk8_m16.0, 0);
        assert_ne!(loaded_a.n32_sk4_m32.0, 0);
        assert_ne!(loaded_a.n64_sk2_m64.0, 0);
        assert_ne!(loaded_a.n64[0].0, 0);
        assert_ne!(loaded_a.n16[0].0, 0);
        assert_ne!(loaded_b.n128[2].0, 0);
        assert_ne!(loaded_a.v2.n32_m16.0, 0);
        assert_ne!(loaded_a.v2.n32_m32.0, 0);
        assert_ne!(loaded_a.v2.n64_m64.0, 0);
        assert_ne!(loaded_a.v2.n64_m80.0, 0);
        assert_ne!(loaded_a.v2.n32_sk2_m64.0, 0);
        assert_ne!(loaded_a.v2.n32_sk2_m80.0, 0);
    }

    #[test]
    fn coalesced_v2_tile_selector_and_fallback_cover_m16_m80_and_2cta_shapes() {
        // Default ON; `"0"` opts out cleanly.
        assert!(parse_coalesced_v2(None));
        assert!(parse_coalesced_v2(Some("")));
        assert!(parse_coalesced_v2(Some("1")));
        assert!(!parse_coalesced_v2(Some("0")));
        assert!(!parse_coalesced_v2(Some(" 0 ")));

        for (m, n, k, n_tile, m_tile, co_n, tile_m, threads, sym) in [
            // Non-split-K wide projections (ssm_qkvz, attn_q, lm_head): keep all winning v2 dispatches.
            (
                10,
                16_384,
                2_560,
                CoalescedNvfp4NTile::N32,
                CoalescedNvfp4MTile::M16,
                32,
                16,
                128,
                "w4a16_gemm_coalesced_v2_n32_m16",
            ),
            (
                20,
                16_384,
                2_560,
                CoalescedNvfp4NTile::N32,
                CoalescedNvfp4MTile::M32,
                32,
                32,
                128,
                "w4a16_gemm_coalesced_v2_n32_m32",
            ),
            (
                16,
                96_000,
                2_560,
                CoalescedNvfp4NTile::N64,
                CoalescedNvfp4MTile::M16,
                64,
                16,
                256,
                "w4a16_gemm_coalesced_v2_n64_m16",
            ),
            // N=2560, K=6144 (ssm_out, attn_o): M=7..32 route to v1 split-K (`n32_sk4_m16`/`m32`);
            // M=33..64 route to v2 `n32_sk2_m64` and M=65..80 to v2 `n32_sk2_m80` (80 CTAs, 2 CTAs/SM,
            // full 48-SM saturation, 2-stage K-prefetch).
            (
                10,
                2_560,
                6_144,
                CoalescedNvfp4NTile::N32Sk4,
                CoalescedNvfp4MTile::M16,
                32,
                16,
                512,
                "w4a16_gemm_coalesced_n32_sk4_m16",
            ),
            (
                20,
                2_560,
                6_144,
                CoalescedNvfp4NTile::N32Sk4,
                CoalescedNvfp4MTile::M32,
                32,
                32,
                512,
                "w4a16_gemm_coalesced_n32_sk4_m32",
            ),
            (
                40,
                2_560,
                6_144,
                CoalescedNvfp4NTile::N32Sk2,
                CoalescedNvfp4MTile::M64,
                32,
                64,
                256,
                "w4a16_gemm_coalesced_v2_n32_sk2_m64",
            ),
            (
                64,
                2_560,
                6_144,
                CoalescedNvfp4NTile::N32Sk2,
                CoalescedNvfp4MTile::M64,
                32,
                64,
                256,
                "w4a16_gemm_coalesced_v2_n32_sk2_m64",
            ),
            (
                80,
                2_560,
                6_144,
                CoalescedNvfp4NTile::N32Sk2,
                CoalescedNvfp4MTile::M80,
                32,
                80,
                256,
                "w4a16_gemm_coalesced_v2_n32_sk2_m80",
            ),
            // N <= 1024 (router, DFlash2 Stage-1): all split-K tiers stay on v1.
            (
                16,
                512,
                2_560,
                CoalescedNvfp4NTile::N8Sk8,
                CoalescedNvfp4MTile::M16,
                8,
                16,
                256,
                "w4a16_gemm_coalesced_n8_sk8_m16",
            ),
            (
                20,
                512,
                2_560,
                CoalescedNvfp4NTile::N16Sk4,
                CoalescedNvfp4MTile::M16,
                16,
                16,
                256,
                "w4a16_gemm_coalesced_n16_sk4_m16",
            ),
            (
                40,
                512,
                2_560,
                CoalescedNvfp4NTile::N32Sk4,
                CoalescedNvfp4MTile::M16,
                32,
                16,
                512,
                "w4a16_gemm_coalesced_n32_sk4_m16",
            ),
            // N=16384 (ssm_qkvz) & lm_head (N >= 65536) at M=33..128: N64 v2 (2 CTAs/SM, double-buffered A_128).
            (
                40,
                16_384,
                2_560,
                CoalescedNvfp4NTile::N64,
                CoalescedNvfp4MTile::M64,
                64,
                64,
                256,
                "w4a16_gemm_coalesced_v2_n64_m64",
            ),
            (
                40,
                65_536,
                2_560,
                CoalescedNvfp4NTile::N64,
                CoalescedNvfp4MTile::M64,
                64,
                64,
                256,
                "w4a16_gemm_coalesced_v2_n64_m64",
            ),
            (
                40,
                248_077,
                2_560,
                CoalescedNvfp4NTile::N64,
                CoalescedNvfp4MTile::M64,
                64,
                64,
                256,
                "w4a16_gemm_coalesced_v2_n64_m64",
            ),
            (
                64,
                248_077,
                2_560,
                CoalescedNvfp4NTile::N64,
                CoalescedNvfp4MTile::M64,
                64,
                64,
                256,
                "w4a16_gemm_coalesced_v2_n64_m64",
            ),
            (
                80,
                16_384,
                2_560,
                CoalescedNvfp4NTile::N64,
                CoalescedNvfp4MTile::M80,
                64,
                80,
                256,
                "w4a16_gemm_coalesced_v2_n64_m80",
            ),
            (
                80,
                248_077,
                2_560,
                CoalescedNvfp4NTile::N64,
                CoalescedNvfp4MTile::M80,
                64,
                80,
                256,
                "w4a16_gemm_coalesced_v2_n64_m80",
            ),
            (
                128,
                16_384,
                2_560,
                CoalescedNvfp4NTile::N64,
                CoalescedNvfp4MTile::M128,
                64,
                128,
                256,
                "w4a16_gemm_coalesced_v2_n64_m128",
            ),
        ] {
            let tile =
                select_coalesced_nvfp4_tile_for_shape_v2(m, n, k).expect("valid v2 shape tile");
            assert_eq!(tile.n_tile, n_tile, "m={m} n={n} k={k}");
            assert_eq!(tile.m_tile, m_tile, "m={m} n={n} k={k}");
            assert_eq!(tile.co_n(), co_n, "m={m} n={n} k={k}");
            assert_eq!(tile.tile_m(), tile_m, "m={m} n={n} k={k}");
            assert_eq!(tile.threads(), threads, "m={m} n={n} k={k}");
            assert_eq!(tile.dispatch_symbol(true), sym, "m={m} n={n} k={k}");
        }

        // ATLAS_QWEN4_COALESCED_V2=0 opt-out keeps v1 selector symbols cleanly for o_proj and lm_head.
        assert_eq!(
            select_coalesced_nvfp4_tile_for_shape(40, 2_560, 6_144)
                .unwrap()
                .dispatch_symbol(false),
            "w4a16_gemm_coalesced_n64_sk2_m64"
        );
        assert_eq!(
            select_coalesced_nvfp4_tile_for_shape(40, 248_077, 2_560)
                .unwrap()
                .dispatch_symbol(false),
            "w4a16_gemm_coalesced_m64"
        );

        // Missing v2 handles degrade cleanly to matching v1 handles/tiles.
        let v1_only = W4a16CoalescedKernels {
            n8_sk8_m16: KernelHandle(1),
            n16_sk4_m16: KernelHandle(2),
            n32_sk4_m16: KernelHandle(3),
            n64_sk4_m16: KernelHandle(4),
            n32_sk4_m32: KernelHandle(5),
            n64_sk2_m64: KernelHandle(6),
            n16: [KernelHandle(10), KernelHandle(11), KernelHandle(12)],
            n32: [KernelHandle(20), KernelHandle(21), KernelHandle(22)],
            n64: [KernelHandle(30), KernelHandle(31), KernelHandle(32)],
            n128: [KernelHandle(40), KernelHandle(41), KernelHandle(42)],
            v2: W4a16CoalescedV2Kernels::empty(),
        };
        let pref_m80 = select_coalesced_nvfp4_tile_for_shape_v2(80, 16_384, 2_560).unwrap();
        let (fb_t, fb_h) = v1_only.resolve_with_v2(pref_m80, 80, true).unwrap();
        assert_eq!(fb_t.n_tile, CoalescedNvfp4NTile::N64);
        assert_eq!(fb_t.m_tile, CoalescedNvfp4MTile::M128);
        assert_eq!(fb_h.0, 32);

        let pref_sk2_64 = select_coalesced_nvfp4_tile_for_shape_v2(40, 2_560, 6_144).unwrap();
        assert_eq!(pref_sk2_64.n_tile, CoalescedNvfp4NTile::N32Sk2);
        assert_eq!(pref_sk2_64.m_tile, CoalescedNvfp4MTile::M64);
        let (fb_t2, fb_h2) = v1_only.resolve_with_v2(pref_sk2_64, 40, true).unwrap();
        assert_eq!(fb_t2.n_tile, CoalescedNvfp4NTile::N64Sk2);
        assert_eq!(fb_t2.m_tile, CoalescedNvfp4MTile::M64);
        assert_eq!(fb_h2.0, 6);

        let pref_sk2_80 = select_coalesced_nvfp4_tile_for_shape_v2(80, 2_560, 6_144).unwrap();
        let (fb_t3, fb_h3) = v1_only.resolve_with_v2(pref_sk2_80, 80, true).unwrap();
        assert_eq!(fb_t3.n_tile, CoalescedNvfp4NTile::N64);
        assert_eq!(fb_t3.m_tile, CoalescedNvfp4MTile::M128);
        assert_eq!(fb_h3.0, 32);

        // When all v2 handles are populated, non-split-K + N32Sk2 M64/M80 dispatch v2,
        // while v1 split-K shapes (N32Sk4 M16/M32, N <= 1024) dispatch v1.
        let with_v2 = W4a16CoalescedKernels {
            v2: W4a16CoalescedV2Kernels {
                n32_m16: KernelHandle(91),
                n64_m16: KernelHandle(92),
                n32_m32: KernelHandle(93),
                n64_m32: KernelHandle(94),
                n64_m64: KernelHandle(95),
                n64_m80: KernelHandle(99),
                n64_m128: KernelHandle(96),
                n32_sk2_m64: KernelHandle(98),
                n32_sk2_m80: KernelHandle(97),
                n8_sk8_m32_grid_v2: KernelHandle(0),
            },
            ..v1_only
        };
        let (v2_t, v2_h) = with_v2.resolve_with_v2(pref_m80, 80, true).unwrap();
        assert_eq!(v2_t, pref_m80);
        assert_eq!(v2_h.0, 99);
        let (v1_t, v1_h) = with_v2.resolve_with_v2(pref_m80, 80, false).unwrap();
        assert_eq!(v1_t.m_tile, CoalescedNvfp4MTile::M128);
        assert_eq!(v1_h.0, 32);

        let (sk2_80_t, sk2_80_h) = with_v2.resolve_with_v2(pref_sk2_80, 80, true).unwrap();
        assert_eq!(sk2_80_t, pref_sk2_80);
        assert_eq!(sk2_80_h.0, 97);

        let pref_sk4_16 = select_coalesced_nvfp4_tile_for_shape_v2(10, 2_560, 6_144).unwrap();
        let (sk4_16_t, sk4_16_h) = with_v2.resolve_with_v2(pref_sk4_16, 10, true).unwrap();
        assert_eq!(sk4_16_t, pref_sk4_16);
        assert_eq!(sk4_16_h.0, 3);

        let pref_sk4_32 = select_coalesced_nvfp4_tile_for_shape_v2(20, 2_560, 6_144).unwrap();
        let (sk4_32_t, sk4_32_h) = with_v2.resolve_with_v2(pref_sk4_32, 20, true).unwrap();
        assert_eq!(sk4_32_t, pref_sk4_32);
        assert_eq!(sk4_32_h.0, 5);

        let (sk2_64_t, sk2_64_h) = with_v2.resolve_with_v2(pref_sk2_64, 40, true).unwrap();
        assert_eq!(sk2_64_t, pref_sk2_64);
        assert_eq!(sk2_64_h.0, 98);
        let (sk2_64_v1_t, sk2_64_v1_h) = with_v2.resolve_with_v2(pref_sk2_64, 40, false).unwrap();
        assert_eq!(sk2_64_v1_t.n_tile, CoalescedNvfp4NTile::N64Sk2);
        assert_eq!(sk2_64_v1_t.m_tile, CoalescedNvfp4MTile::M64);
        assert_eq!(sk2_64_v1_h.0, 6);

        let pref_lm_40 = select_coalesced_nvfp4_tile_for_shape_v2(40, 248_077, 2_560).unwrap();
        let (lm_40_t, lm_40_h) = with_v2.resolve_with_v2(pref_lm_40, 40, true).unwrap();
        assert_eq!(lm_40_t, pref_lm_40);
        assert_eq!(lm_40_h.0, 95);

        let pref_router_40 = select_coalesced_nvfp4_tile_for_shape_v2(40, 512, 2_560).unwrap();
        let (router_t, router_h) = with_v2.resolve_with_v2(pref_router_40, 40, true).unwrap();
        assert_eq!(router_t, pref_router_40);
        assert_eq!(router_h.0, 3);
    }

    #[test]
    fn coalesced_v2_sm_aware_selector_preserves_gb10_and_boosts_sm120_occupancy() {
        assert!(parse_coalesced_sm_tiles(None));
        assert!(parse_coalesced_sm_tiles(Some("")));
        assert!(parse_coalesced_sm_tiles(Some("1")));
        assert!(!parse_coalesced_sm_tiles(Some("0")));
        assert!(!parse_coalesced_sm_tiles(Some(" 0 ")));

        // Out-of-bounds shapes return None on both SM tiers.
        for sm in [48u32, 64, 188] {
            assert_eq!(
                select_coalesced_nvfp4_tile_for_shape_v2_with_sms(0, 2_560, 6_144, sm),
                None
            );
            assert_eq!(
                select_coalesced_nvfp4_tile_for_shape_v2_with_sms(257, 2_560, 6_144, sm),
                None
            );
            assert_eq!(
                select_coalesced_nvfp4_tile_for_shape_v2_with_sms(16, 0, 6_144, sm),
                None
            );
            assert_eq!(
                select_coalesced_nvfp4_tile_for_shape_v2_with_sms(16, 2_560, 0, sm),
                None
            );
        }

        // `sm_count <= 64` matches legacy GB10 `select_coalesced_nvfp4_tile_for_shape_v2` identically.
        for sm in [48u32, 64] {
            for m in [1usize, 6, 10, 16, 20, 32, 40, 64, 80, 96, 128, 160, 256] {
                for (n, k) in [
                    (320usize, 10_240usize),
                    (512, 2_560),
                    (2_560, 1_024),
                    (2_560, 6_144),
                    (2_564, 6_144),
                    (10_240, 320),
                    (12_288, 2_560),
                    (16_384, 2_560),
                    (96_000, 2_560),
                    (248_077, 2_560),
                ] {
                    assert_eq!(
                        select_coalesced_nvfp4_tile_for_shape_v2_with_sms(m, n, k, sm),
                        select_coalesced_nvfp4_tile_for_shape_v2(m, n, k),
                        "sm={sm} m={m} n={n} k={k}"
                    );
                }
            }
        }

        // SM120 (`sm_count == 188`) selects higher-CTA tiles with valid kernel symbols.
        let v1_only = W4a16CoalescedKernels {
            n8_sk8_m16: KernelHandle(1),
            n16_sk4_m16: KernelHandle(2),
            n32_sk4_m16: KernelHandle(3),
            n64_sk4_m16: KernelHandle(4),
            n32_sk4_m32: KernelHandle(5),
            n64_sk2_m64: KernelHandle(6),
            n16: [KernelHandle(10), KernelHandle(11), KernelHandle(12)],
            n32: [KernelHandle(20), KernelHandle(21), KernelHandle(22)],
            n64: [KernelHandle(30), KernelHandle(31), KernelHandle(32)],
            n128: [KernelHandle(40), KernelHandle(41), KernelHandle(42)],
            v2: W4a16CoalescedV2Kernels::empty(),
        };
        let with_v2 = W4a16CoalescedKernels {
            v2: W4a16CoalescedV2Kernels {
                n32_m16: KernelHandle(91),
                n64_m16: KernelHandle(92),
                n32_m32: KernelHandle(93),
                n64_m32: KernelHandle(94),
                n64_m64: KernelHandle(95),
                n64_m80: KernelHandle(99),
                n64_m128: KernelHandle(96),
                n32_sk2_m64: KernelHandle(98),
                n32_sk2_m80: KernelHandle(97),
                n8_sk8_m32_grid_v2: KernelHandle(0),
            },
            ..v1_only
        };

        for (m, n, k, exp_n, exp_m, exp_sym, min_ctas) in [
            // ssm.out / attn.oproj (N=2560, K=6144): 160-320 CTAs across m=1..=128 (vs 40-80 on GB10).
            (
                10usize,
                2_560usize,
                6_144usize,
                CoalescedNvfp4NTile::N8Sk8,
                CoalescedNvfp4MTile::M16,
                "w4a16_gemm_coalesced_n8_sk8_m16",
                320u32,
            ),
            (
                16,
                2_560,
                6_144,
                CoalescedNvfp4NTile::N8Sk8,
                CoalescedNvfp4MTile::M16,
                "w4a16_gemm_coalesced_n8_sk8_m16",
                320,
            ),
            // Non-multiple-of-8 N keeps N32Sk4 M16 at m <= 16.
            (
                16,
                2_564,
                6_144,
                CoalescedNvfp4NTile::N32Sk4,
                CoalescedNvfp4MTile::M16,
                "w4a16_gemm_coalesced_n32_sk4_m16",
                81,
            ),
            (
                20,
                2_560,
                6_144,
                CoalescedNvfp4NTile::N16Sk4,
                CoalescedNvfp4MTile::M16,
                "w4a16_gemm_coalesced_n16_sk4_m16",
                320,
            ),
            (
                32,
                2_560,
                6_144,
                CoalescedNvfp4NTile::N16Sk4,
                CoalescedNvfp4MTile::M16,
                "w4a16_gemm_coalesced_n16_sk4_m16",
                320,
            ),
            (
                40,
                2_560,
                6_144,
                CoalescedNvfp4NTile::N32Sk4,
                CoalescedNvfp4MTile::M32,
                "w4a16_gemm_coalesced_n32_sk4_m32",
                160,
            ),
            (
                64,
                2_560,
                6_144,
                CoalescedNvfp4NTile::N32Sk4,
                CoalescedNvfp4MTile::M32,
                "w4a16_gemm_coalesced_n32_sk4_m32",
                160,
            ),
            (
                80,
                2_560,
                6_144,
                CoalescedNvfp4NTile::N32Sk2,
                CoalescedNvfp4MTile::M64,
                "w4a16_gemm_coalesced_v2_n32_sk2_m64",
                160,
            ),
            (
                128,
                2_560,
                6_144,
                CoalescedNvfp4NTile::N32Sk2,
                CoalescedNvfp4MTile::M64,
                "w4a16_gemm_coalesced_v2_n32_sk2_m64",
                160,
            ),
            (
                160,
                2_560,
                6_144,
                CoalescedNvfp4NTile::N32Sk2,
                CoalescedNvfp4MTile::M64,
                "w4a16_gemm_coalesced_v2_n32_sk2_m64",
                240,
            ),
            (
                256,
                2_560,
                6_144,
                CoalescedNvfp4NTile::N32Sk2,
                CoalescedNvfp4MTile::M64,
                "w4a16_gemm_coalesced_v2_n32_sk2_m64",
                320,
            ),
            // Wide projections (hc.up=10240, attn.q=12288, ssm.qkvz=16384) at m in 33..=64:
            // v2_n32_m32 launches 640..1024 CTAs (4 CTAs/SM, 2-stage K-prefetch).
            (
                40,
                10_240,
                320,
                CoalescedNvfp4NTile::N32,
                CoalescedNvfp4MTile::M32,
                "w4a16_gemm_coalesced_v2_n32_m32",
                640,
            ),
            (
                48,
                12_288,
                2_560,
                CoalescedNvfp4NTile::N32,
                CoalescedNvfp4MTile::M32,
                "w4a16_gemm_coalesced_v2_n32_m32",
                768,
            ),
            (
                64,
                16_384,
                2_560,
                CoalescedNvfp4NTile::N32,
                CoalescedNvfp4MTile::M32,
                "w4a16_gemm_coalesced_v2_n32_m32",
                1_024,
            ),
            // lm_head (> 24576) stays on N64 v2 (already > 3800 CTAs).
            (
                40,
                248_077,
                2_560,
                CoalescedNvfp4NTile::N64,
                CoalescedNvfp4MTile::M64,
                "w4a16_gemm_coalesced_v2_n64_m64",
                3_877,
            ),
        ] {
            let tile = select_coalesced_nvfp4_tile_for_shape_v2_with_sms(m, n, k, 188)
                .expect("valid SM120 tile");
            assert_eq!(tile.n_tile, exp_n, "m={m} n={n} k={k}");
            assert_eq!(tile.m_tile, exp_m, "m={m} n={n} k={k}");
            assert_eq!(tile.dispatch_symbol(true), exp_sym, "m={m} n={n} k={k}");
            let ctas = div_ceil(n as u32, tile.co_n()) * div_ceil(m as u32, tile.tile_m());
            assert!(
                ctas >= min_ctas,
                "m={m} n={n} k={k}: got {ctas} CTAs < {min_ctas}"
            );
            let (resolved, handle) = with_v2
                .resolve_with_v2(tile, m, true)
                .expect("SM120 tile must resolve with v2 handles");
            assert_eq!(resolved, tile, "m={m} n={n} k={k}");
            assert_ne!(handle.0, 0);
            // Also verify clean v1 fallback when v2 handles are absent.
            let (_, fb_handle) = v1_only
                .resolve_with_v2(tile, m, true)
                .expect("SM120 tile must degrade cleanly without v2 handles");
            assert_ne!(fb_handle.0, 0);
        }
    }

    #[test]
    fn shared_expert_nvfp4_t_fast_v2_selects_tiles_launches_zero_alloc_and_fails_closed() {
        use spark_runtime::gpu::mock::MockGpuBackend;

        // C=1 (m=1), m=0, and m>256 are rejected so C=1 greedy parity is untouched.
        assert_eq!(select_shared_expert_nvfp4_t_tiles(0), None);
        assert_eq!(select_shared_expert_nvfp4_t_tiles(1), None);
        assert!(select_shared_expert_nvfp4_t_tiles(129).is_some());
        assert!(select_shared_expert_nvfp4_t_tiles(256).is_some());
        assert_eq!(select_shared_expert_nvfp4_t_tiles(257), None);

        for (m, exp_gu, exp_dn, gu_sym, dn_sym) in [
            (
                2usize,
                SharedExpertGateUpTile::Sk4M16,
                SharedExpertDownTile::M16,
                "w4a16_shared_gateup_silu_t_n16_sk4_m16",
                "w4a16_shared_down_t_n32_m16",
            ),
            (
                8,
                SharedExpertGateUpTile::Sk4M16,
                SharedExpertDownTile::M16,
                "w4a16_shared_gateup_silu_t_n16_sk4_m16",
                "w4a16_shared_down_t_n32_m16",
            ),
            (
                10,
                SharedExpertGateUpTile::Sk4M16,
                SharedExpertDownTile::M16,
                "w4a16_shared_gateup_silu_t_n16_sk4_m16",
                "w4a16_shared_down_t_n32_m16",
            ),
            (
                16,
                SharedExpertGateUpTile::Sk4M16,
                SharedExpertDownTile::M16,
                "w4a16_shared_gateup_silu_t_n16_sk4_m16",
                "w4a16_shared_down_t_n32_m16",
            ),
            (
                17,
                SharedExpertGateUpTile::Sk4M32,
                SharedExpertDownTile::M32,
                "w4a16_shared_gateup_silu_t_n16_sk4_m32",
                "w4a16_shared_down_t_n32_m32",
            ),
            (
                32,
                SharedExpertGateUpTile::Sk4M32,
                SharedExpertDownTile::M32,
                "w4a16_shared_gateup_silu_t_n16_sk4_m32",
                "w4a16_shared_down_t_n32_m32",
            ),
            (
                48,
                SharedExpertGateUpTile::Sk4M32,
                SharedExpertDownTile::M64,
                "w4a16_shared_gateup_silu_t_n16_sk4_m32",
                "w4a16_shared_down_t_n32_m64",
            ),
            (
                49,
                SharedExpertGateUpTile::Sk2M64,
                SharedExpertDownTile::M64,
                "w4a16_shared_gateup_silu_t_n16_sk2_m64",
                "w4a16_shared_down_t_n32_m64",
            ),
            (
                64,
                SharedExpertGateUpTile::Sk2M64,
                SharedExpertDownTile::M64,
                "w4a16_shared_gateup_silu_t_n16_sk2_m64",
                "w4a16_shared_down_t_n32_m64",
            ),
            (
                80,
                SharedExpertGateUpTile::Sk2M64,
                SharedExpertDownTile::M80,
                "w4a16_shared_gateup_silu_t_n16_sk2_m64",
                "w4a16_shared_down_t_n32_m80",
            ),
            (
                96,
                SharedExpertGateUpTile::Sk2M64,
                SharedExpertDownTile::M128,
                "w4a16_shared_gateup_silu_t_n16_sk2_m64",
                "w4a16_shared_down_t_n32_m128",
            ),
            (
                128,
                SharedExpertGateUpTile::Sk2M64,
                SharedExpertDownTile::M128,
                "w4a16_shared_gateup_silu_t_n16_sk2_m64",
                "w4a16_shared_down_t_n32_m128",
            ),
        ] {
            let (gu, dn) = select_shared_expert_nvfp4_t_tiles(m).expect("valid tile pair");
            assert_eq!(gu, exp_gu, "m={m}");
            assert_eq!(dn, exp_dn, "m={m}");
            assert_eq!(gu.symbol(), gu_sym, "m={m}");
            assert_eq!(dn.symbol(), dn_sym, "m={m}");
            assert_eq!(gu.co_n(), 16);
            assert_eq!(dn.co_n(), 32);
        }
        assert_eq!(SharedExpertGateUpTile::Sk2M64.tile_m(), 64);
        assert_eq!(SharedExpertGateUpTile::Sk2M64.threads(), 128);

        // Missing handles fail closed cleanly; missing Sk2M64 falls back to Sk2M32 when present.
        let partial = W4a16SharedFastV2Kernels {
            gu_sk4_m16: KernelHandle(1),
            gu_sk4_m32: KernelHandle(2),
            gu_sk2_m32: KernelHandle(0),
            gu_sk2_m64: KernelHandle(0),
            dn_m16: KernelHandle(4),
            dn_m32: KernelHandle(5),
            dn_m64: KernelHandle(6),
            dn_m80: KernelHandle(0),
            dn_m128: KernelHandle(8),
        };
        assert!(partial.resolve(10).is_some());
        assert!(partial.resolve(48).is_some());
        assert!(partial.resolve(64).is_none());
        assert!(partial.resolve(80).is_none());

        let fallback_m32 = W4a16SharedFastV2Kernels {
            gu_sk2_m32: KernelHandle(3),
            ..partial
        };
        let resolved_fb = fallback_m32
            .resolve(64)
            .expect("falls back to Sk2M32 when Sk2M64 is absent");
        assert_eq!(resolved_fb.gu_tile, SharedExpertGateUpTile::Sk2M32);
        assert_eq!(resolved_fb.gu_kernel.0, 3);

        // MockGpuBackend launches 2 kernels per call with zero allocations/memsets.
        let gpu = MockGpuBackend::new();
        warm_coalesced_nvfp4_kernels(&gpu);
        assert!(!shared_expert_nvfp4_t_fast_v2_available(&gpu, 1));
        assert!(shared_expert_nvfp4_t_fast_v2_available(&gpu, 10));
        assert!(shared_expert_nvfp4_t_fast_v2_available(&gpu, 80));

        let w = QuantizedWeight {
            weight: DevicePtr(0x1000),
            weight_scale: DevicePtr(0x2000),
            weight_scale_2: 0.25,
            input_scale: DevicePtr::NULL,
        };
        for (idx, m) in [2usize, 8, 10, 16, 32, 48, 64, 80, 96, 128, 160, 256]
            .into_iter()
            .enumerate()
        {
            let launched = try_shared_expert_nvfp4_t_fast_v2(
                &gpu,
                DevicePtr(0x4000),
                &w,
                &w,
                &w,
                DevicePtr(0x6000),
                DevicePtr(0x8000),
                m,
                640,
                2_560,
                0,
            )
            .expect("shared fast v2 dispatch");
            assert!(launched, "m={m}");
            assert_eq!(gpu.launch_count(), (idx + 1) * 2);
        }
        assert_eq!(gpu.alloc_count(), 0);
        assert_eq!(gpu.memset_count(), 0);

        // Fail-closed preconditions never launch kernels.
        let launches_before = gpu.launch_count();
        for (input, gate, act, out, m, inter, hidden) in [
            (
                DevicePtr(0x4000),
                w,
                DevicePtr(0x6000),
                DevicePtr(0x8000),
                1,
                640,
                2_560,
            ),
            (
                DevicePtr(0x4000),
                w,
                DevicePtr(0x6000),
                DevicePtr(0x8000),
                257,
                640,
                2_560,
            ),
            (
                DevicePtr(0x4008),
                w,
                DevicePtr(0x6000),
                DevicePtr(0x8000),
                10,
                640,
                2_560,
            ),
            (
                DevicePtr(0x4000),
                w,
                DevicePtr(0x6008),
                DevicePtr(0x8000),
                10,
                640,
                2_560,
            ),
            (
                DevicePtr(0x4000),
                w,
                DevicePtr(0x6000),
                DevicePtr(0x8008),
                10,
                640,
                2_560,
            ),
            (
                DevicePtr::NULL,
                w,
                DevicePtr(0x6000),
                DevicePtr(0x8000),
                10,
                640,
                2_560,
            ),
            (
                DevicePtr(0x4000),
                w,
                DevicePtr(0x6000),
                DevicePtr(0x8000),
                10,
                632,
                2_560,
            ),
            (
                DevicePtr(0x4000),
                w,
                DevicePtr(0x6000),
                DevicePtr(0x8000),
                10,
                640,
                2_544,
            ),
            (
                DevicePtr(0x4000),
                QuantizedWeight {
                    weight_scale: DevicePtr(0x2008),
                    ..w
                },
                DevicePtr(0x6000),
                DevicePtr(0x8000),
                10,
                640,
                2_560,
            ),
            (
                DevicePtr(0x4000),
                QuantizedWeight {
                    weight_scale_2: f32::NAN,
                    ..w
                },
                DevicePtr(0x6000),
                DevicePtr(0x8000),
                10,
                640,
                2_560,
            ),
        ] {
            assert!(
                !try_shared_expert_nvfp4_t_fast_v2(
                    &gpu, input, &gate, &w, &w, act, out, m, inter, hidden, 0
                )
                .expect("fail-closed check")
            );
        }
        assert_eq!(gpu.launch_count(), launches_before);

        // Distinct backend instances resolve independently in W4a16SharedFastV2KernelCache.
        let cache = W4a16SharedFastV2KernelCache::new();
        let gpu_a = Box::new(MockGpuBackend::new());
        let gpu_b = Box::new(MockGpuBackend::new());
        assert!(cache.get_or_load(gpu_a.as_ref()).resolve(10).is_some());
        assert!(cache.get_or_load(gpu_b.as_ref()).resolve(80).is_some());
    }

    #[test]
    fn narrow_router_grid_split_k_preserves_gb10_and_saturates_sm120() {
        // GB10 (<= 64 SMs) never engages narrow grid-Split-K.
        for sm in [48u32, 64] {
            for m in [1usize, 4, 8, 9, 16] {
                assert_eq!(
                    select_narrow_grid_split_k_z(m, 512, 2560, sm, true, true, true),
                    None
                );
            }
        }
        // SM120 (188 SMs): Router GEMM N=512, K=2560, M<=16 selects grid_z=3 -> 64*3 = 192 CTAs.
        for m in [1usize, 4, 8, 9, 16] {
            let gz = select_narrow_grid_split_k_z(m, 512, 2560, 188, true, true, true);
            assert_eq!(gz, Some(3), "m={m}");
            assert_eq!(div_ceil(512, 8) * gz.unwrap(), 192);
        }
        // Guard rails: M > 16, N > 1024, K < 2048, K % 256 != 0, missing kernel/scratch, v2 off.
        assert_eq!(
            select_narrow_grid_split_k_z(17, 512, 2560, 188, true, true, true),
            None
        );
        assert_eq!(
            select_narrow_grid_split_k_z(8, 1025, 2560, 188, true, true, true),
            None
        );
        assert_eq!(
            select_narrow_grid_split_k_z(8, 512, 1024, 188, true, true, true),
            None
        );
        assert_eq!(
            select_narrow_grid_split_k_z(8, 512, 2624, 188, true, true, true),
            None
        );
        assert_eq!(
            select_narrow_grid_split_k_z(8, 512, 2560, 188, false, true, true),
            None
        );
        assert_eq!(
            select_narrow_grid_split_k_z(8, 512, 2560, 188, true, false, true),
            None
        );
        assert_eq!(
            select_narrow_grid_split_k_z(8, 512, 2560, 188, true, true, false),
            None
        );
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn gpu_parity_narrow_router_grid_split_k_n512_k2560() {
        use spark_runtime::cuda_backend::AtlasCudaBackend;

        let Some(ptx) = atlas_kernels::ptx_for_model("flash-next") else {
            eprintln!("skipping GPU parity test: flash-next PTX unavailable");
            return;
        };
        let Ok(gpu) = AtlasCudaBackend::new(0, &ptx.modules) else {
            eprintln!("skipping GPU parity test: CUDA device unavailable");
            return;
        };
        let u16_as_bytes = |s: &[u16]| -> &[u8] {
            // SAFETY: `s` is a valid initialized `&[u16]` POD slice with no padding bytes.
            unsafe { std::slice::from_raw_parts(s.as_ptr().cast::<u8>(), std::mem::size_of_val(s)) }
        };
        let u16_as_bytes_mut = |s: &mut [u16]| -> &mut [u8] {
            // SAFETY: `s` is a valid initialized `&mut [u16]` POD slice with no padding bytes;
            // any written byte pattern is a valid `u16`.
            unsafe {
                std::slice::from_raw_parts_mut(
                    s.as_mut_ptr().cast::<u8>(),
                    std::mem::size_of_val(s),
                )
            }
        };
        let kernels = probe_w4a16_coalesced_kernels(&gpu);
        if kernels.n8_sk8_m16.0 == 0 || kernels.v2.n8_sk8_m32_grid_v2.0 == 0 {
            eprintln!("skipping GPU parity test: w4a16 coalesced kernels unavailable");
            return;
        }
        let scratch =
            alloc_narrow_grid_splitk_scratch(&gpu).expect("narrow grid-Split-K scratch allocation");
        let stream = gpu.default_stream();
        let n = 512usize;
        let k = 2560usize;
        let max_m = 16usize;
        let canary_elems = 1024usize;
        let canary_u16: u16 = 0xBEEF;

        let bf16_from_f32 = |v: f32| -> u16 { (v.to_bits() >> 16) as u16 };
        let f32_from_bf16 = |b: u16| -> f32 { f32::from_bits((b as u32) << 16) };

        let mut act_host = vec![0u16; max_m * k];
        for (i, slot) in act_host.iter_mut().enumerate() {
            let v = (((i.wrapping_mul(41) % 97) as f32) - 48.0) * 0.015;
            *slot = bf16_from_f32(v);
        }
        let mut packed_host = vec![0u8; n * (k / 2)];
        for (i, b) in packed_host.iter_mut().enumerate() {
            let lo = ((i.wrapping_mul(19) + 3) & 0x7) as u8 | (((i >> 3) & 1) as u8) << 3;
            let hi = ((i.wrapping_mul(29) + 5) & 0x7) as u8 | (((i >> 4) & 1) as u8) << 3;
            *b = lo | (hi << 4);
        }
        let mut scale_host = vec![0u8; n * (k / 16)];
        for (i, b) in scale_host.iter_mut().enumerate() {
            *b = 0x30u8 + ((i.wrapping_mul(11) % 16) as u8);
        }
        let scale2: f32 = 0.0625;

        let act_dev = gpu.alloc(act_host.len() * 2).unwrap();
        let packed_dev = gpu.alloc(packed_host.len()).unwrap();
        let scale_dev = gpu.alloc(scale_host.len()).unwrap();
        let out_ref_dev = gpu.alloc((max_m * n + canary_elems) * 2).unwrap();
        let out_sk_dev = gpu.alloc((max_m * n + canary_elems) * 2).unwrap();

        gpu.copy_h2d_async(u16_as_bytes(&act_host), act_dev, stream)
            .unwrap();
        gpu.copy_h2d_async(&packed_host, packed_dev, stream)
            .unwrap();
        gpu.copy_h2d_async(&scale_host, scale_dev, stream).unwrap();

        for &m in &[1usize, 4, 8, 9, 16] {
            let total_elems = m * n + canary_elems;
            let canary_init = vec![canary_u16; total_elems];
            gpu.copy_h2d_async(u16_as_bytes(&canary_init), out_ref_dev, stream)
                .unwrap();
            gpu.copy_h2d_async(u16_as_bytes(&canary_init), out_sk_dev, stream)
                .unwrap();

            KernelLaunch::new(&gpu, kernels.n8_sk8_m16)
                .grid([div_ceil(n as u32, 8), 1, 1])
                .block([256, 1, 1])
                .arg_ptr(act_dev)
                .arg_ptr(packed_dev)
                .arg_ptr(scale_dev)
                .arg_f32(scale2)
                .arg_ptr(out_ref_dev)
                .arg_u32(m as u32)
                .arg_u32(n as u32)
                .arg_u32(k as u32)
                .launch(stream)
                .unwrap();

            let gz = select_narrow_grid_split_k_z(m, n, k, 188, true, true, true).unwrap();
            assert_eq!(gz, 3);
            KernelLaunch::new(&gpu, kernels.v2.n8_sk8_m32_grid_v2)
                .grid([div_ceil(n as u32, 8), 1, gz])
                .block([256, 1, 1])
                .pdl(true)
                .arg_ptr(act_dev)
                .arg_ptr(DevicePtr::NULL)
                .arg_ptr(DevicePtr::NULL)
                .arg_f32(0.0)
                .arg_ptr(DevicePtr::NULL)
                .arg_u32(0)
                .arg_ptr(packed_dev)
                .arg_ptr(scale_dev)
                .arg_f32(scale2)
                .arg_ptr(out_sk_dev)
                .arg_u32(n as u32)
                .arg_u32(m as u32)
                .arg_u32(k as u32)
                .arg_f32(1.0)
                .arg_ptr(scratch.partials)
                .arg_ptr(scratch.counters)
                .launch(stream)
                .unwrap();

            gpu.synchronize(stream).unwrap();

            let mut out_ref = vec![0u16; total_elems];
            let mut out_sk = vec![0u16; total_elems];
            gpu.copy_d2h(out_ref_dev, u16_as_bytes_mut(&mut out_ref))
                .unwrap();
            gpu.copy_d2h(out_sk_dev, u16_as_bytes_mut(&mut out_sk))
                .unwrap();

            // Zero OOB writes past [m, n].
            for idx in (m * n)..total_elems {
                assert_eq!(out_sk[idx], canary_u16, "OOB write at elem {idx} for m={m}");
            }

            let mut max_ulp = 0u32;
            let mut nonzero = 0usize;
            for idx in 0..(m * n) {
                let a = out_ref[idx];
                let b = out_sk[idx];
                let fa = f32_from_bf16(a);
                let fb = f32_from_bf16(b);
                assert!(fa.is_finite() && fb.is_finite());
                if fa != 0.0 {
                    nonzero += 1;
                }
                let ulp = if a == b || (fa == 0.0 && fb == 0.0) {
                    0
                } else if (a ^ b) & 0x8000 == 0 {
                    a.abs_diff(b) as u32
                } else {
                    ((a & 0x7fff) as u32) + ((b & 0x7fff) as u32)
                };
                max_ulp = max_ulp.max(ulp);
            }
            assert!(
                nonzero > (m * n) / 2,
                "expected non-trivial logits for m={m}"
            );
            assert!(
                max_ulp <= 1,
                "narrow router grid-Split-K parity failed for m={m}: max_ulp={max_ulp}"
            );
        }

        for ptr in [
            act_dev,
            packed_dev,
            scale_dev,
            out_ref_dev,
            out_sk_dev,
            scratch.partials,
            scratch.counters,
        ] {
            let _ = gpu.free(ptr);
        }
    }
}
