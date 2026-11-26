// SPDX-License-Identifier: AGPL-3.0-only

//! HC_FUSED's up+mix moved toward its DRAM roof
//! (`ATLAS_QWEN4_PREFILL_MHC_ROOF=1|t64|t128`, default off; K9 item Y).
//!
//! Where the HC_FUSED level-2 path would launch `qwen4_hc_upmix_hcfused[_amax]`
//! (lever `upmix`), an admitted preparation launches
//! `qwen4_hc_upmix_roof_<tile>[_amax]` (module `qwen4_hc_upmix_roof`) with the
//! same arguments instead. Byte-identical by construction: the same
//! `mma.sync` chains over ascending k and the same epilogue expressions; only
//! the data movement changes (hyper' requested into L2 at CTA start, 16-byte
//! staged epilogue copies and stores, L2 priorities, and for `t128` 128-row
//! tiles on a 3-stage ring). Proven by micro 146 (`mixed` and the absmax bits
//! byte-compared against the incumbent at 77 / 2048 / 8192 / 16384 rows, with
//! negative controls); `t128` (the default) runs at 87-88 % of the DRAM roof.
//! Receipts: `MHC_ROOF_ENGAGED ... kernel=<entry>` once per entry,
//! `MHC_ROOF_REFUSED ... reason=<...>` once per reason (the incumbent runs).

use std::cell::Cell;
use std::sync::{Mutex, Once, OnceLock};

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

use super::prefill_upmix_roof_plan::{
    self as plan, FP8_ACT_FROM_BF16_ENTRY, Io, MXFP8_FUSED_SELECTOR, MXFP8_GATE_MAX_Y_TILES,
    Mxfp8GateIo, OutputCachePolicy, Refusal, SELECTOR, THREADS, Tile,
};

/// Parsed once per process (`ATLAS_QWEN4_PREFILL_UPMIX_MXFP8_FUSED`, default ON);
/// a malformed value fails the first prefill.
pub(crate) fn mxfp8_fused_enabled() -> Result<bool> {
    static SEL: OnceLock<Result<bool, String>> = OnceLock::new();
    SEL.get_or_init(|| plan::parse_mxfp8_fused(std::env::var(MXFP8_FUSED_SELECTOR).ok().as_deref()))
        .clone()
        .map_err(anyhow::Error::msg)
}

/// One-shot target buffers for fusing MXFP8 `(xq, xs)` quantization and
/// optional Last-CTA L2-hot shared-expert `gate_scalars` into `upmix_roof`.
#[allow(dead_code)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct UpmixMxfp8Request {
    pub xq: DevicePtr,
    pub xs: DevicePtr,
    pub shared_gate: DevicePtr,
    /// Optional `[rows]` f32 output; `DevicePtr::NULL` skips the Last-CTA gate
    /// reduction while still emitting `(xq, xs)`.
    pub ss_out: DevicePtr,
    pub rows: usize,
    pub stream: u64,
}

/// Record left by `upmix_roof` when it emitted bit-exact MXFP8 `(xq, xs)` (and
/// optionally `ss_out`) directly from the staged BF16 `mixed` tile.
#[allow(dead_code)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PreQuantizedMxfp8 {
    pub mixed: u64,
    pub xq: u64,
    pub xs: u64,
    pub shared_gate: u64,
    pub ss_out: u64,
    pub rows: usize,
    pub stream: u64,
}

impl PreQuantizedMxfp8 {
    /// True when this record also wrote bit-exact `ss_out` using `shared_gate`.
    #[allow(dead_code)]
    pub(crate) fn has_gate_scalars(self, shared_gate: DevicePtr, ss_out: DevicePtr) -> bool {
        !ss_out.is_null() && self.ss_out == ss_out.0 && self.shared_gate == shared_gate.0
    }
}

thread_local! {
    static MXFP8_ARMED: Cell<Option<UpmixMxfp8Request>> = const { Cell::new(None) };
    static MXFP8_READY: Cell<Option<PreQuantizedMxfp8>> = const { Cell::new(None) };
}

/// Scoped, one-shot arm across `mlp.prepare_prefill_exact_packed` and
/// `ffn.forward_prefill`: dropping the guard restores any prior arm and clears
/// any unconsumed [`PreQuantizedMxfp8`] record.
#[allow(dead_code)]
#[must_use]
pub(crate) struct UpmixMxfp8Arm(Option<UpmixMxfp8Request>);

impl UpmixMxfp8Arm {
    /// Arm the next `upmix_roof` launch iff `ATLAS_QWEN4_PREFILL_UPMIX_MXFP8_FUSED`
    /// is enabled (default ON) and `request` is `Some(_)`.
    #[allow(dead_code)]
    pub(crate) fn enter(request: Option<UpmixMxfp8Request>) -> Result<Self> {
        let armed = if request.is_some() && mxfp8_fused_enabled()? {
            request
        } else {
            None
        };
        Ok(Self(MXFP8_ARMED.with(|a| a.replace(armed))))
    }

    /// Inactive guard (does not arm; still restores prior state and clears any
    /// stale ready record on drop).
    #[allow(dead_code)]
    pub(crate) fn inactive() -> Self {
        Self(MXFP8_ARMED.with(|a| a.replace(None)))
    }

    #[cfg(test)]
    fn enter_for_test(request: Option<UpmixMxfp8Request>) -> Self {
        Self(MXFP8_ARMED.with(|a| a.replace(request)))
    }
}

impl Drop for UpmixMxfp8Arm {
    fn drop(&mut self) {
        MXFP8_ARMED.with(|a| a.set(self.0));
        clear_prequant_mxfp8();
    }
}

/// Producer side: consume any armed MXFP8 request once.
pub(super) fn take_mxfp8_armed() -> Option<UpmixMxfp8Request> {
    MXFP8_ARMED.with(Cell::take)
}

/// Producer side: invalidate any earlier [`PreQuantizedMxfp8`] record.
pub(crate) fn clear_prequant_mxfp8() {
    MXFP8_READY.with(|r| r.set(None));
}

pub(super) fn note_prequant_mxfp8(ready: PreQuantizedMxfp8) {
    MXFP8_READY.with(|r| r.set(Some(ready)));
}

/// Consumer side: consume and return the pre-quantized MXFP8 record iff it
/// matches `(mixed, xq, xs, rows, stream)` exactly. Any mismatch also clears
/// the slot so a stale record never survives.
#[allow(dead_code)]
pub(crate) fn take_prequant_mxfp8(
    mixed: DevicePtr,
    xq: DevicePtr,
    xs: DevicePtr,
    rows: usize,
    stream: u64,
) -> Option<PreQuantizedMxfp8> {
    let ready = MXFP8_READY.with(Cell::take)?;
    let hit = ready.mixed == mixed.0
        && ready.xq == xq.0
        && ready.xs == xs.0
        && ready.rows == rows
        && ready.stream == stream;
    if hit {
        static CONSUMED: Once = Once::new();
        CONSUMED.call_once(|| {
            tracing::info!(
                "UPMIX_MXFP8_FUSED_CONSUMED selector={MXFP8_FUSED_SELECTOR} rows={rows} gate_scalars={}",
                ready.ss_out != 0
            );
        });
        Some(ready)
    } else {
        static MISSED: Once = Once::new();
        MISSED.call_once(|| {
            tracing::warn!(
                "UPMIX_MXFP8_FUSED_MISS selector={MXFP8_FUSED_SELECTOR} ready={ready:?} \
                 consumer_rows={rows} consumer_stream={stream}"
            );
        });
        None
    }
}

/// Fixed 4 KiB (`MXFP8_GATE_MAX_Y_TILES` u32 counters) self-resetting arrival
/// counter buffer for the Last-CTA L2-hot gate-scalar epilogue.
///
/// Allocated and zeroed once outside CUDA graph capture; each row-tile's 80th
/// (last) CTA resets `tile_done[blockIdx.y] = 0u` before returning so every
/// subsequent launch or CUDA Graph replay observes an all-zero buffer.
fn tile_done_slot(gpu: &dyn GpuBackend, stream: u64) -> Option<DevicePtr> {
    static SLOT: OnceLock<DevicePtr> = OnceLock::new();
    if let Some(&ptr) = SLOT.get() {
        return Some(ptr);
    }
    if spark_runtime::gpu_capture_region::active_capture_region().is_some() {
        return None;
    }
    static INIT_LOCK: Mutex<()> = Mutex::new(());
    let _guard = INIT_LOCK.lock().ok()?;
    if let Some(&ptr) = SLOT.get() {
        return Some(ptr);
    }
    let bytes = MXFP8_GATE_MAX_Y_TILES * std::mem::size_of::<u32>();
    let ptr = gpu.alloc(bytes).ok()?;
    if gpu.memset_async(ptr, 0, bytes, stream).is_err() || gpu.synchronize(stream).is_err() {
        return None;
    }
    let _ = SLOT.set(ptr);
    Some(ptr)
}

/// Bit-exact (0-ULP) router tensor-scaled FP8 activation quantizer matching
/// `qwen4_mxfp8_quant_rows_v16q4_gate`'s `a8_out` + `a_scale_out` + L2 discard
/// path, without rewriting `(xq, xs)` or `ss_out`.
#[allow(dead_code)]
pub(crate) fn launch_fp8_act_from_bf16(
    gpu: &dyn GpuBackend,
    src: DevicePtr,
    amax_bits: DevicePtr,
    a8_out: DevicePtr,
    a_scale_out: DevicePtr,
    rows: usize,
    stream: u64,
) -> Result<()> {
    anyhow::ensure!(rows > 0, "{MXFP8_FUSED_SELECTOR}: empty fp8_act_from_bf16");
    static K: OnceLock<Result<KernelHandle, String>> = OnceLock::new();
    let k = K
        .get_or_init(|| {
            gpu.kernel("qwen4_hc_upmix_roof", FP8_ACT_FROM_BF16_ENTRY)
                .map_err(|e| e.to_string())
        })
        .clone()
        .map_err(|e| {
            anyhow::anyhow!("{MXFP8_FUSED_SELECTOR}: {FP8_ACT_FROM_BF16_ENTRY} unavailable: {e}")
        })?;
    let rows_u32 = u32::try_from(rows)?;
    let grid_x = rows_u32.clamp(1, 1024);
    KernelLaunch::new(gpu, k)
        .grid([grid_x, 1, 1])
        .block([THREADS, 1, 1])
        .arg_ptr(src)
        .arg_ptr(amax_bits)
        .arg_ptr(a8_out)
        .arg_ptr(a_scale_out)
        .arg_u32(rows_u32)
        .launch(stream)
}

/// Parsed once per process; a malformed value fails the first prefill.
pub(crate) fn tile() -> Result<Option<Tile>> {
    static SEL: OnceLock<Result<Option<Tile>, String>> = OnceLock::new();
    SEL.get_or_init(|| plan::parse_selector(std::env::var(SELECTOR).ok().as_deref()))
        .clone()
        .map_err(anyhow::Error::msg)
}

/// Standard (non-streaming) kernel entry name for `(tile, fused_prep, amax)`.
fn non_streaming_entry(tile: Tile, fused_prep: bool, amax: bool) -> &'static str {
    if matches!(tile, Tile::T128) && fused_prep {
        if amax {
            "qwen4_hc_upmix_roof_t128_r4_amax"
        } else {
            "qwen4_hc_upmix_roof_t128_r4"
        }
    } else {
        tile.entry(amax)
    }
}

/// The four BF16-up entries, `[t64, t64_amax, t128, t128_amax]`, each allowed
/// its dynamic shared memory (t128 is above the 48 KiB default).
fn kernel(gpu: &dyn GpuBackend, tile: Tile, amax: bool) -> Result<KernelHandle> {
    if tile.is_w4() {
        static KW4: OnceLock<Result<[KernelHandle; 2], String>> = OnceLock::new();
        let all = KW4
            .get_or_init(|| {
                let load = |a: bool| {
                    let name = Tile::T128W4.entry(a);
                    let k = gpu
                        .kernel("qwen4_hc_upmix_roof", name)
                        .map_err(|e| format!("{name}: {e}"))?;
                    gpu.set_kernel_max_dynamic_shared_memory(k, Tile::T128W4.smem_bytes())
                        .map_err(|e| format!("{name}: {e}"))?;
                    Ok::<_, String>(k)
                };
                Ok([load(false)?, load(true)?])
            })
            .clone()
            .map_err(|e| {
                anyhow::anyhow!("{SELECTOR}: qwen4_hc_upmix_roof w4 kernels unavailable: {e}")
            })?;
        return Ok(all[usize::from(amax)]);
    }
    static K: OnceLock<Result<[KernelHandle; 4], String>> = OnceLock::new();
    let all = K
        .get_or_init(|| {
            let fused_prep = super::prefill_fused::fused_prep_selected();
            let load = |t: Tile, a: bool| {
                let name = non_streaming_entry(t, fused_prep, a);
                let k = gpu
                    .kernel("qwen4_hc_upmix_roof", name)
                    .map_err(|e| format!("{name}: {e}"))?;
                gpu.set_kernel_max_dynamic_shared_memory(k, t.smem_bytes())
                    .map_err(|e| format!("{name}: {e}"))?;
                Ok::<_, String>(k)
            };
            Ok([
                load(Tile::T64, false)?,
                load(Tile::T64, true)?,
                load(Tile::T128, false)?,
                load(Tile::T128, true)?,
            ])
        })
        .clone()
        .map_err(|e| anyhow::anyhow!("{SELECTOR}: qwen4_hc_upmix_roof kernels unavailable: {e}"))?;
    let index = match tile {
        Tile::T64 => 0,
        Tile::T128 | Tile::T128W4 => 2,
    } + usize::from(amax);
    Ok(all[index])
}

/// Streaming-output (`L2::evict_first` on `mixed`) twins of the `t128` entries
/// for strip-mined wide prefill; returns `None` when the tile has no streaming
/// twin (e.g. `t64`) or the symbol is absent, letting the caller fall back to
/// the standard entry.
fn stream_kernel(gpu: &dyn GpuBackend, tile: Tile, amax: bool) -> Option<KernelHandle> {
    let try_load = |t: Tile, fused_prep: bool, a: bool| -> Option<KernelHandle> {
        let name = t.stream_entry(fused_prep, a)?;
        let k = gpu.kernel("qwen4_hc_upmix_roof", name).ok()?;
        gpu.set_kernel_max_dynamic_shared_memory(k, t.smem_bytes())
            .ok()?;
        Some(k)
    };
    match tile {
        Tile::T64 => None,
        Tile::T128W4 => {
            static KW4_STREAM: OnceLock<[Option<KernelHandle>; 2]> = OnceLock::new();
            let all = KW4_STREAM.get_or_init(|| {
                [
                    try_load(Tile::T128W4, false, false),
                    try_load(Tile::T128W4, false, true),
                ]
            });
            all[usize::from(amax)]
        }
        Tile::T128 => {
            static KT128_STREAM: OnceLock<[Option<KernelHandle>; 2]> = OnceLock::new();
            let all = KT128_STREAM.get_or_init(|| {
                let fused_prep = super::prefill_fused::fused_prep_selected();
                [
                    try_load(Tile::T128, fused_prep, false),
                    try_load(Tile::T128, fused_prep, true),
                ]
            });
            all[usize::from(amax)]
        }
    }
}

/// Fused MXFP8 `(xq, xs)` + optional Last-CTA L2-hot shared-expert gate-scalar
/// twins of the `t128` entries; returns `None` when the tile has no fused twin
/// (e.g. `t64`) or the symbol is absent, letting the caller fall back to the
/// standard entry.
fn mxfp8_gate_kernel(gpu: &dyn GpuBackend, tile: Tile, amax: bool) -> Option<KernelHandle> {
    let try_load = |t: Tile, fused_prep: bool, a: bool| -> Option<KernelHandle> {
        let name = t.mxfp8_gate_entry(fused_prep, a)?;
        let k = gpu.kernel("qwen4_hc_upmix_roof", name).ok()?;
        gpu.set_kernel_max_dynamic_shared_memory(k, t.smem_bytes())
            .ok()?;
        Some(k)
    };
    match tile {
        Tile::T64 => None,
        Tile::T128W4 => {
            static KW4_MXFP8: OnceLock<[Option<KernelHandle>; 2]> = OnceLock::new();
            let all = KW4_MXFP8.get_or_init(|| {
                [
                    try_load(Tile::T128W4, false, false),
                    try_load(Tile::T128W4, false, true),
                ]
            });
            all[usize::from(amax)]
        }
        Tile::T128 => {
            static KT128_MXFP8: OnceLock<[Option<KernelHandle>; 2]> = OnceLock::new();
            let all = KT128_MXFP8.get_or_init(|| {
                let fused_prep = super::prefill_fused::fused_prep_selected();
                [
                    try_load(Tile::T128, fused_prep, false),
                    try_load(Tile::T128, fused_prep, true),
                ]
            });
            all[usize::from(amax)]
        }
    }
}

/// Streaming-output (`L2::evict_first` on `mixed`) + fused MXFP8 `(xq, xs)` +
/// optional Last-CTA gate-scalar twins of the `t128` entries for strip-mined
/// wide prefill.
fn stream_mxfp8_gate_kernel(gpu: &dyn GpuBackend, tile: Tile, amax: bool) -> Option<KernelHandle> {
    let try_load = |t: Tile, fused_prep: bool, a: bool| -> Option<KernelHandle> {
        let name = t.stream_mxfp8_gate_entry(fused_prep, a)?;
        let k = gpu.kernel("qwen4_hc_upmix_roof", name).ok()?;
        gpu.set_kernel_max_dynamic_shared_memory(k, t.smem_bytes())
            .ok()?;
        Some(k)
    };
    match tile {
        Tile::T64 => None,
        Tile::T128W4 => {
            static KW4_STREAM_MXFP8: OnceLock<[Option<KernelHandle>; 2]> = OnceLock::new();
            let all = KW4_STREAM_MXFP8.get_or_init(|| {
                [
                    try_load(Tile::T128W4, false, false),
                    try_load(Tile::T128W4, false, true),
                ]
            });
            all[usize::from(amax)]
        }
        Tile::T128 => {
            static KT128_STREAM_MXFP8: OnceLock<[Option<KernelHandle>; 2]> = OnceLock::new();
            let all = KT128_STREAM_MXFP8.get_or_init(|| {
                let fused_prep = super::prefill_fused::fused_prep_selected();
                [
                    try_load(Tile::T128, fused_prep, false),
                    try_load(Tile::T128, fused_prep, true),
                ]
            });
            all[usize::from(amax)]
        }
    }
}

fn mxfp8_gate_kernel_for(
    gpu: &dyn GpuBackend,
    tile: Tile,
    amax: bool,
    output_cache: OutputCachePolicy,
) -> Option<(KernelHandle, bool)> {
    if output_cache == OutputCachePolicy::StreamEvictFirst
        && let Some(k) = stream_mxfp8_gate_kernel(gpu, tile, amax)
    {
        return Some((k, true));
    }
    mxfp8_gate_kernel(gpu, tile, amax).map(|k| (k, false))
}

fn kernel_for(
    gpu: &dyn GpuBackend,
    tile: Tile,
    amax: bool,
    output_cache: OutputCachePolicy,
) -> Result<(KernelHandle, bool)> {
    if output_cache == OutputCachePolicy::StreamEvictFirst
        && let Some(k) = stream_kernel(gpu, tile, amax)
    {
        return Ok((k, true));
    }
    Ok((kernel(gpu, tile, amax)?, false))
}

fn build_base_launch<'a>(
    gpu: &'a dyn GpuBackend,
    k: KernelHandle,
    tile: Tile,
    l: &UpmixLaunch,
    hidden_u32: u32,
    rows: u32,
) -> KernelLaunch<'a> {
    if tile.is_w4() {
        KernelLaunch::new(gpu, k)
            .grid(tile.grid(hidden_u32, rows))
            .block([THREADS, 1, 1])
            .shared_mem(tile.smem_bytes())
            .arg_ptr(l.act)
            .arg_ptr(l.up_packed)
            .arg_ptr(l.up_scales)
            .arg_f32(l.alpha)
            .arg_ptr(l.hyper)
            .arg_ptr(l.rstd)
            .arg_ptr(l.norm_w)
            .arg_ptr(l.mixed)
            .arg_u32(rows)
    } else {
        KernelLaunch::new(gpu, k)
            .grid(tile.grid(hidden_u32, rows))
            .block([THREADS, 1, 1])
            .shared_mem(tile.smem_bytes())
            .arg_ptr(l.act)
            .arg_ptr(l.up)
            .arg_f32(l.alpha)
            .arg_ptr(l.hyper)
            .arg_ptr(l.rstd)
            .arg_ptr(l.norm_w)
            .arg_ptr(l.mixed)
            .arg_u32(rows)
    }
}

fn refused(r: Refusal, rows: usize) {
    static LOGGED: [Once; Refusal::COUNT] = [const { Once::new() }; Refusal::COUNT];
    LOGGED[r.index()].call_once(|| {
        tracing::info!(
            "MHC_ROOF_REFUSED selector={SELECTOR} reason={} rows={rows} (incumbent up+mix)",
            r.reason()
        );
    });
}

/// One HC_FUSED up+mix launch: `qwen4_hc_upmix_hcfused`'s arguments, plus the
/// raw NVFP4 `up` pointers (for `t128_w4`), the absmax slot when armed, and an
/// optional fused MXFP8 + gate-scalar epilogue request.
#[derive(Clone, Copy)]
pub(super) struct UpmixLaunch {
    pub act: DevicePtr,
    pub up: DevicePtr,
    pub up_packed: DevicePtr,
    pub up_scales: DevicePtr,
    pub alpha: f32,
    pub hyper: DevicePtr,
    pub rstd: DevicePtr,
    pub norm_w: DevicePtr,
    pub mixed: DevicePtr,
    pub amax_slot: Option<DevicePtr>,
    pub output_cache: OutputCachePolicy,
    pub mxfp8_gate: Option<UpmixMxfp8Request>,
    /// When true (opt-in strip mode), promotes `Tile::T128` to `Tile::T128W4`
    /// if NVFP4 `up` admission succeeds, keeping `up` at `1.84 MiB` in L2.
    pub prefer_w4: bool,
    /// When true, skips per-launch `note_prequant_mxfp8` (caller notes once
    /// for the full chunk after all strips finish).
    pub suppress_mxfp8_note: bool,
}

fn launch_io(l: &UpmixLaunch) -> Io {
    Io {
        act: l.act.0,
        up: l.up.0,
        hyper: l.hyper.0,
        rstd: l.rstd.0,
        norm_w: l.norm_w.0,
        mixed: l.mixed.0,
    }
}

fn resolve_tile(l: &UpmixLaunch, upmix_selected: bool) -> Result<Option<Tile>> {
    let Some(base) = tile()? else {
        return Ok(None);
    };
    if l.prefer_w4
        && matches!(base, Tile::T128)
        && plan::admit_w4(upmix_selected, &launch_io(l), l.up_packed.0, l.up_scales.0).is_ok()
    {
        return Ok(Some(Tile::T128W4));
    }
    Ok(Some(base))
}

/// True when `t128_w4` is selected (or promoted via `l.prefer_w4`) and
/// admitted for `l`, so the caller may skip dequantizing `up` into `s.up`.
pub(super) fn w4_admitted(l: &UpmixLaunch, upmix_selected: bool) -> Result<bool> {
    let Some(Tile::T128W4) = resolve_tile(l, upmix_selected)? else {
        return Ok(false);
    };
    let io = launch_io(l);
    Ok(plan::admit_w4(upmix_selected, &io, l.up_packed.0, l.up_scales.0).is_ok())
}

/// Pre-flight check for strip-mined W21-F MXFP8 + gate fusion: returns true
/// only when every strip can launch the fused `mxfp8_gate` roof kernel and the
/// full-chunk `(mixed, xq, xs, shared_gate, ss_out, tile_done)` layout passes
/// alignment and disjointness checks.
pub(super) fn can_fuse_mxfp8_gate_strip(
    l: &UpmixLaunch,
    upmix_selected: bool,
    hidden: usize,
    num_tokens: usize,
    gpu: &dyn GpuBackend,
    stream: u64,
) -> Result<bool> {
    let Some(req) = l.mxfp8_gate else {
        return Ok(false);
    };
    if req.rows != num_tokens || req.stream != stream || !mxfp8_fused_enabled()? {
        return Ok(false);
    }
    let Some(tile) = resolve_tile(l, upmix_selected)? else {
        return Ok(false);
    };
    let io = launch_io(l);
    let roof_ok = if tile.is_w4() {
        plan::admit_w4(upmix_selected, &io, l.up_packed.0, l.up_scales.0).is_ok()
    } else {
        plan::admit(upmix_selected, &io).is_ok()
    };
    if !roof_ok {
        return Ok(false);
    }
    let amax = l.amax_slot.is_some();
    if mxfp8_gate_kernel_for(gpu, tile, amax, l.output_cache).is_none() {
        return Ok(false);
    }
    let tile_done_ptr = if req.ss_out.is_null() {
        DevicePtr::NULL
    } else {
        let Some(ptr) = tile_done_slot(gpu, stream) else {
            return Ok(false);
        };
        ptr
    };
    let mxfp8_io = Mxfp8GateIo {
        xq: req.xq.0,
        xs: req.xs.0,
        shared_gate: req.shared_gate.0,
        ss_out: req.ss_out.0,
        tile_done: tile_done_ptr.0,
        hidden,
        rows: num_tokens,
    };
    Ok(plan::admit_mxfp8_gate(l.mixed.0, &mxfp8_io))
}

/// Launch the selected roof kernel in place of `qwen4_hc_upmix_hcfused[_amax]`.
/// Returns false, launching nothing, when the lever is off or the launch is
/// not admitted (the caller then runs its own up+mix kernel unchanged).
pub(super) fn launch(
    l: &UpmixLaunch,
    upmix_selected: bool,
    hidden: usize,
    num_tokens: usize,
    gpu: &dyn GpuBackend,
    stream: u64,
) -> Result<bool> {
    let Some(tile) = resolve_tile(l, upmix_selected)? else {
        return Ok(false);
    };
    let io = launch_io(l);
    let admitted = if tile.is_w4() {
        plan::admit_w4(upmix_selected, &io, l.up_packed.0, l.up_scales.0)
    } else {
        plan::admit(upmix_selected, &io)
    };
    if let Err(r) = admitted {
        refused(r, num_tokens);
        return Ok(false);
    }
    let hidden_u32 = u32::try_from(hidden)?;
    let rows = u32::try_from(num_tokens)?;
    let amax = l.amax_slot.is_some();
    if let Some(req) = l.mxfp8_gate
        && req.rows == num_tokens
        && req.stream == stream
        && mxfp8_fused_enabled()?
        && let Some((k, streaming_mxfp8)) = mxfp8_gate_kernel_for(gpu, tile, amax, l.output_cache)
    {
        let tile_done = if req.ss_out.is_null() {
            Some(DevicePtr::NULL)
        } else {
            tile_done_slot(gpu, stream)
        };
        if let Some(tile_done_ptr) = tile_done {
            let mxfp8_io = Mxfp8GateIo {
                xq: req.xq.0,
                xs: req.xs.0,
                shared_gate: req.shared_gate.0,
                ss_out: req.ss_out.0,
                tile_done: tile_done_ptr.0,
                hidden,
                rows: num_tokens,
            };
            if plan::admit_mxfp8_gate(l.mixed.0, &mxfp8_io) {
                let kl = build_base_launch(gpu, k, tile, l, hidden_u32, rows);
                let kl = match l.amax_slot {
                    Some(slot) => kl.arg_ptr(slot),
                    None => kl,
                };
                kl.arg_ptr(req.xq)
                    .arg_ptr(req.xs)
                    .arg_ptr(req.shared_gate)
                    .arg_ptr(req.ss_out)
                    .arg_ptr(tile_done_ptr)
                    .launch(stream)?;
                if !l.suppress_mxfp8_note {
                    note_prequant_mxfp8(PreQuantizedMxfp8 {
                        mixed: l.mixed.0,
                        xq: req.xq.0,
                        xs: req.xs.0,
                        shared_gate: req.shared_gate.0,
                        ss_out: req.ss_out.0,
                        rows: num_tokens,
                        stream,
                    });
                }
                static ENGAGED_MXFP8: [Once; 8] = [const { Once::new() }; 8];
                let index = match tile {
                    Tile::T64 | Tile::T128 => 0,
                    Tile::T128W4 => 2,
                } + usize::from(amax)
                    + usize::from(streaming_mxfp8) * 4;
                ENGAGED_MXFP8[index].call_once(|| {
                    let fused_prep = super::prefill_fused::fused_prep_selected();
                    let entry = if streaming_mxfp8 {
                        tile.stream_mxfp8_gate_entry(fused_prep, amax)
                    } else {
                        tile.mxfp8_gate_entry(fused_prep, amax)
                    }
                    .unwrap_or_else(|| tile.entry(amax));
                    tracing::info!(
                        "MHC_ROOF_ENGAGED selector={SELECTOR} mxfp8_selector={MXFP8_FUSED_SELECTOR} \
                         tile={} kernel={entry} rows={num_tokens} mxfp8_gate=true gate_scalars={} stream_out={streaming_mxfp8}",
                        tile.name(),
                        !req.ss_out.is_null(),
                    );
                });
                return Ok(true);
            }
        }
    }
    let (k, streaming) = kernel_for(gpu, tile, amax, l.output_cache)?;
    let kl = build_base_launch(gpu, k, tile, l, hidden_u32, rows);
    match l.amax_slot {
        Some(slot) => kl.arg_ptr(slot).launch(stream)?,
        None => kl.launch(stream)?,
    }
    static ENGAGED: [Once; 12] = [const { Once::new() }; 12];
    let tile_idx = match tile {
        Tile::T64 => 0,
        Tile::T128 => 1,
        Tile::T128W4 => 2,
    };
    let index = (tile_idx * 2 + usize::from(amax)) * 2 + usize::from(streaming);
    ENGAGED[index].call_once(|| {
        let fused_prep = super::prefill_fused::fused_prep_selected();
        let entry = if streaming {
            tile.stream_entry(fused_prep, amax)
                .unwrap_or_else(|| non_streaming_entry(tile, fused_prep, amax))
        } else {
            non_streaming_entry(tile, fused_prep, amax)
        };
        tracing::info!(
            "MHC_ROOF_ENGAGED selector={SELECTOR} tile={} kernel={entry} rows={num_tokens}",
            tile.name(),
        );
    });
    Ok(true)
}

#[inline]
#[must_use]
fn discard_l2_grid_x_for_sm(sm_count: u32) -> u32 {
    sm_count.max(1)
}

/// Discard complete 128-byte L2 cache lines of dead post-upmix scratch buffers
/// (`s.down`, `s.rstd`, `s.folded`, `s.up`) on the 2-kernel fallback path.
/// Launched on the same stream after `upmix_roof` so all 80 H-tiles have
/// finished consuming the buffers before any line is invalidated.
pub(super) fn discard_dead_l2(
    ranges: [(DevicePtr, usize); 4],
    gpu: &dyn GpuBackend,
    stream: u64,
) -> Result<()> {
    if !ranges
        .iter()
        .any(|&(ptr, bytes)| !ptr.is_null() && bytes >= 128)
    {
        return Ok(());
    }
    static K: OnceLock<Option<KernelHandle>> = OnceLock::new();
    let Some(k) = *K.get_or_init(|| {
        gpu.kernel("qwen4_hc_upmix_roof", "qwen4_hc_discard_l2_ranges")
            .ok()
    }) else {
        return Ok(());
    };
    let grid_x = discard_l2_grid_x_for_sm(atlas_core::device::active_sm_count());
    KernelLaunch::new(gpu, k)
        .grid([grid_x, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(ranges[0].0)
        .arg_u64(ranges[0].1 as u64)
        .arg_ptr(ranges[1].0)
        .arg_u64(ranges[1].1 as u64)
        .arg_ptr(ranges[2].0)
        .arg_u64(ranges[2].1 as u64)
        .arg_ptr(ranges[3].0)
        .arg_u64(ranges[3].1 as u64)
        .launch(stream)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_req() -> UpmixMxfp8Request {
        UpmixMxfp8Request {
            xq: DevicePtr(0x1_0000_0000),
            xs: DevicePtr(0x1_0280_0000),
            shared_gate: DevicePtr(0x2_0000_0000),
            ss_out: DevicePtr(0x3_0000_0000),
            rows: 16_384,
            stream: 7,
        }
    }

    fn sample_ready() -> PreQuantizedMxfp8 {
        let req = sample_req();
        PreQuantizedMxfp8 {
            mixed: 0x0_8000_0000,
            xq: req.xq.0,
            xs: req.xs.0,
            shared_gate: req.shared_gate.0,
            ss_out: req.ss_out.0,
            rows: req.rows,
            stream: req.stream,
        }
    }

    #[test]
    fn mxfp8_arm_is_one_shot_and_restores_on_drop() {
        assert_eq!(take_mxfp8_armed(), None);
        {
            let _none = UpmixMxfp8Arm::enter(None).expect("None arm never fails");
            assert_eq!(take_mxfp8_armed(), None);
        }
        let req = sample_req();
        {
            let _outer = UpmixMxfp8Arm::enter_for_test(Some(req));
            assert_eq!(take_mxfp8_armed(), Some(req));
            assert_eq!(take_mxfp8_armed(), None, "arm is one-shot");
            note_prequant_mxfp8(sample_ready());
        }
        assert_eq!(take_mxfp8_armed(), None);
        assert_eq!(
            take_prequant_mxfp8(
                DevicePtr(0x0_8000_0000),
                req.xq,
                req.xs,
                req.rows,
                req.stream
            ),
            None,
            "dropping the arm clears any unconsumed ready record"
        );
    }

    #[test]
    fn mxfp8_ready_serves_only_exact_match_once() {
        let req = sample_req();
        let mixed = DevicePtr(0x0_8000_0000);
        note_prequant_mxfp8(sample_ready());
        assert_eq!(
            take_prequant_mxfp8(mixed, req.xq, req.xs, req.rows + 1, req.stream),
            None,
            "row mismatch refuses"
        );
        assert_eq!(
            take_prequant_mxfp8(mixed, req.xq, req.xs, req.rows, req.stream),
            None,
            "mismatch consumes stale record"
        );

        note_prequant_mxfp8(sample_ready());
        let hit = take_prequant_mxfp8(mixed, req.xq, req.xs, req.rows, req.stream)
            .expect("exact match hits");
        assert!(hit.has_gate_scalars(req.shared_gate, req.ss_out));
        assert!(!hit.has_gate_scalars(req.shared_gate, DevicePtr::NULL));
        assert!(!hit.has_gate_scalars(DevicePtr(0x2_0000_0010), req.ss_out));
        assert_eq!(
            take_prequant_mxfp8(mixed, req.xq, req.xs, req.rows, req.stream),
            None,
            "consumed once"
        );

        note_prequant_mxfp8(sample_ready());
        clear_prequant_mxfp8();
        assert_eq!(
            take_prequant_mxfp8(mixed, req.xq, req.xs, req.rows, req.stream),
            None,
            "clear invalidates record"
        );
    }

    #[test]
    fn inactive_arm_suppresses_nested_arm_and_restores_prior() {
        let req = sample_req();
        let _outer = UpmixMxfp8Arm::enter_for_test(Some(req));
        {
            let _inner = UpmixMxfp8Arm::inactive();
            assert_eq!(take_mxfp8_armed(), None);
        }
        assert_eq!(take_mxfp8_armed(), Some(req));
    }

    #[test]
    fn discard_l2_grid_scales_with_active_sm_count() {
        assert_eq!(discard_l2_grid_x_for_sm(0), 1);
        for sm in [48u32, 110, 170, 188] {
            assert_eq!(discard_l2_grid_x_for_sm(sm), sm);
        }
    }

    #[test]
    fn non_streaming_entry_matches_r4_variant_for_t128_fused_prep() {
        assert_eq!(
            non_streaming_entry(Tile::T128, true, false),
            "qwen4_hc_upmix_roof_t128_r4"
        );
        assert_eq!(
            non_streaming_entry(Tile::T128, true, true),
            "qwen4_hc_upmix_roof_t128_r4_amax"
        );
        assert_eq!(
            non_streaming_entry(Tile::T128, false, false),
            "qwen4_hc_upmix_roof_t128"
        );
        assert_eq!(
            non_streaming_entry(Tile::T64, true, false),
            "qwen4_hc_upmix_roof_t64"
        );
        assert_eq!(
            non_streaming_entry(Tile::T128W4, true, true),
            "qwen4_hc_upmix_roof_t128_w4_amax"
        );
    }
}
