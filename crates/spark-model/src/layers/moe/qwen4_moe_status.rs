// SPDX-License-Identifier: AGPL-3.0-only

//! Default-off (`ATLAS_QWEN4_PREFILL_MOE_STATUS_ASYNC=1`): deferred compact-MoE status checks.
//!
//! Each compact dispatch (routed, and the shared expert through the routed kernels) uploads
//! its contract with a stream-draining H2D group and reads its status word back with a
//! blocking D2H: two host syncs per MoE layer, so the CPU can never queue the next layer
//! while the GPU works. With the selector set, the per-row-count constant bytes (contract,
//! pending/planned word, host-built workspace) are uploaded ONCE to a device buffer and
//! copied in with stream-ordered D2D copies, and each dispatch's final status word is copied
//! (D2D, stream-ordered) into a device slot. `flush` reads every slot with one sync after the
//! chunk's last layer and fails closed on any status that is not PLANNED; a full slot table
//! flushes early. Slots left by a failed chunk are dropped by `discard_stale`.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use anyhow::{Context, Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

use super::qwen4_compact_contract as abi;

pub(crate) const SELECTOR: &str = "ATLAS_QWEN4_PREFILL_MOE_STATUS_ASYNC";
const SLOTS: usize = 4096;
/// Maximum batched-verify row count pre-warmed into the device constant slab at model load
/// (`256`, covering `C=32` DFlash2 `K=8` and `C=64` MTP `K=4`).
const VERIFY_PREWARM_MAX_ROWS: usize = crate::layers::QWEN4_NATIVE_MAX_ROWS;
/// Padded stride (bytes, 16-byte aligned) for one routed `(contract, pending)` constant entry.
const ROUTED_CONST_STRIDE: usize = (abi::CONTRACT_BYTES + 4).next_multiple_of(16);
/// Sticky accumulator bit set when a non-PLANNED status code falls outside `1..=30`
/// (e.g. `PENDING = -1`, `BUILDING = -2`, or `0`).
const STICKY_ERR_OTHER_BIT: u32 = 1u32 << 31;

pub(crate) fn selected() -> bool {
    static SEL: OnceLock<bool> = OnceLock::new();
    *SEL.get_or_init(|| std::env::var(SELECTOR).ok().as_deref() == Some("1"))
}

/// True while either the CUDA stream or an active capture region is recording a CUDA graph.
pub(crate) fn is_capturing(gpu: &dyn GpuBackend, stream: u64) -> bool {
    gpu.stream_is_capturing(stream)
        || spark_runtime::gpu_capture_region::active_capture_region().is_some()
}

struct State {
    slots: DevicePtr,
    slots_dirty: bool,
    labels: Vec<&'static str>,
    /// Single pre-allocated 4-byte device `u32` error bitmask (`0` = all OK).
    sticky_err: DevicePtr,
    /// Single pre-allocated 4-byte device `i32` initialized to `PLANNED` (`1`) for shared-expert
    /// dispatches; untouched on success and latched via `atomicCAS` on failure.
    shared_status: DevicePtr,
    /// True once any sticky-mode dispatch has been queued since the last host check.
    sticky_pending: bool,
    /// True when `sticky_err` / `shared_status` must be re-initialized to `(0, PLANNED)`.
    sticky_dirty: bool,
    /// True once the `rows = 2..=VERIFY_PREWARM_MAX_ROWS` batched-verify constant slab has been uploaded.
    verify_prewarmed: bool,
    constants: HashMap<(&'static str, usize), (DevicePtr, usize)>,
}

impl State {
    fn prewarm_verify_constants(&mut self, gpu: &dyn GpuBackend, stream: u64) -> Result<()> {
        let mut uploaded = false;
        if self.sticky_dirty {
            gpu.copy_h2d_async(&0u32.to_le_bytes(), self.sticky_err, stream)?;
            gpu.copy_h2d_async(&1i32.to_le_bytes(), self.shared_status, stream)?;
            uploaded = true;
        }
        if !self.verify_prewarmed {
            let shared_stride =
                super::qwen4_shared_v7::shared_constant_len(VERIFY_PREWARM_MAX_ROWS)
                    .next_multiple_of(16);
            let count = VERIFY_PREWARM_MAX_ROWS - 1; // rows 2..=256
            let shared_section = count * shared_stride;
            let routed_section = count * ROUTED_CONST_STRIDE;
            let mut slab = vec![0u8; shared_section + routed_section];
            let pending = (-1i32).to_le_bytes();
            for (idx, rows) in (2..=VERIFY_PREWARM_MAX_ROWS).enumerate() {
                let shared_bytes = super::qwen4_shared_v7::shared_constant_bytes(rows)?;
                let sh_off = idx * shared_stride;
                slab[sh_off..sh_off + shared_bytes.len()].copy_from_slice(&shared_bytes);

                let contract = abi::contract(rows).map_err(anyhow::Error::msg)?;
                let rt_off = shared_section + idx * ROUTED_CONST_STRIDE;
                slab[rt_off..rt_off + abi::CONTRACT_BYTES].copy_from_slice(&contract);
                slab[rt_off + abi::CONTRACT_BYTES..rt_off + abi::CONTRACT_BYTES + 4]
                    .copy_from_slice(&pending);
            }
            let base = gpu
                .alloc(slab.len())
                .context("MoE status async: alloc verify constant slab")?;
            gpu.copy_h2d_async(&slab, base, stream)?;
            gpu.synchronize(stream)?;
            if uploaded {
                self.sticky_dirty = false;
                uploaded = false;
            }
            for (idx, rows) in (2..=VERIFY_PREWARM_MAX_ROWS).enumerate() {
                let sh_len = super::qwen4_shared_v7::shared_constant_len(rows);
                self.constants
                    .entry(("shared", rows))
                    .or_insert((base.offset(idx * shared_stride), sh_len));
                self.constants.entry(("routed", rows)).or_insert((
                    base.offset(shared_section + idx * ROUTED_CONST_STRIDE),
                    abi::CONTRACT_BYTES + 4,
                ));
            }
            self.verify_prewarmed = true;
        }
        if uploaded {
            gpu.synchronize(stream)?;
            self.sticky_dirty = false;
        }
        Ok(())
    }

    #[must_use]
    fn is_capture_ready(
        &self,
        rows: usize,
        need_routed_const: bool,
        need_shared_const: bool,
    ) -> bool {
        !self.sticky_err.is_null()
            && !self.shared_status.is_null()
            && !self.sticky_dirty
            && (!need_shared_const || self.constants.contains_key(&("shared", rows)))
            && (!need_routed_const || self.constants.contains_key(&("routed", rows)))
    }
}

static STATE: Mutex<Option<State>> = Mutex::new(None);

fn status_accumulate_kernel(gpu: &dyn GpuBackend) -> Result<KernelHandle> {
    static KERNEL: OnceLock<std::result::Result<KernelHandle, String>> = OnceLock::new();
    KERNEL
        .get_or_init(|| {
            gpu.kernel("qwen4_moe_compact_t_k32_v17", "moe_status_accumulate_err")
                .map_err(|e| format!("qwen4_moe_compact_t_k32_v17::moe_status_accumulate_err: {e}"))
        })
        .clone()
        .map_err(anyhow::Error::msg)
}

/// Eagerly load the fallback status-accumulator kernel handle outside CUDA graph capture.
pub(crate) fn prewarm_kernels(gpu: &dyn GpuBackend) -> Result<()> {
    let _ = status_accumulate_kernel(gpu)?;
    Ok(())
}

fn with_state<T>(
    gpu: &dyn GpuBackend,
    stream: u64,
    f: impl FnOnce(&mut State) -> Result<T>,
) -> Result<T> {
    let mut guard = STATE.lock().expect("MoE status state lock poisoned");
    if guard.is_none() {
        ensure!(
            !is_capturing(gpu, stream),
            "MoE status async: state was not pre-allocated before CUDA graph capture"
        );
        let slots = gpu
            .alloc(SLOTS * 4)
            .context("MoE status async: alloc slot table")?;
        let sticky_err = gpu
            .alloc(4)
            .context("MoE status async: alloc sticky error word")?;
        let shared_status = gpu
            .alloc(4)
            .context("MoE status async: alloc shared status word")?;
        *guard = Some(State {
            slots,
            slots_dirty: true,
            labels: Vec::with_capacity(SLOTS),
            sticky_err,
            shared_status,
            sticky_pending: false,
            sticky_dirty: true,
            verify_prewarmed: false,
            constants: HashMap::new(),
        });
    }
    f(guard.as_mut().expect("MoE status state initialized"))
}

fn ensure_sticky_clean(gpu: &dyn GpuBackend, s: &mut State, stream: u64) -> Result<()> {
    if s.sticky_dirty {
        ensure!(
            !is_capturing(gpu, stream),
            "MoE status async: sticky status words dirty during CUDA graph capture"
        );
        gpu.copy_h2d_async(&0u32.to_le_bytes(), s.sticky_err, stream)?;
        gpu.copy_h2d_async(&1i32.to_le_bytes(), s.shared_status, stream)?;
        s.sticky_dirty = false;
    }
    Ok(())
}

/// Pre-allocate and initialize the sticky status words and the
/// `rows = 2..=VERIFY_PREWARM_MAX_ROWS` (`256`) batched-verify constant slab
/// once at model load time so CUDA graph capture performs zero allocations or
/// H2D uploads for any verify batch size in `2..=256`.
pub(crate) fn prewarm_verify_constants(gpu: &dyn GpuBackend) -> Result<()> {
    if !selected() {
        return Ok(());
    }
    let stream = gpu.default_stream();
    with_state(gpu, stream, |s| s.prewarm_verify_constants(gpu, stream))
}

/// True when `qwen4_moe_status` has all device words and required constants pre-allocated and
/// clean for capturing a CUDA graph at `rows` without performing any allocation or H2D copy.
pub(crate) fn is_capture_ready(
    rows: usize,
    need_routed_const: bool,
    need_shared_const: bool,
) -> bool {
    let guard = STATE.lock().expect("MoE status state lock poisoned");
    let Some(s) = guard.as_ref() else {
        return false;
    };
    s.is_capture_ready(rows, need_routed_const, need_shared_const)
}

/// Device copy of `bytes`, uploaded once per `(tag, rows)` (the bytes are a pure function of
/// both). Later calls must pass the same bytes; a length change is rejected.
///
/// The upload is stream-ordered on `stream`, the stream whose `copy_d2d_async` calls read the
/// constant. The old `gpu.copy_h2d` (pageable, legacy stream) could return before its DMA
/// landed, and the prefill stream is NonBlocking, so the first dispatch at each new row count
/// could copy an unlanded contract/status (fn-decode 2026-10-08). The pageable source is staged
/// before `copy_h2d_async` returns, so `bytes` may be dropped afterwards.
pub(crate) fn constant(
    gpu: &dyn GpuBackend,
    tag: &'static str,
    rows: usize,
    bytes: &[u8],
    stream: u64,
) -> Result<DevicePtr> {
    with_state(gpu, stream, |s| {
        if let Some(&(ptr, len)) = s.constants.get(&(tag, rows)) {
            ensure!(
                len == bytes.len(),
                "MoE status async: {tag} constant changed size"
            );
            return Ok(ptr);
        }
        ensure!(
            !is_capturing(gpu, stream),
            "MoE status async: uncached {tag} constant for rows={rows} during CUDA graph capture"
        );
        let ptr = gpu
            .alloc(bytes.len().next_multiple_of(16))
            .context("MoE status async: alloc constant")?;
        gpu.copy_h2d_async(bytes, ptr, stream)?;
        s.constants.insert((tag, rows), (ptr, bytes.len()));
        Ok(ptr)
    })
}

/// Zero-allocation cached lookup for a per-`(tag, rows)` device constant: runs `build` only on
/// the first miss, avoiding redundant host `Vec` allocations across 48 layers per step.
pub(crate) fn constant_with(
    gpu: &dyn GpuBackend,
    tag: &'static str,
    rows: usize,
    expected_len: usize,
    build: impl FnOnce() -> Result<Vec<u8>>,
    stream: u64,
) -> Result<DevicePtr> {
    with_state(gpu, stream, |s| {
        if let Some(&(ptr, len)) = s.constants.get(&(tag, rows)) {
            ensure!(
                len == expected_len,
                "MoE status async: {tag} constant changed size ({len} != {expected_len})"
            );
            return Ok(ptr);
        }
        ensure!(
            !is_capturing(gpu, stream),
            "MoE status async: uncached {tag} constant for rows={rows} during CUDA graph capture"
        );
        let bytes = build()?;
        ensure!(
            bytes.len() == expected_len,
            "MoE status async: {tag} constant built {} B, expected {expected_len} B",
            bytes.len()
        );
        let ptr = gpu
            .alloc(bytes.len().next_multiple_of(16))
            .context("MoE status async: alloc constant")?;
        gpu.copy_h2d_async(&bytes, ptr, stream)?;
        s.constants.insert((tag, rows), (ptr, bytes.len()));
        Ok(ptr)
    })
}

/// Return the single pre-initialized `PLANNED` (`1`) shared-expert status word for graph-safe
/// sticky status tracking (zero per-layer slot consumption, zero D2D copies, fixed address across
/// CUDA graph replays).
pub(crate) fn sticky_shared_status(gpu: &dyn GpuBackend, stream: u64) -> Result<DevicePtr> {
    with_state(gpu, stream, |s| {
        ensure_sticky_clean(gpu, s, stream)?;
        s.sticky_pending = true;
        Ok(s.shared_status)
    })
}

/// Return `(shared_status, sticky_err)` device pointers for fused post-reduce `atomicOr` status
/// accumulation, ensuring both words are initialized and marking a sticky check pending.
pub(crate) fn sticky_ptrs(gpu: &dyn GpuBackend, stream: u64) -> Result<(DevicePtr, DevicePtr)> {
    with_state(gpu, stream, |s| {
        ensure_sticky_clean(gpu, s, stream)?;
        s.sticky_pending = true;
        Ok((s.shared_status, s.sticky_err))
    })
}

/// Fallback capture-safe status accumulation when the fused unpermute-reduce+blend kernel did not
/// execute (e.g. BF16 slots or absent gate scalars): folds `routed_status` and `shared_status`
/// into `sticky_err` via a 1-thread `atomicOr` kernel without any D2D copy or host sync.
pub(crate) fn accumulate_sticky_status(
    gpu: &dyn GpuBackend,
    routed_status: DevicePtr,
    stream: u64,
) -> Result<()> {
    let (shared_status, sticky_err) = sticky_ptrs(gpu, stream)?;
    let kernel = status_accumulate_kernel(gpu)?;
    KernelLaunch::new(gpu, kernel)
        .grid([1, 1, 1])
        .block([1, 1, 1])
        .arg_ptr(routed_status)
        .arg_ptr(shared_status)
        .arg_ptr(sticky_err)
        .launch(stream)
}

/// Claim a pre-initialized `PLANNED` (`1`) status word slot for a shared-expert dispatch
/// without any per-layer D2D copy.
pub(crate) fn defer_shared(
    gpu: &dyn GpuBackend,
    label: &'static str,
    stream: u64,
) -> Result<DevicePtr> {
    ensure!(
        !is_capturing(gpu, stream),
        "MoE status async: slot-table defer is unsupported during CUDA graph capture"
    );
    let full = with_state(gpu, stream, |s| Ok(s.labels.len() >= SLOTS))?;
    if full {
        flush(gpu, stream)?;
    }
    with_state(gpu, stream, |s| {
        ensure!(
            s.labels.len() < SLOTS,
            "MoE status async: slot table still full after flush"
        );
        if s.slots_dirty {
            let planned: Vec<u8> = (0..SLOTS).flat_map(|_| 1i32.to_le_bytes()).collect();
            gpu.copy_h2d_async(&planned, s.slots, stream)?;
            s.slots_dirty = false;
        }
        let slot = s.slots.offset(s.labels.len() * 4);
        s.labels.push(label);
        Ok(slot)
    })
}

/// Queue this dispatch's status word for the next `flush` (stream-ordered after its GEMMs).
pub(crate) fn defer(
    gpu: &dyn GpuBackend,
    status: DevicePtr,
    label: &'static str,
    stream: u64,
) -> Result<()> {
    ensure!(
        !is_capturing(gpu, stream),
        "MoE status async: slot-table defer is unsupported during CUDA graph capture"
    );
    let full = with_state(gpu, stream, |s| Ok(s.labels.len() >= SLOTS))?;
    if full {
        flush(gpu, stream)?;
    }
    with_state(gpu, stream, |s| {
        ensure!(
            s.labels.len() < SLOTS,
            "MoE status async: slot table still full after flush"
        );
        if s.slots_dirty {
            let planned: Vec<u8> = (0..SLOTS).flat_map(|_| 1i32.to_le_bytes()).collect();
            gpu.copy_h2d_async(&planned, s.slots, stream)?;
            s.slots_dirty = false;
        }
        let slot = s.slots.offset(s.labels.len() * 4);
        gpu.copy_d2d_async(status, slot, 4, stream)?;
        s.labels.push(label);
        Ok(())
    })
}

fn mark_dirty_in(state: &Mutex<Option<State>>, slots_dirty: bool, sticky_dirty: bool) {
    if let Some(s) = state
        .lock()
        .expect("MoE status state lock poisoned")
        .as_mut()
    {
        if slots_dirty {
            s.slots_dirty = true;
        }
        if sticky_dirty {
            s.sticky_dirty = true;
        }
    }
}

fn flush_from_state(gpu: &dyn GpuBackend, stream: u64, state: &Mutex<Option<State>>) -> Result<()> {
    let (slots, labels, sticky_check) = {
        let mut guard = state.lock().expect("MoE status state lock poisoned");
        let Some(s) = guard.as_mut() else {
            return Ok(());
        };
        let labels = if s.labels.is_empty() {
            Vec::new()
        } else {
            std::mem::take(&mut s.labels)
        };
        let sticky_check = if s.sticky_pending {
            s.sticky_pending = false;
            Some(s.sticky_err)
        } else {
            None
        };
        (s.slots, labels, sticky_check)
    };
    if !labels.is_empty() {
        let mut bytes = vec![0u8; labels.len() * 4];
        if let Err(err) = gpu
            .copy_d2h_on_stream(slots, &mut bytes, stream)
            .and_then(|()| check_words(&bytes, &labels))
        {
            mark_dirty_in(state, true, sticky_check.is_some());
            return Err(err);
        }
        static LOGGED: std::sync::Once = std::sync::Once::new();
        LOGGED.call_once(|| {
            tracing::info!(
                "MOE_PREFILL_STATUS_ASYNC_ENGAGED selector={SELECTOR} checked={} per flush",
                labels.len()
            );
        });
    }
    if let Some(sticky_err) = sticky_check {
        let mut word = [0u8; 4];
        if let Err(err) = gpu
            .copy_d2h_on_stream(sticky_err, &mut word, stream)
            .and_then(|()| check_sticky_err(u32::from_le_bytes(word)))
        {
            mark_dirty_in(state, false, true);
            return Err(err);
        }
    }
    Ok(())
}

/// Read every queued status word and any sticky error word (skipped safely during CUDA graph
/// capture) and fail closed on any non-PLANNED status.
pub(crate) fn flush(gpu: &dyn GpuBackend, stream: u64) -> Result<()> {
    if !selected() || is_capturing(gpu, stream) {
        return Ok(());
    }
    flush_from_state(gpu, stream, &STATE)
}

fn check_sticky_err(mask: u32) -> Result<()> {
    if mask == 0 {
        return Ok(());
    }
    let mut codes = Vec::new();
    for bit in 0..31u32 {
        if (mask & (1u32 << bit)) != 0 {
            codes.push(bit.to_string());
        }
    }
    if (mask & STICKY_ERR_OTHER_BIT) != 0 {
        codes.push("pending_or_other".to_string());
    }
    anyhow::bail!(
        "deferred compact-MoE sticky error accumulator non-zero: mask={mask:#010x} codes=[{}]",
        codes.join(",")
    );
}

fn check_words(bytes: &[u8], labels: &[&'static str]) -> Result<()> {
    ensure!(
        bytes.len() == labels.len() * 4,
        "MoE status async: slot count mismatch"
    );
    for (i, (word, label)) in bytes.chunks_exact(4).zip(labels).enumerate() {
        let status = i32::from_le_bytes(word.try_into().expect("4-byte chunk"));
        abi::check_status(status).map_err(|error| {
            anyhow::anyhow!("deferred compact-MoE status #{i} ({label}): {error}: status={status}")
        })?;
    }
    Ok(())
}

/// Drop slots/sticky state queued by a chunk that failed before its flush; returns whether any
/// were dropped.
pub(crate) fn discard_stale() -> bool {
    let mut guard = STATE.lock().expect("MoE status state lock poisoned");
    match guard.as_mut() {
        Some(s) if !s.labels.is_empty() || s.sticky_pending => {
            if !s.labels.is_empty() {
                s.labels.clear();
                s.slots_dirty = true;
            }
            if s.sticky_pending {
                s.sticky_pending = false;
                s.sticky_dirty = true;
            }
            true
        }
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn words(v: &[i32]) -> Vec<u8> {
        v.iter().flat_map(|w| w.to_le_bytes()).collect()
    }

    #[test]
    fn every_word_must_be_planned() {
        assert!(check_words(&words(&[1, 1, 1]), &["routed", "shared", "routed"]).is_ok());
        // a still-PENDING word (-1) or any failure code fails closed and names the dispatch
        let err = check_words(&words(&[1, -1]), &["routed", "shared"])
            .unwrap_err()
            .to_string();
        assert!(err.contains("#1 (shared)"), "{err}");
        assert!(check_words(&words(&[1, 16]), &["routed", "routed"]).is_err());
        assert!(check_words(&words(&[1]), &["routed", "routed"]).is_err());
    }

    #[test]
    fn sticky_error_accumulator_accepts_zero_and_decodes_error_bits() {
        assert!(check_sticky_err(0).is_ok());
        let err_abi = check_sticky_err(1u32 << 10).unwrap_err().to_string();
        assert!(err_abi.contains("codes=[10]"), "{err_abi}");
        let err_multi = check_sticky_err((1u32 << 16) | (1u32 << 18) | STICKY_ERR_OTHER_BIT)
            .unwrap_err()
            .to_string();
        assert!(
            err_multi.contains("codes=[16,18,pending_or_other]"),
            "{err_multi}"
        );
    }

    #[test]
    fn capture_region_marks_status_flush_as_capturing() {
        let gpu = spark_runtime::gpu::mock::MockGpuBackend::new();
        assert!(!is_capturing(&gpu, 0));
        spark_runtime::gpu_capture_region::set_active_capture_region(Some("unit_test_moe_status"));
        assert!(is_capturing(&gpu, 0));
        assert!(flush(&gpu, 0).is_ok());
        spark_runtime::gpu_capture_region::set_active_capture_region(None);
        assert!(!is_capturing(&gpu, 0));
    }

    #[test]
    fn verify_prewarm_covers_256_rows_and_honors_need_shared_const() {
        assert_eq!(VERIFY_PREWARM_MAX_ROWS, 256);
        let gpu = spark_runtime::gpu::mock::MockGpuBackend::new();
        let mut state = State {
            slots: gpu.alloc(SLOTS * 4).unwrap(),
            slots_dirty: true,
            labels: Vec::new(),
            sticky_err: gpu.alloc(4).unwrap(),
            shared_status: gpu.alloc(4).unwrap(),
            sticky_pending: false,
            sticky_dirty: false,
            verify_prewarmed: false,
            constants: HashMap::new(),
        };

        // Before prewarm: when `shared_fast_v2` handles the shared expert and fused sort/plan
        // removes the routed constant, `need_routed_const = false, need_shared_const = false`
        // is capture-ready for rows = 160, 192, 256 without `"shared"` constants cached.
        for rows in [160usize, 192, 256] {
            assert!(
                state.is_capture_ready(rows, false, false),
                "rows={rows} should be capture-ready when neither constant is needed"
            );
            assert!(
                !state.is_capture_ready(rows, false, true),
                "rows={rows} must reject when shared constant is required but uncached"
            );
            assert!(
                !state.is_capture_ready(rows, true, false),
                "rows={rows} must reject when routed constant is required but uncached"
            );
        }

        // Dirty sticky words must always block capture readiness.
        state.sticky_dirty = true;
        assert!(!state.is_capture_ready(160, false, false));

        // Prewarm populates both `"shared"` and `"routed"` constants across `2..=256`
        // and cleans the sticky words.
        state
            .prewarm_verify_constants(&gpu, gpu.default_stream())
            .unwrap();
        assert!(!state.sticky_dirty);
        assert!(state.verify_prewarmed);
        for rows in [2usize, 16, 64, 128, 129, 160, 192, 256] {
            assert!(
                state.is_capture_ready(rows, true, true),
                "rows={rows} must be prewarmed for both routed and shared constants"
            );
            assert!(
                state.is_capture_ready(rows, false, false),
                "rows={rows} must be capture-ready with fast shared + fused sort"
            );
            let &(sh_ptr, sh_len) = state
                .constants
                .get(&("shared", rows))
                .expect("shared constant prewarmed");
            assert!(!sh_ptr.is_null());
            assert_eq!(
                sh_len,
                super::super::qwen4_shared_v7::shared_constant_len(rows)
            );
            let &(rt_ptr, rt_len) = state
                .constants
                .get(&("routed", rows))
                .expect("routed constant prewarmed");
            assert!(!rt_ptr.is_null());
            assert_eq!(rt_len, abi::CONTRACT_BYTES + 4);
        }

        // Above VERIFY_PREWARM_MAX_ROWS (257): only ready when neither constant is required.
        assert!(!state.is_capture_ready(257, true, false));
        assert!(!state.is_capture_ready(257, false, true));
        assert!(state.is_capture_ready(257, false, false));
    }

    struct StatusTestGpu {
        inner: spark_runtime::gpu::mock::MockGpuBackend,
        default_stream_id: u64,
        sync_h2d_calls: Mutex<usize>,
        async_h2d_streams: Mutex<Vec<u64>>,
        sync_streams: Mutex<Vec<u64>>,
        fail_d2h_ptr: Mutex<Option<DevicePtr>>,
    }

    impl StatusTestGpu {
        fn new(default_stream_id: u64) -> Self {
            Self {
                inner: spark_runtime::gpu::mock::MockGpuBackend::new(),
                default_stream_id,
                sync_h2d_calls: Mutex::new(0),
                async_h2d_streams: Mutex::new(Vec::new()),
                sync_streams: Mutex::new(Vec::new()),
                fail_d2h_ptr: Mutex::new(None),
            }
        }

        fn set_fail_d2h(&self, ptr: Option<DevicePtr>) {
            *self
                .fail_d2h_ptr
                .lock()
                .expect("fail_d2h_ptr lock poisoned") = ptr;
        }
    }

    impl GpuBackend for StatusTestGpu {
        fn alloc(&self, bytes: usize) -> Result<DevicePtr> {
            self.inner.alloc(bytes)
        }
        fn alloc_managed(&self, bytes: usize) -> Result<DevicePtr> {
            self.inner.alloc_managed(bytes)
        }
        fn free(&self, ptr: DevicePtr) -> Result<()> {
            self.inner.free(ptr)
        }
        fn copy_h2d(&self, src: &[u8], dst: DevicePtr) -> Result<()> {
            *self
                .sync_h2d_calls
                .lock()
                .expect("sync_h2d_calls lock poisoned") += 1;
            self.inner.copy_h2d(src, dst)
        }
        fn copy_h2d_async(&self, src: &[u8], dst: DevicePtr, stream: u64) -> Result<()> {
            self.async_h2d_streams
                .lock()
                .expect("async_h2d_streams lock poisoned")
                .push(stream);
            self.inner.copy_h2d(src, dst)
        }
        fn copy_d2h(&self, src: DevicePtr, dst: &mut [u8]) -> Result<()> {
            self.inner.copy_d2h(src, dst)
        }
        fn copy_d2h_on_stream(&self, src: DevicePtr, dst: &mut [u8], stream: u64) -> Result<()> {
            if *self
                .fail_d2h_ptr
                .lock()
                .expect("fail_d2h_ptr lock poisoned")
                == Some(src)
            {
                anyhow::bail!("injected D2H failure on stream {stream}");
            }
            self.inner.copy_d2h_on_stream(src, dst, stream)
        }
        fn copy_d2d(&self, src: DevicePtr, dst: DevicePtr, bytes: usize) -> Result<()> {
            self.inner.copy_d2d(src, dst, bytes)
        }
        fn launch(
            &self,
            func: KernelHandle,
            grid: [u32; 3],
            block: [u32; 3],
            shared_mem: u32,
            stream: u64,
            params: &mut [*mut std::ffi::c_void],
        ) -> Result<()> {
            self.inner
                .launch(func, grid, block, shared_mem, stream, params)
        }
        fn synchronize(&self, stream: u64) -> Result<()> {
            self.sync_streams
                .lock()
                .expect("sync_streams lock poisoned")
                .push(stream);
            self.inner.synchronize(stream)
        }
        fn default_stream(&self) -> u64 {
            self.default_stream_id
        }
        #[track_caller]
        fn kernel(&self, module: &str, func_name: &str) -> Result<KernelHandle> {
            self.inner.kernel(module, func_name)
        }
        fn memset(&self, ptr: DevicePtr, value: u8, bytes: usize) -> Result<()> {
            self.inner.memset(ptr, value, bytes)
        }
        fn memset_async(&self, ptr: DevicePtr, value: u8, bytes: usize, stream: u64) -> Result<()> {
            self.inner.memset_async(ptr, value, bytes, stream)
        }
        fn total_memory(&self) -> Result<usize> {
            self.inner.total_memory()
        }
        fn free_memory(&self) -> Result<usize> {
            self.inner.free_memory()
        }
    }

    #[test]
    fn defer_and_defer_shared_reject_cuda_graph_capture() {
        struct ClearCaptureGuard;
        impl Drop for ClearCaptureGuard {
            fn drop(&mut self) {
                spark_runtime::gpu_capture_region::set_active_capture_region(None);
            }
        }
        let gpu = spark_runtime::gpu::mock::MockGpuBackend::new();
        spark_runtime::gpu_capture_region::set_active_capture_region(Some("unit_test_defer"));
        let _guard = ClearCaptureGuard;
        assert!(defer_shared(&gpu, "shared", 0).is_err());
        assert!(defer(&gpu, DevicePtr(0x1000), "routed", 0).is_err());
    }

    #[test]
    fn prewarm_verify_constants_uses_stream_ordered_async_h2d_and_syncs() {
        let gpu = StatusTestGpu::new(0x42);
        let mut state = State {
            slots: gpu.alloc(SLOTS * 4).unwrap(),
            slots_dirty: true,
            labels: Vec::new(),
            sticky_err: gpu.alloc(4).unwrap(),
            shared_status: gpu.alloc(4).unwrap(),
            sticky_pending: false,
            sticky_dirty: true,
            verify_prewarmed: false,
            constants: HashMap::new(),
        };
        let stream = gpu.default_stream();
        state.prewarm_verify_constants(&gpu, stream).unwrap();
        assert_eq!(*gpu.sync_h2d_calls.lock().unwrap(), 0);
        assert_eq!(
            *gpu.async_h2d_streams.lock().unwrap(),
            vec![0x42, 0x42, 0x42]
        );
        assert_eq!(*gpu.sync_streams.lock().unwrap(), vec![0x42]);
        assert!(!state.sticky_dirty);
        assert!(state.verify_prewarmed);
    }

    #[test]
    fn flush_marks_slots_and_sticky_dirty_on_d2h_and_validation_errors() {
        let gpu = StatusTestGpu::new(0);
        let make_state = |slot_val: i32, sticky_val: u32, sticky_pending: bool| {
            let slots = gpu.alloc(SLOTS * 4).unwrap();
            let sticky_err = gpu.alloc(4).unwrap();
            let shared_status = gpu.alloc(4).unwrap();
            gpu.copy_h2d(&slot_val.to_le_bytes(), slots).unwrap();
            gpu.copy_h2d(&sticky_val.to_le_bytes(), sticky_err).unwrap();
            Mutex::new(Some(State {
                slots,
                slots_dirty: false,
                labels: vec!["routed"],
                sticky_err,
                shared_status,
                sticky_pending,
                sticky_dirty: false,
                verify_prewarmed: false,
                constants: HashMap::new(),
            }))
        };

        // 1. D2H failure on slots while sticky_pending=true marks BOTH dirty flags.
        let st = make_state(1, 0, true);
        let slots_ptr = st.lock().unwrap().as_ref().unwrap().slots;
        gpu.set_fail_d2h(Some(slots_ptr));
        assert!(flush_from_state(&gpu, 0, &st).is_err());
        gpu.set_fail_d2h(None);
        {
            let guard = st.lock().unwrap();
            let s = guard.as_ref().unwrap();
            assert!(s.slots_dirty);
            assert!(s.sticky_dirty);
        }

        // 2. check_words failure while sticky_pending=true marks BOTH dirty flags.
        let st = make_state(17, 0, true);
        assert!(flush_from_state(&gpu, 0, &st).is_err());
        {
            let guard = st.lock().unwrap();
            let s = guard.as_ref().unwrap();
            assert!(s.slots_dirty);
            assert!(s.sticky_dirty);
        }

        // 3. D2H failure on sticky_err marks sticky_dirty.
        let st = make_state(1, 0, true);
        let sticky_ptr = st.lock().unwrap().as_ref().unwrap().sticky_err;
        gpu.set_fail_d2h(Some(sticky_ptr));
        assert!(flush_from_state(&gpu, 0, &st).is_err());
        gpu.set_fail_d2h(None);
        {
            let guard = st.lock().unwrap();
            let s = guard.as_ref().unwrap();
            assert!(!s.slots_dirty);
            assert!(s.sticky_dirty);
        }

        // 4. Non-zero sticky_err mask marks sticky_dirty.
        let st = make_state(1, 1u32 << 16, true);
        assert!(flush_from_state(&gpu, 0, &st).is_err());
        {
            let guard = st.lock().unwrap();
            let s = guard.as_ref().unwrap();
            assert!(!s.slots_dirty);
            assert!(s.sticky_dirty);
        }
    }
}
