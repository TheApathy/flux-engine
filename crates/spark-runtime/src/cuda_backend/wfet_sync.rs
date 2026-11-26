// SPDX-License-Identifier: AGPL-3.0-only
//
// ARMv9.2 `WFET` stream completion park over GB10 NVLink-C2C coherent memory.
//
// Replaces driver busy-spin (`cuStreamSynchronize` default `SPIN`) and 50 us
// OS timer-slack sleep (`bounded_copy`) with an inline GPU command-processor
// `cuStreamWriteValue32_v2` write to a per-thread mapped pinned cache line
// followed by ARMv9.2 `WFET` (`Wait For Event with Timeout`).
//
// While the GPU stream executes, the Grace ARM core sleeps at near-zero dynamic
// power (freeing shared 140W SoC thermal headroom) and wakes in < 0.2 us via
// hardware cache-line invalidation the instant the stream completes, before
// handing off to `cuStreamSynchronize` / `cuEventQuery` for final CUDA status
// verification.

use std::cell::Cell;
use std::ffi::c_void;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::time::{Duration, Instant};

use super::{
    cuMemHostAlloc, cuMemHostGetDevicePointer_v2, cuStreamIsCapturing, cuStreamQuery,
    cuStreamWriteValue32_v2,
};

const CU_MEMHOSTALLOC_PORTABLE: u32 = 0x01;
const CU_MEMHOSTALLOC_DEVICEMAP: u32 = 0x02;
const CUDA_ERROR_NOT_READY: i32 = 600;
const SLOT_BYTES: usize = 64;
const MAX_PARK_WALL_TIME: Duration = Duration::from_secs(15);

static RUNTIME_DISABLE: AtomicBool = AtomicBool::new(false);

pub(super) fn enabled() -> bool {
    if RUNTIME_DISABLE.load(Ordering::Relaxed) {
        return false;
    }
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        if std::env::var("ATLAS_CUDA_WFET_SYNC").ok().as_deref() == Some("0") {
            return false;
        }
        let supported = arch_wfxt_supported();
        if supported {
            tracing::info!(
                "ENGAGED ATLAS_CUDA_WFET_SYNC: ARMv9.2 WFET coherent stream completion park active"
            );
        }
        supported
    })
}

#[cfg(all(target_arch = "aarch64", target_os = "linux"))]
fn arch_wfxt_supported() -> bool {
    // SAFETY: glibc/musl `getauxval(unsigned long) -> unsigned long` on 64-bit AArch64 Linux.
    unsafe extern "C" {
        fn getauxval(kind: u64) -> u64;
    }
    const AT_HWCAP2: u64 = 26;
    const HWCAP2_WFXT: u64 = 1 << 31;
    // SAFETY: `getauxval(AT_HWCAP2)` is a pure libc query reading the kernel auxiliary vector.
    let hwcap2 = unsafe { getauxval(AT_HWCAP2) };
    (hwcap2 & HWCAP2_WFXT) != 0
}

#[cfg(not(all(target_arch = "aarch64", target_os = "linux")))]
fn arch_wfxt_supported() -> bool {
    false
}

#[derive(Clone, Copy)]
struct ThreadSlot {
    host_ptr: *const AtomicU32,
    dev_ptr: u64,
    seq: u32,
}

thread_local! {
    /// Deliberately leaked 64-byte portable device-mapped cache line per calling thread.
    static SLOT: Cell<Option<ThreadSlot>> = const { Cell::new(None) };
}

fn acquire_slot() -> Option<(ThreadSlot, u32)> {
    SLOT.with(|cell| {
        let mut slot = match cell.get() {
            Some(existing) => existing,
            None => {
                let mut raw: *mut c_void = std::ptr::null_mut();
                // SAFETY: Allocates a 64-byte portable, device-mapped pinned cache line valid across CUDA contexts.
                let alloc_rc = unsafe {
                    cuMemHostAlloc(
                        &mut raw,
                        SLOT_BYTES,
                        CU_MEMHOSTALLOC_PORTABLE | CU_MEMHOSTALLOC_DEVICEMAP,
                    )
                };
                if alloc_rc != 0 || raw.is_null() {
                    RUNTIME_DISABLE.store(true, Ordering::Relaxed);
                    return None;
                }
                crate::alloc_receipt::GLOBAL.note_pinned_alloc(SLOT_BYTES);
                // SAFETY: `raw` points to `SLOT_BYTES` freshly allocated host bytes.
                unsafe { std::ptr::write_bytes(raw.cast::<u8>(), 0, SLOT_BYTES) };
                let mut dptr: u64 = 0;
                // SAFETY: `raw` was allocated with `CU_MEMHOSTALLOC_DEVICEMAP`.
                let map_rc = unsafe { cuMemHostGetDevicePointer_v2(&mut dptr, raw, 0) };
                if map_rc != 0 || dptr == 0 {
                    RUNTIME_DISABLE.store(true, Ordering::Relaxed);
                    return None;
                }
                ThreadSlot {
                    host_ptr: raw.cast::<AtomicU32>(),
                    dev_ptr: dptr,
                    seq: 0,
                }
            }
        };
        let mut next = slot.seq.wrapping_add(1);
        if next == 0 {
            next = 1;
        }
        slot.seq = next;
        cell.set(Some(slot));
        Some((slot, next))
    })
}

#[cfg(all(target_arch = "aarch64", target_os = "linux"))]
#[inline]
fn cntfrq_ticks_100us() -> u64 {
    static TICKS: OnceLock<u64> = OnceLock::new();
    *TICKS.get_or_init(|| {
        let freq: u64;
        // SAFETY: `cntfrq_el0` is an unprivileged EL0 system register on AArch64 Linux.
        unsafe {
            core::arch::asm!("mrs {}, cntfrq_el0", out(reg) freq, options(nomem, nostack));
        }
        // 100 us window in generic timer ticks (100_000 on 1 GHz Grace timer).
        (freq / 10_000).max(1_000)
    })
}

#[cfg(all(target_arch = "aarch64", target_os = "linux"))]
#[inline]
fn wfet_wait_step(host_ptr: *const AtomicU32, expected: u32, delta_ticks: u64) -> bool {
    // SAFETY: `host_ptr` points to a leaked 64-byte aligned `AtomicU32` slot owned by this thread.
    let atomic = unsafe { &*host_ptr };
    if atomic.load(Ordering::Acquire) == expected {
        return true;
    }
    // SAFETY: `arch_wfxt_supported()` verified `HWCAP2_WFXT` before enabling this module;
    // `host_ptr` is a valid 4-byte aligned address for `ldxr`.
    unsafe {
        let now: u64;
        core::arch::asm!("mrs {}, cntvct_el0", out(reg) now, options(nomem, nostack));
        let deadline = now.wrapping_add(delta_ticks);
        // Clear any pending stale event bit with `sevl; wfe` before arming the
        // exclusive monitor (`ldxr`) and parking until cache-line invalidation
        // or `deadline` (`wfet`).
        core::arch::asm!(
            ".arch armv8.7-a",
            "sevl",
            "wfe",
            "ldxr {val:w}, [{addr}]",
            "cmp {val:w}, {exp:w}",
            "b.eq 2f",
            "wfet {dl}",
            "2:",
            "clrex",
            ".arch armv8-a",
            addr = in(reg) host_ptr,
            exp = in(reg) expected,
            dl = in(reg) deadline,
            val = out(reg) _,
            options(nostack),
        );
    }
    atomic.load(Ordering::Acquire) == expected
}

/// Enqueue a coherent `cuStreamWriteValue32_v2` marker on `stream` (when outside
/// CUDA graph capture) and return `(host_ptr, expected_seq)` for `WFET` parking.
pub(super) fn enqueue_marker(stream: u64) -> Option<(*const AtomicU32, u32)> {
    if !enabled() {
        return None;
    }
    let mut capture_status: u32 = 0;
    // SAFETY: Queries stream capture status; non-zero return or active capture declines marker injection.
    if unsafe { cuStreamIsCapturing(stream, &mut capture_status) } != 0 || capture_status != 0 {
        return None;
    }
    let (slot, target_seq) = acquire_slot()?;
    // SAFETY: `slot.dev_ptr` is a valid 4-byte aligned device-mapped host pointer; flag 0
    // (`CU_STREAM_WRITE_VALUE_DEFAULT`) emits a stream-ordered memory fence before writing.
    let rc = unsafe { cuStreamWriteValue32_v2(stream, slot.dev_ptr, target_seq, 0) };
    if rc != 0 {
        if rc == 801 {
            // CUDA_ERROR_NOT_SUPPORTED
            RUNTIME_DISABLE.store(true, Ordering::Relaxed);
        }
        return None;
    }
    Some((slot.host_ptr, target_seq))
}

/// Park the calling ARMv9.2 core via `WFET` for one bounded slice (up to ~100 us).
/// Returns `true` if the marker has already retired.
#[inline]
pub(super) fn park_slice(marker: (*const AtomicU32, u32)) -> bool {
    #[cfg(all(target_arch = "aarch64", target_os = "linux"))]
    {
        wfet_wait_step(marker.0, marker.1, cntfrq_ticks_100us())
    }
    #[cfg(not(all(target_arch = "aarch64", target_os = "linux")))]
    {
        let _ = marker;
        false
    }
}

/// Park the calling thread at near-zero CPU power until `stream` completes,
/// faults, or reaches a bounded safety deadline, before `cuStreamSynchronize`
/// performs final CUDA status verification.
pub(super) fn park_stream_if_enabled(stream: u64) {
    let Some(marker) = enqueue_marker(stream) else {
        return;
    };
    let start = Instant::now();
    while !park_slice(marker) {
        // SAFETY: Non-blocking stream status poll; exits within <= 100 us if a prior
        // kernel faulted (aborting the stream before `WriteValue32`) or already retired.
        if unsafe { cuStreamQuery(stream) } != CUDA_ERROR_NOT_READY
            || start.elapsed() >= MAX_PARK_WALL_TIME
        {
            break;
        }
    }
}
