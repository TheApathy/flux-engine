// SPDX-License-Identifier: AGPL-3.0-only

//! Opt-in bounded pageable host/device copies through pinned staging.
//!
//! CUDA's synchronous pageable copies and `cuStreamSynchronize` cannot be
//! interrupted. With `ATLAS_BOUNDED_HOST_COPY=1`, every blocking host-copy or
//! D2D completion boundary instead records an event and polls it against a
//! fixed deadline. Host copies additionally use process-lifetime page-locked
//! staging. A timeout poisons the facility: the event and staging are
//! deliberately retained because DMA completion is uncertain, and every later
//! bounded operation fails closed until process restart.

use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, MutexGuard};
use std::time::{Duration, Instant};

use anyhow::{Result, bail};

use super::{
    cuEventCreate, cuEventDestroy_v2, cuEventQuery, cuEventRecord, cuMemAllocHost_v2,
    cuMemcpyDtoDAsync_v2, cuMemcpyDtoHAsync_v2, cuMemcpyHtoDAsync_v2,
};
use crate::gpu::HostToDeviceCopy;

const STAGING_BYTES: usize = 16 * 1024 * 1024;
const TIMEOUT: Duration = Duration::from_secs(60);
static POISONED: AtomicBool = AtomicBool::new(false);
static COPY_GATE: Mutex<()> = Mutex::new(());

fn copy_guard() -> MutexGuard<'static, ()> {
    COPY_GATE
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PollDecision {
    Complete,
    Pending,
    Poison,
}

fn poll_decision(status: i32, timed_out: bool) -> PollDecision {
    match status {
        0 => PollDecision::Complete,
        600 if !timed_out => PollDecision::Pending,
        _ => PollDecision::Poison,
    }
}

fn enabled_from_value(value: Option<&str>) -> bool {
    value == Some("1")
}

pub(super) fn enabled() -> bool {
    static GATE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *GATE.get_or_init(|| {
        let enabled =
            enabled_from_value(std::env::var("ATLAS_BOUNDED_HOST_COPY").ok().as_deref());
        if enabled {
            tracing::info!(
                timeout_seconds = TIMEOUT.as_secs(),
                "ENGAGED ATLAS_BOUNDED_HOST_COPY: pageable host transfers use bounded pinned staging"
            );
        }
        enabled
    })
}

thread_local! {
    /// Deliberately leaked for process lifetime: pending DMA can outlive a timed-out caller.
    static STAGING: std::cell::Cell<*mut u8> = const { std::cell::Cell::new(std::ptr::null_mut()) };
}

fn ensure_healthy() -> Result<()> {
    ensure_state_healthy(&POISONED)
}

fn ensure_state_healthy(state: &AtomicBool) -> Result<()> {
    if state.load(Ordering::Acquire) {
        bail!("bounded host copy disabled after uncertain DMA completion; restart required");
    }
    Ok(())
}

fn poison() {
    poison_state(&POISONED);
}

fn poison_state(state: &AtomicBool) {
    state.store(true, Ordering::Release);
}

fn staging() -> Result<*mut u8> {
    STAGING.with(|cell| {
        if cell.get().is_null() {
            let mut ptr: *mut c_void = std::ptr::null_mut();
            // SAFETY: `&mut ptr` is a valid out-pointer; allocates `STAGING_BYTES` of page-locked
            // host memory on the active CUDA context (retained for process lifetime).
            let status = unsafe { cuMemAllocHost_v2(&mut ptr, STAGING_BYTES) };
            if status != 0 || ptr.is_null() {
                bail!(
                    "cuMemAllocHost_v2({STAGING_BYTES}) for bounded copy failed: status {status}"
                );
            }
            crate::alloc_receipt::GLOBAL.note_pinned_alloc(STAGING_BYTES);
            cell.set(ptr.cast());
        }
        Ok(cell.get())
    })
}

/// Wait for queued work without destroying or forgetting its event on uncertainty.
pub(super) fn drain(stream: u64, what: &str) -> Result<()> {
    let _guard = copy_guard();
    drain_locked(stream, what)
}

fn drain_locked(stream: u64, what: &str) -> Result<()> {
    ensure_healthy()?;
    let wfet_marker = super::wfet_sync::enqueue_marker(stream);
    let mut event: u64 = 0;
    // SAFETY: `&mut event` is a valid out-pointer for `cuEventCreate` with `CU_EVENT_DISABLE_TIMING`.
    let status = unsafe { cuEventCreate(&mut event, 0x02) };
    if status != 0 {
        poison();
        bail!("{what}: cuEventCreate failed: status {status}; context poisoned");
    }
    // SAFETY: `event` was just created above and `stream` is a caller-supplied CUDA stream handle.
    let status = unsafe { cuEventRecord(event, stream) };
    if status != 0 {
        // Event state is uncertain; deliberately leak it and reject every future copy.
        poison();
        bail!(
            "{what}: cuEventRecord failed: status {status}; event {event:#x} leaked, context poisoned"
        );
    }

    let deadline = Instant::now() + TIMEOUT;
    let mut spins = 0u32;
    loop {
        let parked = wfet_marker.is_some_and(|marker| !super::wfet_sync::park_slice(marker));
        // SAFETY: `event` is a live recorded CUDA event handle.
        let status = unsafe { cuEventQuery(event) };
        match poll_decision(status, Instant::now() >= deadline) {
            PollDecision::Complete => {
                // SAFETY: `cuEventQuery` reported `CUDA_SUCCESS`, so the event has completed
                // and is destroyed once here.
                let status = unsafe { cuEventDestroy_v2(event) };
                if status != 0 {
                    poison();
                    bail!("{what}: cuEventDestroy_v2 failed: status {status}; context poisoned");
                }
                return Ok(());
            }
            PollDecision::Pending => {}
            PollDecision::Poison => {
                poison();
                if status == 600 {
                    bail!(
                        "{what}: stream {stream:#x} did not complete within {}s; event {event:#x} leaked, context poisoned",
                        TIMEOUT.as_secs()
                    );
                }
                bail!(
                    "{what}: cuEventQuery failed: status {status}; event {event:#x} leaked, context poisoned"
                );
            }
        }
        if !parked {
            spins += 1;
            if spins < 2000 {
                std::hint::spin_loop();
            } else {
                std::thread::sleep(Duration::from_micros(50));
            }
        }
    }
}

fn d2h_after_drain_locked(src: u64, dst: &mut [u8], stream: u64, what: &str) -> Result<()> {
    let stage = staging()?;
    let mut off = 0usize;
    while off < dst.len() {
        let len = (dst.len() - off).min(STAGING_BYTES);
        // SAFETY: `stage` is a `STAGING_BYTES`-sized page-locked buffer (`len <= STAGING_BYTES`);
        // `COPY_GATE` serializes access and `drain_locked` below completes the DMA before reading.
        let status = unsafe { cuMemcpyDtoHAsync_v2(stage.cast(), src + off as u64, len, stream) };
        if status != 0 {
            poison();
            bail!("{what}: cuMemcpyDtoHAsync_v2 failed: status {status}; context poisoned");
        }
        drain_locked(stream, what)?;
        // SAFETY: (1) bounds: `off + len <= dst.len()` and `len <= STAGING_BYTES`;
        // (2) non-overlap: `stage` is a dedicated thread-local CUDA pinned allocation disjoint
        // from caller's `&mut [u8]` `dst`; (3) exclusivity: caller holds `COPY_GATE`;
        // (4) DMA ordering: `drain_locked` returned `Ok(())`, guaranteeing the device-to-host DMA
        // into `stage[..len]` has completed before host reads.
        unsafe { std::ptr::copy_nonoverlapping(stage, dst.as_mut_ptr().add(off), len) };
        off += len;
    }
    Ok(())
}

fn h2d_after_drain_locked(src: &[u8], dst: u64, stream: u64, what: &str) -> Result<()> {
    let stage = staging()?;
    let mut off = 0usize;
    while off < src.len() {
        let len = (src.len() - off).min(STAGING_BYTES);
        // SAFETY: (1) bounds: `off + len <= src.len()` and `len <= STAGING_BYTES`;
        // (2) non-overlap: `stage` is a dedicated thread-local CUDA pinned allocation disjoint
        // from caller's `&[u8]` `src`; (3) exclusivity: caller holds `COPY_GATE`;
        // (4) DMA ordering: any prior iteration's `drain_locked` completed before returning here,
        // so no in-flight GPU DMA reads `stage` while host writes `stage[..len]`.
        unsafe { std::ptr::copy_nonoverlapping(src.as_ptr().add(off), stage, len) };
        // SAFETY: `stage[..len]` was initialized above, `len <= STAGING_BYTES`, `COPY_GATE` is
        // held, and `drain_locked` below completes the DMA before `stage` is reused.
        let status = unsafe { cuMemcpyHtoDAsync_v2(dst + off as u64, stage.cast(), len, stream) };
        if status != 0 {
            poison();
            bail!("{what}: cuMemcpyHtoDAsync_v2 failed: status {status}; context poisoned");
        }
        // The staging buffer can be reused only after the DMA has completed.
        drain_locked(stream, what)?;
        off += len;
    }
    Ok(())
}

pub(super) fn d2d(src: u64, dst: u64, bytes: usize, stream: u64) -> Result<()> {
    let _guard = copy_guard();
    ensure_healthy()?;
    // SAFETY: Caller supplies valid device pointers `src` and `dst` of at least `bytes` bytes on
    // `stream`; `drain_locked` below polls completion within the bounded deadline.
    let status = unsafe { cuMemcpyDtoDAsync_v2(dst, src, bytes, stream) };
    if status != 0 {
        poison();
        bail!("bounded D2D: cuMemcpyDtoDAsync_v2 failed: status {status}; context poisoned");
    }
    drain_locked(stream, "bounded D2D: copy completion")
}

pub(super) fn d2h(src: u64, dst: &mut [u8], stream: u64) -> Result<()> {
    let _guard = copy_guard();
    ensure_healthy()?;
    drain_locked(stream, "bounded D2H: producer drain")?;
    d2h_after_drain_locked(src, dst, stream, "bounded D2H: copy")
}

pub(super) fn d2h_pair(
    first_src: u64,
    first_dst: &mut [u8],
    second_src: u64,
    second_dst: &mut [u8],
    stream: u64,
) -> Result<()> {
    let _guard = copy_guard();
    ensure_healthy()?;
    drain_locked(stream, "bounded D2H pair: producer drain")?;
    d2h_after_drain_locked(first_src, first_dst, stream, "bounded D2H pair: first copy")?;
    d2h_after_drain_locked(
        second_src,
        second_dst,
        stream,
        "bounded D2H pair: second copy",
    )
}

pub(super) fn h2d(src: &[u8], dst: u64, stream: u64) -> Result<()> {
    let _guard = copy_guard();
    ensure_healthy()?;
    drain_locked(stream, "bounded H2D: prior-work drain")?;
    h2d_after_drain_locked(src, dst, stream, "bounded H2D: copy")
}

pub(super) fn h2d_group(copies: &[HostToDeviceCopy<'_>], stream: u64) -> Result<()> {
    if copies.is_empty() {
        return Ok(());
    }
    let _guard = copy_guard();
    ensure_healthy()?;
    drain_locked(stream, "bounded H2D group: prior-work drain")?;
    for copy in copies.iter().copied() {
        h2d_after_drain_locked(copy.src(), copy.dst().0, stream, "bounded H2D group: copy")?;
    }
    Ok(())
}

#[cfg(test)]
mod state_tests {
    use super::*;

    #[test]
    fn completion_poll_never_treats_timeout_or_query_error_as_ready() {
        assert_eq!(poll_decision(0, false), PollDecision::Complete);
        assert_eq!(poll_decision(0, true), PollDecision::Complete);
        assert_eq!(poll_decision(600, false), PollDecision::Pending);
        assert_eq!(poll_decision(600, true), PollDecision::Poison);
        assert_eq!(poll_decision(719, false), PollDecision::Poison);
    }

    #[test]
    fn poison_is_sticky_and_refuses_future_transfers() {
        let state = AtomicBool::new(false);
        assert!(ensure_state_healthy(&state).is_ok());
        poison_state(&state);
        assert!(ensure_state_healthy(&state).is_err());
        assert!(ensure_state_healthy(&state).is_err());
    }

    #[test]
    fn bounded_mode_is_exact_opt_in() {
        assert!(!enabled_from_value(None));
        assert!(!enabled_from_value(Some("")));
        assert!(!enabled_from_value(Some("0")));
        assert!(!enabled_from_value(Some("true")));
        assert!(enabled_from_value(Some("1")));
    }
}
