// SPDX-License-Identifier: AGPL-3.0-only

//! The name of the CUDA-graph capture region currently open on this thread,
//! if any.
//!
//! `spark-model`'s `CaptureGuard` (`crates/spark-model/src/model/trait_impl/
//! capture_guard.rs`) sets this on `begin`, clears it on `end`/`Drop`, and is
//! the only writer. It exists so a refusal deep inside the CUDA backend —
//! `ensure_bounded_wait_outside_capture` in `cuda_backend/gpu_impl.rs`, fired
//! when `ATLAS_BOUNDED_HOST_COPY` catches a blocking host wait issued while a
//! stream is being captured — can name *which* capture region triggered it.
//! Before this, the error said only "refuses a blocking wait during CUDA
//! graph capture", which was enough to diagnose the original bug (a
//! `tracing::enabled!`-gated debug sync inside the PLE suffix graph) only
//! because there was exactly one capture region to suspect at the time. With
//! nine capture regions across `trait_impl` and GLM-5.3, naming the region
//! is the difference between a five-minute fix and another audit.
//!
//! Thread-local, not global: every capture region in this tree runs its
//! `begin_capture`..`end_capture` span on a single calling thread (the
//! decode/verify loop that owns the stream), so a thread-local avoids any
//! cross-thread synchronization cost on the hot path while still being
//! correct — a region opened on one thread can never be observed as active
//! on another.

use std::cell::Cell;

thread_local! {
    static ACTIVE_CAPTURE_REGION: Cell<Option<&'static str>> = const { Cell::new(None) };
}

/// Set (or clear, with `None`) the name of the capture region open on this
/// thread. Called only by `CaptureGuard`.
pub fn set_active_capture_region(region: Option<&'static str>) {
    ACTIVE_CAPTURE_REGION.with(|cell| cell.set(region));
}

/// The name of the capture region currently open on this thread, or `None`
/// if no `CaptureGuard` is open here.
pub fn active_capture_region() -> Option<&'static str> {
    ACTIVE_CAPTURE_REGION.with(|cell| cell.get())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_to_none_and_round_trips() {
        assert_eq!(active_capture_region(), None);
        set_active_capture_region(Some("unit_test_region"));
        assert_eq!(active_capture_region(), Some("unit_test_region"));
        set_active_capture_region(None);
        assert_eq!(active_capture_region(), None);
    }
}
