// SPDX-License-Identifier: AGPL-3.0-only

//! Shutdown retirement of in-flight work.
//!
//! A shutdown first gives HTTP requests a bounded drain window, then releases
//! the serving state and waits a bounded grace for the scheduler thread to
//! exit. The scheduler only exits with no active or prefilling sequence, so a
//! long prefill or decode outliving the drain window used to keep it running
//! past the grace and into `fail_stop`. [`begin`] marks the point after the
//! drain window: from then on every cooperative boundary (between prefill
//! chunks, after each decode token) retires its sequence.
//!
//! Process-global like `tui::shutdown`: the retiring decision is a property
//! of the process lifecycle, read on the scheduler thread without a carrier.

use std::sync::atomic::{AtomicBool, Ordering};

static RETIRING: AtomicBool = AtomicBool::new(false);

/// Retire in-flight sequences at their next safe boundary. Idempotent.
pub(crate) fn begin() {
    if !RETIRING.swap(true, Ordering::SeqCst) {
        tracing::info!(
            "Scheduler retirement: in-flight sequences stop at their next safe boundary"
        );
    }
}

/// Has shutdown retirement begun?
pub(crate) fn requested() -> bool {
    RETIRING.load(Ordering::Acquire)
}
