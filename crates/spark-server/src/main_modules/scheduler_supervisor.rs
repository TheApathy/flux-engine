// SPDX-License-Identifier: AGPL-3.0-only

//! Panic supervision for the sole scheduler thread.
//!
//! Without this boundary a scheduler panic leaves the HTTP listener alive and
//! apparently ready, even though no thread can execute another request. Because
//! the workspace builds with `panic = "unwind"` (including release), this
//! boundary catches scheduler panics and publishes the process-fatal fault latch.

use std::panic::{AssertUnwindSafe, catch_unwind};

/// Run the scheduler body and latch its panic as a process-fatal serving fault.
///
/// The latch does not attempt in-process recovery. It makes new API requests
/// fail truthfully with 503 and health report the original reason until an
/// operator or supervisor restarts the process. Returns whether `body` panicked.
pub(crate) fn run_supervised(latch: &atlas_core::fault::FaultLatch, body: impl FnOnce()) -> bool {
    match catch_unwind(AssertUnwindSafe(body)) {
        Ok(()) => false,
        Err(payload) => {
            let message = if let Some(message) = payload.downcast_ref::<&str>() {
                *message
            } else if let Some(message) = payload.downcast_ref::<String>() {
                message.as_str()
            } else {
                "non-string panic payload"
            };
            let reason = format!(
                "scheduler thread panicked ({message}); no request can be served until restart"
            );
            if latch.latch(reason.clone()) {
                tracing::error!(target: "atlas::fault", "{reason}");
            }
            true
        }
    }
}

#[cfg(test)]
#[path = "scheduler_supervisor_tests.rs"]
mod tests;
