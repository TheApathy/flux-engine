// SPDX-License-Identifier: AGPL-3.0-only

//! Qwen4 native-batch route accounting.
//!
//! The model decides native versus serial inside `decode_batch` with
//! `spark_model::qwen4_batch_policy` and its per-width native capability.
//! This module asks the model for that same route (`Model::qwen4_decode_route`)
//! only to log and count; it never alters a route. A
//! request's intrinsic reason is logged once, at admission. Frame reasons are
//! counted as attempts and logged once per process per reason. Outcomes describe
//! the returned decode result, not GPU completion, sampling, or an execution-route receipt.

use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use spark_model::qwen4_batch_policy::{self, FrameRoute, SerialReason};
use spark_model::traits::{Model, SequenceState};

use crate::metrics::{
    QWEN4_DECODE_FRAME_OUTCOMES, QWEN4_DECODE_FRAMES, QWEN4_REQUEST_ROUTES, QWEN4_SERIAL_ROW_STEPS,
};

static MAX_BATCH_SIZE: AtomicUsize = AtomicUsize::new(1);
static LOGGED_FRAME_REASONS: Mutex<Vec<SerialReason>> = Mutex::new(Vec::new());

/// Called once when the scheduler thread starts. At C1 no request can share a
/// frame, so request routes are neither logged nor counted.
pub(super) fn configure(max_batch_size: usize) {
    MAX_BATCH_SIZE.store(max_batch_size, Ordering::Relaxed);
}

fn batching_admitted() -> bool {
    MAX_BATCH_SIZE.load(Ordering::Relaxed) > 1
}

/// Log and count the route a newly admitted request will take when it shares
/// a frame with eligible rows. `qwen4_qsa_required` is immutable from here on.
pub(super) fn record_request_admission(
    model: &dyn Model,
    seq: &SequenceState,
    prompt_tokens: usize,
    max_tokens: usize,
) {
    if !batching_admitted() {
        return;
    }
    let plan = match model.qwen4_batch_plan() {
        Ok(Some(plan)) => plan,
        Ok(None) => return,
        Err(error) => {
            // Load admission refuses every plan error; `decode_batch` would
            // also fail this request's first C2+ frame with the same message.
            tracing::error!("Qwen4 batch plan unavailable at admission: {error:#}");
            return;
        }
    };
    match qwen4_batch_policy::request_serial_reason(&plan, seq.qwen4_qsa_required) {
        None => {
            QWEN4_REQUEST_ROUTES
                .with_label_values(&["native", "none"])
                .inc();
            tracing::debug!(
                "QWEN4_REQUEST_ROUTE session={:016x} route=native prompt_tokens={prompt_tokens} max_tokens={max_tokens}",
                seq.session_hash
            );
        }
        Some(reason) => {
            QWEN4_REQUEST_ROUTES
                .with_label_values(&["serial", reason.label()])
                .inc();
            tracing::info!(
                "QWEN4_REQUEST_ROUTE session={:016x} route=serial reason={} prompt_tokens={prompt_tokens} max_tokens={max_tokens} remedy=\"{}\"",
                seq.session_hash,
                reason.label(),
                reason.remedy()
            );
        }
    }
}

/// Count legacy attempts and retain the shared pre-dispatch route query result.
/// A query error or non-Qwen4 route remains outside this route-specific census.
pub(super) fn record_decode_frame(
    model: &dyn Model,
    seqs: &[&mut SequenceState],
) -> Option<FrameRoute> {
    if seqs.len() < 2 {
        return None;
    }
    // A plan, width or capability error is reported by `decode_batch` itself,
    // to every row of this frame; counting it here would double-report it.
    let Ok(Some(route)) = model.qwen4_decode_route(seqs) else {
        return None;
    };
    match &route {
        FrameRoute::Native => QWEN4_DECODE_FRAMES
            .with_label_values(&["native", "none"])
            .inc(),
        FrameRoute::Serial { frame, rows } => {
            QWEN4_DECODE_FRAMES
                .with_label_values(&["serial", frame.label()])
                .inc();
            for reason in rows {
                QWEN4_SERIAL_ROW_STEPS
                    .with_label_values(&[reason.label()])
                    .inc();
            }
            log_first_serial_frame(*frame, rows.len());
        }
        FrameRoute::Single => {}
    }
    Some(route)
}

/// Called once after `decode_batch` returns. Failed serial frames may have
/// partially run rows; no per-row success or failure is inferred here.
pub(super) fn record_decode_frame_result(route: Option<FrameRoute>, succeeded: bool) {
    let (route, reason) = match route {
        Some(FrameRoute::Native) => ("native", "none"),
        Some(FrameRoute::Serial { frame, .. }) => ("serial", frame.label()),
        Some(FrameRoute::Single) | None => return,
    };
    let outcome = if succeeded { "succeeded" } else { "failed" };
    QWEN4_DECODE_FRAME_OUTCOMES
        .with_label_values(&[route, reason, outcome])
        .inc();
}

fn log_first_serial_frame(reason: SerialReason, rows: usize) {
    let mut logged = LOGGED_FRAME_REASONS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if logged.contains(&reason) {
        return;
    }
    logged.push(reason);
    tracing::info!(
        "QWEN4_SERIAL_FRAME_FIRST rows={rows} reason={} remedy=\"{}\" (first C{rows} serial frame for this reason; later frames are counted in atlas_qwen4_decode_frames_total)",
        reason.label(),
        reason.remedy()
    );
}

#[cfg(test)]
mod counter_tests {
    use super::*;

    static TEST_LOCK: Mutex<()> = Mutex::new(());

    fn outcome(route: &str, reason: &str, result: &str) -> u64 {
        QWEN4_DECODE_FRAME_OUTCOMES
            .with_label_values(&[route, reason, result])
            .get()
    }

    #[test]
    fn returned_native_results_have_distinct_outcomes() {
        let _lock = TEST_LOCK.lock().unwrap();
        let success = outcome("native", "none", "succeeded");
        let failure = outcome("native", "none", "failed");
        record_decode_frame_result(Some(FrameRoute::Native), true);
        assert_eq!(outcome("native", "none", "succeeded"), success + 1);
        assert_eq!(outcome("native", "none", "failed"), failure);
        record_decode_frame_result(Some(FrameRoute::Native), false);
        assert_eq!(outcome("native", "none", "succeeded"), success + 1);
        assert_eq!(outcome("native", "none", "failed"), failure + 1);
    }

    #[test]
    fn failed_serial_frame_does_not_infer_successful_rows_or_repeat_attempts() {
        let _lock = TEST_LOCK.lock().unwrap();
        let reason = SerialReason::QsaRequired;
        let success = outcome("serial", reason.label(), "succeeded");
        let failure = outcome("serial", reason.label(), "failed");
        let attempts = QWEN4_DECODE_FRAMES
            .with_label_values(&["serial", reason.label()])
            .get();
        let row_attempts = QWEN4_SERIAL_ROW_STEPS
            .with_label_values(&[reason.label()])
            .get();
        record_decode_frame_result(
            Some(FrameRoute::Serial {
                frame: reason,
                rows: vec![reason, reason],
            }),
            false,
        );
        assert_eq!(outcome("serial", reason.label(), "succeeded"), success);
        assert_eq!(outcome("serial", reason.label(), "failed"), failure + 1);
        assert_eq!(
            QWEN4_DECODE_FRAMES
                .with_label_values(&["serial", reason.label()])
                .get(),
            attempts
        );
        assert_eq!(
            QWEN4_SERIAL_ROW_STEPS
                .with_label_values(&[reason.label()])
                .get(),
            row_attempts
        );
    }

    #[test]
    fn single_and_unresolved_routes_do_not_gain_native_outcomes() {
        let _lock = TEST_LOCK.lock().unwrap();
        let success = outcome("native", "none", "succeeded");
        let failure = outcome("native", "none", "failed");
        record_decode_frame_result(None, true);
        record_decode_frame_result(None, false);
        record_decode_frame_result(Some(FrameRoute::Single), true);
        record_decode_frame_result(Some(FrameRoute::Single), false);
        assert_eq!(outcome("native", "none", "succeeded"), success);
        assert_eq!(outcome("native", "none", "failed"), failure);
    }
}
