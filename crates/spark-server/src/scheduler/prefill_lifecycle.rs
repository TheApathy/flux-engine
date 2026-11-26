// SPDX-License-Identifier: AGPL-3.0-only

//! Cooperative cancellation and deadline retirement for prefill work.
//!
//! CUDA kernels are not preemptible. These checks therefore run immediately
//! before the first allocation, after the first GPU chunk drains, and between
//! every later chunk. Retirement uses `Model::free_sequence`, which also
//! resolves any DFlash async proposal and releases both proposer arms before
//! their per-request buffers can be reused.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use spark_model::traits::{Model, SequenceState};

use super::lifecycle::send_error_to_sink;
use super::prefill_stream_barrier::drain_before_prefill_cleanup;
use super::types::{PrefillInProgress, ResponseSink};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PrefillStop {
    Cancelled,
    TimedOut,
    /// The response sink's receiver is gone (client disconnected or the HTTP
    /// handler was dropped). Nothing can consume the result, so the prompt
    /// must not be prefilled — a queued burst of abandoned requests otherwise
    /// costs one full prefill each before the first failed send notices.
    Disconnected,
    /// Shutdown retirement began ([`super::retirement`]); the request's own
    /// drain window is over and the scheduler must exit.
    Retiring,
}

fn stop_reason(
    cancel_flag: Option<&Arc<AtomicBool>>,
    timeout_at: Option<Instant>,
    sink_closed: bool,
    retiring: bool,
    now: Instant,
) -> Option<PrefillStop> {
    if cancel_flag.is_some_and(|flag| flag.load(Ordering::Acquire)) {
        Some(PrefillStop::Cancelled)
    } else if sink_closed {
        Some(PrefillStop::Disconnected)
    } else if retiring {
        Some(PrefillStop::Retiring)
    } else if timeout_at.is_some_and(|deadline| now >= deadline) {
        Some(PrefillStop::TimedOut)
    } else {
        None
    }
}

fn notify_if_live(sink: &mut ResponseSink, reason: PrefillStop) {
    // A cancelled or disconnected peer has already dropped its receiver.
    // Avoid a guaranteed failed send (and noisy warning); a server-side
    // deadline can still have a live peer and receives an explicit terminal
    // error.
    match reason {
        PrefillStop::TimedOut => {
            tracing::warn!("Request timeout: reason=total_deadline phase=prefill");
            send_error_to_sink(sink, "Request timed out during prefill")
        }
        PrefillStop::Retiring => {
            send_error_to_sink(sink, "Server shutting down; prefill was not completed")
        }
        PrefillStop::Cancelled | PrefillStop::Disconnected => {}
    }
}

pub(super) fn stop_unallocated_prefill(
    cancel_flag: Option<&Arc<AtomicBool>>,
    timeout_at: Option<Instant>,
    sink: &mut ResponseSink,
    stage: &'static str,
) -> bool {
    let Some(reason) = stop_reason(
        cancel_flag,
        timeout_at,
        sink.is_closed(),
        super::retirement::requested(),
        Instant::now(),
    ) else {
        return false;
    };
    notify_if_live(sink, reason);
    tracing::info!(?reason, stage, "Retired prefill before sequence allocation");
    true
}

/// Whether an allocated prefill must stop before its next chunk. The caller
/// stops launching chunks and lets [`stop_allocated_prefill`] retire it at the
/// boundary it already owns.
pub(super) fn allocated_prefill_should_stop(
    cancel_flag: Option<&Arc<AtomicBool>>,
    timeout_at: Option<Instant>,
    sink: &ResponseSink,
) -> bool {
    stop_reason(
        cancel_flag,
        timeout_at,
        sink.is_closed(),
        super::retirement::requested(),
        Instant::now(),
    )
    .is_some()
}

#[allow(clippy::too_many_arguments)]
pub(super) fn stop_allocated_prefill(
    model: &dyn Model,
    prefill_stream: u64,
    seq: &mut SequenceState,
    reset_proof: &mut Option<super::continuation::ResetProofPayload>,
    sink: &mut ResponseSink,
    cancel_flag: Option<&Arc<AtomicBool>>,
    timeout_at: Option<Instant>,
    stage: &'static str,
) -> bool {
    let Some(reason) = stop_reason(
        cancel_flag,
        timeout_at,
        sink.is_closed(),
        super::retirement::requested(),
        Instant::now(),
    ) else {
        return false;
    };
    notify_if_live(sink, reason);
    // The cancellation/deadline may have arrived while this stream's prior
    // chunk was still executing. `free_sequence` releases model-owned buffers;
    // draining both the exact producer stream and the default consumer stream
    // is therefore a mandatory lifetime barrier, not an optimization.
    drain_before_prefill_cleanup(model, prefill_stream, stage);
    if let Err(error) =
        super::continuation::ContinuationRecord::abort_prefill_sequence(model, seq, reset_proof)
    {
        tracing::error!(
            ?reason,
            stage,
            "Prefill retirement free_sequence failed: {error:#}"
        );
    }
    if let Err(error) = model.ep_broadcast_cmd(0xFFFFFFF1) {
        tracing::error!(
            ?reason,
            stage,
            "Prefill retirement EP reset failed: {error:#}"
        );
    }
    crate::metrics::compat_metrics::set_kv_blocks_free(model.num_free_blocks());
    tracing::info!(?reason, stage, "Retired allocated prefill at safe boundary");
    true
}

pub(super) fn retire_stopped_prefills(
    model: &dyn Model,
    prefilling: &mut Vec<PrefillInProgress>,
    prefill_stream: u64,
) {
    for index in (0..prefilling.len()).rev() {
        let reason = {
            let prefill = &prefilling[index];
            stop_reason(
                prefill.cancel_flag.as_ref(),
                prefill.timeout_at,
                prefill.sink.is_closed(),
                super::retirement::requested(),
                Instant::now(),
            )
        };
        let Some(reason) = reason else {
            continue;
        };
        let mut prefill = prefilling.swap_remove(index);
        notify_if_live(&mut prefill.sink, reason);
        drain_before_prefill_cleanup(model, prefill_stream, "chunked-prefill retirement");
        if let Err(error) = super::continuation::ContinuationRecord::abort_prefill_sequence(
            model,
            &mut prefill.seq,
            &mut prefill.reset_proof,
        ) {
            tracing::error!(
                ?reason,
                "Chunked-prefill retirement free_sequence failed: {error:#}"
            );
        }
        if let Err(error) = model.ep_broadcast_cmd(0xFFFFFFF1) {
            tracing::error!(
                ?reason,
                "Chunked-prefill retirement EP reset failed: {error:#}"
            );
        }
        crate::metrics::compat_metrics::set_kv_blocks_free(model.num_free_blocks());
        tracing::info!(
            ?reason,
            completed_tokens = prefill.chunk_offset,
            prompt_tokens = prefill.prompt_tokens.len(),
            "Retired chunked prefill at safe boundary"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scheduler::test_fixtures::{model, prefill_in_progress};

    #[test]
    fn cancellation_precedes_deadline_and_is_acquire_loaded() {
        let flag = Arc::new(AtomicBool::new(true));
        let now = Instant::now();
        assert_eq!(
            stop_reason(Some(&flag), Some(now), true, true, now),
            Some(PrefillStop::Cancelled)
        );
    }

    #[test]
    fn disconnect_precedes_deadline_and_never_notifies() {
        let now = Instant::now();
        assert_eq!(
            stop_reason(None, Some(now), true, true, now),
            Some(PrefillStop::Disconnected)
        );
        assert_eq!(
            stop_reason(None, None, true, false, now),
            Some(PrefillStop::Disconnected)
        );
        // A blocking sink whose oneshot receiver was dropped reads as closed.
        let (tx, rx) = tokio::sync::oneshot::channel();
        let mut sink = ResponseSink::Blocking(Some(tx));
        assert!(!sink.is_closed());
        drop(rx);
        assert!(sink.is_closed());
        notify_if_live(&mut sink, PrefillStop::Disconnected);
        assert!(matches!(sink, ResponseSink::Blocking(Some(_))));
    }

    #[test]
    fn deadlines_are_inclusive_and_future_deadlines_continue() {
        let now = Instant::now();
        assert_eq!(
            stop_reason(None, Some(now), false, false, now),
            Some(PrefillStop::TimedOut)
        );
        assert_eq!(
            stop_reason(
                None,
                Some(now + std::time::Duration::from_secs(1)),
                false,
                false,
                now
            ),
            None
        );
        assert_eq!(stop_reason(None, None, false, false, now), None);
    }

    #[test]
    fn shutdown_retirement_precedes_deadline_and_tells_a_live_peer() {
        let now = Instant::now();
        assert_eq!(
            stop_reason(None, Some(now), false, true, now),
            Some(PrefillStop::Retiring)
        );
        assert_eq!(
            stop_reason(None, None, false, true, now),
            Some(PrefillStop::Retiring)
        );
        let (tx, mut rx) = tokio::sync::oneshot::channel();
        let mut sink = ResponseSink::Blocking(Some(tx));
        notify_if_live(&mut sink, PrefillStop::Retiring);
        let Ok(Err(error)) = rx.try_recv() else {
            panic!("a live peer must receive the shutdown error");
        };
        assert!(error.to_string().contains("shutting down"), "{error}");
    }

    #[test]
    fn every_prefill_stage_is_wired_to_the_retirement_gate() {
        const FIRST_CHUNK: &str = include_str!("prefill_a_step.rs");
        const SINGLE_SHOT: &str = include_str!("prefill_b_step.rs");
        const CONTINUATION: &str = include_str!("phase_continue_prefills.rs");
        assert_eq!(FIRST_CHUNK.matches("stop_unallocated_prefill(").count(), 1);
        assert_eq!(FIRST_CHUNK.matches("stop_allocated_prefill(").count(), 1);
        assert_eq!(SINGLE_SHOT.matches("stop_unallocated_prefill(").count(), 1);
        assert_eq!(SINGLE_SHOT.matches("stop_allocated_prefill(").count(), 1);
        assert_eq!(CONTINUATION.matches("retire_stopped_prefills(").count(), 1);
    }

    /// A disconnect between chunks must drain the exact producer stream and
    /// the default consumer stream before the slot is returned to the pool.
    #[test]
    fn chunked_disconnect_drains_before_freeing_sequence_state() {
        let m = model::recording();
        let (tx, rx) = tokio::sync::mpsc::channel(1);
        drop(rx);
        let mut prefilling = vec![prefill_in_progress(ResponseSink::Streaming(tx))];

        retire_stopped_prefills(&m, &mut prefilling, 7);

        assert!(prefilling.is_empty());
        assert_eq!(
            *m.lifecycle_events.lock().unwrap(),
            ["sync:7", "sync:0", "free:0"],
            "no sequence state is freed while either stream can still reference it"
        );
    }
}
