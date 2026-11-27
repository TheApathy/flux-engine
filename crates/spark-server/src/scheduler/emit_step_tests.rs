// SPDX-License-Identifier: AGPL-3.0-only

//! GPU-free tests for the emit path: cancellation/disconnect observation,
//! the content-token budget and bounded streaming sends.

use super::*;
use crate::scheduler::test_fixtures::active_seq;

/// H2: a sink whose receiver is gone is a cancelled request. Blocking
/// requests carry no `cancel_flag`, so this is their only disconnect signal.
#[test]
fn a_closed_sink_counts_as_cancellation_for_blocking_and_streaming_requests() {
    let (tx, rx) = tokio::sync::oneshot::channel();
    let a = active_seq(ResponseSink::Blocking(Some(tx)), 0);
    assert!(!request_cancelled(&a), "live blocking client");
    drop(rx);
    assert!(request_cancelled(&a), "blocking client went away");

    let (tx, rx) = tokio::sync::mpsc::channel(1);
    let a = active_seq(ResponseSink::Streaming(tx), 0);
    assert!(!request_cancelled(&a), "live streaming client");
    drop(rx);
    assert!(request_cancelled(&a), "streaming client went away");

    // A terminal response already taken is not a disconnect: the sequence
    // is finishing anyway and must not be re-routed through cancellation.
    let a = active_seq(ResponseSink::Blocking(None), 0);
    assert!(!request_cancelled(&a));
}

/// H2: speculative emit retires a sequence whose client is gone before
/// committing the token, exactly like an explicit cancel flag.
#[test]
fn emit_token_retires_a_sequence_whose_blocking_client_disconnected() {
    let (tx, rx) = tokio::sync::oneshot::channel();
    let mut a = active_seq(ResponseSink::Blocking(Some(tx)), 0);
    drop(rx);
    let before = a.output_tokens.clone();

    emit_token(&mut a, 42, None);

    assert!(a.finished);
    assert_eq!(a.output_tokens, before, "the token is not committed");
}

#[test]
fn speculative_think_end_watchdog_is_reported_only_when_it_fires() {
    let ledger = std::sync::Arc::new(std::sync::Mutex::new(
        crate::openai::InterventionLedger::default(),
    ));
    let mut a = active_seq(ResponseSink::Blocking(None), 0);
    a.think_end_token = Some(29);
    a.think_skip_count = 48;
    a.intervention_ledger = Some(std::sync::Arc::clone(&ledger));

    emit_token(&mut a, 29, None);
    assert!(!a.finished);
    assert!(
        crate::openai::snapshot_shared_interventions(Some(&ledger))
            .unwrap()
            .events
            .is_empty()
    );

    emit_token(&mut a, 29, None);
    assert!(a.finished);
    let snapshot = crate::openai::snapshot_shared_interventions(Some(&ledger)).unwrap();
    assert_eq!(snapshot.events.len(), 1);
    assert_eq!(snapshot.events[0].code, "repeated_think_end_watchdog");
    assert_eq!(snapshot.events[0].action, "terminate_generation");
    assert_eq!(snapshot.events[0].count, 1);
}

#[test]
fn speculative_think_end_watchdog_keeps_policy_without_a_ledger() {
    let mut a = active_seq(ResponseSink::Blocking(None), 0);
    a.think_end_token = Some(29);
    a.think_skip_count = 49;

    emit_token(&mut a, 29, None);

    assert!(a.finished);
    assert!(a.intervention_ledger.is_none());
}

/// M3: `remaining` is a budget, not a counter that may wrap. Release
/// builds have overflow checks off, so a `-= 1` at zero became
/// `usize::MAX` and the admission gate then reserved the whole KV pool
/// for this one sequence until it retired.
#[test]
fn emit_token_saturates_an_exhausted_content_budget() {
    let (tx, _rx) = tokio::sync::mpsc::channel(4);
    let mut a = active_seq(ResponseSink::Streaming(tx), 0);
    a.remaining = 0;
    emit_token(&mut a, 42, None);
    assert_eq!(a.remaining, 0, "ordinary content token");

    let (tx, _rx) = tokio::sync::mpsc::channel(4);
    let mut a = active_seq(ResponseSink::Streaming(tx), 0);
    a.remaining = 0;
    let close = a
        .tool_call_end_token
        .expect("fixture has a tool-call close");
    emit_token(&mut a, close, None);
    assert_eq!(a.remaining, 0, "tool-call close token");
}

/// H3: scheduler-owned stream delivery is strictly nonblocking. The channel
/// is the burst tolerance; once it is full, that sequence is retired so one
/// stalled peer cannot add latency to another sequence.
mod bounded_stream_send {
    use super::*;
    use std::time::Duration;

    fn full_channel() -> (
        tokio::sync::mpsc::Sender<StreamEvent>,
        tokio::sync::mpsc::Receiver<StreamEvent>,
    ) {
        let (tx, rx) = tokio::sync::mpsc::channel(1);
        tx.try_send(StreamEvent::Token(0)).expect("one slot");
        (tx, rx)
    }

    #[test]
    fn delivers_immediately_when_the_channel_has_room() {
        let (tx, mut rx) = tokio::sync::mpsc::channel(1);
        assert!(stream_send_bounded(&tx, StreamEvent::Token(7), "t"));
        assert!(matches!(rx.try_recv(), Ok(StreamEvent::Token(7))));
    }

    #[test]
    fn gives_up_immediately_on_a_consumer_that_does_not_drain() {
        let (tx, _rx) = full_channel();
        let started = Instant::now();
        assert!(!stream_send_bounded(&tx, StreamEvent::Token(7), "t"));
        let waited = started.elapsed();
        assert!(waited < Duration::from_millis(50), "{waited:?}");
    }

    /// Hostile two-lane production shape: lane A's client has filled its
    /// channel, but lane B must still publish in the same scheduler pass.
    #[test]
    fn a_healthy_sequence_progresses_while_another_client_is_stalled() {
        let (stalled_tx, _stalled_rx) = full_channel();
        let (healthy_tx, mut healthy_rx) = tokio::sync::mpsc::channel(1);
        let started = Instant::now();

        assert!(!stream_send_bounded(
            &stalled_tx,
            StreamEvent::Token(7),
            "stalled"
        ));
        assert!(stream_send_bounded(
            &healthy_tx,
            StreamEvent::Token(8),
            "healthy"
        ));

        assert!(started.elapsed() < Duration::from_millis(50));
        assert!(matches!(healthy_rx.try_recv(), Ok(StreamEvent::Token(8))));
    }

    #[test]
    fn a_dropped_receiver_is_terminal_without_waiting() {
        let (tx, rx) = tokio::sync::mpsc::channel(1);
        drop(rx);
        let started = Instant::now();
        assert!(!stream_send_bounded(&tx, StreamEvent::Token(7), "t"));
        assert!(started.elapsed() < Duration::from_millis(50));
    }

    /// Every scheduler streaming send, including all three first-token sites
    /// and terminal Done/Error delivery, goes through the nonblocking helper.
    #[test]
    fn every_scheduler_stream_send_is_nonblocking() {
        let emit = include_str!("emit_step.rs");
        let serial = include_str!("decode_logits_step.rs");
        let lifecycle = include_str!("lifecycle.rs");
        let first_a = include_str!("prefill_a_step.rs");
        let first_b = include_str!("prefill_b_step.rs");
        let promote = include_str!("phase_promote_prefills.rs");
        assert_eq!(emit.matches("blocking_send(").count(), 0);
        assert_eq!(serial.matches("blocking_send(").count(), 0);
        assert_eq!(lifecycle.matches("blocking_send(").count(), 0);
        assert_eq!(first_a.matches("blocking_send(").count(), 0);
        assert_eq!(first_b.matches("blocking_send(").count(), 0);
        assert_eq!(promote.matches("blocking_send(").count(), 0);
        assert_eq!(emit.matches("stream_send_bounded(tx, ").count(), 2);
        assert_eq!(emit.matches("publish_stream_event(a, ").count(), 4);
        assert_eq!(emit.matches("try_send(").count(), 1);
        assert_eq!(serial.matches("stream_send_bounded(tx, event").count(), 1);
        assert_eq!(lifecycle.matches("stream_send_bounded(tx, ").count(), 2);
        assert!(lifecycle.contains("stream_send_bounded(\n                tx,"));
    }
}

/// A suppressed EOS cannot outlive an exhausted content budget. This is the
/// speculative emission entry used by MTP/DFlash verify rows.
#[test]
fn speculative_suppressed_eos_retires_at_zero_content_budget() {
    let mut a = active_seq(ResponseSink::Blocking(None), 0);
    a.remaining = 0;
    a.require_tool_call = true;
    let eos = a.eos_tokens[0];

    emit_token(&mut a, eos, None);

    assert!(a.finished);
    assert!(
        !a.output_tokens.contains(&eos),
        "suppressed EOS stays hidden"
    );
    assert_eq!(a.remaining, 0);
}

/// The serial logits path applies the same terminal budget rule as speculative
/// emission, rather than sampling and discarding a suppressed EOS forever.
#[test]
fn serial_suppressed_eos_retires_at_zero_content_budget() {
    let m = crate::scheduler::test_fixtures::model::recording();
    let mut logits = vec![-10.0; 32];
    logits[17] = 10.0;
    *m.host_logits.lock().unwrap() = logits;
    let mut a = active_seq(ResponseSink::Blocking(None), 0);
    a.remaining = 0;
    a.require_tool_call = true;
    let mut active = vec![a];

    process_decode_logits(
        &m,
        &mut active,
        DevicePtr::NULL,
        Instant::now(),
        None,
        None,
        None,
        None,
        None,
        &[],
        false,
        false,
    );

    assert!(active[0].finished);
    assert!(!active[0].output_tokens.contains(&17));
    assert_eq!(active[0].remaining, 0);
}

// ── request timeout: stall + total ceiling (stability action #4) ─────
mod request_timeout {
    use super::*;
    use std::time::{Duration, Instant};

    const LIMIT: Option<Duration> = Some(Duration::from_secs(120));

    fn live_seq() -> ActiveSeq {
        active_seq(ResponseSink::Blocking(None), 0)
    }

    #[test]
    fn first_check_arms_the_stall_clock_without_counting_prefill() {
        let mut a = live_seq();
        let t0 = Instant::now();
        // a first token that arrived long after request start (slow prefill)
        check_request_timeout_at(&mut a, t0 + Duration::from_secs(600), LIMIT);
        assert!(!a.finished, "prefill time is not a stall");
        assert_eq!(a.stall_mark.tokens, a.output_tokens.len());
    }

    /// C2: the pre-first-token deadline (base 300 s) bounds prefill; the
    /// derived decode deadline only takes over at the first token.
    #[test]
    fn the_decode_deadline_replaces_the_base_only_at_the_first_token() {
        let mut a = live_seq();
        let t0 = Instant::now();
        a.timeout_at = Some(t0 + Duration::from_secs(300));
        a.decode_timeout_at = Some(t0 + Duration::from_secs(2088));
        assert_eq!(
            a.timeout_at,
            Some(t0 + Duration::from_secs(300)),
            "prefill sees the base"
        );
        check_request_timeout_at(&mut a, t0 + Duration::from_secs(10), None);
        assert_eq!(a.timeout_at, Some(t0 + Duration::from_secs(2088)));
        assert_eq!(a.decode_timeout_at, None, "switch happens once");
        a.output_tokens.push(5);
        check_request_timeout_at(&mut a, t0 + Duration::from_secs(1000), None);
        assert!(!a.finished, "past the old 300 s base, still decoding");
        a.output_tokens.push(5);
        check_request_timeout_at(&mut a, t0 + Duration::from_secs(2088), None);
        assert!(a.finished);
        assert_eq!(
            a.stop_cause,
            Some(crate::api::request_contract::TIMEOUT_TOTAL_CAUSE)
        );
    }

    #[test]
    fn without_a_decode_deadline_the_base_stays() {
        let mut a = live_seq();
        let t0 = Instant::now();
        a.timeout_at = Some(t0 + Duration::from_secs(300));
        check_request_timeout_at(&mut a, t0 + Duration::from_secs(1), None);
        assert_eq!(a.timeout_at, Some(t0 + Duration::from_secs(300)));
    }

    #[test]
    fn a_request_producing_tokens_is_never_cut_by_the_stall_timer() {
        let mut a = live_seq();
        let t0 = Instant::now();
        check_request_timeout_at(&mut a, t0, LIMIT);
        for i in 1..=36u64 {
            a.output_tokens.push(5);
            check_request_timeout_at(&mut a, t0 + Duration::from_secs(100 * i), LIMIT);
            assert!(!a.finished, "cut at token {i}");
        }
    }

    #[test]
    fn no_new_token_for_the_stall_limit_ends_the_request() {
        let mut a = live_seq();
        let t0 = Instant::now();
        check_request_timeout_at(&mut a, t0, LIMIT);
        check_request_timeout_at(&mut a, t0 + Duration::from_secs(119), LIMIT);
        assert!(!a.finished);
        check_request_timeout_at(&mut a, t0 + Duration::from_secs(120), LIMIT);
        assert!(a.finished, "120 s without a token is a stall");
        assert_eq!(
            a.stop_cause,
            Some(crate::api::request_contract::TIMEOUT_STALL_CAUSE)
        );
    }

    /// C10: progress is tested before the gap. A scheduler held up past the
    /// limit that resumes with a fresh token does not kill the request.
    #[test]
    fn a_token_after_a_scheduler_hiccup_rearms_instead_of_killing() {
        let mut a = live_seq();
        let t0 = Instant::now();
        check_request_timeout_at(&mut a, t0, LIMIT);
        a.output_tokens.push(5);
        check_request_timeout_at(&mut a, t0 + Duration::from_secs(130), LIMIT);
        assert!(!a.finished);
        assert_eq!(a.stall_mark.at, Some(t0 + Duration::from_secs(130)));
    }

    /// C6: a rollback that trims output and re-grows it to the same length
    /// is not progress; only a new high-water mark of committed tokens is.
    #[test]
    fn rollback_regrowth_is_not_progress() {
        let mut a = live_seq();
        let t0 = Instant::now();
        a.output_tokens.extend([5, 6, 7]);
        check_request_timeout_at(&mut a, t0, LIMIT);
        let high = a.output_tokens.len();
        a.output_tokens.truncate(high - 2);
        check_request_timeout_at(&mut a, t0 + Duration::from_secs(60), LIMIT);
        a.output_tokens.extend([6, 7]);
        check_request_timeout_at(&mut a, t0 + Duration::from_secs(120), LIMIT);
        assert!(a.finished, "same-length regrowth did not re-arm the clock");
        assert_eq!(a.stall_mark.tokens, high);
    }

    #[test]
    fn stall_disabled_leaves_only_the_total_deadline() {
        let mut a = live_seq();
        let t0 = Instant::now();
        check_request_timeout_at(&mut a, t0, None);
        check_request_timeout_at(&mut a, t0 + Duration::from_secs(10_000), None);
        assert!(!a.finished);
        a.timeout_at = Some(t0 + Duration::from_secs(10));
        check_request_timeout_at(&mut a, t0 + Duration::from_secs(10), None);
        assert!(a.finished, "total ceiling still applies");
    }

    #[test]
    fn timeout_reason_names_the_cause() {
        assert_eq!(TimeoutReason::Stall.as_str(), "stall");
        assert_eq!(TimeoutReason::TotalDeadline.as_str(), "total_deadline");
    }
}

#[cfg(test)]
mod defer_emit_integration_tests {
    use super::*;
    use crate::scheduler::test_fixtures::active_seq;

    #[test]
    fn deferred_emit_buffers_during_accept_and_flushes_coalesced_token_batch() {
        let (tx, mut rx) = tokio::sync::mpsc::channel(16);
        let mut a = active_seq(ResponseSink::Streaming(tx), 0);
        a.max_output_tokens = 32;
        a.remaining = 32;
        a.defer_emit.begin();

        for tok in [10u32, 11, 12, 13, 14] {
            assert!(emit_token_for_verify(&mut a, tok, None));
        }
        assert!(!a.finished);
        // Zero channel wakeups occurred while deferral was armed (prior to commit+propose).
        assert!(rx.try_recv().is_err());

        flush_deferred_emit(&mut a);
        assert!(!a.defer_emit.is_active());
        assert!(matches!(
            rx.try_recv(),
            Ok(StreamEvent::TokenBatch(batch)) if batch == vec![10, 11, 12, 13, 14]
        ));
        assert!(rx.try_recv().is_err());
    }

    #[test]
    fn deferred_emit_retires_immediately_when_receiver_closes() {
        let (tx, rx) = tokio::sync::mpsc::channel(16);
        drop(rx);
        let mut a = active_seq(ResponseSink::Streaming(tx), 0);
        a.defer_emit.begin();
        assert!(!publish_stream_event(&mut a, StreamEvent::Token(7), "test"));
        assert!(a.finished);
    }

    #[test]
    fn tracked_fuzzy_repetition_outside_tool_matches_full_rposition_scan() {
        let mut a = active_seq(ResponseSink::Blocking(None), 0);
        a.tool_call_start_token = Some(900);
        a.tool_call_end_token = Some(901);

        // Repeat a 20-token block 5 times (100 >= 90 tokens) inside a tool call -> suppressed by both.
        a.output_tokens.push(900);
        for _ in 0..5 {
            a.output_tokens.extend(10..30);
        }
        assert_eq!(
            fuzzy_repetition_outside_tool_tracked(&mut a),
            fuzzy_repetition_outside_tool(&a)
        );
        assert!(fuzzy_repetition_outside_tool_tracked(&mut a).is_none());

        // Close the tool call and repeat a 20-token block 5 times outside -> detected identically by both.
        a.output_tokens.push(901);
        for _ in 0..5 {
            a.output_tokens.extend(40..60);
        }
        let tracked = fuzzy_repetition_outside_tool_tracked(&mut a);
        let reference = fuzzy_repetition_outside_tool(&a);
        assert_eq!(tracked, reference);
        assert!(tracked.is_some());
    }
}
