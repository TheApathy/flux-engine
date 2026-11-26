// SPDX-License-Identifier: AGPL-3.0-only

//! Actual verifier dispatch, terminal-sink, cache, and retirement fault tests.

use super::*;
use crate::scheduler::test_fixtures::{active_seq, model};
use std::sync::atomic::{AtomicBool, Ordering};

static VERIFY_LANES_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn lock_verify_lanes() -> std::sync::MutexGuard<'static, ()> {
    VERIFY_LANES_TEST_LOCK
        .lock()
        .unwrap_or_else(|e| e.into_inner())
}

enum Receiver {
    Blocking(tokio::sync::oneshot::Receiver<Result<InferenceResponse>>),
    Streaming(tokio::sync::mpsc::Receiver<StreamEvent>),
}

fn sink(streaming: bool) -> (ResponseSink, Receiver) {
    if streaming {
        let (tx, rx) = tokio::sync::mpsc::channel(16);
        (ResponseSink::Streaming(tx), Receiver::Streaming(rx))
    } else {
        let (tx, rx) = tokio::sync::oneshot::channel();
        (ResponseSink::Blocking(Some(tx)), Receiver::Blocking(rx))
    }
}

fn context(thinking: bool) -> ThinkSpecCtx<'static> {
    ThinkSpecCtx {
        enabled: thinking,
        code_fence_token: None,
        reflection_suppress_ids: &[],
        adaptive_sampling: false,
        honor_eos_inside_thinking: Some(false),
    }
}

fn request(sink: ResponseSink, width: usize, slot: usize) -> ActiveSeq {
    let mut a = active_seq(sink, slot);
    a.think_ended = false; // neutral raw argmax selects the fixed-width paths
    a.seq.tokens = vec![7; a.seq.seq_len];
    a.pending_drafts = (41..41 + width as u32).collect();
    a
}

fn configure(model: &model::TestModel, stage: &'static str, targets: Vec<u32>) {
    *model.verify_failure.lock().unwrap() = Some(stage);
    *model.verify_targets.lock().unwrap() = targets;
}

fn targets(width: usize, accepted: usize) -> Vec<u32> {
    let mut result = (41..41 + accepted as u32).collect::<Vec<_>>();
    result.resize(width + 1, 77);
    result
}

fn terminal_error(rx: &mut Receiver, expected: &str) {
    let message = match rx {
        Receiver::Blocking(rx) => match rx.try_recv().expect("immediate terminal response") {
            Ok(_) => panic!("an engine failure cannot produce a successful response"),
            Err(error) => error.to_string(),
        },
        Receiver::Streaming(rx) => loop {
            match rx.try_recv().expect("immediate streaming terminal") {
                StreamEvent::Error(message) => break message,
                StreamEvent::Done { .. } => panic!("an engine failure cannot produce Done"),
                _ => {} // already-emitted tokens may precede a commit failure
            }
        },
    };
    assert!(message.starts_with("MTP "), "{message}");
    assert!(message.contains(expected), "{message}");
}

fn no_later_terminal(rx: &mut Receiver) {
    match rx {
        Receiver::Blocking(rx) => assert!(rx.try_recv().is_err()),
        Receiver::Streaming(rx) => assert!(matches!(
            rx.try_recv(),
            Err(tokio::sync::mpsc::error::TryRecvError::Disconnected)
        )),
    }
}

fn error_then_retire(
    model: &model::TestModel,
    active: &mut Vec<ActiveSeq>,
    cache: &mut Vec<bool>,
    receivers: &mut [Receiver],
    expected: &str,
) {
    assert!(active.iter().all(|a| a.finished));
    assert!(cache.iter().all(|cacheable| !cacheable));
    assert!(
        active
            .iter()
            .all(|a| matches!(&a.sink, ResponseSink::Blocking(None)))
    );
    assert!(active.iter().all(|a| a.pending_drafts.is_empty()
        && a.pending_tree_payload.is_none()
        && a.pending_dflash_sampled_proposal.is_none()));
    assert!(model.frees.lock().unwrap().is_empty(), "no early free");
    let request_count = active.len();
    for rx in receivers.iter_mut() {
        terminal_error(rx, expected);
    }
    retire_finished_sequences(model, active, cache);
    retire_finished_sequences(model, active, cache);
    assert!(active.is_empty() && cache.is_empty());
    // Retirement may compact a survivor into the first vacated slot, so slot
    // numbers can legitimately repeat even though each request is freed once.
    assert_eq!(model.frees.lock().unwrap().len(), request_count);
    assert_eq!(model.caches.load(Ordering::SeqCst), 0);
    for rx in receivers {
        no_later_terminal(rx);
    }
}

#[test]
fn fixed_width_engine_failures_are_errors_for_both_sinks_and_every_commit_outcome() {
    for streaming in [false, true] {
        for width in 1..=3 {
            for stage in [
                "sync_secondary",
                "EP command",
                "EP token",
                "decode_verify",
                "EP result",
                "commit",
            ] {
                for accepted in 0..=width {
                    let model = model::recording();
                    configure(&model, stage, targets(width, accepted));
                    let (sink, rx) = sink(streaming);
                    let mut active = vec![request(sink, width, 3)];
                    let mut cache = vec![true];

                    step_mtp(&model, &mut active, width, &context(false), &mut cache);

                    assert_eq!(model.proposals.load(Ordering::SeqCst), 0);
                    assert!(model.verify_calls.lock().unwrap().contains(&stage));
                    error_then_retire(&model, &mut active, &mut cache, &mut [rx], stage);
                }
            }
        }
    }
}

#[test]
fn native_width_verification_and_invalid_rows_fail_without_success_or_cache() {
    for streaming in [false, true] {
        for stage in ["sync_secondary", "decode_verify", "commit", "wrong rows"] {
            for accepted in 0..=4 {
                let model = model::recording();
                let reply = if stage == "wrong rows" {
                    vec![41]
                } else {
                    targets(4, accepted)
                };
                configure(&model, stage, reply);
                let (sink, rx) = sink(streaming);
                let mut active = vec![request(sink, 4, 2)];
                let mut cache = vec![true];

                step_mtp(&model, &mut active, 4, &context(false), &mut cache);

                assert_eq!(model.proposals.load(Ordering::SeqCst), 0);
                let expected = if stage == "wrong rows" {
                    "target row count"
                } else {
                    stage
                };
                error_then_retire(&model, &mut active, &mut cache, &mut [rx], expected);
            }
        }
    }
}

#[test]
fn generic_ep_frame_and_result_failures_send_errors() {
    for streaming in [false, true] {
        for stage in ["EP command", "EP result"] {
            let model = model::recording();
            configure(&model, stage, targets(4, 4));
            model.verify_is_ep.store(true, Ordering::SeqCst);
            let (sink, rx) = sink(streaming);
            let mut active = vec![request(sink, 4, 2)];
            let mut cache = vec![true];

            step_mtp(&model, &mut active, 4, &context(false), &mut cache);

            let expected = if stage == "EP command" {
                "EP frame broadcast"
            } else {
                "EP result broadcast"
            };
            error_then_retire(&model, &mut active, &mut cache, &mut [rx], expected);
        }
    }
}

#[test]
fn generic_policy_download_errors_are_not_confused_with_policy_stops() {
    for streaming in [false, true] {
        for thinking in [false, true] {
            let model = model::recording();
            configure(&model, "logits_download", targets(4, 4));
            *model.host_logits.lock().unwrap() = vec![0.0; 128];
            let (sink, rx) = sink(streaming);
            let mut a = request(sink, 4, 1);
            a.inside_thinking = thinking;
            a.think_ended = !thinking;
            a.presence_penalty = 1.0; // force the distribution-dependent D2H path
            let mut active = vec![a];
            let mut cache = vec![true];

            step_mtp(&model, &mut active, 4, &context(thinking), &mut cache);

            assert!(
                model
                    .verify_calls
                    .lock()
                    .unwrap()
                    .contains(&"logits_download")
            );
            error_then_retire(
                &model,
                &mut active,
                &mut cache,
                &mut [rx],
                "verify logits download",
            );
        }
    }
}

#[test]
fn generic_terminal_commit_error_overrides_only_the_failed_commit_not_the_eos() {
    for streaming in [false, true] {
        for terminal_bonus in [false, true] {
            let model = model::recording();
            let mut reply = targets(4, 4);
            if terminal_bonus {
                reply[4] = 17;
            }
            configure(&model, "commit", reply);
            let (sink, rx) = sink(streaming);
            let mut a = request(sink, 4, 1);
            if !terminal_bonus {
                a.eos_tokens = vec![41];
            }
            let mut active = vec![a];
            let mut cache = vec![true];

            step_mtp(&model, &mut active, 4, &context(false), &mut cache);

            assert!(model.verify_calls.lock().unwrap().contains(&"commit"));
            error_then_retire(&model, &mut active, &mut cache, &mut [rx], "verify commit");
        }
    }
}

#[test]
fn csk_actual_verifiers_share_the_production_cache_suppression_and_single_retirement() {
    for streaming in [false, true] {
        for width in 1..=2 {
            for stage in ["sync_secondary", "decode_verify", "commit"] {
                for accepted in 0..=width {
                    let model = model::recording();
                    configure(&model, stage, targets(width, accepted));
                    let (sink0, rx0) = sink(streaming);
                    let (sink1, rx1) = sink(streaming);
                    let mut active = vec![request(sink0, width, 0), request(sink1, width, 1)];
                    let mut cache = vec![true, true];
                    let indices = [0, 1];

                    if width == 1 {
                        step_verify_k2_batched_csk(&model, &mut active, &indices, width);
                    } else {
                        step_verify_k3_batched_csk(&model, &mut active, &indices, width);
                    }
                    // This is the exact post-dispatch call used by both opt-in
                    // CSK branches; avoid changing process-global env gates.
                    suppress_finished_verify_cache(&active, &indices, &mut cache);

                    assert_eq!(model.proposals.load(Ordering::SeqCst), 0);
                    error_then_retire(&model, &mut active, &mut cache, &mut [rx0, rx1], stage);
                }
            }
        }
    }
}

#[test]
fn batched_actual_verifier_errors_reject_both_requests_and_malformed_results() {
    for streaming in [false, true] {
        for stage in [
            "sync_secondary",
            "decode_verify",
            "commit",
            "wrong rows",
            "outer rows",
        ] {
            for accepted in 0..=4 {
                let model = model::recording();
                let reply = if stage == "wrong rows" {
                    vec![41]
                } else {
                    targets(4, accepted)
                };
                configure(&model, stage, reply);
                let (sink0, rx0) = sink(streaming);
                let (sink1, rx1) = sink(streaming);
                let mut active = vec![request(sink0, 4, 0), request(sink1, 4, 1)];
                let mut cache = vec![true, true];
                let indices = [0, 1];

                step_verify_dflash_batched(&model, &mut active, &indices, 4);
                suppress_finished_verify_cache(&active, &indices, &mut cache);

                assert_eq!(model.proposals.load(Ordering::SeqCst), 0);
                let expected = if stage == "wrong rows" || stage == "outer rows" {
                    "target row count"
                } else {
                    stage
                };
                error_then_retire(&model, &mut active, &mut cache, &mut [rx0, rx1], expected);
            }
        }
    }
}

#[test]
fn batched_terminal_eos_commit_failures_are_errors_for_both_requests() {
    for streaming in [false, true] {
        for terminal_bonus in [false, true] {
            let model = model::recording();
            let mut reply = targets(4, 4);
            if terminal_bonus {
                reply[4] = 17;
            }
            configure(&model, "commit", reply);
            let (sink0, rx0) = sink(streaming);
            let (sink1, rx1) = sink(streaming);
            let mut active = vec![request(sink0, 4, 0), request(sink1, 4, 1)];
            if !terminal_bonus {
                for a in &mut active {
                    a.eos_tokens = vec![41];
                }
            }
            let mut cache = vec![true, true];
            let indices = [0, 1];

            step_verify_dflash_batched(&model, &mut active, &indices, 4);
            suppress_finished_verify_cache(&active, &indices, &mut cache);

            error_then_retire(
                &model,
                &mut active,
                &mut cache,
                &mut [rx0, rx1],
                "terminal batched",
            );
        }
    }
}

#[test]
fn optional_proposer_failure_keeps_the_committed_target_on_plain_bootstrap() {
    for streaming in [false, true] {
        for width in [4] {
            let model =
                model::TestModel::new(Err(anyhow::anyhow!("optional proposer unavailable")), None);
            configure(&model, "success", targets(width, width));
            let (sink, mut rx) = sink(streaming);
            let mut active = vec![request(sink, width, 0)];
            let mut cache = vec![true];

            step_mtp(&model, &mut active, width, &context(false), &mut cache);

            assert!(!active[0].finished);
            assert!(active[0].pending_drafts.is_empty());
            assert!(active[0].pending_tree_payload.is_none());
            assert_eq!(cache, [true]);
            assert_eq!(model.proposals.load(Ordering::SeqCst), 1);
            *model.bootstrap_failure.lock().unwrap() = Some("success");
            step_mtp(&model, &mut active, width, &context(false), &mut cache);
            assert!(active[0].finished);
            retire_finished_sequences(&model, &mut active, &mut cache);
            match &mut rx {
                Receiver::Blocking(rx) => assert!(rx.try_recv().unwrap().is_ok()),
                Receiver::Streaming(rx) => {
                    let mut done = 0;
                    while let Ok(event) = rx.try_recv() {
                        match event {
                            StreamEvent::Error(message) => panic!("unexpected error: {message}"),
                            StreamEvent::Done { .. } => done += 1,
                            _ => {}
                        }
                    }
                    assert_eq!(done, 1);
                }
            }
            assert_eq!(*model.frees.lock().unwrap(), [0]);
            assert_eq!(model.caches.load(Ordering::SeqCst), 1);
        }
    }
}

#[test]
fn bootstrap_checkpoint_start_failure_consumes_the_installed_proposal_and_sink() {
    for streaming in [false, true] {
        let model = model::TestModel::new(Ok(vec![41]), Some(model::tree(41)));
        *model.bootstrap_failure.lock().unwrap() = Some("start_checkpoint_async");
        let (sink, rx) = sink(streaming);
        let mut a = request(sink, 0, 4);
        a.eos_tokens = vec![99];
        let mut active = vec![a];
        let mut cache = vec![true];

        step_mtp(&model, &mut active, 4, &context(false), &mut cache);

        assert_eq!(model.proposals.load(Ordering::SeqCst), 1);
        error_then_retire(
            &model,
            &mut active,
            &mut cache,
            &mut [rx],
            "start_checkpoint_async",
        );
    }
}

#[test]
fn successful_verifier_stops_and_cancellation_keep_successful_terminal_semantics() {
    for streaming in [false, true] {
        for width in 1..=4 {
            for stop in ["length", "EOS", "cancel"] {
                let model = model::recording();
                configure(&model, "success", targets(width, width));
                let (sink, mut rx) = sink(streaming);
                let mut a = request(sink, width, 0);
                match stop {
                    "length" => a.remaining = 1,
                    "EOS" => a.eos_tokens = vec![41],
                    "cancel" => a.cancel_flag = Some(Arc::new(AtomicBool::new(true))),
                    _ => unreachable!(),
                }
                let mut active = vec![a];
                let mut cache = vec![true];

                step_mtp(&model, &mut active, width, &context(false), &mut cache);
                assert!(active[0].finished);
                assert!(matches!(
                    &active[0].sink,
                    ResponseSink::Streaming(_) | ResponseSink::Blocking(Some(_))
                ));
                retire_finished_sequences(&model, &mut active, &mut cache);
                match &mut rx {
                    Receiver::Blocking(rx) => assert!(rx.try_recv().unwrap().is_ok()),
                    Receiver::Streaming(rx) => {
                        let mut done = 0;
                        while let Ok(event) = rx.try_recv() {
                            match event {
                                StreamEvent::Error(message) => {
                                    panic!("unexpected error: {message}")
                                }
                                StreamEvent::Done { .. } => done += 1,
                                _ => {}
                            }
                        }
                        assert_eq!(done, 1);
                    }
                }
                assert_eq!(*model.frees.lock().unwrap(), [0]);
            }
        }
    }
}

/// Cross-sequence batched native MTP (`ATLAS_QWEN4_BATCHED_K5`): an injected
/// failure at every mandatory stage of a real two-sequence frame reaches both
/// clients as an error, proposes nothing, caches nothing and retires cleanly.
#[test]
fn batched_native_mtp_engine_failures_are_errors_for_both_sinks_and_every_commit_outcome() {
    for streaming in [false, true] {
        for stage in [
            "sync_secondary",
            "batch begin",
            "batched verify",
            "batched rows",
            "batched short row",
            "commit",
            "save hidden",
            "trim",
            "batch end",
            "batch finish",
        ] {
            for accepted in [0, 2, 4] {
                let model = model::recording();
                model.batched_k5.store(true, Ordering::SeqCst);
                configure(&model, stage, targets(4, accepted));
                let (sink_a, rx_a) = sink(streaming);
                let (sink_b, rx_b) = sink(streaming);
                let mut active = vec![request(sink_a, 4, 0), request(sink_b, 4, 1)];
                let mut cache = vec![true, true];

                step_mtp(&model, &mut active, 4, &context(false), &mut cache);

                let calls = model.verify_calls.lock().unwrap().clone();
                assert!(
                    calls.contains(&"batched preflight"),
                    "{stage}: the frame must go through the batched verifier, saw {calls:?}"
                );
                assert!(
                    !calls.contains(&"decode_verify"),
                    "{stage}: no per-sequence verify"
                );
                if stage != "sync_secondary" && stage != "batch begin" {
                    assert_eq!(model.batched_frames.load(Ordering::SeqCst), 1, "{stage}");
                }
                assert_eq!(model.proposals.load(Ordering::SeqCst), 0, "{stage}");
                let expected = if stage == "batched rows" || stage == "batched short row" {
                    "row count"
                } else {
                    stage
                };
                error_then_retire(&model, &mut active, &mut cache, &mut [rx_a, rx_b], expected);
            }
        }
    }
}

/// A frame the model refuses at read-only admission (for example a poisoned
/// multi-sequence pointer table) is verified one sequence at a time; a later
/// failure there still reaches every client as an error.
#[test]
fn refused_batched_frames_fall_back_to_per_sequence_verify() {
    let model = model::recording();
    model.batched_k5.store(true, Ordering::SeqCst);
    model.batched_refused.store(true, Ordering::SeqCst);
    configure(&model, "commit", targets(4, 2));
    let (sink_a, rx_a) = sink(false);
    let (sink_b, rx_b) = sink(true);
    let mut active = vec![request(sink_a, 4, 0), request(sink_b, 4, 1)];
    let mut cache = vec![true, true];

    step_mtp(&model, &mut active, 4, &context(false), &mut cache);

    let calls = model.verify_calls.lock().unwrap().clone();
    assert!(calls.contains(&"batched preflight"), "{calls:?}");
    assert_eq!(
        model.batched_frames.load(Ordering::SeqCst),
        0,
        "no batched forward"
    );
    assert_eq!(
        calls
            .iter()
            .filter(|call| **call == "decode_verify")
            .count(),
        2,
        "each sequence was verified on its own: {calls:?}"
    );
    error_then_retire(&model, &mut active, &mut cache, &mut [rx_a, rx_b], "commit");
}

/// A missing verify lane fails the entire owned cohort. Already published
/// state keeps its committed extent; the uncommitted lane is restored.
#[test]
fn a_sequence_without_a_verify_lane_fails_the_cohort_without_rewinding_published_state() {
    let _lane_lock = lock_verify_lanes();
    let _lanes = spark_model::layers::qwen4_verify_lanes::configure_lanes(2).unwrap();
    let model = model::recording();
    model.batched_k5.store(true, Ordering::SeqCst);
    configure(&model, "none", targets(4, 1));
    let (sink_a, mut rx_a) = sink(true);
    let (sink_b, mut rx_b) = sink(false);
    let mut active = vec![request(sink_a, 4, 0), request(sink_b, 4, 5)];
    let pre = (active[1].seq.seq_len, active[1].seq.tokens.len());
    let mut cache = vec![true, true];

    step_mtp(&model, &mut active, 4, &context(false), &mut cache);

    assert_eq!(model.batched_frames.load(Ordering::SeqCst), 1);
    let unlaned = &active[1];
    assert!(unlaned.finished);
    assert_eq!(
        (unlaned.seq.seq_len, unlaned.seq.tokens.len()),
        pre,
        "the verify advance is undone"
    );
    terminal_error(&mut rx_b, "batched verify lane");
    terminal_error(&mut rx_a, "batched verify lane");
    assert!(active[0].finished);
    assert_eq!(
        active[0].seq.seq_len,
        pre.0 + 2,
        "published state is never rewound"
    );
    assert_eq!(cache, [false, false]);
    assert!(
        model.verify_calls.lock().unwrap().contains(&"commit"),
        "the laned sequence still commits"
    );
    assert!(
        model.proposals.load(Ordering::SeqCst) == 0,
        "no proposer may run after a cohort failure"
    );
}

#[test]
fn native_c26_owns_both_c13_frames_until_every_hidden_handoff_is_published() {
    let _lane_lock = lock_verify_lanes();
    let model = model::recording();
    model.batched_k5.store(true, Ordering::SeqCst);
    configure(&model, "success", targets(4, 4));
    let mut active = Vec::new();
    let mut receivers = Vec::new();
    for slot in 0..26 {
        let (sink, rx) = sink(false);
        active.push(request(sink, 4, slot));
        receivers.push(rx);
    }
    let mut cache = vec![true; 26];

    step_mtp(&model, &mut active, 4, &context(false), &mut cache);

    assert_eq!(
        *model.batched_frame_shapes.lock().unwrap(),
        [(13, 65, 5), (13, 65, 5)]
    );
    let calls = model.verify_calls.lock().unwrap();
    assert_eq!(
        calls.iter().filter(|&&call| call == "batch begin").count(),
        1
    );
    assert_eq!(calls.iter().filter(|&&call| call == "batch end").count(), 2);
    assert_eq!(
        calls.iter().filter(|&&call| call == "batch finish").count(),
        1
    );
    assert!(
        calls
            .iter()
            .rposition(|&call| call == "batched preflight")
            .unwrap()
            < calls
                .iter()
                .position(|&call| call == "batch begin")
                .unwrap()
    );
    assert!(
        calls
            .iter()
            .position(|&call| call == "batch finish")
            .unwrap()
            < calls.iter().position(|&call| call == "propose").unwrap()
    );
    assert_eq!(model.proposals.load(Ordering::SeqCst), 26);
    assert!(!model.batched_owner_active.load(Ordering::SeqCst));
    assert!(
        model
            .batched_lanes
            .lock()
            .unwrap()
            .values()
            .all(|(_, phase)| *phase == model::TestBatchPhase::Published)
    );
    assert!(active.iter().all(|a| !a.finished && a.seq.seq_len == 14));
    assert_eq!(cache, [true; 26]);
    assert!(model.batched_abort_attempts.lock().unwrap().is_empty());
}

#[test]
fn later_frame_preflight_refusal_preserves_every_proposal_before_cohort_begin() {
    let _lane_lock = lock_verify_lanes();
    let model = model::recording();
    model.batched_k5.store(true, Ordering::SeqCst);
    model.batched_refuse_preflight.store(1, Ordering::SeqCst);
    configure(&model, "success", targets(4, 4));
    let mut active: Vec<_> = (0..8).map(|slot| request(sink(false).0, 4, slot)).collect();
    let before: Vec<_> = active
        .iter()
        .map(|a| {
            (
                a.seq.seq_len,
                a.seq.tokens.clone(),
                a.pending_drafts.clone(),
            )
        })
        .collect();
    let frames = vec![vec![0, 1, 2, 3], vec![4, 5, 6, 7]];

    assert!(!verify_qwen4_batched_step::step_verify_qwen4_batched(
        &model,
        &mut active,
        &frames,
        4,
        &context(false)
    ));

    assert_eq!(model.batched_preflights.load(Ordering::SeqCst), 2);
    assert_eq!(model.batched_frames.load(Ordering::SeqCst), 0);
    assert!(!model.verify_calls.lock().unwrap().contains(&"batch begin"));
    assert_eq!(
        before,
        active
            .iter()
            .map(|a| (
                a.seq.seq_len,
                a.seq.tokens.clone(),
                a.pending_drafts.clone()
            ))
            .collect::<Vec<_>>()
    );
}

#[test]
fn second_frame_failure_aborts_every_lane_without_rewinding_first_frame_publications() {
    let _lane_lock = lock_verify_lanes();
    for streaming in [false, true] {
        let model = model::recording();
        model.batched_k5.store(true, Ordering::SeqCst);
        model.batched_fail_frame.store(1, Ordering::SeqCst);
        configure(&model, "success", targets(4, 4));
        let mut active = Vec::new();
        let mut receivers = Vec::new();
        for slot in 0..26 {
            let (sink, rx) = sink(streaming);
            active.push(request(sink, 4, slot));
            receivers.push(rx);
        }
        let mut cache = vec![true; 26];

        step_mtp(&model, &mut active, 4, &context(false), &mut cache);

        assert_eq!(model.batched_frames.load(Ordering::SeqCst), 2);
        assert_eq!(model.proposals.load(Ordering::SeqCst), 0);
        assert_eq!(
            *model.batched_abort_attempts.lock().unwrap(),
            (0..26).collect::<Vec<_>>()
        );
        assert!(
            active[..13]
                .iter()
                .all(|a| a.seq.seq_len == 14 && a.seq.tokens.len() == 14)
        );
        assert!(
            active[13..]
                .iter()
                .all(|a| a.seq.seq_len == 9 && a.seq.tokens.len() == 9)
        );
        assert!(
            !model
                .verify_calls
                .lock()
                .unwrap()
                .contains(&"decode_verify")
        );
        error_then_retire(
            &model,
            &mut active,
            &mut cache,
            &mut receivers,
            "later frame failure",
        );
    }
}

#[test]
fn commit_hidden_and_trim_failures_stop_all_proposals_and_preserve_committed_lanes() {
    let _lane_lock = lock_verify_lanes();
    for stage in ["commit", "save hidden", "trim"] {
        let model = model::recording();
        model.batched_k5.store(true, Ordering::SeqCst);
        model.batched_fail_slot.store(1, Ordering::SeqCst);
        configure(&model, stage, targets(4, 2));
        let mut active = Vec::new();
        let mut receivers = Vec::new();
        for slot in 0..4 {
            let (sink, rx) = sink(false);
            active.push(request(sink, 4, slot));
            receivers.push(rx);
        }
        let mut cache = vec![true; 4];

        step_mtp(&model, &mut active, 4, &context(false), &mut cache);

        assert_eq!(model.proposals.load(Ordering::SeqCst), 0);
        assert_eq!(*model.batched_abort_attempts.lock().unwrap(), [0, 1, 2, 3]);
        assert_eq!(
            active[0].seq.seq_len, 12,
            "first lane was published before failure"
        );
        assert_eq!(
            active[1].seq.seq_len,
            if stage == "commit" { 9 } else { 12 }
        );
        assert!(
            active[2..]
                .iter()
                .all(|a| a.seq.seq_len == 9 && a.seq.tokens.len() == 9)
        );
        error_then_retire(&model, &mut active, &mut cache, &mut receivers, stage);
    }
}

#[test]
fn terminal_commit_counts_consumed_rows_and_retires_without_hidden_publication() {
    let _lane_lock = lock_verify_lanes();
    let model = model::recording();
    model.batched_k5.store(true, Ordering::SeqCst);
    configure(&model, "success", targets(4, 4));
    let mut active = Vec::new();
    let mut receivers = Vec::new();
    for slot in 0..2 {
        let (sink, rx) = sink(false);
        let mut a = request(sink, 4, slot);
        a.output_tokens = vec![31; 7];
        a.remaining = 1;
        active.push(a);
        receivers.push(rx);
    }
    let mut cache = vec![true; 2];

    step_mtp(&model, &mut active, 4, &context(false), &mut cache);

    assert!(
        active
            .iter()
            .all(|a| a.finished && a.seq.seq_len == 10 && a.seq.tokens.len() == 10)
    );
    assert_eq!(
        *model.verify_commits.lock().unwrap(),
        [(10, 1, 5), (10, 1, 5)]
    );
    assert!(
        model
            .batched_lanes
            .lock()
            .unwrap()
            .values()
            .all(|(_, phase)| *phase == model::TestBatchPhase::Retired)
    );
    assert!(!model.verify_calls.lock().unwrap().contains(&"save hidden"));
    assert_eq!(model.proposals.load(Ordering::SeqCst), 0);
    assert!(!model.batched_owner_active.load(Ordering::SeqCst));
}

#[test]
fn zero_consumed_terminal_aborts_its_lane_and_still_settles_the_frame() {
    let _lane_lock = lock_verify_lanes();
    let model = model::recording();
    model.batched_k5.store(true, Ordering::SeqCst);
    configure(&model, "success", targets(4, 4));
    let mut active: Vec<_> = (0..2)
        .map(|slot| {
            let mut a = request(sink(false).0, 4, slot);
            a.cancel_flag = Some(Arc::new(AtomicBool::new(true)));
            a
        })
        .collect();

    assert!(verify_qwen4_batched_step::step_verify_qwen4_batched(
        &model,
        &mut active,
        &[vec![0, 1]],
        4,
        &context(false)
    ));

    assert!(model.verify_commits.lock().unwrap().is_empty());
    assert!(
        active
            .iter()
            .all(|a| a.finished && a.seq.seq_len == 9 && a.seq.tokens.len() == 9)
    );
    assert!(
        model
            .batched_lanes
            .lock()
            .unwrap()
            .values()
            .all(|(_, phase)| *phase == model::TestBatchPhase::Aborted)
    );
    assert!(
        !model
            .verify_calls
            .lock()
            .unwrap()
            .contains(&"batch terminal retire")
    );
    assert!(model.verify_calls.lock().unwrap().contains(&"batch finish"));
    assert!(!model.batched_owner_active.load(Ordering::SeqCst));
}

#[test]
fn terminal_retirement_failure_fails_the_cohort_without_fake_hidden_publication() {
    let _lane_lock = lock_verify_lanes();
    let model = model::recording();
    model.batched_k5.store(true, Ordering::SeqCst);
    configure(&model, "batch terminal retire", targets(4, 4));
    let mut active = Vec::new();
    let mut receivers = Vec::new();
    for slot in 0..2 {
        let (sink, rx) = sink(false);
        let mut a = request(sink, 4, slot);
        a.remaining = 1;
        active.push(a);
        receivers.push(rx);
    }
    let mut cache = vec![true; 2];

    step_mtp(&model, &mut active, 4, &context(false), &mut cache);

    assert_eq!(
        active[0].seq.seq_len, 10,
        "positive commit is never rewound"
    );
    assert_eq!(active[1].seq.seq_len, 9, "uncommitted lane is restored");
    assert!(!model.verify_calls.lock().unwrap().contains(&"save hidden"));
    assert_eq!(*model.batched_abort_attempts.lock().unwrap(), [0, 1]);
    assert_eq!(model.proposals.load(Ordering::SeqCst), 0);
    error_then_retire(
        &model,
        &mut active,
        &mut cache,
        &mut receivers,
        "terminal batched verify retirement",
    );
}

#[test]
fn fatal_cohort_cleanup_attempts_every_lane_and_keeps_the_owner_reserved() {
    let _lane_lock = lock_verify_lanes();
    let model = model::recording();
    model.batched_k5.store(true, Ordering::SeqCst);
    model.verify_abort_error.store(true, Ordering::SeqCst);
    configure(&model, "batched verify", targets(4, 4));
    let mut active = Vec::new();
    let mut receivers = Vec::new();
    for slot in 0..4 {
        let (sink, rx) = sink(false);
        active.push(request(sink, 4, slot));
        receivers.push(rx);
    }
    let mut cache = vec![true; 4];

    step_mtp(&model, &mut active, 4, &context(false), &mut cache);

    assert_eq!(*model.batched_abort_attempts.lock().unwrap(), [0, 1, 2, 3]);
    assert!(model.batched_owner_active.load(Ordering::SeqCst));
    assert!(active.iter().all(|a| a.finished));
    assert_eq!(model.proposals.load(Ordering::SeqCst), 0);
    assert_eq!(cache, [false; 4]);
    error_then_retire(
        &model,
        &mut active,
        &mut cache,
        &mut receivers,
        "batched verify",
    );
}
