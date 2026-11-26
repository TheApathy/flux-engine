// SPDX-License-Identifier: AGPL-3.0-only

//! Exercise the actual MTP step, terminal sinks, and retirement without a GPU.

use super::*;
use crate::scheduler::test_fixtures::{active_seq, model};
use std::sync::atomic::{AtomicBool, Ordering};

enum Receiver {
    Blocking(tokio::sync::oneshot::Receiver<Result<InferenceResponse>>),
    Streaming(tokio::sync::mpsc::Receiver<StreamEvent>),
}

fn sink(streaming: bool) -> (ResponseSink, Receiver) {
    if streaming {
        let (tx, rx) = tokio::sync::mpsc::channel(8);
        (ResponseSink::Streaming(tx), Receiver::Streaming(rx))
    } else {
        let (tx, rx) = tokio::sync::oneshot::channel();
        (ResponseSink::Blocking(Some(tx)), Receiver::Blocking(rx))
    }
}

fn step(model: &dyn Model, active: &mut Vec<ActiveSeq>, cache: &mut [bool], thinking: bool) {
    crate::scheduler::mtp_step::step_mtp(
        model,
        active,
        4,
        &ThinkSpecCtx {
            enabled: thinking,
            code_fence_token: None,
            reflection_suppress_ids: &[],
            adaptive_sampling: false,
            honor_eos_inside_thinking: Some(false),
        },
        cache,
    );
}

fn assert_terminal_error(rx: &mut Receiver, stage: &str) {
    let message = match rx {
        Receiver::Blocking(rx) => match rx
            .try_recv()
            .expect("the failure sends the blocking terminal immediately")
        {
            Ok(_) => panic!("a failed bootstrap cannot be a successful completion"),
            Err(error) => error.to_string(),
        },
        Receiver::Streaming(rx) => match rx.try_recv().expect("stream terminal error") {
            StreamEvent::Error(message) => message,
            _ => panic!("a failed bootstrap must not send Token or Done"),
        },
    };
    assert!(message.starts_with("MTP bootstrap "), "{message}");
    assert!(message.contains(stage), "{message}");
}

fn assert_no_later_terminal(rx: &mut Receiver) {
    match rx {
        Receiver::Blocking(rx) => assert!(rx.try_recv().is_err(), "no second blocking response"),
        Receiver::Streaming(rx) => assert!(
            matches!(
                rx.try_recv(),
                Err(tokio::sync::mpsc::error::TryRecvError::Disconnected)
            ),
            "the original stream sender was consumed, with no duplicate Done"
        ),
    }
}

#[test]
fn injected_bootstrap_failures_send_error_skip_cache_and_retire_exactly_once() {
    for streaming in [false, true] {
        for stage in [
            "sync_secondary",
            "reconcile_context",
            "EP broadcast",
            "decode",
            "sample",
        ] {
            let model = model::recording();
            *model.bootstrap_failure.lock().unwrap() = Some(stage);
            let (sink, mut rx) = sink(streaming);
            let mut a = active_seq(sink, 3);
            a.think_ended = false; // neutral sampler reaches the raw-device branch
            let before_len = a.seq.seq_len;
            let mut active = vec![a];
            let mut cache = vec![true];

            step(&model, &mut active, &mut cache, false);

            assert!(active[0].finished, "{stage}");
            assert_eq!(cache, [false], "{stage}: no failed-state cache admission");
            assert!(matches!(&active[0].sink, ResponseSink::Blocking(None)));
            assert!(active[0].pending_drafts.is_empty());
            assert!(active[0].pending_tree_payload.is_none());
            assert!(
                model.frees.lock().unwrap().is_empty(),
                "cleanup is deferred"
            );
            assert_eq!(model.caches.load(Ordering::SeqCst), 0);
            assert_eq!(model.proposals.load(Ordering::SeqCst), 0);
            assert_eq!(active[0].output_tokens, [31], "no invented output");
            assert!(model.bootstrap_calls.lock().unwrap().contains(&stage));
            if stage == "decode" {
                assert_eq!(
                    active[0].seq.seq_len,
                    before_len + 1,
                    "partial forward fixture"
                );
            }
            assert_terminal_error(&mut rx, stage);

            retire_finished_sequences(&model, &mut active, &mut cache);
            retire_finished_sequences(&model, &mut active, &mut cache);
            assert!(active.is_empty() && cache.is_empty());
            assert_eq!(*model.frees.lock().unwrap(), [3], "{stage}: free once");
            assert_eq!(model.caches.load(Ordering::SeqCst), 0, "{stage}: no cache");
            assert_no_later_terminal(&mut rx);
        }
    }
}

#[test]
fn policy_logits_download_failures_are_errors_for_both_bootstrap_helpers() {
    for streaming in [false, true] {
        for thinking in [false, true] {
            let model = model::recording();
            *model.bootstrap_failure.lock().unwrap() = Some("logits_download");
            let (sink, mut rx) = sink(streaming);
            let mut a = active_seq(sink, 2);
            a.inside_thinking = thinking;
            let mut active = vec![a];
            let mut cache = vec![true];

            step(&model, &mut active, &mut cache, thinking);

            assert_terminal_error(&mut rx, "logits download");
            assert!(active[0].finished);
            assert_eq!(cache, [false]);
            assert!(model.frees.lock().unwrap().is_empty());
            assert_eq!(active[0].output_tokens, [31]);
            retire_finished_sequences(&model, &mut active, &mut cache);
            assert_eq!(*model.frees.lock().unwrap(), [2]);
            assert_eq!(model.caches.load(Ordering::SeqCst), 0);
            assert_no_later_terminal(&mut rx);
        }
    }
}

#[test]
fn a_repeated_failure_notification_cannot_duplicate_the_error_or_free_resources() {
    let model = model::recording();
    let (sink, mut rx) = sink(true);
    let mut a = active_seq(sink, 4);
    a.pending_drafts = vec![41];
    a.pending_tree_payload = Some(model::tree(42));
    let mut cacheable = true;

    fail_bootstrap(&mut a, &mut cacheable, "decode", "injected failure");
    fail_bootstrap(&mut a, &mut cacheable, "sample", "another failure");
    assert!(!cacheable);
    assert!(a.pending_drafts.is_empty() && a.pending_tree_payload.is_none());
    assert_terminal_error(&mut rx, "decode");
    assert_no_later_terminal(&mut rx);
    assert!(model.frees.lock().unwrap().is_empty());

    let mut active = vec![a];
    let mut cache = vec![cacheable];
    retire_finished_sequences(&model, &mut active, &mut cache);
    assert_eq!(*model.frees.lock().unwrap(), [4]);
    assert_eq!(model.caches.load(Ordering::SeqCst), 0);
}

#[test]
fn successful_raw_and_policy_bootstraps_keep_normal_stop_and_length_completions() {
    for streaming in [false, true] {
        for policy in [false, true] {
            for length_stop in [false, true] {
                let model = model::recording();
                *model.bootstrap_failure.lock().unwrap() = Some("success");
                let mut logits = vec![0.0; 32];
                logits[17] = 8.0;
                *model.host_logits.lock().unwrap() = logits;
                let (sink, mut rx) = sink(streaming);
                let mut a = active_seq(sink, 0);
                a.think_ended = policy;
                if length_stop {
                    a.eos_tokens = vec![99];
                    a.remaining = 1;
                }
                let mut active = vec![a];
                let mut cache = vec![true];

                step(&model, &mut active, &mut cache, false);
                assert!(active[0].finished);
                assert_eq!(cache, [true], "normal bootstrap finish stays cacheable");
                retire_finished_sequences(&model, &mut active, &mut cache);
                assert_eq!(model.caches.load(Ordering::SeqCst), 1);
                assert_eq!(*model.frees.lock().unwrap(), [0]);
                let expected = if length_stop { "length" } else { "stop" };
                match &mut rx {
                    Receiver::Blocking(rx) => assert_eq!(
                        rx.try_recv()
                            .unwrap()
                            .expect("successful bootstrap")
                            .finish_reason,
                        expected
                    ),
                    Receiver::Streaming(rx) => {
                        let mut done = 0;
                        while let Ok(event) = rx.try_recv() {
                            match event {
                                StreamEvent::Done { finish_reason, .. } => {
                                    assert_eq!(finish_reason, expected);
                                    done += 1;
                                }
                                StreamEvent::Error(message) => {
                                    panic!("unexpected error: {message}")
                                }
                                _ => {}
                            }
                        }
                        assert_eq!(done, 1);
                    }
                }
            }
        }
    }
}

#[test]
fn a_client_cancel_is_not_reclassified_as_an_engine_failure() {
    let model = model::recording();
    let (sink, mut rx) = sink(true);
    let mut a = active_seq(sink, 1);
    a.cancel_flag = Some(Arc::new(AtomicBool::new(true)));
    let mut active = vec![a];
    let mut cache = vec![true];

    step(&model, &mut active, &mut cache, false);

    assert!(active[0].finished);
    assert!(matches!(&active[0].sink, ResponseSink::Streaming(_)));
    assert!(model.bootstrap_calls.lock().unwrap().is_empty());
    retire_finished_sequences(&model, &mut active, &mut cache);
    if let Receiver::Streaming(rx) = &mut rx {
        while let Ok(event) = rx.try_recv() {
            assert!(!matches!(event, StreamEvent::Error(_)));
        }
    }
    assert_eq!(*model.frees.lock().unwrap(), [1]);
}
