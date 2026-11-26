// SPDX-License-Identifier: AGPL-3.0-only

//! Actual fixed-width scheduler calls against a recording, GPU-free model.

use super::*;
use crate::scheduler::test_fixtures::{active_seq, model};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

fn setup(k: usize, accepted: usize) -> (model::TestModel, ActiveSeq, Vec<u32>) {
    let model = model::recording();
    *model.verify_failure.lock().unwrap() = Some("success");
    let drafts: Vec<_> = (41..41 + k as u32 - 1).collect();
    let mut targets = drafts[..accepted].to_vec();
    targets.resize(k, 77);
    *model.verify_targets.lock().unwrap() = targets;
    let mut a = active_seq(ResponseSink::Blocking(None), 3);
    a.seq.tokens = vec![7; a.seq.seq_len];
    a.pending_drafts = drafts.clone();
    (model, a, drafts)
}

fn step(model: &dyn Model, a: &mut ActiveSeq, drafts: &[u32], k: usize) {
    match k {
        3 => step_verify_k3(model, a, drafts, 2),
        4 => step_verify_k4(model, a, drafts, 3),
        _ => unreachable!(),
    }
}

#[test]
fn shortened_failures_abort_once_after_restoring_host_extent() {
    for k in [3, 4] {
        for accepted in 0..k {
            for stage in ["decode_verify", "EP result", "commit"] {
                for abort_error in [false, true] {
                    let (model, mut a, drafts) = setup(k, accepted);
                    *model.verify_failure.lock().unwrap() = Some(stage);
                    model
                        .verify_abort_error
                        .store(abort_error, Ordering::SeqCst);
                    let before = a.seq.seq_len;
                    let (tx, mut rx) = tokio::sync::oneshot::channel();
                    a.sink = ResponseSink::Blocking(Some(tx));
                    step(&model, &mut a, &drafts, k);
                    assert!(a.finished);
                    assert_eq!((a.seq.seq_len, a.seq.tokens.len()), (before, before));
                    assert_eq!(*model.verify_aborts.lock().unwrap(), [(before, before, k)]);
                    let error = match rx.try_recv().unwrap() {
                        Ok(_) => panic!("failed verification cannot succeed"),
                        Err(error) => error.to_string(),
                    };
                    assert!(error.contains(stage), "{error}");
                    assert_eq!(error.contains("verify rollback failure"), abort_error);
                    assert!(a.pending_drafts.is_empty());
                    assert!(model.frees.lock().unwrap().is_empty());
                    assert_eq!(model.proposals.load(Ordering::SeqCst), 0);
                }
            }
        }
    }
}

#[test]
fn preentry_errors_do_not_abort_an_unstarted_transaction() {
    for k in [3, 4] {
        for stage in ["sync_secondary", "EP command", "EP token"] {
            let (model, mut a, drafts) = setup(k, 0);
            *model.verify_failure.lock().unwrap() = Some(stage);
            step(&model, &mut a, &drafts, k);
            assert!(a.finished);
            assert!(model.verify_aborts.lock().unwrap().is_empty());
            assert!(model.verify_commits.lock().unwrap().is_empty());
        }
    }
}

#[test]
fn every_length_terminal_commits_only_the_classified_input_prefix() {
    for k in [3, 4] {
        for accepted in 0..k {
            for terminal_row in 1..=accepted + 1 {
                let (model, mut a, drafts) = setup(k, accepted);
                let before = a.seq.seq_len;
                a.remaining = terminal_row;
                step(&model, &mut a, &drafts, k);
                assert!(a.finished);
                assert_eq!(a.seq.seq_len, before + terminal_row);
                assert_eq!(a.seq.tokens.len(), before + terminal_row);
                assert_eq!(
                    *model.verify_commits.lock().unwrap(),
                    [(before + terminal_row, terminal_row, k)],
                );
                assert!(model.verify_aborts.lock().unwrap().is_empty());
                assert_eq!(model.proposals.load(Ordering::SeqCst), 0);
            }
        }
    }
}

#[test]
fn eos_at_each_accepted_or_bonus_position_consumes_one_terminal_commit() {
    for k in [3, 4] {
        for accepted in 0..k {
            for terminal_row in 1..=accepted + 1 {
                let (model, mut a, drafts) = setup(k, accepted);
                let before = a.seq.seq_len;
                a.eos_tokens = vec![if terminal_row <= accepted {
                    drafts[terminal_row - 1]
                } else {
                    77
                }];
                step(&model, &mut a, &drafts, k);
                assert!(a.finished);
                assert_eq!(
                    *model.verify_commits.lock().unwrap(),
                    [(before + terminal_row, terminal_row, k)],
                );
                assert!(model.verify_aborts.lock().unwrap().is_empty());
            }
        }
    }
}

#[test]
fn hidden_rows_before_terminal_content_are_retained_in_the_target_prefix() {
    for k in [3, 4] {
        for suppressed_eos in [false, true] {
            let (model, mut a, drafts) = setup(k, k - 1);
            let before = a.seq.seq_len;
            if suppressed_eos {
                a.eos_tokens = vec![drafts[0]];
                a.min_tokens = 100;
                a.remaining = 2; // suppressed EOS also consumes the content budget
            } else {
                a.think_end_token = Some(drafts[0]);
                a.remaining = 1;
            }
            step(&model, &mut a, &drafts, k);
            assert!(a.finished);
            assert_eq!(a.output_tokens.len(), 2); // initial token + ordinary content
            assert_eq!(a.seq.seq_len, before + 2);
            assert_eq!(*model.verify_commits.lock().unwrap(), [(before + 2, 2, k)]);
            assert!(model.verify_aborts.lock().unwrap().is_empty());
        }
    }
}

#[test]
fn cancellation_before_first_row_aborts_once_even_when_abort_fails() {
    for k in [3, 4] {
        for abort_error in [false, true] {
            let (model, mut a, drafts) = setup(k, k - 1);
            let before = a.seq.seq_len;
            model
                .verify_abort_error
                .store(abort_error, Ordering::SeqCst);
            a.cancel_flag = Some(Arc::new(AtomicBool::new(true)));
            step(&model, &mut a, &drafts, k);
            assert!(a.finished);
            assert_eq!(a.seq.seq_len, before);
            assert_eq!(a.output_tokens, [31]);
            assert_eq!(*model.verify_aborts.lock().unwrap(), [(before, before, k)]);
            assert!(model.verify_commits.lock().unwrap().is_empty());
        }
    }
}

#[test]
fn cancellation_after_a_hidden_row_keeps_that_row_without_inventing_output() {
    let (model, mut a, _) = setup(3, 2);
    let before = a.seq.seq_len;
    let mut frame = ShortVerifyFrame::capture(&a, 3);
    let tokens = [31, 41, 42];
    a.seq.seq_len += 3;
    a.seq.tokens.extend_from_slice(&tokens);
    a.think_end_token = Some(41);
    frame.emit(&mut a, 41, None);
    assert!(!a.finished);
    a.cancel_flag = Some(Arc::new(AtomicBool::new(true)));
    frame.emit(&mut a, 42, None);
    frame.reconcile_terminal(&model, &mut a, 2, &tokens);
    assert!(a.finished);
    assert_eq!(a.output_tokens, [31]);
    assert_eq!(*model.verify_commits.lock().unwrap(), [(before + 1, 1, 3)]);
    assert!(model.verify_aborts.lock().unwrap().is_empty());
}

#[test]
fn terminal_commit_error_notifies_once_and_defers_retirement() {
    for k in [3, 4] {
        let (model, mut a, drafts) = setup(k, k - 1);
        *model.verify_failure.lock().unwrap() = Some("commit");
        let before = a.seq.seq_len;
        let (tx, mut rx) = tokio::sync::mpsc::channel(8);
        a.sink = ResponseSink::Streaming(tx);
        a.remaining = 1;
        step(&model, &mut a, &drafts, k);
        assert!(a.finished);
        assert_eq!(*model.verify_aborts.lock().unwrap(), [(before, before, k)]);
        assert_eq!(model.verify_commits.lock().unwrap().len(), 1);
        assert!(model.frees.lock().unwrap().is_empty());
        let mut errors = 0;
        while let Ok(event) = rx.try_recv() {
            match event {
                StreamEvent::Error(error) => {
                    assert!(error.contains("short terminal verify commit"));
                    errors += 1;
                }
                StreamEvent::Done { .. } => panic!("failed commit cannot report Done"),
                _ => {}
            }
        }
        assert_eq!(errors, 1);
        assert!(matches!(a.sink, ResponseSink::Blocking(None)));
    }
}

#[test]
fn postcommit_proposer_errors_terminate_without_aborting_consumed_state() {
    for k in [3, 4] {
        for accepted in 0..k {
            for stage in ["save hidden", "trim", "propose"] {
                let (model, mut a, drafts) = setup(k, accepted);
                *model.verify_failure.lock().unwrap() = Some(stage);
                let before = a.seq.seq_len;
                let (tx, mut rx) = tokio::sync::oneshot::channel();
                a.sink = ResponseSink::Blocking(Some(tx));
                step(&model, &mut a, &drafts, k);
                assert!(a.finished);
                assert_eq!(a.seq.seq_len, before + accepted + 1);
                assert_eq!(
                    *model.verify_commits.lock().unwrap(),
                    [(before + accepted + 1, accepted + 1, k)],
                );
                assert!(model.verify_aborts.lock().unwrap().is_empty());
                assert!(model.frees.lock().unwrap().is_empty());
                assert!(a.pending_drafts.is_empty());
                let error = match rx.try_recv().unwrap() {
                    Ok(_) => panic!("postcommit proposer failure cannot succeed"),
                    Err(error) => error.to_string(),
                };
                assert!(error.contains(stage), "{error}");
            }
        }
    }
}
