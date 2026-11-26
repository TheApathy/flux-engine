// SPDX-License-Identifier: AGPL-3.0-only

//! GPU-free K2 scheduler tests; require the shared short-frame test support.

use super::*;
use crate::scheduler::test_fixtures::{active_seq, model};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

fn setup(accepted: bool) -> (model::TestModel, ActiveSeq, [u32; 1]) {
    let model = model::recording();
    *model.verify_failure.lock().unwrap() = Some("success");
    *model.verify_targets.lock().unwrap() = vec![if accepted { 41 } else { 77 }, 77];
    let mut a = active_seq(ResponseSink::Blocking(None), 3);
    a.seq.tokens = vec![7; a.seq.seq_len];
    a.pending_drafts = vec![41];
    (model, a, [41])
}

#[test]
fn verifier_result_and_commit_failures_restore_host_extent_and_abort_once() {
    for accepted in [false, true] {
        for stage in ["decode_verify", "EP result", "commit"] {
            for abort_error in [false, true] {
                let (model, mut a, drafts) = setup(accepted);
                *model.verify_failure.lock().unwrap() = Some(stage);
                model
                    .verify_abort_error
                    .store(abort_error, Ordering::SeqCst);
                let before = a.seq.seq_len;
                let (tx, mut rx) = tokio::sync::oneshot::channel();
                a.sink = ResponseSink::Blocking(Some(tx));
                step_verify_k2(&model, &mut a, &drafts, 1);
                assert!(a.finished);
                assert_eq!((a.seq.seq_len, a.seq.tokens.len()), (before, before));
                assert_eq!(*model.verify_aborts.lock().unwrap(), [(before, before, 2)]);
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

#[test]
fn preentry_failures_do_not_abort_an_unstarted_k2_frame() {
    for stage in ["sync_secondary", "EP command", "EP token"] {
        let (model, mut a, drafts) = setup(false);
        *model.verify_failure.lock().unwrap() = Some(stage);
        step_verify_k2(&model, &mut a, &drafts, 1);
        assert!(a.finished);
        assert!(model.verify_aborts.lock().unwrap().is_empty());
        assert!(model.verify_commits.lock().unwrap().is_empty());
    }
}

#[test]
fn length_terminal_commits_only_classified_k2_input_rows() {
    for accepted in [false, true] {
        for terminal_row in 1..=1 + usize::from(accepted) {
            let (model, mut a, drafts) = setup(accepted);
            let before = a.seq.seq_len;
            a.remaining = terminal_row;
            step_verify_k2(&model, &mut a, &drafts, 1);
            assert!(a.finished);
            assert_eq!(
                (a.seq.seq_len, a.seq.tokens.len()),
                (before + terminal_row, before + terminal_row),
            );
            assert_eq!(
                *model.verify_commits.lock().unwrap(),
                [(before + terminal_row, terminal_row, 2)],
            );
            assert!(model.verify_aborts.lock().unwrap().is_empty());
            assert_eq!(model.proposals.load(Ordering::SeqCst), 0);
        }
    }
}

#[test]
fn eos_at_each_accepted_or_bonus_row_consumes_one_k2_commit() {
    for accepted in [false, true] {
        for terminal_row in 1..=1 + usize::from(accepted) {
            let (model, mut a, drafts) = setup(accepted);
            let before = a.seq.seq_len;
            a.eos_tokens = vec![if accepted && terminal_row == 1 {
                41
            } else {
                77
            }];
            step_verify_k2(&model, &mut a, &drafts, 1);
            assert!(a.finished);
            assert_eq!(
                *model.verify_commits.lock().unwrap(),
                [(before + terminal_row, terminal_row, 2)],
            );
            assert!(model.verify_aborts.lock().unwrap().is_empty());
        }
    }
}

#[test]
fn hidden_control_and_suppressed_eos_rows_count_toward_terminal_prefix() {
    for suppressed_eos in [false, true] {
        let (model, mut a, drafts) = setup(true);
        let before = a.seq.seq_len;
        if suppressed_eos {
            a.eos_tokens = vec![41];
            a.min_tokens = 100;
            a.remaining = 2;
        } else {
            a.think_end_token = Some(41);
            a.remaining = 1;
        }
        step_verify_k2(&model, &mut a, &drafts, 1);
        assert!(a.finished);
        assert_eq!(a.output_tokens.len(), 2);
        assert_eq!(*model.verify_commits.lock().unwrap(), [(before + 2, 2, 2)]);
        assert!(model.verify_aborts.lock().unwrap().is_empty());
    }
}

#[test]
fn pre_row_cancellation_aborts_k2_without_inventing_committed_output() {
    for accepted in [false, true] {
        for abort_error in [false, true] {
            let (model, mut a, drafts) = setup(accepted);
            let before = a.seq.seq_len;
            model
                .verify_abort_error
                .store(abort_error, Ordering::SeqCst);
            a.cancel_flag = Some(Arc::new(AtomicBool::new(true)));
            step_verify_k2(&model, &mut a, &drafts, 1);
            assert!(a.finished);
            assert_eq!((a.seq.seq_len, a.seq.tokens.len()), (before, before));
            assert_eq!(a.output_tokens, [31]);
            assert_eq!(*model.verify_aborts.lock().unwrap(), [(before, before, 2)]);
            assert!(model.verify_commits.lock().unwrap().is_empty());
        }
    }
}

#[test]
fn terminal_commit_failure_notifies_stream_once_without_early_retirement() {
    for accepted in [false, true] {
        let (model, mut a, drafts) = setup(accepted);
        *model.verify_failure.lock().unwrap() = Some("commit");
        let before = a.seq.seq_len;
        let (tx, mut rx) = tokio::sync::mpsc::channel(8);
        a.sink = ResponseSink::Streaming(tx);
        a.remaining = 1;
        step_verify_k2(&model, &mut a, &drafts, 1);
        assert!(a.finished);
        assert_eq!(*model.verify_aborts.lock().unwrap(), [(before, before, 2)]);
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
fn postcommit_proposer_failures_notify_without_aborting_consumed_journal() {
    for accepted in [false, true] {
        for stage in ["save hidden", "trim", "propose"] {
            let (model, mut a, drafts) = setup(accepted);
            *model.verify_failure.lock().unwrap() = Some(stage);
            let before = a.seq.seq_len;
            let consumed = 1 + usize::from(accepted);
            let (tx, mut rx) = tokio::sync::oneshot::channel();
            a.sink = ResponseSink::Blocking(Some(tx));
            step_verify_k2(&model, &mut a, &drafts, 1);
            assert!(a.finished);
            assert_eq!(a.seq.seq_len, before + consumed);
            assert_eq!(
                *model.verify_commits.lock().unwrap(),
                [(before + consumed, consumed, 2)],
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
