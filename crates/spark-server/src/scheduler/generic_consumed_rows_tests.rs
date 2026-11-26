// SPDX-License-Identifier: AGPL-3.0-only

//! GPU-free definitions for explicit classified-row terminal ownership.

use super::*;
use crate::scheduler::test_fixtures::{active_seq, model};
use std::sync::{Arc, atomic::AtomicBool};

fn frame() -> (model::TestModel, ActiveSeq, usize, [u32; 5]) {
    let model = model::recording();
    *model.verify_failure.lock().unwrap() = Some("success");
    let mut a = active_seq(ResponseSink::Blocking(None), 3);
    let before = a.seq.seq_len;
    let tokens = [31, 41, 42, 43, 44];
    a.seq.tokens = vec![7; before];
    // The normal accept rollback already retained root + three drafts.
    a.seq.tokens.extend_from_slice(&tokens[..4]);
    a.seq.seq_len += 4;
    (model, a, before, tokens)
}

#[test]
fn generic_hidden_row_then_length_terminal_keeps_both_classified_inputs() {
    for suppressed_eos in [false, true] {
        let (model, mut a, before, tokens) = frame();
        if suppressed_eos {
            a.eos_tokens = vec![41];
            a.min_tokens = 100;
            a.remaining = 2;
        } else {
            a.think_end_token = Some(41);
            a.remaining = 1;
        }
        let mut consumed = 0;
        for token in [41, 42] {
            if emit_token_for_verify(&mut a, token, None) {
                consumed += 1;
            }
        }
        assert!(a.finished);
        assert_eq!(a.output_tokens, [31, 42]);
        let plan = reconcile_terminal_verify_state(
            &model, &mut a, before, 5, 3, consumed, &tokens, false, None,
        )
        .unwrap();
        assert_eq!(plan.committed_inputs, 2);
        assert_eq!(*model.verify_commits.lock().unwrap(), [(before + 2, 2, 5)]);
    }
}

#[test]
fn generic_cancellation_after_hidden_row_preserves_only_admitted_prefix() {
    let (model, mut a, before, tokens) = frame();
    a.think_end_token = Some(41);
    assert!(emit_token_for_verify(&mut a, 41, None));
    let consumed = 1;
    a.cancel_flag = Some(Arc::new(AtomicBool::new(true)));
    assert!(!emit_token_for_verify(&mut a, 42, None));
    let plan = reconcile_terminal_verify_state(
        &model, &mut a, before, 5, 3, consumed, &tokens, false, None,
    )
    .unwrap();
    assert_eq!(a.output_tokens, [31]);
    assert_eq!(plan.committed_inputs, 1);
    assert_eq!(*model.verify_commits.lock().unwrap(), [(before + 1, 1, 5)]);
}

#[test]
fn generic_preclassification_cancellation_uses_zero_input_abort() {
    let (model, mut a, before, tokens) = frame();
    a.cancel_flag = Some(Arc::new(AtomicBool::new(true)));
    assert!(!emit_token_for_verify(&mut a, 41, None));
    let plan =
        reconcile_terminal_verify_state(&model, &mut a, before, 5, 3, 0, &tokens, false, None)
            .unwrap();
    assert_eq!(plan.committed_inputs, 0);
    assert_eq!(*model.verify_aborts.lock().unwrap(), [(before, before, 5)]);
    assert!(model.verify_commits.lock().unwrap().is_empty());
}

#[test]
fn semantic_tree_slot_uses_classified_rows_not_visible_output() {
    let plan = terminal_verify_commit_plan(100, 104, 104, 8, 3, 0, 3, Some(&[1, 4, 2])).unwrap();
    assert_eq!(plan.committed_inputs, 3);
    assert_eq!(plan.target_seq_len, 103);
    assert_eq!(plan.last_inter_slot, Some(4));
    assert!(terminal_verify_commit_plan(100, 104, 104, 8, 3, 0, 5, None).is_err());
}
