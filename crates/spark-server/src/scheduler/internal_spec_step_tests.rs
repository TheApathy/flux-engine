// SPDX-License-Identifier: AGPL-3.0-only

use std::sync::atomic::Ordering;

use super::*;

#[path = "internal_spec_step_test_support.rs"]
mod fixture;
#[path = "internal_spec_model_test_support.rs"]
mod model_support;
use fixture::active;
use model_support::{LOGITS, TestModel, VOCAB, VerifyReply};

fn step(model: &TestModel, active: &mut Vec<ActiveSeq>, cache: &mut Vec<bool>) {
    step_internal_spec(
        model,
        active,
        cache,
        Some(120),
        Some(121),
        None,
        Some(122),
        Some(123),
        &[],
        false,
    );
}

#[test]
fn dispatches_verify_commits_matching_prefix_and_selects_following_row() {
    let model = TestModel::new(
        VerifyReply::Frame(vec![10, 11, 12], vec![10, 99, 12, 77]),
        99,
    );
    let (mut active, _rx) = active();
    let mut cache = vec![true];
    step(&model, &mut active, &mut cache);

    assert_eq!(model.verify_calls.load(Ordering::SeqCst), 1);
    assert_eq!(model.decode_calls.load(Ordering::SeqCst), 0);
    assert_eq!(*model.commits.lock().unwrap(), vec![1]);
    assert_eq!(active[0].seq.seq_len, 5, "input plus one draft committed");
    assert_eq!(active[0].output_tokens, vec![10, 99]);
    assert_eq!(
        model.argmax_ptr.load(Ordering::SeqCst),
        LOGITS.0 + (VOCAB * 2) as u64,
        "bonus must come from the row after the accepted prefix"
    );
}

#[test]
fn rejection_commits_zero_drafts_and_rolls_verify_frame_back_to_input() {
    let model = TestModel::new(VerifyReply::Frame(vec![10, 11], vec![42, 11, 12]), 42);
    let (mut active, _rx) = active();
    let before = active[0].seq.seq_len;
    let mut cache = vec![true];
    step(&model, &mut active, &mut cache);

    assert_eq!(*model.commits.lock().unwrap(), vec![0]);
    assert_eq!(active[0].seq.seq_len, before + 1);
    assert_eq!(active[0].output_tokens, vec![42]);
    assert_eq!(model.argmax_ptr.load(Ordering::SeqCst), LOGITS.0);
}

#[test]
fn near_max_seq_is_the_only_safe_plain_decode_fallback() {
    let model = TestModel::new(VerifyReply::NearEnd, 55);
    let (mut active, _rx) = active();
    let mut cache = vec![true];
    step(&model, &mut active, &mut cache);

    assert_eq!(model.verify_calls.load(Ordering::SeqCst), 1);
    assert_eq!(model.decode_calls.load(Ordering::SeqCst), 1);
    assert!(model.commits.lock().unwrap().is_empty());
    assert_eq!(active[0].output_tokens, vec![55]);
}

#[test]
fn verify_and_commit_errors_terminate_without_blind_replay() {
    for commit_error in [false, true] {
        let reply = if commit_error {
            VerifyReply::Frame(vec![10], vec![10, 11])
        } else {
            VerifyReply::Error
        };
        let model = TestModel::new(reply, 88);
        model.commit_error.store(commit_error, Ordering::SeqCst);
        let (mut active, mut rx) = active();
        let mut cache = vec![true];
        step(&model, &mut active, &mut cache);

        assert!(active.is_empty() && cache.is_empty());
        assert_eq!(model.decode_calls.load(Ordering::SeqCst), 0);
        assert_eq!(model.free_calls.load(Ordering::SeqCst), 1);
        assert!(rx.try_recv().expect("error response").is_err());
    }
}

#[test]
fn decode_logits_error_clears_cache_on_finish_to_keep_retire_invariant() {
    let model = TestModel::new(VerifyReply::Frame(vec![10], vec![10, 11]), 11);
    model.argmax_error.store(true, Ordering::SeqCst);
    let (mut active, mut rx) = active();
    let mut cache = vec![true];
    step(&model, &mut active, &mut cache);

    assert!(active.is_empty());
    assert!(
        cache.is_empty(),
        "cache_on_finish must stay aligned with active so retire_finished_sequences does not panic"
    );
    retire_finished_sequences(&model, &mut active, &mut cache);
    assert!(rx.try_recv().expect("error response").is_err());
}

#[test]
fn sampled_internal_requests_hold_on_serial_before_ngram_dispatch() {
    let model = TestModel::new(VerifyReply::NearEnd, 1);
    let (mut active, _rx) = active();
    assert!(internal_spec_eligible(&active[0], &model, false));
    active[0].temperature = 1.0;
    assert!(!internal_spec_eligible(&active[0], &model, false));

    const SOURCE: &str = include_str!("mod.rs");
    let internal = SOURCE
        .find("if use_internal_spec")
        .expect("internal dispatch");
    let ngram = SOURCE
        .find("else if use_ngram_speculative")
        .expect("ngram dispatch");
    assert!(
        internal < ngram,
        "internal model dispatch must precede external paths"
    );
    let internal_arm = &SOURCE[internal..ngram];
    assert!(internal_arm.contains("step_internal_spec("));
    assert!(
        internal_arm.contains("step_decode_only("),
        "sampled/ineligible internal requests must stay serial, never fall through to ngram"
    );
}
