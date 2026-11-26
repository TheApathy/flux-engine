// SPDX-License-Identifier: AGPL-3.0-only

const SCHEDULER: &str = include_str!("../src/scheduler/mod.rs");
const DECODE_STEP: &str = include_str!("../src/scheduler/decode_step.rs");
const HELPERS: &str = include_str!("../src/scheduler/mod_helpers.rs");
const LIFECYCLE: &str = include_str!("../src/scheduler/lifecycle.rs");
const MTP: &str = include_str!("../src/scheduler/mtp_step.rs");

fn function_body<'a>(source: &'a str, signature: &str) -> &'a str {
    let start = source.find(signature).expect("function signature missing");
    let open = start
        + source[start..]
            .find('{')
            .expect("function opening brace missing");
    let mut depth = 0usize;
    for (offset, byte) in source.as_bytes()[open..].iter().enumerate() {
        match byte {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return &source[start..open + offset + 1];
                }
            }
            _ => {}
        }
    }
    panic!("function closing brace missing");
}

fn call_bodies<'a>(source: &'a str, callee: &str) -> Vec<&'a str> {
    let needle = format!("{callee}(");
    let mut calls = Vec::new();
    let mut cursor = 0usize;
    while let Some(relative) = source[cursor..].find(&needle) {
        let start = cursor + relative;
        let open = start + needle.len() - 1;
        let mut depth = 0usize;
        for (offset, byte) in source.as_bytes()[open..].iter().enumerate() {
            match byte {
                b'(' => depth += 1,
                b')' => {
                    depth -= 1;
                    if depth == 0 {
                        let end = open + offset + 1;
                        calls.push(&source[start..end]);
                        cursor = end;
                        break;
                    }
                }
                _ => {}
            }
        }
        assert!(cursor > start, "call closing parenthesis missing");
    }
    calls
}

#[test]
fn scheduler_carries_one_cache_policy_per_active_sequence() {
    let scheduler = function_body(SCHEDULER, "pub fn run(");
    assert!(scheduler.contains("let mut cache_on_finish = vec![true; active.len()];"));
    assert_eq!(scheduler.matches("step_mtp(").count(), 3);
    let decode_calls = call_bodies(scheduler, "step_decode_only");
    assert_eq!(decode_calls.len(), 6);
    assert!(
        decode_calls
            .iter()
            .all(|call| call.contains("&mut cache_on_finish"))
    );
    assert!(scheduler.contains("let was_verify = !active[0].pending_drafts.is_empty();"));
    let ngram_finish = &scheduler[scheduler
        .find("if was_verify && active[0].finished")
        .expect("ngram verified-finish policy missing")..];
    let ngram_finish = &ngram_finish[..ngram_finish
        .find('}')
        .expect("ngram verified-finish policy closing brace missing")];
    assert!(ngram_finish.contains("cache_on_finish[0] = false"));
    let self_spec_calls = call_bodies(scheduler, "step_self_spec");
    assert_eq!(self_spec_calls.len(), 1);
    assert!(self_spec_calls[0].contains("&mut self_spec_sync_pending"));
    assert!(self_spec_calls[0].contains("&think_ctx"));
    assert!(
        scheduler.contains("retire_finished_sequences(&*model, &mut active, &mut cache_on_finish)")
    );
}

#[test]
fn every_speculative_verify_exit_suppresses_finished_prefixes() {
    let step = function_body(MTP, "pub fn step_mtp(");
    assert!(step.contains("cache_on_finish: &mut [bool]"));
    // Five verify exits since 295a6aaa1 added the batched K5 frame path.
    assert_eq!(step.matches("suppress_finished_verify_cache(").count(), 5);

    let helper = function_body(MTP, "fn suppress_finished_verify_cache(");
    assert!(helper.contains("for &idx in verify_idxs"));
    assert!(helper.contains("if active[idx].finished"));
    assert!(helper.contains("cache_on_finish[idx] = false"));
}

#[test]
fn decode_failures_drain_the_parallel_cache_policy() {
    let step = function_body(DECODE_STEP, "pub fn step_decode_only(");
    assert!(step.contains("cache_on_finish: &mut Vec<bool>"));
    assert!(step.contains("EP broadcast token"));
    assert!(step.contains("decode_batch error"));
    assert!(step.contains("if active.is_empty()"));
    assert_eq!(step.matches("cache_on_finish.clear();").count(), 3);
}

#[test]
fn retirement_swaps_policy_with_sequence_and_gates_only_cache_insert() {
    let retire = function_body(HELPERS, "pub(super) fn retire_finished_sequences(");
    assert!(retire.contains("assert_eq!(active.len(), cache_on_finish.len())"));
    // Since 823d03146 the policy leaves with its sequence in both retire
    // orders: order-preserving `remove` for hc_mult models, `swap_remove` otherwise.
    assert!(retire.contains("(active.remove(i), cache_on_finish.remove(i))"));
    assert!(retire.contains("(active.swap_remove(i), cache_on_finish.swap_remove(i))"));
    assert!(retire.contains("finish_sequence_with_cache(model, &mut a, cache_sequence)"));

    let finish = function_body(LIFECYCLE, "pub(super) fn finish_sequence_with_cache(");
    assert!(finish.contains("if cache_sequence"));
    assert_eq!(finish.matches("model.cache_sequence(&a.seq)").count(), 1);
    assert!(finish.contains("model.free_sequence(&mut a.seq)"));
    assert!(finish.contains("model.ep_broadcast_cmd(0xFFFFFFF1)"));

    let ordinary = function_body(LIFECYCLE, "pub fn finish_sequence(");
    assert!(ordinary.contains("finish_sequence_with_cache(model, a, true)"));
}
