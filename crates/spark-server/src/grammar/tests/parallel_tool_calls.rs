// SPDX-License-Identifier: AGPL-3.0-only

//! D1 (API audit, 2026-10-21): `parallel_tool_calls=false` must end
//! generation at the first `</tool_call>` in both the blocking and the
//! streaming arm. The blocking arm has no stream-side cancellation, so the
//! grammar itself has to admit nothing but a stop token once the first call
//! closes. `emit_step::commit_tool_call_close` reads that state through
//! `GrammarState::only_stop_tokens_remain` and finishes the sequence on the
//! close token.

use super::*;
use xgrammar::CompiledGrammar;

const TOOL_CALL_OPEN: u32 = 128;
const TOOL_CALL_CLOSE: u32 = 129;
const EOS: u32 = 130;

fn ascii_tokens(text: &str) -> Vec<u32> {
    text.bytes().map(u32::from).collect()
}

fn qwen3_coder_call_tokens() -> Vec<u32> {
    let mut tokens = vec![TOOL_CALL_OPEN];
    tokens.extend(ascii_tokens(
        "\n<function=get_weather>\n<parameter=location>\nParis\n</parameter>\n</function>\n",
    ));
    tokens.push(TOOL_CALL_CLOSE);
    tokens
}

fn hermes_call_tokens() -> Vec<u32> {
    let mut tokens = vec![TOOL_CALL_OPEN];
    tokens.extend(ascii_tokens(
        r#"{"name":"get_weather","arguments":{"location":"Paris"}}"#,
    ));
    tokens.push(TOOL_CALL_CLOSE);
    tokens
}

/// Mirror the scheduler: fill the mask before every token, require the
/// sampled token to be legal, then advance.
fn drive(state: &mut GrammarState, tokens: &[u32]) {
    for (pos, &token) in tokens.iter().enumerate() {
        state.fill_bitmask();
        assert!(
            state.is_token_allowed(token),
            "token {token} at position {pos} must be grammar-legal"
        );
        assert!(
            state.accept_token(token),
            "token {token} at position {pos} accepted"
        );
    }
}

fn state_for(compiled: &CompiledGrammar) -> GrammarState {
    GrammarState::new(compiled, test_vocab().len()).expect("grammar state")
}

fn engine() -> GrammarEngine {
    GrammarEngine::new(&test_vocab(), &[EOS as i32]).unwrap()
}

/// `tool_choice="auto"` + `parallel_tool_calls=false`: after one complete
/// call the matcher admits only the stop token, so the scheduler ends the
/// turn on the close token. `is_terminated()` alone would still be false
/// here (it needs an accepted stop token), which is why the scheduler asks
/// `only_stop_tokens_remain` instead.
#[test]
fn qwen3_coder_auto_stop_after_first_is_terminal_after_one_call() {
    let compiled = engine()
        .compile_qwen3_coder_tool_grammar(&test_tool_defs(), true, true)
        .expect("compile");
    let mut state = state_for(&compiled);
    drive(&mut state, &qwen3_coder_call_tokens());

    assert!(!state.is_terminated(), "no stop token accepted yet");
    assert!(state.only_stop_tokens_remain());
    state.fill_bitmask();
    assert!(state.is_token_allowed(EOS));
    assert!(
        !state.is_token_allowed(u32::from(b'a')),
        "prose after the call is masked"
    );
    assert!(
        !state.is_token_allowed(TOOL_CALL_OPEN),
        "a second call is masked"
    );
}

/// Default `parallel_tool_calls` (omitted/true) is unchanged: the auto
/// grammar loops, so prose, another call, or EOS may follow the first close.
#[test]
fn qwen3_coder_auto_default_keeps_looping_after_one_call() {
    let compiled = engine()
        .compile_qwen3_coder_tool_grammar(&test_tool_defs(), true, false)
        .expect("compile");
    let mut state = state_for(&compiled);
    drive(&mut state, &qwen3_coder_call_tokens());

    assert!(!state.only_stop_tokens_remain());
    state.fill_bitmask();
    assert!(
        state.is_token_allowed(EOS),
        "the model may still end the turn"
    );
    assert!(
        state.is_token_allowed(u32::from(b'a')),
        "prose may continue"
    );
    assert!(
        state.is_token_allowed(TOOL_CALL_OPEN),
        "another call may follow"
    );
}

/// Required/specific `tool_choice` forces the one-call policy whatever the
/// caller passes for `stop_after_first`.
#[test]
fn qwen3_coder_required_is_terminal_after_one_call() {
    for requested in [false, true] {
        let compiled = engine()
            .compile_qwen3_coder_tool_grammar(&test_tool_defs(), false, requested)
            .expect("compile");
        let mut state = state_for(&compiled);
        drive(&mut state, &qwen3_coder_call_tokens());
        assert!(
            state.only_stop_tokens_remain(),
            "stop_after_first={requested}"
        );
    }
}

#[test]
fn hermes_auto_stop_after_first_is_terminal_after_one_call() {
    let compiled = engine()
        .compile_hermes_tool_grammar(&test_tool_defs(), true, true)
        .expect("compile");
    let mut state = state_for(&compiled);
    drive(&mut state, &hermes_call_tokens());
    assert!(state.only_stop_tokens_remain());

    let looping = engine()
        .compile_hermes_tool_grammar(&test_tool_defs(), true, false)
        .expect("compile");
    let mut state = state_for(&looping);
    drive(&mut state, &hermes_call_tokens());
    assert!(!state.only_stop_tokens_remain());
}

/// Before the first call the stop-after-first auto grammar is still an
/// ordinary auto grammar: prose and EOS are legal, and nothing is terminal.
#[test]
fn stop_after_first_does_not_force_a_call() {
    let compiled = engine()
        .compile_qwen3_coder_tool_grammar(&test_tool_defs(), true, true)
        .expect("compile");
    let mut state = state_for(&compiled);
    drive(&mut state, &ascii_tokens("Sure, let me check."));
    assert!(!state.only_stop_tokens_remain());
    state.fill_bitmask();
    assert!(state.is_token_allowed(EOS));
    assert!(state.is_token_allowed(TOOL_CALL_OPEN));
}

/// Once a stop token is accepted the matcher is terminated outright.
#[test]
fn accepted_stop_token_is_terminal() {
    let compiled = engine()
        .compile_qwen3_coder_tool_grammar(&test_tool_defs(), true, true)
        .expect("compile");
    let mut state = state_for(&compiled);
    drive(&mut state, &qwen3_coder_call_tokens());
    assert!(state.accept_token(EOS));
    assert!(state.is_terminated());
    assert!(state.only_stop_tokens_remain());
}

#[test]
fn completed_grammar_without_a_legal_stop_is_not_a_stop_only_state() {
    let vocab = vec!["a".to_string()];
    let mut engine = GrammarEngine::new(&vocab, &[]).unwrap();
    let compiled = engine.compile_ebnf("root ::= \"a\"", "root").unwrap();
    let mut state = GrammarState::new(&compiled, vocab.len()).unwrap();
    state.fill_bitmask();
    assert!(state.accept_token(0));
    assert!(state.fill_bitmask());
    assert!(!state.is_token_allowed(0));
    assert!(
        !state.only_stop_tokens_remain(),
        "an empty mask is not a legal stop"
    );
}

#[test]
fn cached_grammar_keeps_first_call_and_parallel_policies_distinct() {
    for first_policy in [false, true] {
        let mut engine = engine();
        for stop_after_first in [first_policy, !first_policy, first_policy] {
            let compiled = engine
                .compile_qwen3_coder_tool_grammar(&test_tool_defs(), true, stop_after_first)
                .unwrap();
            let mut state = state_for(&compiled);
            drive(&mut state, &qwen3_coder_call_tokens());
            assert_eq!(state.only_stop_tokens_remain(), stop_after_first);
        }
    }
}

fn qwen3_coder_call_with_value(value: &str) -> Vec<u32> {
    let mut tokens = vec![TOOL_CALL_OPEN];
    tokens.extend(ascii_tokens(&format!(
        "\n<function=get_weather>\n<parameter=location>\n{value}\n</parameter>\n</function>\n"
    )));
    tokens.push(TOOL_CALL_CLOSE);
    tokens
}

#[test]
fn qwen3_coder_multiline_value_with_angle_brackets_closes_the_call() {
    for value in [
        "<div class=\"x\">\n  if a < b && c > d { </div> }\n</div>",
        "</param> and </parameter-ish> and <<</parameter but not closed",
        "<",
        "line one\nline two > three\n<parameter=nested>",
    ] {
        for (auto, stop_after_first) in [(true, false), (true, true), (false, false), (false, true)]
        {
            let compiled = engine()
                .compile_qwen3_coder_tool_grammar(&test_tool_defs(), auto, stop_after_first)
                .expect("compile");
            let mut state = state_for(&compiled);
            drive(&mut state, &qwen3_coder_call_with_value(value));
            assert_eq!(
                state.only_stop_tokens_remain(),
                !auto || stop_after_first,
                "value {value:?} auto={auto} stop_after_first={stop_after_first}"
            );
        }
    }
}

#[test]
fn qwen3_coder_value_ends_at_the_parameter_close_tag() {
    let compiled = engine()
        .compile_qwen3_coder_tool_grammar(&test_tool_defs(), false, true)
        .expect("compile");
    let mut state = state_for(&compiled);
    let mut prefix = vec![TOOL_CALL_OPEN];
    prefix.extend(ascii_tokens(
        "\n<function=get_weather>\n<parameter=location>\nParis\n</parameter>",
    ));
    drive(&mut state, &prefix);
    state.fill_bitmask();
    assert!(state.is_token_allowed(u32::from(b'\n')));
    assert!(
        !state.is_token_allowed(u32::from(b'x')),
        "prose after the parameter close is not part of the value"
    );
    drive(&mut state, &ascii_tokens("\n</function>\n"));
    drive(&mut state, &[TOOL_CALL_CLOSE]);
    assert!(state.only_stop_tokens_remain());
}
