// SPDX-License-Identifier: AGPL-3.0-only

//! Terminal watchdog verdicts must survive structural-token early returns.

use super::*;
use crate::scheduler::test_fixtures::active_seq;

fn token_logprobs(token_id: u32) -> crate::api::TokenLogprobs {
    crate::api::TokenLogprobs {
        token_id,
        logprob: -0.5,
        top: vec![(token_id, -0.5)],
    }
}

fn with_ledger(a: &mut ActiveSeq) {
    a.intervention_ledger = Some(std::sync::Arc::new(std::sync::Mutex::new(
        crate::openai::InterventionLedger::default(),
    )));
}

fn assert_single_intervention(a: &ActiveSeq, code: &str) {
    let ledger = crate::openai::snapshot_shared_interventions(a.intervention_ledger.as_ref())
        .expect("audit requested");
    assert_eq!(ledger.events.len(), 1, "{ledger:?}");
    assert_eq!(ledger.events[0].code, code);
    assert_eq!(ledger.events[0].action, "terminate_generation");
    assert_eq!(ledger.events[0].count, 1);
}

fn looping_sequence(sink: ResponseSink) -> ActiveSeq {
    let mut a = active_seq(sink, 0);
    a.output_tokens = vec![42; CATASTROPHIC_LOOP_MIN_TOKENS];
    a.logprobs_data = vec![token_logprobs(42); a.output_tokens.len()];
    a.content_tokens = CATASTROPHIC_LOOP_MIN_TOKENS as u32 - 1;
    a.max_output_tokens = 4096;
    a.remaining = 2048;
    a.require_tool_call = true;
    with_ledger(&mut a);
    a
}

#[test]
fn catastrophic_watchdog_retires_before_suppressed_eos_return() {
    for streaming in [false, true] {
        for report_interventions in [false, true] {
            let (tx, mut rx) = tokio::sync::mpsc::channel(4);
            let sink = if streaming {
                ResponseSink::Streaming(tx)
            } else {
                ResponseSink::Blocking(None)
            };
            let mut a = looping_sequence(sink);
            if !report_interventions {
                a.intervention_ledger = None;
            }
            let eos = a.eos_tokens[0];
            let before = a.output_tokens.clone();

            emit_token_with_policies(&mut a, eos, Some(token_logprobs(eos)), false, false, true);

            assert!(a.finished, "a suppressed EOS cannot bypass the watchdog");
            assert_eq!(a.output_tokens, before, "EOS stays uncommitted");
            assert_eq!(a.logprobs_data.len(), before.len());
            assert!(a.logprobs_data.iter().all(|lp| lp.token_id == 42));
            assert_eq!(a.remaining, 2047, "charge exactly one content position");
            assert!(rx.try_recv().is_err(), "no EOS token enters the stream");
            if report_interventions {
                assert_single_intervention(&a, "catastrophic_content_loop_watchdog");
            } else {
                assert!(a.intervention_ledger.is_none());
            }
        }
    }
}

#[test]
fn configured_watchdog_also_retires_before_suppressed_eos_return() {
    let mut a = looping_sequence(ResponseSink::Blocking(None));
    let eos = a.eos_tokens[0];
    let before = a.output_tokens.clone();

    emit_token_with_policies(&mut a, eos, Some(token_logprobs(eos)), false, true, false);

    assert!(a.finished);
    assert_eq!(a.output_tokens, before);
    assert_eq!(a.logprobs_data.len(), before.len());
    assert_single_intervention(&a, "content_loop_watchdog");
}

#[test]
fn disabled_watchdogs_do_not_retire_suppressed_eos_early() {
    let mut a = looping_sequence(ResponseSink::Blocking(None));
    let eos = a.eos_tokens[0];
    let before = a.output_tokens.clone();

    emit_token_with_policies(&mut a, eos, Some(token_logprobs(eos)), false, false, false);

    assert!(!a.finished);
    assert_eq!(a.output_tokens, before);
    assert_eq!(a.logprobs_data.len(), before.len());
    assert!(
        crate::openai::snapshot_shared_interventions(a.intervention_ledger.as_ref())
            .unwrap()
            .events
            .is_empty()
    );
}

fn json_grammar_after_open_object() -> crate::grammar::GrammarState {
    let mut vocab: Vec<String> = (0u8..128).map(|byte| String::from(byte as char)).collect();
    vocab.push("<eos>".to_string());
    let mut engine = crate::grammar::GrammarEngine::new(&vocab, &[128]).unwrap();
    let compiled = engine.compile_json_grammar().unwrap();
    let mut state = crate::grammar::GrammarState::new(&compiled, engine.vocab_size()).unwrap();
    assert!(state.accept_token(b'{' as u32));
    state
}

fn prose_exhausted_sequence() -> ActiveSeq {
    let mut a = active_seq(ResponseSink::Blocking(None), 0);
    a.output_tokens = vec![b'{' as u32];
    a.logprobs_data = vec![token_logprobs(b'{' as u32)];
    a.eos_tokens = vec![128];
    a.grammar_state = Some(json_grammar_after_open_object());
    a.tool_call_start_token = Some(100);
    a.prose_tokens_since_last_tool = watchdog_params().max_inter_tool_prose;
    with_ledger(&mut a);
    a
}

#[test]
fn response_format_json_grammar_never_triggers_inter_tool_prose_watchdog() {
    let mut a = prose_exhausted_sequence();
    a.tool_call_start_token = None;

    emit_token_with_policies(&mut a, b'"' as u32, None, false, false, false);

    assert!(!a.finished);
    assert!(
        crate::openai::snapshot_shared_interventions(a.intervention_ledger.as_ref())
            .unwrap()
            .events
            .is_empty()
    );
}

#[test]
fn inter_tool_prose_watchdog_retires_at_grammar_suppressed_eos() {
    let mut a = prose_exhausted_sequence();
    let before = a.output_tokens.clone();

    emit_token_with_policies(&mut a, 128, Some(token_logprobs(128)), false, false, false);

    assert!(a.finished);
    assert!(!a.grammar_state.as_ref().unwrap().is_terminated());
    assert_eq!(a.output_tokens, before);
    assert_eq!(a.logprobs_data.len(), before.len());
    assert_single_intervention(&a, "inter_tool_prose_watchdog");
}

#[test]
fn grammar_prose_stop_matches_serial_content_policy_and_ledger() {
    let model = crate::scheduler::test_fixtures::model::recording();
    let mut serial = prose_exhausted_sequence();
    let mut speculative = prose_exhausted_sequence();
    let previous_last_token = serial.last_token;
    let previous_last_token_time = serial.last_token_time;

    assert!(matches!(
        handle_content_token(
            &mut serial,
            128,
            &model,
            previous_last_token,
            previous_last_token_time,
        ),
        ContentTokenDisposition::CommitSample
    ));
    emit_token_with_policies(&mut speculative, 128, None, false, false, false);

    assert!(serial.finished && speculative.finished);
    assert_eq!(serial.output_tokens, speculative.output_tokens);
    assert_eq!(serial.logprobs_data.len(), speculative.logprobs_data.len());
    assert_eq!(serial.remaining, speculative.remaining);
    assert_eq!(serial.content_tokens, speculative.content_tokens);
    assert_eq!(
        crate::openai::snapshot_shared_interventions(serial.intervention_ledger.as_ref()),
        crate::openai::snapshot_shared_interventions(speculative.intervention_ledger.as_ref()),
    );
}

#[test]
fn inter_tool_prose_stop_survives_nonterminal_tool_close() {
    let (tx, mut rx) = tokio::sync::mpsc::channel(4);
    let mut a = prose_exhausted_sequence();
    a.sink = ResponseSink::Streaming(tx);
    a.tool_call_end_token = Some(b'}' as u32);

    emit_token_with_policies(
        &mut a,
        b'}' as u32,
        Some(token_logprobs(b'}' as u32)),
        false,
        false,
        false,
    );

    assert!(a.finished, "nonterminal grammar cannot undo the watchdog");
    assert!(!a.grammar_state.as_ref().unwrap().is_terminated());
    assert_eq!(a.output_tokens, vec![b'{' as u32, b'}' as u32]);
    assert_eq!(a.logprobs_data.len(), 2, "close commits exactly once");
    assert!(
        matches!(rx.try_recv(), Ok(StreamEvent::TokenWithLogprobs(tok, _)) if tok == b'}' as u32)
    );
    assert!(rx.try_recv().is_err(), "one structural-close event");
    assert_single_intervention(&a, "inter_tool_prose_watchdog");
}

#[test]
fn terminal_content_watchdog_does_not_record_a_second_fuzzy_stop() {
    let mut a = looping_sequence(ResponseSink::Blocking(None));
    a.require_tool_call = false;

    emit_token_with_policies(&mut a, 42, Some(token_logprobs(42)), false, true, true);

    assert!(a.finished);
    assert_eq!(a.output_tokens.len(), CATASTROPHIC_LOOP_MIN_TOKENS + 1);
    assert_eq!(a.logprobs_data.len(), a.output_tokens.len());
    assert_single_intervention(&a, "catastrophic_content_loop_watchdog");
}

#[test]
fn implicit_thinking_eos_does_not_gain_a_content_watchdog_stop() {
    let (tx, mut rx) = tokio::sync::mpsc::channel(4);
    let mut a = looping_sequence(ResponseSink::Streaming(tx));
    a.require_tool_call = false;
    a.inside_thinking = true;
    a.enable_thinking = true;
    a.think_ended = false;
    a.thinking_tokens = 1;
    let eos = a.eos_tokens[0];
    let before = a.output_tokens.clone();

    emit_token_with_policies(&mut a, eos, Some(token_logprobs(eos)), true, true, true);

    assert!(!a.finished, "implicit thinking close is not a content stop");
    assert!(!a.inside_thinking && a.think_ended && a.think_just_ended);
    assert_eq!(a.output_tokens, before);
    assert_eq!(a.logprobs_data.len(), before.len());
    assert_eq!(a.remaining, 2048, "thinking does not charge content budget");
    assert!(matches!(rx.try_recv(), Ok(StreamEvent::ThinkingDone)));
    assert!(rx.try_recv().is_err());
    assert!(
        crate::openai::snapshot_shared_interventions(a.intervention_ledger.as_ref())
            .unwrap()
            .events
            .is_empty()
    );
}

#[test]
fn tool_body_stays_exempt_from_content_watchdogs() {
    let mut a = looping_sequence(ResponseSink::Blocking(None));
    a.inside_tool_body = true;
    let eos = a.eos_tokens[0];

    emit_token_with_policies(&mut a, eos, Some(token_logprobs(eos)), false, true, true);

    assert!(!a.finished);
    assert_eq!(a.output_tokens.len(), CATASTROPHIC_LOOP_MIN_TOKENS);
    assert!(
        crate::openai::snapshot_shared_interventions(a.intervention_ledger.as_ref())
            .unwrap()
            .events
            .is_empty()
    );
}
