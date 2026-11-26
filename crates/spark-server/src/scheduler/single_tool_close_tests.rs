// SPDX-License-Identifier: AGPL-3.0-only

//! Recognized single-token tool closes share the serial/speculative boundary.

use super::*;
use crate::scheduler::test_fixtures::active_seq;

const OPEN: u32 = 128;
const CLOSE: u32 = 129;
const EOS: u32 = 130;

fn logprobs(token_id: u32) -> crate::api::TokenLogprobs {
    crate::api::TokenLogprobs {
        token_id,
        logprob: -0.5,
        top: Vec::new(),
    }
}

fn before_close(stop_after_first: bool, sink: ResponseSink) -> ActiveSeq {
    let mut vocab: Vec<String> = (0u8..128).map(|byte| String::from(byte as char)).collect();
    vocab.extend(["<tool_call>".into(), "</tool_call>".into(), "<eos>".into()]);
    let tools =
        serde_json::from_value::<Vec<crate::tool_parser::ToolDefinition>>(serde_json::json!([{
            "type": "function",
            "function": {
                "name": "ping",
                "parameters": {"type": "object", "properties": {}}
            }
        }]))
        .unwrap();
    let mut engine = GrammarEngine::new(&vocab, &[EOS as i32]).unwrap();
    let grammar = engine
        .compile_hermes_tool_grammar(&tools, true, stop_after_first)
        .unwrap();
    let mut state = GrammarState::new(&grammar, vocab.len()).unwrap();
    let prefix: Vec<u32> = std::iter::once(OPEN)
        .chain(r#"{"name":"ping","arguments":{}}"#.bytes().map(u32::from))
        .collect();
    for &token in &prefix {
        state.fill_bitmask();
        assert!(state.is_token_allowed(token));
        assert!(state.accept_token(token));
    }
    let mut a = active_seq(sink, 0);
    a.output_tokens = prefix.clone();
    a.logprobs_data = prefix.into_iter().map(logprobs).collect();
    a.top_logprobs = Some(0);
    a.eos_tokens = vec![EOS];
    a.tool_call_start_token = Some(OPEN);
    a.tool_call_end_token = Some(CLOSE);
    a.inside_tool_body = true;
    a.grammar_state = Some(state);
    a
}

fn commit_close(a: ActiveSeq, speculative: bool) -> ActiveSeq {
    if speculative {
        let mut a = a;
        emit_token(&mut a, CLOSE, Some(logprobs(CLOSE)));
        return a;
    }
    let model = crate::scheduler::test_fixtures::model::recording();
    let mut logits = vec![-10.0; EOS as usize + 1];
    logits[CLOSE as usize] = 10.0;
    *model.host_logits.lock().unwrap() = logits;
    let mut active = vec![a];
    process_decode_logits(
        &model,
        &mut active,
        DevicePtr::NULL,
        Instant::now(),
        None,
        None,
        None,
        Some(OPEN),
        Some(CLOSE),
        &[],
        false,
        false,
    );
    active.pop().unwrap()
}

fn assert_close(a: &ActiveSeq, finished: bool, old_remaining: usize, old_content: u32) {
    assert_eq!(a.finished, finished);
    assert_eq!(a.output_tokens.last(), Some(&CLOSE));
    assert_eq!(a.output_tokens.iter().filter(|&&t| t == CLOSE).count(), 1);
    assert!(!a.output_tokens.contains(&EOS), "no forced EOS decode step");
    assert_eq!(a.output_tokens.len(), a.logprobs_data.len());
    assert_eq!(a.logprobs_data.last().unwrap().token_id, CLOSE);
    assert_eq!(a.remaining, old_remaining - 1);
    assert_eq!(a.content_tokens, old_content + 1);
    assert!(!a.inside_tool_body);
    assert_eq!(
        a.think_ended, finished,
        "only a multi-call close rearms thinking"
    );
    assert!(!a.grammar_state.as_ref().unwrap().is_terminated());
}

#[test]
fn serial_and_speculative_blocking_stop_at_first_recognized_close() {
    for speculative in [false, true] {
        let (tx, mut rx) = tokio::sync::oneshot::channel();
        let a = before_close(true, ResponseSink::Blocking(Some(tx)));
        let (remaining, content) = (a.remaining, a.content_tokens);
        let mut a = commit_close(a, speculative);
        assert_close(&a, true, remaining, content);
        let completion_len = a.output_tokens.len();
        let model = crate::scheduler::test_fixtures::model::recording();
        finish_sequence_with_cache(&model, &mut a, false);
        let response = rx.try_recv().unwrap().unwrap();
        assert_eq!(response.finish_reason, "tool_calls");
        assert_eq!(response.output_tokens.len(), completion_len);
    }
}

#[test]
fn serial_and_speculative_streaming_publish_close_once_then_tool_calls_done() {
    for speculative in [false, true] {
        let (tx, mut rx) = tokio::sync::mpsc::channel(4);
        let a = before_close(true, ResponseSink::Streaming(tx));
        let (remaining, content) = (a.remaining, a.content_tokens);
        let mut a = commit_close(a, speculative);
        assert_close(&a, true, remaining, content);
        assert!(matches!(
            rx.try_recv().unwrap(),
            StreamEvent::TokenWithLogprobs(CLOSE, _)
        ));
        let completion_len = a.output_tokens.len();
        let model = crate::scheduler::test_fixtures::model::recording();
        finish_sequence_with_cache(&model, &mut a, false);
        match rx.try_recv().unwrap() {
            StreamEvent::Done {
                finish_reason,
                completion_tokens,
                ..
            } => {
                assert_eq!(finish_reason, "tool_calls");
                assert_eq!(completion_tokens, completion_len);
            }
            _ => panic!("expected terminal Done after the one close event"),
        }
        assert!(rx.try_recv().is_err(), "no extra close or EOS event");
    }
}

#[test]
fn serial_and_speculative_parallel_grammar_remains_open_for_both_sinks() {
    for speculative in [false, true] {
        let a = before_close(false, ResponseSink::Blocking(None));
        let (remaining, content) = (a.remaining, a.content_tokens);
        let a = commit_close(a, speculative);
        assert_close(&a, false, remaining, content);

        let (tx, mut rx) = tokio::sync::mpsc::channel(4);
        let a = before_close(false, ResponseSink::Streaming(tx));
        let (remaining, content) = (a.remaining, a.content_tokens);
        let a = commit_close(a, speculative);
        assert_close(&a, false, remaining, content);
        assert!(matches!(
            rx.try_recv().unwrap(),
            StreamEvent::TokenWithLogprobs(CLOSE, _)
        ));
        assert!(
            rx.try_recv().is_err(),
            "parallel grammar has no terminal event"
        );
    }
}
