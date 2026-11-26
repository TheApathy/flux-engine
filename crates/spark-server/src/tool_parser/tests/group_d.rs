// SPDX-License-Identifier: AGPL-3.0-only
#![allow(unused_imports, dead_code)]

use super::super::*;

// ────────────────────────────────────────────────────────────────────────
// Streaming `flush()` must not re-emit `ToolCallStart` when the incremental
// path already emitted it.
//
// When the model emits `<tool_call>...<function=NAME>...args` and the stream
// ends without `</tool_call>` (closing tag is the stop token, not streamed),
// emitting a fresh `DetectorOutput::ToolCall` would create a second
// `tool_calls[0]` header with a new `id`, breaking client dispatch.
//
// Instead, when `current_tc_name.is_some()` (incremental start was already
// emitted), `flush()` emits `ToolCallDelta + ToolCallEnd` matching the
// in-stream close path.
// ────────────────────────────────────────────────────────────────────────

#[test]
fn flush_after_incremental_start_does_not_double_emit() {
    let mut det = StreamingToolDetector::new();

    // Simulate token-by-token streaming up to the function header.
    // After this, ToolCallStart has been emitted incrementally.
    let pre_flush =
        det.process("<tool_call>\n<function=web_search>\n<parameter=query>meeting</parameter>\n");
    let starts: Vec<_> = pre_flush
        .iter()
        .filter(|o| matches!(o, DetectorOutput::ToolCallStart { .. }))
        .collect();
    assert_eq!(starts.len(), 1, "expected 1 incremental ToolCallStart");

    // Stream ends here — </tool_call> was the stop token, never streamed.
    // flush() must NOT emit a fresh `ToolCall` (which would carry a new id);
    // it should emit ToolCallDelta + ToolCallEnd against the existing header.
    let flushed = det.flush();
    let starts_after: Vec<_> = flushed
        .iter()
        .filter(|o| matches!(o, DetectorOutput::ToolCallStart { .. }))
        .collect();
    let complete_after: Vec<_> = flushed
        .iter()
        .filter(|o| matches!(o, DetectorOutput::ToolCall(_, _)))
        .collect();
    assert!(
        starts_after.is_empty(),
        "flush() must not re-emit ToolCallStart when incremental start already happened"
    );
    assert!(
        complete_after.is_empty(),
        "flush() must not emit ToolCall (which carries a fresh id) — emit Delta+End instead"
    );

    // Must include the args delta + end so the client gets the complete call.
    let has_delta = flushed
        .iter()
        .any(|o| matches!(o, DetectorOutput::ToolCallDelta { .. }));
    let has_end = flushed
        .iter()
        .any(|o| matches!(o, DetectorOutput::ToolCallEnd { .. }));
    assert!(has_delta, "flush() must emit ToolCallDelta with the args");
    assert!(has_end, "flush() must emit ToolCallEnd to close the call");

    // The args delta should carry the parsed arguments (JSON), not empty.
    let args = flushed
        .iter()
        .find_map(|o| match o {
            DetectorOutput::ToolCallDelta { args, .. } => Some(args.clone()),
            _ => None,
        })
        .unwrap();
    assert!(
        args.contains("meeting"),
        "args should contain 'meeting'; got: {args}"
    );
}

#[test]
fn extract_streaming_name_is_prefix_anchored_per_envelope() {
    // Hermes JSON: never match `"name"` nested inside `"arguments"`.
    assert_eq!(
        extract_streaming_name(
            r#"{"arguments": {"name": "Alice"}"#,
            EnvelopeKind::QwenOrHermes
        ),
        None
    );
    assert_eq!(
        extract_streaming_name(
            r#"{"arguments": {"name": "Alice"}, "name": "create_user"}"#,
            EnvelopeKind::QwenOrHermes
        ),
        Some("create_user".to_string())
    );
    // Qwen XML: ignore `call:`, `[TOOL_CALLS]`, and `"name"` inside arguments.
    assert_eq!(
        extract_streaming_name(
            r#"<function=write><parameter=code>call:fake{x:1}[TOOL_CALLS]fake[ARGS]{"name":"Alice"}</parameter>"#,
            EnvelopeKind::QwenOrHermes
        ),
        Some("write".to_string())
    );
    // MiniMax envelopes never spoof from inner `"name"` or `call:` text.
    assert_eq!(
        extract_streaming_name(
            r#"<invoke name="write"><parameter name="content">{"name":"Alice"} call:fake{</parameter>"#,
            EnvelopeKind::MinimaxCanonical
        ),
        None
    );
    // Gemma-4: anchored to `call:`/`_call:` at the start of the envelope.
    assert_eq!(
        extract_streaming_name(
            r#"  call:get_weather{city:<|"|>Paris<|"|>}"#,
            EnvelopeKind::Gemma4
        ),
        Some("get_weather".to_string())
    );
    assert_eq!(
        extract_streaming_name(r#"{note:<|"|>call:fake{<|"|>}"#, EnvelopeKind::Gemma4),
        None
    );
}

#[test]
fn streaming_detector_hermes_reversed_keys_never_spoofs_inner_name() {
    let mut det = StreamingToolDetector::new();
    let first = det.process(r#"<tool_call>{"arguments": {"name": "Alice"},"#);
    assert!(
        !first
            .iter()
            .any(|o| matches!(o, DetectorOutput::ToolCallStart { .. })),
        "nested argument key must not trigger ToolCallStart"
    );
    let second = det.process(r#" "name": "create_user"}</tool_call>"#);
    let tc = second
        .iter()
        .find_map(|o| match o {
            DetectorOutput::ToolCall(tc, 0) => Some(tc.clone()),
            _ => None,
        })
        .expect("complete ToolCall expected on close");
    assert_eq!(tc.function.name, "create_user");
    let args: serde_json::Value = serde_json::from_str(&tc.function.arguments).unwrap();
    assert_eq!(args["name"], "Alice");
}

#[test]
fn streaming_detector_cleans_up_staged_start_when_body_fails_to_parse() {
    let mut det = StreamingToolDetector::new();
    let first = det.process(r#"<tool_call>{"name": "ghost""#);
    assert!(first.iter().any(
        |o| matches!(o, DetectorOutput::ToolCallStart { name, idx: 0, .. } if name == "ghost")
    ));
    let second = det.process(" invalid trailing garbage</tool_call>");
    assert!(
        !second
            .iter()
            .any(|o| matches!(o, DetectorOutput::ToolCallDelta { .. })),
        "unparseable body must not emit ToolCallDelta"
    );
    assert!(
        second
            .iter()
            .any(|o| matches!(o, DetectorOutput::ToolCallEnd { idx: 0 })),
        "unparseable body after ToolCallStart must emit ToolCallEnd to clear staged state"
    );
    assert!(!det.has_tool_calls());
}

#[test]
fn streaming_detector_ignores_literal_close_tag_inside_qwen_parameter() {
    let mut det = StreamingToolDetector::new();
    let mut events = Vec::new();
    for chunk in [
        "<tool_call>\n<function=write>\n<parameter=code>",
        "if s.contains(\"</tool_call>\") { ok }",
        "</parameter>\n</function>\n</tool_call>",
    ] {
        events.extend(det.process(chunk));
    }
    events.extend(det.flush());
    let args_json = events
        .iter()
        .find_map(|o| match o {
            DetectorOutput::ToolCallDelta { args, idx: 0 } => Some(args.clone()),
            _ => None,
        })
        .expect("expected ToolCallDelta");
    let args: serde_json::Value = serde_json::from_str(&args_json).unwrap();
    assert_eq!(args["code"], r#"if s.contains("</tool_call>") { ok }"#);
}

#[test]
fn streaming_detector_matches_only_active_envelope_close_tag() {
    // Literal `</tool_call>` and `<tool_call|>` inside a MiniMax envelope must not close it.
    let mut mm = StreamingToolDetector::new();
    let mm_events = mm.process(
        "<minimax:tool_call><invoke name=\"write\">\
         <parameter name=\"code\"></tool_call><tool_call|></parameter>\
         </invoke></minimax:tool_call>",
    );
    let mm_tc = mm_events
        .iter()
        .find_map(|o| match o {
            DetectorOutput::ToolCall(tc, 0) => Some(tc.clone()),
            _ => None,
        })
        .expect("expected MiniMax ToolCall");
    let mm_args: serde_json::Value = serde_json::from_str(&mm_tc.function.arguments).unwrap();
    assert_eq!(mm_args["code"], "</tool_call><tool_call|>");

    // Literal `</tool_call>` inside a Gemma-4 envelope must not close it.
    let mut g4 = StreamingToolDetector::new();
    let g4_events = g4.process(
        r#"<|tool_call>call:write{code:<|"|></tool_call></minimax:tool_call><|"|>}<tool_call|>"#,
    );
    let g4_tc = g4_events
        .iter()
        .find_map(|o| match o {
            DetectorOutput::ToolCall(tc, 0) => Some(tc.clone()),
            _ => None,
        })
        .expect("expected Gemma-4 ToolCall");
    let g4_args: serde_json::Value = serde_json::from_str(&g4_tc.function.arguments).unwrap();
    assert_eq!(g4_args["code"], "</tool_call></minimax:tool_call>");
}

#[test]
fn streaming_detector_parses_multiple_functions_in_one_envelope() {
    // Split after the first function header so call 0 uses staged Start/Delta/End
    // and call 1 emits as a complete ToolCall at index 1.
    let mut det = StreamingToolDetector::new();
    let first = det.process("<tool_call><function=read>");
    assert!(first.iter().any(
        |o| matches!(o, DetectorOutput::ToolCallStart { name, idx: 0, .. } if name == "read")
    ));
    let second = det.process(
        "<parameter=path>a.rs</parameter></function>\n\
         <function=read><parameter=path>b.rs</parameter></function></tool_call>",
    );
    let delta0 = second.iter().find_map(|o| match o {
        DetectorOutput::ToolCallDelta { args, idx: 0 } => Some(args.clone()),
        _ => None,
    });
    let call1 = second.iter().find_map(|o| match o {
        DetectorOutput::ToolCall(tc, 1) => Some(tc.clone()),
        _ => None,
    });
    assert!(delta0.as_deref().is_some_and(|a| a.contains("a.rs")));
    let call1 = call1.expect("second sibling function must be emitted at idx 1");
    assert_eq!(call1.function.name, "read");
    assert!(call1.function.arguments.contains("b.rs"));
}

#[test]
fn streaming_detector_does_not_hold_back_python_function_repr() {
    let mut det = StreamingToolDetector::new();
    let out = det.process("Result: <function foo at 0x7f123456> done");
    let emitted: String = out
        .iter()
        .filter_map(|o| match o {
            DetectorOutput::Content(s) => Some(s.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(emitted, "Result: <function foo at 0x7f123456> done");
    assert!(det.flush().is_empty());

    // Split across `<function` boundary: chunk 1 holds back the ambiguous prefix,
    // chunk 2 immediately flushes as ordinary content once ` foo` disambiguates it.
    let mut split_det = StreamingToolDetector::new();
    let c1 = split_det.process("Result: <function");
    let c2 = split_det.process(" foo at 0x7f123456> done");
    let combined: String = c1
        .into_iter()
        .chain(c2)
        .filter_map(|o| match o {
            DetectorOutput::Content(s) => Some(s),
            _ => None,
        })
        .collect();
    assert_eq!(combined, "Result: <function foo at 0x7f123456> done");
    assert!(split_det.flush().is_empty());

    // Real bare `<function name="exec">` still triggers bare-function parsing.
    let mut bare_det = StreamingToolDetector::new();
    let b1 = bare_det.process("Running <function name=");
    let b2 = bare_det.process("\"exec\"><parameter=command>ls</parameter></function>");
    let calls: Vec<_> = b1
        .into_iter()
        .chain(b2)
        .filter_map(|o| match o {
            DetectorOutput::ToolCall(tc, _) => Some(tc),
            _ => None,
        })
        .collect();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].function.name, "exec");
}
