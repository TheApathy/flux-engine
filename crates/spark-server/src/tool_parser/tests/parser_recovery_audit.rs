// SPDX-License-Identifier: AGPL-3.0-only

use super::super::*;

fn qwen_audit(input: &str) -> (Vec<ToolCall>, ToolParseAudit) {
    let (_, calls, audit) = parse_tool_calls_with_audit(input, "qwen3_coder");
    (calls, audit)
}

#[test]
fn blocking_canonical_qwen_has_no_recovery() {
    for input in [
        "prefix\n<tool_call>\n<function=exec>\n<parameter=command>ls</parameter>\n</function>\n</tool_call>",
        // A serving stop boundary may consume only the terminal outer close.
        "<tool_call>\n<function=exec>\n<parameter=command>ls</parameter>\n</function>",
        "<tool_call><function=clock></function></tool_call>",
        "<tool_call><function=a></function></tool_call>\n<tool_call><function=b></function></tool_call>",
    ] {
        let (calls, audit) = qwen_audit(input);
        assert!(
            !calls.is_empty(),
            "canonical fixture did not parse: {input:?}"
        );
        assert_eq!(audit.recovery_count, 0, "canonical fixture: {input:?}");
        assert!(!audit.recovered());
    }
}

#[test]
fn blocking_qwen_alternate_and_salvage_forms_are_audited() {
    let cases = [
        // Foreign inner format.
        r#"<tool_call>{"name":"exec","arguments":{"command":"ls"}}</tool_call>"#,
        // Alternate function openers and repaired names.
        "<tool_call><function exec><parameter=command>ls</parameter></function></tool_call>",
        "<tool_call><|function=exec><parameter=command>ls</parameter></function></tool_call>",
        "<tool_call><function name=\"exec\"><parameter=command>ls</parameter></function></tool_call>",
        "<tool_call><function=exec=exec><parameter=command>ls</parameter></function></tool_call>",
        // Missing inner closures are recovery even if the outer close exists.
        "<tool_call><function=exec><parameter=command>ls</function></tool_call>",
        "<tool_call><function=exec><parameter=command>ls</parameter></tool_call>",
        // Attribute drift and server-side argument deletion.
        "<tool_call><function=exec><parameter name=\"command\">ls</parameter></function></tool_call>",
        "<tool_call><function=exec><parameter=_think>because</parameter><parameter=command>ls</parameter></function></tool_call>",
        // Missing/alternate envelopes and the explicit salvage pipeline.
        "<function=exec><parameter=command>ls</parameter></function>",
        "<tools><function=exec><parameter=command>ls</parameter></function></tools>",
        r#"{"name":"exec","arguments":{"command":"ls"}}"#,
        "<parameter=exec><parameter=command>ls</parameter></function>",
        "[TOOL_CALLS]exec[ARGS]{\"command\":\"ls\"}",
        "<minimax:tool_call><invoke name=\"exec\"><parameter name=\"command\">ls</parameter></invoke></minimax:tool_call>",
    ];

    for input in cases {
        let (calls, audit) = qwen_audit(input);
        assert!(
            !calls.is_empty(),
            "recovery fixture did not parse: {input:?}"
        );
        assert!(audit.recovered(), "recovery was not audited: {input:?}");
        assert_eq!(audit.recovery_count, 1, "recovery fixture: {input:?}");
    }
}

#[test]
fn blocking_qwen_dropped_sibling_envelope_is_audited() {
    let input = "<tool_call><function=exec><parameter=command>ls</parameter></function></tool_call>\
                 <tool_call>unparseable body</tool_call>";
    let (calls, audit) = qwen_audit(input);
    assert_eq!(calls.len(), 1);
    assert_eq!(audit.recovery_count, 1);
}

#[test]
fn blocking_canonical_qwen_cannot_hide_foreign_siblings() {
    let canonical =
        "<tool_call><function=exec><parameter=command>ls</parameter></function></tool_call>";
    for foreign in [
        "[TOOL_CALLS]clock[ARGS]{\"zone\":\"UTC\"}",
        "<function=clock><parameter=zone>UTC</parameter></function>",
        "clock{\"zone\":\"UTC\"}",
    ] {
        for input in [
            format!("{canonical}{foreign}"),
            format!("{foreign}{canonical}"),
        ] {
            let (calls, audit) = qwen_audit(&input);
            assert_eq!(calls.len(), 1, "canonical call must still parse: {input:?}");
            assert_eq!(
                audit.recovery_count, 1,
                "foreign sibling was hidden by canonical call: {input:?}"
            );
        }
    }
}

#[test]
fn blocking_bare_call_audit_shares_the_parser_denylist() {
    let canonical =
        "<tool_call><function=exec><parameter=command>ls</parameter></function></tool_call>";
    for prose in [
        r#"object{"x":1}"#,
        r#"function{"x":1}"#,
        r#"params{"x":1}"#,
        r#"return{"x":1}"#,
        r#"const{"x":1}"#,
        r#"class{"x":1}"#,
    ] {
        for input in [format!("{canonical}{prose}"), format!("{prose}{canonical}")] {
            let (calls, audit) = qwen_audit(&input);
            assert_eq!(calls.len(), 1, "canonical call must still parse: {input:?}");
            assert_eq!(
                audit.recovery_count, 0,
                "parser-denied prose was falsely audited as a call: {input:?}"
            );
        }
    }
}

#[test]
fn blocking_qwen_only_malformed_envelope_is_audited_even_without_a_call() {
    let input = "prefix<tool_call>unparseable body</tool_call>suffix";
    let (calls, audit) = qwen_audit(input);
    assert!(calls.is_empty());
    assert_eq!(audit.recovery_count, 1);
}

#[test]
fn streaming_canonical_qwen_and_stop_stripped_outer_close_are_unflagged() {
    let mut closed = StreamingToolDetector::new_with_parser("qwen3_coder");
    let outputs = closed.process(
        "<tool_call><function=exec><parameter=command>ls</parameter></function></tool_call>",
    );
    assert!(
        outputs
            .iter()
            .any(|event| matches!(event, DetectorOutput::ToolCall(..)))
    );
    assert_eq!(closed.take_recovery_count(), 0);

    let mut stop_stripped = StreamingToolDetector::new_with_parser("qwen3_coder");
    let first = stop_stripped
        .process("<tool_call><function=exec><parameter=command>ls</parameter></function>");
    assert!(
        first
            .iter()
            .any(|event| matches!(event, DetectorOutput::ToolCallStart { .. }))
    );
    let tail = stop_stripped.flush();
    assert!(
        tail.iter()
            .any(|event| matches!(event, DetectorOutput::ToolCallDelta { .. }))
    );
    assert_eq!(stop_stripped.take_recovery_count(), 0);
}

#[test]
fn streaming_qwen_recoveries_and_drops_are_counted_and_drained() {
    let cases = [
        r#"<tool_call>{"name":"exec","arguments":{"command":"ls"}}</tool_call>"#,
        "<tool_call><function exec><parameter=command>ls</parameter></function></tool_call>",
        "<tool_call><function=exec><parameter=command>ls</function></tool_call>",
        "<tool_call>unparseable body</tool_call>",
        "<tool_call>unparseable body",
        "<tool_call>native_name<arg_key>x</arg_key></tool_call>",
        "<tool_call>native_name<arg_key>x</arg_key>",
        "<function=exec><parameter=command>ls</parameter></function>",
        "[TOOL_CALLS]exec[ARGS]{\"command\":\"ls\"}",
        "<minimax:_call><invoke name=\"exec\"><parameter name=\"command\">ls</parameter></invoke></minimax:_call>",
    ];

    for input in cases {
        let mut detector = StreamingToolDetector::new_with_parser("qwen3_coder");
        let _ = detector.process(input);
        let _ = detector.flush();
        assert!(
            detector.take_recovery_count() > 0,
            "stream recovery was not audited: {input:?}"
        );
        assert_eq!(
            detector.take_recovery_count(),
            0,
            "drain must be destructive"
        );
    }
}

#[test]
fn streaming_canonical_qwen_cannot_hide_fragmented_bare_json_siblings() {
    let canonical =
        "<tool_call><function=exec><parameter=command>ls</parameter></function></tool_call>";
    for input in [
        format!(r#"{canonical}clock{{"zone":"UTC"}}"#),
        format!(r#"clock{{"zone":"UTC"}}{canonical}"#),
    ] {
        let mut detector = StreamingToolDetector::new_with_parser("qwen3_coder");
        // Split every UTF-8-safe ASCII byte so the audit cannot depend on the
        // bare name and its JSON body arriving in a single token/chunk.
        for byte in input.as_bytes() {
            let _ = detector.process(std::str::from_utf8(&[*byte]).unwrap());
        }
        let _ = detector.flush();
        assert!(
            detector.take_recovery_count() > 0,
            "bare sibling escaped streaming recovery audit: {input:?}"
        );
    }
}

#[test]
fn streaming_bare_call_audit_shares_the_parser_denylist() {
    let canonical =
        "<tool_call><function=exec><parameter=command>ls</parameter></function></tool_call>";
    for input in [
        format!(r#"{canonical}object{{"x":1}}"#),
        format!(r#"object{{"x":1}}{canonical}"#),
    ] {
        let mut detector = StreamingToolDetector::new_with_parser("qwen3_coder");
        for chunk in input.as_bytes().chunks(3) {
            let _ = detector.process(std::str::from_utf8(chunk).unwrap());
        }
        let _ = detector.flush();
        assert_eq!(
            detector.take_recovery_count(),
            0,
            "parser-denied prose was falsely audited as a streaming call: {input:?}"
        );
    }
}

#[test]
fn legacy_detector_api_does_not_accumulate_provenance() {
    let mut detector = StreamingToolDetector::new();
    let _ =
        detector.process(r#"<tool_call>{"name":"exec","arguments":{"command":"ls"}}</tool_call>"#);
    assert_eq!(detector.take_recovery_count(), 0);
}
