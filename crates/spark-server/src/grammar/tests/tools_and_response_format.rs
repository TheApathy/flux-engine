// SPDX-License-Identifier: AGPL-3.0-only

//! Regression tests for joint `tools + response_format` XGrammar union
//! enforcement (`tool_choice="auto"` with both `tools` and `response_format`).
//!
//! Pins the Category O (`TC-65..TC-69`) failure mode where models wrapped a
//! valid JSON final answer in ` ```json ... ``` ` markdown fences or appended
//! commentary after `}` because `response_format` was previously dropped
//! whenever `tools` were active.

use super::*;
use crate::api::ResponseFormatConstraint;
use xgrammar::{CompiledGrammar, GrammarMatcher};

fn grammar_accepts(compiled: &CompiledGrammar, input: &str) -> bool {
    let mut matcher =
        GrammarMatcher::new(compiled, None, true, -1).expect("GrammarMatcher::new failed");
    if !matcher.accept_string(input, false) {
        return false;
    }
    matcher.is_terminated()
}

fn weather_response_schema() -> String {
    serde_json::json!({
        "type": "object",
        "properties": {
            "location": {"type": "string"},
            "temperature_celsius": {"type": "number"},
            "summary": {"type": "string"}
        },
        "required": ["location", "temperature_celsius", "summary"],
        "additionalProperties": false
    })
    .to_string()
}

#[test]
fn qwen3_coder_joint_union_accepts_tool_calls_and_raw_schema_json() {
    let vocab = test_vocab();
    let stop_ids = vec![130i32];
    let mut engine = GrammarEngine::new(&vocab, &stop_ids).unwrap();
    let tools = test_tool_defs();
    let rf = ResponseFormatConstraint::JsonSchema {
        schema: weather_response_schema(),
        strict: true,
    };

    let compiled = engine
        .compile_qwen3_coder_tool_or_response_format_grammar(&tools, false, &rf)
        .expect("joint qwen3_coder + json_schema compile must succeed");

    // Turn 1: single XML tool call (canonical and with post-</think> leading newlines).
    let single_call = "<tool_call>\n<function=get_weather>\n<parameter=location>\nTokyo\n</parameter>\n</function>\n</tool_call>";
    assert!(
        grammar_accepts(&compiled, single_call),
        "canonical single tool call must be accepted"
    );
    assert!(
        grammar_accepts(&compiled, &format!("\n\n{single_call}\n")),
        "tool call with leading/trailing whitespace must be accepted"
    );

    // Turn 1 parallel: multiple tool calls when stop_after_first=false.
    let parallel_calls = format!("{single_call}\n{single_call}");
    assert!(
        grammar_accepts(&compiled, &parallel_calls),
        "parallel tool calls must be accepted when stop_after_first=false"
    );

    // Turn 2: raw schema-valid JSON (with or without leading/trailing newlines after </think>).
    let raw_json = r#"{"location":"Tokyo","temperature_celsius":28,"summary":"Sunny"}"#;
    assert!(
        grammar_accepts(&compiled, raw_json),
        "exact schema-valid JSON must be accepted"
    );
    assert!(
        grammar_accepts(&compiled, &format!("\n\n{raw_json}\n")),
        "schema-valid JSON with leading/trailing newlines must be accepted"
    );

    // Regression checks for TC-65..TC-69:
    // - Markdown-fenced schema-valid JSON is accepted (preventing greedy fallback
    //   into `<tool_call>` when top-1 post-</think> logit is "```"); the outer
    //   fence is stripped on the OpenAI wire by `JsonFenceStripper` /
    //   `strip_json_markdown_fence`.
    // - Schema violations inside fences, wrong fence languages, preamble prose,
    //   trailing commentary, and free prose MUST still be REJECTED.
    let fenced = format!("```json\n{raw_json}\n```");
    assert!(
        grammar_accepts(&compiled, &fenced),
        "markdown-fenced schema-valid JSON must be accepted in joint union"
    );
    assert!(
        grammar_accepts(&compiled, &format!("\n\n```\n{raw_json}\n```\n")),
        "unlabeled markdown-fenced schema-valid JSON must be accepted"
    );
    let fenced_extra = "```json\n{\"location\":\"Tokyo\",\"temperature_celsius\":28,\"summary\":\"Sunny\",\"extra\":true}\n```";
    assert!(
        !grammar_accepts(&compiled, fenced_extra),
        "markdown-fenced JSON violating additionalProperties=false must be REJECTED"
    );
    let fenced_trailing = format!("{fenced}\nHere is the weather!");
    assert!(
        !grammar_accepts(&compiled, &fenced_trailing),
        "trailing prose after closing fence must be REJECTED"
    );
    let fenced_python = format!("```python\n{raw_json}\n```");
    assert!(
        !grammar_accepts(&compiled, &fenced_python),
        "non-JSON markdown fence language must be REJECTED"
    );
    let with_preamble = format!("Here is the weather:\n{raw_json}");
    assert!(
        !grammar_accepts(&compiled, &with_preamble),
        "preamble prose before JSON must be REJECTED"
    );
    let with_trailing = format!("{raw_json}\nNote: data from get_weather.");
    assert!(
        !grammar_accepts(&compiled, &with_trailing),
        "trailing commentary after JSON must be REJECTED"
    );
    let missing_required = r#"{"location":"Tokyo","temperature_celsius":28}"#;
    assert!(
        !grammar_accepts(&compiled, missing_required),
        "JSON missing required field must be REJECTED"
    );
    let extra_prop =
        r#"{"location":"Tokyo","temperature_celsius":28,"summary":"Sunny","extra":true}"#;
    assert!(
        !grammar_accepts(&compiled, extra_prop),
        "JSON violating additionalProperties=false must be REJECTED"
    );
    let plain_prose = "The weather in Tokyo is 28 degrees and sunny.";
    assert!(
        !grammar_accepts(&compiled, plain_prose),
        "free prose must be REJECTED when response_format is active"
    );
}

#[test]
fn qwen3_coder_joint_union_respects_stop_after_first() {
    let vocab = test_vocab();
    let stop_ids = vec![130i32];
    let mut engine = GrammarEngine::new(&vocab, &stop_ids).unwrap();
    let tools = test_tool_defs();
    let rf = ResponseFormatConstraint::JsonObject;

    let compiled = engine
        .compile_qwen3_coder_tool_or_response_format_grammar(&tools, true, &rf)
        .expect("joint qwen3_coder + json_object compile must succeed");

    let single_call = "<tool_call>\n<function=get_weather>\n<parameter=location>\nParis\n</parameter>\n</function>\n</tool_call>";
    assert!(grammar_accepts(&compiled, single_call));
    assert!(
        !grammar_accepts(&compiled, &format!("{single_call}\n{single_call}")),
        "parallel calls must be rejected when stop_after_first=true"
    );
    assert!(grammar_accepts(&compiled, r#"{"ok":true,"items":[1,2,3]}"#));
    assert!(grammar_accepts(&compiled, "```json\n{\"ok\":true}\n```"));
    assert!(!grammar_accepts(
        &compiled,
        "```json\n{\"ok\":true}\n```\nDone!"
    ));
}

#[test]
fn hermes_and_bare_json_and_gemma4_and_minimax_joint_unions_work() {
    let mut vocab = test_vocab();
    vocab.push("<|tool_call>".to_string());
    vocab.push("<tool_call|>".to_string());
    vocab.push("<minimax:tool_call>".to_string());
    vocab.push("</minimax:tool_call>".to_string());
    let stop_ids = vec![130i32];
    let mut engine = GrammarEngine::new(&vocab, &stop_ids).unwrap();
    let tools = test_tool_defs();
    let rf = ResponseFormatConstraint::JsonSchema {
        schema: weather_response_schema(),
        strict: true,
    };
    let raw_json = r#"{"location":"Oslo","temperature_celsius":12,"summary":"Cloudy"}"#;
    let fenced = format!("```json\n{raw_json}\n```");
    let invalid_prose = format!("Sure!\n{raw_json}");

    // Hermes
    let hermes = engine
        .compile_hermes_tool_or_response_format_grammar(&tools, false, &rf)
        .expect("hermes joint compile");
    assert!(grammar_accepts(
        &hermes,
        r#"<tool_call>{"name":"get_weather","arguments":{"location":"Oslo"}}</tool_call>"#
    ));
    assert!(grammar_accepts(&hermes, raw_json));
    assert!(grammar_accepts(&hermes, &fenced));
    assert!(!grammar_accepts(&hermes, &invalid_prose));

    // BareJson
    let bare = engine
        .compile_bare_json_tool_or_response_format_grammar(&tools, false, &rf)
        .expect("bare_json joint compile");
    assert!(grammar_accepts(
        &bare,
        r#"{"name":"get_weather","arguments":{"location":"Oslo"}}"#
    ));
    assert!(grammar_accepts(&bare, raw_json));
    assert!(grammar_accepts(&bare, &fenced));
    assert!(!grammar_accepts(&bare, &invalid_prose));

    // Gemma4
    let gemma4 = engine
        .compile_gemma4_tool_or_response_format_grammar(&tools, false, &rf)
        .expect("gemma4 joint compile");
    assert!(grammar_accepts(
        &gemma4,
        r#"<|tool_call>call:get_weather{"location":"Oslo"}<tool_call|>"#
    ));
    assert!(grammar_accepts(&gemma4, raw_json));
    assert!(grammar_accepts(&gemma4, &fenced));
    assert!(!grammar_accepts(&gemma4, &invalid_prose));

    // MiniMaxXml
    let minimax = engine
        .compile_minimax_xml_tool_or_response_format_grammar(&tools, false, &rf)
        .expect("minimax joint compile");
    assert!(grammar_accepts(
        &minimax,
        "<minimax:tool_call>\n<invoke name=\"get_weather\">\n<parameter name=\"location\">Oslo</parameter>\n</invoke>\n</minimax:tool_call>"
    ));
    assert!(grammar_accepts(&minimax, raw_json));
    assert!(grammar_accepts(&minimax, &fenced));
    assert!(!grammar_accepts(&minimax, &invalid_prose));

    // GlmXml
    let mut glm_vocab = vocab.clone();
    glm_vocab.push("<arg_key>".to_string());
    glm_vocab.push("</arg_key>".to_string());
    glm_vocab.push("<arg_value>".to_string());
    glm_vocab.push("</arg_value>".to_string());
    let mut glm_engine = GrammarEngine::new(&glm_vocab, &stop_ids).unwrap();
    let glm = glm_engine
        .compile_glm_xml_tool_or_response_format_grammar(&tools, false, &rf)
        .expect("glm_xml joint compile");
    assert!(grammar_accepts(
        &glm,
        "<tool_call>get_weather<arg_key>location</arg_key><arg_value>Oslo</arg_value></tool_call>"
    ));
    assert!(grammar_accepts(&glm, raw_json));
    assert!(grammar_accepts(&glm, &fenced));
    assert!(!grammar_accepts(&glm, &invalid_prose));
}

#[test]
fn response_format_json_schema_respects_strict_false_and_sanitizes() {
    let vocab = test_vocab();
    let stop_ids = vec![130i32];
    let mut engine = GrammarEngine::new(&vocab, &stop_ids).unwrap();
    let tools = test_tool_defs();

    // Schema without explicit `additionalProperties`:
    // strict=true forbids extra keys, strict=false permits them.
    let open_schema = serde_json::json!({
        "type": "object",
        "properties": {
            "name": {"type": "string"}
        },
        "required": ["name"]
    })
    .to_string();
    let with_extra = r#"{"name":"Alice","extra":42}"#;

    let strict_compiled = engine
        .compile_json_schema_with_strict(&open_schema, true)
        .expect("strict schema compile");
    assert!(!grammar_accepts(&strict_compiled, with_extra));

    let relaxed_compiled = engine
        .compile_json_schema_with_strict(&open_schema, false)
        .expect("non-strict schema compile");
    assert!(grammar_accepts(&relaxed_compiled, with_extra));

    let rf_relaxed = ResponseFormatConstraint::JsonSchema {
        schema: open_schema,
        strict: false,
    };
    let joint_relaxed = engine
        .compile_qwen3_coder_tool_or_response_format_grammar(&tools, false, &rf_relaxed)
        .expect("joint non-strict compile");
    assert!(grammar_accepts(&joint_relaxed, with_extra));

    // Schema requiring sanitization (`allOf`, empty `enum`, empty `anyOf`, `$ref`)
    // must compile cleanly in both standalone and joint-union paths.
    let dirty_schema = serde_json::json!({
        "$defs": {
            "Label": {"type": "string", "enum": []}
        },
        "type": "object",
        "allOf": [
            {"properties": {"label": {"$ref": "#/$defs/Label"}}, "required": ["label"]},
            {"properties": {"note": {"type": "string", "anyOf": []}}}
        ]
    })
    .to_string();
    let sanitized_standalone = engine
        .compile_json_schema_with_strict(&dirty_schema, false)
        .expect("sanitized standalone schema compile");
    assert!(grammar_accepts(
        &sanitized_standalone,
        r#"{"label":"ok","note":"hi"}"#
    ));

    let rf_dirty = ResponseFormatConstraint::JsonSchema {
        schema: dirty_schema,
        strict: false,
    };
    let sanitized_joint = engine
        .compile_qwen3_coder_tool_or_response_format_grammar(&tools, false, &rf_dirty)
        .expect("sanitized joint schema compile");
    assert!(grammar_accepts(
        &sanitized_joint,
        r#"{"label":"ok","note":"hi"}"#
    ));
}
