// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

fn url_of(a: &Annotation) -> (usize, usize, &str, &str) {
    match a {
        Annotation::UrlCitation {
            url_citation:
                UrlCitation {
                    start_index,
                    end_index,
                    url,
                    title,
                },
        } => (*start_index, *end_index, url.as_str(), title.as_str()),
    }
}

#[test]
fn bare_url_extracted() {
    let got = extract_url_annotations("see https://example.com/foo for more").unwrap();
    assert_eq!(got.len(), 1);
    let (s, e, u, t) = url_of(&got[0]);
    assert_eq!(u, "https://example.com/foo");
    assert_eq!(t, "https://example.com/foo");
    assert_eq!(s, 4);
    assert_eq!(e, 4 + "https://example.com/foo".len());
}

#[test]
fn trailing_sentence_punct_stripped() {
    let got = extract_url_annotations("go to https://example.com.").unwrap();
    let (_, _, u, _) = url_of(&got[0]);
    assert_eq!(u, "https://example.com");
}

#[test]
fn wikipedia_parens_preserved() {
    let got = extract_url_annotations("see https://en.wikipedia.org/wiki/Foo_(bar) now").unwrap();
    let (_, _, u, _) = url_of(&got[0]);
    assert_eq!(u, "https://en.wikipedia.org/wiki/Foo_(bar)");
}

#[test]
fn markdown_link_uses_title() {
    let got = extract_url_annotations("read [the docs](https://example.com/api) today").unwrap();
    assert_eq!(got.len(), 1);
    let (_, _, u, t) = url_of(&got[0]);
    assert_eq!(u, "https://example.com/api");
    assert_eq!(t, "the docs");
}

#[test]
fn url_in_fenced_code_skipped() {
    let input = "run this:\n```bash\ncurl https://example.com\n```\ndone";
    assert!(extract_url_annotations(input).is_none());
}

#[test]
fn url_in_inline_code_skipped() {
    let input = "use `curl https://example.com` to fetch";
    assert!(extract_url_annotations(input).is_none());
}

#[test]
fn multiple_urls_sorted_by_position() {
    let input = "first https://a.example.com and [second](https://b.example.com)";
    let got = extract_url_annotations(input).unwrap();
    assert_eq!(got.len(), 2);
    let (s0, _, u0, _) = url_of(&got[0]);
    let (s1, _, u1, _) = url_of(&got[1]);
    assert!(s0 < s1);
    assert_eq!(u0, "https://a.example.com");
    assert_eq!(u1, "https://b.example.com");
}

#[test]
fn non_http_ignored() {
    assert!(extract_url_annotations("ftp://example.com not a citation").is_none());
}

#[test]
fn empty_input_returns_none() {
    assert!(extract_url_annotations("").is_none());
    assert!(extract_url_annotations("no URLs here").is_none());
}

#[test]
fn query_and_fragment_preserved() {
    let got = extract_url_annotations("see https://example.com/p?q=1&r=2#frag here").unwrap();
    let (_, _, u, _) = url_of(&got[0]);
    assert_eq!(u, "https://example.com/p?q=1&r=2#frag");
}

#[test]
fn mask_code_spans_preserves_utf8_and_byte_offsets() {
    let input = "café `curl https://ignored.example/日本語` and ```\n🚀 https://fenced.example\n``` see https://real.example/ok";
    let masked = super::annotations::mask_code_spans(input);
    assert_eq!(masked.len(), input.len());
    assert_eq!(masked.matches('\n').count(), input.matches('\n').count());
    assert!(!masked.contains("ignored.example"));
    assert!(!masked.contains("fenced.example"));
    assert!(masked.contains("https://real.example/ok"));

    let got = extract_url_annotations(input).expect("real URL outside code spans");
    assert_eq!(got.len(), 1);
    let (s, e, u, _) = url_of(&got[0]);
    assert_eq!(u, "https://real.example/ok");
    assert_eq!(&input[s..e], "https://real.example/ok");
}

fn lower_with_tools(
    tools: serde_json::Value,
) -> Result<ChatCompletionRequest, LowerResponsesError> {
    let req: ResponsesRequest = serde_json::from_value(serde_json::json!({
        "model": "test-model",
        "input": "ping",
        "tools": tools,
    }))
    .expect("ResponsesRequest deserialize");
    lower_responses_to_chat(req, |_| None)
}

#[test]
fn responses_flat_function_tool_accepted() {
    // OpenAI's official Python SDK sends function tools in the flat
    // shape `{type, name, description, parameters}` — no nested
    // `function` object. Atlas must accept both shapes.
    let chat = lower_with_tools(serde_json::json!([
        {
            "type": "function",
            "name": "get_weather",
            "description": "look up weather",
            "parameters": {"type": "object", "properties": {"loc": {"type": "string"}}, "required": ["loc"]}
        }
    ])).expect("flat-form function tool should parse");
    let tools = chat.tools.expect("tools present");
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].tool_type, "function");
    assert_eq!(tools[0].function.name, "get_weather");
}

#[test]
fn responses_nested_function_tool_still_accepted() {
    // Backwards-compat: chat-completions-style `{type, function:{...}}`
    // must keep working since older clients send it.
    let chat = lower_with_tools(serde_json::json!([
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "parameters": {"type": "object"}
            }
        }
    ]))
    .expect("nested-form function tool should parse");
    let tools = chat.tools.expect("tools present");
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].function.name, "get_weather");
}

#[test]
fn responses_flat_tool_choice_accepted() {
    let req: ResponsesRequest = serde_json::from_value(serde_json::json!({
        "model": "test",
        "input": "ping",
        "tool_choice": {"type": "function", "name": "get_weather"},
    }))
    .unwrap();
    let chat = lower_responses_to_chat(req, |_| None).expect("flat tool_choice");
    match chat.tool_choice {
        Some(crate::tool_parser::ToolChoice::Specific { function }) => {
            assert_eq!(function.name, "get_weather");
        }
        other => panic!("expected Specific tool_choice, got {other:?}"),
    }
}

#[test]
fn responses_string_tool_choice_accepted() {
    let req: ResponsesRequest = serde_json::from_value(serde_json::json!({
        "model": "test",
        "input": "ping",
        "tool_choice": "required",
    }))
    .unwrap();
    let chat = lower_responses_to_chat(req, |_| None).expect("string tool_choice");
    match chat.tool_choice {
        Some(crate::tool_parser::ToolChoice::Mode(s)) => {
            assert_eq!(s, "required");
        }
        other => panic!("expected Mode tool_choice, got {other:?}"),
    }
}

#[test]
fn responses_parallel_tool_calls_false_reaches_chat_pipeline() {
    let req: ResponsesRequest = serde_json::from_value(serde_json::json!({
        "model": "test",
        "input": "ping",
        "parallel_tool_calls": false,
    }))
    .unwrap();
    let chat = lower_responses_to_chat(req, |_| None).expect("responses lowering");
    assert_eq!(chat.parallel_tool_calls, Some(false));
}

#[test]
fn responses_text_format_lowers_to_chat_response_format() {
    let schema = serde_json::json!({
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
        "additionalProperties": false
    });
    let req: ResponsesRequest = serde_json::from_value(serde_json::json!({
        "model": "test",
        "input": "Return JSON",
        "text": {
            "format": {
                "type": "json_schema",
                "name": "reply",
                "strict": true,
                "schema": schema,
            }
        }
    }))
    .unwrap();
    let chat = lower_responses_to_chat(req, |_| None).expect("json_schema lowering");
    match chat.response_format {
        Some(ResponseFormat::JsonSchema { json_schema }) => {
            assert_eq!(json_schema.name, "reply");
            assert!(json_schema.strict);
            assert_eq!(
                json_schema.schema["required"],
                serde_json::json!(["answer"])
            );
        }
        other => panic!("expected JsonSchema response_format, got {other:?}"),
    }

    let req_obj: ResponsesRequest = serde_json::from_value(serde_json::json!({
        "model": "test",
        "input": "Return JSON",
        "text": {"format": {"type": "json_object"}}
    }))
    .unwrap();
    let chat_obj = lower_responses_to_chat(req_obj, |_| None).expect("json_object lowering");
    assert!(matches!(
        chat_obj.response_format,
        Some(ResponseFormat::JsonObject)
    ));

    let req_bad: ResponsesRequest = serde_json::from_value(serde_json::json!({
        "model": "test",
        "input": "Return JSON",
        "text": {"format": {"type": "json_schema", "name": "missing_schema"}}
    }))
    .unwrap();
    assert!(matches!(
        lower_responses_to_chat(req_bad, |_| None),
        Err(LowerResponsesError::BadRequest(_))
    ));
}

#[test]
fn top_level_reasoning_effort_is_explicit_and_template_visible() {
    let req: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
        "model": "test",
        "messages": [{"role": "user", "content": "solve it"}],
        "reasoning_effort": "max"
    }))
    .unwrap();
    assert!(req.thinking_explicitly_requested());
    assert_eq!(req.requested_reasoning_effort(), Some("max"));
    assert_eq!(req.resolve_thinking(false), (true, Some(1024)));
}

#[test]
fn preserve_thinking_template_kwarg_stays_tristate() {
    let absent = ChatTemplateKwargs::from_json(r#"{"enable_thinking":true}"#).unwrap();
    assert_eq!(absent.preserve_thinking, None);
    let disabled = ChatTemplateKwargs::from_json(r#"{"preserve_thinking":false}"#).unwrap();
    assert_eq!(disabled.preserve_thinking, Some(false));
}

#[test]
fn markdown_link_with_parens_in_url_preserved() {
    // Wikipedia URLs contain `(...)` which the bare `find(')')` would
    // truncate. Verify the balanced-paren scan keeps the full URL.
    let got =
        extract_url_annotations("see [Foo (bar)](https://en.wikipedia.org/wiki/Foo_(bar)) here")
            .unwrap();
    assert_eq!(got.len(), 1);
    let (_, _, u, t) = url_of(&got[0]);
    assert_eq!(u, "https://en.wikipedia.org/wiki/Foo_(bar)");
    assert_eq!(t, "Foo (bar)");
}

#[test]
fn responses_in_progress_event_name() {
    let ev = ResponsesStreamEvent::InProgress {
        sequence_number: 1,
        response: ResponsesStreamEnvelope {
            id: "resp_test".into(),
            object: "response",
            created_at: 0,
            model: "m".into(),
            status: "in_progress",
            metadata: None,
        },
    };
    assert_eq!(responses_event_name(&ev), "response.in_progress");
}

// ── ChatTemplateKwargs ────────────────────────────────────────────

#[test]
fn chat_template_kwargs_parse() {
    let kw = ChatTemplateKwargs::from_json(r#"{"enable_thinking":true,"thinking_budget":1024}"#)
        .expect("should parse");
    assert_eq!(kw.enable_thinking, Some(true));
    assert_eq!(kw.thinking_budget, Some(1024));

    assert!(ChatTemplateKwargs::from_json("").is_none());
}

fn empty_chat_request() -> ChatCompletionRequest {
    serde_json::from_value(serde_json::json!({
        "model": "test",
        "messages": [{"role": "user", "content": "hi"}],
    }))
    .expect("valid chat request")
}

#[test]
fn rendered_prompt_evidence_nonce_is_explicit_and_strict() {
    let mut req = empty_chat_request();
    assert!(req.validate_rendered_prompt_evidence_nonce().is_ok());

    req.rendered_prompt_evidence_nonce = Some("1".repeat(32));
    assert!(req.validate_rendered_prompt_evidence_nonce().is_err());
    req.report_interventions = true;
    assert!(req.validate_rendered_prompt_evidence_nonce().is_ok());

    for invalid in ["1".repeat(31), "A".repeat(32), "g".repeat(32)] {
        req.rendered_prompt_evidence_nonce = Some(invalid);
        assert!(req.validate_rendered_prompt_evidence_nonce().is_err());
    }
}

#[test]
fn model_default_thinking_defers_to_the_model_budget() {
    let req = empty_chat_request();
    assert_eq!(req.resolve_thinking(true), (true, None));
    assert_eq!(req.resolve_thinking(false), (false, None));
}

#[test]
fn server_default_merged_when_request_silent() {
    let mut req = empty_chat_request();
    assert!(req.chat_template_kwargs.is_none());

    let server_kw = ChatTemplateKwargs {
        enable_thinking: Some(true),
        thinking_budget: None,
        preserve_thinking: None,
    };
    if !req.thinking_explicitly_requested() {
        req.chat_template_kwargs = Some(server_kw);
    }
    assert!(req.chat_template_kwargs.is_some());

    let (enabled, budget) = req.resolve_thinking(false);
    assert!(enabled);
    assert_eq!(budget, None);
}

#[test]
fn server_default_not_merged_when_request_explicit() {
    let mut req: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
        "model": "test",
        "messages": [{"role": "user", "content": "hi"}],
        "enable_thinking": true,
    }))
    .expect("valid chat request");
    assert!(req.thinking_explicitly_requested());

    let server_kw = ChatTemplateKwargs {
        enable_thinking: Some(false),
        thinking_budget: None,
        preserve_thinking: None,
    };
    if !req.thinking_explicitly_requested() {
        req.chat_template_kwargs = Some(server_kw);
    }
    assert!(req.chat_template_kwargs.is_none());
    assert!(req.resolve_thinking(false).0);
}

/// DeepSeek-V4.1 sends a numeric effort (`"reasoning_effort": 90`), which the
/// Python server accepts. The typed parse used to reject it with "Invalid
/// request JSON" before the model's resolver ran (serve2, request 04).
#[test]
fn integer_reasoning_effort_parses() {
    let body =
        || serde_json::json!({"model": "m", "messages": [{"role": "user", "content": "hi"}]});
    let mut b = body();
    b["reasoning_effort"] = serde_json::json!(90);
    let req: ChatCompletionRequest = serde_json::from_value(b).unwrap();
    assert_eq!(req.reasoning_effort, Some(ReasoningEffort::Integer(90)));
    assert_eq!(req.requested_reasoning_effort_int(), Some(90));
    let mut b = body();
    b["reasoning"] = serde_json::json!({"effort": 33});
    assert!(serde_json::from_value::<ChatCompletionRequest>(b).is_ok());
    // out-of-range integers (< 1 or > 100) are rejected cleanly
    for out_of_range in [0, -1, 101] {
        let mut b = body();
        b["reasoning_effort"] = serde_json::json!(out_of_range);
        assert!(serde_json::from_value::<ChatCompletionRequest>(b).is_err());
    }
    // strings are unchanged
    let mut b = body();
    b["reasoning_effort"] = serde_json::json!("high");
    let req: ChatCompletionRequest = serde_json::from_value(b).unwrap();
    assert_eq!(
        req.reasoning_effort,
        Some(ReasoningEffort::Named("high".into()))
    );
}

/// CONTROL: other types are still refused, so the widening is not "anything goes".
#[test]
fn non_string_non_integer_effort_is_refused() {
    for bad in [
        serde_json::json!(true),
        serde_json::json!(0.5),
        serde_json::json!({"x": 1}),
    ] {
        let mut b =
            serde_json::json!({"model": "m", "messages": [{"role": "user", "content": "hi"}]});
        b["reasoning_effort"] = bad.clone();
        assert!(
            serde_json::from_value::<ChatCompletionRequest>(b).is_err(),
            "{bad} accepted"
        );
    }
}

/// A chat request with no `model` parses (production accepts it and ignores the field).
#[test]
fn missing_model_field_parses() {
    let body = serde_json::json!({"messages": [{"role": "user", "content": "hi"}]});
    let req: ChatCompletionRequest = serde_json::from_value(body).expect("no model field");
    assert_eq!(req.model, "");
    // CONTROL: a malformed field is still an error
    assert!(
        serde_json::from_value::<ChatCompletionRequest>(serde_json::json!({"messages": 5}))
            .is_err()
    );
}

#[test]
fn ignore_eos_parses_and_leaves_min_tokens_alone() {
    let mut body = serde_json::json!({
        "model": "test",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 2048,
    });
    let req: ChatCompletionRequest =
        serde_json::from_value(body.clone()).expect("valid chat request");
    assert!(!req.ignore_eos, "an absent ignore_eos keeps EOS a stop");

    body["ignore_eos"] = serde_json::json!(true);
    let req: ChatCompletionRequest = serde_json::from_value(body).expect("valid chat request");
    assert!(req.ignore_eos);
    assert_eq!(req.max_tokens, 2048);
    assert_eq!(
        req.min_tokens, 0,
        "min_tokens gates the speculative fast paths, so ignore_eos must not set it"
    );

    let completion: CompletionRequest = serde_json::from_value(serde_json::json!({
        "prompt": "hi",
        "max_tokens": 16,
        "ignore_eos": true,
    }))
    .expect("valid completion request");
    assert!(completion.ignore_eos);
}

#[test]
fn body_probe_detects_both_max_tokens_spellings() {
    assert!(body_sets_max_tokens(br#"{"messages":[],"max_tokens":100}"#));
    assert!(body_sets_max_tokens(
        br#"{"messages":[],"max_completion_tokens":100}"#
    ));
    assert!(!body_sets_max_tokens(br#"{"messages":[],"temperature":0}"#));
    // A body the request parser will reject never takes the default path.
    assert!(body_sets_max_tokens(b"not json"));
}

#[test]
fn server_default_applies_only_when_the_client_omits_max_tokens() {
    let parse = |body: &str| -> ChatCompletionRequest {
        let mut req: ChatCompletionRequest = serde_json::from_str(body).unwrap();
        req.apply_default_max_tokens(body_sets_max_tokens(body.as_bytes()), Some(32768));
        req
    };
    let omitted = parse(r#"{"messages":[{"role":"user","content":"hi"}]}"#);
    assert_eq!(
        (omitted.max_tokens, omitted.max_tokens_defaulted),
        (32768, true)
    );
    for body in [
        r#"{"messages":[{"role":"user","content":"hi"}],"max_tokens":4096}"#,
        r#"{"messages":[{"role":"user","content":"hi"}],"max_completion_tokens":512}"#,
    ] {
        let explicit = parse(body);
        assert!(!explicit.max_tokens_defaulted, "{body}");
        assert_ne!(explicit.max_tokens, 32768, "{body}");
    }
}

#[test]
fn unset_server_default_keeps_the_4096_wire_default() {
    let body = r#"{"messages":[{"role":"user","content":"hi"}]}"#;
    let mut req: ChatCompletionRequest = serde_json::from_str(body).unwrap();
    req.apply_default_max_tokens(body_sets_max_tokens(body.as_bytes()), None);
    assert_eq!((req.max_tokens, req.max_tokens_defaulted), (4096, false));
}

#[test]
fn responses_default_applies_when_max_output_tokens_is_omitted() {
    let req: ResponsesRequest =
        serde_json::from_value(serde_json::json!({"model": "m", "input": "hi"})).unwrap();
    let set = req.max_output_tokens.is_some();
    let mut chat = lower_responses_to_chat(req, |_| None).unwrap();
    chat.apply_default_max_tokens(set, Some(32768));
    assert_eq!((chat.max_tokens, chat.max_tokens_defaulted), (32768, true));

    let req: ResponsesRequest = serde_json::from_value(
        serde_json::json!({"model": "m", "input": "hi", "max_output_tokens": 64}),
    )
    .unwrap();
    let set = req.max_output_tokens.is_some();
    let mut chat = lower_responses_to_chat(req, |_| None).unwrap();
    chat.apply_default_max_tokens(set, Some(32768));
    assert_eq!((chat.max_tokens, chat.max_tokens_defaulted), (64, false));
}

#[test]
fn completions_allowance_resolution() {
    let parse = |v: serde_json::Value| -> CompletionRequest { serde_json::from_value(v).unwrap() };
    let omitted = parse(serde_json::json!({"prompt": "hi"}));
    assert_eq!(omitted.max_tokens_or(Some(32768)), (32768, true));
    assert_eq!(omitted.max_tokens_or(None), (4096, false));
    let explicit = parse(serde_json::json!({"prompt": "hi", "max_tokens": 32}));
    assert_eq!(explicit.max_tokens_or(Some(32768)), (32, false));
}

#[test]
fn null_and_coexisting_max_tokens_fields_match_openai_spec() {
    let parse = |body: &str| -> ChatCompletionRequest {
        let mut req: ChatCompletionRequest = serde_json::from_str(body).unwrap();
        req.apply_default_max_tokens(body_sets_max_tokens(body.as_bytes()), Some(32768));
        req
    };

    assert!(!body_sets_max_tokens(
        br#"{"messages":[],"max_tokens":null}"#
    ));
    assert!(!body_sets_max_tokens(
        br#"{"messages":[],"max_completion_tokens":null}"#
    ));

    let null_mt = parse(r#"{"messages":[{"role":"user","content":"hi"}],"max_tokens":null}"#);
    assert_eq!(
        (null_mt.max_tokens, null_mt.max_tokens_defaulted),
        (32768, true)
    );

    let null_mct =
        parse(r#"{"messages":[{"role":"user","content":"hi"}],"max_completion_tokens":null}"#);
    assert_eq!(
        (null_mct.max_tokens, null_mct.max_tokens_defaulted),
        (32768, true)
    );

    let both = parse(
        r#"{"messages":[{"role":"user","content":"hi"}],"max_tokens":256,"max_completion_tokens":512}"#,
    );
    assert_eq!((both.max_tokens, both.max_tokens_defaulted), (512, false));
}

#[test]
fn incoming_message_preserves_reasoning_content_and_alias() {
    let a: IncomingMessage = serde_json::from_str(
        r#"{"role":"assistant","content":"42","reasoning_content":"step by step"}"#,
    )
    .unwrap();
    assert_eq!(a.reasoning_content.as_deref(), Some("step by step"));

    let b: IncomingMessage =
        serde_json::from_str(r#"{"role":"assistant","content":"42","reasoning":"aliased"}"#)
            .unwrap();
    assert_eq!(b.reasoning_content.as_deref(), Some("aliased"));
}
