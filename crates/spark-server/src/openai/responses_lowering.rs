// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

#[derive(Debug)]
pub enum LowerResponsesError {
    /// Request body was malformed (bad `input` shape, etc.).
    BadRequest(String),
    /// `previous_response_id` was set but the store had no entry. Maps to
    /// `400 invalid_request_error, code=response_not_found`.
    PriorNotFound(String),
}

impl LowerResponsesError {
    // Live callers destructure the variants directly via `match`; kept as
    // a stable string-getter for any future consumer that needs the
    // message without inspecting the variant.
    #[allow(dead_code)]
    pub fn message(&self) -> &str {
        match self {
            Self::BadRequest(m) | Self::PriorNotFound(m) => m.as_str(),
        }
    }
}

/// Lower an OpenAI Responses-API request into a ChatCompletionRequest
/// so the existing chat-completions pipeline can satisfy it.
///
/// Prior-turn resume: when `previous_response_id` is set, look it up via
/// `resolve_prior` (the handler passes a closure that consults
/// [`crate::response_store::ResponseStore`]). The stored transcript is
/// prepended to the current input so the pipeline sees the full history.
/// Unknown / expired ids yield [`LowerResponsesError::PriorNotFound`].
pub fn lower_responses_to_chat(
    r: ResponsesRequest,
    resolve_prior: impl FnOnce(&str) -> Option<Vec<IncomingMessage>>,
) -> Result<ChatCompletionRequest, LowerResponsesError> {
    let mut messages: Vec<IncomingMessage> = Vec::new();

    // Prior-turn transcript (when resuming a conversation).
    if let Some(prior_id) = r.previous_response_id.as_deref() {
        match resolve_prior(prior_id) {
            Some(prior) => messages.extend(prior),
            None => {
                return Err(LowerResponsesError::PriorNotFound(format!(
                    "previous_response_id '{prior_id}' not found or expired"
                )));
            }
        }
    }

    if let Some(instr) = r.instructions.clone() {
        // Per the Responses spec, only the CURRENT turn's `instructions`
        // should apply. If we're resuming via previous_response_id, the
        // prior transcript may already contain a synthetic-system
        // message — drop it so the new instructions don't stack.
        // Also insert at position 0 so the chat template sees system
        // first (the conventional layout).
        messages.retain(|m| m.role != "system");
        messages.insert(0, IncomingMessage::synthetic_system(instr));
    }
    match &r.input {
        serde_json::Value::String(s) => {
            messages.push(IncomingMessage::synthetic_user_text(s.clone()));
        }
        serde_json::Value::Array(items) => {
            let input_start = messages.len();
            for it in items {
                if let Some(m) = IncomingMessage::try_from_responses_input_item(it)
                    .map_err(LowerResponsesError::BadRequest)?
                {
                    push_or_coalesce_responses_message(&mut messages, input_start, m);
                }
            }
        }
        _ => {
            return Err(LowerResponsesError::BadRequest(
                "`input` must be a string or array of input items".into(),
            ));
        }
    }

    // Tools pass: separate function tools (supported) from built-in
    // hosted tools (not supported on a self-hosted inference server).
    // Built-ins each get their own error message so the client knows
    // exactly what's missing; function tools are parsed into
    // `ToolDefinition` via serde round-trip.
    let tools: Option<Vec<crate::tool_parser::ToolDefinition>> = match r.tools {
        None => None,
        Some(list) => {
            let mut parsed = Vec::with_capacity(list.len());
            for raw in list {
                let ty = raw.get("type").and_then(|v| v.as_str()).unwrap_or("");
                match ty {
                    "function" | "" => {
                        // Responses API uses a flat `{type, name, description,
                        // parameters, strict}` shape for function tools, while
                        // chat-completions wraps the function fields inside a
                        // nested `function` object. Re-shape the flat form so
                        // the existing chat-format ToolDefinition deserializer
                        // accepts both.
                        let normalized = if raw.get("function").is_some() {
                            raw
                        } else if let Some(obj) = raw.as_object() {
                            let mut function = serde_json::Map::new();
                            for key in ["name", "description", "parameters", "strict"] {
                                if let Some(v) = obj.get(key) {
                                    function.insert(key.to_string(), v.clone());
                                }
                            }
                            serde_json::json!({
                                "type": "function",
                                "function": serde_json::Value::Object(function),
                            })
                        } else {
                            raw
                        };
                        match serde_json::from_value::<crate::tool_parser::ToolDefinition>(
                            normalized,
                        ) {
                            Ok(td) => parsed.push(td),
                            Err(e) => {
                                return Err(LowerResponsesError::BadRequest(format!(
                                    "invalid tool definition: {e}"
                                )));
                            }
                        }
                    }
                    builtin @ ("web_search"
                    | "web_search_preview"
                    | "file_search"
                    | "computer_use_preview"
                    | "code_interpreter"
                    | "image_generation"
                    | "mcp"
                    | "local_shell"
                    | "custom_tool") => {
                        return Err(LowerResponsesError::BadRequest(format!(
                            "built-in tool '{builtin}' is not supported by this server. Flux serves inference only and does not ship hosted tools (web search, file search, code interpreter, computer use, image generation, MCP). Provide your own `function`-type tools instead."
                        )));
                    }
                    other => {
                        return Err(LowerResponsesError::BadRequest(format!(
                            "unknown tool type '{other}'. Supported types: 'function'."
                        )));
                    }
                }
            }
            if parsed.is_empty() {
                None
            } else {
                Some(parsed)
            }
        }
    };
    let response_format = lower_responses_text_format(r.text.as_ref())?;
    Ok(ChatCompletionRequest {
        model: r.model,
        messages,
        max_tokens: r.max_output_tokens.unwrap_or_else(default_max_tokens),
        max_completion_tokens: None,
        max_tokens_defaulted: false,
        runtime_defaults: None,
        temperature: r.temperature,
        top_k: None,
        top_p: r.top_p,
        top_n_sigma: None,
        min_p: None,
        repetition_penalty: None,
        presence_penalty: None,
        frequency_penalty: None,
        logit_bias: None,
        stream: r.stream,
        // Responses API has no token-IDs knob; lowered requests keep
        // the default off (PCND — no implicit behavior change).
        return_token_ids: false,
        // The Responses API currently has no intervention-reporting knob.
        // Preserve its existing wire format and guard behavior.
        report_interventions: false,
        rendered_prompt_evidence_nonce: None,
        intervention_ledger: Default::default(),
        enable_thinking: false,
        thinking: None,
        thinking_token_budget: None,
        thinking_budget: None,
        thinking_reserve: None,
        reasoning: r.reasoning,
        chat_template_kwargs: None,
        tools,
        tool_choice: match r.tool_choice {
            None => None,
            Some(raw) => {
                let normalized = match raw.as_str() {
                    Some(_) => raw,
                    None => match raw.as_object() {
                        Some(obj) if obj.get("function").is_some() => raw,
                        Some(obj)
                            if obj.get("type").and_then(|v| v.as_str()) == Some("function")
                                && obj.get("name").is_some() =>
                        {
                            let mut function = serde_json::Map::new();
                            function.insert(
                                "name".to_string(),
                                obj.get("name").cloned().unwrap_or(serde_json::Value::Null),
                            );
                            serde_json::json!({
                                "type": "function",
                                "function": serde_json::Value::Object(function),
                            })
                        }
                        _ => raw,
                    },
                };
                match serde_json::from_value::<crate::tool_parser::ToolChoice>(normalized) {
                    Ok(tc) => Some(tc),
                    Err(e) => {
                        return Err(LowerResponsesError::BadRequest(format!(
                            "invalid tool_choice: {e}"
                        )));
                    }
                }
            }
        },
        stop: Vec::new(),
        response_format,
        min_tokens: 0,
        ignore_eos: false,
        seed: None,
        logprobs: None,
        top_logprobs: None,
        timeout: None,
        n: 1,
        stream_options: None,
        parallel_tool_calls: r.parallel_tool_calls,
        verbosity: None,
        service_tier: r.service_tier,
        store: r.store,
        metadata: r.metadata,
        safety_identifier: None,
        prompt_cache_key: None,
        user: None,
        modalities: None,
        audio: None,
        prediction: None,
        web_search_options: None,
        reasoning_effort: None,
        raw_body: None,
    })
}

// Canonical `uuid_v4` lives in `openai/mod.rs::uuid_v4`. A verbatim
// duplicate that lived here was dead code (no `responses_lowering`-local
// caller; all live callers reach the openai/mod one via
// `crate::openai::uuid_v4`).

/// Lower OpenAI Responses `text.format` (`{type: "text" | "json_object" | "json_schema"}`)
/// into the shared `ChatCompletionRequest.response_format` so `/v1/responses`
/// inherits strict XGrammar schema enforcement, joint `tools | response_format`
/// union support, and markdown-fence unwrapping.
fn lower_responses_text_format(
    text: Option<&serde_json::Value>,
) -> Result<Option<ResponseFormat>, LowerResponsesError> {
    let Some(text) = text else {
        return Ok(None);
    };
    let Some(text_obj) = text.as_object() else {
        return Err(LowerResponsesError::BadRequest(
            "`text` must be an object".into(),
        ));
    };
    let Some(fmt) = text_obj.get("format") else {
        return Ok(None);
    };
    if fmt.is_null() {
        return Ok(None);
    }
    let Some(fmt_obj) = fmt.as_object() else {
        return Err(LowerResponsesError::BadRequest(
            "`text.format` must be an object".into(),
        ));
    };
    let ty = fmt_obj.get("type").and_then(|v| v.as_str()).unwrap_or("");
    match ty {
        "text" => Ok(Some(ResponseFormat::Text)),
        "json_object" => Ok(Some(ResponseFormat::JsonObject)),
        "json_schema" => {
            // Responses API places `{name, description, schema, strict}` directly
            // on `text.format`; also accept the nested chat-completions form if sent.
            let spec_val = if let Some(nested) = fmt_obj.get("json_schema") {
                nested.clone()
            } else {
                let mut spec = serde_json::Map::new();
                let name = fmt_obj
                    .get("name")
                    .and_then(|v| v.as_str())
                    .filter(|s| !s.is_empty())
                    .unwrap_or("response");
                spec.insert(
                    "name".to_string(),
                    serde_json::Value::String(name.to_string()),
                );
                if let Some(desc) = fmt_obj.get("description") {
                    spec.insert("description".to_string(), desc.clone());
                }
                let Some(schema) = fmt_obj.get("schema") else {
                    return Err(LowerResponsesError::BadRequest(
                        "`text.format` with type 'json_schema' requires a `schema` field".into(),
                    ));
                };
                spec.insert("schema".to_string(), schema.clone());
                if let Some(strict) = fmt_obj.get("strict") {
                    spec.insert("strict".to_string(), strict.clone());
                }
                serde_json::Value::Object(spec)
            };
            let json_schema: JsonSchemaSpec = serde_json::from_value(spec_val).map_err(|e| {
                LowerResponsesError::BadRequest(format!("invalid `text.format` json_schema: {e}"))
            })?;
            Ok(Some(ResponseFormat::JsonSchema { json_schema }))
        }
        other => Err(LowerResponsesError::BadRequest(format!(
            "unsupported `text.format.type` '{other}'; expected 'text', 'json_object', or 'json_schema'"
        ))),
    }
}

/// Append a lowered Responses input item, coalescing consecutive
/// `function_call` items (and an immediately preceding assistant text
/// message in the same `input` array, if any) into a single assistant
/// [`IncomingMessage`] with multiple `tool_calls`.
///
/// Strict chat templates (e.g. `mistral.jinja`, `glm5_next.jinja`) reject
/// consecutive assistant turns, so parallel tool calls replayed as separate
/// `function_call` items must share one assistant turn.
fn push_or_coalesce_responses_message(
    messages: &mut Vec<IncomingMessage>,
    input_start: usize,
    mut msg: IncomingMessage,
) {
    if messages.len() > input_start
        && msg.role == "assistant"
        && msg.content.text.is_empty()
        && msg.content.images.is_empty()
        && let Some(new_calls) = msg.tool_calls.as_mut().filter(|c| !c.is_empty())
        && let Some(prev) = messages.last_mut()
        && prev.role == "assistant"
    {
        prev.tool_calls.get_or_insert_default().append(new_calls);
        return;
    }
    messages.push(msg);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lower_input(input: serde_json::Value) -> Vec<IncomingMessage> {
        let req: ResponsesRequest = serde_json::from_value(serde_json::json!({
            "model": "test-model",
            "input": input,
        }))
        .expect("valid ResponsesRequest");
        lower_responses_to_chat(req, |_| None)
            .expect("lowering succeeds")
            .messages
    }

    #[test]
    fn coalesces_consecutive_function_calls_into_single_assistant_turn() {
        let msgs = lower_input(serde_json::json!([
            {"type": "message", "role": "user", "content": "Check Paris and Tokyo"},
            {"type": "function_call", "call_id": "call_1", "name": "weather", "arguments": "{\"city\":\"Paris\"}"},
            {"type": "function_call", "call_id": "call_2", "name": "weather", "arguments": "{\"city\":\"Tokyo\"}"},
            {"type": "function_call_output", "call_id": "call_1", "output": "18C"},
            {"type": "function_call_output", "call_id": "call_2", "output": "22C"},
        ]));

        assert_eq!(msgs.len(), 4);
        assert_eq!(msgs[0].role, "user");
        assert_eq!(msgs[1].role, "assistant");
        assert_eq!(msgs[1].content.text, "");
        let calls = msgs[1].tool_calls.as_ref().expect("coalesced tool_calls");
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].id.as_deref(), Some("call_1"));
        assert_eq!(calls[0].function.name, "weather");
        assert_eq!(calls[1].id.as_deref(), Some("call_2"));
        assert_eq!(calls[1].function.arguments, "{\"city\":\"Tokyo\"}");
        assert_eq!(msgs[2].role, "tool");
        assert_eq!(msgs[2].tool_call_id.as_deref(), Some("call_1"));
        assert_eq!(msgs[3].role, "tool");
        assert_eq!(msgs[3].tool_call_id.as_deref(), Some("call_2"));
    }

    #[test]
    fn coalesces_preceding_assistant_text_with_consecutive_function_calls() {
        let msgs = lower_input(serde_json::json!([
            {"type": "message", "role": "user", "content": "Check both"},
            {"type": "message", "role": "assistant", "content": "Looking up both cities."},
            {"type": "function_call", "call_id": "call_a", "name": "weather", "arguments": "{\"city\":\"Paris\"}"},
            {"type": "function_call", "call_id": "call_b", "name": "weather", "arguments": "{\"city\":\"Tokyo\"}"},
            {"type": "function_call_output", "call_id": "call_a", "output": "18C"},
            {"type": "function_call_output", "call_id": "call_b", "output": "22C"},
        ]));

        assert_eq!(msgs.len(), 4);
        assert_eq!(msgs[1].role, "assistant");
        assert_eq!(msgs[1].content.text, "Looking up both cities.");
        let calls = msgs[1]
            .tool_calls
            .as_ref()
            .expect("tool_calls merged onto assistant text message");
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].id.as_deref(), Some("call_a"));
        assert_eq!(calls[1].id.as_deref(), Some("call_b"));
    }

    #[test]
    fn keeps_non_consecutive_function_calls_separate_across_tool_outputs() {
        let msgs = lower_input(serde_json::json!([
            {"type": "message", "role": "user", "content": "Step by step"},
            {"type": "function_call", "call_id": "call_1", "name": "step_one", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "call_1", "output": "done 1"},
            {"type": "function_call", "call_id": "call_2", "name": "step_two", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "call_2", "output": "done 2"},
        ]));

        assert_eq!(msgs.len(), 5);
        assert_eq!(msgs[1].role, "assistant");
        assert_eq!(msgs[1].tool_calls.as_ref().map(Vec::len), Some(1));
        assert_eq!(msgs[2].role, "tool");
        assert_eq!(msgs[3].role, "assistant");
        assert_eq!(msgs[3].tool_calls.as_ref().map(Vec::len), Some(1));
        assert_eq!(msgs[4].role, "tool");
    }

    #[test]
    fn does_not_mutate_prior_turn_assistant_message_on_resume() {
        let prior = vec![
            IncomingMessage::synthetic_user_text("Turn 1".to_string()),
            IncomingMessage {
                role: "assistant".to_string(),
                content: ParsedContent {
                    text: "Prior reply".to_string(),
                    images: Vec::new(),
                    image_text_offsets: Vec::new(),
                },
                reasoning_content: None,
                tool_calls: None,
                tool_call_id: None,
                name: None,
            },
        ];
        let req: ResponsesRequest = serde_json::from_value(serde_json::json!({
            "model": "test-model",
            "previous_response_id": "resp_prev",
            "input": [
                {"type": "function_call", "call_id": "call_1", "name": "a", "arguments": "{}"},
                {"type": "function_call", "call_id": "call_2", "name": "b", "arguments": "{}"},
            ],
        }))
        .expect("valid ResponsesRequest");

        let msgs = lower_responses_to_chat(req, |id| {
            assert_eq!(id, "resp_prev");
            Some(prior.clone())
        })
        .expect("lowering succeeds")
        .messages;

        assert_eq!(msgs.len(), 3);
        assert_eq!(msgs[1].role, "assistant");
        assert_eq!(msgs[1].content.text, "Prior reply");
        assert!(msgs[1].tool_calls.is_none());
        assert_eq!(msgs[2].role, "assistant");
        assert_eq!(msgs[2].tool_calls.as_ref().map(Vec::len), Some(2));
    }
}
