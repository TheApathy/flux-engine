// SPDX-License-Identifier: AGPL-3.0-only

//! Anthropic-compatible `POST /v1/messages/count_tokens` endpoint.
//!
//! Unlike `/v1/messages`, Anthropic SDKs (`anthropic-sdk-python`,
//! `@anthropic-ai/sdk`, Claude Code) omit `max_tokens` when calling
//! `/v1/messages/count_tokens`. This dedicated handler accepts optional
//! `max_tokens` and `model`, renders `messages` + `system` + `tools` +
//! `thinking` through the active chat template, and returns
//! `{"input_tokens": N}`.

use axum::extract::rejection::JsonRejection;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Json, Response};
use serde::Deserialize;

use crate::main_modules::model_host::CurrentModel;

#[derive(Debug, Deserialize)]
pub struct CountTokensRequest {
    #[serde(default)]
    #[allow(dead_code)]
    pub model: Option<String>,
    #[serde(default)]
    #[allow(dead_code)]
    pub max_tokens: Option<usize>,
    #[serde(default)]
    pub system: Option<CountSystemContent>,
    pub messages: Vec<CountMessage>,
    #[serde(default)]
    pub tools: Option<Vec<CountTool>>,
    #[serde(default)]
    pub tool_choice: Option<serde_json::Value>,
    #[serde(default)]
    pub thinking: Option<CountThinkingConfig>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum CountSystemContent {
    Text(String),
    Blocks(Vec<CountSystemBlock>),
}

#[derive(Debug, Deserialize)]
pub struct CountSystemBlock {
    #[serde(rename = "type")]
    pub block_type: String,
    #[serde(default)]
    pub text: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct CountMessage {
    pub role: String,
    pub content: CountMessageContent,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum CountMessageContent {
    Text(String),
    Blocks(Vec<CountContentBlock>),
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
pub enum CountContentBlock {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        #[serde(default)]
        input: serde_json::Value,
    },
    #[serde(rename = "tool_result")]
    ToolResult {
        #[allow(dead_code)]
        tool_use_id: String,
        #[serde(default)]
        content: Option<CountToolResultContent>,
        #[serde(default)]
        is_error: Option<bool>,
    },
    #[serde(rename = "thinking")]
    Thinking {
        #[serde(default)]
        #[allow(dead_code)]
        thinking: Option<String>,
    },
    #[serde(rename = "image")]
    Image {},
    #[serde(other)]
    Unknown,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum CountToolResultContent {
    Text(String),
    Blocks(Vec<CountToolResultBlock>),
}

#[derive(Debug, Deserialize)]
pub struct CountToolResultBlock {
    #[serde(rename = "type")]
    pub block_type: String,
    #[serde(default)]
    pub text: Option<String>,
}

impl CountToolResultContent {
    fn contains_image(&self) -> bool {
        match self {
            Self::Text(_) => false,
            Self::Blocks(blocks) => blocks.iter().any(|b| b.block_type == "image"),
        }
    }

    fn to_text(&self) -> String {
        match self {
            Self::Text(s) => s.clone(),
            Self::Blocks(blocks) => blocks
                .iter()
                .filter(|b| b.block_type == "text")
                .filter_map(|b| b.text.clone())
                .collect::<Vec<_>>()
                .join("\n"),
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct CountTool {
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub input_schema: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
pub struct CountThinkingConfig {
    #[serde(rename = "type")]
    pub thinking_type: String,
    #[serde(default)]
    #[allow(dead_code)]
    pub budget_tokens: Option<u32>,
}

impl CountTokensRequest {
    #[must_use]
    pub fn contains_image(&self) -> bool {
        if matches!(&self.system, Some(CountSystemContent::Blocks(blocks))
            if blocks.iter().any(|b| b.block_type == "image"))
        {
            return true;
        }
        self.messages.iter().any(|m| match &m.content {
            CountMessageContent::Text(_) => false,
            CountMessageContent::Blocks(blocks) => blocks.iter().any(|b| match b {
                CountContentBlock::Image {} => true,
                CountContentBlock::ToolResult {
                    content: Some(c), ..
                } => c.contains_image(),
                _ => false,
            }),
        })
    }
}

fn anthropic_error(status: StatusCode, err_type: &str, message: impl Into<String>) -> Response {
    let body = serde_json::json!({
        "type": "error",
        "error": {
            "type": err_type,
            "message": message.into(),
        }
    });
    (status, Json(body)).into_response()
}

fn tool_choice_is_none(tc: Option<&serde_json::Value>) -> bool {
    tc.and_then(|v| v.get("type"))
        .and_then(|t| t.as_str())
        .is_some_and(|t| t == "none")
}

/// Convert Anthropic messages + system + tools into OpenAI-shaped JSON values
/// ready for chat-template tokenization.
pub(crate) fn build_count_template_inputs(
    req: &CountTokensRequest,
    tools_active: bool,
    supports_thinking: bool,
    skip_jinja_tools: bool,
) -> (Vec<serde_json::Value>, Option<Vec<serde_json::Value>>) {
    let mut json_messages: Vec<serde_json::Value> = Vec::with_capacity(req.messages.len() + 1);

    if let Some(ref sys) = req.system {
        let sys_text = match sys {
            CountSystemContent::Text(s) => s.clone(),
            CountSystemContent::Blocks(blocks) => blocks
                .iter()
                .filter(|b| {
                    b.block_type == "text"
                        && !b.text.as_deref().unwrap_or("").starts_with("x-anthropic-")
                })
                .filter_map(|b| b.text.clone())
                .collect::<Vec<_>>()
                .join("\n"),
        };
        json_messages.push(serde_json::json!({
            "role": "system",
            "content": sys_text,
        }));
    }

    for m in &req.messages {
        let role = if m.role == "assistant" {
            "assistant"
        } else {
            "user"
        };
        match &m.content {
            CountMessageContent::Text(s) => {
                let text = if role == "assistant" && supports_thinking {
                    format!("<think>\n\n</think>\n\n{s}")
                } else {
                    s.clone()
                };
                json_messages.push(serde_json::json!({
                    "role": role,
                    "content": text,
                }));
            }
            CountMessageContent::Blocks(blocks) => {
                if tools_active
                    && role == "user"
                    && blocks
                        .iter()
                        .any(|b| matches!(b, CountContentBlock::ToolResult { .. }))
                {
                    for block in blocks {
                        match block {
                            CountContentBlock::ToolResult {
                                content, is_error, ..
                            } => {
                                let mut result_text =
                                    content.as_ref().map_or_else(String::new, |c| c.to_text());
                                if is_error == &Some(true) {
                                    result_text = format!("[tool error]\n{result_text}");
                                }
                                json_messages.push(serde_json::json!({
                                    "role": "tool",
                                    "content": result_text,
                                }));
                            }
                            CountContentBlock::Text { text } if !text.is_empty() => {
                                json_messages.push(serde_json::json!({
                                    "role": "user",
                                    "content": text,
                                }));
                            }
                            _ => {}
                        }
                    }
                    continue;
                }

                let mut text_parts = Vec::new();
                let mut tool_calls = Vec::new();
                for block in blocks {
                    match block {
                        CountContentBlock::Text { text } => text_parts.push(text.clone()),
                        CountContentBlock::ToolUse { id, name, input } => {
                            let args = if input.is_null() {
                                serde_json::json!({})
                            } else {
                                input.clone()
                            };
                            tool_calls.push(serde_json::json!({
                                "id": id,
                                "type": "function",
                                "function": {
                                    "name": name,
                                    "arguments": args,
                                }
                            }));
                        }
                        CountContentBlock::ToolResult {
                            content: Some(c), ..
                        } => {
                            let t = c.to_text();
                            if !t.is_empty() {
                                text_parts.push(t);
                            }
                        }
                        _ => {}
                    }
                }
                let mut joined = text_parts.join("\n");
                if role == "assistant" && supports_thinking {
                    joined = format!("<think>\n\n</think>\n\n{joined}");
                }
                let mut msg = serde_json::json!({
                    "role": role,
                    "content": joined,
                });
                if tools_active && role == "assistant" && !tool_calls.is_empty() {
                    msg["tool_calls"] = serde_json::Value::Array(tool_calls);
                }
                json_messages.push(msg);
            }
        }
    }

    let jinja_tools = if tools_active && !skip_jinja_tools {
        req.tools.as_ref().map(|ts| {
            ts.iter()
                .map(|t| {
                    let params = t
                        .input_schema
                        .clone()
                        .unwrap_or_else(|| serde_json::json!({"type": "object", "properties": {}}));
                    serde_json::json!({
                        "type": "function",
                        "function": {
                            "name": t.name,
                            "description": t.description,
                            "parameters": params,
                        }
                    })
                })
                .collect()
        })
    } else {
        None
    };

    (json_messages, jinja_tools)
}

/// `POST /v1/messages/count_tokens` — render and tokenize Anthropic messages +
/// system + tools, returning `{"input_tokens": N}`.
pub async fn count_tokens(
    CurrentModel(state): CurrentModel,
    req: Result<Json<CountTokensRequest>, JsonRejection>,
) -> Response {
    let Json(req) = match req {
        Ok(r) => r,
        Err(e) => {
            return anthropic_error(
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                format!("Invalid request JSON: {e}"),
            );
        }
    };

    if state.yarn_context && req.contains_image() {
        return anthropic_error(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "Static-YaRN long context is currently text-only; image requests require persistent MRoPE position deltas (unsupported_multimodal_context)",
        );
    }
    if req.contains_image() {
        return anthropic_error(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "This endpoint does not yet count expanded image tokens (unsupported_multimodal_token_count)",
        );
    }
    if req.messages.is_empty() {
        return anthropic_error(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "messages must not be empty",
        );
    }

    let tools_active = (state.tool_call_parser.is_some() || state.dsv41)
        && req.tools.as_ref().is_some_and(|t| !t.is_empty())
        && !tool_choice_is_none(req.tool_choice.as_ref());

    let enable_thinking = if state.disable_thinking {
        false
    } else {
        let from_body = req.thinking.as_ref().map(|t| t.thinking_type.as_str());
        let explicit = from_body.is_some();
        let base = match from_body {
            Some("enabled" | "adaptive") => true,
            Some("disabled") => false,
            _ => {
                let d = state.runtime.snapshot();
                if d.thinking_default_runtime {
                    d.thinking_default
                } else {
                    state.behavior.thinking_default
                }
            }
        };
        if tools_active && !state.behavior.thinking_in_tools && !explicit {
            false
        } else {
            base
        }
    };

    let parser_suppresses = state
        .tool_call_parser
        .as_ref()
        .is_some_and(|p| p.suppresses_jinja_tools());
    let (json_messages, jinja_tools) = build_count_template_inputs(
        &req,
        tools_active,
        !state.dsv41 && state.tokenizer.supports_thinking(),
        parser_suppresses,
    );

    if state.dsv41 {
        let mut wire = serde_json::json!({
            "messages": json_messages,
            "enable_thinking": enable_thinking,
        });
        if let Some(ref tools) = jinja_tools {
            wire["tools"] = serde_json::Value::Array(tools.clone());
        }
        let rendered = match crate::dsv41::request::render_request(&wire) {
            Ok(r) => r,
            Err(e) => {
                return anthropic_error(
                    StatusCode::BAD_REQUEST,
                    "invalid_request_error",
                    e.message,
                );
            }
        };
        return match super::dsv41::escaped_prompt_ids(&state.tokenizer, &wire, &rendered) {
            Ok(ids) => Json(serde_json::json!({ "input_tokens": ids.len() })).into_response(),
            Err(e) => anthropic_error(
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                format!("Tokenization error: {e}"),
            ),
        };
    }

    match state.tokenizer.apply_chat_template_jinja(
        &json_messages,
        jinja_tools.as_deref(),
        enable_thinking,
        state.behavior.disable_tool_steering,
    ) {
        Ok(tokens) => Json(serde_json::json!({ "input_tokens": tokens.len() })).into_response(),
        Err(e) => anthropic_error(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            format!("Template tokenization error: {e}"),
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_sdk_count_tokens_without_max_tokens_or_model() {
        let raw = serde_json::json!({
            "system": [
                {"type": "text", "text": "x-anthropic-billing-header: ignore"},
                {"type": "text", "text": "You are a coding assistant."}
            ],
            "messages": [
                {"role": "user", "content": "Count my tokens"}
            ],
            "tools": [
                {
                    "name": "bash",
                    "description": "Execute shell command",
                    "input_schema": {
                        "type": "object",
                        "properties": {"command": {"type": "string"}},
                        "required": ["command"]
                    }
                },
                {
                    "name": "computer_20250124"
                }
            ],
            "thinking": {"type": "enabled", "budget_tokens": 1024}
        });
        let req: CountTokensRequest = serde_json::from_value(raw)
            .expect("count_tokens payload without max_tokens must parse");
        assert!(!req.contains_image());
        let (msgs, tools) = build_count_template_inputs(&req, true, false, false);
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0]["role"], "system");
        assert_eq!(msgs[0]["content"], "You are a coding assistant.");
        assert_eq!(tools.expect("tools rendered").len(), 2);
    }

    #[test]
    fn detects_images_in_messages_and_nested_tool_results() {
        let with_img: CountTokensRequest = serde_json::from_value(serde_json::json!({
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "look"},
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"}}
                ]
            }]
        }))
        .expect("parse image request");
        assert!(with_img.contains_image());
    }
}
