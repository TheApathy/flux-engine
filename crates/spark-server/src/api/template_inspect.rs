// SPDX-License-Identifier: AGPL-3.0-only

//! `POST /v1/apply-template` — render `messages` + `tools` + `enable_thinking`
//! through the active chat template without running GPU inference, returning
//! `{"prompt": rendered, "token_ids": ids, "token_count": ids.len()}`.

use axum::extract::rejection::JsonRejection;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Json, Response};
use serde::{Deserialize, Serialize};

use super::compact::openai_error_response;
use crate::main_modules::model_host::CurrentModel;
use crate::openai::{ChatTemplateKwargs, ReasoningEffort};

#[derive(Debug, Deserialize)]
pub struct ApplyTemplateRequest {
    pub messages: Vec<serde_json::Value>,
    #[serde(default)]
    pub tools: Option<Vec<serde_json::Value>>,
    #[serde(default)]
    pub enable_thinking: Option<bool>,
    #[serde(default)]
    pub preserve_thinking: Option<bool>,
    #[serde(default)]
    pub reasoning_effort: Option<ReasoningEffort>,
    #[serde(default)]
    pub chat_template_kwargs: Option<ChatTemplateKwargs>,
}

#[derive(Debug, Serialize, PartialEq, Eq)]
pub struct ApplyTemplateResponse {
    pub prompt: String,
    pub token_ids: Vec<u32>,
    pub token_count: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ResolvedTemplateControls {
    pub enable_thinking: bool,
    pub preserve_thinking: Option<bool>,
    pub reasoning_effort: Option<String>,
}

pub(crate) fn resolve_template_controls(
    req: &ApplyTemplateRequest,
    disable_thinking: bool,
    model_default: bool,
    server_default: Option<&ChatTemplateKwargs>,
    model_preserve_thinking: Option<bool>,
) -> ResolvedTemplateControls {
    let kw = req.chat_template_kwargs.as_ref();
    let explicit_req = req.enable_thinking.is_some()
        || req.reasoning_effort.is_some()
        || kw.is_some_and(|k| k.enable_thinking.is_some() || k.thinking_budget.is_some());
    let effective_kw = if explicit_req {
        kw
    } else {
        server_default.or(kw)
    };

    let effort_enables = req.reasoning_effort.as_ref().map(|e| match e {
        ReasoningEffort::Named(s) => {
            let norm = s.trim().to_ascii_lowercase();
            !matches!(norm.as_str(), "none" | "off" | "0" | "disabled")
        }
        ReasoningEffort::Integer(_) => true,
    });

    let requested_thinking = req
        .enable_thinking
        .or_else(|| effective_kw.and_then(|k| k.thinking_budget.map(|b| b > 0)))
        .or_else(|| effective_kw.and_then(|k| k.enable_thinking))
        .or(effort_enables)
        .unwrap_or(model_default);

    let enable_thinking = !disable_thinking && requested_thinking;
    let preserve_thinking = req
        .preserve_thinking
        .or_else(|| effective_kw.and_then(|k| k.preserve_thinking))
        .or(model_preserve_thinking);
    let reasoning_effort = req
        .reasoning_effort
        .as_ref()
        .map(|e| e.as_template_str().to_string());

    ResolvedTemplateControls {
        enable_thinking,
        preserve_thinking,
        reasoning_effort,
    }
}

/// `POST /v1/apply-template` — inspect exact rendered prompt text and token IDs.
pub async fn apply_template(
    CurrentModel(state): CurrentModel,
    req: Result<Json<ApplyTemplateRequest>, JsonRejection>,
) -> Response {
    let Json(req) = match req {
        Ok(r) => r,
        Err(e) => {
            return openai_error_response(
                StatusCode::BAD_REQUEST,
                format!("Invalid request JSON: {e}"),
            );
        }
    };

    if req.messages.is_empty() {
        return openai_error_response(
            StatusCode::BAD_REQUEST,
            "messages must not be empty".to_string(),
        );
    }

    let (model_default, server_kwargs) = state.runtime.snapshot().thinking_inputs(
        state.behavior.thinking_default,
        state.default_chat_template_kwargs.as_ref(),
    );
    let controls = resolve_template_controls(
        &req,
        state.disable_thinking,
        model_default,
        server_kwargs.as_ref(),
        state.behavior.preserve_thinking,
    );

    if state.dsv41 {
        let mut wire = serde_json::json!({
            "messages": req.messages,
            "enable_thinking": controls.enable_thinking,
        });
        if let Some(ref tools) = req.tools {
            wire["tools"] = serde_json::Value::Array(tools.clone());
        }
        let rendered = match crate::dsv41::request::render_request(&wire) {
            Ok(r) => r,
            Err(e) => return openai_error_response(StatusCode::BAD_REQUEST, e.message),
        };
        let token_ids = match super::dsv41::escaped_prompt_ids(&state.tokenizer, &wire, &rendered) {
            Ok(ids) => ids,
            Err(e) => {
                return openai_error_response(
                    StatusCode::BAD_REQUEST,
                    format!("Tokenization error: {e}"),
                );
            }
        };
        let token_count = token_ids.len();
        return Json(ApplyTemplateResponse {
            prompt: rendered.prompt,
            token_ids,
            token_count,
        })
        .into_response();
    }

    let tools_slice = req.tools.as_deref().filter(|t| !t.is_empty());
    let token_ids = match state.tokenizer.apply_chat_template_openai_with_options(
        &req.messages,
        tools_slice,
        controls.enable_thinking,
        state.behavior.disable_tool_steering,
        controls.reasoning_effort.as_deref(),
        controls.preserve_thinking,
    ) {
        Ok(ids) => ids,
        Err(e) => {
            return openai_error_response(
                StatusCode::BAD_REQUEST,
                format!("Template rendering error: {e}"),
            );
        }
    };

    let prompt = match state.tokenizer.decode_with_special(&token_ids) {
        Ok(text) => text,
        Err(e) => {
            tracing::error!("apply_template detokenization error: {e:#}");
            return openai_error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                "Internal detokenization error".to_string(),
            );
        }
    };
    let token_count = token_ids.len();
    Json(ApplyTemplateResponse {
        prompt,
        token_ids,
        token_count,
    })
    .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn template_controls_respect_enable_thinking_kwargs_effort_and_kill_switch() {
        let explicit_off: ApplyTemplateRequest = serde_json::from_value(serde_json::json!({
            "messages": [{"role": "user", "content": "hi"}],
            "enable_thinking": false,
        }))
        .expect("parse");
        let c = resolve_template_controls(&explicit_off, false, true, None, Some(true));
        assert!(!c.enable_thinking);
        assert_eq!(c.preserve_thinking, Some(true));

        let int_effort: ApplyTemplateRequest = serde_json::from_value(serde_json::json!({
            "messages": [{"role": "user", "content": "solve"}],
            "reasoning_effort": 75,
        }))
        .expect("parse int effort");
        let c_int = resolve_template_controls(&int_effort, false, false, None, None);
        assert!(c_int.enable_thinking);
        assert_eq!(c_int.reasoning_effort.as_deref(), Some("xhigh"));

        // Server-wide --disable-thinking kill switch wins over explicit true.
        let explicit_on: ApplyTemplateRequest = serde_json::from_value(serde_json::json!({
            "messages": [{"role": "user", "content": "hi"}],
            "enable_thinking": true,
        }))
        .expect("parse");
        let c_killed = resolve_template_controls(&explicit_on, true, true, None, None);
        assert!(!c_killed.enable_thinking);
    }
}
