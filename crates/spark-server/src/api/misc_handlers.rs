// SPDX-License-Identifier: AGPL-3.0-only

#![allow(unused_imports, dead_code)]

use crate::main_modules::model_host::CurrentModel;
use axum::extract::State;
use axum::extract::rejection::JsonRejection;
use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive};
use axum::response::{IntoResponse, Json, Response, Sse};
use futures::StreamExt;
use std::sync::Arc;
use tokio_stream::wrappers::ReceiverStream;

use super::chat_stream::chat_completions_stream;
use super::responses_stream::responses_endpoint_stream;
use super::responses_translate::{
    build_responses_usage, emit, find_frame_end, translate_chat_response_to_responses,
};
use super::stored::extract_assistant_incoming_message;
use crate::AppState;
use crate::openai::{
    ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse, CompletionChunk,
    CompletionRequest, CompletionResponse, ModelInfo, ModelListResponse, Usage,
};
use crate::tool_parser;

// Sibling-cluster items hoisted from the original `api.rs`. These uses
// give every sub-file access to helpers that the un-split file took for
// granted via single-module visibility.
use super::chat::chat_completions_inner;
use super::compact::{compact_messages, openai_error_response, openai_error_response_with_param};
use super::completions::not_supported;
use super::failures::{
    F23ProgressMetrics, F29EnvironmentFact, F37FailureClass, F39FailureCache,
    F39PermanentFailureMatch, F49DuplicateWrite, append_f7_reminder_to_last_user,
    build_f7_stall_reminder, bump_f12_tool_call_count, check_loop_watchdog,
    collect_f7_stall_buckets, f23_build_reminder, f23_normalize_and_hash, f23_refuse_threshold,
    f23_score_progress, f23_warn_threshold, f28_text_looks_like_error,
    f29_extract_binary_from_error_line, f29_extract_environment_facts,
    f29_inject_environment_facts, f31_inject_hard_refusal, f32_reposition_failed_tool_result,
    f37_classify_failure, f39_build_circuit_breaker_banner, f39_build_failure_cache,
    f39_class_label, f39_detect_recent_retries, f39_extract_binary_name,
    f44_check_permanent_failure, f49_build_banner, f49_detect_duplicate_writes,
    f49_extract_write_path_and_content, f50_append_original_error, f60_disable_mtp_for_request,
    flush_content_sanitizer, prepend_reminder_to_system, recent_message_is_tool_error,
    strip_xml_leaks_from_assistant_content,
};
use super::inference_impl::{extract_thinking, strip_stop_sequences, tokenize_stop_sequences};
use super::inference_types::{
    GrammarSpec, InferenceRequest, InferenceResponse, StreamEvent, TokenLogprobs,
};
use super::sanitizer::{
    F7_STALL_REFUSE_THRESHOLD, F7_STALL_WARN_THRESHOLD, F7StallBuckets, ToolKind, classify_tool,
    extract_bash_final_action, primary_arg_for_tool, sanitize_content_chunk,
};
use super::strip::strip_thinking_tags;

// Re-export sibling helpers via crate::api::* for short paths.
use super::failures::*;
use super::inference_types::*;
use super::sanitizer::*;

pub async fn cancel_response(axum::extract::Path(id): axum::extract::Path<String>) -> Response {
    openai_error_response_with_param(
        StatusCode::BAD_REQUEST,
        format!(
            "Response '{id}' cannot be cancelled: Flux completes responses synchronously. Cancel only applies when the request was created with `background: true`, which this server does not support."
        ),
        Some("id"),
        Some("response_not_cancellable"),
    )
}

/// GET /metrics — Prometheus metrics endpoint.
pub async fn metrics_handler() -> Response {
    use prometheus::Encoder;
    use std::fmt::Write;

    let encoder = prometheus::TextEncoder::new();
    let metric_families = prometheus::gather();
    let mut buffer = Vec::new();
    if let Err(err) = encoder.encode(&metric_families, &mut buffer) {
        tracing::error!("prometheus metrics encode failed: {err}");
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            "Failed to encode Prometheus metrics",
        )
            .into_response();
    }
    let atlas_text = String::from_utf8(buffer).unwrap_or_default();

    // Prefix cache counters (global atomics from spark-runtime)
    let hits = spark_runtime::prefix_cache::cache_hit_count();
    let misses = spark_runtime::prefix_cache::cache_miss_count();
    let hit_tokens = spark_runtime::prefix_cache::cache_hit_tokens_total();
    let total = hits + misses;
    let hit_rate = if total > 0 {
        hits as f64 / total as f64
    } else {
        0.0
    };

    // Prepend compact vllm:* OpenMetrics compatibility block at byte 0 before
    // `atlas_*` metrics so benchmark harnesses probing the leading 2,000 bytes
    // detect vLLM-compatible telemetry immediately while all atlas_* metrics
    // stay intact below.
    let vllm_header = crate::metrics::compat_metrics::render_vllm_openmetrics(total, hits);
    let mut text = String::with_capacity(vllm_header.len() + atlas_text.len() + 768);
    text.push_str(&vllm_header);
    text.push_str(&atlas_text);

    let _ = write!(
        text,
        "\
        # HELP atlas_prefix_cache_hits_total Prefix cache lookups that found cached blocks\n\
        # TYPE atlas_prefix_cache_hits_total counter\n\
        atlas_prefix_cache_hits_total {hits}\n\
        # HELP atlas_prefix_cache_misses_total Prefix cache lookups with no match\n\
        # TYPE atlas_prefix_cache_misses_total counter\n\
        atlas_prefix_cache_misses_total {misses}\n\
        # HELP atlas_prefix_cache_hit_tokens_total Tokens reused from prefix cache\n\
        # TYPE atlas_prefix_cache_hit_tokens_total counter\n\
        atlas_prefix_cache_hit_tokens_total {hit_tokens}\n\
        # HELP atlas_prefix_cache_hit_rate Prefix cache hit rate (0-1)\n\
        # TYPE atlas_prefix_cache_hit_rate gauge\n\
        atlas_prefix_cache_hit_rate {hit_rate:.4}\n"
    );

    // Entropy monitoring (global atomics from spark-runtime sampler)
    let entropy = spark_runtime::sampler::last_entropy();
    let low_entropy = spark_runtime::sampler::low_entropy_token_count();
    let total_sampled = spark_runtime::sampler::total_sampled_token_count();
    let low_ratio = if total_sampled > 0 {
        low_entropy as f64 / total_sampled as f64
    } else {
        0.0
    };

    let _ = write!(
        text,
        "\
        # HELP atlas_token_entropy_last Most recent per-token entropy (nats)\n\
        # TYPE atlas_token_entropy_last gauge\n\
        atlas_token_entropy_last {entropy:.4}\n\
        # HELP atlas_low_entropy_tokens_total Tokens with entropy below 0.3\n\
        # TYPE atlas_low_entropy_tokens_total counter\n\
        atlas_low_entropy_tokens_total {low_entropy}\n\
        # HELP atlas_low_entropy_ratio Fraction of tokens with entropy below 0.3\n\
        # TYPE atlas_low_entropy_ratio gauge\n\
        atlas_low_entropy_ratio {low_ratio:.4}\n"
    );

    (
        [(
            axum::http::header::CONTENT_TYPE,
            "text/plain; version=0.0.4; charset=utf-8",
        )],
        text,
    )
        .into_response()
}

struct ReadinessInputs<'a> {
    ready_model: Option<&'a str>,
    fault: Option<&'a str>,
    draining: bool,
    memory_floor: bool,
}

fn readiness(inputs: ReadinessInputs<'_>) -> (StatusCode, serde_json::Value) {
    if let Some(reason) = inputs.fault {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            serde_json::json!({"status": "faulted", "reason": reason}),
        );
    }
    if inputs.draining {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            serde_json::json!({"status": "draining"}),
        );
    }
    if inputs.memory_floor {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            serde_json::json!({"status": "memory_floor"}),
        );
    }
    match inputs.ready_model {
        Some(model) => (
            StatusCode::OK,
            serde_json::json!({"status": "ready", "model": model}),
        ),
        None => (
            StatusCode::SERVICE_UNAVAILABLE,
            serde_json::json!({"status": "loading"}),
        ),
    }
}

/// GET /health — readiness probe (503 while loading, faulted, draining, or memory-floor breached).
///
/// Answers from the HOST, not through `CurrentModel`. Every other route may be
/// rejected with 503 `model_not_loaded` while a swap is in flight, but health
/// is the route whose whole job is to say so — rejecting it would replace its
/// own answer with the extractor's, and a probe would read the swap as the
/// server being broken rather than as the server loading.
pub async fn health(
    State(host): State<Arc<crate::main_modules::model_host::ModelHost>>,
) -> Response {
    let ready = host
        .current()
        .filter(|s| s.model_ready.load(std::sync::atomic::Ordering::Relaxed));
    let (status, body) = readiness(ReadinessInputs {
        ready_model: ready.as_ref().map(|state| state.model_name.as_str()),
        fault: atlas_core::fault::global().fault(),
        draining: crate::tui::shutdown::requested(),
        memory_floor: crate::main_modules::mem_watch::is_refusing(),
    });
    (status, Json(body)).into_response()
}

/// GET /health/live — liveness fails once restart is required.
pub async fn health_live() -> Response {
    match atlas_core::fault::global().fault() {
        None => "ok".into_response(),
        Some(reason) => (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({"status": "faulted", "reason": reason})),
        )
            .into_response(),
    }
}

/// POST /tokenize — tokenize text or chat messages, return token IDs and count.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct TokenizeTemplateOptions {
    enable_thinking: bool,
    preserve_thinking: Option<bool>,
}

/// Resolve the subset of chat-template policy that `/tokenize` can express.
/// This mirrors the chat path's server-default merge and thinking precedence:
/// an explicit request wins, otherwise `--default-chat-template-kwargs` wins,
/// then the model default.  The global disable switch remains authoritative.
fn tokenize_template_options(
    req: &crate::openai::TokenizeRequest,
    disable_thinking: bool,
    model_default: bool,
    server_default: Option<&crate::openai::ChatTemplateKwargs>,
    model_preserve_thinking: Option<bool>,
) -> TokenizeTemplateOptions {
    let requested = req.chat_template_kwargs.as_ref();
    let request_is_explicit = requested
        .is_some_and(|kwargs| kwargs.thinking_budget.is_some() || kwargs.enable_thinking.is_some());
    // Chat phase 1.5 replaces silent request kwargs with the server default.
    let effective = if request_is_explicit {
        requested
    } else {
        server_default.or(requested)
    };
    let requested_thinking = effective
        .and_then(|kwargs| kwargs.thinking_budget.map(|budget| budget > 0))
        .or_else(|| effective.and_then(|kwargs| kwargs.enable_thinking))
        .unwrap_or(model_default);
    TokenizeTemplateOptions {
        enable_thinking: !disable_thinking && requested_thinking,
        preserve_thinking: effective
            .and_then(|kwargs| kwargs.preserve_thinking)
            .or(model_preserve_thinking),
    }
}

/// Build template-ready JSON messages for `/tokenize`, mirroring the chat path:
/// normalize leading `developer` -> `system`, coalesce consecutive leading
/// `system`/`developer` blocks, map mid-conversation `developer` -> `user`,
/// and preserve historical `tool_calls`, `tool_call_id`, and `reasoning_content`.
fn build_tokenize_json_messages(
    messages: &[crate::openai::IncomingMessage],
) -> Vec<serde_json::Value> {
    let leading_end = messages
        .iter()
        .take_while(|m| m.role == "system" || m.role == "developer")
        .count();
    let mut out = Vec::with_capacity(messages.len());
    if leading_end > 0 {
        let mut combined = String::new();
        for m in &messages[..leading_end] {
            if !m.content.text.is_empty() {
                if !combined.is_empty() {
                    combined.push_str("\n\n");
                }
                combined.push_str(&m.content.text);
            }
        }
        out.push(serde_json::json!({ "role": "system", "content": combined }));
    }
    for m in &messages[leading_end..] {
        let role = if m.role == "developer" {
            "user"
        } else {
            m.role.as_str()
        };
        let mut obj = serde_json::Map::new();
        obj.insert(
            "role".to_string(),
            serde_json::Value::String(role.to_string()),
        );
        obj.insert(
            "content".to_string(),
            serde_json::Value::String(m.content.text.clone()),
        );
        if let Some(rc) = &m.reasoning_content {
            obj.insert(
                "reasoning_content".to_string(),
                serde_json::Value::String(rc.clone()),
            );
        }
        if let Some(id) = &m.tool_call_id {
            obj.insert(
                "tool_call_id".to_string(),
                serde_json::Value::String(id.clone()),
            );
        }
        if m.role == "assistant"
            && let Some(tcs) = m.tool_calls.as_ref().filter(|t| !t.is_empty())
        {
            let parsed: Vec<serde_json::Value> = tcs
                .iter()
                .map(|tc| {
                    let args = serde_json::from_str::<serde_json::Value>(&tc.function.arguments)
                        .unwrap_or_else(|_| serde_json::Value::Object(Default::default()));
                    serde_json::json!({
                        "id": tc.id.as_deref().unwrap_or(""),
                        "type": "function",
                        "function": { "name": tc.function.name, "arguments": args }
                    })
                })
                .collect();
            obj.insert("tool_calls".to_string(), serde_json::Value::Array(parsed));
        }
        out.push(serde_json::Value::Object(obj));
    }
    out
}

pub async fn tokenize(
    CurrentModel(state): CurrentModel,
    req: Result<Json<crate::openai::TokenizeRequest>, JsonRejection>,
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

    if state.dsv41 && req.messages.is_some() {
        // The generic chat template path would render ChatML for this model.
        return openai_error_response(
            StatusCode::BAD_REQUEST,
            "deepseek_v41: render chat messages with POST /v1/debug/prompt".into(),
        );
    }
    if req.messages.as_ref().is_some_and(|messages| {
        messages
            .iter()
            .any(|message| !message.content.images.is_empty())
    }) {
        return openai_error_response(StatusCode::BAD_REQUEST,
            "This endpoint does not yet count expanded image tokens (unsupported_multimodal_token_count)".into());
    }
    let tokens = if let Some(ref prompt) = req.prompt {
        match state.tokenizer.encode(prompt) {
            Ok(t) => t,
            Err(e) => {
                return openai_error_response(
                    StatusCode::BAD_REQUEST,
                    format!("Tokenization error: {e}"),
                );
            }
        }
    } else if let Some(ref messages) = req.messages {
        let json_messages = build_tokenize_json_messages(messages);
        // Render with the thinking default a chat request would resolve now
        // (`/v1/flux/settings`): a runtime default replaces the model default
        // and the launch kwargs' `enable_thinking`, exactly as in chat.
        let (model_default, server_kwargs) = state.runtime.snapshot().thinking_inputs(
            state.behavior.thinking_default,
            state.default_chat_template_kwargs.as_ref(),
        );
        let options = tokenize_template_options(
            &req,
            state.disable_thinking,
            model_default,
            server_kwargs.as_ref(),
            state.behavior.preserve_thinking,
        );
        match state.tokenizer.apply_chat_template_openai_with_options(
            &json_messages,
            None,
            options.enable_thinking,
            state.behavior.disable_tool_steering,
            None,
            options.preserve_thinking,
        ) {
            Ok(t) => t,
            Err(e) => {
                return openai_error_response(
                    StatusCode::BAD_REQUEST,
                    format!("Tokenization error: {e}"),
                );
            }
        }
    } else {
        return openai_error_response(
            StatusCode::BAD_REQUEST,
            "Either 'prompt' or 'messages' is required".to_string(),
        );
    };

    let count = tokens.len();
    Json(crate::openai::TokenizeResponse { tokens, count }).into_response()
}

#[cfg(test)]
mod tokenize_template_tests {
    use super::{TokenizeTemplateOptions, tokenize_template_options};
    use crate::openai::{ChatTemplateKwargs, TokenizeRequest};
    use serde_json::json;

    fn request(kwargs: serde_json::Value) -> TokenizeRequest {
        serde_json::from_value(json!({
            "model": "fixture",
            "messages": [{"role": "user", "content": "hello"}],
            "chat_template_kwargs": kwargs,
        }))
        .expect("valid /tokenize request")
    }

    #[test]
    fn tokenize_request_preserves_exact_thinking_on_off_contract() {
        let on = request(json!({"enable_thinking": true}));
        let off = request(json!({"enable_thinking": false}));

        assert_eq!(
            tokenize_template_options(&on, false, false, None, None),
            TokenizeTemplateOptions {
                enable_thinking: true,
                preserve_thinking: None,
            }
        );
        assert_eq!(
            tokenize_template_options(&off, false, true, None, None),
            TokenizeTemplateOptions {
                enable_thinking: false,
                preserve_thinking: None,
            }
        );
        assert_eq!(
            on.chat_template_kwargs
                .as_ref()
                .and_then(|kwargs| kwargs.enable_thinking),
            Some(true)
        );
    }

    /// `/tokenize` renders with the runtime thinking default, like chat: it
    /// beats the model default and the launch kwargs, and request kwargs
    /// still win.
    #[test]
    fn tokenize_follows_the_runtime_thinking_default() {
        let settings = crate::runtime_settings::RuntimeSettings::for_tests(1024);
        let launch_on = crate::openai::ChatTemplateKwargs::from_json(r#"{"enable_thinking":true}"#);
        let silent: TokenizeRequest = serde_json::from_value(json!({
            "messages": [{"role": "user", "content": "hello"}],
        }))
        .unwrap();
        let render = |req: &TokenizeRequest| {
            let (model_default, kwargs) = settings
                .snapshot()
                .thinking_inputs(false, launch_on.as_ref());
            tokenize_template_options(req, false, model_default, kwargs.as_ref(), None)
                .enable_thinking
        };
        assert!(
            render(&silent),
            "untouched: the launch kwargs default applies"
        );
        settings
            .apply_patch(&json!({"thinking.default": false}))
            .unwrap();
        assert!(
            !render(&silent),
            "the runtime default replaces the launch kwargs"
        );
        settings
            .apply_patch(&json!({"thinking.default": true}))
            .unwrap();
        assert!(render(&silent));
        assert!(
            !render(&request(json!({"enable_thinking": false}))),
            "request wins"
        );
    }

    #[test]
    fn tokenize_thinking_precedence_matches_chat_defaults_and_kill_switch() {
        let silent: TokenizeRequest = serde_json::from_value(json!({
            "messages": [{"role": "user", "content": "hello"}],
        }))
        .expect("valid silent /tokenize request");
        let server_default = ChatTemplateKwargs {
            enable_thinking: Some(true),
            thinking_budget: None,
            preserve_thinking: Some(false),
        };
        assert_eq!(
            tokenize_template_options(&silent, false, false, Some(&server_default), Some(true)),
            TokenizeTemplateOptions {
                enable_thinking: true,
                preserve_thinking: Some(false),
            }
        );
        assert!(
            !tokenize_template_options(&silent, true, true, Some(&server_default), None)
                .enable_thinking
        );

        let zero_budget = request(json!({"enable_thinking": true, "thinking_budget": 0}));
        assert!(!tokenize_template_options(&zero_budget, false, true, None, None).enable_thinking);
    }

    #[test]
    fn tokenize_messages_normalize_developer_roles_and_preserve_tool_metadata() {
        let req: TokenizeRequest = serde_json::from_value(json!({
            "messages": [
                {"role": "system", "content": "Base."},
                {"role": "developer", "content": "Rule."},
                {"role": "user", "content": "Hi"},
                {
                    "role": "assistant",
                    "content": "",
                    "reasoning_content": "plan",
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": "{\"city\":\"Paris\"}"}
                    }]
                },
                {"role": "tool", "tool_call_id": "call_1", "content": "18C"},
                {"role": "developer", "content": "Mid note"}
            ]
        }))
        .expect("valid /tokenize request");
        let built = super::build_tokenize_json_messages(req.messages.as_ref().unwrap());
        assert_eq!(built.len(), 5);
        assert_eq!(built[0]["role"], "system");
        assert_eq!(built[0]["content"], "Base.\n\nRule.");
        assert_eq!(built[2]["reasoning_content"], "plan");
        assert_eq!(
            built[2]["tool_calls"][0]["function"]["arguments"]["city"],
            "Paris"
        );
        assert_eq!(built[3]["tool_call_id"], "call_1");
        assert_eq!(built[4]["role"], "user");
        assert_eq!(built[4]["content"], "Mid note");
    }
}

#[cfg(test)]
mod readiness_tests {
    use super::{ReadinessInputs, readiness};
    use axum::http::StatusCode;

    #[test]
    fn fatal_fault_outranks_a_published_ready_model() {
        let (status, body) = readiness(ReadinessInputs {
            ready_model: Some("fixture"),
            fault: Some("scheduler gone"),
            draining: true,
            memory_floor: true,
        });
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(body["status"], "faulted");
        assert_eq!(body["reason"], "scheduler gone");
    }

    #[test]
    fn draining_and_memory_floor_return_503_for_readiness() {
        let (status, body) = readiness(ReadinessInputs {
            ready_model: Some("fixture"),
            fault: None,
            draining: true,
            memory_floor: false,
        });
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(body["status"], "draining");

        let (status, body) = readiness(ReadinessInputs {
            ready_model: Some("fixture"),
            fault: None,
            draining: true,
            memory_floor: true,
        });
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(body["status"], "draining");

        let (status, body) = readiness(ReadinessInputs {
            ready_model: Some("fixture"),
            fault: None,
            draining: false,
            memory_floor: true,
        });
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(body["status"], "memory_floor");
    }

    #[test]
    fn healthy_readiness_preserves_ready_and_loading_states() {
        assert_eq!(
            readiness(ReadinessInputs {
                ready_model: Some("fixture"),
                fault: None,
                draining: false,
                memory_floor: false,
            })
            .0,
            StatusCode::OK
        );
        assert_eq!(
            readiness(ReadinessInputs {
                ready_model: None,
                fault: None,
                draining: false,
                memory_floor: false,
            })
            .0,
            StatusCode::SERVICE_UNAVAILABLE
        );
    }

    #[tokio::test]
    async fn metrics_handler_succeeds_without_panicking() {
        let resp = super::metrics_handler().await;
        assert_eq!(resp.status(), StatusCode::OK);
    }
}

/// Request body for POST /detokenize.
#[derive(serde::Deserialize)]
pub struct DetokenizeRequest {
    tokens: Vec<u32>,
}

/// POST /detokenize — decode token IDs back to text.
pub async fn detokenize(
    CurrentModel(state): CurrentModel,
    req: Result<Json<DetokenizeRequest>, JsonRejection>,
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
    match state.tokenizer.decode(&req.tokens) {
        Ok(text) => Json(serde_json::json!({"text": text})).into_response(),
        Err(e) => openai_error_response(StatusCode::BAD_REQUEST, e.to_string()),
    }
}
