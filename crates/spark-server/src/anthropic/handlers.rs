// SPDX-License-Identifier: AGPL-3.0-only

use crate::main_modules::model_host::CurrentModel;

use axum::http::StatusCode;
use axum::response::Response;

use crate::openai;

use super::handlers_stream::*;
use super::helpers::*;
use super::translate::*;
use super::types::*;

// ── Handler ──

/// POST /v1/messages — Anthropic Messages API.
///
/// Translates the request into an OpenAI `ChatCompletionRequest`, dispatches
/// through `api::chat_completions_inner` (which runs every fix12-23
/// sanitization, salvage, watchdog, and dump path), and translates the
/// response back into Anthropic format. The Anthropic-specific surface is
/// strictly format conversion — no policy or sampling decisions are made
/// here.
pub async fn messages(
    CurrentModel(state): CurrentModel,
    req_ctx: Option<axum::extract::Extension<crate::rate_limiter::RequestContext>>,
    body: axum::body::Bytes,
) -> Response {
    // 1. Parse the Anthropic request.
    let req: MessagesRequest = match serde_json::from_slice(&body) {
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
            "Static-YaRN long context is currently text-only; image requests require persistent MRoPE position deltas (unsupported_multimodal_context)".to_string(),
        );
    }

    if let Err(error) = req.validate_images() {
        return anthropic_error(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            error.into(),
        );
    }
    tracing::info!(
        "Anthropic request: max_tokens={}, thinking={:?}, tools={}, model={}, stream={}",
        req.max_tokens,
        req.thinking
            .as_ref()
            .map(|t| format!("type={} budget={:?}", t.thinking_type, t.budget_tokens)),
        req.tools.as_ref().map_or(0, |t| t.len()),
        req.model,
        req.stream,
    );

    let stream = req.stream;
    let model_echo = req.model.clone();

    // 2. --dump: capture the raw Anthropic body. We mint our own seq so
    //    the entry shows endpoint="/v1/messages"; chat_completions_inner
    //    is invoked with dump_seq=None so it doesn't double-dump as
    //    "/v1/chat/completions".
    let dump_seq = state.dump_writer.as_ref().and_then(|d| {
        match serde_json::from_slice::<serde_json::Value>(&body) {
            Ok(v) => {
                let seq = d.next_seq();
                d.dump_request("/v1/messages", seq, &v);
                Some(seq)
            }
            Err(_) => None,
        }
    });

    // 3. Translate to a ChatCompletionRequest.
    let chat_json = anthropic_to_chat_request_json(&req);

    // Translation-drift telemetry. Detects whether structural information from
    // the Anthropic request was preserved in the translated OpenAI shape — a
    // count-level audit cheap enough to run on every request. The metric counts
    // mismatches; verbose diff logging is gated behind
    // FLUX_DEBUG_TRANSLATION_DRIFT (or legacy ATLAS_DEBUG_TRANSLATION_DRIFT).
    audit_translation_drift(&req, &chat_json);

    let chat_req: openai::ChatCompletionRequest = match serde_json::from_value(chat_json) {
        Ok(r) => r,
        Err(e) => {
            return anthropic_error(
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                format!("Translation error: {e}"),
            );
        }
    };

    // Drop `req` here — anthropic_to_chat_request_json already cloned what
    // it needed, and we've finished extracting `stream` + `model_echo`.

    // 4. Run the shared OpenAI pipeline. All sanitization, salvage,
    //    watchdog, sampling preset, and prompt mutation logic lives there.
    let chat_resp =
        crate::api::chat_completions_inner(state.clone(), req_ctx, chat_req, None).await;

    // 5. Translate the response back to Anthropic shape.
    if !chat_resp.status().is_success() {
        // Forward the error in Anthropic's error envelope: extract
        // `error.message` when the upstream body is an OpenAI-style JSON
        // envelope, otherwise wrap the raw body text lossily as UTF-8.
        let (parts, body) = chat_resp.into_parts();
        let body_bytes = match axum::body::to_bytes(body, usize::MAX).await {
            Ok(b) => b,
            Err(e) => {
                tracing::error!("anthropic error body collect failed: {e}");
                return anthropic_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "api_error",
                    "Internal response translation error".to_string(),
                );
            }
        };
        let err_msg = serde_json::from_slice::<serde_json::Value>(&body_bytes)
            .ok()
            .and_then(|v| {
                v.get("error")
                    .and_then(|e| e.get("message"))
                    .and_then(|m| m.as_str())
                    .map(|s| s.to_string())
            })
            .unwrap_or_else(|| String::from_utf8_lossy(&body_bytes).into_owned());
        let err_type = match parts.status {
            StatusCode::BAD_REQUEST | StatusCode::UNPROCESSABLE_ENTITY => "invalid_request_error",
            StatusCode::UNAUTHORIZED => "authentication_error",
            StatusCode::FORBIDDEN => "permission_error",
            StatusCode::NOT_FOUND => "not_found_error",
            StatusCode::TOO_MANY_REQUESTS => "rate_limit_error",
            StatusCode::SERVICE_UNAVAILABLE => "overloaded_error",
            _ => "api_error",
        };
        let mut resp = anthropic_error(parts.status, err_type, err_msg);
        if let Some(retry_after) = parts.headers.get(axum::http::header::RETRY_AFTER) {
            resp.headers_mut()
                .insert(axum::http::header::RETRY_AFTER, retry_after.clone());
        }
        return resp;
    }

    if stream {
        let resp = wrap_chat_sse_for_anthropic(chat_resp, model_echo).await;
        // Note: streaming dump from the Anthropic side is best-effort;
        // chat_completions_inner already wrote a synthesized OpenAI-shape
        // response dump entry under endpoint="/v1/chat/completions" if
        // we'd passed a dump_seq. Anthropic-shape response capture is a
        // follow-up.
        let _ = dump_seq;
        resp
    } else {
        let (mut parts, body) = chat_resp.into_parts();
        let body_bytes = match axum::body::to_bytes(body, usize::MAX).await {
            Ok(b) => b,
            Err(e) => {
                tracing::error!("anthropic response body collect failed: {e}");
                return anthropic_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "api_error",
                    "Internal response translation error".to_string(),
                );
            }
        };
        let chat_value: serde_json::Value = match serde_json::from_slice(&body_bytes) {
            Ok(v) => v,
            Err(e) => {
                tracing::error!("anthropic inner response decode failed: {e}");
                return anthropic_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "api_error",
                    "Internal response translation error".to_string(),
                );
            }
        };
        let messages_resp = chat_to_anthropic_response(&chat_value, model_echo);

        if let (Some(seq), Some(dump)) = (dump_seq, state.dump_writer.as_ref()) {
            dump.dump_response("/v1/messages", seq, &messages_resp, false);
        }

        // Preserve status code/headers from chat_completions_inner, but drop
        // any stale Content-Length from the OpenAI JSON body so hyper computes
        // the exact byte length of the translated Anthropic payload.
        parts.headers.remove(axum::http::header::CONTENT_LENGTH);
        let json_bytes = serde_json::to_vec(&messages_resp).unwrap_or_default();
        Response::from_parts(parts, axum::body::Body::from(json_bytes))
    }
}
