// SPDX-License-Identifier: AGPL-3.0-only

#![allow(unused_imports, dead_code)]

use crate::main_modules::model_host::CurrentModel;
use axum::extract::State;
use axum::extract::rejection::JsonRejection;
use axum::http::StatusCode;
use axum::response::sse::Event;
use axum::response::{IntoResponse, Json, Response, Sse};
use futures::StreamExt;
use std::sync::Arc;
use tokio_stream::wrappers::ReceiverStream;

use super::chat_stream::chat_completions_stream;
use super::responses_stream_finalize::{
    CloseOpenCtx, FinalizeCtx, close_open_items, emit_responses_prologue,
    finalize_responses_stream_with_stores,
};
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
// Sibling-cluster items hoisted from the original `api.rs`.
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

pub(super) async fn responses_endpoint_stream(
    CurrentModel(state): CurrentModel,
    req_ctx: Option<axum::extract::Extension<crate::rate_limiter::RequestContext>>,
    mut chat_req: ChatCompletionRequest,
    metadata: Option<std::collections::HashMap<String, String>>,
    store_flag: bool,
    conversation_id: Option<String>,
    conv_new_user_items: Vec<serde_json::Value>,
) -> Response {
    use axum::response::sse::Event;
    use tokio::sync::mpsc;
    // Force streaming on the inner pipeline and disable the
    // stream_options usage sidecar — we want the `done_chunk` shape so
    // the transformer sees a single terminal `finish_reason` event.
    chat_req.stream = true;
    chat_req.stream_options = None;

    let model = chat_req.model.clone();
    let input_messages = chat_req.messages.clone();
    let state_arc = state.clone();

    // Run the chat-completions streaming handler (re-using the full
    // pipeline: scheduler, tool detection, thinking, logprobs, ...).
    let chat_resp = chat_completions_inner(state, req_ctx, chat_req, None).await;
    let (parts, body) = chat_resp.into_parts();
    if !parts.status.is_success() {
        return Response::from_parts(parts, body);
    }

    // Channel the transformer pushes Responses SSE events into. Sized to
    // match chat_stream/mod.rs (~30s decode buffer at 50 tok/s).
    let (tx, rx) = mpsc::channel::<Result<Event, std::convert::Infallible>>(1024);
    let created_at = crate::openai::unix_timestamp();
    let resp_id = format!("resp_{}", crate::openai::uuid_v4());

    tokio::spawn(pump_chat_sse_to_responses(PumpParams {
        body,
        tx,
        response_store: state_arc.response_store.clone(),
        conversation_store: state_arc.conversation_store.clone(),
        resp_id,
        created_at,
        model,
        metadata,
        store_flag,
        input_messages,
        conversation_id,
        conv_new_user_items,
    }));

    let stream = tokio_stream::wrappers::ReceiverStream::new(rx);
    Sse::new(stream)
        .keep_alive(super::chat_stream::sse_keep_alive())
        .into_response()
}

struct PumpParams {
    body: axum::body::Body,
    tx: tokio::sync::mpsc::Sender<Result<Event, std::convert::Infallible>>,
    response_store: Arc<crate::response_store::ResponseStore>,
    conversation_store: Arc<crate::conversation_store::ConversationStore>,
    resp_id: String,
    created_at: u64,
    model: String,
    metadata: Option<std::collections::HashMap<String, String>>,
    store_flag: bool,
    input_messages: Vec<crate::openai::IncomingMessage>,
    conversation_id: Option<String>,
    conv_new_user_items: Vec<serde_json::Value>,
}

struct ResponsesStreamState {
    resp_id: String,
    seq: u64,
    content_text: String,
    tool_args: String,
    current_tool_name: Option<String>,
    current_tool_call_id: Option<String>,
    current_fc_index: Option<u64>,
    output_index: usize,
    // Mint message item ids per output position so a text→fc→text
    // sequence doesn't reuse the same item_id (would collide with the
    // function_call still owning the prior output_index).
    message_item_id: String,
    fc_item_id: Option<String>,
    message_started: bool,
    fc_started: bool,
    fc_done: bool,
    // Accumulated final output items (closed during streaming) so the
    // terminal `response.completed` payload reflects every emitted
    // item, not just the last live one. Required for stateful resume
    // via `previous_response_id` when the model emits text→fc→text.
    completed_items: Vec<crate::openai::ResponsesOutputItem>,
}

impl ResponsesStreamState {
    fn new(resp_id: String) -> Self {
        let message_item_id = format!("msg_{resp_id}_0");
        Self {
            resp_id,
            seq: 0,
            content_text: String::new(),
            tool_args: String::new(),
            current_tool_name: None,
            current_tool_call_id: None,
            current_fc_index: None,
            output_index: 0,
            message_item_id,
            fc_item_id: None,
            message_started: false,
            fc_started: false,
            fc_done: false,
            completed_items: Vec::new(),
        }
    }

    async fn close_open_fc(
        &mut self,
        tx: &tokio::sync::mpsc::Sender<Result<Event, std::convert::Infallible>>,
    ) -> bool {
        if let Some(fcid) = self.fc_item_id.take() {
            let ev = crate::openai::ResponsesStreamEvent::FunctionCallArgumentsDone {
                sequence_number: self.seq,
                item_id: fcid.clone(),
                output_index: self.output_index,
                arguments: self.tool_args.clone(),
            };
            if !emit(tx, &ev).await {
                return false;
            }
            self.seq += 1;
            let done = crate::openai::ResponsesOutputItem::FunctionCall {
                id: fcid,
                call_id: self.current_tool_call_id.clone().unwrap_or_default(),
                name: self.current_tool_name.clone().unwrap_or_default(),
                arguments: self.tool_args.clone(),
                status: "completed",
            };
            self.completed_items.push(done.clone());
            let ev = crate::openai::ResponsesStreamEvent::OutputItemDone {
                sequence_number: self.seq,
                output_index: self.output_index,
                item: done,
            };
            if !emit(tx, &ev).await {
                return false;
            }
            self.seq += 1;
        }
        self.output_index += 1;
        self.message_item_id = format!("msg_{}_{}", self.resp_id, self.output_index);
        self.fc_started = false;
        self.fc_done = false;
        self.current_fc_index = None;
        self.current_tool_name = None;
        self.current_tool_call_id = None;
        self.tool_args.clear();
        true
    }

    async fn ensure_message_open(
        &mut self,
        tx: &tokio::sync::mpsc::Sender<Result<Event, std::convert::Infallible>>,
        initial_part: crate::openai::ResponsesContentPart,
    ) -> bool {
        // If a function_call is currently open, close it before opening a fresh
        // message item — otherwise the message would collide with the
        // function_call on the same `output_index`.
        if self.fc_started && !self.fc_done {
            if !self.close_open_fc(tx).await {
                return false;
            }
            // Reset per-message text so the new message's OutputTextDone
            // carries only the post-fc text.
            self.content_text.clear();
        }
        if !self.message_started {
            self.message_started = true;
            // output_item.added for the message item.
            let item = crate::openai::ResponsesOutputItem::Message {
                id: self.message_item_id.clone(),
                status: "in_progress",
                role: "assistant",
                content: vec![],
            };
            let ev = crate::openai::ResponsesStreamEvent::OutputItemAdded {
                sequence_number: self.seq,
                output_index: self.output_index,
                item,
            };
            if !emit(tx, &ev).await {
                return false;
            }
            self.seq += 1;
            // content_part.added.
            let ev = crate::openai::ResponsesStreamEvent::ContentPartAdded {
                sequence_number: self.seq,
                item_id: self.message_item_id.clone(),
                output_index: self.output_index,
                content_index: 0,
                part: initial_part,
            };
            if !emit(tx, &ev).await {
                return false;
            }
            self.seq += 1;
        }
        true
    }
}

fn normalize_responses_stream_error(err: &serde_json::Value) -> serde_json::Value {
    if let Some(obj) = err.as_object() {
        let mut normalized = obj.clone();
        if normalized
            .get("message")
            .and_then(serde_json::Value::as_str)
            .is_none_or(str::is_empty)
        {
            normalized.insert(
                "message".to_string(),
                serde_json::Value::String(err.to_string()),
            );
        }
        serde_json::Value::Object(normalized)
    } else {
        let message = err
            .as_str()
            .filter(|s| !s.is_empty())
            .map_or_else(|| err.to_string(), str::to_string);
        serde_json::json!({
            "code": "server_error",
            "type": "server_error",
            "message": message,
        })
    }
}

async fn pump_chat_sse_to_responses(params: PumpParams) {
    let PumpParams {
        body,
        tx,
        response_store,
        conversation_store,
        resp_id,
        created_at,
        model,
        metadata,
        store_flag,
        input_messages,
        conversation_id,
        conv_new_user_items,
    } = params;

    let mut stream_state = ResponsesStreamState::new(resp_id);
    let mut final_usage: Option<serde_json::Value> = None;
    let mut finish_reason = "stop".to_string();
    let mut stream_error: Option<serde_json::Value> = None;
    let mut accumulator = Vec::<u8>::new();
    let mut refusal_text: Option<String> = None;
    let mut reasoning_text = String::new();

    let Some(seq) = emit_responses_prologue(
        &tx,
        stream_state.seq,
        &stream_state.resp_id,
        created_at,
        &model,
        &metadata,
    )
    .await
    else {
        return;
    };
    stream_state.seq = seq;

    let mut data_stream = body.into_data_stream();
    'read_stream: loop {
        let next = tokio::select! {
            biased;
            _ = tx.closed() => return,
            next = data_stream.next() => next,
        };
        let Some(next) = next else {
            break;
        };
        let bytes = match next {
            Ok(b) => b,
            Err(e) => {
                tracing::warn!("responses stream read error: {e}");
                stream_error = Some(serde_json::json!({
                    "code": "server_error",
                    "type": "server_error",
                    "message": format!("upstream stream read error: {e}"),
                }));
                finish_reason = "error".to_string();
                break;
            }
        };
        accumulator.extend_from_slice(&bytes);

        // Parse SSE frames: look for "\n\n".
        while let Some(idx) = find_frame_end(&accumulator) {
            let frame_bytes: Vec<u8> = accumulator.drain(..idx + 2).collect();
            // Strip the trailing \n\n; find the data: line.
            let frame_str = std::str::from_utf8(&frame_bytes).unwrap_or("");
            let data_line = frame_str
                .lines()
                .find_map(|l| l.strip_prefix("data:").map(str::trim));
            let Some(data) = data_line else { continue };
            if data.is_empty() || data == "[DONE]" {
                continue;
            }
            // Parse as ChatCompletionChunk JSON.
            let Ok(chunk): Result<serde_json::Value, _> = serde_json::from_str(data) else {
                continue;
            };
            // Inspect mid-stream `{"error": ...}` chunks before `choices` so
            // upstream faults emit `response.failed` instead of finalizing as
            // `response.completed` with `error: null`.
            if let Some(err) = chunk.get("error").filter(|v| !v.is_null()) {
                stream_error = Some(normalize_responses_stream_error(err));
                finish_reason = "error".to_string();
                break 'read_stream;
            }
            // Chat chunks: { choices: [{ delta: {content|tool_calls|reasoning_content}, finish_reason, ... }], usage }
            if let Some(u) = chunk.get("usage")
                && !u.is_null()
            {
                final_usage = Some(u.clone());
            }
            let Some(choice) = chunk
                .get("choices")
                .and_then(|c| c.as_array())
                .and_then(|a| a.first())
            else {
                continue;
            };
            if let Some(fr) = choice.get("finish_reason").and_then(|v| v.as_str()) {
                finish_reason = fr.to_string();
            }
            let delta = choice
                .get("delta")
                .cloned()
                .unwrap_or(serde_json::Value::Null);

            // Capture streaming reasoning trace (`delta.reasoning_content`,
            // falling back to `delta.reasoning`) so stored transcripts persist
            // it for `previous_response_id` resume.
            if let Some(r) = delta
                .get("reasoning_content")
                .and_then(serde_json::Value::as_str)
                .or_else(|| delta.get("reasoning").and_then(serde_json::Value::as_str))
                && !r.is_empty()
            {
                reasoning_text.push_str(r);
            }

            // Refusal delta (post-hoc: one chunk carries the full
            // refusal sentence, emitted by the inner chat streamer).
            if let Some(r) = delta.get("refusal").and_then(|v| v.as_str())
                && !r.is_empty()
            {
                let was_open = stream_state.message_started;
                if !stream_state
                    .ensure_message_open(
                        &tx,
                        crate::openai::ResponsesContentPart::Refusal {
                            refusal: String::new(),
                        },
                    )
                    .await
                {
                    return;
                }
                let content_index = if stream_state.content_text.is_empty() {
                    0
                } else {
                    1
                };
                if was_open && content_index == 1 && refusal_text.is_none() {
                    let ev = crate::openai::ResponsesStreamEvent::ContentPartAdded {
                        sequence_number: stream_state.seq,
                        item_id: stream_state.message_item_id.clone(),
                        output_index: stream_state.output_index,
                        content_index,
                        part: crate::openai::ResponsesContentPart::Refusal {
                            refusal: String::new(),
                        },
                    };
                    if !emit(&tx, &ev).await {
                        return;
                    }
                    stream_state.seq += 1;
                }
                refusal_text.get_or_insert_default().push_str(r);
                let ev = crate::openai::ResponsesStreamEvent::RefusalDelta {
                    sequence_number: stream_state.seq,
                    item_id: stream_state.message_item_id.clone(),
                    output_index: stream_state.output_index,
                    content_index,
                    delta: r.to_string(),
                };
                if !emit(&tx, &ev).await {
                    return;
                }
                stream_state.seq += 1;
            }

            // Text content delta.
            if let Some(text) = delta.get("content").and_then(|v| v.as_str())
                && !text.is_empty()
            {
                if !stream_state
                    .ensure_message_open(
                        &tx,
                        crate::openai::ResponsesContentPart::OutputText {
                            text: String::new(),
                            annotations: None,
                        },
                    )
                    .await
                {
                    return;
                }
                stream_state.content_text.push_str(text);
                let ev = crate::openai::ResponsesStreamEvent::OutputTextDelta {
                    sequence_number: stream_state.seq,
                    item_id: stream_state.message_item_id.clone(),
                    output_index: stream_state.output_index,
                    content_index: 0,
                    delta: text.to_string(),
                };
                if !emit(&tx, &ev).await {
                    return;
                }
                stream_state.seq += 1;
            }

            // Tool-call fragments.
            if let Some(tool_calls) = delta.get("tool_calls").and_then(|v| v.as_array()) {
                for tc in tool_calls {
                    let tc_index = tc.get("index").and_then(|v| v.as_u64());
                    let call_id = tc.get("id").and_then(|v| v.as_str()).map(str::to_string);
                    let name = tc
                        .get("function")
                        .and_then(|f| f.get("name"))
                        .and_then(|v| v.as_str())
                        .map(str::to_string);
                    let args_frag = tc
                        .get("function")
                        .and_then(|f| f.get("arguments"))
                        .and_then(|v| v.as_str())
                        .unwrap_or("");
                    let is_new_fc = stream_state.fc_started
                        && !stream_state.fc_done
                        && (tc_index.is_some_and(|idx| Some(idx) != stream_state.current_fc_index)
                            || (tc_index.is_none() && name.is_some()));
                    if is_new_fc && !stream_state.close_open_fc(&tx).await {
                        return;
                    }
                    if let Some(n) = name {
                        stream_state.current_tool_name = Some(n.clone());
                        if let Some(id) = call_id {
                            stream_state.current_tool_call_id = Some(id);
                        }
                        if !stream_state.fc_started {
                            // Close any open message before starting function call.
                            if stream_state.message_started {
                                let msg_refusal = refusal_text.take();
                                let mut content = Vec::new();
                                if !stream_state.content_text.is_empty() || msg_refusal.is_none() {
                                    let ev = crate::openai::ResponsesStreamEvent::OutputTextDone {
                                        sequence_number: stream_state.seq,
                                        item_id: stream_state.message_item_id.clone(),
                                        output_index: stream_state.output_index,
                                        content_index: 0,
                                        text: stream_state.content_text.clone(),
                                    };
                                    if !emit(&tx, &ev).await {
                                        return;
                                    }
                                    stream_state.seq += 1;
                                    content.push(crate::openai::ResponsesContentPart::OutputText {
                                        text: stream_state.content_text.clone(),
                                        annotations: crate::citation::merged_annotations(
                                            &stream_state.content_text,
                                        ),
                                    });
                                }
                                if let Some(refusal) = msg_refusal {
                                    let content_index = content.len();
                                    let ev = crate::openai::ResponsesStreamEvent::RefusalDone {
                                        sequence_number: stream_state.seq,
                                        item_id: stream_state.message_item_id.clone(),
                                        output_index: stream_state.output_index,
                                        content_index,
                                        refusal: refusal.clone(),
                                    };
                                    if !emit(&tx, &ev).await {
                                        return;
                                    }
                                    stream_state.seq += 1;
                                    content.push(crate::openai::ResponsesContentPart::Refusal {
                                        refusal,
                                    });
                                }
                                let done = crate::openai::ResponsesOutputItem::Message {
                                    id: stream_state.message_item_id.clone(),
                                    status: "completed",
                                    role: "assistant",
                                    content,
                                };
                                stream_state.completed_items.push(done.clone());
                                let ev = crate::openai::ResponsesStreamEvent::OutputItemDone {
                                    sequence_number: stream_state.seq,
                                    output_index: stream_state.output_index,
                                    item: done,
                                };
                                if !emit(&tx, &ev).await {
                                    return;
                                }
                                stream_state.seq += 1;
                                stream_state.output_index += 1;
                                stream_state.message_started = false;
                                stream_state.content_text.clear();
                            }
                            let fcid = format!(
                                "fc_{}_{}",
                                stream_state.resp_id, stream_state.output_index
                            );
                            stream_state.fc_item_id = Some(fcid.clone());
                            stream_state.current_fc_index = tc_index;
                            let item = crate::openai::ResponsesOutputItem::FunctionCall {
                                id: fcid,
                                call_id: stream_state
                                    .current_tool_call_id
                                    .clone()
                                    .unwrap_or_default(),
                                name: n,
                                arguments: String::new(),
                                status: "in_progress",
                            };
                            let ev = crate::openai::ResponsesStreamEvent::OutputItemAdded {
                                sequence_number: stream_state.seq,
                                output_index: stream_state.output_index,
                                item,
                            };
                            if !emit(&tx, &ev).await {
                                return;
                            }
                            stream_state.seq += 1;
                            stream_state.fc_started = true;
                            stream_state.fc_done = false;
                        }
                    }
                    if !args_frag.is_empty() {
                        stream_state.tool_args.push_str(args_frag);
                        if let Some(fcid) = stream_state.fc_item_id.clone() {
                            let ev =
                                crate::openai::ResponsesStreamEvent::FunctionCallArgumentsDelta {
                                    sequence_number: stream_state.seq,
                                    item_id: fcid,
                                    output_index: stream_state.output_index,
                                    delta: args_frag.to_string(),
                                };
                            if !emit(&tx, &ev).await {
                                return;
                            }
                            stream_state.seq += 1;
                        }
                    }
                }
            }
        }
    }

    if tx.is_closed() {
        return;
    }

    if stream_error.is_none() && finish_reason != "error" {
        let Some(seq) = close_open_items(
            &tx,
            &mut stream_state.completed_items,
            CloseOpenCtx {
                seq: stream_state.seq,
                message_started: stream_state.message_started,
                message_item_id: &stream_state.message_item_id,
                content_text: &stream_state.content_text,
                refusal_text: refusal_text.as_deref(),
                fc_started: stream_state.fc_started,
                fc_done: stream_state.fc_done,
                fc_item_id: stream_state.fc_item_id.clone(),
                current_tool_call_id: &stream_state.current_tool_call_id,
                current_tool_name: &stream_state.current_tool_name,
                tool_args: &stream_state.tool_args,
                output_index: stream_state.output_index,
            },
        )
        .await
        else {
            return;
        };
        stream_state.seq = seq;
    }

    finalize_responses_stream_with_stores(
        &tx,
        &response_store,
        &conversation_store,
        FinalizeCtx {
            seq: stream_state.seq,
            completed_items: stream_state.completed_items,
            final_usage,
            finish_reason,
            stream_error,
            reasoning_text,
            resp_id: stream_state.resp_id,
            created_at,
            model,
            metadata_for_done: metadata,
            store_flag,
            input_messages,
            conversation_id,
            conv_new_user_items,
        },
    )
    .await;
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn sse_body_from_chunks(chunks: &[serde_json::Value]) -> axum::body::Body {
        let mut wire = String::new();
        for chunk in chunks {
            wire.push_str("data: ");
            wire.push_str(&chunk.to_string());
            wire.push_str("\n\n");
        }
        wire.push_str("data: [DONE]\n\n");
        axum::body::Body::from(wire)
    }

    async fn run_pump(
        resp_id: &str,
        chunks: &[serde_json::Value],
        input_messages: Vec<crate::openai::IncomingMessage>,
    ) -> (
        Vec<(String, serde_json::Value)>,
        Arc<crate::response_store::ResponseStore>,
    ) {
        let (tx, rx) = tokio::sync::mpsc::channel(64);
        let response_store =
            crate::response_store::ResponseStore::with_config(16, Duration::from_secs(60));
        let conversation_store =
            crate::conversation_store::ConversationStore::with_config(16, Duration::from_secs(60));

        pump_chat_sse_to_responses(PumpParams {
            body: sse_body_from_chunks(chunks),
            tx,
            response_store: response_store.clone(),
            conversation_store,
            resp_id: resp_id.to_string(),
            created_at: 1_700_000_000,
            model: "test-model".to_string(),
            metadata: None,
            store_flag: true,
            input_messages,
            conversation_id: None,
            conv_new_user_items: Vec::new(),
        })
        .await;

        let resp = Sse::new(ReceiverStream::new(rx)).into_response();
        let body = axum::body::to_bytes(resp.into_body(), 1 << 20)
            .await
            .expect("collect SSE body");
        let text = String::from_utf8(body.to_vec()).expect("valid UTF-8 SSE");

        let mut events = Vec::new();
        let mut ev_type = String::new();
        let mut data = String::new();
        for line in text.lines() {
            if line.is_empty() {
                if !data.is_empty() {
                    let json: serde_json::Value =
                        serde_json::from_str(&data).expect("valid JSON SSE data");
                    events.push((std::mem::take(&mut ev_type), json));
                    data.clear();
                }
                continue;
            }
            if let Some(ev) = line.strip_prefix("event:") {
                ev_type = ev.trim().to_string();
            } else if let Some(d) = line.strip_prefix("data:") {
                data = d.trim().to_string();
            }
        }
        if !data.is_empty() {
            let json: serde_json::Value = serde_json::from_str(&data).expect("valid JSON SSE data");
            events.push((ev_type, json));
        }
        (events, response_store)
    }

    #[tokio::test]
    async fn mid_stream_error_emits_response_failed_instead_of_completed() {
        let chunks = [
            serde_json::json!({"choices": [{"delta": {"role": "assistant"}}]}),
            serde_json::json!({"choices": [{"delta": {"content": "partial"}}]}),
            serde_json::json!({
                "error": {"message": "scheduler fault", "type": "server_error", "code": 500}
            }),
        ];
        let (events, store) = run_pump(
            "resp_err",
            &chunks,
            vec![crate::openai::IncomingMessage::synthetic_user_text(
                "hi".to_string(),
            )],
        )
        .await;

        let names: Vec<&str> = events.iter().map(|(e, _)| e.as_str()).collect();
        assert_eq!(
            names,
            [
                "response.created",
                "response.in_progress",
                "response.output_item.added",
                "response.content_part.added",
                "response.output_text.delta",
                "response.failed",
            ]
        );
        for (idx, (_, payload)) in events.iter().enumerate() {
            assert_eq!(payload["sequence_number"], idx as u64);
        }
        let failed = &events.last().expect("failed event").1;
        assert_eq!(failed["response"]["status"], "failed");
        assert_eq!(failed["error"]["message"], "scheduler fault");
        assert_eq!(failed["error"]["type"], "server_error");
        assert_eq!(failed["error"]["code"], 500);

        let stored = store
            .get("resp_err", crate::response_store::StoredKind::Response)
            .expect("failed response stored");
        assert_eq!(stored.body["status"], "failed");
        assert_eq!(stored.body["error"]["message"], "scheduler fault");
        assert_eq!(stored.messages.len(), 1);
    }

    #[tokio::test]
    async fn standalone_refusal_emits_output_item_and_content_part_prologue() {
        let chunks = [
            serde_json::json!({"choices": [{"delta": {"role": "assistant"}}]}),
            serde_json::json!({"choices": [{"delta": {"refusal": "I cannot assist with that."}}]}),
            serde_json::json!({
                "choices": [{"delta": {}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 6, "total_tokens": 16},
            }),
        ];
        let (events, _) = run_pump("resp_refusal", &chunks, Vec::new()).await;

        let names: Vec<&str> = events.iter().map(|(e, _)| e.as_str()).collect();
        let item_added = names
            .iter()
            .position(|&e| e == "response.output_item.added")
            .expect("output_item.added emitted");
        let part_added = names
            .iter()
            .position(|&e| e == "response.content_part.added")
            .expect("content_part.added emitted");
        let refusal_delta = names
            .iter()
            .position(|&e| e == "response.refusal.delta")
            .expect("refusal.delta emitted");
        let refusal_done = names
            .iter()
            .position(|&e| e == "response.refusal.done")
            .expect("refusal.done emitted");
        let item_done = names
            .iter()
            .position(|&e| e == "response.output_item.done")
            .expect("output_item.done emitted");
        let completed = names
            .iter()
            .position(|&e| e == "response.completed")
            .expect("completed emitted");

        assert!(item_added < part_added);
        assert!(part_added < refusal_delta);
        assert!(refusal_delta < refusal_done);
        assert!(refusal_done < item_done);
        assert!(item_done < completed);

        let item_id = events[item_added].1["item"]["id"]
            .as_str()
            .expect("message item id");
        assert_eq!(events[part_added].1["item_id"], item_id);
        assert_eq!(events[refusal_delta].1["item_id"], item_id);
        assert_eq!(
            events[refusal_delta].1["delta"],
            "I cannot assist with that."
        );
        assert_eq!(events[refusal_done].1["item_id"], item_id);
        assert_eq!(
            events[refusal_done].1["refusal"],
            "I cannot assist with that."
        );
        assert_eq!(events[item_done].1["item"]["content"][0]["type"], "refusal");
        assert_eq!(
            events[item_done].1["item"]["content"][0]["refusal"],
            "I cannot assist with that."
        );
        assert_eq!(
            events[completed].1["response"]["output"][0]["content"][0]["type"],
            "refusal"
        );
        assert_eq!(
            events[completed].1["response"]["output"][0]["content"][0]["refusal"],
            "I cannot assist with that."
        );
        for (idx, (_, payload)) in events.iter().enumerate() {
            assert_eq!(payload["sequence_number"], idx as u64);
        }
    }

    #[tokio::test]
    async fn streaming_reasoning_content_persists_in_stored_transcript() {
        let chunks = [
            serde_json::json!({"choices": [{"delta": {"role": "assistant"}}]}),
            serde_json::json!({"choices": [{"delta": {
                "reasoning_content": "First step. ",
                "reasoning": "First step. ",
            }}]}),
            serde_json::json!({"choices": [{"delta": {
                "reasoning": "Fallback alias.",
            }}]}),
            serde_json::json!({"choices": [{"delta": {"content": "42"}}]}),
            serde_json::json!({
                "choices": [{"delta": {}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 8, "completion_tokens": 5, "total_tokens": 13},
            }),
        ];
        let (events, store) = run_pump(
            "resp_reason",
            &chunks,
            vec![crate::openai::IncomingMessage::synthetic_user_text(
                "What is 6*7?".to_string(),
            )],
        )
        .await;

        let completed = &events.last().expect("completed event").1;
        assert_eq!(completed["type"], "response.completed");
        assert!(
            completed["response"].get("reasoning").is_none(),
            "Response.reasoning is a ReasoningConfig object and must not be overloaded with raw trace text"
        );

        let stored = store
            .get("resp_reason", crate::response_store::StoredKind::Response)
            .expect("response stored");
        assert_eq!(stored.messages.len(), 2);
        assert_eq!(stored.messages[1].role, "assistant");
        assert_eq!(stored.messages[1].content.text, "42");
        assert_eq!(
            stored.messages[1].reasoning_content.as_deref(),
            Some("First step. Fallback alias.")
        );
    }

    struct PendingDropStream {
        dropped: Arc<std::sync::atomic::AtomicBool>,
    }

    impl Drop for PendingDropStream {
        fn drop(&mut self) {
            self.dropped
                .store(true, std::sync::atomic::Ordering::Release);
        }
    }

    impl futures::Stream for PendingDropStream {
        type Item = Result<Event, std::convert::Infallible>;

        fn poll_next(
            self: std::pin::Pin<&mut Self>,
            _cx: &mut std::task::Context<'_>,
        ) -> std::task::Poll<Option<Self::Item>> {
            std::task::Poll::Pending
        }
    }

    #[tokio::test]
    async fn prefill_phase_disconnect_drops_upstream_body_and_skips_store() {
        let dropped = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let body = Sse::new(PendingDropStream {
            dropped: dropped.clone(),
        })
        .into_response()
        .into_body();
        let (tx, mut rx) = tokio::sync::mpsc::channel(16);
        let response_store =
            crate::response_store::ResponseStore::with_config(16, Duration::from_secs(60));
        let conversation_store =
            crate::conversation_store::ConversationStore::with_config(16, Duration::from_secs(60));

        let handle = tokio::spawn(pump_chat_sse_to_responses(PumpParams {
            body,
            tx,
            response_store: response_store.clone(),
            conversation_store,
            resp_id: "resp_disconnect_prefill".to_string(),
            created_at: 1_700_000_000,
            model: "test-model".to_string(),
            metadata: None,
            store_flag: true,
            input_messages: Vec::new(),
            conversation_id: None,
            conv_new_user_items: Vec::new(),
        }));

        // Drain the two prologue frames so the pump is parked waiting on upstream prefill.
        assert!(rx.recv().await.is_some());
        assert!(rx.recv().await.is_some());
        assert!(!dropped.load(std::sync::atomic::Ordering::Acquire));

        // Simulate client disconnect.
        drop(rx);

        tokio::time::timeout(Duration::from_secs(2), handle)
            .await
            .expect("pump must exit promptly on client disconnect")
            .expect("pump task must not panic");

        assert!(
            dropped.load(std::sync::atomic::Ordering::Acquire),
            "upstream stream must be dropped to cancel scheduler work"
        );
        assert!(
            response_store
                .get(
                    "resp_disconnect_prefill",
                    crate::response_store::StoredKind::Response,
                )
                .is_none(),
            "abandoned stream must not be persisted to response_store"
        );
    }

    #[tokio::test]
    async fn prologue_disconnect_drops_upstream_body_immediately() {
        let dropped = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let body = Sse::new(PendingDropStream {
            dropped: dropped.clone(),
        })
        .into_response()
        .into_body();
        let (tx, rx) = tokio::sync::mpsc::channel(16);
        drop(rx);

        let response_store =
            crate::response_store::ResponseStore::with_config(16, Duration::from_secs(60));
        let conversation_store =
            crate::conversation_store::ConversationStore::with_config(16, Duration::from_secs(60));

        tokio::time::timeout(
            Duration::from_secs(2),
            pump_chat_sse_to_responses(PumpParams {
                body,
                tx,
                response_store: response_store.clone(),
                conversation_store,
                resp_id: "resp_disconnect_prologue".to_string(),
                created_at: 1_700_000_000,
                model: "test-model".to_string(),
                metadata: None,
                store_flag: true,
                input_messages: Vec::new(),
                conversation_id: None,
                conv_new_user_items: Vec::new(),
            }),
        )
        .await
        .expect("pump must exit immediately when receiver is already dropped");

        assert!(dropped.load(std::sync::atomic::Ordering::Acquire));
        assert!(
            response_store
                .get(
                    "resp_disconnect_prologue",
                    crate::response_store::StoredKind::Response,
                )
                .is_none()
        );
    }
}
