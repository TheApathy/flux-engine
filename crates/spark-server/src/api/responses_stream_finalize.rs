// SPDX-License-Identifier: AGPL-3.0-only

#![allow(unused_imports, dead_code)]

//! Prologue, item-closing, and post-stream finalization helpers for the
//! Responses streaming endpoint (`responses_stream.rs`).
//!
//! [`emit_responses_prologue`] opens the SSE lifecycle, [`close_open_items`]
//! closes any in-flight message or function-call output items, and
//! [`finalize_responses_stream_with_stores`] builds the terminal
//! `response.completed` / `response.failed` payload, persists for
//! `previous_response_id` resume, and appends to any linked conversation.

use axum::response::sse::Event;
use tokio::sync::mpsc;

use super::responses_translate::{build_responses_usage, emit};

/// Emit the opening `response.created` and `response.in_progress` SSE
/// frames. Returns `Some(next_seq)` on success, or `None` if the client
/// disconnected (receiver dropped) so the caller can abort immediately.
pub(super) async fn emit_responses_prologue(
    tx: &mpsc::Sender<Result<Event, std::convert::Infallible>>,
    seq_start: u64,
    resp_id: &str,
    created_at: u64,
    model: &str,
    metadata: &Option<std::collections::HashMap<String, String>>,
) -> Option<u64> {
    let mut seq = seq_start;
    // Emit response.created first (status: in_progress).
    let created = crate::openai::ResponsesStreamEvent::Created {
        sequence_number: seq,
        response: crate::openai::ResponsesStreamEnvelope {
            id: resp_id.to_string(),
            object: "response",
            created_at,
            model: model.to_string(),
            status: "in_progress",
            metadata: metadata.clone(),
        },
    };
    if !emit(tx, &created).await {
        return None;
    }
    seq += 1;

    // Per the Responses spec, `response.in_progress` follows
    // `response.created` once generation actually starts. Emit it
    // immediately so spec-strict clients (e.g. the OpenAI Python SDK)
    // don't hang waiting for it.
    let in_progress = crate::openai::ResponsesStreamEvent::InProgress {
        sequence_number: seq,
        response: crate::openai::ResponsesStreamEnvelope {
            id: resp_id.to_string(),
            object: "response",
            created_at,
            model: model.to_string(),
            status: "in_progress",
            metadata: metadata.clone(),
        },
    };
    if !emit(tx, &in_progress).await {
        return None;
    }
    seq += 1;
    Some(seq)
}

/// State for any "in-flight" output item that may need closing once the
/// upstream chat stream drains. Mirrors the locals in
/// `responses_endpoint_stream`'s tail block 1:1.
pub(super) struct CloseOpenCtx<'a> {
    pub seq: u64,
    pub message_started: bool,
    pub message_item_id: &'a str,
    pub content_text: &'a str,
    pub refusal_text: Option<&'a str>,
    pub fc_started: bool,
    pub fc_done: bool,
    pub fc_item_id: Option<String>,
    pub current_tool_call_id: &'a Option<String>,
    pub current_tool_name: &'a Option<String>,
    pub tool_args: &'a str,
    pub output_index: usize,
}

/// Close any open message- or function-call output item left behind when
/// the inner chat stream ended. Emits the matching part-done (`OutputTextDone`,
/// `RefusalDone`, `FunctionCallArgumentsDone`) and `OutputItemDone` SSE events,
/// records the completed `ResponsesOutputItem`s in `completed_items`, and
/// returns `Some(next_seq)`, or `None` if the client disconnected.
pub(super) async fn close_open_items(
    tx: &mpsc::Sender<Result<Event, std::convert::Infallible>>,
    completed_items: &mut Vec<crate::openai::ResponsesOutputItem>,
    ctx: CloseOpenCtx<'_>,
) -> Option<u64> {
    let mut seq = ctx.seq;
    if ctx.message_started {
        let mut content = Vec::new();
        if !ctx.content_text.is_empty() || ctx.refusal_text.is_none() {
            let ev = crate::openai::ResponsesStreamEvent::OutputTextDone {
                sequence_number: seq,
                item_id: ctx.message_item_id.to_string(),
                output_index: ctx.output_index,
                content_index: 0,
                text: ctx.content_text.to_string(),
            };
            if !emit(tx, &ev).await {
                return None;
            }
            seq += 1;
            content.push(crate::openai::ResponsesContentPart::OutputText {
                text: ctx.content_text.to_string(),
                annotations: crate::citation::merged_annotations(ctx.content_text),
            });
        }
        if let Some(refusal) = ctx.refusal_text {
            let content_index = content.len();
            let ev = crate::openai::ResponsesStreamEvent::RefusalDone {
                sequence_number: seq,
                item_id: ctx.message_item_id.to_string(),
                output_index: ctx.output_index,
                content_index,
                refusal: refusal.to_string(),
            };
            if !emit(tx, &ev).await {
                return None;
            }
            seq += 1;
            content.push(crate::openai::ResponsesContentPart::Refusal {
                refusal: refusal.to_string(),
            });
        }
        let done = crate::openai::ResponsesOutputItem::Message {
            id: ctx.message_item_id.to_string(),
            status: "completed",
            role: "assistant",
            content,
        };
        completed_items.push(done.clone());
        let ev = crate::openai::ResponsesStreamEvent::OutputItemDone {
            sequence_number: seq,
            output_index: ctx.output_index,
            item: done,
        };
        if !emit(tx, &ev).await {
            return None;
        }
        seq += 1;
    }
    if ctx.fc_started
        && !ctx.fc_done
        && let Some(fcid) = ctx.fc_item_id.clone()
    {
        let ev = crate::openai::ResponsesStreamEvent::FunctionCallArgumentsDone {
            sequence_number: seq,
            item_id: fcid.clone(),
            output_index: ctx.output_index,
            arguments: ctx.tool_args.to_string(),
        };
        if !emit(tx, &ev).await {
            return None;
        }
        seq += 1;
        let done = crate::openai::ResponsesOutputItem::FunctionCall {
            id: fcid,
            call_id: ctx.current_tool_call_id.clone().unwrap_or_default(),
            name: ctx.current_tool_name.clone().unwrap_or_default(),
            arguments: ctx.tool_args.to_string(),
            status: "completed",
        };
        completed_items.push(done.clone());
        let ev = crate::openai::ResponsesStreamEvent::OutputItemDone {
            sequence_number: seq,
            output_index: ctx.output_index,
            item: done,
        };
        if !emit(tx, &ev).await {
            return None;
        }
        seq += 1;
    }
    Some(seq)
}

/// Inputs collected during the streaming loop that the finalize stage
/// consumes. Bundled into a struct purely to keep the call site readable —
/// each field corresponds 1:1 to a local in `responses_endpoint_stream`'s
/// post-loop block.
pub(super) struct FinalizeCtx {
    pub seq: u64,
    pub completed_items: Vec<crate::openai::ResponsesOutputItem>,
    pub final_usage: Option<serde_json::Value>,
    pub finish_reason: String,
    pub stream_error: Option<serde_json::Value>,
    pub reasoning_text: String,
    pub resp_id: String,
    pub created_at: u64,
    pub model: String,
    pub metadata_for_done: Option<std::collections::HashMap<String, String>>,
    pub store_flag: bool,
    pub input_messages: Vec<crate::openai::IncomingMessage>,
    pub conversation_id: Option<String>,
    pub conv_new_user_items: Vec<serde_json::Value>,
}

/// Emit `response.completed` (or `response.failed` on upstream error), persist
/// the response if `store_flag` is set, and append items to the linked
/// conversation when `conversation_id` is present.
pub(super) async fn finalize_responses_stream_with_stores(
    tx: &mpsc::Sender<Result<Event, std::convert::Infallible>>,
    response_store: &crate::response_store::ResponseStore,
    conversation_store: &crate::conversation_store::ConversationStore,
    ctx: FinalizeCtx,
) {
    let FinalizeCtx {
        seq,
        completed_items,
        final_usage,
        finish_reason,
        stream_error,
        reasoning_text,
        resp_id,
        created_at,
        model,
        metadata_for_done,
        store_flag,
        input_messages,
        conversation_id,
        conv_new_user_items,
    } = ctx;

    // Build the final ResponsesResponse for the `completed` event from
    // the running `completed_items` log so multi-item streams (text→fc→text)
    // surface every emitted item, not just the last live one.
    let final_output = completed_items;
    let usage =
        final_usage
            .as_ref()
            .map(build_responses_usage)
            .unwrap_or(crate::openai::ResponsesUsage {
                input_tokens: 0,
                input_tokens_details: None,
                output_tokens: 0,
                output_tokens_details: None,
                total_tokens: 0,
            });
    let failed_error = if let Some(err) = stream_error {
        Some(err)
    } else if finish_reason == "error" {
        Some(serde_json::json!({
            "code": "server_error",
            "type": "server_error",
            "message": "upstream stream failed",
        }))
    } else {
        None
    };
    let final_status: &'static str = if failed_error.is_some() {
        "failed"
    } else if finish_reason == "length" {
        "incomplete"
    } else {
        "completed"
    };
    let final_resp = crate::openai::ResponsesResponse {
        id: resp_id.clone(),
        object: "response",
        created_at,
        model: model.clone(),
        status: final_status,
        error: failed_error.clone(),
        output: final_output,
        reasoning: None,
        usage,
        metadata: metadata_for_done,
    };

    if let Some(error) = failed_error {
        if store_flag && let Ok(body) = serde_json::to_value(&final_resp) {
            response_store.insert(crate::response_store::StoredEntry {
                id: resp_id.clone(),
                kind: crate::response_store::StoredKind::Response,
                model: model.clone(),
                created_at,
                messages: input_messages,
                body,
                last_access: std::time::Instant::now(),
            });
        }
        let failed = crate::openai::ResponsesStreamEvent::Failed {
            sequence_number: seq,
            response: crate::openai::ResponsesStreamEnvelope {
                id: resp_id,
                object: "response",
                created_at,
                model,
                status: "failed",
                metadata: final_resp.metadata,
            },
            error,
        };
        let _ = emit(tx, &failed).await;
        return;
    }

    // Aggregate every emitted message's text so multi-item streams
    // (text→fc→text) round-trip cleanly through previous_response_id
    // resume and the Conversations API.
    let mut transcript_text = String::new();
    for item in &final_resp.output {
        if let crate::openai::ResponsesOutputItem::Message { content, .. } = item {
            for part in content {
                if let crate::openai::ResponsesContentPart::OutputText { text, .. } = part {
                    transcript_text.push_str(text);
                }
            }
        }
    }
    // Persist for previous_response_id resume (Responses API defaults
    // to store=true; we respect the caller's store field).
    if store_flag && let Ok(body) = serde_json::to_value(&final_resp) {
        let mut transcript = input_messages;
        // Rebuild structured tool_calls from the accumulated
        // function-call output items so previous_response_id
        // resume sees the model's prior tool calls. Without
        // this, multi-turn streaming Responses chains lose
        // every function_call across turns and the model
        // re-issues them on resume.
        let stored_tool_calls: Vec<crate::tool_parser::IncomingToolCall> = final_resp
            .output
            .iter()
            .filter_map(|item| match item {
                crate::openai::ResponsesOutputItem::FunctionCall {
                    call_id,
                    name,
                    arguments,
                    ..
                } => Some(crate::tool_parser::IncomingToolCall {
                    id: Some(call_id.clone()),
                    function: crate::tool_parser::IncomingFunction {
                        name: name.clone(),
                        arguments: arguments.clone(),
                    },
                }),
                _ => None,
            })
            .collect();
        let stored_reasoning = (!reasoning_text.is_empty()).then_some(reasoning_text);
        if !transcript_text.is_empty()
            || !stored_tool_calls.is_empty()
            || stored_reasoning.is_some()
        {
            transcript.push(crate::openai::IncomingMessage {
                role: "assistant".to_string(),
                content: crate::openai::ParsedContent {
                    text: transcript_text.clone(),
                    images: Vec::new(),
                    image_text_offsets: Vec::new(),
                },
                reasoning_content: stored_reasoning,
                tool_calls: if stored_tool_calls.is_empty() {
                    None
                } else {
                    Some(stored_tool_calls)
                },
                tool_call_id: None,
                name: None,
            });
        }
        response_store.insert(crate::response_store::StoredEntry {
            id: resp_id.clone(),
            kind: crate::response_store::StoredKind::Response,
            model: model.clone(),
            created_at,
            messages: transcript,
            body,
            last_access: std::time::Instant::now(),
        });
    }
    // Append the new user items + assistant output to the linked
    // conversation (best-effort; silent on failure so the primary
    // SSE stream is never disrupted by conversation-store issues).
    if let Some(cid) = conversation_id.as_ref() {
        let mut batch = conv_new_user_items.clone();
        batch.extend(
            final_resp
                .output
                .iter()
                .filter_map(|item| serde_json::to_value(item).ok()),
        );
        if !batch.is_empty()
            && let Err(e) = conversation_store.add_items(cid, batch)
        {
            tracing::warn!(
                "responses_stream_finalize: conversation_store.add_items failed for {cid}: {e:?}"
            );
        }
    }
    let completed = crate::openai::ResponsesStreamEvent::Completed {
        sequence_number: seq,
        response: final_resp,
    };
    let _ = emit(tx, &completed).await;
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::response::{IntoResponse, Sse};
    use std::time::Duration;
    use tokio_stream::wrappers::ReceiverStream;

    async fn collect_events(
        rx: mpsc::Receiver<Result<Event, std::convert::Infallible>>,
    ) -> Vec<(String, serde_json::Value)> {
        let resp = Sse::new(ReceiverStream::new(rx)).into_response();
        let body = axum::body::to_bytes(resp.into_body(), 1 << 20)
            .await
            .expect("collect SSE body");
        let text = String::from_utf8(body.to_vec()).expect("valid UTF-8 SSE");
        let mut out = Vec::new();
        let mut ev_type = String::new();
        let mut data = String::new();
        for line in text.lines() {
            if line.is_empty() {
                if !data.is_empty() {
                    let json: serde_json::Value =
                        serde_json::from_str(&data).expect("valid JSON SSE data");
                    out.push((std::mem::take(&mut ev_type), json));
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
            out.push((ev_type, json));
        }
        out
    }

    #[tokio::test]
    async fn failed_finalize_emits_response_failed_and_skips_conversation_append() {
        let (tx, rx) = mpsc::channel(16);
        let response_store =
            crate::response_store::ResponseStore::with_config(16, Duration::from_secs(60));
        let conversation_store =
            crate::conversation_store::ConversationStore::with_config(16, Duration::from_secs(60));
        let conv_id = conversation_store.create(Vec::new(), std::collections::HashMap::new());

        finalize_responses_stream_with_stores(
            &tx,
            &response_store,
            &conversation_store,
            FinalizeCtx {
                seq: 3,
                completed_items: Vec::new(),
                final_usage: None,
                finish_reason: "error".to_string(),
                stream_error: Some(serde_json::json!({
                    "message": "decode timeout",
                    "type": "server_error",
                    "code": 500,
                })),
                reasoning_text: "partial thought".to_string(),
                resp_id: "resp_fail".to_string(),
                created_at: 1_700_000_000,
                model: "test-model".to_string(),
                metadata_for_done: None,
                store_flag: true,
                input_messages: vec![crate::openai::IncomingMessage::synthetic_user_text(
                    "hi".to_string(),
                )],
                conversation_id: Some(conv_id.clone()),
                conv_new_user_items: vec![serde_json::json!({
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "hi"}],
                })],
            },
        )
        .await;
        drop(tx);

        let events = collect_events(rx).await;
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].0, "response.failed");
        assert_eq!(events[0].1["sequence_number"], 3);
        assert_eq!(events[0].1["response"]["status"], "failed");
        assert_eq!(events[0].1["error"]["message"], "decode timeout");

        let stored = response_store
            .get("resp_fail", crate::response_store::StoredKind::Response)
            .expect("failed response stored for retrieval");
        assert_eq!(stored.body["status"], "failed");
        assert_eq!(stored.body["error"]["message"], "decode timeout");
        assert_eq!(
            stored.messages.len(),
            1,
            "failed turn must not append a partial assistant message"
        );

        let conv = conversation_store.get(&conv_id).expect("conversation");
        assert!(
            conv.items.is_empty(),
            "failed turn must not append items to conversation"
        );
    }

    #[tokio::test]
    async fn reasoning_only_stream_persists_assistant_reasoning_content() {
        let (tx, rx) = mpsc::channel(16);
        let response_store =
            crate::response_store::ResponseStore::with_config(16, Duration::from_secs(60));
        let conversation_store =
            crate::conversation_store::ConversationStore::with_config(16, Duration::from_secs(60));

        finalize_responses_stream_with_stores(
            &tx,
            &response_store,
            &conversation_store,
            FinalizeCtx {
                seq: 2,
                completed_items: Vec::new(),
                final_usage: None,
                finish_reason: "length".to_string(),
                stream_error: None,
                reasoning_text: "deep chain of thought".to_string(),
                resp_id: "resp_think".to_string(),
                created_at: 1_700_000_000,
                model: "test-model".to_string(),
                metadata_for_done: None,
                store_flag: true,
                input_messages: vec![crate::openai::IncomingMessage::synthetic_user_text(
                    "think".to_string(),
                )],
                conversation_id: None,
                conv_new_user_items: Vec::new(),
            },
        )
        .await;
        drop(tx);

        let events = collect_events(rx).await;
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].0, "response.completed");
        assert_eq!(events[0].1["response"]["status"], "incomplete");
        assert!(events[0].1["response"].get("reasoning").is_none());

        let stored = response_store
            .get("resp_think", crate::response_store::StoredKind::Response)
            .expect("response stored");
        assert_eq!(stored.messages.len(), 2);
        assert_eq!(stored.messages[1].role, "assistant");
        assert_eq!(
            stored.messages[1].reasoning_content.as_deref(),
            Some("deep chain of thought")
        );
    }
}
