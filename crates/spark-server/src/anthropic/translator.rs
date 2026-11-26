// SPDX-License-Identifier: AGPL-3.0-only

use axum::response::sse::Event;

use super::helpers::*;

/// Open-block tracker for the streaming translator's state machine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum OpenBlock {
    /// No block currently open. Either before the first delta arrives
    /// or right after a `content_block_stop`.
    None,
    /// A text content block is open at index `block_idx`.
    Text,
    /// A `thinking` content block is open at index `block_idx`. Atlas
    /// emits its reasoning trace as `delta.reasoning_content` chunks
    /// inside the OpenAI stream; we map those to Anthropic's
    /// `content_block_start{type:"thinking"}` + `content_block_delta
    /// {delta.type:"thinking_delta"}` events so Claude Code can show
    /// the model's thinking progress in real time. Without this,
    /// Claude Code displayed only "Brewed for Ns" with no progress
    /// for the entire thinking phase, and users cancelled because
    /// they thought the server had stalled (2026-04-25 incident).
    Thinking,
    /// A tool_use block is open at index `block_idx` for the given
    /// OpenAI tool-call index (the `index` field on
    /// `delta.tool_calls[]`). We track it so subsequent argument
    /// fragments for the same OpenAI index get routed into the same
    /// Anthropic block, and so we know to close+reopen if a new
    /// OpenAI index appears (multi-tool turn).
    ToolUse(usize),
}

/// State machine kept alive for the lifetime of one /v1/messages
/// streaming response. Each incoming OpenAI chunk JSON is fed in via
/// `process_openai_chunk`, which pushes zero or more Anthropic
/// `Event`s into the shared sender. The state ensures correct
/// `content_block_{start,stop}` framing across:
/// - text blocks (single per turn, but produced over many OpenAI
///   `delta.content` chunks)
/// - tool_use blocks (one per OpenAI tool-call index, args streamed
///   as `input_json_delta` partials)
/// - mixed turns (text first, then tool_use; or vice versa)
/// - `finish_reason` chunk (closes any still-open block, emits
///   `message_delta` + `message_stop`)
pub struct AnthropicTranslator {
    model: String,
    msg_started: bool,
    msg_id: String,
    block_idx: u32,
    open_block: OpenBlock,
    /// Per-OpenAI-tool-index: have we emitted `content_block_start`
    /// for this tool yet? Needed because the OpenAI stream may emit
    /// the `id`+`name` (start fragment) once and then many
    /// `arguments` fragments — each subsequent `delta.tool_calls[i]`
    /// shouldn't re-trigger another `content_block_start`.
    tool_started: std::collections::HashMap<usize, ()>,
    completion_tokens: usize,
    prompt_tokens: usize,
    /// Deferred OpenAI `finish_reason` when the finish chunk arrived without
    /// `usage`, awaiting the terminal `usage_only_chunk` (or `finalize()`).
    pending_finish_reason: Option<String>,
    finished: bool,
}

impl AnthropicTranslator {
    pub(super) fn new(model: String) -> Self {
        Self {
            model,
            msg_started: false,
            msg_id: String::new(),
            block_idx: 0,
            open_block: OpenBlock::None,
            tool_started: std::collections::HashMap::new(),
            completion_tokens: 0,
            prompt_tokens: 0,
            pending_finish_reason: None,
            finished: false,
        }
    }

    fn make_event(ev_type: &str, data: serde_json::Value) -> Event {
        let json_str = serde_json::to_string(&data).unwrap_or_default();
        Event::default().event(ev_type).data(json_str)
    }

    /// Record token counts from `val["usage"]` when present.
    /// Returns `true` when `completion_tokens` was observed on this chunk.
    fn capture_usage(&mut self, val: &serde_json::Value) -> bool {
        let Some(u) = val.get("usage").and_then(serde_json::Value::as_object) else {
            return false;
        };
        if let Some(p) = u.get("prompt_tokens").and_then(serde_json::Value::as_u64) {
            self.prompt_tokens = p as usize;
        }
        if let Some(c) = u
            .get("completion_tokens")
            .and_then(serde_json::Value::as_u64)
        {
            self.completion_tokens = c as usize;
            true
        } else {
            false
        }
    }

    fn emit_finish(&mut self, finish_reason: &str, out: &mut Vec<Event>) {
        self.ensure_message_start(out);
        if let Some(stop) = self.close_open_block() {
            out.push(stop);
        }
        out.push(Self::make_event(
            "message_delta",
            serde_json::json!({
                "type": "message_delta",
                "delta": {
                    "stop_reason": convert_stop_reason(finish_reason),
                    "stop_sequence": serde_json::Value::Null,
                },
                "usage": {"output_tokens": self.completion_tokens},
            }),
        ));
        out.push(Self::make_event(
            "message_stop",
            serde_json::json!({"type": "message_stop"}),
        ));
        self.finished = true;
    }

    /// Close the currently open block (if any) and increment the
    /// block index. Returns the matching `content_block_stop` event,
    /// or `None` if no block was open.
    ///
    /// When the closed block is a `ToolUse(oa_idx)`, also remove
    /// `oa_idx` from `tool_started` so a fresh start chunk carrying a
    /// non-empty `name` for that same OpenAI tool index can open a new
    /// `tool_use` block cleanly. Late argument-only fragments (empty
    /// `name`) for a closed block are dropped instead of re-opening a
    /// duplicate block; interleaved fragments for a still-tracked
    /// different index are dropped with a warning below.
    pub(super) fn close_open_block(&mut self) -> Option<Event> {
        match self.open_block {
            OpenBlock::None => None,
            OpenBlock::Text | OpenBlock::Thinking => {
                let ev = Self::make_event(
                    "content_block_stop",
                    serde_json::json!({
                        "type": "content_block_stop",
                        "index": self.block_idx,
                    }),
                );
                self.open_block = OpenBlock::None;
                self.block_idx += 1;
                Some(ev)
            }
            OpenBlock::ToolUse(oa_idx) => {
                self.tool_started.remove(&oa_idx);
                let ev = Self::make_event(
                    "content_block_stop",
                    serde_json::json!({
                        "type": "content_block_stop",
                        "index": self.block_idx,
                    }),
                );
                self.open_block = OpenBlock::None;
                self.block_idx += 1;
                Some(ev)
            }
        }
    }

    pub(super) fn ensure_message_start(&mut self, out: &mut Vec<Event>) {
        if self.msg_started {
            return;
        }
        let id = if self.msg_id.is_empty() {
            "msg_unknown".to_string()
        } else {
            format!("msg_{}", self.msg_id.trim_start_matches("chatcmpl-"))
        };
        out.push(Self::make_event(
            "message_start",
            serde_json::json!({
                "type": "message_start",
                "message": {
                    "id": id,
                    "type": "message",
                    "role": "assistant",
                    "model": self.model,
                    "content": [],
                    "stop_reason": serde_json::Value::Null,
                    "stop_sequence": serde_json::Value::Null,
                    "usage": {
                        "input_tokens": self.prompt_tokens,
                        "output_tokens": 0,
                    },
                },
            }),
        ));
        self.msg_started = true;
    }

    /// Process one OpenAI chat-completion chunk JSON value, push
    /// resulting Anthropic events into `out`.
    pub(super) fn process_openai_chunk(&mut self, val: &serde_json::Value, out: &mut Vec<Event>) {
        if self.finished {
            return;
        }

        // Propagate mid-stream OpenAI `{"error": ...}` chunks before inspecting
        // `choices` so upstream faults surface as Anthropic `event: error`
        // rather than silently terminating with a bogus `end_turn`.
        if let Some(err) = val.get("error").filter(|v| !v.is_null()) {
            let message = err
                .get("message")
                .and_then(serde_json::Value::as_str)
                .or_else(|| err.as_str())
                .filter(|s| !s.is_empty())
                .map_or_else(|| err.to_string(), str::to_string);
            out.push(Self::make_event(
                "error",
                serde_json::json!({
                    "type": "error",
                    "error": {
                        "type": "api_error",
                        "message": message,
                    },
                }),
            ));
            self.pending_finish_reason = None;
            self.finished = true;
            return;
        }

        // Pick up the chat-completion id on the first chunk that
        // carries it (typically the role chunk).
        if self.msg_id.is_empty()
            && let Some(s) = val.get("id").and_then(|v| v.as_str())
        {
            self.msg_id = s.to_string();
        }

        // Capture `usage` whenever it appears (on role_chunk when continuous
        // usage is enabled, on choice chunks, or on the terminal usage-only chunk).
        let saw_completion_usage = self.capture_usage(val);

        let Some(choice) = val.get("choices").and_then(|c| c.get(0)) else {
            // Some OpenAI chunks (the separate-usage chunk) carry
            // no `choices`. If a prior chunk deferred `finish_reason`
            // waiting for usage, flush `message_delta` + `message_stop` now.
            if saw_completion_usage && let Some(fr) = self.pending_finish_reason.take() {
                self.emit_finish(&fr, out);
            }
            return;
        };

        let delta = choice.get("delta");

        // Lazy message_start. We delay it until we have something
        // meaningful (the role chunk) so the emitted `prompt_tokens`
        // reflect the chat-completion's reported input usage if the
        // first chunk happens to carry it. Most servers send role +
        // usage in the first / penultimate chunks respectively.
        let has_delta_signal = delta
            .and_then(|d| d.as_object())
            .map(|o| {
                o.contains_key("role")
                    || o.contains_key("content")
                    || o.contains_key("tool_calls")
                    || o.contains_key("reasoning_content")
            })
            .unwrap_or(false);
        if has_delta_signal {
            self.ensure_message_start(out);
        }

        // Reasoning / thinking delta — Atlas emits these as
        // `delta.reasoning_content` chunks during the model's
        // thinking phase. Anthropic's streaming spec wraps thinking
        // in its own content block (`content_block.type="thinking"`)
        // with `delta.type="thinking_delta"`. Without this mapping
        // Claude Code shows "Brewed for Ns" with no visible progress
        // for the entire thinking phase, and users cancel thinking
        // the server has stalled (2026-04-25 incident).
        if let Some(d) = delta
            && let Some(text) = d.get("reasoning_content").and_then(|v| v.as_str())
            && !text.is_empty()
        {
            if !matches!(self.open_block, OpenBlock::Thinking) {
                if let Some(stop) = self.close_open_block() {
                    out.push(stop);
                }
                out.push(Self::make_event(
                    "content_block_start",
                    serde_json::json!({
                        "type": "content_block_start",
                        "index": self.block_idx,
                        "content_block": {"type": "thinking", "thinking": ""},
                    }),
                ));
                self.open_block = OpenBlock::Thinking;
            }
            out.push(Self::make_event(
                "content_block_delta",
                serde_json::json!({
                    "type": "content_block_delta",
                    "index": self.block_idx,
                    "delta": {"type": "thinking_delta", "thinking": text},
                }),
            ));
        }

        // Text content delta.
        if let Some(d) = delta {
            if let Some(text) = d.get("content").and_then(|v| v.as_str())
                && !text.is_empty()
            {
                if !matches!(self.open_block, OpenBlock::Text) {
                    if let Some(stop) = self.close_open_block() {
                        out.push(stop);
                    }
                    out.push(Self::make_event(
                        "content_block_start",
                        serde_json::json!({
                            "type": "content_block_start",
                            "index": self.block_idx,
                            "content_block": {"type": "text", "text": ""},
                        }),
                    ));
                    self.open_block = OpenBlock::Text;
                }
                out.push(Self::make_event(
                    "content_block_delta",
                    serde_json::json!({
                        "type": "content_block_delta",
                        "index": self.block_idx,
                        "delta": {"type": "text_delta", "text": text},
                    }),
                ));
            }

            // Tool-call deltas. OpenAI emits these as an array; each
            // entry has `index` (which OpenAI tool-call slot it
            // refers to) plus optionally `id`, `function.name`,
            // `function.arguments` (a string fragment).
            if let Some(tcs) = d.get("tool_calls").and_then(|v| v.as_array()) {
                for tc in tcs {
                    let oa_idx = tc.get("index").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                    let id = tc
                        .get("id")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    let function = tc.get("function");
                    let name = function
                        .and_then(|f| f.get("name"))
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    let arguments_fragment = function
                        .and_then(|f| f.get("arguments"))
                        .and_then(|v| v.as_str())
                        .unwrap_or("");

                    let need_start = !self.tool_started.contains_key(&oa_idx);
                    if need_start {
                        // Only open a new block once we have at least
                        // a name (the model's first emit per tool
                        // typically carries `id`+`name` together).
                        if name.is_empty() {
                            // Drop argument-only fragments until a start chunk
                            // carrying a non-empty `name` arrives.
                            continue;
                        }
                        if !matches!(self.open_block, OpenBlock::ToolUse(idx) if idx == oa_idx) {
                            if let Some(stop) = self.close_open_block() {
                                out.push(stop);
                            }
                            out.push(Self::make_event(
                                "content_block_start",
                                serde_json::json!({
                                    "type": "content_block_start",
                                    "index": self.block_idx,
                                    "content_block": {
                                        "type": "tool_use",
                                        "id": id,
                                        "name": name,
                                        "input": {},
                                    },
                                }),
                            ));
                            self.open_block = OpenBlock::ToolUse(oa_idx);
                            self.tool_started.insert(oa_idx, ());
                        }
                    } else if !matches!(self.open_block, OpenBlock::ToolUse(idx) if idx == oa_idx) {
                        // Should be unreachable now that
                        // `close_open_block` clears `tool_started`
                        // for `ToolUse` (F3, 2026-04-26): if the
                        // block were closed, `need_start` would be
                        // true and we'd have taken the branch above.
                        // The only way to reach here is the upstream
                        // OpenAI stream interleaving args for tool
                        // index `oa_idx` while a *different* tool
                        // block is open without the prior block ever
                        // being closed — protocol-level upstream
                        // bug, not something we can synthesise a
                        // safe fix for. Drop the late fragment with
                        // a warning rather than emit a duplicate
                        // `tool_use` block (which Claude Code would
                        // execute twice).
                        tracing::warn!(
                            target: "anthropic_translator",
                            oa_idx,
                            current_block = ?self.open_block,
                            arg_fragment_len = arguments_fragment.len(),
                            "dropping interleaved tool-call argument fragment for already-open different tool block"
                        );
                        continue;
                    }

                    if !arguments_fragment.is_empty() {
                        out.push(Self::make_event(
                            "content_block_delta",
                            serde_json::json!({
                                "type": "content_block_delta",
                                "index": self.block_idx,
                                "delta": {
                                    "type": "input_json_delta",
                                    "partial_json": arguments_fragment,
                                },
                            }),
                        ));
                    }
                }
            }
        }

        // Finish reason: close the current block (if any) and emit
        // message_delta + message_stop. When `finish_reason` arrives without
        // `usage` (e.g. OpenAI `include_usage=true` layout), defer emission
        // until the terminal `usage_only_chunk` (or `finalize()`) so
        // `output_tokens` is accurate.
        if let Some(fr) = choice.get("finish_reason").and_then(|v| v.as_str())
            && !self.finished
        {
            if saw_completion_usage {
                self.pending_finish_reason = None;
                self.emit_finish(fr, out);
            } else {
                self.pending_finish_reason = Some(fr.to_string());
            }
        }
    }

    /// Stream ended without an explicit `finish_reason` (or with a deferred
    /// `finish_reason` whose trailing usage chunk never arrived). Best-effort
    /// flush so the client sees a coherent end of message.
    pub(super) fn finalize(&mut self, out: &mut Vec<Event>) {
        if self.finished {
            return;
        }
        let finish_reason = self.pending_finish_reason.take();
        self.emit_finish(finish_reason.as_deref().unwrap_or("stop"), out);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::response::{IntoResponse, Sse};

    async fn render_sse_events(events: Vec<Event>) -> Vec<(String, serde_json::Value)> {
        let stream =
            futures::stream::iter(events.into_iter().map(Ok::<_, std::convert::Infallible>));
        let resp = Sse::new(stream).into_response();
        let body = axum::body::to_bytes(resp.into_body(), 1 << 20)
            .await
            .expect("collect SSE body");
        let text = String::from_utf8(body.to_vec()).expect("valid UTF-8 SSE");

        let mut parsed = Vec::new();
        let mut ev_type = String::new();
        let mut data = String::new();
        for line in text.lines() {
            if line.is_empty() {
                if !data.is_empty() {
                    let json: serde_json::Value =
                        serde_json::from_str(&data).expect("valid JSON SSE data");
                    parsed.push((std::mem::take(&mut ev_type), json));
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
            parsed.push((ev_type, json));
        }
        parsed
    }

    #[tokio::test]
    async fn role_chunk_sets_input_tokens_and_deferred_usage_sets_output_tokens() {
        let mut tr = AnthropicTranslator::new("claude-sonnet-4-6".to_string());
        let mut events = Vec::new();

        tr.process_openai_chunk(
            &serde_json::json!({
                "id": "chatcmpl-usage1",
                "choices": [{"delta": {"role": "assistant"}}],
                "usage": {"prompt_tokens": 42, "completion_tokens": 0, "total_tokens": 42},
            }),
            &mut events,
        );
        tr.process_openai_chunk(
            &serde_json::json!({
                "id": "chatcmpl-usage1",
                "choices": [{"delta": {"content": "hello"}}],
            }),
            &mut events,
        );
        // Finish chunk without usage must defer message_delta + message_stop.
        tr.process_openai_chunk(
            &serde_json::json!({
                "id": "chatcmpl-usage1",
                "choices": [{"delta": {}, "finish_reason": "stop"}],
            }),
            &mut events,
        );
        assert_eq!(
            events.len(),
            3,
            "finish without usage should defer message_delta and message_stop"
        );

        // Terminal usage-only chunk triggers deferred close + accurate output_tokens.
        tr.process_openai_chunk(
            &serde_json::json!({
                "id": "chatcmpl-usage1",
                "choices": [],
                "usage": {"prompt_tokens": 42, "completion_tokens": 17, "total_tokens": 59},
            }),
            &mut events,
        );
        // Subsequent finalize() must be a no-op.
        tr.finalize(&mut events);

        let rendered = render_sse_events(events).await;
        assert_eq!(rendered.len(), 6);
        assert_eq!(rendered[0].0, "message_start");
        assert_eq!(rendered[0].1["message"]["id"], "msg_usage1");
        assert_eq!(rendered[0].1["message"]["usage"]["input_tokens"], 42);
        assert_eq!(rendered[0].1["message"]["usage"]["output_tokens"], 0);
        assert_eq!(rendered[1].0, "content_block_start");
        assert_eq!(rendered[2].0, "content_block_delta");
        assert_eq!(rendered[3].0, "content_block_stop");
        assert_eq!(rendered[4].0, "message_delta");
        assert_eq!(rendered[4].1["delta"]["stop_reason"], "end_turn");
        assert_eq!(rendered[4].1["usage"]["output_tokens"], 17);
        assert_eq!(rendered[5].0, "message_stop");
    }

    #[tokio::test]
    async fn deferred_finish_reason_is_preserved_when_stream_ends_without_usage_chunk() {
        let mut tr = AnthropicTranslator::new("claude-sonnet-4-6".to_string());
        let mut events = Vec::new();

        tr.process_openai_chunk(
            &serde_json::json!({
                "id": "chatcmpl-tool",
                "choices": [{"delta": {"role": "assistant"}}],
            }),
            &mut events,
        );
        tr.process_openai_chunk(
            &serde_json::json!({
                "id": "chatcmpl-tool",
                "choices": [{"delta": {
                    "tool_calls": [{
                        "index": 0,
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "search", "arguments": "{\"q\":\"x\"}"}
                    }]
                }}],
            }),
            &mut events,
        );
        tr.process_openai_chunk(
            &serde_json::json!({
                "id": "chatcmpl-tool",
                "choices": [{"delta": {}, "finish_reason": "tool_calls"}],
            }),
            &mut events,
        );
        tr.finalize(&mut events);

        let rendered = render_sse_events(events).await;
        assert_eq!(rendered.len(), 6);
        assert_eq!(rendered[4].0, "message_delta");
        assert_eq!(rendered[4].1["delta"]["stop_reason"], "tool_use");
        assert_eq!(rendered[5].0, "message_stop");
    }

    #[tokio::test]
    async fn mid_stream_error_emits_anthropic_error_and_suppresses_finalize() {
        let mut tr = AnthropicTranslator::new("claude-sonnet-4-6".to_string());
        let mut events = Vec::new();

        tr.process_openai_chunk(
            &serde_json::json!({
                "id": "chatcmpl-err",
                "choices": [{"delta": {"role": "assistant"}}],
                "usage": {"prompt_tokens": 11, "completion_tokens": 0, "total_tokens": 11},
            }),
            &mut events,
        );
        tr.process_openai_chunk(
            &serde_json::json!({
                "id": "chatcmpl-err",
                "choices": [{"delta": {"content": "partial"}}],
            }),
            &mut events,
        );
        tr.process_openai_chunk(
            &serde_json::json!({
                "error": {"message": "scheduler OOM", "type": "server_error", "code": 500}
            }),
            &mut events,
        );
        // Trailing chunks and finalize() after an error must not emit end_turn / message_stop.
        tr.process_openai_chunk(
            &serde_json::json!({
                "id": "chatcmpl-err",
                "choices": [{"delta": {}, "finish_reason": "stop"}],
            }),
            &mut events,
        );
        tr.finalize(&mut events);

        let rendered = render_sse_events(events).await;
        assert_eq!(rendered.len(), 4);
        assert_eq!(rendered[0].0, "message_start");
        assert_eq!(rendered[1].0, "content_block_start");
        assert_eq!(rendered[2].0, "content_block_delta");
        assert_eq!(rendered[3].0, "error");
        assert_eq!(rendered[3].1["type"], "error");
        assert_eq!(rendered[3].1["error"]["type"], "api_error");
        assert_eq!(rendered[3].1["error"]["message"], "scheduler OOM");
        assert!(!rendered.iter().any(|(ev, _)| ev == "message_delta"));
        assert!(!rendered.iter().any(|(ev, _)| ev == "message_stop"));
    }
}
