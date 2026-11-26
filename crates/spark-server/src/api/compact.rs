// SPDX-License-Identifier: AGPL-3.0-only

//! Progressive context compaction + shared OpenAI-compatible error helpers
//! (extracted from `api.rs`, lines 20-225).

use axum::http::StatusCode;
use axum::response::{IntoResponse, Json, Response};

/// Progressive context compaction (stages 2–5, per arXiv:2603.05344 OpenDev).
///
/// Uses actual `prompt_tokens` (from trial tokenization) over `max_seq_len` to
/// select the appropriate compaction stage. Always keeps system message + a
/// recent tail of messages.
///
/// - Stage 2 (`< 80%`): Truncate middle messages (`> 500` bytes and `> 6` lines)
///   to first + last 3 lines with `... [N lines truncated] ...` (keeps system + last 6)
/// - Stage 3 (`80%..85%`): Replace middle `tool` and `user` messages (`> 200` bytes)
///   with `"[Tool output truncated — {len} chars]"` (keeps system + last 6)
/// - Stage 4 (`85%..95%`): Drop oldest middle messages, keeping system + last 6
///   (expanded backward if needed to include a real user query)
/// - Stage 5 (`>= 95%`): Trim system prompt (`> 4000` bytes -> head ~2000 + tail ~1000)
///   + keep last 4 messages (expanded backward if needed for a real user query)
pub fn compact_messages(
    msgs: &[serde_json::Value],
    prompt_tokens: usize,
    max_seq_len: usize,
) -> Vec<serde_json::Value> {
    let ratio = prompt_tokens as f32 / max_seq_len as f32;

    let (result, stage) = if ratio < 0.80 {
        // Stage 2: truncate long tool responses in middle messages
        let keep_tail = 6.min(msgs.len());
        let tail_start = msgs.len().saturating_sub(keep_tail);
        let mut out = Vec::with_capacity(msgs.len());
        for (i, msg) in msgs.iter().enumerate() {
            if i == 0 || i >= tail_start {
                out.push(msg.clone());
            } else {
                let content = msg["content"].as_str().unwrap_or("");
                if content.len() > 500 {
                    let lines: Vec<&str> = content.lines().collect();
                    let truncated = if lines.len() > 6 {
                        format!(
                            "{}\n... [{} lines truncated] ...\n{}",
                            lines[..3].join("\n"),
                            lines.len() - 6,
                            lines[lines.len() - 3..].join("\n")
                        )
                    } else {
                        content.to_string()
                    };
                    let mut m = msg.clone();
                    m["content"] = serde_json::Value::String(truncated);
                    out.push(m);
                } else {
                    out.push(msg.clone());
                }
            }
        }
        (out, 2)
    } else if ratio < 0.85 {
        // Stage 3: mask observations — replace tool response content with pointer
        let keep_tail = 6.min(msgs.len());
        let tail_start = msgs.len().saturating_sub(keep_tail);
        let mut out = Vec::with_capacity(msgs.len());
        for (i, msg) in msgs.iter().enumerate() {
            if i == 0 || i >= tail_start {
                out.push(msg.clone());
            } else {
                let role = msg["role"].as_str().unwrap_or("");
                let content = msg["content"].as_str().unwrap_or("");
                if (role == "tool" || role == "user") && content.len() > 200 {
                    let mut m = msg.clone();
                    m["content"] = serde_json::Value::String(format!(
                        "[Tool output truncated — {} chars]",
                        content.len()
                    ));
                    out.push(m);
                } else {
                    out.push(msg.clone());
                }
            }
        }
        (out, 3)
    } else if ratio < 0.95 {
        // Stage 4: drop oldest middle messages, keep system + last 6
        // Ensure tail starts on a user message (not tool/assistant) to avoid
        // Jinja "No user query found" error and orphaned tool_response messages.
        let keep_tail = 6.min(msgs.len().saturating_sub(1));
        let mut tail_start = msgs.len().saturating_sub(keep_tail);
        // Walk backward to find a real user message in the tail
        let has_user_query = (tail_start..msgs.len()).any(|i| {
            let role = msgs[i]["role"].as_str().unwrap_or("");
            let content = msgs[i]["content"].as_str().unwrap_or("");
            role == "user" && !content.starts_with("<tool_response>")
        });
        if !has_user_query {
            // Expand tail backwards until we find a real user message
            while tail_start > 1 {
                tail_start -= 1;
                let role = msgs[tail_start]["role"].as_str().unwrap_or("");
                let content = msgs[tail_start]["content"].as_str().unwrap_or("");
                if role == "user" && !content.starts_with("<tool_response>") {
                    break;
                }
            }
        }
        // Don't start tail on a "tool" message — it needs a preceding assistant
        while tail_start < msgs.len() && msgs[tail_start]["role"].as_str() == Some("tool") {
            tail_start += 1;
        }
        let mut out = Vec::with_capacity(msgs.len() - tail_start + 1);
        out.push(msgs[0].clone()); // system
        for msg in &msgs[tail_start..] {
            out.push(msg.clone());
        }
        (out, 4)
    } else {
        // Stage 5: trim system prompt + keep only last 4 messages
        // Same safety: ensure a real user message is present and no orphaned tool messages.
        let keep_tail = 4.min(msgs.len().saturating_sub(1));
        let mut tail_start = msgs.len().saturating_sub(keep_tail);
        let has_user_query = (tail_start..msgs.len()).any(|i| {
            let role = msgs[i]["role"].as_str().unwrap_or("");
            let content = msgs[i]["content"].as_str().unwrap_or("");
            role == "user" && !content.starts_with("<tool_response>")
        });
        if !has_user_query {
            while tail_start > 1 {
                tail_start -= 1;
                let role = msgs[tail_start]["role"].as_str().unwrap_or("");
                let content = msgs[tail_start]["content"].as_str().unwrap_or("");
                if role == "user" && !content.starts_with("<tool_response>") {
                    break;
                }
            }
        }
        while tail_start < msgs.len() && msgs[tail_start]["role"].as_str() == Some("tool") {
            tail_start += 1;
        }
        let mut out = Vec::with_capacity(msgs.len() - tail_start + 1);
        // Trim system prompt: keep first ~2000 + last ~1000 chars.
        // Use floor/ceil_char_boundary to avoid panics on multi-byte UTF-8.
        let sys_content = msgs[0]["content"].as_str().unwrap_or("");
        let trimmed_sys = if sys_content.len() > 4000 {
            let head_end = sys_content.floor_char_boundary(2000);
            let tail_start = sys_content.ceil_char_boundary(sys_content.len().saturating_sub(1000));
            format!(
                "{}...\n[System prompt truncated — {} chars removed]\n...{}",
                &sys_content[..head_end],
                sys_content.len() - head_end - (sys_content.len() - tail_start),
                &sys_content[tail_start..]
            )
        } else {
            sys_content.to_string()
        };
        let mut sys = msgs[0].clone();
        sys["content"] = serde_json::Value::String(trimmed_sys);
        out.push(sys);
        for msg in &msgs[tail_start..] {
            out.push(msg.clone());
        }
        (out, 5)
    };

    tracing::info!(
        "Auto-compact stage {}: {} → {} messages (was {:.0}% of {})",
        stage,
        msgs.len(),
        result.len(),
        ratio * 100.0,
        max_seq_len,
    );
    result
}

/// Opt-in hard context-overflow truncation helper.
///
/// Distinct from [`compact_messages`], which implements progressive
/// auto-compaction (also disabled by default; see `template.rs`). When
/// enabled, this last-resort truncation fires only when the rendered prompt
/// would not leave `output_reserve` tokens under `max_seq_len`.
///
/// It drops the oldest middle messages, always keeping the system message
/// (index 0) plus the most recent `keep_tail` messages, and guarantees the
/// retained tail begins on a real user query (not an orphaned `tool` /
/// `<tool_response>` message) so the Jinja template cannot raise
/// "No user query found". Returns `None` when nothing more can be dropped
/// (already at system + minimal tail), letting the caller return a 4xx.
///
/// Gated by `ATLAS_CTX_OVERFLOW_TRUNCATE=1` (default OFF; returns HTTP 400 on
/// context overflow unless opted in).
pub fn truncate_to_fit(
    msgs: &[serde_json::Value],
    keep_tail: usize,
) -> Option<Vec<serde_json::Value>> {
    // Need at least a system message + one droppable middle message + tail.
    if msgs.len() <= keep_tail.saturating_add(1) {
        return None;
    }
    let mut tail_start = msgs.len().saturating_sub(keep_tail);

    // Walk backward until the tail begins on a genuine user query, so we never
    // strand a `tool`/assistant reply without its preceding user turn.
    let has_user_query = |from: usize| {
        (from..msgs.len()).any(|i| {
            let role = msgs[i]["role"].as_str().unwrap_or("");
            let content = msgs[i]["content"].as_str().unwrap_or("");
            role == "user" && !content.starts_with("<tool_response>")
        })
    };
    if !has_user_query(tail_start) {
        while tail_start > 1 {
            tail_start -= 1;
            let role = msgs[tail_start]["role"].as_str().unwrap_or("");
            let content = msgs[tail_start]["content"].as_str().unwrap_or("");
            if role == "user" && !content.starts_with("<tool_response>") {
                break;
            }
        }
    }
    // Never start the tail on a bare `tool` message (needs a preceding assistant).
    while tail_start < msgs.len() && msgs[tail_start]["role"].as_str() == Some("tool") {
        tail_start += 1;
    }
    // If the safety walks pushed us back to "keep everything but system", there's
    // nothing left to drop — signal the caller to fast-fail instead of looping.
    if tail_start <= 1 {
        return None;
    }
    let mut out = Vec::with_capacity(msgs.len() - tail_start + 1);
    out.push(msgs[0].clone()); // system
    out.extend_from_slice(&msgs[tail_start..]);
    Some(out)
}

/// OpenAI-compatible JSON error response.
/// Coding agents (OpenCode, Cline, nanobot) expect this exact structure.
pub(super) fn openai_error_response(status: StatusCode, message: String) -> Response {
    openai_error_response_with_param(status, message, None, None)
}

/// OpenAI-compatible error with optional `param` (field path like
/// `messages[0].role`) and `code` (e.g. `"context_length_exceeded"`).
pub(super) fn openai_error_response_with_param(
    status: StatusCode,
    message: String,
    param: Option<&str>,
    code: Option<&str>,
) -> Response {
    let body = serde_json::json!({
        "error": {
            "message": message,
            "type": match status {
                StatusCode::BAD_REQUEST => "invalid_request_error",
                StatusCode::UNAUTHORIZED => "authentication_error",
                StatusCode::FORBIDDEN => "permission_error",
                StatusCode::NOT_FOUND => "not_found_error",
                StatusCode::TOO_MANY_REQUESTS => "rate_limit_exceeded",
                StatusCode::SERVICE_UNAVAILABLE => "server_error",
                _ => "server_error",
            },
            "param": param,
            "code": code,
        }
    });
    let mut resp = (status, Json(body)).into_response();
    let retry_after = match status {
        StatusCode::TOO_MANY_REQUESTS => Some("1"),
        StatusCode::SERVICE_UNAVAILABLE => Some("5"),
        _ => None,
    };
    if let Some(secs) = retry_after
        && !resp.headers().contains_key(axum::http::header::RETRY_AFTER)
    {
        resp.headers_mut().insert(
            axum::http::header::RETRY_AFTER,
            axum::http::HeaderValue::from_static(secs),
        );
    }
    resp
}

#[cfg(test)]
mod truncate_tests {
    use super::truncate_to_fit;
    use serde_json::json;

    fn m(role: &str, content: &str) -> serde_json::Value {
        json!({"role": role, "content": content})
    }

    fn roles(v: &[serde_json::Value]) -> Vec<String> {
        v.iter()
            .map(|x| x["role"].as_str().unwrap_or("").to_string())
            .collect()
    }

    #[test]
    fn keeps_system_plus_tail_and_drops_oldest() {
        // system + 10 alternating user/assistant turns.
        let mut msgs = vec![m("system", "you are helpful")];
        for i in 0..5 {
            msgs.push(m("user", &format!("q{i}")));
            msgs.push(m("assistant", &format!("a{i}")));
        }
        // 11 msgs, keep_tail=6 → drop 4 middle, keep system + last 6.
        let out = truncate_to_fit(&msgs, 6).expect("should truncate");
        assert_eq!(out.len(), 7, "system + 6 tail");
        assert_eq!(out[0]["role"], "system");
        // newest content is preserved.
        assert_eq!(out.last().unwrap()["content"], "a4");
    }

    #[test]
    fn none_when_nothing_to_drop() {
        // Only system + tail already at/below keep_tail+1 → cannot shrink.
        let msgs = vec![m("system", "s"), m("user", "hi"), m("assistant", "yo")];
        assert!(truncate_to_fit(&msgs, 6).is_none());
    }

    #[test]
    fn tail_never_starts_on_bare_tool_message() {
        // Deep conversation whose keep_tail window would begin on a `tool` reply.
        let mut msgs = vec![m("system", "s")];
        for i in 0..6 {
            msgs.push(m("user", &format!("q{i}")));
            msgs.push(m("assistant", &format!("call{i}")));
            msgs.push(m("tool", &format!("result{i}")));
        }
        let out = truncate_to_fit(&msgs, 2).expect("should truncate");
        assert_eq!(out[0]["role"], "system");
        // The first non-system retained message must not be an orphaned tool reply.
        assert_ne!(
            out[1]["role"], "tool",
            "tail must not strand a bare tool msg"
        );
    }

    #[test]
    fn tail_begins_on_real_user_query() {
        // Force the safety walk: last messages are assistant/tool only.
        let msgs = vec![
            m("system", "s"),
            m("user", "old question"),
            m("assistant", "old answer"),
            m("user", "real question"),
            m("assistant", "thinking"),
            m("tool", "<tool_response>data"),
        ];
        let out = truncate_to_fit(&msgs, 2).expect("should truncate");
        // A genuine user query must be present in the retained tail (not just a
        // <tool_response> placeholder) so Jinja won't raise "No user query found".
        let has_user = out.iter().skip(1).any(|x| {
            x["role"] == "user"
                && !x["content"]
                    .as_str()
                    .unwrap_or("")
                    .starts_with("<tool_response>")
        });
        assert!(
            has_user,
            "retained tail must contain a real user query; roles={:?}",
            roles(&out)
        );
    }

    #[test]
    fn openai_error_response_sets_retry_after_on_429_and_503() {
        use axum::http::{StatusCode, header};

        let too_many =
            super::openai_error_response(StatusCode::TOO_MANY_REQUESTS, "queue full".to_string());
        assert_eq!(too_many.status(), StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(
            too_many
                .headers()
                .get(header::RETRY_AFTER)
                .and_then(|v| v.to_str().ok()),
            Some("1")
        );

        let unavailable = super::openai_error_response(
            StatusCode::SERVICE_UNAVAILABLE,
            "scheduler unavailable".to_string(),
        );
        assert_eq!(unavailable.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(
            unavailable
                .headers()
                .get(header::RETRY_AFTER)
                .and_then(|v| v.to_str().ok()),
            Some("5")
        );

        let bad_req =
            super::openai_error_response(StatusCode::BAD_REQUEST, "invalid field".to_string());
        assert_eq!(bad_req.status(), StatusCode::BAD_REQUEST);
        assert!(bad_req.headers().get(header::RETRY_AFTER).is_none());
    }
}
