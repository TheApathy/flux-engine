// SPDX-License-Identifier: AGPL-3.0-only

use super::completion_token_ids::terminal_chunks as exact_terminal_chunks;
use super::continuous_usage::{ContinuousUsageTracker, terminal_chunks};
use super::dsv41::delta_events as dsv41_delta_events;
use super::{chunks_to_events, sse_response};
use crate::api::dsv41::Delta;
use crate::openai::{ChatCompletionChunk, Usage};
use crate::tool_parser::{FunctionCall, ToolCall};
use axum::body::to_bytes;

fn usage(prompt_tokens: usize, completion_tokens: usize) -> Usage {
    Usage {
        prompt_tokens,
        completion_tokens,
        total_tokens: prompt_tokens + completion_tokens,
        prompt_tokens_details: None,
        completion_tokens_details: None,
        time_to_first_token_ms: 0.0,
        response_tokens_per_second: 0.0,
    }
}

fn json(chunk: &ChatCompletionChunk) -> serde_json::Value {
    serde_json::to_value(chunk).expect("chunk serializes")
}

fn completion_count(chunk: &ChatCompletionChunk) -> Option<usize> {
    json(chunk)["usage"]["completion_tokens"]
        .as_u64()
        .map(|n| n as usize)
}

async fn collect_sse_payloads(
    role: &ChatCompletionChunk,
    chunks: Vec<ChatCompletionChunk>,
) -> Vec<String> {
    let role_json = serde_json::to_string(role).expect("role serializes");
    let stream = futures::stream::iter(chunks_to_events(chunks));
    let response = sse_response(
        role_json,
        stream,
        std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
        std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
    );
    let body = to_bytes(response.into_body(), 1 << 20)
        .await
        .expect("SSE body collects");
    String::from_utf8(body.to_vec())
        .expect("SSE is UTF-8")
        .lines()
        .filter_map(|line| line.strip_prefix("data:").map(str::trim))
        .map(str::to_string)
        .collect()
}

#[derive(Default)]
struct LilUsageCounter {
    tokens: usize,
    fallback_chunks: usize,
}

impl LilUsageCounter {
    fn observe(&mut self, chunk: &ChatCompletionChunk) {
        let value = json(chunk);
        if let Some(total) = value["usage"]["completion_tokens"].as_u64() {
            self.tokens = self.tokens.max(total as usize);
            return;
        }
        let Some(choice) = value["choices"]
            .as_array()
            .and_then(|choices| choices.first())
        else {
            return;
        };
        let delta = &choice["delta"];
        let has_text = ["reasoning", "reasoning_content", "content"]
            .iter()
            .any(|key| delta[*key].as_str().is_some_and(|text| !text.is_empty()));
        let has_tool = delta["tool_calls"]
            .as_array()
            .is_some_and(|calls| !calls.is_empty());
        if has_text || has_tool {
            self.tokens += 1;
            self.fallback_chunks += 1;
        }
    }
}

#[test]
fn continuous_usage_requires_both_flags() {
    for (include_usage, continuous_usage, expected) in [
        (false, false, false),
        (false, true, false),
        (true, false, false),
        (true, true, true),
    ] {
        assert_eq!(
            ContinuousUsageTracker::new(include_usage, continuous_usage, 8, "m", "id").enabled(),
            expected,
        );
    }
}

#[test]
fn disabled_wire_is_unchanged_and_buffered_tokens_stay_silent() {
    for (include_usage, continuous_usage) in [(false, false), (false, true), (true, false)] {
        let mut tracker =
            ContinuousUsageTracker::new(include_usage, continuous_usage, 8, "m", "id");
        let role = ChatCompletionChunk::role_chunk("m", "id");
        let content = ChatCompletionChunk::content_chunk("m", "id", "x".into());
        let expected_role = json(&role);
        let expected_content = json(&content);

        assert_eq!(json(&tracker.decorate_role(role)), expected_role);
        let output = tracker.observe_token(vec![content]);
        assert_eq!(output.len(), 1);
        assert_eq!(json(&output[0]), expected_content);
        assert!(tracker.observe_token(Vec::new()).is_empty());
    }
}

#[test]
fn buffered_token_emits_normal_choice_progress_chunk() {
    let mut tracker = ContinuousUsageTracker::new(true, true, 11, "m", "id");
    let chunks = tracker.observe_token(Vec::new());
    assert_eq!(chunks.len(), 1);
    let value = json(&chunks[0]);
    assert_eq!(value["choices"].as_array().map(Vec::len), Some(1));
    assert_eq!(value["choices"][0]["delta"], serde_json::json!({}));
    assert_eq!(
        value["choices"][0]["finish_reason"],
        serde_json::Value::Null
    );
    assert_eq!(value["usage"]["completion_tokens"], 1);
}

#[test]
fn visible_token_counts_are_strictly_monotonic() {
    let mut tracker = ContinuousUsageTracker::new(true, true, 8, "m", "id");
    let counts: Vec<usize> = (0..4)
        .map(|_| {
            let chunks = tracker.observe_token(vec![ChatCompletionChunk::content_chunk(
                "m",
                "id",
                "x".into(),
            )]);
            completion_count(&chunks[0]).expect("continuous count")
        })
        .collect();
    assert_eq!(counts, vec![1, 2, 3, 4]);
    assert!(counts.windows(2).all(|pair| pair[0] < pair[1]));
}

#[tokio::test]
async fn generic_sse_sequence_is_exact_for_lil_and_reconciles_done() {
    let mut tracker = ContinuousUsageTracker::new(true, true, 8, "m", "id");
    let role = tracker.decorate_role(ChatCompletionChunk::role_chunk("m", "id"));
    let mut chunks = Vec::new();
    chunks.extend(
        tracker.observe_token(vec![ChatCompletionChunk::content_chunk(
            "m",
            "id",
            "hello".into(),
        )]),
    );
    chunks.extend(tracker.observe_token(Vec::new()));
    chunks.extend(tracker.reconcile_done(3, terminal_chunks("m", "id", "stop", usage(8, 3), true)));

    assert_eq!(
        std::iter::once(&role)
            .chain(chunks.iter())
            .map(completion_count)
            .collect::<Vec<_>>(),
        vec![Some(0), Some(1), Some(2), Some(3), Some(3)],
    );
    assert_eq!(json(&role)["usage"]["prompt_tokens"], 8);
    assert_eq!(json(&role)["usage"]["total_tokens"], 8);
    let finish = json(&chunks[chunks.len() - 2]);
    let final_usage = json(chunks.last().expect("usage chunk"));
    assert_eq!(finish["choices"][0]["finish_reason"], "stop");
    assert_eq!(final_usage["choices"], serde_json::json!([]));

    let mut lil = LilUsageCounter::default();
    lil.observe(&role);
    for chunk in &chunks {
        lil.observe(chunk);
    }
    assert_eq!(lil.tokens, 3);
    assert_eq!(lil.fallback_chunks, 0);

    let payloads = collect_sse_payloads(&role, chunks).await;
    assert_eq!(payloads.last().map(String::as_str), Some("[DONE]"));
    let finish: serde_json::Value =
        serde_json::from_str(&payloads[payloads.len() - 3]).expect("finish JSON");
    let final_usage: serde_json::Value =
        serde_json::from_str(&payloads[payloads.len() - 2]).expect("usage JSON");
    assert_eq!(finish["choices"][0]["finish_reason"], "stop");
    assert_eq!(finish["usage"]["completion_tokens"], 3);
    assert_eq!(final_usage["choices"], serde_json::json!([]));
    assert_eq!(final_usage["usage"]["completion_tokens"], 3);
}

#[tokio::test]
async fn dsv41_tool_multiframe_uses_one_cumulative_count() {
    let mut tracker = ContinuousUsageTracker::new(true, true, 5, "m", "id");
    let call = ToolCall {
        id: "call_1".into(),
        call_type: "function".into(),
        function: FunctionCall {
            name: "lookup".into(),
            arguments: r#"{"q":"x"}"#.into(),
        },
    };
    let role = tracker.decorate_role(ChatCompletionChunk::role_chunk("m", "id"));
    let mut chunks = Vec::new();
    let mut reasoning = Vec::new();
    dsv41_delta_events("m", "id", Delta::Reasoning("think".into()), &mut reasoning);
    chunks.extend(tracker.observe_token(reasoning));
    let mut tool = Vec::new();
    dsv41_delta_events("m", "id", Delta::ToolCall { index: 0, call }, &mut tool);
    chunks.extend(tracker.observe_token(tool));
    chunks.extend(tracker.reconcile_done(
        3,
        terminal_chunks("m", "id", "tool_calls", usage(5, 3), true),
    ));

    assert_eq!(
        std::iter::once(&role)
            .chain(chunks.iter())
            .map(completion_count)
            .collect::<Vec<_>>(),
        vec![Some(0), Some(1), Some(2), Some(2), Some(3), Some(3)],
    );
    let mut lil = LilUsageCounter::default();
    lil.observe(&role);
    for chunk in &chunks {
        lil.observe(chunk);
    }
    assert_eq!(lil.tokens, 3);
    assert_eq!(lil.fallback_chunks, 0);

    let payloads = collect_sse_payloads(&role, chunks).await;
    assert_eq!(payloads.last().map(String::as_str), Some("[DONE]"));
    let tool_start: serde_json::Value =
        serde_json::from_str(&payloads[2]).expect("tool start JSON");
    let tool_args: serde_json::Value = serde_json::from_str(&payloads[3]).expect("tool args JSON");
    assert_eq!(tool_start["usage"]["completion_tokens"], 2);
    assert_eq!(tool_args["usage"]["completion_tokens"], 2);
}

#[test]
fn scheduler_suppressed_token_progress_is_counted_without_fallback() {
    let mut tracker = ContinuousUsageTracker::new(true, true, 4, "m", "id");
    let role = tracker.decorate_role(ChatCompletionChunk::role_chunk("m", "id"));
    let progress = tracker.observe_completion_progress();
    assert_eq!(completion_count(&role), Some(0));
    assert_eq!(progress.len(), 1);
    assert_eq!(completion_count(&progress[0]), Some(1));

    let mut lil = LilUsageCounter::default();
    lil.observe(&role);
    lil.observe(&progress[0]);
    assert_eq!(lil.tokens, 1);
    assert_eq!(lil.fallback_chunks, 0);
}

#[test]
fn continuous_usage_preserves_scheduler_authoritative_terminal_token_ids() {
    let mut tracker = ContinuousUsageTracker::new(true, true, 4, "m", "id");
    tracker.observe_token(Vec::new());
    tracker.observe_token(Vec::new());
    let terminal = exact_terminal_chunks(
        "m",
        "id",
        "stop",
        usage(4, 2),
        true,
        true,
        vec![17, 23],
        None,
        None,
        None,
    )
    .expect("scheduler token-ID snapshot matches usage");
    let chunks = tracker.reconcile_done(2, terminal);

    assert_eq!(chunks.len(), 2);
    let finish = json(&chunks[0]);
    let final_usage = json(&chunks[1]);
    assert_eq!(finish["choices"][0]["finish_reason"], "stop");
    assert_eq!(
        finish["choices"][0]["token_ids"],
        serde_json::json!([17, 23])
    );
    assert_eq!(finish["usage"]["completion_tokens"], 2);
    assert_eq!(final_usage["choices"], serde_json::json!([]));
    assert_eq!(final_usage["usage"]["completion_tokens"], 2);
}
