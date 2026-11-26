// SPDX-License-Identifier: AGPL-3.0-only
//
// deepseek_v41: the Python server's stream shape (api/dsv41.rs) on the shared
// scheduler events, encoded as this server's chat-completion chunks.

use crate::api::dsv41::{Delta, StreamState};
use crate::openai::{ChatCompletionChunk, Usage};

use super::super::chat::repair_json::JsonFenceStripper;
use super::ChunkVec;
use super::continuous_usage::terminal_chunks;
use super::ctx::StreamCtx;

fn enforce_parallel_tool_call_policy(deltas: &mut Vec<Delta>, allow_parallel: bool) -> bool {
    if allow_parallel {
        return false;
    }
    let mut saw_tool_call = false;
    let mut dropped_tool_call = false;
    deltas.retain(|delta| {
        if !matches!(delta, Delta::ToolCall { .. }) {
            return true;
        }
        if saw_tool_call {
            dropped_tool_call = true;
            false
        } else {
            saw_tool_call = true;
            true
        }
    });
    dropped_tool_call
}

/// Wire chunks for one piece of the reply. `Finish` is handled by [`done`].
pub(super) fn delta_events(model: &str, id: &str, delta: Delta, out: &mut ChunkVec) {
    delta_events_with_stripper(model, id, delta, None, out);
}

fn delta_events_with_stripper(
    model: &str,
    id: &str,
    delta: Delta,
    fence_stripper: Option<&mut JsonFenceStripper>,
    out: &mut ChunkVec,
) {
    match delta {
        Delta::Reasoning(text) => {
            let mut chunk = ChatCompletionChunk::reasoning_chunk(model, id, text);
            // app.py streams `reasoning_content` only.
            for c in &mut chunk.choices {
                c.delta.reasoning = None;
            }
            out.push(chunk);
        }
        Delta::Content(text) => {
            let text = if let Some(stripper) = fence_stripper {
                stripper.push(&text)
            } else {
                text
            };
            if !text.is_empty() {
                out.push(ChatCompletionChunk::content_chunk(model, id, text));
            }
        }
        Delta::ToolCall { index, call } => {
            let start = ChatCompletionChunk::tool_call_start_chunk(model, id, &call, index);
            out.push(start);
            let args = ChatCompletionChunk::tool_call_args_fragment(
                model,
                id,
                index,
                &call.function.arguments,
            );
            out.push(args);
        }
        Delta::Finish(_) => {}
    }
}

pub(super) fn on_token(
    st: &mut StreamState,
    ctx: &StreamCtx,
    token: u32,
    mut fence_stripper: Option<&mut JsonFenceStripper>,
) -> ChunkVec {
    let mut out = ChunkVec::new();
    for d in st.on_token(&ctx.state.tokenizer, token) {
        delta_events_with_stripper(
            &ctx.model,
            &ctx.id,
            d,
            fence_stripper.as_deref_mut(),
            &mut out,
        );
    }
    out
}

#[allow(clippy::too_many_arguments)]
pub(super) fn on_done(
    st: &mut StreamState,
    ctx: &StreamCtx,
    finish_reason: String,
    completion_tokens: usize,
    time_to_first_token_ms: f64,
    decode_time_ms: f64,
    reasoning_tokens: u32,
    cached_prompt_tokens: u32,
    mut fence_stripper: Option<&mut JsonFenceStripper>,
) -> ChunkVec {
    let mut out = ChunkVec::new();
    let mut reason = finish_reason;
    let mut deltas = st.on_done(&ctx.state.tokenizer, reason.clone());
    if enforce_parallel_tool_call_policy(&mut deltas, ctx.allow_parallel_tool_calls) {
        crate::openai::record_shared_intervention(
            ctx.intervention_ledger.as_ref(),
            "tool_output_policy",
            "truncate_parallel_calls",
        );
        st.cancel_flag()
            .store(true, std::sync::atomic::Ordering::Release);
    }
    for d in deltas {
        if let Delta::Finish(r) = &d {
            reason = r.clone();
        }
        delta_events_with_stripper(
            &ctx.model,
            &ctx.id,
            d,
            fence_stripper.as_deref_mut(),
            &mut out,
        );
    }
    if let Some(stripper) = fence_stripper {
        let remainder = stripper.flush();
        if !remainder.is_empty() {
            out.push(ChatCompletionChunk::content_chunk(
                &ctx.model, &ctx.id, remainder,
            ));
        }
    }

    let tps = if decode_time_ms > 0.0 {
        completion_tokens.saturating_sub(1) as f64 / (decode_time_ms / 1000.0)
    } else {
        0.0
    };
    let usage = Usage {
        prompt_tokens: ctx.prompt_len,
        completion_tokens,
        total_tokens: ctx.prompt_len + completion_tokens,
        prompt_tokens_details: Some(crate::openai::PromptTokensDetails {
            cached_tokens: cached_prompt_tokens as usize,
            audio_tokens: 0,
        }),
        completion_tokens_details: Some(crate::openai::CompletionTokensDetails {
            reasoning_tokens: reasoning_tokens as usize,
            audio_tokens: 0,
            accepted_prediction_tokens: 0,
            rejected_prediction_tokens: 0,
            think_nudges: 0,
        }),
        time_to_first_token_ms,
        response_tokens_per_second: tps,
    };
    out.extend(terminal_chunks(
        &ctx.model,
        &ctx.id,
        &reason,
        usage.clone(),
        ctx.req_stream_include_usage,
    ));

    if !ctx
        .active_guard
        .swap(true, std::sync::atomic::Ordering::SeqCst)
    {
        crate::metrics::REQUESTS_ACTIVE.dec();
    }
    crate::metrics::PROMPT_TOKENS_TOTAL.inc_by(ctx.prompt_len as u64);
    crate::metrics::GENERATION_TOKENS_TOTAL.inc_by(completion_tokens as u64);
    crate::metrics::TTFT_SECONDS.observe(time_to_first_token_ms / 1000.0);
    if let Some(ref rctx) = ctx.req_ctx {
        let actual = (ctx.prompt_len + completion_tokens) as u64;
        let refund = rctx.reserved_tokens.saturating_sub(actual);
        if refund > 0 {
            ctx.state.rate_limiter.refund_tokens(&rctx.identity, refund);
        }
    }
    if let (Some(seq), Some(dump)) = (ctx.dump_seq, ctx.state.dump_writer.as_ref()) {
        let body = serde_json::json!({
            "id": ctx.id,
            "model": ctx.model,
            "object": "chat.completion.synthesized",
            "finish_reason": reason,
            "usage": usage,
        });
        dump.dump_response("/v1/chat/completions", seq, &body, true);
    }
    out
}

#[cfg(test)]
mod parallel_tool_tests {
    use super::enforce_parallel_tool_call_policy;
    use crate::api::dsv41::Delta;
    use crate::tool_parser::{FunctionCall, ToolCall};

    fn call(index: usize) -> Delta {
        Delta::ToolCall {
            index,
            call: ToolCall {
                id: format!("call_{index}"),
                call_type: "function".into(),
                function: FunctionCall {
                    name: format!("tool_{index}"),
                    arguments: "{}".into(),
                },
            },
        }
    }

    #[test]
    fn explicit_serial_policy_keeps_only_first_complete_call() {
        let mut deltas = vec![call(0), call(1), Delta::Finish("tool_calls".into())];
        assert!(enforce_parallel_tool_call_policy(&mut deltas, false));
        assert_eq!(
            deltas
                .iter()
                .filter(|delta| matches!(delta, Delta::ToolCall { .. }))
                .count(),
            1
        );
        assert!(matches!(deltas.last(), Some(Delta::Finish(reason)) if reason == "tool_calls"));
    }

    #[test]
    fn default_parallel_policy_preserves_all_calls() {
        let mut deltas = vec![call(0), call(1)];
        assert!(!enforce_parallel_tool_call_policy(&mut deltas, true));
        assert_eq!(deltas.len(), 2);
    }

    #[test]
    fn dsv41_delta_events_strip_markdown_json_fence_when_enabled() {
        let mut stripper = super::JsonFenceStripper::new();
        let mut chunks = Vec::new();
        for piece in ["\n\n```json\n", "{\"ok\":", "true}", "\n```\n"] {
            super::delta_events_with_stripper(
                "m",
                "id",
                Delta::Content(piece.into()),
                Some(&mut stripper),
                &mut chunks,
            );
        }
        let remainder = stripper.flush();
        if !remainder.is_empty() {
            chunks.push(crate::openai::ChatCompletionChunk::content_chunk(
                "m", "id", remainder,
            ));
        }
        let text: String = chunks
            .into_iter()
            .filter_map(|c| c.choices.into_iter().next()?.delta.content)
            .collect();
        assert_eq!(text, "{\"ok\":true}");
    }
}
