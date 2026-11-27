// SPDX-License-Identifier: AGPL-3.0-only

#![allow(unused_imports, dead_code)]

//! Streaming `/v1/chat/completions` SSE handler.
//!
//! Sub-files:
//! - `state`        — `StreamState`: every captured-mutable local
//! - `ctx`          — `StreamCtx`: every captured-read-only value
//! - `handle_token` — Token / TokenWithLogprobs arm + tool-call
//!                    helpers shared with the Done arm's flush
//! - `handle_done`  — Done arm (flush, salvage, usage, dump, metrics)
//! - `handle_error` — Error arm

mod completion_token_ids;
mod continuous_usage;
mod ctx;
mod dsv41;
mod handle_done;
mod handle_error;
mod handle_token;
mod logprobs;
mod state;
mod tool_handlers;

#[cfg(test)]
mod glm_tool_publication_stream_tests;

#[cfg(test)]
mod continuous_usage_contract_tests;

pub(in crate::api) use handle_token::filter_stop_delta;

use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive};
use axum::response::{IntoResponse, Response, Sse};
use futures::StreamExt;
use std::sync::Arc;
use tokio_stream::wrappers::ReceiverStream;

use crate::AppState;
use crate::openai::ChatCompletionChunk;
use crate::tool_parser;

use super::failures::{F39FailureCache, f60_disable_mtp_for_request};
use super::inference_impl::try_admit_scheduler;
use super::inference_types::{GrammarSpec, InferenceRequest, StreamEvent};

use continuous_usage::ContinuousUsageTracker;
use ctx::StreamCtx;
use state::StreamState;

pub(super) type ChunkVec = Vec<ChatCompletionChunk>;
type StreamEvents = Vec<Result<Event, std::convert::Infallible>>;

#[allow(clippy::too_many_arguments)]
pub(crate) async fn chat_completions_stream(
    state: Arc<AppState>,
    prompt_tokens: Vec<u32>,
    session_hash: u64,
    image_pixels: Vec<(Vec<f32>, usize, usize)>,
    max_tokens: usize,
    min_tokens: usize,
    ignore_eos: bool,
    temperature: f32,
    top_k: u32,
    top_p: f32,
    top_n_sigma: f32,
    min_p: f32,
    repetition_penalty: f32,
    presence_penalty: f32,
    frequency_penalty: f32,
    dry_multiplier: f32,
    dry_base: f32,
    dry_allowed_length: u32,
    lz_penalty: f32,
    logit_bias: Vec<(u32, f32)>,
    enable_thinking: bool,
    thinking_budget: Option<u32>,
    tools_active: bool,
    tool_choice_required: bool,
    suppress_tool_call: bool,
    tool_defs: Vec<tool_parser::ToolDefinition>,
    cwd_hint: Option<String>,
    stop_tokens: Vec<u32>,
    grammar_spec: Option<GrammarSpec>,
    seed: Option<u64>,
    top_logprobs: Option<u8>,
    timeout_at: Option<std::time::Instant>,
    decode_timeout_at: Option<std::time::Instant>,
    contract_warnings: Vec<String>,
    stop_strings: Vec<String>,
    allow_parallel_tool_calls: bool,
    req_stream_include_usage: bool,
    req_stream_continuous_usage: bool,
    req_return_token_ids: bool,
    req_expects_json_response: bool,
    req_service_tier: Option<String>,
    req_metadata: Option<std::collections::HashMap<String, String>>,
    req_ctx: Option<crate::rate_limiter::RequestContext>,
    dump_seq: Option<u64>,
    f44_cache: F39FailureCache,
    intervention_ledger: Option<crate::openai::SharedInterventionLedger>,
    rendered_prompt_evidence_source: Option<(String, String)>,
    think_nudge: Option<std::sync::Arc<crate::scheduler::think_nudge::ThinkNudgeConfig>>,
) -> Result<Response, (StatusCode, String)> {
    // `service_tier` and `metadata` are accepted for OpenAI request schema
    // compatibility and discarded on streaming responses (chat-completion
    // chunk frames do not carry them).
    let _ = (&req_service_tier, &req_metadata);
    // Channel capacity sized for ~30s of decode at 50 tok/s. The previous
    // 64-slot buffer would fill in <2s under any HTTP-flush stall and silently
    // drop the seq via emit_step.rs's `try_send().is_err()` (now fixed to
    // discriminate Full from Closed). Larger capacity = fewer Full→blocking_send
    // round-trips in the steady state.
    let (token_tx, token_rx) = tokio::sync::mpsc::channel::<StreamEvent>(1024);
    let prompt_len = prompt_tokens.len();
    // Cooperative cancellation flag shared with the scheduler. Flipped
    // by stream-side loop guards (Bug-2 name-run cap, F11/F44 dedup,
    // loop-watchdog) so the scheduler stops generating instead of just
    // having its output suppressed — without it a degenerate-loop
    // response keeps generating until max_tokens (or hangs on a
    // channel-full blocking_send) while the client waits for `[DONE]`.
    let cancel_flag = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));

    // Scheduler tracks thinking only when the template actually opens it.
    // When enable_thinking=false, the template inserts closed
    // `<think>\n\n</think>\n\n` and the model generates no thinking tokens —
    // no need for scheduler tracking.
    let scheduler_thinking = enable_thinking;
    // Strip markdown JSON fences whenever the client requested JSON output
    // (`response_format` = `json_object` / `json_schema`), even when
    // `disable_tool_grammar=true` leaves `grammar_spec == None`, as well as
    // when a JSON grammar is active.
    let strip_json_fence = req_expects_json_response
        || matches!(
            &grammar_spec,
            Some(
                GrammarSpec::JsonSchema { .. }
                    | GrammarSpec::JsonObject
                    | GrammarSpec::ToolCall {
                        response_format: Some(_),
                        ..
                    }
            )
        );
    // Native GLM publication validates against the grammar's exact tool set.
    let native_tool_defs = match &grammar_spec {
        Some(GrammarSpec::ToolCall { tools, parser, .. }) if parser.name() == "glm_xml" => {
            Some(tools.clone())
        }
        _ => None,
    };
    let request = InferenceRequest::Streaming {
        prompt_tokens,
        session_hash,
        image_pixels,
        max_tokens,
        min_tokens,
        ignore_eos,
        temperature,
        top_k,
        top_p,
        top_n_sigma,
        min_p,
        repetition_penalty,
        presence_penalty,
        frequency_penalty,
        dry_multiplier,
        dry_base,
        dry_allowed_length,
        lz_penalty,
        logit_bias,
        stop_tokens,
        enable_thinking: scheduler_thinking,
        thinking_budget,
        require_tool_call: tool_choice_required,
        suppress_tool_call,
        disable_mtp: f60_disable_mtp_for_request(tools_active),
        grammar_spec,
        seed,
        top_logprobs,
        timeout_at,
        decode_timeout_at,
        intervention_ledger: intervention_ledger.clone(),
        think_nudge,
        token_tx,
        cancel_flag: cancel_flag.clone(),
    };

    try_admit_scheduler(&state.request_tx, request)?;

    let chunk_id = crate::openai::new_chunk_id();
    let model_name = state.model_name.clone();
    let rendered_prompt_evidence = rendered_prompt_evidence_source.map(|(nonce, prompt_sha256)| {
        crate::openai::RenderedPromptEvidence::from_digest(nonce, chunk_id.clone(), prompt_sha256)
    });

    // Only engage tool-markup leak sanitization when tools are active on this
    // request; non-tool streams must pass code/XML/HTML/JSX/SQL verbatim.
    let leak_markers: tool_parser::LeakMarkers = if tools_active {
        state
            .tool_call_parser
            .as_ref()
            .map(|p| p.leak_markers())
            .unwrap_or(tool_parser::LeakMarkers::EMPTY)
    } else {
        tool_parser::LeakMarkers::EMPTY
    };

    let agent_guards = crate::api::chat_phases::agent_guards_enabled();
    let max_tool_calls_per_response: usize = std::env::var("FLUX_MAX_TOOL_CALLS_PER_RESPONSE")
        .ok()
        .or_else(|| std::env::var("ATLAS_MAX_TOOL_CALLS_PER_RESPONSE").ok())
        .and_then(|s| s.parse().ok())
        .unwrap_or(if agent_guards { 12 } else { usize::MAX });

    // Log permanent-failure cache state at stream entry (active only under opt-in agent guards).
    let f44_cache_active = agent_guards
        && (!f44_cache.direct.is_empty() || !f44_cache.missing_bins_by_tool.is_empty());
    if f44_cache_active {
        tracing::info!(
            direct_entries = f44_cache.direct.len(),
            missing_bin_tools = f44_cache.missing_bins_by_tool.len(),
            sample_direct = ?f44_cache.direct.keys().take(3).collect::<Vec<_>>(),
            "permanent-failure cache: streaming closure entered with non-empty cache"
        );
    }

    let active_guard = Arc::new(std::sync::atomic::AtomicBool::new(false));

    let ctx = StreamCtx {
        state: state.clone(),
        model: model_name.clone(),
        id: chunk_id.clone(),
        prompt_len,
        min_tokens,
        ignore_eos,
        enable_thinking,
        tool_defs_for_backfill: tool_defs,
        native_tool_defs,
        cwd_for_normalize: cwd_hint,
        stop_strings,
        leak_markers,
        max_tool_calls_per_response,
        agent_guards,
        allow_parallel_tool_calls,
        req_stream_include_usage,
        req_return_token_ids,
        req_ctx,
        dump_seq,
        f44_cache,
        f44_cache_active,
        intervention_ledger: intervention_ledger.clone(),
        rendered_prompt_evidence,
        contract_warnings,
        active_guard: active_guard.clone(),
    };

    if state.dsv41 {
        let mut continuous_usage = ContinuousUsageTracker::new(
            req_stream_include_usage,
            req_stream_continuous_usage,
            prompt_len,
            &model_name,
            &chunk_id,
        );
        let role_chunk =
            continuous_usage.decorate_role(ChatCompletionChunk::role_chunk(&model_name, &chunk_id));
        let role_json = serde_json::to_string(&role_chunk).unwrap_or_default();
        let mut st = crate::api::dsv41::StreamState::new(
            &state.tokenizer,
            enable_thinking,
            ctx.stop_strings.clone(),
            cancel_flag.clone(),
        );
        let mut dsv41_fence_stripper =
            strip_json_fence.then(super::chat::repair_json::JsonFenceStripper::new);
        let token_stream = ReceiverStream::new(token_rx).flat_map(move |event| {
            let events = match event {
                StreamEvent::Token(tok) => chunks_to_events(continuous_usage.observe_token(
                    dsv41::on_token(&mut st, &ctx, tok, dsv41_fence_stripper.as_mut()),
                )),
                StreamEvent::TokenBatch(toks) => {
                    let token_count = toks.len();
                    let mut chunks = ChunkVec::new();
                    for tok in toks {
                        chunks.extend(dsv41::on_token(
                            &mut st,
                            &ctx,
                            tok,
                            dsv41_fence_stripper.as_mut(),
                        ));
                    }
                    chunks_to_events(continuous_usage.observe_token_batch(token_count, chunks))
                }
                StreamEvent::TokenWithLogprobs(tok, token_logprobs) => {
                    let chunks = dsv41::on_token(&mut st, &ctx, tok, dsv41_fence_stripper.as_mut());
                    let chunks = logprobs::attach_token_logprobs(
                        chunks,
                        &ctx.state.tokenizer,
                        &ctx.model,
                        &ctx.id,
                        token_logprobs,
                    );
                    chunks_to_events(continuous_usage.observe_token(chunks))
                }
                StreamEvent::CompletionProgress => {
                    chunks_to_events(continuous_usage.observe_completion_progress())
                }
                StreamEvent::CompletionProgressWithLogprobs(token_logprobs) => {
                    let chunks = logprobs::attach_token_logprobs(
                        Vec::new(),
                        &ctx.state.tokenizer,
                        &ctx.model,
                        &ctx.id,
                        token_logprobs,
                    );
                    chunks_to_events(continuous_usage.observe_token(chunks))
                }
                // The Qwen implicit-close policy is disabled for DSV41. Keep
                // the endpoint exhaustive without changing its Python router.
                StreamEvent::ThinkingDone => Vec::new(),
                StreamEvent::Done {
                    finish_reason,
                    completion_tokens,
                    time_to_first_token_ms,
                    decode_time_ms,
                    reasoning_tokens,
                    cached_prompt_tokens,
                    ..
                } => {
                    let mut chunks = dsv41::on_done(
                        &mut st,
                        &ctx,
                        finish_reason,
                        completion_tokens,
                        time_to_first_token_ms,
                        decode_time_ms,
                        reasoning_tokens,
                        cached_prompt_tokens,
                        dsv41_fence_stripper.as_mut(),
                    );
                    if let Some(ledger) = crate::openai::snapshot_shared_interventions(
                        ctx.intervention_ledger.as_ref(),
                    ) && let Some(chunk) = chunks.iter_mut().find(|chunk| {
                        chunk
                            .choices
                            .iter()
                            .any(|choice| choice.finish_reason.is_some())
                    }) {
                        chunk.interventions = Some(ledger);
                    }
                    chunks_to_events(continuous_usage.reconcile_done(completion_tokens, chunks))
                }
                StreamEvent::Error(msg) => handle_error::handle_error(&ctx, st.cancel_flag(), msg),
            };
            futures::stream::iter(events)
        });
        return Ok(sse_response(
            role_json,
            token_stream,
            active_guard,
            cancel_flag,
        ));
    }

    let parser_audit_name = intervention_ledger.as_ref().and_then(|_| {
        state
            .tool_call_parser
            .as_ref()
            .map(|parser| parser.name().to_string())
    });
    let mut stream_state = StreamState::new(
        tools_active,
        enable_thinking,
        cancel_flag.clone(),
        intervention_ledger,
    );
    if strip_json_fence {
        stream_state.json_fence_stripper = Some(super::chat::repair_json::JsonFenceStripper::new());
    }
    if let (Some(detector), Some(parser_name)) =
        (stream_state.detector.as_mut(), parser_audit_name.as_deref())
    {
        *detector = tool_parser::StreamingToolDetector::new_with_parser(parser_name);
    }
    let mut continuous_usage = ContinuousUsageTracker::new(
        req_stream_include_usage,
        req_stream_continuous_usage,
        prompt_len,
        &model_name,
        &chunk_id,
    );
    let role_chunk =
        continuous_usage.decorate_role(ChatCompletionChunk::role_chunk(&model_name, &chunk_id));
    let role_json = serde_json::to_string(&role_chunk).unwrap_or_default();

    let token_stream = ReceiverStream::new(token_rx).flat_map(move |event| {
        let events = match event {
            StreamEvent::Token(tok) => {
                chunks_to_events(continuous_usage.observe_token(handle_token::handle_token(
                    &mut stream_state,
                    &ctx,
                    tok,
                )))
            }
            StreamEvent::TokenBatch(toks) => {
                let token_count = toks.len();
                let chunks = handle_token::handle_token_batch(&mut stream_state, &ctx, &toks);
                chunks_to_events(continuous_usage.observe_token_batch(token_count, chunks))
            }
            StreamEvent::TokenWithLogprobs(tok, token_logprobs) => {
                let chunks = handle_token::handle_token(&mut stream_state, &ctx, tok);
                let chunks = logprobs::attach_token_logprobs(
                    chunks,
                    &ctx.state.tokenizer,
                    &ctx.model,
                    &ctx.id,
                    token_logprobs,
                );
                chunks_to_events(continuous_usage.observe_token(chunks))
            }
            StreamEvent::CompletionProgress => {
                chunks_to_events(continuous_usage.observe_completion_progress())
            }
            StreamEvent::CompletionProgressWithLogprobs(token_logprobs) => {
                let chunks = logprobs::attach_token_logprobs(
                    Vec::new(),
                    &ctx.state.tokenizer,
                    &ctx.model,
                    &ctx.id,
                    token_logprobs,
                );
                chunks_to_events(continuous_usage.observe_token(chunks))
            }
            StreamEvent::ThinkingDone => {
                chunks_to_events(handle_token::handle_thinking_done(&mut stream_state, &ctx))
            }
            StreamEvent::Done {
                finish_reason,
                completion_token_ids,
                prompt_tokens: _,
                completion_tokens,
                time_to_first_token_ms,
                decode_time_ms,
                reasoning_tokens,
                cached_prompt_tokens,
                think_nudges,
                stop_cause,
            } => {
                match handle_done::handle_done(
                    &mut stream_state,
                    &ctx,
                    finish_reason,
                    completion_token_ids,
                    completion_tokens,
                    time_to_first_token_ms,
                    decode_time_ms,
                    reasoning_tokens,
                    cached_prompt_tokens,
                    think_nudges,
                    stop_cause,
                ) {
                    Ok(chunks) => {
                        chunks_to_events(continuous_usage.reconcile_done(completion_tokens, chunks))
                    }
                    Err(message) => {
                        handle_error::handle_error(&ctx, &stream_state.cancel_flag, message)
                    }
                }
            }
            StreamEvent::Error(msg) => {
                handle_error::handle_error(&ctx, &stream_state.cancel_flag, msg)
            }
        };
        futures::stream::iter(events)
    });

    Ok(sse_response(
        role_json,
        token_stream,
        active_guard,
        cancel_flag,
    ))
}

fn chunks_to_events(chunks: ChunkVec) -> StreamEvents {
    chunks
        .into_iter()
        .map(|chunk| {
            let json = serde_json::to_string(&chunk).unwrap_or_default();
            Ok(Event::default().data(json))
        })
        .collect()
}

struct StreamGuard {
    decremented: Arc<std::sync::atomic::AtomicBool>,
    cancel_flag: Arc<std::sync::atomic::AtomicBool>,
}

impl Drop for StreamGuard {
    fn drop(&mut self) {
        self.cancel_flag
            .store(true, std::sync::atomic::Ordering::Release);
        if !self
            .decremented
            .swap(true, std::sync::atomic::Ordering::SeqCst)
        {
            crate::metrics::REQUESTS_ACTIVE.dec();
        }
    }
}

struct DropGuardedStream<S> {
    inner: S,
    _guard: StreamGuard,
}

impl<S> DropGuardedStream<S> {
    fn new(inner: S, guard: StreamGuard) -> Self {
        Self {
            inner,
            _guard: guard,
        }
    }
}

impl<S: futures::Stream + Unpin> futures::Stream for DropGuardedStream<S> {
    type Item = S::Item;

    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        std::pin::Pin::new(&mut self.inner).poll_next(cx)
    }
}

const SSE_KEEPALIVE_ENV: &str = "FLUX_SSE_KEEPALIVE_SECS";
const DEFAULT_SSE_KEEPALIVE_SECS: u64 = 2;
const MIN_SSE_KEEPALIVE_SECS: u64 = 1;
const MAX_SSE_KEEPALIVE_SECS: u64 = 60;

fn parse_sse_keepalive_interval(raw: Option<&str>) -> std::time::Duration {
    let secs = raw
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .and_then(|s| s.parse::<u64>().ok())
        .map_or(DEFAULT_SSE_KEEPALIVE_SECS, |n| {
            n.clamp(MIN_SSE_KEEPALIVE_SECS, MAX_SSE_KEEPALIVE_SECS)
        });
    std::time::Duration::from_secs(secs)
}

/// SSE keep-alive policy (`FLUX_SSE_KEEPALIVE_SECS`, default `2`, clamped `1..=60`).
///
/// Tightening the keep-alive comment ping from axum's 15s default to 2s detects
/// dead client sockets during long prefills promptly so the scheduler aborts
/// abandoned GPU work within `<= 2s`.
pub(crate) fn sse_keep_alive() -> KeepAlive {
    static INTERVAL: std::sync::OnceLock<std::time::Duration> = std::sync::OnceLock::new();
    let interval = *INTERVAL.get_or_init(|| {
        parse_sse_keepalive_interval(std::env::var(SSE_KEEPALIVE_ENV).ok().as_deref())
    });
    KeepAlive::new().interval(interval)
}

/// Prepend the role chunk and append the `[DONE]` sentinel.
fn sse_response(
    role_json: String,
    token_stream: impl futures::Stream<Item = Result<Event, std::convert::Infallible>> + Send + 'static,
    active_guard: Arc<std::sync::atomic::AtomicBool>,
    cancel_flag: Arc<std::sync::atomic::AtomicBool>,
) -> Response {
    let role_event = futures::stream::once(async move { Ok(Event::default().data(role_json)) });
    let done_event = futures::stream::once(async {
        Ok::<_, std::convert::Infallible>(Event::default().data("[DONE]"))
    });
    let full_stream = role_event.chain(token_stream).chain(done_event);
    let guarded_stream = DropGuardedStream::new(
        full_stream.boxed(),
        StreamGuard {
            decremented: active_guard,
            cancel_flag,
        },
    );

    Sse::new(guarded_stream)
        .keep_alive(sse_keep_alive())
        .into_response()
}

#[cfg(test)]
mod sse_keepalive_tests {
    use std::time::Duration;

    use super::parse_sse_keepalive_interval;

    #[test]
    fn keepalive_interval_defaults_to_2s_and_clamps_to_1_to_60s() {
        assert_eq!(parse_sse_keepalive_interval(None), Duration::from_secs(2));
        assert_eq!(
            parse_sse_keepalive_interval(Some("")),
            Duration::from_secs(2)
        );
        assert_eq!(
            parse_sse_keepalive_interval(Some("   ")),
            Duration::from_secs(2)
        );
        assert_eq!(
            parse_sse_keepalive_interval(Some("abc")),
            Duration::from_secs(2)
        );
        assert_eq!(
            parse_sse_keepalive_interval(Some("0")),
            Duration::from_secs(1)
        );
        assert_eq!(
            parse_sse_keepalive_interval(Some("1")),
            Duration::from_secs(1)
        );
        assert_eq!(
            parse_sse_keepalive_interval(Some(" 5 ")),
            Duration::from_secs(5)
        );
        assert_eq!(
            parse_sse_keepalive_interval(Some("60")),
            Duration::from_secs(60)
        );
        assert_eq!(
            parse_sse_keepalive_interval(Some("61")),
            Duration::from_secs(60)
        );
        assert_eq!(
            parse_sse_keepalive_interval(Some("999")),
            Duration::from_secs(60)
        );
    }
}
