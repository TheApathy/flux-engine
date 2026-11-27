// SPDX-License-Identifier: AGPL-3.0-only

#![allow(unused_imports, dead_code)]

use axum::extract::State;
use axum::extract::rejection::JsonRejection;
use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive};
use axum::response::{IntoResponse, Json, Response, Sse};
use futures::StreamExt;
use std::sync::Arc;
use tokio_stream::wrappers::ReceiverStream;

use crate::AppState;
use crate::openai::{
    ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse, CompletionChunk,
    CompletionRequest, CompletionResponse, ModelInfo, ModelListResponse, Usage,
};
use crate::tool_parser;

/// Structured JSON output constraint carried by `response_format` when
/// paired with `tools` under `tool_choice="auto"`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ResponseFormatConstraint {
    /// Any valid JSON value (`response_format={"type":"json_object"}`).
    JsonObject,
    /// JSON conforming to a schema (`response_format={"type":"json_schema",...}`).
    JsonSchema { schema: String, strict: bool },
}

/// Grammar specification for constrained decoding.
///
/// Either tool-call grammar (Hermes/Qwen3-Coder format), response-format
/// grammar (generic JSON or JSON-schema), or a joint union of both when a
/// request supplies `tools` alongside `response_format` under `tool_choice="auto"`.
#[derive(Clone)]
pub enum GrammarSpec {
    /// Tool-call constrained decoding: tools, parser instance, use_triggers flag.
    ///
    /// F69 (2026-04-29): carries the `Arc<dyn ToolCallParser>` itself
    /// rather than a stringly-typed `parser_name`. The scheduler
    /// dispatches `parser.compile_tool_grammar(engine, tools, use_triggers,
    /// stop_after_first)`
    /// — the parser is the single source of truth for both response
    /// scanning AND grammar generation. This eliminates the prior
    /// `match parser_name.as_str()` in
    /// `scheduler.rs::compile_grammar_state` that could drift from the
    /// parser identity stored in `AppState.tool_call_parser`.
    ToolCall {
        tools: Vec<tool_parser::ToolDefinition>,
        parser: std::sync::Arc<dyn tool_parser::ToolCallParser>,
        use_triggers: bool,
        /// Make the structural-tag matcher admit only a stop token after the
        /// first complete call. `commit_tool_call_close` then finishes the
        /// sequence at a recognized single-token close for both sinks. Set
        /// for `tool_choice=required|specific` (as before) and for
        /// `parallel_tool_calls=false`, which previously only truncated the
        /// published list while the scheduler kept decoding.
        stop_after_first: bool,
        /// Optional `response_format` constraint to union with the tool-call
        /// grammar when `tool_choice="auto"` and both `tools` and
        /// `response_format` are present. Allows the model to emit either a
        /// valid tool call (on tool-invocation turns) OR schema-valid JSON
        /// (on final-answer turns, allowing optional ```` ```json ```` fences
        /// stripped by `repair_json`) while forbidding surrounding prose.
        response_format: Option<ResponseFormatConstraint>,
    },
    /// Response format: any valid JSON (json_object).
    JsonObject,
    /// Response format: JSON matching a specific schema.
    JsonSchema { schema: String, strict: bool },
}

/// Request submitted to the scheduler.
pub enum InferenceRequest {
    /// Blocking: waits for full response.
    Blocking {
        prompt_tokens: Vec<u32>,
        /// Session hash for SSM snapshot isolation (hash of first 64 prompt tokens).
        session_hash: u64,
        /// Preprocessed image data: (pixels `[P,1536]` f32, grid_h, grid_w) per image.
        image_pixels: Vec<(Vec<f32>, usize, usize)>,
        max_tokens: usize,
        /// Minimum tokens before allowing EOS/stop (0 = no minimum).
        min_tokens: usize,
        /// vLLM `ignore_eos`: model EOS ids do not end the turn.
        ignore_eos: bool,
        temperature: f32,
        /// Top-k: keep only the k highest-probability tokens (0 = disabled).
        top_k: u32,
        /// Top-p (nucleus): cumulative probability threshold (1.0 = disabled).
        top_p: f32,
        /// Top-n-sigma: filter tokens by logit z-score (0.0 = disabled).
        top_n_sigma: f32,
        /// Min-p: keep tokens with prob >= min_p * max_prob (0.0 = disabled).
        min_p: f32,
        /// Repetition penalty (1.0 = disabled).
        repetition_penalty: f32,
        /// Presence penalty — OpenAI-style, additive (0.0 = disabled).
        presence_penalty: f32,
        /// Frequency penalty — OpenAI-style, additive (0.0 = disabled).
        frequency_penalty: f32,
        /// DRY (Don't-Repeat-Yourself) n-gram sampler. Populated from
        /// `sampling_presets.tools.dry_multiplier` etc. when the model's
        /// MODEL.toml opts in (e.g. qwen3.5-35b-a3b/MODEL.toml sets
        /// `dry_multiplier = 0.8` for the tools preset). 0.0 = disabled.
        dry_multiplier: f32,
        dry_base: f32,
        dry_allowed_length: u32,
        /// LZ penalty (A.1, arXiv:2504.20131). Per-extension n-gram
        /// penalty over the recent 256-token window. Populated from
        /// MODEL.toml `[sampling.*].lz_penalty`. 0.0 = disabled.
        lz_penalty: f32,
        /// Per-token logit bias: (token_id, bias_value) pairs.
        logit_bias: Vec<(u32, f32)>,
        /// Per-request stop token IDs (from OpenAI `stop` parameter).
        stop_tokens: Vec<u32>,
        /// Whether thinking mode is enabled for this request.
        enable_thinking: bool,
        /// Max thinking tokens before forcing `</think>`. None = unlimited.
        thinking_budget: Option<u32>,
        /// Whether a tool call is required (tool_choice="required").
        require_tool_call: bool,
        /// Suppress `<tool_call>` token when tool call loop detected (≥3 identical).
        suppress_tool_call: bool,
        /// Disable MTP speculative decoding for this sequence. Set when the
        /// request has tools active and `ATLAS_DISABLE_MTP_FOR_TOOLS=1` is set
        /// (default OFF; see `api::failures::circuit_f60`).
        disable_mtp: bool,
        /// Grammar specification for constrained decoding (tools or response_format).
        grammar_spec: Option<GrammarSpec>,
        /// Seed for deterministic sampling. None = non-deterministic.
        seed: Option<u64>,
        /// Number of top logprobs to return per token. None = disabled.
        top_logprobs: Option<u8>,
        /// Request timeout as absolute deadline. None = no timeout.
        timeout_at: Option<std::time::Instant>,
        decode_timeout_at: Option<std::time::Instant>,
        /// Opt-in report-only audit state shared with the scheduler.
        intervention_ledger: Option<crate::openai::SharedInterventionLedger>,
        /// Thinking-nudge config of the request's settings snapshot (`None`:
        /// off). Carried with the request so the nudge and the other request
        /// defaults always come from the same snapshot.
        think_nudge: Option<std::sync::Arc<crate::scheduler::think_nudge::ThinkNudgeConfig>>,
        response_tx: tokio::sync::oneshot::Sender<anyhow::Result<InferenceResponse>>,
    },
    /// Streaming: sends tokens as they're generated.
    Streaming {
        prompt_tokens: Vec<u32>,
        /// Session hash for SSM snapshot isolation (hash of first 64 prompt tokens).
        session_hash: u64,
        /// Preprocessed image data: (pixels `[P,1536]` f32, grid_h, grid_w) per image.
        image_pixels: Vec<(Vec<f32>, usize, usize)>,
        max_tokens: usize,
        /// Minimum tokens before allowing EOS/stop (0 = no minimum).
        min_tokens: usize,
        /// vLLM `ignore_eos`: model EOS ids do not end the turn.
        ignore_eos: bool,
        temperature: f32,
        /// Top-k: keep only the k highest-probability tokens (0 = disabled).
        top_k: u32,
        /// Top-p (nucleus): cumulative probability threshold (1.0 = disabled).
        top_p: f32,
        /// Top-n-sigma: filter tokens by logit z-score (0.0 = disabled).
        top_n_sigma: f32,
        /// Min-p: keep tokens with prob >= min_p * max_prob (0.0 = disabled).
        min_p: f32,
        /// Repetition penalty (1.0 = disabled).
        repetition_penalty: f32,
        /// Presence penalty — OpenAI-style, additive (0.0 = disabled).
        presence_penalty: f32,
        /// Frequency penalty — OpenAI-style, additive (0.0 = disabled).
        frequency_penalty: f32,
        /// DRY (Don't-Repeat-Yourself) n-gram sampler (see Blocking
        /// variant for rationale).
        dry_multiplier: f32,
        dry_base: f32,
        dry_allowed_length: u32,
        /// LZ penalty (A.1, see Blocking variant).
        lz_penalty: f32,
        /// Per-token logit bias: (token_id, bias_value) pairs.
        logit_bias: Vec<(u32, f32)>,
        /// Per-request stop token IDs (from OpenAI `stop` parameter).
        stop_tokens: Vec<u32>,
        /// Whether thinking mode is enabled for this request.
        enable_thinking: bool,
        /// Max thinking tokens before forcing `</think>`. None = unlimited.
        thinking_budget: Option<u32>,
        /// Whether a tool call is required (tool_choice="required").
        require_tool_call: bool,
        /// Suppress `<tool_call>` token when tool call loop detected (≥3 identical).
        suppress_tool_call: bool,
        /// Disable MTP speculative decoding for this sequence. Set when the
        /// request has tools active and `ATLAS_DISABLE_MTP_FOR_TOOLS=1` is set
        /// (default OFF; see `api::failures::circuit_f60`).
        disable_mtp: bool,
        /// Grammar specification for constrained decoding (tools or response_format).
        grammar_spec: Option<GrammarSpec>,
        /// Seed for deterministic sampling. None = non-deterministic.
        seed: Option<u64>,
        /// Number of top logprobs to return per token. None = disabled.
        top_logprobs: Option<u8>,
        /// Request timeout as absolute deadline. None = no timeout.
        timeout_at: Option<std::time::Instant>,
        decode_timeout_at: Option<std::time::Instant>,
        /// Opt-in report-only audit state shared with the scheduler.
        intervention_ledger: Option<crate::openai::SharedInterventionLedger>,
        /// Thinking-nudge config of the request's settings snapshot (`None`:
        /// off). Carried with the request so the nudge and the other request
        /// defaults always come from the same snapshot.
        think_nudge: Option<std::sync::Arc<crate::scheduler::think_nudge::ThinkNudgeConfig>>,
        token_tx: tokio::sync::mpsc::Sender<StreamEvent>,
        /// Cooperative cancellation flag, shared with the streaming
        /// pipeline. Set true by chat_stream guards (tool-call loop
        /// cap, watchdog, etc.) to ask the scheduler to terminate
        /// the sequence at the next decode boundary. The scheduler
        /// reads it in `emit_step::emit_token`; flipping it true is
        /// equivalent to receiving an EOS — `a.finished = true`, the
        /// usual finalize path runs, and `handle_done` emits the
        /// proper final-chunk (`finish_reason="length"` via the
        /// `tool_loop_capped` override) plus `[DONE]`. Without this,
        /// `stop_string_triggered` only suppresses *output*; the
        /// scheduler keeps generating until natural EOS / max_tokens,
        /// which on a degenerate-loop response can hang.
        cancel_flag: std::sync::Arc<std::sync::atomic::AtomicBool>,
    },
}

/// Per-token logprobs data extracted from the logits buffer.
#[derive(Clone)]
pub struct TokenLogprobs {
    /// The sampled token ID.
    pub token_id: u32,
    /// Log-probability of the sampled token.
    pub logprob: f32,
    /// Top-K alternative tokens with their logprobs, sorted descending.
    pub top: Vec<(u32, f32)>,
}

/// Response from the scheduler (blocking path).
pub struct InferenceResponse {
    pub output_tokens: Vec<u32>,
    /// Visible-output index where an implicitly/silently closed reasoning
    /// phase ends. A literal `</think>` remains self-describing in
    /// `output_tokens`; this fallback covers hidden EOS/control boundaries.
    pub thinking_output_boundary: Option<usize>,
    pub finish_reason: String,
    pub time_to_first_token_ms: f64,
    pub decode_time_ms: f64,
    /// Per-token logprobs (populated when top_logprobs is requested).
    pub logprobs: Vec<TokenLogprobs>,
    /// Number of generated tokens that were inside the thinking/reasoning
    /// block (counted AS PART OF `output_tokens.len()`). Reported to clients
    /// as `usage.completion_tokens_details.reasoning_tokens`.
    pub reasoning_tokens: u32,
    /// Number of prompt tokens served by the prefix cache (no prefill compute
    /// cost). Reported as `usage.prompt_tokens_details.cached_tokens`.
    pub cached_prompt_tokens: u32,
    /// Thinking nudges injected into the reasoning (`FLUX_THINK_NUDGE`).
    /// Reported as `usage.completion_tokens_details.think_nudges` when > 0.
    pub think_nudges: u32,
    /// The server guard that ended generation (`"watchdog"` or
    /// `"catastrophic_guard"` on `"stop"` finishes, or `"request_timeout_stall"`
    /// / `"request_timeout_total"` on `"length"` finishes), reported as
    /// `flux_stop_reason`.
    pub stop_cause: Option<&'static str>,
}

/// Events sent during streaming generation.
pub enum StreamEvent {
    Token(u32),
    /// Multiple contiguous plain tokens emitted in one speculative verify step
    /// (`ATLAS_DFLASH_DEFER_EMIT=1`). Decoded in a single `incremental_decode`
    /// pass (with automatic per-token fallback on incomplete trailing UTF-8).
    TokenBatch(Vec<u32>),
    /// Token with logprobs data (when top_logprobs is requested).
    TokenWithLogprobs(u32, TokenLogprobs),
    /// One committed completion token intentionally has no text/token delta
    /// on the wire (for example, suppressed spontaneous reasoning). Chat SSE
    /// uses this to advance opt-in cumulative usage without exposing content.
    CompletionProgress,
    /// A committed completion token intentionally hidden from the text decoder,
    /// with the logprob payload requested by the client.  Stop/EOS and other
    /// structural terminal tokens use this event so their probability and
    /// usage are published without ever decoding or leaking their text.
    CompletionProgressWithLogprobs(TokenLogprobs),
    /// The scheduler ended reasoning without publishing a literal `</think>`
    /// token (for example, Qwen's first in-thinking EOS). This is a pure phase
    /// transition: it does not advance completion usage or carry logprobs.
    ThinkingDone,
    Done {
        finish_reason: String,
        prompt_tokens: usize,
        /// Authoritative ordered completion sequence owned by the scheduler.
        /// Its length is exactly `completion_tokens`, including committed
        /// tokens intentionally suppressed from text events.
        completion_token_ids: Vec<u32>,
        completion_tokens: usize,
        time_to_first_token_ms: f64,
        decode_time_ms: f64,
        /// Tokens inside the thinking/reasoning block (for usage details).
        reasoning_tokens: u32,
        /// Prefix-cached prompt tokens (for usage details).
        cached_prompt_tokens: u32,
        /// Thinking nudges injected into the reasoning (for usage details).
        think_nudges: u32,
        /// The server guard that ended generation, if any.
        stop_cause: Option<&'static str>,
    },
    Error(String),
}
