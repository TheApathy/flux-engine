// SPDX-License-Identifier: AGPL-3.0-only

//! Shared scheduler-internal type definitions. Visibility is `pub(super)`
//! throughout so sibling modules (`decode_step`, `mtp_step`, `lifecycle`, etc.)
//! access them via `super::*`.

#![allow(dead_code)]

use std::time::Instant;

use anyhow::Result;
use spark_model::traits::SequenceState;

use crate::api::{InferenceRequest, InferenceResponse, StreamEvent};
use crate::grammar::GrammarState;

/// Shared queue between receiver thread and scheduler.
pub(super) struct PendingQueue {
    pub requests: Vec<InferenceRequest>,
    pub closed: bool,
}

/// How to deliver results for an active sequence.
pub(super) enum ResponseSink {
    Blocking(Option<tokio::sync::oneshot::Sender<Result<InferenceResponse>>>),
    Streaming(tokio::sync::mpsc::Sender<StreamEvent>),
}

impl ResponseSink {
    /// True once nobody can receive what this sink produces: the HTTP handler
    /// future was dropped (client disconnect / proxy timeout), so the
    /// streaming receiver or the blocking oneshot receiver is gone. A sequence
    /// whose sink is closed is pure GPU waste and must be retired like a
    /// cancelled one. Cheap: both tokio senders answer from an atomic.
    ///
    /// Blocking requests have no `cancel_flag`, so this is their ONLY
    /// disconnect signal; before it existed a disconnected blocking client
    /// kept its slot busy until `max_tokens`.
    pub fn is_closed(&self) -> bool {
        match self {
            ResponseSink::Streaming(tx) => tx.is_closed(),
            // `None` means the terminal response was already taken and sent;
            // the sequence is finishing anyway, so it is not "disconnected".
            ResponseSink::Blocking(tx) => tx.as_ref().is_some_and(|tx| tx.is_closed()),
        }
    }
}

/// An in-progress chunked prefill (prompt being processed in chunks).
pub(super) struct PrefillInProgress {
    pub prompt_tokens: Vec<u32>,
    pub session_hash: u64,
    /// Deterministic prefix hash over prompt_tokens + multimodal image digest.
    pub prompt_prefix_hash: u64,
    /// Incremental deterministic prefix hash computed alongside session_hash.
    pub prefix_hash: u64,
    /// Canary/debug reset proof initialized at prefill slot claim time.
    pub reset_proof: Option<super::continuation::ResetProofPayload>,
    pub seq: SequenceState,
    pub chunk_offset: usize,
    /// Per-request chunk stride established at chunk 0 so continuation chunks
    /// remain aligned even if active decode streams drain mid-prefill.
    pub chunk_budget: usize,
    pub max_tokens: usize,
    pub min_tokens: usize,
    pub eos_tokens: Vec<u32>,
    /// vLLM `ignore_eos`: `eos_tokens` holds only the request's stop tokens.
    pub ignore_eos: bool,
    pub sink: ResponseSink,
    /// Opt-in report-only audit state shared with the API response path.
    pub intervention_ledger: Option<crate::openai::SharedInterventionLedger>,
    /// Thinking-nudge config the request carried; becomes the sequence state.
    pub think_nudge: Option<std::sync::Arc<super::think_nudge::ThinkNudgeConfig>>,
    /// Cooperative cancellation flag — see ActiveSeq for the contract.
    /// Propagated to ActiveSeq when this PrefillInProgress promotes.
    pub cancel_flag: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
    pub request_start: Instant,
    pub temperature: f32,
    pub top_k: u32,
    pub top_p: f32,
    pub top_n_sigma: f32,
    pub min_p: f32,
    pub repetition_penalty: f32,
    pub repetition_penalty_window: u32,
    pub presence_penalty: f32,
    pub frequency_penalty: f32,
    pub lz_penalty: f32,
    pub dry_multiplier: f32,
    pub dry_base: f32,
    pub dry_allowed_length: u32,
    pub dry_sequence_breakers: Vec<u32>,
    pub logit_bias: Vec<(u32, f32)>,
    pub enable_thinking: bool,
    pub thinking_budget: Option<u32>,
    /// Per-server spontaneous-thinking budget (from MODEL.toml
    /// `[behavior].max_thinking_budget`). When the model emits a
    /// `<think>` token without the request having explicitly enabled
    /// thinking, this caps how many thinking tokens it can produce
    /// before `</think>` is force-emitted. Replaces a previous
    /// hard-coded 512-token fallback.
    pub spontaneous_think_budget: u32,
    pub require_tool_call: bool,
    pub suppress_tool_call: bool,
    /// F60 (2026-04-27): MTP-disable flag (propagated to ActiveSeq).
    pub disable_mtp: bool,
    pub grammar_state: Option<GrammarState>,
    /// Effective request seed. Positive-temperature requests materialize this
    /// once from either the explicit API seed or OS entropy before sampling.
    pub seed: Option<u64>,
    pub top_logprobs: Option<u8>,
    /// Logprobs of the first generated token, sampled from the last prefill
    /// chunk's logits (set when that token is sampled; `top_logprobs` only).
    pub first_logprobs: Option<crate::api::TokenLogprobs>,
    pub timeout_at: Option<Instant>,
    /// Deadline that replaces `timeout_at` once the first token is emitted
    /// (request-lifecycle contract). None = keep `timeout_at`.
    pub decode_timeout_at: Option<Instant>,
}

/// Last observed output length and when it was observed, for the stall
/// timeout (`FLUX_STALL_TIMEOUT`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct StallMark {
    pub tokens: usize,
    pub at: Option<Instant>,
}

/// An in-flight sequence participating in batched decode.
pub(super) struct ActiveSeq {
    pub seq: SequenceState,
    pub session_hash: u64,
    /// Deterministic prefix hash over prompt_tokens + multimodal image digest.
    pub prompt_prefix_hash: u64,
    /// Incremental deterministic prefix hash computed alongside session_hash
    /// per CONTINUATION_RECORD_DESIGN.md §5.1.
    pub prefix_hash: u64,
    /// Lifecycle state of the sequence's continuation record.
    pub record_state: super::continuation::RecordState,
    /// Canary/debug reset proof tracking when enabled.
    pub reset_proof: Option<super::continuation::ResetProofPayload>,
    pub last_token: u32,
    pub output_tokens: Vec<u32>,
    /// Absolute API response cap, including reasoning/thinking tokens.
    /// `remaining` separately tracks the content-token budget so thinking can
    /// remain free within this hard envelope.
    pub max_output_tokens: usize,
    pub remaining: usize,
    pub min_tokens: usize,
    pub eos_tokens: Vec<u32>,
    /// vLLM `ignore_eos`: `eos_tokens` holds only the request's stop tokens
    /// and the ChatML role boundary does not end the turn.
    pub ignore_eos: bool,
    pub finished: bool,
    pub sink: ResponseSink,
    /// Opt-in report-only audit state shared with the API response path.
    pub intervention_ledger: Option<crate::openai::SharedInterventionLedger>,
    /// Cooperative cancellation flag from the streaming pipeline.
    /// `Some` for streaming requests with the flag wired through;
    /// `None` for blocking requests. `emit_step::emit_token` reads
    /// it on every token and finalises the sequence when set —
    /// equivalent to receiving an EOS. Set by `chat_stream` guards
    /// (tool-call loop cap, loop-watchdog) so the scheduler stops
    /// generating instead of just having its output suppressed.
    pub cancel_flag: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
    pub temperature: f32,
    pub top_k: u32,
    pub top_p: f32,
    pub top_n_sigma: f32,
    pub min_p: f32,
    pub repetition_penalty: f32,
    pub repetition_penalty_window: u32,
    pub presence_penalty: f32,
    pub frequency_penalty: f32,
    pub lz_penalty: f32,
    pub dry_multiplier: f32,
    pub dry_base: f32,
    pub dry_allowed_length: u32,
    pub dry_sequence_breakers: Vec<u32>,
    pub logit_bias: Vec<(u32, f32)>,
    /// Tracks whether the model is inside `<think>...</think>` reasoning.
    pub inside_thinking: bool,
    /// Whether the request opted into thinking mode (`enable_thinking=true`).
    /// When false but the model spontaneously emits `<think>`, the thinking-content
    /// tokens MUST NOT be streamed to the client.
    pub enable_thinking: bool,
    /// Max thinking tokens before forcing `</think>`. None = unlimited.
    pub thinking_budget: Option<u32>,
    /// Per-server spontaneous-thinking budget (from MODEL.toml
    /// `[behavior].max_thinking_budget`).
    pub spontaneous_think_budget: u32,
    /// Number of thinking tokens generated so far (counted while inside_thinking).
    pub thinking_tokens: u32,
    /// When true, the next decode step must produce the `</think>` token.
    pub force_end_thinking: bool,
    /// Consecutive tokens where top-1 softmax prob >= 0.95 (for confidence early stop).
    pub consecutive_confident: u32,
    /// True while the model is inside an unclosed ``` code fence within
    /// the current thinking block. Toggled on each sampled code-fence
    /// token. F2 still tracks confidence and can arm `force_end_thinking`
    /// while this is set, but `</think>` injection is deferred
    /// (`should_inject_think_end`) until the fence closes (unless
    /// hard-overridden), so coherent code blocks are not truncated
    /// mid-statement. Per-seq state, persisted across decode steps and
    /// snapshots.
    pub in_code_fence: bool,
    /// Token ID for `</think>` (needed for budget enforcement in emit_token).
    pub think_end_token: Option<u32>,
    /// Token ID for `<think>` (needed for spontaneous thinking detection in emit_token).
    pub think_start_token: Option<u32>,
    /// True after the first `</think>` token is generated.
    pub think_ended: bool,
    /// One-shot signal: set when `</think>` was the most recently emitted token.
    pub think_just_ended: bool,
    /// Consecutive `</think>` tokens skipped outside thinking. Safety limit: 50.
    pub think_skip_count: u32,
    /// Committed content tokens since the latest `</think>` transition. The
    /// spec-entry dispatch pin consumes this counter; requests that never think
    /// count from the start of their response.
    pub post_think_gate_steps: u8,
    /// Token ID for `</tool_call>` — acts as a stop token for one-call-per-response.
    pub tool_call_end_token: Option<u32>,
    /// When true AND grammar_state is None, EOS tokens are suppressed until
    /// `<tool_call>` is generated (legacy fallback).
    pub require_tool_call: bool,
    /// Token ID for `<tool_call>` (legacy fallback when grammar is unavailable).
    pub tool_call_start_token: Option<u32>,
    /// True after `<tool_call>` generated in output (not inside thinking).
    pub tool_call_opened: bool,
    /// True between emission of `<tool_call>`/`<function=…>` (open) and
    /// `</tool_call>`/`</function>` (close).
    pub inside_tool_body: bool,
    /// When true, `<tool_call>` token logit is biased by `-12.0` during decode.
    pub suppress_tool_call: bool,
    /// F60 (2026-04-27): when true, MTP speculative decoding is bypassed.
    pub disable_mtp: bool,
    /// True after the first non-thinking content token has been generated.
    pub content_started: bool,
    /// Number of content tokens emitted post-`</think>`.
    pub content_tokens: u32,
    /// Free-text tokens emitted since the last `<tool_call>` opened.
    pub prose_tokens_since_last_tool: u32,
    /// F10 (2026-04-26): how many times the thinking-loop watchdog has fired.
    pub think_watchdog_fires: u32,
    /// Phase-C: how many times a degeneration watchdog has rolled this
    /// sequence back to a boundary and re-steered. Capped at
    /// [`atlas_kernels::ROLLBACK_RESTEER_CAP`]; once the cap is hit the
    /// watchdog reverts to a hard stop. See
    /// [`super::rollback::rollback_to_boundary`].
    pub rollback_count: u32,
    /// Phase-C: decode-time SSM-snapshot ring for hybrid (attention +
    /// Mamba/SSM) models. Records a bounded set of SSM `h_state` +
    /// `conv_state` snapshots taken at boundary tokens so a watchdog
    /// rollback can restore the recurrent state — not just the KV
    /// cache — to the chosen boundary. Disabled (`capacity == 0`,
    /// every op a no-op) for pure-attention models. See
    /// [`super::ssm_decode_ring::SsmDecodeRing`].
    pub ssm_rollback_ring: super::ssm_decode_ring::SsmDecodeRing,
    /// Grammar state for constrained decoding (tool_choice="required").
    pub grammar_state: Option<GrammarState>,
    /// MTP draft tokens awaiting verification.
    pub pending_drafts: Vec<u32>,
    /// Where `pending_drafts` came from. Consumed (and reset) by the
    /// verify path purely to attribute accepted tokens to the tier that
    /// proposed them; nothing reads it to make a decision. See
    /// [`crate::rest_store`].
    pub draft_origin: DraftOrigin,
    /// Drafts the neural proposer got accepted in this sequence's most
    /// recent γ-block verify. Read by the retrieval tiers as a one-sample
    /// estimate of how well the drafter is currently doing on THIS text:
    /// a drafter that just filled its frame must not be pre-empted. Zero
    /// before the first verify, which is when there is no evidence either
    /// way and retrieval is free to try.
    pub last_verify_accepted: u16,
    /// Self-context draft index over this sequence's own history.
    /// Empty and unallocated until a drafting tier syncs it, and
    /// dropped with the sequence. See
    /// [`crate::rest_store::self_context`].
    pub self_context: crate::rest_store::self_context::SelfContextIndex,
    /// DDTree M3: optional tree payload paired with `pending_drafts`.
    /// `None` for flat DFlash / MTP / ngram paths. When set by a
    /// DDTree-capable proposer (M4B+), the verifier reads ancestor
    /// metadata + parent ids from this payload instead of treating
    /// `pending_drafts` as a flat chain. Cleared in the same
    /// scheduler step the drafts are consumed.
    pub pending_tree_payload: Option<spark_model::layers::DDTreePayload>,
    /// Drain-once sparse-q frame paired atomically with `pending_drafts` for
    /// probabilistically sampled DFlash2 proposals. No verifier may retain or
    /// move this independently of its draft frame.
    pub pending_dflash_sampled_proposal:
        Option<spark_model::dflash_sampling::DflashSampledProposalFrame>,
    /// Timestamp of the last token emission (for TBT deadline tracking).
    pub last_token_time: Instant,
    /// Timestamp when the request entered prefill (for TTFT).
    pub request_start: Instant,
    /// Decode start time (set after prefill completes, for decode throughput).
    pub decode_start: Instant,
    /// Stable request-scoped base seed. Positive-temperature requests always
    /// have one; each output position derives its effective seed from it.
    pub seed: Option<u64>,
    /// Number of top logprobs to return per token. None = disabled.
    pub top_logprobs: Option<u8>,
    /// Accumulated logprobs data for blocking responses.
    pub logprobs_data: Vec<crate::api::TokenLogprobs>,
    /// Request timeout deadline (the derived total ceiling). None = no timeout.
    pub timeout_at: Option<Instant>,
    /// Deadline that replaces `timeout_at` once the first token is emitted
    /// (request-lifecycle contract). None = keep `timeout_at`.
    pub decode_timeout_at: Option<Instant>,
    /// Inter-token progress clock for the stall timeout. Unarmed until the
    /// first post-token check, so prefill time never counts as a stall.
    pub stall_mark: StallMark,
    /// Adaptive sampling state.
    pub adaptive: crate::adaptive_sampler::AdaptiveSamplingState,
    /// Number of prompt tokens served by the prefix cache (no prefill cost).
    pub cached_prompt_tokens: u32,
    /// Lever 3 (ATLAS_ADAPTIVE_THINK): per-sequence difficulty probe over the
    /// first ~48 thinking tokens. Accumulates top-1 confidence; once full it
    /// commits an adaptively-scaled `thinking_budget`. Inert (never observed,
    /// never committed) unless `ATLAS_ADAPTIVE_THINK=1`. See
    /// [`super::thinking_efficiency::DifficultyProbe`].
    pub difficulty_probe: super::thinking_efficiency::DifficultyProbe,
    /// Thinking-nudge state (`FLUX_THINK_NUDGE=on`). `None` when the nudge
    /// is off, which makes every hook a single `is_none` check. See
    /// [`super::think_nudge`].
    pub think_nudge: Option<Box<super::think_nudge::ThinkNudgeState>>,
    /// In-flight thinking-budget nudge (forced before the budget's forced
    /// `</think>`). See [`super::think_budget_nudge`].
    pub budget_nudge: Option<super::think_budget_nudge::BudgetNudge>,
    /// Why a server guard ended generation (`"watchdog"` or
    /// `"catastrophic_guard"`). `None` when the sequence ends on EOS, a stop
    /// token or its token budget. A set cause reports `finish_reason: "stop"`
    /// plus `flux_stop_reason`; `"length"` stays reserved for exhaustion.
    pub stop_cause: Option<&'static str>,
}

impl ActiveSeq {
    /// Return the deterministic prompt prefix hash (including multimodal digest).
    #[inline]
    pub fn prompt_prefix_hash(&self) -> u64 {
        self.prompt_prefix_hash
    }

    /// Return the deterministic incremental prefix hash.
    #[inline]
    pub fn prefix_hash(&self) -> u64 {
        self.prefix_hash
    }

    /// Return the session hash.
    #[inline]
    pub fn session_hash(&self) -> u64 {
        self.session_hash
    }

    /// Return the slot-isolated adaptive-depth controller key for this sequence.
    #[inline]
    pub(super) fn depth_controller_key(&self) -> u64 {
        super::adaptive_depth::seq_controller_key(self.session_hash, self.seq.slot_idx)
    }

    /// Return the lifecycle state of the sequence's continuation record.
    #[inline]
    pub fn record_state(&self) -> super::continuation::RecordState {
        self.record_state
    }

    /// Advance the deterministic incremental prefix hash with a newly committed token.
    #[inline]
    pub fn update_prefix_hash(&mut self, token: u32) {
        self.prefix_hash = super::continuation::hash_token_incremental(self.prefix_hash, token);
    }

    /// Return a borrowed reference view of this sequence as a ContinuationRecord.
    pub fn continuation_record(&self) -> super::continuation::ContinuationRecordRef<'_> {
        super::continuation::ContinuationRecordRef {
            seq: &self.seq,
            ssm_rollback_ring: &self.ssm_rollback_ring,
            session_hash: self.session_hash,
            prompt_prefix_hash: self.prompt_prefix_hash,
            prefix_hash: self.prefix_hash,
            state: self.record_state,
            reset_proof: self.reset_proof.as_ref(),
        }
    }

    /// Return a mutable borrowed view of this sequence as a ContinuationRecord.
    pub fn continuation_record_mut(&mut self) -> super::continuation::ContinuationRecordMut<'_> {
        super::continuation::ContinuationRecordMut {
            seq: &mut self.seq,
            ssm_rollback_ring: &mut self.ssm_rollback_ring,
            session_hash: self.session_hash,
            prompt_prefix_hash: self.prompt_prefix_hash,
            prefix_hash: &mut self.prefix_hash,
            state: &mut self.record_state,
            reset_proof: &mut self.reset_proof,
        }
    }
}

#[path = "swapped_seq.rs"]
mod swapped_seq;
pub(super) use swapped_seq::SwappedSeq;

/// Which drafting tier proposed the current `pending_drafts` frame.
///
/// Accounting only. The verifier treats every frame identically — a
/// retrieved chain is accepted token-by-token against the target's argmax
/// exactly as a neural draft is.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum DraftOrigin {
    /// The neural proposer (DFlash or native MTP).
    #[default]
    Proposer,
    /// The static `.rest` retrieval store.
    RestStore,
    /// The sequence's own token history.
    SelfContext,
}
