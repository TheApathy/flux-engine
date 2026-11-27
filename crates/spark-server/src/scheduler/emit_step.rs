// SPDX-License-Identifier: AGPL-3.0-only

//! emit_token + compile_grammar_state + StartPrefillResult enum.

use super::*;

#[path = "rethink_budget.rs"]
mod rethink_budget;
pub(super) use rethink_budget::resolve_rethink_budget;

/// Whether the ordinary EOS path must defer termination for this request.
///
/// This deliberately contains no blanket post-thinking minimum.  Required
/// tool calls already have two authoritative protections (the grammar matcher
/// and the legacy `require_tool_call` fallback), while an ordinary answer must
/// be allowed to stop on its first EOS after `</think>`.  Ignoring that EOS can
/// make the model continue into the next ChatML role boundary.
pub(super) fn eos_is_suppressed_by_non_thinking_guards(
    grammar_state: Option<&crate::grammar::GrammarState>,
    require_tool_call: bool,
    output_len: usize,
    min_tokens: usize,
) -> bool {
    grammar_state.is_some_and(|gs| !gs.is_terminated())
        || require_tool_call
        || output_len < min_tokens
}

fn eos_is_suppressed_without_thinking(a: &ActiveSeq, output_len: usize) -> bool {
    eos_is_suppressed_by_non_thinking_guards(
        a.grammar_state.as_ref(),
        a.require_tool_call,
        output_len,
        a.min_tokens,
    )
}

pub(super) fn eos_is_suppressed(a: &ActiveSeq, output_len: usize) -> bool {
    eos_is_suppressed_without_thinking(a, output_len)
        // deepseek_v41: the Python engine stops on EOS inside thinking too.
        || (a.inside_thinking && !crate::dsv41::serving())
}

/// If thinking is the only reason an EOS would be suppressed, optionally
/// treat that EOS as an implicit `</think>` boundary.
///
/// The sampled EOS itself remains hidden/discarded. Feeding it back gives the
/// model one clean content-phase position; if the model immediately samples
/// EOS again, thinking no longer suppresses it and the turn ends normally.
/// Other stop constraints (grammar, required tool use, or `min_tokens`) keep
/// their existing authority and prevent this transition.
pub(super) fn maybe_implicit_close_thinking_on_eos(
    a: &mut ActiveSeq,
    tok: u32,
    output_len: usize,
    honor_eos_inside_thinking: bool,
) -> bool {
    if !honor_eos_inside_thinking
        || crate::dsv41::serving()
        || !a.inside_thinking
        || !a.eos_tokens.contains(&tok)
        || eos_is_suppressed_without_thinking(a, output_len)
    {
        return false;
    }

    a.inside_thinking = false;
    a.force_end_thinking = false;
    super::think_budget_nudge::on_think_closed(a);
    a.consecutive_confident = 0;
    a.in_code_fence = false;
    a.think_ended = true;
    a.think_just_ended = true;
    a.post_think_gate_steps = 0;
    tracing::debug!(
        target: "flux::eos",
        tok,
        thinking_tokens = a.thinking_tokens,
        "EOS implicitly closed thinking; discarding this EOS before content"
    );
    publish_thinking_done(a);
    true
}

/// Tell streaming consumers that reasoning ended without a visible delimiter.
/// This event is structural, not a completion token, so it carries neither
/// usage nor logprobs. Blocking responses recover the same boundary from the
/// scheduler's content-token count at finalization.
pub(super) fn publish_thinking_done(a: &mut ActiveSeq) {
    if !matches!(a.sink, ResponseSink::Streaming(_)) {
        return;
    }
    if !publish_stream_event(a, StreamEvent::ThinkingDone, "thinking_done") {
        tracing::debug!("stream dropped or stalled before hidden thinking boundary");
    }
}

/// ChatML role boundaries end the assistant turn even when ordinary EOS is
/// constrained. The only exception is a boundary sampled inside `<think>`,
/// where special end tokens remain suppressed until `</think>`. An
/// `ignore_eos` request opts out: as in vLLM, it runs on to `max_tokens`.
pub(super) fn is_chatml_role_boundary(a: &ActiveSeq, tok: u32, im_start: Option<u32>) -> bool {
    im_start == Some(tok) && !a.inside_thinking && !a.ignore_eos
}

/// Commit a ChatML role boundary before any ordinary token policy runs.
///
/// Both serial decode and speculative emission use this edge.  In particular,
/// grammar / required-tool / minimum-token EOS suppression must never discard
/// `<|im_start|>` and allow the following role text onto the assistant stream.
pub(super) fn commit_chatml_role_boundary(
    a: &mut ActiveSeq,
    tok: u32,
    im_start: Option<u32>,
    logprobs: &Option<crate::api::TokenLogprobs>,
) -> bool {
    if !is_chatml_role_boundary(a, tok, im_start) {
        return false;
    }
    // Keep the stop token in output_tokens so lifecycle reports `stop`.  The
    // API text path strips registered stop tokens, so it is never client text.
    a.output_tokens.push(tok);
    a.update_prefix_hash(tok);
    if let Some(lp) = logprobs {
        a.logprobs_data.push(lp.clone());
    }
    publish_hidden_committed_token(a, logprobs, "ChatML role boundary");
    a.finished = true;
    tracing::debug!(
        id = im_start.unwrap_or_default(),
        "<|im_start|> hard-stop fired; ending turn before grammar/suppress_eos"
    );
    true
}

/// Deliver one event to a streaming sink without ever parking the scheduler.
///
/// There is exactly one scheduler thread for every live sequence. Waiting for
/// one full SSE channel here, even with a deadline, delays every healthy lane
/// by that deadline. The channel is already a 1024-event burst buffer; once it
/// is full the consumer has fallen behind by far more than one decode step, so
/// retire that sequence immediately and let the other lanes progress.
///
/// This deliberately has no environment-controlled blocking fallback. The old
/// timeout was process-latched with `OnceLock` and `0` restored an unbounded
/// wait, making a live recipe change ineffective and allowing one peer to halt
/// the engine. Backpressure belongs at HTTP admission, not on the GPU owner.
pub(super) fn stream_send_bounded(
    tx: &tokio::sync::mpsc::Sender<StreamEvent>,
    event: StreamEvent,
    context: &str,
) -> bool {
    use tokio::sync::mpsc::error::TrySendError;
    match tx.try_send(event) {
        Ok(()) => true,
        Err(TrySendError::Closed(_)) => {
            tracing::debug!(context, "Streaming receiver dropped, finishing seq");
            false
        }
        Err(TrySendError::Full(_)) => {
            tracing::warn!(
                context,
                "Streaming channel exhausted its burst buffer; retiring the sequence without \
                 blocking the scheduler"
            );
            false
        }
    }
}

/// A suppressed EOS still consumes a target position. Once the content or
/// absolute completion budget is already exhausted, no policy guard may keep
/// the request alive to sample and discard EOS forever.
pub(super) fn retire_if_budget_exhausted(a: &mut ActiveSeq, context: &str) -> bool {
    if a.remaining > 0 && a.output_tokens.len() < a.max_output_tokens {
        return false;
    }
    tracing::info!(
        context,
        remaining = a.remaining,
        output_tokens = a.output_tokens.len(),
        max_output_tokens = a.max_output_tokens,
        "Retiring sequence at exhausted completion budget"
    );
    a.finished = true;
    true
}

/// Deliver one streaming event immediately (default), or queue it inside
/// `a.defer_emit` when `ATLAS_DFLASH_DEFER_EMIT=1` is armed for the active
/// verify step. Retires `a.finished = true` immediately when the receiver has
/// already closed so commit/propose work is never wasted on a dead stream.
pub(super) fn publish_stream_event(a: &mut ActiveSeq, event: StreamEvent, context: &str) -> bool {
    if !matches!(a.sink, ResponseSink::Streaming(_)) {
        return true;
    }
    if a.defer_emit.is_active() {
        if a.sink.is_closed() {
            tracing::debug!(context, "Streaming receiver dropped, finishing seq");
            a.finished = true;
            return false;
        }
        a.defer_emit.push_event(event);
        return true;
    }
    let ResponseSink::Streaming(tx) = &a.sink else {
        return true;
    };
    if !stream_send_bounded(tx, event, context) {
        a.finished = true;
        return false;
    }
    true
}

/// Flush any queued deferred SSE events (`ATLAS_DFLASH_DEFER_EMIT=1`) in exact
/// FIFO order and disarm deferral. No-op when deferral is inactive and empty.
pub(super) fn flush_deferred_emit(a: &mut ActiveSeq) {
    if !a.defer_emit.has_pending() {
        return;
    }
    let events = a.defer_emit.drain_events();
    if events.is_empty() {
        return;
    }
    let ResponseSink::Streaming(tx) = &a.sink else {
        return;
    };
    for event in events {
        if !stream_send_bounded(tx, event, "deferred_emit_flush") {
            a.finished = true;
            break;
        }
    }
}

/// Publish one committed scheduler token which must never enter either text
/// decoder. This is used for structural/terminal tokens and for suppressed
/// reasoning. The event still carries requested logprobs and advances opt-in
/// continuous usage exactly once.
pub(super) fn publish_hidden_committed_token(
    a: &mut ActiveSeq,
    logprobs: &Option<crate::api::TokenLogprobs>,
    context: &str,
) {
    if !matches!(a.sink, ResponseSink::Streaming(_)) {
        return;
    }
    let event = committed_token_event(0, logprobs, true);
    let _ = publish_stream_event(a, event, context);
}

/// Advance the sampler's tool-body phase after committing one content token.
/// The content watchdog must observe the phase on entry (so a closing tag still
/// belongs to the body), while the next token's sampler must observe the new
/// phase. Both serial and speculative emission call this helper at that edge.
pub(super) fn update_tool_body_phase(a: &mut ActiveSeq, tok: u32) {
    if a.inside_thinking {
        return;
    }
    if a.tool_call_start_token == Some(tok) {
        a.inside_tool_body = true;
        a.prose_tokens_since_last_tool = 0;
    } else if a.tool_call_end_token == Some(tok) {
        a.inside_tool_body = false;
    }
}

/// Safety valve: if `require_tool_call` is still set after 512 committed
/// tokens, the model isn't generating a tool call (grammar may have failed to
/// compile). Clear the flag so EOS is no longer suppressed and generation
/// cannot run on indefinitely.
///
/// `committed_len` is `output_tokens.len()` before the current token is
/// pushed, matching the plain decoder's frame across serial decode,
/// `emit_step`, and `think_spec_accept`.
pub(super) fn clear_stale_require_tool_call(a: &mut ActiveSeq, committed_len: usize) {
    if a.require_tool_call && committed_len > 512 {
        tracing::warn!(
            "require_tool_call safety: no <tool_call> after 512 tokens, clearing EOS suppression"
        );
        a.require_tool_call = false;
    }
}

/// Advance a content grammar exactly once for a committed non-thinking token.
/// Both serial decode and speculative emission use this edge; special token
/// handlers must not consume the same token again.
pub(super) fn advance_content_grammar(a: &mut ActiveSeq, tok: u32) {
    if !a.inside_thinking
        && let Some(ref mut gs) = a.grammar_state
    {
        gs.accept_token(tok);
    }
}

/// Commit a non-thinking tool-call close with one shared serial/spec contract.
///
/// Callers own the per-token content counters and must advance the grammar and
/// tool-body phase first.  This helper owns the one output push, the one stream
/// event, and all close-boundary termination decisions.
pub(super) fn commit_tool_call_close(a: &mut ActiveSeq, tok: u32, was_inside_thinking: bool) {
    debug_assert!(!a.inside_thinking);
    debug_assert_eq!(a.tool_call_end_token, Some(tok));

    a.output_tokens.push(tok);
    a.update_prefix_hash(tok);
    if a.think_ended && !was_inside_thinking {
        a.post_think_gate_steps = mtp_gate::advance_entry_counter(a.post_think_gate_steps);
    }

    if matches!(a.sink, ResponseSink::Streaming(_)) {
        let event = if let Some(lp) = a.logprobs_data.last().cloned() {
            StreamEvent::TokenWithLogprobs(tok, lp)
        } else {
            StreamEvent::Token(tok)
        };
        let _ = publish_stream_event(a, event, "tool_call_end");
    }

    // A stop-after-first grammar (required/specific tool_choice, or
    // `parallel_tool_calls=false`) admits nothing but a stop token once this
    // close commits; ending here saves the grammar-forced EOS step and keeps
    // `finish_reason` on the close token. A looping grammar still admits
    // prose or another call.
    let grammar_terminal = a
        .grammar_state
        .as_mut()
        .is_some_and(|gs| gs.only_stop_tokens_remain());
    if a.grammar_state.is_none() || grammar_terminal {
        // Legacy mode is one-call-per-response. A stop-after-first grammar has
        // the same boundary once its close token completes the matcher.
        a.finished = true;
    } else {
        // A nonterminal multi-call grammar may legitimately think again before
        // its next call; post-think masks must not remain latched across calls.
        a.think_ended = false;
    }

    if a.output_tokens.len() >= a.max_output_tokens || a.remaining == 0 {
        a.finished = true;
    }
}

/// Return a fuzzy tail-loop only when the current token stream is outside a
/// tool call. This is shared by serial decode and speculative emission so both
/// paths use identical start/end-token framing.
pub(super) fn fuzzy_repetition_outside_tool(a: &ActiveSeq) -> Option<(usize, usize, usize)> {
    if a.inside_thinking {
        return None;
    }
    let last_tc_start = a
        .tool_call_start_token
        .and_then(|t| a.output_tokens.iter().rposition(|&tok| tok == t));
    let last_tc_end = a
        .tool_call_end_token
        .and_then(|t| a.output_tokens.iter().rposition(|&tok| tok == t));
    let inside_tool_call = match (last_tc_start, last_tc_end) {
        (Some(start), Some(end)) => start > end,
        (Some(_), None) => true,
        _ => false,
    };
    if inside_tool_call {
        None
    } else {
        detect_fuzzy_repetition(&a.output_tokens)
    }
}

/// O(1) amortized variant of [`fuzzy_repetition_outside_tool`] that tracks the
/// latest `<tool_call>` / `</tool_call>` positions incrementally in
/// `a.defer_emit` instead of running two O(N) reverse scans per token.
pub(super) fn fuzzy_repetition_outside_tool_tracked(
    a: &mut ActiveSeq,
) -> Option<(usize, usize, usize)> {
    if a.inside_thinking {
        return None;
    }
    let (last_tc_start, last_tc_end) = a.defer_emit.sync_tool_call_positions(
        &a.output_tokens,
        a.tool_call_start_token,
        a.tool_call_end_token,
    );
    let inside_tool_call = match (last_tc_start, last_tc_end) {
        (Some(start), Some(end)) => start > end,
        (Some(_), None) => true,
        _ => false,
    };
    if inside_tool_call {
        None
    } else {
        detect_fuzzy_repetition(&a.output_tokens)
    }
}

/// Stop speculative emission on a fuzzy tail-loop. The serial path can roll
/// back to a per-token SSM snapshot; a multi-token verify batch cannot safely
/// do that after commit, so its parity behavior is the conservative hard stop.
pub(super) fn apply_speculative_fuzzy_stop(a: &mut ActiveSeq) -> bool {
    let Some((pattern_len, mis_a, mis_b)) = fuzzy_repetition_outside_tool_tracked(a) else {
        return false;
    };
    record_watchdog_stop(a, "fuzzy_repeat_watchdog");
    tracing::warn!(
        pattern_len,
        mismatches = mis_a + mis_b,
        output_len = a.output_tokens.len(),
        "Fuzzy repetition detected during speculative emit; ending response early (batched rollback unavailable)"
    );
    a.finished = true;
    true
}

/// Why a request was ended by the timeout contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum TimeoutReason {
    /// No new token for `FLUX_STALL_TIMEOUT` seconds.
    Stall,
    /// The derived total ceiling (`timeout_at`) passed.
    TotalDeadline,
}

impl TimeoutReason {
    pub(super) fn as_str(self) -> &'static str {
        match self {
            Self::Stall => "stall",
            Self::TotalDeadline => "total_deadline",
        }
    }

    /// The `stop_cause` / `flux_stop_reason` reported to the client.
    pub(super) fn stop_cause(self) -> &'static str {
        match self {
            Self::Stall => crate::api::request_contract::TIMEOUT_STALL_CAUSE,
            Self::TotalDeadline => crate::api::request_contract::TIMEOUT_TOTAL_CAUSE,
        }
    }
}

/// Apply the request deadline at the same post-token boundary used by serial
/// decode. Thinking and content-speculation paths share this implementation.
pub(super) fn check_request_timeout(a: &mut ActiveSeq) {
    let stall_s = crate::api::request_contract::stall_timeout_s();
    let stall = (stall_s > 0).then(|| std::time::Duration::from_secs(stall_s));
    check_request_timeout_at(a, Instant::now(), stall);
}

/// The timeout contract at `now`, run at the post-token boundary.
///
/// * The first call (the first emitted token) arms the stall clock and
///   switches the total ceiling from the pre-first-token deadline (server
///   base or explicit request timeout, which also bounds prefill) to the
///   derived decode deadline when the request has one.
/// * Progress is the high-water mark of committed output: draft tokens live
///   in `pending_drafts` until verified, and a rollback that trims the
///   vector and re-grows it to the same length is not progress.
/// * Progress is tested before the gap, so a scheduler that was itself held
///   up past the limit does not kill a request that just produced a token.
/// * A timeout sets `stop_cause` so the client sees `flux_stop_reason` and a
///   `warnings` entry; `finish_reason` stays `length`.
pub(super) fn check_request_timeout_at(
    a: &mut ActiveSeq,
    now: Instant,
    stall: Option<std::time::Duration>,
) {
    if a.finished {
        return;
    }
    let produced = a.output_tokens.len();
    if a.stall_mark.at.is_none()
        && let Some(decode_deadline) = a.decode_timeout_at.take()
    {
        a.timeout_at = Some(decode_deadline);
    }
    let progressed = a.stall_mark.at.is_none() || produced > a.stall_mark.tokens;
    if progressed {
        a.stall_mark = StallMark {
            tokens: produced.max(a.stall_mark.tokens),
            at: Some(now),
        };
    }
    let stalled = !progressed
        && stall.is_some_and(|limit| {
            a.stall_mark
                .at
                .is_some_and(|at| now.saturating_duration_since(at) >= limit)
        });
    let reason = if stalled {
        Some(TimeoutReason::Stall)
    } else {
        a.timeout_at
            .filter(|&deadline| now >= deadline)
            .map(|_| TimeoutReason::TotalDeadline)
    };
    if let Some(reason) = reason {
        tracing::warn!(
            "Request timeout: reason={} elapsed_s={:.2} output_tokens={produced} stall_s={}",
            reason.as_str(),
            now.saturating_duration_since(a.request_start).as_secs_f64(),
            stall.map_or_else(|| "off".to_string(), |d| d.as_secs().to_string()),
        );
        a.stop_cause = a.stop_cause.or(Some(reason.stop_cause()));
        a.finished = true;
    }
}

/// Observe the request's monotonic cooperative-cancellation flag, or a sink
/// whose receiver has gone away (client disconnect). Both are terminal for
/// the sequence: nothing can consume further tokens, so generating them only
/// burns GPU time and holds a batch slot. The sink check is what gives
/// blocking requests (no `cancel_flag`) disconnect detection at all.
pub(super) fn request_cancelled(a: &ActiveSeq) -> bool {
    a.cancel_flag
        .as_ref()
        .is_some_and(|flag| flag.load(std::sync::atomic::Ordering::Acquire))
        || a.sink.is_closed()
        // Shutdown retirement: the drain window is over, stop at this token.
        || super::retirement::requested()
}

/// Retire cancellation as an ordinary scheduler lifecycle outcome.
///
/// This is deliberately distinct from deadline, budget, and policy failures:
/// those remain fail-closed errors.  The first observer emits one INFO receipt;
/// subsequent boundaries only report that the request is already retired.
pub(super) fn retire_cancelled_speculation(a: &mut ActiveSeq, stage: &'static str) -> bool {
    if !request_cancelled(a) {
        return false;
    }
    let newly_retired = !a.finished;
    proposal_lifecycle::clear_if_cancelled(
        true,
        &mut a.pending_drafts,
        &mut a.pending_tree_payload,
        &mut a.pending_dflash_sampled_proposal,
    );
    a.finished = true;
    if newly_retired {
        tracing::info!(stage, outcome = "client_cancelled", "SPEC_CANCEL_RETIRED");
    }
    true
}

/// Emit a token for an active sequence (stream + bookkeeping).
///
/// Per OpenAI spec, stop/EOS tokens are NOT streamed to the client —
/// the returned text must not contain the stop sequence. The token is
/// still recorded in output_tokens for accurate token counting.
///
/// When `logprobs` is Some, the logprobs data is accumulated for blocking
/// responses and sent via `StreamEvent::TokenWithLogprobs` for streaming.
pub fn emit_token(a: &mut ActiveSeq, tok: u32, logprobs: Option<crate::api::TokenLogprobs>) {
    emit_token_with_eos_policy(a, tok, logprobs, honor_eos_inside_thinking());
}

/// Whether this verify row reached token classification. Hidden control rows
/// still advance the accepted target prefix; a pre-row cancellation does not.
pub(super) fn emit_token_for_verify(
    a: &mut ActiveSeq,
    tok: u32,
    logprobs: Option<crate::api::TokenLogprobs>,
) -> bool {
    emit_token_with_policies(
        a,
        tok,
        logprobs,
        honor_eos_inside_thinking(),
        enable_loop_watchdog(),
        enable_catastrophic_loop_watchdog(),
    )
}

/// [`emit_token`] with an explicit per-model EOS policy.  Production reads
/// the refreshable model behavior above; the explicit form keeps policy
/// regression tests independent of process-global model-load state.
pub(super) fn emit_token_with_eos_policy(
    a: &mut ActiveSeq,
    tok: u32,
    logprobs: Option<crate::api::TokenLogprobs>,
    honor_eos_inside_thinking: bool,
) {
    emit_token_with_policies(
        a,
        tok,
        logprobs,
        honor_eos_inside_thinking,
        enable_loop_watchdog(),
        enable_catastrophic_loop_watchdog(),
    );
}

fn emit_token_with_policies(
    a: &mut ActiveSeq,
    tok: u32,
    logprobs: Option<crate::api::TokenLogprobs>,
    honor_eos_inside_thinking: bool,
    loop_watchdog_enabled: bool,
    catastrophic_loop_watchdog_enabled: bool,
) -> bool {
    let (now, stall) = if a.defer_emit.is_active() {
        a.defer_emit.step_clock()
    } else {
        let stall_s = crate::api::request_contract::stall_timeout_s();
        let stall = (stall_s > 0).then(|| std::time::Duration::from_secs(stall_s));
        (Instant::now(), stall)
    };
    emit_token_with_policies_at(
        a,
        tok,
        logprobs,
        honor_eos_inside_thinking,
        loop_watchdog_enabled,
        catastrophic_loop_watchdog_enabled,
        now,
        stall,
    )
}

#[allow(clippy::too_many_arguments)]
fn emit_token_with_policies_at(
    a: &mut ActiveSeq,
    tok: u32,
    logprobs: Option<crate::api::TokenLogprobs>,
    honor_eos_inside_thinking: bool,
    loop_watchdog_enabled: bool,
    catastrophic_loop_watchdog_enabled: bool,
    now: Instant,
    stall: Option<std::time::Duration>,
) -> bool {
    let output_len_before = a.output_tokens.len();
    let classified = emit_token_inner(
        a,
        tok,
        logprobs,
        honor_eos_inside_thinking,
        loop_watchdog_enabled,
        catastrophic_loop_watchdog_enabled,
    );
    let committed = a.output_tokens.len().saturating_sub(output_len_before);
    if committed > 0 {
        crate::metrics::compat_metrics::add_generation_tokens(committed as u64);
    }
    if classified {
        // Single post-classification timeout choke point so suppressed EOS and
        // structural tokens cannot spin past `timeout_at` or the stall limit.
        // Finished sequences short-circuit inside `check_request_timeout_at`.
        check_request_timeout_at(a, now, stall);
    }
    classified
}

fn emit_token_inner(
    a: &mut ActiveSeq,
    tok: u32,
    logprobs: Option<crate::api::TokenLogprobs>,
    honor_eos_inside_thinking: bool,
    loop_watchdog_enabled: bool,
    catastrophic_loop_watchdog_enabled: bool,
) -> bool {
    let output_len_before_emit = a.output_tokens.len();
    let was_inside_thinking = a.inside_thinking;
    let current_logprobs = logprobs.clone();
    // Cooperative cancellation from the streaming pipeline (PR #89). The
    // stream-side loop guards (Bug-2 name-run cap, F11 within-dedup, F44
    // perm-fail, loop-watchdog) flip this flag when they decide the
    // response should end. Treat it like an EOS: finalise now so
    // `handle_done` runs with the proper `tool_loop_capped` /
    // `finish_reason="length"` machinery, instead of letting the model
    // keep emitting tokens that just get suppressed.
    if retire_cancelled_speculation(a, "spec_emit") {
        return false;
    }

    // ── ATLAS_DUMP_HIDDEN: append token-emit record (catch-all spec path) ──
    // verify_k2/k3/k4/dflash all funnel through emit_token, so this hook
    // covers the speculative codepaths that bypass process_decode_logits.
    // Path lookup is cached via OnceLock in `dump_hidden_path` so the
    // common production case (var unset) is a single relaxed pointer load.
    if let Some(path) = super::helpers::dump_hidden_path() {
        const TOKEN_DUMP_MAGIC: u32 = 0xA71B5DEE;
        use std::io::Write;
        if let Ok(mut f) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
        {
            let _ = f.write_all(&TOKEN_DUMP_MAGIC.to_le_bytes());
            let _ = f.write_all(&tok.to_le_bytes());
            let _ = f.write_all(&0u32.to_le_bytes());
            let _ = f.write_all(&0u32.to_le_bytes());
        }
    }

    // ChatML role-boundary HARD stop (`<|im_start|>`).
    //
    // Handled BEFORE grammar advance / EOS suppression: if the model
    // hallucinated a `<|im_start|>` mid-turn, we must end the turn regardless
    // of grammar / require_tool_call / min_tokens. The regular EOS path at
    // line ~3020 honors `suppress_eos`, which is true while a tool-call
    // grammar is active — so if we fell through to it, the tokenizer would
    // strip `<|im_start|>` (special-token) but the following role literal
    // (`user` / `assistant` — regular tokens) would stream to the client,
    // poisoning its context and causing the observed multi-turn drift /
    // "file was corrupted" hallucinations in opencode.
    if commit_chatml_role_boundary(a, tok, im_start_hard_stop(), &current_logprobs) {
        return true;
    }

    // Spontaneous <think>: model generates <think> even when thinking was not
    // requested. Enter thinking mode so EOS is suppressed and thinking content
    // is stripped. This handles MTP bootstrap/verify paths.
    //
    // Spontaneous <think>: model generates <think> even when thinking was not
    // requested. Enter thinking mode so EOS is suppressed and thinking content
    // is stripped. This handles MTP bootstrap/verify paths.
    //
    // DDTree guard: bonus tokens bypass the sample-time logit mask that normally
    // blocks <think> when think_ended=true. When think_ended=true and <think>
    // arrives here, we must still enter thinking mode (the token is already
    // committed to the KV cache by the verify pass) but force an immediate exit
    // via force_end_thinking. The very next token becomes </think> (0 thinking
    // content tokens, 1 </think> overhead), and coding output continues cleanly.
    // deepseek_v41: `<think>` and a stray `</think>` are ordinary output
    // tokens for the Python engine (its router splits on the text).
    let dsv41 = crate::dsv41::serving();
    if !dsv41 && !a.inside_thinking && a.think_start_token == Some(tok) {
        // Re-entering thinking re-arms the response-entry counter for the
        // next `</think>` boundary.
        a.post_think_gate_steps = 0;
        if !a.think_ended {
            a.inside_thinking = true;
            a.think_ended = false;
            a.think_skip_count = 0;
            let spontaneous_base =
                crate::api::chat::effective_spontaneous_think_budget(a.spontaneous_think_budget);
            a.thinking_budget = Some(resolve_rethink_budget(
                spontaneous_base,
                a.think_watchdog_fires,
            ));
            tracing::debug!("Spontaneous <think> detected in emit_token, entering thinking mode");
        } else {
            // DDTree cliff-path <think> bonus with think_ended=true: force-exit.
            a.inside_thinking = true;
            a.think_skip_count = 0;
            a.thinking_budget = Some(0);
            a.force_end_thinking = true;
            tracing::debug!(
                "DDTree <think> bonus (think_ended=true): force-exit thinking (0 content tokens)"
            );
        }
        return true; // don't emit <think> as content
    }

    // Silently skip </think> tokens outside thinking mode (same as process_decode_logits).
    if !dsv41 && !a.inside_thinking && a.think_end_token == Some(tok) {
        a.think_skip_count += 1;
        if a.think_skip_count >= 50 {
            a.finished = true;
            record_watchdog_stop(a, "repeated_think_end_watchdog");
        }
        return true;
    }
    // Match serial decode: a real post-thinking token breaks the aggregate
    // stray-</think> run. Without this reset, speculative serving stopped after
    // 50 non-consecutive close tags accumulated across otherwise valid output.
    if a.think_ended {
        a.think_skip_count = 0;
    }

    // Track <tool_call> token: once seen, legacy tool call requirement is satisfied.
    // Guard with !inside_thinking — tool calls inside thinking are spurious.
    if a.require_tool_call && a.tool_call_start_token == Some(tok) && !a.inside_thinking {
        a.require_tool_call = false;
        a.tool_call_opened = true;
    }
    clear_stale_require_tool_call(a, output_len_before_emit);

    // Preserve the phase on entry for the content watchdog below.  A closing
    // </tool_call> token still belongs to the tool body; flipping the flag
    // first would incorrectly charge that token to the inter-tool prose
    // budget (the plain decoder updates this flag after content handling).
    let was_inside_tool_body = a.inside_tool_body;

    // Track CURRENT tool-body phase (P3.1, 2026-04-25). Set on the
    // open token, clear on the close. The flag drives sampler
    // scoping: when true, the main decode path zeroes
    // repetition/presence/frequency/DRY so legitimate JSON
    // micro-repetition (`":"`, `","`, key names) is not penalised.
    update_tool_body_phase(a, tok);

    // Advance grammar state with the emitted token — skip while the
    // sequence is inside `<think>`…`</think>` so the matcher only
    // sees the final-output token stream.
    advance_content_grammar(a, tok);

    // Accumulate logprobs data for blocking responses.
    if let Some(lp) = logprobs {
        a.logprobs_data.push(lp);
    }

    // Silent exit for zero-content thinking episodes (DDTree cliff or spontaneous
    // <think> with budget=0). Skip pushing </think> to output_tokens — the tag
    // would appear as a spurious delimiter and cause strip_thinking_tags to
    // truncate everything before it, treating the real output as "reasoning".
    if a.inside_thinking && a.think_end_token == Some(tok) && a.thinking_budget == Some(0) {
        a.inside_thinking = false;
        a.force_end_thinking = false;
        super::think_budget_nudge::on_think_closed(a);
        a.think_ended = true;
        a.think_just_ended = true;
        a.post_think_gate_steps = 0;
        discard_uncommitted_logprobs(a, tok, current_logprobs.is_some());
        publish_thinking_done(a);
        tracing::info!("Thinking ended after 0 tokens (budget=Some(0)) [silent]");
        return true;
    }

    // Speculative verify paths funnel through emit_token rather than
    // process_decode_logits, so they must advance the same content counters
    // and run the same degeneration detectors.  Historically they skipped
    // this entire policy layer: a plain decode could stop/rollback at 96
    // repeated tokens while DFlash/MTP emitted until max_tokens, making both
    // serving behaviour and benchmark token counts mechanism-dependent.
    //
    // A speculative verify may have committed several target/SSM positions
    // before reaching this function.  Calling rollback_to_boundary here would
    // therefore be unsafe: the per-token boundary can sit in the middle of
    // that already-committed batch.  Fail closed by ending the response.  A
    // later design can replace this with batched commit truncation plus exact
    // SSM snapshots; silent bypass is not an acceptable fallback.
    let mut speculative_watchdog_stop = false;
    if !a.inside_thinking {
        a.content_started = true;
        a.content_tokens = a.content_tokens.saturating_add(1);

        // Not for deepseek_v41 (the Python engine has only its cycle breaker).
        let allow_content_loop_stop = !a.ignore_eos && a.output_tokens.len() >= a.min_tokens;
        let catastrophic_loop = allow_content_loop_stop
            && catastrophic_loop_watchdog_enabled
            && !crate::dsv41::serving()
            && a.content_tokens >= CATASTROPHIC_LOOP_MIN_TOKENS as u32
            && a.content_tokens.is_multiple_of(CONTENT_LOOP_CHECK_STRIDE)
            && detect_catastrophic_content_loop(&a.output_tokens);
        let configured_loop = allow_content_loop_stop
            && loop_watchdog_enabled
            && a.content_tokens >= CONTENT_LOOP_MIN_TOKENS
            && a.content_tokens.is_multiple_of(CONTENT_LOOP_CHECK_STRIDE)
            && (detect_content_token_loop(&a.output_tokens)
                || numeric_token_mask()
                    .as_deref()
                    .is_some_and(|m| detect_content_token_loop_normalized(&a.output_tokens, m)));
        if a.grammar_state.is_none()
            && !was_inside_tool_body
            && (catastrophic_loop || configured_loop)
        {
            record_watchdog_stop(
                a,
                if catastrophic_loop {
                    "catastrophic_content_loop_watchdog"
                } else {
                    "content_loop_watchdog"
                },
            );
            tracing::warn!(
                content_tokens = a.content_tokens,
                output_len = a.output_tokens.len(),
                catastrophic = catastrophic_loop,
                "Content-loop watchdog fired during speculative emit; ending response early (batched rollback unavailable)"
            );
            speculative_watchdog_stop = true;
        }

        if !was_inside_tool_body && a.grammar_state.is_some() && a.tool_call_start_token.is_some() {
            a.prose_tokens_since_last_tool = a.prose_tokens_since_last_tool.saturating_add(1);
            let max_prose = watchdog_params().max_inter_tool_prose;
            if a.prose_tokens_since_last_tool > max_prose {
                record_watchdog_stop(a, "inter_tool_prose_watchdog");
                tracing::warn!(
                    prose_tokens = a.prose_tokens_since_last_tool,
                    max = max_prose,
                    "Inter-tool prose budget exhausted during speculative emit; ending response (batched rollback unavailable)"
                );
                speculative_watchdog_stop = true;
            }
        }
    }

    // Classify the sampled row below, but latch the terminal verdict before
    // any structural-token return can bypass it (notably suppressed EOS).
    a.finished |= speculative_watchdog_stop;

    // Tool-call close is a structural boundary, not ordinary content. The
    // serial decoder and every speculative verifier share the exact push /
    // stream / finish contract through this helper.
    if a.tool_call_end_token == Some(tok) && !a.inside_thinking {
        // Saturating: release builds have overflow checks off (`panic =
        // "abort"`, no `overflow-checks`), so a plain `-= 1` at zero wraps to
        // `usize::MAX` and the admission gate then reserves the whole KV
        // pool for this one sequence until it retires.
        a.remaining = a.remaining.saturating_sub(1);
        a.think_just_ended = false;
        commit_tool_call_close(a, tok, was_inside_thinking);
        return true;
    }

    let pre_tok_prefix_hash = a.prefix_hash;
    a.output_tokens.push(tok);
    a.update_prefix_hash(tok);

    // Thinking tokens are "free" (don't decrement remaining).
    // Detect </think> transition. Track thinking token count for budget enforcement.
    if a.inside_thinking {
        if a.think_end_token == Some(tok) {
            a.inside_thinking = false;
            a.force_end_thinking = false;
            super::think_budget_nudge::on_think_closed(a);
            a.consecutive_confident = 0;
            a.in_code_fence = false;
            a.think_ended = true;
            a.post_think_gate_steps = 0;
            // One-shot for the next decode step: pin to
            // tool_call_start_token if require_tool_call (Change 3b).
            a.think_just_ended = true;
            tracing::info!(
                "Thinking ended after {} tokens (budget={:?})",
                a.thinking_tokens,
                a.thinking_budget,
            );
        } else {
            a.thinking_tokens += 1;
            let min_runway = super::confidence::min_answer_runway(a.max_output_tokens);
            let runway_exhausted = a.max_output_tokens > 0
                && a.output_tokens.len() >= a.max_output_tokens.saturating_sub(min_runway);
            if !a.force_end_thinking
                && (a.thinking_budget.is_some_and(|b| a.thinking_tokens >= b) || runway_exhausted)
            {
                let effective_budget = a.thinking_budget.unwrap_or(a.thinking_tokens);
                super::think_budget_nudge::arm_budget_close(a, effective_budget, 0);
            }
            if !a.force_end_thinking
                && im_start_hard_stop().is_some_and(|im_start| {
                    detect_chatml_thinking_cycle(&a.output_tokens, None, im_start)
                })
            {
                a.force_end_thinking = true;
                crate::openai::record_shared_intervention(
                    a.intervention_ledger.as_ref(),
                    "chatml_thinking_cycle_watchdog",
                    "force_end_thinking",
                );
                tracing::warn!(
                    thinking_tokens = a.thinking_tokens,
                    "ChatML control-token cycle detected during speculative thinking; forcing </think>"
                );
            }
            super::think_nudge::on_thinking_token(a, tok);
            super::think_budget_nudge::on_thinking_token(a, tok);
        }
    } else {
        a.remaining = a.remaining.saturating_sub(1);
        // Clear think_just_ended one-shot now that we've consumed the
        // token after </think>.
        a.think_just_ended = false;
    }

    // Single-pass over eos_tokens: previously this Vec was linearly
    // scanned twice per emit (once for each branch of the suppress_eos
    // dispatch). On a 256-token response that's 256 extra scans of the
    // ~3-5 entry vec — small per call but mechanical to remove.
    let is_eos = a.eos_tokens.contains(&tok);
    let implicit_think_close = is_eos
        && maybe_implicit_close_thinking_on_eos(
            a,
            tok,
            output_len_before_emit,
            honor_eos_inside_thinking,
        );
    // Required tools remain protected by grammar / legacy-required mode;
    // ordinary post-thinking answers are free to terminate immediately.
    let suppress_eos = eos_is_suppressed(a, output_len_before_emit);
    if implicit_think_close {
        // `tok` was staged above so speculative and serial bookkeeping stay
        // aligned. The implicit boundary is structural, never client text.
        debug_assert_eq!(a.output_tokens.last(), Some(&tok));
        a.output_tokens.pop();
        a.prefix_hash = pre_tok_prefix_hash;
        discard_uncommitted_logprobs(a, tok, current_logprobs.is_some());
        retire_if_budget_exhausted(a, "speculative implicit thinking close");
        return true;
    }
    if is_eos && !suppress_eos {
        publish_hidden_committed_token(a, &current_logprobs, "speculative terminal EOS");
        a.finished = true;
        return true;
    }
    if is_eos {
        // EOS suppressed: grammar not terminated, legacy tool call not yet seen,
        // min_tokens not reached, or thinking is still active.
        // The plain decode path discards this token rather than including it in
        // output history or response logprobs; undo the speculative path's
        // earlier bookkeeping pushes.
        debug_assert_eq!(a.output_tokens.last(), Some(&tok));
        a.output_tokens.pop();
        a.prefix_hash = pre_tok_prefix_hash;
        discard_uncommitted_logprobs(a, tok, current_logprobs.is_some());
        retire_if_budget_exhausted(a, "speculative suppressed EOS");
        return true;
    }
    // Count only committed non-EOS content. Suppressed EOS rows are popped
    // above and therefore cannot age the entry pin accidentally.
    if a.think_ended && !was_inside_thinking && !a.inside_thinking {
        a.post_think_gate_steps = mtp_gate::advance_entry_counter(a.post_think_gate_steps);
    }
    // OPENCODE FIX: see process_decode_logits — same gate. Suppress streaming
    // of spontaneous-thinking content (and its closing `</think>`) so it
    // doesn't pollute opencode's history.
    let suppress_stream = (a.inside_thinking
        || (was_inside_thinking && a.think_end_token == Some(tok)))
        && !a.enable_thinking;
    if matches!(a.sink, ResponseSink::Streaming(_)) {
        let event = committed_token_event(tok, &current_logprobs, suppress_stream);
        // The channel is the burst buffer. A full or closed consumer retires
        // this sequence immediately instead of parking the sole scheduler.
        if !publish_stream_event(a, event, "spec_emit") {
            return true;
        }
    }
    // Grammar termination is itself a sequence boundary. Serial decode stops
    // immediately after the token that completes a stop-after-first grammar;
    // speculative emission must not continue walking already-verified rows.
    if a.grammar_state
        .as_ref()
        .is_some_and(|gs| gs.is_terminated())
    {
        a.finished = true;
        return true;
    }
    // `remaining` deliberately excludes reasoning tokens, but the API's
    // max_tokens/max_completion_tokens envelope includes every generated
    // completion token.  Keep thinking free for the content budget while
    // enforcing the absolute response cap for both blocking and streaming
    // speculative paths.
    if a.output_tokens.len() >= a.max_output_tokens {
        tracing::info!(
            "emit_token: max_output_tokens={} reached, output_tokens={}, thinking_tokens={}",
            a.max_output_tokens,
            a.output_tokens.len(),
            a.thinking_tokens,
        );
        a.finished = true;
        return true;
    }
    if a.remaining == 0 {
        tracing::info!(
            "emit_token: remaining=0, output_tokens={}, thinking_tokens={}",
            a.output_tokens.len(),
            a.thinking_tokens
        );
        a.finished = true;
    }
    if loop_watchdog_enabled && !a.finished {
        apply_speculative_fuzzy_stop(a);
    }
    true
}

/// Remove logprobs staged before a speculative token was classified as a
/// discarded EOS.  Blocking metadata must describe exactly `output_tokens`;
/// the serial path never stages logprobs for suppressed or implicit-close EOS.
fn discard_uncommitted_logprobs(a: &mut ActiveSeq, tok: u32, staged: bool) {
    if !staged {
        return;
    }
    debug_assert_eq!(a.logprobs_data.last().map(|lp| lp.token_id), Some(tok));
    a.logprobs_data.pop();
}

// F72 (byte-level partial-trigger anchor) was removed in F73 / fix42.
// The sampler-side anchor hung the server in production; the broken
// envelope is now recovered at the streaming-sanitizer + parser
// layer. xgrammar's non-anchored TagDispatch limitation is pinned
// for documentation by
// `grammar.rs::tests::test_minimax_xml_grammar_masks_trigger_breaking_multibyte_token`.

/// Compile a grammar state from a grammar specification + engine.
///
/// Returns `Some(GrammarState)` if compilation succeeds, `None` otherwise
/// (logging a warning on failure so the request falls back to legacy tool_call
/// suppression). Called once per request during prefill.
pub fn compile_grammar_state(
    engine: &mut Option<GrammarEngine>,
    grammar_spec: &Option<GrammarSpec>,
) -> Option<GrammarState> {
    let spec = grammar_spec.as_ref()?;
    let engine = engine.as_mut()?;

    // F69 (2026-04-29): symmetric dispatch via the trait. The parser
    // is the single source of truth for both runtime parsing and
    // grammar compilation; no string match keyed on `parser_name`.
    // Mistral's default trait impl returns `None`, which we treat as
    // "no constraint, fall through to unconstrained decoding."
    let compiled = match spec {
        GrammarSpec::ToolCall {
            tools,
            parser,
            use_triggers,
            stop_after_first,
            response_format,
        } => {
            let compiled_opt = match (response_format, *use_triggers) {
                (Some(rf), true) => parser.compile_tool_or_response_format_grammar(
                    engine,
                    tools,
                    *stop_after_first,
                    rf,
                ),
                _ => parser.compile_tool_grammar(engine, tools, *use_triggers, *stop_after_first),
            };
            match compiled_opt {
                Some(result) => result,
                None => {
                    tracing::debug!(
                        "Grammar: parser '{}' opted out of constrained decoding for this request",
                        parser.name(),
                    );
                    return None;
                }
            }
        }
        GrammarSpec::JsonObject => engine.compile_json_grammar(),
        GrammarSpec::JsonSchema { schema, strict } => {
            engine.compile_json_schema_with_strict(schema, *strict)
        }
    };

    let label = match spec {
        GrammarSpec::ToolCall {
            parser,
            tools,
            response_format: Some(crate::api::ResponseFormatConstraint::JsonObject),
            use_triggers: true,
            ..
        } => format!(
            "parser={}, tools={}, response_format=json_object (joint union)",
            parser.name(),
            tools.len()
        ),
        GrammarSpec::ToolCall {
            parser,
            tools,
            response_format: Some(crate::api::ResponseFormatConstraint::JsonSchema { .. }),
            use_triggers: true,
            ..
        } => format!(
            "parser={}, tools={}, response_format=json_schema (joint union)",
            parser.name(),
            tools.len()
        ),
        GrammarSpec::ToolCall { parser, tools, .. } => {
            format!("parser={}, tools={}", parser.name(), tools.len())
        }
        GrammarSpec::JsonObject => "response_format=json_object".to_string(),
        GrammarSpec::JsonSchema { .. } => "response_format=json_schema".to_string(),
    };

    match compiled {
        Ok(grammar) => {
            let vocab_size = engine.vocab_size();
            match GrammarState::new(&grammar, vocab_size) {
                Ok(state) => {
                    tracing::info!("Grammar constrained decoding active: {label}");
                    Some(state)
                }
                Err(e) => {
                    tracing::warn!("Grammar state creation failed: {e}");
                    None
                }
            }
        }
        Err(e) => {
            tracing::warn!("Grammar compilation failed: {e}");
            None
        }
    }
}

/// Result of starting a chunked prefill.
pub enum StartPrefillResult {
    /// Prompt fit in one chunk → ready for decode.
    Active(ActiveSeq),
    /// Prompt needs more chunks → add to prefilling[].
    InProgress(PrefillInProgress),
    /// Completed during first chunk (EOS on first token).
    Finished,
}

#[cfg(test)]
#[path = "emit_step_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "emit_watchdog_boundary_tests.rs"]
mod watchdog_boundary_tests;

#[cfg(test)]
mod timeout_branch_tests {
    use super::*;
    use crate::api::request_contract::{TIMEOUT_STALL_CAUSE, TIMEOUT_TOTAL_CAUSE};
    use crate::scheduler::test_fixtures::active_seq;
    use std::time::Duration;

    #[test]
    fn speculative_suppressed_eos_loop_trips_stall_timeout() {
        let mut a = active_seq(ResponseSink::Blocking(None), 0);
        a.require_tool_call = true;
        let eos = a.eos_tokens[0];
        let t0 = Instant::now();
        let stall = Some(Duration::from_secs(10));

        emit_token_with_policies_at(&mut a, eos, None, false, false, false, t0, stall);
        assert!(!a.finished, "first suppressed EOS arms the stall clock");
        assert_eq!(a.stall_mark.tokens, 1);

        emit_token_with_policies_at(
            &mut a,
            eos,
            None,
            false,
            false,
            false,
            t0 + Duration::from_secs(9),
            stall,
        );
        assert!(!a.finished, "below stall threshold");

        emit_token_with_policies_at(
            &mut a,
            eos,
            None,
            false,
            false,
            false,
            t0 + Duration::from_secs(10),
            stall,
        );
        assert!(a.finished, "suppressed-EOS loop must trip stall timeout");
        assert_eq!(a.stop_cause, Some(TIMEOUT_STALL_CAUSE));
        assert!(!a.output_tokens.contains(&eos));
    }

    #[test]
    fn speculative_structural_and_suppressed_eos_branches_enforce_total_timeout() {
        let past = Instant::now()
            .checked_sub(Duration::from_secs(1))
            .unwrap_or_else(Instant::now);

        // 1. Suppressed EOS.
        let mut suppressed = active_seq(ResponseSink::Blocking(None), 0);
        suppressed.require_tool_call = true;
        suppressed.timeout_at = Some(past);
        emit_token(&mut suppressed, 17, None);
        assert!(suppressed.finished);
        assert_eq!(suppressed.stop_cause, Some(TIMEOUT_TOTAL_CAUSE));

        // 2. Spontaneous <think>.
        let mut spontaneous = active_seq(ResponseSink::Blocking(None), 0);
        spontaneous.think_start_token = Some(28);
        spontaneous.think_ended = false;
        spontaneous.timeout_at = Some(past);
        emit_token(&mut spontaneous, 28, None);
        assert!(spontaneous.finished);
        assert_eq!(spontaneous.stop_cause, Some(TIMEOUT_TOTAL_CAUSE));

        // 3. Stray </think> outside thinking.
        let mut stray = active_seq(ResponseSink::Blocking(None), 0);
        stray.think_end_token = Some(29);
        stray.timeout_at = Some(past);
        emit_token(&mut stray, 29, None);
        assert!(stray.finished);
        assert_eq!(stray.stop_cause, Some(TIMEOUT_TOTAL_CAUSE));

        // 4. Zero-budget silent </think> exit.
        let mut silent_close = active_seq(ResponseSink::Blocking(None), 0);
        silent_close.inside_thinking = true;
        silent_close.think_end_token = Some(29);
        silent_close.thinking_budget = Some(0);
        silent_close.timeout_at = Some(past);
        emit_token(&mut silent_close, 29, None);
        assert!(silent_close.finished);
        assert_eq!(silent_close.stop_cause, Some(TIMEOUT_TOTAL_CAUSE));

        // 5. Implicit thinking close on EOS.
        let mut implicit = active_seq(ResponseSink::Blocking(None), 0);
        implicit.inside_thinking = true;
        implicit.think_ended = false;
        implicit.timeout_at = Some(past);
        emit_token_with_eos_policy(&mut implicit, 17, None, true);
        assert!(implicit.finished);
        assert_eq!(implicit.stop_cause, Some(TIMEOUT_TOTAL_CAUSE));
    }

    #[test]
    fn speculative_legitimate_completion_is_not_overwritten_by_timeout() {
        let past = Instant::now()
            .checked_sub(Duration::from_secs(1))
            .unwrap_or_else(Instant::now);

        // Normal EOS.
        let mut eos_seq = active_seq(ResponseSink::Blocking(None), 0);
        eos_seq.timeout_at = Some(past);
        emit_token(&mut eos_seq, 17, None);
        assert!(eos_seq.finished);
        assert_eq!(eos_seq.stop_cause, None);

        // Normal token hitting max_output_tokens.
        let mut max_seq = active_seq(ResponseSink::Blocking(None), 0);
        max_seq.max_output_tokens = 2;
        max_seq.timeout_at = Some(past);
        emit_token(&mut max_seq, 5, None);
        assert!(max_seq.finished);
        assert_eq!(max_seq.stop_cause, None);
    }

    #[test]
    fn spontaneous_think_end_is_suppressed_on_sse_when_enable_thinking_false() {
        let (tx, mut rx) = tokio::sync::mpsc::channel(8);
        let mut a = active_seq(ResponseSink::Streaming(tx), 0);
        a.enable_thinking = false;
        a.inside_thinking = false;
        a.think_ended = false;
        a.think_start_token = Some(28);
        a.think_end_token = Some(29);

        // Spontaneous `<think>` enters thinking without emitting to SSE.
        emit_token(&mut a, 28, None);
        assert!(a.inside_thinking);
        assert!(matches!(
            rx.try_recv(),
            Err(tokio::sync::mpsc::error::TryRecvError::Empty)
        ));

        // Thinking content stays hidden (`CompletionProgress`).
        emit_token(&mut a, 40, None);
        assert!(matches!(rx.try_recv(), Ok(StreamEvent::CompletionProgress)));

        // Closing `</think>` clears `inside_thinking` AND must stay hidden on SSE.
        emit_token(&mut a, 29, None);
        assert!(!a.inside_thinking && a.think_ended);
        assert!(
            matches!(rx.try_recv(), Ok(StreamEvent::CompletionProgress)),
            "spontaneous </think> must not leak as StreamEvent::Token when enable_thinking=false"
        );

        // First real answer token streams normally.
        emit_token(&mut a, 41, None);
        assert!(matches!(rx.try_recv(), Ok(StreamEvent::Token(41))));

        // Control: when `enable_thinking=true`, closing `</think>` still streams.
        let (tx_on, mut rx_on) = tokio::sync::mpsc::channel(4);
        let mut enabled = active_seq(ResponseSink::Streaming(tx_on), 0);
        enabled.enable_thinking = true;
        enabled.inside_thinking = true;
        enabled.think_ended = false;
        enabled.think_end_token = Some(29);
        emit_token(&mut enabled, 29, None);
        assert!(matches!(rx_on.try_recv(), Ok(StreamEvent::Token(29))));
    }

    #[test]
    fn speculative_require_tool_call_safety_valve_clears_after_512_tokens() {
        let mut a = active_seq(ResponseSink::Blocking(None), 0);
        let eos = a.eos_tokens[0];
        a.require_tool_call = true;
        a.inside_thinking = false;
        a.max_output_tokens = 1_000;
        a.remaining = 1_000;
        a.output_tokens = vec![5; 512];

        // At exactly 512 committed tokens, EOS stays suppressed.
        emit_token(&mut a, eos, None);
        assert!(a.require_tool_call);
        assert!(!a.finished);
        assert_eq!(a.output_tokens.len(), 512);

        // Past 512 committed tokens, the safety valve clears `require_tool_call`
        // so EOS terminates instead of spinning forever.
        a.output_tokens.push(5);
        emit_token(&mut a, eos, None);
        assert!(!a.require_tool_call);
        assert!(a.finished);
        assert_eq!(a.output_tokens.last(), Some(&eos));
    }
}
