// SPDX-License-Identifier: AGPL-3.0-only

//! Sequence lifecycle: finish, errors, swap-out, resume.

use super::*;

#[allow(clippy::too_many_arguments)]
fn completion_done_event(
    output_tokens: &mut Vec<u32>,
    finish_reason: &str,
    time_to_first_token_ms: f64,
    decode_time_ms: f64,
    reasoning_tokens: u32,
    cached_prompt_tokens: u32,
    think_nudges: u32,
    stop_cause: Option<&'static str>,
) -> StreamEvent {
    let completion_tokens = output_tokens.len();
    let completion_token_ids = std::mem::take(output_tokens);
    StreamEvent::Done {
        finish_reason: finish_reason.to_string(),
        prompt_tokens: 0,
        completion_token_ids,
        completion_tokens,
        time_to_first_token_ms,
        decode_time_ms,
        reasoning_tokens,
        cached_prompt_tokens,
        think_nudges,
        stop_cause,
    }
}

/// `"length"` means only that the token budget or context ran out: a stop
/// made by a server guard (`stop_cause`) reports `"stop"`.
fn finish_reason(
    output_tokens: &[u32],
    eos_tokens: &[u32],
    tool_call_end_token: Option<u32>,
    ignore_eos: bool,
    stop_cause: Option<&str>,
) -> &'static str {
    let last_tok = output_tokens.last().copied();
    let is_eos = last_tok.is_some_and(|t| eos_tokens.contains(&t));
    let is_chatml_role_boundary =
        !ignore_eos && last_tok.is_some_and(|token| im_start_hard_stop() == Some(token));
    let is_tool_call_end = last_tok.is_some_and(|token| tool_call_end_token == Some(token));
    if is_eos || is_chatml_role_boundary {
        "stop"
    } else if is_tool_call_end {
        "tool_calls"
    } else if stop_cause.is_some_and(|c| !crate::api::request_contract::is_timeout_cause(c)) {
        "stop"
    } else {
        "length"
    }
}

struct CompletionOutcome<'a> {
    reason: &'a str,
    stop_cause: Option<&'a str>,
    finished: bool,
    cancelled: bool,
    delivered: bool,
}

/// True when a sequence reached a normal terminal completion (`"stop"`,
/// `"tool_calls"`, or non-timeout `"length"`) and delivered its response,
/// excluding client cancellations/disconnects, shutdown retirement, timeouts,
/// KV evictions (`!finished`), and error causes.
fn is_successful_completion(outcome: CompletionOutcome<'_>) -> bool {
    if !outcome.finished || outcome.cancelled || !outcome.delivered {
        return false;
    }
    if !matches!(outcome.reason, "stop" | "tool_calls" | "length") {
        return false;
    }
    if outcome.stop_cause.is_some_and(|cause| {
        crate::api::request_contract::is_timeout_cause(cause)
            || matches!(
                cause,
                "cancelled" | "aborted" | "error" | "timeout" | "kv_exhausted"
            )
    }) {
        return false;
    }
    true
}

/// Recover the visible reasoning/content split when the scheduler closed
/// thinking without committing a literal delimiter. Content bookkeeping is
/// rollback-aware and counts every committed post-boundary token, including a
/// terminal EOS, so subtracting it from the visible sequence is exact even
/// when one or more in-thinking EOS samples were discarded.
fn hidden_thinking_output_boundary(a: &ActiveSeq) -> Option<usize> {
    if !a.enable_thinking {
        return None;
    }
    if !a.think_ended {
        // Sequence finished (e.g. hit max_tokens) while still inside `<think>`
        // without closing: every committed token is reasoning, matching the
        // streaming path where pre-close tokens go to `reasoning_content`.
        return (a.inside_thinking && !a.output_tokens.is_empty()).then_some(a.output_tokens.len());
    }
    if a.think_end_token
        .is_some_and(|end| a.output_tokens.contains(&end))
    {
        return None;
    }
    Some(
        a.output_tokens
            .len()
            .saturating_sub(a.content_tokens as usize),
    )
}

fn emit_qwen4_target_state_receipt(model: &dyn Model, a: &ActiveSeq) {
    match model.qwen4_target_state_receipt(&a.seq) {
        Ok(None) => {}
        Ok(Some(receipt)) => tracing::info!(
            "QWEN4_NATIVE_MTP_REQUEST_RECEIPT schema={} mode={} scheduler_path={} proposer_loaded={} session_hash={:016x} seq_len={} token_count={} output_tokens={} gdn_layers={} attention_layers={} logical_kv_blocks={} kv_token_positions={} compared_buffers={} compared_bytes={} state_scope=live-gdn-ple-plus-logical-bf16-kv checkpoints_excluded=true fed_tokens_sha256={} target_state_sha256={}",
            receipt.schema,
            receipt.mode,
            receipt.scheduler_path,
            receipt.proposer_loaded,
            a.session_hash,
            receipt.seq_len,
            receipt.token_count,
            a.output_tokens.len(),
            receipt.gdn_layers,
            receipt.attention_layers,
            receipt.logical_kv_blocks,
            receipt.kv_token_positions,
            receipt.compared_buffers,
            receipt.compared_bytes,
            receipt.fed_tokens_sha256,
            receipt.target_state_sha256,
        ),
        Err(error) => tracing::error!("QWEN4_NATIVE_MTP_REQUEST_RECEIPT_FAILED detail={error:#}"),
    }
}

/// Send final response and free GPU resources for a completed sequence.
pub fn finish_sequence(model: &dyn Model, a: &mut ActiveSeq) {
    finish_sequence_with_cache(model, a, true);
}

/// Finish a sequence while explicitly controlling final prefix-cache admission.
///
/// Speculative verification may stop while only a prefix of an over-planned
/// verify frame has been emitted. Until that frame has an exact recurrent/KV
/// commit boundary, its `seq.tokens` must never become a reusable prefix.
pub(super) fn finish_sequence_with_cache(
    model: &dyn Model,
    a: &mut ActiveSeq,
    cache_sequence: bool,
) {
    // Flush any deferred streaming events before terminal `StreamEvent::Done`
    // so tokens accepted on a terminal verify step are never lost or reordered.
    flush_deferred_emit(a);
    // SPEC_CYCLE_V2 is C=1-only and keeps an allocation-free thread-local
    // identity across cycles.  Retire it at every request boundary, including
    // client cancellation after a fully recorded cycle; otherwise a later
    // request can inherit the old id when it reaches the stale expected seq_len.
    spec_timing::retire_request();
    super::adaptive_depth::retire_seq(a.depth_controller_key());
    super::verify_dflash_step::prune_branch_audit_stash(a.session_hash);
    proposal_lifecycle::clear(
        &mut a.pending_drafts,
        &mut a.pending_tree_payload,
        &mut a.pending_dflash_sampled_proposal,
    );
    let reason = finish_reason(
        &a.output_tokens,
        &a.eos_tokens,
        a.tool_call_end_token,
        a.ignore_eos,
        a.stop_cause,
    );
    let thinking_output_boundary = hidden_thinking_output_boundary(a);
    let completion_tokens = a.output_tokens.len();
    // Qualification-only: capture before streaming completion takes the token
    // vector and before prefix-cache admission/freeing releases physical KV
    // blocks. The model method is a zero-cost None unless its exact receipt
    // environment is armed.
    emit_qwen4_target_state_receipt(model, a);
    super::decode_logits_seq::finish_logits_fnv_request();
    super::lookup_receipt::emit(a, "finish");
    let think_nudges = think_nudge::completed(a);
    let cancelled = request_cancelled(a);
    let delivered = match &mut a.sink {
        ResponseSink::Streaming(tx) => {
            let ttft_ms = a.decode_start.duration_since(a.request_start).as_secs_f64() * 1000.0;
            let decode_ms = a.decode_start.elapsed().as_secs_f64() * 1000.0;
            let done = completion_done_event(
                &mut a.output_tokens,
                reason,
                ttft_ms,
                decode_ms,
                a.thinking_tokens,
                a.cached_prompt_tokens,
                think_nudges,
                a.stop_cause,
            );
            // Bounded: this runs right after a stall retirement, when the
            // channel is still full; an unbounded send here would re-park
            // the scheduler on the very consumer we just gave up on.
            if !stream_send_bounded(tx, done, "finish_sequence") {
                tracing::warn!(
                    "finish_sequence: streaming Done not delivered (receiver gone or stalled)"
                );
                false
            } else {
                true
            }
        }
        ResponseSink::Blocking(tx) => {
            if let Some(tx) = tx.take() {
                let ttft_ms = a.decode_start.duration_since(a.request_start).as_secs_f64() * 1000.0;
                let decode_ms = a.decode_start.elapsed().as_secs_f64() * 1000.0;
                if tx
                    .send(Ok(InferenceResponse {
                        output_tokens: a.output_tokens.clone(),
                        thinking_output_boundary,
                        finish_reason: reason.to_string(),
                        time_to_first_token_ms: ttft_ms,
                        decode_time_ms: decode_ms,
                        logprobs: std::mem::take(&mut a.logprobs_data),
                        reasoning_tokens: a.thinking_tokens,
                        cached_prompt_tokens: a.cached_prompt_tokens,
                        think_nudges,
                        stop_cause: a.stop_cause,
                    }))
                    .is_err()
                {
                    tracing::warn!(
                        "finish_sequence: blocking response send failed (receiver dropped)"
                    );
                    false
                } else {
                    true
                }
            } else {
                false
            }
        }
    };
    let decode_s = a.decode_start.elapsed().as_secs_f64();
    let n = completion_tokens;
    let tps = if decode_s > 0.0 {
        n as f64 / decode_s
    } else {
        0.0
    };
    let ttft_ms = a.decode_start.duration_since(a.request_start).as_secs_f64() * 1000.0;
    tracing::info!("Done: {n} tokens ({reason}) {tps:.1} tok/s, TTFT={ttft_ms:.1}ms");
    // ATLAS_FULL_PROFILE=1: dump per-kernel timing report after every
    // generation. Zero overhead when env var is unset.
    if spark_model::full_profile::is_enabled() {
        spark_model::full_profile::dump();
    }
    // Cache the full sequence (prompt + generated) in the prefix cache.
    // Must happen BEFORE free_sequence() so block indices are still valid.
    // Enables multi-turn sessions to reuse KV cache for prior assistant responses.
    // Delegated through ContinuationRecordMut::finish_or_release:
    // if cache_sequence { model.cache_sequence(&a.seq); }
    // model.free_sequence(&mut a.seq)
    if let Err(e) = a
        .continuation_record_mut()
        .finish_or_release(model, cache_sequence)
    {
        tracing::error!("finish_sequence finish_or_release: {e:#}");
    }
    if is_successful_completion(CompletionOutcome {
        reason,
        stop_cause: a.stop_cause,
        finished: a.finished,
        cancelled,
        delivered,
    }) {
        crate::metrics::compat_metrics::inc_request_success();
    }
    crate::metrics::compat_metrics::set_kv_blocks_free(model.num_free_blocks());
    // EP: signal worker to free+realloc its mirrored sequence.
    if let Err(e) = model.ep_broadcast_cmd(0xFFFFFFF1) {
        tracing::error!("EP broadcast free+realloc: {e:#}");
    }
}

/// Send error to client and free GPU resources.
pub fn send_error(model: &dyn Model, a: &mut ActiveSeq, msg: &str) {
    flush_deferred_emit(a);
    spec_timing::retire_request();
    super::adaptive_depth::retire_seq(a.depth_controller_key());
    super::verify_dflash_step::prune_branch_audit_stash(a.session_hash);
    super::lookup_receipt::emit(a, "error");
    proposal_lifecycle::clear(
        &mut a.pending_drafts,
        &mut a.pending_tree_payload,
        &mut a.pending_dflash_sampled_proposal,
    );
    match &mut a.sink {
        ResponseSink::Streaming(tx) => {
            if !stream_send_bounded(tx, StreamEvent::Error(msg.to_string()), "send_error") {
                tracing::warn!(
                    "send_error: streaming Error not delivered (receiver gone or stalled)"
                );
            }
        }
        ResponseSink::Blocking(tx) => {
            if let Some(tx) = tx.take()
                && tx.send(Err(anyhow::anyhow!("{msg}"))).is_err()
            {
                tracing::warn!("send_error: blocking Error send failed (receiver dropped)");
            }
        }
    }
    if let Err(e) = a.continuation_record_mut().finish_or_release(model, false) {
        tracing::error!("send_error: finish_or_release: {e:#}");
    }
    crate::metrics::compat_metrics::set_kv_blocks_free(model.num_free_blocks());
    if let Err(e) = model.ep_broadcast_cmd(0xFFFFFFF1) {
        tracing::error!("send_error: ep_broadcast free+realloc: {e:#}");
    }
}

/// Send an error directly to a ResponseSink that hasn't been attached
/// to an ActiveSeq yet. Used by prefill_request when it fails AFTER
/// extracting the sink from the InferenceRequest but BEFORE building
/// an ActiveSeq. Without this the sender is silently dropped, producing
/// a misleading "Inference cancelled" error on the client side.
pub fn send_error_to_sink(sink: &mut ResponseSink, msg: &str) {
    match sink {
        ResponseSink::Streaming(tx) => {
            if !stream_send_bounded(
                tx,
                StreamEvent::Error(msg.to_string()),
                "send_error_to_sink",
            ) {
                tracing::warn!(
                    "send_error_to_sink: streaming Error not delivered (receiver gone or stalled)"
                );
            }
        }
        ResponseSink::Blocking(tx) => {
            if let Some(tx) = tx.take()
                && tx.send(Err(anyhow::anyhow!("{msg}"))).is_err()
            {
                tracing::warn!("send_error_to_sink: blocking Error send failed (receiver dropped)");
            }
        }
    }
}

#[path = "swap_lifecycle.rs"]
mod swap_lifecycle;
pub(super) use swap_lifecycle::{resume_swapped_seq, swap_out_sequence};

#[cfg(test)]
#[path = "lifecycle_tests.rs"]
mod tests;

#[cfg(test)]
mod request_success_metrics_tests {
    use super::{CompletionOutcome, is_successful_completion};
    use crate::api::request_contract::{TIMEOUT_STALL_CAUSE, TIMEOUT_TOTAL_CAUSE};

    fn outcome<'a>(reason: &'a str, stop_cause: Option<&'a str>) -> CompletionOutcome<'a> {
        CompletionOutcome {
            reason,
            stop_cause,
            finished: true,
            cancelled: false,
            delivered: true,
        }
    }

    #[test]
    fn request_success_counts_only_normal_delivered_completions() {
        assert!(is_successful_completion(outcome("stop", None)));
        assert!(is_successful_completion(outcome("tool_calls", None)));
        assert!(is_successful_completion(outcome("length", None)));
        assert!(is_successful_completion(outcome("stop", Some("watchdog"))));

        // Cancellations, disconnects/undelivered, and unfinished KV evictions do not count.
        assert!(!is_successful_completion(CompletionOutcome {
            cancelled: true,
            ..outcome("stop", None)
        }));
        assert!(!is_successful_completion(CompletionOutcome {
            cancelled: true,
            ..outcome("length", None)
        }));
        assert!(!is_successful_completion(CompletionOutcome {
            delivered: false,
            ..outcome("stop", None)
        }));
        assert!(!is_successful_completion(CompletionOutcome {
            finished: false,
            ..outcome("length", None)
        }));

        // Timeouts and non-normal causes/reasons do not count.
        assert!(!is_successful_completion(outcome(
            "length",
            Some(TIMEOUT_TOTAL_CAUSE)
        )));
        assert!(!is_successful_completion(outcome(
            "length",
            Some(TIMEOUT_STALL_CAUSE)
        )));
        for abnormal in ["cancelled", "aborted", "error", "timeout", "kv_exhausted"] {
            assert!(
                !is_successful_completion(outcome(abnormal, None)),
                "finish_reason={abnormal} must not count as request_success"
            );
            assert!(
                !is_successful_completion(outcome("length", Some(abnormal))),
                "stop_cause={abnormal} must not count as request_success"
            );
        }
    }
}
