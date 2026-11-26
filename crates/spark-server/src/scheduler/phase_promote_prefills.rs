// SPDX-License-Identifier: AGPL-3.0-only

//! Promote completed (fully prefilled) prefills into the active queue,
//! or surface error to the sink + free the sequence. Extracted from
//! `phase_continue_prefills` for the ≤500 LoC cap.

use spark_model::traits::Model;
use std::time::Instant;

use super::*;

#[allow(clippy::too_many_arguments)]
pub(super) fn promote_completed_prefills(
    model: &dyn Model,
    prefilling: &mut Vec<PrefillInProgress>,
    mut completed_indices: Vec<(usize, Option<u32>)>,
    active: &mut Vec<ActiveSeq>,
    prefill_stream: u64,
    think_end_token: Option<u32>,
    think_start_token: Option<u32>,
    code_fence_token: Option<u32>,
    tool_call_start_token: Option<u32>,
    tool_call_end_token: Option<u32>,
) {
    // Process in reverse order so swap_remove indices stay valid, then
    // restore ascending order so promoted sequences enter `active` FIFO.
    completed_indices.sort_unstable_by_key(|x| std::cmp::Reverse(x.0));
    let mut removed = Vec::with_capacity(completed_indices.len());
    for (idx, maybe_token) in completed_indices {
        removed.push((prefilling.swap_remove(idx), maybe_token));
    }
    removed.reverse();
    let defer_first_stream_event = removed.len() > 1;
    let active_start = active.len();
    for (mut p, maybe_token) in removed {
        if stop_allocated_prefill(
            model,
            prefill_stream,
            &mut p.seq,
            &mut p.reset_proof,
            &mut p.sink,
            p.cancel_flag.as_ref(),
            p.timeout_at,
            "prefill promotion",
        ) {
            continue;
        }
        let Some(first) = maybe_token else {
            // Error path: tell the client, then free the sequence. Every
            // producer of `(idx, None)` (chunk/mixed/batched prefill failure,
            // NULL logits, first-token sampling failure, two-phase failure)
            // only logged the error; the sink was dropped silently and the
            // API layer turned the closed channel into a misleading
            // "Inference cancelled" / dropped stream.
            send_error_to_sink(
                &mut p.sink,
                "prefill failed after the first chunk (see server log for the model error)",
            );
            if let Err(e) = super::continuation::ContinuationRecord::abort_prefill_sequence(
                model,
                &mut p.seq,
                &mut p.reset_proof,
            ) {
                tracing::error!("phase_promote_prefills: free_sequence (error path): {e:#}");
            }
            if let Err(e) = model.ep_broadcast_cmd(0xFFFFFFF1) {
                tracing::error!(
                    "phase_promote_prefills: ep_broadcast free+realloc (error path): {e:#}"
                );
            }
            continue;
        };
        // deepseek_v41: a first <think> is an ordinary token for the Python engine.
        let spontaneous_think =
            !p.enable_thinking && think_start_token == Some(first) && !crate::dsv41::serving();
        let use_legacy_tool_call =
            p.require_tool_call && p.grammar_state.is_none() && tool_call_start_token.is_some();
        let eos_suppressed_outside_thinking = eos_is_suppressed_by_non_thinking_guards(
            p.grammar_state.as_ref(),
            use_legacy_tool_call,
            0,
            p.min_tokens,
        );
        let initial_thinking = initial_thinking_state(
            p.enable_thinking,
            spontaneous_think,
            first,
            think_end_token,
            &p.eos_tokens,
            p.thinking_budget,
            crate::api::chat::effective_spontaneous_think_budget(p.spontaneous_think_budget),
            code_fence_token,
            p.max_tokens,
            honor_eos_inside_thinking(),
            eos_suppressed_outside_thinking,
        );
        let first_is_visible = initial_thinking.first_is_visible(spontaneous_think);
        let grammar_terminated = advance_prefill_first_token_grammar(
            &mut p.grammar_state,
            first,
            first_is_visible,
            initial_thinking.inside_thinking || initial_thinking.closed_on_first,
        );
        let immediate_finish = crate::api::disagg::has_pending_prefill_export(p.session_hash)
            || (first_is_visible
                && (p.eos_tokens.contains(&first) || p.max_tokens <= 1 || grammar_terminated));
        let send_first_now = immediate_finish
            || !defer_first_stream_event
            || initial_thinking.publishes_hidden_boundary();
        // Visible first tokens enter the text decoder. A terminal EOS instead
        // becomes hidden progress so usage/logprobs stay cardinality-exact
        // without exposing the stop token as content.
        if send_first_now
            && first_is_visible
            && let ResponseSink::Streaming(ref tx) = p.sink
            && !stream_send_bounded(
                tx,
                first_token_event(first, &p.first_logprobs, p.eos_tokens.contains(&first)),
                "phase_promote_prefills first token",
            )
        {
            // The first event could not enter an otherwise dedicated channel:
            // the peer is gone or deliberately supplied a full sink. Do not
            // retain its GPU slot or run the deferred prefill tail.
            if let Err(error) = super::continuation::ContinuationRecord::abort_prefill_sequence(
                model,
                &mut p.seq,
                &mut p.reset_proof,
            ) {
                tracing::error!(
                    "phase_promote_prefills: free_sequence after first-token rejection: {error:#}"
                );
            }
            if let Err(error) = model.ep_broadcast_cmd(0xFFFFFFF1) {
                tracing::error!(
                    "phase_promote_prefills: EP reset after first-token rejection: {error:#}"
                );
            }
            continue;
        }
        let now = Instant::now();
        let cached_prompt_tok = p.seq.cached_prefix_tokens as u32;
        // K8 item A: the MTP last-K replay deferred past the first token. The first
        // token was sampled (phase_continue_prefills) with these same arguments.
        if !immediate_finish {
            let params = p.sampling_params();
            if let Err(e) = finish_deferred_prefill_tail(
                model,
                &mut p.seq,
                first,
                &params,
                &p.eos_tokens,
                p.top_logprobs,
            ) {
                let msg = format!("deferred prefill tail failed: {e:#}");
                send_error_to_sink(&mut p.sink, &msg);
                if let Err(free_err) =
                    super::continuation::ContinuationRecord::abort_prefill_sequence(
                        model,
                        &mut p.seq,
                        &mut p.reset_proof,
                    )
                {
                    tracing::error!(
                        "phase_promote_prefills: free_sequence (after deferred tail error): {free_err:#}"
                    );
                }
                if let Err(bcast_err) = model.ep_broadcast_cmd(0xFFFFFFF1) {
                    tracing::error!(
                        "phase_promote_prefills: ep_broadcast (after deferred tail error): {bcast_err:#}"
                    );
                }
                continue;
            }
        }

        let mut a = build_active_seq_from_prefill(
            p,
            first,
            spontaneous_think,
            use_legacy_tool_call,
            cached_prompt_tok,
            immediate_finish,
            initial_thinking,
            now,
            think_end_token,
            think_start_token,
            tool_call_start_token,
            tool_call_end_token,
            model.decode_rollback_ring_slots(),
        );
        super::logits_row_receipt::prefill_output(&a);
        if initial_thinking.publishes_hidden_boundary() {
            publish_thinking_done(&mut a);
        }
        if immediate_finish || a.finished {
            crate::api::disagg::maybe_fulfill_prefill_export(
                model,
                a.session_hash,
                &a.seq,
                a.last_token,
                a.cached_prompt_tokens,
                a.request_start,
            );
            finish_sequence(model, &mut a);
        } else {
            if a.think_just_ended {
                rollback::snapshot_boundary_if_ssm(&mut a, model);
            }
            active.push(a);
        }
    }
    if defer_first_stream_event && active.len() > active_start {
        let cohort_start = Instant::now();
        let mut idx = active_start;
        while idx < active.len() {
            let a = &mut active[idx];
            a.decode_start = cohort_start;
            a.last_token_time = cohort_start;
            let deferred = !a.output_tokens.is_empty();
            let send_ok = if deferred && let ResponseSink::Streaming(ref tx) = a.sink {
                let first = a.last_token;
                let first_lp = a.logprobs_data.first().cloned();
                let is_eos = a.eos_tokens.contains(&first);
                stream_send_bounded(
                    tx,
                    first_token_event(first, &first_lp, is_eos),
                    "phase_promote_prefills cohort first token",
                )
            } else {
                true
            };
            if !send_ok {
                let mut rejected = active.remove(idx);
                if let Err(error) = rejected
                    .continuation_record_mut()
                    .finish_or_release(model, false)
                {
                    tracing::error!(
                        "phase_promote_prefills: free_sequence after cohort first-token rejection: {error:#}"
                    );
                }
                if let Err(error) = model.ep_broadcast_cmd(0xFFFFFFF1) {
                    tracing::error!(
                        "phase_promote_prefills: EP reset after cohort first-token rejection: {error:#}"
                    );
                }
            } else {
                idx += 1;
            }
        }
    }
    crate::metrics::compat_metrics::set_kv_blocks_free(model.num_free_blocks());
}

/// Construct an `ActiveSeq` from a finished prefill plus its first sampled
/// token. The two original branches (immediate finish + continue) used
/// near-identical field initialisation; this helper reuses the same record
/// and the caller decides whether to push onto active or finish_sequence.
#[allow(clippy::too_many_arguments)]
fn build_active_seq_from_prefill(
    p: PrefillInProgress,
    first: u32,
    spontaneous_think: bool,
    use_legacy_tool_call: bool,
    cached_prompt_tok: u32,
    immediate_finish: bool,
    initial_thinking: InitialThinkingState,
    now: Instant,
    think_end_token: Option<u32>,
    think_start_token: Option<u32>,
    tool_call_start_token: Option<u32>,
    tool_call_end_token: Option<u32>,
    // Phase-C decode-rollback ring capacity (`model.decode_rollback_ring_slots()`).
    ssm_ring_capacity: usize,
) -> ActiveSeq {
    let temperature = p.temperature;
    // Pre-allocate output_tokens to the final response length so we
    // avoid the ~10 Vec growth-realloc cycles (1 → 2 → 4 → ... → 1024)
    // that grow() would do over a long response. Saves ~10 mallocs +
    // copies per request. Capacity guess uses max_tokens directly; under
    // mimalloc the slop from over-allocation is reclaimed quickly.
    let output_capacity = p.max_tokens.max(1);
    let mut output_tokens = Vec::with_capacity(output_capacity);
    let prefix_hash = if immediate_finish || initial_thinking.first_is_visible(spontaneous_think) {
        output_tokens.push(first);
        crate::metrics::compat_metrics::add_generation_tokens(1);
        super::continuation::hash_token_incremental(p.prefix_hash, first)
    } else {
        p.prefix_hash
    };
    let slot_idx = p.seq.slot_idx;
    let prompt_prefix_hash = p.prompt_prefix_hash;
    let reset_proof = p.reset_proof.or_else(|| {
        super::continuation::is_reset_proof_enabled()
            .then(|| super::continuation::ResetProofPayload::new(slot_idx, prompt_prefix_hash))
    });
    ActiveSeq {
        stop_cause: None,
        seq: p.seq,
        session_hash: p.session_hash,
        prompt_prefix_hash,
        prefix_hash,
        record_state: super::continuation::RecordState::Live,
        reset_proof,
        last_token: first,
        output_tokens,
        max_output_tokens: p.max_tokens,
        remaining: if immediate_finish {
            0
        } else {
            initial_thinking.remaining
        },
        min_tokens: p.min_tokens,
        eos_tokens: p.eos_tokens,
        ignore_eos: p.ignore_eos,
        finished: immediate_finish,
        sink: p.sink,
        intervention_ledger: p.intervention_ledger,
        cancel_flag: p.cancel_flag,
        temperature: p.temperature,
        top_k: p.top_k,
        top_p: p.top_p,
        top_n_sigma: p.top_n_sigma,
        min_p: p.min_p,
        repetition_penalty: p.repetition_penalty,
        presence_penalty: p.presence_penalty,
        frequency_penalty: p.frequency_penalty,
        repetition_penalty_window: 256,
        lz_penalty: p.lz_penalty,
        dry_multiplier: p.dry_multiplier,
        dry_base: p.dry_base,
        dry_allowed_length: p.dry_allowed_length,
        dry_sequence_breakers: Vec::new(),
        logit_bias: p.logit_bias,
        pending_drafts: Vec::new(),
        draft_origin: DraftOrigin::default(),
        last_verify_accepted: 0,
        self_context: Default::default(),
        pending_tree_payload: None,
        pending_dflash_sampled_proposal: None,
        inside_thinking: initial_thinking.inside_thinking,
        enable_thinking: p.enable_thinking,
        thinking_budget: initial_thinking.thinking_budget,
        spontaneous_think_budget: p.spontaneous_think_budget,
        thinking_tokens: initial_thinking.thinking_tokens,
        cached_prompt_tokens: cached_prompt_tok,
        difficulty_probe: Default::default(),
        think_nudge: super::think_nudge::new_state(p.think_nudge),
        budget_nudge: None,
        force_end_thinking: initial_thinking.force_end_thinking,
        consecutive_confident: 0,
        in_code_fence: initial_thinking.in_code_fence,
        think_end_token,
        think_start_token,
        // When thinking is disabled but model supports thinking, the template
        // pre-closes with `<think>\n\n</think>\n\n`. Set think_ended=true so
        // the </think> logit suppression is active from the start.
        think_ended: initial_thinking.think_ended,
        think_just_ended: initial_thinking.think_just_ended,
        think_skip_count: 0,
        post_think_gate_steps: 0,
        require_tool_call: use_legacy_tool_call,
        tool_call_start_token,
        tool_call_opened: false,
        inside_tool_body: false,
        suppress_tool_call: p.suppress_tool_call,
        disable_mtp: p.disable_mtp,
        content_started: false,
        content_tokens: 0,
        prose_tokens_since_last_tool: 0,
        think_watchdog_fires: 0,
        rollback_count: 0,
        ssm_rollback_ring: SsmDecodeRing::new(ssm_ring_capacity),
        tool_call_end_token,
        grammar_state: p.grammar_state,
        last_token_time: now,
        request_start: p.request_start,
        decode_start: now,
        seed: p.seed,
        top_logprobs: p.top_logprobs,
        logprobs_data: if initial_thinking.first_is_visible(spontaneous_think) {
            p.first_logprobs.into_iter().collect()
        } else {
            Vec::new()
        },
        timeout_at: p.timeout_at,
        decode_timeout_at: p.decode_timeout_at,
        stall_mark: StallMark::default(),
        adaptive: crate::adaptive_sampler::AdaptiveSamplingState::new(temperature),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scheduler::test_fixtures::{model, prefill_in_progress};

    fn promote_failed(prefilling: &mut Vec<PrefillInProgress>, m: &model::TestModel) {
        let mut active = Vec::new();
        promote_completed_prefills(
            m,
            prefilling,
            vec![(0, None)],
            &mut active,
            0,
            None,
            None,
            None,
            None,
            None,
        );
        assert!(active.is_empty());
        assert!(prefilling.is_empty());
        assert_eq!(*m.frees.lock().unwrap(), vec![0], "the sequence is freed");
    }

    /// M2: a chunked prefill that fails after chunk 0 reports `(idx, None)`.
    /// The client must receive an error frame; before the fix the sink was
    /// dropped silently (blocking: HTTP 500 "Inference cancelled";
    /// streaming: a truncated stream with no error event).
    #[test]
    fn a_failed_chunked_prefill_tells_the_blocking_client() {
        let m = model::recording();
        let (tx, mut rx) = tokio::sync::oneshot::channel();
        let mut prefilling = vec![prefill_in_progress(ResponseSink::Blocking(Some(tx)))];

        promote_failed(&mut prefilling, &m);

        let err = match rx.try_recv() {
            Ok(Err(err)) => err,
            Ok(Ok(_)) => panic!("a prefill failure is an error"),
            Err(_) => panic!("the client is answered, not dropped"),
        };
        assert!(err.to_string().contains("prefill failed"), "{err:#}");
    }

    #[test]
    fn a_failed_chunked_prefill_tells_the_streaming_client() {
        let m = model::recording();
        let (tx, mut rx) = tokio::sync::mpsc::channel(4);
        let mut prefilling = vec![prefill_in_progress(ResponseSink::Streaming(tx))];

        promote_failed(&mut prefilling, &m);

        assert!(
            matches!(rx.try_recv(), Ok(StreamEvent::Error(msg)) if msg.contains("prefill failed")),
            "the stream carries an error frame before it closes"
        );
    }

    /// H2: a prefill whose deadline expired (or client disconnected/cancelled)
    /// during its final chunk must retire before sending the first token or
    /// entering `active`.
    #[test]
    fn stopped_prefill_is_retired_before_promotion() {
        let m = model::recording();
        let (tx, mut rx) = tokio::sync::mpsc::channel(4);
        let mut p = prefill_in_progress(ResponseSink::Streaming(tx));
        p.timeout_at = Some(
            Instant::now()
                .checked_sub(std::time::Duration::from_secs(1))
                .unwrap_or_else(Instant::now),
        );
        let mut prefilling = vec![p];
        let mut active = Vec::new();

        promote_completed_prefills(
            &m,
            &mut prefilling,
            vec![(0, Some(42))],
            &mut active,
            7,
            None,
            None,
            None,
            None,
            None,
        );

        assert!(active.is_empty(), "timed-out prefill must not enter active");
        assert!(prefilling.is_empty());
        assert!(
            matches!(rx.try_recv(), Ok(StreamEvent::Error(msg)) if msg.contains("timed out")),
            "live timed-out stream receives a timeout error instead of a first token"
        );
        assert_eq!(
            *m.lifecycle_events.lock().expect("lifecycle_events lock"),
            ["sync:7", "sync:0", "free:0"],
            "stream drain barrier precedes free_sequence"
        );
    }

    /// M3: when `publish_thinking_done` marks the sequence finished (full or
    /// dropped stream), promotion must release the sequence immediately rather
    /// than pushing a finished sequence into `active`.
    #[test]
    fn full_sink_on_hidden_thinking_boundary_finishes_without_entering_active() {
        let m = model::recording();
        let (tx, _rx) = tokio::sync::mpsc::channel(1);
        tx.try_send(StreamEvent::Token(0))
            .expect("fill channel to force publish_thinking_done rejection");
        let mut p = prefill_in_progress(ResponseSink::Streaming(tx));
        p.enable_thinking = true;
        p.thinking_budget = Some(0);
        let mut prefilling = vec![p];
        let mut active = Vec::new();

        promote_completed_prefills(
            &m,
            &mut prefilling,
            vec![(0, Some(99))],
            &mut active,
            0,
            Some(99),
            None,
            None,
            None,
            None,
        );

        assert!(
            active.is_empty(),
            "sequence finished by publish_thinking_done must not be pushed to active"
        );
        assert!(prefilling.is_empty());
        assert_eq!(
            *m.frees.lock().expect("frees lock"),
            vec![0],
            "sequence slot is released immediately"
        );
    }

    /// Regression for cohort deferral when a thinking request samples a visible
    /// `</think>` as its first token (`thinking_budget != Some(0)`): deferred
    /// flush must still emit `StreamEvent::Token(</think>)` so subsequent
    /// visible tokens are not trapped inside reasoning_content.
    #[test]
    fn deferred_cohort_emits_visible_first_token_think_end() {
        let m = model::recording();
        let (tx_a, mut rx_a) = tokio::sync::mpsc::channel(4);
        let (tx_b, mut rx_b) = tokio::sync::mpsc::channel(4);
        let mut p_a = prefill_in_progress(ResponseSink::Streaming(tx_a));
        p_a.enable_thinking = true;
        p_a.thinking_budget = Some(64);
        p_a.max_tokens = 16;
        let mut p_b = prefill_in_progress(ResponseSink::Streaming(tx_b));
        p_b.max_tokens = 16;
        let mut prefilling = vec![p_a, p_b];
        let mut active = Vec::new();
        let think_end = 99u32;

        promote_completed_prefills(
            &m,
            &mut prefilling,
            vec![(0, Some(think_end)), (1, Some(42))],
            &mut active,
            0,
            Some(think_end),
            None,
            None,
            None,
            None,
        );

        assert_eq!(active.len(), 2);
        assert!(active[0].think_just_ended);
        assert_eq!(active[0].output_tokens, vec![think_end]);
        assert!(
            matches!(rx_a.try_recv(), Ok(StreamEvent::Token(tok)) if tok == think_end),
            "deferred cohort flush must send visible first-token </think>"
        );
        assert!(
            matches!(rx_b.try_recv(), Ok(StreamEvent::Token(42))),
            "sibling cohort stream receives its deferred first token"
        );
    }
}
