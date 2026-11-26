// SPDX-License-Identifier: AGPL-3.0-only

//! prefill_request (single-shot prefill).

use super::prefill_stream_barrier::drain_before_prefill_cleanup;
use super::*;

/// Prefill a new request and return an ActiveSeq ready for batched decode.
/// Returns None if the sequence completed during prefill (EOS on first token).
pub fn prefill_request(
    think_end_token: Option<u32>,
    think_start_token: Option<u32>,
    code_fence_token: Option<u32>,
    tool_call_start_token: Option<u32>,
    tool_call_end_token: Option<u32>,
    model: &dyn Model,
    mut req: InferenceRequest,
    eos_tokens: &[u32],
    grammar_engine: &mut Option<GrammarEngine>,
    spontaneous_think_budget: u32,
) -> Result<Option<ActiveSeq>> {
    // Merge user-supplied stop tokens with model EOS tokens.
    let req_ignore_eos = req.ignore_eos();
    let eos_tokens = sequence_stop_ids(eos_tokens, req.take_stop_tokens(), req_ignore_eos);
    let eos_tokens = &eos_tokens;

    let top_k = req.top_k();
    let top_p = req.top_p();
    let top_n_sigma = req.top_n_sigma();
    let min_p = req.min_p();
    let repetition_penalty = req.repetition_penalty();
    let presence_penalty = req.presence_penalty();
    let frequency_penalty = req.frequency_penalty();
    let dry_multiplier = req.dry_multiplier();
    let dry_base = req.dry_base();
    let dry_allowed_length = req.dry_allowed_length();
    let req_lz_penalty = req.lz_penalty();
    let logit_bias = req.logit_bias().to_vec();
    let req_min_tokens = req.min_tokens();
    let req_session_hash = req.session_hash();
    let req_enable_thinking = req.enable_thinking();
    let req_thinking_budget = req.thinking_budget();
    if req_enable_thinking {
        tracing::info!("Thinking enabled, budget={:?}", req_thinking_budget);
    }
    let req_require_tool_call = req.require_tool_call();
    let req_suppress_tool_call = req.suppress_tool_call();
    let req_disable_mtp = req.disable_mtp();
    let mut req_seed = req.seed();
    let req_top_logprobs = req.top_logprobs();
    let req_timeout_at = req.timeout_at();
    let req_decode_timeout_at = req.decode_timeout_at();
    let req_intervention_ledger = req.intervention_ledger();
    let req_think_nudge = req.think_nudge();
    let grammar_spec = req.take_grammar_spec();
    let mut grammar_state = compile_grammar_state(grammar_engine, &grammar_spec);
    let mut first_params = request_sampling_params(&req);
    let (prompt_tokens, max_tokens, mut sink, image_pixels, temperature, cancel_flag) = match req {
        InferenceRequest::Streaming {
            prompt_tokens,
            max_tokens,
            temperature,
            token_tx,
            image_pixels,
            cancel_flag,
            ..
        } => (
            prompt_tokens,
            max_tokens,
            ResponseSink::Streaming(token_tx),
            image_pixels,
            temperature,
            Some(cancel_flag),
        ),
        InferenceRequest::Blocking {
            prompt_tokens,
            max_tokens,
            temperature,
            response_tx,
            image_pixels,
            ..
        } => (
            prompt_tokens,
            max_tokens,
            ResponseSink::Blocking(Some(response_tx)),
            image_pixels,
            temperature,
            None,
        ),
    };

    let request_start = Instant::now();
    materialize_prompt_sampling_seed(temperature, &mut req_seed, &prompt_tokens);
    if let Err(error) = materialize_effective_sampling_seed(temperature, &mut req_seed) {
        send_error_to_sink(
            &mut sink,
            &format!("request RNG initialization failed: {error:#}"),
        );
        return Err(error);
    }
    first_params.seed = req_seed;
    if let Err(reason) = spark_runtime::sampler::device_topk::admit(&first_params) {
        send_error_to_sink(&mut sink, &reason);
        return Err(anyhow::anyhow!(reason));
    }
    let req_prefix_hash =
        super::continuation::compute_prefix_hash_with_images(&prompt_tokens, &image_pixels);
    tracing::info!(
        "Prefilling: {} prompt tokens, max_tokens={max_tokens}",
        prompt_tokens.len(),
    );
    // `sink` was extracted from `req` above. From here on, ANY error must
    // be surfaced to the client via `sink` BEFORE returning Err, otherwise
    // the client sees the misleading "Inference cancelled" error from the
    // API layer when the channel drops silently.
    if stop_unallocated_prefill(
        cancel_flag.as_ref(),
        req_timeout_at,
        &mut sink,
        "single-shot",
    ) {
        return Ok(None);
    }
    let (mut seq, mut reset_proof) =
        match super::continuation::ContinuationRecord::claim_sequence(model, req_prefix_hash) {
            Ok(claimed) => claimed,
            Err(e) => {
                let msg = format!("alloc_sequence failed: {e:#}");
                send_error_to_sink(&mut sink, &msg);
                return Err(e);
            }
        };
    seq.session_hash = req_session_hash;
    seq.qwen4_qsa_required = prefill_request_can_reach_qwen4_qsa(prompt_tokens.len(), max_tokens);
    super::qwen4_batch_route::record_request_admission(
        model,
        &seq,
        prompt_tokens.len(),
        max_tokens,
    );

    // Guard: free SSM slot on any error after allocation (Bug #16).
    let prefill_result = (|| -> Result<(u32, Option<crate::api::TokenLogprobs>)> {
        // Vision: encode images and store embeddings for prefill token overwrite.
        // Empty input invalidates the prior request's published images.
        model.prepare_vision_embed(&image_pixels)?;

        // EP: broadcast prefill command + tokens to worker (bulk, single NCCL op).
        model.ep_broadcast_cmd(0xFFFFFFF0)?;
        model.ep_broadcast_cmd(prompt_tokens.len() as u32)?;
        model.ep_broadcast_cmd(0)?; // chunk_start = 0 (non-chunked)
        model.ep_broadcast_cmd(prompt_tokens.len() as u32)?; // full prompt length
        model.ep_broadcast_tokens(&prompt_tokens)?;

        let logits = model.prefill(&prompt_tokens, &mut seq, 0)?;
        let _gumbel_guard = (super::stochastic_verify_walk::spec_verify_policy()
            == super::stochastic_verify_walk::SpecVerifyPolicy::Gumbel
            && first_params.temperature > 0.0)
            .then_some(first_params.seed)
            .flatten()
            .map(|seed| {
                spark_runtime::sampler::gumbel::GumbelPosGuard::new(
                    seed,
                    prompt_tokens.len() as u64,
                )
            });
        let sampled = sample_token_with_logprobs_and_grammar(
            model,
            logits,
            &first_params,
            eos_tokens,
            &[],
            req_top_logprobs,
            grammar_state.as_mut(),
            prefill_first_token_applies_grammar(req_enable_thinking, think_end_token),
        )?;
        super::logits_row_receipt::prefill_raw(model, logits, req_session_hash, sampled.0);
        Ok(sampled)
    })();

    let (first, first_lp) = match prefill_result {
        Ok(token) => token,
        Err(e) => {
            drain_before_prefill_cleanup(model, 0, "single-shot prefill failure");
            let msg = format!("prefill failed: {e:#}");
            send_error_to_sink(&mut sink, &msg);
            if let Err(free_err) = super::continuation::ContinuationRecord::abort_prefill_sequence(
                model,
                &mut seq,
                &mut reset_proof,
            ) {
                tracing::error!(
                    "prefill_b_step: free_sequence (after prefill error): {free_err:#}"
                );
            }
            if let Err(bcast_err) = model.ep_broadcast_cmd(0xFFFFFFF1) {
                tracing::error!(
                    "prefill_b_step: ep_broadcast (after prefill error): {bcast_err:#}"
                );
            }
            return Err(e);
        }
    };

    if stop_allocated_prefill(
        model,
        0,
        &mut seq,
        &mut reset_proof,
        &mut sink,
        cancel_flag.as_ref(),
        req_timeout_at,
        "single-shot",
    ) {
        return Ok(None);
    }

    crate::metrics::compat_metrics::add_prompt_tokens(
        u64::try_from(prompt_tokens.len()).unwrap_or(u64::MAX),
    );

    // Spontaneous <think>: if the first token is <think> and thinking was not
    // requested, suppress it and enter thinking mode on the ActiveSeq.
    // deepseek_v41: a first <think> is an ordinary token for the Python engine.
    let spontaneous_think =
        !req_enable_thinking && think_start_token == Some(first) && !crate::dsv41::serving();
    // When grammar is active, disable legacy require_tool_call (grammar handles EOS).
    let use_legacy_tool_call =
        req_require_tool_call && grammar_state.is_none() && tool_call_start_token.is_some();
    let eos_suppressed_outside_thinking = eos_is_suppressed_by_non_thinking_guards(
        grammar_state.as_ref(),
        use_legacy_tool_call,
        0,
        req_min_tokens,
    );
    let initial_thinking = initial_thinking_state(
        req_enable_thinking,
        spontaneous_think,
        first,
        think_end_token,
        eos_tokens,
        req_thinking_budget,
        crate::api::chat::effective_spontaneous_think_budget(spontaneous_think_budget),
        code_fence_token,
        max_tokens,
        honor_eos_inside_thinking(),
        eos_suppressed_outside_thinking,
    );
    let first_is_visible = initial_thinking.first_is_visible(spontaneous_think);
    let grammar_terminated = advance_prefill_first_token_grammar(
        &mut grammar_state,
        first,
        first_is_visible,
        initial_thinking.inside_thinking || initial_thinking.closed_on_first,
    );
    if first_is_visible
        && let ResponseSink::Streaming(ref tx) = sink
        && !stream_send_bounded(
            tx,
            first_token_event(first, &first_lp, eos_tokens.contains(&first)),
            "prefill_b_step first token",
        )
    {
        if let Err(error) = super::continuation::ContinuationRecord::abort_prefill_sequence(
            model,
            &mut seq,
            &mut reset_proof,
        ) {
            tracing::error!("prefill_b_step: free_sequence after first-token rejection: {error:#}");
        }
        if let Err(error) = model.ep_broadcast_cmd(0xFFFFFFF1) {
            tracing::error!("prefill_b_step: EP reset after first-token rejection: {error:#}");
        }
        return Ok(None);
    }

    let output_tokens = if first_is_visible {
        crate::metrics::compat_metrics::add_generation_tokens(1);
        vec![first]
    } else {
        vec![]
    };

    let now = Instant::now();
    let cached_prompt_tok = seq.cached_prefix_tokens as u32;

    if crate::api::disagg::has_pending_prefill_export(req_session_hash)
        || (first_is_visible
            && (eos_tokens.contains(&first) || max_tokens <= 1 || grammar_terminated))
    {
        let prefix_hash = super::continuation::hash_token_incremental(req_prefix_hash, first);
        let mut a = ActiveSeq {
            stop_cause: None,
            seq,
            session_hash: req_session_hash,
            prompt_prefix_hash: req_prefix_hash,
            prefix_hash,
            record_state: super::continuation::RecordState::Live,
            reset_proof,
            last_token: first,
            output_tokens,
            max_output_tokens: max_tokens,
            remaining: 0,
            min_tokens: req_min_tokens,
            eos_tokens: eos_tokens.to_vec(),
            ignore_eos: req_ignore_eos,
            finished: true,
            sink,
            intervention_ledger: req_intervention_ledger.clone(),
            cancel_flag: cancel_flag.clone(),
            temperature,
            top_k,
            top_p,
            top_n_sigma,
            min_p,
            repetition_penalty,
            repetition_penalty_window: 256,
            presence_penalty,
            frequency_penalty,
            lz_penalty: req_lz_penalty,
            dry_multiplier,
            dry_base,
            dry_allowed_length,
            dry_sequence_breakers: Vec::new(),
            logit_bias: logit_bias.clone(),
            pending_drafts: Vec::new(),
            draft_origin: DraftOrigin::default(),
            last_verify_accepted: 0,
            self_context: Default::default(),
            pending_tree_payload: None,
            pending_dflash_sampled_proposal: None,
            inside_thinking: initial_thinking.inside_thinking,
            enable_thinking: req_enable_thinking,
            thinking_budget: initial_thinking.thinking_budget,
            spontaneous_think_budget,
            thinking_tokens: initial_thinking.thinking_tokens,
            cached_prompt_tokens: cached_prompt_tok,
            difficulty_probe: Default::default(),
            think_nudge: super::think_nudge::new_state(req_think_nudge.clone()),
            budget_nudge: None,
            force_end_thinking: initial_thinking.force_end_thinking,
            consecutive_confident: 0,
            in_code_fence: initial_thinking.in_code_fence,
            think_end_token,
            think_start_token,
            think_ended: initial_thinking.think_ended,
            think_just_ended: initial_thinking.think_just_ended,
            think_skip_count: 0,
            post_think_gate_steps: 0,
            require_tool_call: use_legacy_tool_call,
            suppress_tool_call: req_suppress_tool_call,
            disable_mtp: req_disable_mtp,
            content_started: false,
            content_tokens: 0,
            prose_tokens_since_last_tool: 0,
            think_watchdog_fires: 0,
            rollback_count: 0,
            ssm_rollback_ring: SsmDecodeRing::new(model.decode_rollback_ring_slots()),
            tool_call_start_token,
            tool_call_opened: false,
            inside_tool_body: false,
            tool_call_end_token,
            grammar_state,
            last_token_time: now,
            request_start,
            decode_start: now,
            seed: req_seed,
            top_logprobs: req_top_logprobs,
            logprobs_data: first_lp
                .clone()
                .filter(|_| initial_thinking.first_is_visible(spontaneous_think))
                .into_iter()
                .collect(),
            timeout_at: req_timeout_at,
            decode_timeout_at: req_decode_timeout_at,
            stall_mark: StallMark::default(),
            adaptive: crate::adaptive_sampler::AdaptiveSamplingState::new(temperature),
        };
        super::logits_row_receipt::prefill_output(&a);
        crate::api::disagg::maybe_fulfill_prefill_export(
            model,
            a.session_hash,
            &a.seq,
            a.last_token,
            a.cached_prompt_tokens,
            a.request_start,
        );
        finish_sequence(model, &mut a);
        return Ok(None);
    }

    // K8 item A: the MTP last-K replay deferred past the first token.
    if let Err(e) = finish_deferred_prefill_tail(
        model,
        &mut seq,
        first,
        &first_params,
        eos_tokens,
        req_top_logprobs,
    ) {
        let msg = format!("deferred prefill tail failed: {e:#}");
        send_error_to_sink(&mut sink, &msg);
        if let Err(free_err) = super::continuation::ContinuationRecord::abort_prefill_sequence(
            model,
            &mut seq,
            &mut reset_proof,
        ) {
            tracing::error!(
                "prefill_b_step: free_sequence (after deferred tail error): {free_err:#}"
            );
        }
        if let Err(bcast_err) = model.ep_broadcast_cmd(0xFFFFFFF1) {
            tracing::error!(
                "prefill_b_step: ep_broadcast (after deferred tail error): {bcast_err:#}"
            );
        }
        return Err(e);
    }

    let prefix_hash = if initial_thinking.first_is_visible(spontaneous_think) {
        super::continuation::hash_token_incremental(req_prefix_hash, first)
    } else {
        req_prefix_hash
    };
    let mut a = ActiveSeq {
        stop_cause: None,
        seq,
        session_hash: req_session_hash,
        prompt_prefix_hash: req_prefix_hash,
        prefix_hash,
        record_state: super::continuation::RecordState::Live,
        reset_proof,
        last_token: first,
        output_tokens,
        max_output_tokens: max_tokens,
        remaining: initial_thinking.remaining,
        min_tokens: req_min_tokens,
        eos_tokens: eos_tokens.to_vec(),
        ignore_eos: req_ignore_eos,
        finished: false,
        sink,
        intervention_ledger: req_intervention_ledger,
        cancel_flag,
        temperature,
        top_k,
        top_p,
        top_n_sigma,
        min_p,
        repetition_penalty,
        repetition_penalty_window: 256,
        presence_penalty,
        frequency_penalty,
        lz_penalty: req_lz_penalty,
        dry_multiplier,
        dry_base,
        dry_allowed_length,
        dry_sequence_breakers: Vec::new(),
        logit_bias,
        pending_drafts: Vec::new(),
        draft_origin: DraftOrigin::default(),
        last_verify_accepted: 0,
        self_context: Default::default(),
        pending_tree_payload: None,
        pending_dflash_sampled_proposal: None,
        inside_thinking: initial_thinking.inside_thinking,
        enable_thinking: req_enable_thinking,
        thinking_budget: initial_thinking.thinking_budget,
        spontaneous_think_budget,
        thinking_tokens: initial_thinking.thinking_tokens,
        cached_prompt_tokens: cached_prompt_tok,
        difficulty_probe: Default::default(),
        think_nudge: super::think_nudge::new_state(req_think_nudge.clone()),
        budget_nudge: None,
        force_end_thinking: initial_thinking.force_end_thinking,
        consecutive_confident: 0,
        in_code_fence: initial_thinking.in_code_fence,
        think_end_token,
        think_start_token,
        think_ended: initial_thinking.think_ended,
        think_just_ended: initial_thinking.think_just_ended,
        think_skip_count: 0,
        post_think_gate_steps: 0,
        require_tool_call: use_legacy_tool_call,
        suppress_tool_call: req_suppress_tool_call,
        disable_mtp: req_disable_mtp,
        content_started: false,
        content_tokens: 0,
        prose_tokens_since_last_tool: 0,
        think_watchdog_fires: 0,
        rollback_count: 0,
        ssm_rollback_ring: SsmDecodeRing::new(model.decode_rollback_ring_slots()),
        tool_call_start_token,
        tool_call_opened: false,
        inside_tool_body: false,
        tool_call_end_token,
        grammar_state,
        last_token_time: now,
        request_start,
        decode_start: now,
        seed: req_seed,
        top_logprobs: req_top_logprobs,
        logprobs_data: first_lp
            .clone()
            .filter(|_| initial_thinking.first_is_visible(spontaneous_think))
            .into_iter()
            .collect(),
        timeout_at: req_timeout_at,
        decode_timeout_at: req_decode_timeout_at,
        stall_mark: StallMark::default(),
        adaptive: crate::adaptive_sampler::AdaptiveSamplingState::new(temperature),
    };
    super::logits_row_receipt::prefill_output(&a);
    if initial_thinking.publishes_hidden_boundary() {
        publish_thinking_done(&mut a);
    }
    if a.finished {
        finish_sequence(model, &mut a);
        return Ok(None);
    }
    if initial_thinking.closed_on_first {
        rollback::snapshot_boundary_if_ssm(&mut a, model);
    }
    Ok(Some(a))
}
