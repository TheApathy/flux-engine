// SPDX-License-Identifier: AGPL-3.0-only

//! start_chunked_prefill (chunked prefill orchestration) and shared
//! request-preparation / first-token finalization helpers used by both the
//! serial single-request entry point and the Qwen4 short-prefill burst path.

use super::prefill_stream_barrier::{complete_prefill_handoff, drain_before_prefill_cleanup};
use super::*;

mod qwen4_qsa;
use qwen4_qsa::request_can_reach_qwen4_qsa;

/// Whether a prompt/budget pair can cross the Qwen4 QSA dense-attention boundary.
#[must_use]
pub(super) fn prefill_request_can_reach_qwen4_qsa(prompt_tokens: usize, max_tokens: usize) -> bool {
    request_can_reach_qwen4_qsa(prompt_tokens, max_tokens)
}

fn prefill_phase_profile_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED
        .get_or_init(|| std::env::var("ATLAS_PREFILL_PHASE_PROFILE").ok().as_deref() == Some("1"))
}

/// Static special-token and thinking-budget knobs threaded into first-token finalization.
#[derive(Clone, Copy, Debug)]
pub(super) struct PrefillTokenConfig {
    pub think_end_token: Option<u32>,
    pub think_start_token: Option<u32>,
    pub code_fence_token: Option<u32>,
    pub tool_call_start_token: Option<u32>,
    pub tool_call_end_token: Option<u32>,
    pub spontaneous_think_budget: u32,
}

/// Validated, seed-materialized prefill request ready for sequence allocation
/// and first-token finalization.
pub(super) struct PreparedPrefillRequest {
    pub prompt_tokens: Vec<u32>,
    pub max_tokens: usize,
    pub sink: ResponseSink,
    pub image_pixels: Vec<(Vec<f32>, usize, usize)>,
    pub temperature: f32,
    pub cancel_flag: Option<Arc<std::sync::atomic::AtomicBool>>,
    pub eos_tokens: Vec<u32>,
    pub ignore_eos: bool,
    pub top_k: u32,
    pub top_p: f32,
    pub top_n_sigma: f32,
    pub min_p: f32,
    pub repetition_penalty: f32,
    pub presence_penalty: f32,
    pub frequency_penalty: f32,
    pub dry_multiplier: f32,
    pub dry_base: f32,
    pub dry_allowed_length: u32,
    pub lz_penalty: f32,
    pub logit_bias: Vec<(u32, f32)>,
    pub min_tokens: usize,
    pub session_hash: u64,
    pub enable_thinking: bool,
    pub thinking_budget: Option<u32>,
    pub require_tool_call: bool,
    pub suppress_tool_call: bool,
    pub disable_mtp: bool,
    pub seed: Option<u64>,
    pub top_logprobs: Option<u8>,
    pub timeout_at: Option<Instant>,
    pub decode_timeout_at: Option<Instant>,
    pub intervention_ledger: Option<crate::openai::SharedInterventionLedger>,
    pub think_nudge: Option<Arc<super::think_nudge::ThinkNudgeConfig>>,
    pub grammar_state: Option<GrammarState>,
    pub first_params: SamplingParams,
    pub request_start: Instant,
    pub prefix_hash: u64,
}

/// Materialize stop IDs, grammar state, effective sampling seed, device top-k
/// admission, and prefix hash for one incoming [`InferenceRequest`].
///
/// Any error is reported on the request's sink before returning `Err`.
pub(super) fn prepare_prefill_request(
    mut req: InferenceRequest,
    eos_tokens: &[u32],
    grammar_engine: &mut Option<GrammarEngine>,
) -> Result<PreparedPrefillRequest> {
    // Merge user-supplied stop tokens with model EOS tokens.
    let ignore_eos = req.ignore_eos();
    let eos_tokens = sequence_stop_ids(eos_tokens, req.take_stop_tokens(), ignore_eos);

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
    let lz_penalty = req.lz_penalty();
    let logit_bias = req.logit_bias().to_vec();
    let min_tokens = req.min_tokens();
    let session_hash = req.session_hash();
    let enable_thinking = req.enable_thinking();
    let thinking_budget = req.thinking_budget();
    if enable_thinking {
        tracing::info!("Thinking enabled, budget={:?}", thinking_budget);
    }
    let require_tool_call = req.require_tool_call();
    let suppress_tool_call = req.suppress_tool_call();
    let disable_mtp = req.disable_mtp();
    let mut req_seed = req.seed();
    let top_logprobs = req.top_logprobs();
    let timeout_at = req.timeout_at();
    let decode_timeout_at = req.decode_timeout_at();
    let intervention_ledger = req.intervention_ledger();
    let think_nudge = req.think_nudge();
    let grammar_spec = req.take_grammar_spec();
    let grammar_state = compile_grammar_state(grammar_engine, &grammar_spec);
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
    let prefix_hash =
        super::continuation::compute_prefix_hash_with_images(&prompt_tokens, &image_pixels);

    Ok(PreparedPrefillRequest {
        prompt_tokens,
        max_tokens,
        sink,
        image_pixels,
        temperature,
        cancel_flag,
        eos_tokens,
        ignore_eos,
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
        min_tokens,
        session_hash,
        enable_thinking,
        thinking_budget,
        require_tool_call,
        suppress_tool_call,
        disable_mtp,
        seed: req_seed,
        top_logprobs,
        timeout_at,
        decode_timeout_at,
        intervention_ledger,
        think_nudge,
        grammar_state,
        first_params,
        request_start,
        prefix_hash,
    })
}

/// Start a chunked prefill: process chunk 0, return result.
pub fn start_chunked_prefill(
    think_end_token: Option<u32>,
    think_start_token: Option<u32>,
    code_fence_token: Option<u32>,
    tool_call_start_token: Option<u32>,
    tool_call_end_token: Option<u32>,
    model: &dyn Model,
    req: InferenceRequest,
    eos_tokens: &[u32],
    max_prefill_tokens: usize,
    prefill_stream: u64,
    prefill_event: u64,
    grammar_engine: &mut Option<GrammarEngine>,
    spontaneous_think_budget: u32,
) -> Result<StartPrefillResult> {
    start_chunked_prefill_with_defer(
        think_end_token,
        think_start_token,
        code_fence_token,
        tool_call_start_token,
        tool_call_end_token,
        model,
        req,
        eos_tokens,
        max_prefill_tokens,
        prefill_stream,
        prefill_event,
        grammar_engine,
        spontaneous_think_budget,
        false,
        false,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn start_chunked_prefill_with_defer(
    think_end_token: Option<u32>,
    think_start_token: Option<u32>,
    code_fence_token: Option<u32>,
    tool_call_start_token: Option<u32>,
    tool_call_end_token: Option<u32>,
    model: &dyn Model,
    req: InferenceRequest,
    eos_tokens: &[u32],
    max_prefill_tokens: usize,
    prefill_stream: u64,
    prefill_event: u64,
    grammar_engine: &mut Option<GrammarEngine>,
    spontaneous_think_budget: u32,
    defer_first_stream_event: bool,
    complete_all_chunks: bool,
    next_admission_tokens: Option<&[u32]>,
) -> Result<StartPrefillResult> {
    let mut prep = prepare_prefill_request(req, eos_tokens, grammar_engine)?;
    let image_pixels = std::mem::take(&mut prep.image_pixels);
    let total = prep.prompt_tokens.len();
    let chunk_len = total.min(max_prefill_tokens);
    let req_prefix_hash = prep.prefix_hash;
    let req_session_hash = prep.session_hash;
    let max_tokens = prep.max_tokens;
    let request_start = prep.request_start;
    let req_timeout_at = prep.timeout_at;

    let pstep_on = prefill_phase_profile_enabled();
    let mut pstep_ms: Vec<(&str, f64)> = Vec::new();
    macro_rules! pstep_mark {
        ($name:expr) => {
            if pstep_on {
                pstep_ms.push(($name, request_start.elapsed().as_secs_f64() * 1000.0));
            }
        };
    }

    tracing::info!(
        "Chunked prefill start: {} prompt tokens, chunk_size={}, max_tokens={max_tokens}",
        total,
        chunk_len,
    );

    // From here on, `prep.sink` holds the client's response channel. Any
    // error MUST be reported via send_error_to_sink before returning,
    // otherwise the API layer will turn the dropped channel into a
    // misleading "Inference cancelled" error.
    if stop_unallocated_prefill(
        prep.cancel_flag.as_ref(),
        req_timeout_at,
        &mut prep.sink,
        "first-chunk",
    ) {
        return Ok(StartPrefillResult::Finished);
    }
    // K9 item W (ATLAS_QWEN4_PLE_EARLY_ISSUE=1, default off; a no-op for every
    // other model): start chunk 0's PLE row read now so it overlaps slot
    // allocation. The prefill below consumes it only for these exact tokens.
    model.begin_prefill_admission(&prep.prompt_tokens, chunk_len);
    // ATLAS_ALLOC_SEQ_ASYNC (K10 item A): the model may order the prefill
    // stream after the slot's zeroing with an event instead of a host sync.
    let (mut seq, mut reset_proof) =
        match super::continuation::ContinuationRecord::claim_sequence_on(
            model,
            prefill_stream,
            req_prefix_hash,
        ) {
            Ok(claimed) => claimed,
            Err(e) => {
                let msg = format!("alloc_sequence failed: {e:#}");
                send_error_to_sink(&mut prep.sink, &msg);
                return Err(e);
            }
        };
    seq.session_hash = req_session_hash;
    // Qwen4 dense attention is exact below 2048 tokens. Avoid building all
    // twelve QSA side indexes when the immutable request budget cannot cross
    // that boundary; requests that can cross it still index the entire prefix.
    seq.qwen4_qsa_required = request_can_reach_qwen4_qsa(total, max_tokens);
    super::qwen4_batch_route::record_request_admission(model, &seq, total, max_tokens);
    pstep_mark!("alloc_sequence");

    let disagg_import = crate::api::disagg::take_decode_import(req_session_hash);
    let disagg_imported = disagg_import.is_some();
    let mut chunk_offset = if disagg_imported { total } else { chunk_len };
    let mut is_last = chunk_offset >= total;
    // ATLAS_QWEN4_SEQSTATE_CACHE: the model may restore a saved prefix of this
    // prompt; the first chunk then starts where that prefix ends.
    let mut first_len = chunk_len;

    let imported_first_token = if let Some(import_pkt) = disagg_import {
        let mut cursor = std::io::Cursor::new(&import_pkt.state_bytes);
        if let Err(e) = model.restore_sequence_state(&mut seq, import_pkt.num_blocks, &mut cursor) {
            let err = super::continuation::reclaim_or_quarantine_sequence(
                model,
                seq,
                e,
                "start_chunked_prefill(disagg_restore_sequence_state)",
            );
            send_error_to_sink(
                &mut prep.sink,
                &format!("disagg state import failed: {err:#}"),
            );
            return Err(err);
        }
        seq.tokens = prep.prompt_tokens.clone();
        seq.prompt_len = prep.prompt_tokens.len();
        seq.seq_len = import_pkt.seq_len;
        seq.cached_prefix_tokens = import_pkt.cached_prompt_tokens as usize;
        Some(import_pkt.first_token)
    } else {
        None
    };

    let mut resumed_start = 0usize;
    let logits = if imported_first_token.is_some() {
        resumed_start = chunk_offset;
        DevicePtr::NULL
    } else {
        // Guard: free SSM slot on any error after allocation.
        let prefill_result = (|| -> Result<DevicePtr> {
            // Vision: encode images and store embeddings for chunk 0 token overwrite.
            // Empty input invalidates the prior request's published images.
            model.prepare_vision_embed(&image_pixels)?;

            let first_start = model.resume_conversation(&mut seq, &prep.prompt_tokens)?;
            resumed_start = first_start;
            if first_start > 0 {
                first_len = (total - first_start).min(max_prefill_tokens);
                chunk_offset = first_start + first_len;
                is_last = chunk_offset >= total;
            }

            // EP: broadcast chunk 0 tokens to worker.
            // Send full prompt length + all tokens so worker can do
            // identical Marconi prefix-cache lookups (bug #33 fix).
            // Uses bulk broadcast (single NCCL op) instead of per-token broadcast
            // which caused NCCL timeouts on long prompts (6K+ tokens = 6K+ broadcasts).
            model.ep_broadcast_cmd(0xFFFFFFF0)?;
            model.ep_broadcast_cmd(first_len as u32)?;
            model.ep_broadcast_cmd(first_start as u32)?; // chunk_start
            model.ep_broadcast_cmd(prep.prompt_tokens.len() as u32)?; // full prompt length
            model.ep_broadcast_tokens(&prep.prompt_tokens)?;
            pstep_mark!("ep_broadcast");

            let mut r = model.prefill_chunk(
                &prep.prompt_tokens,
                &mut seq,
                first_start,
                first_len,
                is_last,
                prefill_stream,
            )?;
            pstep_mark!("prefill_chunk");

            if complete_all_chunks {
                while !is_last {
                    // One call can otherwise run a whole 300K prompt with no
                    // cooperative boundary: stop here and let the post-chunk
                    // `stop_allocated_prefill` retire the sequence.
                    if allocated_prefill_should_stop(
                        prep.cancel_flag.as_ref(),
                        req_timeout_at,
                        &prep.sink,
                    ) {
                        break;
                    }
                    let next_len = (total - chunk_offset).min(max_prefill_tokens);
                    let next_is_last = chunk_offset + next_len >= total;
                    model.ep_broadcast_cmd(0xFFFFFFF0)?;
                    model.ep_broadcast_cmd(next_len as u32)?;
                    model.ep_broadcast_cmd(chunk_offset as u32)?;
                    model.ep_broadcast_cmd(prep.prompt_tokens.len() as u32)?;
                    model.ep_broadcast_tokens(&prep.prompt_tokens)?;
                    r = model.prefill_chunk(
                        &prep.prompt_tokens,
                        &mut seq,
                        chunk_offset,
                        next_len,
                        next_is_last,
                        prefill_stream,
                    )?;
                    chunk_offset += next_len;
                    is_last = next_is_last;
                }
            }
            Ok(r)
        })();

        match prefill_result {
            Ok(l) => l,
            Err(e) => {
                tracing::error!("prefill_chunk failed: {e:#}");
                drain_before_prefill_cleanup(model, prefill_stream, "first-chunk prefill failure");
                let msg = format!("prefill_chunk failed: {e:#}");
                send_error_to_sink(&mut prep.sink, &msg);
                if let Err(free_err) =
                    super::continuation::ContinuationRecord::abort_prefill_sequence(
                        model,
                        &mut seq,
                        &mut reset_proof,
                    )
                {
                    tracing::error!(
                        "prefill_a_step: free_sequence (after prefill error): {free_err:#}"
                    );
                }
                if let Err(bcast_err) = model.ep_broadcast_cmd(0xFFFFFFF1) {
                    tracing::error!(
                        "prefill_a_step: ep_broadcast (after prefill error): {bcast_err:#}"
                    );
                }
                return Err(e);
            }
        }
    };

    let newly_prefilled = chunk_offset.saturating_sub(resumed_start);
    if newly_prefilled > 0 {
        crate::metrics::record_prefill_progress(
            req_session_hash,
            chunk_offset,
            total,
            newly_prefilled,
        );
    }

    if is_last && !disagg_imported {
        if let Some(next_tokens) = next_admission_tokens {
            let next_chunk_len = next_tokens.len().min(max_prefill_tokens);
            model.begin_prefill_admission(next_tokens, next_chunk_len);
        }
        // Sync prefill stream before sampling on default_stream. Event
        // failures fall back to a host-blocking drain; a failed drain is fatal.
        complete_prefill_handoff(model, prefill_stream, prefill_event, "first-chunk handoff");
    }
    pstep_mark!("event_wait");

    if stop_allocated_prefill(
        model,
        prefill_stream,
        &mut seq,
        &mut reset_proof,
        &mut prep.sink,
        prep.cancel_flag.as_ref(),
        req_timeout_at,
        "first-chunk",
    ) {
        return Ok(StartPrefillResult::Finished);
    }

    if pstep_on {
        let mut joined: Vec<String> = Vec::with_capacity(pstep_ms.len());
        let mut prev = 0.0f64;
        for (name, t) in &pstep_ms {
            joined.push(format!("{name}={:.1}", t - prev));
            prev = *t;
        }
        tracing::info!("PREFILL_STEPS | {}", joined.join(" "));
    }

    if is_last {
        let tokens_cfg = PrefillTokenConfig {
            think_end_token,
            think_start_token,
            code_fence_token,
            tool_call_start_token,
            tool_call_end_token,
            spontaneous_think_budget,
        };
        finalize_prefill_first_token(
            tokens_cfg,
            model,
            prep,
            seq,
            reset_proof,
            logits,
            imported_first_token,
            disagg_imported,
            prefill_stream,
            defer_first_stream_event,
        )
    } else {
        Ok(StartPrefillResult::InProgress(PrefillInProgress {
            prompt_tokens: prep.prompt_tokens,
            session_hash: prep.session_hash,
            prompt_prefix_hash: prep.prefix_hash,
            prefix_hash: prep.prefix_hash,
            reset_proof,
            seq,
            chunk_offset,
            chunk_budget: max_prefill_tokens,
            max_tokens: prep.max_tokens,
            min_tokens: prep.min_tokens,
            eos_tokens: prep.eos_tokens,
            ignore_eos: prep.ignore_eos,
            sink: prep.sink,
            intervention_ledger: prep.intervention_ledger,
            think_nudge: prep.think_nudge,
            cancel_flag: prep.cancel_flag,
            request_start: prep.request_start,
            temperature: prep.temperature,
            top_k: prep.top_k,
            top_p: prep.top_p,
            top_n_sigma: prep.top_n_sigma,
            min_p: prep.min_p,
            repetition_penalty: prep.repetition_penalty,
            repetition_penalty_window: 256,
            presence_penalty: prep.presence_penalty,
            frequency_penalty: prep.frequency_penalty,
            lz_penalty: prep.lz_penalty,
            // DRY comes from the request (populated at
            // api.rs from `sampling_presets.tools.dry_*` when tools
            // are active). 0.0 for other presets leaves DRY inert.
            dry_multiplier: prep.dry_multiplier,
            dry_base: prep.dry_base,
            dry_allowed_length: prep.dry_allowed_length,
            dry_sequence_breakers: Vec::new(),
            logit_bias: prep.logit_bias,
            enable_thinking: prep.enable_thinking,
            thinking_budget: prep.thinking_budget,
            spontaneous_think_budget,
            require_tool_call: prep.require_tool_call,
            suppress_tool_call: prep.suppress_tool_call,
            disable_mtp: prep.disable_mtp,
            grammar_state: prep.grammar_state,
            seed: prep.seed,
            top_logprobs: prep.top_logprobs,
            first_logprobs: None,
            timeout_at: prep.timeout_at,
            decode_timeout_at: prep.decode_timeout_at,
        }))
    }
}

/// Whether prefill first-token sampling must enforce an active content grammar.
/// Mirrors [`initial_thinking_state`]: skip only when the prompt opened `<think>`.
#[must_use]
pub(super) fn prefill_first_token_applies_grammar(
    enable_thinking: bool,
    think_end_token: Option<u32>,
) -> bool {
    !(enable_thinking && think_end_token.is_some())
}

/// Advance a prefill request's content grammar on its first sampled token.
///
/// Mirrors [`advance_content_grammar`]: skips hidden and thinking-phase tokens
/// (including a first-token `</think>` close). Returns `true` if terminated.
pub(super) fn advance_prefill_first_token_grammar(
    grammar_state: &mut Option<GrammarState>,
    first: u32,
    first_is_visible: bool,
    inside_thinking: bool,
) -> bool {
    if !first_is_visible || inside_thinking {
        return false;
    }
    let Some(gs) = grammar_state.as_mut() else {
        return false;
    };
    if !gs.accept_token(first) {
        tracing::warn!(first, "Prefill first token rejected by grammar matcher");
    }
    gs.is_terminated()
}

/// Shared post-prefill first-token sampling, SSE/blocking emission, deferred
/// MTP tail execution, and [`ActiveSeq`] construction for both single-request
/// chunked prefill and batched short-prefill bursts.
#[allow(clippy::too_many_arguments)]
pub(super) fn finalize_prefill_first_token(
    tokens_cfg: PrefillTokenConfig,
    model: &dyn Model,
    prep: PreparedPrefillRequest,
    mut seq: SequenceState,
    mut reset_proof: Option<super::continuation::ResetProofPayload>,
    logits: DevicePtr,
    imported_first_token: Option<u32>,
    disagg_imported: bool,
    prefill_stream: u64,
    defer_first_stream_event: bool,
) -> Result<StartPrefillResult> {
    let PrefillTokenConfig {
        think_end_token,
        think_start_token,
        code_fence_token,
        tool_call_start_token,
        tool_call_end_token,
        spontaneous_think_budget,
    } = tokens_cfg;
    let PreparedPrefillRequest {
        prompt_tokens,
        max_tokens,
        mut sink,
        image_pixels: _,
        temperature,
        cancel_flag,
        eos_tokens,
        ignore_eos: req_ignore_eos,
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
        lz_penalty: req_lz_penalty,
        logit_bias,
        min_tokens: req_min_tokens,
        session_hash: req_session_hash,
        enable_thinking: req_enable_thinking,
        thinking_budget: req_thinking_budget,
        require_tool_call: req_require_tool_call,
        suppress_tool_call: req_suppress_tool_call,
        disable_mtp: req_disable_mtp,
        seed: req_seed,
        top_logprobs: req_top_logprobs,
        timeout_at: req_timeout_at,
        decode_timeout_at: req_decode_timeout_at,
        intervention_ledger: req_intervention_ledger,
        think_nudge: req_think_nudge,
        mut grammar_state,
        first_params,
        request_start,
        prefix_hash: req_prefix_hash,
    } = prep;
    let total = prompt_tokens.len();
    let eos_tokens = &eos_tokens;

    // Single chunk covered the entire prompt — get first token.
    let (first, first_lp) = if let Some(first_tok) = imported_first_token {
        (first_tok, None)
    } else {
        let _gumbel_guard = (super::stochastic_verify_walk::spec_verify_policy()
            == super::stochastic_verify_walk::SpecVerifyPolicy::Gumbel
            && first_params.temperature > 0.0)
            .then_some(first_params.seed)
            .flatten()
            .map(|seed| spark_runtime::sampler::gumbel::GumbelPosGuard::new(seed, total as u64));
        match sample_token_with_logprobs_and_grammar(
            model,
            logits,
            &first_params,
            eos_tokens,
            &[],
            req_top_logprobs,
            grammar_state.as_mut(),
            prefill_first_token_applies_grammar(req_enable_thinking, think_end_token),
        ) {
            Ok((t, lp)) => {
                super::logits_row_receipt::prefill_raw(model, logits, req_session_hash, t);
                tracing::info!("Prefill first token: {t}");
                (t, lp)
            }
            Err(e) => {
                tracing::error!("sample_token failed: {e:#}");
                drain_before_prefill_cleanup(model, prefill_stream, "first-chunk sampling failure");
                let msg = format!("sample_token failed: {e:#}");
                send_error_to_sink(&mut sink, &msg);
                if let Err(free_err) =
                    super::continuation::ContinuationRecord::abort_prefill_sequence(
                        model,
                        &mut seq,
                        &mut reset_proof,
                    )
                {
                    tracing::error!(
                        "prefill_a_step: free_sequence (after sample error): {free_err:#}"
                    );
                }
                if let Err(bcast_err) = model.ep_broadcast_cmd(0xFFFFFFF1) {
                    tracing::error!(
                        "prefill_a_step: ep_broadcast (after sample error): {bcast_err:#}"
                    );
                }
                return Err(e);
            }
        }
    };

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
    let finish_on_first =
        first_is_visible && (eos_tokens.contains(&first) || max_tokens <= 1 || grammar_terminated);
    let send_first_now = finish_on_first
        || !defer_first_stream_event
        || initial_thinking.publishes_hidden_boundary();
    if send_first_now
        && first_is_visible
        && let ResponseSink::Streaming(ref tx) = sink
        && !stream_send_bounded(
            tx,
            first_token_event(first, &first_lp, eos_tokens.contains(&first)),
            "prefill_a_step first token",
        )
    {
        if let Err(error) = super::continuation::ContinuationRecord::abort_prefill_sequence(
            model,
            &mut seq,
            &mut reset_proof,
        ) {
            tracing::error!("prefill_a_step: free_sequence after first-token rejection: {error:#}");
        }
        if let Err(error) = model.ep_broadcast_cmd(0xFFFFFFF1) {
            tracing::error!("prefill_a_step: EP reset after first-token rejection: {error:#}");
        }
        return Ok(StartPrefillResult::Finished);
    }

    let now = Instant::now();
    let cached_prompt_tok = seq.cached_prefix_tokens as u32;
    if crate::api::disagg::has_pending_prefill_export(req_session_hash) || finish_on_first {
        crate::metrics::compat_metrics::add_generation_tokens(1);
        let prefix_hash = super::continuation::hash_token_incremental(req_prefix_hash, first);
        let mut a = ActiveSeq {
            stop_cause: None,
            defer_emit: Default::default(),
            seq,
            session_hash: req_session_hash,
            prompt_prefix_hash: req_prefix_hash,
            prefix_hash,
            record_state: super::continuation::RecordState::Live,
            reset_proof,
            last_token: first,
            output_tokens: vec![first],
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
            tool_call_start_token,
            tool_call_opened: false,
            inside_tool_body: false,
            suppress_tool_call: req_suppress_tool_call,
            disable_mtp: req_disable_mtp,
            content_started: false,
            content_tokens: 0,
            prose_tokens_since_last_tool: 0,
            think_watchdog_fires: 0,
            rollback_count: 0,
            ssm_rollback_ring: SsmDecodeRing::new(model.decode_rollback_ring_slots()),
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
        Ok(StartPrefillResult::Finished)
    } else {
        // K8 item A: the MTP last-K replay deferred past the first token.
        if !disagg_imported
            && let Err(e) = finish_deferred_prefill_tail(
                model,
                &mut seq,
                first,
                &first_params,
                eos_tokens,
                req_top_logprobs,
            )
        {
            let msg = format!("deferred prefill tail failed: {e:#}");
            send_error_to_sink(&mut sink, &msg);
            if let Err(free_err) = super::continuation::ContinuationRecord::abort_prefill_sequence(
                model,
                &mut seq,
                &mut reset_proof,
            ) {
                tracing::error!(
                    "prefill_a_step: free_sequence (after deferred tail error): {free_err:#}"
                );
            }
            if let Err(bcast_err) = model.ep_broadcast_cmd(0xFFFFFFF1) {
                tracing::error!(
                    "prefill_a_step: ep_broadcast (after deferred tail error): {bcast_err:#}"
                );
            }
            return Err(e);
        }
        let prefix_hash = if initial_thinking.first_is_visible(spontaneous_think) {
            crate::metrics::compat_metrics::add_generation_tokens(1);
            super::continuation::hash_token_incremental(req_prefix_hash, first)
        } else {
            req_prefix_hash
        };
        let mut a = ActiveSeq {
            stop_cause: None,
            defer_emit: Default::default(),
            seq,
            session_hash: req_session_hash,
            prompt_prefix_hash: req_prefix_hash,
            prefix_hash,
            record_state: super::continuation::RecordState::Live,
            reset_proof,
            last_token: first,
            output_tokens: if initial_thinking.first_is_visible(spontaneous_think) {
                vec![first]
            } else {
                vec![]
            },
            max_output_tokens: max_tokens,
            remaining: initial_thinking.remaining,
            min_tokens: req_min_tokens,
            eos_tokens: eos_tokens.to_vec(),
            ignore_eos: req_ignore_eos,
            finished: false,
            sink,
            intervention_ledger: req_intervention_ledger.clone(),
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
            tool_call_start_token,
            tool_call_opened: false,
            inside_tool_body: false,
            suppress_tool_call: req_suppress_tool_call,
            disable_mtp: req_disable_mtp,
            content_started: false,
            content_tokens: 0,
            prose_tokens_since_last_tool: 0,
            think_watchdog_fires: 0,
            rollback_count: 0,
            ssm_rollback_ring: SsmDecodeRing::new(model.decode_rollback_ring_slots()),
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
            return Ok(StartPrefillResult::Finished);
        }
        if initial_thinking.closed_on_first {
            rollback::snapshot_boundary_if_ssm(&mut a, model);
        }
        Ok(StartPrefillResult::Active(a))
    }
}
