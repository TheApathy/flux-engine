// SPDX-License-Identifier: AGPL-3.0-only

//! Token sampling helpers (resample + sample + grammar-constrained sample).

use super::*;

/// Resolve the one request-scoped seed used by every stochastic target and
/// speculative sample in this sequence.
///
/// Explicit API seeds are preserved byte-for-byte. An unseeded stochastic
/// request captures OS entropy exactly once, before its first target sample;
/// later output positions derive their effective seed from this base in
/// [`ActiveSeq::sampling_params`]. Failure to obtain OS entropy is fatal: a
/// per-token clock/thread RNG fallback would make target and DFlash disagree.
pub(super) fn materialize_effective_sampling_seed(
    temperature: f32,
    seed: &mut Option<u64>,
) -> Result<()> {
    materialize_effective_sampling_seed_with(temperature, seed, crate::openai::getrandom)
}

fn materialize_effective_sampling_seed_with(
    temperature: f32,
    seed: &mut Option<u64>,
    fill_entropy: impl FnOnce(&mut [u8]) -> std::result::Result<(), ()>,
) -> Result<()> {
    if temperature <= 0.0 || seed.is_some() {
        return Ok(());
    }
    let mut bytes = [0u8; 8];
    fill_entropy(&mut bytes)
        .map_err(|()| anyhow::anyhow!("OS entropy unavailable for stochastic request seed"))?;
    *seed = Some(u64::from_le_bytes(bytes));
    Ok(())
}

/// Populate `seed` from `prompt_tokens` when `temperature > 0.0`, `seed` is
/// `None`, and `FLUX_PROMPT_SEED=1` (`ATLAS_PROMPT_SEED=1`) is enabled.
///
/// Called ahead of [`materialize_effective_sampling_seed`] so unseeded
/// stochastic requests derive the exact same deterministic prompt seed on
/// non-speculative (`--no-mtp` / `draft=false`) and speculative (`step_mtp` /
/// `draft=true`) routes instead of consuming OS entropy.
pub(super) fn materialize_prompt_sampling_seed(
    temperature: f32,
    seed: &mut Option<u64>,
    prompt_tokens: &[u32],
) {
    materialize_prompt_sampling_seed_with(
        temperature,
        seed,
        prompt_tokens,
        spark_runtime::sampler::gumbel::prompt_seed_enabled(),
    );
}

pub(super) fn materialize_prompt_sampling_seed_with(
    temperature: f32,
    seed: &mut Option<u64>,
    prompt_tokens: &[u32],
    prompt_seed_enabled: bool,
) {
    if temperature > 0.0 && seed.is_none() && prompt_seed_enabled {
        *seed = Some(spark_runtime::sampler::seed_from_prompt_tokens(
            prompt_tokens,
        ));
    }
}

/// A request's sampler settings for its first token, as decode applies them to every later
/// one (`ActiveSeq::sampling_params`). The first token used to get temperature/top-k/top-p
/// only, so logit_bias, top_n_sigma, min_p and the seed were ignored at position 0.
/// Penalties see no history yet; they are carried so the two paths cannot drift.
pub(super) fn request_sampling_params(req: &InferenceRequest) -> SamplingParams {
    let temperature = match req {
        InferenceRequest::Blocking { temperature, .. } => *temperature,
        InferenceRequest::Streaming { temperature, .. } => *temperature,
    };
    SamplingParams {
        temperature,
        top_k: req.top_k(),
        top_p: req.top_p(),
        top_n_sigma: req.top_n_sigma(),
        min_p: req.min_p(),
        logit_bias: req.logit_bias().to_vec(),
        repetition_penalty: req.repetition_penalty(),
        repetition_penalty_window: 256,
        presence_penalty: req.presence_penalty(),
        frequency_penalty: req.frequency_penalty(),
        lz_penalty: req.lz_penalty(),
        dry_multiplier: req.dry_multiplier(),
        dry_base: req.dry_base(),
        dry_allowed_length: req.dry_allowed_length(),
        dry_sequence_breakers: Vec::new(),
        max_tokens: 0,
        stop_token_ids: Vec::new(),
        seed: req.seed(),
    }
}

impl PrefillInProgress {
    /// [`request_sampling_params`] for a chunked prefill's first token.
    pub(super) fn sampling_params(&self) -> SamplingParams {
        SamplingParams {
            temperature: self.temperature,
            top_k: self.top_k,
            top_p: self.top_p,
            top_n_sigma: self.top_n_sigma,
            min_p: self.min_p,
            logit_bias: self.logit_bias.clone(),
            repetition_penalty: self.repetition_penalty,
            repetition_penalty_window: self.repetition_penalty_window,
            presence_penalty: self.presence_penalty,
            frequency_penalty: self.frequency_penalty,
            lz_penalty: self.lz_penalty,
            dry_multiplier: self.dry_multiplier,
            dry_base: self.dry_base,
            dry_allowed_length: self.dry_allowed_length,
            dry_sequence_breakers: self.dry_sequence_breakers.clone(),
            max_tokens: 0,
            stop_token_ids: Vec::new(),
            seed: self.seed,
        }
    }
}

impl ActiveSeq {
    /// The sampler settings for this sequence's next token at `temperature`.
    ///
    /// Phase-gated (P3.1, 2026-04-25): inside the tool-call body (between `<tool_call>` and
    /// `</tool_call>`) the JSON is dense with legitimate short repetitions — `":"`, `","`,
    /// key tokens — that DRY/presence/frequency penalties would punish, breaking schema
    /// validity. XGrammar already guarantees the structure there, so the penalties are off;
    /// outside the body (free text + `<think>`), where prose loops live, the full preset
    /// applies. LZ is off whenever a grammar is active.
    pub(super) fn sampling_params(&self, temperature: f32) -> SamplingParams {
        let in_tool = self.inside_tool_body && !self.inside_thinking;
        SamplingParams {
            temperature,
            top_k: self.top_k,
            top_p: self.top_p,
            top_n_sigma: self.top_n_sigma,
            min_p: self.min_p,
            logit_bias: self.logit_bias.clone(),
            repetition_penalty: if in_tool {
                1.0
            } else {
                self.repetition_penalty
            },
            repetition_penalty_window: self.repetition_penalty_window,
            presence_penalty: if in_tool { 0.0 } else { self.presence_penalty },
            frequency_penalty: if in_tool { 0.0 } else { self.frequency_penalty },
            lz_penalty: if self.grammar_state.is_some() {
                0.0
            } else {
                self.lz_penalty
            },
            dry_multiplier: if in_tool { 0.0 } else { self.dry_multiplier },
            dry_base: self.dry_base,
            dry_allowed_length: self.dry_allowed_length,
            dry_sequence_breakers: self.dry_sequence_breakers.clone(),
            max_tokens: 0,
            stop_token_ids: Vec::new(),
            // Advance the seed per token: deterministic but varying.
            seed: self
                .seed
                .map(|s| s.wrapping_add(self.output_tokens.len() as u64)),
        }
    }
}

/// Whether sampling `params` after `history` can pick something other than the raw argmax
/// even at temperature 0: a logit bias, or a penalty with history to act on.
fn needs_sampler(params: &SamplingParams, history: &[u32]) -> bool {
    let penalties = (params.repetition_penalty != 1.0 && params.repetition_penalty > 0.0)
        || params.presence_penalty != 0.0
        || params.frequency_penalty != 0.0
        || params.lz_penalty != 0.0
        || params.dry_multiplier != 0.0;
    params.temperature != 0.0 || !params.logit_bias.is_empty() || (penalties && !history.is_empty())
}

/// Sample one token from device logits with the request's full sampler settings (`params`),
/// penalties acting on `history` (the tokens generated so far).
///
/// `suppress_ids`: token IDs to mask to -inf before sampling (e.g. EOS on first token).
pub fn sample_token(
    model: &dyn Model,
    logits: DevicePtr,
    params: &SamplingParams,
    suppress_ids: &[u32],
    history: &[u32],
) -> Result<u32> {
    sample_token_with_logprobs(model, logits, params, suppress_ids, history, None)
        .map(|(tok, _)| tok)
}

/// [`sample_token`] that also returns the token's logprobs when the request
/// asked for them (`top_logprobs`). The first generated token is sampled here,
/// from the prefill logits, so this is where its logprobs entry comes from;
/// they are taken over the same (suppressed) distribution the token was drawn
/// from, as the decode path does.
pub fn sample_token_with_logprobs(
    model: &dyn Model,
    logits: DevicePtr,
    params: &SamplingParams,
    suppress_ids: &[u32],
    history: &[u32],
    top_logprobs: Option<u8>,
) -> Result<(u32, Option<crate::api::TokenLogprobs>)> {
    // deepseek_v41: the Python engine may end on its very first token.
    let suppress_ids = if crate::dsv41::serving() {
        &[][..]
    } else {
        suppress_ids
    };
    if sample_uses_device_argmax(params, suppress_ids, history, top_logprobs) {
        let tok = model.argmax_on_device(logits, 0)?;
        if !suppress_ids.contains(&tok) {
            return Ok((tok, None));
        }
        // Raw device argmax landed on a suppressed EOS token at position 0:
        // fall through to host D2H so `suppress_ids` can be masked out.
    }
    let vocab_size = model.vocab_size();
    // Fast host fallback for greedy BF16 when raw device argmax hit `suppress_ids`:
    // scan BF16 directly without allocating the 993 KB `Vec<f32>`.
    if !needs_sampler(params, history)
        && top_logprobs.is_none()
        && !model.logits_ptr_is_fp32(logits)
    {
        let mut bf16_buf = vec![0u8; vocab_size * 2];
        model.copy_logits_to_host(logits, &mut bf16_buf)?;
        let tok = spark_runtime::sampler::argmax_bf16_first_wins(&bf16_buf, |id| {
            !suppress_ids.contains(&id)
        })
        .unwrap_or(0);
        return Ok((tok, None));
    }
    // Read logits from device. Gemma-4 dense single-token decode produces FP32
    // logits via the FP32 lm_head + softcap path (margin between top-1 and
    // top-2 sits on a BF16 representable boundary at value 16-32, so storing
    // BF16 there flips the greedy argmax). Other paths still produce BF16
    // and need expansion. Dispatch by `logits_ptr_is_fp32`.
    let mut f32_logits: Vec<f32> = if model.logits_ptr_is_fp32(logits) {
        let mut buf = vec![0u8; vocab_size * 4];
        model.copy_logits_to_host(logits, &mut buf)?;
        buf.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    } else {
        let mut bf16_buf = vec![0u8; vocab_size * 2];
        model.copy_logits_to_host(logits, &mut bf16_buf)?;
        (0..vocab_size)
            .map(|i| {
                let lo = bf16_buf[i * 2];
                let hi = bf16_buf[i * 2 + 1];
                bf16_to_f32(lo, hi)
            })
            .collect()
    };
    let tok = sample_host_logits(&mut f32_logits, params, suppress_ids, history);
    let logprobs = top_logprobs
        .map(|k| super::logprobs::extract_logprobs_from_f32(&f32_logits, tok, k as usize));
    Ok((tok, logprobs))
}

/// Whether [`sample_token_with_logprobs`] takes its device branch for these arguments
/// (`Model::argmax_on_device`, which writes the first bytes of the scratch buffer)
/// before any host fallback. K8 item A's deferred MTP replay redoes the device write
/// after the replay whenever this returns true.
pub(super) fn sample_uses_device_argmax(
    params: &SamplingParams,
    _suppress_ids: &[u32],
    history: &[u32],
    top_logprobs: Option<u8>,
) -> bool {
    !needs_sampler(params, history) && top_logprobs.is_none()
}

/// The host half of [`sample_token`]: mask `suppress_ids`, then sample with the full
/// `params`, or take the plain argmax when nothing in them can move it.
fn sample_host_logits(
    f32_logits: &mut [f32],
    params: &SamplingParams,
    suppress_ids: &[u32],
    history: &[u32],
) -> u32 {
    // Suppress EOS tokens on first token by setting to -inf.
    for &id in suppress_ids {
        if let Some(l) = f32_logits.get_mut(id as usize) {
            *l = f32::NEG_INFINITY;
        }
    }
    if !needs_sampler(params, history) {
        // Greedy argmax over FP32, first id of a tie (the device rule).
        return spark_runtime::sampler::first_max_index(f32_logits);
    }
    // SAFETY: an f32 slice viewed as its bytes; same length in bytes, no alignment demand.
    let f32_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(f32_logits.as_ptr() as *const u8, f32_logits.len() * 4)
    };
    if params.temperature > 0.0
        && !spark_runtime::sampler::gumbel::gumbel_sampling_enabled()
        && let Some((base_seed, abs_pos)) = spark_runtime::sampler::gumbel::current_gumbel_pos()
    {
        return spark_runtime::sampler::choose_gumbel(
            f32_bytes, params, history, base_seed, abs_pos,
        );
    }
    sample_with_params_history(f32_bytes, params, history)
}

/// Sample one token from device logits with optional grammar constraint.
///
/// Like `sample_token` but also applies grammar bitmask when `grammar_state`
/// is provided. Always uses host-side sampling when grammar is active (can't
/// use GPU argmax since grammar bitmask is CPU-side).
pub fn sample_token_with_grammar(
    model: &dyn Model,
    logits: DevicePtr,
    params: &SamplingParams,
    suppress_ids: &[u32],
    history: &[u32],
    grammar_state: Option<&mut GrammarState>,
) -> Result<u32> {
    let Some(gs) = grammar_state else {
        return sample_token(model, logits, params, suppress_ids, history);
    };

    // ── Tier 3b: forced-token short-circuit (xgrammar "Coalescence") ──
    //
    // When the grammar admits exactly one legal next token (very common in
    // JSON tool-call syntax: literal `{`, key strings, `:`, `,`, closing
    // braces), xgrammar can compute the token directly without filling a
    // vocab-wide bitmask. Skip the D2H + BF16→f32 + mask + argmax loop
    // entirely — saves ~5 ms per forced position. On a typical tool-call
    // response, ~30-50% of positions are forced.
    if let Some(forced) = gs.forced_token() {
        return Ok(forced as u32);
    }

    let vocab_size = model.vocab_size();
    if model.logits_ptr_is_fp32(logits) {
        let mut buf = vec![0u8; vocab_size * 4];
        model.copy_logits_to_host(logits, &mut buf)?;
        let mut f32_logits: Vec<f32> = buf
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        if gs.fill_bitmask() {
            gs.apply_bitmask_to_logits(&mut f32_logits);
        }
        return Ok(sample_host_logits(
            &mut f32_logits,
            params,
            suppress_ids,
            history,
        ));
    }
    let mut bf16_buf = vec![0u8; vocab_size * 2];
    model.copy_logits_to_host(logits, &mut bf16_buf)?;
    gs.fill_bitmask();
    let bitmask = gs.bitmask_data();

    // ── Greedy fused fast path (temperature == 0) ──
    //
    // One pass over the BF16 row: the grammar bitmask and suppress_ids are
    // folded into the admit filter of the shared greedy pick, so no f32
    // scratch buffer is needed. A fully-masked row falls back to token 0.
    if !needs_sampler(params, history) {
        let allowed = |tok: u32| {
            let (word, bit) = ((tok / 32) as usize, tok % 32);
            word < bitmask.len()
                && (bitmask[word] & (1i32 << bit)) != 0
                && !suppress_ids.contains(&tok)
        };
        return Ok(spark_runtime::sampler::argmax_bf16_first_wins(&bf16_buf, allowed).unwrap_or(0));
    }

    // Stochastic sampling path: needs f32 logits for the sampler. Keep the
    // original BF16→f32 + apply_bitmask_to_logits pattern here; this path
    // is rare in greedy production usage and the dispatching overhead is
    // dominated by `sample_with_params` regardless.
    let mut f32_logits: Vec<f32> = (0..vocab_size)
        .map(|i| {
            let lo = bf16_buf[i * 2];
            let hi = bf16_buf[i * 2 + 1];
            bf16_to_f32(lo, hi)
        })
        .collect();
    for &id in suppress_ids {
        if (id as usize) < vocab_size {
            f32_logits[id as usize] = f32::NEG_INFINITY;
        }
    }
    // Apply grammar bitmask to logits before sampling (`sample_with_params`
    // uses deterministic tie-breaking argmax whenever temperature <= 0).
    gs.apply_bitmask_to_logits(&mut f32_logits);
    // SAFETY: `f32_logits` has `vocab_size` initialized `f32` elements (`vocab_size * 4` bytes)
    // and `u8` has alignment 1, valid for the duration of this call.
    let f32_bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(f32_logits.as_ptr() as *const u8, vocab_size * 4) };
    if params.temperature > 0.0
        && !spark_runtime::sampler::gumbel::gumbel_sampling_enabled()
        && let Some((base_seed, abs_pos)) = spark_runtime::sampler::gumbel::current_gumbel_pos()
    {
        return Ok(spark_runtime::sampler::choose_gumbel(
            f32_bytes, params, history, base_seed, abs_pos,
        ));
    }
    Ok(sample_with_params_history(f32_bytes, params, history))
}

/// [`sample_token_with_logprobs`] with optional grammar bitmask enforcement.
///
/// Delegates to [`sample_token_with_logprobs`] when no grammar applies
/// (`grammar_state` is `None`, `!apply_grammar`, terminated, or unconstrained).
pub fn sample_token_with_logprobs_and_grammar(
    model: &dyn Model,
    logits: DevicePtr,
    params: &SamplingParams,
    suppress_ids: &[u32],
    history: &[u32],
    top_logprobs: Option<u8>,
    grammar_state: Option<&mut GrammarState>,
    apply_grammar: bool,
) -> Result<(u32, Option<crate::api::TokenLogprobs>)> {
    let Some(gs) = grammar_state.and_then(|gs| {
        let active = apply_grammar && !gs.is_terminated() && gs.fill_bitmask();
        active.then_some(gs)
    }) else {
        return sample_token_with_logprobs(
            model,
            logits,
            params,
            suppress_ids,
            history,
            top_logprobs,
        );
    };
    let suppress_ids = if crate::dsv41::serving() {
        &[][..]
    } else {
        suppress_ids
    };
    let vocab_size = model.vocab_size();
    let mut f32_logits: Vec<f32> = if model.logits_ptr_is_fp32(logits) {
        let mut buf = vec![0u8; vocab_size * 4];
        model.copy_logits_to_host(logits, &mut buf)?;
        buf.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    } else {
        let mut bf16_buf = vec![0u8; vocab_size * 2];
        model.copy_logits_to_host(logits, &mut bf16_buf)?;
        bf16_buf
            .chunks_exact(2)
            .map(|c| bf16_to_f32(c[0], c[1]))
            .collect()
    };
    gs.apply_bitmask_to_logits(&mut f32_logits);
    let tok = sample_host_logits(&mut f32_logits, params, suppress_ids, history);
    let logprobs = top_logprobs
        .map(|k| super::logprobs::extract_logprobs_from_f32(&f32_logits, tok, k as usize));
    Ok((tok, logprobs))
}

/// The stream event for a first generated token (with its logprobs when the
/// request asked for them).
pub(super) fn first_token_event(
    tok: u32,
    logprobs: &Option<crate::api::TokenLogprobs>,
    suppress_stream: bool,
) -> crate::api::StreamEvent {
    committed_token_event(tok, logprobs, suppress_stream)
}

/// Wire event for one committed completion token. Suppressed spontaneous
/// reasoning still advances the continuous-usage stream without publishing
/// its token id or text.
pub(super) fn committed_token_event(
    tok: u32,
    logprobs: &Option<crate::api::TokenLogprobs>,
    suppress_stream: bool,
) -> crate::api::StreamEvent {
    if suppress_stream {
        return match logprobs {
            Some(lp) => crate::api::StreamEvent::CompletionProgressWithLogprobs(lp.clone()),
            None => crate::api::StreamEvent::CompletionProgress,
        };
    }
    match logprobs {
        Some(lp) => crate::api::StreamEvent::TokenWithLogprobs(tok, lp.clone()),
        None => crate::api::StreamEvent::Token(tok),
    }
}

#[cfg(test)]
#[path = "sample_step_tests.rs"]
mod tests;
