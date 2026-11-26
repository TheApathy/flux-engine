// SPDX-License-Identifier: AGPL-3.0-only

//! process_decode_logits per-sequence helper (extracted to keep parent file ≤500 LoC).

use super::*;

#[path = "decode_logits_diagnostic.rs"]
mod diagnostic;

/// Process logits for a single active sequence: dequant, adjust, sample, return token + optional logprobs.
///
/// This is the single source of truth for the per-token thinking-span logit
/// interventions (F1 reflection suppression, efficiency wave, F2 confidence
/// early stop, tool-call mask, forced `</think>` injection). Besides the
/// plain decode path (`process_decode_logits`), the `ATLAS_THINK_SPEC=1`
/// accept filter (`think_spec_accept`) feeds individual verify-logits rows
/// through this same function so the speculative and plain paths cannot
/// drift — do NOT fork the intervention logic out of here.
#[allow(clippy::too_many_arguments)]
pub fn process_seq_logits(
    a: &mut ActiveSeq,
    buf: &[u8],
    i: usize,
    vocab_size: usize,
    elem_bytes: usize,
    logits_fp32: bool,
    think_end_token: Option<u32>,
    _think_start_token: Option<u32>,
    tool_call_start_token: Option<u32>,
    _tool_call_end_token: Option<u32>,
    reflection_suppress_ids: &[u32],
    adaptive_sampling: bool,
) -> (u32, Option<crate::api::TokenLogprobs>) {
    let slice = &buf[i * vocab_size * elem_bytes..(i + 1) * vocab_size * elem_bytes];
    log_logits_fnv(a.session_hash, a.output_tokens.len(), slice);
    let f32_logits: Vec<f32> = if logits_fp32 {
        // Direct FP32: 4 bytes/element little-endian.
        (0..vocab_size)
            .map(|j| {
                let off = j * 4;
                f32::from_le_bytes([slice[off], slice[off + 1], slice[off + 2], slice[off + 3]])
            })
            .collect()
    } else {
        // BF16 → FP32 expansion.
        (0..vocab_size)
            .map(|j| {
                let lo = slice[j * 2];
                let hi = slice[j * 2 + 1];
                bf16_to_f32(lo, hi)
            })
            .collect()
    };
    let raw_diagnostic = logits_fnv_requires_row().then(|| diagnostic::summarize(&f32_logits));
    process_seq_f32_logits(
        a,
        f32_logits,
        raw_diagnostic,
        None,
        think_end_token,
        tool_call_start_token,
        reflection_suppress_ids,
        adaptive_sampling,
    )
}

/// [`process_seq_logits`] for a row built by the device top-k route
/// (`ATLAS_SAMPLER_DEVICE_TOPK=1`, see `device_topk_route`): candidates at
/// their raw values, -inf elsewhere. The route only admits rows whose logits
/// receipts are off, so there is no raw row to hash or summarize.
#[allow(clippy::too_many_arguments)]
pub(super) fn process_seq_sparse_logits(
    a: &mut ActiveSeq,
    f32_logits: Vec<f32>,
    f2_decision: Option<bool>,
    think_end_token: Option<u32>,
    tool_call_start_token: Option<u32>,
    reflection_suppress_ids: &[u32],
    adaptive_sampling: bool,
) -> (u32, Option<crate::api::TokenLogprobs>) {
    process_seq_f32_logits(
        a,
        f32_logits,
        None,
        f2_decision,
        think_end_token,
        tool_call_start_token,
        reflection_suppress_ids,
        adaptive_sampling,
    )
}

/// The plain path's edits that precede F2, in application order: F1
/// reflection suppression (`-= 10.0` per id while thinking, not for
/// deepseek_v41) and the min-thinking floor (`</think>` = -inf). Ids at or
/// past `vocab` are skipped, as the plain path skipped them.
pub(super) fn pre_f2_edits(
    a: &ActiveSeq,
    think_end_token: Option<u32>,
    reflection_suppress_ids: &[u32],
    vocab: usize,
) -> Vec<(u32, spark_runtime::sampler::f2_certify::PreF2Edit)> {
    use spark_runtime::sampler::f2_certify::PreF2Edit;
    let mut edits = Vec::new();
    if a.inside_thinking && !crate::dsv41::serving() {
        for &rid in reflection_suppress_ids {
            if (rid as usize) < vocab {
                edits.push((rid, PreF2Edit::Sub(10.0)));
            }
        }
    }
    if a.inside_thinking && !a.force_end_thinking && !a.think_ended {
        let floor = crate::scheduler::helpers::min_thinking_floor();
        if floor > 0
            && a.thinking_tokens < floor
            && let Some(end) = think_end_token
            && (end as usize) < vocab
        {
            edits.push((end, PreF2Edit::NegInf));
        }
    }
    edits
}

/// The interventions + sampler half of [`process_seq_logits`], over an
/// already-expanded f32 row. Every route shares this body (SSOT).
#[allow(clippy::too_many_arguments)]
fn process_seq_f32_logits(
    a: &mut ActiveSeq,
    mut f32_logits: Vec<f32>,
    raw_diagnostic: Option<diagnostic::RawSummary>,
    f2_decision: Option<bool>,
    think_end_token: Option<u32>,
    tool_call_start_token: Option<u32>,
    reflection_suppress_ids: &[u32],
    adaptive_sampling: bool,
) -> (u32, Option<crate::api::TokenLogprobs>) {
    // Thinking nudge (FLUX_THINK_NUDGE, default off): while a nudge is being
    // injected, this position commits the next nudge token instead of a
    // sample. Returned before every thinking intervention below, so the
    // injected text neither feeds F2/the difficulty probe nor gets shaped.
    // Logprobs report the model's own distribution at this position.
    // The thinking-budget nudge (default on) is forced the same way, ahead
    // of the budget's forced `</think>` below.
    let hard_override = super::confidence::compute_think_defer_hard_override(
        a.thinking_tokens,
        a.thinking_budget,
        a.output_tokens.len(),
        a.max_output_tokens,
    );
    if a.inside_thinking
        && !a.force_end_thinking
        && a.thinking_budget.is_none()
        && a.max_output_tokens > 0
        && a.output_tokens.len()
            >= a.max_output_tokens
                .saturating_sub(super::confidence::min_answer_runway(a.max_output_tokens))
    {
        a.force_end_thinking = true;
    }
    let close_allowed_now = super::think_budget_nudge::close_allowed(a)
        || (a.inside_thinking
            && super::confidence::should_inject_think_end(
                a.force_end_thinking,
                a.in_code_fence,
                hard_override,
            ));
    let runway_nudge_forced = a
        .budget_nudge
        .as_ref()
        .filter(|_| {
            close_allowed_now && a.output_tokens.len().saturating_add(1) < a.max_output_tokens
        })
        .and_then(|n| n.pending_token());
    let forced = super::think_budget_nudge::forced_token(a)
        .or(runway_nudge_forced)
        .or_else(|| super::think_nudge::forced_token(a));
    if let Some(forced) = forced {
        log_logits_selection(a, raw_diagnostic.as_ref(), forced, None, false);
        let logprobs = a
            .top_logprobs
            .map(|k| extract_logprobs_from_f32(&f32_logits, forced, k as usize));
        return (forced, logprobs);
    }

    // F1 (reflection suppression) and the min-thinking floor: the edits that
    // precede F2, from `pre_f2_edits` (SSOT: the device top-k route hands the
    // same list to the certified-F2 kernel).
    for (id, edit) in pre_f2_edits(
        a,
        think_end_token,
        reflection_suppress_ids,
        f32_logits.len(),
    ) {
        edit.apply(&mut f32_logits[id as usize]);
    }

    // ── Decoding-efficiency wave (env-gated, default-OFF, OUTPUT-SHAPING) ──
    // Lever 1 (hesitation penalty) + Lever 2 (soft </think> exit bias) shape
    // the thinking-span logits; Lever 3 (adaptive budget) probes difficulty
    // over the first ~48 thinking tokens and rescales `thinking_budget`.
    //
    // Fully inert when no ATLAS_* var is set: `is_active()` is false, the
    // shaping apply is a guarded no-op, and the probe never observes — so the
    // committed token stream (and counting-eval md5) is byte-identical.
    if a.inside_thinking {
        let cfg = crate::scheduler::think_efficiency_config();
        if cfg.is_active() {
            // Lever 3: adaptive thinking budget. Observe this token's top-1
            // confidence into the difficulty probe; once the probe window is
            // full, commit a difficulty-scaled budget (once). Confidence is the
            // only difficulty signal available *during* thinking — speculative
            // decode (and thus the drafter accept-rate) is bypassed while
            // inside_thinking by the MTP gate in scheduler::run, unless
            // ATLAS_THINK_SPEC=1, in which case the think-spec accept filter
            // slow-paths every position through this function while the wave
            // is active, so the probe observes identically. High mean
            // confidence ⇒ easy reasoning ⇒ shorter budget.
            if cfg.adaptive_think {
                let top1 = crate::scheduler::top1_confidence(&f32_logits);
                a.difficulty_probe.observe(top1);
                if let Some(base) = a.thinking_budget
                    && let Some(scaled) = a.difficulty_probe.commit(base)
                {
                    if scaled != base {
                        tracing::info!(
                            seq_thinking_tokens = a.thinking_tokens,
                            mean_confidence = a.difficulty_probe.mean_confidence().unwrap_or(0.0),
                            base_budget = base,
                            adaptive_budget = scaled,
                            "ADAPTIVE_THINK: rescaled thinking budget from difficulty probe"
                        );
                    }
                    a.thinking_budget = Some(scaled);
                }
            }

            // Levers 1+2: hesitation penalty + progressive </think> exit bias.
            let touched = crate::scheduler::apply_think_logit_shaping(
                &mut f32_logits,
                cfg,
                think_end_token,
                a.thinking_tokens,
                a.thinking_budget,
            );
            if touched > 0 && tracing::enabled!(tracing::Level::TRACE) {
                tracing::trace!(
                    thinking_tokens = a.thinking_tokens,
                    touched,
                    "THINK_EFFICIENCY: shaped {touched} logit slots"
                );
            }
        }
    }

    // F2: Confidence-based early stop during thinking.
    // When top-1 prob >= 0.95 for 30 consecutive tokens, force </think>.
    // Only kicks in after 400 thinking tokens — the model needs room to
    // plan (numbered lists, step-by-step reasoning have high per-token
    // confidence but are NOT signs the model is done thinking).
    // Previous thresholds (200 tokens, 10 consecutive) were too aggressive
    // and caused premature thinking termination in agentic coding sessions.
    //
    // Code-fence handling: a ``` block inside the reasoning is even
    // MORE confident than prose (Python/JSON syntax is near-
    // deterministic: `def`/`(`/`:`/indent/`return`), so 30 consecutive
    // ≥0.95 tokens trips trivially while the model is *productively*
    // drafting code. We still ARM the brake here (a model can ramble in
    // code forever — it must eventually stop), but the forced </think>
    // injection is DEFERRED until the fence closes (see
    // `should_inject_think_end` at the injection site below), so the
    // boundary lands cleanly right after the code block instead of
    // splitting a statement. The token-period THINK_LOOP watchdog
    // (decode_logits_step) also stays active in fences.
    if a.inside_thinking
        && !a.force_end_thinking
        && a.thinking_tokens >= CONFIDENCE_EARLY_STOP_MIN_THINKING
        && crate::scheduler::helpers::watchdog_params().confidence_early_stop
    {
        // `f2_decision` is the device route's certified (or exact host) answer
        // for this same edited row; otherwise compute it here.
        let confident = f2_decision
            .unwrap_or_else(|| spark_runtime::sampler::f2_certify::host_f2_confident(&f32_logits));
        let (run, force_end) = confidence_run_step(confident, a.consecutive_confident);
        a.consecutive_confident = run;
        if force_end {
            a.force_end_thinking = true;
            tracing::info!(
                "Confidence early stop armed: top-1 prob >= 0.95 for {} tokens (after {} thinking tokens){}",
                crate::scheduler::helpers::watchdog_params().confidence_run_length,
                a.thinking_tokens,
                if a.in_code_fence {
                    " — deferred until ``` fence closes"
                } else {
                    ""
                }
            );
        }
    }

    // After thinking is done, suppress the </think> token to prevent
    // degenerate loops where the model generates hundreds of </think>.
    // Not for deepseek_v41: the Python engine masks neither think token.
    if a.think_ended && !crate::dsv41::serving() {
        if let Some(end_tok) = think_end_token {
            let end_idx = end_tok as usize;
            if end_idx < f32_logits.len() {
                f32_logits[end_idx] = f32::NEG_INFINITY;
            }
        }
        // F9 (2026-04-26): symmetric mask for the START token.
        // Once `think_ended` is true (watchdog forced close OR
        // model emitted </think> naturally), the model must not
        // re-enter thinking in the same response. Without this
        // mask, the spontaneous-<think> re-entry path at the
        // emit site flips `inside_thinking=true` again on any
        // sampled <think>, and the watchdog fires again ~8s
        // later — observed three rapid re-entries on
        // 2026-04-26 fix29 logs. arXiv evidence: s1
        // (2501.19393), DeepSeek-R1, Qwen3 (2505.09388),
        // Production Repetition (2512.04419) all mask the
        // open token after first close. Chain-of-Draft
        // (2502.18600) ablates penalty stacking (12% drop) vs
        // hard masking (94% drop) — masking dominates.
        if let Some(start_tok) = a.think_start_token {
            let start_idx = start_tok as usize;
            if start_idx < f32_logits.len() {
                f32_logits[start_idx] = f32::NEG_INFINITY;
            }
        }
    }

    // Suppress <tool_call> during thinking (prevents KV cache contamination
    // from think-leak bug) AND when tool call loop detected (≥4 identical
    // calls — see `api::chat::loop_detect`). For the loop case, use a STRONG NEGATIVE
    // BIAS (−12.0) instead of `-inf` so the model can still escape if its
    // evidence for a tool call is overwhelming (e.g. user explicitly says
    // "actually run the tests"). For thinking, hard-mask remains: tool
    // calls inside <think> are unparsable per the (canonical) qwen3_coder
    // dialect, so they must be physically blocked.
    if a.inside_thinking {
        if let Some(tc_start) = tool_call_start_token {
            let idx = tc_start as usize;
            if idx < f32_logits.len() {
                f32_logits[idx] = f32::NEG_INFINITY;
            }
        }
    } else if a.suppress_tool_call
        && let Some(tc_start) = tool_call_start_token
    {
        let idx = tc_start as usize;
        if idx < f32_logits.len() {
            f32_logits[idx] -= 12.0;
        }
    }

    // Force </think> when budget exhausted OR confidence early stop
    // triggered — but DEFER while inside a ``` code fence so the
    // injection never splits a code statement (2026-05-17 thinkbrake
    // fix). The fence closes within a bounded number of tokens, then
    // this fires cleanly at the block boundary.
    // Bound the in-fence deferral: a model that writes its whole answer
    // as a code block inside <think> never closes the fence, so an
    // unbounded defer traps the deliverable in reasoning. Past
    // THINK_DEFER_BUDGET_FACTOR× the budget (or the absolute ceiling
    // when budget is None), inject </think> even mid-fence.
    let post_f2_hard_override = super::confidence::compute_think_defer_hard_override(
        a.thinking_tokens,
        a.thinking_budget,
        a.output_tokens.len(),
        a.max_output_tokens,
    );
    let post_f2_close_allowed = super::think_budget_nudge::close_allowed(a)
        || (a.inside_thinking
            && super::confidence::should_inject_think_end(
                a.force_end_thinking,
                a.in_code_fence,
                post_f2_hard_override,
            ));
    if post_f2_close_allowed && let Some(end_tok) = think_end_token {
        let end_idx = end_tok as usize;
        if end_idx < f32_logits.len() {
            for logit in f32_logits.iter_mut() {
                *logit = f32::NEG_INFINITY;
            }
            f32_logits[end_idx] = 0.0;
        }
    }

    // Change 3b: one-shot pin-to-tool-call-start.
    // When the previous token was `</think>` AND the request
    // requires a tool call AND no tool-call has been opened yet,
    // mask all logits to -inf except `tool_call_start_token`.
    // This prevents architectures like MiniMax M2 (which always
    // thinks via the chat template) from wandering into prose
    // after `</think>` instead of emitting the structured tool
    // call. Models that don't have `require_tool_call` set
    // (i.e. the request didn't pass tools) skip this entirely.
    if a.think_just_ended
        && a.require_tool_call
        && !a.tool_call_opened
        && !a.inside_thinking
        && let Some(start_tok) = tool_call_start_token
    {
        let idx = start_tok as usize;
        if idx < f32_logits.len() {
            for logit in f32_logits.iter_mut() {
                *logit = f32::NEG_INFINITY;
            }
            f32_logits[idx] = 0.0;
            tracing::debug!("Forced tool_call_start_token after </think> (require_tool_call set)");
        }
    }

    // F70 (2026-04-29, attempted): canonical-opener anchor
    // bias was REVERTED. xgrammar's TagDispatch is non-anchored
    // (verified by
    // `grammar.rs::test_minimax_xml_grammar_masks_trigger_breaking_multibyte_token`),
    // and a flat +2.5 logit boost on `tool_call_start_token`
    // pushes the model into the tool body too aggressively —
    // observed live: 1-tool prompts produce
    // `tool_calls[0].function.arguments = {"command":""}`
    // because the model rushes through the envelope with no
    // parameter values. The proper fix is byte-level partial
    // trigger anchoring (mask trigger-breaker tokens only when
    // a partial-match suffix is actually present in recent
    // output) but that's a follow-up — for now we accept the
    // "model occasionally drifts on stressed prompts"
    // limitation and rely on F26/F2 to terminate the response
    // cleanly when it happens.

    // ── Forced-token fast-path (xgrammar Tier 3b, Coalescence) ──
    // When the active tool-call grammar admits exactly one legal next
    // token, the model sample is redundant: the token is determined.
    // `forced_token()` returns `Some(id)` ONLY when the authoritative
    // next-token bitmask has a single set bit — so emitting `id`
    // directly is bit-identical to sampling from an all-but-`id`-masked
    // logit vector (every other token would be `-inf`). We skip the
    // O(vocab) bitmask fill *and* the O(vocab) CPU sampling scan for
    // these positions; this is the big win for structured tool-call
    // scaffolding (literal `<function=`, `</parameter>`, JSON
    // punctuation emit with no sampling work).
    //
    // GUARDS — the fast-path fires only when ALL hold:
    //  * not inside `<think>` — thinking is unconstrained (mirrors the
    //    bitmask-skip below; thinking tokens never advance the grammar).
    //  * the request actually has an active grammar (`grammar_state`).
    //  * the kill-switch is off (default; set `ATLAS_DISABLE_FORCED_TOKEN=1` to disable).
    //  * `top_logprobs` is NOT requested — logprobs are extracted from
    //    the model's logit distribution; the fast-path never builds it.
    //    Falling through to the normal masked-sample path keeps logprobs
    //    byte-identical (the all-but-one mask makes the sample return
    //    the same forced token anyway, so output is unchanged).
    //
    // The returned forced token still flows through the SAME caller
    // accounting as a sampled token — `decode_logits_step` pushes it to
    // `output_tokens`, calls `gs.accept_token`, runs stop-token / EOS /
    // streaming handling — so all downstream state is identical.
    if !a.inside_thinking
        && a.top_logprobs.is_none()
        && crate::scheduler::helpers::forced_token_fastpath_enabled()
        && let Some(ref mut gs) = a.grammar_state
        && let Some(forced) = gs.forced_token()
    {
        // `forced` is the sole grammar-legal token; `forced_token`
        // returns only non-negative vocab ids (it reads them off the
        // packed bitmask). Emit directly — no mask fill, no sample.
        log_logits_selection(a, raw_diagnostic.as_ref(), forced as u32, None, true);
        return (forced as u32, None);
    }

    // Apply grammar bitmask BEFORE sampling — but NOT during
    // `<think>`…`</think>`. Thinking is free-form reasoning that
    // is stripped from the final API response, so forcing it
    // through a JSON-tool-call grammar produces garbage
    // punctuation streams (observed with opencode: the assistant
    // thinking channel filled with `!.,),,,***` before the
    // model recovered after `</think>`).
    if !a.inside_thinking
        && let Some(ref mut gs) = a.grammar_state
        && gs.fill_bitmask()
    {
        gs.apply_bitmask_to_logits(&mut f32_logits);
    }

    // deepseek_v41: the Python engine's cycle breaker / no-repeat-ngram bans
    // (a no-op for every other model). Python applies them after the
    // penalties; a -inf ban survives every penalty and bias, so applying it
    // before the sampler's penalty stage is the same distribution.
    crate::dsv41::repetition::apply(&mut f32_logits, &a.output_tokens);

    // F72 (byte-level partial-trigger anchor) was removed — see
    // F73 / fix42. The sampler-side anchor hung the server in
    // production despite passing isolated unit tests; the
    // model's broken-envelope case is now recovered at the
    // streaming-sanitizer + parser layer (F73 + F71). The
    // xgrammar non-anchored TagDispatch limitation is pinned
    // by `grammar.rs::test_minimax_xml_grammar_masks_trigger_breaking_multibyte_token`
    // for documentation only.

    // ── Adaptive sampling: update zone, observe entropy, check greedy gate ──
    // Disabled by default (--adaptive-sampling flag). Each call scans the
    // full vocab (262k) on CPU: entropy O(V) exp+log, greedy gate O(V) exp.
    // Cost: ~300-400µs per token → 2-3x throughput regression when enabled.
    let greedy_gate = if adaptive_sampling {
        a.adaptive.update_zone(
            a.tool_call_opened,
            a.inside_thinking,
            a.grammar_state.is_some(),
        );
        a.adaptive.observe_entropy(&f32_logits);
        a.adaptive.update_lz_ratio(&a.output_tokens);
        a.adaptive.should_use_greedy(&f32_logits)
    } else {
        false
    };
    let effective_temp = if adaptive_sampling {
        a.adaptive.effective_temperature()
    } else {
        a.temperature
    };

    // Unified sampling path: stochastic OR greedy (temp==0 or
    // adaptive greedy_gate) both go through
    // `sample_with_params_history`. The function applies all
    // configured penalties (repetition / presence / frequency /
    // LZ / DRY) and logit_bias BEFORE the temperature decision,
    // so greedy argmax respects MODEL.toml-configured penalties
    // — matching HF Transformers / vLLM / llama.cpp behavior.
    //
    // The earlier "Pure greedy argmax — NO penalties" bypass
    // here was the load-bearing bug for Gemma-4-31B's greedy
    // fib failure: `MODEL.toml` configures rep_penalty=1.1 but
    // the bypass dropped it. After this change, the configured
    // penalty applies at temp=0.
    // SAFETY: `f32_logits` holds `f32_logits.len()` initialized `f32` values
    // (`size_of_val` bytes) and `u8` has alignment 1, valid across
    // `sample_with_params_history` / `choose_gumbel`.
    let f32_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            f32_logits.as_ptr() as *const u8,
            std::mem::size_of_val(f32_logits.as_slice()),
        )
    };
    // Force temp=0 for greedy_gate path (adaptive override) so
    // sample_with_params_seeded takes the post-penalty argmax
    // branch instead of running the full stochastic pipeline.
    let sampling_temp = if greedy_gate { 0.0 } else { effective_temp };
    let _gumbel_guard = (super::stochastic_verify_walk::spec_verify_policy()
        == super::stochastic_verify_walk::SpecVerifyPolicy::Gumbel
        && sampling_temp > 0.0)
        .then(|| {
            let seed = super::stochastic_verify_walk::resolve_seq_gumbel_seed(a);
            spark_runtime::sampler::gumbel::GumbelPosGuard::new(
                seed,
                super::stochastic_verify_walk::seq_abs_pos(a),
            )
        });
    let params = a.sampling_params(sampling_temp);
    let sampled = if sampling_temp > 0.0
        && !spark_runtime::sampler::gumbel::gumbel_sampling_enabled()
        && let Some((base_seed, abs_pos)) = spark_runtime::sampler::gumbel::current_gumbel_pos()
    {
        spark_runtime::sampler::choose_gumbel(
            f32_bytes,
            &params,
            &a.output_tokens,
            base_seed,
            abs_pos,
        )
    } else {
        sample_with_params_history(f32_bytes, &params, &a.output_tokens)
    };
    log_logits_selection(
        a,
        raw_diagnostic.as_ref(),
        sampled,
        Some(sampling_temp),
        false,
    );

    // Extract top-K logprobs from f32_logits if requested.
    let logprobs = a
        .top_logprobs
        .map(|k| extract_logprobs_from_f32(&f32_logits, sampled, k as usize));
    (sampled, logprobs)
}

fn log_logits_selection(
    a: &ActiveSeq,
    raw: Option<&diagnostic::RawSummary>,
    selected: u32,
    sampling_temperature: Option<f32>,
    grammar_forced: bool,
) {
    let Some(raw) = raw else {
        return;
    };
    tracing::info!(
        session = format_args!("{:016x}", a.session_hash),
        idx = a.output_tokens.len(),
        raw_top1_id = ?raw.top1.map(|candidate| candidate.token),
        raw_top1_value = ?raw.top1.map(|candidate| candidate.value),
        raw_top1_bits = ?raw.top1.map(|candidate| candidate.value.to_bits()),
        raw_top2_id = ?raw.top2.map(|candidate| candidate.token),
        raw_top2_value = ?raw.top2.map(|candidate| candidate.value),
        raw_top2_bits = ?raw.top2.map(|candidate| candidate.value.to_bits()),
        raw_finite_margin = ?raw.finite_margin(),
        finite_count = raw.finite_count,
        nonfinite_count = raw.nan_count + raw.positive_infinity_count + raw.negative_infinity_count,
        nan_count = raw.nan_count,
        positive_infinity_count = raw.positive_infinity_count,
        negative_infinity_count = raw.negative_infinity_count,
        selected,
        request_temperature = a.temperature,
        sampling_temperature = ?sampling_temperature,
        inside_thinking = a.inside_thinking,
        think_ended = a.think_ended,
        think_just_ended = a.think_just_ended,
        force_end_thinking = a.force_end_thinking,
        ignore_eos = a.ignore_eos,
        grammar_active = a.grammar_state.is_some(),
        grammar_forced,
        require_tool_call = a.require_tool_call,
        suppress_tool_call = a.suppress_tool_call,
        "LOGITS_FNV_DETAIL"
    );
}

#[derive(Default)]
struct LogitsReceiptBudget {
    request_open: bool,
    completed_requests: usize,
}

static LOGITS_FNV_ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
static LOGITS_FNV_LIMIT: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
static LOGITS_FNV_BUDGET: std::sync::OnceLock<parking_lot::Mutex<LogitsReceiptBudget>> =
    std::sync::OnceLock::new();
static LOGITS_FNV_EXHAUSTED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

fn logits_fnv_enabled() -> bool {
    *LOGITS_FNV_ON.get_or_init(|| std::env::var("ATLAS_LOGITS_FNV").ok().as_deref() == Some("1"))
}

fn logits_fnv_limit() -> usize {
    *LOGITS_FNV_LIMIT.get_or_init(|| {
        let Ok(raw) = std::env::var("ATLAS_LOGITS_FNV_REQUEST_LIMIT") else {
            return usize::MAX;
        };
        match raw
            .parse::<usize>()
            .ok()
            .filter(|value| *value > 0 && value.to_string() == raw)
        {
            Some(value) => value,
            None => {
                tracing::error!(
                    "ATLAS_LOGITS_FNV_REQUEST_LIMIT must be a canonical positive usize; disabling logits receipts"
                );
                0
            }
        }
    })
}

/// Whether the active qualification receipt still needs the next logits row.
///
/// Speculative greedy accept may otherwise commit the verifier's already
/// computed raw argmax without copying its full logits row to the host. Keep
/// receipt mode observable by forcing that copy while its request budget is
/// active. The exhausted flag and completed-request count are monotonic, so a
/// `false` result cannot become `true` later in the same process.
pub(super) fn logits_fnv_requires_row() -> bool {
    if LOGITS_FNV_EXHAUSTED.load(std::sync::atomic::Ordering::Acquire) || !logits_fnv_enabled() {
        return false;
    }
    let limit = logits_fnv_limit();
    if limit == 0 {
        LOGITS_FNV_EXHAUSTED.store(true, std::sync::atomic::Ordering::Release);
        return false;
    }
    LOGITS_FNV_BUDGET
        .get_or_init(|| parking_lot::Mutex::new(LogitsReceiptBudget::default()))
        .lock()
        .completed_requests
        < limit
}

/// Close the qualification receipt for the just-finished C=1 request. This
/// explicit lifecycle edge avoids guessing request boundaries from repeated
/// output indices and turns the diagnostic into one atomic load after its
/// configured endpoint-request budget is exhausted.
pub(super) fn finish_logits_fnv_request() {
    if LOGITS_FNV_EXHAUSTED.load(std::sync::atomic::Ordering::Acquire) || !logits_fnv_enabled() {
        return;
    }
    let limit = logits_fnv_limit();
    if limit == 0 {
        LOGITS_FNV_EXHAUSTED.store(true, std::sync::atomic::Ordering::Release);
        return;
    }
    let mut budget = LOGITS_FNV_BUDGET
        .get_or_init(|| parking_lot::Mutex::new(LogitsReceiptBudget::default()))
        .lock();
    if !budget.request_open {
        return;
    }
    budget.request_open = false;
    budget.completed_requests = budget.completed_requests.saturating_add(1);
    if budget.completed_requests >= limit {
        LOGITS_FNV_EXHAUSTED.store(true, std::sync::atomic::Ordering::Release);
    }
}

/// `ATLAS_LOGITS_FNV=1`: log FNV-1a and SHA-256 hashes of every complete
/// logits row the sampler sees. The prompt-derived session hash and output
/// index bind rows under concurrent scheduling without relying on log order.
/// SHA-256 is the qualification comparison; FNV remains a quick divergence
/// locator for existing diagnostics.
///
/// `ATLAS_LOGITS_FNV_REQUEST_LIMIT=N` restricts this diagnostic to the first N
/// completed requests. A concurrent qualifier must leave it unset because
/// request completions can interleave.
pub(super) fn log_logits_fnv(session_hash: u64, output_index: usize, row: &[u8]) {
    if LOGITS_FNV_EXHAUSTED.load(std::sync::atomic::Ordering::Acquire) || !logits_fnv_enabled() {
        return;
    }
    let limit = logits_fnv_limit();
    if limit == 0 {
        LOGITS_FNV_EXHAUSTED.store(true, std::sync::atomic::Ordering::Release);
        return;
    }
    let mut budget = LOGITS_FNV_BUDGET
        .get_or_init(|| parking_lot::Mutex::new(LogitsReceiptBudget::default()))
        .lock();
    if !budget.request_open {
        budget.request_open = true;
    }
    let active = budget.completed_requests < limit;
    drop(budget);
    if !active {
        return;
    }
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for &byte in row {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0100_0000_01b3);
    }
    let sha256 = ring::digest::digest(&ring::digest::SHA256, row);
    let mut sha256_hex = String::with_capacity(64);
    for byte in sha256.as_ref() {
        use std::fmt::Write as _;
        let _ = write!(sha256_hex, "{byte:02x}");
    }
    tracing::info!(
        "LOGITS_FNV session={session_hash:016x} idx={output_index} fnv={hash:016x} sha256={sha256_hex}"
    );
}

#[cfg(test)]
mod gumbel_coordinate_parity_tests {
    use super::*;
    use crate::scheduler::spec_policy_accept::VerifyLogitsFormat;
    use crate::scheduler::stochastic_verify::DraftDistribution;
    use crate::scheduler::stochastic_verify_walk::{
        SpecVerifyPolicy, resolve_seq_gumbel_seed, seq_abs_pos,
        stochastic_content_accept_with_policy,
    };
    use spark_runtime::sampler::gumbel::GumbelPosGuard;

    fn bf16_bytes_for_pos(abs_pos: u64, vocab: usize) -> Vec<u8> {
        let mut out = Vec::with_capacity(vocab * 2);
        for id in 0..vocab {
            let mixed = spark_runtime::sampler::gumbel::splitmix64_mix(
                (abs_pos.wrapping_mul(0x517C_C1B7_2722_0A95)) ^ (id as u64),
            );
            // Smooth finite logits in [-1.5, 2.5] so multiple candidates compete at T=0.85.
            let unit = ((mixed >> 40) as f32) * (1.0 / ((1u32 << 24) as f32));
            let val = -1.5 + 4.0 * unit;
            let bits = (val.to_bits() >> 16) as u16;
            out.extend_from_slice(&bits.to_le_bytes());
        }
        out
    }

    fn setup_seq(prompt: &[u32], seed: u64) -> ActiveSeq {
        let mut a = crate::scheduler::test_fixtures::active_seq(
            crate::scheduler::ResponseSink::Blocking(None),
            0,
        );
        a.seq.prompt_len = prompt.len();
        a.seq.tokens = prompt.to_vec();
        a.seq.seq_len = prompt.len();
        a.output_tokens.clear();
        a.max_output_tokens = 64;
        a.remaining = 64;
        a.eos_tokens = vec![999_999];
        a.temperature = 0.85;
        a.top_k = 6;
        a.top_p = 0.92;
        a.min_p = 0.02;
        a.repetition_penalty = 1.15;
        a.presence_penalty = 0.35;
        a.frequency_penalty = 0.1;
        a.dry_multiplier = 0.0;
        a.lz_penalty = 0.0;
        a.think_ended = true;
        a.inside_thinking = false;
        a.seed = Some(seed);
        a
    }

    #[test]
    fn prompt_seed_materializes_identically_for_serial_and_speculative_requests() {
        let prompt = [101u32, 202, 303, 404, 505];
        let mut req_seed = None;
        super::super::sample_step::materialize_prompt_sampling_seed_with(
            0.8,
            &mut req_seed,
            &prompt,
            true,
        );
        super::super::sample_step::materialize_effective_sampling_seed(0.8, &mut req_seed)
            .expect("prompt seed already materialized without needing entropy");

        let mut spec_seq = setup_seq(&prompt, 0);
        spec_seq.seed = None;
        assert_eq!(req_seed, Some(resolve_seq_gumbel_seed(&spec_seq)));
    }

    #[test]
    fn serial_k1_and_mtp_gumbel_verify_are_bit_identical_across_partial_accepts() {
        let vocab = 16usize;
        let prompt = [11u32, 22, 33, 44, 55, 66, 77]; // prompt_len = 7 (non-zero!)
        let seed = spark_runtime::sampler::seed_from_prompt_tokens(&prompt);

        // ── Arm 1: Plain serial K=1 prefill first-token + 14 decode steps (`draft=false`) ──
        let mut serial = setup_seq(&prompt, seed);
        let first_row = bf16_bytes_for_pos(prompt.len() as u64, vocab);
        let first_tok = {
            let mut first_params = serial.sampling_params(serial.temperature);
            first_params.seed = Some(seed);
            let _guard = GumbelPosGuard::new(seed, prompt.len() as u64);
            let mut f32_row: Vec<f32> = first_row
                .chunks_exact(2)
                .map(|b| bf16_to_f32(b[0], b[1]))
                .collect();
            for &eos in &serial.eos_tokens {
                if (eos as usize) < vocab {
                    f32_row[eos as usize] = f32::NEG_INFINITY;
                }
            }
            // SAFETY: `f32_row` holds `vocab` initialized `f32` values (`vocab * 4` bytes).
            let bytes =
                unsafe { std::slice::from_raw_parts(f32_row.as_ptr() as *const u8, vocab * 4) };
            let (g_seed, g_pos) = spark_runtime::sampler::gumbel::current_gumbel_pos()
                .expect("prefill first-token guard bound");
            spark_runtime::sampler::choose_gumbel(bytes, &first_params, &[], g_seed, g_pos)
        };
        serial.output_tokens.push(first_tok);
        serial.last_token = first_tok;

        for _step in 1..15 {
            let abs_pos = seq_abs_pos(&serial);
            let row = bf16_bytes_for_pos(abs_pos, vocab);
            let (tok, lp) = {
                let _guard = GumbelPosGuard::new(resolve_seq_gumbel_seed(&serial), abs_pos);
                process_seq_logits(
                    &mut serial,
                    &row,
                    0,
                    vocab,
                    2,
                    false,
                    None,
                    None,
                    None,
                    None,
                    &[],
                    false,
                )
            };
            emit_token(&mut serial, tok, lp);
            serial.last_token = tok;
        }
        assert_eq!(serial.output_tokens.len(), 15);

        // ── Arm 2: Speculative `step_mtp` Gumbel walk across partial accepts 0, 2, 4, 1, 2 ──
        let mut spec = setup_seq(&prompt, seed);
        spec.output_tokens.push(first_tok);
        spec.last_token = first_tok;

        let wrong_than = |truth: u32| (truth + 1) % (vocab as u32);
        // Scripted (num_drafts, desired_accepts) summing to 14 post-prefill tokens:
        //   0 accepted -> emits 1 (idx 1)
        //   2 accepted -> emits 3 (idxs 2, 3, 4)
        //   4 accepted -> emits 5 (idxs 5, 6, 7, 8, 9)
        //   1 accepted -> emits 2 (idxs 10, 11)
        //   2 accepted -> emits 3 (idxs 12, 13, 14)
        let schedule = [(4usize, 0usize), (4, 2), (4, 4), (4, 1), (3, 2)];
        let mut cursor = 1usize;
        for &(draft_len, want_accept) in &schedule {
            let base_abs_pos = (prompt.len() + cursor) as u64;
            let mut drafts = Vec::with_capacity(draft_len);
            for d in 0..draft_len {
                let truth = serial.output_tokens.get(cursor + d).copied().unwrap_or(0);
                if d < want_accept {
                    drafts.push(truth);
                } else if d == want_accept {
                    drafts.push(wrong_than(truth));
                } else {
                    drafts.push((d as u32) % (vocab as u32));
                }
            }
            let verified_rows = draft_len + 1;
            let outcome = stochastic_content_accept_with_policy(
                &mut spec,
                &drafts,
                verified_rows,
                VerifyLogitsFormat::Bf16,
                vocab,
                SpecVerifyPolicy::Gumbel,
                |_| DraftDistribution::PointMass,
                |row_idx, buf| {
                    *buf = bf16_bytes_for_pos(base_abs_pos + row_idx as u64, vocab);
                    true
                },
            );
            assert_eq!(
                outcome.num_accepted, want_accept,
                "cursor={cursor} draft_len={draft_len}"
            );
            assert!(outcome.bonus.is_some());
            cursor += want_accept + 1;
            assert_eq!(spec.output_tokens.len(), cursor);
        }

        assert_eq!(
            spec.output_tokens, serial.output_tokens,
            "draft=true Gumbel verification must match draft=false serial K=1 bit-for-bit"
        );

        // Negative control: sampling with unguarded out_pos=0..14 (missing prompt_len=7) diverges.
        let mut unguarded = setup_seq(&prompt, seed);
        for step in 0..15 {
            let abs_pos = (prompt.len() + step) as u64;
            let row = bf16_bytes_for_pos(abs_pos, vocab);
            let (tok, lp) = {
                let _guard = GumbelPosGuard::new(seed, step as u64);
                process_seq_logits(
                    &mut unguarded,
                    &row,
                    0,
                    vocab,
                    2,
                    false,
                    None,
                    None,
                    None,
                    None,
                    &[],
                    false,
                )
            };
            emit_token(&mut unguarded, tok, lp);
        }
        assert_ne!(
            unguarded.output_tokens, serial.output_tokens,
            "missing prompt_len offset must observably change the Gumbel coordinate stream"
        );
    }

    #[test]
    fn source_contract_pins_gumbel_guards_on_prefill_and_serial_decode() {
        for src in [
            include_str!("prefill_a_step.rs"),
            include_str!("prefill_b_step.rs"),
        ] {
            assert!(src.contains("materialize_prompt_sampling_seed("));
            assert!(src.contains("GumbelPosGuard::new("));
        }
        let decode_src = include_str!("decode_logits_seq.rs");
        assert!(decode_src.contains("GumbelPosGuard::new("));
    }
}
