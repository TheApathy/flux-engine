// SPDX-License-Identifier: AGPL-3.0-only

//! process_decode_logits: post-decode logits processing.

use super::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GreedyRowDisposition {
    /// Raw GPU argmax is mathematically identical to `process_seq_logits`.
    AcceptRawArgmax,
    /// Only the F2 confidence early-stop window is active; certify F2 on GPU
    /// via `f2_decision` without copying the full vocabulary to host.
    CertifyF2OnDevice,
    /// Copy only this single row to host and run `process_seq_logits`.
    HostSingleRow,
}

pub(super) fn bootstrap_trace_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_SPEC_BOOTSTRAP_TRACE").ok().as_deref() == Some("1"))
}

/// Decide whether a greedy (`fast_path_seq_eligible`) row's raw GPU `argmax`
/// can be committed without a full-vocabulary D2H copy.
///
/// Mirrors every intervention in [`super::decode_logits_seq::process_seq_logits`]:
/// - Whole-row observers (`think_efficiency`, `f2_window_active`) or host-forced
///   tokens (runway close, budget/think nudges, `close_allowed`, post-`</think>`
///   required `<tool_call>` pin) cannot blindly accept `raw_tok`.
/// - Decrease-only masks/penalties (`reflection_suppress_ids`, min-thinking
///   floor `</think>`, post-thinking `</think>`/`<think>`, `<tool_call>` mask
///   or `-12` bias) can only move a greedy argmax if `raw_tok` itself is in
///   that decrease-only set.
fn classify_greedy_row(
    a: &ActiveSeq,
    raw_tok: u32,
    think_end_token: Option<u32>,
    tool_call_start_token: Option<u32>,
    reflection_suppress_ids: &[u32],
    can_certify_f2_on_device: bool,
) -> GreedyRowDisposition {
    let runway_arms_close = a.inside_thinking
        && !a.force_end_thinking
        && a.thinking_budget.is_none()
        && a.max_output_tokens > 0
        && a.output_tokens.len()
            >= a.max_output_tokens
                .saturating_sub(super::confidence::min_answer_runway(a.max_output_tokens));
    let hard_override = super::confidence::compute_think_defer_hard_override(
        a.thinking_tokens,
        a.thinking_budget,
        a.output_tokens.len(),
        a.max_output_tokens,
    );
    let close_allowed_now = super::think_budget_nudge::close_allowed(a)
        || (a.inside_thinking
            && super::confidence::should_inject_think_end(
                a.force_end_thinking,
                a.in_code_fence,
                hard_override,
            ));
    let has_forced = runway_arms_close
        || close_allowed_now
        || super::think_budget_nudge::forced_token(a).is_some()
        || super::think_nudge::forced_token(a).is_some();
    let pin_tool_call = a.think_just_ended
        && a.require_tool_call
        && !a.tool_call_opened
        && !a.inside_thinking
        && tool_call_start_token.is_some();
    let wave_active = a.inside_thinking && crate::scheduler::think_efficiency_config().is_active();
    if has_forced || pin_tool_call || wave_active {
        return GreedyRowDisposition::HostSingleRow;
    }

    // Check decrease-only token collisions: lowering any id != raw_tok cannot
    // change the greedy maximum.
    let dsv41 = crate::dsv41::serving();
    let collides_reflection =
        a.inside_thinking && !dsv41 && reflection_suppress_ids.contains(&raw_tok);
    let collides_min_think_floor = a.inside_thinking
        && !a.force_end_thinking
        && !a.think_ended
        && crate::scheduler::helpers::min_thinking_floor() > 0
        && a.thinking_tokens < crate::scheduler::helpers::min_thinking_floor()
        && think_end_token == Some(raw_tok);
    let collides_post_think_mask = a.think_ended
        && !dsv41
        && (think_end_token == Some(raw_tok) || a.think_start_token == Some(raw_tok));
    let collides_tool_call_suppress =
        (a.inside_thinking || a.suppress_tool_call) && tool_call_start_token == Some(raw_tok);
    if collides_reflection
        || collides_min_think_floor
        || collides_post_think_mask
        || collides_tool_call_suppress
    {
        return GreedyRowDisposition::HostSingleRow;
    }

    if a.inside_thinking && super::think_spec_accept::f2_window_active(a) {
        if can_certify_f2_on_device
            && reflection_suppress_ids.len() < spark_runtime::sampler::f2_certify::F2_EDITS_MAX
        {
            return GreedyRowDisposition::CertifyF2OnDevice;
        }
        return GreedyRowDisposition::HostSingleRow;
    }

    GreedyRowDisposition::AcceptRawArgmax
}

/// Resolve a batch of greedy (`fast_path_seq_eligible`) sequences via GPU
/// `argmax_batch`, falling back to single-row D2H only for individual rows
/// whose raw argmax collides with a suppressed/masked token or requires a
/// host-side intervention.
#[expect(
    clippy::too_many_arguments,
    reason = "mirrors process_decode_logits intervention parameter set"
)]
fn sample_greedy_batch_with_device_argmax(
    model: &dyn Model,
    active: &mut [ActiveSeq],
    logits: DevicePtr,
    think_end_token: Option<u32>,
    think_start_token: Option<u32>,
    tool_call_start_token: Option<u32>,
    tool_call_end_token: Option<u32>,
    reflection_suppress_ids: &[u32],
    can_certify_f2_on_device: bool,
) -> Result<Vec<(u32, Option<crate::api::TokenLogprobs>)>> {
    let n = active.len();
    let raw_tokens = model.argmax_batch(logits, n, 0)?;
    let vocab_size = model.vocab_size();
    let row_bytes = vocab_size * 2;
    let mut host_row_scratch: Option<Vec<u8>> = None;
    let mut out = Vec::with_capacity(n);

    for (i, (a, raw_tok)) in active.iter_mut().zip(raw_tokens).enumerate() {
        match classify_greedy_row(
            a,
            raw_tok,
            think_end_token,
            tool_call_start_token,
            reflection_suppress_ids,
            can_certify_f2_on_device,
        ) {
            GreedyRowDisposition::AcceptRawArgmax => out.push((raw_tok, None)),
            GreedyRowDisposition::CertifyF2OnDevice => {
                let row_ptr = logits.offset(i * row_bytes);
                let edits = super::decode_logits_seq::pre_f2_edits(
                    a,
                    think_end_token,
                    reflection_suppress_ids,
                    vocab_size,
                );
                let confident =
                    super::device_topk_route::f2_decision(model, row_ptr, &edits, row_bytes)?;
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
                if post_f2_close_allowed
                    && let Some(end_tok) = think_end_token
                    && (end_tok as usize) < vocab_size
                {
                    out.push((end_tok, None));
                } else {
                    out.push((raw_tok, None));
                }
            }
            GreedyRowDisposition::HostSingleRow => {
                let row_buf = host_row_scratch.get_or_insert_with(|| vec![0u8; row_bytes]);
                model.copy_logits_to_host(logits.offset(i * row_bytes), row_buf)?;
                let sampled = process_seq_logits(
                    a,
                    row_buf,
                    0,
                    vocab_size,
                    2,
                    false,
                    think_end_token,
                    think_start_token,
                    tool_call_start_token,
                    tool_call_end_token,
                    reflection_suppress_ids,
                    false,
                );
                out.push(sampled);
            }
        }
    }
    Ok(out)
}

/// Sample and process decode logits for all active sequences.
///
/// Factored out of `step_decode_only` so that `mixed_forward` can reuse
/// the same sampling + token-processing logic without duplication (SSOT).
/// `logits` must point to `[n, vocab_size]` BF16 on device where n = active.len().
pub fn process_decode_logits(
    model: &dyn Model,
    active: &mut Vec<ActiveSeq>,
    logits: DevicePtr,
    t0: std::time::Instant,
    think_end_token: Option<u32>,
    think_start_token: Option<u32>,
    code_fence_token: Option<u32>,
    tool_call_start_token: Option<u32>,
    tool_call_end_token: Option<u32>,
    reflection_suppress_ids: &[u32],
    adaptive_sampling: bool,
    honor_eos_inside_thinking: bool,
) {
    let n = active.len();

    // Grammar bitmask is CPU-side, so any sequence with active grammar forces
    // the host-side sampling path for its logits slice.
    let any_grammar = active.iter().any(|a| a.grammar_state.is_some());
    let any_logprobs = active.iter().any(|a| a.top_logprobs.is_some());
    // FP32 lm_head models (Gemma-4 dense) MUST use the host-side path —
    // `argmax_batch` assumes BF16 layout and would interpret 4-byte FP32
    // values as 2-byte BF16 pairs, returning garbage tokens.
    let model_logits_fp32 = model.decode_logits_fp32();
    let any_thinking_state = active.iter().any(|a| a.inside_thinking || a.think_ended);
    let device_greedy_thinking_enabled = spark_runtime::sampler::device_topk::device_topk_enabled()
        && model.device_topk_supported()
        && !adaptive_sampling
        && !super::decode_logits_seq::logits_fnv_requires_row()
        && !(n == 1 && bootstrap_trace_enabled());
    let needs_host_logits = (any_thinking_state && !device_greedy_thinking_enabled)
        || any_grammar
        || any_logprobs
        || model_logits_fp32
        // deepseek_v41's cycle breaker bans tokens on host logits, greedy included.
        || crate::dsv41::repetition::active();

    let new_tokens: Vec<(u32, Option<crate::api::TokenLogprobs>)> =
        if active.iter().all(fast_path_seq_eligible) && !any_grammar && !needs_host_logits {
            if any_thinking_state {
                // Device-greedy thinking/post-thinking fast path (auto-enabled on
                // SM120 via `device_topk_mode() == Auto`): runs GPU `argmax_batch`
                // once for the whole batch and only copies an individual row on a
                // decrease-only mask collision or active forced intervention.
                match sample_greedy_batch_with_device_argmax(
                    model,
                    active,
                    logits,
                    think_end_token,
                    think_start_token,
                    tool_call_start_token,
                    tool_call_end_token,
                    reflection_suppress_ids,
                    true,
                ) {
                    Ok(t) => t,
                    Err(e) => {
                        tracing::error!("argmax_batch error: {e:#}");
                        for mut a in active.drain(..) {
                            send_error(model, &mut a, &format!("{e:#}"));
                        }
                        return;
                    }
                }
            } else {
                // Fast path: all greedy, no grammar, no thinking — GPU argmax for the full batch.
                match model.argmax_batch(logits, n, 0) {
                    Ok(t) => t.into_iter().map(|tok| (tok, None)).collect(),
                    Err(e) => {
                        tracing::error!("argmax_batch error: {e:#}");
                        for mut a in active.drain(..) {
                            send_error(model, &mut a, &format!("{e:#}"));
                        }
                        return;
                    }
                }
            }
        } else {
            // Host-side path: copy all batch logits to host, sample per-sequence.
            // Required when any sequence has temperature > 0 or grammar constraints.
            let vocab_size = model.vocab_size();
            // FP32 lm_head dispatch (Gemma-4 dense + ATLAS_GEMMA4_FP32_LMHEAD=1).
            // When the model writes FP32 logits to its decode-logits buffer, we
            // copy 4 bytes/element and skip the BF16→FP32 expansion. Earlier
            // bisection at model.rs:1192-1201 incorrectly concluded FP32 lm_head
            // had no effect on Gemma-4 because this dispatch was never wired —
            // the scheduler always read the (stale) BF16 logits buffer.
            // FP32 lm_head dispatch (Gemma-4 dense). When `use_fp32_logits` is
            // on, the per-token decode lm_head writes 4 bytes/element. The
            // passed `logits` pointer is whatever the most-recent forward
            // returned — that's already the correct buffer (prefill or decode).
            // We just need to read it with the matching width.
            let logits_fp32 = model.decode_logits_fp32();
            let elem_bytes = if logits_fp32 { 4 } else { 2 };
            let routes = super::device_topk_route::plain_batch_routes(
                active,
                logits_fp32,
                adaptive_sampling,
                reflection_suppress_ids,
            );
            let route_t0 = std::time::Instant::now();
            if let Some(routes) = routes.as_ref().filter(|r| {
                matches!(
                    r.first(),
                    Some(super::device_topk_route::TopkRoute::Device { .. })
                )
            }) {
                // ATLAS_SAMPLER_DEVICE_TOPK=1 and every row eligible (see
                // `plain_batch_routes`): no full-vocabulary D2H.
                let rows = match super::device_topk_route::fetch_device_rows(
                    model,
                    logits,
                    routes,
                    active,
                    think_end_token,
                    reflection_suppress_ids,
                ) {
                    Ok(rows) => rows,
                    Err(e) => {
                        tracing::error!("device top-k sampling failed: {e:#}");
                        for mut a in active.drain(..) {
                            send_error(
                                model,
                                &mut a,
                                &format!("device top-k sampling failed: {e:#}"),
                            );
                        }
                        return;
                    }
                };
                let fetch_share = route_t0.elapsed() / n.max(1) as u32;
                let mut out = Vec::with_capacity(n);
                for (i, (a, row)) in active.iter_mut().zip(rows).enumerate() {
                    let row_t0 = std::time::Instant::now();
                    let (sampled, route) = match row {
                        super::device_topk_route::FetchedRow::Sparse(sparse, f2_decision) => (
                            process_seq_sparse_logits(
                                a,
                                sparse,
                                f2_decision,
                                think_end_token,
                                tool_call_start_token,
                                reflection_suppress_ids,
                                adaptive_sampling,
                            ),
                            routes[i],
                        ),
                        super::device_topk_route::FetchedRow::FullBf16(full) => (
                            process_seq_logits(
                                a,
                                &full,
                                0,
                                vocab_size,
                                2,
                                false,
                                think_end_token,
                                think_start_token,
                                tool_call_start_token,
                                tool_call_end_token,
                                reflection_suppress_ids,
                                adaptive_sampling,
                            ),
                            super::device_topk_route::TopkRoute::Host("few_finite"),
                        ),
                    };
                    super::device_topk_route::note_route(a, route, fetch_share + row_t0.elapsed());
                    out.push(sampled);
                }
                out
            } else {
                let mut buf = vec![0u8; n * vocab_size * elem_bytes];
                if let Err(e) = model.copy_logits_to_host(logits, &mut buf) {
                    tracing::error!("copy_logits_to_host error: {e:#}");
                    for mut a in active.drain(..) {
                        send_error(model, &mut a, &format!("{e:#}"));
                    }
                    return;
                }
                let sampled_rows = active
                    .iter_mut()
                    .enumerate()
                    .map(|(i, a)| {
                        let sampled = process_seq_logits(
                            a,
                            &buf,
                            i,
                            vocab_size,
                            elem_bytes,
                            logits_fp32,
                            think_end_token,
                            think_start_token,
                            tool_call_start_token,
                            tool_call_end_token,
                            reflection_suppress_ids,
                            adaptive_sampling,
                        );
                        if n == 1 && bootstrap_trace_enabled() {
                            let row = &buf
                                [i * vocab_size * elem_bytes..(i + 1) * vocab_size * elem_bytes];
                            let read = |j: usize| {
                                if logits_fp32 {
                                    let off = j * 4;
                                    f32::from_le_bytes([
                                        row[off],
                                        row[off + 1],
                                        row[off + 2],
                                        row[off + 3],
                                    ])
                                } else {
                                    bf16_to_f32(row[j * 2], row[j * 2 + 1])
                                }
                            };
                            let raw_argmax = (0..vocab_size)
                                .max_by(|&x, &y| read(x).total_cmp(&read(y)))
                                .unwrap_or(0);
                            tracing::info!(
                                raw_argmax,
                                sampled = sampled.0,
                                think_ended = a.think_ended,
                                "SERIAL_POLICY_DECODE"
                            );
                        }
                        sampled
                    })
                    .collect::<Vec<_>>();
                if let Some(routes) = routes.as_ref() {
                    // Host rows share one D2H; attribute the batch wall time evenly.
                    let per_row = route_t0.elapsed() / n.max(1) as u32;
                    for (a, route) in active.iter().zip(routes) {
                        super::device_topk_route::note_route(a, *route, per_row);
                    }
                }
                sampled_rows
            }
        };

    let row_receipts = if super::logits_row_receipt::enabled() && active.len() == 1 {
        let elem_bytes = if model.logits_ptr_is_fp32(logits) {
            4
        } else {
            2
        };
        Some(
            active
                .iter()
                .zip(&new_tokens)
                .enumerate()
                .map(|(i, (a, (selected, _)))| {
                    super::logits_row_receipt::capture_device(
                        model,
                        logits.offset(i * model.vocab_size() * elem_bytes),
                        a.session_hash,
                        &a.output_tokens,
                        *selected,
                        super::logits_row_receipt::single_location("plain_decode", a, i),
                    )
                })
                .collect::<Vec<_>>(),
        )
    } else {
        None
    };

    // ── ATLAS_DUMP_HIDDEN: append token records (catch-all path) ──
    // Pairs decode-step hidden states (dumped per-layer) with the token that
    // was sampled at this step. Fires for both fast path (argmax_batch) and
    // host-side path (process_seq_logits). Matches the model-side hook that
    // dumps records only when ATLAS_DUMP_HIDDEN env var is set.
    // Cached via `dump_hidden_path` OnceLock so the var-unset case is a
    // single relaxed load per scheduler step.
    if let Some(path) = super::helpers::dump_hidden_path() {
        const TOKEN_DUMP_MAGIC: u32 = 0xA71B5DEE;
        use std::io::Write;
        if let Ok(mut f) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
        {
            for (tok, _lp) in &new_tokens {
                let _ = f.write_all(&TOKEN_DUMP_MAGIC.to_le_bytes());
                let _ = f.write_all(&tok.to_le_bytes());
                let _ = f.write_all(&0u32.to_le_bytes());
                let _ = f.write_all(&0u32.to_le_bytes());
            }
        }
    }

    let step_ms = t0.elapsed().as_secs_f64() * 1000.0;
    if tracing::enabled!(tracing::Level::DEBUG) {
        let token_ids: Vec<u32> = new_tokens.iter().map(|(t, _)| *t).collect();
        tracing::debug!(
            "DECODE: n={n} step={step_ms:.1}ms ({:.1} tok/s) tokens={:?}",
            1000.0 * n as f64 / step_ms,
            token_ids,
        );
    }

    let now = Instant::now();
    let mut gen_tokens_emitted: u64 = 0;
    for (i, (tok, mut logprobs)) in new_tokens.into_iter().enumerate() {
        let a = &mut active[i];
        // Cooperative cancellation (a stream-side stop string or loop guard):
        // end here, like speculative `emit_token` does, and drop the sample.
        // The serial path never read the flag, so a matched stop string only
        // hid the text while generation ran on to max_tokens.
        if retire_cancelled_speculation(a, "serial_decode") {
            continue;
        }
        let was_inside_thinking = a.inside_thinking;
        let previous_last_token = a.last_token;
        let previous_last_token_time = a.last_token_time;
        a.last_token = tok;
        a.last_token_time = now;

        // A ChatML role boundary ends the assistant turn even when an active
        // grammar, required tool call, or min_tokens would suppress ordinary
        // EOS. Keep this before every phase/grammar side effect, exactly like
        // speculative `emit_token`.
        if commit_chatml_role_boundary(a, tok, im_start_hard_stop(), &logprobs) {
            gen_tokens_emitted += 1;
            continue;
        }

        // Spontaneous <think>: model generates <think> even when thinking
        // was not requested. Enter thinking mode so EOS is suppressed and
        // thinking content is stripped. Matches vLLM's behavior of always
        // parsing <think>...</think> regardless of enable_thinking setting.
        //
        // F9+F10 (2026-04-26): the sample-time logit mask at line ~1716
        // hard-blocks `<think>` when `think_ended=true`, so this branch
        // should not fire after a watchdog has force-closed thinking.
        // Defence-in-depth: if the model somehow still emits <think>
        // (e.g. the start token differs from the masked one in edge
        // cases), decay the budget by `>> watchdog_fires.min(4)` so
        // each successive re-entry has a tighter window. After 4+
        // fires, the budget is 1/16 of normal — the watchdog kills
        // re-entry within a handful of tokens.
        // DDTree guard: bonus tokens bypass the sample-time logit mask that
        // blocks <think> when think_ended=true. If think_ended=true and <think>
        // arrives, the token is already in the KV cache (committed by verify),
        // so we must enter thinking mode but force an immediate exit: next token
        // is forced to </think> (0 thinking content tokens, 1 </think> overhead).
        // deepseek_v41: `<think>` and a stray `</think>` are ordinary output
        // tokens for the Python engine (its router splits on the text).
        let dsv41 = crate::dsv41::serving();
        if !dsv41 && !a.inside_thinking && think_start_token == Some(tok) {
            // Re-entering thinking re-arms the response-entry counter for the
            // next `</think>` boundary.
            a.post_think_gate_steps = 0;
            if !a.think_ended {
                let spontaneous_base = crate::api::chat::effective_spontaneous_think_budget(
                    a.spontaneous_think_budget,
                );
                let decayed = resolve_rethink_budget(spontaneous_base, a.think_watchdog_fires);
                a.inside_thinking = true;
                a.think_ended = false; // reset so </think> detection path works
                a.think_skip_count = 0;
                a.thinking_budget = Some(decayed);
                if a.think_watchdog_fires > 0 {
                    tracing::debug!(
                        fires = a.think_watchdog_fires,
                        decayed_budget = decayed,
                        "Spontaneous <think> re-entry after watchdog; decayed budget"
                    );
                } else {
                    tracing::debug!("Spontaneous <think> detected, entering thinking mode");
                }
            } else {
                // DDTree cliff-path <think> (think_ended=true): force-exit.
                a.inside_thinking = true;
                a.think_skip_count = 0;
                a.thinking_budget = Some(0);
                a.force_end_thinking = true;
                tracing::debug!(
                    "DDTree <think> (think_ended=true): force-exit thinking (0 content tokens)"
                );
            }
            check_request_timeout(a);
            continue; // don't emit <think> as content
        }

        // Silently skip </think> tokens outside thinking mode.
        // At long context (37k+), models degenerate into repeating </think>.
        // Skip up to 50 occurrences, then force-stop. This gives cached
        // prompts a chance to produce content while limiting degenerate loops.
        if !dsv41 && !a.inside_thinking && think_end_token == Some(tok) {
            a.think_skip_count += 1;
            if a.think_skip_count >= 50 {
                a.finished = true;
                record_watchdog_stop(a, "repeated_think_end_watchdog");
            }
            check_request_timeout(a);
            continue;
        }
        // Reset skip counter when a real content token is generated.
        if a.think_ended {
            a.think_skip_count = 0;
        }

        // Thinking tokens don't count toward remaining (thinking is "free").
        if a.inside_thinking {
            if think_end_token == Some(tok) {
                a.inside_thinking = false;
                a.force_end_thinking = false;
                super::think_budget_nudge::on_think_closed(a);
                a.consecutive_confident = 0;
                a.in_code_fence = false;
                a.think_ended = true;
                a.post_think_gate_steps = 0;
                // One-shot: pin the next sampled token to the
                // tool-call-start token if the request requires a
                // tool call (Change 3b). Cleared in the `else`
                // branch below on the next emit.
                a.think_just_ended = true;
            } else {
                a.thinking_tokens += 1;
                // Track ``` code-fence parity within the thinking block:
                // each fence token flips in/out of a fenced code span.
                // F2 still tracks confidence and can arm `force_end_thinking`
                // while `in_code_fence` is set, but `</think>` injection is
                // deferred (`should_inject_think_end`) until the fence closes
                // (unless hard-overridden), so coherent code blocks are not
                // truncated mid-statement. THINK_LOOP (below) deliberately
                // stays active even inside fences: it catches *repeating*
                // fence-narration, not one coherent block.
                a.in_code_fence = toggle_code_fence(a.in_code_fence, tok, code_fence_token);
                // Set force_end_thinking when budget exhausted (picked up next iteration)
                if let Some(budget) = a.thinking_budget
                    && a.thinking_tokens >= budget
                    && !a.force_end_thinking
                {
                    // `tok` is pushed below, after the phase bookkeeping.
                    super::think_budget_nudge::arm_budget_close(a, budget, 1);
                }
                // Structural control-token attractor. Unlike the broad
                // phrase-loop watchdog, this remains enabled for quality
                // profiles: repeated ChatML role openers are never valid
                // reasoning content, while prose/code repetition can be.
                if !a.force_end_thinking
                    && im_start_hard_stop().is_some_and(|im_start| {
                        detect_chatml_thinking_cycle(&a.output_tokens, Some(tok), im_start)
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
                        "ChatML control-token cycle detected during thinking; forcing </think>"
                    );
                }
                // Token-level fence-loop detection. Catches the Qwen3.5-35B
                // phrase attractor (`Running:\`\`\`bash cmd\`\`\`Executing:…`
                // cycling) within ~24-60 tokens of the loop starting,
                // instead of waiting for the 256-token thinking budget.
                if enable_thinking_loop_watchdog()
                    && !a.force_end_thinking
                    && a.thinking_tokens >= THINK_LOOP_MIN_TOKENS
                    && a.thinking_tokens.is_multiple_of(THINK_LOOP_CHECK_STRIDE)
                    && detect_thinking_token_loop(&a.output_tokens)
                {
                    a.force_end_thinking = true;
                    a.think_watchdog_fires = a.think_watchdog_fires.saturating_add(1);
                    crate::openai::record_shared_intervention(
                        a.intervention_ledger.as_ref(),
                        "thinking_loop_watchdog",
                        "force_end_thinking",
                    );
                    tracing::warn!(
                        thinking_tokens = a.thinking_tokens,
                        watchdog_fires = a.think_watchdog_fires,
                        "Thinking-loop watchdog fired (period-{}…{} repeat in tail); forcing </think> early",
                        THINK_LOOP_PERIOD_MIN,
                        THINK_LOOP_PERIOD_MAX,
                    );
                }
                super::think_nudge::on_thinking_token(a, tok);
                super::think_budget_nudge::on_thinking_token(a, tok);
            }
        } else {
            // Content-phase token: budget bookkeeping + the content-loop
            // and inter-tool-prose watchdogs. Extracted to
            // `decode_logits_content.rs` to keep this file ≤500 LoC.
            // `model` is threaded through so a watchdog rollback can
            // restore SSM recurrent state on hybrid models (Phase-C).
            match handle_content_token(a, tok, model, previous_last_token, previous_last_token_time)
            {
                ContentTokenDisposition::CommitSample => {
                    advance_content_grammar(a, tok);
                }
                ContentTokenDisposition::DiscardSampleAndStop { dropped } => {
                    tracing::debug!(dropped, "Discarded sample after rollback hard-stop");
                    continue;
                }
            }
        }

        // Track <tool_call> token: once seen, legacy tool call requirement is satisfied.
        // Guard with !inside_thinking — a <tool_call> inside thinking is spurious
        // and must not clear require_tool_call (which would allow premature EOS).
        if a.require_tool_call && tool_call_start_token == Some(tok) && !a.inside_thinking {
            a.require_tool_call = false;
            a.tool_call_opened = true;
        }
        // Update the tool-body phase AFTER content handling: the opening token
        // is the boundary into the body, while the closing token still belongs
        // to the body for watchdog accounting. `process_seq_logits` observes
        // this flag on the next position and disables prose penalties inside
        // structured JSON exactly like speculative emission.
        update_tool_body_phase(a, tok);
        // Safety: if require_tool_call is still set after 512 tokens, the model
        // isn't generating a tool call (grammar may have failed to compile).
        // Clear the flag so EOS is no longer suppressed — prevents infinite gen.
        clear_stale_require_tool_call(a, a.output_tokens.len());

        // </tool_call> stop: in legacy mode (no grammar), stop after first tool call.
        // When grammar is active, allow the model to generate multiple tool calls —
        // the grammar controls when EOS is valid.
        if tool_call_end_token == Some(tok) && !a.inside_thinking {
            if let Some(lp) = logprobs.take() {
                a.logprobs_data.push(lp);
            }
            commit_tool_call_close(a, tok, was_inside_thinking);
            gen_tokens_emitted += 1;
            check_request_timeout(a);
            continue;
        }

        // Single-pass over eos_tokens (was 2× linear scan).
        let is_eos = a.eos_tokens.contains(&tok);
        let implicit_think_close = is_eos
            && maybe_implicit_close_thinking_on_eos(
                a,
                tok,
                a.output_tokens.len(),
                honor_eos_inside_thinking,
            );
        // Keep this identical to speculative emission. Required tools are
        // protected by grammar / legacy-required mode, but an ordinary answer
        // may terminate on its first EOS after `</think>`.
        let suppress_eos = eos_is_suppressed(a, a.output_tokens.len());
        if implicit_think_close {
            // Discard the first in-thinking EOS after using it as the implicit
            // phase boundary. `last_token` remains EOS so the next model step
            // consumes the exact sampled token, matching normal suppression.
            retire_if_budget_exhausted(a, "serial implicit thinking close");
            check_request_timeout(a);
            continue;
        }
        if is_eos && !suppress_eos {
            // Stop/EOS token: do NOT stream to client (OpenAI spec: returned text
            // must not contain the stop sequence). The token is still added to
            // output_tokens for correct token count; the API layer strips the
            // decoded text for blocking responses.
            let terminal_logprobs = logprobs.clone();
            if let Some(lp) = logprobs.take() {
                a.logprobs_data.push(lp);
            }
            a.output_tokens.push(tok);
            a.update_prefix_hash(tok);
            gen_tokens_emitted += 1;
            publish_hidden_committed_token(a, &terminal_logprobs, "serial terminal EOS");
            a.finished = true;
        } else if is_eos {
            // EOS suppressed: grammar not terminated or legacy tool call not yet seen.
            // Don't stop, don't stream the EOS — the model must keep generating.
            // Don't add to output_tokens (EOS is discarded).
            retire_if_budget_exhausted(a, "serial suppressed EOS");
        } else {
            if let Some(lp) = logprobs.take() {
                a.logprobs_data.push(lp);
            }
            a.output_tokens.push(tok);
            a.update_prefix_hash(tok);
            gen_tokens_emitted += 1;
            if a.think_ended && !was_inside_thinking && !a.inside_thinking {
                a.post_think_gate_steps = mtp_gate::advance_entry_counter(a.post_think_gate_steps);
            }
            // Phase-C: if this committed token is a content-phase
            // boundary token (sentence end / newline) and the model is
            // hybrid (attention + SSM), snapshot the recurrent SSM
            // state now so a later watchdog rollback to this boundary
            // can also rewind h_state/conv_state — not just the KV
            // cache. Gated to content tokens because the watchdogs that
            // roll back all fire post-`</think>`, and `apply_rollback`
            // requires every dropped token to be a content token. No-op
            // for pure-attention models / disabled rings (see
            // `rollback::snapshot_boundary_if_ssm`).
            if !a.inside_thinking {
                rollback::snapshot_boundary_if_ssm(a, model);
            }
            // OPENCODE FIX: when the model spontaneously emits `<think>` even
            // though the request didn't ask for thinking (`enable_thinking=false`),
            // the `<think>` open token itself is suppressed (line ~1356), but
            // the thinking-content tokens that follow (and the closing `</think>`)
            // MUST also be kept off the wire — otherwise opencode persists them
            // as `assistant.content` and on the next turn the model sees its own
            // past garbage (fake `<function=…>`, fake `<tool_response>`) as a
            // "format example" and continues the pattern. Tokens stay in
            // `output_tokens` for the blocking response path's reasoning_content
            // extraction.
            let suppress_stream = (a.inside_thinking
                || (was_inside_thinking
                    && (think_end_token == Some(tok) || a.think_end_token == Some(tok))))
                && !a.enable_thinking;
            if let ResponseSink::Streaming(ref tx) = a.sink {
                let event =
                    committed_token_event(tok, &a.logprobs_data.last().cloned(), suppress_stream);
                // Bounded wait on a full channel: one stalled SSE consumer must
                // not park the scheduler thread (and every other sequence).
                if !super::emit_step::stream_send_bounded(tx, event, "decode_logits") {
                    a.finished = true;
                }
            }
            if a.output_tokens.len() >= a.max_output_tokens {
                tracing::info!(
                    "process_decode_logits: max_output_tokens={} reached, output_tokens={}, thinking_tokens={}",
                    a.max_output_tokens,
                    a.output_tokens.len(),
                    a.thinking_tokens,
                );
                a.finished = true;
            }
            if a.remaining == 0 {
                tracing::info!(
                    "process_decode_logits: remaining=0, output_tokens={}, thinking_tokens={}",
                    a.output_tokens.len(),
                    a.thinking_tokens
                );
                a.finished = true;
            }
            // Grammar termination = end of sequence. With `stop_after_first=true`
            // (tool_choice="required"), the structural-tag matcher transitions
            // to its terminal state right after the single tool call closes.
            // The model's free distribution past that point can be degenerate
            // (Nemotron-Super-120B emits a `</parameter>` loop and never
            // samples EOS naturally). Finish here instead of letting it run.
            if a.grammar_state
                .as_ref()
                .is_some_and(|gs| gs.is_terminated())
            {
                a.finished = true;
            }
        }

        // Check request timeout (total + stall). Short-circuits immediately
        // when the sequence finished legitimately above (EOS, max_tokens,
        // grammar termination, or exhausted budget).
        check_request_timeout(a);
    }
    if gen_tokens_emitted > 0 {
        crate::metrics::compat_metrics::add_generation_tokens(gen_tokens_emitted);
    }
    crate::metrics::compat_metrics::set_kv_blocks_free(model.num_free_blocks());
    if let Some(receipts) = row_receipts {
        for (a, receipt) in active.iter().zip(receipts) {
            if let Some(receipt) = receipt {
                super::logits_row_receipt::output_map(&receipt, &a.output_tokens);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::request_contract::TIMEOUT_TOTAL_CAUSE;
    use crate::scheduler::test_fixtures::{active_seq, model};
    use std::time::Duration;

    fn run_serial_step(
        seq: ActiveSeq,
        tok: u32,
        think_end_token: Option<u32>,
        think_start_token: Option<u32>,
        honor_eos_inside_thinking: bool,
    ) -> ActiveSeq {
        let m = model::recording();
        *m.argmax.lock().expect("argmax lock") = vec![tok];
        let mut logits = vec![-10.0f32; 32];
        logits[tok as usize] = 10.0;
        *m.host_logits.lock().expect("host_logits lock") = logits;
        let mut active = vec![seq];
        process_decode_logits(
            &m,
            &mut active,
            DevicePtr::NULL,
            Instant::now(),
            think_end_token,
            think_start_token,
            None,
            None,
            None,
            &[],
            false,
            honor_eos_inside_thinking,
        );
        active.pop().expect("single active sequence")
    }

    #[test]
    fn serial_suppressed_eos_and_structural_tokens_enforce_request_timeout() {
        let past = Instant::now()
            .checked_sub(Duration::from_secs(1))
            .unwrap_or_else(Instant::now);

        // 1. Suppressed EOS outside thinking (require_tool_call).
        let mut suppressed_eos = active_seq(ResponseSink::Blocking(None), 0);
        suppressed_eos.require_tool_call = true;
        suppressed_eos.timeout_at = Some(past);
        let out = run_serial_step(suppressed_eos, 17, None, None, false);
        assert!(out.finished);
        assert_eq!(out.stop_cause, Some(TIMEOUT_TOTAL_CAUSE));
        assert!(!out.output_tokens.contains(&17));

        // 2. Suppressed EOS inside thinking (honor_eos_inside_thinking = false).
        let mut suppressed_think_eos = active_seq(ResponseSink::Blocking(None), 0);
        suppressed_think_eos.inside_thinking = true;
        suppressed_think_eos.think_ended = false;
        suppressed_think_eos.timeout_at = Some(past);
        let out = run_serial_step(suppressed_think_eos, 17, Some(29), Some(28), false);
        assert!(out.finished);
        assert_eq!(out.stop_cause, Some(TIMEOUT_TOTAL_CAUSE));

        // 3. Implicit thinking close on EOS.
        let mut implicit_close = active_seq(ResponseSink::Blocking(None), 0);
        implicit_close.inside_thinking = true;
        implicit_close.think_ended = false;
        implicit_close.timeout_at = Some(past);
        let out = run_serial_step(implicit_close, 17, Some(29), Some(28), true);
        assert!(out.finished);
        assert_eq!(out.stop_cause, Some(TIMEOUT_TOTAL_CAUSE));

        // 4. Spontaneous <think> entry.
        let mut spontaneous_think = active_seq(ResponseSink::Blocking(None), 0);
        spontaneous_think.think_ended = false;
        spontaneous_think.timeout_at = Some(past);
        let out = run_serial_step(spontaneous_think, 28, Some(29), Some(28), false);
        assert!(out.finished);
        assert_eq!(out.stop_cause, Some(TIMEOUT_TOTAL_CAUSE));

        // 5. Stray </think> outside thinking (< 50 skip limit).
        let mut stray_close = active_seq(ResponseSink::Blocking(None), 0);
        stray_close.think_ended = false;
        stray_close.timeout_at = Some(past);
        let out = run_serial_step(stray_close, 29, Some(29), Some(28), false);
        assert!(out.finished);
        assert_eq!(out.stop_cause, Some(TIMEOUT_TOTAL_CAUSE));
    }

    #[test]
    fn serial_legitimate_eos_and_max_tokens_do_not_become_timeout_errors() {
        let past = Instant::now()
            .checked_sub(Duration::from_secs(1))
            .unwrap_or_else(Instant::now);

        // Normal terminal EOS on the exact step the deadline passes.
        let mut eos_seq = active_seq(ResponseSink::Blocking(None), 0);
        eos_seq.timeout_at = Some(past);
        let out = run_serial_step(eos_seq, 17, None, None, false);
        assert!(out.finished);
        assert_eq!(out.stop_cause, None);
        assert_eq!(out.output_tokens.last(), Some(&17));

        // Normal content token hitting max_output_tokens on the exact step the deadline passes.
        let mut max_seq = active_seq(ResponseSink::Blocking(None), 0);
        max_seq.max_output_tokens = 2;
        max_seq.timeout_at = Some(past);
        let out = run_serial_step(max_seq, 5, None, None, false);
        assert!(out.finished);
        assert_eq!(out.stop_cause, None);
        assert_eq!(out.output_tokens, vec![31, 5]);
    }

    #[test]
    fn serial_spontaneous_think_end_is_suppressed_on_sse_when_enable_thinking_false() {
        let (tx, mut rx) = tokio::sync::mpsc::channel(8);
        let mut seq = active_seq(ResponseSink::Streaming(tx), 0);
        seq.enable_thinking = false;
        seq.inside_thinking = false;
        seq.think_ended = false;
        seq.think_start_token = Some(28);
        seq.think_end_token = Some(29);

        // Spontaneous `<think>` enters thinking without emitting to SSE.
        seq = run_serial_step(seq, 28, Some(29), Some(28), false);
        assert!(seq.inside_thinking);
        assert!(matches!(
            rx.try_recv(),
            Err(tokio::sync::mpsc::error::TryRecvError::Empty)
        ));

        // Thinking content stays hidden (`CompletionProgress`).
        seq = run_serial_step(seq, 10, Some(29), Some(28), false);
        assert!(matches!(rx.try_recv(), Ok(StreamEvent::CompletionProgress)));

        // Closing `</think>` clears `inside_thinking` AND must stay hidden on SSE.
        seq = run_serial_step(seq, 29, Some(29), Some(28), false);
        assert!(!seq.inside_thinking && seq.think_ended);
        assert!(
            matches!(rx.try_recv(), Ok(StreamEvent::CompletionProgress)),
            "spontaneous </think> must not leak as StreamEvent::Token when enable_thinking=false"
        );

        // Post-thinking answer token streams normally.
        let _seq = run_serial_step(seq, 11, Some(29), Some(28), false);
        assert!(matches!(rx.try_recv(), Ok(StreamEvent::Token(11))));

        // Control: when `enable_thinking=true`, closing `</think>` still streams.
        let (tx_on, mut rx_on) = tokio::sync::mpsc::channel(4);
        let mut enabled = active_seq(ResponseSink::Streaming(tx_on), 0);
        enabled.enable_thinking = true;
        enabled.inside_thinking = true;
        enabled.think_ended = false;
        enabled.think_end_token = Some(29);
        let _enabled = run_serial_step(enabled, 29, Some(29), Some(28), false);
        assert!(matches!(rx_on.try_recv(), Ok(StreamEvent::Token(29))));
    }

    #[test]
    fn greedy_row_classifier_accepts_non_colliding_thinking_and_post_thinking_rows() {
        let mut seq = active_seq(ResponseSink::Blocking(None), 0);
        seq.inside_thinking = true;
        seq.think_ended = false;
        seq.thinking_budget = None;
        seq.thinking_tokens = 10;
        seq.max_output_tokens = 2048;
        seq.think_start_token = Some(28);
        seq.think_end_token = Some(29);

        // Normal thinking token with no collision avoids full-vocab host copy.
        assert_eq!(
            classify_greedy_row(&seq, 7, Some(29), Some(25), &[11, 12], true),
            GreedyRowDisposition::AcceptRawArgmax
        );
        // Collision with reflection suppression or `<tool_call>` falls back to single-row host.
        assert_eq!(
            classify_greedy_row(&seq, 11, Some(29), Some(25), &[11, 12], true),
            GreedyRowDisposition::HostSingleRow
        );
        assert_eq!(
            classify_greedy_row(&seq, 25, Some(29), Some(25), &[11, 12], true),
            GreedyRowDisposition::HostSingleRow
        );

        // Past F2 confidence window (>= 400 thinking tokens): certifies F2 on device when supported.
        seq.thinking_tokens = CONFIDENCE_EARLY_STOP_MIN_THINKING;
        assert_eq!(
            classify_greedy_row(&seq, 7, Some(29), Some(25), &[11, 12], true),
            GreedyRowDisposition::CertifyF2OnDevice
        );
        assert_eq!(
            classify_greedy_row(&seq, 7, Some(29), Some(25), &[11, 12], false),
            GreedyRowDisposition::HostSingleRow
        );

        // Forced close always goes through host single-row handling.
        seq.force_end_thinking = true;
        assert_eq!(
            classify_greedy_row(&seq, 7, Some(29), Some(25), &[11, 12], true),
            GreedyRowDisposition::HostSingleRow
        );

        // Post-thinking (`think_ended = true`): ordinary token accepts raw argmax;
        // masked `</think>` or `<think>` collides and falls back to single-row host.
        seq.inside_thinking = false;
        seq.force_end_thinking = false;
        seq.think_ended = true;
        assert_eq!(
            classify_greedy_row(&seq, 7, Some(29), Some(25), &[11, 12], true),
            GreedyRowDisposition::AcceptRawArgmax
        );
        assert_eq!(
            classify_greedy_row(&seq, 29, Some(29), Some(25), &[11, 12], true),
            GreedyRowDisposition::HostSingleRow
        );
        assert_eq!(
            classify_greedy_row(&seq, 28, Some(29), Some(25), &[11, 12], true),
            GreedyRowDisposition::HostSingleRow
        );
    }

    #[test]
    fn greedy_device_argmax_batch_copies_only_colliding_row() {
        let m = model::recording();
        // Row 0 is post-thinking and emits normal token 7 (no collision -> 0 D2H).
        // Row 1 is post-thinking and raw argmax collides with masked `</think>` (29),
        // so host fallback copies that row and picks the runner-up token 13!
        *m.argmax.lock().expect("argmax lock") = vec![7, 29];
        let mut logits = vec![-10.0f32; 32];
        logits[29] = 10.0;
        logits[13] = 8.0;
        *m.host_logits.lock().expect("host_logits lock") = logits;

        let mut s0 = active_seq(ResponseSink::Blocking(None), 0);
        s0.inside_thinking = false;
        s0.think_ended = true;
        s0.think_start_token = Some(28);
        s0.think_end_token = Some(29);
        let mut s1 = active_seq(ResponseSink::Blocking(None), 1);
        s1.inside_thinking = false;
        s1.think_ended = true;
        s1.think_start_token = Some(28);
        s1.think_end_token = Some(29);

        let mut active = vec![s0, s1];
        let out = sample_greedy_batch_with_device_argmax(
            &m,
            &mut active,
            DevicePtr::NULL,
            Some(29),
            Some(28),
            None,
            None,
            &[],
            true,
        )
        .expect("greedy batch");
        let summary: Vec<(u32, bool)> = out.iter().map(|(tok, lp)| (*tok, lp.is_some())).collect();
        assert_eq!(summary, vec![(7, false), (13, false)]);
    }
}
