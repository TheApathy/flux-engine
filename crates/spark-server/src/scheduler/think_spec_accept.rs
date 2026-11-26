// SPDX-License-Identifier: AGPL-3.0-only

//! Policy-aware speculative decode inside `<think>` spans.
//!
//! The scheduler historically forced every batch containing a thinking
//! sequence into plain `step_decode_only` (see the `!a.inside_thinking`
//! MTP gate in `scheduler::run`): the plain decode path applies per-token
//! logit interventions during thinking — F1 reflection suppression, the
//! thinking-efficiency wave, the F2 confidence early stop, the tool-call
//! hard mask, the forced `</think>` injection — that the DFlash K=γ
//! verify (raw GPU argmax) does not replicate. Thinking is the majority
//! of tokens on a reasoning model, so the whole batch collapsed to
//! ~13-17 tok/s whenever any sequence was mid-`<think>`.
//!
//! This module replicates the interventions in a post-verify CPU accept
//! filter (same shape as `dflash_masked_accept`, the grammar sibling) so
//! speculation runs through thinking spans LOSSLESSLY:
//!
//!   * SLOW PATH: a verify-logits row is lazily D2H'd (cheap on GB10
//!     unified memory) and fed through [`process_seq_logits`] — the EXACT
//!     function the plain path samples with, including F1/F2/wave state
//!     evolution, penalties, logit_bias and seeded sampling — so the two
//!     paths share one implementation and cannot drift.
//!   * FAST PATH: no D2H when the plain-path token is PROVABLY the raw
//!     verify argmax — greedy request with neutral penalties, wave off,
//!     F2 window closed, and the raw argmax outside the
//!     intervention-sensitive id set (see [`position_fast_path_ok`]).
//!     The forced `</think>` injection is distribution-independent
//!     (masks everything but `</think>`), so it stays on the fast path.
//!
//! Acceptance TRUNCATES (never rewrites): a draft is accepted only while
//! it equals the plain-path token; the first divergence becomes the
//! bonus. `</think>` and EOS are phase-boundary tokens — they always end
//! the walk as the bonus (never as an accepted draft) so `a.last_token`
//! keeps the plain-path contract of "committed but not yet fed to the
//! model" across the transition.
//!
//! Native MTP uses this path by default when the BF16 row oracle is available.
//! DFlash keeps the historical `ATLAS_THINK_SPEC=1` opt-in because its wider
//! raw verify path has a separate correctness contract.

use super::*;

/// `ATLAS_THINK_SPEC=1` — read once at first use.
pub(super) fn think_spec_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_THINK_SPEC").ok().as_deref() == Some("1"))
}

/// Per-run parameters threaded from `scheduler::run` into `step_mtp` /
/// `step_verify_dflash` (the tokens not carried on `ActiveSeq`).
#[derive(Clone, Copy)]
pub(super) struct ThinkSpecCtx<'a> {
    /// The BF16 per-row policy oracle is available. Scheduler dispatch enables
    /// native MTP thinking by default and separately keeps DFlash behind
    /// `ATLAS_THINK_SPEC=1`.
    pub enabled: bool,
    /// ``` fence token — `in_code_fence` parity (gates the forced
    /// `</think>` injection deferral).
    pub code_fence_token: Option<u32>,
    /// F1 reflection-suppression id set (`-10.0` during thinking).
    pub reflection_suppress_ids: &'a [u32],
    /// Mirror the scheduler's request-wide adaptive sampler setting.
    pub adaptive_sampling: bool,
    /// Per-model EOS-in-thinking policy. Production uses `None` so every
    /// boundary reads the refreshable model-load value; tests may pin a value.
    pub honor_eos_inside_thinking: Option<bool>,
}

impl ThinkSpecCtx<'_> {
    pub(super) fn resolved_honor_eos_inside_thinking(&self) -> bool {
        self.honor_eos_inside_thinking
            .unwrap_or_else(super::helpers::honor_eos_inside_thinking)
    }
}

/// Result of a sampler-policy accept walk. Tokens (accepted prefix AND bonus)
/// are already committed/streamed by the walk itself — the caller must skip
/// its own emit loop and only run the seq/SSM bookkeeping.
pub(super) struct SpecAcceptOutcome {
    /// Drafts accepted (excludes the bonus). Same contract as the legacy
    /// accept-prefix: `seq` rollback and the SSM commit use it unchanged.
    pub num_accepted: usize,
    /// The committed bonus token. `None` only when the sequence finished
    /// (or a logits D2H failed) mid-walk — the caller must bail out
    /// exactly like the legacy emit loop's early return.
    pub bonus: Option<u32>,
    /// Classified rows, including hidden controls; pre-row cancellation is zero.
    pub consumed_rows: usize,
}

/// Whether a thinking-span accept walk (`a.inside_thinking == true` on entry)
/// closed thinking on a non-terminal `</think>` token.
///
/// Multi-row speculative verify callers must defer
/// [`super::rollback::snapshot_boundary_if_ssm`] until AFTER
/// `commit_verify_state_async` / `commit_verify_batch_cached` and
/// `model.sync_secondary()` complete, rather than snapshotting inside the
/// accept walk while live recurrent state still holds the verify kernel's
/// rejected tail.
pub(super) fn outcome_ended_thinking(a: &ActiveSeq, outcome: &SpecAcceptOutcome) -> bool {
    !a.finished
        && !a.inside_thinking
        && a.think_end_token.is_some()
        && outcome.bonus == a.think_end_token
}

/// Request-level fast-path eligibility: the plain path's temp-0 sample is
/// a pure post-intervention argmax only when no penalty / bias / logprob
/// machinery can re-order or observe the distribution
/// (`sample_with_params_seeded` applies these BEFORE its greedy bypass).
pub(super) fn fast_path_seq_eligible(a: &ActiveSeq) -> bool {
    a.temperature == 0.0
        && a.repetition_penalty == 1.0
        && a.presence_penalty == 0.0
        && a.frequency_penalty == 0.0
        && a.lz_penalty <= 0.0
        && a.dry_multiplier <= 0.0
        && a.logit_bias.is_empty()
        && a.top_logprobs.is_none()
}

/// Position-level fast-path gate. The interventions can only move the
/// argmax when (a) the wave shapes logits, (b) the F2 window needs the
/// distribution for its state evolution, (c) the raw argmax is itself a
/// suppressed id (a -10 on a NON-argmax id can never promote another
/// token), or (d) the raw argmax is the hard-masked `<tool_call>` start.
/// Qualification receipt mode also closes the shortcut: every committed
/// token needs the exact logits row which [`process_seq_logits`] hashes.
pub(super) fn position_fast_path_ok(
    raw_argmax: u32,
    reflection_suppress_ids: &[u32],
    tool_call_start_token: Option<u32>,
    wave_active: bool,
    f2_window_active: bool,
    logits_fnv_requires_row: bool,
) -> bool {
    !logits_fnv_requires_row
        && !wave_active
        && !f2_window_active
        && !reflection_suppress_ids.contains(&raw_argmax)
        && tool_call_start_token != Some(raw_argmax)
}

/// Whether the F2 confidence early-stop would observe THIS position on the
/// plain path (`process_seq_logits`): armed sequences skip it, and it only
/// opens past [`CONFIDENCE_EARLY_STOP_MIN_THINKING`] thinking tokens.
pub(super) fn f2_window_active(a: &ActiveSeq) -> bool {
    !a.force_end_thinking
        && a.thinking_tokens >= CONFIDENCE_EARLY_STOP_MIN_THINKING
        && watchdog_params().confidence_early_stop
}

/// Whether the plain path would force-inject `</think>` at this position
/// (mirrors the defer + answer-runway hard-override arithmetic in
/// `process_seq_logits` via `think_budget_nudge::close_allowed`).
fn forced_injection_pending(a: &ActiveSeq) -> bool {
    super::think_budget_nudge::close_allowed(a)
}

/// Thinking-span accept walk over the verify positions.
///
/// Row `i` of the verify logits is the target's prediction for the slot
/// `drafts[i]` occupies (row `drafts.len()` is the bonus slot) — the same
/// indexing as `dflash_masked_accept`. Per position: derive the
/// plain-path token (fast path = raw argmax, slow path = the full
/// [`process_seq_logits`] pipeline over the D2H'd row), COMMIT it via
/// [`commit_thinking_token`] (streaming + the plain path's per-token side
/// effects), and accept while it equals `drafts[i]`. The first
/// divergence — or a `</think>`/EOS boundary — becomes the bonus and ends
/// the walk. State (F2 run, difficulty probe, fence parity, THINK_LOOP
/// watchdog, budget arming) evolves on `a` exactly as one plain-decode
/// step per committed token.
///
/// `fetch_row(i, buf)` fills `buf` with the BF16 bytes of verify row `i`
/// and returns `true` (`false` = fatal D2H failure → the sequence is
/// finished, mirroring the other verify error paths). The fill-buffer
/// shape lets one ~vocab*2-byte allocation serve every slow-path row in
/// the walk. `on_think_end` runs after a committed `</think>`
/// (production: the plain path's SSM boundary snapshot).
pub(super) fn dflash_thinking_accept(
    a: &mut ActiveSeq,
    drafts: &[u32],
    verified: &[u32],
    ctx: &ThinkSpecCtx<'_>,
    fetch_row: impl FnMut(usize, &mut Vec<u8>) -> bool,
    on_think_end: impl FnMut(&mut ActiveSeq),
) -> SpecAcceptOutcome {
    dflash_thinking_accept_routed(
        a,
        drafts,
        verified,
        ctx,
        fetch_row,
        |_, _| super::device_topk_spec::DeviceRow::Host,
        on_think_end,
    )
}

/// [`dflash_thinking_accept`] with a device top-k row source: `device_row(i,
/// a)` is asked first for every slow-path row and may sample it on the device
/// route (`ATLAS_SAMPLER_DEVICE_TOPK=1`); `Host` falls back to `fetch_row`.
pub(super) fn dflash_thinking_accept_routed(
    a: &mut ActiveSeq,
    drafts: &[u32],
    verified: &[u32],
    ctx: &ThinkSpecCtx<'_>,
    mut fetch_row: impl FnMut(usize, &mut Vec<u8>) -> bool,
    mut device_row: impl FnMut(usize, &mut ActiveSeq) -> super::device_topk_spec::DeviceRow,
    mut on_think_end: impl FnMut(&mut ActiveSeq),
) -> SpecAcceptOutcome {
    debug_assert!(a.inside_thinking);
    let seq_fast =
        fast_path_seq_eligible(a) && !ctx.adaptive_sampling && !crate::dsv41::repetition::active();
    let wave_active = think_efficiency_config().is_active();
    let logits_fnv_requires_row = super::decode_logits_seq::logits_fnv_requires_row();
    let mut row_buf: Vec<u8> = Vec::new();
    let mut num_accepted = 0usize;
    let mut consumed_rows = 0usize;
    for i in 0..verified.len() {
        let raw = verified[i];
        // Mirror plain decode (`decode_logits_seq.rs`): arm forced `</think>`
        // once `output_tokens` enters the reserved visible-answer runway when
        // no explicit `thinking_budget` is set, BEFORE `f2_window_active` and
        // `forced_injection_pending` run so fast-path greedy speculation
        // triggers `hard_override` identically to plain decode.
        let min_runway = super::confidence::min_answer_runway(a.max_output_tokens);
        let runway_exhausted = a.max_output_tokens > 0
            && a.output_tokens.len() >= a.max_output_tokens.saturating_sub(min_runway);
        if a.inside_thinking
            && !a.force_end_thinking
            && a.thinking_budget.is_none()
            && runway_exhausted
        {
            a.force_end_thinking = true;
        }
        let fast = seq_fast
            && a.thinking_tokens >= crate::scheduler::helpers::min_thinking_floor()
            && position_fast_path_ok(
                raw,
                ctx.reflection_suppress_ids,
                a.tool_call_start_token,
                wave_active,
                f2_window_active(a),
                logits_fnv_requires_row,
            );
        // Forced `</think>` / budget-nudge / repetition-nudge injection masks
        // every other logit to `-inf`, making the emitted token distribution-
        // independent at any temperature — no D2H or device top-k is needed
        // unless logprobs or an FNV row receipt is requested.
        let deterministic_forced = if !logits_fnv_requires_row && a.top_logprobs.is_none() {
            match (
                super::think_budget_nudge::forced_token(a),
                forced_injection_pending(a),
                a.think_end_token,
            ) {
                (Some(nudge), _, _) => Some(nudge),
                (None, true, Some(end)) => Some(end),
                _ => super::think_nudge::forced_token(a),
            }
        } else {
            None
        };
        let (target, logprobs) = if let Some(forced_tok) = deterministic_forced {
            (forced_tok, None)
        } else if fast {
            (raw, None)
        } else {
            let _gumbel_pos = (super::stochastic_verify_walk::spec_verify_policy()
                == super::stochastic_verify_walk::SpecVerifyPolicy::Gumbel
                && a.temperature > 0.0)
                .then(|| {
                    spark_runtime::sampler::gumbel::GumbelPosGuard::new(
                        super::stochastic_verify_walk::resolve_seq_gumbel_seed(a),
                        super::stochastic_verify_walk::seq_abs_pos(a),
                    )
                });
            if let Some(sampled) = match device_row(i, a) {
                super::device_topk_spec::DeviceRow::Sampled(sampled) => Some(sampled),
                super::device_topk_spec::DeviceRow::Host => None,
                super::device_topk_spec::DeviceRow::Failed => {
                    tracing::error!(
                        "think-spec verify: device top-k row {i} failed; finishing seq"
                    );
                    a.finished = true;
                    return SpecAcceptOutcome {
                        num_accepted,
                        bonus: None,
                        consumed_rows,
                    };
                }
            } {
                sampled
            } else {
                if !fetch_row(i, &mut row_buf) {
                    // Mirrors the fatal-error handling of the surrounding
                    // verify step: without the row we cannot re-derive the
                    // plain-path token, and committing the raw argmax would
                    // be lossy — end the sequence instead.
                    tracing::error!("think-spec verify: logits row {i} unavailable; finishing seq");
                    a.finished = true;
                    return SpecAcceptOutcome {
                        num_accepted,
                        bonus: None,
                        consumed_rows,
                    };
                }
                let vocab = row_buf.len() / 2;
                process_seq_logits(
                    a,
                    &row_buf,
                    0,
                    vocab,
                    2,
                    false,
                    a.think_end_token,
                    a.think_start_token,
                    a.tool_call_start_token,
                    a.tool_call_end_token,
                    ctx.reflection_suppress_ids,
                    ctx.adaptive_sampling,
                )
            }
        };
        // `</think>` / EOS are phase-boundary tokens: always the bonus,
        // never an accepted draft — keeps `a.last_token` on the
        // plain-path contract (committed, not yet fed to the model), so
        // the next step feeds it exactly like the plain path would.
        let boundary = a.think_end_token == Some(target) || a.eos_tokens.contains(&target);
        let draft_match = i < drafts.len() && i + 1 < verified.len() && drafts[i] == target;
        if !commit_thinking_token(
            a,
            target,
            logprobs,
            ctx.code_fence_token,
            ctx.resolved_honor_eos_inside_thinking(),
            &mut on_think_end,
        ) {
            return SpecAcceptOutcome {
                num_accepted,
                bonus: None,
                consumed_rows,
            };
        }
        consumed_rows += 1;
        if boundary || !draft_match {
            if !a.finished {
                a.last_token = target;
            }
            return SpecAcceptOutcome {
                num_accepted,
                bonus: Some(target),
                consumed_rows,
            };
        }
        if a.finished {
            return SpecAcceptOutcome {
                num_accepted: num_accepted + 1,
                bonus: None,
                consumed_rows,
            };
        }
        num_accepted += 1;
    }
    // Unreachable with a well-formed `verified` (the `i + 1 <
    // verified.len()` guard turns the last row into the bonus); an empty
    // `verified` is rejected by the production wrapper. If a logic error
    // ever lands here, the caller's `bonus: None` early-return skips the
    // SSM commit — finish the sequence so it cannot keep decoding on an
    // uncommitted state.
    debug_assert!(
        verified.is_empty(),
        "think-spec walk fell off a non-empty verified loop"
    );
    a.finished = true;
    SpecAcceptOutcome {
        num_accepted,
        bonus: None,
        consumed_rows,
    }
}

/// Commit one thinking-span token with the plain path's per-token side
/// effects (`process_decode_logits`'s thinking branch, decode_logits_step
/// ~202-253 + the EOS suppression at ~364), on top of what `emit_token`
/// already does for the speculative paths:
///
///   * `</think>`: after emission admission, reset the F2 run + fence parity
///     (the plain path does, `emit_token` doesn't); `on_think_end` notifies
///     the caller of the non-terminal `</think>` boundary (single-row
///     bootstrap takes the SSM snapshot immediately; multi-row verify defers
///     it via [`outcome_ended_thinking`] until after verify commit +
///     `sync_secondary`).
///   * EOS inside thinking: counted as a thinking token and fed back as
///     `last_token` without pushing to `output_tokens` or finishing the
///     sequence; [`maybe_implicit_close_thinking_on_eos`] then either keeps
///     thinking open (when `honor_eos_inside_thinking` is false) or closes
///     thinking and streams `ThinkingDone` (when `honor_eos_inside_thinking`
///     is true).
///   * regular token: `emit_token` (push + stream + `thinking_tokens` +
///     budget arming), then the two plain-path effects `emit_token`
///     lacks — ``` fence parity and the THINK_LOOP watchdog.
///
/// Returns whether classification occurred. Bootstrap and sampled-DFlash
/// callers retain their own lifecycle and deliberately ignore this count.
pub(super) fn commit_thinking_token(
    a: &mut ActiveSeq,
    tok: u32,
    logprobs: Option<crate::api::TokenLogprobs>,
    code_fence_token: Option<u32>,
    honor_eos_inside_thinking: bool,
    on_think_end: &mut impl FnMut(&mut ActiveSeq),
) -> bool {
    debug_assert!(a.inside_thinking);
    if retire_cancelled_speculation(a, "spec_emit") {
        return false;
    }
    if a.think_end_token == Some(tok) {
        if !emit_token_for_verify(a, tok, logprobs) {
            return false;
        }
        // Plain-path resets belong only to the successfully classified row.
        a.consecutive_confident = 0;
        a.in_code_fence = false;
        clear_stale_require_tool_call(a, a.output_tokens.len().saturating_sub(1));
        if !a.finished {
            // Plain path: `</think>` is a non-thinking committed token →
            // SSM boundary snapshot opportunity (rollback ring).
            on_think_end(a);
            check_request_timeout(a);
        }
        return true;
    }
    if a.eos_tokens.contains(&tok) {
        // In-thinking EOS is never pushed to `output_tokens` and never ends
        // the sequence directly: it counts as a thinking token and is fed back
        // as `last_token`. Below, `maybe_implicit_close_thinking_on_eos` either
        // keeps thinking open (`honor_eos_inside_thinking == false`) or closes
        // thinking and emits `ThinkingDone` (`honor_eos_inside_thinking == true`).
        // Suppressed and implicit-close EOS are absent from output_tokens, so
        // they must also be absent from blocking response logprobs.  This is
        // the serial decoder's contract; `logprobs` is intentionally dropped.
        a.thinking_tokens += 1;
        a.in_code_fence = toggle_code_fence(a.in_code_fence, tok, code_fence_token);
        if let Some(budget) = a.thinking_budget
            && a.thinking_tokens >= budget
            && !a.force_end_thinking
        {
            // A suppressed EOS is never pushed.
            super::think_budget_nudge::arm_budget_close(a, budget, 0);
        }
        super::think_budget_nudge::on_thinking_token(a, tok);
        run_think_loop_watchdog(a, false);
        clear_stale_require_tool_call(a, a.output_tokens.len());
        maybe_implicit_close_thinking_on_eos(
            a,
            tok,
            a.output_tokens.len(),
            honor_eos_inside_thinking,
        );
        return true;
    }
    if !emit_token_for_verify(a, tok, logprobs) {
        return false;
    }
    if a.finished {
        return true;
    }
    a.in_code_fence = toggle_code_fence(a.in_code_fence, tok, code_fence_token);
    run_think_loop_watchdog(a, true);
    clear_stale_require_tool_call(a, a.output_tokens.len().saturating_sub(1));
    check_request_timeout(a);
    true
}

/// Token-period THINK_LOOP watchdog — verbatim from the plain path
/// (decode_logits_step ~238-252). `exclude_last`: the plain path scans
/// `output_tokens` BEFORE the current token is pushed; `emit_token` has
/// already pushed it, so the emitted branches drop the tail element to
/// keep the scan frame identical. The suppressed-EOS branch never pushes,
/// so it scans the full vec.
fn run_think_loop_watchdog(a: &mut ActiveSeq, exclude_last: bool) {
    if !enable_thinking_loop_watchdog()
        || a.force_end_thinking
        || a.thinking_tokens < THINK_LOOP_MIN_TOKENS
        || !a.thinking_tokens.is_multiple_of(THINK_LOOP_CHECK_STRIDE)
    {
        return;
    }
    let tokens: &[u32] = if exclude_last {
        &a.output_tokens[..a.output_tokens.len().saturating_sub(1)]
    } else {
        &a.output_tokens
    };
    if detect_thinking_token_loop(tokens) {
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
}

/// Production wrapper: bind the walk to the model's resident `[K, vocab]`
/// BF16 verify-logits buffer (same lazy per-row D2H as
/// `dflash_masked_accept`). The scheduler gate should already exclude
/// FP32-logits models. If that gate and the actual resident buffer ever
/// disagree, fail closed: falling through to the legacy raw accept would
/// silently bypass all thinking interventions.
///
/// Callers must defer [`super::rollback::snapshot_boundary_if_ssm`] until after
/// the verify commit and `model.sync_secondary()` complete whenever
/// [`outcome_ended_thinking`] returns `true`.
pub(super) fn run_dflash_thinking_accept(
    model: &dyn Model,
    a: &mut ActiveSeq,
    drafts: &[u32],
    verified: &[u32],
    ctx: &ThinkSpecCtx<'_>,
) -> Option<SpecAcceptOutcome> {
    run_dflash_thinking_accept_at_row(model, a, 0, drafts, verified, ctx)
}

/// Per-row dynamic lossless guard for pre-sampled GPU Gumbel-Max tokens inside
/// `<think>`. Any state transition that could alter the target distribution or
/// force a token invalidates pre-sampled tokens for this row and all remaining
/// rows of the sequence so the walk falls back cleanly to the exact per-row
/// sampler.
pub(super) fn presampled_thinking_row_ok(
    a: &ActiveSeq,
    tok: u32,
    vocab: usize,
    expected_abs_pos: u64,
    ctx: &ThinkSpecCtx<'_>,
) -> bool {
    if (tok as usize) >= vocab {
        return false;
    }
    if !a.inside_thinking || a.think_ended {
        return false;
    }
    if super::stochastic_verify_walk::seq_abs_pos(a) != expected_abs_pos {
        return false;
    }
    if a.force_end_thinking
        || a.budget_nudge.is_some()
        || super::think_budget_nudge::forced_token(a).is_some()
        || super::think_nudge::forced_token(a).is_some()
        || super::think_budget_nudge::close_allowed(a)
    {
        return false;
    }
    if a.thinking_budget.is_none()
        && a.max_output_tokens > 0
        && a.output_tokens.len()
            >= a.max_output_tokens
                .saturating_sub(super::confidence::min_answer_runway(a.max_output_tokens))
    {
        return false;
    }
    if f2_window_active(a) {
        return false;
    }
    let floor = crate::scheduler::helpers::min_thinking_floor();
    if floor > 0 && a.thinking_tokens < floor && a.think_end_token.is_some() {
        return false;
    }
    if !ctx.reflection_suppress_ids.is_empty()
        || ctx.adaptive_sampling
        || crate::scheduler::think_efficiency_config().is_active()
    {
        return false;
    }
    if a.tool_call_start_token == Some(tok) {
        return false;
    }
    true
}

pub(super) fn run_dflash_thinking_accept_at_row(
    model: &dyn Model,
    a: &mut ActiveSeq,
    row_base: usize,
    drafts: &[u32],
    verified: &[u32],
    ctx: &ThinkSpecCtx<'_>,
) -> Option<SpecAcceptOutcome> {
    run_dflash_thinking_accept_at_row_with_presampled(
        model, a, row_base, drafts, verified, ctx, None,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn run_dflash_thinking_accept_at_row_with_presampled(
    model: &dyn Model,
    a: &mut ActiveSeq,
    row_base: usize,
    drafts: &[u32],
    verified: &[u32],
    ctx: &ThinkSpecCtx<'_>,
    presampled: Option<&[u32]>,
) -> Option<SpecAcceptOutcome> {
    if verified.is_empty() {
        tracing::error!("ATLAS_THINK_SPEC: empty target result; finishing seq");
        a.finished = true;
        return Some(SpecAcceptOutcome {
            num_accepted: 0,
            bonus: None,
            consumed_rows: 0,
        });
    }
    let logits_base = model.logits_buffer_ptr();
    if model.logits_ptr_is_fp32(logits_base) {
        // The scheduler gate clears `ctx.enabled` for FP32-logits models.
        // Reaching here means the gate and resident buffer disagree. Do not
        // return None: the caller interprets that as permission to use the
        // legacy raw accept, which is lossy inside thinking.
        tracing::error!("ATLAS_THINK_SPEC: FP32 verify logits despite BF16 gate; finishing seq");
        a.finished = true;
        return Some(SpecAcceptOutcome {
            num_accepted: 0,
            bonus: None,
            consumed_rows: 0,
        });
    }
    let local_presampled = if presampled.is_none() {
        match super::spec_policy_accept::presample_single_seq_gumbel(
            model,
            a,
            row_base,
            verified.len(),
            ctx,
        ) {
            Ok(tokens) => tokens,
            Err(error) => {
                fail_speculative_step(a, "batched gumbel thinking verify", error);
                return Some(SpecAcceptOutcome {
                    num_accepted: 0,
                    bonus: None,
                    consumed_rows: 0,
                });
            }
        }
    } else {
        None
    };
    let effective_presampled = presampled.or(local_presampled.as_deref());
    if effective_presampled.is_none()
        && super::stochastic_verify_walk::spec_verify_policy()
            == super::stochastic_verify_walk::SpecVerifyPolicy::Gumbel
        && !super::device_topk_route::seq_uses_device_topk(
            model,
            a,
            false,
            ctx.adaptive_sampling,
            ctx.reflection_suppress_ids,
        )
        && super::stochastic_verify_walk::gumbel_thinking_seq_eligible(a, ctx)
    {
        return Some(
            super::stochastic_verify_walk::run_gumbel_thinking_accept_at_row(
                model,
                a,
                row_base,
                drafts,
                verified.len(),
                ctx,
            ),
        );
    }
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        tracing::info!(
            "ATLAS_THINK_SPEC: policy-aware speculative decode active inside <think> spans \
             (native MTP default; DFlash env opt-in; post-verify plain-path accept filter)"
        );
    });
    let vocab = model.vocab_size();
    let base_abs_pos = super::stochastic_verify_walk::seq_abs_pos(a);
    let mut presampled_active = effective_presampled.is_some_and(|p| p.len() >= verified.len());
    let mut download_error = None;
    let outcome = dflash_thinking_accept_routed(
        a,
        drafts,
        verified,
        ctx,
        |i, buf| {
            buf.resize(vocab * 2, 0);
            match model.copy_logits_to_host(logits_base.offset((row_base + i) * vocab * 2), buf) {
                Ok(()) => true,
                Err(e) => {
                    tracing::error!("think-spec verify: logits D2H failed at row {i}: {e:#}");
                    download_error = Some(e);
                    false
                }
            }
        },
        |i, a: &mut ActiveSeq| {
            if presampled_active {
                if let Some(tokens) = effective_presampled
                    && let Some(&tok) = tokens.get(i)
                    && presampled_thinking_row_ok(
                        a,
                        tok,
                        vocab,
                        base_abs_pos.saturating_add(i as u64),
                        ctx,
                    )
                {
                    return super::device_topk_spec::DeviceRow::Sampled((tok, None));
                }
                presampled_active = false;
            }
            let _gumbel_pos = (super::stochastic_verify_walk::spec_verify_policy()
                == super::stochastic_verify_walk::SpecVerifyPolicy::Gumbel
                && a.temperature > 0.0)
                .then(|| {
                    spark_runtime::sampler::gumbel::GumbelPosGuard::new(
                        super::stochastic_verify_walk::resolve_seq_gumbel_seed(a),
                        super::stochastic_verify_walk::seq_abs_pos(a),
                    )
                });
            if model.device_topk_supported() {
                super::device_topk_spec::sample_spec_row(
                    model,
                    a,
                    logits_base.offset((row_base + i) * vocab * 2),
                    false,
                    ctx,
                )
            } else {
                super::device_topk_spec::DeviceRow::Host
            }
        },
        // Defer the SSM boundary snapshot until the caller commits the accepted
        // prefix and drains `sync_secondary()` (checked via `outcome_ended_thinking`):
        // live recurrent state still holds the over-extended verify tail here.
        |_| {},
    );
    if let Some(error) = download_error {
        fail_speculative_step(a, "thinking verify logits download", error);
    }
    super::stochastic_verify_walk::publish_seq_gumbel_draft_spec(a);
    Some(outcome)
}

/// Bootstrap-decode sibling of the verify filter: sample ONE token from a
/// single-row decode-logits buffer through the full plain-path pipeline
/// (`process_seq_logits`) and commit it with the thinking side effects.
/// Used by `step_mtp`'s Phase A when a thinking sequence has no pending
/// drafts — byte-identical to one `step_decode_only` token for this seq.
///
/// Returns the committed token, or `None` on a fatal D2H error (the
/// sequence is finished, mirroring the bootstrap decode error handling).
pub(super) fn bootstrap_thinking_token(
    model: &dyn Model,
    a: &mut ActiveSeq,
    logits: DevicePtr,
    ctx: &ThinkSpecCtx<'_>,
) -> Option<u32> {
    let _gumbel_pos = (super::stochastic_verify_walk::spec_verify_policy()
        == super::stochastic_verify_walk::SpecVerifyPolicy::Gumbel
        && a.temperature > 0.0)
        .then(|| {
            spark_runtime::sampler::gumbel::GumbelPosGuard::new(
                super::stochastic_verify_walk::resolve_seq_gumbel_seed(a),
                super::stochastic_verify_walk::seq_abs_pos(a),
            )
        });
    let device = if model.device_topk_supported() {
        super::device_topk_spec::sample_spec_row(model, a, logits, false, ctx)
    } else {
        super::device_topk_spec::DeviceRow::Host
    };
    let (tok, lp) = match device {
        super::device_topk_spec::DeviceRow::Sampled(sampled) => sampled,
        super::device_topk_spec::DeviceRow::Failed => {
            a.finished = true;
            return None;
        }
        super::device_topk_spec::DeviceRow::Host => bootstrap_thinking_host(model, a, logits, ctx)?,
    };
    commit_thinking_token(
        a,
        tok,
        lp,
        ctx.code_fence_token,
        ctx.resolved_honor_eos_inside_thinking(),
        &mut |a| rollback::snapshot_boundary_if_ssm(a, model),
    );
    Some(tok)
}

/// The host half of [`bootstrap_thinking_token`]: full-row D2H + plain path.
fn bootstrap_thinking_host(
    model: &dyn Model,
    a: &mut ActiveSeq,
    logits: DevicePtr,
    ctx: &ThinkSpecCtx<'_>,
) -> Option<(u32, Option<crate::api::TokenLogprobs>)> {
    let vocab = model.vocab_size();
    let mut buf = vec![0u8; vocab * 2];
    if let Err(e) = model.copy_logits_to_host(logits, &mut buf) {
        tracing::error!("think-spec bootstrap: logits D2H failed: {e:#}");
        a.finished = true;
        return None;
    }
    Some(process_seq_logits(
        a,
        &buf,
        0,
        vocab,
        2,
        false,
        a.think_end_token,
        a.think_start_token,
        a.tool_call_start_token,
        a.tool_call_end_token,
        ctx.reflection_suppress_ids,
        ctx.adaptive_sampling,
    ))
}

#[cfg(test)]
#[path = "think_spec_accept_tests.rs"]
mod think_spec_accept_tests;

#[cfg(test)]
mod runway_override_tests {
    use super::*;

    fn mid_fence_greedy_seq(
        max_output: usize,
        output_len: usize,
        thinking_budget: Option<u32>,
        thinking_tokens: u32,
        force_end_thinking: bool,
    ) -> ActiveSeq {
        let mut a = crate::scheduler::test_fixtures::active_seq(ResponseSink::Blocking(None), 0);
        a.inside_thinking = true;
        a.enable_thinking = true;
        a.think_ended = false;
        a.think_end_token = Some(29);
        a.max_output_tokens = max_output;
        a.remaining = max_output;
        a.thinking_budget = thinking_budget;
        a.thinking_tokens = thinking_tokens;
        a.force_end_thinking = force_end_thinking;
        a.in_code_fence = true;
        a.output_tokens = vec![1; output_len];
        a
    }

    #[test]
    fn greedy_speculative_thinking_mid_fence_honors_answer_runway_override() {
        const MAX_OUTPUT: usize = 4_096;
        let cutoff = MAX_OUTPUT - crate::scheduler::confidence::min_answer_runway(MAX_OUTPUT);
        let ctx = ThinkSpecCtx {
            enabled: true,
            code_fence_token: Some(3),
            reflection_suppress_ids: &[],
            adaptive_sampling: false,
            honor_eos_inside_thinking: Some(false),
        };

        // Case 1: explicit budget whose 3x defer ceiling exceeds max_output_tokens.
        let mut budgeted =
            mid_fence_greedy_seq(MAX_OUTPUT, cutoff, Some(3_072), cutoff as u32, true);
        let mut ended = false;
        let out = dflash_thinking_accept(
            &mut budgeted,
            &[1, 1],
            &[1, 1, 1],
            &ctx,
            |_, _| panic!("greedy runway override must stay on deterministic fast path"),
            |_| ended = true,
        );
        assert_eq!(out.num_accepted, 0);
        assert_eq!(out.bonus, Some(29));
        assert!(outcome_ended_thinking(&budgeted, &out));
        assert!(!budgeted.inside_thinking && budgeted.think_ended && ended);
        assert_eq!(budgeted.output_tokens.len(), cutoff + 1);

        // Case 2: no explicit thinking_budget and force_end_thinking not yet armed;
        // runway exhaustion arms force_end_thinking and overrides fence deferral
        // identically to plain decode (`decode_logits_seq.rs`).
        let mut unbounded = mid_fence_greedy_seq(MAX_OUTPUT, cutoff, None, 100, false);
        let mut unbounded_ended = false;
        let out = dflash_thinking_accept(
            &mut unbounded,
            &[1, 1],
            &[1, 1, 1],
            &ctx,
            |_, _| panic!("greedy unbounded runway override must not fetch logits row"),
            |_| unbounded_ended = true,
        );
        assert_eq!(out.num_accepted, 0);
        assert_eq!(out.bonus, Some(29));
        assert!(outcome_ended_thinking(&unbounded, &out));
        assert!(!unbounded.inside_thinking && unbounded.think_ended && unbounded_ended);
        assert_eq!(unbounded.output_tokens.len(), cutoff + 1);
    }

    #[test]
    fn verify_thinking_wrappers_defer_ssm_snapshot_until_after_commit() {
        // Multi-row speculative verify wrappers must never snapshot SSM state
        // inside the accept walk (while live recurrent state still holds the
        // verify kernel's rejected tail). Only single-row bootstrap decode may
        // snapshot immediately.
        let think_src = include_str!("think_spec_accept.rs");
        let run_wrapper = think_src
            .split("pub(super) fn run_dflash_thinking_accept_at_row_with_presampled(")
            .nth(1)
            .unwrap()
            .split("pub(super) fn bootstrap_thinking_token(")
            .next()
            .unwrap();
        assert!(
            !run_wrapper.contains("snapshot_boundary_if_ssm"),
            "run_dflash_thinking_accept_at_row_with_presampled must defer SSM snapshot"
        );
        let bootstrap = think_src
            .split("pub(super) fn bootstrap_thinking_token(")
            .nth(1)
            .unwrap()
            .split("#[cfg(test)]")
            .next()
            .unwrap();
        assert!(
            bootstrap.contains("snapshot_boundary_if_ssm"),
            "single-row bootstrap_thinking_token must keep immediate SSM snapshot"
        );

        let policy_src = include_str!("spec_policy_accept.rs");
        let rej_wrapper = policy_src
            .split("pub(super) fn run_dflash_thinking_rejection_accept_at_row(")
            .nth(1)
            .unwrap()
            .split("#[cfg(test)]")
            .next()
            .unwrap();
        assert!(
            !rej_wrapper.contains("snapshot_boundary_if_ssm"),
            "run_dflash_thinking_rejection_accept_at_row must defer SSM snapshot"
        );

        let stoch_src = include_str!("stochastic_verify_walk.rs");
        let gumbel_wrapper = stoch_src
            .split("pub(super) fn run_gumbel_thinking_accept_at_row(")
            .nth(1)
            .unwrap()
            .split("#[cfg(test)]")
            .next()
            .unwrap();
        assert!(
            !gumbel_wrapper.contains("snapshot_boundary_if_ssm"),
            "run_gumbel_thinking_accept_at_row must defer SSM snapshot"
        );
    }
}
