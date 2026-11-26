// SPDX-License-Identifier: AGPL-3.0-only

//! `FLUX_SPEC_VERIFY=stochastic|gumbel|rejection`: sequence-level speculative accept walks.
//!
//! - `stochastic`: standard rejection-sampling / seeded sample-then-match rule
//!   in [`super::stochastic_verify`] for eligible content positions.
//! - `gumbel`: stateless `(seed, abs_pos, token_id)` SplitMix64 Gumbel-Max rule
//!   ([`spark_runtime::sampler::choose_gumbel`]) across both content AND
//!   `<think>` positions, guaranteeing 100% bit-identical output to `K=1`
//!   serial decoding at any `temperature > 0` via simple `draft == target`
//!   equality and bulk single-copy host verification.
//! - `rejection`: GPU maximal-coupling rejection sampling
//!   (`target_rejection_verify_rows_bf16`) pairing device draft `(q_cand_ids, q_probs)`
//!   capture with exact target post-policy acceptance and residual sampling across
//!   both content and `<think>` spans.

use super::spec_policy_accept::VerifyLogitsFormat;
use super::stochastic_verify::{
    DraftDistribution, PositionDraft, PositionVerdict, position_uniforms, target_probabilities,
    verify_position,
};
use super::*;

/// The verify policy chosen by `FLUX_SPEC_VERIFY`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum SpecVerifyPolicy {
    /// Default rule: argmax / standard policy walk.
    Argmax,
    /// Rejection sampling against the target's post-policy distribution.
    Stochastic,
    /// Position + token-ID keyed Gumbel-Max (`draft == choose_gumbel`).
    Gumbel,
    /// GPU maximal-coupling rejection sampling (`target_rejection_verify_rows_bf16`).
    Rejection,
}

/// Parse `FLUX_SPEC_VERIFY`. Anything other than `stochastic`, `gumbel`, or `rejection` keeps `Argmax`.
pub(super) fn parse_spec_verify_policy(value: Option<&str>) -> SpecVerifyPolicy {
    if spark_runtime::sampler::gumbel::is_rejection_verify_env(value) {
        return SpecVerifyPolicy::Rejection;
    }
    match value.map(str::trim) {
        Some("stochastic") => SpecVerifyPolicy::Stochastic,
        Some("gumbel") => SpecVerifyPolicy::Gumbel,
        Some("") | Some("argmax") | None => SpecVerifyPolicy::Argmax,
        Some(other) => {
            tracing::warn!(
                "FLUX_SPEC_VERIFY={other:?} is not argmax|stochastic|gumbel|rejection; using argmax"
            );
            SpecVerifyPolicy::Argmax
        }
    }
}

/// The process-wide policy, read once.
pub(super) fn spec_verify_policy() -> SpecVerifyPolicy {
    static POLICY: std::sync::OnceLock<SpecVerifyPolicy> = std::sync::OnceLock::new();
    *POLICY.get_or_init(|| {
        let raw = std::env::var("FLUX_SPEC_VERIFY")
            .or_else(|_| std::env::var("ATLAS_SPEC_VERIFY"))
            .ok();
        let policy = parse_spec_verify_policy(raw.as_deref());
        match policy {
            SpecVerifyPolicy::Stochastic => {
                tracing::info!("FLUX_SPEC_VERIFY_ENGAGED policy=stochastic rule=sample-then-match");
            }
            SpecVerifyPolicy::Gumbel => {
                tracing::info!("FLUX_SPEC_VERIFY_ENGAGED policy=gumbel rule=keyed-gumbel-max");
            }
            SpecVerifyPolicy::Rejection => {
                tracing::info!(
                    "FLUX_SPEC_VERIFY_ENGAGED policy=rejection rule=gpu-maximal-coupling"
                );
            }
            SpecVerifyPolicy::Argmax => {}
        }
        policy
    })
}

/// Resolve the request-scoped base seed for `a` (honoring `FLUX_PROMPT_SEED=1`).
pub(super) fn resolve_seq_gumbel_seed(a: &ActiveSeq) -> u64 {
    if let Some(seed) = a.seed {
        return seed;
    }
    let prompt_end = a.seq.prompt_len.min(a.seq.tokens.len());
    let prompt_slice = &a.seq.tokens[..prompt_end];
    spark_runtime::sampler::seed_from_prompt_tokens(prompt_slice)
}

/// Exact committed absolute position `prompt_len + output_tokens.len()` for `a`'s next sample.
#[inline]
pub(super) fn seq_abs_pos(a: &ActiveSeq) -> u64 {
    (a.seq.prompt_len + a.output_tokens.len()) as u64
}

/// Register `a`'s Gumbel / GPU maximal-coupling rejection draft spec before the
/// upcoming MTP or DFlash proposal, or explicitly clear/invalidate stale specs
/// and rejection-slot stamps when greedy (`temperature <= 0.0`), finished, or
/// outside `FLUX_SPEC_VERIFY=gumbel|rejection`.
pub(super) fn publish_seq_gumbel_draft_spec(a: &ActiveSeq) {
    publish_seq_gumbel_draft_spec_with_policy(a, spec_verify_policy());
}

/// Pure computation of `(GumbelDraftSpec, pending_rejection_slot)` for `a` under
/// `policy` (returns `(None, None)` for greedy, finished, or non-Gumbel/Rejection policies).
#[must_use]
fn resolve_seq_draft_spec_and_slot(
    a: &ActiveSeq,
    policy: SpecVerifyPolicy,
) -> (
    Option<spark_runtime::sampler::gumbel::GumbelDraftSpec>,
    Option<usize>,
) {
    if !matches!(
        policy,
        SpecVerifyPolicy::Gumbel | SpecVerifyPolicy::Rejection
    ) || !a.temperature.is_finite()
        || a.temperature <= 0.0
        || a.finished
    {
        return (None, None);
    }
    let params = a.sampling_params(a.temperature);
    let spec = spark_runtime::sampler::gumbel::GumbelDraftSpec {
        seed: resolve_seq_gumbel_seed(a),
        temperature: params.temperature,
        top_k: params.top_k,
        top_p: params.top_p,
        min_p: params.min_p,
        step0_abs_pos: seq_abs_pos(a),
    };
    let rejection_slot = (policy == SpecVerifyPolicy::Rejection).then_some(a.seq.slot_idx);
    (Some(spec), rejection_slot)
}

pub(super) fn publish_seq_gumbel_draft_spec_with_policy(a: &ActiveSeq, policy: SpecVerifyPolicy) {
    let propose_pos = (a.seq.prompt_len + a.output_tokens.len()).saturating_sub(1);
    let stamp_proposer_state = |spec: Option<spark_runtime::sampler::gumbel::GumbelDraftSpec>,
                                rejection_slot: Option<usize>| {
        if let Some(prop_state) = a.seq.proposer_state.as_ref() {
            if let Some(mtp_state) = prop_state
                .as_any()
                .downcast_ref::<spark_model::layers::qwen4_mtp::Qwen4MtpState>(
            ) {
                mtp_state.set_gumbel_spec(spec);
            } else if let Some(dflash_state) = prop_state
                .as_any()
                .downcast_ref::<spark_model::layers::dflash_head::DflashProposerState>(
            ) {
                dflash_state.set_gumbel_spec(spec);
                dflash_state.set_pending_rejection_slot(rejection_slot);
            }
        }
    };
    if !matches!(
        policy,
        SpecVerifyPolicy::Gumbel | SpecVerifyPolicy::Rejection
    ) {
        stamp_proposer_state(None, None);
        spark_runtime::sampler::gumbel::clear_gumbel_draft_specs();
        return;
    }
    let (Some(spec), rejection_slot) = resolve_seq_draft_spec_and_slot(a, policy) else {
        stamp_proposer_state(None, None);
        spark_runtime::sampler::gumbel::invalidate_gumbel_draft_spec(a.seq.slot_idx, propose_pos);
        if a.seq.seq_len != propose_pos {
            spark_runtime::sampler::gumbel::invalidate_gumbel_draft_spec(
                a.seq.slot_idx,
                a.seq.seq_len,
            );
        }
        return;
    };
    stamp_proposer_state(Some(spec), rejection_slot);
    spark_runtime::sampler::gumbel::register_gumbel_draft_spec_for_slot(
        a.seq.slot_idx,
        propose_pos,
        spec,
    );
    if a.seq.seq_len != propose_pos {
        spark_runtime::sampler::gumbel::register_gumbel_draft_spec_for_slot(
            a.seq.slot_idx,
            a.seq.seq_len,
            spec,
        );
    }
}

/// Read the active GPU rejection-sampling `(slot_idx, step0_abs_pos, valid_len)`
/// fence recorded on `a`'s [`spark_model::layers::dflash_head::DflashProposerState`].
#[inline]
pub(super) fn seq_rejection_q_fence(a: &ActiveSeq) -> Option<(usize, u64, usize)> {
    a.seq
        .proposer_state
        .as_ref()
        .and_then(|s| {
            s.as_any()
                .downcast_ref::<spark_model::layers::dflash_head::DflashProposerState>()
        })
        .and_then(spark_model::layers::dflash_head::DflashProposerState::last_rejection_q)
}

/// Invalidate any pending GPU rejection-sampling `(q_cand_ids, q_probs)` fence
/// on `a`'s proposer state so stale or externally replaced drafts never read
/// mismatched device `q` rows.
#[inline]
pub(super) fn invalidate_seq_rejection_q(a: &ActiveSeq) {
    if let Some(dflash_state) = a.seq.proposer_state.as_ref().and_then(|s| {
        s.as_any()
            .downcast_ref::<spark_model::layers::dflash_head::DflashProposerState>()
    }) {
        dflash_state.invalidate_rejection_q();
    }
}

/// Clamp any pending GPU rejection-sampling `(q_cand_ids, q_probs)` fence on
/// `a`'s proposer state to at most `max_len` prefix draft rows (`0` invalidates).
#[inline]
pub(super) fn truncate_seq_rejection_q(a: &ActiveSeq, max_len: usize) {
    if let Some(dflash_state) = a.seq.proposer_state.as_ref().and_then(|s| {
        s.as_any()
            .downcast_ref::<spark_model::layers::dflash_head::DflashProposerState>()
    }) {
        if max_len == 0 {
            dflash_state.invalidate_rejection_q();
        } else {
            dflash_state.truncate_rejection_q(max_len);
        }
    }
}

/// First index where `before` and `after` diverge (or `None` when identical).
/// Used to clamp the GPU rejection `q` fence to the unmodified prefix after an
/// in-place draft splice (`cfg_jf`, forced budget nudge, or grammar splice).
#[inline]
#[must_use]
pub(super) fn first_spliced_draft_index(before: &[u32], after: &[u32]) -> Option<usize> {
    let common = before.len().min(after.len());
    for i in 0..common {
        if before[i] != after[i] {
            return Some(i);
        }
    }
    (before.len() != after.len()).then_some(common)
}

/// Truncate (or invalidate if index 0 changed) `a`'s GPU rejection `q` fence
/// to the unmodified prefix before the first spliced draft position.
#[inline]
pub(super) fn truncate_seq_rejection_q_after_splice(a: &ActiveSeq, before: &[u32], after: &[u32]) {
    if let Some(first_diff) = first_spliced_draft_index(before, after) {
        truncate_seq_rejection_q(a, first_diff);
    }
}

/// Whether this content position may use the stochastic/gumbel content walk.
pub(super) fn stochastic_seq_eligible(a: &ActiveSeq, ctx: &ThinkSpecCtx<'_>) -> bool {
    a.temperature > 0.0
        && !ctx.adaptive_sampling
        && !a.inside_thinking
        && !a.think_just_ended
        && !a.suppress_tool_call
        && !a.require_tool_call
        && !a.inside_tool_body
        && a.grammar_state.is_none()
        && a.top_logprobs.is_none()
        && a.output_tokens.len() >= a.min_tokens
        && super::think_nudge::forced_token(a).is_none()
        && !super::decode_logits_seq::logits_fnv_requires_row()
        && !crate::dsv41::serving()
}

/// Whether an `inside_thinking` sequence may use fast bulk-D2H keyed Gumbel verification.
pub(super) fn gumbel_thinking_seq_eligible(a: &ActiveSeq, ctx: &ThinkSpecCtx<'_>) -> bool {
    a.temperature > 0.0
        && a.inside_thinking
        && !ctx.adaptive_sampling
        && a.top_logprobs.is_none()
        && super::think_nudge::forced_token(a).is_none()
        && super::think_budget_nudge::forced_token(a).is_none()
        && !crate::scheduler::think_efficiency_config().is_active()
        && !super::decode_logits_seq::logits_fnv_requires_row()
        && !crate::dsv41::serving()
}

/// Whether `a` (either content or `inside_thinking`) is eligible under `FLUX_SPEC_VERIFY=gumbel`.
#[cfg_attr(not(test), allow(dead_code))]
pub(super) fn gumbel_seq_eligible(a: &ActiveSeq, ctx: &ThinkSpecCtx<'_>) -> bool {
    if a.inside_thinking {
        gumbel_thinking_seq_eligible(a, ctx)
    } else {
        stochastic_seq_eligible(a, ctx)
    }
}

/// Widen one verify row to FP32 bytes and apply the post-thinking hard mask.
pub(super) fn masked_f32_row(
    row: &[u8],
    format: VerifyLogitsFormat,
    vocab: usize,
    masked: &[u32],
) -> Option<Vec<u8>> {
    let logits: Vec<f32> = match format {
        VerifyLogitsFormat::Fp32 => row
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect(),
        VerifyLogitsFormat::Bf16 => row
            .chunks_exact(2)
            .map(|b| super::helpers::bf16_to_f32(b[0], b[1]))
            .collect(),
    };
    if logits.len() != vocab {
        return None;
    }
    Some(
        logits
            .iter()
            .enumerate()
            .map(|(id, &v)| {
                if masked.contains(&(id as u32)) {
                    f32::NEG_INFINITY
                } else {
                    v
                }
            })
            .flat_map(f32::to_le_bytes)
            .collect(),
    )
}

/// Ids the plain path hard-masks at a content position.
fn content_mask(a: &ActiveSeq) -> Vec<u32> {
    if !a.think_ended {
        return Vec::new();
    }
    [a.think_end_token, a.think_start_token]
        .into_iter()
        .flatten()
        .collect()
}

/// Widen `row` into `out` in place and apply `masked` in `O(masked.len())`.
fn widen_and_mask_f32_into(
    row: &[u8],
    format: VerifyLogitsFormat,
    vocab: usize,
    masked: &[u32],
    out: &mut Vec<f32>,
) -> bool {
    out.clear();
    out.reserve(vocab);
    match format {
        VerifyLogitsFormat::Fp32 => {
            if row.len() != vocab * 4 {
                return false;
            }
            out.extend(
                row.chunks_exact(4)
                    .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])),
            );
        }
        VerifyLogitsFormat::Bf16 => {
            if row.len() != vocab * 2 {
                return false;
            }
            out.extend(
                row.chunks_exact(2)
                    .map(|b| super::helpers::bf16_to_f32(b[0], b[1])),
            );
        }
    }
    for &id in masked {
        if let Some(slot) = out.get_mut(id as usize) {
            *slot = f32::NEG_INFINITY;
        }
    }
    true
}

/// The token one content walk position commits, and whether it accepted `draft`.
fn choose_token_with_policy(
    a: &ActiveSeq,
    row: &[u8],
    format: VerifyLogitsFormat,
    vocab: usize,
    draft: Option<PositionDraft<'_>>,
    policy: SpecVerifyPolicy,
    f32_scratch: &mut Vec<f32>,
) -> Result<(u32, bool)> {
    let mask = content_mask(a);
    let params = a.sampling_params(a.temperature);
    if policy == SpecVerifyPolicy::Gumbel {
        if !widen_and_mask_f32_into(row, format, vocab, &mask, f32_scratch) {
            anyhow::bail!("stochastic verify: malformed logits row");
        }
        let seed = resolve_seq_gumbel_seed(a);
        let abs_pos = seq_abs_pos(a);
        let token = spark_runtime::sampler::choose_gumbel_f32_in_place(
            f32_scratch,
            &params,
            &a.output_tokens,
            seed,
            abs_pos,
        );
        return Ok((token, draft.is_some_and(|d| d.token == token)));
    }
    let bytes = masked_f32_row(row, format, vocab, &mask)
        .ok_or_else(|| anyhow::anyhow!("stochastic verify: malformed logits row"))?;
    match draft {
        None
        | Some(PositionDraft {
            q: DraftDistribution::PointMass,
            ..
        }) => {
            let token = sample_with_params_history(&bytes, &params, &a.output_tokens);
            Ok((token, draft.is_some_and(|d| d.token == token)))
        }
        Some(d) => {
            let p = target_probabilities(&bytes, &params, &a.output_tokens)?;
            match verify_position(&p, &d, position_uniforms(params.seed))? {
                PositionVerdict::Accept => Ok((d.token, true)),
                PositionVerdict::Reject { token, .. } => Ok((token, false)),
            }
        }
    }
}

fn finish_with(a: &mut ActiveSeq, num_accepted: usize, consumed_rows: usize) -> SpecAcceptOutcome {
    a.finished = true;
    SpecAcceptOutcome {
        num_accepted,
        bonus: None,
        consumed_rows,
    }
}

/// Walk content verify rows with the active policy (`Stochastic` or `Gumbel`).
pub(super) fn stochastic_content_accept<'q>(
    a: &mut ActiveSeq,
    drafts: &[u32],
    verified_rows: usize,
    format: VerifyLogitsFormat,
    vocab: usize,
    q_at: impl Fn(usize) -> DraftDistribution<'q>,
    fetch_row: impl FnMut(usize, &mut Vec<u8>) -> bool,
) -> SpecAcceptOutcome {
    stochastic_content_accept_with_policy(
        a,
        drafts,
        verified_rows,
        format,
        vocab,
        spec_verify_policy(),
        q_at,
        fetch_row,
    )
}

/// Parameterized content walk (tests drive both `Stochastic` and `Gumbel` directly).
#[allow(clippy::too_many_arguments)]
pub(super) fn stochastic_content_accept_with_policy<'q>(
    a: &mut ActiveSeq,
    drafts: &[u32],
    verified_rows: usize,
    format: VerifyLogitsFormat,
    vocab: usize,
    policy: SpecVerifyPolicy,
    q_at: impl Fn(usize) -> DraftDistribution<'q>,
    mut fetch_row: impl FnMut(usize, &mut Vec<u8>) -> bool,
) -> SpecAcceptOutcome {
    let mut row = Vec::new();
    let mut f32_scratch = Vec::with_capacity(if policy == SpecVerifyPolicy::Gumbel {
        vocab
    } else {
        0
    });
    let mut num_accepted = 0usize;
    let mut consumed_rows = 0usize;
    for i in 0..verified_rows {
        if !fetch_row(i, &mut row) {
            tracing::error!("stochastic verify: logits row {i} unavailable; finishing seq");
            return finish_with(a, num_accepted, consumed_rows);
        }
        let draft = (i < drafts.len() && i + 1 < verified_rows).then(|| PositionDraft {
            token: drafts[i],
            q: q_at(i),
        });
        let (token, accepted) =
            match choose_token_with_policy(a, &row, format, vocab, draft, policy, &mut f32_scratch)
            {
                Ok(choice) => choice,
                Err(error) => {
                    tracing::error!("{error:#}; finishing seq");
                    return finish_with(a, num_accepted, consumed_rows);
                }
            };
        let was_inside_thinking = a.inside_thinking;
        if !emit_token_for_verify(a, token, None) {
            return SpecAcceptOutcome {
                num_accepted,
                bonus: None,
                consumed_rows,
            };
        }
        consumed_rows += 1;
        let phase_boundary = was_inside_thinking != a.inside_thinking
            || a.think_start_token == Some(token)
            || a.think_end_token == Some(token);
        if phase_boundary || !accepted {
            if !a.finished {
                a.last_token = token;
            }
            return SpecAcceptOutcome {
                num_accepted,
                bonus: Some(token),
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
    finish_with(a, num_accepted, consumed_rows)
}

/// Fast keyed Gumbel thinking-span accept walk reproducing the exact `process_seq_logits`
/// interventions (forced budget/think nudge, F1 reflection suppression, min-thinking
/// floor, F2 confidence early stop, `<tool_call>` hard mask, and `close_allowed` `</think>`
/// injection) and committing via [`super::think_spec_accept::commit_thinking_token`].
pub(super) fn gumbel_thinking_accept(
    a: &mut ActiveSeq,
    drafts: &[u32],
    verified_rows: usize,
    format: VerifyLogitsFormat,
    vocab: usize,
    ctx: &ThinkSpecCtx<'_>,
    mut fetch_row: impl FnMut(usize, &mut Vec<u8>) -> bool,
    mut on_think_end: impl FnMut(&mut ActiveSeq),
) -> SpecAcceptOutcome {
    let mut row = Vec::new();
    let mut f32_logits = Vec::new();
    let mut num_accepted = 0usize;
    let mut consumed_rows = 0usize;
    let honor_eos = ctx.resolved_honor_eos_inside_thinking();

    for i in 0..verified_rows {
        // Mirror plain decode (`decode_logits_seq.rs`) and `dflash_thinking_accept`:
        // arm forced `</think>` once `output_tokens` enters the reserved
        // visible-answer runway when no explicit `thinking_budget` is set, BEFORE
        // `close_allowed` and `f2_window_active` run.
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
        // Forced budget-nudge tokens, armed `</think>` close (when `force_end_thinking`
        // is already set, so `f2_window_active` is false), and repetition-nudge tokens
        // are distribution-independent — short-circuit BEFORE any logits D2H or F32
        // widening allocation.
        let forced = super::think_budget_nudge::forced_token(a)
            .or_else(|| {
                a.think_end_token.filter(|&tok| {
                    (tok as usize) < vocab && super::think_budget_nudge::close_allowed(a)
                })
            })
            .or_else(|| super::think_nudge::forced_token(a));
        let target = if let Some(tok) = forced {
            tok
        } else {
            if !fetch_row(i, &mut row) {
                tracing::error!(
                    "gumbel thinking verify: logits row {i} unavailable; finishing seq"
                );
                return finish_with(a, num_accepted, consumed_rows);
            }
            f32_logits.clear();
            if f32_logits.capacity() < vocab {
                f32_logits.reserve(vocab - f32_logits.len());
            }
            match format {
                VerifyLogitsFormat::Fp32 => {
                    for b in row.chunks_exact(4) {
                        f32_logits.push(f32::from_le_bytes([b[0], b[1], b[2], b[3]]));
                    }
                }
                VerifyLogitsFormat::Bf16 => {
                    for b in row.chunks_exact(2) {
                        f32_logits.push(super::helpers::bf16_to_f32(b[0], b[1]));
                    }
                }
            }
            if f32_logits.len() != vocab {
                tracing::error!("gumbel thinking verify: malformed logits row {i}; finishing seq");
                return finish_with(a, num_accepted, consumed_rows);
            }

            // 1. Pre-F2 thinking edits (F1 reflection `-= 10.0` + min-thinking `</think>` = -inf).
            for (id, edit) in super::decode_logits_seq::pre_f2_edits(
                a,
                a.think_end_token,
                ctx.reflection_suppress_ids,
                vocab,
            ) {
                edit.apply(&mut f32_logits[id as usize]);
            }

            // 2. F2 confidence early-stop state evolution (identical to `process_seq_f32_logits`).
            if super::think_spec_accept::f2_window_active(a) {
                let confident = spark_runtime::sampler::f2_certify::host_f2_confident(&f32_logits);
                let (run, force_end) = confidence_run_step(confident, a.consecutive_confident);
                a.consecutive_confident = run;
                if force_end {
                    a.force_end_thinking = true;
                }
            }

            // 3. Hard-mask `<tool_call>` inside thinking.
            if let Some(tc_start) = a.tool_call_start_token
                && (tc_start as usize) < vocab
            {
                f32_logits[tc_start as usize] = f32::NEG_INFINITY;
            }

            // 4. Forced `</think>` when budget or F2 armed and outside code fence.
            if super::think_budget_nudge::close_allowed(a)
                && let Some(end_tok) = a.think_end_token
                && (end_tok as usize) < vocab
            {
                end_tok
            } else {
                let params = a.sampling_params(a.temperature);
                let seed = resolve_seq_gumbel_seed(a);
                let abs_pos = seq_abs_pos(a);
                spark_runtime::sampler::choose_gumbel_f32_in_place(
                    &mut f32_logits,
                    &params,
                    &a.output_tokens,
                    seed,
                    abs_pos,
                )
            }
        };

        let boundary = a.think_end_token == Some(target) || a.eos_tokens.contains(&target);
        let draft_match = i < drafts.len() && i + 1 < verified_rows && drafts[i] == target;
        if !super::think_spec_accept::commit_thinking_token(
            a,
            target,
            None,
            ctx.code_fence_token,
            honor_eos,
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
    finish_with(a, num_accepted, consumed_rows)
}

/// Bind the stochastic/gumbel content walk to the resident verify-logits rows at `row_base`.
pub(super) fn run_stochastic_content_accept_at_row(
    model: &dyn Model,
    a: &mut ActiveSeq,
    row_base: usize,
    drafts: &[u32],
    verified_rows: usize,
) -> SpecAcceptOutcome {
    let logits_base = model.logits_buffer_ptr();
    let format = if model.logits_ptr_is_fp32(logits_base) {
        VerifyLogitsFormat::Fp32
    } else {
        VerifyLogitsFormat::Bf16
    };
    let vocab = model.vocab_size();
    let row_bytes = vocab
        * if format == VerifyLogitsFormat::Fp32 {
            4
        } else {
            2
        };
    let mut all = vec![0u8; verified_rows * row_bytes];
    if let Err(error) =
        model.copy_logits_to_host(logits_base.offset(row_base * row_bytes), &mut all)
    {
        fail_speculative_step(a, "stochastic verify logits download", error);
        return SpecAcceptOutcome {
            num_accepted: 0,
            bonus: None,
            consumed_rows: 0,
        };
    }
    let outcome = stochastic_content_accept(
        a,
        drafts,
        verified_rows,
        format,
        vocab,
        |_| DraftDistribution::PointMass,
        |i, buf| {
            let Some(src) = all.get(i * row_bytes..(i + 1) * row_bytes) else {
                return false;
            };
            buf.clear();
            buf.extend_from_slice(src);
            true
        },
    );
    publish_seq_gumbel_draft_spec(a);
    outcome
}

/// Bind the keyed Gumbel thinking walk to resident BF16 verify-logits rows at
/// `row_base` with lazy per-row D2H on demand: forced budget-nudge and armed
/// `</think>` rows never touch GPU logits (`0 bytes` D2H), and unforced walks
/// copy only the rows actually reached before mismatch or boundary.
pub(super) fn run_gumbel_thinking_accept_at_row(
    model: &dyn Model,
    a: &mut ActiveSeq,
    row_base: usize,
    drafts: &[u32],
    verified_rows: usize,
    ctx: &ThinkSpecCtx<'_>,
) -> SpecAcceptOutcome {
    let logits_base = model.logits_buffer_ptr();
    let vocab = model.vocab_size();
    let row_bytes = vocab * 2;
    let mut download_error: Option<anyhow::Error> = None;
    let outcome = gumbel_thinking_accept(
        a,
        drafts,
        verified_rows,
        VerifyLogitsFormat::Bf16,
        vocab,
        ctx,
        |i, buf| {
            if i >= verified_rows {
                return false;
            }
            buf.resize(row_bytes, 0);
            match model.copy_logits_to_host(logits_base.offset((row_base + i) * row_bytes), buf) {
                Ok(()) => true,
                Err(error) => {
                    download_error = Some(error);
                    false
                }
            }
        },
        // Defer the SSM boundary snapshot until the caller commits the accepted
        // prefix and drains `sync_secondary()` (checked via `outcome_ended_thinking`):
        // live recurrent state still holds the over-extended verify tail here.
        |_| {},
    );
    if let Some(error) = download_error {
        fail_speculative_step(a, "gumbel thinking verify logits download", error);
    }
    publish_seq_gumbel_draft_spec(a);
    outcome
}

#[cfg(test)]
mod draft_spec_publish_tests {
    use super::*;

    #[test]
    fn greedy_step_after_gumbel_step_at_same_propose_position_clears_draft_spec() {
        spark_runtime::sampler::gumbel::clear_gumbel_draft_specs();
        let mut a = crate::scheduler::test_fixtures::active_seq(
            crate::scheduler::ResponseSink::Blocking(None),
            0,
        );
        a.seq.prompt_len = 8;
        a.seq.seq_len = 9;
        a.output_tokens = vec![11, 22];
        a.seed = Some(0x1234_5678);
        a.temperature = 0.8;

        publish_seq_gumbel_draft_spec_with_policy(&a, SpecVerifyPolicy::Gumbel);
        let propose_pos = (a.seq.prompt_len + a.output_tokens.len()).saturating_sub(1);
        assert_eq!(propose_pos, 9);

        // A subsequent greedy request on the same slot/propose_position must clear it.
        a.temperature = 0.0;
        publish_seq_gumbel_draft_spec_with_policy(&a, SpecVerifyPolicy::Gumbel);
        assert_eq!(
            spark_runtime::sampler::gumbel::take_gumbel_draft_spec_for_slot(0, propose_pos),
            None
        );
        assert_eq!(
            spark_runtime::sampler::gumbel::lookup_gumbel_draft_spec(propose_pos),
            None
        );
    }

    #[test]
    fn gumbel_thinking_accept_forced_nudge_and_close_allowed_skip_all_row_fetches() {
        let mut a = crate::scheduler::test_fixtures::active_seq(
            crate::scheduler::ResponseSink::Blocking(None),
            0,
        );
        a.seq.prompt_len = 8;
        a.seq.seq_len = 8;
        a.output_tokens.clear();
        a.max_output_tokens = 64;
        a.remaining = 64;
        a.inside_thinking = true;
        a.force_end_thinking = true;
        a.think_end_token = Some(5);
        a.temperature = 0.8;
        a.seed = Some(777);
        a.budget_nudge = super::super::think_budget_nudge::BudgetNudge::plan(
            Some(std::sync::Arc::from([30u32, 31, 32])),
            0,
            64,
        );
        let ctx = ThinkSpecCtx {
            enabled: true,
            code_fence_token: None,
            reflection_suppress_ids: &[],
            adaptive_sampling: false,
            honor_eos_inside_thinking: Some(false),
        };

        // All 3 forced nudge tokens + final forced `</think>` match drafts with ZERO logits fetches.
        let mut fetches = 0usize;
        let outcome = gumbel_thinking_accept(
            &mut a,
            &[30, 31, 32, 5],
            5,
            VerifyLogitsFormat::Bf16,
            128,
            &ctx,
            |_i, _buf| {
                fetches += 1;
                false
            },
            |_| {},
        );
        assert_eq!(
            fetches, 0,
            "forced nudge + armed </think> rows must require zero logits D2H fetches"
        );
        assert_eq!(
            (outcome.num_accepted, outcome.bonus, outcome.consumed_rows),
            (3, Some(5), 4)
        );
        assert!(super::super::think_spec_accept::outcome_ended_thinking(
            &a, &outcome
        ));
        assert_eq!(a.output_tokens, vec![30, 31, 32, 5]);
        assert!(!a.inside_thinking);
        assert!(a.think_ended);
    }

    #[test]
    fn gumbel_thinking_accept_arms_force_end_thinking_on_answer_runway_exhaustion() {
        const MAX_OUTPUT: usize = 4_096;
        let cutoff = MAX_OUTPUT - super::super::confidence::min_answer_runway(MAX_OUTPUT);
        let mut a = crate::scheduler::test_fixtures::active_seq(
            crate::scheduler::ResponseSink::Blocking(None),
            0,
        );
        a.seq.prompt_len = 8;
        a.seq.seq_len = 8;
        a.max_output_tokens = MAX_OUTPUT;
        a.remaining = MAX_OUTPUT;
        a.inside_thinking = true;
        a.thinking_budget = None;
        a.thinking_tokens = 100;
        a.force_end_thinking = false;
        a.in_code_fence = true;
        a.think_end_token = Some(29);
        a.temperature = 0.8;
        a.seed = Some(42);
        a.output_tokens = vec![1; cutoff];
        let ctx = ThinkSpecCtx {
            enabled: true,
            code_fence_token: Some(3),
            reflection_suppress_ids: &[],
            adaptive_sampling: false,
            honor_eos_inside_thinking: Some(false),
        };

        let outcome = gumbel_thinking_accept(
            &mut a,
            &[1, 1],
            3,
            VerifyLogitsFormat::Bf16,
            128,
            &ctx,
            |_i, _buf| panic!("runway-exhausted forced </think> must not fetch logits"),
            |_| {},
        );
        assert_eq!(outcome.num_accepted, 0);
        assert_eq!(outcome.bonus, Some(29));
        assert!(super::super::think_spec_accept::outcome_ended_thinking(
            &a, &outcome
        ));
        assert!(!a.inside_thinking && a.think_ended);
        assert_eq!(a.output_tokens.len(), cutoff + 1);
    }

    #[test]
    fn parse_spec_verify_policy_recognizes_rejection_and_falls_back_on_unknown() {
        assert_eq!(
            parse_spec_verify_policy(Some("rejection")),
            SpecVerifyPolicy::Rejection
        );
        assert_eq!(
            parse_spec_verify_policy(Some("  rejection \n")),
            SpecVerifyPolicy::Rejection
        );
        assert_eq!(
            parse_spec_verify_policy(Some("Rejection")),
            SpecVerifyPolicy::Argmax
        );
        assert_eq!(
            parse_spec_verify_policy(Some("reject")),
            SpecVerifyPolicy::Argmax
        );
        assert_eq!(
            parse_spec_verify_policy(Some("gumbel")),
            SpecVerifyPolicy::Gumbel
        );
        assert_eq!(
            parse_spec_verify_policy(Some("stochastic")),
            SpecVerifyPolicy::Stochastic
        );
        assert_eq!(parse_spec_verify_policy(None), SpecVerifyPolicy::Argmax);
    }

    #[test]
    fn rejection_policy_publishes_and_clears_draft_spec_for_greedy_and_finished_seqs() {
        let mut a = crate::scheduler::test_fixtures::active_seq(
            crate::scheduler::ResponseSink::Blocking(None),
            2,
        );
        a.seq.prompt_len = 12;
        a.seq.seq_len = 13;
        a.output_tokens = vec![41, 42];
        a.seed = Some(0xABCD_EF01);
        a.temperature = 0.7;
        a.top_k = 20;
        a.top_p = 0.95;
        a.min_p = 0.05;

        // Rejection policy resolves both active draft spec and pending rejection slot.
        let (spec, slot) = resolve_seq_draft_spec_and_slot(&a, SpecVerifyPolicy::Rejection);
        let spec = spec.expect("rejection policy must produce active draft spec for T > 0");
        assert_eq!(slot, Some(2));
        assert_eq!(spec.seed, 0xABCD_EF01);
        assert_eq!(spec.step0_abs_pos, 14);
        assert_eq!(spec.temperature, 0.7);
        assert_eq!(spec.top_k, 20);
        assert_eq!(spec.top_p, 0.95);
        assert_eq!(spec.min_p, 0.05);

        // Gumbel policy resolves draft spec with rejection slot = None.
        let (g_spec, g_slot) = resolve_seq_draft_spec_and_slot(&a, SpecVerifyPolicy::Gumbel);
        assert_eq!(g_spec, Some(spec));
        assert_eq!(g_slot, None);

        // Argmax / Stochastic policies resolve (None, None).
        assert_eq!(
            resolve_seq_draft_spec_and_slot(&a, SpecVerifyPolicy::Argmax),
            (None, None)
        );
        assert_eq!(
            resolve_seq_draft_spec_and_slot(&a, SpecVerifyPolicy::Stochastic),
            (None, None)
        );

        // Finished and greedy sequences resolve (None, None) and invalidate ring entries.
        let propose_pos = (a.seq.prompt_len + a.output_tokens.len()).saturating_sub(1);
        assert_eq!(propose_pos, 13);
        a.finished = true;
        assert_eq!(
            resolve_seq_draft_spec_and_slot(&a, SpecVerifyPolicy::Rejection),
            (None, None)
        );
        publish_seq_gumbel_draft_spec_with_policy(&a, SpecVerifyPolicy::Rejection);
        assert_eq!(
            spark_runtime::sampler::gumbel::take_gumbel_draft_spec_for_slot(2, propose_pos),
            None
        );

        a.finished = false;
        a.temperature = 0.0;
        assert_eq!(
            resolve_seq_draft_spec_and_slot(&a, SpecVerifyPolicy::Rejection),
            (None, None)
        );
        publish_seq_gumbel_draft_spec_with_policy(&a, SpecVerifyPolicy::Rejection);
        assert_eq!(
            spark_runtime::sampler::gumbel::take_gumbel_draft_spec_for_slot(2, propose_pos),
            None
        );
    }

    #[test]
    fn first_spliced_draft_index_finds_exact_unmodified_prefix() {
        assert_eq!(
            first_spliced_draft_index(&[10, 11, 12], &[10, 11, 12]),
            None
        );
        assert_eq!(
            first_spliced_draft_index(&[10, 11, 12], &[99, 11, 12]),
            Some(0)
        );
        assert_eq!(
            first_spliced_draft_index(&[10, 11, 12], &[10, 11, 99]),
            Some(2)
        );
        assert_eq!(first_spliced_draft_index(&[10, 11, 12], &[10, 11]), Some(2));
        assert_eq!(first_spliced_draft_index(&[10, 11], &[10, 11, 12]), Some(2));
    }
}
