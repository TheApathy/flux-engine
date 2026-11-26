// SPDX-License-Identifier: AGPL-3.0-only

//! Fail-closed scheduler transaction for probabilistically sampled DFlash2.
//!
//! This module intentionally owns the complete narrow path: exact request
//! staging, drain-once sparse-q verification, target p/q rejection sampling,
//! normal token emission, one recurrent-state commit, and the next proposal.
//! Greedy DFlash never enters this file.

use super::*;
use anyhow::Context as _;
use spark_model::dflash_sampling::{
    DFLASH_SAMPLING_RECEIPT_SCHEMA, DflashCanonicalCommitRule, DflashCommitReceipt,
    DflashDraftSampleMethod, DflashDraftSamplingRequest, DflashNextTokenDisposition,
    DflashSampledProposalFrame, DflashSamplingReceipt, DflashTargetLogitsFormat,
    DflashTargetPolicyPhase, evaluate_sampled_dflash_with_evidence,
};

const SAMPLED_RECEIPT_MODE_ENV: &str = "ATLAS_DFLASH_SAMPLING_RECEIPTS";
const BOUNDED_FULL_RECEIPT_LIMIT: u64 = 16;
static SAMPLED_COMMIT_ORDINAL: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SampledReceiptMode {
    Bounded,
    Full,
}

impl SampledReceiptMode {
    fn as_str(self) -> &'static str {
        match self {
            Self::Bounded => "bounded",
            Self::Full => "full",
        }
    }
}

fn parse_sampled_receipt_mode(value: Option<&std::ffi::OsStr>) -> Result<SampledReceiptMode> {
    let Some(value) = value else {
        anyhow::bail!("sampled DFlash requires explicit {SAMPLED_RECEIPT_MODE_ENV}=full|bounded");
    };
    let Some(value) = value.to_str() else {
        anyhow::bail!("sampled DFlash {SAMPLED_RECEIPT_MODE_ENV} must be valid UTF-8");
    };
    match value {
        "bounded" => Ok(SampledReceiptMode::Bounded),
        "full" => Ok(SampledReceiptMode::Full),
        _ => anyhow::bail!(
            "sampled DFlash {SAMPLED_RECEIPT_MODE_ENV} must be exactly full or bounded"
        ),
    }
}

fn sampled_receipt_mode() -> Result<SampledReceiptMode> {
    parse_sampled_receipt_mode(std::env::var_os(SAMPLED_RECEIPT_MODE_ENV).as_deref())
}

fn should_emit_full_sampling_receipt(mode: SampledReceiptMode, ordinal: u64) -> bool {
    mode == SampledReceiptMode::Full || ordinal < BOUNDED_FULL_RECEIPT_LIMIT
}

const FORBIDDEN_FLAG_ENVS: &[&str] = &[
    "ATLAS_DFLASH_ASYNC",
    "ATLAS_DFLASH_BATCHED_VERIFY",
    "ATLAS_DFLASH_ADAPTIVE_GAMMA",
    "ATLAS_DFLASH_TPS_ROUTER",
    "ATLAS_DFLASH_PLD",
    "ATLAS_DFLASH_RETRIEVAL",
    "ATLAS_DFLASH_SAM",
    "ATLAS_DFLASH_RECYCLE",
    "ATLAS_DFLASH_ECHO",
    "ATLAS_DFLASH_SAM_CORROBORATE",
    "ATLAS_DFLASH_ACCEPT_FALLBACK",
    "ATLAS_DFLASH_BRANCH",
    "ATLAS_DFLASH_CATERPILLAR",
    "ATLAS_DFLASH_PORTFOLIO",
    "ATLAS_DFLASH_CFG_JF",
    "ATLAS_DFLASH_RELAX_ACCEPT",
    "ATLAS_DFLASH_SERIAL_COMMIT",
    "ATLAS_DFLASH_SKIP_REPROPOSE",
    "ATLAS_DDTREE_NONFLAT",
    "ATLAS_SELF_CONTEXT_DRAFT",
];

const FORBIDDEN_DEBUG_ENVS: &[&str] = &[
    "ATLAS_DFLASH_DEBUG_FORCE_PATTERN",
    "ATLAS_DFLASH_DEBUG_FORCE_NOISE_PATTERN",
    "ATLAS_DFLASH_MASK_OVERRIDE",
    "ATLAS_DFLASH_MARGIN_GATE",
    "ATLAS_DFLASH_TYPICAL_ACCEPT",
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum SampledProposalOutcome {
    /// The loaded proposer uses the released greedy selector.
    Greedy,
    /// Temperature zero stays on ordinary single-token decode.  Calling the
    /// probabilistic proposer without a sampled request is forbidden.
    PlainTemperatureZero,
    /// A typed sampled-q frame was atomically installed.
    Installed,
    /// The current thinking-policy boundary cannot be represented by one
    /// sampled frame.  The caller must take exactly one ordinary decode step
    /// and may try again from the resulting state.
    PlainThinkingBoundary,
    /// The policy-safe sampled width would verify outside the tensor-core
    /// parity window.  The caller must stay plain for this step so decode and
    /// verify never use non-identical kernel families while parity is engaged.
    PlainTcParityWidth,
    /// The proposed chain contains a structural/EOS token whose side effects
    /// require an ordinary one-token decode boundary.
    PlainStructuralBoundary,
    /// The producer returned no usable frame; scheduler state was cleared.
    Empty,
    /// The client cancelled at a scheduler transaction boundary.  All proposal
    /// auxiliaries were drained and the sequence was marked for retirement.
    Cancelled,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SampledThinkingPhase {
    Outside,
    Inside,
    Post,
}

const fn sampled_thinking_transition_requires_plain(
    force_end_thinking: bool,
    think_just_ended: bool,
) -> bool {
    force_end_thinking || think_just_ended
}

/// An engaged decode/verify parity contract may only publish widths served by
/// the tensor-core verify family that plain decode mirrors.  Keep this helper
/// explicitly inert when parity is off: sampled policy-width behavior must not
/// change for ordinary deployments.
const fn sampled_width_preserves_tc_parity(parity_engaged: bool, drafts: usize) -> bool {
    if !parity_engaged {
        return true;
    }
    match drafts.checked_add(1) {
        Some(verify_rows) => spark_model::layers::rows_in_tc_verify_window(verify_rows),
        None => false,
    }
}

impl SampledThinkingPhase {
    fn as_str(self) -> &'static str {
        match self {
            Self::Outside => "outside",
            Self::Inside => "inside",
            Self::Post => "post",
        }
    }

    fn receipt_phase(self) -> DflashTargetPolicyPhase {
        match self {
            Self::Outside => DflashTargetPolicyPhase::OutsideThinking,
            Self::Inside => DflashTargetPolicyPhase::InsideThinking,
            Self::Post => DflashTargetPolicyPhase::PostThinking,
        }
    }
}

/// Resolve the request's thinking phase without treating the request-level
/// `enable_thinking` bit as a permanent speculation ban.  That bit remains
/// true after `</think>`; only the live transition flags decide whether the
/// ordinary content sampler has reached a quiescent post-thinking state.
fn sampled_thinking_phase(
    enable_thinking: bool,
    inside_thinking: bool,
    force_end_thinking: bool,
    think_ended: bool,
    think_just_ended: bool,
) -> Result<SampledThinkingPhase> {
    if sampled_thinking_transition_requires_plain(force_end_thinking, think_just_ended) {
        if force_end_thinking {
            anyhow::bail!("sampled DFlash holds while a forced </think> transition is pending");
        }
        anyhow::bail!("sampled DFlash holds for the one-token post-thinking transition");
    }
    if inside_thinking {
        anyhow::ensure!(
            !think_ended,
            "sampled DFlash rejected contradictory inside/ended thinking state"
        );
        return Ok(SampledThinkingPhase::Inside);
    }
    if think_ended {
        return Ok(SampledThinkingPhase::Post);
    }
    anyhow::ensure!(
        !enable_thinking,
        "sampled DFlash holds before an enabled thinking span has ended"
    );
    Ok(SampledThinkingPhase::Outside)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct SampledCommitPlan {
    verify_rows: usize,
    committed_drafts: usize,
    total_target_slots: usize,
}

fn sampled_commit_plan(verify_rows: usize, committed_drafts: usize) -> Result<SampledCommitPlan> {
    anyhow::ensure!(
        verify_rows > 1,
        "sampled DFlash verify needs root plus drafts"
    );
    anyhow::ensure!(
        committed_drafts < verify_rows,
        "sampled DFlash committed draft count exceeds verify width"
    );
    Ok(SampledCommitPlan {
        verify_rows,
        committed_drafts,
        // The verifier inputs are [previous bonus/root, drafts...]. The newly
        // sampled next token is emitted as the next bonus but is not an input
        // row until the following step. Keep root + actually emitted drafts.
        total_target_slots: committed_drafts
            .checked_add(1)
            .context("sampled DFlash target-slot count overflow")?,
    })
}

fn env_is_one(name: &str) -> bool {
    std::env::var(name).ok().as_deref() == Some("1")
}

fn ensure_no_sampled_shortcuts() -> Result<()> {
    anyhow::ensure!(
        std::env::var("ATLAS_BOUNDED_HOST_COPY").ok().as_deref() == Some("1"),
        "sampled DFlash requires ATLAS_BOUNDED_HOST_COPY=1"
    );
    for &name in FORBIDDEN_FLAG_ENVS {
        anyhow::ensure!(!env_is_one(name), "sampled DFlash holds while {name}=1");
    }
    for &name in FORBIDDEN_DEBUG_ENVS {
        anyhow::ensure!(
            std::env::var_os(name).is_none(),
            "sampled DFlash holds while {name} is configured"
        );
    }
    anyhow::ensure!(
        std::env::var("ATLAS_DFLASH_METHOD").ok().as_deref() != Some("ddtree"),
        "sampled DFlash holds for DDTree"
    );
    anyhow::ensure!(
        std::env::var("ATLAS_DFLASH_FREE_SLOTS")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(0)
            == 0,
        "sampled DFlash holds for free-slot tree expansion"
    );
    anyhow::ensure!(
        std::env::var("ATLAS_DFLASH_RETR_WIDE")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(0)
            == 0,
        "sampled DFlash holds for wide retrieval"
    );
    anyhow::ensure!(
        std::env::var("ATLAS_DFLASH_DENOISE_STEPS")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(1)
            == 1,
        "sampled DFlash initially admits one denoise pass only"
    );
    Ok(())
}

fn neutral_inside_thinking_penalties(a: &ActiveSeq) -> bool {
    a.repetition_penalty == 1.0
        && a.presence_penalty == 0.0
        && a.frequency_penalty == 0.0
        && a.lz_penalty == 0.0
        && a.dry_multiplier == 0.0
}

/// Maximum draft width whose target rows all observe the same thinking
/// policy state as row zero.  The correction/bonus is the last sampled row,
/// so it may land on a boundary; no later target row consumes that mutation.
fn inside_policy_stable_width(
    requested: usize,
    thinking_tokens: u32,
    thinking_budget: Option<u32>,
    min_thinking_floor: u32,
    confidence_early_stop: bool,
    thinking_loop_watchdog: bool,
) -> usize {
    let mut width = requested;
    let cap_before =
        |boundary: u32| boundary.saturating_sub(thinking_tokens).saturating_sub(1) as usize;

    if let Some(budget) = thinking_budget {
        width = width.min(cap_before(budget));
    }
    if thinking_tokens < min_thinking_floor {
        width = width.min(cap_before(min_thinking_floor));
    }
    if confidence_early_stop {
        width = width.min(cap_before(
            super::confidence::CONFIDENCE_EARLY_STOP_MIN_THINKING,
        ));
    }
    if thinking_loop_watchdog {
        let next_check = if thinking_tokens < THINK_LOOP_MIN_TOKENS {
            THINK_LOOP_MIN_TOKENS
        } else {
            thinking_tokens
                .checked_div(THINK_LOOP_CHECK_STRIDE)
                .and_then(|bucket| bucket.checked_add(1))
                .and_then(|bucket| bucket.checked_mul(THINK_LOOP_CHECK_STRIDE))
                .unwrap_or(thinking_tokens)
        };
        width = width.min(cap_before(next_check));
    }
    width
}

fn sampled_policy_width(
    a: &ActiveSeq,
    requested: usize,
    phase: SampledThinkingPhase,
) -> Result<usize> {
    if phase != SampledThinkingPhase::Inside {
        return Ok(requested);
    }
    anyhow::ensure!(
        neutral_inside_thinking_penalties(a),
        "sampled DFlash inside-thinking target policy requires neutral history penalties"
    );
    anyhow::ensure!(
        !think_efficiency_config().is_active(),
        "sampled DFlash inside-thinking efficiency-wave policy is not receipted"
    );
    anyhow::ensure!(
        a.output_tokens.last().copied() == Some(a.last_token),
        "sampled DFlash holds after a silently consumed thinking token"
    );
    Ok(inside_policy_stable_width(
        requested,
        a.thinking_tokens,
        a.thinking_budget,
        crate::scheduler::helpers::min_thinking_floor(),
        watchdog_params().confidence_early_stop,
        enable_thinking_loop_watchdog(),
    ))
}

/// Build the exact sampler represented by a schema-7 receipt.  During an
/// active thinking span the plain path applies the reflection penalty before
/// the request sampler.  With neutral history penalties, prepending those
/// additive entries to `logit_bias` is bit-for-bit the same operation order:
/// reflection entries first, then the request's original bias entries.
fn sampled_policy_params(
    a: &ActiveSeq,
    think: &ThinkSpecCtx<'_>,
    phase: SampledThinkingPhase,
    vocab_size: usize,
) -> spark_runtime::sampler::SamplingParams {
    let mut params = a.sampling_params(a.temperature);
    if phase == SampledThinkingPhase::Inside {
        params.logit_bias = prepend_inside_thinking_biases(
            &params.logit_bias,
            think.reflection_suppress_ids,
            vocab_size,
        );
    }
    params
}

fn prepend_inside_thinking_biases(
    request_biases: &[(u32, f32)],
    reflection_suppress_ids: &[u32],
    vocab_size: usize,
) -> Vec<(u32, f32)> {
    let mut biases = Vec::with_capacity(reflection_suppress_ids.len() + request_biases.len());
    biases.extend(
        reflection_suppress_ids
            .iter()
            .copied()
            .filter(|&token| (token as usize) < vocab_size)
            .map(|token| (token, -10.0)),
    );
    biases.extend_from_slice(request_biases);
    biases
}

fn sampled_policy_forbidden_tokens(a: &ActiveSeq, phase: SampledThinkingPhase) -> Vec<u32> {
    sampled_policy_forbidden_tokens_for(
        phase,
        a.thinking_tokens,
        crate::scheduler::helpers::min_thinking_floor(),
        a.think_start_token,
        a.think_end_token,
        a.tool_call_start_token,
    )
}

fn sampled_policy_forbidden_tokens_for(
    phase: SampledThinkingPhase,
    thinking_tokens: u32,
    min_thinking_floor: u32,
    think_start_token: Option<u32>,
    think_end_token: Option<u32>,
    tool_call_start_token: Option<u32>,
) -> Vec<u32> {
    let mut tokens = match phase {
        SampledThinkingPhase::Post => [think_start_token, think_end_token]
            .into_iter()
            .flatten()
            .collect::<Vec<_>>(),
        SampledThinkingPhase::Inside => {
            let mut tokens = Vec::with_capacity(2);
            // The plain sampler always hard-masks a tool opener in reasoning.
            if let Some(token) = tool_call_start_token {
                tokens.push(token);
            }
            // Before the configured floor, ordinary decode also hard-masks
            // `</think>`. Width planning prevents this frame crossing the
            // point where that mask is removed.
            if thinking_tokens < min_thinking_floor
                && let Some(token) = think_end_token
            {
                tokens.push(token);
            }
            tokens
        }
        SampledThinkingPhase::Outside => Vec::new(),
    };
    tokens.sort_unstable();
    tokens.dedup();
    tokens
}

/// A sampled frame cannot roll target-policy state across a structural or
/// reasoning-policy boundary. Outside/post phases are deliberately
/// conservative. Inside thinking, ordinary control-looking content is allowed;
/// only a real phase close, suppressed stop, or ChatML-cycle transition holds.
fn sampled_proposal_requires_plain(
    phase: SampledThinkingPhase,
    drafts: &[u32],
    output_tokens: &[u32],
    think_start_token: Option<u32>,
    think_end_token: Option<u32>,
    tool_call_start_token: Option<u32>,
    tool_call_end_token: Option<u32>,
    eos_tokens: &[u32],
    im_start: Option<u32>,
) -> bool {
    match phase {
        SampledThinkingPhase::Outside => drafts.iter().any(|&token| {
            eos_tokens.contains(&token)
                || think_start_token == Some(token)
                || think_end_token == Some(token)
                || tool_call_start_token == Some(token)
                || tool_call_end_token == Some(token)
                || im_start == Some(token)
        }),
        SampledThinkingPhase::Post => drafts.iter().any(|&token| {
            eos_tokens.contains(&token)
                || tool_call_start_token == Some(token)
                || tool_call_end_token == Some(token)
                || im_start == Some(token)
        }),
        SampledThinkingPhase::Inside => {
            let mut history = output_tokens.to_vec();
            for &token in drafts {
                if eos_tokens.contains(&token) || think_end_token == Some(token) {
                    return true;
                }
                history.push(token);
                if im_start
                    .is_some_and(|marker| detect_chatml_thinking_cycle(&history, None, marker))
                {
                    return true;
                }
            }
            false
        }
    }
}

fn validate_sampled_frame_policy(
    model: &dyn Model,
    a: &ActiveSeq,
    think: &ThinkSpecCtx<'_>,
    drafts: &[u32],
    frame: &DflashSampledProposalFrame,
) -> Result<(SampledReceiptMode, SampledThinkingPhase)> {
    let receipt_mode = validate_sampled_route(model, a, think)?;
    let phase = sampled_thinking_phase(
        a.enable_thinking,
        a.inside_thinking,
        a.force_end_thinking,
        a.think_ended,
        a.think_just_ended,
    )?;
    frame.validate(model.vocab_size())?;
    let params = sampled_policy_params(a, think, phase, model.vocab_size());
    frame.request.validate_sampler(&params)?;
    frame.request.validate_history(&a.output_tokens)?;
    anyhow::ensure!(
        frame
            .proposal
            .q_rows
            .as_ref()
            .and_then(|rows| rows.first())
            .is_some_and(|row| row.predecessor == a.last_token),
        "sampled DFlash sparse-q anchor disagrees with live last token"
    );
    anyhow::ensure!(
        frame.proposal.tokens.as_slice() == drafts,
        "sampled DFlash q does not match the verify draft frame"
    );
    anyhow::ensure!(
        frame.request.target_policy_phase == phase.receipt_phase(),
        "sampled DFlash target policy phase drifted after proposal"
    );
    anyhow::ensure!(
        frame.request.target_forbidden_token_ids == sampled_policy_forbidden_tokens(a, phase),
        "sampled DFlash target policy mask drifted after proposal"
    );
    let safe_width = sampled_policy_width(a, drafts.len(), phase)?;
    anyhow::ensure!(
        !drafts.is_empty() && drafts.len() <= safe_width,
        "sampled DFlash frame crossed a thinking-policy boundary"
    );
    Ok((receipt_mode, phase))
}

fn validate_sampled_route(
    model: &dyn Model,
    a: &ActiveSeq,
    think: &ThinkSpecCtx<'_>,
) -> Result<SampledReceiptMode> {
    anyhow::ensure!(model.proposer_is_dflash(), "sampled route requires DFlash");
    anyhow::ensure!(!model.is_ep(), "sampled DFlash EP route is not implemented");
    anyhow::ensure!(
        !crate::dsv41::serving(),
        "sampled DFlash DSV4.1 target-policy route is not implemented"
    );
    anyhow::ensure!(
        model.dflash_draft_sample_method() == DflashDraftSampleMethod::Probabilistic,
        "sampled route requires the probabilistic DFlash selector"
    );
    anyhow::ensure!(
        a.temperature.is_finite() && a.temperature > 0.0,
        "sampled DFlash requires positive finite request temperature"
    );
    anyhow::ensure!(
        a.seed.is_some(),
        "sampled DFlash requires a scheduler-resolved effective seed"
    );
    anyhow::ensure!(!a.finished, "sampled DFlash cannot run a finished sequence");
    anyhow::ensure!(
        a.timeout_at
            .is_none_or(|deadline| Instant::now() < deadline),
        "sampled DFlash request deadline already elapsed"
    );
    anyhow::ensure!(
        a.remaining > 0,
        "sampled DFlash has no remaining content budget"
    );
    anyhow::ensure!(
        a.output_tokens.len() < a.max_output_tokens,
        "sampled DFlash reached max output tokens"
    );
    anyhow::ensure!(
        a.output_tokens.len() >= a.min_tokens,
        "sampled DFlash holds while EOS suppression can discard a token"
    );
    anyhow::ensure!(
        a.grammar_state.is_none(),
        "sampled DFlash grammar route is not implemented"
    );
    anyhow::ensure!(
        a.top_logprobs.is_none(),
        "sampled DFlash logprobs route is not implemented"
    );
    anyhow::ensure!(
        !think.adaptive_sampling,
        "sampled DFlash adaptive sampler is not implemented"
    );
    let phase = sampled_thinking_phase(
        a.enable_thinking,
        a.inside_thinking,
        a.force_end_thinking,
        a.think_ended,
        a.think_just_ended,
    )?;
    if phase == SampledThinkingPhase::Inside {
        anyhow::ensure!(
            think.enabled,
            "sampled DFlash inside-thinking route requires the policy-oracle gate"
        );
        // Validate the static policy here; the exact frame-width transition
        // guard is applied after the caller supplies its requested gamma.
        sampled_policy_width(a, 1, phase)?;
    }
    anyhow::ensure!(
        !a.require_tool_call && !a.suppress_tool_call && !a.tool_call_opened && !a.inside_tool_body,
        "sampled DFlash tool route is not implemented"
    );
    anyhow::ensure!(
        a.pending_tree_payload.is_none(),
        "sampled DFlash cannot consume tree topology"
    );
    anyhow::ensure!(
        !super::verify_dflash_step::dflash_tree_method_active()
            && !super::verify_dflash_step::dflash_portfolio_active(),
        "sampled DFlash tree/portfolio route is not implemented"
    );
    ensure_no_sampled_shortcuts()?;
    sampled_receipt_mode()
}

fn sampled_frame_boundary_after_prefix(
    phase: SampledThinkingPhase,
    accepted_prefix: &[u32],
    token: u32,
    output_tokens: &[u32],
    think_start_token: Option<u32>,
    think_end_token: Option<u32>,
    tool_call_start_token: Option<u32>,
    tool_call_end_token: Option<u32>,
    eos_tokens: &[u32],
    im_start: Option<u32>,
) -> bool {
    let mut boundary_probe = Vec::with_capacity(accepted_prefix.len() + 1);
    boundary_probe.extend_from_slice(accepted_prefix);
    boundary_probe.push(token);
    sampled_proposal_requires_plain(
        phase,
        &boundary_probe,
        output_tokens,
        think_start_token,
        think_end_token,
        tool_call_start_token,
        tool_call_end_token,
        eos_tokens,
        im_start,
    )
}

fn emit_sampled_token(
    a: &mut ActiveSeq,
    token: u32,
    phase: SampledThinkingPhase,
    think: &ThinkSpecCtx<'_>,
) -> bool {
    if phase == SampledThinkingPhase::Inside {
        let mut defer_think_end_snapshot = false;
        super::think_spec_accept::commit_thinking_token(
            a,
            token,
            None,
            think.code_fence_token,
            think.resolved_honor_eos_inside_thinking(),
            &mut |_| defer_think_end_snapshot = true,
        );
        defer_think_end_snapshot
    } else {
        emit_token(a, token, None);
        false
    }
}

/// Stage exactly one request and atomically install its proposal.  This is the
/// only production caller allowed for probabilistic DFlash, so server-side
/// retrieval and producer shortcuts cannot publish a q-less frame.
pub(super) fn propose_sampled_dflash(
    model: &dyn Model,
    a: &mut ActiveSeq,
    num_drafts: usize,
    think: &ThinkSpecCtx<'_>,
) -> Result<SampledProposalOutcome> {
    if model.dflash_draft_sample_method() != DflashDraftSampleMethod::Probabilistic {
        return Ok(SampledProposalOutcome::Greedy);
    }
    if retire_cancelled_speculation(a, "dflash_before_route") {
        return Ok(SampledProposalOutcome::Cancelled);
    }
    // These are expected one-token plain transitions, not transaction errors.
    // Drain every scheduler-owned proposal field before returning so neither a
    // stale draft nor its sampled-q frame can cross the policy boundary.
    if sampled_thinking_transition_requires_plain(a.force_end_thinking, a.think_just_ended) {
        proposal_lifecycle::clear(
            &mut a.pending_drafts,
            &mut a.pending_tree_payload,
            &mut a.pending_dflash_sampled_proposal,
        );
        return Ok(SampledProposalOutcome::PlainThinkingBoundary);
    }
    if a.temperature == 0.0 {
        proposal_lifecycle::clear(
            &mut a.pending_drafts,
            &mut a.pending_tree_payload,
            &mut a.pending_dflash_sampled_proposal,
        );
        return Ok(SampledProposalOutcome::PlainTemperatureZero);
    }
    validate_sampled_route(model, a, think)?;
    if retire_cancelled_speculation(a, "dflash_after_route_validation") {
        return Ok(SampledProposalOutcome::Cancelled);
    }
    let phase = sampled_thinking_phase(
        a.enable_thinking,
        a.inside_thinking,
        a.force_end_thinking,
        a.think_ended,
        a.think_just_ended,
    )?;
    anyhow::ensure!(
        a.pending_drafts.is_empty() && a.pending_dflash_sampled_proposal.is_none(),
        "sampled DFlash proposal slot is not empty"
    );
    let safe_requested = sampled_policy_width(a, num_drafts, phase)?;
    if safe_requested == 0 {
        proposal_lifecycle::clear(
            &mut a.pending_drafts,
            &mut a.pending_tree_payload,
            &mut a.pending_dflash_sampled_proposal,
        );
        return Ok(SampledProposalOutcome::PlainThinkingBoundary);
    }
    let gamma = model.dflash_sampled_proposal_width(safe_requested)?;
    anyhow::ensure!(
        gamma > 0 && gamma <= safe_requested,
        "sampled DFlash proposer widened policy-safe gamma {safe_requested} to {gamma}"
    );
    let parity_engaged = spark_model::layers::decode_tc_parity_enabled();
    if !sampled_width_preserves_tc_parity(parity_engaged, gamma) {
        tracing::warn!(
            drafts = gamma,
            verify_rows = ?gamma.checked_add(1),
            "DFLASH_TC_PARITY_PLAIN: sampled proposal width is outside the engaged tensor-core parity window"
        );
        proposal_lifecycle::clear(
            &mut a.pending_drafts,
            &mut a.pending_tree_payload,
            &mut a.pending_dflash_sampled_proposal,
        );
        return Ok(SampledProposalOutcome::PlainTcParityWidth);
    }
    let params = sampled_policy_params(a, think, phase, model.vocab_size());
    let request = DflashDraftSamplingRequest::from_seeded_sampling_params(
        &params,
        &a.output_tokens,
        a.output_tokens.len(),
        gamma,
    )?
    .with_target_forbidden_token_ids(sampled_policy_forbidden_tokens(a, phase))?
    .with_target_policy_phase(phase.receipt_phase())?;
    model.stage_dflash_sampling_request(&mut a.seq, request)?;
    let proposal_token = a.last_token;
    let (installed, cancelled_after_propose) = proposal_lifecycle::propose_and_install_with(
        model,
        a,
        proposal_token,
        gamma,
        None,
        true,
        |a, _proposal| request_cancelled(a),
    );
    let installed = installed?;
    if cancelled_after_propose || request_cancelled(a) {
        retire_cancelled_speculation(a, "dflash_after_propose");
        return Ok(SampledProposalOutcome::Cancelled);
    }
    if installed {
        anyhow::ensure!(
            a.pending_tree_payload.is_none() && a.pending_dflash_sampled_proposal.is_some(),
            "sampled DFlash proposal published without exactly one q frame"
        );
        let installed_validation = a
            .pending_dflash_sampled_proposal
            .as_ref()
            .context("sampled DFlash proposal omitted its q frame")
            .and_then(|frame| {
                validate_sampled_frame_policy(model, a, think, &a.pending_drafts, frame)
            });
        let (_, installed_phase) = match installed_validation {
            Ok(validated) => validated,
            Err(error) => {
                proposal_lifecycle::clear(
                    &mut a.pending_drafts,
                    &mut a.pending_tree_payload,
                    &mut a.pending_dflash_sampled_proposal,
                );
                return Err(error.context("validate freshly installed sampled DFlash frame"));
            }
        };
        if retire_cancelled_speculation(a, "dflash_after_installed_validation") {
            return Ok(SampledProposalOutcome::Cancelled);
        }
        if sampled_proposal_requires_plain(
            installed_phase,
            &a.pending_drafts,
            &a.output_tokens,
            a.think_start_token,
            a.think_end_token,
            a.tool_call_start_token,
            a.tool_call_end_token,
            &a.eos_tokens,
            im_start_hard_stop(),
        ) {
            proposal_lifecycle::clear(
                &mut a.pending_drafts,
                &mut a.pending_tree_payload,
                &mut a.pending_dflash_sampled_proposal,
            );
            return Ok(SampledProposalOutcome::PlainStructuralBoundary);
        }
        a.draft_origin = DraftOrigin::Proposer;
        Ok(SampledProposalOutcome::Installed)
    } else {
        proposal_lifecycle::clear(
            &mut a.pending_drafts,
            &mut a.pending_tree_payload,
            &mut a.pending_dflash_sampled_proposal,
        );
        Ok(SampledProposalOutcome::Empty)
    }
}

fn fail_sampled_step(
    model: &dyn Model,
    a: &mut ActiveSeq,
    pre_verify_len: usize,
    verify_k: usize,
    verify_started: bool,
    target_state_committed: bool,
    committed_drafts: usize,
    irreversible: bool,
    error: anyhow::Error,
) {
    tracing::error!("sampled DFlash transaction failed closed: {error:#}");
    if verify_started && !target_state_committed && irreversible {
        let reconcile = (|| -> Result<()> {
            let plan = sampled_commit_plan(verify_k, committed_drafts)?;
            let target_seq_len = pre_verify_len
                .checked_add(plan.total_target_slots)
                .ok_or_else(|| anyhow::anyhow!("sampled abort target length overflow"))?;
            anyhow::ensure!(
                target_seq_len <= a.seq.seq_len && target_seq_len <= a.seq.tokens.len(),
                "sampled abort cannot retain emitted prefix from short verifier state"
            );
            a.seq.seq_len = target_seq_len;
            a.seq.tokens.truncate(target_seq_len);
            model.commit_verify_state_async(&mut a.seq, plan.total_target_slots, plan.verify_rows)
        })();
        if let Err(reconcile_error) = reconcile {
            tracing::error!(
                "sampled DFlash emitted-prefix reconciliation failed: {reconcile_error:#}"
            );
            a.seq.seq_len = pre_verify_len;
            a.seq.tokens.truncate(pre_verify_len);
            if let Err(rollback_error) =
                super::verify_dflash_step::rollback_failed_verify_frame(model, &mut a.seq, verify_k)
            {
                tracing::error!("sampled DFlash abort rollback failed: {rollback_error:#}");
            }
        }
    } else if verify_started && !target_state_committed {
        a.seq.seq_len = pre_verify_len;
        a.seq.tokens.truncate(pre_verify_len);
        if let Err(restore_error) =
            super::verify_dflash_step::rollback_failed_verify_frame(model, &mut a.seq, verify_k)
        {
            tracing::error!("sampled DFlash checkpoint restore failed: {restore_error:#}");
        }
    }
    proposal_lifecycle::clear(
        &mut a.pending_drafts,
        &mut a.pending_tree_payload,
        &mut a.pending_dflash_sampled_proposal,
    );
    a.finished = true;
}

/// Consume one sampled-q frame exactly once and execute exact target p/q
/// rejection sampling.  Every target-row read is a single bounded-copy-aware
/// `copy_logits_to_host` call; any failure restores the pre-verify checkpoint
/// and terminates instead of falling back to greedy acceptance.
pub(super) fn step_verify_sampled_dflash(
    model: &dyn Model,
    a: &mut ActiveSeq,
    drafts: &[u32],
    num_drafts: usize,
    think: &ThinkSpecCtx<'_>,
    frame: DflashSampledProposalFrame,
) {
    let pre_verify_len = a.seq.seq_len;
    if retire_cancelled_speculation(a, "dflash_before_verify") {
        return;
    }
    // Validate all live/frame identities before classifying a valid structural
    // proposal as a plain boundary. A stale or forged frame must fail closed,
    // never escape validation through the no-GPU fallback.
    let (receipt_mode, phase) = match validate_sampled_frame_policy(model, a, think, drafts, &frame)
    {
        Ok(validated) => validated,
        Err(error) => {
            fail_sampled_step(model, a, pre_verify_len, 0, false, false, 0, false, error);
            return;
        }
    };
    if retire_cancelled_speculation(a, "dflash_after_verify_validation") {
        return;
    }
    let parity_engaged = spark_model::layers::decode_tc_parity_enabled();
    if !sampled_width_preserves_tc_parity(parity_engaged, drafts.len()) {
        tracing::warn!(
            drafts = drafts.len(),
            verify_rows = ?drafts.len().checked_add(1),
            "DFLASH_TC_PARITY_PLAIN: sampled frame discarded before out-of-window verify"
        );
        proposal_lifecycle::clear(
            &mut a.pending_drafts,
            &mut a.pending_tree_payload,
            &mut a.pending_dflash_sampled_proposal,
        );
        return;
    }
    let crosses_structural_boundary = |tokens: &[u32]| {
        sampled_proposal_requires_plain(
            phase,
            tokens,
            &a.output_tokens,
            a.think_start_token,
            a.think_end_token,
            a.tool_call_start_token,
            a.tool_call_end_token,
            &a.eos_tokens,
            im_start_hard_stop(),
        )
    };
    if crosses_structural_boundary(drafts) || crosses_structural_boundary(&frame.proposal.tokens) {
        tracing::debug!(
            "sampled DFlash structural proposal discarded before target verify; next step stays plain"
        );
        proposal_lifecycle::clear(
            &mut a.pending_drafts,
            &mut a.pending_tree_payload,
            &mut a.pending_dflash_sampled_proposal,
        );
        return;
    }
    let mut irreversible = false;
    let mut verify_started = false;
    let mut target_state_committed = false;
    let mut emitted_committed_drafts = 0usize;
    let transaction = (|| -> Result<()> {
        super::verify_dflash_step::dflash_capacity::preflight_frame_then(
            true,
            || model.dflash_verify_capacity_k(),
            drafts.len(),
            drafts.len(),
            None,
            || model.sync_secondary(),
        )??;

        let mut tokens = Vec::with_capacity(drafts.len() + 1);
        tokens.push(a.last_token);
        tokens.extend_from_slice(drafts);
        verify_started = true;
        let verified = model.decode_verify_dflash(&tokens, &mut a.seq, 0)?;
        anyhow::ensure!(
            verified.len() == tokens.len(),
            "sampled DFlash verifier row-count drift"
        );

        let vocab_size = model.vocab_size();
        let logits_ptr = model.logits_buffer_ptr();
        let format = if model.logits_ptr_is_fp32(logits_ptr) {
            DflashTargetLogitsFormat::Fp32
        } else {
            DflashTargetLogitsFormat::Bf16
        };
        let element_bytes = match format {
            DflashTargetLogitsFormat::Bf16 => 2usize,
            DflashTargetLogitsFormat::Fp32 => 4usize,
        };
        let logits_bytes = tokens
            .len()
            .checked_mul(vocab_size)
            .and_then(|elements| elements.checked_mul(element_bytes))
            .ok_or_else(|| anyhow::anyhow!("sampled DFlash logit readback size overflow"))?;
        let mut host_logits = vec![0u8; logits_bytes];
        model.copy_logits_to_host(logits_ptr, &mut host_logits)?;

        let params = sampled_policy_params(a, think, phase, vocab_size);
        let (acceptance, target_evidence) = evaluate_sampled_dflash_with_evidence(
            &host_logits,
            format,
            vocab_size,
            &frame,
            &params,
            &a.output_tokens,
        )?;
        anyhow::ensure!(
            !sampled_proposal_requires_plain(
                phase,
                &drafts[..acceptance.accepted],
                &a.output_tokens,
                a.think_start_token,
                a.think_end_token,
                a.tool_call_start_token,
                a.tool_call_end_token,
                &a.eos_tokens,
                im_start_hard_stop(),
            ),
            "sampled DFlash target accepted through a structural draft"
        );
        let output_position = frame.request.output_position;
        let output_tokens_before = a.output_tokens.len();
        let last_token_before = a.last_token;

        // The correction/bonus token is allowed to be a terminal or structural
        // boundary, but it must flow through `emit_token` below.  That keeps
        // EOS/stop/max and silent <think> transitions identical to ordinary
        // decode.  A boundary ends speculation for this frame; the next step
        // resumes on the plain decoder.
        let next_is_structural = sampled_frame_boundary_after_prefix(
            phase,
            &drafts[..acceptance.accepted],
            acceptance.next_token,
            &a.output_tokens,
            a.think_start_token,
            a.think_end_token,
            a.tool_call_start_token,
            a.tool_call_end_token,
            &a.eos_tokens,
            im_start_hard_stop(),
        );
        let next_is_think_end = phase == SampledThinkingPhase::Inside
            && a.think_end_token == Some(acceptance.next_token);
        a.last_token_time = Instant::now();
        a.last_verify_accepted = u16::try_from(acceptance.accepted).unwrap_or(u16::MAX);
        a.draft_origin = DraftOrigin::Proposer;

        let mut committed_drafts = 0usize;
        let mut deferred_think_end_snapshot = false;
        for &token in drafts.iter().take(acceptance.accepted) {
            irreversible = true;
            let output_len = a.output_tokens.len();
            deferred_think_end_snapshot |= emit_sampled_token(a, token, phase, think);
            if a.output_tokens.len() == output_len + 1 {
                committed_drafts += 1;
                emitted_committed_drafts = committed_drafts;
                a.last_token = token;
            } else {
                anyhow::ensure!(
                    a.finished,
                    "sampled accepted token was suppressed without terminating"
                );
            }
            if a.finished {
                break;
            }
        }

        let mut next_disposition = DflashNextTokenDisposition::NotAttempted;
        if !a.finished && committed_drafts == acceptance.accepted {
            irreversible = true;
            let output_len = a.output_tokens.len();
            deferred_think_end_snapshot |=
                emit_sampled_token(a, acceptance.next_token, phase, think);
            if a.output_tokens.len() == output_len + 1 {
                a.last_token = acceptance.next_token;
                next_disposition = DflashNextTokenDisposition::OutputAppendedAndArmed;
            } else if next_is_structural && !a.finished {
                // Ordinary decode advances `last_token` even for structural
                // tokens that `emit_token` consumes silently (for example
                // `<think>`).  Preserve that exact target-input boundary.
                a.last_token = acceptance.next_token;
                next_disposition = DflashNextTokenDisposition::ConsumedSilentlyAndArmed;
            } else {
                anyhow::ensure!(
                    a.finished,
                    "sampled next token was suppressed without terminating"
                );
                next_disposition = DflashNextTokenDisposition::AttemptedTerminalWithoutFeedback;
            }
        }

        let plan = sampled_commit_plan(tokens.len(), committed_drafts)?;
        let target_seq_len = pre_verify_len
            .checked_add(plan.total_target_slots)
            .ok_or_else(|| anyhow::anyhow!("sampled DFlash target length overflow"))?;
        anyhow::ensure!(
            target_seq_len <= a.seq.seq_len,
            "sampled DFlash target length exceeds verifier state"
        );
        a.seq.seq_len = target_seq_len;
        a.seq.tokens.truncate(target_seq_len);
        // Exactly one commit call per successful verify transaction, including
        // EOS/stop/max/cancel terminal outcomes.
        model.commit_verify_state_async(&mut a.seq, plan.total_target_slots, plan.verify_rows)?;
        target_state_committed = true;
        irreversible = true;
        let mut think_end_snapshot_ordered = false;
        if deferred_think_end_snapshot {
            // `</think>` is emitted before recurrent-state commit so all
            // ordinary token side effects happen exactly once.  The rollback
            // snapshot must not observe the verify kernel's rejected tail,
            // though: make the default stream wait for the secondary-stream
            // accepted-intermediate restore, then enqueue the snapshot there.
            model.sync_secondary()?;
            rollback::snapshot_boundary_if_ssm(a, model);
            think_end_snapshot_ordered = true;
        }

        let receipt_ordinal =
            SAMPLED_COMMIT_ORDINAL.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let receipt = DflashSamplingReceipt::new(
            frame.request,
            vocab_size,
            frame.proposal,
            acceptance.clone(),
            target_evidence,
            DflashCommitReceipt {
                ordinal: receipt_ordinal,
                rule: DflashCanonicalCommitRule::RootAndCommittedDraftsNextTokenUnfed,
                verify_rows: plan.verify_rows,
                committed_drafts: plan.committed_drafts,
                total_target_slots: plan.total_target_slots,
                next_disposition,
                next_was_structural: next_is_structural,
                think_end_token: a.think_end_token,
                next_was_think_end: next_is_think_end,
                commit_call_returned_ok: true,
                seq_len_before: pre_verify_len,
                seq_len_after: target_seq_len,
                finished: a.finished,
                output_tokens_before,
                output_tokens_after: a.output_tokens.len(),
                last_token_before,
                last_token_after: a.last_token,
                think_end_snapshot_requested: deferred_think_end_snapshot,
                think_end_snapshot_ordered,
            },
        )?;
        let next_armed = matches!(
            next_disposition,
            DflashNextTokenDisposition::OutputAppendedAndArmed
                | DflashNextTokenDisposition::ConsumedSilentlyAndArmed
        );
        let execution_sha256 = receipt.execution_sha256_hex();
        let emit_full_receipt = should_emit_full_sampling_receipt(receipt_mode, receipt_ordinal);
        tracing::info!(
            accepted = acceptance.accepted,
            committed_drafts = plan.committed_drafts,
            emitted_next = next_armed,
            next_disposition = ?next_disposition,
            finished = a.finished,
            schema = DFLASH_SAMPLING_RECEIPT_SCHEMA,
            receipt_mode = %receipt_mode.as_str(),
            full_receipt = emit_full_receipt,
            receipt_ordinal = receipt_ordinal,
            output_position = output_position,
            drafts = drafts.len(),
            thinking_phase = phase.as_str(),
            execution_sha256 = %execution_sha256,
            "DFLASH_SAMPLED_COMMIT"
        );
        if emit_full_receipt {
            tracing::info!("{}", receipt.to_json_line()?);
        } else if receipt_ordinal == BOUNDED_FULL_RECEIPT_LIMIT {
            tracing::info!(
                schema = DFLASH_SAMPLING_RECEIPT_SCHEMA,
                receipt_mode = %receipt_mode.as_str(),
                full_receipt_limit = BOUNDED_FULL_RECEIPT_LIMIT,
                "DFLASH_SAMPLING_RECEIPTS_BOUNDED"
            );
        }
        crate::metrics::SPEC_DECODE_VERIFY
            .with_label_values(&[
                "dflash-sampled",
                if acceptance.accepted == drafts.len() {
                    "accept_all"
                } else {
                    "accept_partial"
                },
            ])
            .inc();
        crate::metrics::compat_metrics::record_spec_decode(
            drafts.len() as u64,
            acceptance.accepted as u64,
        );
        crate::metrics::compat_metrics::set_kv_blocks_free(model.num_free_blocks());

        if a.finished {
            return Ok(());
        }
        anyhow::ensure!(
            next_armed,
            "nonterminal sampled step did not emit its next token"
        );
        if next_is_structural || a.force_end_thinking {
            proposal_lifecycle::clear(
                &mut a.pending_drafts,
                &mut a.pending_tree_payload,
                &mut a.pending_dflash_sampled_proposal,
            );
            return Ok(());
        }
        if let Err(error) = model.save_hidden_for_dflash(a.last_token, &mut a.seq, 0) {
            tracing::error!("sampled DFlash hidden save failed; next step stays plain: {error:#}");
            return Ok(());
        }
        if let Err(error) = model.trim_proposer_state(&mut a.seq, committed_drafts, 0) {
            tracing::error!(
                "sampled DFlash proposer trim failed; next step stays plain: {error:#}"
            );
            return Ok(());
        }
        match propose_sampled_dflash(model, a, num_drafts, think) {
            Ok(SampledProposalOutcome::Installed) => {}
            Ok(SampledProposalOutcome::Cancelled) => {}
            Ok(other) => tracing::warn!(
                "sampled DFlash reproposal stayed plain after committed output: {other:?}"
            ),
            Err(error) => {
                tracing::error!(
                    "sampled DFlash reproposal failed after committed output; next step stays plain: {error:#}"
                );
                proposal_lifecycle::clear(
                    &mut a.pending_drafts,
                    &mut a.pending_tree_payload,
                    &mut a.pending_dflash_sampled_proposal,
                );
            }
        }
        Ok(())
    })();

    if let Err(error) = transaction {
        fail_sampled_step(
            model,
            a,
            pre_verify_len,
            drafts.len() + 1,
            verify_started,
            target_state_committed,
            emitted_committed_drafts,
            irreversible,
            error,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn receipt_mode_is_explicit_and_exact() {
        assert_eq!(
            parse_sampled_receipt_mode(Some(std::ffi::OsStr::new("bounded"))).unwrap(),
            SampledReceiptMode::Bounded
        );
        assert_eq!(
            parse_sampled_receipt_mode(Some(std::ffi::OsStr::new("full"))).unwrap(),
            SampledReceiptMode::Full
        );
        assert!(parse_sampled_receipt_mode(None).is_err());
        for invalid in ["", "off", "FULL", "1", "bounded "] {
            assert!(
                parse_sampled_receipt_mode(Some(std::ffi::OsStr::new(invalid))).is_err(),
                "accepted invalid receipt mode {invalid:?}"
            );
        }
    }

    #[test]
    fn bounded_receipts_keep_an_exact_prefix_while_full_never_suppresses() {
        assert!(should_emit_full_sampling_receipt(
            SampledReceiptMode::Bounded,
            0
        ));
        assert!(should_emit_full_sampling_receipt(
            SampledReceiptMode::Bounded,
            BOUNDED_FULL_RECEIPT_LIMIT - 1
        ));
        assert!(!should_emit_full_sampling_receipt(
            SampledReceiptMode::Bounded,
            BOUNDED_FULL_RECEIPT_LIMIT
        ));
        assert!(should_emit_full_sampling_receipt(
            SampledReceiptMode::Full,
            u64::MAX
        ));
    }

    #[test]
    fn commit_plan_keeps_root_and_exact_committed_draft_prefix() {
        assert_eq!(
            sampled_commit_plan(16, 0).unwrap(),
            SampledCommitPlan {
                verify_rows: 16,
                committed_drafts: 0,
                total_target_slots: 1,
            }
        );
        assert_eq!(sampled_commit_plan(16, 7).unwrap().total_target_slots, 8);
        assert_eq!(sampled_commit_plan(16, 15).unwrap().total_target_slots, 16);
        assert!(sampled_commit_plan(16, 16).is_err());
    }

    #[test]
    fn thinking_phase_truth_table_rejects_only_transitions_and_contradictions() {
        for bits in 0u8..32 {
            let enable = bits & 1 != 0;
            let inside = bits & 2 != 0;
            let force = bits & 4 != 0;
            let ended = bits & 8 != 0;
            let just_ended = bits & 16 != 0;
            let expected =
                if force || just_ended || (inside && ended) || (enable && !inside && !ended) {
                    None
                } else if inside {
                    Some(SampledThinkingPhase::Inside)
                } else if ended {
                    Some(SampledThinkingPhase::Post)
                } else {
                    Some(SampledThinkingPhase::Outside)
                };
            assert_eq!(
                sampled_thinking_phase(enable, inside, force, ended, just_ended).ok(),
                expected,
                "phase truth-table mismatch for {bits:05b}"
            );
        }
        assert_eq!(
            sampled_thinking_phase(true, false, false, true, false).unwrap(),
            SampledThinkingPhase::Post,
            "enable_thinking remains true after a legitimate close"
        );
    }

    #[test]
    fn forced_and_just_ended_transitions_are_plain_boundaries() {
        assert!(!sampled_thinking_transition_requires_plain(false, false));
        assert!(sampled_thinking_transition_requires_plain(true, false));
        assert!(sampled_thinking_transition_requires_plain(false, true));
        assert!(sampled_thinking_transition_requires_plain(true, true));
    }

    #[test]
    fn sampled_width_gate_is_inert_without_parity_and_exact_with_parity() {
        for drafts in [0, 1, 2, 3, 15, 31, 32, usize::MAX] {
            assert!(
                sampled_width_preserves_tc_parity(false, drafts),
                "parity-off changed sampled width {drafts}"
            );
        }
        for drafts in [0, 1, 2, 32, usize::MAX] {
            assert!(!sampled_width_preserves_tc_parity(true, drafts));
        }
        for drafts in 3..=31 {
            assert!(sampled_width_preserves_tc_parity(true, drafts));
        }
    }

    #[test]
    fn source_contract_downgrades_boundaries_and_parity_widths_before_gpu_work() {
        let source = include_str!("sampled_dflash.rs");
        let propose = source
            .split("pub(super) fn propose_sampled_dflash(")
            .nth(1)
            .unwrap()
            .split("fn fail_sampled_step(")
            .next()
            .unwrap();

        let transition = propose
            .find("sampled_thinking_transition_requires_plain(")
            .unwrap();
        let route = propose
            .find("validate_sampled_route(model, a, think)?")
            .unwrap();
        let transition_fallback = &propose[transition..route];
        assert!(transition_fallback.contains("proposal_lifecycle::clear("));
        assert!(
            transition_fallback
                .contains("return Ok(SampledProposalOutcome::PlainThinkingBoundary)")
        );

        let gamma = propose
            .find("let gamma = model.dflash_sampled_proposal_width(")
            .unwrap();
        let parity = propose[gamma..]
            .find("sampled_width_preserves_tc_parity(")
            .unwrap()
            + gamma;
        let stage = propose
            .find("model.stage_dflash_sampling_request(")
            .unwrap();
        let install = propose
            .find("proposal_lifecycle::propose_and_install_with(")
            .unwrap();
        assert!(gamma < parity && parity < stage && stage < install);
        let parity_fallback = &propose[parity..stage];
        assert!(parity_fallback.contains("proposal_lifecycle::clear("));
        assert!(parity_fallback.contains("return Ok(SampledProposalOutcome::PlainTcParityWidth)"));

        let verify = source
            .split("pub(super) fn step_verify_sampled_dflash(")
            .nth(1)
            .unwrap()
            .split("#[cfg(test)]")
            .next()
            .unwrap();
        let validation = verify.find("validate_sampled_frame_policy(").unwrap();
        let parity = verify.find("sampled_width_preserves_tc_parity(").unwrap();
        let gpu_verify = verify.find("model.decode_verify_dflash(").unwrap();
        assert!(validation < parity && parity < gpu_verify);
        let parity_fallback = &verify[parity..gpu_verify];
        assert!(parity_fallback.contains("proposal_lifecycle::clear("));
        assert!(parity_fallback.contains("return;"));
    }

    #[test]
    fn cancellation_is_a_normal_transaction_outcome_around_propose_and_validation() {
        let source = include_str!("sampled_dflash.rs");
        let propose = source
            .split("pub(super) fn propose_sampled_dflash(")
            .nth(1)
            .unwrap()
            .split("fn fail_sampled_step(")
            .next()
            .unwrap();

        let route = propose
            .find("validate_sampled_route(model, a, think)?")
            .unwrap();
        let after_route = propose.find("dflash_after_route_validation").unwrap();
        let run_proposer = propose.find("propose_and_install_with(").unwrap();
        let after_propose = propose.find("dflash_after_propose").unwrap();
        let validate_installed = propose
            .find("validate_sampled_frame_policy(model, a, think")
            .unwrap();
        let after_validation = propose.find("dflash_after_installed_validation").unwrap();

        assert!(route < after_route && after_route < run_proposer);
        assert!(run_proposer < after_propose && after_propose < validate_installed);
        assert!(validate_installed < after_validation);
        assert!(propose.contains("return Ok(SampledProposalOutcome::Cancelled)"));
        assert!(
            !propose.contains("sampled DFlash request is already cancelled"),
            "client cancellation must not be converted back into an anyhow failure"
        );
    }

    #[test]
    fn inside_width_stops_before_every_stateful_policy_boundary() {
        assert_eq!(
            inside_policy_stable_width(4, 6, Some(8), 0, false, false),
            1
        );
        assert_eq!(inside_policy_stable_width(4, 8, None, 10, false, false), 1);
        assert_eq!(inside_policy_stable_width(4, 397, None, 0, true, false), 2);
        assert_eq!(inside_policy_stable_width(4, 399, None, 0, true, false), 0);
        assert_eq!(inside_policy_stable_width(3, 47, None, 0, false, true), 0);
        assert_eq!(inside_policy_stable_width(3, 48, None, 0, false, true), 3);
        assert_eq!(inside_policy_stable_width(4, 55, None, 0, false, true), 0);
        assert_eq!(inside_policy_stable_width(4, 17, None, 0, false, false), 4);
    }

    #[test]
    fn inside_biases_preserve_plain_path_order_duplicates_and_vocab_filter() {
        assert_eq!(
            prepend_inside_thinking_biases(&[(7, 0.25), (9, 1.0)], &[7, 3, 100, 7], 10,),
            vec![(7, -10.0), (3, -10.0), (7, -10.0), (7, 0.25), (9, 1.0)]
        );
    }

    #[test]
    fn phase_masks_are_sorted_deduplicated_and_floor_sensitive() {
        assert_eq!(
            sampled_policy_forbidden_tokens_for(
                SampledThinkingPhase::Post,
                0,
                0,
                Some(248_069),
                Some(248_068),
                None,
            ),
            vec![248_068, 248_069]
        );
        assert_eq!(
            sampled_policy_forbidden_tokens_for(
                SampledThinkingPhase::Post,
                0,
                0,
                Some(7),
                Some(7),
                None,
            ),
            vec![7]
        );
        assert_eq!(
            sampled_policy_forbidden_tokens_for(
                SampledThinkingPhase::Inside,
                7,
                8,
                Some(1),
                Some(9),
                Some(3),
            ),
            vec![3, 9]
        );
        assert_eq!(
            sampled_policy_forbidden_tokens_for(
                SampledThinkingPhase::Inside,
                8,
                8,
                Some(1),
                Some(9),
                Some(3),
            ),
            vec![3]
        );
        assert!(
            sampled_policy_forbidden_tokens_for(
                SampledThinkingPhase::Outside,
                0,
                0,
                Some(1),
                Some(2),
                Some(3),
            )
            .is_empty()
        );
    }

    #[test]
    fn phase_specific_structural_preflight_matches_plain_state_transitions() {
        const IM_START: u32 = 248_045;
        for boundary in [2, 7, 9, 10, 11, IM_START] {
            assert!(sampled_proposal_requires_plain(
                SampledThinkingPhase::Outside,
                &[21, boundary, 22],
                &[],
                Some(7),
                Some(9),
                Some(10),
                Some(11),
                &[2],
                Some(IM_START),
            ));
        }
        assert!(!sampled_proposal_requires_plain(
            SampledThinkingPhase::Outside,
            &[21, 22, 23],
            &[],
            Some(7),
            Some(9),
            Some(10),
            Some(11),
            &[2],
            Some(IM_START),
        ));

        for boundary in [2, 9] {
            assert!(sampled_proposal_requires_plain(
                SampledThinkingPhase::Inside,
                &[21, boundary, 22],
                &[],
                Some(7),
                Some(9),
                Some(10),
                Some(11),
                &[2],
                Some(IM_START),
            ));
        }
        for ordinary_inside in [7, 11, IM_START] {
            assert!(!sampled_proposal_requires_plain(
                SampledThinkingPhase::Inside,
                &[21, ordinary_inside, 22],
                &[],
                Some(7),
                Some(9),
                Some(10),
                Some(11),
                &[2],
                Some(IM_START),
            ));
        }
        assert!(sampled_proposal_requires_plain(
            SampledThinkingPhase::Inside,
            &[IM_START],
            &[IM_START, 3, IM_START, 4, IM_START],
            Some(7),
            Some(9),
            Some(10),
            Some(11),
            &[2],
            Some(IM_START),
        ));
        let two_prior_openers = [IM_START, 3, IM_START];
        assert!(!sampled_proposal_requires_plain(
            SampledThinkingPhase::Inside,
            &[IM_START],
            &two_prior_openers,
            Some(7),
            Some(9),
            Some(10),
            Some(11),
            &[2],
            Some(IM_START),
        ));
        assert!(sampled_proposal_requires_plain(
            SampledThinkingPhase::Inside,
            &[IM_START, IM_START],
            &two_prior_openers,
            Some(7),
            Some(9),
            Some(10),
            Some(11),
            &[2],
            Some(IM_START),
        ));

        for boundary in [2, 10, 11, IM_START] {
            assert!(sampled_proposal_requires_plain(
                SampledThinkingPhase::Post,
                &[21, boundary, 22],
                &[],
                Some(7),
                Some(9),
                Some(10),
                Some(11),
                &[2],
                Some(IM_START),
            ));
        }
        for hard_masked_think_marker in [7, 9] {
            assert!(!sampled_proposal_requires_plain(
                SampledThinkingPhase::Post,
                &[hard_masked_think_marker],
                &[],
                Some(7),
                Some(9),
                Some(10),
                Some(11),
                &[2],
                Some(IM_START),
            ));
        }
        assert!(sampled_proposal_requires_plain(
            SampledThinkingPhase::Inside,
            &[7],
            &[],
            Some(7),
            Some(7),
            None,
            None,
            &[],
            None,
        ));
        assert!(sampled_proposal_requires_plain(
            SampledThinkingPhase::Post,
            &[7],
            &[],
            Some(7),
            Some(7),
            None,
            Some(7),
            &[],
            None,
        ));
    }

    #[test]
    fn correction_boundary_includes_the_accepted_prefix_history() {
        const IM_START: u32 = 248_045;
        let old_history = [IM_START, 21];
        let common = (
            Some(7),
            Some(9),
            Some(10),
            Some(11),
            &[2][..],
            Some(IM_START),
        );
        assert!(!sampled_frame_boundary_after_prefix(
            SampledThinkingPhase::Inside,
            &[],
            IM_START,
            &old_history,
            common.0,
            common.1,
            common.2,
            common.3,
            common.4,
            common.5,
        ));
        assert!(sampled_frame_boundary_after_prefix(
            SampledThinkingPhase::Inside,
            &[IM_START, IM_START],
            IM_START,
            &old_history,
            common.0,
            common.1,
            common.2,
            common.3,
            common.4,
            common.5,
        ));
    }

    #[test]
    fn source_contract_commits_after_normal_emission_and_before_terminal_return() {
        let source = include_str!("sampled_dflash.rs");
        let body = source
            .split("pub(super) fn step_verify_sampled_dflash(")
            .nth(1)
            .unwrap()
            .split("#[cfg(test)]")
            .next()
            .unwrap();
        let emit = body
            .find("emit_sampled_token(a, acceptance.next_token")
            .unwrap();
        let commit = body.find("model.commit_verify_state_async(").unwrap();
        let terminal = body[commit..].find("if a.finished").unwrap() + commit;
        assert!(emit < commit && commit < terminal);
        assert_eq!(body.matches("model.commit_verify_state_async(").count(), 1);
    }

    #[test]
    fn source_contract_snapshots_think_end_only_after_canonical_commit_is_ordered() {
        let source = include_str!("sampled_dflash.rs");
        let emit_impl = source
            .split("fn emit_sampled_token(")
            .nth(1)
            .unwrap()
            .split("/// Stage exactly one request")
            .next()
            .unwrap();
        assert!(emit_impl.contains("defer_think_end_snapshot = true"));
        assert!(!emit_impl.contains("snapshot_boundary_if_ssm"));

        let body = source
            .split("pub(super) fn step_verify_sampled_dflash(")
            .nth(1)
            .unwrap()
            .split("#[cfg(test)]")
            .next()
            .unwrap();
        let emit = body
            .find("emit_sampled_token(a, acceptance.next_token")
            .unwrap();
        let commit = body.find("model.commit_verify_state_async(").unwrap();
        let sync = body[commit..].find("model.sync_secondary()?").unwrap() + commit;
        let snapshot = body[commit..]
            .find("rollback::snapshot_boundary_if_ssm(a, model)")
            .unwrap()
            + commit;
        assert!(emit < commit && commit < sync && sync < snapshot);
    }

    #[test]
    fn source_contract_validates_every_receipt_and_logs_compact_accounting_every_cycle() {
        let source = include_str!("sampled_dflash.rs");
        let body = source
            .split("pub(super) fn step_verify_sampled_dflash(")
            .nth(1)
            .unwrap()
            .split("#[cfg(test)]")
            .next()
            .unwrap();
        let receipt = body.find("DflashSamplingReceipt::new(").unwrap();
        let commit = body.find("model.commit_verify_state_async(").unwrap();
        let compact = body.find("\"DFLASH_SAMPLED_COMMIT\"").unwrap();
        let conditional_full = body.find("if emit_full_receipt").unwrap();
        let serialize = body.find("receipt.to_json_line()?").unwrap();
        assert!(commit < receipt && receipt < compact && compact < conditional_full);
        assert!(conditional_full < serialize);
        let compact_fields = &body[commit..compact];
        let accepted = compact_fields
            .find("accepted = acceptance.accepted")
            .unwrap();
        let finished = compact_fields.find("finished = a.finished").unwrap();
        let schema = compact_fields
            .find("schema = DFLASH_SAMPLING_RECEIPT_SCHEMA")
            .unwrap();
        assert!(
            accepted < finished && finished < schema,
            "new compact fields must trail the legacy parser prefix"
        );
        for field in [
            "schema = DFLASH_SAMPLING_RECEIPT_SCHEMA",
            "receipt_mode = %receipt_mode.as_str()",
            "full_receipt = emit_full_receipt",
            "receipt_ordinal = receipt_ordinal",
            "drafts = drafts.len()",
            "thinking_phase = phase.as_str()",
            "execution_sha256 = %execution_sha256",
            "accepted = acceptance.accepted",
            "committed_drafts = plan.committed_drafts",
            "emitted_next = next_armed",
            "next_disposition = ?next_disposition",
            "finished = a.finished",
        ] {
            assert!(
                body.contains(field),
                "missing compact receipt field: {field}"
            );
        }
    }

    #[test]
    fn source_contract_routes_structural_next_through_normal_emit_then_plain() {
        let source = include_str!("sampled_dflash.rs");
        let body = source
            .split("pub(super) fn step_verify_sampled_dflash(")
            .nth(1)
            .unwrap()
            .split("#[cfg(test)]")
            .next()
            .unwrap();
        let validation = body.find("validate_sampled_frame_policy(").unwrap();
        let preflight = body.find("if crosses_structural_boundary(drafts)").unwrap();
        let target_verify = body.find("model.decode_verify_dflash(").unwrap();
        let emit = body
            .find("emit_sampled_token(a, acceptance.next_token")
            .unwrap();
        let commit = body.find("model.commit_verify_state_async(").unwrap();
        let plain_boundary = body[commit..].find("if next_is_structural").unwrap() + commit;
        assert!(validation < preflight && preflight < target_verify && target_verify < emit);
        assert!(emit < commit && commit < plain_boundary);
        assert!(body.contains("target accepted through a structural draft"));

        let propose = source
            .split("pub(super) fn propose_sampled_dflash(")
            .nth(1)
            .unwrap()
            .split("fn fail_sampled_step(")
            .next()
            .unwrap();
        assert!(propose.contains("if sampled_proposal_requires_plain("));
        assert!(!propose.contains("phase == SampledThinkingPhase::Inside\n            &&"));
    }

    #[test]
    fn source_contract_has_no_greedy_or_structural_fallback() {
        let source = include_str!("sampled_dflash.rs");
        let implementation = source
            .split("#[cfg(test)]")
            .next()
            .expect("sampled DFlash implementation before tests");
        assert!(!implementation.contains("argmax_on_device"));
        assert!(!implementation.contains("unwrap_or_else(|_|"));
        for gate in [
            "sampled route requires DFlash",
            "sampled DFlash EP route is not implemented",
            "sampled route requires the probabilistic DFlash selector",
            "sampled DFlash requires positive finite request temperature",
            "sampled DFlash requires a scheduler-resolved effective seed",
            "grammar route is not implemented",
            "logprobs route is not implemented",
            "inside-thinking route requires the policy-oracle gate",
            "inside-thinking target policy requires neutral history penalties",
            "inside-thinking efficiency-wave policy is not receipted",
            "holds while a forced </think> transition is pending",
            "adaptive sampler is not implemented",
            "tool route is not implemented",
            "tree/portfolio route is not implemented",
        ] {
            assert!(implementation.contains(gate), "missing HOLD gate: {gate}");
        }
    }

    #[test]
    fn source_contract_reads_logits_once_and_drains_frame_at_dispatch() {
        let source = include_str!("sampled_dflash.rs");
        let body = source
            .split("pub(super) fn step_verify_sampled_dflash(")
            .nth(1)
            .unwrap()
            .split("#[cfg(test)]")
            .next()
            .unwrap();
        assert_eq!(body.matches("model.copy_logits_to_host(").count(), 1);
        let copy = body.find("model.copy_logits_to_host(").unwrap();
        let evaluate = body.find("evaluate_sampled_dflash_with_evidence(").unwrap();
        let emit = body.find("emit_sampled_token(a,").unwrap();
        let commit = body.find("model.commit_verify_state_async(").unwrap();
        assert!(copy < evaluate && evaluate < emit && emit < commit);
        let verify = include_str!("verify_dflash_step.rs");
        assert!(verify.contains("a.pending_dflash_sampled_proposal.take()"));
        assert!(verify.contains(
            "probabilistic DFlash draft frame reached verify without its sampled-q proposal"
        ));
        assert!(
            verify.contains("greedy DFlash verifier rejected an unexpected sampled-q proposal")
        );
    }
}
