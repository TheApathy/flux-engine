// SPDX-License-Identifier: AGPL-3.0-only

//! Scheduler admission, physical slot ownership guards, and fail-closed device
//! outcome interpretation for opt-in hardened on-GPU exact speculative
//! acceptance (`ATLAS_SPEC_EXACT_GPU_ACCEPT=1`, default OFF).
//!
//! When `ATLAS_SPEC_EXACT_GPU_ACCEPT` is unset or `"0"`, this module returns
//! [`ExactGpuAcceptEligibility::Disabled`] without touching any GPU state,
//! keeping the baseline verify path byte-identical.

use anyhow::{Context, Result, ensure};
use spark_model::layers::ops::exact_spec_accept::{
    EXACT_SPEC_MAX_WIDTH, ExactSpecGpuAcceptPolicy, ExactSpecSlotGuard, validate_slot_guards_host,
};

use super::think_spec_accept::ThinkSpecCtx;
use super::types::ActiveSeq;

/// Fail-closed scheduler admission decision for `exact_spec_accept_sm120`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ExactGpuAcceptEligibility {
    /// Sequence and window width satisfy every exact-equivalence precondition.
    Eligible,
    /// `ATLAS_SPEC_EXACT_GPU_ACCEPT` is disabled (default OFF).
    Disabled,
    /// Sampling parameters are greedy, non-finite, missing a Philox seed, or
    /// include unsupported truncation filters (`top_k`, `top_p`, `top_n_sigma`,
    /// `min_p`).
    UnsupportedSampling,
    /// Verify window width `W` is outside `1..=16`.
    UnsupportedWidth,
    /// Sequence or runtime carries unsupported modifiers (windowed penalties,
    /// repetition/DRY/LZ penalties, logit bias, grammar, logprobs, DDTree
    /// payload, min-tokens, think/tool/nudge/adaptive/dsv41/receipt policies,
    /// FP32 decode logits, or is already finished).
    UnsupportedModifiers,
}

/// Evaluate whether `a` is eligible for hardened on-GPU exact acceptance over a
/// verify window of width `verify_width` under `policy`.
///
/// Strictly fail-closed: admits only finite positive-temperature, explicitly
/// seeded BF16-logit sampling whose exact target distribution is fully captured
/// by `exact_spec_accept_sparse_sm120` (temperature scaling plus full-history
/// prefix-aware presence & frequency penalties).
#[must_use]
pub(super) fn evaluate_exact_gpu_accept_eligibility(
    a: &ActiveSeq,
    verify_width: usize,
    think: &ThinkSpecCtx<'_>,
    logits_fp32: bool,
    policy: ExactSpecGpuAcceptPolicy,
) -> ExactGpuAcceptEligibility {
    if policy != ExactSpecGpuAcceptPolicy::Enabled {
        return ExactGpuAcceptEligibility::Disabled;
    }
    if !(1..=(EXACT_SPEC_MAX_WIDTH as usize)).contains(&verify_width) {
        return ExactGpuAcceptEligibility::UnsupportedWidth;
    }
    if !a.temperature.is_finite()
        || a.temperature <= 0.0
        || !a.presence_penalty.is_finite()
        || !a.frequency_penalty.is_finite()
        || a.top_k != 0
        || !a.top_p.is_finite()
        || a.top_p != 1.0
        || !a.top_n_sigma.is_finite()
        || a.top_n_sigma != 0.0
        || !a.min_p.is_finite()
        || a.min_p != 0.0
        || a.seed.is_none()
    {
        return ExactGpuAcceptEligibility::UnsupportedSampling;
    }
    let windowed_history_penalty = (a.presence_penalty != 0.0 || a.frequency_penalty != 0.0)
        && a.repetition_penalty_window != 0;
    if logits_fp32
        || windowed_history_penalty
        || a.inside_thinking
        || a.think_ended
        || a.think_just_ended
        || a.force_end_thinking
        || a.budget_nudge.is_some()
        || super::think_nudge::forced_token(a).is_some()
        || super::think_budget_nudge::forced_token(a).is_some()
        || think.adaptive_sampling
        || super::decode_logits_seq::logits_fnv_requires_row()
        || super::logits_row_receipt::enabled()
        || crate::dsv41::serving()
        || crate::dsv41::repetition::active()
        || a.finished
        || !a.repetition_penalty.is_finite()
        || a.repetition_penalty != 1.0
        || !a.lz_penalty.is_finite()
        || a.lz_penalty != 0.0
        || !a.dry_multiplier.is_finite()
        || a.dry_multiplier != 0.0
        || !a.logit_bias.is_empty()
        || a.grammar_state.is_some()
        || a.top_logprobs.is_some()
        || a.pending_tree_payload.is_some()
        || a.suppress_tool_call
        || a.require_tool_call
        || a.output_tokens.len() < a.min_tokens
    {
        return ExactGpuAcceptEligibility::UnsupportedModifiers;
    }
    ExactGpuAcceptEligibility::Eligible
}

/// Construct the physical slot ownership guard for `a` before launching
/// `exact_spec_prefix_reduce_sm120`.
///
/// Consumed exclusively by callers holding real pool generations and refcounts
/// (never fabricated by diagnostic hooks).
///
/// # Errors
/// Fails closed if `a.seq.slot_idx` exceeds `u32::MAX`.
#[allow(dead_code)]
pub(super) fn build_slot_guard(
    a: &ActiveSeq,
    slot_generation: u64,
    expected_generation: u64,
    slot_refcount: u32,
) -> Result<ExactSpecSlotGuard> {
    let slot_id = u32::try_from(a.seq.slot_idx)
        .with_context(|| format!("slot_idx {} exceeds u32::MAX", a.seq.slot_idx))?;
    Ok(ExactSpecSlotGuard {
        slot_id,
        slot_generation,
        expected_generation,
        slot_refcount,
    })
}

/// Validate that every sequence in a batch holds an exclusive (`refcount == 1`),
/// generation-matched, pairwise-distinct physical slot before zero-host-sync commit.
///
/// Consumed exclusively by callers supplying real pool generations and refcounts.
///
/// # Errors
/// Returns an error identifying the first lane that violated `Generation` or
/// `sharing <= 1`.
#[allow(dead_code)]
pub(super) fn validate_batch_slot_guards(guards: &[ExactSpecSlotGuard]) -> Result<()> {
    ensure!(!guards.is_empty(), "empty exact GPU accept batch");
    let statuses = validate_slot_guards_host(guards);
    for (lane, (&guard, &status)) in guards.iter().zip(statuses.iter()).enumerate() {
        ensure!(
            status.is_ok(),
            "exact GPU accept slot guard rejected lane {lane} (slot_id={} gen={} expected={} refcount={}): {status}",
            guard.slot_id,
            guard.slot_generation,
            guard.expected_generation,
            guard.slot_refcount
        );
    }
    Ok(())
}

/// Record a single-sequence opt-in admission diagnostic when
/// `ATLAS_SPEC_EXACT_GPU_ACCEPT=1` is active; strict no-op when disabled.
pub(super) fn audit_single_seq_exact_gpu_accept_if_enabled(
    a: &ActiveSeq,
    verify_width: usize,
    think: &ThinkSpecCtx<'_>,
    logits_fp32: bool,
    policy: ExactSpecGpuAcceptPolicy,
) -> ExactGpuAcceptEligibility {
    let eligibility =
        evaluate_exact_gpu_accept_eligibility(a, verify_width, think, logits_fp32, policy);
    if policy == ExactSpecGpuAcceptPolicy::Enabled {
        tracing::debug!(
            slot_idx = a.seq.slot_idx,
            verify_width,
            ?eligibility,
            "ATLAS_SPEC_EXACT_GPU_ACCEPT single-seq verify admission audit"
        );
    }
    eligibility
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use spark_model::layers::ops::exact_spec_accept::{
        CpuExactSpecRowInput, CpuExactSpecRowKind, EXACT_SPEC_INVALID_TOKEN, ExactGpuAcceptOutcome,
        ExactSpecStatus, cpu_exact_spec_accept_sparse_row, cpu_exact_spec_prefix_reduce,
        f32_to_bf16_bits, interpret_device_accept_outcome,
    };

    use super::*;
    use crate::scheduler::test_fixtures;
    use crate::scheduler::think_budget_nudge::BudgetNudge;
    use crate::scheduler::types::ResponseSink;

    fn neutral_think_ctx() -> ThinkSpecCtx<'static> {
        ThinkSpecCtx {
            enabled: true,
            code_fence_token: None,
            reflection_suppress_ids: &[],
            adaptive_sampling: false,
            honor_eos_inside_thinking: None,
        }
    }

    fn sampled_active_seq(slot: usize) -> ActiveSeq {
        let (tx, _rx) = tokio::sync::oneshot::channel();
        let mut a = test_fixtures::active_seq(ResponseSink::Blocking(Some(tx)), slot);
        a.temperature = 0.8;
        a.presence_penalty = 0.25;
        a.frequency_penalty = 0.1;
        a.repetition_penalty_window = 0;
        a.think_ended = false;
        a.seed = Some(0xCAFE_BABE_0000_0001);
        a
    }

    #[test]
    fn admission_defaults_off_and_rejects_every_unsupported_modifier() {
        let think = neutral_think_ctx();
        let base = sampled_active_seq(0);
        assert_eq!(
            evaluate_exact_gpu_accept_eligibility(
                &base,
                8,
                &think,
                false,
                ExactSpecGpuAcceptPolicy::Disabled
            ),
            ExactGpuAcceptEligibility::Disabled
        );
        for w in [1_usize, 8, 16] {
            assert_eq!(
                evaluate_exact_gpu_accept_eligibility(
                    &base,
                    w,
                    &think,
                    false,
                    ExactSpecGpuAcceptPolicy::Enabled
                ),
                ExactGpuAcceptEligibility::Eligible
            );
        }

        // Width outside 1..=16.
        for bad_w in [0_usize, 17, 32] {
            assert_eq!(
                evaluate_exact_gpu_accept_eligibility(
                    &base,
                    bad_w,
                    &think,
                    false,
                    ExactSpecGpuAcceptPolicy::Enabled
                ),
                ExactGpuAcceptEligibility::UnsupportedWidth
            );
        }

        // Sampling disqualifiers: greedy, non-finite, top_k, top_p != 1.0,
        // top_n_sigma != 0.0, min_p != 0.0, missing Philox seed.
        for mutate in [
            (|s: &mut ActiveSeq| s.temperature = 0.0) as fn(&mut ActiveSeq),
            |s| s.temperature = f32::NAN,
            |s| s.top_k = 40,
            |s| s.top_p = 0.95,
            |s| s.top_p = 1.5,
            |s| s.top_n_sigma = 1.0,
            |s| s.top_n_sigma = -0.5,
            |s| s.min_p = 0.05,
            |s| s.min_p = -0.1,
            |s| s.seed = None,
        ] {
            let mut s = sampled_active_seq(0);
            mutate(&mut s);
            assert_eq!(
                evaluate_exact_gpu_accept_eligibility(
                    &s,
                    8,
                    &think,
                    false,
                    ExactSpecGpuAcceptPolicy::Enabled
                ),
                ExactGpuAcceptEligibility::UnsupportedSampling
            );
        }

        // Windowed presence/frequency penalties must fail closed while
        // neutral zero penalties with a non-zero window remain eligible.
        let mut win_pres = sampled_active_seq(0);
        win_pres.frequency_penalty = 0.0;
        win_pres.repetition_penalty_window = 64;
        assert_eq!(
            evaluate_exact_gpu_accept_eligibility(
                &win_pres,
                8,
                &think,
                false,
                ExactSpecGpuAcceptPolicy::Enabled
            ),
            ExactGpuAcceptEligibility::UnsupportedModifiers
        );
        let mut win_freq = sampled_active_seq(0);
        win_freq.presence_penalty = 0.0;
        win_freq.repetition_penalty_window = 64;
        assert_eq!(
            evaluate_exact_gpu_accept_eligibility(
                &win_freq,
                8,
                &think,
                false,
                ExactSpecGpuAcceptPolicy::Enabled
            ),
            ExactGpuAcceptEligibility::UnsupportedModifiers
        );
        let mut win_zero_pen = sampled_active_seq(0);
        win_zero_pen.presence_penalty = 0.0;
        win_zero_pen.frequency_penalty = 0.0;
        win_zero_pen.repetition_penalty_window = 64;
        assert_eq!(
            evaluate_exact_gpu_accept_eligibility(
                &win_zero_pen,
                8,
                &think,
                false,
                ExactSpecGpuAcceptPolicy::Enabled
            ),
            ExactGpuAcceptEligibility::Eligible
        );

        // FP32 decode logits and adaptive sampling must fail closed.
        assert_eq!(
            evaluate_exact_gpu_accept_eligibility(
                &base,
                8,
                &think,
                true,
                ExactSpecGpuAcceptPolicy::Enabled
            ),
            ExactGpuAcceptEligibility::UnsupportedModifiers
        );
        let mut adaptive_think = neutral_think_ctx();
        adaptive_think.adaptive_sampling = true;
        assert_eq!(
            evaluate_exact_gpu_accept_eligibility(
                &base,
                8,
                &adaptive_think,
                false,
                ExactSpecGpuAcceptPolicy::Enabled
            ),
            ExactGpuAcceptEligibility::UnsupportedModifiers
        );

        // Think / nudge / penalty / bias / logprobs / tool / min-token / finished modifiers.
        for mutate in [
            (|m: &mut ActiveSeq| m.inside_thinking = true) as fn(&mut ActiveSeq),
            |m| m.think_ended = true,
            |m| m.think_just_ended = true,
            |m| m.force_end_thinking = true,
            |m| m.budget_nudge = BudgetNudge::plan(Some(Arc::from([10_u32, 11])), 1, 100),
            |m| m.repetition_penalty = 1.1,
            |m| m.dry_multiplier = 0.5,
            |m| m.lz_penalty = 0.2,
            |m| m.logit_bias.push((42, -5.0)),
            |m| m.top_logprobs = Some(5),
            |m| m.suppress_tool_call = true,
            |m| m.require_tool_call = true,
            |m| m.min_tokens = m.output_tokens.len() + 1,
            |m| m.finished = true,
        ] {
            let mut m = sampled_active_seq(0);
            mutate(&mut m);
            assert_eq!(
                evaluate_exact_gpu_accept_eligibility(
                    &m,
                    8,
                    &think,
                    false,
                    ExactSpecGpuAcceptPolicy::Enabled
                ),
                ExactGpuAcceptEligibility::UnsupportedModifiers
            );
        }
    }

    #[test]
    fn slot_guard_enforces_generation_and_exclusivity_across_batch() {
        let a0 = sampled_active_seq(2);
        let a1 = sampled_active_seq(5);
        let a2 = sampled_active_seq(2); // collides physically with a0!

        let g0 = build_slot_guard(&a0, 10, 10, 1).unwrap();
        let g1 = build_slot_guard(&a1, 11, 11, 1).unwrap();
        assert!(validate_batch_slot_guards(&[g0, g1]).is_ok());

        // Stale generation must fail closed.
        let g_stale = build_slot_guard(&a1, 10, 11, 1).unwrap();
        assert!(validate_batch_slot_guards(&[g0, g_stale]).is_err());

        // Refcount != 1 must fail closed.
        let g_shared = build_slot_guard(&a1, 11, 11, 2).unwrap();
        assert!(validate_batch_slot_guards(&[g0, g_shared]).is_err());

        // Duplicate physical slot_id across batch must fail closed.
        let g_dup = build_slot_guard(&a2, 10, 10, 1).unwrap();
        assert!(validate_batch_slot_guards(&[g0, g_dup]).is_err());
    }

    #[test]
    fn outcome_interpreter_accepts_valid_and_fails_closed_on_faults() {
        assert_eq!(
            interpret_device_accept_outcome(3, 42, ExactSpecStatus::OK, 8, 1000),
            ExactGpuAcceptOutcome::Accepted {
                num_accepted: 3,
                bonus_token: 42,
                fallback_used: false,
            }
        );
        assert_eq!(
            interpret_device_accept_outcome(0, 7, ExactSpecStatus::FALLBACK_USED, 8, 1000),
            ExactGpuAcceptOutcome::Accepted {
                num_accepted: 0,
                bonus_token: 7,
                fallback_used: true,
            }
        );
        assert_eq!(
            interpret_device_accept_outcome(0, 42, ExactSpecStatus::DRAFT_SUM_ERR, 8, 1000),
            ExactGpuAcceptOutcome::FailClosedFallback {
                status: ExactSpecStatus::DRAFT_SUM_ERR,
            }
        );
        assert_eq!(
            interpret_device_accept_outcome(8, 42, ExactSpecStatus::OK, 8, 1000),
            ExactGpuAcceptOutcome::FailClosedFallback {
                status: ExactSpecStatus::BAD_PARAM,
            }
        );
        assert_eq!(
            interpret_device_accept_outcome(1, 1000, ExactSpecStatus::OK, 8, 1000),
            ExactGpuAcceptOutcome::FailClosedFallback {
                status: ExactSpecStatus::BAD_PARAM,
            }
        );
        assert_eq!(
            interpret_device_accept_outcome(
                1,
                EXACT_SPEC_INVALID_TOKEN,
                ExactSpecStatus::OK,
                8,
                1000
            ),
            ExactGpuAcceptOutcome::FailClosedFallback {
                status: ExactSpecStatus::BAD_PARAM,
            }
        );
    }

    #[test]
    fn scheduler_end_to_end_oracle_roundtrip_matches_interpret_outcome() {
        let think = neutral_think_ctx();
        let a = sampled_active_seq(1);
        assert_eq!(
            audit_single_seq_exact_gpu_accept_if_enabled(
                &a,
                2,
                &think,
                false,
                ExactSpecGpuAcceptPolicy::Enabled
            ),
            ExactGpuAcceptEligibility::Eligible
        );
        let guard = build_slot_guard(&a, 7, 7, 1).unwrap();
        let logits: Vec<u16> = [0.2_f32, 3.0, -0.5, 1.0, 0.0, 0.5, -1.0, 2.0]
            .into_iter()
            .map(f32_to_bf16_bits)
            .collect();
        let row0 = cpu_exact_spec_accept_sparse_row(&CpuExactSpecRowInput {
            logits_bf16: &logits,
            kind: CpuExactSpecRowKind::Draft {
                draft_token: 1,
                sparse_ids: &[1, 7],
                sparse_probs: &[0.5, 0.5],
            },
            prefix_tokens: &[],
            pre_verify_counts: None,
            seed: a.seed.unwrap(),
            abs_pos: 50,
            temperature: a.temperature,
            presence_penalty: a.presence_penalty,
            frequency_penalty: a.frequency_penalty,
        });
        let row1 = cpu_exact_spec_accept_sparse_row(&CpuExactSpecRowInput {
            logits_bf16: &logits,
            kind: CpuExactSpecRowKind::Bonus,
            prefix_tokens: &[1],
            pre_verify_counts: None,
            seed: a.seed.unwrap(),
            abs_pos: 51,
            temperature: a.temperature,
            presence_penalty: a.presence_penalty,
            frequency_penalty: a.frequency_penalty,
        });
        let seq_out = cpu_exact_spec_prefix_reduce(&[row0, row1], &[guard], 2).unwrap()[0];
        let interpreted = interpret_device_accept_outcome(
            seq_out.num_accepted,
            seq_out.bonus_token,
            seq_out.status,
            2,
            logits.len() as u32,
        );
        assert!(matches!(
            interpreted,
            ExactGpuAcceptOutcome::Accepted { bonus_token, .. } if bonus_token < 8
        ));
    }
}
