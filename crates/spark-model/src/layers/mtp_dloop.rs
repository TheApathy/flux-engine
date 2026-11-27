// SPDX-License-Identifier: AGPL-3.0-only

//! Zero-training confidence-gated Native-MTP DLoop (`ATLAS_MTP_DLOOP=1`, default OFF).
//!
//! Implements the two-stage looped drafting gate from DLoop (arXiv:2610.07659, Eq. 2 &
//! Table A5) for dense native-MTP targets: after completing Stage-1 (`k = 4` drafts,
//! `K = 5` verify width), compute the cumulative Stage-1 log-probability confidence
//!
//! ```text
//! c_1 = sum_{i=1..k} ln(p_i)
//! ```
//!
//! When `c_1 >= threshold` (default `-0.5`), extend drafting by a second stage
//! (`+4` drafts -> `8` total drafts, `K = 9` verify width) before target verification.
//!
//! ## Strict Safety & Roofline Guards (Fail-Closed)
//!
//! 1. **Default OFF**: requires explicit `ATLAS_MTP_DLOOP=1`.
//! 2. **Dense-only (strictly refused on MoE)**: on sparse MoE targets (e.g., 512-expert
//!    Qwen3.8-Flash-Next), widening `K = 5 -> K = 9` increases unique activated experts
//!    per layer and adds +25..32 ms DRAM traffic on GB10 (net throughput regression).
//!    Any target with `num_experts > 1` is strictly refused fail-closed.
//! 3. **Low concurrency (`C <= 2`)**: widening `K = 5 -> 9` is only memory-bound-free
//!    when live concurrency `<= 2`; higher concurrency stays at Stage 1 (`S = 1`).
//! 4. **Physical verify capacity**: requires `max_verify_k >= 9` (`total_drafts + 1`).
//! 5. **Lossless**: drafts only propose; target verification remains bit-exact.

use std::cell::Cell;

/// Master opt-in environment variable for zero-training native-MTP DLoop (default OFF).
pub const MTP_DLOOP_ENV: &str = "ATLAS_MTP_DLOOP";

/// Optional Stage-1 cumulative log-probability gate threshold override (default `-0.5`).
pub const MTP_DLOOP_THRESH_ENV: &str = "ATLAS_MTP_DLOOP_THRESH";

/// Maximum live sequence concurrency admitted for native-MTP DLoop (`C <= 2`).
pub const MTP_DLOOP_MAX_CONCURRENCY: usize = 2;

/// Default Stage-1 cumulative log-probability threshold (`c_1 = sum ln p_i >= -0.5`).
pub const MTP_DLOOP_DEFAULT_LOGPROB_THRESH: f32 = -0.5;

/// Stage-1 draft count (`K = 5` verify width = 1 anchor + 4 drafts).
pub const MTP_DLOOP_STAGE1_DRAFTS: usize = 4;

/// Additional Stage-2 drafts when the Stage-1 confidence gate passes.
pub const MTP_DLOOP_STAGE2_EXTRA_DRAFTS: usize = 4;

/// Combined `S = 2` total draft count (`K = 9` verify width = 1 anchor + 8 drafts).
pub const MTP_DLOOP_STAGE2_TOTAL_DRAFTS: usize =
    MTP_DLOOP_STAGE1_DRAFTS + MTP_DLOOP_STAGE2_EXTRA_DRAFTS;

/// Physical verify width `K = 9` required to verify an extended `S = 2` proposal.
pub const MTP_DLOOP_STAGE2_VERIFY_K: usize = MTP_DLOOP_STAGE2_TOTAL_DRAFTS + 1;

/// Target model FFN topology used to enforce dense-only DLoop admission.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TargetFfnTopology {
    /// Dense FFN target (`num_experts <= 1`, e.g., Qwen3.8-27B).
    Dense,
    /// Sparse Mixture-of-Experts target (`num_experts > 1`, e.g., Flash-Next 512E).
    Moe { num_experts: usize },
}

impl TargetFfnTopology {
    /// Classify a model topology from its configured `num_experts`.
    #[inline]
    #[must_use]
    pub const fn from_num_experts(num_experts: usize) -> Self {
        if num_experts > 1 {
            Self::Moe { num_experts }
        } else {
            Self::Dense
        }
    }

    /// Returns `true` for sparse MoE topologies (which DLoop must refuse).
    #[inline]
    #[must_use]
    pub const fn is_moe(self) -> bool {
        matches!(self, Self::Moe { .. })
    }
}

/// Fail-closed refusal reason when native-MTP DLoop stays at Stage 1 (`S = 1`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MtpDloopRefusal {
    /// `ATLAS_MTP_DLOOP` is not set to `"1"`.
    Disabled,
    /// Sparse MoE targets (`num_experts > 1`) are strictly refused to avoid
    /// unique-expert DRAM expansion on `K = 5 -> 9`.
    MoeTargetRefused { num_experts: usize },
    /// Active concurrency exceeds `MTP_DLOOP_MAX_CONCURRENCY` (or cohort guard absent).
    ConcurrencyTooHigh {
        live_seqs: usize,
        max_concurrency: usize,
    },
    /// Stage-1 draft width is not eligible for `K = 5 -> 9` extension.
    InsufficientStage1Drafts { stage1_drafts: usize },
    /// Target verify pool lacks physical capacity for `K = 9`.
    VerifyCapacityExceeded {
        required_k: usize,
        max_verify_k: usize,
    },
    /// Stage-1 probabilities were empty, mismatched, non-finite, or outside `[0, 1]`.
    InvalidProbabilities,
    /// Cumulative Stage-1 log-probability `c_1 = sum ln p_i` was below threshold.
    ConfidenceBelowThreshold,
}

/// Pure admission decision for Stage-2 native-MTP DLoop extension.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MtpDloopDecision {
    /// Extend Stage 1 with `extra_drafts` additional drafts (`K = 5 -> K = 9`).
    ExtendStage2 {
        extra_drafts: usize,
        total_drafts: usize,
    },
    /// Keep Stage 1 drafts unchanged (`S = 1`) for the reported reason.
    StayStage1(MtpDloopRefusal),
}

impl MtpDloopDecision {
    /// Returns `Some(extra_drafts)` when Stage 2 is admitted.
    #[inline]
    #[must_use]
    pub const fn extra_drafts(self) -> Option<usize> {
        match self {
            Self::ExtendStage2 { extra_drafts, .. } => Some(extra_drafts),
            Self::StayStage1(_) => None,
        }
    }
}

/// Pre-Stage-1 admission inputs checked before requesting probabilities on GPU.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MtpDloopStage1PrecheckInput {
    pub enabled: bool,
    pub topology: TargetFfnTopology,
    pub live_seqs: usize,
    pub stage1_drafts: usize,
    pub max_verify_k: usize,
}

/// Full Stage-2 extension evaluation inputs after Stage-1 probabilities arrive.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MtpDloopInput<'a> {
    pub enabled: bool,
    pub topology: TargetFfnTopology,
    pub live_seqs: usize,
    pub stage1_drafts: usize,
    pub max_verify_k: usize,
    pub stage1_probs: &'a [f32],
    pub threshold: f32,
}

/// Scoped cohort metadata delivered from `step_mtp` to native MTP proposers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MtpDloopCohortContext {
    pub live_seqs: usize,
    pub max_verify_k: usize,
}

thread_local! {
    static ACTIVE_COHORT: Cell<Option<MtpDloopCohortContext>> = const { Cell::new(None) };
}

/// RAII scope publishing the active step's `(live_seqs, max_verify_k)` to native
/// MTP proposers. Restores the previous context on drop.
pub struct MtpDloopCohortGuard {
    prev: Option<MtpDloopCohortContext>,
}

impl MtpDloopCohortGuard {
    /// Enter a scoped native-MTP DLoop cohort context for the current scheduler step.
    #[must_use]
    pub fn enter(live_seqs: usize, max_verify_k: usize) -> Self {
        let prev = ACTIVE_COHORT.with(|cell| {
            cell.replace(Some(MtpDloopCohortContext {
                live_seqs,
                max_verify_k,
            }))
        });
        Self { prev }
    }
}

impl Drop for MtpDloopCohortGuard {
    fn drop(&mut self) {
        let prev = self.prev;
        ACTIVE_COHORT.with(|cell| cell.set(prev));
    }
}

/// Read the active scheduler cohort context (`None` outside `MtpDloopCohortGuard`,
/// failing closed).
#[inline]
#[must_use]
pub fn active_cohort_context() -> Option<MtpDloopCohortContext> {
    ACTIVE_COHORT.with(Cell::get)
}

/// Parse `ATLAS_MTP_DLOOP`: exact `"1"` (trimmed) enables; everything else stays OFF.
#[inline]
#[must_use]
pub fn parse_mtp_dloop_enabled(raw: Option<&str>) -> bool {
    raw.map(str::trim) == Some("1")
}

/// Parse `ATLAS_MTP_DLOOP_THRESH`: finite `<= 0.0` log-probability threshold,
/// failing closed to [`MTP_DLOOP_DEFAULT_LOGPROB_THRESH`] (`-0.5`) on unset/invalid.
#[inline]
#[must_use]
pub fn parse_mtp_dloop_thresh(raw: Option<&str>) -> f32 {
    let Some(trimmed) = raw.map(str::trim).filter(|s| !s.is_empty()) else {
        return MTP_DLOOP_DEFAULT_LOGPROB_THRESH;
    };
    match trimmed.parse::<f32>() {
        Ok(val) if val.is_finite() && val <= 0.0 => val,
        _ => {
            tracing::warn!(
                "{MTP_DLOOP_THRESH_ENV}={raw:?} invalid (want finite float <= 0.0); \
                 using default {MTP_DLOOP_DEFAULT_LOGPROB_THRESH}"
            );
            MTP_DLOOP_DEFAULT_LOGPROB_THRESH
        }
    }
}

/// Process-lifetime cached gate for `ATLAS_MTP_DLOOP=1` (default OFF).
#[must_use]
pub fn mtp_dloop_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| parse_mtp_dloop_enabled(std::env::var(MTP_DLOOP_ENV).ok().as_deref()))
}

/// Process-lifetime cached Stage-1 confidence threshold (`ATLAS_MTP_DLOOP_THRESH`,
/// default `-0.5`).
#[must_use]
pub fn mtp_dloop_thresh() -> f32 {
    static THRESH: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
    *THRESH
        .get_or_init(|| parse_mtp_dloop_thresh(std::env::var(MTP_DLOOP_THRESH_ENV).ok().as_deref()))
}

/// Compute the required minimum SSM intermediate capacity for a model at load time.
///
/// Widens to [`MTP_DLOOP_STAGE2_VERIFY_K`] (`9`) ONLY when `enabled`, `has_mtp`,
/// `!dflash_enabled`, and `topology == TargetFfnTopology::Dense`; returns `0`
/// (no extra allocation) on MoE targets or when disabled.
#[inline]
#[must_use]
pub const fn required_verify_k_for_model(
    enabled: bool,
    has_mtp: bool,
    dflash_enabled: bool,
    topology: TargetFfnTopology,
) -> usize {
    if enabled && has_mtp && !dflash_enabled && !topology.is_moe() {
        MTP_DLOOP_STAGE2_VERIFY_K
    } else {
        0
    }
}

/// Compute Stage-1 cumulative log-probability confidence `c_1 = sum_{i=1..k} ln(p_i)`.
///
/// - Returns `None` if `probs` is empty or any entry is non-finite or outside `[0.0, 1.0]`.
/// - Returns `Some(f32::NEG_INFINITY)` if any valid entry equals `0.0`.
/// - Otherwise returns `Some(sum)` where `sum <= 0.0`.
#[must_use]
pub fn stage1_log_prob_sum(probs: &[f32]) -> Option<f32> {
    if probs.is_empty() {
        return None;
    }
    let mut sum = 0.0f32;
    let mut saw_zero = false;
    for &p in probs {
        if !p.is_finite() || !(0.0..=1.0).contains(&p) {
            return None;
        }
        if p == 0.0 {
            saw_zero = true;
        } else {
            sum += p.ln();
        }
    }
    if saw_zero {
        Some(f32::NEG_INFINITY)
    } else {
        sum.is_finite().then_some(sum)
    }
}

/// Pre-Stage-1 admission gate checked before computing Stage-1 probabilities.
///
/// Returns `Ok(MTP_DLOOP_STAGE2_EXTRA_DRAFTS)` when Stage 1 is eligible for a
/// confidence-gated Stage-2 extension, or `Err(refusal)` otherwise.
pub fn precheck_stage1_admission(
    input: &MtpDloopStage1PrecheckInput,
) -> Result<usize, MtpDloopRefusal> {
    if !input.enabled {
        return Err(MtpDloopRefusal::Disabled);
    }
    if let TargetFfnTopology::Moe { num_experts } = input.topology {
        return Err(MtpDloopRefusal::MoeTargetRefused { num_experts });
    }
    if input.live_seqs == 0 || input.live_seqs > MTP_DLOOP_MAX_CONCURRENCY {
        return Err(MtpDloopRefusal::ConcurrencyTooHigh {
            live_seqs: input.live_seqs,
            max_concurrency: MTP_DLOOP_MAX_CONCURRENCY,
        });
    }
    if input.stage1_drafts < MTP_DLOOP_STAGE1_DRAFTS {
        return Err(MtpDloopRefusal::InsufficientStage1Drafts {
            stage1_drafts: input.stage1_drafts,
        });
    }
    let extra = MTP_DLOOP_STAGE2_EXTRA_DRAFTS;
    let required_k = input.stage1_drafts.saturating_add(extra).saturating_add(1);
    if input.max_verify_k < required_k {
        return Err(MtpDloopRefusal::VerifyCapacityExceeded {
            required_k,
            max_verify_k: input.max_verify_k,
        });
    }
    Ok(extra)
}

/// Evaluate whether Stage-1 native-MTP drafts should extend to Stage 2 (`K = 5 -> 9`).
///
/// Strictly fail-closed on MoE targets (`num_experts > 1`), high concurrency (`C > 2`),
/// insufficient verify capacity, invalid/non-finite probabilities, or `c_1 < threshold`.
#[must_use]
pub fn evaluate_stage2_extension(input: &MtpDloopInput<'_>) -> MtpDloopDecision {
    let precheck = MtpDloopStage1PrecheckInput {
        enabled: input.enabled,
        topology: input.topology,
        live_seqs: input.live_seqs,
        stage1_drafts: input.stage1_drafts,
        max_verify_k: input.max_verify_k,
    };
    let extra_drafts = match precheck_stage1_admission(&precheck) {
        Ok(extra) => extra,
        Err(refusal) => return MtpDloopDecision::StayStage1(refusal),
    };
    if input.stage1_probs.len() != input.stage1_drafts
        || !input.threshold.is_finite()
        || input.threshold > 0.0
    {
        return MtpDloopDecision::StayStage1(MtpDloopRefusal::InvalidProbabilities);
    }
    let Some(c1) = stage1_log_prob_sum(input.stage1_probs) else {
        return MtpDloopDecision::StayStage1(MtpDloopRefusal::InvalidProbabilities);
    };
    if c1 >= input.threshold {
        MtpDloopDecision::ExtendStage2 {
            extra_drafts,
            total_drafts: input.stage1_drafts + extra_drafts,
        }
    } else {
        MtpDloopDecision::StayStage1(MtpDloopRefusal::ConfidenceBelowThreshold)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flash_next_moe_c1_is_strictly_refused_fail_closed() {
        // Qwen3.8-Flash-Next: 512 experts, C=1, ultra-confident Stage-1 probs.
        // Must NEVER extend K=5 -> K=9 because MoE unique-expert expansion regresses GB10.
        let decision = evaluate_stage2_extension(&MtpDloopInput {
            enabled: true,
            topology: TargetFfnTopology::from_num_experts(512),
            live_seqs: 1,
            stage1_drafts: 4,
            max_verify_k: 16,
            stage1_probs: &[0.99, 0.99, 0.99, 0.99],
            threshold: MTP_DLOOP_DEFAULT_LOGPROB_THRESH,
        });
        assert_eq!(
            decision,
            MtpDloopDecision::StayStage1(MtpDloopRefusal::MoeTargetRefused { num_experts: 512 })
        );
        assert_eq!(decision.extra_drafts(), None);
        assert_eq!(
            required_verify_k_for_model(
                true,
                true,
                false,
                TargetFfnTopology::from_num_experts(512),
            ),
            0
        );
    }

    #[test]
    fn any_moe_expert_count_above_one_is_refused() {
        for num_experts in [2, 8, 64, 128, 512] {
            let decision = evaluate_stage2_extension(&MtpDloopInput {
                enabled: true,
                topology: TargetFfnTopology::from_num_experts(num_experts),
                live_seqs: 1,
                stage1_drafts: 4,
                max_verify_k: 9,
                stage1_probs: &[0.98, 0.98, 0.98, 0.98],
                threshold: -0.5,
            });
            assert_eq!(
                decision,
                MtpDloopDecision::StayStage1(MtpDloopRefusal::MoeTargetRefused { num_experts })
            );
        }
    }

    #[test]
    fn dense_27b_c1_and_c2_extend_k5_to_k9_when_confident() {
        // 4 * ln(0.90) = -0.42144 >= -0.5 -> ExtendStage2 (+4 drafts, total 8, K=9).
        for num_experts in [0, 1] {
            for live_seqs in [1, 2] {
                let decision = evaluate_stage2_extension(&MtpDloopInput {
                    enabled: true,
                    topology: TargetFfnTopology::from_num_experts(num_experts),
                    live_seqs,
                    stage1_drafts: 4,
                    max_verify_k: 9,
                    stage1_probs: &[0.90, 0.90, 0.90, 0.90],
                    threshold: MTP_DLOOP_DEFAULT_LOGPROB_THRESH,
                });
                assert_eq!(
                    decision,
                    MtpDloopDecision::ExtendStage2 {
                        extra_drafts: 4,
                        total_drafts: 8,
                    }
                );
            }
            assert_eq!(
                required_verify_k_for_model(
                    true,
                    true,
                    false,
                    TargetFfnTopology::from_num_experts(num_experts),
                ),
                9
            );
        }
    }

    #[test]
    fn concurrency_above_two_or_zero_refuses_stage2() {
        for live_seqs in [0, 3, 4, 16] {
            let decision = evaluate_stage2_extension(&MtpDloopInput {
                enabled: true,
                topology: TargetFfnTopology::Dense,
                live_seqs,
                stage1_drafts: 4,
                max_verify_k: 9,
                stage1_probs: &[0.99, 0.99, 0.99, 0.99],
                threshold: -0.5,
            });
            assert_eq!(
                decision,
                MtpDloopDecision::StayStage1(MtpDloopRefusal::ConcurrencyTooHigh {
                    live_seqs,
                    max_concurrency: 2,
                })
            );
        }
    }

    #[test]
    fn confidence_threshold_and_invalid_probs_fail_closed() {
        // 4 * ln(0.85) = -0.65008 < -0.5 -> refused.
        let low = evaluate_stage2_extension(&MtpDloopInput {
            enabled: true,
            topology: TargetFfnTopology::Dense,
            live_seqs: 1,
            stage1_drafts: 4,
            max_verify_k: 9,
            stage1_probs: &[0.85, 0.85, 0.85, 0.85],
            threshold: -0.5,
        });
        assert_eq!(
            low,
            MtpDloopDecision::StayStage1(MtpDloopRefusal::ConfidenceBelowThreshold)
        );

        // Zero probability yields -inf -> ConfidenceBelowThreshold.
        let zero = evaluate_stage2_extension(&MtpDloopInput {
            enabled: true,
            topology: TargetFfnTopology::Dense,
            live_seqs: 1,
            stage1_drafts: 4,
            max_verify_k: 9,
            stage1_probs: &[0.99, 0.0, 0.99, 0.99],
            threshold: -0.5,
        });
        assert_eq!(
            zero,
            MtpDloopDecision::StayStage1(MtpDloopRefusal::ConfidenceBelowThreshold)
        );

        // NaN, negative, >1.0, or length mismatch -> InvalidProbabilities.
        for bad in [
            &[f32::NAN, 0.9, 0.9, 0.9][..],
            &[-0.1, 0.9, 0.9, 0.9][..],
            &[1.01, 0.9, 0.9, 0.9][..],
            &[0.9, 0.9, 0.9][..],
        ] {
            let decision = evaluate_stage2_extension(&MtpDloopInput {
                enabled: true,
                topology: TargetFfnTopology::Dense,
                live_seqs: 1,
                stage1_drafts: 4,
                max_verify_k: 9,
                stage1_probs: bad,
                threshold: -0.5,
            });
            assert_eq!(
                decision,
                MtpDloopDecision::StayStage1(MtpDloopRefusal::InvalidProbabilities)
            );
        }
    }

    #[test]
    fn disabled_and_capacity_limits_fail_closed() {
        let disabled = evaluate_stage2_extension(&MtpDloopInput {
            enabled: false,
            topology: TargetFfnTopology::Dense,
            live_seqs: 1,
            stage1_drafts: 4,
            max_verify_k: 9,
            stage1_probs: &[0.99, 0.99, 0.99, 0.99],
            threshold: -0.5,
        });
        assert_eq!(
            disabled,
            MtpDloopDecision::StayStage1(MtpDloopRefusal::Disabled)
        );

        let low_cap = evaluate_stage2_extension(&MtpDloopInput {
            enabled: true,
            topology: TargetFfnTopology::Dense,
            live_seqs: 1,
            stage1_drafts: 4,
            max_verify_k: 5,
            stage1_probs: &[0.99, 0.99, 0.99, 0.99],
            threshold: -0.5,
        });
        assert_eq!(
            low_cap,
            MtpDloopDecision::StayStage1(MtpDloopRefusal::VerifyCapacityExceeded {
                required_k: 9,
                max_verify_k: 5,
            })
        );
    }

    #[test]
    fn env_parsers_and_cohort_guard_behave_fail_closed() {
        assert!(!parse_mtp_dloop_enabled(None));
        assert!(!parse_mtp_dloop_enabled(Some("0")));
        assert!(!parse_mtp_dloop_enabled(Some("true")));
        assert!(parse_mtp_dloop_enabled(Some("1")));
        assert!(parse_mtp_dloop_enabled(Some(" 1 ")));

        assert_eq!(parse_mtp_dloop_thresh(None), -0.5);
        assert_eq!(parse_mtp_dloop_thresh(Some("-0.35")), -0.35);
        assert_eq!(parse_mtp_dloop_thresh(Some("0.0")), 0.0);
        // Positive, NaN, or garbage falls back to default -0.5.
        assert_eq!(parse_mtp_dloop_thresh(Some("0.2")), -0.5);
        assert_eq!(parse_mtp_dloop_thresh(Some("NaN")), -0.5);
        assert_eq!(parse_mtp_dloop_thresh(Some("bogus")), -0.5);

        assert_eq!(active_cohort_context(), None);
        {
            let _outer = MtpDloopCohortGuard::enter(2, 9);
            assert_eq!(
                active_cohort_context(),
                Some(MtpDloopCohortContext {
                    live_seqs: 2,
                    max_verify_k: 9,
                })
            );
            {
                let _inner = MtpDloopCohortGuard::enter(4, 5);
                assert_eq!(
                    active_cohort_context(),
                    Some(MtpDloopCohortContext {
                        live_seqs: 4,
                        max_verify_k: 5,
                    })
                );
            }
            assert_eq!(
                active_cohort_context(),
                Some(MtpDloopCohortContext {
                    live_seqs: 2,
                    max_verify_k: 9,
                })
            );
        }
        assert_eq!(active_cohort_context(), None);
    }
}
