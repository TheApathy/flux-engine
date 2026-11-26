// SPDX-License-Identifier: AGPL-3.0-only

//! Pure host-side speculative-sampling verifier (`FLUX_SPEC_VERIFY=stochastic`).
//!
//! Standard rejection sampling (Leviathan et al. 2023; Chen et al. 2023):
//! accept draft `x` with probability `min(1, p(x)/q(x))`; on rejection emit a
//! sample of `norm(max(0, p - q))`; when every draft is accepted emit a bonus
//! sample of the next target row. `p` is the target's *post-policy*
//! distribution (penalties, bias, temperature, top-k, min-p, top-p) built by
//! the same `prepare_sampling_distribution_with_history` the plain sampler
//! uses, so the emitted stream is distributed exactly like plain sampling.
//!
//! Nothing here touches `ActiveSeq`, the device or global state: every input
//! is a value, so the rule is unit-testable in isolation. Design and proof
//! sketch: `docs/design/stochastic-verify.md`.

// Dense q, the frame-level entry and reject-branch reporting are exercised by
// the unit tests now and by the walk once the MTP chain exposes q rows
// (GPU TODO 1 in the design doc); until then they are unused in non-test builds.
#![cfg_attr(not(test), allow(dead_code))]

use anyhow::{Result, ensure};
use spark_runtime::sampler::{
    PreparedSamplingDistribution, SamplingParams, prepare_sampling_distribution_with_history,
    sampling_uniforms_from_seed,
};

/// The drafter's proposal distribution `q` at one position.
#[derive(Clone, Copy, Debug)]
pub(super) enum DraftDistribution<'a> {
    /// A deterministic draft (today's greedy MTP argmax, or an n-gram/history
    /// draft): `q` is a point mass on the drafted token.
    PointMass,
    /// A dense normalized `q` row over the vocabulary; the draft token must
    /// have been sampled from it.
    Dense(&'a [f32]),
}

/// One drafted position: the token and the distribution it was drawn from.
#[derive(Clone, Copy, Debug)]
pub(super) struct PositionDraft<'a> {
    pub token: u32,
    pub q: DraftDistribution<'a>,
}

/// The two uniforms one verify position may consume. `accept` is draw 0 of
/// the position's seeded stream, which is also the draw plain decode uses
/// for its sample at that output index; `residual` is draw 1.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct PositionUniforms {
    pub accept: f32,
    pub residual: f32,
}

/// Why a rejected position emitted its token.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum RejectBranch {
    /// Sampled from `norm(max(0, p - q))`.
    Residual,
    /// The residual had no mass (only reachable through rounding when
    /// `q >= p` everywhere); sampled from `p` itself.
    TargetFallback,
}

/// The outcome of verifying one drafted position.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum PositionVerdict {
    Accept,
    Reject { token: u32, branch: RejectBranch },
}

/// The outcome of a whole frame: `accepted` drafts, then `next_token`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct FrameOutcome {
    pub accepted: usize,
    pub next_token: u32,
    /// `None` when every draft was accepted and `next_token` is the bonus.
    pub reject_branch: Option<RejectBranch>,
}

/// Uniforms for the output index whose plain-decode seed is `seed`.
///
/// `seed` is already advanced by the caller (`ActiveSeq::sampling_params`
/// adds `output_tokens.len()`), matching plain decode. `None` (unseeded
/// request) draws a fresh process-random seed.
pub(super) fn position_uniforms(seed: Option<u64>) -> PositionUniforms {
    let draws = sampling_uniforms_from_seed(seed.unwrap_or_else(fresh_seed), 2);
    PositionUniforms {
        accept: draws.first().copied().unwrap_or(0.0),
        residual: draws.get(1).copied().unwrap_or(0.0),
    }
}

fn fresh_seed() -> u64 {
    use std::hash::{BuildHasher as _, Hasher as _};
    let mut hasher = std::collections::hash_map::RandomState::new().build_hasher();
    hasher.write_u64(0x5bd1_e995);
    hasher.finish()
}

/// The dense normalized target `p` for one logits row under `params`.
///
/// `logits_f32` is little-endian FP32 bytes after the caller's hard masks.
/// Fails closed when the sampler produced a degenerate distribution.
pub(super) fn target_probabilities(
    logits_f32: &[u8],
    params: &SamplingParams,
    history: &[u32],
) -> Result<Vec<f32>> {
    let prepared: PreparedSamplingDistribution =
        prepare_sampling_distribution_with_history(logits_f32, params, history);
    prepared
        .normalized_dense_probabilities()
        .ok_or_else(|| anyhow::anyhow!("stochastic verify: degenerate target distribution"))
}

/// Verify one drafted position against the target row `p`.
pub(super) fn verify_position(
    p: &[f32],
    draft: &PositionDraft<'_>,
    uniforms: PositionUniforms,
) -> Result<PositionVerdict> {
    validate_uniform(uniforms.accept)?;
    let x = draft.token as usize;
    ensure!(
        x < p.len(),
        "stochastic verify: draft {x} outside vocab {}",
        p.len()
    );
    let q_x = draft_mass(draft, p.len())?;
    let p_x = f64::from(p[x]);
    // u < p/q  <=>  u*q < p, strict so p(x) = 0 is never accepted.
    if f64::from(uniforms.accept) * q_x < p_x {
        return Ok(PositionVerdict::Accept);
    }
    let residual = residual_weights(p, draft);
    let (token, branch) = if residual.iter().any(|&w| w > 0.0) {
        (
            sample_weights(&residual, uniforms.residual)?,
            RejectBranch::Residual,
        )
    } else {
        (
            sample_weights(p, uniforms.residual)?,
            RejectBranch::TargetFallback,
        )
    };
    Ok(PositionVerdict::Reject { token, branch })
}

/// Sample the bonus token from `p` with the position's draw 0 (the same
/// draw plain decode consumes at this output index).
pub(super) fn sample_bonus(p: &[f32], uniforms: PositionUniforms) -> Result<u32> {
    sample_weights(p, uniforms.accept)
}

/// Verify a frame: `targets` holds `drafts.len() + 1` rows.
pub(super) fn verify_frame(
    targets: &[Vec<f32>],
    drafts: &[PositionDraft<'_>],
    uniforms: impl Fn(usize) -> PositionUniforms,
) -> Result<FrameOutcome> {
    ensure!(
        targets.len() == drafts.len() + 1,
        "stochastic verify: need drafts+1 target rows ({} vs {})",
        targets.len(),
        drafts.len()
    );
    for (i, (p, draft)) in targets.iter().zip(drafts).enumerate() {
        if let PositionVerdict::Reject { token, branch } = verify_position(p, draft, uniforms(i))? {
            return Ok(FrameOutcome {
                accepted: i,
                next_token: token,
                reject_branch: Some(branch),
            });
        }
    }
    let last = drafts.len();
    Ok(FrameOutcome {
        accepted: last,
        next_token: sample_bonus(&targets[last], uniforms(last))?,
        reject_branch: None,
    })
}

fn draft_mass(draft: &PositionDraft<'_>, vocab: usize) -> Result<f64> {
    match draft.q {
        DraftDistribution::PointMass => Ok(1.0),
        DraftDistribution::Dense(q) => {
            ensure!(
                q.len() == vocab,
                "stochastic verify: q width {} != vocab {vocab}",
                q.len()
            );
            let q_x = f64::from(q[draft.token as usize]);
            // A token drawn from q has q(x) > 0; anything else means the
            // proposal and its distribution disagree, so fail closed.
            ensure!(
                q_x > 0.0 && q_x.is_finite(),
                "stochastic verify: draft has q(x) = {q_x}"
            );
            Ok(q_x)
        }
    }
}

fn residual_weights(p: &[f32], draft: &PositionDraft<'_>) -> Vec<f32> {
    match draft.q {
        DraftDistribution::PointMass => p
            .iter()
            .enumerate()
            .map(|(i, &w)| if i == draft.token as usize { 0.0 } else { w })
            .collect(),
        DraftDistribution::Dense(q) => p
            .iter()
            .zip(q)
            .map(|(&pi, &qi)| (pi - qi).max(0.0))
            .collect(),
    }
}

fn validate_uniform(u: f32) -> Result<()> {
    ensure!(
        (0.0..1.0).contains(&u),
        "stochastic verify: uniform {u} outside [0, 1)"
    );
    Ok(())
}

/// Inverse-CDF sample over unnormalized non-negative `weights`, vocab order.
/// A zero-weight id can never be returned: the CDF only crosses the target
/// on a positive step, and the rounding fallthrough returns the last
/// positive id.
pub(super) fn sample_weights(weights: &[f32], uniform: f32) -> Result<u32> {
    validate_uniform(uniform)?;
    ensure!(
        weights.iter().all(|w| w.is_finite() && *w >= 0.0),
        "stochastic verify: weights contain a non-finite or negative value"
    );
    let total: f64 = weights.iter().map(|&w| f64::from(w)).sum();
    ensure!(total > 0.0, "stochastic verify: weights sum to zero");
    let target = f64::from(uniform) * total;
    let mut cumulative = 0.0f64;
    let mut last_positive = None;
    for (i, &w) in weights.iter().enumerate() {
        if w > 0.0 {
            last_positive = Some(i as u32);
            cumulative += f64::from(w);
            if cumulative > target {
                return Ok(i as u32);
            }
        }
    }
    last_positive.ok_or_else(|| anyhow::anyhow!("stochastic verify: no positive weight"))
}
