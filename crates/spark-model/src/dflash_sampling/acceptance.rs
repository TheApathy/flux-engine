// SPDX-License-Identifier: AGPL-3.0-only

//! Sparse-q speculative rejection sampling for DFlash2.

use anyhow::{Result, ensure};

use super::types::{
    DflashDraftSampleMethod, DflashResidualBranch, DflashSampledAcceptance, DflashSampledProposal,
    validate_probability_row, validate_uniform,
};

pub fn accept_sparse_dflash(
    target_probabilities: &[Vec<f32>],
    proposal: &DflashSampledProposal,
    accept_uniforms: &[f32],
    residual_uniform: f32,
) -> Result<DflashSampledAcceptance> {
    let gamma = proposal.tokens.len();
    ensure!(
        target_probabilities.len() == gamma + 1,
        "sampled DFlash needs gamma+1 target probability rows"
    );
    ensure!(
        accept_uniforms.len() == gamma,
        "sampled DFlash needs one acceptance uniform per draft"
    );
    validate_uniform(f64::from(residual_uniform), "residual uniform")?;
    let vocab_size = target_probabilities[0].len();
    ensure!(vocab_size > 0, "target vocabulary must be positive");
    for row in target_probabilities {
        ensure!(
            row.len() == vocab_size,
            "target probability row width drift"
        );
        validate_probability_row(row)?;
    }
    proposal.validate(DflashDraftSampleMethod::Probabilistic, vocab_size)?;
    let q_rows = proposal.q_rows.as_ref().expect("validated sparse q rows");

    let mut acceptance_probabilities = Vec::with_capacity(gamma);
    for i in 0..gamma {
        validate_uniform(f64::from(accept_uniforms[i]), "acceptance uniform")?;
        let draft = proposal.tokens[i] as usize;
        let q_row = &q_rows[i];
        let q = q_row.probabilities[q_row.selected_index];
        let p = target_probabilities[i][draft];
        let acceptance_probability = (p / q).min(1.0);
        ensure!(
            acceptance_probability.is_finite(),
            "DFlash acceptance probability is not finite"
        );
        acceptance_probabilities.push(acceptance_probability);
        if f64::from(accept_uniforms[i]) * f64::from(q) < f64::from(p) {
            continue;
        }

        let mut residual = target_probabilities[i].clone();
        for (&candidate, &probability) in q_row.candidates.iter().zip(q_row.probabilities.iter()) {
            let value = &mut residual[candidate as usize];
            *value = (*value - probability).max(0.0);
        }
        let residual_mass: f64 = residual.iter().map(|&value| f64::from(value)).sum();
        let (next_token, branch) = if residual_mass > 0.0 {
            (
                sample_weights(&residual, residual_uniform)?,
                DflashResidualBranch::RejectionResidual,
            )
        } else {
            (
                sample_weights(&target_probabilities[i], residual_uniform)?,
                DflashResidualBranch::RejectionTargetFallback,
            )
        };
        return Ok(DflashSampledAcceptance {
            accepted: i,
            next_token,
            branch,
            residual_mass_before_normalize: Some(residual_mass),
            acceptance_probabilities,
        });
    }

    Ok(DflashSampledAcceptance {
        accepted: gamma,
        next_token: sample_weights(&target_probabilities[gamma], residual_uniform)?,
        branch: DflashResidualBranch::AllAcceptedBonus,
        residual_mass_before_normalize: None,
        acceptance_probabilities,
    })
}

fn sample_weights(weights: &[f32], uniform: f32) -> Result<u32> {
    validate_uniform(f64::from(uniform), "sampling uniform")?;
    ensure!(
        weights
            .iter()
            .all(|weight| weight.is_finite() && *weight >= 0.0),
        "sampling weights contain a non-finite or negative value"
    );
    let total: f64 = weights.iter().map(|&weight| weight as f64).sum();
    ensure!(
        total.is_finite() && total > 0.0,
        "sampling weights sum to zero"
    );
    let target = f64::from(uniform) * total;
    let mut cumulative = 0.0f64;
    let mut last_nonzero = 0u32;
    for (index, &weight) in weights.iter().enumerate() {
        if weight > 0.0 {
            last_nonzero = index as u32;
        }
        cumulative += weight as f64;
        if cumulative > target {
            return Ok(index as u32);
        }
    }
    Ok(last_nonzero)
}
