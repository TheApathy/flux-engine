// SPDX-License-Identifier: AGPL-3.0-only

//! Exact host reference for DFlash2's sequential candidate-selector walk.

use anyhow::{Result, ensure};

use super::types::{
    DFLASH2_MAX_Q_MODIFIERS, DFLASH2_SELECTOR_TOP_K, DflashDraftSampleMethod,
    DflashDraftSamplingRequest, DflashSampledProposal, DflashSampledProposalFrame,
    DflashSparseQRow, select_sparse_q_f32, validate_uniform,
};

/// Decode the bounded CUDA selector outputs into the same typed proposal used
/// by the host oracle. Every byte count and row invariant is checked before q
/// can cross the proposer boundary.
pub fn decode_dflash2_device_proposal(
    request: DflashDraftSamplingRequest,
    anchor_id: u32,
    tokens: Vec<u32>,
    candidate_bytes: &[u8],
    probability_bytes: &[u8],
    vocab_size: usize,
) -> Result<DflashSampledProposalFrame> {
    let gamma = tokens.len();
    request.validate(gamma)?;
    ensure!(
        request.request_sampler.temperature > 0.0,
        "device sampled proposal requires positive temperature"
    );
    let row_elements = gamma
        .checked_mul(DFLASH2_SELECTOR_TOP_K)
        .ok_or_else(|| anyhow::anyhow!("DFlash selector row count overflow"))?;
    let expected_bytes = row_elements
        .checked_mul(size_of::<u32>())
        .ok_or_else(|| anyhow::anyhow!("DFlash selector byte count overflow"))?;
    ensure!(
        candidate_bytes.len() == expected_bytes,
        "DFlash selector candidate byte count drift"
    );
    ensure!(
        probability_bytes.len() == expected_bytes,
        "DFlash selector probability byte count drift"
    );

    let candidates: Vec<u32> = candidate_bytes
        .chunks_exact(4)
        .map(|bytes| u32::from_le_bytes(bytes.try_into().expect("four-byte chunk")))
        .collect();
    let probabilities: Vec<f32> = probability_bytes
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("four-byte chunk")))
        .collect();
    let mut predecessor = anchor_id;
    let mut q_rows = Vec::with_capacity(gamma);
    for position in 0..gamma {
        let base = position * DFLASH2_SELECTOR_TOP_K;
        let row_candidates: [u32; DFLASH2_SELECTOR_TOP_K] = candidates
            [base..base + DFLASH2_SELECTOR_TOP_K]
            .try_into()
            .expect("checked selector row width");
        let row_probabilities: [f32; DFLASH2_SELECTOR_TOP_K] = probabilities
            [base..base + DFLASH2_SELECTOR_TOP_K]
            .try_into()
            .expect("checked selector row width");
        for (candidate, probability) in row_candidates.iter().zip(row_probabilities.iter()) {
            ensure!(
                !request.target_forbidden_token_ids.contains(candidate) || *probability == 0.0,
                "DFlash target-forbidden candidate retained positive sparse-q mass"
            );
        }
        let selected_token = tokens[position];
        let selected_index = row_candidates
            .iter()
            .position(|&candidate| candidate == selected_token)
            .ok_or_else(|| anyhow::anyhow!("selected token is absent from sparse-q candidates"))?;
        let expected_selected =
            select_sparse_q_f32(&row_probabilities, request.draft_uniforms[position]);
        ensure!(
            selected_index == expected_selected,
            "selected token disagrees with the recorded uniform and sparse-q CDF"
        );
        let probability_sum = row_probabilities.iter().sum();
        q_rows.push(DflashSparseQRow {
            position,
            predecessor,
            candidates: row_candidates,
            probabilities: row_probabilities,
            selected_index,
            selected_token,
            probability_sum,
        });
        predecessor = selected_token;
    }
    let frame = DflashSampledProposalFrame {
        request,
        proposal: DflashSampledProposal {
            tokens,
            q_rows: Some(q_rows),
        },
    };
    frame.validate(vocab_size)?;
    Ok(frame)
}

#[allow(clippy::too_many_arguments)]
pub fn select_dflash2_path(
    method: DflashDraftSampleMethod,
    temperature: f32,
    anchor_id: u32,
    unary_rows: &[[f32; DFLASH2_SELECTOR_TOP_K]],
    candidate_rows: &[[u32; DFLASH2_SELECTOR_TOP_K]],
    hidden_rows: &[Vec<f32>],
    predecessor_codebook: &[Vec<f32>],
    successor_codebook: &[Vec<f32>],
    draft_uniforms: &[f64],
) -> Result<DflashSampledProposal> {
    select_dflash2_path_with_q_policy(
        method,
        temperature,
        0,
        0.0,
        1.0,
        anchor_id,
        unary_rows,
        candidate_rows,
        hidden_rows,
        predecessor_codebook,
        successor_codebook,
        &[],
        &[],
        draft_uniforms,
    )
}

/// Host oracle for the bounded sampled-selector policy. Candidate membership
/// remains the drafter's raw top-16. Finite request biases and hard masks alter
/// only those candidate scores, then the request's scalar filters run over the
/// final pair-adjusted scores in ordinary sampler order:
/// top-k -> softmax -> min-p -> top-p.
#[allow(clippy::too_many_arguments)]
pub fn select_dflash2_path_with_q_policy(
    method: DflashDraftSampleMethod,
    temperature: f32,
    top_k: u32,
    min_p: f32,
    top_p: f32,
    anchor_id: u32,
    unary_rows: &[[f32; DFLASH2_SELECTOR_TOP_K]],
    candidate_rows: &[[u32; DFLASH2_SELECTOR_TOP_K]],
    hidden_rows: &[Vec<f32>],
    predecessor_codebook: &[Vec<f32>],
    successor_codebook: &[Vec<f32>],
    logit_bias: &[(u32, f32)],
    forbidden_token_ids: &[u32],
    draft_uniforms: &[f64],
) -> Result<DflashSampledProposal> {
    ensure!(
        temperature.is_finite() && temperature >= 0.0,
        "DFlash draft temperature must be finite and non-negative"
    );
    ensure!(
        min_p.is_finite() && (0.0..=1.0).contains(&min_p),
        "DFlash draft min_p must be finite and in [0, 1]"
    );
    ensure!(
        top_p.is_finite() && top_p > 0.0 && top_p <= 1.0,
        "DFlash draft top_p must be finite and in (0, 1]"
    );
    let gamma = unary_rows.len();
    ensure!(gamma > 0, "DFlash selector needs at least one draft row");
    ensure!(
        candidate_rows.len() == gamma && hidden_rows.len() == gamma,
        "DFlash selector row counts disagree"
    );
    let sampled = method == DflashDraftSampleMethod::Probabilistic && temperature > 0.0;
    ensure!(
        sampled || (logit_bias.is_empty() && forbidden_token_ids.is_empty()),
        "q policy modifiers require probabilistic DFlash"
    );
    let modifier_count = logit_bias
        .len()
        .checked_add(forbidden_token_ids.len())
        .ok_or_else(|| anyhow::anyhow!("DFlash q modifier count overflow"))?;
    ensure!(
        modifier_count <= DFLASH2_MAX_Q_MODIFIERS,
        "DFlash q modifier count exceeds bounded selector ABI"
    );
    ensure!(
        logit_bias.iter().all(|(_, bias)| bias.is_finite()),
        "DFlash q logit bias contains a non-finite value"
    );
    ensure!(
        forbidden_token_ids.windows(2).all(|pair| pair[0] < pair[1]),
        "DFlash q hard-mask token ids must be sorted and unique"
    );
    if sampled {
        ensure!(
            draft_uniforms.len() == gamma,
            "probabilistic DFlash needs one draft uniform per row"
        );
    } else {
        ensure!(
            draft_uniforms.is_empty(),
            "greedy DFlash must not consume draft RNG draws"
        );
    }

    ensure!(
        !predecessor_codebook.is_empty() && predecessor_codebook.len() == successor_codebook.len(),
        "DFlash selector codebook vocabularies disagree"
    );
    let vocab_size = predecessor_codebook.len();
    ensure!(
        (anchor_id as usize) < vocab_size,
        "DFlash selector anchor is outside the codebook vocabulary"
    );
    let rank = hidden_rows[0].len();
    ensure!(rank > 0, "DFlash selector rank must be positive");

    let mut predecessor = anchor_id;
    let mut tokens = Vec::with_capacity(gamma);
    let mut q_rows = sampled.then(|| Vec::with_capacity(gamma));
    for position in 0..gamma {
        ensure!(
            hidden_rows[position].len() == rank,
            "DFlash selector hidden rank drift at row {position}"
        );
        validate_finite(&hidden_rows[position], "selector hidden")?;
        validate_codebook_row(predecessor_codebook, predecessor, rank, "predecessor")?;

        let candidates = candidate_rows[position];
        let mut scores = [0.0f32; DFLASH2_SELECTOR_TOP_K];
        for k in 0..DFLASH2_SELECTOR_TOP_K {
            let candidate = candidates[k];
            ensure!(
                !candidates[..k].contains(&candidate),
                "DFlash selector candidates must be unique"
            );
            validate_codebook_row(successor_codebook, candidate, rank, "successor")?;
            ensure!(
                unary_rows[position][k].is_finite(),
                "DFlash selector unary score is not finite"
            );
            let mut pairwise = 0.0f32;
            for r in 0..rank {
                pairwise += predecessor_codebook[predecessor as usize][r]
                    * hidden_rows[position][r]
                    * successor_codebook[candidate as usize][r];
            }
            let mut score = unary_rows[position][k] + pairwise;
            for &(token, bias) in logit_bias {
                if token == candidate {
                    score += bias;
                }
            }
            if forbidden_token_ids.binary_search(&candidate).is_ok() {
                score = f32::NEG_INFINITY;
            }
            ensure!(
                score.is_finite() || score == f32::NEG_INFINITY,
                "DFlash selector score is invalid after q policy"
            );
            scores[k] = score;
        }

        let selected_index = if sampled {
            validate_uniform(draft_uniforms[position], "draft uniform")?;
            let probabilities =
                filtered_softmax_top16(scores, &candidates, temperature, top_k, min_p, top_p)?;
            let selected = sample_top16(&probabilities, draft_uniforms[position]);
            let probability_sum = probabilities.iter().sum();
            q_rows
                .as_mut()
                .expect("sampled q rows allocated")
                .push(DflashSparseQRow {
                    position,
                    predecessor,
                    candidates,
                    probabilities,
                    selected_index: selected,
                    selected_token: candidates[selected],
                    probability_sum,
                });
            selected
        } else {
            first_wins_argmax(&scores)
        };
        predecessor = candidates[selected_index];
        tokens.push(predecessor);
    }

    let proposal = DflashSampledProposal { tokens, q_rows };
    proposal.validate(
        if sampled {
            DflashDraftSampleMethod::Probabilistic
        } else {
            DflashDraftSampleMethod::Greedy
        },
        vocab_size,
    )?;
    Ok(proposal)
}

fn validate_finite(values: &[f32], label: &str) -> Result<()> {
    ensure!(
        values.iter().all(|value| value.is_finite()),
        "{label} contains a non-finite value"
    );
    Ok(())
}

fn validate_codebook_row(
    codebook: &[Vec<f32>],
    token: u32,
    rank: usize,
    label: &str,
) -> Result<()> {
    let row = codebook
        .get(token as usize)
        .ok_or_else(|| anyhow::anyhow!("DFlash selector {label} is outside the codebook"))?;
    ensure!(
        row.len() == rank,
        "DFlash selector {label} codebook rank drift"
    );
    validate_finite(row, label)
}

fn first_wins_argmax(scores: &[f32; DFLASH2_SELECTOR_TOP_K]) -> usize {
    let mut selected = 0usize;
    for i in 1..DFLASH2_SELECTOR_TOP_K {
        if scores[i] > scores[selected] {
            selected = i;
        }
    }
    selected
}

fn filtered_softmax_top16(
    scores: [f32; DFLASH2_SELECTOR_TOP_K],
    candidates: &[u32; DFLASH2_SELECTOR_TOP_K],
    temperature: f32,
    top_k: u32,
    min_p: f32,
    top_p: f32,
) -> Result<[f32; DFLASH2_SELECTOR_TOP_K]> {
    ensure!(
        temperature > 0.0,
        "sampled DFlash temperature must be positive"
    );
    let mut order = (0..DFLASH2_SELECTOR_TOP_K)
        .filter(|&index| scores[index].is_finite())
        .collect::<Vec<_>>();
    order.sort_unstable_by(|&left, &right| {
        if scores[left] > scores[right] {
            std::cmp::Ordering::Less
        } else if scores[left] < scores[right] {
            std::cmp::Ordering::Greater
        } else {
            candidates[left].cmp(&candidates[right])
        }
    });
    ensure!(!order.is_empty(), "DFlash q softmax is degenerate");
    if top_k > 0 {
        order.truncate(order.len().min(top_k as usize));
    }

    let max = scores[order[0]];
    let mut weights = [0.0f32; DFLASH2_SELECTOR_TOP_K];
    for &index in &order {
        weights[index] = ((scores[index] - max) / temperature).exp();
    }

    if min_p > 0.0 {
        let threshold = min_p * weights[order[0]];
        order.retain(|&index| weights[index] >= threshold);
    }
    ensure!(!order.is_empty(), "DFlash q min-p filter is degenerate");

    if top_p < 1.0 {
        let sum: f32 = order.iter().map(|&index| weights[index]).sum();
        ensure!(
            sum.is_finite() && sum > 0.0,
            "DFlash q top-p input is degenerate"
        );
        let mut cumulative = 0.0f32;
        let mut keep = order.len();
        for (rank, &index) in order.iter().enumerate() {
            cumulative += weights[index] / sum;
            if cumulative >= top_p {
                keep = rank + 1;
                break;
            }
        }
        for &index in &order[keep..] {
            weights[index] = 0.0;
        }
        order.truncate(keep);
    }

    let sum: f32 = order.iter().map(|&index| weights[index]).sum();
    ensure!(
        sum.is_finite() && sum > 0.0,
        "DFlash q softmax is degenerate"
    );
    let mut probabilities = [0.0f32; DFLASH2_SELECTOR_TOP_K];
    for &index in &order {
        probabilities[index] = weights[index] / sum;
    }
    Ok(probabilities)
}

fn sample_top16(probabilities: &[f32; DFLASH2_SELECTOR_TOP_K], uniform: f64) -> usize {
    let uniform = uniform as f32;
    let mut cumulative = 0.0f32;
    let mut last_nonzero = 0usize;
    for (index, &probability) in probabilities.iter().enumerate() {
        if probability > 0.0 {
            last_nonzero = index;
        }
        cumulative += probability;
        if cumulative > uniform {
            return index;
        }
    }
    last_nonzero
}
