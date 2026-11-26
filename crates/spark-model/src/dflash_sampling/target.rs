// SPDX-License-Identifier: AGPL-3.0-only

//! Exact target-`p` preparation for sampled DFlash verification.

use std::borrow::Cow;

use anyhow::{Context, Result, ensure};
use ring::digest::{Context as DigestContext, SHA256};
use serde::{Deserialize, Serialize};
use spark_runtime::sampler::{SamplingParams, prepare_sampling_distribution_with_history};

use super::types::validate_probability_row;
use super::{DflashSampledAcceptance, DflashSampledProposalFrame, accept_sparse_dflash};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DflashTargetLogitsFormat {
    Bf16,
    Fp32,
}

impl DflashTargetLogitsFormat {
    fn element_bytes(self) -> usize {
        match self {
            Self::Bf16 => 2,
            Self::Fp32 => 4,
        }
    }
}

/// One exact non-zero entry in a normalized target-`p` row. Entries are
/// serialized in strictly increasing token order; omitted vocabulary entries
/// are exactly zero.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DflashTargetProbabilityEntry {
    pub token: u32,
    pub probability: f32,
}

/// Self-contained normalized target-`p` evidence.  This is sparse only for
/// receipt size: validation expands the implicit zeros and replays the exact
/// acceptance rule.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DflashTargetProbabilityRow {
    pub entries: Vec<DflashTargetProbabilityEntry>,
}

/// Cryptographic identity plus self-contained replay evidence for the exact
/// target rows consumed by rejection sampling. The raw-logit digest is a
/// provenance hook for an external capture; the normalized target `p` rows
/// are carried here and independently validated/replayed.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DflashTargetEvidence {
    pub logits_format: DflashTargetLogitsFormat,
    pub row_count: usize,
    pub vocab_size: usize,
    pub logits_bytes: usize,
    pub target_probability_values: usize,
    pub target_probability_nonzero_values: usize,
    pub logits_sha256: [u8; 32],
    pub target_probabilities_sha256: [u8; 32],
    pub target_probability_rows: Vec<DflashTargetProbabilityRow>,
}

impl DflashTargetEvidence {
    pub fn from_evaluation(
        verify_logits: &[u8],
        format: DflashTargetLogitsFormat,
        vocab_size: usize,
        target_probabilities: &[Vec<f32>],
    ) -> Result<Self> {
        ensure!(vocab_size > 0, "DFlash target vocabulary must be positive");
        ensure!(
            u32::try_from(vocab_size - 1).is_ok(),
            "DFlash target vocabulary exceeds token-id width"
        );
        ensure!(
            !target_probabilities.is_empty(),
            "DFlash target evidence needs at least one row"
        );
        let row_count = target_probabilities.len();
        let expected_logits_bytes = row_count
            .checked_mul(vocab_size)
            .and_then(|values| values.checked_mul(format.element_bytes()))
            .ok_or_else(|| anyhow::anyhow!("DFlash target evidence byte count overflow"))?;
        ensure!(
            verify_logits.len() == expected_logits_bytes,
            "DFlash target evidence logit byte count drift"
        );
        for row in target_probabilities {
            ensure!(
                row.len() == vocab_size,
                "DFlash target evidence probability row width drift"
            );
            validate_probability_row(row)?;
        }
        let target_probability_values = row_count
            .checked_mul(vocab_size)
            .ok_or_else(|| anyhow::anyhow!("DFlash target evidence value count overflow"))?;

        let mut logits_digest = DigestContext::new(&SHA256);
        logits_digest.update(b"atlas-dflash-target-logits-v1\0");
        logits_digest.update(&[match format {
            DflashTargetLogitsFormat::Bf16 => 0,
            DflashTargetLogitsFormat::Fp32 => 1,
        }]);
        logits_digest.update(&(row_count as u64).to_le_bytes());
        logits_digest.update(&(vocab_size as u64).to_le_bytes());
        logits_digest.update(&(verify_logits.len() as u64).to_le_bytes());
        logits_digest.update(verify_logits);

        let target_probability_rows = target_probabilities
            .iter()
            .map(|row| DflashTargetProbabilityRow {
                entries: row
                    .iter()
                    .enumerate()
                    .filter(|(_, probability)| **probability > 0.0)
                    .map(|(token, &probability)| DflashTargetProbabilityEntry {
                        token: token as u32,
                        probability,
                    })
                    .collect(),
            })
            .collect::<Vec<_>>();
        let target_probability_nonzero_values = target_probability_rows
            .iter()
            .try_fold(0usize, |count, row| count.checked_add(row.entries.len()))
            .ok_or_else(|| anyhow::anyhow!("DFlash target evidence nonzero count overflow"))?;
        let target_probabilities_sha256 =
            target_probability_digest(row_count, vocab_size, &target_probability_rows);

        Ok(Self {
            logits_format: format,
            row_count,
            vocab_size,
            logits_bytes: verify_logits.len(),
            target_probability_values,
            target_probability_nonzero_values,
            logits_sha256: logits_digest
                .finish()
                .as_ref()
                .try_into()
                .expect("SHA-256 width"),
            target_probabilities_sha256,
            target_probability_rows,
        })
    }

    pub fn validate_shape(&self, expected_rows: usize, expected_vocab_size: usize) -> Result<()> {
        self.validated_target_probabilities(expected_rows, expected_vocab_size)
            .map(|_| ())
    }

    pub(crate) fn validated_target_probabilities(
        &self,
        expected_rows: usize,
        expected_vocab_size: usize,
    ) -> Result<Vec<Vec<f32>>> {
        ensure!(
            self.row_count == expected_rows && self.vocab_size == expected_vocab_size,
            "DFlash target evidence shape drift"
        );
        let expected_logits_bytes = expected_rows
            .checked_mul(expected_vocab_size)
            .and_then(|values| values.checked_mul(self.logits_format.element_bytes()))
            .ok_or_else(|| anyhow::anyhow!("DFlash target evidence byte count overflow"))?;
        ensure!(
            self.logits_bytes == expected_logits_bytes,
            "DFlash target evidence logit byte count drift"
        );
        ensure!(
            self.target_probability_values
                == expected_rows.checked_mul(expected_vocab_size).ok_or_else(
                    || anyhow::anyhow!("DFlash target evidence value count overflow")
                )?,
            "DFlash target evidence probability value count drift"
        );
        ensure!(
            self.target_probability_rows.len() == expected_rows,
            "DFlash target evidence sparse row count drift"
        );
        let mut dense_rows = Vec::with_capacity(expected_rows);
        let mut nonzero_values = 0usize;
        for sparse in &self.target_probability_rows {
            let mut dense = vec![0.0f32; expected_vocab_size];
            let mut previous = None;
            for entry in &sparse.entries {
                let token = entry.token as usize;
                ensure!(
                    token < expected_vocab_size,
                    "DFlash target evidence token is outside the vocabulary"
                );
                ensure!(
                    previous.is_none_or(|prior| prior < entry.token),
                    "DFlash target evidence entries must be sorted and unique"
                );
                ensure!(
                    entry.probability.is_finite() && entry.probability > 0.0,
                    "DFlash target evidence sparse probability must be finite and positive"
                );
                dense[token] = entry.probability;
                previous = Some(entry.token);
            }
            validate_probability_row(&dense)?;
            nonzero_values = nonzero_values
                .checked_add(sparse.entries.len())
                .ok_or_else(|| anyhow::anyhow!("DFlash target evidence nonzero count overflow"))?;
            dense_rows.push(dense);
        }
        ensure!(
            self.target_probability_nonzero_values == nonzero_values,
            "DFlash target evidence nonzero value count drift"
        );
        ensure!(
            self.logits_sha256 != [0; 32] && self.target_probabilities_sha256 != [0; 32],
            "DFlash target evidence digest is empty"
        );
        ensure!(
            self.target_probabilities_sha256
                == target_probability_digest(
                    expected_rows,
                    expected_vocab_size,
                    &self.target_probability_rows,
                ),
            "DFlash target probability digest drift"
        );
        Ok(dense_rows)
    }
}

fn target_probability_digest(
    row_count: usize,
    vocab_size: usize,
    rows: &[DflashTargetProbabilityRow],
) -> [u8; 32] {
    let mut digest = DigestContext::new(&SHA256);
    digest.update(b"atlas-dflash-target-probabilities-v2-sparse\0");
    digest.update(&(row_count as u64).to_le_bytes());
    digest.update(&(vocab_size as u64).to_le_bytes());
    digest.update(&(rows.len() as u64).to_le_bytes());
    for row in rows {
        digest.update(&(row.entries.len() as u64).to_le_bytes());
        for entry in &row.entries {
            digest.update(&entry.token.to_le_bytes());
            digest.update(&entry.probability.to_bits().to_le_bytes());
        }
    }
    digest.finish().as_ref().try_into().expect("SHA-256 width")
}

/// Prepare `p_0..p_gamma` from resident verify logits using ordinary
/// decoding's sampler pipeline as the single source of truth.
///
/// Row `i` is evaluated against a private counterfactual history
/// `base_history + drafts[..i]`. The caller's live history is never mutated.
pub fn prepare_dflash_target_probabilities(
    verify_logits: &[u8],
    format: DflashTargetLogitsFormat,
    vocab_size: usize,
    frame: &DflashSampledProposalFrame,
    params: &SamplingParams,
    base_history: &[u32],
) -> Result<Vec<Vec<f32>>> {
    ensure!(vocab_size > 0, "DFlash target vocabulary must be positive");
    let gamma = frame.proposal.tokens.len();
    frame.validate(vocab_size)?;
    frame.request.validate_sampler(params)?;
    frame.request.validate_history(base_history)?;
    if let Some(&last) = base_history.last() {
        ensure!(
            frame
                .proposal
                .q_rows
                .as_ref()
                .and_then(|rows| rows.first())
                .is_some_and(|row| row.predecessor == last),
            "DFlash sparse-q anchor disagrees with generated history"
        );
    }

    let row_bytes = vocab_size
        .checked_mul(format.element_bytes())
        .ok_or_else(|| anyhow::anyhow!("DFlash target row byte count overflow"))?;
    let expected_bytes = row_bytes
        .checked_mul(
            gamma
                .checked_add(1)
                .ok_or_else(|| anyhow::anyhow!("DFlash target row count overflow"))?,
        )
        .ok_or_else(|| anyhow::anyhow!("DFlash target buffer byte count overflow"))?;
    ensure!(
        verify_logits.len() == expected_bytes,
        "DFlash verify-logit byte count drift: got {}, expected {expected_bytes}",
        verify_logits.len()
    );

    let mut rolling_history = Vec::with_capacity(base_history.len() + gamma);
    rolling_history.extend_from_slice(base_history);
    let mut fp32_row = Vec::with_capacity(vocab_size * size_of::<f32>());
    let mut target = Vec::with_capacity(gamma + 1);

    for row_index in 0..=gamma {
        let start = row_index * row_bytes;
        let raw_row = &verify_logits[start..start + row_bytes];
        let mut sampler_row: Cow<'_, [u8]> = match format {
            DflashTargetLogitsFormat::Fp32 => Cow::Borrowed(raw_row),
            DflashTargetLogitsFormat::Bf16 => {
                fp32_row.clear();
                for bytes in raw_row.chunks_exact(2) {
                    let bits = u16::from_le_bytes([bytes[0], bytes[1]]) as u32;
                    fp32_row.extend_from_slice(&f32::from_bits(bits << 16).to_le_bytes());
                }
                Cow::Borrowed(fp32_row.as_slice())
            }
        };
        if !frame.request.target_forbidden_token_ids.is_empty() {
            let row = sampler_row.to_mut();
            for &token in &frame.request.target_forbidden_token_ids {
                let start = token as usize * size_of::<f32>();
                row[start..start + size_of::<f32>()]
                    .copy_from_slice(&f32::NEG_INFINITY.to_le_bytes());
            }
        }
        let probabilities = prepare_sampling_distribution_with_history(
            sampler_row.as_ref(),
            params,
            &rolling_history,
        )
        .normalized_dense_probabilities()
        .with_context(|| format!("DFlash target row {row_index} is degenerate"))?;
        ensure!(
            probabilities.len() == vocab_size,
            "DFlash target row vocabulary drift"
        );
        validate_probability_row(&probabilities)?;
        target.push(probabilities);
        if row_index < gamma {
            rolling_history.push(frame.proposal.tokens[row_index]);
        }
    }
    Ok(target)
}

/// Prepare the rolling target rows and execute exact sparse-`q` rejection
/// sampling with the frame's drain-once RNG draws.
pub fn evaluate_sampled_dflash(
    verify_logits: &[u8],
    format: DflashTargetLogitsFormat,
    vocab_size: usize,
    frame: &DflashSampledProposalFrame,
    params: &SamplingParams,
    base_history: &[u32],
) -> Result<DflashSampledAcceptance> {
    evaluate_sampled_dflash_with_evidence(
        verify_logits,
        format,
        vocab_size,
        frame,
        params,
        base_history,
    )
    .map(|(acceptance, _)| acceptance)
}

/// Evaluate exact target rejection sampling and return the evidence identity
/// consumed by the strict execution receipt.
pub fn evaluate_sampled_dflash_with_evidence(
    verify_logits: &[u8],
    format: DflashTargetLogitsFormat,
    vocab_size: usize,
    frame: &DflashSampledProposalFrame,
    params: &SamplingParams,
    base_history: &[u32],
) -> Result<(DflashSampledAcceptance, DflashTargetEvidence)> {
    let target = prepare_dflash_target_probabilities(
        verify_logits,
        format,
        vocab_size,
        frame,
        params,
        base_history,
    )?;
    let evidence =
        DflashTargetEvidence::from_evaluation(verify_logits, format, vocab_size, &target)?;
    let acceptance = accept_sparse_dflash(
        &target,
        &frame.proposal,
        &frame.request.accept_uniforms,
        frame.request.residual_uniform,
    )?;
    Ok((acceptance, evidence))
}
