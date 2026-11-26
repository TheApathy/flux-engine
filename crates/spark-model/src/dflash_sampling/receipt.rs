// SPDX-License-Identifier: AGPL-3.0-only

//! Strict schema-7 execution receipt for sampled DFlash2 steps.

use anyhow::{Context, Result, ensure};
use ring::digest::{Context as DigestContext, SHA256};
use serde::{Deserialize, Serialize};

use super::target::DflashTargetEvidence;
use super::types::{
    DflashDraftSampleMethod, DflashDraftSamplingRequest, DflashResidualBranch,
    DflashSampledAcceptance, DflashSampledProposal, DflashSampledProposalFrame,
    DflashTargetPolicyPhase,
};

pub const DFLASH_SAMPLING_RECEIPT_SCHEMA: u32 = 7;
pub const DFLASH_SAMPLING_RECEIPT_PREFIX: &str = "DFLASH_SAMPLING_V7 ";

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DflashCandidateSource {
    /// The selector shortlist is still formed from raw drafter logits. Request
    /// policy cannot bring an outside token into this fixed top-16 yet.
    RawDraftLogitsTop16BeforeRequestPolicy,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DflashQNormalization {
    PairAdjustedThenBiasMaskTopKSoftmaxMinPTopP,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DflashTargetDistribution {
    PostRequestSamplerRollingHistory,
    PostRequestSamplerRollingHistoryWithForbiddenTokens,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DflashAcceptanceRule {
    MinOnePOverQ,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DflashResidualRule {
    NormalizePositivePMinusSparseQ,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DflashRngRule {
    /// One `StdRng::seed_from_u64` stream represented as `f32`: draft rows,
    /// acceptance rows, then the residual/bonus draw.
    StdRngSequentialF32DraftAcceptResidual,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DflashCanonicalCommitRule {
    /// Canonical recurrent/KV state contains the prior bonus/root and the
    /// actually committed draft prefix. The newly sampled correction/bonus
    /// remains `last_token` and is fed by the following decode step.
    RootAndCommittedDraftsNextTokenUnfed,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DflashNextTokenDisposition {
    /// The response terminated while walking the accepted draft prefix.
    NotAttempted,
    /// The correction/bonus was appended to output and armed as `last_token`.
    OutputAppendedAndArmed,
    /// A structural token was consumed without output and armed as `last_token`.
    ConsumedSilentlyAndArmed,
    /// Token handling terminated without output or feedback state.
    AttemptedTerminalWithoutFeedback,
}

impl DflashNextTokenDisposition {
    fn attempted(self) -> bool {
        self != Self::NotAttempted
    }

    fn output_appended(self) -> bool {
        self == Self::OutputAppendedAndArmed
    }

    fn armed_as_last_token(self) -> bool {
        matches!(
            self,
            Self::OutputAppendedAndArmed | Self::ConsumedSilentlyAndArmed
        )
    }
}

/// Actual scheduler outcome after token side effects and canonical-state
/// commit. This deliberately differs from the mathematical acceptance record:
/// cancellation, length, EOS, or streaming termination may stop emission
/// before every mathematically accepted draft is committed.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DflashCommitReceipt {
    pub ordinal: u64,
    pub rule: DflashCanonicalCommitRule,
    pub verify_rows: usize,
    pub committed_drafts: usize,
    pub total_target_slots: usize,
    pub next_disposition: DflashNextTokenDisposition,
    pub next_was_structural: bool,
    pub think_end_token: Option<u32>,
    pub next_was_think_end: bool,
    /// The asynchronous canonical-commit call returned success. This does not
    /// claim device completion; stream ordering is handled by the caller.
    pub commit_call_returned_ok: bool,
    pub seq_len_before: usize,
    pub seq_len_after: usize,
    pub finished: bool,
    pub output_tokens_before: usize,
    pub output_tokens_after: usize,
    pub last_token_before: u32,
    pub last_token_after: u32,
    /// Exact callback outcome captured while committing the next token.  This
    /// is intentionally not reconstructed from final `finished`: the plain
    /// thinking path requests the boundary snapshot before its final timeout
    /// check, so a deadline can expire after this becomes true.
    pub think_end_snapshot_requested: bool,
    /// True only when an emitted `</think>` requested a rollback-ring
    /// snapshot and the canonical-state event was ordered onto the snapshot's
    /// stream before the save was attempted.
    pub think_end_snapshot_ordered: bool,
}

impl DflashCommitReceipt {
    fn validate(
        &self,
        request: &DflashDraftSamplingRequest,
        proposal: &DflashSampledProposal,
        acceptance: &DflashSampledAcceptance,
    ) -> Result<()> {
        let gamma = proposal.tokens.len();
        ensure!(
            self.rule == DflashCanonicalCommitRule::RootAndCommittedDraftsNextTokenUnfed,
            "receipt canonical commit rule drifted"
        );
        ensure!(
            self.commit_call_returned_ok,
            "receipt cannot claim a failed canonical commit call"
        );
        ensure!(
            self.verify_rows == gamma + 1,
            "receipt verify-row count disagrees with proposal width"
        );
        ensure!(
            self.committed_drafts <= acceptance.accepted,
            "receipt committed more drafts than target accepted"
        );
        let expected_target_slots = self
            .committed_drafts
            .checked_add(1)
            .ok_or_else(|| anyhow::anyhow!("receipt canonical target-slot count overflow"))?;
        ensure!(
            self.total_target_slots == expected_target_slots
                && self.total_target_slots <= self.verify_rows,
            "receipt canonical target-slot count is inconsistent"
        );
        ensure!(
            self.seq_len_after
                == self
                    .seq_len_before
                    .checked_add(self.total_target_slots)
                    .ok_or_else(|| anyhow::anyhow!("receipt sequence length overflow"))?,
            "receipt canonical sequence-length transition is inconsistent"
        );
        ensure!(
            self.output_tokens_before == request.output_position,
            "receipt pre-commit output position drift"
        );
        let next_attempted = self.next_disposition.attempted();
        ensure!(
            !next_attempted || self.committed_drafts == acceptance.accepted,
            "receipt attempted next token before committing the accepted draft prefix"
        );
        ensure!(
            self.committed_drafts == acceptance.accepted
                || (self.finished
                    && self.next_disposition == DflashNextTokenDisposition::NotAttempted),
            "receipt truncated accepted drafts without a terminal outcome"
        );
        ensure!(
            self.finished || next_attempted,
            "nonterminal receipt did not attempt its correction/bonus token"
        );
        ensure!(
            self.next_disposition != DflashNextTokenDisposition::NotAttempted
                || (self.finished && acceptance.accepted > 0),
            "not-attempted disposition requires termination while walking a nonempty accepted prefix"
        );
        ensure!(
            self.next_disposition != DflashNextTokenDisposition::AttemptedTerminalWithoutFeedback
                || self.finished,
            "terminal-without-feedback disposition requires a terminal outcome"
        );
        ensure!(
            self.next_disposition != DflashNextTokenDisposition::ConsumedSilentlyAndArmed
                || (self.next_was_structural && !self.finished),
            "silently consumed disposition requires a nonterminal structural boundary"
        );
        let expected_output_after = self
            .output_tokens_before
            .checked_add(self.committed_drafts)
            .and_then(|value| {
                value.checked_add(usize::from(self.next_disposition.output_appended()))
            })
            .ok_or_else(|| anyhow::anyhow!("receipt output length overflow"))?;
        ensure!(
            self.output_tokens_after == expected_output_after,
            "receipt output-token delta disagrees with exact token dispositions"
        );
        let q_rows = proposal
            .q_rows
            .as_ref()
            .context("receipt probabilistic proposal omitted sparse q rows")?;
        ensure!(
            q_rows
                .first()
                .is_some_and(|row| row.predecessor == self.last_token_before),
            "receipt prior last token disagrees with sparse-q anchor"
        );
        let expected_last = if self.next_disposition.armed_as_last_token() {
            acceptance.next_token
        } else if self.committed_drafts > 0 {
            proposal.tokens[self.committed_drafts - 1]
        } else {
            self.last_token_before
        };
        ensure!(
            self.last_token_after == expected_last,
            "receipt last-token transition disagrees with actual commit fields"
        );
        let next_is_think_end = request.target_policy_phase
            == DflashTargetPolicyPhase::InsideThinking
            && self.think_end_token == Some(acceptance.next_token);
        ensure!(
            self.next_was_think_end == next_is_think_end
                && (!self.next_was_think_end || self.next_was_structural),
            "receipt think-end classification drift"
        );
        ensure!(
            !self.think_end_snapshot_requested
                || (self.next_was_think_end
                    && self.next_disposition.armed_as_last_token()
                    && request.target_policy_phase == DflashTargetPolicyPhase::InsideThinking),
            "receipt thinking-boundary snapshot request is inconsistent"
        );
        ensure!(
            !(self.next_was_think_end
                && self.next_disposition.armed_as_last_token()
                && !self.finished)
                || self.think_end_snapshot_requested,
            "nonterminal armed think-end omitted its snapshot request"
        );
        ensure!(
            self.think_end_snapshot_ordered == self.think_end_snapshot_requested,
            "receipt thinking-boundary snapshot outcome is inconsistent"
        );
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DflashSamplingReceipt {
    pub schema: u32,
    pub method: DflashDraftSampleMethod,
    pub request: DflashDraftSamplingRequest,
    /// Duplicated at the receipt top level so compact consumers can establish
    /// semantic phase without interpreting scheduler logs or nested policy.
    pub target_policy_phase: DflashTargetPolicyPhase,
    /// Exact effective sampler/mask policy identity, copied from the
    /// producer request and revalidated against it on every parse.
    pub target_policy_sha256: [u8; 32],
    pub vocab_size: usize,
    pub candidate_source: DflashCandidateSource,
    pub q_normalization: DflashQNormalization,
    pub target_distribution: DflashTargetDistribution,
    pub acceptance_rule: DflashAcceptanceRule,
    pub residual_rule: DflashResidualRule,
    pub rng_rule: DflashRngRule,
    pub proposal: DflashSampledProposal,
    pub acceptance: DflashSampledAcceptance,
    pub target_evidence: DflashTargetEvidence,
    pub commit: DflashCommitReceipt,
    /// Digest of the entire schema-7 receipt with this field zeroed. Compact
    /// commit logs carry the same value, binding them to an exact standalone
    /// receipt even when concurrent requests interleave their log lines.
    pub execution_sha256: [u8; 32],
}

impl DflashSamplingReceipt {
    pub fn new(
        request: DflashDraftSamplingRequest,
        vocab_size: usize,
        proposal: DflashSampledProposal,
        acceptance: DflashSampledAcceptance,
        target_evidence: DflashTargetEvidence,
        commit: DflashCommitReceipt,
    ) -> Result<Self> {
        let target_distribution = if request.target_forbidden_token_ids.is_empty() {
            DflashTargetDistribution::PostRequestSamplerRollingHistory
        } else {
            DflashTargetDistribution::PostRequestSamplerRollingHistoryWithForbiddenTokens
        };
        let target_policy_phase = request.target_policy_phase;
        let target_policy_sha256 = request.target_policy_sha256;
        let receipt = Self {
            schema: DFLASH_SAMPLING_RECEIPT_SCHEMA,
            method: DflashDraftSampleMethod::Probabilistic,
            request,
            target_policy_phase,
            target_policy_sha256,
            vocab_size,
            candidate_source: DflashCandidateSource::RawDraftLogitsTop16BeforeRequestPolicy,
            q_normalization: DflashQNormalization::PairAdjustedThenBiasMaskTopKSoftmaxMinPTopP,
            target_distribution,
            acceptance_rule: DflashAcceptanceRule::MinOnePOverQ,
            residual_rule: DflashResidualRule::NormalizePositivePMinusSparseQ,
            rng_rule: DflashRngRule::StdRngSequentialF32DraftAcceptResidual,
            proposal,
            acceptance,
            target_evidence,
            commit,
            execution_sha256: [0; 32],
        };
        let mut receipt = receipt;
        receipt.execution_sha256 = receipt.calculated_execution_sha256()?;
        receipt.validate()?;
        Ok(receipt)
    }

    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == DFLASH_SAMPLING_RECEIPT_SCHEMA,
            "unsupported DFlash sampling receipt schema {}",
            self.schema
        );
        ensure!(
            self.method == DflashDraftSampleMethod::Probabilistic,
            "schema-7 sampled receipt must name the probabilistic method"
        );
        ensure!(
            self.target_policy_phase == self.request.target_policy_phase,
            "receipt target-policy phase disagrees with its producer request"
        );
        ensure!(
            self.target_policy_sha256 == self.request.target_policy_sha256,
            "receipt target-policy digest disagrees with its producer request"
        );
        ensure!(self.vocab_size > 0, "receipt vocabulary must be positive");
        ensure!(
            self.candidate_source == DflashCandidateSource::RawDraftLogitsTop16BeforeRequestPolicy,
            "receipt candidate source does not disclose the raw top-16 membership limit"
        );
        ensure!(
            self.q_normalization
                == DflashQNormalization::PairAdjustedThenBiasMaskTopKSoftmaxMinPTopP,
            "receipt q normalization is not the post-request-policy distribution"
        );
        ensure!(
            self.acceptance_rule == DflashAcceptanceRule::MinOnePOverQ
                && self.residual_rule == DflashResidualRule::NormalizePositivePMinusSparseQ
                && self.rng_rule == DflashRngRule::StdRngSequentialF32DraftAcceptResidual,
            "receipt sampled-DFlash rules drifted"
        );
        ensure!(
            self.target_distribution
                == if self.request.target_forbidden_token_ids.is_empty() {
                    DflashTargetDistribution::PostRequestSamplerRollingHistory
                } else {
                    DflashTargetDistribution::PostRequestSamplerRollingHistoryWithForbiddenTokens
                },
            "receipt target distribution disagrees with its forbidden-token policy"
        );
        DflashSampledProposalFrame {
            request: self.request.clone(),
            proposal: self.proposal.clone(),
        }
        .validate(self.vocab_size)?;
        let gamma = self.proposal.tokens.len();
        self.request.validate(gamma)?;
        let target_probabilities = self
            .target_evidence
            .validated_target_probabilities(gamma + 1, self.vocab_size)?;
        ensure!(
            self.acceptance.accepted <= gamma,
            "receipt accepted count exceeds gamma"
        );
        ensure!(
            (self.acceptance.next_token as usize) < self.vocab_size,
            "receipt next token is outside the vocabulary"
        );
        let evaluated = if self.acceptance.accepted == gamma {
            gamma
        } else {
            self.acceptance.accepted + 1
        };
        ensure!(
            self.acceptance.acceptance_probabilities.len() == evaluated,
            "receipt acceptance-probability count is inconsistent"
        );
        ensure!(
            self.acceptance
                .acceptance_probabilities
                .iter()
                .all(|p| p.is_finite() && (0.0..=1.0).contains(p)),
            "receipt acceptance probabilities are invalid"
        );
        match self.acceptance.branch {
            DflashResidualBranch::AllAcceptedBonus => {
                ensure!(
                    self.acceptance.accepted == gamma
                        && self.acceptance.residual_mass_before_normalize.is_none(),
                    "all-accepted receipt has inconsistent residual fields"
                );
            }
            DflashResidualBranch::RejectionResidual => {
                ensure!(
                    self.acceptance.accepted < gamma
                        && self
                            .acceptance
                            .residual_mass_before_normalize
                            .is_some_and(|mass| mass.is_finite() && mass > 0.0),
                    "positive-residual receipt has inconsistent fields"
                );
            }
            DflashResidualBranch::RejectionTargetFallback => {
                ensure!(
                    self.acceptance.accepted < gamma
                        && self.acceptance.residual_mass_before_normalize == Some(0.0),
                    "target-fallback receipt requires a zero residual mass"
                );
            }
        }
        let replayed = super::accept_sparse_dflash(
            &target_probabilities,
            &self.proposal,
            &self.request.accept_uniforms,
            self.request.residual_uniform,
        )?;
        ensure!(
            replayed == self.acceptance,
            "receipt acceptance disagrees with exact target-p/q replay"
        );
        self.commit
            .validate(&self.request, &self.proposal, &self.acceptance)?;
        ensure!(
            self.execution_sha256 == self.calculated_execution_sha256()?,
            "receipt execution digest drift"
        );
        Ok(())
    }

    fn calculated_execution_sha256(&self) -> Result<[u8; 32]> {
        let mut payload = self.clone();
        payload.execution_sha256 = [0; 32];
        let encoded = serde_json::to_vec(&payload)
            .context("serialize canonical DFlash execution receipt payload")?;
        let mut digest = DigestContext::new(&SHA256);
        digest.update(b"atlas-dflash-execution-receipt-v7\0");
        digest.update(&(encoded.len() as u64).to_le_bytes());
        digest.update(&encoded);
        Ok(digest.finish().as_ref().try_into().expect("SHA-256 width"))
    }

    pub fn execution_sha256_hex(&self) -> String {
        use std::fmt::Write as _;

        let mut encoded = String::with_capacity(64);
        for byte in self.execution_sha256 {
            write!(&mut encoded, "{byte:02x}").expect("write to String");
        }
        encoded
    }

    pub fn to_json_line(&self) -> Result<String> {
        self.validate()?;
        let json = serde_json::to_string(self).context("serialize DFlash sampling receipt")?;
        Ok(format!("{DFLASH_SAMPLING_RECEIPT_PREFIX}{json}"))
    }

    pub fn parse_json_line(line: &str) -> Result<Self> {
        ensure!(
            line.trim() == line,
            "DFlash sampling receipt has outer whitespace"
        );
        let json = line
            .strip_prefix(DFLASH_SAMPLING_RECEIPT_PREFIX)
            .context("missing DFLASH_SAMPLING_V7 receipt prefix")?;
        ensure!(!json.is_empty(), "DFlash sampling receipt payload is empty");
        let receipt: Self =
            serde_json::from_str(json).context("parse strict DFlash sampling receipt")?;
        receipt.validate()?;
        Ok(receipt)
    }
}
