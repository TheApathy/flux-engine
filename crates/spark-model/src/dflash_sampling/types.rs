// SPDX-License-Identifier: AGPL-3.0-only

use std::{fmt, str::FromStr};

use anyhow::{Result, ensure};
use ring::digest::{Context as DigestContext, SHA256};
use serde::{Deserialize, Serialize};
use spark_runtime::sampler::{SamplingParams, sampling_uniforms_from_seed};

pub const DFLASH2_SELECTOR_TOP_K: usize = 16;
/// Maximum number of request bias entries plus target hard masks admitted by
/// the sampled-selector CUDA ABI. Request bias entries remain ordered so
/// duplicate-token additions retain ordinary sampling's f32 semantics.
pub const DFLASH2_MAX_Q_MODIFIERS: usize = 64;
pub(crate) const PROBABILITY_SUM_TOLERANCE: f32 = 2.0e-5;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DflashDraftSampleMethod {
    #[default]
    Greedy,
    Probabilistic,
}

impl fmt::Display for DflashDraftSampleMethod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Greedy => f.write_str("greedy"),
            Self::Probabilistic => f.write_str("probabilistic"),
        }
    }
}

impl FromStr for DflashDraftSampleMethod {
    type Err = String;

    fn from_str(value: &str) -> std::result::Result<Self, Self::Err> {
        match value {
            "greedy" => Ok(Self::Greedy),
            "probabilistic" => Ok(Self::Probabilistic),
            other => Err(format!(
                "unknown DFlash draft sample method {other:?}; expected `greedy` or `probabilistic`"
            )),
        }
    }
}

/// Admit only runtime modes whose end-to-end producer/consumer ABI exists.
///
/// Source carries the CUDA selector, typed sparse-q transport, and narrow
/// target p/q scheduler transaction. Probabilistic execution is admitted only
/// behind the explicit qualification flag and an actual DFlash2 selector;
/// this source gate is not endpoint or promotion evidence.
pub fn validate_dflash_runtime_admission(
    method: DflashDraftSampleMethod,
    dflash2_selector_present: bool,
    probabilistic_qualification_enabled: bool,
) -> Result<()> {
    if method == DflashDraftSampleMethod::Greedy {
        ensure!(
            !probabilistic_qualification_enabled,
            "probabilistic DFlash qualification was enabled for the greedy selector"
        );
        return Ok(());
    }
    ensure!(
        probabilistic_qualification_enabled,
        "--dflash-draft-sample-method probabilistic is experimental and requires \
         --experimental-dflash-probabilistic"
    );
    ensure!(
        dflash2_selector_present,
        "--dflash-draft-sample-method probabilistic requires a DFlash2 checkpoint with a \
         candidate selector; DSpark/legacy DFlash selectors are not admitted"
    );
    Ok(())
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RequestSamplerReceipt {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: u32,
    pub top_n_sigma: f32,
    pub min_p: f32,
    pub logit_bias: Vec<(u32, f32)>,
    pub repetition_penalty: f32,
    pub repetition_penalty_window: u32,
    pub presence_penalty: f32,
    pub frequency_penalty: f32,
    pub lz_penalty: f32,
    pub dry_multiplier: f32,
    pub dry_base: f32,
    pub dry_allowed_length: u32,
    pub dry_sequence_breakers: Vec<u32>,
    pub max_tokens: usize,
    pub stop_token_ids: Vec<u32>,
    /// Effective per-output seed. `ActiveSeq::sampling_params` advances the
    /// request seed by the current generated-token position before this
    /// receipt is captured.
    pub seed: Option<u64>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DflashRngSeedSource {
    /// The effective seed carried by `SamplingParams` for this output
    /// position. Its request-scoped base may be explicit API input or entropy
    /// captured once by the scheduler before the first target sample.
    RequestEffectiveSeed,
    /// Reserved for frame-local entropy. Runtime admission remains closed for
    /// this source because it cannot also drive ordinary target sampling.
    CapturedEntropy,
}

/// Scheduler policy phase under which both sparse draft `q` and target `p`
/// were sampled.  This is carried by the producer request (rather than only
/// an adjacent server log) so a standalone execution receipt can prove which
/// thinking policy was in force.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DflashTargetPolicyPhase {
    #[default]
    OutsideThinking,
    InsideThinking,
    PostThinking,
}

/// Per-frame DFlash selector sampling inputs staged by the serving layer.
///
/// The uniforms are explicit rather than generated inside CUDA so a receipt
/// can replay the exact selector walk independently of device RNG state.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DflashDraftSamplingRequest {
    pub request_sampler: RequestSamplerReceipt,
    /// Hard masks that ordinary decoding applies outside the request sampler.
    /// The serving integration uses this for the `<think>` / `</think>`
    /// re-entry fence after a non-thinking chat template. Both target p and
    /// sparse draft q apply these masks before normalization.
    #[serde(default)]
    pub target_forbidden_token_ids: Vec<u32>,
    /// Explicit semantic phase for the effective target policy.
    pub target_policy_phase: DflashTargetPolicyPhase,
    /// Domain-separated SHA-256 over `target_policy_phase`, the complete
    /// effective request sampler, and `target_forbidden_token_ids`.
    /// Validation recomputes it, so a stale or partially mutated frame is
    /// rejected before either selector or verifier can consume it.
    pub target_policy_sha256: [u8; 32],
    /// Generated-output index whose sampler settings and RNG stream this
    /// frame represents.
    pub output_position: usize,
    /// Exact live-history identity at proposal time. The verifier recomputes
    /// this digest before constructing rolling counterfactual histories.
    pub history_len: usize,
    pub history_sha256: [u8; 32],
    pub seed_source: DflashRngSeedSource,
    /// Seed for the one frame-local `StdRng` stream.
    pub seed: u64,
    pub draft_uniforms: Vec<f32>,
    pub accept_uniforms: Vec<f32>,
    pub residual_uniform: f32,
}

impl DflashDraftSamplingRequest {
    /// Construct a replayable frame from the exact seeded sampler settings
    /// used by ordinary decode at `output_position`.
    ///
    /// Draw order is fixed: `gamma` draft draws, then `gamma` acceptance
    /// draws, then one residual/bonus draw. All are materialized as `f32`,
    /// matching ordinary decode and the CUDA selector CDF boundary.
    pub fn from_seeded_sampling_params(
        params: &SamplingParams,
        history: &[u32],
        output_position: usize,
        gamma: usize,
    ) -> Result<Self> {
        ensure!(gamma > 0, "DFlash sampled request requires gamma > 0");
        ensure!(
            output_position == history.len(),
            "DFlash output position must equal generated history length"
        );
        let seed = params.seed.ok_or_else(|| {
            anyhow::anyhow!("seeded DFlash request requires ActiveSeq's effective per-output seed")
        })?;
        let draw_count = gamma
            .checked_mul(2)
            .and_then(|count| count.checked_add(1))
            .ok_or_else(|| anyhow::anyhow!("DFlash RNG draw count overflow"))?;
        let draws = sampling_uniforms_from_seed(seed, draw_count);
        let request_sampler = RequestSamplerReceipt::from(params);
        let target_forbidden_token_ids = Vec::new();
        let target_policy_phase = DflashTargetPolicyPhase::OutsideThinking;
        let target_policy_sha256 = dflash_target_policy_sha256(
            target_policy_phase,
            &request_sampler,
            &target_forbidden_token_ids,
        );
        let request = Self {
            request_sampler,
            target_forbidden_token_ids,
            target_policy_phase,
            target_policy_sha256,
            output_position,
            history_len: history.len(),
            history_sha256: dflash_history_sha256(history),
            seed_source: DflashRngSeedSource::RequestEffectiveSeed,
            seed,
            draft_uniforms: draws[..gamma].to_vec(),
            accept_uniforms: draws[gamma..gamma * 2].to_vec(),
            residual_uniform: draws[gamma * 2],
        };
        request.validate(gamma)?;
        Ok(request)
    }

    /// Attach a sorted, duplicate-free target policy mask to the replayable
    /// frame. The same mask is applied to target p and sparse draft q.
    pub fn with_target_forbidden_token_ids(mut self, mut token_ids: Vec<u32>) -> Result<Self> {
        token_ids.sort_unstable();
        token_ids.dedup();
        self.target_forbidden_token_ids = token_ids;
        self.refresh_target_policy_digest();
        self.validate(self.draft_uniforms.len())?;
        Ok(self)
    }

    /// Bind this frame to the scheduler policy phase that produced its
    /// effective sampler and masks.
    pub fn with_target_policy_phase(mut self, phase: DflashTargetPolicyPhase) -> Result<Self> {
        self.target_policy_phase = phase;
        self.refresh_target_policy_digest();
        self.validate(self.draft_uniforms.len())?;
        Ok(self)
    }

    fn refresh_target_policy_digest(&mut self) {
        self.target_policy_sha256 = dflash_target_policy_sha256(
            self.target_policy_phase,
            &self.request_sampler,
            &self.target_forbidden_token_ids,
        );
    }

    pub fn validate(&self, gamma: usize) -> Result<()> {
        ensure!(gamma > 0, "DFlash sampled request requires gamma > 0");
        self.request_sampler.validate()?;
        let q_modifier_count = self
            .request_sampler
            .logit_bias
            .len()
            .checked_add(self.target_forbidden_token_ids.len())
            .ok_or_else(|| anyhow::anyhow!("DFlash q modifier count overflow"))?;
        ensure!(
            q_modifier_count <= DFLASH2_MAX_Q_MODIFIERS,
            "DFlash q modifier count {q_modifier_count} exceeds bounded ABI capacity {DFLASH2_MAX_Q_MODIFIERS}"
        );
        ensure!(
            self.request_sampler.top_n_sigma == 0.0,
            "sampled DFlash q does not yet support top_n_sigma"
        );
        ensure!(
            self.target_forbidden_token_ids
                .windows(2)
                .all(|pair| pair[0] < pair[1]),
            "DFlash target-forbidden token ids must be sorted and unique"
        );
        ensure!(
            self.target_policy_sha256
                == dflash_target_policy_sha256(
                    self.target_policy_phase,
                    &self.request_sampler,
                    &self.target_forbidden_token_ids,
                ),
            "DFlash target-policy digest drift"
        );
        ensure!(
            self.request_sampler.temperature > 0.0,
            "sampled DFlash request requires positive temperature"
        );
        ensure!(
            self.output_position == self.history_len,
            "DFlash output position/history length drift"
        );
        ensure!(
            self.history_sha256 != [0; 32],
            "DFlash history digest must be populated"
        );
        match self.seed_source {
            DflashRngSeedSource::RequestEffectiveSeed => ensure!(
                self.request_sampler.seed == Some(self.seed),
                "DFlash frame seed disagrees with the effective request seed"
            ),
            DflashRngSeedSource::CapturedEntropy => ensure!(
                self.request_sampler.seed.is_none(),
                "captured-entropy DFlash frame cannot claim a request seed"
            ),
        }
        ensure!(
            self.draft_uniforms.len() == gamma,
            "sampled DFlash request needs one draft uniform per row"
        );
        ensure!(
            self.accept_uniforms.len() == gamma,
            "sampled DFlash request needs one acceptance uniform per row"
        );
        for &uniform in self
            .draft_uniforms
            .iter()
            .chain(self.accept_uniforms.iter())
            .chain(std::iter::once(&self.residual_uniform))
        {
            ensure!(
                uniform.is_finite() && (0.0..1.0).contains(&uniform),
                "DFlash RNG uniform must be finite and in [0, 1)"
            );
        }
        let draw_count = gamma
            .checked_mul(2)
            .and_then(|count| count.checked_add(1))
            .ok_or_else(|| anyhow::anyhow!("DFlash RNG draw count overflow"))?;
        let expected = sampling_uniforms_from_seed(self.seed, draw_count);
        ensure!(
            self.draft_uniforms == expected[..gamma],
            "DFlash draft RNG draws disagree with the receipted seed"
        );
        ensure!(
            self.accept_uniforms == expected[gamma..gamma * 2],
            "DFlash acceptance RNG draws disagree with the receipted seed"
        );
        ensure!(
            self.residual_uniform == expected[gamma * 2],
            "DFlash residual RNG draw disagrees with the receipted seed"
        );
        Ok(())
    }

    pub fn validate_history(&self, history: &[u32]) -> Result<()> {
        ensure!(
            self.history_len == history.len(),
            "DFlash generated history length drift"
        );
        ensure!(
            self.history_sha256 == dflash_history_sha256(history),
            "DFlash generated history digest drift"
        );
        Ok(())
    }

    pub fn validate_sampler(&self, params: &SamplingParams) -> Result<()> {
        let current = RequestSamplerReceipt::from(params);
        current.validate()?;
        ensure!(
            self.request_sampler == current,
            "DFlash request sampler receipt drift"
        );
        Ok(())
    }
}

impl From<&SamplingParams> for RequestSamplerReceipt {
    fn from(params: &SamplingParams) -> Self {
        Self {
            temperature: params.temperature,
            top_p: params.top_p,
            top_k: params.top_k,
            top_n_sigma: params.top_n_sigma,
            min_p: params.min_p,
            logit_bias: params.logit_bias.clone(),
            repetition_penalty: params.repetition_penalty,
            repetition_penalty_window: params.repetition_penalty_window,
            presence_penalty: params.presence_penalty,
            frequency_penalty: params.frequency_penalty,
            lz_penalty: params.lz_penalty,
            dry_multiplier: params.dry_multiplier,
            dry_base: params.dry_base,
            dry_allowed_length: params.dry_allowed_length,
            dry_sequence_breakers: params.dry_sequence_breakers.clone(),
            max_tokens: params.max_tokens,
            stop_token_ids: params.stop_token_ids.clone(),
            seed: params.seed,
        }
    }
}

impl RequestSamplerReceipt {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.temperature.is_finite() && self.temperature >= 0.0,
            "request temperature must be finite and non-negative"
        );
        ensure!(
            self.top_p.is_finite() && self.top_p > 0.0 && self.top_p <= 1.0,
            "request top_p must be finite and in (0, 1]"
        );
        ensure!(
            self.min_p.is_finite() && (0.0..=1.0).contains(&self.min_p),
            "request min_p must be finite and in [0, 1]"
        );
        ensure!(
            self.top_n_sigma.is_finite() && self.top_n_sigma >= 0.0,
            "request top_n_sigma must be finite and non-negative"
        );
        ensure!(
            self.logit_bias.iter().all(|(_, bias)| bias.is_finite()),
            "request logit_bias contains a non-finite value"
        );
        ensure!(
            self.presence_penalty.is_finite() && self.frequency_penalty.is_finite(),
            "request penalties must be finite"
        );
        ensure!(
            self.repetition_penalty.is_finite() && self.repetition_penalty > 0.0,
            "request repetition_penalty must be finite and positive"
        );
        ensure!(
            self.lz_penalty.is_finite() && self.dry_multiplier.is_finite(),
            "request LZ/DRY multipliers must be finite"
        );
        ensure!(
            self.dry_base.is_finite() && self.dry_base > 0.0,
            "request dry_base must be finite and positive"
        );
        Ok(())
    }
}

/// Stable history identity for the proposal/verify hand-off. Token IDs are
/// hashed as little-endian u32 values under a versioned domain separator.
pub fn dflash_history_sha256(history: &[u32]) -> [u8; 32] {
    let mut context = DigestContext::new(&SHA256);
    context.update(b"atlas-dflash-history-v1\0");
    context.update(&(history.len() as u64).to_le_bytes());
    for &token in history {
        context.update(&token.to_le_bytes());
    }
    context.finish().as_ref().try_into().expect("SHA-256 width")
}

/// Stable identity for every input that changes the effective target sampling
/// distribution.  Values use fixed-width little-endian encodings and f32 bit
/// patterns; vector lengths are included to prevent concatenation ambiguity.
pub fn dflash_target_policy_sha256(
    phase: DflashTargetPolicyPhase,
    sampler: &RequestSamplerReceipt,
    forbidden_token_ids: &[u32],
) -> [u8; 32] {
    fn update_u32(context: &mut DigestContext, value: u32) {
        context.update(&value.to_le_bytes());
    }
    fn update_u64(context: &mut DigestContext, value: u64) {
        context.update(&value.to_le_bytes());
    }
    fn update_f32(context: &mut DigestContext, value: f32) {
        update_u32(context, value.to_bits());
    }
    fn update_u32_slice(context: &mut DigestContext, values: &[u32]) {
        update_u64(context, values.len() as u64);
        for &value in values {
            update_u32(context, value);
        }
    }

    let mut context = DigestContext::new(&SHA256);
    context.update(b"atlas-dflash-target-policy-v1\0");
    context.update(&[match phase {
        DflashTargetPolicyPhase::OutsideThinking => 0,
        DflashTargetPolicyPhase::InsideThinking => 1,
        DflashTargetPolicyPhase::PostThinking => 2,
    }]);
    update_f32(&mut context, sampler.temperature);
    update_f32(&mut context, sampler.top_p);
    update_u32(&mut context, sampler.top_k);
    update_f32(&mut context, sampler.top_n_sigma);
    update_f32(&mut context, sampler.min_p);
    update_u64(&mut context, sampler.logit_bias.len() as u64);
    for &(token, bias) in &sampler.logit_bias {
        update_u32(&mut context, token);
        update_f32(&mut context, bias);
    }
    update_f32(&mut context, sampler.repetition_penalty);
    update_u32(&mut context, sampler.repetition_penalty_window);
    update_f32(&mut context, sampler.presence_penalty);
    update_f32(&mut context, sampler.frequency_penalty);
    update_f32(&mut context, sampler.lz_penalty);
    update_f32(&mut context, sampler.dry_multiplier);
    update_f32(&mut context, sampler.dry_base);
    update_u32(&mut context, sampler.dry_allowed_length);
    update_u32_slice(&mut context, &sampler.dry_sequence_breakers);
    update_u64(&mut context, sampler.max_tokens as u64);
    update_u32_slice(&mut context, &sampler.stop_token_ids);
    match sampler.seed {
        Some(seed) => {
            context.update(&[1]);
            update_u64(&mut context, seed);
        }
        None => context.update(&[0]),
    }
    update_u32_slice(&mut context, forbidden_token_ids);
    context.finish().as_ref().try_into().expect("SHA-256 width")
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DflashSparseQRow {
    pub position: usize,
    pub predecessor: u32,
    pub candidates: [u32; DFLASH2_SELECTOR_TOP_K],
    pub probabilities: [f32; DFLASH2_SELECTOR_TOP_K],
    pub selected_index: usize,
    pub selected_token: u32,
    pub probability_sum: f32,
}

impl DflashSparseQRow {
    pub fn validate(&self, expected_position: usize, vocab_size: usize) -> Result<()> {
        ensure!(
            self.position == expected_position,
            "sparse q position drift"
        );
        ensure!(
            (self.predecessor as usize) < vocab_size,
            "sparse q predecessor is outside the vocabulary"
        );
        ensure!(
            self.selected_index < DFLASH2_SELECTOR_TOP_K,
            "sparse q selected index is outside top-16"
        );
        for (i, &candidate) in self.candidates.iter().enumerate() {
            ensure!(
                (candidate as usize) < vocab_size,
                "sparse q candidate is outside the vocabulary"
            );
            ensure!(
                !self.candidates[..i].contains(&candidate),
                "sparse q candidates must be unique"
            );
        }
        ensure!(
            self.selected_token == self.candidates[self.selected_index],
            "sparse q selected token disagrees with selected index"
        );
        validate_probability_row(&self.probabilities)?;
        let sum: f32 = self.probabilities.iter().sum();
        ensure!(
            self.probability_sum.is_finite()
                && (self.probability_sum - sum).abs() <= PROBABILITY_SUM_TOLERANCE,
            "sparse q recorded probability sum is inconsistent"
        );
        ensure!(
            self.probabilities[self.selected_index] > 0.0,
            "selected draft must have positive q probability"
        );
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DflashSampledProposal {
    pub tokens: Vec<u32>,
    /// Effective normalized q after pair adjustment, ordered request
    /// `logit_bias`, target hard masks, top-k, min-p and top-p. Candidate
    /// membership is still the raw drafter top-16. These exact post-policy
    /// rows drive p/q acceptance and are serialized in the sampling receipt.
    pub q_rows: Option<Vec<DflashSparseQRow>>,
}

/// Drain-once producer frame. The request that generated q travels with the
/// proposal so a later target-p consumer cannot accidentally pair q with a
/// different temperature, seed, or RNG sequence.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DflashSampledProposalFrame {
    pub request: DflashDraftSamplingRequest,
    pub proposal: DflashSampledProposal,
}

impl DflashSampledProposalFrame {
    pub fn validate(&self, vocab_size: usize) -> Result<()> {
        self.request.validate(self.proposal.tokens.len())?;
        ensure!(
            self.request
                .target_forbidden_token_ids
                .iter()
                .all(|&token| (token as usize) < vocab_size),
            "DFlash target-forbidden token is outside the vocabulary"
        );
        ensure!(
            self.request.request_sampler.temperature > 0.0,
            "sampled DFlash frame requires positive temperature"
        );
        self.proposal
            .validate(DflashDraftSampleMethod::Probabilistic, vocab_size)?;
        let q_rows = self
            .proposal
            .q_rows
            .as_ref()
            .expect("validated probabilistic q rows");
        for (position, (row, &uniform)) in q_rows
            .iter()
            .zip(self.request.draft_uniforms.iter())
            .enumerate()
        {
            for (candidate, probability) in row.candidates.iter().zip(row.probabilities.iter()) {
                ensure!(
                    !self.request.target_forbidden_token_ids.contains(candidate)
                        || *probability == 0.0,
                    "DFlash target-forbidden candidate retained positive sparse-q mass"
                );
            }
            if position > 0 {
                ensure!(
                    row.predecessor == self.proposal.tokens[position - 1],
                    "sparse q predecessor chain drift"
                );
            }
            let expected = select_sparse_q_f32(&row.probabilities, uniform);
            ensure!(
                row.selected_index == expected,
                "sparse q selection disagrees with the receipted f32 CDF draw"
            );
        }
        Ok(())
    }
}

impl DflashSampledProposal {
    pub fn validate(&self, method: DflashDraftSampleMethod, vocab_size: usize) -> Result<()> {
        ensure!(!self.tokens.is_empty(), "DFlash proposal must not be empty");
        ensure!(
            self.tokens
                .iter()
                .all(|&token| (token as usize) < vocab_size),
            "DFlash proposal token is outside the vocabulary"
        );
        match (method, self.q_rows.as_deref()) {
            (DflashDraftSampleMethod::Greedy, None) => Ok(()),
            (DflashDraftSampleMethod::Probabilistic, Some(rows)) => {
                ensure!(rows.len() == self.tokens.len(), "sparse q row count drift");
                for (position, row) in rows.iter().enumerate() {
                    row.validate(position, vocab_size)?;
                    ensure!(
                        row.selected_token == self.tokens[position],
                        "sparse q row does not describe the proposed token"
                    );
                }
                Ok(())
            }
            (DflashDraftSampleMethod::Greedy, Some(_)) => {
                anyhow::bail!("greedy DFlash proposal must not claim sampled q rows")
            }
            (DflashDraftSampleMethod::Probabilistic, None) => {
                anyhow::bail!("probabilistic DFlash proposal omitted sparse q rows")
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DflashResidualBranch {
    RejectionResidual,
    RejectionTargetFallback,
    AllAcceptedBonus,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DflashSampledAcceptance {
    pub accepted: usize,
    pub next_token: u32,
    pub branch: DflashResidualBranch,
    pub residual_mass_before_normalize: Option<f64>,
    pub acceptance_probabilities: Vec<f32>,
}

pub(crate) fn validate_probability_row(row: &[f32]) -> Result<()> {
    ensure!(!row.is_empty(), "probability row must not be empty");
    ensure!(
        row.iter().all(|p| p.is_finite() && *p >= 0.0),
        "probability row contains a non-finite or negative value"
    );
    // Accumulate in f64: a valid full-vocabulary f32 distribution (Qwen is
    // ~152K entries) can drift well beyond this tolerance under sequential
    // f32 addition even though every stored probability is correct.
    let sum: f64 = row.iter().map(|&probability| probability as f64).sum();
    ensure!(
        sum.is_finite() && (sum - 1.0).abs() <= PROBABILITY_SUM_TOLERANCE as f64,
        "probability row is not normalized: sum={sum}"
    );
    Ok(())
}

pub(crate) fn validate_uniform(value: f64, label: &str) -> Result<()> {
    ensure!(
        value.is_finite() && (0.0..1.0).contains(&value),
        "{label} must be finite and in [0, 1)"
    );
    Ok(())
}

/// Replay the CUDA selector's strict-`>` f32 CDF, including its fallback to
/// the final positive-probability candidate when rounding leaves the CDF
/// below the draw. Trailing zero entries must never become the fallback.
pub(crate) fn select_sparse_q_f32(
    probabilities: &[f32; DFLASH2_SELECTOR_TOP_K],
    uniform: f32,
) -> usize {
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

#[cfg(test)]
mod cdf_tests {
    use super::*;

    #[test]
    fn f32_cdf_fallback_is_last_nonzero_not_physical_tail() {
        let mut probabilities = [0.0; DFLASH2_SELECTOR_TOP_K];
        probabilities[0] = 0.25;
        probabilities[3] = 0.25;
        assert_eq!(select_sparse_q_f32(&probabilities, 0.75), 3);
    }
}
