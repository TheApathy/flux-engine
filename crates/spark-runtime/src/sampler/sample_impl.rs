// SPDX-License-Identifier: AGPL-3.0-only
//
// Core sampling pipeline (`sample_with_params_history` + seeded
// implementation). Split out of `sampler.rs` to keep the parent file
// under the 500-line cap. The parent re-exports these via `pub use`.

use super::{SamplingParams, read_f32, record_entropy};

/// The exact post-processor distribution used by Atlas sampling.
///
/// We retain weights in the sampler's descending-logit order rather than
/// normalizing them in-place.  That preserves the existing seeded cumulative
/// sum and its floating-point boundary behavior.  Speculative rejection
/// sampling can independently request a dense normalized view of the same
/// distribution, so target `p` cannot drift from ordinary decode policy.
#[derive(Clone, Debug, PartialEq)]
pub struct PreparedSamplingDistribution {
    vocab_size: usize,
    /// A sampler result that is already determined (greedy or a degenerate
    /// filter fallback). Keep it out of `ordered_weights` so ordinary decode
    /// preserves its allocation-free, no-RNG fast return while probability
    /// consumers can still materialize the equivalent one-hot row.
    deterministic_token: Option<u32>,
    ordered_weights: Vec<(u32, f32)>,
    total_weight: f32,
}

impl PreparedSamplingDistribution {
    fn from_ordered_weights(vocab_size: usize, ordered_weights: Vec<(u32, f32)>) -> Self {
        let total_weight = ordered_weights.iter().map(|(_, weight)| *weight).sum();
        Self {
            vocab_size,
            deterministic_token: None,
            ordered_weights,
            total_weight,
        }
    }

    fn one_hot(vocab_size: usize, token: u32) -> Self {
        Self {
            vocab_size,
            deterministic_token: Some(token),
            ordered_weights: Vec::new(),
            total_weight: 1.0,
        }
    }

    /// The sampler-order (token, weight) list; tests compare orders with it.
    #[cfg(test)]
    pub(super) fn ordered_weights(&self) -> &[(u32, f32)] {
        &self.ordered_weights
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Return a dense normalized probability row, or `None` if hostile or
    /// invalid sampler inputs produced a degenerate distribution.  Serving
    /// keeps its historical fallback behavior; speculative acceptance must
    /// fail closed rather than invent `p` in that case.
    pub fn normalized_dense_probabilities(&self) -> Option<Vec<f32>> {
        if let Some(token) = self.deterministic_token {
            let mut dense = vec![0.0f32; self.vocab_size];
            *dense.get_mut(token as usize)? = 1.0;
            return Some(dense);
        }
        if !self.total_weight.is_finite() || self.total_weight <= 0.0 {
            return None;
        }
        let inverse = 1.0 / self.total_weight;
        let mut dense = vec![0.0f32; self.vocab_size];
        for &(token, weight) in &self.ordered_weights {
            let slot = dense.get_mut(token as usize)?;
            if !weight.is_finite() || weight < 0.0 {
                return None;
            }
            *slot += weight * inverse;
        }
        // Validate in f64.  A sequential f32 sum over a full Qwen vocabulary
        // can drift by more than 1e-3 even when every represented probability
        // is correct.  Moving that residue onto one token would silently turn
        // a near-uniform target row into a biased one, so acceptance consumers
        // must use the same f64 accumulation contract instead.
        let sum: f64 = dense
            .iter()
            .map(|&probability| f64::from(probability))
            .sum();
        if !sum.is_finite() || (sum - 1.0).abs() > 2.0e-5 {
            return None;
        }
        Some(dense)
    }

    pub(super) fn sample_with_uniform(&self, uniform: f32) -> u32 {
        if let Some(token) = self.deterministic_token {
            return token;
        }
        let threshold = uniform * self.total_weight;
        let mut cumulative = 0.0f32;
        for &(token, weight) in &self.ordered_weights {
            cumulative += weight;
            if cumulative >= threshold {
                return token;
            }
        }
        self.ordered_weights.last().map_or(0, |entry| entry.0)
    }
}

pub fn sample_with_params_history(
    data: &[u8],
    params: &SamplingParams,
    token_history: &[u32],
) -> u32 {
    sample_with_params_seeded(data, params, token_history, params.seed)
}

/// Core sampling pipeline with explicit seed control.
/// `seed` overrides the RNG for deterministic sampling. None = thread_rng.
pub fn sample_with_params_seeded(
    data: &[u8],
    params: &SamplingParams,
    token_history: &[u32],
    seed: Option<u64>,
) -> u32 {
    sample_seeded_impl(
        data,
        params,
        token_history,
        seed,
        super::device_topk::canonical_ties_enabled(),
    )
}

/// [`sample_with_params_seeded`] with the tie order explicit (tests drive both).
pub(super) fn sample_seeded_impl(
    data: &[u8],
    params: &SamplingParams,
    token_history: &[u32],
    seed: Option<u64>,
    canonical_ties: bool,
) -> u32 {
    if params.temperature > 0.0 && super::gumbel::gumbel_sampling_enabled() {
        return sample_gumbel_from_seeded_args(data, params, token_history, seed);
    }
    let distribution =
        prepare_sampling_distribution_impl(data, params, token_history, true, canonical_ties);
    // Preserve the historical greedy/degenerate fast return: these paths did
    // not allocate a categorical row, instantiate StdRng, or consume thread
    // RNG state. DFlash target-p preparation still receives the same one-hot
    // distribution through `normalized_dense_probabilities` above.
    if let Some(token) = distribution.deterministic_token {
        return token;
    }
    let random_val: f32 = if let Some(s) = seed {
        sampling_uniform_from_seed(s)
    } else {
        rand::random::<f32>()
    };
    distribution.sample_with_uniform(random_val)
}

/// Return the exact first `f32` draw used by seeded ordinary decoding.
///
/// DFlash rejection sampling uses this as its RNG compatibility anchor: a
/// frame seeded for output position `n` begins with the same draw that plain
/// decode would consume for output position `n`, then consumes explicitly
/// receipted proposal/acceptance/residual draws from that same `StdRng`
/// stream. Keeping this helper next to the ordinary sampler prevents the two
/// seeded paths from silently choosing different RNG algorithms.
pub fn sampling_uniform_from_seed(seed: u64) -> f32 {
    use rand::{Rng as _, SeedableRng as _};

    rand::rngs::StdRng::seed_from_u64(seed).r#gen::<f32>()
}

/// Return `count` sequential `f32` draws from ordinary decode's seeded RNG.
///
/// This deliberately returns the represented `f32` values, not widened
/// `f64` approximations, because selector CDF boundaries are part of the
/// execution contract.
pub fn sampling_uniforms_from_seed(seed: u64, count: usize) -> Vec<f32> {
    use rand::{Rng as _, SeedableRng as _};

    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    (0..count).map(|_| rng.r#gen::<f32>()).collect()
}

fn sample_gumbel_from_seeded_args(
    data: &[u8],
    params: &SamplingParams,
    token_history: &[u32],
    seed: Option<u64>,
) -> u32 {
    let (base_seed, abs_pos) = if let Some(ctx) = super::gumbel::current_gumbel_pos() {
        ctx
    } else {
        let out_pos = token_history.len() as u64;
        let s = seed
            .map(|v| v.wrapping_sub(out_pos))
            .unwrap_or_else(rand::random::<u64>);
        (s, out_pos)
    };
    choose_gumbel_impl(data, params, token_history, base_seed, abs_pos, true)
}

/// Deterministic position- and token-ID-keyed Gumbel-Max sampling over the exact
/// Atlas post-penalty, post-(`top_n_sigma`, `top_k`, `min_p`, `top_p`) support.
///
/// Picks `argmax_{j in kept} ((logit_j / max(T, 1e-6)) as f64 + gumbel_f64(seed, abs_pos, id_j))`
/// using [`super::candidate_order::CandidateOrder::Total`] (`(logit desc, token_id asc)`
/// O(V) `select_nth_unstable_by`). At `temperature <= 0.0`, reduces to exact
/// post-penalty greedy [`first_max_index`].
pub fn choose_gumbel(
    data: &[u8],
    params: &SamplingParams,
    token_history: &[u32],
    seed: u64,
    abs_pos: u64,
) -> u32 {
    choose_gumbel_impl(data, params, token_history, seed, abs_pos, false)
}

/// Zero-copy in-place variant of [`choose_gumbel`] operating directly on a mutable
/// `f32` logits slice (avoiding `Vec<u8>` serialization and `Vec<f32>` re-allocation
/// on multi-row speculative verify walks).
pub fn choose_gumbel_f32_in_place(
    raw_logits: &mut [f32],
    params: &SamplingParams,
    token_history: &[u32],
    seed: u64,
    abs_pos: u64,
) -> u32 {
    choose_gumbel_f32_impl(raw_logits, params, token_history, seed, abs_pos, false)
}

fn choose_gumbel_impl(
    data: &[u8],
    params: &SamplingParams,
    token_history: &[u32],
    seed: u64,
    abs_pos: u64,
    record_entropy_metric: bool,
) -> u32 {
    let n = data.len() / 4;
    if n == 0 {
        return 0;
    }
    let mut raw_logits: Vec<f32> = (0..n).map(|i| read_f32(data, i)).collect();
    choose_gumbel_f32_impl(
        &mut raw_logits,
        params,
        token_history,
        seed,
        abs_pos,
        record_entropy_metric,
    )
}

fn choose_gumbel_f32_impl(
    raw_logits: &mut [f32],
    params: &SamplingParams,
    token_history: &[u32],
    seed: u64,
    abs_pos: u64,
    record_entropy_metric: bool,
) -> u32 {
    if raw_logits.is_empty() {
        return 0;
    }
    super::apply_penalties_and_bias(raw_logits, params, token_history);

    if params.temperature <= 0.0 {
        return first_max_index(raw_logits);
    }

    if params.top_n_sigma > 0.0
        && let Some(threshold) = top_n_sigma_threshold(raw_logits, params.top_n_sigma)
    {
        for logit in raw_logits.iter_mut() {
            if *logit < threshold {
                *logit = f32::NEG_INFINITY;
            }
        }
    }

    let eff_temp = params
        .temperature
        .max(super::gumbel::GUMBEL_MIN_TEMPERATURE);
    let logits: Vec<(u32, f32)> = raw_logits
        .iter()
        .enumerate()
        .filter(|(_, v)| v.is_finite())
        .map(|(i, v)| (i as u32, v / eff_temp))
        .collect();
    if logits.is_empty() {
        return first_max_index(raw_logits);
    }

    choose_gumbel_from_scaled(logits, params, seed, abs_pos, record_entropy_metric)
}

/// Shared Gumbel-Max tail over finite temperature-scaled `(token_id, scaled_logit)`
/// candidates in ascending token-ID order. Used by both dense [`choose_gumbel_f32_impl`]
/// and sparse [`choose_gumbel_sparse_impl`] so both paths are bit-identical by construction.
fn choose_gumbel_from_scaled(
    mut logits: Vec<(u32, f32)>,
    params: &SamplingParams,
    seed: u64,
    abs_pos: u64,
    record_entropy_metric: bool,
) -> u32 {
    super::candidate_order::order_candidates(
        &mut logits,
        params.top_k as usize,
        super::candidate_order::CandidateOrder::Total,
    );

    let max_val = logits[0].1;
    let mut kept: Vec<(u32, f32, f32)> = logits
        .iter()
        .map(|&(id, scaled)| (id, scaled, (scaled - max_val).exp()))
        .collect();

    if record_entropy_metric {
        let sum: f32 = kept.iter().map(|c| c.2).sum();
        if sum > 0.0 {
            let inv = 1.0 / sum;
            let h: f32 = kept
                .iter()
                .map(|&(_, _, w)| {
                    let p = w * inv;
                    if p > 1e-10 { -p * p.ln() } else { 0.0 }
                })
                .sum();
            record_entropy(h);
        }
    }

    if params.min_p > 0.0 {
        let threshold = params.min_p * kept[0].2;
        kept.retain(|c| c.2 >= threshold);
    }

    if params.top_p < 1.0 && !kept.is_empty() {
        let sum: f32 = kept.iter().map(|c| c.2).sum();
        if sum > 0.0 {
            let mut cumsum = 0.0f32;
            let mut cutoff = kept.len();
            for (i, c) in kept.iter().enumerate() {
                cumsum += c.2 / sum;
                if cumsum >= params.top_p {
                    cutoff = i + 1;
                    break;
                }
            }
            kept.truncate(cutoff);
        }
    }

    if kept.is_empty() {
        return logits[0].0;
    }

    let mut best_token = kept[0].0;
    let mut best_score = f64::NEG_INFINITY;
    for &(id, scaled, _) in &kept {
        let score = f64::from(scaled) + super::gumbel::gumbel_f64(seed, abs_pos, id);
        if score > best_score {
            best_score = score;
            best_token = id;
        }
    }
    best_token
}

/// Apply the complete Atlas request sampler policy without drawing a token.
///
/// This is the single source of truth for both ordinary decoding and a
/// speculative verifier that needs the target probability `p`: penalties,
/// bias, top-n-sigma, temperature, top-k, min-p and top-p are all included in
/// exactly the same order as the token sampler.
pub fn prepare_sampling_distribution_with_history(
    data: &[u8],
    params: &SamplingParams,
    token_history: &[u32],
) -> PreparedSamplingDistribution {
    // A verifier may prepare several counterfactual target rows for one
    // emitted token.  They must not overwrite `last_entropy`, which describes
    // the distribution the ordinary serving sampler actually consumed.
    prepare_sampling_distribution_impl(
        data,
        params,
        token_history,
        false,
        super::device_topk::canonical_ties_enabled(),
    )
}

fn prepare_sampling_distribution_impl(
    data: &[u8],
    params: &SamplingParams,
    token_history: &[u32],
    record_entropy_metric: bool,
    canonical_ties: bool,
) -> PreparedSamplingDistribution {
    let order = if canonical_ties {
        super::candidate_order::CandidateOrder::Total
    } else {
        super::candidate_order::candidate_order()
    };
    prepare_sampling_distribution_ordered(data, params, token_history, record_entropy_metric, order)
}

pub(super) fn prepare_sampling_distribution_ordered(
    data: &[u8],
    params: &SamplingParams,
    token_history: &[u32],
    record_entropy_metric: bool,
    order: super::candidate_order::CandidateOrder,
) -> PreparedSamplingDistribution {
    let n = data.len() / 4;
    let top_n_sigma = params.top_n_sigma;

    // Read raw logits into a mutable vec for in-place modifications.
    // Penalties (repetition / presence / frequency / LZ / DRY) and
    // logit_bias are applied to `raw_logits` BEFORE the greedy bypass
    // below, so they take effect even at `temperature == 0.0`. Atlas
    // previously short-circuited to `argmax(raw_logits)` for greedy,
    // silently dropping caller-configured penalties — the 2026-05-01
    // sweep showed this caused Gemma-4-31B's haiku to enter a
    // repetition loop ("la... la... laaaL!") even with the model's
    // configured `repetition_penalty=1.1` because the harness uses
    // `temperature=0`. HF Transformers, vLLM, and llama.cpp all run
    // LogitsProcessor (penalties + bias) before greedy argmax — Atlas
    // is the outlier here.
    let mut raw_logits: Vec<f32> = (0..n).map(|i| read_f32(data, i)).collect();
    super::apply_penalties_and_bias(&mut raw_logits, params, token_history);

    // ── Greedy bypass (post-penalty argmax) ──
    // At `temperature == 0.0` we return argmax of the penalty/bias-modified
    // logits. We bypass top_n_sigma, temperature scaling, top_k, top_p,
    // min_p — all of those either filter (set values to -inf) or apply
    // monotonic transforms, neither of which can re-order the maximum. Only
    // penalties + logit_bias actually re-order logits, so as long as those
    // ran first, this argmax is correct AND respects caller config.
    //
    // Ties keep the FIRST maximal id, like the device argmax and the verify
    // path (`argmax_first_wins_f32` is the SSOT); `Iterator::max_by` would
    // keep the LAST and fork plain decode from speculative verify on exact
    // BF16 ties.
    if params.temperature <= 0.0 {
        return PreparedSamplingDistribution::one_hot(n, first_max_index(&raw_logits));
    }
    let temperature = params.temperature;

    // ── 1. Top-n-sigma: filter noise in logit space (temperature-invariant) ──
    // Keep tokens with logit >= max - n*sigma (Tang et al., arXiv:2411.07641; vLLM).
    if top_n_sigma > 0.0
        && let Some(threshold) = top_n_sigma_threshold(&raw_logits, top_n_sigma)
    {
        for logit in raw_logits.iter_mut() {
            if *logit < threshold {
                *logit = f32::NEG_INFINITY;
            }
        }
    }

    // ── 2. Temperature scaling ──
    let logits: Vec<(u32, f32)> = raw_logits
        .iter()
        .enumerate()
        .filter(|(_, v)| v.is_finite()) // Skip -inf tokens from top-n-sigma
        .map(|(i, v)| (i as u32, v / temperature))
        .collect();

    if logits.is_empty() {
        // Fallback: if top-n-sigma filtered everything, use argmax of original
        return PreparedSamplingDistribution::one_hot(n, first_max_index(&raw_logits));
    }

    finish_distribution_from_scaled(n, logits, params, record_entropy_metric, order)
}

/// Shared softmax / entropy / min-p / top-p tail over finite temperature-scaled
/// `(token_id, scaled_logit)` candidates in ascending token-ID order.
fn finish_distribution_from_scaled(
    vocab_size: usize,
    mut logits: Vec<(u32, f32)>,
    params: &SamplingParams,
    record_entropy_metric: bool,
    order: super::candidate_order::CandidateOrder,
) -> PreparedSamplingDistribution {
    let top_k = params.top_k as usize;
    let top_p = params.top_p;
    let min_p = params.min_p;

    // ── 3. Sort descending + top-k ──
    // Default (`CandidateOrder::Total`): total order (logit desc, id asc) with
    // a partial top-k select. Opt-in `FLUX_SAMPLER_TIE_ORDER=legacy` restores
    // the historical full unstable value-only sort; see `candidate_order`.
    super::candidate_order::order_candidates(&mut logits, top_k, order);

    // ── 4. Softmax ──
    let max_val = logits[0].1;
    let mut probs: Vec<(u32, f32)> = logits
        .iter()
        .map(|&(idx, logit)| (idx, (logit - max_val).exp()))
        .collect();

    // ── 4b. Entropy: H = -Σ p·ln(p) over the post-softmax distribution ──
    if record_entropy_metric {
        let sum: f32 = probs.iter().map(|p| p.1).sum();
        if sum > 0.0 {
            let inv = 1.0 / sum;
            let h: f32 = probs
                .iter()
                .map(|&(_, w)| {
                    let p = w * inv;
                    if p > 1e-10 { -p * p.ln() } else { 0.0 }
                })
                .sum();
            record_entropy(h);
        }
    }

    // ── 5. Min-p: keep tokens with prob >= min_p * max_prob ──
    if min_p > 0.0 {
        let max_prob = probs[0].1; // Already sorted descending
        let threshold = min_p * max_prob;
        probs.retain(|p| p.1 >= threshold);
    }

    // ── 6. Top-p (nucleus) ──
    if top_p < 1.0 {
        let sum: f32 = probs.iter().map(|p| p.1).sum();
        let mut cumsum = 0.0f32;
        let mut cutoff = probs.len();
        for (i, &(_, prob)) in probs.iter().enumerate() {
            cumsum += prob / sum;
            if cumsum >= top_p {
                cutoff = i + 1;
                break;
            }
        }
        probs.truncate(cutoff);
    }

    PreparedSamplingDistribution::from_ordered_weights(vocab_size, probs)
}

/// Apply repetition / presence / frequency / LZ / DRY penalties and per-token
/// logit bias directly to a sparse row (`entries` strictly ascending by token ID;
/// every absent vocab index is implicitly `-inf`).
///
/// Exact O(m) counterpart to [`super::apply_penalties_and_bias`]: finite penalties
/// and additive biases leave absent `-inf` entries unchanged (`-inf`), while every
/// present entry receives the exact same arithmetic in the exact same order.
fn apply_penalties_and_bias_sparse(
    entries: &mut [(u32, f32)],
    params: &SamplingParams,
    token_history: &[u32],
) {
    if entries.is_empty() {
        return;
    }
    let min_id = entries[0].0;
    let max_id = entries[entries.len() - 1].0;

    // ── 0. Windowed repetition penalty (once per distinct token) ──
    let rep_penalty = params.repetition_penalty;
    if rep_penalty != 1.0 && rep_penalty > 0.0 && !token_history.is_empty() {
        let window = params.repetition_penalty_window as usize;
        let effective = if window > 0 && window < token_history.len() {
            &token_history[token_history.len() - window..]
        } else {
            token_history
        };
        let mut seen = vec![false; entries.len()];
        for &tid in effective {
            if (min_id..=max_id).contains(&tid)
                && let Ok(idx) = entries.binary_search_by_key(&tid, |&(id, _)| id)
            {
                seen[idx] = true;
            }
        }
        for (entry, was_seen) in entries.iter_mut().zip(seen) {
            if was_seen {
                if entry.1 > 0.0 {
                    entry.1 /= rep_penalty;
                } else {
                    entry.1 *= rep_penalty;
                }
            }
        }
    }

    // ── 0b. OpenAI-style additive penalties (presence + frequency) ──
    let freq_pen = params.frequency_penalty;
    let pres_pen = params.presence_penalty;
    if (freq_pen != 0.0 || pres_pen != 0.0) && !token_history.is_empty() {
        let window = params.repetition_penalty_window as usize;
        let effective = if window > 0 && window < token_history.len() {
            &token_history[token_history.len() - window..]
        } else {
            token_history
        };
        let mut counts = vec![0u32; entries.len()];
        for &tid in effective {
            if (min_id..=max_id).contains(&tid)
                && let Ok(idx) = entries.binary_search_by_key(&tid, |&(id, _)| id)
            {
                counts[idx] += 1;
            }
        }
        for (entry, count) in entries.iter_mut().zip(counts) {
            if count > 0 {
                entry.1 -= freq_pen * count as f32 + pres_pen;
            }
        }
    }

    // ── 0c. LZ penalty (3/4/5-gram extension within last 256 tokens) ──
    if params.lz_penalty > 0.0 && token_history.len() >= 4 {
        const LZ_WINDOW: usize = 256;
        let history = if token_history.len() > LZ_WINDOW {
            &token_history[token_history.len() - LZ_WINDOW..]
        } else {
            token_history
        };
        for &mut (candidate, ref mut logit) in entries.iter_mut() {
            if !history.contains(&candidate) {
                continue;
            }
            for ngram_len in 3..=5usize {
                if history.len() < ngram_len {
                    continue;
                }
                let suffix = &history[history.len() - (ngram_len - 1)..];
                let count = history
                    .windows(ngram_len)
                    .filter(|w| w[..ngram_len - 1] == *suffix && w[ngram_len - 1] == candidate)
                    .count();
                if count > 0 {
                    *logit -= params.lz_penalty * (ngram_len as f32 - 2.0) * count as f32;
                }
            }
        }
    }

    // ── 0d. DRY penalty (exponential suffix-match extension) ──
    if params.dry_multiplier > 0.0 && token_history.len() >= 3 {
        apply_dry_penalty_sparse(
            entries,
            token_history,
            params.dry_multiplier,
            params.dry_base,
            params.dry_allowed_length,
            &params.dry_sequence_breakers,
        );
    }

    // ── 0e. Logit bias: additive per-token bias ──
    for &(tid, bias) in &params.logit_bias {
        if (min_id..=max_id).contains(&tid)
            && let Ok(idx) = entries.binary_search_by_key(&tid, |&(id, _)| id)
        {
            entries[idx].1 += bias;
        }
    }
}

fn apply_dry_penalty_sparse(
    entries: &mut [(u32, f32)],
    history: &[u32],
    multiplier: f32,
    base: f32,
    allowed_length: u32,
    breakers: &[u32],
) {
    if entries.is_empty() || history.is_empty() || multiplier == 0.0 {
        return;
    }
    let min_id = entries[0].0;
    let max_id = entries[entries.len() - 1].0;
    let allowed = allowed_length as usize;
    for (token, len) in super::dry_match_lengths(history, allowed_length, breakers) {
        if (min_id..=max_id).contains(&token)
            && let Ok(idx) = entries.binary_search_by_key(&token, |&(id, _)| id)
        {
            let exp = i32::try_from(len.saturating_sub(allowed)).unwrap_or(i32::MAX);
            let penalty = multiplier * base.powi(exp);
            entries[idx].1 -= penalty;
        }
    }
}

fn top_n_sigma_threshold_sparse(entries: &[(u32, f32)], n: f32) -> Option<f32> {
    let finite = || entries.iter().map(|&(_, v)| v).filter(|x| x.is_finite());
    let count = finite().count();
    if count == 0 {
        return None;
    }
    let max = finite().fold(f32::NEG_INFINITY, f32::max);
    let mean = finite().sum::<f32>() / count as f32;
    let sigma = (finite().map(|x| (x - mean).powi(2)).sum::<f32>() / count as f32).sqrt();
    (sigma > 0.0).then_some(max - n * sigma)
}

/// Exact sparse counterpart to [`first_max_index`] over a vocab-length row whose
/// entries outside `entries` (strictly ascending by token ID) are `-inf`.
fn first_max_index_sparse(entries: &[(u32, f32)], vocab_size: usize) -> u32 {
    if vocab_size == 0 {
        return 0;
    }
    let mut best: Option<(u32, f32)> = None;
    for &(id, v) in entries {
        if v > f32::NEG_INFINITY && best.is_none_or(|(_, b)| v > b) {
            best = Some((id, v));
        }
    }
    if let Some((id, _)) = best {
        return id;
    }
    // No present entry exceeds `-inf`, so the dense maximum is `-inf` at the
    // lowest index in `0..vocab_size` that is not `NaN` (or `0` if every index
    // in the vocabulary is `NaN`).
    let mut candidate_id = 0u32;
    for &(id, v) in entries {
        if id == candidate_id {
            if !v.is_nan() {
                return candidate_id;
            }
            let Some(next) = candidate_id.checked_add(1) else {
                return 0;
            };
            candidate_id = next;
        } else if id > candidate_id {
            return candidate_id;
        }
    }
    if (candidate_id as usize) < vocab_size {
        candidate_id
    } else {
        0
    }
}

/// Direct O(m) Gumbel-Max sampling over sparse candidates (`entries` strictly
/// ascending by token ID; absent vocab entries are implicitly `-inf`).
pub(super) fn choose_gumbel_sparse_impl(
    vocab_size: usize,
    entries: &mut [(u32, f32)],
    params: &SamplingParams,
    token_history: &[u32],
    seed: u64,
    abs_pos: u64,
    record_entropy_metric: bool,
) -> u32 {
    if vocab_size == 0 {
        return 0;
    }
    apply_penalties_and_bias_sparse(entries, params, token_history);

    if params.temperature <= 0.0 {
        return first_max_index_sparse(entries, vocab_size);
    }

    if params.top_n_sigma > 0.0
        && let Some(threshold) = top_n_sigma_threshold_sparse(entries, params.top_n_sigma)
    {
        for (_, logit) in entries.iter_mut() {
            if *logit < threshold {
                *logit = f32::NEG_INFINITY;
            }
        }
    }

    let eff_temp = params
        .temperature
        .max(super::gumbel::GUMBEL_MIN_TEMPERATURE);
    let logits: Vec<(u32, f32)> = entries
        .iter()
        .filter(|(_, v)| v.is_finite())
        .map(|&(id, v)| (id, v / eff_temp))
        .collect();
    if logits.is_empty() {
        return first_max_index_sparse(entries, vocab_size);
    }

    choose_gumbel_from_scaled(logits, params, seed, abs_pos, record_entropy_metric)
}

/// Direct O(m) target-distribution preparation over sparse candidates (`entries`
/// strictly ascending by token ID; absent vocab entries are implicitly `-inf`).
pub(super) fn prepare_sparse_sampling_distribution_ordered(
    vocab_size: usize,
    entries: &mut [(u32, f32)],
    params: &SamplingParams,
    token_history: &[u32],
    record_entropy_metric: bool,
    order: super::candidate_order::CandidateOrder,
) -> PreparedSamplingDistribution {
    apply_penalties_and_bias_sparse(entries, params, token_history);

    if params.temperature <= 0.0 {
        return PreparedSamplingDistribution::one_hot(
            vocab_size,
            first_max_index_sparse(entries, vocab_size),
        );
    }
    let temperature = params.temperature;

    if params.top_n_sigma > 0.0
        && let Some(threshold) = top_n_sigma_threshold_sparse(entries, params.top_n_sigma)
    {
        for (_, logit) in entries.iter_mut() {
            if *logit < threshold {
                *logit = f32::NEG_INFINITY;
            }
        }
    }

    let logits: Vec<(u32, f32)> = entries
        .iter()
        .filter(|(_, v)| v.is_finite())
        .map(|&(id, v)| (id, v / temperature))
        .collect();

    if logits.is_empty() {
        return PreparedSamplingDistribution::one_hot(
            vocab_size,
            first_max_index_sparse(entries, vocab_size),
        );
    }

    finish_distribution_from_scaled(vocab_size, logits, params, record_entropy_metric, order)
}

/// Direct O(m) counterpart to [`sample_seeded_impl`] over sparse candidates.
pub(super) fn sample_sparse_seeded_impl(
    vocab_size: usize,
    entries: &mut [(u32, f32)],
    params: &SamplingParams,
    token_history: &[u32],
    seed: Option<u64>,
    canonical_ties: bool,
) -> u32 {
    if params.temperature > 0.0 && super::gumbel::gumbel_sampling_enabled() {
        let (base_seed, abs_pos) = if let Some(ctx) = super::gumbel::current_gumbel_pos() {
            ctx
        } else {
            let out_pos = token_history.len() as u64;
            let s = seed
                .map(|v| v.wrapping_sub(out_pos))
                .unwrap_or_else(rand::random::<u64>);
            (s, out_pos)
        };
        return choose_gumbel_sparse_impl(
            vocab_size,
            entries,
            params,
            token_history,
            base_seed,
            abs_pos,
            true,
        );
    }
    let order = if canonical_ties {
        super::candidate_order::CandidateOrder::Total
    } else {
        super::candidate_order::candidate_order()
    };
    let distribution = prepare_sparse_sampling_distribution_ordered(
        vocab_size,
        entries,
        params,
        token_history,
        true,
        order,
    );
    if let Some(token) = distribution.deterministic_token {
        return token;
    }
    let random_val: f32 = if let Some(s) = seed {
        sampling_uniform_from_seed(s)
    } else {
        rand::random::<f32>()
    };
    distribution.sample_with_uniform(random_val)
}

/// The top-n-sigma cut: `max - n*sigma` over the finite logits (a bias, grammar mask or
/// suppression leaves -inf entries that would otherwise poison the statistics). `None` when
/// there is nothing to cut: no finite logits, or no spread.
///
/// It used to be `mean - n*sigma`, which only trims the far tail; with the paper's `max`
/// the cut is relative to the leader, which is what makes it temperature-invariant.
pub(super) fn top_n_sigma_threshold(logits: &[f32], n: f32) -> Option<f32> {
    let finite = || logits.iter().copied().filter(|x| x.is_finite());
    let count = finite().count();
    if count == 0 {
        return None;
    }
    let max = finite().fold(f32::NEG_INFINITY, f32::max);
    let mean = finite().sum::<f32>() / count as f32;
    let sigma = (finite().map(|x| (x - mean).powi(2)).sum::<f32>() / count as f32).sqrt();
    (sigma > 0.0).then_some(max - n * sigma)
}

/// Greedy argmax with torch/vLLM tie-breaking: the LOWEST index among equal maxima.
/// (`Iterator::max_by` returns the LAST of equal elements, which flipped exact bf16 ties.)
/// NaN is never selected and never blocks a later value; all-NaN or empty returns 0.
pub fn first_max_index(logits: &[f32]) -> u32 {
    let mut best: Option<(usize, f32)> = None;
    for (i, &v) in logits.iter().enumerate() {
        if v.is_nan() {
            continue;
        }
        if best.is_none_or(|(_, b)| v > b) {
            best = Some((i, v));
        }
    }
    best.map_or(0, |(i, _)| i as u32)
}

#[cfg(test)]
mod deterministic_fast_path_tests {
    use super::*;

    #[test]
    fn greedy_preparation_keeps_target_one_hot_without_a_weight_allocation() {
        let logits: Vec<u8> = [0.5f32, 1.7, 0.3, 1.2]
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        let mut params = SamplingParams::greedy(10);
        params.repetition_penalty = 1.5;

        let prepared = prepare_sampling_distribution_impl(&logits, &params, &[1], false, false);
        assert_eq!(prepared.deterministic_token, Some(3));
        assert!(prepared.ordered_weights.is_empty());
        assert_eq!(prepared.ordered_weights.capacity(), 0);
        assert_eq!(
            prepared.normalized_dense_probabilities().unwrap(),
            vec![0.0, 0.0, 0.0, 1.0]
        );
    }

    #[test]
    fn ordinary_deterministic_return_precedes_both_rng_paths() {
        let source = include_str!("sample_impl.rs");
        let body = source
            .split("pub fn sample_with_params_seeded(")
            .nth(1)
            .unwrap()
            .split("/// Return the exact first")
            .next()
            .unwrap();
        let deterministic = body.find("distribution.deterministic_token").unwrap();
        let seeded_rng = body.find("sampling_uniform_from_seed").unwrap();
        let unseeded_rng = body.find("rand::random").unwrap();
        assert!(deterministic < seeded_rng && deterministic < unseeded_rng);
    }
}
