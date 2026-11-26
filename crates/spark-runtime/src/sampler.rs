// SPDX-License-Identifier: AGPL-3.0-only

//! Token sampling strategies.
//!
//! Supports greedy argmax (FP32 and BF16 with first-index-wins tie-breaking),
//! temperature scaling, top-k, top-p (nucleus), top-n-sigma, min-p, logit bias,
//! repetition/presence/frequency/LZ/DRY penalties, Gumbel-Max, and device top-k
//! candidate selection.

use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

use crate::gpu::{DevicePtr, GpuBackend};
use anyhow::Result;

// ── Global entropy tracking ──
// Stores the latest per-token entropy and counts of low-entropy streaks.
// AtomicU32 stores f32 bits for lock-free reads.

static LAST_ENTROPY: AtomicU32 = AtomicU32::new(0);
static LOW_ENTROPY_TOKENS: AtomicU64 = AtomicU64::new(0);
static TOTAL_SAMPLED_TOKENS: AtomicU64 = AtomicU64::new(0);

/// Read the most recent per-token entropy (nats).
pub fn last_entropy() -> f32 {
    f32::from_bits(LAST_ENTROPY.load(Ordering::Relaxed))
}

/// Total tokens with entropy < 0.3 (potential degeneration).
pub fn low_entropy_token_count() -> u64 {
    LOW_ENTROPY_TOKENS.load(Ordering::Relaxed)
}

/// Total tokens sampled (for computing low-entropy ratio).
pub fn total_sampled_token_count() -> u64 {
    TOTAL_SAMPLED_TOKENS.load(Ordering::Relaxed)
}

pub(super) fn record_entropy(entropy: f32) {
    LAST_ENTROPY.store(entropy.to_bits(), Ordering::Relaxed);
    TOTAL_SAMPLED_TOKENS.fetch_add(1, Ordering::Relaxed);
    if entropy < 0.3 {
        LOW_ENTROPY_TOKENS.fetch_add(1, Ordering::Relaxed);
    }
}

/// Sampling parameters for a request.
#[derive(Debug, Clone)]
pub struct SamplingParams {
    /// Temperature (0.0 = greedy).
    pub temperature: f32,
    /// Top-k: keep only the k highest-probability tokens before sampling.
    /// 0 = disabled (use all tokens).
    pub top_k: u32,
    /// Top-p (nucleus): keep smallest set of tokens whose cumulative probability >= p.
    /// 1.0 = disabled.
    pub top_p: f32,
    /// Top-n-sigma: filter tokens in logit space before temperature scaling.
    /// Keep only tokens with logit >= max - n*sigma (Tang et al.). Temperature-invariant.
    /// 0.0 = disabled. Recommended: 1.0 for NVFP4 models.
    pub top_n_sigma: f32,
    /// Min-p: keep tokens with prob >= min_p * max_prob (post-softmax).
    /// 0.0 = disabled. Recommended: 0.05-0.1.
    pub min_p: f32,
    /// Per-token logit bias: (token_id, bias_value) pairs.
    /// Applied additively to raw logits before any filtering.
    pub logit_bias: Vec<(u32, f32)>,
    /// Repetition penalty: multiply logits of previously-seen tokens.
    /// 1.0 = disabled. Recommended: 1.05-1.1.
    pub repetition_penalty: f32,
    /// Repetition penalty window: only consider the last N tokens.
    /// 0 = full history (default). Recommended: 64 for long-form generation.
    pub repetition_penalty_window: u32,
    /// Presence penalty (OpenAI-style): flat additive penalty for each token that
    /// appeared at least once. Range [-2.0, 2.0], 0.0 = disabled.
    pub presence_penalty: f32,
    /// Frequency penalty (OpenAI-style): additive penalty proportional to occurrence
    /// count. Range [-2.0, 2.0], 0.0 = disabled.
    pub frequency_penalty: f32,
    /// LZ penalty: penalize tokens that extend repeated n-gram patterns.
    /// 0.0 = disabled. 1.0 = moderate (default). Based on arXiv:2504.20131.
    pub lz_penalty: f32,
    /// DRY (Don't Repeat Yourself) penalty multiplier. From llama.cpp.
    /// Uses backward suffix matching with exponential penalty.
    /// 0.0 = disabled. Recommended: 0.8.
    pub dry_multiplier: f32,
    /// DRY penalty base for exponential scaling. penalty = multiplier * base^(match_len - allowed_len).
    /// Recommended: 1.75.
    pub dry_base: f32,
    /// DRY minimum match length before penalty applies. Sequences shorter than this are ignored.
    /// Recommended: 2.
    pub dry_allowed_length: u32,
    /// DRY sequence breaker token IDs. Delimiters (newlines, colons, quotes, braces) that
    /// reset sequence tracking. Critical for JSON/tool call output where structural tokens repeat.
    pub dry_sequence_breakers: Vec<u32>,
    /// Maximum tokens to generate.
    pub max_tokens: usize,
    /// Stop token IDs.
    pub stop_token_ids: Vec<u32>,
    /// Seed for deterministic sampling. When Some, the RNG is seeded with this
    /// value for reproducible output. None = non-deterministic (thread_rng).
    pub seed: Option<u64>,
}

impl SamplingParams {
    /// Greedy sampling with a max token limit.
    pub fn greedy(max_tokens: usize) -> Self {
        Self {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            top_n_sigma: 0.0,
            min_p: 0.0,
            logit_bias: Vec::new(),
            repetition_penalty: 1.0,
            repetition_penalty_window: 0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            lz_penalty: 0.0,
            dry_multiplier: 0.0,
            dry_base: 1.75,
            dry_allowed_length: 2,
            dry_sequence_breakers: Vec::new(),
            max_tokens,
            stop_token_ids: Vec::new(),
            seed: None,
        }
    }

    pub fn is_greedy(&self) -> bool {
        self.temperature == 0.0
    }
}

/// Sampler that picks tokens from logits.
pub struct Sampler {
    /// Reusable host buffer for BF16 logits D2H copy.
    logits_host: Vec<u8>,
    /// FP32 expanded logits for accurate sampling.
    logits_f32: Vec<f32>,
    /// Vocab size.
    vocab_size: usize,
}

impl Sampler {
    pub fn new(vocab_size: usize) -> Self {
        let logits_host = vec![0u8; vocab_size * 2]; // BF16 from GPU
        let logits_f32 = vec![0.0f32; vocab_size]; // FP32 for sampling
        Self {
            logits_host,
            logits_f32,
            vocab_size,
        }
    }

    /// Copy BF16 logits from GPU, expand to FP32, return FP32 slice.
    fn fetch_logits_f32(&mut self, logits_ptr: DevicePtr, gpu: &dyn GpuBackend) -> Result<&[f32]> {
        let byte_len = self.vocab_size * 2;
        gpu.copy_d2h(logits_ptr, &mut self.logits_host[..byte_len])?;
        // BF16 → FP32 expansion: full precision for sampling
        for i in 0..self.vocab_size {
            self.logits_f32[i] = bf16_to_f32(self.logits_host[i * 2], self.logits_host[i * 2 + 1]);
        }
        Ok(&self.logits_f32[..self.vocab_size])
    }

    /// Sample a token from logits on the GPU.
    ///
    /// `logits_ptr` points to `[vocab_size]` BF16 values on device.
    /// Reads BF16, expands to FP32, then samples with full precision.
    pub fn sample(
        &mut self,
        logits_ptr: DevicePtr,
        params: &SamplingParams,
        gpu: &dyn GpuBackend,
    ) -> Result<u32> {
        if params.is_greedy() && params.logit_bias.is_empty() {
            // Greedy without logit_bias: BF16 argmax is fine (argmax is robust to BF16 quantization)
            let byte_len = self.vocab_size * 2;
            gpu.copy_d2h(logits_ptr, &mut self.logits_host[..byte_len])?;
            return Ok(argmax_bf16(&self.logits_host[..byte_len]));
        }
        // Stochastic (or greedy with logit_bias): expand to FP32 for accurate sampling
        let f32_logits = self.fetch_logits_f32(logits_ptr, gpu)?;
        // SAFETY: `f32_logits` is a valid, initialized `&[f32]` slice, `u8` has
        // alignment 1, and `std::mem::size_of_val(f32_logits)` is its exact byte length.
        let f32_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                f32_logits.as_ptr().cast::<u8>(),
                std::mem::size_of_val(f32_logits),
            )
        };
        Ok(sample_with_params(f32_bytes, params))
    }

    /// Sample a batch of tokens (one per sequence in the batch).
    ///
    /// `logits_ptr` points to [batch_size, vocab_size] BF16 values.
    pub fn sample_batch(
        &mut self,
        logits_ptr: DevicePtr,
        batch_size: usize,
        params: &[&SamplingParams],
        gpu: &dyn GpuBackend,
    ) -> Result<Vec<u32>> {
        let total_bytes = batch_size * self.vocab_size * 2; // BF16
        if self.logits_host.len() < total_bytes {
            self.logits_host.resize(total_bytes, 0);
        }
        gpu.copy_d2h(logits_ptr, &mut self.logits_host[..total_bytes])?;

        let stride_bf16 = self.vocab_size * 2;
        let mut tokens = Vec::with_capacity(batch_size);
        for i in 0..batch_size {
            let start = i * stride_bf16;
            let end = start + stride_bf16;
            let p = params.get(i).copied().unwrap_or(params[0]);
            tokens.push(if p.is_greedy() && p.logit_bias.is_empty() {
                argmax_bf16(&self.logits_host[start..end])
            } else {
                // Expand BF16 → FP32 for accurate sampling / logit_bias application
                if self.logits_f32.len() < self.vocab_size {
                    self.logits_f32.resize(self.vocab_size, 0.0);
                }
                for j in 0..self.vocab_size {
                    self.logits_f32[j] = bf16_to_f32(
                        self.logits_host[start + j * 2],
                        self.logits_host[start + j * 2 + 1],
                    );
                }
                let f32_slice = &self.logits_f32[..self.vocab_size];
                // SAFETY: `f32_slice` is a valid, initialized `&[f32]` of length
                // `self.vocab_size`, `u8` has alignment 1, and `size_of_val` gives
                // its exact byte length.
                let f32_bytes: &[u8] = unsafe {
                    std::slice::from_raw_parts(
                        f32_slice.as_ptr().cast::<u8>(),
                        std::mem::size_of_val(f32_slice),
                    )
                };
                sample_with_params(f32_bytes, p)
            });
        }
        Ok(tokens)
    }
}

/// LZ penalty: penalize tokens that would extend repeated n-gram patterns
/// in the recent token history. Based on arXiv:2504.20131.
///
/// For each candidate token that appears in the history, check if appending it
/// creates a repeated 3/4/5-gram. Penalize proportional to n-gram length and
/// frequency: `logit -= penalty * (ngram_len - 2) * count`.
pub fn apply_lz_penalty(logits: &mut [f32], history: &[u32], penalty: f32) {
    use std::collections::HashSet;
    // Window the history to last 256 tokens to avoid penalizing
    // cross-turn structural repetition (e.g., JSON keys in tool calls).
    const LZ_WINDOW: usize = 256;
    let history = if history.len() > LZ_WINDOW {
        &history[history.len() - LZ_WINDOW..]
    } else {
        history
    };
    let n = logits.len();
    // Only check tokens that appear in history (others can't form repeats)
    let token_set: HashSet<u32> = history.iter().copied().collect();
    for &candidate in &token_set {
        if (candidate as usize) >= n {
            continue;
        }
        for ngram_len in 3..=5usize {
            if history.len() < ngram_len {
                continue;
            }
            // The n-gram that would form: history[-(ngram_len-1)..] ++ [candidate]
            let suffix = &history[history.len() - (ngram_len - 1)..];
            let count = history
                .windows(ngram_len)
                .filter(|w| w[..ngram_len - 1] == *suffix && w[ngram_len - 1] == candidate)
                .count();
            if count > 0 {
                logits[candidate as usize] -= penalty * (ngram_len as f32 - 2.0) * count as f32;
            }
        }
    }
}

/// Compute the longest suffix match length (`>= allowed_length` and `> 0`) for
/// each candidate continuation token in `history` via an O(n^2) backward scan.
///
/// Taking the maximum match length per candidate ensures repeated historical
/// occurrences apply the DRY penalty once per token (matching llama.cpp DRY).
pub(super) fn dry_match_lengths(
    history: &[u32],
    allowed_length: u32,
    breakers: &[u32],
) -> std::collections::HashMap<u32, usize> {
    let mut max_match_len = std::collections::HashMap::<u32, usize>::new();
    let hist_len = history.len();
    if hist_len < 2 {
        return max_match_len;
    }
    let allowed = allowed_length as usize;
    let last = hist_len - 1;
    // If the trailing token is itself a sequence breaker, no suffix match can
    // start from the end of `history`.
    if breakers.contains(&history[last]) {
        return max_match_len;
    }

    for i in (0..last).rev() {
        // Scan backward from `i` against the trailing suffix ending at `last`.
        // When `len` tokens match, `history[i + 1 - len..=i]` equals
        // `history[hist_len - len..hist_len]`, so `history[i + 1]` is the
        // historical continuation token that would extend the repeat if sampled next.
        let mut len = 0usize;
        let mut j = i;
        let mut k = last;
        while j < k && history[j] == history[k] {
            if breakers.contains(&history[j]) {
                break;
            }
            len += 1;
            if j == 0 {
                break;
            }
            j -= 1;
            k -= 1;
        }
        if len > 0 && len >= allowed {
            let extend_token = history[i + 1];
            max_match_len
                .entry(extend_token)
                .and_modify(|best| {
                    if len > *best {
                        *best = len;
                    }
                })
                .or_insert(len);
        }
    }
    max_match_len
}

/// DRY (Don't Repeat Yourself) penalty. Ported from llama.cpp PR #9702.
///
/// Uses O(n^2) backward suffix matching to find, for each candidate continuation
/// token, the longest repeated sequence ending at the current position in
/// `history` that the token would extend. Applies exponential penalty once per
/// candidate using its maximum match length (`>= allowed_length`):
///   `penalty = multiplier * base^(match_length - allowed_length)`
///
/// Sequence breakers (e.g., newlines, quotes, braces) halt suffix matching
/// without being counted in `match_length`, preventing false positives in
/// structured output like JSON tool calls.
pub fn apply_dry_penalty(
    logits: &mut [f32],
    history: &[u32],
    multiplier: f32,
    base: f32,
    allowed_length: u32,
    breakers: &[u32],
) {
    if history.is_empty() || multiplier == 0.0 {
        return;
    }
    let n = logits.len();
    let allowed = allowed_length as usize;
    for (token, len) in dry_match_lengths(history, allowed_length, breakers) {
        let token_idx = token as usize;
        if token_idx < n {
            let exp = i32::try_from(len.saturating_sub(allowed)).unwrap_or(i32::MAX);
            let penalty = multiplier * base.powi(exp);
            logits[token_idx] -= penalty;
        }
    }
}

/// Apply repetition / presence / frequency / LZ / DRY penalties and
/// per-token logit bias to `logits` IN PLACE, using `token_history`.
///
/// SSOT for the pre-filter logit-modification block. Extracted verbatim
/// from `sample_with_params_seeded` (the non-MTP sampling path) so the
/// MTP verify path (`verify_pick_with_pipeline`) and bootstrap path
/// (`sample_token_with_grammar`) apply the *same* penalties+bias the
/// non-MTP path does — previously those two paths emitted tokens with no
/// penalties (hardcoded `repetition_penalty=1.0`, empty history), so the
/// configured `repetition_penalty`/`dry_multiplier` from MODEL.toml never
/// reached MTP-emitted tokens and the model degenerated into repeated
/// tool-call argument junk.
///
/// BACKWARD-COMPATIBLE / ADDITIVE: a mathematical no-op when
/// `repetition_penalty == 1.0`, `presence_penalty == 0.0`,
/// `frequency_penalty == 0.0`, `lz_penalty <= 0.0`, `dry_multiplier <= 0.0`
/// and `logit_bias` is empty — every branch below is individually gated on
/// its parameter being non-neutral, so the NVFP4 / Gemma / Mistral presets
/// (which use those neutral values) are byte-for-byte unchanged.
pub fn apply_penalties_and_bias(
    logits: &mut [f32],
    params: &SamplingParams,
    token_history: &[u32],
) {
    let n = logits.len();

    // ── 0. Windowed repetition penalty: penalize recently seen tokens ──
    // Window=0 uses full history; window>0 uses only the last N tokens.
    // Skip when rep_penalty <= 0.0 — the divide at the next branch would
    // produce inf for positive logits and 0 for negative, poisoning the
    // distribution. (Caller intent for 0.0 is unclear; treat as no-op.)
    let rep_penalty = params.repetition_penalty;
    if rep_penalty != 1.0 && rep_penalty > 0.0 && !token_history.is_empty() {
        let window = params.repetition_penalty_window as usize;
        let effective = if window > 0 && window < token_history.len() {
            &token_history[token_history.len() - window..]
        } else {
            token_history
        };
        // Once per DISTINCT token, as in `prepare_sampling_distribution_impl`
        // (HF / vLLM / llama.cpp semantics); walking the raw history
        // compounded the penalty per occurrence.
        let mut seen: Vec<u32> = effective.to_vec();
        seen.sort_unstable();
        seen.dedup();
        for tid in seen {
            if (tid as usize) < n {
                let logit = &mut logits[tid as usize];
                if *logit > 0.0 {
                    *logit /= rep_penalty;
                } else {
                    *logit *= rep_penalty;
                }
            }
        }
    }

    // ── 0b. OpenAI-style additive penalties (presence + frequency) ──
    // Presence: z'ⱼ = zⱼ − β (flat, if token appeared at all)
    // Frequency: z'ⱼ = zⱼ − α · cⱼ (proportional to occurrence count)
    let freq_pen = params.frequency_penalty;
    let pres_pen = params.presence_penalty;
    if (freq_pen != 0.0 || pres_pen != 0.0) && !token_history.is_empty() {
        let window = params.repetition_penalty_window as usize;
        let effective = if window > 0 && window < token_history.len() {
            &token_history[token_history.len() - window..]
        } else {
            token_history
        };
        // Count occurrences per token
        let mut counts = std::collections::HashMap::<u32, u32>::new();
        for &tid in effective {
            *counts.entry(tid).or_insert(0) += 1;
        }
        for (&tid, &count) in &counts {
            if (tid as usize) < n {
                logits[tid as usize] -= freq_pen * count as f32 + pres_pen;
            }
        }
    }

    // ── 0c. LZ penalty: penalize tokens that extend repeated n-gram patterns ──
    if params.lz_penalty > 0.0 && token_history.len() >= 4 {
        apply_lz_penalty(logits, token_history, params.lz_penalty);
    }

    // ── 0d. DRY penalty: exponential penalty for extending repeated sequences ──
    if params.dry_multiplier > 0.0 && token_history.len() >= 3 {
        apply_dry_penalty(
            logits,
            token_history,
            params.dry_multiplier,
            params.dry_base,
            params.dry_allowed_length,
            &params.dry_sequence_breakers,
        );
    }

    // ── 0e. Logit bias: additive per-token bias ──
    for &(tid, bias) in &params.logit_bias {
        if (tid as usize) < n {
            logits[tid as usize] += bias;
        }
    }
}

/// Argmax over an f32 slice with the strict-`>` FIRST-index-wins tie-break.
///
/// SSOT for this pick: the verify path (`spark-server`'s
/// `crates/spark-server/src/scheduler/spec_policy_accept.rs`) calls here too. The naive
/// `if v > best { best = v; idx = i }` loop carries a dependency through BOTH
/// the running value and the index, which blocks vectorisation — measured
/// 1.19 ms for 4x248k on the verify path before the two-pass rewrite (5.95x).
///
/// Equivalence to that loop, including the awkward cases: `>` is false for
/// NaN in both passes so NaN never wins (all-NaN => -inf max, pass 2 finds no
/// equal, falls back to 0 — same as the loop); IEEE -0.0 == +0.0 so neither
/// `>` nor `==` separates them and the first zero encountered is returned
/// either way. `f32::max` is deliberately avoided (it returns the non-NaN
/// operand, which would let a NaN-adjacent value win where `>` ignored it).
pub fn argmax_first_wins_f32(v: &[f32]) -> u32 {
    const LANES: usize = 8;
    let mut acc = [f32::NEG_INFINITY; LANES];
    let mut chunks = v.chunks_exact(LANES);
    for c in &mut chunks {
        for (a, &x) in acc.iter_mut().zip(c) {
            if x > *a {
                *a = x;
            }
        }
    }
    let mut best = f32::NEG_INFINITY;
    for &a in acc.iter() {
        if a > best {
            best = a;
        }
    }
    for &x in chunks.remainder() {
        if x > best {
            best = x;
        }
    }
    v.iter()
        .position(|&x| x == best)
        .unwrap_or(0)
        .try_into()
        .unwrap_or(0)
}

/// Greedy pick over a little-endian BF16 logits row, restricted to the ids
/// `admit` accepts. SSOT for every host-side BF16 greedy pick.
///
/// Tie rule: the FIRST maximal id wins — the same rule as the device
/// reduction (`kernels/gb10/common/argmax_bf16.cu`), [`argmax_first_wins_f32`]
/// and torch.argmax. Plain decode and speculative verify must agree on it, or
/// they fork on exact BF16 ties. Values compare as f32 (BF16 is the high half
/// of an f32), so negative logits order numerically — comparing the raw bit
/// patterns as i16 does not: sign-magnitude makes -2.0 look larger than -1.0 —
/// and -0.0 == +0.0. NaN never wins, matching the kernel's `>` scan. `None`
/// when no admitted id carries a non-NaN value.
pub fn argmax_bf16_first_wins(row: &[u8], mut admit: impl FnMut(u32) -> bool) -> Option<u32> {
    let mut best: Option<(u32, f32)> = None;
    for (id, bytes) in row.chunks_exact(2).enumerate() {
        let id = id as u32;
        if !admit(id) {
            continue;
        }
        let v = bf16_to_f32(bytes[0], bytes[1]);
        if !v.is_nan() && best.is_none_or(|(_, best_v)| v > best_v) {
            best = Some((id, v));
        }
    }
    best.map(|(id, _)| id)
}

mod candidate_order;
#[cfg(test)]
mod candidate_order_tests;
pub mod device_topk;
pub mod f2_certify;
pub mod gumbel;
mod sample_impl;

/// Resolve `FLUX_SAMPLER_TIE_ORDER` now and emit its receipt line, so the
/// choice is visible at server start rather than at the first sample.
pub fn sampler_tie_order_receipt() {
    let _ = candidate_order::candidate_order();
}
pub use gumbel::{gumbel_f32, gumbel_f64, key_bits, seed_from_prompt_tokens};
pub use sample_impl::{
    PreparedSamplingDistribution, choose_gumbel, choose_gumbel_f32_in_place, first_max_index,
    prepare_sampling_distribution_with_history, sample_with_params_history,
    sample_with_params_seeded, sampling_uniform_from_seed, sampling_uniforms_from_seed,
};

/// Convenience wrapper: sample without token history (no repetition penalty).
pub fn sample_with_params(data: &[u8], params: &SamplingParams) -> u32 {
    sample_with_params_history(data, params, &[])
}

/// Argmax over FP32 values stored as raw bytes (4 bytes per element, little-endian).
///
/// Skips `NaN` values and breaks exact ties to the lowest token index; returns `0`
/// if empty or all-`NaN`.
pub fn argmax_f32(data: &[u8]) -> u32 {
    debug_assert!(data.len().is_multiple_of(4));
    let mut best: Option<(u32, f32)> = None;
    for (i, chunk) in data.chunks_exact(4).enumerate() {
        let val = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        if !val.is_nan() && best.is_none_or(|(_, best_v)| val > best_v) {
            best = Some((i as u32, val));
        }
    }
    best.map_or(0, |(idx, _)| idx)
}

/// Read an f32 value from a byte slice at element index `i`.
#[inline]
pub(super) fn read_f32(data: &[u8], i: usize) -> f32 {
    let off = i * 4;
    let bytes = [data[off], data[off + 1], data[off + 2], data[off + 3]];
    f32::from_le_bytes(bytes)
}

/// Legacy: argmax over BF16 values (still used by argmax_on_device fallback).
///
/// Skips `NaN` values and breaks exact ties to the lowest token index; returns `0`
/// if empty or all-`NaN`.
pub fn argmax_bf16(data: &[u8]) -> u32 {
    debug_assert!(data.len().is_multiple_of(2));
    argmax_bf16_first_wins(data, |_| true).unwrap_or(0)
}

/// Convert BF16 (2 bytes, little-endian) to f32.
#[inline]
fn bf16_to_f32(lo: u8, hi: u8) -> f32 {
    let bits = (lo as u32) | ((hi as u32) << 8);
    f32::from_bits(bits << 16)
}

#[cfg(test)]
mod tests;
