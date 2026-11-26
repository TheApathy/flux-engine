// SPDX-License-Identifier: AGPL-3.0-only

// Module is gated by parent's `#[cfg(test)] mod tests;` declaration —
// no inner `#![cfg(test)]` needed (and nesting them is a duplicated
// attribute under recent rustc).

use super::*;

#[test]
fn test_bf16_to_f32() {
    // BF16 for 1.0: 0x3F80 → f32 bits 0x3F800000 = 1.0
    assert_eq!(bf16_to_f32(0x80, 0x3F), 1.0);
    // BF16 for -1.0: 0xBF80 → f32 bits 0xBF800000 = -1.0
    assert_eq!(bf16_to_f32(0x80, 0xBF), -1.0);
    // BF16 for 0.0: 0x0000
    assert_eq!(bf16_to_f32(0x00, 0x00), 0.0);
}

#[test]
fn test_argmax_bf16() {
    // 3 values: 1.0, 2.0, 0.5
    // BF16(1.0) = 0x3F80, BF16(2.0) = 0x4000, BF16(0.5) = 0x3F00
    let data: Vec<u8> = vec![
        0x80, 0x3F, // 1.0
        0x00, 0x40, // 2.0
        0x00, 0x3F, // 0.5
    ];
    assert_eq!(argmax_bf16(&data), 1);
}

#[test]
fn test_argmax_negative() {
    // Values: -1.0, -0.5, -2.0 → argmax should be index 1 (-0.5)
    let data: Vec<u8> = vec![
        0x80, 0xBF, // -1.0
        0x00, 0xBF, // -0.5
        0x00, 0xC0, // -2.0
    ];
    assert_eq!(argmax_bf16(&data), 1);
}

#[test]
fn test_greedy_params() {
    let params = SamplingParams::greedy(100);
    assert!(params.is_greedy());
    assert_eq!(params.max_tokens, 100);
    assert!(params.stop_token_ids.is_empty());
}

#[test]
fn test_argmax_f32() {
    // 3 values: 1.0, 2.0, 0.5 as FP32 little-endian
    let data: Vec<u8> = [1.0f32, 2.0f32, 0.5f32]
        .iter()
        .flat_map(|f| f.to_le_bytes())
        .collect();
    assert_eq!(argmax_f32(&data), 1);
}

#[test]
fn test_sampler_with_mock() {
    use crate::gpu::mock::MockGpuBackend;

    let gpu = MockGpuBackend::new();
    let vocab_size = 4;
    let mut sampler = Sampler::new(vocab_size);

    // Sampler reads BF16 from device (2 bytes/element), not FP32.
    // BF16 encoding: upper 2 bytes of the IEEE 754 FP32 representation.
    // BF16(0.5) = 0x3F00, BF16(3.0) = 0x4040, BF16(1.0) = 0x3F80, BF16(2.0) = 0x4000
    let ptr = gpu.alloc(vocab_size * 2).unwrap();
    let logits: Vec<u8> = vec![
        0x00, 0x3F, // 0.5
        0x40, 0x40, // 3.0
        0x80, 0x3F, // 1.0
        0x00, 0x40, // 2.0
    ];
    gpu.copy_h2d(&logits, ptr).unwrap();

    let params = SamplingParams::greedy(10);
    let token = sampler.sample(ptr, &params, &gpu).unwrap();
    assert_eq!(token, 1); // index 1 = BF16(3.0) = max
}

/// The cut is the paper's `max - n*sigma`, not `mean - n*sigma`. On [2, 1, 1, 1, 1]
/// (mean 1.2, sigma 0.4) they keep different sets: max-based keeps only the leader
/// (cut 1.6), the old mean-based formula (the control) keeps everything (cut 0.8).
#[test]
fn test_top_n_sigma_is_relative_to_the_max() {
    let logits_f32 = [2.0f32, 1.0, 1.0, 1.0, 1.0];
    let survivors = |cut: f32| logits_f32.iter().filter(|&&x| x >= cut).count();
    let cut = super::sample_impl::top_n_sigma_threshold(&logits_f32, 1.0).unwrap();
    assert!((cut - 1.6).abs() < 1e-5, "{cut}");
    assert_eq!(survivors(cut), 1);
    let old_mean_cut = 1.2 - 0.4;
    assert_eq!(
        survivors(old_mean_cut),
        5,
        "control: the mean-based cut keeps all five"
    );

    let logits: Vec<u8> = logits_f32.iter().flat_map(|f| f.to_le_bytes()).collect();
    let params = SamplingParams {
        temperature: 1.0,
        top_k: 0,
        top_p: 1.0,
        top_n_sigma: 1.0,
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
        max_tokens: 10,
        stop_token_ids: Vec::new(),
        seed: None,
    };
    // Unfiltered, tokens 1-4 have ~58% between them; with the cut they never appear.
    for seed in 0..500u64 {
        let p = SamplingParams {
            seed: Some(seed),
            ..params.clone()
        };
        assert_eq!(sample_with_params(&logits, &p), 0);
    }
}

/// Masked (-inf) logits don't poison the statistics: the cut is taken over the finite ones.
#[test]
fn test_top_n_sigma_ignores_masked_logits() {
    let masked = [2.0f32, 1.0, 1.0, 1.0, 1.0, f32::NEG_INFINITY];
    let cut = super::sample_impl::top_n_sigma_threshold(&masked, 1.0).unwrap();
    assert!((cut - 1.6).abs() < 1e-5, "{cut}");
    assert_eq!(
        super::sample_impl::top_n_sigma_threshold(&[3.0, 3.0], 1.0),
        None
    );
}

#[test]
fn test_top_n_sigma_disabled_at_zero() {
    // With top_n_sigma=0.0, no filtering should occur.
    // Use moderate logits so softmax gives reasonable probabilities.
    let logits_f32 = [1.0f32, 1.0, 1.0, 1.0, 1.5];
    let logits: Vec<u8> = logits_f32.iter().flat_map(|f| f.to_le_bytes()).collect();
    let params = SamplingParams {
        temperature: 1.0,
        top_k: 0,
        top_p: 1.0,
        top_n_sigma: 0.0, // disabled
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
        max_tokens: 10,
        stop_token_ids: Vec::new(),
        seed: None,
    };
    // P(token 0-3) = exp(1.0)/Z ≈ 0.19 each, P(token 4) = exp(1.5)/Z ≈ 0.24
    // With 500 samples, P(never see token < 4) ≈ 0.24^500 ≈ 0.
    let mut saw_low = false;
    for _ in 0..500 {
        let token = sample_with_params(&logits, &params);
        if token < 4 {
            saw_low = true;
            break;
        }
    }
    assert!(saw_low, "top_n_sigma=0.0 should not filter any tokens");
}

#[test]
fn test_sample_with_params_seeded_temperature_zero_returns_argmax() {
    // Direct call with temperature=0.0 must NOT divide-by-zero.
    // Should return the argmax of raw logits.
    let logits_f32 = [0.5f32, 1.7, 0.3, 1.2];
    let logits: Vec<u8> = logits_f32.iter().flat_map(|f| f.to_le_bytes()).collect();
    let mut params = SamplingParams::greedy(10);
    params.temperature = 0.0; // explicit
    for _ in 0..10 {
        assert_eq!(sample_with_params_seeded(&logits, &params, &[], None), 1);
    }
}

#[test]
fn prepared_distribution_exposes_exact_post_filter_target_p() {
    let logits: Vec<u8> = [2.0f32, 1.0, 0.0, -1.0]
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    let mut params = SamplingParams::greedy(10);
    params.temperature = 1.0;
    params.top_k = 3;
    params.top_p = 0.8;
    params.min_p = 0.2;

    let prepared = prepare_sampling_distribution_with_history(&logits, &params, &[]);
    let dense = prepared
        .normalized_dense_probabilities()
        .expect("valid target probability row");
    assert_eq!(prepared.vocab_size(), 4);
    assert!((dense.iter().sum::<f32>() - 1.0).abs() < 1.0e-6);
    assert!((dense[0] - 0.731_058_6).abs() < 1.0e-6);
    assert!((dense[1] - 0.268_941_43).abs() < 1.0e-6);
    assert_eq!(dense[2], 0.0);
    assert_eq!(dense[3], 0.0);
}

#[test]
fn prepared_greedy_distribution_applies_history_penalties() {
    let logits: Vec<u8> = [0.5f32, 1.7, 0.3, 1.2]
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    let mut params = SamplingParams::greedy(10);
    params.repetition_penalty = 1.5;

    let dense = prepare_sampling_distribution_with_history(&logits, &params, &[1])
        .normalized_dense_probabilities()
        .expect("greedy one-hot target probability row");
    assert_eq!(dense, vec![0.0, 0.0, 0.0, 1.0]);
}

#[test]
fn prepared_distribution_fails_closed_on_degenerate_filter_settings() {
    let logits: Vec<u8> = [2.0f32, 1.0]
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    let mut params = SamplingParams::greedy(10);
    params.temperature = 1.0;
    params.min_p = 2.0;

    let prepared = prepare_sampling_distribution_with_history(&logits, &params, &[]);
    assert!(prepared.normalized_dense_probabilities().is_none());
    // Preserve the historical serving fallback for invalid external input;
    // only the speculative probability consumer is required to reject it.
    assert_eq!(sample_with_params_seeded(&logits, &params, &[], Some(7)), 0);
}

#[test]
fn prepared_full_vocab_distribution_does_not_move_rounding_mass_to_argmax() {
    const QWEN_VOCAB: usize = 151_936;
    let logits = vec![0u8; QWEN_VOCAB * std::mem::size_of::<f32>()];
    let mut params = SamplingParams::greedy(10);
    params.temperature = 1.0;
    params.top_k = 0;

    let dense = prepare_sampling_distribution_with_history(&logits, &params, &[])
        .normalized_dense_probabilities()
        .expect("valid full-vocabulary target probability row");
    let expected = 1.0 / QWEN_VOCAB as f32;
    assert_eq!(dense.len(), QWEN_VOCAB);
    assert!(dense.iter().all(|&probability| probability == expected));
    assert!(
        (dense
            .iter()
            .map(|&probability| f64::from(probability))
            .sum::<f64>()
            - 1.0)
            .abs()
            < 1.0e-6
    );
}

fn frozen_neutral_sampler(logits: &[f32], params: &SamplingParams, seed: u64) -> u32 {
    use rand::{Rng, SeedableRng};

    if params.temperature <= 0.0 {
        return first_max_index(logits);
    }
    let mut ranked: Vec<(u32, f32)> = logits
        .iter()
        .enumerate()
        .filter(|(_, value)| value.is_finite())
        .map(|(token, value)| (token as u32, value / params.temperature))
        .collect();
    ranked.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let top_k = params.top_k as usize;
    if top_k > 0 && top_k < ranked.len() {
        ranked.truncate(top_k);
    }
    let max = ranked[0].1;
    let mut weights: Vec<(u32, f32)> = ranked
        .iter()
        .map(|&(token, logit)| (token, (logit - max).exp()))
        .collect();
    if params.min_p > 0.0 {
        let threshold = params.min_p * weights[0].1;
        weights.retain(|entry| entry.1 >= threshold);
    }
    if params.top_p < 1.0 {
        let sum: f32 = weights.iter().map(|entry| entry.1).sum();
        let mut cumulative = 0.0f32;
        let mut cutoff = weights.len();
        for (index, &(_, weight)) in weights.iter().enumerate() {
            cumulative += weight / sum;
            if cumulative >= params.top_p {
                cutoff = index + 1;
                break;
            }
        }
        weights.truncate(cutoff);
    }
    let sum: f32 = weights.iter().map(|entry| entry.1).sum();
    let uniform = rand::rngs::StdRng::seed_from_u64(seed).r#gen::<f32>();
    let threshold = uniform * sum;
    let mut cumulative = 0.0f32;
    for &(token, weight) in &weights {
        cumulative += weight;
        if cumulative >= threshold {
            return token;
        }
    }
    weights.last().map_or(0, |entry| entry.0)
}

#[test]
fn prepared_sampler_preserves_seeded_neutral_token_path() {
    let values = [2.0f32, -0.25, 1.0, 1.75, -3.0, 0.5];
    let logits: Vec<u8> = values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    for (top_k, top_p, min_p, temperature) in [
        (0, 1.0, 0.0, 1.0),
        (3, 1.0, 0.0, 0.7),
        (4, 0.8, 0.0, 1.0),
        (0, 0.9, 0.2, 1.3),
    ] {
        let mut params = SamplingParams::greedy(10);
        params.temperature = temperature;
        params.top_k = top_k;
        params.top_p = top_p;
        params.min_p = min_p;
        for seed in 0..256 {
            assert_eq!(
                sample_with_params_seeded(&logits, &params, &[], Some(seed)),
                frozen_neutral_sampler(&values, &params, seed),
                "seeded path drift at seed={seed}, top_k={top_k}, top_p={top_p}, min_p={min_p}, temperature={temperature}",
            );
        }
    }
}

#[test]
fn dflash_rng_anchor_matches_frozen_ordinary_seeded_draws() {
    use rand::{Rng as _, SeedableRng as _};

    for seed in [0, 1, 7, 42, 0x5eed, u64::MAX] {
        let frozen = rand::rngs::StdRng::seed_from_u64(seed).r#gen::<f32>();
        assert_eq!(sampling_uniform_from_seed(seed), frozen);
        assert_eq!(sampling_uniforms_from_seed(seed, 7)[0], frozen);
    }
}

#[test]
fn test_greedy_with_neutral_history_penalties_returns_raw_argmax() {
    // Qwen3.8 code generation intentionally disables every history-based
    // processor. Even when the winning token occurred repeatedly, neutral
    // settings must leave temperature-zero decoding at the raw target argmax.
    let logits_f32 = [0.5f32, 1.7, 0.3, 1.2];
    let logits: Vec<u8> = logits_f32.iter().flat_map(|f| f.to_le_bytes()).collect();
    let params = SamplingParams::greedy(10);
    let history = vec![1u32, 2, 1, 2, 1, 2, 1];

    assert_eq!(argmax_f32(&logits), 1);
    assert_eq!(
        sample_with_params_seeded(&logits, &params, &history, None),
        1,
        "neutral repetition/presence/frequency/LZ/DRY settings must preserve raw argmax"
    );
}

#[test]
fn test_greedy_applies_repetition_penalty_before_argmax() {
    // Regression for 2026-05-01 Gemma-4-31B greedy creative collapse.
    // At temperature=0, repetition_penalty MUST shift argmax when the
    // previous-argmax token is in history. Before the fix, greedy
    // bypassed all penalty processing → infinite repetition loops on
    // models with `repetition_penalty=1.1` configured in MODEL.toml.
    //
    // Setup: token 1 has highest raw logit (1.7). With rep_penalty=1.5
    // applied to history [1, 1] (token 1 twice), its logit becomes
    // 1.7 / 1.5 = 1.133, which drops below token 3's 1.2. Argmax
    // should flip from 1 → 3.
    let logits_f32 = [0.5f32, 1.7, 0.3, 1.2];
    let logits: Vec<u8> = logits_f32.iter().flat_map(|f| f.to_le_bytes()).collect();
    let mut params = SamplingParams::greedy(10);
    params.temperature = 0.0;
    params.repetition_penalty = 1.5;
    let history = vec![1u32, 1u32];
    let token = sample_with_params_seeded(&logits, &params, &history, None);
    assert_eq!(
        token, 3,
        "rep_penalty must shift greedy argmax away from history-repeated token"
    );

    // Sanity: same prompt without history → original argmax (token 1).
    let token_no_hist = sample_with_params_seeded(&logits, &params, &[], None);
    assert_eq!(token_no_hist, 1, "no history → no penalty → raw argmax");

    // Sanity: rep_penalty=1.0 (default) → original argmax even with history.
    params.repetition_penalty = 1.0;
    let token_no_pen = sample_with_params_seeded(&logits, &params, &history, None);
    assert_eq!(
        token_no_pen, 1,
        "rep_penalty=1.0 is no-op even with history"
    );
}

#[test]
fn test_repetition_penalty_applies_once_per_distinct_token() {
    // Token 1 appears 3 times in history. One application: 2.0 / 1.5 = 1.33
    // stays above token 3 (1.2), so greedy keeps token 1. Compounding per
    // occurrence (2.0 / 1.5^3 = 0.59) would wrongly flip it to token 3.
    let logits_f32 = [0.5f32, 2.0, 0.3, 1.2];
    let logits: Vec<u8> = logits_f32.iter().flat_map(|f| f.to_le_bytes()).collect();
    let mut params = SamplingParams::greedy(10);
    params.temperature = 0.0;
    params.repetition_penalty = 1.5;
    let history = vec![1u32, 1, 1];
    assert_eq!(
        sample_with_params_seeded(&logits, &params, &history, None),
        1
    );
    assert_eq!(
        sample_with_params_seeded(&logits, &params, &[1u32], None),
        sample_with_params_seeded(&logits, &params, &history, None),
        "repeats in history must not change the penalty"
    );

    // Negative logits likewise move once: -1.0 * 1.5 = -1.5 beats -2.0.
    let neg_f32 = [-2.0f32, -1.0];
    let neg: Vec<u8> = neg_f32.iter().flat_map(|f| f.to_le_bytes()).collect();
    assert_eq!(
        sample_with_params_seeded(&neg, &params, &[1u32, 1, 1], None),
        1
    );
}

#[test]
fn test_neutral_repetition_penalty_leaves_sampling_unchanged() {
    // repetition_penalty == 1.0 (every Flash-Next / Qwen / GLM preset) skips
    // the penalty block entirely: history, repeats and window must not move
    // a single greedy or seeded-sampled pick, nor any logit.
    let logits_f32: Vec<f32> = (0..64)
        .map(|i| ((i * 37 % 64) as f32) * 0.11 - 3.0)
        .collect();
    let logits: Vec<u8> = logits_f32.iter().flat_map(|f| f.to_le_bytes()).collect();
    let histories: [&[u32]; 4] = [&[], &[5], &[5, 5, 5, 5, 9, 9, 63], &[63; 300]];
    for temperature in [0.0f32, 0.7, 1.0] {
        let mut params = SamplingParams::greedy(10);
        params.temperature = temperature;
        params.repetition_penalty = 1.0;
        for seed in 0..32u64 {
            let reference = sample_with_params_seeded(&logits, &params, &[], Some(seed));
            for history in histories {
                assert_eq!(
                    sample_with_params_seeded(&logits, &params, history, Some(seed)),
                    reference,
                    "T={temperature} seed={seed} history={history:?}"
                );
            }
        }
        for history in histories {
            let mut penalised = logits_f32.clone();
            apply_penalties_and_bias(&mut penalised, &params, history);
            assert!(
                penalised
                    .iter()
                    .zip(&logits_f32)
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                "neutral penalties changed a logit (history {history:?})"
            );
        }
    }
}

#[test]
fn test_apply_penalties_and_bias_penalises_once_per_distinct_token() {
    // The in-place SSOT must agree with the sampler: one application per token.
    let mut params = SamplingParams::greedy(10);
    params.repetition_penalty = 1.5;
    let mut once = vec![0.5f32, 2.0, 0.3, -1.0];
    let mut repeated = once.clone();
    apply_penalties_and_bias(&mut once, &params, &[1, 3]);
    apply_penalties_and_bias(&mut repeated, &params, &[1, 1, 1, 3, 3]);
    assert_eq!(once, repeated);
    assert_eq!(once, vec![0.5, 2.0 / 1.5, 0.3, -1.5]);

    let mut single = vec![0.5f32, 2.0, 0.3, 1.2];
    let mut triple = single.clone();
    apply_penalties_and_bias(&mut single, &params, &[1]);
    apply_penalties_and_bias(&mut triple, &params, &[1, 1, 1]);
    assert_eq!(
        single, triple,
        "history [1, 1, 1] must penalise exactly like [1]"
    );
}

#[test]
fn test_greedy_applies_logit_bias_before_argmax() {
    // logit_bias must shift greedy argmax (e.g. for tool-call grammar).
    let logits_f32 = [0.5f32, 1.7, 0.3, 1.2];
    let logits: Vec<u8> = logits_f32.iter().flat_map(|f| f.to_le_bytes()).collect();
    let mut params = SamplingParams::greedy(10);
    params.temperature = 0.0;
    // Bias token 0 by +5.0 → it should become argmax (0.5+5.0 = 5.5 > 1.7)
    params.logit_bias = vec![(0, 5.0)];
    let token = sample_with_params_seeded(&logits, &params, &[], None);
    assert_eq!(token, 0, "logit_bias must shift greedy argmax");
}

#[test]
fn test_sample_with_params_seeded_repetition_penalty_zero_doesnt_div_by_zero() {
    // repetition_penalty=0.0 used to produce inf/0 logits. Now skipped.
    let logits_f32 = [0.5f32, 1.7, 0.3, 1.2];
    let logits: Vec<u8> = logits_f32.iter().flat_map(|f| f.to_le_bytes()).collect();
    let mut params = SamplingParams::greedy(10);
    params.temperature = 1.0;
    params.repetition_penalty = 0.0; // pathological
    // Token 1 was "seen" — under broken code its logit would have been
    // divided by 0, producing inf. With the guard, no penalty applies.
    let history = vec![1u32];
    let token = sample_with_params_seeded(&logits, &params, &history, Some(42));
    // Just assert we got a valid token (no panic, no infinite loop).
    assert!(token < 4);
}

#[test]
fn test_top_n_sigma_filters_extreme_outliers() {
    // Logits: [100.0, -100.0, -100.0, -100.0, -100.0]: max 100, sigma 80, so at
    // n=0.1 the cut is max - 0.1*sigma = 92 and kills tokens 1-4.
    let logits_f32 = [100.0f32, -100.0, -100.0, -100.0, -100.0];
    let logits: Vec<u8> = logits_f32.iter().flat_map(|f| f.to_le_bytes()).collect();
    let params = SamplingParams {
        temperature: 1.0,
        top_k: 0,
        top_p: 1.0,
        top_n_sigma: 0.1, // tight filter
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
        max_tokens: 10,
        stop_token_ids: Vec::new(),
        seed: None,
    };
    // Token 0 (100.0) should always be selected since others are far below threshold
    for _ in 0..50 {
        let token = sample_with_params(&logits, &params);
        assert_eq!(
            token, 0,
            "extreme low-logit tokens should be filtered at tight sigma"
        );
    }
}

#[test]
fn test_greedy_breaks_exact_ties_toward_the_lowest_index() {
    use super::sample_impl::first_max_index;
    assert_eq!(first_max_index(&[1.0, 3.0, 3.0]), 1);
    // Exact bf16 tie seen on DeepSeek-V4.1 (25.375 twice): torch.argmax takes the first.
    assert_eq!(first_max_index(&[0.0, 25.375, 1.0, 25.375]), 1);
    // Through the full greedy path (penalties/bias are no-ops here).
    let logits: Vec<u8> = [1.0f32, 3.0, 3.0]
        .iter()
        .flat_map(|f| f.to_le_bytes())
        .collect();
    let params = SamplingParams::greedy(3);
    assert_eq!(sample_with_params_history(&logits, &params, &[]), 1);
    // NaN is skipped wherever it sits; it never wins and never hides a later max.
    assert_eq!(first_max_index(&[f32::NAN, 1.0, 3.0, 3.0]), 2);
    assert_eq!(first_max_index(&[1.0, 3.0, f32::NAN]), 1);
    assert_eq!(first_max_index(&[f32::NAN, f32::NAN]), 0);
    assert_eq!(first_max_index(&[]), 0);
    // Negative control: the previous max_by selection picks the LAST of the tie.
    let old = [1.0f32, 3.0, 3.0]
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap();
    assert_eq!(old, 2);
}

// ── Atlas greedy tie rule: FIRST maximal id wins, compared numerically ──────
//
// The device reduction (`argmax_bf16.cu`), `argmax_first_wins_f32` and
// torch.argmax all keep the FIRST id of an exact tie. Every host greedy pick
// must agree, or plain decode and speculative verify fork on exact BF16 ties
// (measured on Qwen3.8-27B: " to"/" answer" tied at -0.7212).

fn bf16_bytes(vals: &[f32]) -> Vec<u8> {
    vals.iter()
        .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
        .collect()
}

fn f32_bytes(vals: &[f32]) -> Vec<u8> {
    vals.iter().flat_map(|v| v.to_le_bytes()).collect()
}

#[test]
fn greedy_sampler_breaks_exact_ties_to_the_first_id() {
    let vals = [1.0f32, 3.0, -2.0, 3.0, 0.5];
    let row_f32 = f32_bytes(&vals);
    let params = SamplingParams::greedy(10);
    assert_eq!(sample_with_params(&row_f32, &params), 1);

    let row_bf16 = bf16_bytes(&vals);
    assert_eq!(argmax_bf16_first_wins(&row_bf16, |_| true), Some(1));
}

#[test]
fn argmax_f32_and_bf16_skip_leading_nan_and_respect_greedy_logit_bias() {
    use crate::gpu::mock::MockGpuBackend;

    let f32_with_nan = f32_bytes(&[f32::NAN, -5.0, 2.5, 2.5]);
    assert_eq!(argmax_f32(&f32_with_nan), 2);
    let f32_nan_neginf = f32_bytes(&[f32::NAN, f32::NEG_INFINITY]);
    assert_eq!(argmax_f32(&f32_nan_neginf), 1);

    let bf16_with_nan = bf16_bytes(&[f32::NAN, -5.0, 2.5, 2.5]);
    assert_eq!(argmax_bf16(&bf16_with_nan), 2);

    // Greedy `Sampler::sample` and `sample_batch` must honor `logit_bias`.
    let gpu = MockGpuBackend::new();
    let vocab_size = 4;
    let mut sampler = Sampler::new(vocab_size);
    let row_bf16 = bf16_bytes(&[0.5, 3.0, 1.0, 2.0]);
    let ptr = gpu.alloc(row_bf16.len()).unwrap();
    gpu.copy_h2d(&row_bf16, ptr).unwrap();

    let mut params = SamplingParams::greedy(10);
    params.logit_bias = vec![(0, 5.0)];
    assert_eq!(sampler.sample(ptr, &params, &gpu).unwrap(), 0);
    assert_eq!(
        sampler.sample_batch(ptr, 1, &[&params], &gpu).unwrap(),
        vec![0]
    );
}

#[test]
fn dry_penalty_targets_continuation_excludes_breakers_and_fires_at_allowed_length() {
    // History [1, 2, 3, 4, 1, 2, 3]: trailing suffix [1, 2, 3] (len 3) was followed by 4.
    let mut logits = vec![10.0f32; 8];
    apply_dry_penalty(&mut logits, &[1, 2, 3, 4, 1, 2, 3], 0.8, 2.0, 2, &[]);
    // Token 4 gets penalty 0.8 * 2^(3 - 2) = 1.6; all others untouched.
    assert!((logits[4] - 8.4).abs() < 1e-6, "got {}", logits[4]);
    for (tok, &val) in logits.iter().enumerate() {
        if tok != 4 {
            assert_eq!(val, 10.0, "unexpected penalty on token {tok}");
        }
    }

    // Exact boundary `len == allowed_length` must fire with exponent 0 (penalty = multiplier).
    let mut logits_eq = vec![10.0f32; 8];
    apply_dry_penalty(&mut logits_eq, &[1, 2, 4, 1, 2], 0.8, 2.0, 2, &[]);
    assert!((logits_eq[4] - 9.2).abs() < 1e-6, "got {}", logits_eq[4]);

    // Sequence breaker inside the matching suffix must stop counting BEFORE incrementing len.
    let mut logits_brk = vec![10.0f32; 8];
    apply_dry_penalty(&mut logits_brk, &[1, 7, 3, 4, 1, 7, 3], 0.8, 2.0, 2, &[7]);
    assert_eq!(
        logits_brk[4], 10.0,
        "breaker 7 must limit match length to 1 (< allowed 2)"
    );

    // Repeated occurrences apply max match length ONCE per candidate token.
    let mut logits_dup = vec![10.0f32; 8];
    apply_dry_penalty(&mut logits_dup, &[1, 2, 4, 1, 2, 4, 1, 2], 0.8, 2.0, 2, &[]);
    // Longest match before `4` is `[1, 2, 4, 1, 2]` (len 5) -> 0.8 * 2^(5-2) = 6.4 (applied once).
    assert!((logits_dup[4] - 3.6).abs() < 1e-6, "got {}", logits_dup[4]);
}

#[test]
fn argmax_bf16_first_wins_orders_negative_logits_numerically() {
    // Every logit negative: -1.0 must beat -2.0 even though -2.0's
    // sign-magnitude bits compare larger as i16.
    let row = bf16_bytes(&[-8.0, -1.0, -2.0, -8.0]);
    assert_eq!(argmax_bf16_first_wins(&row, |_| true), Some(1));
}

#[test]
fn argmax_bf16_first_wins_keeps_the_first_of_an_exact_tie() {
    let row = bf16_bytes(&[1.0, 3.0, -2.0, 3.0]);
    assert_eq!(argmax_bf16_first_wins(&row, |_| true), Some(1));
}

#[test]
fn argmax_bf16_first_wins_treats_signed_zeros_as_equal() {
    // -0.0 == +0.0 numerically: the first one wins, whichever sign it has.
    let row = bf16_bytes(&[-8.0, -0.0, 0.0, -8.0]);
    assert_eq!(argmax_bf16_first_wins(&row, |_| true), Some(1));
}

#[test]
fn argmax_bf16_first_wins_honours_the_admit_filter_and_skips_nan() {
    let row = bf16_bytes(&[f32::NAN, 5.0, 4.0, 4.0]);
    assert_eq!(argmax_bf16_first_wins(&row, |id| id != 1), Some(2));
    assert_eq!(argmax_bf16_first_wins(&row, |id| id == 0), None);
    assert_eq!(argmax_bf16_first_wins(&row, |_| false), None);
}
