// SPDX-License-Identifier: AGPL-3.0-only

use super::*;
use crate::sampler::{
    SamplingParams, choose_gumbel, first_max_index, prepare_sampling_distribution_with_history,
};

fn f32_bytes(vals: &[f32]) -> Vec<u8> {
    vals.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn bf16_bytes(vals: &[f32]) -> Vec<u8> {
    vals.iter()
        .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
        .collect()
}

#[test]
fn splitmix64_and_key_bits_match_exact_specification_formula() {
    let seed = 0x1234_5678_9ABC_DEF0u64;
    let pos = 42u64;
    let tok = 73_727u32;

    let mix = |mut x: u64| {
        x ^= x >> 30;
        x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
        x ^= x >> 27;
        x = x.wrapping_mul(0x94D0_49BB_1331_11EB);
        x ^ (x >> 31)
    };
    let expected_base = mix(
        mix(seed.wrapping_add(0x9E37_79B9_7F4A_7C15)) ^ pos.wrapping_mul(0xD1B5_4A32_D192_ED03)
    );
    let expected_bits = mix(expected_base ^ u64::from(tok));

    assert_eq!(position_base(seed, pos), expected_base);
    assert_eq!(key_bits(seed, pos, tok), expected_bits);
    assert_ne!(key_bits(seed, pos, tok), key_bits(seed, pos, tok + 1));
    assert_ne!(key_bits(seed, pos, tok), key_bits(seed, pos + 1, tok));
    assert_ne!(key_bits(seed, pos, tok), key_bits(seed + 1, pos, tok));
}

#[test]
fn gumbel_f64_and_f32_are_finite_and_coupled_via_common_random_numbers() {
    // Euler-Mascheroni constant gamma ~ 0.5772156649, Var(Gumbel(0,1)) = pi^2/6 ~ 1.644934.
    const SAMPLES: u32 = 50_000;
    let mut sum64 = 0.0f64;
    let mut sq64 = 0.0f64;
    let mut sum_abs_diff = 0.0f64;
    let mut max_rel_diff = 0.0f64;

    for tid in 0..SAMPLES {
        let g64 = gumbel_f64(99, 7, tid);
        let g32 = gumbel_f32(99, 7, tid);
        assert!(g64.is_finite(), "non-finite g64 at tid={tid}");
        assert!(g32.is_finite(), "non-finite g32 at tid={tid}");
        sum64 += g64;
        sq64 += g64 * g64;
        let diff = (g64 - f64::from(g32)).abs();
        sum_abs_diff += diff;
        let rel = diff / (1.0 + g64.abs());
        if rel > max_rel_diff {
            max_rel_diff = rel;
        }
    }

    let mean = sum64 / f64::from(SAMPLES);
    let var = sq64 / f64::from(SAMPLES) - mean * mean;
    let mean_abs_diff = sum_abs_diff / f64::from(SAMPLES);
    assert!(
        (mean - 0.577_215_664_9).abs() < 0.02,
        "unexpected Gumbel mean: {mean}"
    );
    assert!(
        (var - std::f64::consts::PI.powi(2) / 6.0).abs() < 0.05,
        "unexpected Gumbel variance: {var}"
    );
    // Because u24 uses the top 24 bits of u53, f32 and f64 share the same Gumbel spike!
    assert!(
        mean_abs_diff < 1.0e-4 && max_rel_diff < 0.01,
        "CRN f32/f64 Gumbel coupling drifted: mean_abs_diff={mean_abs_diff} max_rel_diff={max_rel_diff}"
    );
}

#[test]
fn gumbel_max_samples_from_the_exact_post_filter_categorical_distribution() {
    let logits = [2.1f32, 1.6, 1.1, 0.4, -0.5, -4.0, 0.9, 1.3];
    let bytes = f32_bytes(&logits);
    let mut params = SamplingParams::greedy(0);
    params.temperature = 0.85;
    params.top_k = 5;
    params.top_p = 0.90;
    params.min_p = 0.08;

    let target = prepare_sampling_distribution_with_history(&bytes, &params, &[])
        .normalized_dense_probabilities()
        .expect("valid target distribution");

    const N: u64 = 80_000;
    let mut counts = vec![0u64; logits.len()];
    for pos in 0..N {
        let tok = choose_gumbel(&bytes, &params, &[], 0xC0FFEE, pos);
        counts[tok as usize] += 1;
    }

    for (i, &p) in target.iter().enumerate() {
        if p == 0.0 {
            assert_eq!(counts[i], 0, "filtered token {i} must never be sampled");
        }
    }
    let chi2: f64 = target
        .iter()
        .zip(&counts)
        .filter(|(p, _)| **p > 0.0)
        .map(|(&p, &c)| {
            let expected = f64::from(p) * N as f64;
            (c as f64 - expected).powi(2) / expected
        })
        .sum();
    // Critical value for df <= 5 at alpha = 1e-3 is 20.52.
    assert!(
        chi2 < 20.52,
        "Gumbel-Max empirical frequencies diverged from target softmax: chi2={chi2} counts={counts:?} p={target:?}"
    );
}

#[test]
fn choose_gumbel_respects_penalties_bias_and_tiny_temperature_floor() {
    let logits = [0.5f32, 2.0, 0.3, 1.5];
    let bytes = f32_bytes(&logits);
    let mut params = SamplingParams::greedy(0);
    params.repetition_penalty = 2.0;

    // Greedy fallback at T=0: token 1 penalized to 1.0, so token 3 (1.5) wins.
    assert_eq!(choose_gumbel(&bytes, &params, &[1], 7, 0), 3);

    // Tiny positive temperature uses max(T, 1e-6) without overflowing to NaN/+inf,
    // and picks the unique post-penalty argmax (token 3).
    params.temperature = 1.0e-12;
    assert_eq!(choose_gumbel(&bytes, &params, &[1], 7, 0), 3);
    assert_eq!(
        first_max_index(&[0.5, 1.0, 0.3, 1.5]),
        3,
        "matches post-penalty argmax"
    );
}

#[test]
fn prompt_seed_is_deterministic_63bit_and_order_sensitive() {
    let a = seed_from_prompt_tokens(&[101, 202, 303]);
    let b = seed_from_prompt_tokens(&[101, 202, 303]);
    let swapped = seed_from_prompt_tokens(&[101, 303, 202]);
    let extended = seed_from_prompt_tokens(&[101, 202, 303, 0]);
    assert_eq!(a, b);
    assert_eq!(a & !PROMPT_SEED_MASK, 0, "top bit must be cleared");
    assert_ne!(a, swapped);
    assert_ne!(a, extended);
}

#[test]
fn shortlist_gumbel_selector_maps_compact_ids_and_couples_with_target() {
    // Shortlist covers 4 compact tokens mapping to global IDs [10, 50, 120, 999].
    let shortlist_ids = [10u32, 50, 120, 999];
    let shortlist_logits = [1.5f32, 1.4, 1.3, -2.0];
    let mut full_logits = vec![f32::NEG_INFINITY; 1024];
    for (&gid, &v) in shortlist_ids.iter().zip(&shortlist_logits) {
        full_logits[gid as usize] = v;
    }
    let spec = GumbelDraftSpec {
        seed: 0xBEEF,
        temperature: 0.9,
        top_k: 3,
        top_p: 0.95,
        min_p: 0.05,
        step0_abs_pos: 100,
    };
    let mut params = SamplingParams::greedy(0);
    params.temperature = spec.temperature;
    params.top_k = spec.top_k;
    params.top_p = spec.top_p;
    params.min_p = spec.min_p;

    let shortlist_bf16 = bf16_bytes(&shortlist_logits);
    let full_f32: Vec<f32> = full_logits
        .iter()
        .map(|&v| {
            if v.is_finite() {
                f32::from_bits(v.to_bits() & 0xFFFF_0000)
            } else {
                v
            }
        })
        .collect();
    let full_bytes = f32_bytes(&full_f32);

    let mut matches = 0usize;
    for step in 0..200u64 {
        let (draft_id, prob) =
            select_gumbel_shortlist_bf16(&shortlist_bf16, Some(&shortlist_ids), spec, 100 + step)
                .expect("shortlist draft");
        assert!(shortlist_ids.contains(&draft_id));
        assert!(prob > 0.0 && prob <= 1.0);
        let target_id = choose_gumbel(&full_bytes, &params, &[], spec.seed, 100 + step);
        if draft_id == target_id {
            matches += 1;
        }
    }
    // Because u24 and u53 are coupled to the exact same SplitMix64 draw on identical BF16
    // logits, draft and target agree > 99.9% of the time even at T = 0.9!
    assert!(
        matches >= 199,
        "CRN shortlist drafter and target agreed on only {matches}/200 steps"
    );
}
