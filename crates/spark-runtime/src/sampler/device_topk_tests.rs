// SPDX-License-Identifier: AGPL-3.0-only

use super::*;
use crate::sampler::sample_impl::sample_seeded_impl;
use rand::{Rng, SeedableRng, rngs::StdRng};

fn bf16_bits(v: f32) -> u16 {
    (v.to_bits() >> 16) as u16
}

fn row_bytes(vals: &[u16]) -> Vec<u8> {
    vals.iter().flat_map(|b| b.to_le_bytes()).collect()
}

fn expand(vals: &[u16]) -> Vec<f32> {
    vals.iter()
        .map(|&b| f32::from_bits((b as u32) << 16))
        .collect()
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn params(temperature: f32, top_k: u32, top_p: f32, min_p: f32, seed: u64) -> SamplingParams {
    let mut p = SamplingParams::greedy(16);
    p.temperature = temperature;
    p.top_k = top_k;
    p.top_p = top_p;
    p.min_p = min_p;
    p.seed = Some(seed);
    p
}

/// A random BF16 row on a coarse grid (exact ties are common), with a few
/// -inf / +inf / NaN / -0.0 / +0.0 entries mixed in.
fn random_row(rng: &mut StdRng, vocab: usize, levels: u32) -> Vec<u16> {
    (0..vocab)
        .map(|_| match rng.gen_range(0..100) {
            0 => bf16_bits(f32::NEG_INFINITY),
            1 => bf16_bits(f32::INFINITY),
            2 => 0x7FC0, // NaN
            3 => bf16_bits(-0.0),
            4 => bf16_bits(0.0),
            _ => {
                let level = rng.gen_range(0..levels) as f32;
                bf16_bits(level * 0.125 - 4.0)
            }
        })
        .collect()
}

#[test]
fn selection_key_orders_by_value_then_lower_id_and_skips_non_finite() {
    let k = |id, v| selection_key(id, v, 1.0).unwrap();
    assert!(k(5, 2.0) > k(1, 1.0));
    assert!(k(1, -1.0) > k(0, -2.0), "negative values order numerically");
    assert!(k(3, 1.0) > k(9, 1.0), "equal value: lower id wins");
    assert!(k(3, -0.0) > k(9, 0.0), "-0 and +0 are equal values");
    assert!(k(9, -0.0) < k(3, 0.0));
    assert_eq!(selection_key(0, f32::NAN, 1.0), None);
    assert_eq!(selection_key(0, f32::INFINITY, 1.0), None);
    assert_eq!(selection_key(0, f32::NEG_INFINITY, 1.0), None);
}

#[test]
fn selection_key_ranks_by_the_temperature_scaled_value() {
    // Two raw values that collapse to the same f32 after division must tie
    // and fall back to the id order, exactly as the host comparator does.
    let t = 3.0f32;
    let (a, b) = (1.0f32, f32::from_bits(1.0f32.to_bits() + 1));
    if a / t == b / t {
        assert!(selection_key(2, a, t) > selection_key(7, b, t));
    }
}

#[test]
fn reference_matches_a_full_canonical_sort_on_10k_rows() {
    let mut rng = StdRng::seed_from_u64(0x5eed);
    for row_i in 0..10_000 {
        let vocab = rng.gen_range(1..300);
        let vals = random_row(&mut rng, vocab, if row_i % 2 == 0 { 8 } else { 64 });
        let t = [0.6f32, 1.0, 1.3][row_i % 3];
        let m = rng.gen_range(1..=40);
        let got = topk_candidates_reference(&row_bytes(&vals), t, m);
        let mut want: Vec<(u32, f32)> = expand(&vals)
            .into_iter()
            .enumerate()
            .filter(|(_, v)| v.is_finite())
            .map(|(i, v)| (i as u32, v))
            .collect();
        want.sort_by(|a, b| cmp_desc(&(a.0, a.1 / t), &(b.0, b.1 / t), true));
        want.truncate(m);
        let ids = |v: &[(u32, f32)]| v.iter().map(|c| c.0).collect::<Vec<_>>();
        assert_eq!(ids(&got), ids(&want), "row {row_i}");
        for (g, w) in got.iter().zip(&want) {
            assert_eq!(
                g.1.to_bits(),
                w.1.to_bits(),
                "raw value carried bit-exactly"
            );
        }
    }
}

#[test]
fn crafted_ties_select_and_order_by_lower_id() {
    // ids 7, 3, 11 tie at the top; 5 and 2 tie below. m=4 must take 3,7,11,2.
    let mut vals = vec![bf16_bits(-5.0); 16];
    for id in [7, 3, 11] {
        vals[id] = bf16_bits(2.5);
    }
    for id in [5, 2] {
        vals[id] = bf16_bits(1.0);
    }
    let got = topk_candidates_reference(&row_bytes(&vals), 1.0, 4);
    assert_eq!(
        got.iter().map(|c| c.0).collect::<Vec<_>>(),
        vec![3, 7, 11, 2]
    );
}

#[test]
fn crafted_ties_sample_identically_from_the_sparse_row_for_every_seed() {
    let mut vals = vec![bf16_bits(-5.0); 64];
    for id in [40, 9, 33, 12] {
        vals[id] = bf16_bits(2.0);
    }
    vals[50] = bf16_bits(1.5);
    vals[1] = bf16_bits(1.5);
    let full = f32_bytes(&expand(&vals));
    for top_k in [1u32, 2, 3, 5, 6] {
        let m = top_k as usize;
        let cands = topk_candidates_reference(&row_bytes(&vals), 1.0, m);
        let sparse = f32_bytes(&sparse_row_from_candidates(vals.len(), &cands).unwrap());
        for seed in 0..2000 {
            let p = params(1.0, top_k, 1.0, 0.0, seed);
            let a = sample_seeded_impl(&full, &p, &[], p.seed, true);
            let b = sample_seeded_impl(&sparse, &p, &[], p.seed, true);
            assert_eq!(a, b, "top_k={top_k} seed={seed}");
        }
    }
    // Canonical order puts the lowest tied id first: with top_k=1 that is 9.
    let p = params(1.0, 1, 1.0, 0.0, 0);
    assert_eq!(sample_seeded_impl(&full, &p, &[], p.seed, true), 9);
}

/// Decrease-only edits on random ids, applied identically to both rows.
fn apply_edits(row: &mut [f32], edits: &[(usize, u8)]) {
    for &(id, kind) in edits {
        match kind {
            0 => row[id] = f32::NEG_INFINITY,
            1 => row[id] -= 10.0,
            _ => row[id] -= 12.0,
        }
    }
}

#[test]
fn sparse_row_samples_equal_full_row_samples_under_penalties_and_edits() {
    let mut rng = StdRng::seed_from_u64(0xD7);
    let mut checked = 0usize;
    for case in 0..6000 {
        let vocab = rng.gen_range(40..400);
        let vals = random_row(&mut rng, vocab, if case % 2 == 0 { 6 } else { 200 });
        let t = [0.6f32, 1.0, 1.3, 0.7][case % 4];
        let top_k = rng.gen_range(1..=30);
        let top_p = [1.0f32, 0.95, 0.8, 0.5][case % 4];
        let min_p = [0.0f32, 0.0, 0.05, 0.1][(case / 4) % 4];
        let mut p = params(t, top_k, top_p, min_p, rng.r#gen());
        let history: Vec<u32> = (0..rng.gen_range(0..24))
            .map(|_| rng.gen_range(0..vocab as u32))
            .collect();
        match case % 5 {
            1 => p.repetition_penalty = 1.1,
            2 => p.presence_penalty = 0.5,
            3 => {
                p.frequency_penalty = 0.3;
                p.lz_penalty = 1.0;
            }
            4 => p.dry_multiplier = 0.8,
            _ => {}
        }
        let edits: Vec<(usize, u8)> = (0..rng.gen_range(0..6))
            .map(|_| (rng.gen_range(0..vocab), rng.gen_range(0..3u8)))
            .collect();
        let Ok(m) = candidate_budget(&p, &history, edits.len()) else {
            continue;
        };
        let mut full = expand(&vals);
        let cands = topk_candidates_reference(&row_bytes(&vals), t, m);
        let mut sparse = sparse_row_from_candidates(vocab, &cands).unwrap();
        apply_edits(&mut full, &edits);
        apply_edits(&mut sparse, &edits);
        if cands.is_empty() {
            continue; // the scheduler routes an all-non-finite row to the host
        }
        let a = sample_seeded_impl(&f32_bytes(&full), &p, &history, p.seed, true);
        let b = sample_seeded_impl(&f32_bytes(&sparse), &p, &history, p.seed, true);
        assert_eq!(a, b, "case {case}: m={m} top_k={top_k} t={t}");
        checked += 1;
    }
    assert!(checked > 5000, "only {checked} cases exercised");
}

#[test]
fn admit_refuses_disabled_or_oversized_top_k() {
    assert!(admit_top_k(0).is_err());
    assert!(admit_top_k(DEVICE_TOPK_K_MAX as u32 + 1).is_err());
    assert!(admit_top_k(20).is_ok());
    assert!(admit_top_k(DEVICE_TOPK_K_MAX as u32).is_ok());
}

#[test]
fn candidate_budget_routes_anything_that_can_raise_or_read_the_row() {
    let base = params(1.0, 20, 0.95, 0.0, 1);
    assert_eq!(candidate_budget(&base, &[], 0), Ok(20));
    assert_eq!(candidate_budget(&base, &[1, 2, 2], 3), Ok(23));
    let reason = |f: &dyn Fn(&mut SamplingParams)| {
        let mut p = base.clone();
        f(&mut p);
        candidate_budget(&p, &[1, 2, 3, 4], 0).unwrap_err()
    };
    assert_eq!(reason(&|p| p.temperature = 0.0), "greedy");
    assert_eq!(reason(&|p| p.top_k = 0), "top_k");
    assert_eq!(reason(&|p| p.top_n_sigma = 1.0), "top_n_sigma");
    assert_eq!(reason(&|p| p.logit_bias = vec![(1, 0.5)]), "logit_bias");
    assert_eq!(
        reason(&|p| p.repetition_penalty = 0.9),
        "repetition_penalty_below_one"
    );
    assert_eq!(
        reason(&|p| p.presence_penalty = -0.5),
        "negative_additive_penalty"
    );
    let mut p = base.clone();
    p.repetition_penalty = 1.1;
    assert_eq!(
        candidate_budget(&p, &[4, 4, 9], 0),
        Ok(22),
        "distinct history ids"
    );
    assert_eq!(
        candidate_budget(&p, &[], DEVICE_TOPK_M_MAX),
        Err("edit_budget")
    );
}

#[test]
fn sparse_row_rejects_out_of_range_ids() {
    assert!(sparse_row_from_candidates(4, &[(4, 1.0)]).is_none());
    let row = sparse_row_from_candidates(4, &[(2, 1.5)]).unwrap();
    assert_eq!(row[2], 1.5);
    assert!(row[0] == f32::NEG_INFINITY && row[3] == f32::NEG_INFINITY);
}

#[test]
fn production_preset_windows_the_history_and_stays_on_the_device() {
    // Flash-Next default thinking preset: presence 1.5, window 256, top_k 20,
    // 12 reflection ids + 4 fixed slots; a long history of distinct tokens.
    let mut p = params(1.0, 20, 0.95, 0.0, 1);
    p.presence_penalty = 1.5;
    p.repetition_penalty_window = 256;
    let history: Vec<u32> = (0..16_000u32).collect();
    assert_eq!(candidate_budget(&p, &history, 16), Ok(20 + 256 + 16));
    // Repeats inside the window shrink E; tokens outside it do not count.
    let mut repeats: Vec<u32> = (0..15_000u32).collect();
    repeats.extend((0..1000u32).map(|i| 90_000 + i % 50));
    assert_eq!(candidate_budget(&p, &repeats, 16), Ok(20 + 50 + 16));
    // LZ adds its own 256 window; DRY covers the whole history.
    p.lz_penalty = 1.0;
    assert_eq!(candidate_budget(&p, &repeats, 16), Ok(20 + 50 + 16));
    p.dry_multiplier = 0.8;
    assert_eq!(candidate_budget(&p, &repeats, 16), Err("edit_budget"));
}

#[test]
fn windowed_budget_is_exact_against_the_full_sampler_on_long_histories() {
    let mut rng = StdRng::seed_from_u64(0xD71);
    let mut checked = 0;
    for case in 0..800 {
        let vocab = rng.gen_range(300..600);
        let vals = random_row(&mut rng, vocab, 200);
        let mut p = params(1.0, rng.gen_range(1..=20), 0.95, 0.0, rng.r#gen());
        p.presence_penalty = 1.5;
        p.frequency_penalty = if case % 2 == 0 { 0.2 } else { 0.0 };
        p.repetition_penalty = if case % 3 == 0 { 1.1 } else { 1.0 };
        p.repetition_penalty_window = [0u32, 16, 64][case % 3];
        p.lz_penalty = if case % 5 == 0 { 1.0 } else { 0.0 };
        let history: Vec<u32> = (0..rng.gen_range(0..400))
            .map(|_| rng.gen_range(0..vocab as u32))
            .collect();
        let Ok(m) = candidate_budget(&p, &history, 0) else {
            continue;
        };
        let cands = topk_candidates_reference(&row_bytes(&vals), 1.0, m);
        if cands.len() < m {
            continue; // the scheduler takes the full row for these
        }
        let full = f32_bytes(&expand(&vals));
        let sparse = f32_bytes(&sparse_row_from_candidates(vocab, &cands).unwrap());
        let a = sample_seeded_impl(&full, &p, &history, p.seed, true);
        let b = sample_seeded_impl(&sparse, &p, &history, p.seed, true);
        assert_eq!(a, b, "case {case} m={m}");
        checked += 1;
    }
    assert!(checked > 400, "only {checked}");
}

#[test]
fn nan_infinite_and_raising_penalty_parameters_route_to_the_host() {
    let base = params(1.0, 20, 0.95, 0.0, 1);
    let reason = |f: &dyn Fn(&mut SamplingParams)| {
        let mut p = base.clone();
        f(&mut p);
        candidate_budget(&p, &[1, 2, 3, 4], 0).unwrap_err()
    };
    assert_eq!(reason(&|p| p.presence_penalty = f32::NAN), "nan_param");
    assert_eq!(reason(&|p| p.repetition_penalty = f32::NAN), "nan_param");
    assert_eq!(reason(&|p| p.dry_base = f32::NAN), "nan_param");
    assert_eq!(reason(&|p| p.temperature = f32::NAN), "nan_param");
    assert_eq!(reason(&|p| p.lz_penalty = f32::INFINITY), "infinite_param");
    assert_eq!(
        reason(&|p| {
            p.dry_multiplier = 0.8;
            p.dry_base = -1.75;
        }),
        "negative_dry_base"
    );
    // A negative base with DRY off edits nothing.
    let mut p = base.clone();
    p.dry_base = -1.75;
    assert_eq!(candidate_budget(&p, &[1, 2, 3, 4], 0), Ok(20));
}

#[test]
fn sliced_topk_geometry_maps_248k_vocab_to_31_ctas_for_m_up_to_64() {
    assert_eq!(topk_slice_grid(248_320), 31);
    assert!(use_sliced_topk(248_320, 1));
    assert!(use_sliced_topk(248_320, 64));
    assert!(!use_sliced_topk(248_320, 0));
    assert!(!use_sliced_topk(248_320, 65));
    assert!(
        !use_sliced_topk(8192, 20),
        "single slice needs no stage2 merge"
    );
    assert!(use_sliced_topk(8193, 20));
    assert!(!use_sliced_topk(DEVICE_TOPK_SLICE_WIDTH * 65, 20));
}

#[test]
fn sliced_topk_matches_single_cta_reference_across_boundaries_ties_and_248k_rows() {
    let mut rng = StdRng::seed_from_u64(0x511CE);
    for &(vocab, m, t) in &[
        (8191usize, 16usize, 1.0f32),
        (8192, 32, 0.7),
        (8193, 64, 1.3),
        (24_577, 64, 0.9),
        (248_320, 20, 1.0),
        (248_320, 64, 0.6),
    ] {
        let vals = random_row(&mut rng, vocab, 12); // heavy BF16 ties + NaN/+-inf/+-0.0
        let bytes = row_bytes(&vals);
        let single = topk_candidates_reference(&bytes, t, m);
        let sliced = topk_sliced_candidates_reference(&bytes, t, m);
        assert_eq!(single.len(), sliced.len(), "vocab={vocab} m={m}");
        for ((id_a, v_a), (id_b, v_b)) in single.iter().zip(&sliced) {
            assert_eq!(id_a, id_b, "vocab={vocab} m={m} id mismatch");
            assert_eq!(v_a.to_bits(), v_b.to_bits(), "vocab={vocab} m={m} val bits");
        }
    }
}

#[test]
fn device_topk_mode_parses_off_strict_and_auto() {
    assert_eq!(parse_device_topk_mode(None), DeviceTopkMode::Off);
    assert_eq!(parse_device_topk_mode(Some("")), DeviceTopkMode::Off);
    assert_eq!(parse_device_topk_mode(Some("0")), DeviceTopkMode::Off);
    assert_eq!(parse_device_topk_mode(Some("off")), DeviceTopkMode::Off);
    assert_eq!(parse_device_topk_mode(Some("2")), DeviceTopkMode::Off);

    assert_eq!(parse_device_topk_mode(Some("1")), DeviceTopkMode::Strict);
    assert_eq!(parse_device_topk_mode(Some(" 1 ")), DeviceTopkMode::Strict);

    assert_eq!(parse_device_topk_mode(Some("auto")), DeviceTopkMode::Auto);
    assert_eq!(parse_device_topk_mode(Some("AUTO")), DeviceTopkMode::Auto);
    assert_eq!(
        parse_device_topk_mode(Some(" Auto \n")),
        DeviceTopkMode::Auto
    );

    assert!(!DeviceTopkMode::Off.is_enabled());
    assert!(!DeviceTopkMode::Off.is_strict());
    assert!(DeviceTopkMode::Strict.is_enabled());
    assert!(DeviceTopkMode::Strict.is_strict());
    assert!(DeviceTopkMode::Auto.is_enabled());
    assert!(!DeviceTopkMode::Auto.is_strict());
}

#[test]
fn admit_with_mode_refuses_unsupported_only_in_strict_and_admits_all_in_auto() {
    let greedy_zero = params(0.0, 0, 1.0, 0.0, 1);
    let stoch_zero = params(1.0, 0, 0.95, 0.0, 1);
    let stoch_oversized = params(1.0, DEVICE_TOPK_K_MAX as u32 + 1, 0.95, 0.0, 1);
    let stoch_valid = params(1.0, 20, 0.95, 0.0, 1);

    // Off admits everything.
    for p in [&greedy_zero, &stoch_zero, &stoch_oversized, &stoch_valid] {
        assert!(admit_with_mode(p, DeviceTopkMode::Off).is_ok());
    }

    // Strict (=1) keeps exact refusal for top_k == 0 or top_k > K_MAX on stochastic requests.
    assert!(admit_with_mode(&greedy_zero, DeviceTopkMode::Strict).is_ok());
    assert!(admit_with_mode(&stoch_valid, DeviceTopkMode::Strict).is_ok());
    let err_zero = admit_with_mode(&stoch_zero, DeviceTopkMode::Strict).unwrap_err();
    assert!(
        err_zero.contains("ATLAS_SAMPLER_DEVICE_TOPK=1"),
        "{err_zero}"
    );
    assert!(admit_with_mode(&stoch_oversized, DeviceTopkMode::Strict).is_err());

    // Auto (=auto) admits unsupported stochastic requests so runtime routing falls back to host.
    for p in [&greedy_zero, &stoch_zero, &stoch_oversized, &stoch_valid] {
        assert!(
            admit_with_mode(p, DeviceTopkMode::Auto).is_ok(),
            "auto mode must admit top_k={} T={}",
            p.top_k,
            p.temperature
        );
    }
}

#[test]
fn can_use_device_topk_matches_static_request_eligibility() {
    let valid = params(1.0, 20, 0.95, 0.0, 1);
    assert!(can_use_device_topk(&valid));
    assert!(!can_use_device_topk(&params(0.0, 20, 0.95, 0.0, 1)));
    assert!(!can_use_device_topk(&params(1.0, 0, 0.95, 0.0, 1)));
    assert!(!can_use_device_topk(&params(
        1.0,
        DEVICE_TOPK_K_MAX as u32 + 1,
        0.95,
        0.0,
        1
    )));

    let mut with_sigma = valid.clone();
    with_sigma.top_n_sigma = 1.0;
    assert!(!can_use_device_topk(&with_sigma));

    let mut with_bias = valid.clone();
    with_bias.logit_bias = vec![(42, -1.0)];
    assert!(!can_use_device_topk(&with_bias));

    let mut raising_rep = valid;
    raising_rep.repetition_penalty = 0.9;
    assert!(!can_use_device_topk(&raising_rep));
}
