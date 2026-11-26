// SPDX-License-Identifier: AGPL-3.0-only
//
// Tests for the opt-in total candidate order and its partial top-k select.

use super::SamplingParams;
use super::candidate_order::{
    CandidateOrder, order_candidates, parse_candidate_order, resolve_candidate_order, total_desc,
};
use super::sample_impl::{PreparedSamplingDistribution, prepare_sampling_distribution_ordered};
use rand::{Rng, SeedableRng, rngs::StdRng};

const ROWS: usize = 10_000;

fn bytes(logits: &[f32]) -> Vec<u8> {
    logits.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// A row with many exact duplicates: values on a coarse 0.25 grid, plus some
/// -inf entries.
fn dup_row(rng: &mut StdRng, vocab: usize) -> Vec<f32> {
    (0..vocab)
        .map(|_| {
            if rng.gen_bool(0.05) {
                f32::NEG_INFINITY
            } else {
                (rng.gen_range(-8..=8) as f32) * 0.25
            }
        })
        .collect()
}

fn distinct_row(rng: &mut StdRng, vocab: usize) -> Vec<f32> {
    // Distinct by construction: a shuffled ramp with a tiny random offset.
    let mut row: Vec<f32> = (0..vocab).map(|i| i as f32 * 0.37 - 5.0).collect();
    for i in (1..vocab).rev() {
        row.swap(i, rng.gen_range(0..=i));
    }
    row
}

fn random_params(rng: &mut StdRng) -> SamplingParams {
    SamplingParams {
        temperature: [0.6f32, 1.0, 1.3][rng.gen_range(0..3)],
        top_k: [0u32, 1, 5, 20][rng.gen_range(0..4)],
        top_p: [1.0f32, 0.95, 0.5][rng.gen_range(0..3)],
        min_p: [0.0f32, 0.05][rng.gen_range(0..2)],
        top_n_sigma: [0.0f32, 1.0][rng.gen_range(0..2)],
        ..SamplingParams::greedy(0)
    }
}

fn prepare(
    row: &[f32],
    params: &SamplingParams,
    order: CandidateOrder,
) -> PreparedSamplingDistribution {
    prepare_sampling_distribution_ordered(&bytes(row), params, &[], false, order)
}

#[test]
fn flag_parse_defaults_to_total_and_keeps_explicit_legacy_escape_hatch() {
    assert_eq!(parse_candidate_order(None), CandidateOrder::Total);
    assert_eq!(parse_candidate_order(Some("")), CandidateOrder::Total);
    assert_eq!(parse_candidate_order(Some("total")), CandidateOrder::Total);
    assert_eq!(parse_candidate_order(Some("1")), CandidateOrder::Total);
    assert_eq!(
        parse_candidate_order(Some("legacy")),
        CandidateOrder::Legacy
    );
    assert_eq!(
        parse_candidate_order(Some(" legacy ")),
        CandidateOrder::Legacy
    );
    assert_eq!(
        resolve_candidate_order(Some("legacy"), Some("gumbel")),
        CandidateOrder::Total,
        "gumbel verification always enforces total candidate order"
    );
    assert_eq!(
        resolve_candidate_order(Some("legacy"), Some("stochastic")),
        CandidateOrder::Legacy
    );
}

#[test]
fn partial_select_equals_full_total_sort_then_truncate() {
    let mut rng = StdRng::seed_from_u64(1);
    for case in 0..ROWS {
        let vocab = rng.gen_range(1..96);
        let candidates: Vec<(u32, f32)> = dup_row(&mut rng, vocab)
            .into_iter()
            .enumerate()
            .filter(|(_, v)| v.is_finite())
            .map(|(i, v)| (i as u32, v))
            .collect();
        let top_k = [0usize, 1, 2, 5, 20, 200][rng.gen_range(0..6)];
        let mut reference = candidates.clone();
        reference.sort_by(total_desc);
        if top_k > 0 && top_k < reference.len() {
            reference.truncate(top_k);
        }
        let mut selected = candidates;
        order_candidates(&mut selected, top_k, CandidateOrder::Total);
        assert_eq!(selected, reference, "case {case} top_k {top_k}");
    }
}

#[test]
fn total_order_breaks_ties_by_ascending_id() {
    let mut c = vec![
        (4, 1.0f32),
        (2, 1.0),
        (9, 2.0),
        (1, 1.0),
        (3, -0.0),
        (0, 0.0),
    ];
    order_candidates(&mut c, 3, CandidateOrder::Total);
    assert_eq!(c, vec![(9, 2.0), (1, 1.0), (2, 1.0)]);
    let mut z = vec![(3, -0.0f32), (0, 0.0)];
    order_candidates(&mut z, 0, CandidateOrder::Total);
    assert_eq!(z.iter().map(|e| e.0).collect::<Vec<_>>(), vec![0, 3]);
}

/// On rows with duplicated logits the two orders may permute ids inside a
/// group of equal logits, but the weight sequence (and so the CDF and the
/// position a draw lands on) is bit-identical, and the sampled tokens have
/// identical logits. On rows without duplicates the tokens are identical.
#[test]
fn sampler_outputs_match_legacy_up_to_equal_logit_permutation() {
    let mut rng = StdRng::seed_from_u64(2);
    let mut permuted = 0usize;
    for case in 0..ROWS {
        let vocab = rng.gen_range(2..64);
        let distinct = case % 2 == 0;
        let row = if distinct {
            distinct_row(&mut rng, vocab)
        } else {
            dup_row(&mut rng, vocab)
        };
        if row.iter().all(|v| !v.is_finite()) {
            continue;
        }
        let params = random_params(&mut rng);
        let legacy = prepare(&row, &params, CandidateOrder::Legacy);
        let total = prepare(&row, &params, CandidateOrder::Total);
        let weights = |d: &PreparedSamplingDistribution| -> Vec<u32> {
            d.ordered_weights().iter().map(|e| e.1.to_bits()).collect()
        };
        assert_eq!(
            weights(&legacy),
            weights(&total),
            "case {case}: weight sequence"
        );
        for _ in 0..4 {
            let u: f32 = rng.r#gen();
            let a = legacy.sample_with_uniform(u);
            let b = total.sample_with_uniform(u);
            if distinct {
                assert_eq!(a, b, "case {case} u {u}");
            } else {
                assert_eq!(
                    row[a as usize].to_bits(),
                    row[b as usize].to_bits(),
                    "case {case}"
                );
                permuted += usize::from(a != b);
            }
        }
    }
    // The duplicate rows must actually exercise tie permutations.
    assert!(permuted > 0, "no tie permutation was exercised");
}

#[test]
fn top_p_min_p_top_n_sigma_are_unaffected_by_order() {
    let row = [3.0f32, 3.0, 2.5, 2.5, 2.5, 1.0, -1.0, f32::NEG_INFINITY];
    for top_p in [1.0f32, 0.9, 0.5] {
        for min_p in [0.0f32, 0.3] {
            for top_n_sigma in [0.0f32, 0.5] {
                for top_k in [0u32, 2, 3, 4] {
                    let params = SamplingParams {
                        temperature: 1.0,
                        top_k,
                        top_p,
                        min_p,
                        top_n_sigma,
                        ..SamplingParams::greedy(0)
                    };
                    let legacy = prepare(&row, &params, CandidateOrder::Legacy);
                    let total = prepare(&row, &params, CandidateOrder::Total);
                    let lw: Vec<f32> = legacy.ordered_weights().iter().map(|e| e.1).collect();
                    let tw: Vec<f32> = total.ordered_weights().iter().map(|e| e.1).collect();
                    assert_eq!(
                        lw, tw,
                        "p {top_p} min_p {min_p} sigma {top_n_sigma} k {top_k}"
                    );
                    let ids: Vec<u32> = total.ordered_weights().iter().map(|e| e.0).collect();
                    let mut sorted = ids.clone();
                    sorted.sort_by(|a, b| {
                        total_desc(&(*a, row[*a as usize]), &(*b, row[*b as usize]))
                    });
                    assert_eq!(ids, sorted, "total order is deterministic");
                }
            }
        }
    }
}

/// Host cost per sampled position on a Flash-Next-sized row (248,320).
/// Run with `--release -- --ignored --nocapture candidate_order_bench`.
#[test]
#[ignore = "micro-benchmark; run explicitly in release"]
fn candidate_order_bench() {
    const VOCAB: usize = 248_320;
    const ITERS: u32 = 40;
    let mut rng = StdRng::seed_from_u64(3);
    let row: Vec<f32> = (0..VOCAB)
        .map(|_| {
            let bits = (rng.gen_range(-12.0f32..12.0)).to_bits() & 0xffff_0000; // BF16 grid
            f32::from_bits(bits)
        })
        .collect();
    let data = bytes(&row);
    let params = SamplingParams {
        temperature: 1.0,
        top_k: 20,
        top_p: 0.95,
        ..SamplingParams::greedy(0)
    };
    for order in [
        CandidateOrder::Legacy,
        CandidateOrder::Total,
        CandidateOrder::Legacy,
        CandidateOrder::Total,
    ] {
        let start = std::time::Instant::now();
        let mut sink = 0u32;
        for _ in 0..ITERS {
            let d = prepare_sampling_distribution_ordered(&data, &params, &[], false, order);
            sink ^= d.sample_with_uniform(0.5);
        }
        let us = start.elapsed().as_secs_f64() * 1e6 / f64::from(ITERS);
        println!(
            "CANDIDATE_ORDER_BENCH order={order:?} vocab={VOCAB} us_per_position={us:.1} sink={sink}"
        );
    }
}
