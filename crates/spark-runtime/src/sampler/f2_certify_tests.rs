// SPDX-License-Identifier: AGPL-3.0-only

use super::*;
use rand::{Rng, SeedableRng, rngs::StdRng};

fn bf16(v: f32) -> u16 {
    (v.to_bits() >> 16) as u16
}

fn bytes(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|b| b.to_le_bytes()).collect()
}

fn expand(v: &[u16]) -> Vec<f32> {
    v.iter()
        .map(|&b| f32::from_bits((b as u32) << 16))
        .collect()
}

/// A row whose top probability sweeps across 0.95: one leader at `lead`
/// above a noise floor, plus random edits.
fn row(rng: &mut StdRng, n: usize, lead: f32) -> Vec<u16> {
    let mut v: Vec<u16> = (0..n)
        .map(|_| bf16(rng.gen_range(-12.0f32..-2.0)))
        .collect();
    // Leaders at low ids are the worst case for the summation bound.
    let at = if rng.gen_bool(0.5) {
        rng.gen_range(0..256)
    } else {
        rng.gen_range(0..n)
    };
    v[at] = bf16(lead);
    v
}

#[test]
fn certified_decisions_always_equal_the_host_decision() {
    let mut rng = StdRng::seed_from_u64(0xF2);
    let (mut certified, mut total) = (0usize, 0usize);
    for case in 0..600 {
        let n = [5_000usize, 50_000, 248_320][case % 3];
        let lead = rng.gen_range(0.0f32..14.0);
        let v = row(&mut rng, n, lead);
        let edits: Vec<(u32, PreF2Edit)> = (0..rng.gen_range(0..6))
            .map(|_| {
                let e = if rng.gen_bool(0.7) {
                    PreF2Edit::Sub(10.0)
                } else {
                    PreF2Edit::NegInf
                };
                (rng.gen_range(0..n as u32), e)
            })
            .collect();
        let stats = edited_sum_exp_reference(&bytes(&v), &edits);
        let mut host = expand(&v);
        for &(id, e) in &edits {
            e.apply(&mut host[id as usize]);
        }
        total += 1;
        if let Some(decision) = certify(&stats) {
            certified += 1;
            assert_eq!(
                decision,
                host_f2_confident(&host),
                "case {case} n={n} S={}",
                stats.sum
            );
        }
    }
    // The sweep puts many rows near the boundary; most must still certify.
    assert!(
        certified * 10 > total * 7,
        "only {certified}/{total} certified"
    );
}

fn stats(sum: f64) -> SumExpStats {
    SumExpStats {
        max: 3.0,
        sum,
        weighted: 1000.0,
        term_err: 10.0,
        n: 248_320,
        nonfinite: false,
    }
}

#[test]
fn rows_on_the_boundary_are_never_certified() {
    let threshold = 1.0 / f64::from(0.95f32);
    for s in [
        threshold,
        threshold * (1.0 + 1e-6),
        threshold * (1.0 - 1e-6),
    ] {
        assert_eq!(certify(&stats(s)), None);
    }
    // The bound grows with the summation weight: a leader at id 0 of a
    // 248k row widens the band to about 1.5 %.
    let wide = SumExpStats {
        weighted: 248_319.0,
        ..stats(threshold * 0.99)
    };
    assert_eq!(certify(&wide), None);
    assert_eq!(certify(&stats(threshold * 0.99)), Some(true));
}

#[test]
fn nonfinite_or_degenerate_rows_are_uncertain() {
    let ok = stats(1.001);
    assert_eq!(certify(&ok), Some(true));
    assert_eq!(
        certify(&SumExpStats {
            nonfinite: true,
            ..ok
        }),
        None
    );
    assert_eq!(
        certify(&SumExpStats {
            max: f32::NEG_INFINITY,
            ..ok
        }),
        None
    );
    assert_eq!(
        certify(&SumExpStats {
            sum: f64::NAN,
            ..ok
        }),
        None
    );
    assert_eq!(certify(&SumExpStats { sum: 0.5, ..ok }), None);
    assert_eq!(certify(&stats(40.0)), Some(false));
}

#[test]
fn bound_dominates_the_observed_host_device_gap() {
    let mut rng = StdRng::seed_from_u64(7);
    for _ in 0..24 {
        let n = 248_320;
        let lead = rng.gen_range(0.0f32..12.0);
        let v = row(&mut rng, n, lead);
        let s = edited_sum_exp_reference(&bytes(&v), &[]);
        let host = expand(&v);
        let max = host.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let sh: f32 = host.iter().map(|&l| (l - max).exp()).sum();
        assert!((f64::from(sh) - s.sum).abs() <= absolute_error_bound(&s));
    }
}

#[test]
fn reference_flags_nan_and_positive_infinity_after_edits() {
    let v = [bf16(1.0), 0x7FC0, bf16(2.0)];
    assert!(edited_sum_exp_reference(&bytes(&v), &[]).nonfinite);
    // Masking the NaN on the host makes it -inf; the stats see the edit.
    assert!(!edited_sum_exp_reference(&bytes(&v), &[(1, PreF2Edit::NegInf)]).nonfinite);
    let inf = [bf16(1.0), bf16(f32::INFINITY)];
    assert!(edited_sum_exp_reference(&bytes(&inf), &[]).nonfinite);
}

#[test]
fn edits_apply_in_order_exactly_like_the_host() {
    let v = [bf16(3.0), bf16(1.0)];
    let s = edited_sum_exp_reference(
        &bytes(&v),
        &[(0, PreF2Edit::Sub(10.0)), (0, PreF2Edit::Sub(10.0))],
    );
    assert_eq!(s.max, 1.0);
    let mut x = 3.0f32;
    PreF2Edit::Sub(10.0).apply(&mut x);
    PreF2Edit::Sub(10.0).apply(&mut x);
    assert_eq!(x, -17.0);
    assert_eq!(PreF2Edit::NegInf.encode(), (1, 0.0));
    assert_eq!(PreF2Edit::Sub(10.0).encode(), (0, 10.0));
}
