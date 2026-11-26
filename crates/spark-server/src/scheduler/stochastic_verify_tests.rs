// SPDX-License-Identifier: AGPL-3.0-only

//! Unit tests for the pure stochastic verifier.

use super::stochastic_verify::*;
use spark_runtime::sampler::{SamplingParams, first_max_index, sampling_uniforms_from_seed};

fn f32_bytes(logits: &[f32]) -> Vec<u8> {
    logits.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn greedy() -> SamplingParams {
    SamplingParams::greedy(0)
}

/// The checkpoint's production sampling (T=1.0, top_p 0.95, top_k 20).
fn checkpoint_params() -> SamplingParams {
    SamplingParams {
        temperature: 1.0,
        top_k: 20,
        top_p: 0.95,
        ..SamplingParams::greedy(0)
    }
}

fn p_row(logits: &[f32], params: &SamplingParams) -> Vec<f32> {
    target_probabilities(&f32_bytes(logits), params, &[]).expect("non-degenerate row")
}

/// A pseudo-random stream of uniforms in [0, 1) from the production RNG.
struct Uniforms {
    draws: Vec<f32>,
    next: usize,
}

impl Uniforms {
    fn new(seed: u64, count: usize) -> Self {
        Self {
            draws: sampling_uniforms_from_seed(seed, count),
            next: 0,
        }
    }
    fn take(&mut self) -> f32 {
        let u = self.draws[self.next % self.draws.len()];
        self.next += 1;
        u
    }
    fn pair(&mut self) -> PositionUniforms {
        PositionUniforms {
            accept: self.take(),
            residual: self.take(),
        }
    }
}

/// Today's argmax accept rule with the house first-wins tie rule.
fn argmax_rule(rows: &[Vec<f32>], drafts: &[u32]) -> (usize, u32) {
    for (i, draft) in drafts.iter().enumerate() {
        let target = first_max_index(&rows[i]);
        if *draft != target {
            return (i, target);
        }
    }
    (drafts.len(), first_max_index(&rows[drafts.len()]))
}

fn sample_from(weights: &[f32], u: f32) -> u32 {
    sample_weights(weights, u).expect("valid weights")
}

#[test]
fn greedy_reduces_to_first_wins_argmax_rule_including_ties() {
    let mut rng = Uniforms::new(7, 1 << 16);
    for case in 0..500 {
        // Coarse values (multiples of 0.125, like BF16 near 25) make exact
        // ties common, which is where a last-wins rule would diverge.
        let rows: Vec<Vec<f32>> = (0..4)
            .map(|_| (0..8).map(|_| (rng.take() * 4.0).floor() * 0.125).collect())
            .collect();
        let drafts: Vec<u32> = (0..3)
            .map(|i| {
                if rng.take() < 0.6 {
                    first_max_index(&rows[i])
                } else {
                    (rng.take() * 8.0) as u32
                }
            })
            .collect();
        let targets: Vec<Vec<f32>> = rows.iter().map(|r| p_row(r, &greedy())).collect();
        let point: Vec<PositionDraft<'_>> = drafts
            .iter()
            .map(|&token| PositionDraft {
                token,
                q: DraftDistribution::PointMass,
            })
            .collect();
        let pairs: Vec<PositionUniforms> = (0..4).map(|_| rng.pair()).collect();
        let outcome = verify_frame(&targets, &point, |i| pairs[i]).expect("frame");
        let (accepted, next) = argmax_rule(&rows, &drafts);
        assert_eq!(
            (outcome.accepted, outcome.next_token),
            (accepted, next),
            "case {case}"
        );
    }
}

#[test]
fn greedy_tie_keeps_the_first_maximal_id() {
    let row = vec![1.0, 3.0, 3.0, 0.0];
    let p = p_row(&row, &greedy());
    assert_eq!(p, vec![0.0, 1.0, 0.0, 0.0]);
    let draft_second = PositionDraft {
        token: 2,
        q: DraftDistribution::PointMass,
    };
    let u = PositionUniforms {
        accept: 0.0,
        residual: 0.999,
    };
    assert_eq!(
        verify_position(&p, &draft_second, u).expect("verdict"),
        PositionVerdict::Reject {
            token: 1,
            branch: RejectBranch::Residual
        }
    );
}

#[test]
fn greedy_dense_q_also_reduces_to_argmax() {
    let p = p_row(&[0.0, 2.0, 1.0], &greedy());
    let q = [0.2f32, 0.5, 0.3];
    for token in 0..3u32 {
        let draft = PositionDraft {
            token,
            q: DraftDistribution::Dense(&q),
        };
        let verdict = verify_position(
            &p,
            &draft,
            PositionUniforms {
                accept: 0.99,
                residual: 0.5,
            },
        )
        .expect("verdict");
        let expected = if token == 1 {
            PositionVerdict::Accept
        } else {
            PositionVerdict::Reject {
                token: 1,
                branch: RejectBranch::Residual,
            }
        };
        assert_eq!(verdict, expected, "token {token}");
    }
}

fn chi_square(counts: &[u64], p: &[f32], n: u64) -> f64 {
    p.iter()
        .zip(counts)
        .filter(|(pi, _)| **pi > 0.0)
        .map(|(&pi, &c)| {
            let expected = f64::from(pi) * n as f64;
            (c as f64 - expected).powi(2) / expected
        })
        .sum()
}

/// Run one position N times; returns emitted-token counts and accept count.
fn simulate_position(p: &[f32], q: Option<&[f32]>, n: u64, seed: u64) -> (Vec<u64>, u64) {
    let mut rng = Uniforms::new(seed, 3 * n as usize);
    let mut counts = vec![0u64; p.len()];
    let mut accepts = 0u64;
    for _ in 0..n {
        let (token, dist) = match q {
            Some(q) => (sample_from(q, rng.take()), DraftDistribution::Dense(q)),
            None => (first_max_index(p), DraftDistribution::PointMass),
        };
        let emitted = match verify_position(p, &PositionDraft { token, q: dist }, rng.pair())
            .expect("verdict")
        {
            PositionVerdict::Accept => {
                accepts += 1;
                token
            }
            PositionVerdict::Reject { token, .. } => token,
        };
        counts[emitted as usize] += 1;
    }
    (counts, accepts)
}

// Chi-square critical value at alpha = 1e-3 for df <= 7 is 24.32.
const CHI2_CRIT_DF7: f64 = 24.32;
const N: u64 = 100_000;

#[test]
fn accepted_stream_matches_target_with_dense_draft() {
    let params = checkpoint_params();
    let p = p_row(&[2.0, 1.5, 1.0, 0.2, -1.0, -6.0, 0.7, 1.2], &params);
    // q disagrees with p, and puts mass on ids p has truncated away.
    let q = p_row(
        &[0.5, 2.2, -0.3, 1.8, 1.0, 1.4, -2.0, 0.0],
        &SamplingParams {
            temperature: 1.0,
            ..SamplingParams::greedy(0)
        },
    );
    let (counts, _) = simulate_position(&p, Some(&q), N, 11);
    for (i, &pi) in p.iter().enumerate() {
        if pi == 0.0 {
            assert_eq!(counts[i], 0, "emitted id {i} with p = 0");
        }
    }
    let chi2 = chi_square(&counts, &p, N);
    assert!(
        chi2 < CHI2_CRIT_DF7,
        "chi2 {chi2} counts {counts:?} p {p:?}"
    );
}

#[test]
fn accepted_stream_matches_target_with_point_mass_draft() {
    let p = p_row(&[1.0, 0.9, 0.4, -0.5, 0.0, -3.0], &checkpoint_params());
    let (counts, accepts) = simulate_position(&p, None, N, 12);
    let chi2 = chi_square(&counts, &p, N);
    assert!(chi2 < CHI2_CRIT_DF7, "chi2 {chi2}");
    // Point-mass acceptance rate is p(argmax).
    let expected = f64::from(p[first_max_index(&p) as usize]);
    let rate = accepts as f64 / N as f64;
    assert!((rate - expected).abs() < 0.01, "rate {rate} vs {expected}");
}

#[test]
fn acceptance_rate_is_overlap_of_p_and_q() {
    let p = p_row(&[1.0, 0.5, 0.0, -0.5], &checkpoint_params());
    let q = p_row(&[0.2, 0.9, 0.1, -0.1], &checkpoint_params());
    let overlap: f64 = p.iter().zip(&q).map(|(&a, &b)| f64::from(a.min(b))).sum();
    let (_, accepts) = simulate_position(&p, Some(&q), N, 13);
    let rate = accepts as f64 / N as f64;
    assert!(
        (rate - overlap).abs() < 0.01,
        "rate {rate} vs overlap {overlap}"
    );
}

#[test]
fn q_equal_to_p_always_accepts() {
    let p = p_row(&[1.0, 0.5, 0.2, -0.5], &checkpoint_params());
    let (_, accepts) = simulate_position(&p, Some(&p), 10_000, 14);
    assert_eq!(accepts, 10_000);
}

#[test]
fn rejection_never_emits_zero_probability_token() {
    let p = vec![0.0, 0.5, 0.0, 0.5];
    let q = vec![0.25, 0.0, 0.75, 0.0];
    for k in 0..1000 {
        let u = k as f32 / 1000.0;
        let draft = PositionDraft {
            token: 2,
            q: DraftDistribution::Dense(&q),
        };
        match verify_position(
            &p,
            &draft,
            PositionUniforms {
                accept: u,
                residual: u,
            },
        )
        .expect("verdict")
        {
            PositionVerdict::Reject { token, .. } => assert!(p[token as usize] > 0.0),
            PositionVerdict::Accept => panic!("accepted a p = 0 draft"),
        }
    }
}

#[test]
fn zero_probability_draft_is_rejected_even_at_u_zero() {
    let p = vec![0.0, 1.0];
    let draft = PositionDraft {
        token: 0,
        q: DraftDistribution::PointMass,
    };
    let verdict = verify_position(
        &p,
        &draft,
        PositionUniforms {
            accept: 0.0,
            residual: 0.0,
        },
    )
    .unwrap();
    assert_eq!(
        verdict,
        PositionVerdict::Reject {
            token: 1,
            branch: RejectBranch::Residual
        }
    );
}

#[test]
fn residual_without_mass_falls_back_to_target() {
    // q >= p everywhere except rounding: residual empty, fall back to p.
    let p = vec![0.5, 0.5];
    let q = vec![0.5, 0.5];
    let draft = PositionDraft {
        token: 0,
        q: DraftDistribution::Dense(&q),
    };
    // accept = 1 - eps with p == q would accept; force the reject branch with
    // a q(x) slightly above p(x) instead.
    let q_hi = vec![0.6, 0.5];
    let draft_hi = PositionDraft {
        token: 0,
        q: DraftDistribution::Dense(&q_hi),
    };
    let u = PositionUniforms {
        accept: 0.95,
        residual: 0.75,
    };
    assert_eq!(
        verify_position(&p, &draft, u).unwrap(),
        PositionVerdict::Accept
    );
    assert_eq!(
        verify_position(&p, &draft_hi, u).unwrap(),
        PositionVerdict::Reject {
            token: 1,
            branch: RejectBranch::TargetFallback
        }
    );
}

#[test]
fn bonus_is_sampled_from_last_row_when_all_accepted() {
    let targets = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![0.25, 0.75]];
    let drafts = [
        PositionDraft {
            token: 0,
            q: DraftDistribution::PointMass,
        },
        PositionDraft {
            token: 1,
            q: DraftDistribution::PointMass,
        },
    ];
    let low = verify_frame(&targets, &drafts, |_| PositionUniforms {
        accept: 0.1,
        residual: 0.9,
    })
    .unwrap();
    assert_eq!(
        low,
        FrameOutcome {
            accepted: 2,
            next_token: 0,
            reject_branch: None
        }
    );
    let high = verify_frame(&targets, &drafts, |_| PositionUniforms {
        accept: 0.5,
        residual: 0.0,
    })
    .unwrap();
    assert_eq!(
        high,
        FrameOutcome {
            accepted: 2,
            next_token: 1,
            reject_branch: None
        }
    );
}

#[test]
fn acceptance_length_counts_prefix_before_first_reject() {
    let targets = vec![
        vec![1.0, 0.0],
        vec![1.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0],
    ];
    let drafts: Vec<PositionDraft<'_>> = [0u32, 0, 0]
        .iter()
        .map(|&token| PositionDraft {
            token,
            q: DraftDistribution::PointMass,
        })
        .collect();
    let outcome = verify_frame(&targets, &drafts, |_| PositionUniforms {
        accept: 0.5,
        residual: 0.5,
    })
    .unwrap();
    assert_eq!(
        outcome,
        FrameOutcome {
            accepted: 2,
            next_token: 1,
            reject_branch: Some(RejectBranch::Residual)
        }
    );
}

#[test]
fn zero_drafts_is_a_plain_sample() {
    let targets = vec![vec![0.0, 0.0, 1.0]];
    let outcome = verify_frame(&targets, &[], |_| PositionUniforms {
        accept: 0.3,
        residual: 0.3,
    })
    .unwrap();
    assert_eq!(
        outcome,
        FrameOutcome {
            accepted: 0,
            next_token: 2,
            reject_branch: None
        }
    );
}

#[test]
fn invalid_inputs_fail_closed() {
    let p = vec![0.5, 0.5];
    let u = PositionUniforms {
        accept: 0.5,
        residual: 0.5,
    };
    let out_of_vocab = PositionDraft {
        token: 9,
        q: DraftDistribution::PointMass,
    };
    assert!(verify_position(&p, &out_of_vocab, u).is_err());
    let q_zero = [1.0f32, 0.0];
    let zero_q = PositionDraft {
        token: 1,
        q: DraftDistribution::Dense(&q_zero),
    };
    assert!(verify_position(&p, &zero_q, u).is_err());
    let bad_u = PositionUniforms {
        accept: 1.0,
        residual: 0.5,
    };
    let ok = PositionDraft {
        token: 0,
        q: DraftDistribution::PointMass,
    };
    assert!(verify_position(&p, &ok, bad_u).is_err());
    assert!(verify_frame(std::slice::from_ref(&p), &[ok], |_| u).is_err());
    assert!(sample_weights(&[0.0, 0.0], 0.5).is_err());
    assert!(sample_weights(&[f32::NAN, 1.0], 0.5).is_err());
}

#[test]
fn seeded_uniforms_are_deterministic_and_start_with_plain_draw() {
    let a = position_uniforms(Some(42));
    assert_eq!(a, position_uniforms(Some(42)));
    assert_eq!(
        a.accept,
        spark_runtime::sampler::sampling_uniform_from_seed(42)
    );
    assert_ne!(a, position_uniforms(Some(43)));
}
