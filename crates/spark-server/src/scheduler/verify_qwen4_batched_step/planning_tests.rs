// SPDX-License-Identifier: AGPL-3.0-only

use super::{
    BatchedK5Candidate, BatchedK5Plan, CohortAcceptEwma, DraftDepthPolicy, adaptive_drafts,
    batched_k5_policy_required, batched_k5_width, batched_k5_width_for, choose_harmonized_width,
    ladder_drafts, narrow_pending_drafts, parse_ladder, plan_batched_k5, plan_batched_uniform,
    plan_ragged_cohort, resolve_cohort_width_capped, resolve_cohort_width_with_controller,
    resolve_effective_depth,
};

#[test]
fn non_thinking_qwen4_sequence_with_think_ended_is_eligible_for_batched_k5() {
    let mut a = crate::scheduler::test_fixtures::active_seq(
        crate::scheduler::ResponseSink::Blocking(None),
        1,
    );
    a.inside_thinking = false;
    a.think_ended = true;
    a.think_just_ended = false;
    let think = crate::scheduler::ThinkSpecCtx {
        enabled: false,
        code_fence_token: None,
        reflection_suppress_ids: &[],
        adaptive_sampling: false,
        honor_eos_inside_thinking: Some(false),
    };
    assert!(!batched_k5_policy_required(&a, &think));

    a.think_just_ended = true;
    assert!(!batched_k5_policy_required(&a, &think));
    a.think_just_ended = false;

    a.inside_thinking = true;
    assert!(batched_k5_policy_required(&a, &think));
    let think_enabled = crate::scheduler::ThinkSpecCtx {
        enabled: true,
        ..think
    };
    assert!(!batched_k5_policy_required(&a, &think_enabled));
}

#[test]
fn ladder_parses_ascending_positive_rungs_only() {
    assert_eq!(
        parse_ladder("2:4,4:3,8:2,32:1").unwrap(),
        vec![(2, 4), (4, 3), (8, 2), (32, 1)]
    );
    for bad in [
        "", "2", "2:0", "0:4", "4:3,2:4", "4:3,4:2", "a:1", "2:b", "2:4,",
    ] {
        assert!(parse_ladder(bad).is_err(), "{bad:?}");
    }
}

#[test]
fn ladder_picks_the_first_covering_rung_and_never_exceeds_configured() {
    let ladder = parse_ladder("2:4,4:3,8:2,32:1").unwrap();
    let at = |c| ladder_drafts(&ladder, c, 4);
    assert_eq!(
        (at(1), at(2), at(3), at(4), at(5), at(8), at(9), at(64)),
        (4, 4, 3, 3, 2, 2, 1, 1)
    );
    assert_eq!(
        ladder_drafts(&ladder, 1, 2),
        2,
        "configured width is a ceiling"
    );
}

fn neutral(draft_len: usize) -> BatchedK5Candidate {
    BatchedK5Candidate {
        draft_len,
        configured_num_drafts: 4,
        grammar_active: false,
        finished: false,
        policy_required: false,
        has_tree: false,
        has_sampled: false,
    }
}

#[test]
fn neutral_flat_chains_batch_at_their_own_width() {
    assert_eq!(batched_k5_width(neutral(4)), Some(5));
    assert_eq!(batched_k5_width(neutral(3)), Some(4));
    assert_eq!(batched_k5_width(neutral(1)), Some(2));
}

#[test]
fn policy_tree_sampled_finished_and_empty_frames_stay_per_sequence_while_grammar_batches() {
    assert_eq!(
        batched_k5_width(BatchedK5Candidate {
            grammar_active: true,
            ..neutral(4)
        }),
        Some(5),
        "active grammar sequences share batched Qwen4 K5 verify"
    );
    let cases = [
        BatchedK5Candidate {
            finished: true,
            ..neutral(4)
        },
        BatchedK5Candidate {
            policy_required: true,
            ..neutral(4)
        },
        BatchedK5Candidate {
            has_tree: true,
            ..neutral(4)
        },
        BatchedK5Candidate {
            has_sampled: true,
            ..neutral(4)
        },
        neutral(0),
        neutral(5),
    ];
    for case in cases {
        assert_eq!(batched_k5_width(case), None, "{case:?}");
    }
}

#[test]
fn wider_than_k9_never_batches() {
    let wide = BatchedK5Candidate {
        configured_num_drafts: 15,
        ..neutral(15)
    };
    assert_eq!(batched_k5_width(wide), None);
    let k9 = BatchedK5Candidate {
        configured_num_drafts: 8,
        ..neutral(8)
    };
    assert_eq!(batched_k5_width(k9), Some(9));
}

#[test]
fn ragged_width_admits_k1_anchor_only_through_k8_and_plan_ragged_cohort_packs_mixed_widths() {
    // Under ragged mode (allow_k1 = true), draft_len=0 maps to Some(1) (K=1 anchor-only)
    // and widths 1..=8 are admitted; K=9 is excluded from ragged bin packing.
    assert_eq!(batched_k5_width_for(neutral(0), false), None);
    assert_eq!(batched_k5_width_for(neutral(0), true), Some(1));
    for drafts in 0..=7usize {
        let cand = BatchedK5Candidate {
            configured_num_drafts: 7,
            ..neutral(drafts)
        };
        assert_eq!(batched_k5_width_for(cand, true), Some(drafts + 1));
    }
    let k9 = BatchedK5Candidate {
        configured_num_drafts: 8,
        ..neutral(8)
    };
    assert_eq!(batched_k5_width_for(k9, true), None);

    // Mixed [7, 3, 1, None, 5, 8, 2] packs into ONE ragged frame (26 rows <= 128).
    let mixed = [
        (0, Some(7)),
        (1, Some(3)),
        (2, Some(1)),
        (3, None),
        (4, Some(5)),
        (5, Some(8)),
        (6, Some(2)),
    ];
    assert_eq!(
        plan_ragged_cohort(&mixed),
        BatchedK5Plan {
            frames: vec![vec![0, 1, 2, 4, 5, 6]],
            leftover: vec![3],
        }
    );

    // Single eligible sequence stays in leftover (no singleton frames).
    assert_eq!(
        plan_ragged_cohort(&[(0, Some(1)), (1, None)]),
        BatchedK5Plan {
            frames: Vec::new(),
            leftover: vec![0, 1],
        }
    );

    // 17 K=8 sequences (> 128 rows) split into 2 valid frames with zero singletons.
    let over_rows: Vec<(usize, Option<usize>)> = (0..17).map(|i| (i, Some(8))).collect();
    let plan_over = plan_ragged_cohort(&over_rows);
    assert_eq!(plan_over.frames.len(), 2);
    assert!(plan_over.leftover.is_empty());
    assert!(
        plan_over
            .frames
            .iter()
            .all(|f| f.len() >= 2 && f.len() * 8 <= 128)
    );

    // 65 K=1 anchor-only sequences (> 64 seqs) split into 2 valid frames (63 + 2).
    let over_seqs: Vec<(usize, Option<usize>)> = (0..65).map(|i| (i, Some(1))).collect();
    let plan_seqs = plan_ragged_cohort(&over_seqs);
    assert_eq!(plan_seqs.frames.len(), 2);
    assert!(plan_seqs.leftover.is_empty());
    assert!(plan_seqs.frames.iter().all(|f| (2..=64).contains(&f.len())));

    // With ragged compaction active (cohort_adaptive_k = false), C=2..16 keeps
    // configured 6 drafts (K_s up to 7); turning ragged off (cohort_adaptive_k = true)
    // restores CohortAcceptEwma uniform throttling on low acceptance.
    let ctrl = CohortAcceptEwma::new();
    for live in [2usize, 4, 8, 16] {
        assert_eq!(
            resolve_cohort_width_with_controller(true, true, false, None, live, 6, &ctrl),
            6,
            "ragged mode drafts up to K=7 ceiling at C={live}"
        );
    }
}

#[test]
fn plan_groups_by_width_and_returns_singletons() {
    let candidates = [
        (0, Some(5)),
        (1, Some(5)),
        (2, Some(4)),
        (3, None),
        (4, Some(5)),
        (5, Some(3)),
        (6, Some(3)),
    ];
    let plan = plan_batched_k5(&candidates);
    assert_eq!(
        plan,
        BatchedK5Plan {
            frames: vec![vec![5, 6], vec![0, 1, 4]],
            leftover: vec![2, 3],
        }
    );
}

#[test]
fn plan_splits_c26_k5_into_two_balanced_frames_while_c8_k5_is_one() {
    let c8: Vec<_> = (0..8).map(|idx| (idx, Some(5))).collect();
    let plan_c8 = plan_batched_k5(&c8);
    assert_eq!(plan_c8.frames, vec![(0..8).collect::<Vec<_>>()]);
    assert!(plan_c8.leftover.is_empty());

    let c26: Vec<_> = (0..26).map(|idx| (idx, Some(5))).collect();
    let plan_c26 = plan_batched_k5(&c26);
    assert_eq!(
        plan_c26.frames,
        vec![(0..13).collect::<Vec<_>>(), (13..26).collect::<Vec<_>>()]
    );
    assert!(plan_c26.leftover.is_empty());
}

#[test]
fn plan_covers_every_index_exactly_once() {
    for n in 0..80usize {
        let candidates: Vec<_> = (0..n)
            .map(|idx| (idx, [Some(5), Some(4), None, Some(5), Some(3)][idx % 5]))
            .collect();
        let plan = plan_batched_k5(&candidates);
        let mut seen: Vec<usize> = plan.frames.iter().flatten().copied().collect();
        seen.extend(&plan.leftover);
        seen.sort_unstable();
        assert_eq!(seen, (0..n).collect::<Vec<_>>(), "n={n}");
        assert!(plan.frames.iter().all(|frame| frame.len() >= 2));
    }
}

#[test]
fn adaptive_drafts_per_concurrency() {
    use spark_model::layers::qwen4_verify_lanes::{
        ADAPTIVE_MAX_FRAME_ROWS, ADAPTIVE_MAX_K, AdaptiveVerifyProfile, adaptive_verify_profile,
        adaptive_verify_width, adaptive_verify_width_for_profile,
    };
    let max_drafts = ADAPTIVE_MAX_K - 1;
    let use_k4_profile = matches!(
        adaptive_verify_profile(),
        Ok(AdaptiveVerifyProfile::K4C7C16)
    );
    for c in 0..=160 {
        let expected_width = if use_k4_profile {
            adaptive_verify_width_for_profile(c, AdaptiveVerifyProfile::K4C7C16)
        } else {
            adaptive_verify_width(c)
        };
        for configured in [2usize, 4, 6, max_drafts] {
            let expected = expected_width
                .map_or(1, |k| k.saturating_sub(1).max(1))
                .min(configured);
            assert_eq!(
                adaptive_drafts(c, configured),
                expected,
                "C{c} configured={configured}"
            );
            if (1..=64).contains(&c) {
                assert!(
                    c * (expected + 1) <= ADAPTIVE_MAX_FRAME_ROWS,
                    "C{c} configured={configured} drafts={expected} exceeds 128-row frame"
                );
            }
        }
    }
    if !use_k4_profile {
        for c in 1..=16usize {
            assert_eq!(
                adaptive_drafts(c, 6),
                max_drafts.min(6),
                "C{c} with --num-drafts 6 must not clamp below single-frame adaptive ceiling"
            );
        }
    }
    assert_eq!(adaptive_drafts(2, 2), 2, "configured width is a ceiling");
    assert_eq!(
        adaptive_drafts(2, 0),
        0,
        "zero configured drafts stays zero"
    );
}

/// Every C2..=C64 cohort plans one <=128-row frame, every C65..=C128 cohort
/// plans two balanced <=128-row K2 sub-frames (up to 256 verify rows/cycle),
/// and C129+ has no admitted adaptive width.
#[test]
fn adaptive_plans_one_frame_to_c64_and_two_balanced_frames_to_c128() {
    use spark_model::layers::qwen4_verify_lanes::{
        ADAPTIVE_MAX_FRAME_ROWS, ADAPTIVE_MAX_K, AdaptiveVerifyProfile, adaptive_verify_profile,
        adaptive_verify_width,
    };
    let max_drafts = ADAPTIVE_MAX_K - 1;
    let use_k4_profile = matches!(
        adaptive_verify_profile(),
        Ok(AdaptiveVerifyProfile::K4C7C16)
    );
    for c in 1..=140usize {
        let Some(k) = adaptive_verify_width(c) else {
            assert!(c >= 129, "C{c} must speculate");
            assert_eq!(adaptive_drafts(c, max_drafts), 1, "C{c} defensive floor");
            continue;
        };
        let drafts = adaptive_drafts(c, max_drafts);
        let k_eff = drafts + 1;
        if !use_k4_profile {
            assert_eq!(k_eff, k, "C{c}");
        } else {
            assert!(k_eff <= k, "C{c}: k_eff={k_eff} > k={k}");
        }
        let width = batched_k5_width(BatchedK5Candidate {
            configured_num_drafts: max_drafts,
            ..neutral(drafts)
        });
        assert_eq!(width, Some(k_eff), "C{c}");
        let candidates: Vec<_> = (0..c).map(|idx| (idx, width)).collect();
        let plan = plan_batched_uniform(&candidates);
        if c == 1 {
            assert!(plan.frames.is_empty() && plan.leftover == vec![0]);
            continue;
        }
        if c <= 64 {
            assert_eq!(plan.frames.len(), 1, "C{c} K{k_eff}: {plan:?}");
            assert_eq!(plan.frames[0], (0..c).collect::<Vec<_>>(), "C{c}");
            assert!(
                plan.frames[0].len() * k_eff <= ADAPTIVE_MAX_FRAME_ROWS,
                "C{c} K{k_eff}"
            );
        } else {
            assert_eq!(k_eff, 2, "C{c}");
            assert_eq!(plan.frames.len(), 2, "C{c} K{k_eff}: {plan:?}");
            let mid = c.div_ceil(2);
            assert_eq!(plan.frames[0], (0..mid).collect::<Vec<_>>(), "C{c}");
            assert_eq!(plan.frames[1], (mid..c).collect::<Vec<_>>(), "C{c}");
            assert!(
                plan.frames[0].len() * k_eff <= ADAPTIVE_MAX_FRAME_ROWS,
                "C{c}"
            );
            assert!(
                plan.frames[1].len() * k_eff <= ADAPTIVE_MAX_FRAME_ROWS,
                "C{c}"
            );
        }
        assert!(plan.leftover.is_empty(), "C{c}");
    }
}

#[test]
fn fixed_k5_c26_needs_two_frames_and_adaptive_k4_needs_one() {
    let k5: Vec<_> = (0..26).map(|idx| (idx, Some(5))).collect();
    assert_eq!(plan_batched_k5(&k5).frames.len(), 2);
    let k4: Vec<_> = (0..26).map(|idx| (idx, Some(4))).collect();
    assert_eq!(
        plan_batched_uniform(&k4).frames,
        vec![(0..26).collect::<Vec<_>>()]
    );
}

#[test]
fn uniform_plan_keeps_the_dominant_width_and_leaves_the_rest() {
    // A transition step: two sequences still carry a shorter retrieval
    // chain, one is ineligible.
    let candidates = [
        (0, Some(3)),
        (1, Some(3)),
        (2, Some(2)),
        (3, Some(3)),
        (4, None),
        (5, Some(2)),
        (6, Some(3)),
    ];
    let plan = plan_batched_uniform(&candidates);
    assert_eq!(
        plan,
        BatchedK5Plan {
            frames: vec![vec![0, 1, 3, 6]],
            leftover: vec![2, 4, 5],
        }
    );
    // Ties go to the wider width; singletons never form a cohort.
    let tie = [
        (0, Some(2)),
        (1, Some(2)),
        (2, Some(4)),
        (3, Some(4)),
        (4, Some(5)),
    ];
    assert_eq!(plan_batched_uniform(&tie).frames, vec![vec![2, 3]]);
    let singles = [(0, Some(2)), (1, Some(3)), (2, None)];
    let plan = plan_batched_uniform(&singles);
    assert!(plan.frames.is_empty());
    assert_eq!(plan.leftover, vec![0, 1, 2]);
}

#[test]
fn uniform_plan_covers_every_index_exactly_once_with_one_width() {
    for n in 0..40usize {
        let candidates: Vec<_> = (0..n)
            .map(|idx| (idx, [Some(2), Some(3), None, Some(2), Some(5)][idx % 5]))
            .collect();
        let plan = plan_batched_uniform(&candidates);
        let mut seen: Vec<usize> = plan.frames.iter().flatten().copied().collect();
        seen.extend(&plan.leftover);
        seen.sort_unstable();
        assert_eq!(seen, (0..n).collect::<Vec<_>>(), "n={n}");
        let widths: std::collections::BTreeSet<_> = plan
            .frames
            .iter()
            .flatten()
            .map(|&idx| candidates[idx].1)
            .collect();
        assert!(widths.len() <= 1, "n={n} {widths:?}");
    }
}

#[test]
fn narrowing_shortens_only_longer_flat_chains() {
    let mut active: Vec<_> = (0..4)
        .map(|slot| {
            crate::scheduler::test_fixtures::active_seq(
                crate::scheduler::ResponseSink::Blocking(None),
                slot,
            )
        })
        .collect();
    active[0].pending_drafts = vec![11, 12, 13, 14];
    active[1].pending_drafts = vec![21, 22];
    active[2].pending_drafts = vec![31, 32, 33, 34];
    active[3].pending_drafts = vec![41, 42, 43, 44];
    narrow_pending_drafts(&mut active, &[0, 1, 2], 2);
    assert_eq!(active[0].pending_drafts, vec![11, 12], "prefix is kept");
    assert_eq!(active[1].pending_drafts, vec![21, 22]);
    assert_eq!(active[2].pending_drafts, vec![31, 32]);
    assert_eq!(
        active[3].pending_drafts,
        vec![41, 42, 43, 44],
        "not in the cohort"
    );
    narrow_pending_drafts(&mut active, &[0], 0);
    assert_eq!(
        active[0].pending_drafts,
        vec![11],
        "a chain never narrows to empty"
    );
}

#[test]
fn hidden_saves_precede_every_proposal() {
    let source = include_str!("verify_exec.rs");
    let step = source
        .split("fn step_verify_qwen4_batched(")
        .nth(1)
        .unwrap()
        .split("fn batch_sequences_mut")
        .next()
        .unwrap();
    let phase1 = step.find("verify_frame(").unwrap();
    let finish = step.find("finish_verify_qwen4_batch(").unwrap();
    let kv_metric = step
        .find("crate::metrics::compat_metrics::set_kv_blocks_free(model.num_free_blocks());")
        .expect("batched verify must refresh kv_blocks_free metric");
    let phase2 = step.find("propose_next(model").unwrap();
    assert!(
        phase1 < finish && finish < kv_metric && kv_metric < phase2,
        "every frame must settle, refresh kv_blocks_free, and release its cohort before proposals"
    );
    let finalize = source.split("fn finalize_commit(").nth(1).unwrap();
    let finalize = finalize.split("fn propose_next(").next().unwrap();
    let commit = finalize.find("commit_verify_state_async(").unwrap();
    let save = finalize
        .find("save_hidden_for_mtp_sequence_in_frame(")
        .unwrap();
    assert!(commit < save);
    assert!(save < finalize.find("trim_proposer_state(").unwrap());
    assert!(!finalize.contains("propose_and_install"));
}

#[test]
fn commits_use_the_slot_lane_and_cohort_abort_owns_state_aware_rollback() {
    let source = include_str!("verify_exec.rs");
    let step = source
        .split("fn verify_frame(")
        .nth(1)
        .unwrap()
        .split("fn finalize_commit(")
        .next()
        .unwrap();
    let lane = step
        .find("let lane = LaneGuard::enter_for_slot(a.seq.slot_idx)")
        .unwrap();
    let commit = step.find("let propose = finalize_commit(").unwrap();
    let release = step.find("drop(lane);").unwrap();
    assert!(lane < commit && commit < release);
    let abort = source
        .split("fn abort_cohort(")
        .nth(1)
        .unwrap()
        .split("fn verify_frame(")
        .next()
        .unwrap();
    assert!(abort.contains("model.abort_verify_qwen4_batch(&mut seqs)"));
    assert!(!abort.contains("tokens.truncate") && !abort.contains("seq_len ="));
}

#[test]
fn harmonize_merges_singleton_k4_and_k5_into_one_k4_frame() {
    let candidates = [(0, Some(4)), (1, Some(5))];
    assert_eq!(
        choose_harmonized_width(&candidates),
        Some(4),
        "1xK4 + 1xK5 (2 serial fwds, 4.5 rows/fwd) must harmonize to 2xK4 (1 batched fwd, 8.0 rows/fwd)"
    );
}

#[test]
fn harmonize_merges_balanced_k4_k5_split_into_one_k4_frame() {
    let candidates = [
        (0, Some(5)),
        (1, Some(5)),
        (2, Some(5)),
        (3, Some(4)),
        (4, Some(4)),
        (5, Some(4)),
    ];
    assert_eq!(
        choose_harmonized_width(&candidates),
        Some(4),
        "3xK5 + 3xK4 (4 fwds, 6.75 rows/fwd) must harmonize to 6xK4 (1 fwd, 24.0 rows/fwd)"
    );
}

#[test]
fn harmonize_keeps_k5_when_single_k2_outlier_would_hurt_throughput() {
    let candidates = [
        (0, Some(5)),
        (1, Some(5)),
        (2, Some(5)),
        (3, Some(5)),
        (4, Some(5)),
        (5, Some(2)),
    ];
    assert_eq!(
        choose_harmonized_width(&candidates),
        None,
        "5xK5 + 1xK2 (2 fwds, 13.5 rows/fwd) beats 6xK2 (1 fwd, 12.0 rows/fwd)"
    );
}

#[test]
fn harmonize_is_noop_when_cohort_already_uniform_or_singleton() {
    assert_eq!(choose_harmonized_width(&[(0, Some(5)), (1, Some(5))]), None);
    assert_eq!(choose_harmonized_width(&[(0, Some(4)), (1, None)]), None);
}

#[test]
fn draft_depth_policy_is_uniform_only_when_batched_and_concurrent() {
    assert_eq!(
        DraftDepthPolicy::for_cohort(true, 2),
        DraftDepthPolicy::CohortUniform
    );
    assert_eq!(
        DraftDepthPolicy::for_cohort(true, 6),
        DraftDepthPolicy::CohortUniform
    );
    assert_eq!(
        DraftDepthPolicy::for_cohort(true, 1),
        DraftDepthPolicy::PerSequence
    );
    assert_eq!(
        DraftDepthPolicy::for_cohort(false, 6),
        DraftDepthPolicy::PerSequence
    );
    assert_eq!(DraftDepthPolicy::CohortUniform.resolve(0x1234, 4), 4);
}

#[test]
fn dflash_preserves_gamma7_under_gumbel_while_native_mtp_uses_stochastic_depth() {
    // Under FLUX_SPEC_VERIFY=gumbel, adaptive_depth::stochastic_depth may
    // recommend a truncated depth (e.g. 3 or 1). DFlash emits its full
    // block in one pass and must keep gamma=7, whereas native MTP must
    // respect stochastic_depth at C=1 and fixed width at C>=2 fixed-K5.
    assert_eq!(
        resolve_effective_depth(true, false, false, 1, 7, Some(3)),
        7
    );
    assert_eq!(
        resolve_effective_depth(true, false, false, 1, 7, Some(1)),
        7
    );
    assert_eq!(
        resolve_effective_depth(false, false, false, 1, 7, Some(3)),
        3
    );
    assert_eq!(
        resolve_effective_depth(false, true, false, 1, 4, Some(2)),
        2
    );
    assert_eq!(
        resolve_effective_depth(false, true, false, 2, 4, Some(2)),
        4
    );
    assert_eq!(resolve_effective_depth(false, true, true, 2, 4, Some(2)), 4);
    assert_eq!(resolve_effective_depth(false, false, false, 1, 7, None), 7);
}

#[test]
fn cohort_ewma_and_roofline_width_controller_adapt_by_acceptance_and_respect_overrides() {
    use spark_model::layers::qwen4_verify_lanes::{
        ADAPTIVE_MAX_FRAME_ROWS, ADAPTIVE_MAX_K, optimal_cohort_k,
    };
    let max_batched_drafts = ADAPTIVE_MAX_K - 1;
    let k7_or_cap_drafts = 6usize.min(max_batched_drafts);
    let ewma = CohortAcceptEwma::new();
    assert!((ewma.ewma_p_step() - 0.80).abs() < 1e-6);

    // Solo C=1 keeps configured width (e.g. 6 drafts -> K=7, or 10 -> K=11), even on a batched server.
    assert_eq!(
        resolve_cohort_width_capped(true, true, true, None, 1, 6, 0.55),
        6
    );
    assert_eq!(
        resolve_cohort_width_capped(true, true, true, None, 1, 10, 0.55),
        10
    );
    // Zero configured drafts returns 0 cleanly without panicking.
    assert_eq!(
        resolve_cohort_width_capped(true, true, true, None, 8, 0, 0.80),
        0
    );

    // High-acceptance cohort with configured=6 reaches K=7 (6 drafts, bounded by ADAPTIVE_MAX_K-1)
    // across c=2..16, and wider solo configs (e.g. 10) cap to ADAPTIVE_MAX_K-1 (K=8 -> 7 drafts).
    for live in [2usize, 4, 8, 16] {
        let expected_c = adaptive_drafts(live, k7_or_cap_drafts);
        assert_eq!(
            resolve_cohort_width_capped(true, true, true, None, live, 6, 1.0),
            expected_c,
            "high-accept configured=6 live={live}"
        );
        let expected_wide = adaptive_drafts(live, max_batched_drafts);
        assert_eq!(
            resolve_cohort_width_capped(true, true, true, None, live, 10, 1.0),
            expected_wide,
            "high-accept configured=10 caps to ADAPTIVE_MAX_K-1 live={live}"
        );
    }

    // Feed genuinely low-acceptance frames (8 seqs, 2 accepted, 8 rejections -> p=0.20).
    for _ in 0..16 {
        ewma.observe_frame_seqs(8, 2, 48, 8);
    }
    let p_low = ewma.ewma_p_step();
    assert!(
        p_low < 0.30,
        "ewma should converge below 0.30 on low-accept frames, got {p_low}"
    );
    for live in [2usize, 4, 8, 16] {
        assert!(
            resolve_cohort_width_capped(true, true, true, None, live, 6, p_low) <= 2,
            "low-accept p={p_low} live={live} must narrow cleanly"
        );
    }

    // Exact roofline delegation & single-frame <=128-row invariant across concurrency and p.
    for live in [2usize, 4, 8, 16, 24, 36, 64] {
        for configured in [4usize, 6] {
            let profile_drafts = adaptive_drafts(live, configured.min(max_batched_drafts));
            let max_k = (profile_drafts + 1).min(ADAPTIVE_MAX_K);
            for p in [0.25_f32, 0.60, 0.80, 0.98] {
                let got = resolve_cohort_width_capped(true, true, true, None, live, configured, p);
                let want = optimal_cohort_k(live, p, max_k)
                    .saturating_sub(1)
                    .clamp(1, profile_drafts);
                assert_eq!(got, want, "live={live} configured={configured} p={p}");
                assert!(
                    live * (got + 1) <= ADAPTIVE_MAX_FRAME_ROWS,
                    "live={live} K={} exceeds 128-row single frame",
                    got + 1
                );
            }
        }
    }

    // Disabling ATLAS_QWEN4_COHORT_ADAPTIVE_K restores profile ceiling for A/B runs.
    assert_eq!(
        resolve_cohort_width_capped(true, true, false, None, 8, 6, p_low),
        adaptive_drafts(8, k7_or_cap_drafts)
    );
    // Fixed-K5 mode without ATLAS_QWEN4_BATCHED_ADAPTIVE_K=1 caps batched cohorts to K=5 (4 drafts).
    assert_eq!(
        resolve_cohort_width_capped(true, false, true, None, 8, 6, 0.95),
        4
    );
    assert_eq!(
        resolve_cohort_width_capped(true, false, true, None, 8, 4, p_low),
        4
    );
    // Explicit ATLAS_QWEN4_BATCHED_K5_LADDER always overrides the roofline controller.
    let ladder = [(4usize, 3usize), (16usize, 1usize)];
    assert_eq!(
        resolve_cohort_width_capped(true, true, true, Some(&ladder), 4, 6, 0.90),
        3
    );
    assert_eq!(
        resolve_cohort_width_capped(true, true, true, Some(&ladder), 16, 6, 0.90),
        1
    );
}

#[test]
fn cohort_controller_enforces_dwell_hysteresis_and_unbiased_ewma() {
    let max_k = spark_model::layers::qwen4_verify_lanes::ADAPTIVE_MAX_K.min(7);
    let ctrl = CohortAcceptEwma::new();
    // Warm-start at c=8 opens at full cohort max_k (K=7 when ADAPTIVE_MAX_K=8).
    assert_eq!(ctrl.select_k_deterministic(8, max_k, false), max_k);
    // Repeated calls without a new settled frame are idempotent.
    assert_eq!(ctrl.select_k_deterministic(8, max_k, false), max_k);

    // Sustained 0-accept frames must NOT prematurely drop K before MIN_DWELL_STEPS (6).
    let mut cur_k = max_k;
    for step in 1..=5 {
        let d = cur_k - 1;
        ctrl.observe_frame_seqs(8, 0, 8 * d, 8);
        cur_k = ctrl.select_k_deterministic(8, max_k, false);
        assert_eq!(cur_k, max_k, "dwell must hold K={max_k} on step {step}");
    }
    // Continued 0-accept frames collapse cleanly to K=2 with minimal transitions.
    let mut down_transitions = 0usize;
    for _ in 6..=16 {
        let d = cur_k - 1;
        ctrl.observe_frame_seqs(8, 0, 8 * d, 8);
        let next_k = ctrl.select_k_deterministic(8, max_k, false);
        if next_k != cur_k {
            down_transitions += 1;
            cur_k = next_k;
        }
    }
    assert_eq!(cur_k, 2, "sustained zero accepts must collapse to K=2");
    assert!(
        down_transitions <= 2,
        "collapse should avoid intermediate graph churn, got {down_transitions} transitions"
    );

    // Sustained full-accept frames recover smoothly all the way back up to max_k (K=7).
    let mut recovered = cur_k;
    let mut up_transitions = 0usize;
    for _ in 0..24 {
        let d = recovered - 1;
        ctrl.observe_frame_seqs(8, 8 * d, 8 * d, 0);
        let next_k = ctrl.select_k_deterministic(8, max_k, false);
        if next_k != recovered {
            up_transitions += 1;
            recovered = next_k;
        }
    }
    assert_eq!(
        recovered, max_k,
        "sustained full accepts must recover back to K={max_k}"
    );
    assert!(
        up_transitions <= 3,
        "recovery should keep CUDA graph transitions minimal, got {up_transitions}"
    );

    // Unbiased ratio-of-EWMAs avoids Jensen downward bias on small cohorts (c=2):
    // alternating 0/2 (1 trial/seq) and 2*d/2*d (d trials/seq) has true MLE d/(d+1)
    // (0.80 at K=5, 0.857 at K=7), whereas naive per-frame ratio averaging collapses to 0.50.
    for d in [4usize, 6] {
        let unbiased = CohortAcceptEwma::new();
        for _ in 0..40 {
            unbiased.observe_frame_seqs(2, 0, 2 * d, 2);
            unbiased.observe_frame_seqs(2, 2 * d, 2 * d, 0);
        }
        let true_mle = d as f32 / (d + 1) as f32;
        assert!(
            unbiased.ewma_p_step() > true_mle - 0.025,
            "d={d}: ratio-of-EWMAs must stay near true MLE {true_mle:.3}, got {}",
            unbiased.ewma_p_step()
        );
    }
}

#[test]
fn cohort_controller_warm_starts_on_new_concurrency_bursts_and_idle_gaps() {
    let max_k = spark_model::layers::qwen4_verify_lanes::ADAPTIVE_MAX_K.min(7);
    let ctrl = CohortAcceptEwma::new();
    // Drive c=2 down to K=2 on sustained zero-accept frames.
    assert_eq!(ctrl.select_k_deterministic(2, max_k, false), max_k);
    for _ in 0..16 {
        ctrl.observe_frame_seqs(2, 0, 2 * (max_k - 1), 2);
        let _ = ctrl.select_k_deterministic(2, max_k, false);
    }
    assert_eq!(ctrl.select_k_deterministic(2, max_k, false), 2);
    assert!(ctrl.ewma_p_step() < 0.30);

    // 1. Sharp concurrency ramp (c=2 -> c=16) warm-starts at max_k, p=0.80 instead of K=2.
    assert_eq!(ctrl.select_k_deterministic(16, max_k, false), max_k);
    assert!((ctrl.ewma_p_step() - 0.80).abs() < 1e-6);

    // Drive c=16 down to K=2, then drain 16 -> 15 -> 14: drain keeps learned K without reset.
    for _ in 0..16 {
        ctrl.observe_frame_seqs(16, 0, 16 * (max_k - 1), 16);
        let _ = ctrl.select_k_deterministic(16, max_k, false);
    }
    assert_eq!(ctrl.select_k_deterministic(16, max_k, false), 2);
    assert_eq!(ctrl.select_k_deterministic(15, max_k, false), 2);
    assert_eq!(ctrl.select_k_deterministic(14, max_k, false), 2);

    // 2. Idle gap (`idle_expired = true`) warm-starts a fresh burst back to max_k.
    assert_eq!(ctrl.select_k_deterministic(14, max_k, true), max_k);
    assert!((ctrl.ewma_p_step() - 0.80).abs() < 1e-6);

    // 3. Dropping through solo C=1 resets the next concurrent burst (e.g. C=24 -> K=5) warm.
    for _ in 0..16 {
        ctrl.observe_frame_seqs(14, 0, 14 * (max_k - 1), 14);
        let _ = ctrl.select_k_deterministic(14, max_k, false);
    }
    assert_eq!(ctrl.select_k_deterministic(14, max_k, false), 2);
    assert_eq!(
        resolve_cohort_width_with_controller(true, true, true, None, 1, 6, &ctrl),
        6
    );
    assert_eq!(ctrl.select_k_deterministic(24, 5, false), 5);
    assert!((ctrl.ewma_p_step() - 0.80).abs() < 1e-6);
}

#[test]
fn scheduler_env_parsers_handle_enabled_disabled_and_invalid_values() {
    use super::{
        parse_batched_k5_log_enabled, parse_dflash_gumbel_gamma_override,
        parse_dflash_pcum_thresh_override, parse_grammar_serial_verify,
    };

    assert!(!parse_grammar_serial_verify(None));
    assert!(parse_grammar_serial_verify(Some("1")));
    assert!(parse_grammar_serial_verify(Some(" 1 ")));
    assert!(!parse_grammar_serial_verify(Some("0")));
    assert!(!parse_grammar_serial_verify(Some("true")));

    assert!(!parse_batched_k5_log_enabled(None));
    assert!(parse_batched_k5_log_enabled(Some("1")));
    assert!(!parse_batched_k5_log_enabled(Some("0")));
    assert!(!parse_batched_k5_log_enabled(Some(" 1 ")));

    assert_eq!(parse_dflash_gumbel_gamma_override(None), None);
    assert_eq!(parse_dflash_gumbel_gamma_override(Some("5")), Some(5));
    assert_eq!(parse_dflash_gumbel_gamma_override(Some("0")), Some(0));
    assert_eq!(parse_dflash_gumbel_gamma_override(Some("99")), None);

    assert_eq!(parse_dflash_pcum_thresh_override(None), None);
    assert_eq!(parse_dflash_pcum_thresh_override(Some("0.25")), Some(0.25));
    assert_eq!(parse_dflash_pcum_thresh_override(Some("0")), Some(0.0));
    assert_eq!(parse_dflash_pcum_thresh_override(Some("2.5")), None);
}

#[test]
fn sm120_256_row_frame_budget_unlocks_c48_c64_k4_and_c24_c32_dflash2_k8() {
    use super::{
        adaptive_drafts_with_max_rows, plan_batched_uniform_with_max_rows,
        plan_ragged_cohort_with_max_rows,
    };

    // Native MTP C=48/64 under 256-row budget: K=4 (3 drafts) fits in ONE frame (192/256 rows).
    assert_eq!(adaptive_drafts_with_max_rows(48, 3, 256), 3);
    assert_eq!(adaptive_drafts_with_max_rows(48, 6, 256), 4);
    assert_eq!(adaptive_drafts_with_max_rows(64, 6, 256), 3);
    let c64_k4: Vec<_> = (0..64).map(|i| (i, Some(4))).collect();
    let plan_c64 = plan_batched_uniform_with_max_rows(&c64_k4, 256);
    assert_eq!(plan_c64.frames, vec![(0..64).collect::<Vec<_>>()]);
    assert!(plan_c64.leftover.is_empty());

    // DFlash2 C=24/32 under 256-row budget: full gamma=7, K=8 fits in ONE frame (192/256 rows).
    assert_eq!(adaptive_drafts_with_max_rows(24, 7, 256), 7);
    assert_eq!(adaptive_drafts_with_max_rows(32, 7, 256), 7);
    let c32_k8: Vec<_> = (0..32).map(|i| (i, Some(8))).collect();
    let plan_c32 = plan_batched_uniform_with_max_rows(&c32_k8, 256);
    assert_eq!(plan_c32.frames, vec![(0..32).collect::<Vec<_>>()]);
    assert!(plan_c32.leftover.is_empty());

    // Ragged packing: 32 x K=8 packs into 1 frame (256 rows); 33 x K=8 splits into 2 frames.
    let plan_r32 = plan_ragged_cohort_with_max_rows(&c32_k8, 256);
    assert_eq!(plan_r32.frames.len(), 1);
    assert_eq!(plan_r32.frames[0].len(), 32);
    let c33_k8: Vec<_> = (0..33).map(|i| (i, Some(8))).collect();
    let plan_r33 = plan_ragged_cohort_with_max_rows(&c33_k8, 256);
    assert_eq!(plan_r33.frames.len(), 2);
    assert!(plan_r33.leftover.is_empty());
    assert!(
        plan_r33
            .frames
            .iter()
            .all(|f| f.len() >= 2 && f.len() * 8 <= 256)
    );

    // GB10 128-row budget keeps C=64 at K=2 (1 draft) and C=32 at K=4 (3 drafts).
    assert_eq!(adaptive_drafts_with_max_rows(64, 6, 128), 1);
    assert_eq!(adaptive_drafts_with_max_rows(32, 7, 128), 3);
}
