// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

#[test]
fn dflash_gumbel_gamma_override_parser_accepts_zero_and_one_through_seven() {
    assert_eq!(parse_dflash_gumbel_gamma_override(None), None);
    assert_eq!(parse_dflash_gumbel_gamma_override(Some("")), None);
    assert_eq!(parse_dflash_gumbel_gamma_override(Some("   ")), None);
    assert_eq!(parse_dflash_gumbel_gamma_override(Some("0")), Some(0));
    assert_eq!(parse_dflash_gumbel_gamma_override(Some(" 0 ")), Some(0));
    for v in 1..=7usize {
        assert_eq!(
            parse_dflash_gumbel_gamma_override(Some(&v.to_string())),
            Some(v),
            "gamma={v}"
        );
    }
    for bad in ["8", "15", "-1", "auto", "4.0"] {
        assert_eq!(
            parse_dflash_gumbel_gamma_override(Some(bad)),
            None,
            "invalid override {bad:?} must fall back to default"
        );
    }
}

#[test]
fn dflash_gumbel_width_keeps_greedy_full_and_narrows_non_greedy_with_overrides() {
    use crate::scheduler::stochastic_verify_walk::SpecVerifyPolicy;

    // Greedy (`T == 0`) keeps full γ=7 across C=1, C=5, C=8, C=16.
    for live in [1usize, 2, 5, 8, 9, 16] {
        assert_eq!(
            resolve_dflash_gumbel_num_drafts(true, SpecVerifyPolicy::Gumbel, false, live, 7, None),
            7,
            "greedy cohort at live={live} must keep full gamma=7"
        );
    }

    // Non-greedy (`T > 0`) under Gumbel + DFlash follows the calibrated
    // cohort ladder: C<=2 -> 6, C<=4 -> 5, C>4 -> 4.
    for (live, expected) in [
        (1usize, 6usize),
        (2, 6),
        (3, 5),
        (4, 5),
        (5, 4),
        (8, 4),
        (12, 4),
        (16, 4),
    ] {
        assert_eq!(
            resolve_dflash_gumbel_num_drafts(true, SpecVerifyPolicy::Gumbel, true, live, 7, None),
            expected,
            "non-greedy DFlash2 Gumbel at live={live} must resolve to gamma={expected}"
        );
    }

    // Non-greedy (`T > 0`) under Rejection + DFlash keeps gamma=7 for C<=8 and
    // narrows C>8 (C=16) to gamma=5 to eliminate wasted high-C drafter rows.
    for (live, expected) in [(1usize, 7usize), (2, 7), (4, 7), (8, 7), (16, 5)] {
        assert_eq!(
            resolve_dflash_gumbel_num_drafts(
                true,
                SpecVerifyPolicy::Rejection,
                true,
                live,
                7,
                None
            ),
            expected,
            "non-greedy DFlash2 Rejection at live={live} must resolve to gamma={expected}"
        );
    }

    // Configured ceiling is respected when narrower than default.
    assert_eq!(
        resolve_dflash_gumbel_num_drafts(true, SpecVerifyPolicy::Gumbel, true, 1, 2, None),
        2
    );
    assert_eq!(
        resolve_dflash_gumbel_num_drafts(true, SpecVerifyPolicy::Gumbel, true, 16, 2, None),
        2
    );

    // Explicit `ATLAS_QWEN4_DFLASH_GUMBEL_GAMMA=0` disables narrowing.
    for live in [1usize, 4, 8, 16] {
        assert_eq!(
            resolve_dflash_gumbel_num_drafts(
                true,
                SpecVerifyPolicy::Gumbel,
                true,
                live,
                7,
                Some(0),
            ),
            7,
            "override=0 must keep full configured gamma=7 at live={live}"
        );
    }

    // Explicit `ATLAS_QWEN4_DFLASH_GUMBEL_GAMMA=5` overrides cleanly (capped by configured).
    for live in [1usize, 4, 16] {
        assert_eq!(
            resolve_dflash_gumbel_num_drafts(
                true,
                SpecVerifyPolicy::Gumbel,
                true,
                live,
                7,
                Some(5),
            ),
            5,
            "override=5 must cap to 5 at live={live}"
        );
        assert_eq!(
            resolve_dflash_gumbel_num_drafts(
                true,
                SpecVerifyPolicy::Gumbel,
                true,
                live,
                4,
                Some(5),
            ),
            4,
            "override=5 must not exceed configured=4 at live={live}"
        );
    }

    // Non-DFlash proposers and non-Gumbel policies stay untouched.
    assert_eq!(
        resolve_dflash_gumbel_num_drafts(false, SpecVerifyPolicy::Gumbel, true, 16, 7, None),
        7
    );
    assert_eq!(
        resolve_dflash_gumbel_num_drafts(true, SpecVerifyPolicy::Argmax, true, 16, 7, None),
        7
    );
    assert_eq!(
        resolve_dflash_gumbel_num_drafts(true, SpecVerifyPolicy::Stochastic, true, 16, 7, None),
        7
    );
}

#[test]
fn dflash_gumbel_cohort_temperature_check_requires_all_live_seqs_non_greedy() {
    let mk = |slot: usize, temp: f32, finished: bool| {
        let mut a = crate::scheduler::test_fixtures::active_seq(
            crate::scheduler::ResponseSink::Blocking(None),
            slot,
        );
        a.temperature = temp;
        a.finished = finished;
        a
    };

    // Empty or all-finished -> not non-greedy.
    assert_eq!(cohort_live_and_all_non_greedy(&[]), (0, false));
    assert_eq!(
        cohort_live_and_all_non_greedy(&[mk(0, 0.8, true)]),
        (0, false)
    );

    // All live sequences T > 0 -> non-greedy; finished greedy seq is ignored.
    let non_greedy = [mk(0, 0.8, false), mk(1, 0.6, false), mk(2, 0.0, true)];
    assert_eq!(cohort_live_and_all_non_greedy(&non_greedy), (2, true));

    // Any live greedy (or NaN) sequence keeps cohort on full greedy width.
    let mixed = [mk(0, 0.8, false), mk(1, 0.0, false)];
    assert_eq!(cohort_live_and_all_non_greedy(&mixed), (2, false));

    let nan_temp = [mk(0, 0.8, false), mk(1, f32::NAN, false)];
    assert_eq!(cohort_live_and_all_non_greedy(&nan_temp), (2, false));
}

#[test]
fn dflash_pcum_truncation_narrows_uncertain_keeps_high_prob_and_respects_guards() {
    // Parser handles unset, empty, disable (0 / <=0), valid overrides, and invalid strings.
    assert_eq!(parse_dflash_pcum_thresh_override(None), None);
    assert_eq!(parse_dflash_pcum_thresh_override(Some("")), None);
    assert_eq!(parse_dflash_pcum_thresh_override(Some("   ")), None);
    assert_eq!(parse_dflash_pcum_thresh_override(Some("0")), Some(0.0));
    assert_eq!(parse_dflash_pcum_thresh_override(Some("0.0")), Some(0.0));
    assert_eq!(parse_dflash_pcum_thresh_override(Some("-0.25")), Some(0.0));
    assert_eq!(parse_dflash_pcum_thresh_override(Some("0.32")), Some(0.32));
    assert_eq!(parse_dflash_pcum_thresh_override(Some("1.0")), Some(1.0));
    for bad in ["1.01", "2.0", "NaN", "inf", "auto"] {
        assert_eq!(
            parse_dflash_pcum_thresh_override(Some(bad)),
            None,
            "invalid threshold {bad:?} must fall back to default"
        );
    }

    use crate::scheduler::stochastic_verify_walk::SpecVerifyPolicy;

    // Policy- and cohort-aware defaults for T > 0 vs T == 0.
    for (c, want_rej) in [
        (1usize, 0.55f32),
        (2, 0.55),
        (4, 0.58),
        (8, 0.58),
        (16, 0.58),
    ] {
        assert_eq!(
            super::dflash_pcum_default_for_cohort_with_policy(0.8, c, SpecVerifyPolicy::Rejection),
            want_rej,
            "Rejection T>0 must match calibrated threshold across cohort_size={c}"
        );
        assert_eq!(
            super::dflash_pcum_default_for_cohort_with_policy(0.0, c, SpecVerifyPolicy::Rejection),
            0.25,
            "greedy T==0 must stay at 0.25 across cohort_size={c}"
        );
        assert_eq!(
            super::dflash_pcum_default_for_cohort_with_policy(0.0, c, SpecVerifyPolicy::Gumbel),
            0.25,
            "greedy T==0 under Gumbel must stay at 0.25 across cohort_size={c}"
        );
    }
    assert_eq!(
        super::dflash_pcum_default_for_cohort_with_policy(0.8, 1, SpecVerifyPolicy::Gumbel),
        0.05
    );
    assert_eq!(
        super::dflash_pcum_default_for_cohort_with_policy(0.8, 2, SpecVerifyPolicy::Gumbel),
        0.05
    );
    assert_eq!(
        super::dflash_pcum_default_for_cohort_with_policy(0.8, 4, SpecVerifyPolicy::Gumbel),
        0.06
    );
    assert_eq!(
        super::dflash_pcum_default_for_cohort_with_policy(0.8, 8, SpecVerifyPolicy::Gumbel),
        0.10
    );
    assert_eq!(
        super::dflash_pcum_default_for_cohort_with_policy(0.8, 16, SpecVerifyPolicy::Gumbel),
        0.13
    );

    // T > 0 (C=1 Gumbel default tau = 0.05): narrows low-prob tails, keeps >= 1 draft (K_s >= 2).
    let low_probs = [0.20f32, 0.20, 0.20, 0.15];
    // p_cum: [0.20, 0.04 (< 0.05)] -> keeps 1 draft (K = 2).
    assert_eq!(dflash_pcum_keep_len(4, &low_probs, 0.7, None), 1);
    let mid_probs = [0.60f32, 0.50, 0.35, 0.25];
    // p_cum: [0.60, 0.30, 0.105, 0.02625].
    assert_eq!(dflash_pcum_keep_len(4, &mid_probs, 0.7, None), 3);
    let high_probs = [0.80f32, 0.70, 0.55, 0.45];
    // p_cum: [0.80, 0.56, 0.308, 0.1386 (>= 0.13)] -> keeps all 4 drafts (K = 5).
    assert_eq!(dflash_pcum_keep_len(4, &high_probs, 0.7, None), 4);
    // Cohort scaling under Gumbel: C=4 (0.06) and C=8 (0.10) keep 3 on mid_probs (0.105);
    // C=16 (0.13) trims mid_probs to 2 while keeping high_probs (0.1386) at 4.
    assert_eq!(
        super::dflash_pcum_keep_len_for_cohort(4, &mid_probs, 1.0, 4, None),
        3
    );
    assert_eq!(
        super::dflash_pcum_keep_len_for_cohort(4, &mid_probs, 1.0, 8, None),
        3
    );
    assert_eq!(
        super::dflash_pcum_keep_len_for_cohort(4, &mid_probs, 1.0, 16, None),
        2
    );
    assert_eq!(
        super::dflash_pcum_keep_len_for_cohort(4, &high_probs, 1.0, 16, None),
        4
    );
    // Under Rejection (calibrated for prefix-only `max_z q_t(z)`), C<=2 uses 0.55
    // and C>=4 uses 0.58: `high_probs` [0.80, 0.70, 0.55, 0.45] has p_cum [0.80, 0.56,
    // 0.308, 0.1386] so C=2 keeps 2 (0.56 >= 0.55) while C=16 keeps 1 (0.56 < 0.58),
    // and confident code [0.95, 0.90, 0.85, 0.80] (p_cum 0.5814 >= 0.58) keeps all 4.
    assert_eq!(
        super::dflash_pcum_keep_len_for_cohort_with_policy(
            4,
            &high_probs,
            1.0,
            2,
            SpecVerifyPolicy::Rejection,
            None,
        ),
        2
    );
    assert_eq!(
        super::dflash_pcum_keep_len_for_cohort_with_policy(
            4,
            &high_probs,
            1.0,
            16,
            SpecVerifyPolicy::Rejection,
            None,
        ),
        1
    );
    let confident_probs = [0.95f32, 0.90, 0.85, 0.80];
    assert_eq!(
        super::dflash_pcum_keep_len_for_cohort_with_policy(
            4,
            &confident_probs,
            1.0,
            16,
            SpecVerifyPolicy::Rejection,
            None,
        ),
        4
    );
    // Explicit override takes precedence over policy/cohort default.
    assert_eq!(
        super::dflash_pcum_keep_len_for_cohort_with_policy(
            4,
            &high_probs,
            1.0,
            16,
            SpecVerifyPolicy::Rejection,
            Some(0.25),
        ),
        3
    );

    // T == 0 greedy (default tau = 0.25): keeps full K=8 on high-prob code, narrows prose.
    let code_probs = [0.95f32, 0.92, 0.90, 0.88, 0.86, 0.85, 0.84];
    assert_eq!(dflash_pcum_keep_len(7, &code_probs, 0.0, None), 7);
    let prose_probs = [0.55f32, 0.50, 0.48, 0.45, 0.40, 0.35, 0.30];
    // p_cum: [0.55, 0.275 (>= 0.25), 0.132 (< 0.25)] -> keeps 2 drafts (K = 3).
    assert_eq!(dflash_pcum_keep_len(7, &prose_probs, 0.0, None), 2);

    // Override = 0.0 disables truncation completely; empty probs are a no-op.
    assert_eq!(dflash_pcum_keep_len(7, &prose_probs, 0.0, Some(0.0)), 7);
    assert_eq!(dflash_pcum_keep_len(4, &low_probs, 0.7, Some(0.0)), 4);
    assert_eq!(dflash_pcum_keep_len(7, &[], 0.7, None), 7);

    // ActiveSeq guards: non-DFlash, retrieval chains, forced nudge, and length-1 stay untouched.
    let mk_seq = |temp: f32, origin: crate::scheduler::DraftOrigin| {
        let mut a = crate::scheduler::test_fixtures::active_seq(
            crate::scheduler::ResponseSink::Blocking(None),
            0,
        );
        a.temperature = temp;
        a.draft_origin = origin;
        a.pending_drafts = vec![10, 11, 12, 13, 14, 15, 16];
        a
    };

    let mut non_dflash = mk_seq(0.7, crate::scheduler::DraftOrigin::Proposer);
    apply_dflash_pcum_truncation_with_probs(&mut non_dflash, false, &prose_probs, None);
    assert_eq!(non_dflash.pending_drafts.len(), 7);

    for origin in [
        crate::scheduler::DraftOrigin::RestStore,
        crate::scheduler::DraftOrigin::SelfContext,
    ] {
        let mut retrieved = mk_seq(0.7, origin);
        apply_dflash_pcum_truncation_with_probs(&mut retrieved, true, &prose_probs, None);
        assert_eq!(
            retrieved.pending_drafts.len(),
            7,
            "retrieval origin {origin:?} must never be truncated"
        );
    }

    let mut forced = mk_seq(0.7, crate::scheduler::DraftOrigin::Proposer);
    forced.inside_thinking = true;
    forced.force_end_thinking = true;
    forced.in_code_fence = false;
    apply_dflash_pcum_truncation_with_probs(&mut forced, true, &prose_probs, None);
    assert_eq!(
        forced.pending_drafts.len(),
        7,
        "forced close chain must never be truncated"
    );

    let mut dflash_prose = mk_seq(0.0, crate::scheduler::DraftOrigin::Proposer);
    apply_dflash_pcum_truncation_with_probs(&mut dflash_prose, true, &prose_probs, None);
    assert_eq!(dflash_prose.pending_drafts, vec![10, 11]);

    let mut dflash_low = mk_seq(0.8, crate::scheduler::DraftOrigin::Proposer);
    dflash_low.pending_drafts = vec![20, 21, 22, 23];
    apply_dflash_pcum_truncation_with_probs(&mut dflash_low, true, &low_probs, None);
    assert_eq!(
        dflash_low.pending_drafts,
        vec![20],
        "must always keep >= 1 draft (K_s >= 2)"
    );
}

#[test]
fn dflash_sm120_vs_gb10_gamma_pcum_and_ragged_bucket_parity() {
    use crate::scheduler::stochastic_verify_walk::SpecVerifyPolicy;

    // 1. Draft width under Rejection: GB10 (<=64 SMs) caps C>8 at gamma=5;
    //    SM120 (>64 SMs) preserves full gamma=7 (K=8) through C<=32.
    for gb10_sms in [48u32, 64u32] {
        for (live, want) in [(1, 7), (4, 7), (8, 7), (9, 5), (16, 5), (32, 5)] {
            assert_eq!(
                super::resolve_dflash_gumbel_num_drafts_for_sm(
                    true,
                    SpecVerifyPolicy::Rejection,
                    true,
                    live,
                    7,
                    None,
                    gb10_sms,
                ),
                want,
                "GB10 sm={gb10_sms} live={live}"
            );
        }
    }
    for sm120_sms in [65u32, 148u32, 188u32] {
        for live in [1usize, 2, 4, 8, 9, 16, 24, 32] {
            assert_eq!(
                super::resolve_dflash_gumbel_num_drafts_for_sm(
                    true,
                    SpecVerifyPolicy::Rejection,
                    true,
                    live,
                    7,
                    None,
                    sm120_sms,
                ),
                7,
                "SM120 sm={sm120_sms} live={live} must preserve gamma=7"
            );
        }
        assert_eq!(
            super::resolve_dflash_gumbel_num_drafts_for_sm(
                true,
                SpecVerifyPolicy::Rejection,
                true,
                33,
                7,
                None,
                sm120_sms,
            ),
            5,
            "SM120 sm={sm120_sms} live=33 exceeds wide cap"
        );
        // Explicit env override still wins on SM120.
        assert_eq!(
            super::resolve_dflash_gumbel_num_drafts_for_sm(
                true,
                SpecVerifyPolicy::Rejection,
                true,
                16,
                7,
                Some(5),
                sm120_sms,
            ),
            5
        );
    }

    // 2. Rejection p_cum defaults: GB10 keeps 0.55/0.58/0.58/0.58;
    //    SM120 sets 0.0 (disabled, steady uniform K=8) for C<=24 and 0.15 for C>24.
    for gb10_sms in [48u32, 64u32] {
        assert_eq!(
            super::dflash_pcum_default_for_cohort_with_policy_for_sm(
                0.8,
                2,
                SpecVerifyPolicy::Rejection,
                gb10_sms
            ),
            0.55
        );
        for c in [4usize, 8, 16, 24, 32] {
            assert_eq!(
                super::dflash_pcum_default_for_cohort_with_policy_for_sm(
                    0.8,
                    c,
                    SpecVerifyPolicy::Rejection,
                    gb10_sms
                ),
                0.58,
                "GB10 sm={gb10_sms} c={c}"
            );
        }
    }
    for sm120_sms in [65u32, 188u32] {
        for c in [1usize, 2, 4, 8, 16, 24] {
            assert_eq!(
                super::dflash_pcum_default_for_cohort_with_policy_for_sm(
                    0.8,
                    c,
                    SpecVerifyPolicy::Rejection,
                    sm120_sms
                ),
                0.0,
                "SM120 sm={sm120_sms} c={c} must default Rejection p_cum to 0.0"
            );
        }
        for c in [25usize, 32] {
            assert_eq!(
                super::dflash_pcum_default_for_cohort_with_policy_for_sm(
                    0.8,
                    c,
                    SpecVerifyPolicy::Rejection,
                    sm120_sms
                ),
                0.15,
                "SM120 sm={sm120_sms} c={c} must default Rejection p_cum to 0.15"
            );
        }
        // Greedy T==0 stays 0.25 on SM120.
        assert_eq!(
            super::dflash_pcum_default_for_cohort_with_policy_for_sm(
                0.0,
                16,
                SpecVerifyPolicy::Rejection,
                sm120_sms
            ),
            0.25
        );
    }

    // 3. Ragged row bucket granularity: GB10 keeps 8 at C=16 and 16 at C>=24;
    //    SM120 widens to 16 at C=12..23 and 32 at C>=24.
    assert_eq!(super::dflash_ragged_row_bucket_step_for_sm(8, 48), 8);
    assert_eq!(super::dflash_ragged_row_bucket_step_for_sm(16, 48), 8);
    assert_eq!(super::dflash_ragged_row_bucket_step_for_sm(24, 48), 16);
    assert_eq!(super::dflash_ragged_row_bucket_step_for_sm(32, 48), 16);

    assert_eq!(super::dflash_ragged_row_bucket_step_for_sm(2, 188), 2);
    assert_eq!(super::dflash_ragged_row_bucket_step_for_sm(4, 188), 4);
    assert_eq!(super::dflash_ragged_row_bucket_step_for_sm(8, 188), 8);
    assert_eq!(super::dflash_ragged_row_bucket_step_for_sm(12, 188), 16);
    assert_eq!(super::dflash_ragged_row_bucket_step_for_sm(16, 188), 16);
    assert_eq!(super::dflash_ragged_row_bucket_step_for_sm(23, 188), 16);
    assert_eq!(super::dflash_ragged_row_bucket_step_for_sm(24, 188), 32);
    assert_eq!(super::dflash_ragged_row_bucket_step_for_sm(32, 188), 32);
}

#[test]
fn dflash_ragged_bucket_quantization_matches_verify_graph_grid_and_preserves_floor() {
    assert_eq!(dflash_ragged_row_bucket_step(1), 1);
    assert_eq!(dflash_ragged_row_bucket_step(2), 2);
    assert_eq!(dflash_ragged_row_bucket_step(3), 4);
    assert_eq!(dflash_ragged_row_bucket_step(4), 4);
    assert_eq!(dflash_ragged_row_bucket_step(5), 4);
    assert_eq!(dflash_ragged_row_bucket_step(6), 8);
    assert_eq!(dflash_ragged_row_bucket_step(8), 8);
    assert_eq!(dflash_ragged_row_bucket_step(16), 8);
    assert_eq!(dflash_ragged_row_bucket_step(24), 16);

    // Uniform cohorts are always graph-admitted and must NEVER be trimmed.
    let mut uniform3 = [4usize, 4, 4]; // total_rows = 15 (not % 4), uniform -> untouched
    let pcum3 = vec![vec![0.9, 0.8, 0.7, 0.6]; 3];
    quantize_ragged_lens_to_graph_bucket(&mut uniform3, &pcum3);
    assert_eq!(uniform3, [4, 4, 4]);

    // b = 2 (step = 2): ragged [4, 1] -> total_rows = 7 -> snaps to 6 ([3, 1]).
    let mut b2 = [4usize, 1];
    let pcum2 = vec![vec![0.9, 0.8, 0.7, 0.4], vec![0.2]];
    quantize_ragged_lens_to_graph_bucket(&mut b2, &pcum2);
    assert_eq!(b2, [3, 1]);
    assert_eq!(b2.iter().map(|&l| l + 1).sum::<usize>() % 2, 0);

    // b = 4 (step = 4): trims lowest tail-p_cum drafts first while keeping >= 1 per lane.
    let mut b4 = [4usize, 3, 2, 1]; // total_rows = 14 (rem = 2) -> snaps to 12.
    let pcum4 = vec![
        vec![0.95, 0.90, 0.85, 0.80], // tail 0.80 (high confidence -> preserved)
        vec![0.70, 0.55, 0.35],       // tail 0.35 (lowest -> trimmed first, then 0.55)
        vec![0.60, 0.40],             // tail 0.40 (second lowest -> trimmed second)
        vec![0.20],                   // at K=2 floor (len=1 -> protected)
    ];
    quantize_ragged_lens_to_graph_bucket(&mut b4, &pcum4);
    assert_eq!(b4, [4, 2, 1, 1]);
    assert_eq!(b4.iter().map(|&l| l + 1).sum::<usize>(), 12);

    // b = 8 (step = 8) and b = 16 (step = 8): total_rows aligns to multiple of 8.
    let mut b8 = [4usize, 4, 3, 3, 2, 2, 1, 1]; // total_rows = 28 (rem = 4) -> 24
    let pcum8: Vec<Vec<f32>> = b8
        .iter()
        .enumerate()
        .map(|(s, &l)| {
            (0..l)
                .map(|i| 0.9 - 0.05 * s as f32 - 0.1 * i as f32)
                .collect()
        })
        .collect();
    quantize_ragged_lens_to_graph_bucket(&mut b8, &pcum8);
    assert_eq!(b8.iter().map(|&l| l + 1).sum::<usize>(), 24);
    assert!(b8.iter().all(|&l| l >= 1));

    let mut b16 = [3usize, 3, 3, 3, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1]; // total_rows = 46 (rem = 6) -> 40
    let pcum16: Vec<Vec<f32>> = b16
        .iter()
        .map(|&l| (0..l).map(|i| 0.85 - 0.15 * i as f32).collect())
        .collect();
    quantize_ragged_lens_to_graph_bucket(&mut b16, &pcum16);
    assert_eq!(b16.iter().map(|&l| l + 1).sum::<usize>(), 40);
    assert!(b16.iter().all(|&l| l >= 1));

    // Floor case (b = 6, total_rows = 13, rem = 5): stops at K=2 floor (12 == 2 * 6).
    let mut floor6 = [2usize, 1, 1, 1, 1, 1];
    let pcum6: Vec<Vec<f32>> = floor6.iter().map(|&l| vec![0.5; l]).collect();
    quantize_ragged_lens_to_graph_bucket(&mut floor6, &pcum6);
    assert_eq!(floor6, [1, 1, 1, 1, 1, 1]);
}

#[test]
fn dflash_ragged_bucket_quantization_skips_r56_cliff_for_large_cohorts() {
    // b = 16, exact r = 56 (ragged): must snap down by 8 rows to r = 48 while keeping len >= 1.
    let mut exact56 = [4usize, 4, 4, 4, 3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1];
    assert_eq!(exact56.iter().map(|&l| l + 1).sum::<usize>(), 56);
    let pcum56: Vec<Vec<f32>> = exact56
        .iter()
        .map(|&l| (0..l).map(|i| 0.90 - 0.15 * i as f32).collect())
        .collect();
    quantize_ragged_lens_to_graph_bucket(&mut exact56, &pcum56);
    assert_eq!(exact56.iter().map(|&l| l + 1).sum::<usize>(), 48);
    assert!(exact56.iter().all(|&l| l >= 1));

    // b = 16, r = 60 (quantizes to 56 -> snaps down to 48 by trimming 12 rows).
    let mut over56 = [4usize, 4, 4, 4, 4, 4, 3, 3, 2, 2, 2, 2, 2, 2, 1, 1];
    assert_eq!(over56.iter().map(|&l| l + 1).sum::<usize>(), 60);
    let pcum60: Vec<Vec<f32>> = over56
        .iter()
        .map(|&l| (0..l).map(|i| 0.90 - 0.15 * i as f32).collect())
        .collect();
    quantize_ragged_lens_to_graph_bucket(&mut over56, &pcum60);
    assert_eq!(over56.iter().map(|&l| l + 1).sum::<usize>(), 48);
    assert!(over56.iter().all(|&l| l >= 1));

    // b = 12 (threshold boundary), r = 56 -> snaps down to 48.
    let mut b12 = [5usize, 5, 5, 5, 4, 4, 4, 4, 2, 2, 2, 2];
    assert_eq!(b12.iter().map(|&l| l + 1).sum::<usize>(), 56);
    let pcum12: Vec<Vec<f32>> = b12
        .iter()
        .map(|&l| (0..l).map(|i| 0.85 - 0.10 * i as f32).collect())
        .collect();
    quantize_ragged_lens_to_graph_bucket(&mut b12, &pcum12);
    assert_eq!(b12.iter().map(|&l| l + 1).sum::<usize>(), 48);
    assert!(b12.iter().all(|&l| l >= 1));

    // b = 10 (< 12), r = 56 stays at 56; b = 16 at r = 64 or r = 48 stays untouched.
    let mut b10 = [6usize, 6, 6, 5, 5, 4, 4, 4, 3, 3];
    assert_eq!(b10.iter().map(|&l| l + 1).sum::<usize>(), 56);
    let pcum10: Vec<Vec<f32>> = b10.iter().map(|&l| vec![0.5; l]).collect();
    quantize_ragged_lens_to_graph_bucket(&mut b10, &pcum10);
    assert_eq!(b10.iter().map(|&l| l + 1).sum::<usize>(), 56);

    let mut b16_64 = [5usize, 5, 5, 5, 4, 4, 4, 4, 2, 2, 2, 2, 1, 1, 1, 1];
    assert_eq!(b16_64.iter().map(|&l| l + 1).sum::<usize>(), 64);
    let pcum64: Vec<Vec<f32>> = b16_64.iter().map(|&l| vec![0.5; l]).collect();
    quantize_ragged_lens_to_graph_bucket(&mut b16_64, &pcum64);
    assert_eq!(b16_64.iter().map(|&l| l + 1).sum::<usize>(), 64);

    // End-to-end `quantize_dflash_ragged_drafts_to_graph_bucket` must NOT
    // early-return when ragged `total_rows == 56` (`56 % 8 == 0`) at `b == 16`.
    let mut active: Vec<crate::scheduler::ActiveSeq> =
        [4usize, 4, 4, 4, 3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1]
            .into_iter()
            .enumerate()
            .map(|(slot, len)| {
                let mut a = crate::scheduler::test_fixtures::active_seq(
                    crate::scheduler::ResponseSink::Blocking(None),
                    slot,
                );
                a.pending_drafts = (0..len as u32).map(|i| 100 + i).collect();
                a
            })
            .collect();
    let idxs: Vec<usize> = (0..16).collect();
    super::quantize_dflash_ragged_drafts_to_graph_bucket(&mut active, &idxs);
    let after_rows: usize = active.iter().map(|a| a.pending_drafts.len() + 1).sum();
    assert_eq!(after_rows, 48);
    assert!(active.iter().all(|a| !a.pending_drafts.is_empty()));
}

#[test]
fn dflash_ragged_bucket_quantization_prefers_ahead_streams_before_stragglers() {
    // b = 4 (step = 4): total_rows = 14 (rem = 2) -> must trim 2 rows to hit 12.
    // Stream 0 and 1 are ahead of cohort median remaining (`median = 120`, threshold `112`),
    // with higher `tail_p` than straggler stream 2 (`rem = 120`, `tail_p = 0.20`).
    // Ahead-first tie-break must trim ahead streams first (lowest `tail_p` among ahead)
    // while preserving straggler stream 2 and respecting the `len >= 1` (`K >= 2`) floor.
    let mut lens = [4usize, 2, 3, 1];
    let pcum = vec![
        vec![0.95, 0.90, 0.85, 0.80], // ahead (rem=40), tail_p=0.80 -> trimmed second (after lane 1 hits floor)
        vec![0.75, 0.65], // ahead (rem=50), tail_p=0.65 -> trimmed first (hits floor len=1)
        vec![0.40, 0.30, 0.20], // straggler (rem=120), tail_p=0.20 -> preserved!
        vec![0.10],       // straggler (rem=130), at K=2 floor
    ];
    let remaining = [40usize, 50, 120, 130];
    quantize_ragged_lens_to_graph_bucket_with_remaining(&mut lens, &pcum, Some(&remaining));
    assert_eq!(lens, [3, 1, 3, 1]);
    assert_eq!(lens.iter().map(|&l| l + 1).sum::<usize>(), 12);

    // Fallback when all ahead streams hit the `len == 1` (`K == 2`) floor before `rem == 0`:
    // b = 4 (step = 4), total_rows = 15 (rem = 3) -> snaps to 12.
    // Ahead stream 0 starts at len=2 (can only give 1 row before K=2 floor); the remaining
    // 2 trims fall back to lowest `tail_p` across non-ahead streams (lane 2 then lane 1).
    let mut fallback = [2usize, 4, 4, 1];
    let pcum_fb = vec![
        vec![0.90, 0.85],             // ahead (rem=30): gives 1 row -> len=1 (floor)
        vec![0.95, 0.90, 0.85, 0.80], // non-ahead (rem=108 >= 102), tail=0.80 -> preserved
        vec![0.70, 0.55, 0.40, 0.30], // straggler (rem=110), tails 0.30, 0.40 -> gives 2 rows
        vec![0.10],                   // straggler (rem=120), at floor
    ];
    let rem_fb = [30usize, 108, 110, 120];
    quantize_ragged_lens_to_graph_bucket_with_remaining(&mut fallback, &pcum_fb, Some(&rem_fb));
    assert_eq!(fallback, [1, 4, 2, 1]);
    assert_eq!(fallback.iter().map(|&l| l + 1).sum::<usize>(), 12);

    // When all streams have equal remaining tokens (none < median - 8), behaves identically
    // to pure lowest-`tail_p` trimming and never touches uniform cohorts.
    let mut equal_rem = [4usize, 3, 2, 1];
    let mut baseline = equal_rem;
    let pcum_eq = vec![
        vec![0.95, 0.90, 0.85, 0.80],
        vec![0.70, 0.55, 0.35],
        vec![0.60, 0.40],
        vec![0.20],
    ];
    let same = [100usize; 4];
    quantize_ragged_lens_to_graph_bucket(&mut baseline, &pcum_eq);
    quantize_ragged_lens_to_graph_bucket_with_remaining(&mut equal_rem, &pcum_eq, Some(&same));
    assert_eq!(equal_rem, baseline);

    let mut uniform = [3usize, 3, 3, 3];
    let pcum_u = vec![vec![0.8, 0.6, 0.4]; 4];
    quantize_ragged_lens_to_graph_bucket_with_remaining(
        &mut uniform,
        &pcum_u,
        Some(&[20, 100, 110, 120]),
    );
    assert_eq!(uniform, [3, 3, 3, 3]);
}

#[test]
fn dflash_ragged_upward_bucket_quantization_aligns_grid_skips_r56_and_never_trims_down() {
    // b = 2 (step = 2): ragged [4, 1] -> total_rows = 7 -> snaps UP to 8 ([4, 2] or [5, 1]).
    // Stream 0 has higher stopping-frontier pcum (0.055 vs 0.020), so stream 0 extends to 5.
    let mut b2 = [4usize, 1];
    let max2 = [7usize, 7];
    let fp2 = [0.055f32, 0.020];
    super::quantize_ragged_lens_upward_to_graph_bucket_with_remaining(&mut b2, &max2, &fp2, None);
    assert_eq!(b2, [5, 1]);
    assert_eq!(b2.iter().map(|&l| l + 1).sum::<usize>(), 8);

    // b = 4 (step = 4): prefers stragglers (remaining >= median - 8) first,
    // then smallest increment, then highest frontier_pcum.
    let mut b4 = [3usize, 2, 2, 1]; // total_rows = 12 (aligned) -> untouched
    let max4 = [7usize; 4];
    let fp4 = [0.04f32, 0.05, 0.03, 0.02];
    super::quantize_ragged_lens_upward_to_graph_bucket_with_remaining(&mut b4, &max4, &fp4, None);
    assert_eq!(b4, [3, 2, 2, 1]);

    let mut b4_unaligned = [4usize, 2, 2, 1]; // total_rows = 13 -> needs +3 to hit 16
    let rem4 = [30usize, 115, 120, 130]; // sorted[2] = 120, threshold = 112: lane 0 ahead, lanes 1,2,3 behind
    super::quantize_ragged_lens_upward_to_graph_bucket_with_remaining(
        &mut b4_unaligned,
        &max4,
        &fp4,
        Some(&rem4),
    );
    // Behind lanes 1, 2, 3 each receive +1 (smallest added first), lane 0 untouched.
    assert_eq!(b4_unaligned, [4, 3, 3, 2]);
    assert_eq!(b4_unaligned.iter().map(|&l| l + 1).sum::<usize>(), 16);

    // b = 16, r = 56 already aligns to step = 8: upward quantization stays at 56
    // (never adds a wasteful +8 jump to 64 and never trims downward below p_cum stop).
    let mut exact56 = [4usize, 4, 4, 4, 3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1];
    let orig56 = exact56;
    let max16 = [5usize; 16];
    let fp16 = [0.08f32; 16];
    assert_eq!(exact56.iter().map(|&l| l + 1).sum::<usize>(), 56);
    super::quantize_ragged_lens_upward_to_graph_bucket_with_remaining(
        &mut exact56,
        &max16,
        &fp16,
        None,
    );
    assert_eq!(exact56.iter().map(|&l| l + 1).sum::<usize>(), 56);
    assert_eq!(exact56, orig56);

    // b = 16, r = 53 (gamma = 5 ceiling): snaps UP minimally by +3 to 56 (not +11 to 64).
    let mut r53 = [4usize, 4, 4, 3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1, 1];
    let orig53 = r53;
    assert_eq!(r53.iter().map(|&l| l + 1).sum::<usize>(), 53);
    super::quantize_ragged_lens_upward_to_graph_bucket_with_remaining(
        &mut r53, &max16, &fp16, None,
    );
    assert_eq!(r53.iter().map(|&l| l + 1).sum::<usize>(), 56);
    for (&after, &before) in r53.iter().zip(orig53.iter()) {
        assert!(
            after >= before,
            "upward quantization must never trim below p_cum stop"
        );
        assert!(after <= 5, "upward quantization must respect gamma=5 cap");
    }

    // Insufficient slack: leaves lens untouched instead of trimming downward.
    let mut tight = [4usize, 1];
    let max_tight = [4usize, 1];
    super::quantize_ragged_lens_upward_to_graph_bucket_with_remaining(
        &mut tight, &max_tight, &fp2, None,
    );
    assert_eq!(tight, [4, 1]);
}

#[test]
fn dflash_rejection_upward_quantization_preserves_exact_markov_marginal_tv_zero() {
    // Non-i.i.d. 3-state Markov chain (spiky repetition-prone state 0,
    // flat state 1, medium state 2) with mismatched flattened drafter Q.
    // Verifies that prefix-only max_z q_t(z) + `dflash_pcum_keep_len_and_frontier_*`
    // + upward bucket quantization preserves exact target marginal (TV < 0.01).
    const P: [[f64; 3]; 3] = [[0.90, 0.05, 0.05], [0.35, 0.35, 0.30], [0.55, 0.25, 0.20]];
    const Q: [[f64; 3]; 3] = [[0.84, 0.08, 0.08], [0.34, 0.34, 0.32], [0.50, 0.27, 0.23]];
    let mut rng_state = 0x9E37_79B9_7F4A_7C15u64;
    let mut next_f64 = || -> f64 {
        rng_state = rng_state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = rng_state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        ((z >> 11) as f64) * (1.0 / ((1u64 << 53) as f64))
    };
    let sample_cat = |probs: [f64; 3], u: f64| -> usize {
        if u < probs[0] {
            0
        } else if u < probs[0] + probs[1] {
            1
        } else {
            2
        }
    };

    const B: usize = 4;
    const GAMMA: usize = 7;
    let mut states = [0usize; B];
    let mut counts = [[0u64; 3]; 3];
    for _ in 0..25_000 {
        let mut drafts = [[0usize; GAMMA]; B];
        let mut max_probs = [[0.0f32; GAMMA]; B];
        for s in 0..B {
            let mut cur = states[s];
            for t in 0..GAMMA {
                let u = next_f64();
                let x = sample_cat(Q[cur], u);
                drafts[s][t] = x;
                // Prefix-only max_z q_t(z | x_{<t}) (matches dflash2_selector.cu s_w[0]).
                max_probs[s][t] = Q[cur][0].max(Q[cur][1]).max(Q[cur][2]) as f32;
                cur = x;
            }
        }
        let mut lens = [GAMMA; B];
        let max_lens = [GAMMA; B];
        let mut frontiers = [1.0f32; B];
        for s in 0..B {
            let (keep, frontier) = super::dflash_pcum_keep_len_and_frontier_for_cohort_with_policy(
                GAMMA,
                &max_probs[s],
                1.0,
                B,
                crate::scheduler::stochastic_verify_walk::SpecVerifyPolicy::Rejection,
                Some(0.15),
            );
            lens[s] = keep;
            frontiers[s] = frontier;
        }
        super::quantize_ragged_lens_upward_to_graph_bucket_with_remaining(
            &mut lens, &max_lens, &frontiers, None,
        );
        for s in 0..B {
            let mut st = states[s];
            let mut rejected = false;
            for &x in &drafts[s][..lens[s]] {
                let px = P[st][x];
                let qx = Q[st][x];
                if px >= qx || next_f64() * qx < px {
                    counts[st][x] += 1;
                    st = x;
                } else {
                    let mut res = [0.0f64; 3];
                    let mut sum = 0.0f64;
                    for j in 0..3 {
                        res[j] = (P[st][j] - Q[st][j]).max(0.0);
                        sum += res[j];
                    }
                    for r in &mut res {
                        *r /= sum;
                    }
                    let y = sample_cat(res, next_f64());
                    counts[st][y] += 1;
                    st = y;
                    rejected = true;
                    break;
                }
            }
            if !rejected {
                let y = sample_cat(P[st], next_f64());
                counts[st][y] += 1;
                st = y;
            }
            states[s] = st;
        }
    }
    for i in 0..3 {
        let total = (counts[i][0] + counts[i][1] + counts[i][2]) as f64;
        let tv: f64 = 0.5
            * (0..3)
                .map(|j| ((counts[i][j] as f64) / total - P[i][j]).abs())
                .sum::<f64>();
        assert!(
            tv < 0.01,
            "row {i} marginal TV={tv:.5} exceeded 0.01 (counts={:?}, true={:?})",
            counts[i],
            P[i]
        );
    }
}
