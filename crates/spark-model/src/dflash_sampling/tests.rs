// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

struct SplitMix64(u64);

impl SplitMix64 {
    fn next(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = self.0;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        ((value ^ (value >> 31)) >> 11) as f64 / (1u64 << 53) as f64
    }
}

fn selector_fixture() -> (
    Vec<[f32; DFLASH2_SELECTOR_TOP_K]>,
    Vec<[u32; DFLASH2_SELECTOR_TOP_K]>,
    Vec<Vec<f32>>,
    Vec<Vec<f32>>,
    Vec<Vec<f32>>,
) {
    let unary = vec![[0.0; DFLASH2_SELECTOR_TOP_K]; 2];
    let ascending = std::array::from_fn(|i| i as u32);
    let descending = std::array::from_fn(|i| (DFLASH2_SELECTOR_TOP_K - 1 - i) as u32);
    let candidates = vec![ascending, descending];
    let hidden = vec![vec![1.0], vec![1.0]];
    let predecessor = (0..20).map(|token| vec![token as f32 * 0.1]).collect();
    let successor = (0..20).map(|token| vec![token as f32 * 0.05]).collect();
    (unary, candidates, hidden, predecessor, successor)
}

fn sampling_params(seed: u64) -> spark_runtime::sampler::SamplingParams {
    spark_runtime::sampler::SamplingParams {
        temperature: 1.0,
        top_k: 20,
        top_p: 0.95,
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
        max_tokens: 0,
        stop_token_ids: Vec::new(),
        seed: Some(seed),
    }
}

#[test]
fn default_and_temperature_zero_preserve_the_legacy_greedy_path() {
    assert_eq!(
        DflashDraftSampleMethod::default(),
        DflashDraftSampleMethod::Greedy
    );
    let (unary, candidates, hidden, predecessor, successor) = selector_fixture();
    let greedy = select_dflash2_path(
        DflashDraftSampleMethod::Greedy,
        1.0,
        1,
        &unary,
        &candidates,
        &hidden,
        &predecessor,
        &successor,
        &[],
    )
    .expect("legacy greedy selector");
    let sampled_flag_at_zero = select_dflash2_path(
        DflashDraftSampleMethod::Probabilistic,
        0.0,
        1,
        &unary,
        &candidates,
        &hidden,
        &predecessor,
        &successor,
        &[],
    )
    .expect("temperature zero remains greedy");
    assert_eq!(greedy.tokens, vec![15, 15]);
    assert_eq!(greedy.q_rows, None);
    assert_eq!(sampled_flag_at_zero, greedy);
    let bytes = |tokens: &[u32]| {
        tokens
            .iter()
            .flat_map(|token| token.to_le_bytes())
            .collect::<Vec<_>>()
    };
    assert_eq!(bytes(&sampled_flag_at_zero.tokens), bytes(&greedy.tokens));
}

#[test]
fn seeded_probabilistic_walk_is_sequential_and_retains_normalized_sparse_q() {
    let seed = 0x5eed_u64;
    let mut rng = SplitMix64(seed);
    let uniforms = vec![rng.next(), rng.next()];
    let (unary, candidates, hidden, predecessor, successor) = selector_fixture();
    let run = || {
        select_dflash2_path(
            DflashDraftSampleMethod::Probabilistic,
            1.0,
            1,
            &unary,
            &candidates,
            &hidden,
            &predecessor,
            &successor,
            &uniforms,
        )
        .expect("sampled selector")
    };
    let first = run();
    let second = run();
    assert_eq!(first, second, "same seed-derived draws must replay exactly");
    let rows = first.q_rows.as_ref().expect("probabilistic q rows");
    assert_eq!(rows[0].predecessor, 1);
    assert_eq!(rows[1].predecessor, first.tokens[0]);
    for row in rows {
        assert!((row.probability_sum - 1.0).abs() <= 2.0e-5);
        assert_eq!(row.selected_token, row.candidates[row.selected_index]);
    }
}

#[test]
fn sampled_q_applies_ordered_sparse_bias_and_hard_masks_before_softmax() {
    let unary = vec![[0.0; DFLASH2_SELECTOR_TOP_K]];
    let candidates = vec![std::array::from_fn(|index| index as u32)];
    let hidden = vec![vec![0.0]];
    let predecessor = vec![vec![0.0]; 20];
    let successor = vec![vec![0.0]; 20];
    let proposal = select_dflash2_path_with_q_policy(
        DflashDraftSampleMethod::Probabilistic,
        1.0,
        0,
        0.0,
        1.0,
        19,
        &unary,
        &candidates,
        &hidden,
        &predecessor,
        &successor,
        &[(0, -100.0), (1, 2.0), (2, 100.0), (3, -0.5), (3, -0.5)],
        &[2],
        &[0.1],
    )
    .unwrap();
    let row = &proposal.q_rows.as_ref().unwrap()[0];
    assert_eq!(proposal.tokens, vec![1]);
    assert_eq!(row.probabilities[2], 0.0);
    assert!(row.probabilities[0] < f32::EPSILON);
    assert!(row.probabilities[1] > row.probabilities[4]);
    let duplicate_bias_ratio = row.probabilities[3] / row.probabilities[4];
    assert!((duplicate_bias_ratio - (-1.0f32).exp()).abs() < 1.0e-6);
}

#[test]
fn sampled_q_modifier_overflow_or_degenerate_mask_fails_closed() {
    let unary = vec![[0.0; DFLASH2_SELECTOR_TOP_K]];
    let candidates = vec![std::array::from_fn(|index| index as u32)];
    let hidden = vec![vec![0.0]];
    let predecessor = vec![vec![0.0]; 20];
    let successor = vec![vec![0.0]; 20];
    let oversized = (0..=DFLASH2_MAX_Q_MODIFIERS)
        .map(|token| (token as u32, 0.0))
        .collect::<Vec<_>>();
    assert!(
        select_dflash2_path_with_q_policy(
            DflashDraftSampleMethod::Probabilistic,
            1.0,
            0,
            0.0,
            1.0,
            19,
            &unary,
            &candidates,
            &hidden,
            &predecessor,
            &successor,
            &oversized,
            &[],
            &[0.5],
        )
        .is_err()
    );
    let all_masked = (0..DFLASH2_SELECTOR_TOP_K as u32).collect::<Vec<_>>();
    assert!(
        select_dflash2_path_with_q_policy(
            DflashDraftSampleMethod::Probabilistic,
            1.0,
            0,
            0.0,
            1.0,
            19,
            &unary,
            &candidates,
            &hidden,
            &predecessor,
            &successor,
            &[],
            &all_masked,
            &[0.5],
        )
        .is_err()
    );
}

#[test]
fn sampled_q_applies_top_k_to_final_pair_adjusted_scores() {
    let mut unary = vec![[0.0; DFLASH2_SELECTOR_TOP_K]];
    unary[0][3] = 3.0;
    unary[0][7] = 2.0;
    unary[0][11] = 1.0;
    let candidates = vec![std::array::from_fn(|index| index as u32)];
    let hidden = vec![vec![0.0]];
    let codebook = vec![vec![0.0]; 20];
    let proposal = select_dflash2_path_with_q_policy(
        DflashDraftSampleMethod::Probabilistic,
        1.0,
        2,
        0.0,
        1.0,
        19,
        &unary,
        &candidates,
        &hidden,
        &codebook,
        &codebook,
        &[],
        &[],
        &[0.0],
    )
    .unwrap();
    let row = &proposal.q_rows.unwrap()[0];
    assert!(row.probabilities[3] > 0.0);
    assert!(row.probabilities[7] > 0.0);
    assert_eq!(row.probabilities[11], 0.0);
    assert_eq!(row.probabilities.iter().filter(|&&p| p > 0.0).count(), 2);
}

#[test]
fn sampled_q_applies_min_p_before_top_p_and_keeps_the_crossing_token() {
    let mut unary = vec![[-100.0; DFLASH2_SELECTOR_TOP_K]];
    unary[0][5] = 0.0;
    unary[0][1] = 0.2f32.ln();
    unary[0][9] = 0.19f32.ln();
    let candidates = vec![std::array::from_fn(|index| index as u32)];
    let hidden = vec![vec![0.0]];
    let codebook = vec![vec![0.0]; 20];

    let min_p_then_top_p = select_dflash2_path_with_q_policy(
        DflashDraftSampleMethod::Probabilistic,
        1.0,
        0,
        0.195,
        0.82,
        19,
        &unary,
        &candidates,
        &hidden,
        &codebook,
        &codebook,
        &[],
        &[],
        &[0.0],
    )
    .unwrap();
    let row = &min_p_then_top_p.q_rows.unwrap()[0];
    assert!(row.probabilities[5] > 0.0);
    assert_eq!(row.probabilities[1], 0.0);
    assert_eq!(row.probabilities[9], 0.0);

    let crossing = select_dflash2_path_with_q_policy(
        DflashDraftSampleMethod::Probabilistic,
        1.0,
        0,
        0.0,
        0.82,
        19,
        &unary,
        &candidates,
        &hidden,
        &codebook,
        &codebook,
        &[],
        &[],
        &[0.0],
    )
    .unwrap();
    let row = &crossing.q_rows.unwrap()[0];
    assert!(row.probabilities[5] > 0.0);
    assert!(
        row.probabilities[1] > 0.0,
        "top-p must retain its crossing token"
    );
    assert_eq!(row.probabilities[9], 0.0);
}

#[test]
fn sampled_q_top_k_ties_use_token_id_and_top_n_sigma_fails_closed() {
    let unary = vec![[0.0; DFLASH2_SELECTOR_TOP_K]];
    let candidates = vec![std::array::from_fn(|index| (15 - index) as u32)];
    let hidden = vec![vec![0.0]];
    let codebook = vec![vec![0.0]; 20];
    let proposal = select_dflash2_path_with_q_policy(
        DflashDraftSampleMethod::Probabilistic,
        1.0,
        1,
        0.0,
        1.0,
        19,
        &unary,
        &candidates,
        &hidden,
        &codebook,
        &codebook,
        &[],
        &[],
        &[0.0],
    )
    .unwrap();
    let row = &proposal.q_rows.unwrap()[0];
    assert_eq!(proposal.tokens, vec![0]);
    assert_eq!(row.probabilities[15], 1.0);
    assert_eq!(row.probabilities[..15], [0.0; 15]);

    let mut params = sampling_params(7);
    params.top_n_sigma = 0.5;
    let error =
        DflashDraftSamplingRequest::from_seeded_sampling_params(&params, &[], 0, 1).unwrap_err();
    assert!(error.to_string().contains("top_n_sigma"), "{error:#}");
}

#[test]
fn sparse_q_rejection_subtracts_only_candidates_and_samples_the_residual() {
    let (unary, candidates, hidden, predecessor, successor) = selector_fixture();
    let proposal = select_dflash2_path(
        DflashDraftSampleMethod::Probabilistic,
        1.0,
        1,
        &unary,
        &candidates,
        &hidden,
        &predecessor,
        &successor,
        &[0.0, 0.0],
    )
    .expect("sampled selector");
    assert_eq!(proposal.tokens, vec![0, 15]);
    let mut p0 = vec![0.0; 20];
    p0[16] = 1.0;
    let mut p1 = vec![0.0; 20];
    p1[17] = 1.0;
    let mut p2 = vec![0.0; 20];
    p2[18] = 1.0;
    let accepted = accept_sparse_dflash(&[p0, p1, p2], &proposal, &[0.5, 0.5], 0.25)
        .expect("sparse rejection sample");
    assert_eq!(accepted.accepted, 0);
    assert_eq!(accepted.next_token, 16);
    assert_eq!(accepted.branch, DflashResidualBranch::RejectionResidual);
    assert_eq!(accepted.residual_mass_before_normalize, Some(1.0));
    assert_eq!(accepted.acceptance_probabilities, vec![0.0]);
}

#[test]
fn schema_seven_receipt_round_trips_and_rejects_unknown_or_drifted_fields() {
    let (unary, candidates, hidden, predecessor, successor) = selector_fixture();
    let params = sampling_params(0x5eed);
    let request = DflashDraftSamplingRequest::from_seeded_sampling_params(&params, &[], 0, 2)
        .expect("seeded request")
        .with_target_forbidden_token_ids(vec![7, 3])
        .expect("target policy mask")
        .with_target_policy_phase(DflashTargetPolicyPhase::InsideThinking)
        .expect("thinking policy phase");
    let draft_uniforms = request
        .draft_uniforms
        .iter()
        .map(|&value| f64::from(value))
        .collect::<Vec<_>>();
    let proposal = select_dflash2_path_with_q_policy(
        DflashDraftSampleMethod::Probabilistic,
        1.0,
        request.request_sampler.top_k,
        request.request_sampler.min_p,
        request.request_sampler.top_p,
        1,
        &unary,
        &candidates,
        &hidden,
        &predecessor,
        &successor,
        &request.request_sampler.logit_bias,
        &request.target_forbidden_token_ids,
        &draft_uniforms,
    )
    .expect("sampled selector");
    let mut target = vec![vec![0.0; 20]; 3];
    target[0][proposal.tokens[0] as usize] = 1.0;
    target[1][proposal.tokens[1] as usize] = 1.0;
    target[2][18] = 1.0;
    let acceptance = accept_sparse_dflash(
        &target,
        &proposal,
        &request.accept_uniforms,
        request.residual_uniform,
    )
    .expect("sparse acceptance");
    let target_evidence = DflashTargetEvidence::from_evaluation(
        &vec![0; target.len() * 20 * size_of::<f32>()],
        DflashTargetLogitsFormat::Fp32,
        20,
        &target,
    )
    .expect("target evidence");
    let committed_drafts = acceptance.accepted;
    let next_token = acceptance.next_token;
    let verify_rows = proposal.tokens.len() + 1;
    let last_token_before = proposal.q_rows.as_ref().unwrap()[0].predecessor;
    let seq_len_before = 41;
    let commit = DflashCommitReceipt {
        ordinal: 3,
        rule: DflashCanonicalCommitRule::RootAndCommittedDraftsNextTokenUnfed,
        verify_rows,
        committed_drafts,
        total_target_slots: committed_drafts + 1,
        next_disposition: DflashNextTokenDisposition::OutputAppendedAndArmed,
        next_was_structural: false,
        think_end_token: Some(9),
        next_was_think_end: false,
        commit_call_returned_ok: true,
        seq_len_before,
        seq_len_after: seq_len_before + committed_drafts + 1,
        finished: false,
        output_tokens_before: 0,
        output_tokens_after: committed_drafts + 1,
        last_token_before,
        last_token_after: next_token,
        think_end_snapshot_requested: false,
        think_end_snapshot_ordered: false,
    };
    let receipt = DflashSamplingReceipt::new(
        request.clone(),
        20,
        proposal.clone(),
        acceptance.clone(),
        target_evidence.clone(),
        commit.clone(),
    )
    .expect("valid receipt");
    assert_eq!(
        receipt.target_policy_phase,
        DflashTargetPolicyPhase::InsideThinking
    );
    assert_eq!(
        receipt.target_policy_sha256,
        receipt.request.target_policy_sha256
    );
    assert_eq!(
        receipt.target_distribution,
        DflashTargetDistribution::PostRequestSamplerRollingHistoryWithForbiddenTokens
    );
    assert_eq!(receipt.request.target_forbidden_token_ids, vec![3, 7]);
    assert_eq!(
        receipt.candidate_source,
        DflashCandidateSource::RawDraftLogitsTop16BeforeRequestPolicy
    );
    assert_eq!(
        receipt.q_normalization,
        DflashQNormalization::PairAdjustedThenBiasMaskTopKSoftmaxMinPTopP
    );
    let line = receipt.to_json_line().expect("serialize receipt");
    assert_eq!(
        DflashSamplingReceipt::parse_json_line(&line).unwrap(),
        receipt
    );

    let unknown = line.replacen('{', "{\"unknown\":true,", 1);
    assert!(DflashSamplingReceipt::parse_json_line(&unknown).is_err());
    let stale = line.replacen("\"schema\":7", "\"schema\":6", 1);
    assert!(DflashSamplingReceipt::parse_json_line(&stale).is_err());
    assert!(DflashSamplingReceipt::parse_json_line(&format!(" {line}")).is_err());

    let mut phase_drift = receipt.clone();
    phase_drift.target_policy_phase = DflashTargetPolicyPhase::PostThinking;
    assert!(phase_drift.validate().is_err());
    let mut digest_drift = receipt.clone();
    digest_drift.target_policy_sha256[0] ^= 1;
    assert!(digest_drift.validate().is_err());
    let mut nested_policy_drift = receipt.clone();
    nested_policy_drift
        .request
        .request_sampler
        .logit_bias
        .push((9, -1.0));
    assert!(nested_policy_drift.validate().is_err());
    let mut nested_phase_drift = receipt.request.clone();
    nested_phase_drift.target_policy_phase = DflashTargetPolicyPhase::PostThinking;
    assert!(nested_phase_drift.validate(2).is_err());
    let mut nested_mask_drift = receipt.request.clone();
    nested_mask_drift.target_forbidden_token_ids.push(8);
    assert!(nested_mask_drift.validate(2).is_err());
    let mut target_digest_drift = receipt.clone();
    target_digest_drift
        .target_evidence
        .target_probabilities_sha256[0] ^= 1;
    assert!(
        target_digest_drift
            .validate()
            .unwrap_err()
            .to_string()
            .contains("target probability digest drift")
    );
    let mut raw_logits_provenance_drift = receipt.clone();
    raw_logits_provenance_drift.target_evidence.logits_sha256[0] ^= 1;
    assert!(
        raw_logits_provenance_drift
            .validate()
            .unwrap_err()
            .to_string()
            .contains("execution digest drift")
    );
    let mut fabricated_evidence = target_evidence.clone();
    fabricated_evidence.target_probabilities_sha256 = [1; 32];
    assert!(
        DflashSamplingReceipt::new(
            request.clone(),
            20,
            proposal.clone(),
            acceptance.clone(),
            fabricated_evidence,
            commit.clone(),
        )
        .unwrap_err()
        .to_string()
        .contains("target probability digest drift")
    );
    let mut altered_acceptance = acceptance.clone();
    altered_acceptance.next_token = (altered_acceptance.next_token + 1) % 20;
    let mut altered_acceptance_commit = commit.clone();
    altered_acceptance_commit.last_token_after = altered_acceptance.next_token;
    assert!(
        DflashSamplingReceipt::new(
            request.clone(),
            20,
            proposal.clone(),
            altered_acceptance,
            target_evidence.clone(),
            altered_acceptance_commit,
        )
        .unwrap_err()
        .to_string()
        .contains("exact target-p/q replay")
    );
    let mut commit_drift = receipt.clone();
    commit_drift.commit.committed_drafts += 1;
    assert!(commit_drift.validate().is_err());
    let mut premature_next = receipt.clone();
    premature_next.commit.committed_drafts -= 1;
    premature_next.commit.total_target_slots -= 1;
    premature_next.commit.seq_len_after -= 1;
    assert!(
        premature_next
            .validate()
            .unwrap_err()
            .to_string()
            .contains("attempted next token before committing")
    );
    let mut zero_growth = receipt.clone();
    zero_growth.commit.output_tokens_after = zero_growth.commit.output_tokens_before;
    assert!(
        zero_growth
            .validate()
            .unwrap_err()
            .to_string()
            .contains("output-token delta")
    );
    let mut bogus_snapshot = receipt.clone();
    bogus_snapshot.commit.next_was_structural = true;
    bogus_snapshot.commit.next_was_think_end = true;
    bogus_snapshot.commit.think_end_token = Some((next_token + 1) % 20);
    bogus_snapshot.commit.think_end_snapshot_requested = true;
    bogus_snapshot.commit.think_end_snapshot_ordered = true;
    assert!(
        bogus_snapshot
            .validate()
            .unwrap_err()
            .to_string()
            .contains("think-end classification drift")
    );
    // The ordinary thinking path requests its boundary snapshot before its
    // final timeout check.  A timeout may therefore make the final receipt
    // terminal without invalidating the already-requested, ordered snapshot.
    let mut timed_out_after_snapshot_request = commit.clone();
    timed_out_after_snapshot_request.next_was_structural = true;
    timed_out_after_snapshot_request.think_end_token = Some(next_token);
    timed_out_after_snapshot_request.next_was_think_end = true;
    timed_out_after_snapshot_request.finished = true;
    timed_out_after_snapshot_request.think_end_snapshot_requested = true;
    timed_out_after_snapshot_request.think_end_snapshot_ordered = true;
    DflashSamplingReceipt::new(
        request.clone(),
        20,
        proposal.clone(),
        acceptance.clone(),
        target_evidence.clone(),
        timed_out_after_snapshot_request,
    )
    .expect("snapshot request preceding final timeout remains an honest receipt");

    let mut ordered_without_request = commit.clone();
    ordered_without_request.think_end_snapshot_ordered = true;
    assert!(
        DflashSamplingReceipt::new(
            request.clone(),
            20,
            proposal.clone(),
            acceptance.clone(),
            target_evidence.clone(),
            ordered_without_request,
        )
        .unwrap_err()
        .to_string()
        .contains("snapshot outcome is inconsistent")
    );
    let mut requested_for_non_boundary = commit.clone();
    requested_for_non_boundary.think_end_snapshot_requested = true;
    requested_for_non_boundary.think_end_snapshot_ordered = true;
    assert!(
        DflashSamplingReceipt::new(
            request.clone(),
            20,
            proposal.clone(),
            acceptance.clone(),
            target_evidence.clone(),
            requested_for_non_boundary,
        )
        .unwrap_err()
        .to_string()
        .contains("snapshot request is inconsistent")
    );
    let mut nonterminal_terminal_disposition = commit.clone();
    nonterminal_terminal_disposition.next_disposition =
        DflashNextTokenDisposition::AttemptedTerminalWithoutFeedback;
    nonterminal_terminal_disposition.output_tokens_after =
        nonterminal_terminal_disposition.output_tokens_before + committed_drafts;
    nonterminal_terminal_disposition.last_token_after = if committed_drafts > 0 {
        proposal.tokens[committed_drafts - 1]
    } else {
        last_token_before
    };
    assert!(
        DflashSamplingReceipt::new(
            request.clone(),
            20,
            proposal.clone(),
            acceptance.clone(),
            target_evidence.clone(),
            nonterminal_terminal_disposition,
        )
        .unwrap_err()
        .to_string()
        .contains("terminal-without-feedback disposition requires")
    );
    for (next_was_structural, finished) in [(false, false), (true, true)] {
        let mut impossible_silent = commit.clone();
        impossible_silent.next_disposition = DflashNextTokenDisposition::ConsumedSilentlyAndArmed;
        impossible_silent.next_was_structural = next_was_structural;
        impossible_silent.finished = finished;
        impossible_silent.output_tokens_after =
            impossible_silent.output_tokens_before + committed_drafts;
        assert!(
            DflashSamplingReceipt::new(
                request.clone(),
                20,
                proposal.clone(),
                acceptance.clone(),
                target_evidence.clone(),
                impossible_silent,
            )
            .unwrap_err()
            .to_string()
            .contains("silently consumed disposition requires")
        );
    }
    let mut missing_snapshot_request = commit.clone();
    missing_snapshot_request.next_was_structural = true;
    missing_snapshot_request.think_end_token = Some(next_token);
    missing_snapshot_request.next_was_think_end = true;
    assert!(
        DflashSamplingReceipt::new(
            request.clone(),
            20,
            proposal.clone(),
            acceptance.clone(),
            target_evidence.clone(),
            missing_snapshot_request,
        )
        .unwrap_err()
        .to_string()
        .contains("armed think-end omitted its snapshot request")
    );

    let mut first_row_rejects = target.clone();
    first_row_rejects[0].fill(0.0);
    first_row_rejects[0][((proposal.tokens[0] + 1) % 20) as usize] = 1.0;
    let zero_acceptance = accept_sparse_dflash(
        &first_row_rejects,
        &proposal,
        &request.accept_uniforms,
        request.residual_uniform,
    )
    .expect("first-row rejection");
    assert_eq!(zero_acceptance.accepted, 0);
    let zero_evidence = DflashTargetEvidence::from_evaluation(
        &vec![0; first_row_rejects.len() * 20 * size_of::<f32>()],
        DflashTargetLogitsFormat::Fp32,
        20,
        &first_row_rejects,
    )
    .expect("zero-acceptance evidence");
    let mut impossible_not_attempted = commit.clone();
    impossible_not_attempted.committed_drafts = 0;
    impossible_not_attempted.total_target_slots = 1;
    impossible_not_attempted.next_disposition = DflashNextTokenDisposition::NotAttempted;
    impossible_not_attempted.seq_len_after = impossible_not_attempted.seq_len_before + 1;
    impossible_not_attempted.finished = true;
    impossible_not_attempted.output_tokens_after = impossible_not_attempted.output_tokens_before;
    impossible_not_attempted.last_token_after = impossible_not_attempted.last_token_before;
    assert!(
        DflashSamplingReceipt::new(
            request.clone(),
            20,
            proposal.clone(),
            zero_acceptance,
            zero_evidence,
            impossible_not_attempted,
        )
        .unwrap_err()
        .to_string()
        .contains("walking a nonempty accepted prefix")
    );
    let mut compact_binding_drift = receipt.clone();
    compact_binding_drift.execution_sha256[0] ^= 1;
    assert!(compact_binding_drift.validate().is_err());
}

#[test]
fn malformed_sampling_inputs_fail_closed() {
    let (unary, candidates, hidden, predecessor, successor) = selector_fixture();
    let missing_draw = select_dflash2_path(
        DflashDraftSampleMethod::Probabilistic,
        1.0,
        1,
        &unary,
        &candidates,
        &hidden,
        &predecessor,
        &successor,
        &[0.5],
    );
    assert!(missing_draw.is_err());
    let bad_temperature = select_dflash2_path(
        DflashDraftSampleMethod::Probabilistic,
        f32::NAN,
        1,
        &unary,
        &candidates,
        &hidden,
        &predecessor,
        &successor,
        &[0.5, 0.5],
    );
    assert!(bad_temperature.is_err());

    validate_dflash_runtime_admission(DflashDraftSampleMethod::Greedy, true, false)
        .expect("released greedy runtime remains admitted");
    let inert_qualification =
        validate_dflash_runtime_admission(DflashDraftSampleMethod::Greedy, true, true)
            .expect_err("sampled qualification must not be inert on greedy");
    assert!(
        inert_qualification.to_string().contains("greedy selector"),
        "{inert_qualification:#}"
    );
    let missing_qualification =
        validate_dflash_runtime_admission(DflashDraftSampleMethod::Probabilistic, true, false)
            .expect_err("probabilistic mode must be explicitly qualified");
    assert!(
        missing_qualification
            .to_string()
            .contains("--experimental-dflash-probabilistic"),
        "{missing_qualification:#}"
    );
    let missing_selector =
        validate_dflash_runtime_admission(DflashDraftSampleMethod::Probabilistic, false, true)
            .expect_err("probabilistic mode needs DFlash2");
    assert!(missing_selector.to_string().contains("candidate selector"));
    validate_dflash_runtime_admission(DflashDraftSampleMethod::Probabilistic, true, true)
        .expect("qualified DFlash2 selector is admitted to the narrow runtime path");
}

#[test]
fn softmax_subtracts_the_max_before_temperature_scaling() {
    let mut unary = vec![[f32::MAX; DFLASH2_SELECTOR_TOP_K]];
    unary[0][1] = f32::MAX / 2.0;
    let candidates = vec![std::array::from_fn(|index| index as u32)];
    let hidden = vec![vec![0.0]];
    let codebook = vec![vec![0.0]; DFLASH2_SELECTOR_TOP_K];
    let proposal = select_dflash2_path(
        DflashDraftSampleMethod::Probabilistic,
        0.5,
        0,
        &unary,
        &candidates,
        &hidden,
        &codebook,
        &codebook,
        &[0.0],
    )
    .expect("stable max-before-scale softmax");
    let row = &proposal.q_rows.unwrap()[0];
    assert_eq!(row.selected_index, 0);
    assert!(row.probabilities.iter().all(|value| value.is_finite()));
}

fn uniform_q_frame(
    params: &spark_runtime::sampler::SamplingParams,
    history: &[u32],
    gamma: usize,
) -> DflashSampledProposalFrame {
    uniform_q_frame_with_masks(params, history, gamma, &[])
}

fn uniform_q_frame_with_masks(
    params: &spark_runtime::sampler::SamplingParams,
    history: &[u32],
    gamma: usize,
    forbidden_token_ids: &[u32],
) -> DflashSampledProposalFrame {
    let request = DflashDraftSamplingRequest::from_seeded_sampling_params(
        params,
        history,
        history.len(),
        gamma,
    )
    .unwrap()
    .with_target_forbidden_token_ids(forbidden_token_ids.to_vec())
    .unwrap();
    let candidates = std::array::from_fn(|index| index as u32);
    let mut probabilities = [1.0; DFLASH2_SELECTOR_TOP_K];
    for (candidate, probability) in candidates.iter().zip(probabilities.iter_mut()) {
        if request.target_forbidden_token_ids.contains(candidate) {
            *probability = 0.0;
        }
    }
    let probability_sum = probabilities.iter().sum::<f32>();
    assert!(probability_sum > 0.0);
    for probability in &mut probabilities {
        *probability /= probability_sum;
    }
    let selected_index = |uniform: f32| {
        let mut cumulative = 0.0f32;
        for (index, probability) in probabilities.iter().enumerate() {
            cumulative += probability;
            if cumulative > uniform {
                return index;
            }
        }
        DFLASH2_SELECTOR_TOP_K - 1
    };
    let mut predecessor = history.last().copied().unwrap_or(0);
    let mut tokens = Vec::with_capacity(gamma);
    let mut q_rows = Vec::with_capacity(gamma);
    for (position, &uniform) in request.draft_uniforms.iter().enumerate() {
        let selected_index = selected_index(uniform);
        let selected_token = candidates[selected_index];
        tokens.push(selected_token);
        q_rows.push(DflashSparseQRow {
            position,
            predecessor,
            candidates,
            probabilities,
            selected_index,
            selected_token,
            probability_sum: probabilities.iter().sum(),
        });
        predecessor = selected_token;
    }
    DflashSampledProposalFrame {
        request,
        proposal: DflashSampledProposal {
            tokens,
            q_rows: Some(q_rows),
        },
    }
}

fn fp32_rows(rows: &[Vec<f32>]) -> Vec<u8> {
    rows.iter()
        .flatten()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn bf16_rows(rows: &[Vec<f32>]) -> Vec<u8> {
    rows.iter()
        .flatten()
        .flat_map(|value| ((value.to_bits() >> 16) as u16).to_le_bytes())
        .collect()
}

#[test]
fn target_p_uses_exact_sampler_receipt_and_rolling_counterfactual_history() {
    let history = vec![19];
    let mut params = sampling_params(0x5eed);
    params.top_k = 0;
    params.top_p = 1.0;
    params.repetition_penalty = 2.0;
    let frame = uniform_q_frame(&params, &history, 2);
    let first_draft = frame.proposal.tokens[0] as usize;
    let alternative = 16usize;
    let mut row = vec![0.0f32; 20];
    row[first_draft] = 4.0;
    row[alternative] = 3.0;
    let raw = fp32_rows(&[row.clone(), row.clone(), row]);
    let history_before = history.clone();

    let target = prepare_dflash_target_probabilities(
        &raw,
        DflashTargetLogitsFormat::Fp32,
        20,
        &frame,
        &params,
        &history,
    )
    .unwrap();

    assert_eq!(history, history_before, "live history must not be mutated");
    assert!(target[0][first_draft] > target[0][alternative]);
    assert!(
        target[1][first_draft] < target[1][alternative],
        "draft prefix must be present in row-1 penalty history"
    );

    let target_bf16 = prepare_dflash_target_probabilities(
        &bf16_rows(&[vec![0.0; 20], vec![0.0; 20], vec![0.0; 20]]),
        DflashTargetLogitsFormat::Bf16,
        20,
        &uniform_q_frame(&sampling_params(7), &history, 2),
        &sampling_params(7),
        &history,
    )
    .unwrap();
    assert_eq!(target_bf16.len(), 3);
    assert!(target_bf16.iter().all(|row| {
        (row.iter().map(|&value| f64::from(value)).sum::<f64>() - 1.0).abs() < 2.0e-5
    }));
}

#[test]
fn target_p_applies_replayable_forbidden_tokens_before_sampling() {
    let history = vec![19];
    let mut params = sampling_params(0x5eed);
    params.top_k = 1;
    params.top_p = 1.0;
    let frame = uniform_q_frame_with_masks(&params, &history, 1, &[3, 7]);

    let mut row0 = vec![0.0f32; 20];
    row0[3] = 100.0;
    row0[7] = 90.0;
    row0[11] = 4.0;
    let row1 = row0.clone();
    for (format, raw) in [
        (
            DflashTargetLogitsFormat::Fp32,
            fp32_rows(&[row0.clone(), row1.clone()]),
        ),
        (DflashTargetLogitsFormat::Bf16, bf16_rows(&[row0, row1])),
    ] {
        let target =
            prepare_dflash_target_probabilities(&raw, format, 20, &frame, &params, &history)
                .unwrap();
        assert_eq!(target.len(), 2);
        for probabilities in target {
            assert_eq!(probabilities[3], 0.0);
            assert_eq!(probabilities[7], 0.0);
            assert!(probabilities[11] > 0.0);
        }
    }
}

#[test]
fn sparse_q_with_positive_mass_on_a_forbidden_candidate_fails_closed() {
    let history = vec![19];
    let mut params = sampling_params(0x5eed);
    params.top_k = 1;
    params.top_p = 1.0;
    let mut frame = uniform_q_frame(&params, &history, 1);
    let forbidden = frame.proposal.tokens[0] as usize;
    let allowed = (forbidden + 1) % 20;
    frame.request = frame
        .request
        .clone()
        .with_target_forbidden_token_ids(vec![forbidden as u32])
        .unwrap();

    let mut row = vec![0.0f32; 20];
    row[forbidden] = 100.0;
    row[allowed] = 4.0;
    let error = prepare_dflash_target_probabilities(
        &fp32_rows(&[row.clone(), row]),
        DflashTargetLogitsFormat::Fp32,
        20,
        &frame,
        &params,
        &history,
    )
    .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("retained positive sparse-q mass")
    );
}

#[test]
fn target_policy_mask_is_canonical_and_vocabulary_bounded() {
    let history = vec![19];
    let params = sampling_params(7);
    let request = DflashDraftSamplingRequest::from_seeded_sampling_params(
        &params,
        &history,
        history.len(),
        1,
    )
    .unwrap()
    .with_target_forbidden_token_ids(vec![7, 3, 7])
    .unwrap();
    assert_eq!(request.target_forbidden_token_ids, vec![3, 7]);

    let mut oversized_params = params.clone();
    oversized_params.logit_bias = (0..DFLASH2_MAX_Q_MODIFIERS)
        .map(|token| (token as u32, 0.0))
        .collect();
    let overflow = DflashDraftSamplingRequest::from_seeded_sampling_params(
        &oversized_params,
        &history,
        history.len(),
        1,
    )
    .unwrap()
    .with_target_forbidden_token_ids(vec![19]);
    assert!(overflow.is_err());

    let mut frame = uniform_q_frame(&params, &history, 1);
    frame.request = request.with_target_forbidden_token_ids(vec![20]).unwrap();
    let error = prepare_dflash_target_probabilities(
        &fp32_rows(&[vec![0.0; 20], vec![0.0; 20]]),
        DflashTargetLogitsFormat::Fp32,
        20,
        &frame,
        &params,
        &history,
    )
    .unwrap_err();
    assert!(error.to_string().contains("outside the vocabulary"));
}

#[test]
fn target_p_fails_closed_on_sampler_history_or_logit_drift() {
    let history = vec![3, 4];
    let params = sampling_params(11);
    let frame = uniform_q_frame(&params, &history, 1);
    let raw = fp32_rows(&[vec![0.0; 20], vec![0.0; 20]]);

    let mut drifted_params = params.clone();
    drifted_params.top_n_sigma = 1.0;
    assert!(
        prepare_dflash_target_probabilities(
            &raw,
            DflashTargetLogitsFormat::Fp32,
            20,
            &frame,
            &drifted_params,
            &history,
        )
        .unwrap_err()
        .to_string()
        .contains("sampler receipt drift")
    );
    assert!(
        prepare_dflash_target_probabilities(
            &raw,
            DflashTargetLogitsFormat::Fp32,
            20,
            &frame,
            &params,
            &[3, 5],
        )
        .unwrap_err()
        .to_string()
        .contains("history digest drift")
    );
    assert!(
        prepare_dflash_target_probabilities(
            &raw[..raw.len() - 1],
            DflashTargetLogitsFormat::Fp32,
            20,
            &frame,
            &params,
            &history,
        )
        .unwrap_err()
        .to_string()
        .contains("byte count drift")
    );
}

#[test]
fn sampled_frame_rng_is_seeded_f32_atomic_and_tamper_evident() {
    let history = vec![1, 2, 3];
    let params = sampling_params(0x1234_5678);
    let request = DflashDraftSamplingRequest::from_seeded_sampling_params(
        &params,
        &history,
        history.len(),
        3,
    )
    .unwrap();
    assert_eq!(
        request.draft_uniforms[0],
        spark_runtime::sampler::sampling_uniform_from_seed(params.seed.unwrap()),
        "first frame draw must match ordinary seeded decode"
    );
    assert_eq!(
        request,
        DflashDraftSamplingRequest::from_seeded_sampling_params(
            &params,
            &history,
            history.len(),
            3,
        )
        .unwrap()
    );
    let mut tampered = request.clone();
    tampered.accept_uniforms[1] = 0.0;
    assert!(
        tampered
            .validate(3)
            .unwrap_err()
            .to_string()
            .contains("acceptance RNG draws")
    );
    let mut next_position = params.clone();
    next_position.seed = Some(params.seed.unwrap().wrapping_add(1));
    assert_ne!(
        request.draft_uniforms,
        DflashDraftSamplingRequest::from_seeded_sampling_params(
            &next_position,
            &[1, 2, 3, 4],
            4,
            3,
        )
        .unwrap()
        .draft_uniforms
    );
}

#[test]
fn sampler_receipt_covers_every_sampling_params_field() {
    let history = vec![1, 2, 3];
    let mut params = sampling_params(99);
    params.top_n_sigma = 0.0;
    params.logit_bias = vec![(7, -1.25)];
    params.repetition_penalty_window = 64;
    params.lz_penalty = 0.3;
    params.dry_multiplier = 0.8;
    params.dry_base = 1.9;
    params.dry_allowed_length = 3;
    params.dry_sequence_breakers = vec![10, 11];
    params.max_tokens = 512;
    params.stop_token_ids = vec![2, 151_645];
    let request = DflashDraftSamplingRequest::from_seeded_sampling_params(
        &params,
        &history,
        history.len(),
        1,
    )
    .unwrap();
    let json = serde_json::to_value(&request.request_sampler).unwrap();
    for field in [
        "temperature",
        "top_k",
        "top_p",
        "top_n_sigma",
        "min_p",
        "logit_bias",
        "repetition_penalty",
        "repetition_penalty_window",
        "presence_penalty",
        "frequency_penalty",
        "lz_penalty",
        "dry_multiplier",
        "dry_base",
        "dry_allowed_length",
        "dry_sequence_breakers",
        "max_tokens",
        "stop_token_ids",
        "seed",
    ] {
        assert!(json.get(field).is_some(), "receipt omitted {field}");
    }

    let assert_drift = |drifted: &spark_runtime::sampler::SamplingParams| {
        assert!(request.validate_sampler(drifted).is_err());
    };
    let mut drifted = params.clone();
    drifted.top_n_sigma += 0.1;
    assert_drift(&drifted);
    let mut drifted = params.clone();
    drifted.logit_bias.push((8, 0.5));
    assert_drift(&drifted);
    let mut drifted = params.clone();
    drifted.repetition_penalty_window += 1;
    assert_drift(&drifted);
    let mut drifted = params.clone();
    drifted.lz_penalty += 0.1;
    assert_drift(&drifted);
    let mut drifted = params.clone();
    drifted.dry_multiplier += 0.1;
    assert_drift(&drifted);
    let mut drifted = params.clone();
    drifted.dry_base += 0.1;
    assert_drift(&drifted);
    let mut drifted = params.clone();
    drifted.dry_allowed_length += 1;
    assert_drift(&drifted);
    let mut drifted = params.clone();
    drifted.dry_sequence_breakers.push(12);
    assert_drift(&drifted);
    let mut drifted = params.clone();
    drifted.max_tokens += 1;
    assert_drift(&drifted);
    let mut drifted = params.clone();
    drifted.stop_token_ids.push(3);
    assert_drift(&drifted);
    let mut drifted = params;
    drifted.seed = Some(100);
    assert_drift(&drifted);
}
