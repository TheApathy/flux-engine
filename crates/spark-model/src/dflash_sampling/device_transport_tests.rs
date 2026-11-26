// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

fn request(seed: u64, gamma: usize) -> DflashDraftSamplingRequest {
    let params = spark_runtime::sampler::SamplingParams {
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
    };
    DflashDraftSamplingRequest::from_seeded_sampling_params(&params, &[], 0, gamma).unwrap()
}

fn cdf_selected(probabilities: &[f32], uniform: f32) -> usize {
    let mut cumulative = 0.0f32;
    for (index, &probability) in probabilities.iter().enumerate() {
        cumulative += probability;
        if cumulative > uniform {
            return index;
        }
    }
    probabilities.len() - 1
}

fn encode_u32(values: impl IntoIterator<Item = u32>) -> Vec<u8> {
    values.into_iter().flat_map(u32::to_le_bytes).collect()
}

fn encode_f32(values: impl IntoIterator<Item = f32>) -> Vec<u8> {
    values.into_iter().flat_map(f32::to_le_bytes).collect()
}

#[test]
fn device_rows_decode_to_a_valid_drain_once_frame() {
    let first = std::array::from_fn::<_, DFLASH2_SELECTOR_TOP_K, _>(|i| i as u32);
    let second = std::array::from_fn::<_, DFLASH2_SELECTOR_TOP_K, _>(|i| (16 + i) as u32);
    let q0 = [1.0 / DFLASH2_SELECTOR_TOP_K as f32; DFLASH2_SELECTOR_TOP_K];
    let mut q1 = [0.0; DFLASH2_SELECTOR_TOP_K];
    q1[3] = 1.0;
    let request = request(0x5eed, 2);
    let first_token = cdf_selected(&q0, request.draft_uniforms[0]) as u32;
    let frame = decode_dflash2_device_proposal(
        request,
        7,
        vec![first_token, 19],
        &encode_u32(first.into_iter().chain(second)),
        &encode_f32(q0.into_iter().chain(q1)),
        64,
    )
    .unwrap();
    assert_eq!(frame.proposal.q_rows.as_ref().unwrap()[0].predecessor, 7);
    assert_eq!(
        frame.proposal.q_rows.as_ref().unwrap()[1].predecessor,
        first_token
    );
    assert_eq!(frame.proposal.q_rows.as_ref().unwrap()[1].selected_index, 3);
}

#[test]
fn device_decode_rejects_truncation_and_a_token_outside_its_sparse_row() {
    let candidates = (0..DFLASH2_SELECTOR_TOP_K as u32).collect::<Vec<_>>();
    let probabilities = [1.0 / DFLASH2_SELECTOR_TOP_K as f32; DFLASH2_SELECTOR_TOP_K];
    let valid_request = request(7, 1);
    let valid_token = cdf_selected(&probabilities, valid_request.draft_uniforms[0]) as u32;
    assert!(
        decode_dflash2_device_proposal(
            valid_request,
            0,
            vec![valid_token],
            &encode_u32(candidates.iter().copied()),
            &encode_f32(probabilities),
            32,
        )
        .is_ok()
    );
    assert!(
        decode_dflash2_device_proposal(
            request(7, 1),
            0,
            vec![31],
            &encode_u32(candidates.iter().copied()),
            &encode_f32(probabilities),
            32,
        )
        .is_err()
    );
    assert!(
        decode_dflash2_device_proposal(
            request(7, 1),
            0,
            vec![valid_token],
            &encode_u32(candidates),
            &[],
            32,
        )
        .is_err()
    );
}

#[test]
fn device_decode_rejects_a_positive_q_token_not_selected_by_the_uniform() {
    let candidates = (0..DFLASH2_SELECTOR_TOP_K as u32).collect::<Vec<_>>();
    let probabilities = [1.0 / DFLASH2_SELECTOR_TOP_K as f32; DFLASH2_SELECTOR_TOP_K];
    let request = request(11, 1);
    let expected = cdf_selected(&probabilities, request.draft_uniforms[0]) as u32;
    let wrong = (expected + 1) % DFLASH2_SELECTOR_TOP_K as u32;
    let error = decode_dflash2_device_proposal(
        request,
        0,
        vec![wrong],
        &encode_u32(candidates),
        &encode_f32(probabilities),
        32,
    )
    .unwrap_err();
    assert!(error.to_string().contains("sparse-q CDF"));
}

#[test]
fn device_decode_rejects_positive_q_mass_on_a_hard_masked_candidate() {
    let candidates = (0..DFLASH2_SELECTOR_TOP_K as u32).collect::<Vec<_>>();
    let probabilities = [1.0 / DFLASH2_SELECTOR_TOP_K as f32; DFLASH2_SELECTOR_TOP_K];
    let request = request(11, 1)
        .with_target_forbidden_token_ids(vec![0])
        .unwrap();
    let selected = cdf_selected(&probabilities, request.draft_uniforms[0]) as u32;
    let error = decode_dflash2_device_proposal(
        request,
        0,
        vec![selected],
        &encode_u32(candidates),
        &encode_f32(probabilities),
        32,
    )
    .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("retained positive sparse-q mass")
    );
}

#[test]
fn device_decode_replays_strict_f32_cdf_for_seeded_draws() {
    let candidates = (0..DFLASH2_SELECTOR_TOP_K as u32).collect::<Vec<_>>();
    let probabilities = [1.0 / DFLASH2_SELECTOR_TOP_K as f32; DFLASH2_SELECTOR_TOP_K];
    for seed in [0, 1, 7, 11, 0x5eed, u64::MAX] {
        let request = request(seed, 1);
        let uniform = request.draft_uniforms[0];
        let selected = cdf_selected(&probabilities, uniform) as u32;
        let frame = decode_dflash2_device_proposal(
            request,
            0,
            vec![selected],
            &encode_u32(candidates.iter().copied()),
            &encode_f32(probabilities),
            32,
        )
        .unwrap_or_else(|error| panic!("uniform {uniform:?}: {error:#}"));
        assert_eq!(
            frame.proposal.q_rows.as_ref().unwrap()[0].selected_token,
            selected
        );
    }
}

#[test]
fn sampled_readback_is_bounded_and_fails_before_publication() {
    let source = include_str!("../layers/dflash_head/forward_block.rs");
    let body = source
        .split_once("BOUNDED_SAMPLED_READBACK:")
        .and_then(|(_, tail)| tail.split_once("BOUNDED_SAMPLED_READBACK_END"))
        .map(|(body, _)| body)
        .expect("bounded sampled readback markers");
    assert_eq!(body.matches("copy_d2h_pair_on_stream(").count(), 1);
    assert_eq!(body.matches("copy_d2h_on_stream(").count(), 1);
    assert!(!body.contains("gpu.synchronize("));
    assert!(!body.contains("gpu.copy_d2h("));
    let pair = body.find("copy_d2h_pair_on_stream(").unwrap();
    let q = body.find("copy_d2h_on_stream(").unwrap();
    let decode = body.find("decode_dflash2_device_proposal(").unwrap();
    let publish = body.find("publish_after_success(").unwrap();
    assert!(
        pair < q,
        "tokens/candidates must fail before the q readback"
    );
    assert!(
        publish < pair && q < decode,
        "publication guard must own both readbacks and typed decode"
    );
    assert!(body[pair..q].contains(")?;"));
    assert!(body[q..decode].contains(")?;"));
    assert!(body[decode..].contains(")?;"));
}

#[test]
fn sampled_selector_fails_closed_before_any_markov_rewrite() {
    let source = include_str!("../layers/dflash_head/forward_block.rs");
    let guard = source
        .find("probabilistic DFlash2 q would be invalidated by the DSpark Markov rewrite")
        .expect("sampled/Markov guard");
    let launch = source
        .find("self.enqueue_markov_sequential")
        .expect("Markov launch");
    assert!(guard < launch);
}
