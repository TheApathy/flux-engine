// SPDX-License-Identifier: AGPL-3.0-only

use super::*;
use crate::scheduler::test_fixtures;
/// Small deterministic xorshift (spark-server has no `rand` dependency).
struct XorShift(u64);
impl XorShift {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}
use spark_runtime::sampler::device_topk::topk_candidates_reference;

type Rx = tokio::sync::oneshot::Receiver<anyhow::Result<crate::api::InferenceResponse>>;

fn stochastic_seq(top_k: u32) -> (ActiveSeq, Rx) {
    let (tx, rx) = tokio::sync::oneshot::channel();
    let mut a = test_fixtures::active_seq(ResponseSink::Blocking(Some(tx)), 0);
    a.temperature = 1.0;
    a.top_k = top_k;
    a.top_p = 0.95;
    a.seed = Some(11);
    (a, rx)
}

const REFLECT: [u32; 3] = [5, 6, 7];

fn route(a: &ActiveSeq) -> TopkRoute {
    device_topk_route(a, false, false, &REFLECT)
}

#[test]
fn plain_t1_top20_row_takes_the_device_with_the_fixed_edit_budget() {
    let (a, _rx) = stochastic_seq(20);
    assert_eq!(
        route(&a),
        TopkRoute::Device {
            m: 20 + REFLECT.len() + FIXED_LOWERED_IDS,
            temperature: 1.0,
            f2: false,
        }
    );
}

#[test]
fn rows_the_device_cannot_reproduce_route_to_the_host_with_a_reason() {
    let cases: Vec<(&str, Box<dyn Fn(&mut ActiveSeq)>)> = vec![
        ("greedy", Box::new(|a| a.temperature = 0.0)),
        ("top_k", Box::new(|a| a.top_k = 0)),
        ("logprobs", Box::new(|a| a.top_logprobs = Some(5))),
        ("top_n_sigma", Box::new(|a| a.top_n_sigma = 1.0)),
        ("logit_bias", Box::new(|a| a.logit_bias = vec![(3, 1.0)])),
    ];
    for (want, mutate) in cases {
        let (mut a, _rx) = stochastic_seq(20);
        mutate(&mut a);
        assert_eq!(route(&a), TopkRoute::Host(want));
    }
    let (a, _rx) = stochastic_seq(20);
    assert_eq!(
        device_topk_route(&a, true, false, &[]),
        TopkRoute::Host("fp32_logits")
    );
    assert_eq!(
        device_topk_route(&a, false, true, &[]),
        TopkRoute::Host("adaptive_sampling")
    );
}

#[test]
fn early_thinking_rows_stay_on_the_device() {
    let (mut a, _rx) = stochastic_seq(20);
    a.inside_thinking = true;
    a.thinking_tokens = 3;
    assert!(matches!(route(&a), TopkRoute::Device { .. }));
}

/// A BF16 row of DISTINCT positive values in [0.5, 8): the scheduler's
/// sampler reads the canonical-ties flag from the environment, so this test
/// keeps ties out (the -10 / -12 / -inf edits land below every unedited
/// value). Tie and penalty equivalence is proven with canonical ties in
/// spark-runtime's device_topk tests.
fn distinct_row(rng: &mut XorShift, vocab: usize) -> Vec<u8> {
    let mut bits: Vec<u16> = (0..vocab as u16).map(|i| 0x3F00u16 + i).collect();
    for i in (1..bits.len()).rev() {
        bits.swap(i, rng.below(i + 1));
    }
    bits.iter().flat_map(|b| b.to_le_bytes()).collect()
}

#[test]
fn sparse_row_through_the_plain_interventions_matches_the_full_row() {
    let vocab = 512usize;
    let mut rng = XorShift(0x9E37_79B9_7F4A_7C15);
    let mut compared = 0;
    for case in 0..400u64 {
        let row = distinct_row(&mut rng, vocab);
        let setup = |a: &mut ActiveSeq| {
            a.seed = Some(case);
            a.top_k = 1 + (case % 30) as u32;
            a.top_p = [1.0, 0.95, 0.7][(case % 3) as usize];
            a.min_p = [0.0, 0.05][(case % 2) as usize];
            a.output_tokens = vec![1, 2, 3, 2];
            match case % 3 {
                0 => {
                    a.inside_thinking = true;
                    a.thinking_tokens = 10;
                    a.tool_call_start_token = Some(9);
                }
                1 => {
                    a.think_ended = true;
                    a.think_start_token = Some(10);
                    a.think_end_token = Some(11);
                    a.suppress_tool_call = true;
                }
                _ => {}
            }
        };
        let (mut full_seq, _r1) = stochastic_seq(20);
        let (mut sparse_seq, _r2) = stochastic_seq(20);
        setup(&mut full_seq);
        setup(&mut sparse_seq);
        let TopkRoute::Device { m, temperature, .. } = route(&sparse_seq) else {
            panic!("case {case} unexpectedly routed to the host");
        };
        let cands = topk_candidates_reference(&row, temperature, m);
        let sparse =
            spark_runtime::sampler::device_topk::sparse_row_from_candidates(vocab, &cands).unwrap();
        let te = full_seq.think_end_token;
        let tcs = Some(9u32);
        let full_tok = process_seq_logits(
            &mut full_seq,
            &row,
            0,
            vocab,
            2,
            false,
            te,
            None,
            tcs,
            None,
            &REFLECT,
            false,
        );
        let sparse_tok =
            process_seq_sparse_logits(&mut sparse_seq, sparse, None, te, tcs, &REFLECT, false);
        assert_eq!(full_tok.0, sparse_tok.0, "case {case}");
        compared += 1;
    }
    assert_eq!(compared, 400);
}

fn in_f2_window(a: &mut ActiveSeq) {
    a.inside_thinking = true;
    a.thinking_tokens = crate::scheduler::confidence::CONFIDENCE_EARLY_STOP_MIN_THINKING + 7;
    a.think_end_token = Some(11);
}

#[test]
fn f2_window_rows_take_the_device_with_certified_f2() {
    let (mut a, _rx) = stochastic_seq(20);
    in_f2_window(&mut a);
    assert!(matches!(route(&a), TopkRoute::Device { f2: true, .. }));
    let (mut b, _rx) = stochastic_seq(20);
    in_f2_window(&mut b);
    let many: Vec<u32> = (0..64).collect();
    assert_eq!(
        device_topk_route(&b, false, false, &many),
        TopkRoute::Host("f2_edit_budget")
    );
}

/// A thinking row inside the F2 window: a leader whose top probability sweeps
/// across 0.95 over the case index, over DISTINCT negative noise values (this
/// test runs with canonical ties off, so it must not contain exact ties; tie
/// equivalence is proven in spark-runtime with canonical ties on).
fn f2_row(rng: &mut XorShift, vocab: usize, case: u64) -> Vec<u8> {
    // +0.0625 keeps the lead off 2.0, the edited value of id 3.
    let lead = 0.5625 + (case % 40) as f32 * 0.25;
    // Distinct bf16 patterns from -2.0 downward (0xC000 + k), shuffled.
    let mut noise: Vec<u16> = (0..vocab as u16).map(|k| 0xC000u16 + k).collect();
    for i in (1..noise.len()).rev() {
        noise.swap(i, rng.below(i + 1));
    }
    let bf = |v: f32| (v.to_bits() >> 16) as u16;
    // id 3 / id 17 values are not of the 0xC000.. form, so they tie nothing.
    noise[3] = bf(12.0);
    noise[17] = bf(lead);
    noise.iter().flat_map(|b| b.to_le_bytes()).collect()
}

#[test]
fn f2_window_sparse_route_matches_the_full_row_token_and_f2_state() {
    use spark_runtime::sampler::f2_certify::{certify, edited_sum_exp_reference};
    let vocab = 4096usize;
    let reflect = [3u32, 5, 6]; // id 3 is the raw leader: F1 pulls it down by 10
    let mut rng = XorShift(0xF2F2_0001);
    let (mut certified, mut runs_seen) = (0, 0);
    for case in 0..240u64 {
        let row = f2_row(&mut rng, vocab, case);
        let (mut full_seq, _r1) = stochastic_seq(20);
        let (mut sparse_seq, _r2) = stochastic_seq(20);
        for a in [&mut full_seq, &mut sparse_seq] {
            in_f2_window(a);
            a.seed = Some(case);
            a.consecutive_confident = (case % 31) as u32;
        }
        let TopkRoute::Device {
            m,
            temperature,
            f2: true,
        } = device_topk_route(&sparse_seq, false, false, &reflect)
        else {
            panic!("case {case}: F2 row should take the device");
        };
        let edits =
            super::super::decode_logits_seq::pre_f2_edits(&sparse_seq, Some(11), &reflect, vocab);
        // Certified when the bound allows; otherwise the exact host fallback.
        let stats = edited_sum_exp_reference(&row, &edits);
        let exact = host_f2_from_bf16(&row, &edits);
        let decision = match certify(&stats) {
            Some(d) => {
                certified += 1;
                assert_eq!(
                    d, exact,
                    "case {case}: certified decision differs from the host"
                );
                d
            }
            None => exact,
        };
        let cands = topk_candidates_reference(&row, temperature, m);
        let sparse =
            spark_runtime::sampler::device_topk::sparse_row_from_candidates(vocab, &cands).unwrap();
        let full_tok = process_seq_logits(
            &mut full_seq,
            &row,
            0,
            vocab,
            2,
            false,
            Some(11),
            None,
            None,
            None,
            &reflect,
            false,
        );
        let sparse_tok = process_seq_sparse_logits(
            &mut sparse_seq,
            sparse,
            Some(decision),
            Some(11),
            None,
            &reflect,
            false,
        );
        assert_eq!(full_tok.0, sparse_tok.0, "case {case}: token");
        assert_eq!(
            full_seq.consecutive_confident, sparse_seq.consecutive_confident,
            "case {case}: F2 run"
        );
        assert_eq!(
            full_seq.force_end_thinking, sparse_seq.force_end_thinking,
            "case {case}: F2 arm"
        );
        runs_seen += usize::from(full_seq.consecutive_confident > 0);
    }
    assert!(certified > 150, "only {certified}/240 certified");
    assert!(
        runs_seen > 20 && runs_seen < 220,
        "the sweep must cross the F2 threshold ({runs_seen})"
    );
}

#[test]
fn flash_next_default_preset_stays_on_the_device_through_long_thinking() {
    // presence 1.5 / window 256 (the FN thinking preset), 12 reflection ids,
    // thousands of distinct output tokens, inside the F2 window.
    let reflect: Vec<u32> = (100..112).collect();
    let (mut a, _rx) = stochastic_seq(20);
    a.presence_penalty = 1.5;
    a.repetition_penalty_window = 256;
    a.output_tokens = (0..6000).collect();
    in_f2_window(&mut a);
    assert_eq!(
        device_topk_route(&a, false, false, &reflect),
        TopkRoute::Device {
            m: 20 + 256 + reflect.len() + FIXED_LOWERED_IDS,
            temperature: 1.0,
            f2: true,
        }
    );
}

#[test]
fn mode_off_strict_auto_startup_and_per_request_eligibility() {
    use device_topk::DeviceTopkMode;

    // Startup decisions across Off / Strict / Auto and model support.
    assert_eq!(
        startup_receipt_decision(false, DeviceTopkMode::Off).unwrap(),
        StartupOutcome::Disabled
    );
    assert_eq!(
        startup_receipt_decision(true, DeviceTopkMode::Off).unwrap(),
        StartupOutcome::Disabled
    );
    assert_eq!(
        startup_receipt_decision(true, DeviceTopkMode::Strict).unwrap(),
        StartupOutcome::Engaged {
            mode_label: "strict",
        }
    );
    assert!(startup_receipt_decision(false, DeviceTopkMode::Strict).is_err());
    assert_eq!(
        startup_receipt_decision(true, DeviceTopkMode::Auto).unwrap(),
        StartupOutcome::Engaged { mode_label: "auto" }
    );
    assert_eq!(
        startup_receipt_decision(false, DeviceTopkMode::Auto).unwrap(),
        StartupOutcome::AutoHostOnly
    );

    // Per-request eligibility under Off / Strict / Auto.
    let (eligible, _r1) = stochastic_seq(20);
    let (top_k_zero, _r2) = stochastic_seq(0);
    let (top_k_over, _r3) = stochastic_seq(device_topk::DEVICE_TOPK_K_MAX as u32 + 1);
    let (mut with_logprobs, _r4) = stochastic_seq(20);
    with_logprobs.top_logprobs = Some(5);

    // Off never routes to device.
    assert!(!seq_uses_device_topk_with_mode(
        true,
        &eligible,
        false,
        false,
        &REFLECT,
        DeviceTopkMode::Off
    ));

    // Auto and Strict route eligible rows to device when model supports it,
    // and transparently route unsupported rows (top_k=0, oversized top_k, logprobs,
    // fp32 logits, unsupported model) to host.
    for mode in [DeviceTopkMode::Strict, DeviceTopkMode::Auto] {
        assert!(seq_uses_device_topk_with_mode(
            true, &eligible, false, false, &REFLECT, mode
        ));
        assert!(!seq_uses_device_topk_with_mode(
            false, &eligible, false, false, &REFLECT, mode
        ));
        assert!(!seq_uses_device_topk_with_mode(
            true, &eligible, true, false, &REFLECT, mode
        ));
        assert!(!seq_uses_device_topk_with_mode(
            true,
            &top_k_zero,
            false,
            false,
            &REFLECT,
            mode
        ));
        assert!(!seq_uses_device_topk_with_mode(
            true,
            &top_k_over,
            false,
            false,
            &REFLECT,
            mode
        ));
        assert!(!seq_uses_device_topk_with_mode(
            true,
            &with_logprobs,
            false,
            false,
            &REFLECT,
            mode
        ));
    }

    // Plain batch routing under Auto: homogeneous eligible batch takes Device;
    // mixed batch containing a top_k=0 request falls back cleanly to Host.
    let (eligible_b, _r5) = stochastic_seq(20);
    let all_eligible = [eligible, eligible_b];
    let auto_ok = plain_batch_routes_with_mode(
        &all_eligible,
        false,
        false,
        &REFLECT,
        DeviceTopkMode::Auto,
        true,
    )
    .unwrap();
    assert!(
        auto_ok
            .iter()
            .all(|r| matches!(r, TopkRoute::Device { .. })),
        "expected all Device routes, got {auto_ok:?}"
    );

    let (eligible_c, _r6) = stochastic_seq(20);
    let mixed = [eligible_c, top_k_zero];
    let auto_mixed =
        plain_batch_routes_with_mode(&mixed, false, false, &REFLECT, DeviceTopkMode::Auto, true)
            .unwrap();
    assert_eq!(
        auto_mixed,
        vec![TopkRoute::Host("mixed_batch"), TopkRoute::Host("top_k")]
    );
}
