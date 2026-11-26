// SPDX-License-Identifier: AGPL-3.0-only

//! The first sampled token honors the request's full sampler settings. Each test's control is
//! what the first-token path used to build: temperature/top-k/top-p and nothing else.

use super::{
    materialize_effective_sampling_seed_with, needs_sampler, sample_host_logits,
    sample_token_with_logprobs_and_grammar,
};
use crate::grammar::{GrammarEngine, GrammarState};
use crate::scheduler::prefill_a_step::{
    advance_prefill_first_token_grammar, prefill_first_token_applies_grammar,
};
use spark_runtime::gpu::DevicePtr;
use spark_runtime::sampler::SamplingParams;

fn params(temperature: f32) -> SamplingParams {
    SamplingParams {
        temperature,
        top_k: 0,
        top_p: 1.0,
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
        seed: None,
    }
}

/// The parameters the first-token path built before the fix, from a request's `full` set.
fn old_first_token(full: &SamplingParams) -> SamplingParams {
    SamplingParams {
        seed: full.seed,
        ..params(full.temperature)
    }
}

/// A logit_bias that forbids the argmax is honored at position 0, greedy.
#[test]
fn logit_bias_is_honored_on_the_first_token() {
    let logits = vec![1.0, 2.0, 5.0, 3.0];
    let full = SamplingParams {
        logit_bias: vec![(2, -100.0)],
        ..params(0.0)
    };
    assert!(
        needs_sampler(&full, &[]),
        "a bias must leave the device argmax fast path"
    );
    assert_eq!(sample_host_logits(&mut logits.clone(), &full, &[], &[]), 3);
    // Control: the old first-token parameters pick the forbidden token.
    assert_eq!(
        sample_host_logits(&mut logits.clone(), &old_first_token(&full), &[], &[]),
        2
    );
}

/// top_n_sigma filters a token the request excluded; the old path samples it.
#[test]
fn top_n_sigma_is_honored_on_the_first_token() {
    // Nine tokens at 5.0 and a probe at 3.5: sigma 0.45, so the 1-sigma cut (max - sigma =
    // 4.55) drops the probe, which plain T=1 picks ~2.4% of the time.
    let mut logits = vec![5.0f32; 10];
    logits[9] = 3.5;
    let picks = |p: &SamplingParams| -> Vec<u32> {
        (0..1000u64)
            .map(|seed| {
                let p = SamplingParams {
                    seed: Some(seed),
                    ..p.clone()
                };
                sample_host_logits(&mut logits.clone(), &p, &[], &[])
            })
            .collect()
    };
    let full = SamplingParams {
        top_n_sigma: 1.0,
        ..params(1.0)
    };
    assert!(!picks(&full).contains(&9), "sigma must drop the probe");
    assert!(
        picks(&old_first_token(&full)).contains(&9),
        "control: the old path samples the probe"
    );
}

/// Plain greedy with no bias stays on the fast path; penalties only matter once there is
/// history for them to act on.
#[test]
fn only_settings_that_can_move_the_argmax_leave_the_fast_path() {
    let penalised = SamplingParams {
        repetition_penalty: 1.1,
        ..params(0.0)
    };
    assert!(!needs_sampler(&params(0.0), &[]));
    assert!(!needs_sampler(&penalised, &[]));
    assert!(needs_sampler(&penalised, &[7]));
    assert!(needs_sampler(&params(0.7), &[]));
}

#[test]
fn explicit_stochastic_seed_is_preserved_without_entropy() {
    let mut seed = Some(0x5eed);
    materialize_effective_sampling_seed_with(1.0, &mut seed, |_| {
        panic!("explicit seed must not consult OS entropy")
    })
    .unwrap();
    assert_eq!(seed, Some(0x5eed));
}

#[test]
fn unseeded_stochastic_request_captures_entropy_exactly_once() {
    let bytes = 0x0123_4567_89ab_cdefu64.to_le_bytes();
    let mut seed = None;
    materialize_effective_sampling_seed_with(0.7, &mut seed, |out| {
        out.copy_from_slice(&bytes);
        Ok(())
    })
    .unwrap();
    assert_eq!(seed, Some(0x0123_4567_89ab_cdef));

    materialize_effective_sampling_seed_with(0.7, &mut seed, |_| {
        panic!("an established effective seed must not be replaced")
    })
    .unwrap();
    assert_eq!(seed, Some(0x0123_4567_89ab_cdef));
}

#[test]
fn greedy_request_does_not_consume_entropy() {
    let mut seed = None;
    materialize_effective_sampling_seed_with(0.0, &mut seed, |_| {
        panic!("greedy sampling must not consult OS entropy")
    })
    .unwrap();
    assert_eq!(seed, None);
}

#[test]
fn stochastic_request_fails_closed_when_entropy_is_unavailable() {
    let mut seed = None;
    let error = materialize_effective_sampling_seed_with(1.0, &mut seed, |_| Err(()))
        .expect_err("missing OS entropy must abort stochastic request setup");
    assert!(error.to_string().contains("OS entropy unavailable"));
    assert_eq!(seed, None);
}

#[test]
fn both_prefill_routes_seed_before_the_first_target_sample() {
    for source in [
        include_str!("prefill_a_step.rs"),
        include_str!("prefill_b_step.rs"),
    ] {
        let capture = source
            .find("materialize_effective_sampling_seed(temperature, &mut req_seed)")
            .expect("prefill route must establish the effective request seed");
        let publish = source
            .find("first_params.seed = req_seed")
            .expect("prefill route must give the target sampler the effective seed");
        let first_sample = source
            .find("sample_token_with_logprobs_and_grammar(")
            .expect("prefill route must sample a first target token");
        assert!(capture < publish && publish < first_sample);
        assert!(source[publish..].contains("seed: req_seed"));
    }
}

const GRAMMAR_EOS: u32 = 128;
const GRAMMAR_FENCE: u32 = 129;
const THINK_END: u32 = 130;
const TEST_JSON_SCHEMA: &str = r#"{"type":"object","properties":{"name":{"type":"string"}},"required":["name"],"additionalProperties":false}"#;

fn test_grammar(schema: Option<&str>) -> GrammarState {
    let mut vocab: Vec<String> = (0u8..128).map(|b| String::from(b as char)).collect();
    vocab.push("<eos>".to_string());
    vocab.push("```".to_string());
    let mut engine = GrammarEngine::new(&vocab, &[GRAMMAR_EOS as i32]).expect("grammar engine");
    let compiled = match schema {
        Some(s) => engine.compile_json_schema(s).expect("compile json schema"),
        None => engine.compile_json_grammar().expect("compile json grammar"),
    };
    GrammarState::new(&compiled, engine.vocab_size()).expect("grammar state")
}

fn strict_object_grammar() -> GrammarState {
    test_grammar(Some(TEST_JSON_SCHEMA))
}

fn any_json_grammar() -> GrammarState {
    test_grammar(None)
}

fn fence_over_open_brace_logits() -> Vec<f32> {
    let mut logits = vec![-8.0f32; 130];
    logits[b'{' as usize] = 10.0;
    logits[b'H' as usize] = 40.0;
    logits[GRAMMAR_FENCE as usize] = 50.0;
    logits
}

#[test]
fn sample_token_with_logprobs_and_grammar_masks_illegal_logits_and_delegates_cleanly() {
    let model = crate::scheduler::test_fixtures::model::recording();
    *model.host_logits.lock().expect("host_logits lock") = fence_over_open_brace_logits();
    let sample = |p: &SamplingParams, k: Option<u8>, gs: Option<&mut GrammarState>, apply: bool| {
        sample_token_with_logprobs_and_grammar(
            &model,
            DevicePtr::NULL,
            p,
            &[GRAMMAR_EOS],
            &[],
            k,
            gs,
            apply,
        )
        .expect("sample")
    };
    let greedy = params(0.0);
    let mut gs = strict_object_grammar();

    // 1. Grammar active (greedy, no logprobs): masks ``` (50.0) & 'H' (40.0), picks '{' (10.0).
    let (tok, lp) = sample(&greedy, None, Some(&mut gs), true);
    assert_eq!(tok, b'{' as u32, "must pick legal '{{' over fence");
    assert!(lp.is_none());

    // 2. Grammar active with top_logprobs: logprobs reflect the masked distribution.
    let (tok, lp) = sample(&greedy, Some(2), Some(&mut gs), true);
    assert_eq!(tok, b'{' as u32);
    let lp = lp.expect("requested first-token logprobs");
    assert_eq!(lp.token_id, b'{' as u32);
    assert_eq!(lp.top[0].0, b'{' as u32);
    assert!(
        lp.top[0].1.abs() < 1e-6,
        "only legal token '{{' must have logprob 0.0: {:?}",
        lp.top
    );
    assert!(
        !lp.top
            .iter()
            .any(|&(id, p)| (id == GRAMMAR_FENCE || id == b'H' as u32) && p.is_finite()),
        "masked illegal tokens must not have finite logprob: {:?}",
        lp.top
    );

    // 3. Grammar active on stochastic path: obeys mask and preserves logprobs.
    let stochastic = SamplingParams {
        seed: Some(42),
        ..params(0.7)
    };
    let (tok, lp) = sample(&stochastic, Some(1), Some(&mut gs), true);
    assert_eq!(tok, b'{' as u32);
    assert_eq!(lp.expect("stochastic logprobs").token_id, tok);

    // 4. apply_grammar=false (thinking enabled): delegates without masking.
    let (tok, lp) = sample(&greedy, Some(1), Some(&mut gs), false);
    assert_eq!(tok, GRAMMAR_FENCE, "apply_grammar=false must not mask");
    assert_eq!(lp.expect("delegated logprobs").token_id, GRAMMAR_FENCE);

    // 5. grammar_state=None: delegates cleanly to sample_token_with_logprobs.
    assert_eq!(sample(&greedy, Some(1), None, true).0, GRAMMAR_FENCE);

    // 6. Terminated grammar delegates without calling fill_next_token_bitmask.
    let mut done = any_json_grammar();
    assert!(done.accept_token(b'{' as u32));
    assert!(done.accept_token(b'}' as u32));
    assert!(done.accept_token(GRAMMAR_EOS));
    assert!(done.is_terminated());
    assert_eq!(
        sample(&greedy, Some(1), Some(&mut done), true).0,
        GRAMMAR_FENCE
    );
}

#[test]
fn advance_prefill_first_token_grammar_prevents_double_brace_and_skips_thinking() {
    // Non-thinking visible '{' on general JSON advances past '{': rejects '{', allows '"' & '}'.
    let mut json_gs = Some(any_json_grammar());
    let terminated = advance_prefill_first_token_grammar(&mut json_gs, b'{' as u32, true, false);
    assert!(!terminated);
    let state = json_gs.as_mut().expect("grammar state retained");
    assert!(state.fill_bitmask());
    assert!(
        !state.is_token_allowed(b'{' as u32),
        "second '{{' must be rejected after first token advanced grammar"
    );
    assert!(state.is_token_allowed(b'"' as u32));
    assert!(state.is_token_allowed(b'}' as u32));

    // Non-thinking visible '{' on strict object schema rejects second '{' and allows '"'.
    let mut schema_gs = Some(strict_object_grammar());
    assert!(!advance_prefill_first_token_grammar(
        &mut schema_gs,
        b'{' as u32,
        true,
        false
    ));
    let state = schema_gs.as_mut().expect("schema grammar retained");
    assert!(state.fill_bitmask());
    assert!(!state.is_token_allowed(b'{' as u32));
    assert!(state.is_token_allowed(b'"' as u32));

    // Inside thinking: grammar must NOT advance (remains at root expecting '{').
    let mut thinking_gs = Some(strict_object_grammar());
    assert!(!advance_prefill_first_token_grammar(
        &mut thinking_gs,
        b'{' as u32,
        true,
        true
    ));
    let state = thinking_gs.as_mut().expect("thinking grammar retained");
    assert!(state.fill_bitmask());
    assert!(state.is_token_allowed(b'{' as u32));
    assert!(!state.is_token_allowed(b'"' as u32));

    // Hidden first token (e.g. spontaneous <think>): grammar must NOT advance.
    let mut hidden_gs = Some(strict_object_grammar());
    assert!(!advance_prefill_first_token_grammar(
        &mut hidden_gs,
        b'{' as u32,
        false,
        false
    ));
    let state = hidden_gs.as_mut().expect("hidden grammar retained");
    assert!(state.fill_bitmask());
    assert!(state.is_token_allowed(b'{' as u32));
    assert!(!state.is_token_allowed(b'"' as u32));

    // None grammar is a no-op returning false.
    let mut no_gs = None;
    assert!(!advance_prefill_first_token_grammar(
        &mut no_gs,
        b'{' as u32,
        true,
        false
    ));
}

#[test]
fn prefill_first_token_applies_grammar_matches_thinking_phase() {
    assert!(prefill_first_token_applies_grammar(false, Some(THINK_END)));
    assert!(prefill_first_token_applies_grammar(false, None));
    assert!(!prefill_first_token_applies_grammar(true, Some(THINK_END)));
    assert!(prefill_first_token_applies_grammar(true, None));
}

#[test]
fn prefill_handoffs_mask_and_advance_grammar_without_double_brace() {
    use crate::scheduler::emit_step::StartPrefillResult;
    use crate::scheduler::phase_promote_prefills::promote_completed_prefills;
    use crate::scheduler::prefill_a_step::{
        PrefillTokenConfig, finalize_prefill_first_token, prepare_prefill_request,
    };
    use crate::scheduler::test_fixtures::{blocking_request, model, prefill_in_progress};
    use crate::scheduler::types::ResponseSink;

    let tokens_cfg = PrefillTokenConfig {
        think_end_token: Some(THINK_END),
        think_start_token: None,
        code_fence_token: None,
        tool_call_start_token: None,
        tool_call_end_token: None,
        spontaneous_think_budget: 0,
    };

    let m = model::recording();
    *m.host_logits.lock().expect("host_logits lock") = fence_over_open_brace_logits();
    let finalize = |imported: Option<u32>| {
        let (req, rx) = blocking_request(vec![1, 2, 3]);
        let mut prep = prepare_prefill_request(req, &[GRAMMAR_EOS], &mut None).expect("prepare");
        prep.grammar_state = Some(strict_object_grammar());
        let res = finalize_prefill_first_token(
            tokens_cfg,
            &m,
            prep,
            model::seq_state(),
            None,
            DevicePtr::NULL,
            imported,
            imported.is_some(),
            0,
            false,
        )
        .expect("finalize prefill first token");
        let StartPrefillResult::Active(a) = res else {
            panic!("expected Active sequence");
        };
        (a, rx)
    };
    let promote = |enable_thinking: bool, thinking_budget: Option<u32>, first: u32| {
        let (tx, rx) = tokio::sync::oneshot::channel();
        let mut p = prefill_in_progress(ResponseSink::Blocking(Some(tx)));
        p.enable_thinking = enable_thinking;
        p.thinking_budget = thinking_budget;
        p.eos_tokens = vec![GRAMMAR_EOS];
        p.grammar_state = Some(strict_object_grammar());
        let mut prefilling = vec![p];
        let mut active = Vec::new();
        promote_completed_prefills(
            &m,
            &mut prefilling,
            vec![(0, Some(first))],
            &mut active,
            0,
            Some(THINK_END),
            None,
            None,
            None,
            None,
        );
        (active.pop().expect("promoted ActiveSeq"), rx)
    };

    // 1. finalize_prefill_first_token: masks fence -> samples '{' -> advances grammar past '{'.
    let (mut active_a, _rx_a) = finalize(None);
    assert_eq!(active_a.output_tokens, vec![b'{' as u32]);
    let gs_a = active_a.grammar_state.as_mut().expect("active grammar");
    assert!(gs_a.fill_bitmask());
    assert!(!gs_a.is_token_allowed(b'{' as u32));
    assert!(gs_a.is_token_allowed(b'"' as u32));

    // 2. Disagg imported_first_token also advances grammar state on handoff.
    let (mut active_imp, _rx_imp) = finalize(Some(b'{' as u32));
    let gs_imp = active_imp.grammar_state.as_mut().expect("imported grammar");
    assert!(gs_imp.fill_bitmask());
    assert!(!gs_imp.is_token_allowed(b'{' as u32));
    assert!(gs_imp.is_token_allowed(b'"' as u32));

    // 3. promote_completed_prefills (chunked continuation handoff) advances grammar past '{'.
    let (mut active_p, _rx_p) = promote(false, None, b'{' as u32);
    let gs_p = active_p.grammar_state.as_mut().expect("promoted grammar");
    assert!(gs_p.fill_bitmask());
    assert!(!gs_p.is_token_allowed(b'{' as u32));
    assert!(gs_p.is_token_allowed(b'"' as u32));

    // 4. First-token explicit </think> close (closed_on_first=true) must NOT feed </think> to grammar.
    let (mut active_t, _rx_t) = promote(true, Some(8), THINK_END);
    assert!(!active_t.inside_thinking && active_t.think_ended);
    let gs_t = active_t
        .grammar_state
        .as_mut()
        .expect("post-think-close grammar");
    assert!(gs_t.fill_bitmask());
    assert!(
        gs_t.is_token_allowed(b'{' as u32),
        "first-token </think> must leave grammar at root expecting '{{'"
    );
    assert!(!gs_t.is_token_allowed(b'"' as u32));
}

#[test]
fn all_prefill_sampling_and_handoff_sites_enforce_first_token_grammar() {
    for (name, source) in [
        ("prefill_a_step.rs", include_str!("prefill_a_step.rs")),
        ("prefill_b_step.rs", include_str!("prefill_b_step.rs")),
        (
            "phase_continue_prefills.rs",
            include_str!("phase_continue_prefills.rs"),
        ),
        (
            "run_standard.rs",
            include_str!("phase_continue_prefills/run_standard.rs"),
        ),
        (
            "run_batched_prefill.rs",
            include_str!("phase_continue_prefills/run_batched_prefill.rs"),
        ),
        (
            "run_batched_mixed.rs",
            include_str!("phase_continue_prefills/run_batched_mixed.rs"),
        ),
    ] {
        assert!(
            source.contains("sample_token_with_logprobs_and_grammar("),
            "{name} must sample first tokens through sample_token_with_logprobs_and_grammar"
        );
        assert!(
            !source.contains("sample_token_with_logprobs("),
            "{name} must not bypass grammar masking with raw sample_token_with_logprobs"
        );
    }

    for (name, source) in [
        ("prefill_a_step.rs", include_str!("prefill_a_step.rs")),
        ("prefill_b_step.rs", include_str!("prefill_b_step.rs")),
        (
            "phase_promote_prefills.rs",
            include_str!("phase_promote_prefills.rs"),
        ),
    ] {
        assert!(
            source.contains("advance_prefill_first_token_grammar("),
            "{name} must advance grammar state on prefill-to-ActiveSeq handoff"
        );
    }
}
