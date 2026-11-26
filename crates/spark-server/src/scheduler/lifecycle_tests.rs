// SPDX-License-Identifier: AGPL-3.0-only

//! GPU-free finish-reason parity tests shared by blocking and streaming sinks.

use super::*;

mod model_support {
    mod implementation {
        use crate::scheduler::proposal_lifecycle::SchedulerProposalFrame;
        use spark_model::layers::DDTreePayload;

        include!("proposal_lifecycle_model_test_support.rs");
    }
    pub(super) fn model() -> impl spark_model::traits::Model {
        implementation::TestModel::new(Ok(Vec::new()), None)
    }

    pub(super) fn seq_state() -> spark_model::traits::SequenceState {
        implementation::seq_state()
    }

    /// A model whose (single) logits row is `row`.
    pub(super) fn logits_model(row: Vec<f32>) -> impl spark_model::traits::Model {
        let m = implementation::TestModel::new(Ok(Vec::new()), None);
        *m.host_logits.lock().unwrap() = row;
        m
    }

    /// A model whose greedy decode step returns `tokens`.
    pub(super) fn greedy_model(tokens: Vec<u32>) -> impl spark_model::traits::Model {
        let m = implementation::TestModel::new(Ok(Vec::new()), None);
        *m.argmax.lock().unwrap() = tokens;
        m
    }
}

const EOS: u32 = 17;
const TOOL_CLOSE: u32 = 23;
const IM_START: u32 = u32::MAX - 1;
const CONTENT: u32 = 31;
static HARD_STOP_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn token_logprobs(token_id: u32) -> crate::api::TokenLogprobs {
    crate::api::TokenLogprobs {
        token_id,
        logprob: -0.25,
        top: vec![(token_id, -0.25)],
    }
}

struct HardStopReset;

impl Drop for HardStopReset {
    fn drop(&mut self) {
        set_im_start_hard_stop(0);
    }
}

fn active_seq(sink: ResponseSink, last_token: u32) -> ActiveSeq {
    let now = std::time::Instant::now();
    ActiveSeq {
        stop_cause: None,
        seq: model_support::seq_state(),
        session_hash: 0,
        prompt_prefix_hash: 0,
        prefix_hash: 0,
        record_state: super::continuation::RecordState::Live,
        reset_proof: None,
        last_token,
        output_tokens: vec![last_token],
        max_output_tokens: 1,
        remaining: 0,
        min_tokens: 0,
        eos_tokens: vec![EOS],
        ignore_eos: false,
        finished: true,
        sink,
        intervention_ledger: None,
        cancel_flag: None,
        temperature: 0.0,
        top_k: 0,
        top_p: 1.0,
        top_n_sigma: 0.0,
        min_p: 0.0,
        repetition_penalty: 1.0,
        repetition_penalty_window: 0,
        presence_penalty: 0.0,
        frequency_penalty: 0.0,
        lz_penalty: 0.0,
        dry_multiplier: 0.0,
        dry_base: 1.75,
        dry_allowed_length: 3,
        dry_sequence_breakers: Vec::new(),
        logit_bias: Vec::new(),
        inside_thinking: false,
        enable_thinking: false,
        thinking_budget: None,
        spontaneous_think_budget: 512,
        thinking_tokens: 0,
        force_end_thinking: false,
        consecutive_confident: 0,
        in_code_fence: false,
        think_end_token: None,
        think_start_token: None,
        think_ended: true,
        think_just_ended: false,
        think_skip_count: 0,
        post_think_gate_steps: 0,
        tool_call_end_token: Some(TOOL_CLOSE),
        require_tool_call: false,
        tool_call_start_token: None,
        tool_call_opened: false,
        inside_tool_body: false,
        suppress_tool_call: false,
        disable_mtp: false,
        content_started: true,
        content_tokens: 1,
        prose_tokens_since_last_tool: 0,
        think_watchdog_fires: 0,
        rollback_count: 0,
        ssm_rollback_ring: SsmDecodeRing::new(0),
        grammar_state: None,
        pending_drafts: Vec::new(),
        draft_origin: DraftOrigin::default(),
        last_verify_accepted: 0,
        self_context: Default::default(),
        pending_tree_payload: None,
        pending_dflash_sampled_proposal: None,
        last_token_time: now,
        request_start: now,
        decode_start: now,
        seed: None,
        top_logprobs: None,
        logprobs_data: Vec::new(),
        timeout_at: None,
        decode_timeout_at: None,
        stall_mark: StallMark::default(),
        adaptive: crate::adaptive_sampler::AdaptiveSamplingState::new(0.0),
        cached_prompt_tokens: 0,
        difficulty_probe: Default::default(),
        think_nudge: None,
        budget_nudge: None,
    }
}

fn delivered_finish_reasons(last_token: u32) -> (String, String) {
    delivered_finish_reasons_with_controls(last_token, false, &[EOS])
}

fn delivered_finish_reasons_with_controls(
    last_token: u32,
    ignore_eos: bool,
    stop_tokens: &[u32],
) -> (String, String) {
    let model = model_support::model();

    let (blocking_tx, mut blocking_rx) = tokio::sync::oneshot::channel();
    let mut blocking = active_seq(ResponseSink::Blocking(Some(blocking_tx)), last_token);
    blocking.ignore_eos = ignore_eos;
    blocking.eos_tokens = stop_tokens.to_vec();
    finish_sequence_with_cache(&model, &mut blocking, false);
    let blocking_reason = blocking_rx
        .try_recv()
        .expect("blocking response must be delivered")
        .expect("blocking finish must succeed")
        .finish_reason;

    let (streaming_tx, mut streaming_rx) = tokio::sync::mpsc::channel(1);
    let mut streaming = active_seq(ResponseSink::Streaming(streaming_tx), last_token);
    streaming.ignore_eos = ignore_eos;
    streaming.eos_tokens = stop_tokens.to_vec();
    finish_sequence_with_cache(&model, &mut streaming, false);
    let streaming_reason = match streaming_rx
        .try_recv()
        .expect("streaming Done must be delivered")
    {
        StreamEvent::Done { finish_reason, .. } => finish_reason,
        _ => panic!("finish must deliver a Done event"),
    };

    (blocking_reason, streaming_reason)
}

#[test]
fn blocking_hidden_thinking_boundary_uses_visible_content_count() {
    let (tx, _rx) = tokio::sync::oneshot::channel();
    let mut a = active_seq(ResponseSink::Blocking(Some(tx)), CONTENT);
    a.enable_thinking = true;
    a.think_ended = true;
    a.think_end_token = Some(TOOL_CLOSE);
    a.output_tokens = vec![1, 2, 30, EOS];
    a.content_tokens = 2;
    assert_eq!(hidden_thinking_output_boundary(&a), Some(2));

    a.output_tokens = vec![1, TOOL_CLOSE, 30, EOS];
    assert_eq!(hidden_thinking_output_boundary(&a), None);

    a.enable_thinking = false;
    a.output_tokens = vec![30, EOS];
    assert_eq!(hidden_thinking_output_boundary(&a), None);

    // Truncated while still inside `<think>` (never emitted `</think>`):
    // route all tokens to `reasoning_content` (matching streaming).
    a.enable_thinking = true;
    a.think_ended = false;
    a.inside_thinking = true;
    a.output_tokens = vec![1, 2, 3];
    assert_eq!(hidden_thinking_output_boundary(&a), Some(3));
}

#[test]
fn streaming_done_takes_the_authoritative_completion_sequence_once() {
    let cases = [
        ("promoted-prefill-eos", vec![EOS]),
        ("serial-terminal-eos", vec![CONTENT, EOS]),
        ("chatml-hard-stop", vec![CONTENT, IM_START]),
        ("suppressed-thinking", vec![7, 11, 13, CONTENT]),
        ("length", vec![CONTENT, CONTENT + 1, CONTENT + 2]),
        (
            "uneven-speculative-commits",
            [vec![43, 47, 53, 59, 61], vec![67, 71], vec![73, 79, 83, 89]]
                .into_iter()
                .flatten()
                .collect(),
        ),
    ];

    for (name, expected) in cases {
        let reason = finish_reason(&expected, &[EOS], Some(TOOL_CLOSE), false, None);
        let mut scheduler_output = expected.clone();
        let event = completion_done_event(&mut scheduler_output, reason, 1.0, 2.0, 0, 0, 0, None);
        assert!(scheduler_output.is_empty(), "{name}: ownership transfer");
        match event {
            StreamEvent::Done {
                completion_token_ids,
                completion_tokens,
                ..
            } => {
                assert_eq!(completion_token_ids, expected, "{name}: ordered IDs");
                assert_eq!(completion_tokens, expected.len(), "{name}: usage");
            }
            _ => panic!("{name}: expected Done"),
        }
    }
}

#[test]
fn blocking_and_streaming_sinks_deliver_the_shared_finish_reason() {
    let _guard = HARD_STOP_TEST_LOCK.lock().expect("hard-stop test lock");
    let _reset = HardStopReset;
    set_im_start_hard_stop(IM_START);

    for (last_token, expected) in [(IM_START, "stop"), (TOOL_CLOSE, "tool_calls")] {
        let (blocking_reason, streaming_reason) = delivered_finish_reasons(last_token);
        assert_eq!(blocking_reason, expected);
        assert_eq!(streaming_reason, blocking_reason);
    }
}

#[test]
fn ignore_eos_length_boundary_is_reported_by_both_sinks() {
    let _guard = HARD_STOP_TEST_LOCK.lock().expect("hard-stop test lock");
    let _reset = HardStopReset;
    set_im_start_hard_stop(IM_START);

    // Ignored model boundaries are ordinary tokens. Explicit request stop IDs
    // and tool closure still end the turn under ignore_eos.
    for (last_token, ignore_eos, stop_tokens, expected) in [
        (IM_START, true, vec![], "length"),
        (EOS, true, vec![], "length"),
        (IM_START, false, vec![EOS], "stop"),
        (IM_START, true, vec![IM_START], "stop"),
        (TOOL_CLOSE, true, vec![], "tool_calls"),
    ] {
        let (blocking, streaming) =
            delivered_finish_reasons_with_controls(last_token, ignore_eos, &stop_tokens);
        assert_eq!(blocking, expected);
        assert_eq!(streaming, expected);
    }
}

#[test]
fn shared_sink_finish_reason_obeys_protocol_stop_boundaries() {
    let _guard = HARD_STOP_TEST_LOCK.lock().expect("hard-stop test lock");
    let _reset = HardStopReset;
    set_im_start_hard_stop(0);

    assert_eq!(
        finish_reason(&[EOS], &[EOS], Some(TOOL_CLOSE), false, None),
        "stop"
    );
    assert_eq!(
        finish_reason(&[TOOL_CLOSE], &[EOS], Some(TOOL_CLOSE), false, None),
        "tool_calls"
    );
    assert_eq!(
        finish_reason(&[CONTENT], &[EOS], Some(TOOL_CLOSE), false, None),
        "length",
        "an ordinary token at the request cap must remain a length stop"
    );
    assert_eq!(
        finish_reason(&[], &[EOS], Some(TOOL_CLOSE), false, None),
        "length",
        "an empty history must not alias an unregistered ChatML boundary"
    );
    assert_eq!(
        finish_reason(&[IM_START], &[EOS], Some(TOOL_CLOSE), false, None),
        "length",
        "an unregistered ChatML token must not become a stop"
    );
    assert_eq!(
        finish_reason(&[], &[EOS], None, false, None),
        "length",
        "an empty history must not alias an absent tool-close token"
    );
    assert_eq!(
        finish_reason(&[CONTENT], &[EOS], None, false, None),
        "length",
        "an absent tool-close token can never classify as tool_calls"
    );

    set_im_start_hard_stop(IM_START);
    assert_eq!(
        finish_reason(&[IM_START], &[EOS], Some(TOOL_CLOSE), false, None),
        "stop",
        "the exact registered ChatML boundary must be an ordinary stop"
    );
    assert_eq!(
        finish_reason(&[IM_START - 1], &[EOS], Some(TOOL_CLOSE), false, None),
        "length",
        "a token different from the registered boundary must remain length"
    );
    assert_eq!(
        finish_reason(&[IM_START], &[EOS], Some(IM_START), false, None),
        "stop",
        "the registered ChatML stop must take precedence over tool close"
    );
}

/// A stream-side stop string sets the request's cancel flag. The serial decode
/// path must end the sequence there (dropping the sample), as speculative
/// `emit_token` already did; before the fix it committed the token and ran on.
#[test]
fn a_cancelled_sequence_stops_on_the_serial_decode_path() {
    let _guard = HARD_STOP_TEST_LOCK.lock().expect("hard-stop test lock");
    let _reset = HardStopReset;
    set_im_start_hard_stop(0);
    let run = |cancelled: bool| {
        let model = model_support::greedy_model(vec![CONTENT]);
        let (tx, _rx) = tokio::sync::oneshot::channel();
        let mut a = active_seq(ResponseSink::Blocking(Some(tx)), CONTENT);
        a.finished = false;
        a.max_output_tokens = 100;
        a.remaining = 100;
        a.think_ended = false;
        a.cancel_flag = Some(std::sync::Arc::new(std::sync::atomic::AtomicBool::new(
            cancelled,
        )));
        let mut active = vec![a];
        process_decode_logits(
            &model,
            &mut active,
            spark_runtime::gpu::DevicePtr::NULL,
            std::time::Instant::now(),
            None,
            None,
            None,
            None,
            Some(TOOL_CLOSE),
            &[],
            false,
            false,
        );
        let a = active.pop().expect("sequence");
        (a.finished, a.output_tokens.len())
    };
    // cancelled: finished, the sampled token is not committed
    assert_eq!(run(true), (true, 1));
    // CONTROL: without the flag the same step commits the token and continues
    assert_eq!(run(false), (false, 2));
}

/// A blocking client that went away (its oneshot receiver dropped) has no
/// cancel flag. The serial decode path must still retire the sequence at
/// the next token instead of generating to `max_tokens` for nobody.
#[test]
fn a_disconnected_blocking_client_stops_the_serial_decode_path() {
    let _guard = HARD_STOP_TEST_LOCK.lock().expect("hard-stop test lock");
    let _reset = HardStopReset;
    set_im_start_hard_stop(0);
    let run = |disconnected: bool| {
        let model = model_support::greedy_model(vec![CONTENT]);
        let (tx, rx) = tokio::sync::oneshot::channel();
        let _live_receiver = if disconnected {
            drop(rx);
            None
        } else {
            Some(rx)
        };
        let mut a = active_seq(ResponseSink::Blocking(Some(tx)), CONTENT);
        a.finished = false;
        a.max_output_tokens = 100;
        a.remaining = 100;
        a.think_ended = false;
        let mut active = vec![a];
        process_decode_logits(
            &model,
            &mut active,
            spark_runtime::gpu::DevicePtr::NULL,
            std::time::Instant::now(),
            None,
            None,
            None,
            None,
            Some(TOOL_CLOSE),
            &[],
            false,
            false,
        );
        let a = active.pop().expect("sequence");
        (a.finished, a.output_tokens.len())
    };
    // disconnected: finished, the sampled token is not committed
    assert_eq!(run(true), (true, 1));
    // CONTROL: with a live receiver the same step commits the token and continues
    assert_eq!(run(false), (false, 2));
}

/// The first generated token is sampled from the prefill logits; with
/// `top_logprobs` requested its logprobs entry must come with it (it used to be
/// missing, so logprobs.content had completion_tokens - 1 entries), and its
/// stream event must carry them.
#[test]
fn the_first_token_gets_its_logprobs_from_the_prefill_logits() {
    let at = |temperature: f32| spark_runtime::sampler::SamplingParams {
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
    };
    let model = model_support::logits_model(vec![0.0, 3.0, 1.0, 2.0]);
    let null = spark_runtime::gpu::DevicePtr::NULL;
    let (tok, lp) = sample_token_with_logprobs(&model, null, &at(0.0), &[], &[], Some(2)).unwrap();
    assert_eq!(tok, 1);
    let lp = lp.expect("first-token logprobs");
    assert_eq!(lp.token_id, 1);
    assert_eq!(lp.top.iter().map(|t| t.0).collect::<Vec<_>>(), vec![1, 3]);
    let z = [0.0f32, 3.0, 1.0, 2.0]
        .iter()
        .map(|v| v.exp())
        .sum::<f32>()
        .ln();
    assert!((lp.logprob - (3.0 - z)).abs() < 1e-5);
    assert!(matches!(
        first_token_event(tok, &Some(lp), false),
        StreamEvent::TokenWithLogprobs(1, _)
    ));
    // Sampled (T>0) path returns an entry for whatever it drew.
    let (t2, lp2) = sample_token_with_logprobs(&model, null, &at(1.0), &[], &[], Some(1)).unwrap();
    assert_eq!(lp2.expect("sampled first-token logprobs").token_id, t2);
    // CONTROL: without top_logprobs nothing is produced (and the plain event is sent),
    // which is what every request got before -- one entry short.
    let (_, none) = sample_token_with_logprobs(&model, null, &at(0.5), &[], &[], None).unwrap();
    assert!(none.is_none());
    assert!(matches!(
        first_token_event(1, &None, false),
        StreamEvent::Token(1)
    ));
}

#[test]
fn committed_suppressed_reasoning_emits_usage_progress() {
    assert!(matches!(
        committed_token_event(7, &None, true),
        StreamEvent::CompletionProgress
    ));
    assert!(matches!(
        committed_token_event(7, &None, false),
        StreamEvent::Token(7)
    ));
}

#[test]
fn hidden_first_token_keeps_logprobs_without_exposing_a_token_event() {
    let event = first_token_event(EOS, &Some(token_logprobs(EOS)), true);
    match event {
        StreamEvent::CompletionProgressWithLogprobs(lp) => assert_eq!(lp.token_id, EOS),
        _ => panic!("terminal first token must be hidden and retain its logprobs"),
    }
}

#[test]
fn speculative_hidden_tokens_publish_one_logprob_progress_event() {
    let cases = [(EOS, false, true), (CONTENT, true, false)];
    for (token, suppress_reasoning, should_finish) in cases {
        let (tx, mut rx) = tokio::sync::mpsc::channel(4);
        let mut a = active_seq(ResponseSink::Streaming(tx), CONTENT);
        a.output_tokens.clear();
        a.logprobs_data.clear();
        a.finished = false;
        a.remaining = 10;
        a.max_output_tokens = 10;
        a.inside_thinking = suppress_reasoning;
        a.enable_thinking = !suppress_reasoning;
        a.think_ended = false;

        emit_token(&mut a, token, Some(token_logprobs(token)));

        assert_eq!(a.output_tokens, vec![token]);
        assert_eq!(a.logprobs_data.len(), 1);
        assert_eq!(a.logprobs_data[0].token_id, token);
        assert_eq!(a.finished, should_finish);
        match rx.try_recv().expect("one hidden-token event") {
            StreamEvent::CompletionProgressWithLogprobs(lp) => {
                assert_eq!(lp.token_id, token)
            }
            _ => panic!("hidden token must use logprob-bearing progress"),
        }
        assert!(matches!(
            rx.try_recv(),
            Err(tokio::sync::mpsc::error::TryRecvError::Empty)
        ));
    }
}

#[test]
fn serial_terminal_eos_publishes_one_logprob_progress_event() {
    let _guard = HARD_STOP_TEST_LOCK.lock().expect("hard-stop test lock");
    let _reset = HardStopReset;
    set_im_start_hard_stop(0);

    let mut logits = vec![0.0; CONTENT as usize + 1];
    logits[EOS as usize] = 8.0;
    let model = model_support::logits_model(logits);
    let (tx, mut rx) = tokio::sync::mpsc::channel(4);
    let mut a = active_seq(ResponseSink::Streaming(tx), CONTENT);
    a.output_tokens.clear();
    a.logprobs_data.clear();
    a.finished = false;
    a.remaining = 10;
    a.max_output_tokens = 10;
    a.top_logprobs = Some(2);
    a.think_ended = false;
    let mut active = vec![a];

    process_decode_logits(
        &model,
        &mut active,
        spark_runtime::gpu::DevicePtr::NULL,
        std::time::Instant::now(),
        None,
        None,
        None,
        None,
        Some(TOOL_CLOSE),
        &[],
        false,
        false,
    );

    let a = active.pop().expect("sequence");
    assert_eq!(a.output_tokens, vec![EOS]);
    assert_eq!(a.logprobs_data.len(), 1);
    assert_eq!(a.logprobs_data[0].token_id, EOS);
    assert!(a.finished);
    match rx.try_recv().expect("terminal logprob progress") {
        StreamEvent::CompletionProgressWithLogprobs(lp) => assert_eq!(lp.token_id, EOS),
        _ => panic!("serial EOS must use logprob-bearing progress"),
    }
    assert!(matches!(
        rx.try_recv(),
        Err(tokio::sync::mpsc::error::TryRecvError::Empty)
    ));
}

fn first_thinking_eos(
    honor_eos_inside_thinking: bool,
) -> (
    impl spark_model::traits::Model,
    ActiveSeq,
    tokio::sync::oneshot::Receiver<anyhow::Result<crate::api::InferenceResponse>>,
) {
    let mut logits = vec![0.0; CONTENT as usize + 1];
    logits[EOS as usize] = 8.0;
    let model = model_support::logits_model(logits);
    let (tx, rx) = tokio::sync::oneshot::channel();
    let mut a = active_seq(ResponseSink::Blocking(Some(tx)), CONTENT);
    a.finished = false;
    a.output_tokens = vec![CONTENT];
    a.logprobs_data = vec![token_logprobs(CONTENT)];
    a.max_output_tokens = 10;
    a.remaining = 10;
    a.inside_thinking = true;
    a.enable_thinking = true;
    a.think_ended = false;
    a.think_just_ended = false;
    a.thinking_tokens = 1;
    a.think_end_token = Some(TOOL_CLOSE);
    a.top_logprobs = Some(2);
    let mut active = vec![a];

    process_decode_logits(
        &model,
        &mut active,
        spark_runtime::gpu::DevicePtr::NULL,
        std::time::Instant::now(),
        Some(TOOL_CLOSE),
        None,
        None,
        None,
        Some(TOOL_CLOSE),
        &[],
        false,
        honor_eos_inside_thinking,
    );
    // The receiver must outlive every decode step under test. Dropping it
    // here correctly makes the next step retire a disconnected client.
    (model, active.pop().expect("sequence"), rx)
}

#[test]
fn plain_decode_qwen_policy_closes_thinking_on_first_eos_then_stops_on_second() {
    let (_, baseline, _baseline_rx) = first_thinking_eos(false);
    assert!(baseline.inside_thinking && !baseline.think_ended);
    assert!(!baseline.finished);
    assert_eq!(baseline.output_tokens, vec![CONTENT]);
    assert_eq!(baseline.logprobs_data.len(), 1);
    assert_eq!(baseline.logprobs_data[0].token_id, CONTENT);

    let (model, mut honored, honored_rx) = first_thinking_eos(true);
    assert!(!honored.inside_thinking && honored.think_ended);
    assert!(honored.think_just_ended);
    assert!(
        !honored.finished,
        "the implicit close discards the first EOS"
    );
    assert_eq!(honored.output_tokens, vec![CONTENT]);
    assert_eq!(honored.logprobs_data.len(), 1);
    assert_eq!(honored.logprobs_data[0].token_id, CONTENT);
    assert_eq!(honored.last_token, EOS, "the sampled EOS is fed back");
    assert!(!honored.sink.is_closed(), "the client is still connected");

    let mut active = vec![honored];
    process_decode_logits(
        &model,
        &mut active,
        spark_runtime::gpu::DevicePtr::NULL,
        std::time::Instant::now(),
        Some(TOOL_CLOSE),
        None,
        None,
        None,
        Some(TOOL_CLOSE),
        &[],
        false,
        true,
    );
    honored = active.pop().expect("sequence");
    assert!(
        honored.finished,
        "the second EOS terminates in content phase"
    );
    assert_eq!(honored.output_tokens, vec![CONTENT, EOS]);
    assert_eq!(honored.logprobs_data.len(), 2);
    assert_eq!(honored.logprobs_data[0].token_id, CONTENT);
    assert_eq!(honored.logprobs_data[1].token_id, EOS);
    drop(honored_rx);
}

#[test]
fn plain_decode_retires_disconnect_after_implicit_thinking_close_without_committing_eos() {
    let (model, honored, rx) = first_thinking_eos(true);
    assert!(!honored.finished && honored.think_just_ended);
    assert!(!honored.sink.is_closed());
    let remaining = honored.remaining;
    drop(rx);
    assert!(honored.sink.is_closed());
    let mut active = vec![honored];

    process_decode_logits(
        &model,
        &mut active,
        spark_runtime::gpu::DevicePtr::NULL,
        std::time::Instant::now(),
        Some(TOOL_CLOSE),
        None,
        None,
        None,
        Some(TOOL_CLOSE),
        &[],
        false,
        true,
    );

    let retired = active.pop().expect("sequence");
    assert!(retired.finished, "a disconnected client is terminal");
    assert_eq!(retired.output_tokens, vec![CONTENT]);
    assert_eq!(retired.logprobs_data.len(), 1);
    assert_eq!(retired.logprobs_data[0].token_id, CONTENT);
    assert_eq!(retired.remaining, remaining, "no content position commits");
}

#[test]
fn speculative_implicit_close_eos_discards_token_logprob_and_stream_event() {
    let (tx, mut rx) = tokio::sync::mpsc::channel(4);
    let mut a = active_seq(ResponseSink::Streaming(tx), CONTENT);
    a.finished = false;
    a.output_tokens = vec![CONTENT];
    a.logprobs_data = vec![token_logprobs(CONTENT)];
    a.max_output_tokens = 10;
    a.remaining = 10;
    a.inside_thinking = true;
    a.enable_thinking = true;
    a.think_ended = false;
    a.think_just_ended = false;
    a.thinking_tokens = 1;
    a.think_end_token = Some(TOOL_CLOSE);

    emit_token_with_eos_policy(&mut a, EOS, Some(token_logprobs(EOS)), true);

    assert_eq!(a.output_tokens, vec![CONTENT]);
    assert_eq!(a.logprobs_data.len(), 1);
    assert_eq!(a.logprobs_data[0].token_id, CONTENT);
    assert!(!a.finished);
    assert!(!a.inside_thinking && a.think_ended && a.think_just_ended);
    assert_eq!(a.last_token, CONTENT, "caller owns speculative feedback");
    assert!(matches!(rx.try_recv(), Ok(StreamEvent::ThinkingDone)));
    assert!(matches!(
        rx.try_recv(),
        Err(tokio::sync::mpsc::error::TryRecvError::Empty)
    ));
}

#[test]
fn speculative_zero_budget_close_discards_logprob_and_publishes_boundary() {
    let (tx, mut rx) = tokio::sync::mpsc::channel(4);
    let mut a = active_seq(ResponseSink::Streaming(tx), CONTENT);
    a.finished = false;
    a.output_tokens = vec![CONTENT];
    a.logprobs_data = vec![token_logprobs(CONTENT)];
    a.max_output_tokens = 10;
    a.remaining = 10;
    a.inside_thinking = true;
    a.enable_thinking = true;
    a.think_ended = false;
    a.think_just_ended = false;
    a.thinking_budget = Some(0);
    a.think_end_token = Some(TOOL_CLOSE);

    emit_token_with_eos_policy(&mut a, TOOL_CLOSE, Some(token_logprobs(TOOL_CLOSE)), true);

    assert_eq!(a.output_tokens, vec![CONTENT]);
    assert_eq!(a.logprobs_data.len(), 1);
    assert_eq!(a.logprobs_data[0].token_id, CONTENT);
    assert!(!a.inside_thinking && a.think_ended && a.think_just_ended);
    assert!(matches!(rx.try_recv(), Ok(StreamEvent::ThinkingDone)));
    assert!(matches!(
        rx.try_recv(),
        Err(tokio::sync::mpsc::error::TryRecvError::Empty)
    ));
}

#[test]
fn chatml_hard_stop_publishes_hidden_logprobs_once() {
    let _guard = HARD_STOP_TEST_LOCK.lock().expect("hard-stop test lock");
    let _reset = HardStopReset;
    set_im_start_hard_stop(IM_START);

    let (tx, mut rx) = tokio::sync::mpsc::channel(4);
    let mut a = active_seq(ResponseSink::Streaming(tx), CONTENT);
    a.output_tokens.clear();
    a.logprobs_data.clear();
    a.finished = false;

    emit_token(&mut a, IM_START, Some(token_logprobs(IM_START)));

    assert_eq!(a.output_tokens, vec![IM_START]);
    assert_eq!(a.logprobs_data.len(), 1);
    assert!(a.finished);
    match rx.try_recv().expect("hard-stop logprob progress") {
        StreamEvent::CompletionProgressWithLogprobs(lp) => assert_eq!(lp.token_id, IM_START),
        _ => panic!("ChatML hard stop must not enter the text decoder"),
    }
    assert!(matches!(
        rx.try_recv(),
        Err(tokio::sync::mpsc::error::TryRecvError::Empty)
    ));
}

#[test]
fn an_ignore_eos_sequence_runs_through_the_chatml_role_boundary() {
    let (tx, _rx) = tokio::sync::oneshot::channel();
    let mut a = active_seq(ResponseSink::Blocking(Some(tx)), CONTENT);
    a.finished = false;
    assert!(is_chatml_role_boundary(&a, IM_START, Some(IM_START)));

    a.ignore_eos = true;
    assert!(!is_chatml_role_boundary(&a, IM_START, Some(IM_START)));
    assert!(!commit_chatml_role_boundary(
        &mut a,
        IM_START,
        Some(IM_START),
        &None
    ));
    assert!(!a.finished, "vLLM ignore_eos runs on to max_tokens");
    assert_eq!(
        a.output_tokens,
        vec![CONTENT],
        "the boundary is not committed as a stop"
    );
}

/// Finish one sequence through each sink with a guard stop cause set.
fn delivered_guard_stop(
    last_token: u32,
    cause: &'static str,
) -> [(String, Option<&'static str>); 2] {
    let model = model_support::model();

    let (blocking_tx, mut blocking_rx) = tokio::sync::oneshot::channel();
    let mut blocking = active_seq(ResponseSink::Blocking(Some(blocking_tx)), last_token);
    blocking.eos_tokens = vec![EOS];
    blocking.stop_cause = Some(cause);
    finish_sequence_with_cache(&model, &mut blocking, false);
    let response = blocking_rx
        .try_recv()
        .expect("blocking response must be delivered")
        .expect("blocking finish must succeed");

    let (streaming_tx, mut streaming_rx) = tokio::sync::mpsc::channel(1);
    let mut streaming = active_seq(ResponseSink::Streaming(streaming_tx), last_token);
    streaming.eos_tokens = vec![EOS];
    streaming.stop_cause = Some(cause);
    finish_sequence_with_cache(&model, &mut streaming, false);
    let streamed = match streaming_rx
        .try_recv()
        .expect("streaming Done must be delivered")
    {
        StreamEvent::Done {
            finish_reason,
            stop_cause,
            ..
        } => (finish_reason, stop_cause),
        _ => panic!("finish must deliver a Done event"),
    };

    [(response.finish_reason, response.stop_cause), streamed]
}

#[test]
fn guard_stops_finish_as_stop_with_their_cause_on_both_sinks() {
    let _guard = HARD_STOP_TEST_LOCK.lock().expect("hard-stop test lock");
    let _reset = HardStopReset;
    set_im_start_hard_stop(IM_START);

    for cause in ["watchdog", "catastrophic_guard"] {
        for (reason, delivered_cause) in delivered_guard_stop(CONTENT, cause) {
            assert_eq!(reason, "stop", "{cause}: a guard stop is not exhaustion");
            assert_eq!(delivered_cause, Some(cause));
        }
    }
}

#[test]
fn length_is_reserved_for_budget_exhaustion() {
    let _guard = HARD_STOP_TEST_LOCK.lock().expect("hard-stop test lock");
    let _reset = HardStopReset;
    set_im_start_hard_stop(IM_START);

    let reason = |last, cause| finish_reason(&[last], &[EOS], Some(TOOL_CLOSE), false, cause);
    assert_eq!(reason(CONTENT, None), "length");
    assert_eq!(reason(CONTENT, Some("watchdog")), "stop");
    // A tool call that closed keeps its own reason even if a guard fired.
    assert_eq!(reason(TOOL_CLOSE, Some("watchdog")), "tool_calls");
    let (blocking, streaming) = delivered_finish_reasons(CONTENT);
    assert_eq!(
        (blocking.as_str(), streaming.as_str()),
        ("length", "length")
    );
}

#[test]
fn watchdog_interventions_name_their_stop_cause() {
    assert_eq!(
        stop_cause_for("catastrophic_content_loop_watchdog"),
        "catastrophic_guard"
    );
    for code in [
        "content_loop_watchdog",
        "fuzzy_repeat_watchdog",
        "inter_tool_prose_watchdog",
        "repeated_think_end_watchdog",
    ] {
        assert_eq!(stop_cause_for(code), "watchdog", "{code}");
    }
}

/// Stability action #4: a client cancel or disconnect mid-generation must
/// release the request's state rows (KV blocks, recurrent slot, PLE history,
/// position) exactly like a natural finish. Both go `finished = true` ->
/// `retire_finished_sequences` -> `finish_sequence_with_cache` ->
/// `Model::free_sequence`; this pins that the pool traffic is identical and
/// leaves nothing owned.
#[test]
fn a_cancelled_request_releases_state_exactly_like_a_natural_finish() {
    use crate::scheduler::test_fixtures::model::recording;
    let _guard = HARD_STOP_TEST_LOCK.lock().expect("hard-stop test lock");
    let _reset = HardStopReset;
    set_im_start_hard_stop(0);
    #[derive(Debug, PartialEq)]
    enum End {
        Natural,
        Cancelled,
        Disconnected,
    }
    let run = |end: End| {
        let m = recording();
        let token = if end == End::Natural { EOS } else { CONTENT };
        *m.argmax.lock().unwrap() = vec![token];
        let (tx, rx) = tokio::sync::oneshot::channel();
        let _live = if end == End::Disconnected {
            drop(rx);
            None
        } else {
            Some(rx)
        };
        let mut a = active_seq(ResponseSink::Blocking(Some(tx)), CONTENT);
        a.seq.slot_idx = 2;
        a.finished = false;
        a.max_output_tokens = 100;
        a.remaining = 100;
        a.think_ended = false;
        a.cancel_flag = Some(std::sync::Arc::new(std::sync::atomic::AtomicBool::new(
            end == End::Cancelled,
        )));
        let before = (
            m.frees.lock().unwrap().len(),
            m.compacts.lock().unwrap().len(),
        );
        assert_eq!(before, (0, 0), "pool idle before the request");
        let mut active = vec![a];
        process_decode_logits(
            &m,
            &mut active,
            spark_runtime::gpu::DevicePtr::NULL,
            std::time::Instant::now(),
            None,
            None,
            None,
            None,
            Some(TOOL_CLOSE),
            &[],
            false,
            false,
        );
        assert!(
            active[0].finished,
            "{end:?} must end the sequence this step"
        );
        let mut cache = vec![true];
        retire_finished_sequences(&m, &mut active, &mut cache);
        assert!(
            active.is_empty() && cache.is_empty(),
            "{end:?}: nothing left in the batch"
        );
        let frees = m.frees.lock().unwrap().clone();
        let compacts = m.compacts.lock().unwrap().clone();
        let events: Vec<String> = m
            .lifecycle_events
            .lock()
            .unwrap()
            .iter()
            .filter(|e| e.starts_with("free:"))
            .cloned()
            .collect();
        (frees, compacts, events)
    };
    let natural = run(End::Natural);
    assert_eq!(natural.0, vec![2], "natural finish frees its slot once");
    assert!(natural.1.is_empty());
    assert_eq!(
        run(End::Cancelled),
        natural,
        "cancel releases exactly like a finish"
    );
    assert_eq!(
        run(End::Disconnected),
        natural,
        "disconnect releases exactly like a finish"
    );
}

/// C3: a timed-out request is truncated, so `finish_reason` stays `length`
/// even though `stop_cause` carries the timeout for `flux_stop_reason`.
#[test]
fn a_timeout_stop_cause_keeps_finish_reason_length() {
    use crate::api::request_contract::{TIMEOUT_STALL_CAUSE, TIMEOUT_TOTAL_CAUSE};
    for cause in [TIMEOUT_STALL_CAUSE, TIMEOUT_TOTAL_CAUSE] {
        assert_eq!(
            finish_reason(&[CONTENT], &[EOS], Some(TOOL_CLOSE), false, Some(cause)),
            "length"
        );
    }
    assert_eq!(
        finish_reason(
            &[CONTENT],
            &[EOS],
            Some(TOOL_CLOSE),
            false,
            Some("watchdog")
        ),
        "stop"
    );
}
