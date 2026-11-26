// SPDX-License-Identifier: AGPL-3.0-only

//! GPU-free policy and real speculative-emission regression tests.

use super::super::*;
use super::*;
use crate::grammar::{GrammarEngine, GrammarState};
use spark_model::traits::SequenceState;
use std::time::Instant;

const THINK_START: u32 = 128;
const THINK_END: u32 = 129;
const EOS: u32 = 130;

fn seq_state() -> SequenceState {
    SequenceState {
        tokens: Vec::new(),
        block_table: Vec::new(),
        seq_len: 0,
        qwen4_qsa_required: false,
        layer_states: Vec::new(),
        rotary_positions: spark_model::traits::RotaryPositions::identity(),
        proposer_state: None,
        proposer_state_alt: None,
        slot_idx: 0,
        marconi_skip_to: 0,
        session_hash: 0,
        verify_frame: None,
        chunked_prefill_meta: None,
        qwen4_prefill_lineage: spark_model::traits::Qwen4PrefillLineage::Unset,
        cached_prefix_tokens: 0,
        kv_valid_tokens: 0,
        prompt_len: 0,
        disk_block_ids: Vec::new(),
        mtp_lastk_host_buf: Vec::new(),
        mtp_lastk_host_filled: 0,
        mtp_lastk_end_abs: 0,
        disk_last_offloaded_per_layer: Vec::new(),
    }
}

const TOOL_OPEN: u32 = 131;
const TOOL_CLOSE: u32 = 132;

/// A LOOPING (multi-call) Hermes tool grammar, walked up to the close tag of
/// its first call. A single-object JSON grammar is stop-after-first: since
/// `commit_tool_call_close` ends a stop-after-first grammar at its close, it
/// cannot exercise the multi-call re-entry this file is about (same fix as
/// think_spec_accept_tests, 8b4a4d11c).
fn looping_tool_grammar_before_close() -> (GrammarState, Vec<u32>) {
    let mut vocab: Vec<String> = (0u8..128).map(|byte| String::from(byte as char)).collect();
    vocab.push("<think>".to_string());
    vocab.push("</think>".to_string());
    vocab.push("<eos>".to_string());
    vocab.push("<tool_call>".to_string());
    vocab.push("</tool_call>".to_string());
    let tools =
        serde_json::from_value::<Vec<crate::tool_parser::ToolDefinition>>(serde_json::json!([{
            "type": "function",
            "function": {
                "name": "ping",
                "parameters": {"type": "object", "properties": {}}
            }
        }]))
        .unwrap();
    let mut engine = GrammarEngine::new(&vocab, &[EOS as i32]).unwrap();
    let compiled = engine
        .compile_hermes_tool_grammar(&tools, true, false)
        .unwrap();
    let mut state = GrammarState::new(&compiled, engine.vocab_size()).unwrap();
    let prefix: Vec<u32> = std::iter::once(TOOL_OPEN)
        .chain(r#"{"name":"ping","arguments":{}}"#.bytes().map(u32::from))
        .collect();
    for &token in &prefix {
        state.fill_bitmask();
        assert!(state.is_token_allowed(token));
        assert!(state.accept_token(token));
    }
    state.fill_bitmask();
    assert!(state.is_token_allowed(TOOL_CLOSE));
    assert!(!state.only_stop_tokens_remain());
    (state, prefix)
}

fn multi_call_seq() -> (ActiveSeq, tokio::sync::mpsc::Receiver<StreamEvent>) {
    let (tx, rx) = tokio::sync::mpsc::channel(4);
    let (grammar, prefix) = looping_tool_grammar_before_close();
    let now = Instant::now();
    (
        ActiveSeq {
            stop_cause: None,
            seq: seq_state(),
            session_hash: 0,
            prompt_prefix_hash: 0,
            prefix_hash: 0,
            record_state: crate::scheduler::continuation::RecordState::Live,
            reset_proof: None,
            last_token: 0,
            output_tokens: prefix,
            max_output_tokens: 100,
            remaining: 100,
            min_tokens: 0,
            eos_tokens: vec![EOS],
            ignore_eos: false,
            finished: false,
            sink: ResponseSink::Streaming(tx),
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
            thinking_budget: Some(512),
            spontaneous_think_budget: 512,
            thinking_tokens: 0,
            force_end_thinking: false,
            consecutive_confident: 0,
            in_code_fence: false,
            think_end_token: Some(THINK_END),
            think_start_token: Some(THINK_START),
            think_ended: true,
            think_just_ended: false,
            think_skip_count: 0,
            post_think_gate_steps: 1,
            tool_call_end_token: Some(TOOL_CLOSE),
            require_tool_call: false,
            tool_call_start_token: Some(TOOL_OPEN),
            tool_call_opened: true,
            inside_tool_body: true,
            suppress_tool_call: false,
            disable_mtp: false,
            content_started: true,
            content_tokens: 1,
            prose_tokens_since_last_tool: 0,
            think_watchdog_fires: 2,
            rollback_count: 0,
            ssm_rollback_ring: SsmDecodeRing::new(0),
            grammar_state: Some(grammar),
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
        },
        rx,
    )
}

#[test]
fn resolver_matches_the_closed_decay_table() {
    const BASES: [u32; 7] = [0, 1, 7, 8, 9, 64, 512];
    const FIRES: [u32; 7] = [0, 1, 2, 3, 4, 5, u32::MAX];
    const EXPECTED: [[u32; 7]; 7] = [
        [8, 8, 8, 8, 9, 64, 512],
        [8, 8, 8, 8, 8, 32, 256],
        [8, 8, 8, 8, 8, 16, 128],
        [8, 8, 8, 8, 8, 8, 64],
        [8, 8, 8, 8, 8, 8, 32],
        [8, 8, 8, 8, 8, 8, 32],
        [8, 8, 8, 8, 8, 8, 32],
    ];
    for (fire_idx, fires) in FIRES.into_iter().enumerate() {
        for (base_idx, base) in BASES.into_iter().enumerate() {
            assert_eq!(
                resolve_rethink_budget(base, fires),
                EXPECTED[fire_idx][base_idx]
            );
        }
    }
}

#[test]
fn speculative_multicall_reentry_uses_decayed_budget_without_emitting_start() {
    let (mut active, mut rx) = multi_call_seq();

    emit_token(&mut active, TOOL_CLOSE, None);
    assert!(!active.finished);
    assert!(!active.think_ended);
    assert!(matches!(rx.try_recv(), Ok(StreamEvent::Token(tok)) if tok == TOOL_CLOSE));
    let output_after_close = active.output_tokens.clone();

    emit_token(&mut active, THINK_START, None);

    assert!(active.inside_thinking);
    assert_eq!(active.thinking_budget, Some(128));
    assert_eq!(active.output_tokens, output_after_close);
    assert!(matches!(
        rx.try_recv(),
        Err(tokio::sync::mpsc::error::TryRecvError::Empty)
    ));
}

#[test]
fn speculative_suppressed_thinking_emits_completion_progress() {
    let (mut active, mut rx) = multi_call_seq();
    active.inside_thinking = true;
    active.think_ended = false;
    active.enable_thinking = false;

    let before = active.output_tokens.clone();
    emit_token(&mut active, b'x' as u32, None);

    assert_eq!(active.output_tokens[..before.len()], before[..]);
    assert_eq!(active.output_tokens[before.len()..], [b'x' as u32]);
    assert!(matches!(rx.try_recv(), Ok(StreamEvent::CompletionProgress)));
}
