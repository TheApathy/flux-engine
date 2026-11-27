// SPDX-License-Identifier: AGPL-3.0-only

//! GPU-free fixtures shared by the scheduler lifecycle tests: a mock
//! `Model` that records slot traffic, plus minimal `ActiveSeq`,
//! `PrefillInProgress` and `InferenceRequest` values.

use super::*;

/// The recording mock model (`compacts`, `frees`, `fail_compact`,
/// `alloc_error`). Included so every test file sees one definition.
pub(super) mod model {
    use crate::scheduler::proposal_lifecycle::SchedulerProposalFrame;
    use spark_model::layers::DDTreePayload;

    include!("proposal_lifecycle_model_test_support.rs");

    pub(crate) fn recording() -> TestModel {
        TestModel::new(Ok(Vec::new()), None)
    }
}

/// A decode-phase sequence that owns SSM pool slot `slot`.
pub(super) fn active_seq(sink: ResponseSink, slot: usize) -> ActiveSeq {
    active_seq_with_ssm_ring(sink, slot, SsmDecodeRing::new(0))
}

/// Same as [`active_seq`] but with an explicit Phase-C decode-ring, so
/// hybrid-model (SSM) rollback tests can pre-populate boundary snapshots.
pub(super) fn active_seq_with_ssm_ring(
    sink: ResponseSink,
    slot: usize,
    ssm_rollback_ring: SsmDecodeRing,
) -> ActiveSeq {
    let now = Instant::now();
    let mut seq = model::seq_state();
    seq.slot_idx = slot;
    ActiveSeq {
        stop_cause: None,
        defer_emit: Default::default(),
        seq,
        session_hash: 0,
        prompt_prefix_hash: 0,
        prefix_hash: 0,
        record_state: super::continuation::RecordState::Live,
        reset_proof: None,
        last_token: 31,
        output_tokens: vec![31],
        max_output_tokens: 100,
        remaining: 99,
        min_tokens: 0,
        eos_tokens: vec![17],
        ignore_eos: false,
        finished: false,
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
        tool_call_end_token: Some(23),
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
        ssm_rollback_ring,
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

/// A chunked prefill that has already claimed a pool slot.
pub(super) fn prefill_in_progress(sink: ResponseSink) -> PrefillInProgress {
    PrefillInProgress {
        prompt_tokens: vec![1, 2, 3, 4],
        session_hash: 0,
        prompt_prefix_hash: 0,
        prefix_hash: 0,
        reset_proof: None,
        seq: model::seq_state(),
        chunk_offset: 2,
        chunk_budget: 0,
        max_tokens: 16,
        min_tokens: 0,
        eos_tokens: vec![17],
        ignore_eos: false,
        sink,
        intervention_ledger: None,
        think_nudge: None,
        cancel_flag: None,
        request_start: Instant::now(),
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
        enable_thinking: false,
        thinking_budget: None,
        spontaneous_think_budget: 512,
        require_tool_call: false,
        suppress_tool_call: false,
        disable_mtp: false,
        grammar_state: None,
        seed: None,
        top_logprobs: None,
        first_logprobs: None,
        timeout_at: None,
        decode_timeout_at: None,
    }
}

/// A blocking request as the API layer hands it to the scheduler.
pub(super) fn blocking_request(
    prompt_tokens: Vec<u32>,
) -> (
    InferenceRequest,
    tokio::sync::oneshot::Receiver<Result<InferenceResponse>>,
) {
    let (response_tx, rx) = tokio::sync::oneshot::channel();
    let req = InferenceRequest::Blocking {
        prompt_tokens,
        session_hash: 0,
        image_pixels: Vec::new(),
        max_tokens: 8,
        min_tokens: 0,
        ignore_eos: false,
        temperature: 0.0,
        top_k: 0,
        top_p: 1.0,
        top_n_sigma: 0.0,
        min_p: 0.0,
        repetition_penalty: 1.0,
        presence_penalty: 0.0,
        frequency_penalty: 0.0,
        dry_multiplier: 0.0,
        dry_base: 1.75,
        dry_allowed_length: 3,
        lz_penalty: 0.0,
        logit_bias: Vec::new(),
        stop_tokens: Vec::new(),
        enable_thinking: false,
        thinking_budget: None,
        require_tool_call: false,
        suppress_tool_call: false,
        disable_mtp: false,
        grammar_spec: None,
        seed: None,
        top_logprobs: None,
        timeout_at: None,
        decode_timeout_at: None,
        intervention_ledger: None,
        think_nudge: None,
        response_tx,
    };
    (req, rx)
}
