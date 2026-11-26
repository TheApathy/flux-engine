// SPDX-License-Identifier: AGPL-3.0-only

//! Swap-out/resume I/O and the authoritative CPU metadata mapping.

use super::*;

fn ensure_swappable_victim(active: &[ActiveSeq], victim_idx: usize) -> Result<()> {
    anyhow::ensure!(
        active
            .get(victim_idx)
            .is_some_and(|victim| victim.grammar_state.is_none()),
        "refusing to swap a missing or grammar-active sequence"
    );
    Ok(())
}

/// Swap out an active sequence to disk, freeing its GPU blocks.
///
/// Removes the sequence at `victim_idx` from `active`, saves its state
/// to a swap file, frees GPU resources, and returns a `SwappedSeq`.
pub(in crate::scheduler) fn swap_out_sequence(
    model: &dyn Model,
    active: &mut Vec<ActiveSeq>,
    victim_idx: usize,
    spill: &mut KvSpillManager,
) -> Result<SwappedSeq> {
    ensure_swappable_victim(active, victim_idx)?;
    // Persist while the victim is still active and owns its original SSM slot.
    // Compacting first copies the swap-remove replacement into that slot, so
    // serializing afterwards pairs the victim's KV with the replacement's SSM
    // recurrent state. It also makes a save error destructive by removing the
    // request before any durable snapshot exists.
    let (swap_id, mut writer) = spill.create_file()?;
    if let Err(e) = model.save_sequence_state(&active[victim_idx].seq, &mut writer) {
        drop(writer);
        let _ = spill.remove_file(swap_id);
        return Err(e);
    }
    drop(writer);
    spill.record_usage(swap_id);

    let mut a = active.swap_remove(victim_idx);

    let num_blocks = a.seq.block_table.len();
    let seq_len = a.seq.seq_len;
    let tokens = a.seq.tokens.clone();
    proposal_lifecycle::clear(
        &mut a.pending_drafts,
        &mut a.pending_tree_payload,
        &mut a.pending_dflash_sampled_proposal,
    );

    // Free GPU resources (KV blocks + SSM slot) BEFORE compacting the
    // swapped-in survivor into the victim's slot: `compact_sequence` only
    // accepts a target it can claim from the pool free list (upstream
    // 11e83f5b, claim-first), so the victim must have released its slot.
    // The destination is the VICTIM's own slot, never the vector index,
    // which may belong to a chunked prefill that is not in `active`. A
    // refused or failed compaction leaves the survivor in place; the victim
    // is already saved and freed, so nothing is lost either way.
    let victim_slot = a.seq.slot_idx;
    if let Err(err) = a.continuation_record_mut().finish_or_release(model, false) {
        let _ = spill.remove_file(swap_id);
        send_error_to_sink(&mut a.sink, &format!("Swap-out failed: {err:#}"));
        return Err(err);
    }
    if let Err(err) = model.ep_broadcast_cmd(0xFFFFFFF1) {
        tracing::error!("Swap-out EP broadcast failed: {err:#}");
    }
    if model.hc_mult() == 0
        && victim_idx < active.len()
        && victim_slot != usize::MAX
        && active[victim_idx].seq.slot_idx > victim_slot
    {
        super::mod_helpers::compact_survivor_into_vacated_slot(
            model,
            &mut active[victim_idx],
            victim_slot,
            "swap-out",
        );
    }

    Ok(pack_swapped_seq(a, tokens, seq_len, num_blocks, swap_id))
}

fn pack_swapped_seq(
    a: ActiveSeq,
    tokens: Vec<u32>,
    seq_len: usize,
    num_blocks: usize,
    swap_id: u64,
) -> SwappedSeq {
    SwappedSeq {
        tokens,
        session_hash: a.session_hash,
        prompt_prefix_hash: a.prompt_prefix_hash,
        prefix_hash: a.prefix_hash,
        seq_len,
        num_blocks,
        last_token: a.last_token,
        output_tokens: a.output_tokens,
        max_output_tokens: a.max_output_tokens,
        remaining: a.remaining,
        min_tokens: a.min_tokens,
        eos_tokens: a.eos_tokens,
        ignore_eos: a.ignore_eos,
        sink: a.sink,
        intervention_ledger: a.intervention_ledger,
        cancel_flag: a.cancel_flag,
        temperature: a.temperature,
        top_k: a.top_k,
        top_p: a.top_p,
        top_n_sigma: a.top_n_sigma,
        min_p: a.min_p,
        repetition_penalty: a.repetition_penalty,
        presence_penalty: a.presence_penalty,
        frequency_penalty: a.frequency_penalty,
        repetition_penalty_window: a.repetition_penalty_window,
        lz_penalty: a.lz_penalty,
        dry_multiplier: a.dry_multiplier,
        dry_base: a.dry_base,
        dry_allowed_length: a.dry_allowed_length,
        dry_sequence_breakers: a.dry_sequence_breakers,
        logit_bias: a.logit_bias,
        inside_thinking: a.inside_thinking,
        enable_thinking: a.enable_thinking,
        thinking_budget: a.thinking_budget,
        spontaneous_think_budget: a.spontaneous_think_budget,
        thinking_tokens: a.thinking_tokens,
        force_end_thinking: a.force_end_thinking,
        consecutive_confident: a.consecutive_confident,
        in_code_fence: a.in_code_fence,
        think_end_token: a.think_end_token,
        think_start_token: a.think_start_token,
        think_ended: a.think_ended,
        think_just_ended: a.think_just_ended,
        think_skip_count: a.think_skip_count,
        post_think_gate_steps: a.post_think_gate_steps,
        require_tool_call: a.require_tool_call,
        suppress_tool_call: a.suppress_tool_call,
        disable_mtp: a.disable_mtp,
        content_started: a.content_started,
        content_tokens: a.content_tokens,
        prose_tokens_since_last_tool: a.prose_tokens_since_last_tool,
        think_watchdog_fires: a.think_watchdog_fires,
        rollback_count: a.rollback_count,
        tool_call_start_token: a.tool_call_start_token,
        tool_call_opened: a.tool_call_opened,
        inside_tool_body: a.inside_tool_body,
        tool_call_end_token: a.tool_call_end_token,
        grammar_state: a.grammar_state,
        last_token_time: a.last_token_time,
        request_start: a.request_start,
        decode_start: a.decode_start,
        seed: a.seed,
        top_logprobs: a.top_logprobs,
        logprobs_data: a.logprobs_data,
        timeout_at: a.timeout_at,
        decode_timeout_at: a.decode_timeout_at,
        swap_id,
        cached_prompt_tokens: a.cached_prompt_tokens,
        adaptive: a.adaptive,
        difficulty_probe: a.difficulty_probe,
        think_nudge: a.think_nudge,
        budget_nudge: a.budget_nudge,
    }
}

/// Resume a swapped-out sequence by restoring its state from disk.
///
/// Since the scheduler removes `s` from the swapped queue before calling `resume_swapped_seq`,
/// any failure path here must:
/// 1. Deliver a terminal error to the request sink (`send_error_to_sink`) so the caller does not hang, and
/// 2. Reclaim any allocated `SequenceState` via `model.free_sequence` or, if reclamation fails,
///    move the `SequenceState` into the ownership-preserving quarantine registry (`quarantine_unreclaimed_sequence`),
///    activating permanent fail-stop (process restart required for production recovery) so subsequent
///    allocations do not blindly retry after a possibly-partial free.
pub(in crate::scheduler) fn resume_swapped_seq(
    _think_end_token: Option<u32>,
    _think_start_token: Option<u32>,
    model: &dyn Model,
    mut s: SwappedSeq,
    spill: &mut KvSpillManager,
) -> Result<ActiveSeq> {
    let (mut seq, _) = match super::continuation::ContinuationRecord::claim_sequence(
        model,
        s.prompt_prefix_hash,
    ) {
        Ok(pair) => pair,
        Err(err) => {
            send_error_to_sink(&mut s.sink, &format!("Swap-in failed: {err:#}"));
            return Err(err);
        }
    };
    let mut reader = match spill.open_file(s.swap_id) {
        Ok(r) => r,
        Err(err) => {
            let err = super::continuation::reclaim_or_quarantine_sequence(
                model,
                seq,
                err,
                "resume_swapped_seq(open_file)",
            );
            send_error_to_sink(&mut s.sink, &format!("Swap-in failed: {err:#}"));
            return Err(err);
        }
    };
    if let Err(err) = model.restore_sequence_state(&mut seq, s.num_blocks, &mut reader) {
        drop(reader);
        let err = super::continuation::reclaim_or_quarantine_sequence(
            model,
            seq,
            err,
            "resume_swapped_seq(restore_sequence_state)",
        );
        send_error_to_sink(&mut s.sink, &format!("Swap-in failed: {err:#}"));
        return Err(err);
    }
    drop(reader);
    if let Err(err) = spill.remove_file(s.swap_id) {
        let err = super::continuation::reclaim_or_quarantine_sequence(
            model,
            seq,
            err,
            "resume_swapped_seq(remove_file)",
        );
        send_error_to_sink(&mut s.sink, &format!("Swap-in failed: {err:#}"));
        return Err(err);
    }

    let mut resumed =
        restore_swapped_seq(seq, s, model.decode_rollback_ring_slots(), Instant::now());
    if let Some(ref proof) = resumed.reset_proof
        && let Err(err) = proof.verify_claim()
    {
        let slot_idx = resumed.seq.slot_idx;
        match resumed
            .continuation_record_mut()
            .finish_or_release(model, false)
        {
            Ok(()) => {
                send_error_to_sink(&mut resumed.sink, &format!("Swap-in failed: {err:#}"));
                return Err(err);
            }
            Err(free_err) => {
                let reason = format!(
                    "{err:#}; additionally failed to reclaim restored sequence slot {slot_idx}: {free_err:#}"
                );
                send_error_to_sink(&mut resumed.sink, &format!("Swap-in failed: {reason}"));
                let quarantined_total = super::continuation::quarantine_unreclaimed_sequence(
                    model,
                    resumed.seq,
                    reason.clone(),
                );
                return Err(anyhow::anyhow!(
                    "{reason} (SequenceState quarantined; fail-stop active with {quarantined_total} quarantined sequence(s); process restart required)"
                ));
            }
        }
    }
    Ok(resumed)
}

fn restore_swapped_seq(
    mut seq: SequenceState,
    s: SwappedSeq,
    rollback_ring_slots: usize,
    last_token_time: Instant,
) -> ActiveSeq {
    // Restore CPU-side metadata after the model has rebuilt device state.
    seq.tokens = s.tokens;
    seq.seq_len = s.seq_len;
    let reset_proof = if super::continuation::is_reset_proof_enabled() {
        Some(super::continuation::ResetProofPayload::from_restored_seq(
            &seq,
            s.prefix_hash,
        ))
    } else {
        None
    };

    ActiveSeq {
        stop_cause: None,
        seq,
        session_hash: s.session_hash,
        prompt_prefix_hash: s.prompt_prefix_hash,
        prefix_hash: s.prefix_hash,
        record_state: super::continuation::RecordState::Live,
        reset_proof,
        last_token: s.last_token,
        output_tokens: s.output_tokens,
        max_output_tokens: s.max_output_tokens,
        remaining: s.remaining,
        min_tokens: s.min_tokens,
        eos_tokens: s.eos_tokens,
        ignore_eos: s.ignore_eos,
        finished: false,
        sink: s.sink,
        intervention_ledger: s.intervention_ledger,
        cancel_flag: s.cancel_flag,
        temperature: s.temperature,
        top_k: s.top_k,
        top_p: s.top_p,
        top_n_sigma: s.top_n_sigma,
        min_p: s.min_p,
        repetition_penalty: s.repetition_penalty,
        presence_penalty: s.presence_penalty,
        frequency_penalty: s.frequency_penalty,
        repetition_penalty_window: s.repetition_penalty_window,
        lz_penalty: s.lz_penalty,
        dry_multiplier: s.dry_multiplier,
        dry_base: s.dry_base,
        dry_allowed_length: s.dry_allowed_length,
        dry_sequence_breakers: s.dry_sequence_breakers,
        logit_bias: s.logit_bias,
        inside_thinking: s.inside_thinking,
        enable_thinking: s.enable_thinking,
        thinking_budget: s.thinking_budget,
        spontaneous_think_budget: s.spontaneous_think_budget,
        thinking_tokens: s.thinking_tokens,
        force_end_thinking: s.force_end_thinking,
        consecutive_confident: s.consecutive_confident,
        in_code_fence: s.in_code_fence,
        think_end_token: s.think_end_token,
        think_start_token: s.think_start_token,
        think_ended: s.think_ended,
        think_just_ended: s.think_just_ended,
        think_skip_count: s.think_skip_count,
        post_think_gate_steps: s.post_think_gate_steps,
        require_tool_call: s.require_tool_call,
        suppress_tool_call: s.suppress_tool_call,
        disable_mtp: s.disable_mtp,
        content_started: s.content_started,
        content_tokens: s.content_tokens,
        prose_tokens_since_last_tool: s.prose_tokens_since_last_tool,
        think_watchdog_fires: s.think_watchdog_fires,
        rollback_count: s.rollback_count,
        // Decode-rollback SSM snapshots are GPU-resident and not part of
        // the disk swap image — a resumed sequence starts with an empty ring.
        ssm_rollback_ring: SsmDecodeRing::new(rollback_ring_slots),
        tool_call_start_token: s.tool_call_start_token,
        tool_call_opened: s.tool_call_opened,
        inside_tool_body: s.inside_tool_body,
        tool_call_end_token: s.tool_call_end_token,
        // The matcher is CPU-owned metadata, so pack/restore moves the exact
        // live FSM without serializing, cloning, recompiling, or resetting it.
        grammar_state: s.grammar_state,
        pending_drafts: Vec::new(),
        draft_origin: DraftOrigin::default(),
        last_verify_accepted: 0,
        self_context: Default::default(),
        pending_tree_payload: None,
        pending_dflash_sampled_proposal: None,
        last_token_time,
        request_start: s.request_start,
        decode_start: s.decode_start,
        seed: s.seed,
        top_logprobs: s.top_logprobs,
        logprobs_data: s.logprobs_data,
        timeout_at: s.timeout_at,
        decode_timeout_at: s.decode_timeout_at,
        stall_mark: StallMark::default(),
        adaptive: s.adaptive,
        cached_prompt_tokens: s.cached_prompt_tokens,
        difficulty_probe: s.difficulty_probe,
        think_nudge: s.think_nudge,
        budget_nudge: s.budget_nudge,
    }
}

#[cfg(test)]
#[path = "swap_lifecycle_grammar_tests.rs"]
mod grammar_tests;
#[cfg(test)]
#[path = "swap_lifecycle_tests.rs"]
mod tests;
