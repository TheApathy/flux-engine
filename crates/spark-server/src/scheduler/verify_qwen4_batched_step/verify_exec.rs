// SPDX-License-Identifier: AGPL-3.0-only

//! Frame execution, slot-lane commit, rollback/abort, and token emission for batched verify.

use anyhow::Context;
use spark_model::layers::qwen4_verify_lanes::LaneGuard;
use spark_model::traits::Model;

use super::depth_controller::COHORT_ACCEPT_EWMA;
use super::dflash_budget::dflash_gumbel_cohort_num_drafts;
use super::planning::{
    DraftDepthPolicy, batched_k5_log_enabled, batched_ragged_enabled,
    batched_verify_proposer_eligible, indices_distinct_in_range,
};
use super::propose::{propose_next, propose_next_batch};
use super::stochastic_staging::{
    presample_batched_gumbel_frame_ragged, presample_batched_rejection_frame_ragged,
    stage_batched_gumbel_frame_if_eligible, stage_batched_rejection_frame_if_eligible,
};
use crate::scheduler::spec_accept_log::record_settled_frame;
use crate::scheduler::verify_dflash_step::{
    dflash_think_mask_verified_at_row, exact_accept_prefix, reargmax_bonus_skip_think,
    reconcile_terminal_verify_state, wide_verify_metric_k,
};
use crate::scheduler::*;

/// Run every planned frame under one physical-lane reservation. A read-only
/// refusal leaves all proposals untouched and permits scalar fallback. Once
/// begin is attempted, every failure is terminal for the whole cohort.
pub(in super::super) fn step_verify_qwen4_batched(
    model: &dyn Model,
    active: &mut [ActiveSeq],
    frames: &[Vec<usize>],
    num_drafts: usize,
    think: &ThinkSpecCtx<'_>,
) -> bool {
    if frames.is_empty() || !batched_verify_proposer_eligible(model) {
        return false;
    }
    let cohort_owned: Vec<usize>;
    let cohort: &[usize] = if frames.len() == 1 {
        &frames[0]
    } else {
        cohort_owned = frames.iter().flatten().copied().collect();
        &cohort_owned
    };
    if frames.iter().any(|frame| frame.len() < 2)
        || !indices_distinct_in_range(cohort, active.len())
    {
        let error = "batch cohort indices must be distinct, in range and grouped into frames";
        for &idx in cohort {
            if let Some(a) = active.get_mut(idx) {
                fail_speculative_step(a, "batched verify cohort plan", error);
            }
        }
        return true;
    }
    let windows: Vec<Vec<Vec<u32>>> = frames
        .iter()
        .map(|idxs| {
            idxs.iter()
                .map(|&idx| {
                    let a = &active[idx];
                    let mut window = Vec::with_capacity(a.pending_drafts.len() + 1);
                    window.push(a.last_token);
                    window.extend_from_slice(&a.pending_drafts);
                    window
                })
                .collect()
        })
        .collect();

    // All frame admissions are read-only and complete before any reservation,
    // proposal consumption or target mutation. C8 K5 is one C8 owner with two
    // C4 target frames, never two independent owners interleaved with drafts.
    for (idxs, tokens_per_seq) in frames.iter().zip(&windows) {
        let views: Vec<&spark_model::traits::SequenceState> =
            idxs.iter().map(|&idx| &active[idx].seq).collect();
        if let Err(error) = model.preflight_verify_qwen4_batched(tokens_per_seq, &views) {
            static REFUSED: std::sync::Once = std::sync::Once::new();
            REFUSED.call_once(|| {
                tracing::warn!(
                    cohort_sequences = cohort.len(),
                    fallback = true,
                    "QWEN4_BATCHED_K5_REFUSED cohort verified per sequence instead: {error:#}"
                );
            });
            return false;
        }
    }
    // In uniform mode (or TP=2), every lane in a cohort must share one width
    // K. Under ragged TP=1 mode (native MTP and DFlash2), heterogeneous
    // `K_s in 1..=8` pass through via `RaggedCohortWidthsGuard`.
    let mut cohort_ks_stack = [0usize; spark_model::layer::QWEN4_CROSS_SEQ_MAX_ROWS];
    let cohort_ks_heap;
    let mut min_k = usize::MAX;
    let mut max_k = 0usize;
    let cohort_ks: &[usize] = if cohort.len() <= cohort_ks_stack.len() {
        let mut cohort_ks_len = 0usize;
        for w in windows.iter().flatten() {
            let k = w.len();
            min_k = min_k.min(k);
            max_k = max_k.max(k);
            cohort_ks_stack[cohort_ks_len] = k;
            cohort_ks_len += 1;
        }
        &cohort_ks_stack[..cohort_ks_len]
    } else {
        let mut heap = Vec::with_capacity(cohort.len());
        for w in windows.iter().flatten() {
            let k = w.len();
            min_k = min_k.min(k);
            max_k = max_k.max(k);
            heap.push(k);
        }
        cohort_ks_heap = heap;
        &cohort_ks_heap
    };
    if cohort_ks.is_empty() {
        min_k = 0;
    }
    let is_ragged = min_k != max_k;
    if (is_ragged || min_k == 1) && (!batched_ragged_enabled() || model.is_ep()) {
        return false;
    }
    if let Err(error) = model.sync_secondary() {
        tracing::error!("sync_secondary (batched native MTP): {error:#}");
        for &idx in cohort {
            fail_speculative_step(&mut active[idx], "batched verify sync_secondary", &error);
        }
        return true;
    }

    let begin = {
        let _ragged_guard = (is_ragged || min_k == 1)
            .then(|| spark_model::layers::RaggedCohortWidthsGuard::enter(cohort_ks));
        batch_sequences_mut(active, cohort)
            .and_then(|mut seqs| model.begin_verify_qwen4_batch(&mut seqs, max_k))
    };
    if let Err(error) = begin {
        abort_cohort(model, active, cohort, "batched verify cohort begin", &error);
        return true;
    }
    let mut propose_idxs = Vec::with_capacity(cohort.len());
    for (idxs, tokens_per_seq) in frames.iter().zip(&windows) {
        match verify_frame(model, active, idxs, tokens_per_seq, cohort.len(), think) {
            Ok(proposals) => propose_idxs.extend(proposals),
            Err(error) => {
                abort_cohort(model, active, cohort, "batched verify frame", &error);
                return true;
            }
        }
        let views: Vec<&spark_model::traits::SequenceState> =
            idxs.iter().map(|&idx| &active[idx].seq).collect();
        if let Err(error) = model.end_verify_qwen4_batch_frame(&views) {
            abort_cohort(model, active, cohort, "batched verify frame end", &error);
            return true;
        }
    }
    let views: Vec<&spark_model::traits::SequenceState> =
        cohort.iter().map(|&idx| &active[idx].seq).collect();
    if let Err(error) = model.finish_verify_qwen4_batch(&views) {
        abort_cohort(
            model,
            active,
            cohort,
            "batched verify cohort finish",
            &error,
        );
        return true;
    }
    // Keep `vllm:kv_cache_usage_perc` live during sustained batched decode
    // (C=2..16 cohorts otherwise skip the scalar verify metric refresh).
    crate::metrics::compat_metrics::set_kv_blocks_free(model.num_free_blocks());

    // Every frame has settled, every live hidden handoff is sequence-owned,
    // and the single cohort reservation has been released before proposals.
    let propose_num_drafts = dflash_gumbel_cohort_num_drafts(model, active, num_drafts);
    let t_draft = batched_k5_log_enabled().then(Instant::now);
    let n_propose = propose_idxs.len();
    if n_propose >= 2 {
        propose_next_batch(model, active, &propose_idxs, propose_num_drafts);
    } else {
        let live_remaining = active.iter().filter(|a| !a.finished).count();
        let depth_policy =
            DraftDepthPolicy::for_cohort(model.qwen4_batched_k5_armed(), live_remaining);
        for idx in propose_idxs {
            propose_next(model, &mut active[idx], propose_num_drafts, depth_policy);
        }
    }
    if let Some(t_draft) = t_draft
        && n_propose > 0
    {
        tracing::info!(
            propose_seqs = n_propose,
            num_drafts = propose_num_drafts,
            draft_us = t_draft.elapsed().as_micros() as u64,
            "QWEN4_BATCHED_K5 draft frame"
        );
    }
    true
}

pub(in super::super) fn batch_sequences_mut<'a>(
    active: &'a mut [ActiveSeq],
    idxs: &[usize],
) -> anyhow::Result<Vec<&'a mut spark_model::traits::SequenceState>> {
    anyhow::ensure!(
        indices_distinct_in_range(idxs, active.len()),
        "batch sequence indices are not distinct and in range"
    );
    let base = active.as_mut_ptr();
    Ok(idxs
        .iter()
        .map(|&idx| {
            // SAFETY: the complete index list was checked above. The returned
            // mutable field borrows are disjoint and bounded by `active`'s borrow.
            let a = unsafe { &mut *base.add(idx) };
            &mut a.seq
        })
        .collect())
}

fn abort_cohort(
    model: &dyn Model,
    active: &mut [ActiveSeq],
    cohort: &[usize],
    stage: &str,
    error: impl std::fmt::Display,
) {
    if let Err(cleanup_error) = batch_sequences_mut(active, cohort)
        .and_then(|mut seqs| model.abort_verify_qwen4_batch(&mut seqs))
    {
        tracing::error!("batched native-MTP all-lane cleanup failed: {cleanup_error:#}");
    }
    // The model restores host extent only for uncommitted owner states. A
    // committed, published or retired lane must never be rewound here.
    for &idx in cohort {
        crate::scheduler::stochastic_verify_walk::invalidate_seq_rejection_q(&active[idx]);
        fail_speculative_step(&mut active[idx], stage, &error);
    }
    crate::metrics::compat_metrics::set_kv_blocks_free(model.num_free_blocks());
}

pub(super) fn verify_frame(
    model: &dyn Model,
    active: &mut [ActiveSeq],
    idxs: &[usize],
    tokens_per_seq: &[Vec<u32>],
    cohort_sequences: usize,
    think: &ThinkSpecCtx<'_>,
) -> anyhow::Result<Vec<usize>> {
    // Take each proposal frame; eligibility guarantees no tree or sampled-q.
    let mut drafts_per_seq: Vec<Vec<u32>> = Vec::with_capacity(idxs.len());
    let mut origins: Vec<DraftOrigin> = Vec::with_capacity(idxs.len());
    for &idx in idxs {
        let a = &mut active[idx];
        drafts_per_seq.push(std::mem::take(&mut a.pending_drafts));
        a.pending_tree_payload = None;
        a.pending_dflash_sampled_proposal = None;
        origins.push(a.draft_origin);
    }
    let mut k_per_seq_stack = [0usize; spark_model::layer::QWEN4_CROSS_SEQ_MAX_ROWS];
    let mut row_bases_stack = [0usize; spark_model::layer::QWEN4_CROSS_SEQ_MAX_ROWS];
    let k_per_seq_heap;
    let row_bases_heap;
    let mut total_rows = 0usize;
    let mut max_k = 0usize;
    let (k_per_seq, row_bases): (&[usize], &[usize]) =
        if tokens_per_seq.len() <= k_per_seq_stack.len() {
            for (s, window) in tokens_per_seq.iter().enumerate() {
                let k_s = window.len();
                k_per_seq_stack[s] = k_s;
                row_bases_stack[s] = total_rows;
                total_rows += k_s;
                max_k = max_k.max(k_s);
            }
            (
                &k_per_seq_stack[..tokens_per_seq.len()],
                &row_bases_stack[..tokens_per_seq.len()],
            )
        } else {
            let mut ks = Vec::with_capacity(tokens_per_seq.len());
            let mut bases = Vec::with_capacity(tokens_per_seq.len());
            for window in tokens_per_seq {
                let k_s = window.len();
                ks.push(k_s);
                bases.push(total_rows);
                total_rows += k_s;
                max_k = max_k.max(k_s);
            }
            k_per_seq_heap = ks;
            row_bases_heap = bases;
            (&k_per_seq_heap, &row_bases_heap)
        };
    let policy = crate::scheduler::stochastic_verify_walk::spec_verify_policy();
    let (seq_in_graph_gumbel, seq_in_graph_rejection) =
        if policy == crate::scheduler::stochastic_verify_walk::SpecVerifyPolicy::Rejection {
            model.stage_next_verify_gumbel_rows(&[]);
            (
                None,
                stage_batched_rejection_frame_if_eligible(
                    model,
                    active,
                    idxs,
                    &drafts_per_seq,
                    k_per_seq,
                    row_bases,
                    total_rows,
                    think,
                ),
            )
        } else {
            model.stage_next_verify_rejection_rows(&[]);
            (
                stage_batched_gumbel_frame_if_eligible(
                    model, active, idxs, k_per_seq, row_bases, total_rows, think,
                ),
                None,
            )
        };
    let t_verify = batched_k5_log_enabled().then(Instant::now);
    let verified = {
        let mut seqs = batch_sequences_mut(active, idxs)?;
        model.decode_verify_qwen4_batched(tokens_per_seq, &mut seqs)
    }
    .context("decode_verify_qwen4_batched")?;
    // Validate the complete result before any request can emit or index a row.
    if verified.len() != tokens_per_seq.len()
        || verified
            .iter()
            .zip(tokens_per_seq)
            .any(|(rows, window)| rows.len() != window.len())
    {
        anyhow::bail!(
            "batched verify target row count: expected {:?}, received {:?}",
            tokens_per_seq.iter().map(Vec::len).collect::<Vec<_>>(),
            verified.iter().map(Vec::len).collect::<Vec<_>>(),
        );
    }
    if let Some(t_verify) = t_verify {
        tracing::info!(
            seqs = idxs.len(),
            k = max_k,
            ks = ?k_per_seq,
            actual_frame_sequences = idxs.len(),
            actual_frame_rows = total_rows,
            cohort_sequences,
            fallback = false,
            forward_us = t_verify.elapsed().as_micros() as u64,
            seq_len_0 = active[idxs[0]].seq.seq_len,
            verify_input_0 = ?tokens_per_seq[0],
            verified_0 = ?verified[0],
            "QWEN4_BATCHED_K5 verify frame"
        );
    }

    // Consume in-graph GPU Gumbel-Max tokens or packed rejection words per
    // eligible sequence (falling back to one batched launch when eager verify
    // or dedup skipped in-graph capture). Always invalidate every sequence's
    // pending GPU rejection `q` fence once consumed/settled.
    let batched_rejection = presample_batched_rejection_frame_ragged(
        model,
        active,
        idxs,
        &drafts_per_seq,
        k_per_seq,
        row_bases,
        think,
        seq_in_graph_rejection.as_deref(),
    )?;
    for &idx in idxs {
        crate::scheduler::stochastic_verify_walk::invalidate_seq_rejection_q(&active[idx]);
    }
    let batched_gumbel = presample_batched_gumbel_frame_ragged(
        model,
        active,
        idxs,
        k_per_seq,
        row_bases,
        think,
        seq_in_graph_gumbel.as_deref(),
    )?;
    let presampled_for = |s: usize| -> Option<&[u32]> {
        let frame = batched_gumbel.as_ref()?;
        let start = (*frame.seq_offsets.get(s)?)?;
        frame.tokens.get(start..start + k_per_seq[s])
    };
    let presampled_rejection_for = |s: usize| -> Option<&[spark_model::PackedRejectionWord]> {
        let frame = batched_rejection.as_ref()?;
        let start = (*frame.seq_offsets.get(s)?)?;
        frame.words.get(start..start + k_per_seq[s])
    };

    // Phase 1: accept, emit, commit and save every sequence's hidden frame.
    let mut propose_idxs: Vec<usize> = Vec::with_capacity(idxs.len());
    if !model.supports_batched_verify_commit() || idxs.len() < 2 {
        for (s, &idx) in idxs.iter().enumerate() {
            let a = &mut active[idx];
            let lane = LaneGuard::enter_for_slot(a.seq.slot_idx).context("batched verify lane")?;
            let propose = finalize_commit(
                model,
                a,
                row_bases[s],
                &drafts_per_seq[s],
                &tokens_per_seq[s],
                &verified[s],
                origins[s],
                think,
                presampled_for(s),
                presampled_rejection_for(s),
            )?;
            drop(lane);
            if propose {
                propose_idxs.push(idx);
            }
        }
    } else {
        let mut pending: Vec<(usize, PendingBatchedCommit)> = Vec::with_capacity(idxs.len());
        for (s, &idx) in idxs.iter().enumerate() {
            let a = &mut active[idx];
            let lane = LaneGuard::enter_for_slot(a.seq.slot_idx).context("batched verify lane")?;
            let maybe_commit = accept_and_emit_row(
                model,
                a,
                row_bases[s],
                &drafts_per_seq[s],
                &tokens_per_seq[s],
                &verified[s],
                origins[s],
                think,
                presampled_for(s),
                presampled_rejection_for(s),
            )?;
            drop(lane);
            if let Some(item) = maybe_commit {
                pending.push((idx, item));
            }
        }
        if !pending.is_empty() {
            let pending_idxs: Vec<usize> = pending.iter().map(|(idx, _)| *idx).collect();
            let mut seqs = batch_sequences_mut(active, &pending_idxs)?;
            let mut commit_entries: Vec<(&mut spark_model::traits::SequenceState, usize, usize)> =
                seqs.iter_mut()
                    .zip(&pending)
                    .map(|(seq, (_, item))| (&mut **seq, item.num_accepted + 1, item.k_verify))
                    .collect();
            model
                .commit_verify_batch_cached(&mut commit_entries, 0)
                .context("batched verify commit")?;
        }
        for (idx, item) in &pending {
            let idx = *idx;
            let a = &mut active[idx];
            let lane = LaneGuard::enter_for_slot(a.seq.slot_idx).context("batched verify lane")?;
            let propose = finalize_post_commit(model, a, item)?;
            drop(lane);
            if propose {
                propose_idxs.push(idx);
            }
        }
    }

    let mut frame_accepted = 0usize;
    let mut frame_drafted = 0usize;
    let mut frame_rejected_seqs = 0usize;
    let mut frame_drafted_seqs = 0usize;
    for (s, &idx) in idxs.iter().enumerate() {
        let drafted = drafts_per_seq[s].len();
        if drafted > 0 && !active[idx].finished {
            let accepted = usize::from(active[idx].last_verify_accepted).min(drafted);
            frame_drafted_seqs += 1;
            frame_drafted += drafted;
            frame_accepted += accepted;
            if accepted < drafted {
                frame_rejected_seqs += 1;
            }
        }
    }
    if !model.proposer_is_dflash() {
        COHORT_ACCEPT_EWMA.observe_frame_seqs(
            frame_drafted_seqs,
            frame_accepted,
            frame_drafted,
            frame_rejected_seqs,
        );
    }

    Ok(propose_idxs)
}

/// Host acceptance metadata staged between batched verify commit and per-slot
/// hidden-frame save / proposer trim.
struct PendingBatchedCommit {
    row_base: usize,
    k_verify: usize,
    drafted: usize,
    thinking_entry: bool,
    emitted_before: usize,
    num_accepted: usize,
    frame_consumed_rows: usize,
    think_ended_this_step: bool,
}

/// Accept / emit / rollback / commit / hidden save for one sequence, the
/// neutral flat-chain or policy-walk path of `step_verify_dflash`. Runs inside
/// the sequence's verify lane. Returns whether the sequence should propose.
#[allow(clippy::too_many_arguments)]
fn finalize_commit(
    model: &dyn Model,
    a: &mut ActiveSeq,
    row_base: usize,
    drafts: &[u32],
    verify_tokens: &[u32],
    verified: &[u32],
    origin: DraftOrigin,
    think: &ThinkSpecCtx<'_>,
    presampled: Option<&[u32]>,
    presampled_rejection: Option<&[spark_model::PackedRejectionWord]>,
) -> anyhow::Result<bool> {
    let Some(pending) = accept_and_emit_row(
        model,
        a,
        row_base,
        drafts,
        verify_tokens,
        verified,
        origin,
        think,
        presampled,
        presampled_rejection,
    )?
    else {
        return Ok(false);
    };
    model
        .commit_verify_state_async(&mut a.seq, pending.num_accepted + 1, pending.k_verify)
        .context("batched verify commit")?;
    finalize_post_commit(model, a, &pending)
}

/// Save MTP hidden handoff, trim proposer state, and record the settled frame
/// receipt after the sequence's verify commit has been submitted.
fn finalize_post_commit(
    model: &dyn Model,
    a: &mut ActiveSeq,
    pending: &PendingBatchedCommit,
) -> anyhow::Result<bool> {
    let _depth_session =
        crate::scheduler::adaptive_depth::ActiveDepthSessionGuard::new(a.depth_controller_key());
    if pending.think_ended_this_step && !a.finished {
        // `</think>` was emitted during `accept_and_emit_row` before the
        // batched verify commit. Order the default stream after the
        // secondary-stream accepted-intermediate restore so the rollback
        // snapshot captures the committed prefix, not the rejected tail.
        model
            .sync_secondary()
            .context("batched verify think-end snapshot sync")?;
        rollback::snapshot_boundary_if_ssm(a, model);
    }
    let receipt = |a: &ActiveSeq, accepted: usize, consumed_rows: usize, terminal: bool| {
        crate::scheduler::spec_accept_log::CommittedFrameReceipt {
            verified_rows: pending.k_verify,
            drafted: pending.drafted,
            thinking_entry: pending.thinking_entry,
            consumed_rows,
            accepted,
            emitted: a.output_tokens.len().saturating_sub(pending.emitted_before),
            terminal,
        }
    };
    let num_accepted = pending.num_accepted;
    let frame_consumed_rows = pending.frame_consumed_rows;
    if model.proposer_is_dflash() {
        model
            .retire_dflash_frame_captures(pending.row_base, num_accepted, &mut a.seq, 0)
            .context("batched verify retire dflash captures")?;
    } else {
        model
            .save_hidden_for_mtp_sequence_in_frame(pending.row_base, num_accepted, &a.seq, 0)
            .context("batched verify save hidden")?;
    }
    model
        .trim_proposer_state(&mut a.seq, num_accepted, 0)
        .context("batched verify trim proposer")?;
    // Settled: count it (ATLAS_MTP_ACCEPT_LOG=1 only).
    record_settled_frame(&receipt(a, num_accepted, frame_consumed_rows, false));
    Ok(true)
}

/// Host accept, emit, extent rollback, and terminal retirement for one verified
/// sequence. Returns `Ok(None)` when the sequence finished terminally, or
/// `Ok(Some(..))` with the commit metadata for a non-terminal sequence.
#[allow(clippy::too_many_arguments)]
fn accept_and_emit_row(
    model: &dyn Model,
    a: &mut ActiveSeq,
    row_base: usize,
    drafts: &[u32],
    verify_tokens: &[u32],
    verified: &[u32],
    origin: DraftOrigin,
    think: &ThinkSpecCtx<'_>,
    presampled: Option<&[u32]>,
    presampled_rejection: Option<&[spark_model::PackedRejectionWord]>,
) -> anyhow::Result<Option<PendingBatchedCommit>> {
    let _depth_session =
        crate::scheduler::adaptive_depth::ActiveDepthSessionGuard::new(a.depth_controller_key());
    a.last_token_time = Instant::now();
    let k_verify = verify_tokens.len();
    let pre_verify_len = a
        .seq
        .seq_len
        .checked_sub(k_verify)
        .context("batched verify state extent underflow")?;
    anyhow::ensure!(
        a.seq.tokens.len() == a.seq.seq_len,
        "batched verify host token extent differs from its sequence length"
    );

    let mut masked_stack = [0u32; spark_model::layer::QWEN4_CROSS_SEQ_MAX_K];
    let mut masked_heap = Vec::new();
    let needs_think_mask = presampled.is_none()
        && presampled_rejection.is_none()
        && a.think_ended
        && !a.inside_thinking
        && (a.think_start_token.is_some_and(|t| verified.contains(&t))
            || a.think_end_token.is_some_and(|t| verified.contains(&t)));
    let verified: &[u32] = if needs_think_mask {
        if verified.len() <= masked_stack.len() {
            let dst = &mut masked_stack[..verified.len()];
            dst.copy_from_slice(verified);
            let _ = dflash_think_mask_verified_at_row(model, a, row_base, dst);
            dst
        } else {
            masked_heap.extend_from_slice(verified);
            let _ = dflash_think_mask_verified_at_row(model, a, row_base, &mut masked_heap);
            &masked_heap
        }
    } else {
        verified
    };
    // ATLAS_MTP_ACCEPT_LOG=1 receipt inputs, taken at frame entry.
    let thinking_frame = a.inside_thinking;
    let emitted_before = a.output_tokens.len();
    let receipt = |a: &ActiveSeq, accepted: usize, consumed_rows: usize, terminal: bool| {
        crate::scheduler::spec_accept_log::CommittedFrameReceipt {
            verified_rows: k_verify,
            drafted: drafts.len(),
            thinking_entry: thinking_frame,
            consumed_rows,
            accepted,
            emitted: a.output_tokens.len().saturating_sub(emitted_before),
            terminal,
        }
    };

    let terminal = |a: &mut ActiveSeq,
                    num_accepted: usize,
                    consumed_rows: usize|
     -> anyhow::Result<Option<PendingBatchedCommit>> {
        let plan = reconcile_terminal_verify_state(
            model,
            a,
            pre_verify_len,
            k_verify,
            num_accepted,
            consumed_rows,
            verify_tokens,
            false,
            None,
        )
        .context("terminal batched verify commit")?;
        if plan.committed_inputs > 0 {
            model
                .retire_verify_qwen4_batch_terminal(&a.seq)
                .context("terminal batched verify retirement")?;
        }
        record_settled_frame(&receipt(a, num_accepted, consumed_rows, true));
        Ok(None)
    };

    let policy_outcome = if think.enabled && a.inside_thinking {
        if presampled_rejection.is_some() {
            crate::scheduler::spec_policy_accept::run_dflash_thinking_rejection_accept_at_row(
                model,
                a,
                row_base,
                drafts,
                verified,
                think,
                presampled_rejection,
            )
        } else {
            crate::scheduler::think_spec_accept::run_dflash_thinking_accept_at_row_with_presampled(
                model, a, row_base, drafts, verified, think, presampled,
            )
        }
    } else if !a.inside_thinking
        && (content_policy_required(a, think)
            || a.require_tool_call
            || a.inside_tool_body
            || a.output_tokens.len() < a.min_tokens)
    {
        if presampled_rejection.is_some() {
            crate::scheduler::spec_policy_accept::run_dflash_content_accept_at_row_with_rejection(
                model,
                a,
                row_base,
                drafts,
                verified,
                think,
                presampled_rejection,
            )
        } else {
            crate::scheduler::spec_policy_accept::run_dflash_content_accept_at_row_with_presampled(
                model, a, row_base, drafts, verified, think, presampled,
            )
        }
    } else {
        None
    };

    let (num_accepted, frame_consumed_rows, think_ended_this_step) = if let Some(outcome) =
        policy_outcome
    {
        let num_accepted = outcome.num_accepted.min(k_verify.saturating_sub(1));
        let target_seq_len = pre_verify_len
            .checked_add(num_accepted + 1)
            .context("batched verify target extent overflow")?;
        let to_drop = a.seq.seq_len.saturating_sub(target_seq_len);
        if to_drop > 0 {
            a.seq.seq_len = target_seq_len;
            let pop_n = to_drop.min(a.seq.tokens.len());
            a.seq.tokens.truncate(a.seq.tokens.len() - pop_n);
        }
        a.last_verify_accepted = u16::try_from(num_accepted).unwrap_or(u16::MAX);
        match origin {
            DraftOrigin::RestStore => crate::rest_store::record_accepted(num_accepted),
            DraftOrigin::SelfContext => {
                crate::rest_store::self_context::record_accepted(num_accepted)
            }
            DraftOrigin::Proposer => {}
        }
        if a.finished || outcome.bonus.is_none() {
            return terminal(a, num_accepted, outcome.consumed_rows);
        }
        (
            num_accepted,
            outcome.consumed_rows,
            thinking_frame
                && crate::scheduler::think_spec_accept::outcome_ended_thinking(a, &outcome),
        )
    } else {
        let num_accepted = exact_accept_prefix(drafts, verified).min(k_verify.saturating_sub(1));
        let target_seq_len = pre_verify_len
            .checked_add(num_accepted + 1)
            .context("batched verify target extent overflow")?;
        let to_drop = a.seq.seq_len.saturating_sub(target_seq_len);
        if to_drop > 0 {
            a.seq.seq_len = target_seq_len;
            let pop_n = to_drop.min(a.seq.tokens.len());
            a.seq.tokens.truncate(a.seq.tokens.len() - pop_n);
        }
        a.last_verify_accepted = u16::try_from(num_accepted).unwrap_or(u16::MAX);
        match origin {
            DraftOrigin::RestStore => crate::rest_store::record_accepted(num_accepted),
            DraftOrigin::SelfContext => {
                crate::rest_store::self_context::record_accepted(num_accepted)
            }
            DraftOrigin::Proposer => {}
        }
        let mut consumed_rows = 0usize;
        for &tok in &drafts[..num_accepted] {
            if emit_token_for_verify(a, tok, None) {
                consumed_rows += 1;
            }
            if a.finished {
                return terminal(a, num_accepted, consumed_rows);
            }
        }
        let Some(&raw_bonus) = verified.get(num_accepted) else {
            anyhow::bail!("batched verify bonus row is missing");
        };
        let bonus = reargmax_bonus_skip_think(model, a, row_base + num_accepted, raw_bonus);
        if emit_token_for_verify(a, bonus, None) {
            consumed_rows += 1;
        }
        if a.finished {
            return terminal(a, num_accepted, consumed_rows);
        }
        a.last_token = bonus;
        (num_accepted, consumed_rows, false)
    };

    let metric_k = wide_verify_metric_k(
        model.proposer_arm_kind(model.proposer_arm()),
        origin,
        k_verify,
    );
    crate::metrics::SPEC_DECODE_VERIFY
        .with_label_values(&[
            &metric_k,
            if num_accepted == drafts.len() {
                "accept_all"
            } else {
                "accept_partial"
            },
        ])
        .inc();
    crate::metrics::compat_metrics::record_spec_decode(drafts.len() as u64, num_accepted as u64);

    Ok(Some(PendingBatchedCommit {
        row_base,
        k_verify,
        drafted: drafts.len(),
        thinking_entry: thinking_frame,
        emitted_before,
        num_accepted,
        frame_consumed_rows,
        think_ended_this_step,
    }))
}
