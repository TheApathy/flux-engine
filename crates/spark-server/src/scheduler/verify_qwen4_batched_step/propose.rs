// SPDX-License-Identifier: AGPL-3.0-only

//! Post-verify proposal, batched drafting, PLE prefetch, and grammar draft splicing.

use spark_model::traits::Model;

use super::dflash_budget::{
    apply_dflash_pcum_and_upward_bucket_quantize_cohort, apply_dflash_pcum_truncation,
    apply_dflash_pcum_truncation_for_cohort, dflash_gumbel_cohort_num_drafts,
    quantize_dflash_ragged_drafts_to_graph_bucket,
};
use super::planning::{DraftDepthPolicy, batched_ragged_enabled};
use super::verify_exec::batch_sequences_mut;
use crate::scheduler::verify_dflash_step::cfg_jf_splice_drafts;
use crate::scheduler::*;

/// Next proposal for one committed sequence: the retrieval cascade first,
/// then the native MTP head, followed by zero-GPU-cost history fallback or
/// suffix grafting when remaining slots up to `num_drafts` are open.
pub(super) fn propose_next(
    model: &dyn Model,
    a: &mut ActiveSeq,
    num_drafts: usize,
    depth_policy: DraftDepthPolicy,
) {
    crate::scheduler::mtp_step::history_ring::sync_one_validated_tokens(
        a.session_hash,
        &a.output_tokens,
    );
    let seq_drafts = if model.proposer_is_dflash() {
        num_drafts
    } else {
        depth_policy.resolve(a.depth_controller_key(), num_drafts)
    };
    propose_next_with_seq_drafts(model, a, num_drafts, seq_drafts, depth_policy);
    warm_ple_draft_ahead_for_seq(model, a);
}

fn propose_next_with_seq_drafts(
    model: &dyn Model,
    a: &mut ActiveSeq,
    num_drafts: usize,
    seq_drafts: usize,
    depth_policy: DraftDepthPolicy,
) {
    propose_next_with_seq_drafts_impl(model, a, num_drafts, seq_drafts, depth_policy, false);
}

fn propose_next_with_seq_drafts_impl(
    model: &dyn Model,
    a: &mut ActiveSeq,
    num_drafts: usize,
    seq_drafts: usize,
    depth_policy: DraftDepthPolicy,
    defer_pcum_truncation: bool,
) {
    if dflash_grammar_skip_propose(model, a) {
        let _ = crate::scheduler::mtp_step::reconcile_dflash_before_external_draft(model, a);
        crate::scheduler::stochastic_verify_walk::invalidate_seq_rejection_q(a);
        return;
    }
    let _depth_session =
        crate::scheduler::adaptive_depth::ActiveDepthSessionGuard::new(a.depth_controller_key());
    crate::scheduler::stochastic_verify_walk::publish_seq_gumbel_draft_spec(a);
    let token = a.last_token;
    let grammar_masked = a.grammar_state.is_some();
    let allow_history_graft = depth_policy == DraftDepthPolicy::PerSequence;
    let mut retrieved =
        crate::scheduler::think_budget_nudge::full_forced_draft_chain(a, seq_drafts)
            .map(|chain| (DraftOrigin::Proposer, chain))
            .or_else(|| crate::scheduler::mtp_step::retrieval_chain(a, token, num_drafts))
            .or_else(|| {
                (allow_history_graft && !grammar_masked)
                    .then(|| {
                        crate::scheduler::mtp_step::history_ring::try_explicit_preempt(
                            a.session_hash,
                            &a.seq.tokens,
                            token,
                            a.output_tokens.len(),
                            a.last_verify_accepted,
                            num_drafts,
                        )
                        .map(|chain| (DraftOrigin::Proposer, chain))
                    })
                    .flatten()
            });
    let should_propose = retrieved.is_none() && seq_drafts > 0;
    if !should_propose
        && !crate::scheduler::mtp_step::reconcile_dflash_before_external_draft(model, a)
    {
        retrieved = None;
    }
    let vocab_size = model.vocab_size();
    let (result, ()) = proposal_lifecycle::propose_and_install_with(
        model,
        a,
        token,
        seq_drafts,
        None,
        should_propose,
        |a, proposal| {
            if let Some((origin, chain)) = retrieved {
                notify_qwen4_external_draft(a, &chain);
                crate::scheduler::stochastic_verify_walk::invalidate_seq_rejection_q(a);
                *proposal = Ok(chain);
                a.draft_origin = origin;
            } else if !should_propose {
                notify_qwen4_external_draft(a, &[]);
                crate::scheduler::stochastic_verify_walk::invalidate_seq_rejection_q(a);
            }
            match proposal {
                Ok(drafts) if !drafts.is_empty() => {
                    let before_splice = drafts.clone();
                    cfg_jf_splice_drafts(a, drafts);
                    crate::scheduler::think_budget_nudge::splice_forced_drafts(a, drafts);
                    splice_grammar_drafts(a, drafts, vocab_size);
                    crate::scheduler::stochastic_verify_walk::truncate_seq_rejection_q_after_splice(
                        a,
                        &before_splice,
                        drafts,
                    );
                }
                Ok(_) => {}
                Err(error) => {
                    tracing::error!("run_mtp_propose_multi (batched native MTP): {error:#}")
                }
            }
        },
    );
    if let Err(error) = result {
        tracing::error!("install proposal (batched native MTP): {error:#}");
        return;
    }
    if should_propose && !defer_pcum_truncation {
        apply_dflash_pcum_truncation(a, model.proposer_is_dflash());
    }
    if allow_history_graft {
        crate::scheduler::mtp_step::apply_history_fallback_or_graft(
            model,
            a,
            token,
            num_drafts,
            grammar_masked,
        );
    }
}

pub(in super::super) fn propose_next_batch(
    model: &dyn Model,
    active: &mut [ActiveSeq],
    propose_idxs: &[usize],
    num_drafts: usize,
) {
    let num_drafts = dflash_gumbel_cohort_num_drafts(model, active, num_drafts);
    let ragged = batched_ragged_enabled() && !model.is_ep();
    let has_stochastic_lane = propose_idxs.iter().any(|&idx| {
        active
            .get(idx)
            .is_some_and(|a| !a.finished && a.temperature.is_finite() && a.temperature > 0.0)
    });
    let use_upward_rejection_pcum = model.proposer_is_dflash()
        && ragged
        && propose_idxs.len() >= 2
        && has_stochastic_lane
        && crate::scheduler::stochastic_verify_walk::spec_verify_policy()
            == crate::scheduler::stochastic_verify_walk::SpecVerifyPolicy::Rejection;
    crate::scheduler::mtp_step::history_ring::sync_active_validated_tokens(active);
    let mut mtp_idxs: Vec<(usize, usize)> = Vec::with_capacity(propose_idxs.len());
    let vocab_size = model.vocab_size();
    for &idx in propose_idxs {
        let a = &mut active[idx];
        if dflash_grammar_skip_propose(model, a) {
            let _ = crate::scheduler::mtp_step::reconcile_dflash_before_external_draft(model, a);
            crate::scheduler::stochastic_verify_walk::invalidate_seq_rejection_q(a);
            continue;
        }
        // Every sequence in a multi-sequence batched cohort proposes the uniform
        // cohort width `num_drafts` so the next tick verifies all sequences in
        // one batched target forward instead of splintering into serial forwards.
        let seq_drafts = num_drafts;
        let token = a.last_token;
        let preempt = crate::scheduler::think_budget_nudge::full_forced_draft_chain(a, seq_drafts)
            .map(|chain| (DraftOrigin::Proposer, chain))
            .or_else(|| crate::scheduler::mtp_step::retrieval_chain(a, token, num_drafts));
        if let Some((origin, chain)) = preempt {
            if !crate::scheduler::mtp_step::reconcile_dflash_before_external_draft(model, a) {
                crate::scheduler::stochastic_verify_walk::invalidate_seq_rejection_q(a);
                continue;
            }
            let _depth_session = crate::scheduler::adaptive_depth::ActiveDepthSessionGuard::new(
                a.depth_controller_key(),
            );
            let (result, ()) = proposal_lifecycle::propose_and_install_with(
                model,
                a,
                token,
                num_drafts,
                None,
                false,
                |a, proposal| {
                    notify_qwen4_external_draft(a, &chain);
                    crate::scheduler::stochastic_verify_walk::invalidate_seq_rejection_q(a);
                    *proposal = Ok(chain);
                    a.draft_origin = origin;
                    if let Ok(drafts) = proposal
                        && !drafts.is_empty()
                    {
                        cfg_jf_splice_drafts(a, drafts);
                        crate::scheduler::think_budget_nudge::splice_forced_drafts(a, drafts);
                        splice_grammar_drafts(a, drafts, vocab_size);
                    }
                },
            );
            if let Err(error) = result {
                tracing::error!("install proposal (batched native MTP): {error:#}");
            }
        } else if seq_drafts == 0 || a.finished || a.pending_dflash_sampled_proposal.is_some() {
            propose_next_with_seq_drafts_impl(
                model,
                a,
                num_drafts,
                seq_drafts,
                DraftDepthPolicy::CohortUniform,
                use_upward_rejection_pcum,
            );
        } else {
            mtp_idxs.push((idx, seq_drafts));
        }
    }
    if mtp_idxs.len() <= 1 {
        for (idx, seq_drafts) in mtp_idxs {
            propose_next_with_seq_drafts_impl(
                model,
                &mut active[idx],
                num_drafts,
                seq_drafts,
                DraftDepthPolicy::CohortUniform,
                use_upward_rejection_pcum,
            );
        }
    } else {
        for chunk in mtp_idxs.chunks(64) {
            if chunk.len() == 1 {
                let (idx, seq_drafts) = chunk[0];
                propose_next_with_seq_drafts_impl(
                    model,
                    &mut active[idx],
                    num_drafts,
                    seq_drafts,
                    DraftDepthPolicy::CohortUniform,
                    use_upward_rejection_pcum,
                );
                continue;
            }
            let chunk_idxs: Vec<usize> = chunk.iter().map(|&(idx, _)| idx).collect();
            let chunk_max_drafts = chunk
                .iter()
                .map(|&(_, d)| d)
                .max()
                .unwrap_or(num_drafts)
                .max(1);
            let tokens: Vec<u32> = chunk_idxs
                .iter()
                .map(|&idx| active[idx].last_token)
                .collect();
            let positions: Vec<usize> = chunk_idxs
                .iter()
                .map(|&idx| active[idx].seq.seq_len)
                .collect();
            for &idx in &chunk_idxs {
                let a = &mut active[idx];
                crate::scheduler::stochastic_verify_walk::invalidate_seq_rejection_q(a);
                crate::scheduler::stochastic_verify_walk::publish_seq_gumbel_draft_spec(a);
                a.pending_drafts.clear();
                a.pending_tree_payload = None;
                a.pending_dflash_sampled_proposal = None;
                a.draft_origin = crate::scheduler::DraftOrigin::Proposer;
            }
            let batch_result = {
                let Ok(mut seqs) = batch_sequences_mut(active, &chunk_idxs) else {
                    for &(idx, seq_drafts) in chunk {
                        propose_next_with_seq_drafts_impl(
                            model,
                            &mut active[idx],
                            num_drafts,
                            seq_drafts,
                            DraftDepthPolicy::CohortUniform,
                            use_upward_rejection_pcum,
                        );
                    }
                    continue;
                };
                model.run_mtp_propose_multi_batched(
                    &tokens,
                    &positions,
                    chunk_max_drafts,
                    &mut seqs,
                    0,
                )
            };
            match batch_result {
                Ok(all_drafts) => {
                    for (&(idx, seq_drafts), mut drafts) in chunk.iter().zip(all_drafts) {
                        let a = &mut active[idx];
                        let is_async_placeholder = model.dflash_seq_has_async_placeholder(&a.seq);
                        if drafts.len() > seq_drafts {
                            drafts.truncate(seq_drafts);
                            if !is_async_placeholder {
                                crate::scheduler::stochastic_verify_walk::truncate_seq_rejection_q(
                                    a,
                                    drafts.len(),
                                );
                            }
                        }
                        if !is_async_placeholder && !drafts.is_empty() {
                            let before_splice = drafts.clone();
                            cfg_jf_splice_drafts(a, &mut drafts);
                            crate::scheduler::think_budget_nudge::splice_forced_drafts(
                                a,
                                &mut drafts,
                            );
                            splice_grammar_drafts(a, &mut drafts, vocab_size);
                            crate::scheduler::stochastic_verify_walk::truncate_seq_rejection_q_after_splice(
                                a,
                                &before_splice,
                                &drafts,
                            );
                        }
                        if let Err(error) =
                            proposal_lifecycle::install_collected(model, a, Ok(drafts))
                        {
                            tracing::error!("install proposal (batched native MTP): {error:#}");
                            continue;
                        }
                        if !is_async_placeholder && !use_upward_rejection_pcum {
                            apply_dflash_pcum_truncation_for_cohort(
                                a,
                                model.proposer_is_dflash(),
                                chunk_idxs.len(),
                            );
                        }
                    }
                }
                Err(error) => {
                    tracing::error!(
                        "run_mtp_propose_multi_batched (batched native MTP): {error:#}"
                    );
                    for &idx in &chunk_idxs {
                        let a = &mut active[idx];
                        if let Err(install_err) = proposal_lifecycle::install_collected(
                            model,
                            a,
                            Err(anyhow::anyhow!("{error:#}")),
                        ) {
                            tracing::error!(
                                "install proposal (batched native MTP): {install_err:#}"
                            );
                        }
                    }
                }
            }
        }
    }
    let sync_propose_idxs: Vec<usize> = propose_idxs
        .iter()
        .copied()
        .filter(|&idx| {
            active
                .get(idx)
                .is_some_and(|a| !model.dflash_seq_has_async_placeholder(&a.seq))
        })
        .collect();
    if !sync_propose_idxs.is_empty() {
        if use_upward_rejection_pcum {
            apply_dflash_pcum_and_upward_bucket_quantize_cohort(
                active,
                &sync_propose_idxs,
                model.proposer_is_dflash(),
            );
        } else if model.proposer_is_dflash() && ragged {
            quantize_dflash_ragged_drafts_to_graph_bucket(active, &sync_propose_idxs);
        }
        if sync_propose_idxs.len() < 2 {
            warm_ple_draft_ahead_for_seqs(model, active, &sync_propose_idxs);
        } else {
            warm_ple_verify_prefetch_for_seqs(model, active, &sync_propose_idxs, ragged);
        }
    }
}

/// Apply post-proposal splicing, `p_cum` truncation, and ragged verify-graph
/// bucket quantization to freshly collected asynchronous DFlash drafts at the
/// start of the next scheduler step (deferred from launch time while the
/// placeholder chain was in flight).
pub(in super::super) fn finalize_collected_async_dflash_drafts(
    model: &dyn Model,
    active: &mut [ActiveSeq],
    collected_idxs: &[usize],
) {
    if collected_idxs.is_empty() || !model.proposer_is_dflash() {
        return;
    }
    let vocab_size = model.vocab_size();
    for &idx in collected_idxs {
        let Some(a) = active.get_mut(idx) else {
            continue;
        };
        if a.finished || a.pending_drafts.is_empty() || a.pending_tree_payload.is_some() {
            continue;
        }
        let before_splice = a.pending_drafts.clone();
        let mut drafts = std::mem::take(&mut a.pending_drafts);
        cfg_jf_splice_drafts(a, &mut drafts);
        crate::scheduler::think_budget_nudge::splice_forced_drafts(a, &mut drafts);
        splice_grammar_drafts(a, &mut drafts, vocab_size);
        let after_splice = drafts.clone();
        a.pending_drafts = drafts;
        crate::scheduler::stochastic_verify_walk::truncate_seq_rejection_q_after_splice(
            a,
            &before_splice,
            &after_splice,
        );
    }
    let ragged = batched_ragged_enabled() && !model.is_ep();
    let has_stochastic_lane = collected_idxs.iter().any(|&idx| {
        active
            .get(idx)
            .is_some_and(|a| !a.finished && a.temperature.is_finite() && a.temperature > 0.0)
    });
    let use_upward_rejection_pcum = ragged
        && collected_idxs.len() >= 2
        && has_stochastic_lane
        && crate::scheduler::stochastic_verify_walk::spec_verify_policy()
            == crate::scheduler::stochastic_verify_walk::SpecVerifyPolicy::Rejection;
    if use_upward_rejection_pcum {
        apply_dflash_pcum_and_upward_bucket_quantize_cohort(active, collected_idxs, true);
    } else {
        let cohort_size = collected_idxs.len();
        for &idx in collected_idxs {
            if let Some(a) = active.get_mut(idx) {
                apply_dflash_pcum_truncation_for_cohort(a, true, cohort_size);
            }
        }
        if ragged && cohort_size >= 2 {
            quantize_dflash_ragged_drafts_to_graph_bucket(active, collected_idxs);
        }
    }
}

/// Row cap for one background PLE draft-ahead pre-warm submission (matches
/// `qwen4_ple`'s `PLE_MAX_BATCH_ROWS = QWEN4_CROSS_SEQ_MAX_ROWS = 256`).
const PLE_DRAFT_AHEAD_MAX_ROWS: usize = spark_model::layer::QWEN4_CROSS_SEQ_MAX_ROWS;
/// Each speculating sequence carries at least 2 rows (`last_token + draft[0]`),
/// bounding distinct sequence entries inside `PLE_DRAFT_AHEAD_MAX_ROWS` at
/// `256 / 2 = 128` (cross-sequence verify frames themselves remain capped at
/// `QWEN4_CROSS_SEQ_MAX_SEQS = 64`).
const PLE_DRAFT_AHEAD_MAX_SEQS: usize = PLE_DRAFT_AHEAD_MAX_ROWS / 2;

/// Submit an exact-order, un-deduplicated cross-sequence PLE verify prefetch
/// immediately after `propose_next_batch` finishes so background storage I/O
/// overlaps with tick-end host work, KV allocation, metadata staging, and
/// Layer 0, and `decode_verify_qwen4_batched` adopts the ticket on exact key
/// match without preemption or a second read.
///
/// Preserves exact cohort order and duplicates (matching un-deduplicated
/// DFlash2 `unique_leaders`) and includes `K = 1` anchor-only lanes when
/// `allow_k1` is enabled for ragged cohorts. Uses fixed stack buffers only.
pub(in super::super) fn warm_ple_verify_prefetch_for_seqs(
    model: &dyn Model,
    active: &[ActiveSeq],
    idxs: &[usize],
    allow_k1: bool,
) {
    let mut flat_tokens = [0u32; PLE_DRAFT_AHEAD_MAX_ROWS];
    let mut spans = [(0usize, 0usize, 0usize); PLE_DRAFT_AHEAD_MAX_SEQS];
    let mut token_cursor = 0usize;
    let mut seq_count = 0usize;

    for &idx in idxs {
        if seq_count >= PLE_DRAFT_AHEAD_MAX_SEQS {
            break;
        }
        let Some(a) = active.get(idx) else {
            continue;
        };
        if a.finished
            || a.pending_tree_payload.is_some()
            || a.pending_dflash_sampled_proposal.is_some()
            || (!allow_k1 && a.pending_drafts.is_empty())
        {
            continue;
        }
        let window_len = 1 + a.pending_drafts.len();
        if window_len > spark_model::layer::QWEN4_CROSS_SEQ_MAX_K {
            continue;
        }
        if token_cursor + window_len > PLE_DRAFT_AHEAD_MAX_ROWS {
            break;
        }
        let start = token_cursor;
        flat_tokens[start] = a.last_token;
        flat_tokens[start + 1..start + window_len].copy_from_slice(&a.pending_drafts);
        token_cursor += window_len;
        spans[seq_count] = (start, token_cursor, idx);
        seq_count += 1;
    }

    if seq_count == 0 {
        return;
    }
    let mut entries: [(&[u32], &[u32]); PLE_DRAFT_AHEAD_MAX_SEQS] =
        [(&[], &[]); PLE_DRAFT_AHEAD_MAX_SEQS];
    for (entry, &(start, end, idx)) in entries.iter_mut().zip(&spans[..seq_count]) {
        *entry = (&flat_tokens[start..end], active[idx].seq.tokens.as_slice());
    }
    model.warm_ple_draft_ahead_seqs(&entries[..seq_count]);
}

/// Pre-warm `ParkedPleCache` on the background PLE worker for every live
/// sequence's upcoming verify window `[a.last_token, ...a.pending_drafts]`
/// paired with its exact prior token history `a.seq.tokens`.
///
/// Pairing each window with `a.seq.tokens` matches
/// `PleWarmSeq::from_slices(&tokens_per_seq[s], &seqs[s].tokens)` in
/// `decode_verify_qwen4_batched`, warming all 8 bigram + 8 trigram PLE heads
/// accurately before the next verify step's Layer 0 begins. Uses fixed stack
/// buffers (zero heap allocations) and deduplicates identical `(prior_tail,
/// window)` tuples so every slot in the `PLE_DRAFT_AHEAD_MAX_ROWS`-row (256-row)
/// budget covers distinct rows.
pub(in super::super) fn warm_ple_draft_ahead_for_seqs(
    model: &dyn Model,
    active: &[ActiveSeq],
    idxs: &[usize],
) {
    let mut flat_tokens = [0u32; PLE_DRAFT_AHEAD_MAX_ROWS];
    let mut spans = [(0usize, 0usize, 0usize); PLE_DRAFT_AHEAD_MAX_SEQS];
    let mut token_cursor = 0usize;
    let mut seq_count = 0usize;

    for &idx in idxs {
        if seq_count >= PLE_DRAFT_AHEAD_MAX_SEQS {
            break;
        }
        let Some(a) = active.get(idx) else {
            continue;
        };
        if a.finished || a.pending_drafts.is_empty() {
            continue;
        }
        let window_len = 1 + a.pending_drafts.len();
        if token_cursor + window_len > PLE_DRAFT_AHEAD_MAX_ROWS {
            if seq_count == 0 {
                let clamped = PLE_DRAFT_AHEAD_MAX_ROWS;
                flat_tokens[0] = a.last_token;
                flat_tokens[1..clamped].copy_from_slice(&a.pending_drafts[..clamped - 1]);
                spans[0] = (0, clamped, idx);
                seq_count = 1;
            }
            break;
        }
        let prior_tail = &a.seq.tokens[a.seq.tokens.len().saturating_sub(2)..];
        let is_dup = spans[..seq_count].iter().any(|&(start, end, prev_idx)| {
            let prev = &active[prev_idx];
            let prev_tail = &prev.seq.tokens[prev.seq.tokens.len().saturating_sub(2)..];
            prev_tail == prior_tail
                && (end - start) == window_len
                && flat_tokens[start] == a.last_token
                && flat_tokens[start + 1..end] == a.pending_drafts[..]
        });
        if is_dup {
            continue;
        }
        let start = token_cursor;
        flat_tokens[start] = a.last_token;
        flat_tokens[start + 1..start + window_len].copy_from_slice(&a.pending_drafts);
        token_cursor += window_len;
        spans[seq_count] = (start, token_cursor, idx);
        seq_count += 1;
    }

    if seq_count == 0 {
        return;
    }
    let mut entries: [(&[u32], &[u32]); PLE_DRAFT_AHEAD_MAX_SEQS] =
        [(&[], &[]); PLE_DRAFT_AHEAD_MAX_SEQS];
    for (entry, &(start, end, idx)) in entries.iter_mut().zip(&spans[..seq_count]) {
        *entry = (&flat_tokens[start..end], active[idx].seq.tokens.as_slice());
    }
    model.warm_ple_draft_ahead_seqs(&entries[..seq_count]);
}

/// Single-sequence variant of [`warm_ple_draft_ahead_for_seqs`] for `C=1`
/// verify steps (`verify_dflash_step`) and singleton post-verify proposals.
pub(in super::super) fn warm_ple_draft_ahead_for_seq(model: &dyn Model, a: &ActiveSeq) {
    if a.finished || a.pending_drafts.is_empty() {
        return;
    }
    let mut window = [0u32; PLE_DRAFT_AHEAD_MAX_ROWS];
    let len = (1 + a.pending_drafts.len()).min(PLE_DRAFT_AHEAD_MAX_ROWS);
    window[0] = a.last_token;
    window[1..len].copy_from_slice(&a.pending_drafts[..len - 1]);
    model.warm_ple_draft_ahead_seqs(&[(&window[..len], a.seq.tokens.as_slice())]);
}

fn first_allowed_token(bitmask: &[i32], vocab_size: usize) -> Option<u32> {
    for (word_idx, &word) in bitmask.iter().enumerate() {
        let mut bits = word as u32;
        while bits != 0 {
            let bit = bits.trailing_zeros();
            bits &= bits - 1;
            let token_id = (word_idx as u32) * 32 + bit;
            if (token_id as usize) < vocab_size {
                return Some(token_id);
            }
        }
    }
    None
}

/// Splice grammar-forced tokens (`forced_token` coalescence) and repair an
/// illegal `drafts[0]` in-place without changing `drafts.len()` or advancing
/// `a.grammar_state` permanently.
///
/// Lossless: `finalize_commit -> run_dflash_content_accept_at_row` still
/// applies `a.grammar_state` row-by-row against target logits, while keeping
/// `drafts.len()` uniform across mixed grammar + plain cohorts and ensuring
/// `drafts[0]` is grammar-legal if the cohort narrows to `C = 1` next tick.
fn splice_grammar_drafts(a: &mut ActiveSeq, drafts: &mut [u32], vocab_size: usize) {
    if a.inside_thinking || drafts.is_empty() {
        return;
    }
    let Some(ref mut gs) = a.grammar_state else {
        return;
    };
    if gs.is_terminated() {
        return;
    }
    let history_before = gs.num_history_steps();
    let mut spliced = false;
    for (i, slot) in drafts.iter_mut().enumerate() {
        if gs.is_terminated() {
            break;
        }
        if let Some(forced) = gs.forced_token()
            && forced >= 0
        {
            let forced_u32 = forced as u32;
            if *slot != forced_u32 {
                *slot = forced_u32;
                spliced = true;
            }
        } else if gs.fill_bitmask() && !gs.is_token_allowed(*slot) {
            if i == 0
                && let Some(legal) = first_allowed_token(gs.bitmask_data(), vocab_size)
            {
                *slot = legal;
                spliced = true;
            } else {
                break;
            }
        }
        if !gs.accept_token(*slot) {
            break;
        }
    }
    let recorded = gs.num_history_steps() - history_before;
    if recorded > 0 {
        gs.rollback(recorded);
    }
    if spliced
        && let Some(prop_state) = a.seq.proposer_state.as_mut()
        && let Some(mtp_state) = prop_state
            .as_any_mut()
            .downcast_mut::<spark_model::layers::qwen4_mtp::Qwen4MtpState>()
    {
        mtp_state.sync_spliced_drafts(drafts);
    }
}

fn notify_qwen4_external_draft(a: &mut ActiveSeq, chain: &[u32]) {
    a.seq.retire_committed_verify_handoff();
    if let Some(prop_state) = a.seq.proposer_state.as_mut()
        && let Some(mtp_state) = prop_state
            .as_any_mut()
            .downcast_mut::<spark_model::layers::qwen4_mtp::Qwen4MtpState>()
    {
        mtp_state.notify_external_draft_with_tokens(chain);
    }
}
