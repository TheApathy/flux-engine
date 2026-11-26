// SPDX-License-Identifier: AGPL-3.0-only

//! Batched Gumbel-Max and maximal-coupling rejection staging/presampling for verify frames.

use anyhow::Context;
use spark_model::traits::Model;

use crate::scheduler::{ActiveSeq, ThinkSpecCtx};

/// Stage per-sequence unified tail specs (`temp > 0` Gumbel-Max, `temp == 0`
/// 0-ULP greedy argmax, `temp == -1` 0-cycle host-policy skip) before
/// `decode_verify_qwen4_batched` so mixed cohorts (e.g. 15 Gumbel streams +
/// 1 `budget_nudge` / `force_end_thinking` / penalty / greedy stream) run in
/// ONE unified suffix CUDA graph without disabling in-graph sampling for the
/// rest of the batch.
///
/// Returns `Some(seq_in_graph_gumbel)` recording which sequences were staged
/// with `temp > 0` Gumbel parameters, or `None` when all sequences are greedy
/// (`verify_qwen4_batched` auto-stages all `temp == 0.0` with zero heap alloc)
/// or staging does not apply.
pub(super) fn stage_batched_gumbel_frame_if_eligible(
    model: &dyn Model,
    active: &[ActiveSeq],
    idxs: &[usize],
    k_per_seq: &[usize],
    row_bases: &[usize],
    total_rows: usize,
    think: &ThinkSpecCtx<'_>,
) -> Option<Vec<bool>> {
    stage_batched_gumbel_frame_if_eligible_with_policy(
        model,
        active,
        idxs,
        k_per_seq,
        row_bases,
        total_rows,
        think,
        crate::scheduler::stochastic_verify_walk::spec_verify_policy(),
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn stage_batched_gumbel_frame_if_eligible_with_policy(
    model: &dyn Model,
    active: &[ActiveSeq],
    idxs: &[usize],
    k_per_seq: &[usize],
    row_bases: &[usize],
    total_rows: usize,
    think: &ThinkSpecCtx<'_>,
    policy: crate::scheduler::stochastic_verify_walk::SpecVerifyPolicy,
) -> Option<Vec<bool>> {
    use crate::scheduler::spec_policy_accept::{
        VerifyTailRowMode, classify_verify_tail_row_mode, push_verify_tail_row_specs,
    };
    if total_rows == 0
        || total_rows > spark_model::layer::QWEN4_CROSS_SEQ_MAX_ROWS
        || idxs.len() > spark_model::layer::QWEN4_CROSS_SEQ_MAX_ROWS
        || idxs.len() != k_per_seq.len()
        || idxs.len() != row_bases.len()
    {
        return None;
    }
    let logits_base = model.logits_buffer_ptr();
    if model.decode_logits_fp32() || model.logits_ptr_is_fp32(logits_base) {
        return None;
    }
    let mut modes = [VerifyTailRowMode::GreedyArgmax; spark_model::layer::QWEN4_CROSS_SEQ_MAX_ROWS];
    let mut any_non_greedy = false;
    for (s, &idx) in idxs.iter().enumerate() {
        let mode = classify_verify_tail_row_mode(&active[idx], k_per_seq[s], think, policy);
        modes[s] = mode;
        if mode != VerifyTailRowMode::GreedyArgmax {
            any_non_greedy = true;
        }
    }
    if !any_non_greedy {
        return None;
    }
    let mut reqs = Vec::with_capacity(total_rows);
    let mut seq_in_graph_gumbel = Vec::with_capacity(idxs.len());
    for (s, &idx) in idxs.iter().enumerate() {
        let mode = modes[s];
        seq_in_graph_gumbel.push(mode == VerifyTailRowMode::Gumbel);
        push_verify_tail_row_specs(&mut reqs, &active[idx], row_bases[s], k_per_seq[s], mode);
    }
    if reqs.len() == total_rows {
        model.stage_next_verify_gumbel_rows(&reqs);
        Some(seq_in_graph_gumbel)
    } else {
        None
    }
}

/// Frame-level pre-sampled GPU target Gumbel-Max tokens for eligible sequences
/// in a batched verify frame.
pub(super) struct PresampledGumbelFrame {
    pub(super) tokens: Vec<u32>,
    pub(super) seq_offsets: Vec<Option<usize>>,
}

/// Consume per-sequence in-graph GPU Gumbel-Max tokens from the unified verify
/// tail (zero extra launches or syncs, even in mixed cohorts where some
/// sequences ran `temp == 0` greedy or `temp == -1` host-policy skip), or fall
/// back to ONE batched `Model::sample_gumbel_verify_rows_batched` call over the
/// eligible sequences when dedup or an unsupported backend bypassed in-graph
/// sampling.
#[allow(clippy::too_many_arguments)]
pub(super) fn presample_batched_gumbel_frame_ragged(
    model: &dyn Model,
    active: &[ActiveSeq],
    idxs: &[usize],
    k_per_seq: &[usize],
    row_bases: &[usize],
    think: &ThinkSpecCtx<'_>,
    seq_in_graph_gumbel: Option<&[bool]>,
) -> anyhow::Result<Option<PresampledGumbelFrame>> {
    presample_batched_gumbel_frame_ragged_with_policy(
        model,
        active,
        idxs,
        k_per_seq,
        row_bases,
        think,
        seq_in_graph_gumbel,
        crate::scheduler::stochastic_verify_walk::spec_verify_policy(),
    )
}

#[cfg(test)]
pub(super) fn presample_batched_gumbel_frame_with_policy(
    model: &dyn Model,
    active: &[ActiveSeq],
    idxs: &[usize],
    k_verify: usize,
    think: &ThinkSpecCtx<'_>,
    policy: crate::scheduler::stochastic_verify_walk::SpecVerifyPolicy,
) -> anyhow::Result<Option<PresampledGumbelFrame>> {
    let k_per_seq = vec![k_verify; idxs.len()];
    let row_bases: Vec<usize> = (0..idxs.len()).map(|s| s * k_verify).collect();
    presample_batched_gumbel_frame_ragged_with_policy(
        model, active, idxs, &k_per_seq, &row_bases, think, None, policy,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn presample_batched_gumbel_frame_ragged_with_policy(
    model: &dyn Model,
    active: &[ActiveSeq],
    idxs: &[usize],
    k_per_seq: &[usize],
    row_bases: &[usize],
    think: &ThinkSpecCtx<'_>,
    seq_in_graph_gumbel: Option<&[bool]>,
    policy: crate::scheduler::stochastic_verify_walk::SpecVerifyPolicy,
) -> anyhow::Result<Option<PresampledGumbelFrame>> {
    if policy != crate::scheduler::stochastic_verify_walk::SpecVerifyPolicy::Gumbel {
        return Ok(None);
    }
    let logits_base = model.logits_buffer_ptr();
    if model.decode_logits_fp32() || model.logits_ptr_is_fp32(logits_base) {
        return Ok(None);
    }
    // Always drain any in-graph buffer published by the immediately preceding
    // verify forward so tokens never leak across steps.
    let in_graph_tokens = model.take_in_graph_gumbel_verify_tokens();
    if idxs.len() != k_per_seq.len() || idxs.len() != row_bases.len() {
        return Ok(None);
    }
    let total_rows: usize = k_per_seq.iter().sum();
    // Defer heap allocations until the first eligible sequence so greedy /
    // penalty-ineligible frames pay zero allocation overhead.
    let mut seq_eligible: Vec<bool> = Vec::new();
    let mut eligible_seqs = 0usize;
    let mut eligible_rows = 0usize;
    for (s, &idx) in idxs.iter().enumerate() {
        let k_s = k_per_seq[s];
        if crate::scheduler::spec_policy_accept::batched_gumbel_seq_eligible(
            &active[idx],
            k_s,
            think,
        ) {
            if seq_eligible.is_empty() {
                seq_eligible.resize(idxs.len(), false);
            }
            seq_eligible[s] = true;
            eligible_seqs += 1;
            eligible_rows += k_s;
        }
    }
    if eligible_seqs == 0 {
        return Ok(None);
    }
    let vocab = model.vocab_size();
    if let Some(tokens) = in_graph_tokens
        && tokens.len() == total_rows
    {
        let mut seq_offsets = vec![None; idxs.len()];
        let mut all_eligible_satisfied = true;
        for (s, &eligible) in seq_eligible.iter().enumerate() {
            if !eligible {
                continue;
            }
            let staged_ok =
                seq_in_graph_gumbel.is_none_or(|mask| mask.get(s).copied().unwrap_or(false));
            let base = row_bases[s];
            let k_s = k_per_seq[s];
            let slice_ok = tokens
                .get(base..base + k_s)
                .is_some_and(|slice| slice.iter().all(|&tok| (tok as usize) < vocab));
            if staged_ok && slice_ok {
                seq_offsets[s] = Some(base);
            } else {
                all_eligible_satisfied = false;
                break;
            }
        }
        if all_eligible_satisfied {
            crate::scheduler::spec_policy_accept::log_batched_gumbel_engaged_once(
                eligible_rows,
                eligible_seqs,
            );
            return Ok(Some(PresampledGumbelFrame {
                tokens,
                seq_offsets,
            }));
        }
    }
    // Fallback (dedup or backend without in-graph Gumbel): launch one batched
    // GPU Gumbel-Max pass over ONLY the eligible sequences.
    let mut reqs: Vec<spark_runtime::sampler::device_topk::GumbelVerifyRowSpec> =
        Vec::with_capacity(eligible_rows);
    let mut seq_offsets: Vec<Option<usize>> = vec![None; idxs.len()];
    for (s, &idx) in idxs.iter().enumerate() {
        if !seq_eligible[s] {
            continue;
        }
        let a = &active[idx];
        let k_s = k_per_seq[s];
        let row_base = row_bases[s];
        seq_offsets[s] = Some(reqs.len());
        let seed = crate::scheduler::stochastic_verify_walk::resolve_seq_gumbel_seed(a);
        let base_pos = crate::scheduler::stochastic_verify_walk::seq_abs_pos(a);
        for r in 0..k_s {
            reqs.push(spark_runtime::sampler::device_topk::GumbelVerifyRowSpec {
                row_idx: row_base + r,
                seed,
                abs_pos: base_pos.saturating_add(r as u64),
                temperature: a.temperature,
                top_k: a.top_k,
                top_p: a.top_p,
            });
        }
    }
    let Some(tokens) = model
        .sample_gumbel_verify_rows_batched(logits_base, vocab, &reqs)
        .context("sample_gumbel_verify_rows_batched")?
    else {
        return Ok(None);
    };
    anyhow::ensure!(
        tokens.len() == reqs.len(),
        "sample_gumbel_verify_rows_batched returned {} tokens for {} rows",
        tokens.len(),
        reqs.len()
    );
    crate::scheduler::spec_policy_accept::log_batched_gumbel_engaged_once(
        reqs.len(),
        eligible_seqs,
    );
    Ok(Some(PresampledGumbelFrame {
        tokens,
        seq_offsets,
    }))
}

/// Stage per-sequence unified GPU maximal-coupling rejection verify row specs
/// (`temp > 0` maximal coupling / bonus, `temp == 0` greedy argmax,
/// `temp == -1` 0-cycle host-policy skip) before `decode_verify_qwen4_batched`.
///
/// Always clears stale staged rows with `&[]` when outside
/// `SpecVerifyPolicy::Rejection`, when the rejection kernel is unavailable, or
/// when every sequence in the frame is greedy.
#[allow(clippy::too_many_arguments)]
pub(super) fn stage_batched_rejection_frame_if_eligible(
    model: &dyn Model,
    active: &[ActiveSeq],
    idxs: &[usize],
    drafts_per_seq: &[Vec<u32>],
    k_per_seq: &[usize],
    row_bases: &[usize],
    total_rows: usize,
    think: &ThinkSpecCtx<'_>,
) -> Option<Vec<bool>> {
    stage_batched_rejection_frame_if_eligible_with_policy(
        model,
        active,
        idxs,
        drafts_per_seq,
        k_per_seq,
        row_bases,
        total_rows,
        think,
        crate::scheduler::stochastic_verify_walk::spec_verify_policy(),
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn stage_batched_rejection_frame_if_eligible_with_policy(
    model: &dyn Model,
    active: &[ActiveSeq],
    idxs: &[usize],
    drafts_per_seq: &[Vec<u32>],
    k_per_seq: &[usize],
    row_bases: &[usize],
    total_rows: usize,
    think: &ThinkSpecCtx<'_>,
    policy: crate::scheduler::stochastic_verify_walk::SpecVerifyPolicy,
) -> Option<Vec<bool>> {
    use crate::scheduler::spec_policy_accept::{
        RejectionTailRowMode, classify_rejection_tail_row_mode, push_rejection_tail_row_specs,
    };
    if policy != crate::scheduler::stochastic_verify_walk::SpecVerifyPolicy::Rejection
        || !model.gpu_rejection_verify_available()
        || total_rows == 0
        || total_rows > spark_model::REJECTION_MAX_ROWS
        || idxs.len() > spark_model::REJECTION_MAX_ROWS
        || idxs.len() != drafts_per_seq.len()
        || idxs.len() != k_per_seq.len()
        || idxs.len() != row_bases.len()
    {
        model.stage_next_verify_rejection_rows(&[]);
        return None;
    }
    let logits_base = model.logits_buffer_ptr();
    if model.decode_logits_fp32() || model.logits_ptr_is_fp32(logits_base) {
        model.stage_next_verify_rejection_rows(&[]);
        return None;
    }
    let mut modes = [RejectionTailRowMode::GreedyArgmax; spark_model::REJECTION_MAX_ROWS];
    let mut any_rejection = false;
    for (s, &idx) in idxs.iter().enumerate() {
        let mode = classify_rejection_tail_row_mode(&active[idx], k_per_seq[s], think, policy);
        modes[s] = mode;
        if mode == RejectionTailRowMode::Rejection {
            any_rejection = true;
        }
    }
    if !any_rejection {
        model.stage_next_verify_rejection_rows(&[]);
        return None;
    }
    let mut reqs = Vec::with_capacity(total_rows);
    let mut seq_in_graph_rejection = Vec::with_capacity(idxs.len());
    for (s, &idx) in idxs.iter().enumerate() {
        let mode = modes[s];
        seq_in_graph_rejection.push(mode == RejectionTailRowMode::Rejection);
        push_rejection_tail_row_specs(
            &mut reqs,
            &active[idx],
            &drafts_per_seq[s],
            row_bases[s],
            k_per_seq[s],
            mode,
        );
    }
    if reqs.len() == total_rows {
        model.stage_next_verify_rejection_rows(&reqs);
        Some(seq_in_graph_rejection)
    } else {
        model.stage_next_verify_rejection_rows(&[]);
        None
    }
}

/// Frame-level pre-verified GPU maximal-coupling rejection words for eligible
/// sequences in a batched verify frame.
pub(super) struct PresampledRejectionFrame {
    pub(super) words: Vec<spark_model::PackedRejectionWord>,
    pub(super) seq_offsets: Vec<Option<usize>>,
}

/// Consume per-sequence in-graph [`spark_model::PackedRejectionWord`] words
/// from the unified verify suffix tail (zero extra launches or syncs), or fall
/// back to ONE batched [`Model::sample_rejection_verify_rows_batched`] launch
/// over ONLY the eligible sequences when eager verify bypassed in-graph capture.
#[allow(clippy::too_many_arguments)]
pub(super) fn presample_batched_rejection_frame_ragged(
    model: &dyn Model,
    active: &[ActiveSeq],
    idxs: &[usize],
    drafts_per_seq: &[Vec<u32>],
    k_per_seq: &[usize],
    row_bases: &[usize],
    think: &ThinkSpecCtx<'_>,
    seq_in_graph_rejection: Option<&[bool]>,
) -> anyhow::Result<Option<PresampledRejectionFrame>> {
    presample_batched_rejection_frame_ragged_with_policy(
        model,
        active,
        idxs,
        drafts_per_seq,
        k_per_seq,
        row_bases,
        think,
        seq_in_graph_rejection,
        crate::scheduler::stochastic_verify_walk::spec_verify_policy(),
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn presample_batched_rejection_frame_ragged_with_policy(
    model: &dyn Model,
    active: &[ActiveSeq],
    idxs: &[usize],
    drafts_per_seq: &[Vec<u32>],
    k_per_seq: &[usize],
    row_bases: &[usize],
    think: &ThinkSpecCtx<'_>,
    seq_in_graph_rejection: Option<&[bool]>,
    policy: crate::scheduler::stochastic_verify_walk::SpecVerifyPolicy,
) -> anyhow::Result<Option<PresampledRejectionFrame>> {
    // Always drain any in-graph buffer published by the immediately preceding
    // verify forward so packed words can never leak across steps.
    let in_graph_words = model.take_in_graph_rejection_verify_words();
    if policy != crate::scheduler::stochastic_verify_walk::SpecVerifyPolicy::Rejection
        || !model.gpu_rejection_verify_available()
    {
        return Ok(None);
    }
    let logits_base = model.logits_buffer_ptr();
    if model.decode_logits_fp32() || model.logits_ptr_is_fp32(logits_base) {
        return Ok(None);
    }
    if idxs.len() != drafts_per_seq.len()
        || idxs.len() != k_per_seq.len()
        || idxs.len() != row_bases.len()
    {
        return Ok(None);
    }
    let total_rows: usize = k_per_seq.iter().sum();
    let mut seq_eligible: Vec<bool> = Vec::new();
    let mut eligible_seqs = 0usize;
    let mut eligible_rows = 0usize;
    for (s, &idx) in idxs.iter().enumerate() {
        let k_s = k_per_seq[s];
        if crate::scheduler::spec_policy_accept::batched_rejection_seq_eligible(
            &active[idx],
            k_s,
            think,
        ) {
            if seq_eligible.is_empty() {
                seq_eligible.resize(idxs.len(), false);
            }
            seq_eligible[s] = true;
            eligible_seqs += 1;
            eligible_rows += k_s;
        }
    }
    if eligible_seqs == 0 {
        return Ok(None);
    }
    let vocab = model.vocab_size();
    if let Some(words) = in_graph_words
        && words.len() == total_rows
    {
        let mut seq_offsets = vec![None; idxs.len()];
        let mut all_eligible_satisfied = true;
        for (s, &eligible) in seq_eligible.iter().enumerate() {
            if !eligible {
                continue;
            }
            let staged_ok =
                seq_in_graph_rejection.is_none_or(|mask| mask.get(s).copied().unwrap_or(false));
            let base = row_bases[s];
            let k_s = k_per_seq[s];
            let slice_ok = words.get(base..base + k_s).is_some_and(|slice| {
                slice
                    .iter()
                    .all(|w| w.is_host_fallback() || (w.token_id() as usize) < vocab)
            });
            if staged_ok && slice_ok {
                seq_offsets[s] = Some(base);
            } else {
                all_eligible_satisfied = false;
                break;
            }
        }
        if all_eligible_satisfied {
            crate::scheduler::spec_policy_accept::log_batched_rejection_engaged_once(
                eligible_rows,
                eligible_seqs,
            );
            return Ok(Some(PresampledRejectionFrame { words, seq_offsets }));
        }
    }
    // Eager fallback: launch ONE batched GPU rejection pass over ONLY the eligible sequences.
    let mut reqs: Vec<spark_model::RejectionVerifyRowSpec> = Vec::with_capacity(eligible_rows);
    let mut seq_offsets: Vec<Option<usize>> = vec![None; idxs.len()];
    for (s, &idx) in idxs.iter().enumerate() {
        if !seq_eligible[s] {
            continue;
        }
        seq_offsets[s] = Some(reqs.len());
        crate::scheduler::spec_policy_accept::push_rejection_tail_row_specs(
            &mut reqs,
            &active[idx],
            &drafts_per_seq[s],
            row_bases[s],
            k_per_seq[s],
            crate::scheduler::spec_policy_accept::RejectionTailRowMode::Rejection,
        );
    }
    let Some(words) = model
        .sample_rejection_verify_rows_batched(logits_base, vocab, &reqs)
        .context("sample_rejection_verify_rows_batched")?
    else {
        return Ok(None);
    };
    anyhow::ensure!(
        words.len() == reqs.len(),
        "sample_rejection_verify_rows_batched returned {} words for {} rows",
        words.len(),
        reqs.len()
    );
    for (idx, word) in words.iter().enumerate() {
        anyhow::ensure!(
            word.is_host_fallback() || (word.token_id() as usize) < vocab,
            "sample_rejection_verify_rows_batched returned out-of-vocab token {} (vocab={vocab}) at row {idx}",
            word.token_id()
        );
    }
    crate::scheduler::spec_policy_accept::log_batched_rejection_engaged_once(
        reqs.len(),
        eligible_seqs,
    );
    Ok(Some(PresampledRejectionFrame { words, seq_offsets }))
}
