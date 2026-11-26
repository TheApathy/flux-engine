// SPDX-License-Identifier: AGPL-3.0-only

//! Cohort eligibility, width harmonization, max-seq-len clamping, and frame planning for batched verify.

use spark_model::traits::Model;

use super::depth_controller::{adaptive_selected, narrow_pending_drafts};
use super::dflash_budget::{
    dflash_gumbel_cohort_num_drafts, quantize_dflash_ragged_drafts_to_graph_bucket,
};
use crate::scheduler::{ActiveSeq, ThinkSpecCtx};

/// Per-sequence facts that decide batched eligibility.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in super::super) struct BatchedK5Candidate {
    pub draft_len: usize,
    pub configured_num_drafts: usize,
    pub grammar_active: bool,
    pub finished: bool,
    pub policy_required: bool,
    pub has_tree: bool,
    pub has_sampled: bool,
}

/// Verify width K of an eligible frame, or `None`. Flat chains (including
/// strict/non-strict grammar sequences and think/content policy walks whose
/// per-row logits live in the batched frame buffer) batch by verify width;
/// trees, sampled-q frames, finished sequences, and `policy_required`
/// exclusions stay on the per-sequence verifier.
///
/// When `allow_k1` is true (ragged mixed-`K` mode), `draft_len == 0` maps to
/// `Some(1)` (`K_s = 1`, anchor-only) and widths `1..=8` are admitted.
pub(in super::super) fn batched_k5_width_for(
    c: BatchedK5Candidate,
    allow_k1: bool,
) -> Option<usize> {
    if c.finished || c.policy_required || c.has_tree || c.has_sampled {
        return None;
    }
    if c.draft_len > c.configured_num_drafts {
        return None;
    }
    if c.draft_len == 0 {
        return allow_k1.then_some(1);
    }
    let k = c.draft_len + 1;
    let max_k = if allow_k1 {
        spark_model::layer::Qwen4CrossSeqFrame::RAGGED_PLAN_MAX_K
    } else {
        spark_model::layer::QWEN4_CROSS_SEQ_MAX_K
    };
    (spark_model::layer::QWEN4_CROSS_SEQ_MIN_K..=max_k)
        .contains(&k)
        .then_some(k)
}

pub(in super::super) fn batched_k5_width(c: BatchedK5Candidate) -> Option<usize> {
    batched_k5_width_for(c, false)
}

/// True when true ragged mixed-`K` (`K_s in 1..=8`) verify frames and
/// active-subset compacted MTP drafting are enabled (default ON under
/// `ATLAS_QWEN4_BATCHED_ADAPTIVE_K=1`; set `ATLAS_QWEN4_BATCHED_RAGGED_MTP=0`
/// to restore uniform `CohortAcceptEwma` widths for A/B testing).
#[inline]
pub(super) fn batched_ragged_enabled() -> bool {
    adaptive_selected() && spark_model::layers::qwen4_mtp::batched_ragged_mtp_enabled()
}

/// Frames to run and the sequences left for the per-sequence verifier.
#[derive(Debug, Default, PartialEq, Eq)]
pub(in super::super) struct BatchedK5Plan {
    pub frames: Vec<Vec<usize>>,
    pub leftover: Vec<usize>,
}

/// Group `(index, width)` candidates by width (ascending, stable within a
/// width) and split each group into balanced frames bounded by `max_frame_rows`.
/// `None` widths and any width with a single sequence are left over.
pub(in super::super) fn plan_batched_k5_with_max_rows(
    candidates: &[(usize, Option<usize>)],
    max_frame_rows: usize,
) -> BatchedK5Plan {
    let mut by_width: std::collections::BTreeMap<usize, Vec<usize>> = Default::default();
    let mut leftover = Vec::new();
    for &(idx, width) in candidates {
        match width {
            Some(k) => by_width.entry(k).or_default().push(idx),
            None => leftover.push(idx),
        }
    }
    let mut frames = Vec::new();
    for (k, idxs) in by_width {
        let sizes = spark_model::layer::Qwen4CrossSeqFrame::plan_frames_with_max_rows(
            idxs.len(),
            k,
            max_frame_rows,
        );
        let mut cursor = 0usize;
        for size in sizes {
            frames.push(idxs[cursor..cursor + size].to_vec());
            cursor += size;
        }
        leftover.extend_from_slice(&idxs[cursor..]);
    }
    leftover.sort_unstable();
    BatchedK5Plan { frames, leftover }
}

/// Group `(index, width)` candidates by width (ascending, stable within a
/// width) and split each group into balanced frames using the active GPU's
/// adaptive frame row cap (`128` on GB10, `256` on `> 64`-SM GPUs).
pub(in super::super) fn plan_batched_k5(candidates: &[(usize, Option<usize>)]) -> BatchedK5Plan {
    plan_batched_k5_with_max_rows(
        candidates,
        spark_model::layers::qwen4_verify_lanes::adaptive_max_frame_rows(),
    )
}

/// Parse `ATLAS_GRAMMAR_SERIAL_VERIFY`: `"1"` (trimmed) restores per-sequence
/// verification for grammar-active sequences during A/B comparisons.
pub(in super::super) fn parse_grammar_serial_verify(raw: Option<&str>) -> bool {
    raw.map(str::trim) == Some("1")
}

/// Opt-in escape hatch (`ATLAS_GRAMMAR_SERIAL_VERIFY=1`) restoring per-sequence
/// verification for grammar-active sequences during A/B comparisons.
pub(super) fn grammar_serial_verify() -> bool {
    static SERIAL: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *SERIAL.get_or_init(|| {
        parse_grammar_serial_verify(std::env::var("ATLAS_GRAMMAR_SERIAL_VERIFY").ok().as_deref())
    })
}

/// Parse `ATLAS_QWEN4_BATCHED_K5_LOG`: exact `"1"` enables batched-K5 step logs.
pub(in super::super) fn parse_batched_k5_log_enabled(raw: Option<&str>) -> bool {
    raw == Some("1")
}

/// Cached `ATLAS_QWEN4_BATCHED_K5_LOG=1` observability gate (read once instead
/// of twice per verify step on the scheduler hot path).
#[inline]
pub(super) fn batched_k5_log_enabled() -> bool {
    static LOG_ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *LOG_ENABLED.get_or_init(|| {
        parse_batched_k5_log_enabled(std::env::var("ATLAS_QWEN4_BATCHED_K5_LOG").ok().as_deref())
    })
}

/// Whether a sequence cannot use the batched verifier.
///
/// `decode_verify_qwen4_batched` writes all `rows = seqs * k_verify` BF16
/// logits rows into `buffers.logits()` at row offset `s * k_verify`, so
/// `finalize_commit` runs both `run_dflash_thinking_accept_at_row` (when
/// `think.enabled && a.inside_thinking`) and `run_dflash_content_accept_at_row`
/// (penalties, min_tokens, temperature > 0, Gumbel, think masks, and per-row
/// `grammar_state` bitmask + advance) directly on the sequence's slice of the
/// frame's logits. Only thinking without `think.enabled` (or an explicit
/// `ATLAS_GRAMMAR_SERIAL_VERIFY=1` override) stays on the per-sequence path.
pub(in super::super) fn batched_k5_policy_required(
    a: &ActiveSeq,
    think: &ThinkSpecCtx<'_>,
) -> bool {
    (a.grammar_state.is_some() && grammar_serial_verify()) || (a.inside_thinking && !think.enabled)
}

/// [`plan_batched_k5_with_max_rows`] restricted to one verify width: the width
/// held by the most candidates (ties go to the wider K).
pub(in super::super) fn plan_batched_uniform_with_max_rows(
    candidates: &[(usize, Option<usize>)],
    max_frame_rows: usize,
) -> BatchedK5Plan {
    let mut counts = [0usize; 16];
    let mut wide_counts: Option<std::collections::BTreeMap<usize, usize>> = None;
    for &(_, width) in candidates {
        if let Some(k) = width {
            if k < counts.len() {
                counts[k] += 1;
            } else {
                *wide_counts
                    .get_or_insert_with(Default::default)
                    .entry(k)
                    .or_default() += 1;
            }
        }
    }
    let mut chosen: Option<(usize, usize)> = None;
    for (k, &n) in counts.iter().enumerate() {
        if n >= 2 && chosen.is_none_or(|best| (n, k) > best) {
            chosen = Some((n, k));
        }
    }
    if let Some(ref map) = wide_counts {
        for (&k, &n) in map {
            if n >= 2 && chosen.is_none_or(|best| (n, k) > best) {
                chosen = Some((n, k));
            }
        }
    }
    let Some((_, chosen_k)) = chosen else {
        let mut leftover: Vec<usize> = candidates.iter().map(|&(idx, _)| idx).collect();
        leftover.sort_unstable();
        return BatchedK5Plan {
            frames: Vec::new(),
            leftover,
        };
    };
    let mut chosen_idxs = Vec::with_capacity(candidates.len());
    let mut leftover = Vec::new();
    for &(idx, width) in candidates {
        if width == Some(chosen_k) {
            chosen_idxs.push(idx);
        } else {
            leftover.push(idx);
        }
    }
    let sizes = spark_model::layer::Qwen4CrossSeqFrame::plan_frames_with_max_rows(
        chosen_idxs.len(),
        chosen_k,
        max_frame_rows,
    );
    let mut frames = Vec::with_capacity(sizes.len());
    let mut cursor = 0usize;
    for size in sizes {
        frames.push(chosen_idxs[cursor..cursor + size].to_vec());
        cursor += size;
    }
    leftover.extend_from_slice(&chosen_idxs[cursor..]);
    leftover.sort_unstable();
    BatchedK5Plan { frames, leftover }
}

/// [`plan_batched_k5`] restricted to one verify width: the width held by the
/// most candidates (ties go to the wider K). A cohort reserves one K for all
/// of its lanes; every other width is left over for the per-sequence
/// verifier. Used only under `ATLAS_QWEN4_BATCHED_ADAPTIVE_K=1`.
pub(in super::super) fn plan_batched_uniform(
    candidates: &[(usize, Option<usize>)],
) -> BatchedK5Plan {
    plan_batched_uniform_with_max_rows(
        candidates,
        spark_model::layers::qwen4_verify_lanes::adaptive_max_frame_rows(),
    )
}

/// Bin-pack heterogeneous eligible `(index, width)` candidates (`width in 1..=8`)
/// into minimal ragged frames satisfying `2 <= frame.len() <= 64` and
/// `sum(k_s) <= max_frame_rows`.
pub(in super::super) fn plan_ragged_cohort_with_max_rows(
    candidates: &[(usize, Option<usize>)],
    max_frame_rows: usize,
) -> BatchedK5Plan {
    let mut eligible_idxs: Vec<usize> = Vec::with_capacity(candidates.len());
    let mut eligible_ks: Vec<usize> = Vec::with_capacity(candidates.len());
    let mut leftover: Vec<usize> = Vec::new();
    for &(idx, width) in candidates {
        match width {
            Some(k)
                if (1..=spark_model::layer::Qwen4CrossSeqFrame::RAGGED_PLAN_MAX_K).contains(&k) =>
            {
                eligible_idxs.push(idx);
                eligible_ks.push(k);
            }
            _ => leftover.push(idx),
        }
    }
    let packed_bins = spark_model::layer::Qwen4CrossSeqFrame::plan_ragged_frames_with_max_rows(
        &eligible_ks,
        max_frame_rows,
    );
    let mut packed_mask = vec![false; eligible_idxs.len()];
    let mut frames = Vec::with_capacity(packed_bins.len());
    for bin in packed_bins {
        let mut frame = Vec::with_capacity(bin.len());
        for pos in bin {
            packed_mask[pos] = true;
            frame.push(eligible_idxs[pos]);
        }
        frames.push(frame);
    }
    for (pos, &packed) in packed_mask.iter().enumerate() {
        if !packed {
            leftover.push(eligible_idxs[pos]);
        }
    }
    leftover.sort_unstable();
    BatchedK5Plan { frames, leftover }
}

/// Bin-pack heterogeneous eligible `(index, width)` candidates (`width in 1..=8`)
/// into minimal ragged frames bounded by the active GPU's adaptive frame row cap.
pub(in super::super) fn plan_ragged_cohort(candidates: &[(usize, Option<usize>)]) -> BatchedK5Plan {
    plan_ragged_cohort_with_max_rows(
        candidates,
        spark_model::layers::qwen4_verify_lanes::adaptive_max_frame_rows(),
    )
}

/// Whether proposal depth is chosen per sequence (`C=1` or non-batched) or
/// held uniform across the active batched native-MTP cohort (`C>=2`).
///
/// On memory-bandwidth-bound GPUs, splitting a concurrent cohort onto
/// heterogeneous widths (`K4` vs `K5`) or dropping one member to depth `0`
/// (`bootstrap`) forces extra full target weight reads per tick. Holding one
/// uniform cohort width (`num_drafts`) keeps every active sequence inside the
/// single batched target forward.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in super::super) enum DraftDepthPolicy {
    PerSequence,
    CohortUniform,
}

impl DraftDepthPolicy {
    #[inline]
    pub(in super::super) const fn for_cohort(batched_armed: bool, live_seqs: usize) -> Self {
        if batched_armed && live_seqs >= 2 {
            Self::CohortUniform
        } else {
            Self::PerSequence
        }
    }

    #[inline]
    pub(in super::super) fn resolve(self, depth_key: u64, num_drafts: usize) -> usize {
        match self {
            Self::CohortUniform => num_drafts,
            Self::PerSequence => {
                crate::scheduler::adaptive_depth::seq_next_draft_depth(depth_key, num_drafts)
                    .unwrap_or(num_drafts)
            }
        }
    }
}

/// Choose a common verify width `k_target` when eligible flat-chain candidates
/// arrive with mixed widths (e.g. a `C1 -> C2` transition tick or post-grammar
/// splice) under `ATLAS_QWEN4_BATCHED_ADAPTIVE_K=1`.
///
/// Truncating eligible candidates wider than `k_target` down to `k_target` is
/// lossless (`narrow_pending_drafts` invariant). We compare exact
/// `verified_rows / total_forwards` against the no-truncation `plan_batched_uniform`
/// baseline and only return `Some(k_target)` when harmonization strictly
/// improves rows verified per target forward.
pub(in super::super) fn choose_harmonized_width(
    candidates: &[(usize, Option<usize>)],
) -> Option<usize> {
    let mut counts: std::collections::BTreeMap<usize, usize> = Default::default();
    let mut total_eligible = 0usize;
    let mut total_rows = 0usize;
    for &(_, width) in candidates {
        if let Some(k) = width {
            *counts.entry(k).or_default() += 1;
            total_eligible += 1;
            total_rows += k;
        }
    }
    if counts.len() < 2 || total_eligible < 2 {
        return None;
    }

    // Baseline: no truncation (`plan_batched_uniform` picks the single widest modal k with n >= 2).
    let baseline_plan = plan_batched_uniform(candidates);
    let baseline_batched: usize = baseline_plan.frames.iter().map(Vec::len).sum();
    let baseline_serial = total_eligible.saturating_sub(baseline_batched);
    let baseline_fwds = baseline_plan.frames.len() + baseline_serial;
    let mut best_rows = total_rows;
    let mut best_fwds = baseline_fwds.max(1);
    let mut best_k: Option<usize> = None;

    // Evaluate descending distinct widths `k_target`: every eligible seq with
    // `k >= k_target` narrows to `k_target` and batches together; narrower seqs
    // stay serial.
    for &k_target in counts.keys().rev() {
        let mut batched_n = 0usize;
        let mut serial_n = 0usize;
        let mut serial_rows = 0usize;
        for (&k, &n) in &counts {
            if k >= k_target {
                batched_n += n;
            } else {
                serial_n += n;
                serial_rows += k * n;
            }
        }
        if batched_n < 2 {
            continue;
        }
        let frames = spark_model::layer::Qwen4CrossSeqFrame::plan_frames_with_max_rows(
            batched_n,
            k_target,
            spark_model::layers::qwen4_verify_lanes::adaptive_max_frame_rows(),
        );
        let batched_in_frames: usize = frames.iter().sum();
        let leftover_from_plan = batched_n.saturating_sub(batched_in_frames);
        let fwds = frames.len() + serial_n + leftover_from_plan;
        if fwds == 0 {
            continue;
        }
        let rows = batched_n * k_target + serial_rows;
        // Exact rational comparison `rows / fwds > best_rows / best_fwds`.
        if (rows as u128) * (best_fwds as u128) > (best_rows as u128) * (fwds as u128) {
            best_rows = rows;
            best_fwds = fwds;
            best_k = Some(k_target);
        }
    }
    best_k
}

/// True when the model's active proposer and topology are eligible for
/// cross-sequence batched Qwen4 verification (`Qwen4NativeMtp`,
/// non-probabilistic `DFlash`, or `SpecVerifyPolicy::Rejection` with the GPU
/// rejection verify kernel available).
#[inline]
pub(super) fn batched_verify_proposer_eligible(model: &dyn Model) -> bool {
    batched_verify_proposer_eligible_with_policy(
        model,
        crate::scheduler::stochastic_verify_walk::spec_verify_policy(),
    )
}

#[inline]
pub(super) fn batched_verify_proposer_eligible_with_policy(
    model: &dyn Model,
    policy: crate::scheduler::stochastic_verify_walk::SpecVerifyPolicy,
) -> bool {
    if model.is_ep() && !model.qwen4_tp_batched_verify_supported() {
        return false;
    }
    match model.proposer_arm_kind(model.proposer_arm()) {
        Some(spark_model::speculative::DraftProposerKind::Qwen4NativeMtp) => {
            !model.proposer_is_dflash()
        }
        Some(spark_model::speculative::DraftProposerKind::DFlash) => {
            model.proposer_is_dflash()
                && (model.dflash_draft_sample_method()
                    != spark_model::dflash_sampling::DflashDraftSampleMethod::Probabilistic
                    || (policy
                        == crate::scheduler::stochastic_verify_walk::SpecVerifyPolicy::Rejection
                        && model.gpu_rejection_verify_available()))
        }
        _ => false,
    }
}

/// Physical KV sequence-length ceiling (`max_blocks_per_seq * block_size`)
/// exposed by models with resident paged KV caches, or `None` when absent.
#[inline]
pub(super) fn model_kv_max_seq_len(model: &dyn Model) -> Option<usize> {
    let dims = model.high_speed_swap_dims()?;
    let blocks = dims.max_blocks_per_layer as usize;
    let block_size = dims.block_size as usize;
    (blocks > 0 && block_size > 0).then(|| blocks.saturating_mul(block_size))
}

/// Maximum flat draft length that keeps `seq_len + 1 + draft_len <= max_seq_len`
/// (P0-3 belt-and-suspenders guard so a sequence reaching `max_seq_len` finishes
/// cleanly without overshooting `max_blocks_per_seq` and aborting co-batched peers).
#[inline]
#[must_use]
pub(in super::super) fn clamp_drafts_to_max_seq_len_count(
    seq_len: usize,
    draft_len: usize,
    max_seq_len: usize,
) -> usize {
    if max_seq_len == 0 {
        return draft_len;
    }
    draft_len.min(max_seq_len.saturating_sub(seq_len.saturating_add(1)))
}

/// Clamp flat `pending_drafts` of sequences in `idxs` to
/// `max_seq_len.saturating_sub(a.seq.seq_len + 1)` before cohort framing,
/// preserving tree/sampled-q frames and clamping any GPU rejection `q` fence.
pub(in super::super) fn clamp_pending_drafts_for_max_seq_len(
    active: &mut [ActiveSeq],
    idxs: &[usize],
    max_seq_len: usize,
) {
    if max_seq_len == 0 {
        return;
    }
    for &idx in idxs {
        let Some(a) = active.get_mut(idx) else {
            continue;
        };
        if a.pending_tree_payload.is_some() || a.pending_dflash_sampled_proposal.is_some() {
            continue;
        }
        let max_drafts =
            clamp_drafts_to_max_seq_len_count(a.seq.seq_len, a.pending_drafts.len(), max_seq_len);
        if max_drafts < a.pending_drafts.len() {
            a.pending_drafts.truncate(max_drafts);
            crate::scheduler::stochastic_verify_walk::truncate_seq_rejection_q(a, max_drafts);
        }
    }
}

/// Apply [`clamp_pending_drafts_for_max_seq_len`] using `model`'s physical KV ceiling.
pub(in super::super) fn clamp_pending_drafts_to_model_max_seq_len(
    model: &dyn Model,
    active: &mut [ActiveSeq],
    idxs: &[usize],
) {
    if let Some(max_seq_len) = model_kv_max_seq_len(model) {
        clamp_pending_drafts_for_max_seq_len(active, idxs, max_seq_len);
    }
}

/// Harmonize mixed eligible flat-chain draft lengths on transition ticks under
/// `ATLAS_QWEN4_BATCHED_ADAPTIVE_K=1` so the cohort verifies in one batched
/// forward instead of falling back to serial per-sequence forwards.
pub(in super::super) fn harmonize_cohort_widths(
    model: &dyn Model,
    active: &mut [ActiveSeq],
    verify_idxs: &[usize],
    num_drafts: usize,
    think: &ThinkSpecCtx<'_>,
) {
    clamp_pending_drafts_to_model_max_seq_len(model, active, verify_idxs);
    if !model.qwen4_batched_k5_armed()
        || !adaptive_selected()
        || verify_idxs.len() < 2
        || !batched_verify_proposer_eligible(model)
    {
        return;
    }
    let gumbel_drafts = dflash_gumbel_cohort_num_drafts(model, active, num_drafts);
    if gumbel_drafts < num_drafts {
        narrow_pending_drafts(active, verify_idxs, gumbel_drafts);
    }
    if batched_ragged_enabled() && !model.is_ep() {
        if model.proposer_is_dflash() {
            let is_eligible = |&idx: &usize| {
                let a = &active[idx];
                batched_k5_width_for(
                    BatchedK5Candidate {
                        draft_len: a.pending_drafts.len(),
                        configured_num_drafts: num_drafts,
                        grammar_active: a.grammar_state.is_some(),
                        finished: a.finished,
                        policy_required: batched_k5_policy_required(a, think),
                        has_tree: a.pending_tree_payload.is_some(),
                        has_sampled: a.pending_dflash_sampled_proposal.is_some(),
                    },
                    true,
                )
                .is_some()
            };
            let mut eligible_stack = [0usize; spark_model::layer::QWEN4_CROSS_SEQ_MAX_ROWS];
            let eligible_heap;
            let eligible_idxs: &[usize] = if verify_idxs.len() <= eligible_stack.len() {
                let mut eligible_len = 0usize;
                for &idx in verify_idxs {
                    if is_eligible(&idx) {
                        eligible_stack[eligible_len] = idx;
                        eligible_len += 1;
                    }
                }
                &eligible_stack[..eligible_len]
            } else {
                eligible_heap = verify_idxs
                    .iter()
                    .copied()
                    .filter(is_eligible)
                    .collect::<Vec<_>>();
                &eligible_heap
            };
            quantize_dflash_ragged_drafts_to_graph_bucket(active, eligible_idxs);
        }
        return;
    }
    let candidates: Vec<(usize, Option<usize>)> = verify_idxs
        .iter()
        .map(|&idx| {
            let a = &active[idx];
            let width = batched_k5_width(BatchedK5Candidate {
                draft_len: a.pending_drafts.len(),
                configured_num_drafts: num_drafts,
                grammar_active: a.grammar_state.is_some(),
                finished: a.finished,
                policy_required: batched_k5_policy_required(a, think),
                has_tree: a.pending_tree_payload.is_some(),
                has_sampled: a.pending_dflash_sampled_proposal.is_some(),
            });
            (idx, width)
        })
        .collect();
    let Some(target_k) = choose_harmonized_width(&candidates) else {
        return;
    };
    let target_drafts = target_k.saturating_sub(1).max(1);
    for (idx, width) in candidates {
        if width.is_some_and(|k| k > target_k) {
            let a = &mut active[idx];
            if a.pending_tree_payload.is_none()
                && a.pending_dflash_sampled_proposal.is_none()
                && a.pending_drafts.len() > target_drafts
            {
                a.pending_drafts.truncate(target_drafts);
                crate::scheduler::stochastic_verify_walk::truncate_seq_rejection_q(
                    a,
                    target_drafts,
                );
            }
        }
    }
}

/// Build the plan for the current verify cohort.
pub(in super::super) fn plan_for_cohort(
    model: &dyn Model,
    active: &[ActiveSeq],
    verify_idxs: &[usize],
    num_drafts: usize,
    think: &ThinkSpecCtx<'_>,
) -> BatchedK5Plan {
    let model_ok = model.qwen4_batched_k5_armed() && batched_verify_proposer_eligible(model);
    let ragged = batched_ragged_enabled() && !model.is_ep();
    let kv_max_seq_len = model_kv_max_seq_len(model);
    let candidates: Vec<(usize, Option<usize>)> = verify_idxs
        .iter()
        .map(|&idx| {
            let a = &active[idx];
            let fits_kv_ceiling = kv_max_seq_len.is_none_or(|cap| {
                a.seq
                    .seq_len
                    .saturating_add(a.pending_drafts.len())
                    .saturating_add(1)
                    <= cap
            });
            let width = (model_ok && fits_kv_ceiling)
                .then(|| {
                    batched_k5_width_for(
                        BatchedK5Candidate {
                            draft_len: a.pending_drafts.len(),
                            configured_num_drafts: num_drafts,
                            grammar_active: a.grammar_state.is_some(),
                            finished: a.finished,
                            policy_required: batched_k5_policy_required(a, think),
                            has_tree: a.pending_tree_payload.is_some(),
                            has_sampled: a.pending_dflash_sampled_proposal.is_some(),
                        },
                        ragged,
                    )
                })
                .flatten();
            (idx, width)
        })
        .collect();
    if ragged {
        plan_ragged_cohort(&candidates)
    } else if adaptive_selected() {
        plan_batched_uniform(&candidates)
    } else {
        plan_batched_k5(&candidates)
    }
}

/// Zero-allocation distinctness + bounds check for scheduler cohort indices
/// (`len <= 256` uses a stack bitset; larger active tables fall back to a set).
#[inline]
pub(super) fn indices_distinct_in_range(idxs: &[usize], len: usize) -> bool {
    if len <= 256 {
        let mut bits = [0u64; 4];
        for &idx in idxs {
            if idx >= len {
                return false;
            }
            let word = &mut bits[idx >> 6];
            let mask = 1u64 << (idx & 63);
            if (*word & mask) != 0 {
                return false;
            }
            *word |= mask;
        }
        true
    } else {
        let mut seen = std::collections::HashSet::with_capacity(idxs.len());
        idxs.iter().all(|&idx| idx < len && seen.insert(idx))
    }
}
