// SPDX-License-Identifier: AGPL-3.0-only

//! Cross-sequence batched native-MTP verify step (`ATLAS_QWEN4_BATCHED_K5=1`).
//!
//! Every eligible speculating flat-chain sequence — including neutral greedy,
//! grammar-constrained, thinking-span (`think.enabled`), stochastic/Gumbel,
//! penalized, and `min_tokens` requests — is grouped by verify width K (or
//! packed into ragged mixed-K frames in `1..=8` when `batched_ragged_enabled()`
//! holds) into frames bounded by the active GPU's adaptive row cap (`128` rows
//! on GB10, `256` rows on `> 64`-SM GPUs, clamped to `QWEN4_CROSS_SEQ_MAX_ROWS
//! = 256` and `QWEN4_CROSS_SEQ_MAX_SEQS = 64` via
//! `spark_model::layer::Qwen4CrossSeqFrame::plan_frames_with_max_rows`). One
//! target forward verifies a whole frame (`Model::decode_verify_qwen4_batched`).
//! Each sequence then accepts its own exact prefix, emits, rolls back its host
//! extent and commits its own recurrent/QSA/PLE state from its own intermediates
//! (the accepted row is promoted, the rest discarded), all inside the verify
//! lane of its slot. A cohort owns every physical lane until all frames have
//! settled. Hidden handoffs for every sequence are saved before the cohort is
//! released and any proposer runs, because a proposal reuses the shared target
//! scratch.
//!
//! Sequences that are not eligible (trees, sampled-q frames, finished
//! sequences, thinking when `!think.enabled`, or `ATLAS_GRAMMAR_SERIAL_VERIFY=1`
//! grammar overrides), singleton widths, and frames the model refuses at
//! read-only admission fall back to the per-sequence verifier. After cohort
//! begin, every failure aborts the whole cohort without fallback.
//!
//! `ATLAS_QWEN4_BATCHED_ADAPTIVE_K=1` (opt-in) narrows the verify width with
//! the number of live sequences so one frame fits inside the active row cap
//! (`qwen4_verify_lanes::adaptive_verify_width` /
//! `adaptive_verify_width_for_profile_and_max_rows`): under the 128-row default
//! profile, C1-C16 K8, C17-C18 K7, C19-C21 K6, C22-C25 K5, C26-C32 K4,
//! C33-C42 K3, C43-C128 K2 (`C129+` returns `None`); 256-row GPUs scale widths
//! upward within `K <= 8`. Narrowing only shortens a draft chain, exactly as
//! the MTP confidence gate does, so verify still decides every token.

use super::spec_accept_log::record_settled_frame;
use super::verify_dflash_step::{
    cfg_jf_splice_drafts, dflash_think_mask_verified_at_row, exact_accept_prefix,
    reargmax_bonus_skip_think, reconcile_terminal_verify_state, wide_verify_metric_k,
};
use super::*;
use anyhow::Context;
use spark_model::layers::qwen4_verify_lanes::LaneGuard;

/// Per-sequence facts that decide batched eligibility.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct BatchedK5Candidate {
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
pub(super) fn batched_k5_width_for(c: BatchedK5Candidate, allow_k1: bool) -> Option<usize> {
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

pub(super) fn batched_k5_width(c: BatchedK5Candidate) -> Option<usize> {
    batched_k5_width_for(c, false)
}

/// True when true ragged mixed-`K` (`K_s in 1..=8`) verify frames and
/// active-subset compacted MTP drafting are enabled (default ON under
/// `ATLAS_QWEN4_BATCHED_ADAPTIVE_K=1`; set `ATLAS_QWEN4_BATCHED_RAGGED_MTP=0`
/// to restore uniform `CohortAcceptEwma` widths for A/B testing).
#[inline]
fn batched_ragged_enabled() -> bool {
    adaptive_selected() && spark_model::layers::qwen4_mtp::batched_ragged_mtp_enabled()
}

/// Frames to run and the sequences left for the per-sequence verifier.
#[derive(Debug, Default, PartialEq, Eq)]
pub(super) struct BatchedK5Plan {
    pub frames: Vec<Vec<usize>>,
    pub leftover: Vec<usize>,
}

/// Group `(index, width)` candidates by width (ascending, stable within a
/// width) and split each group into balanced frames bounded by `max_frame_rows`.
/// `None` widths and any width with a single sequence are left over.
pub(super) fn plan_batched_k5_with_max_rows(
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
pub(super) fn plan_batched_k5(candidates: &[(usize, Option<usize>)]) -> BatchedK5Plan {
    plan_batched_k5_with_max_rows(
        candidates,
        spark_model::layers::qwen4_verify_lanes::adaptive_max_frame_rows(),
    )
}

/// Parse `ATLAS_GRAMMAR_SERIAL_VERIFY`: `"1"` (trimmed) restores per-sequence
/// verification for grammar-active sequences during A/B comparisons.
pub(super) fn parse_grammar_serial_verify(raw: Option<&str>) -> bool {
    raw.map(str::trim) == Some("1")
}

/// Opt-in escape hatch (`ATLAS_GRAMMAR_SERIAL_VERIFY=1`) restoring per-sequence
/// verification for grammar-active sequences during A/B comparisons.
fn grammar_serial_verify() -> bool {
    static SERIAL: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *SERIAL.get_or_init(|| {
        parse_grammar_serial_verify(std::env::var("ATLAS_GRAMMAR_SERIAL_VERIFY").ok().as_deref())
    })
}

/// Parse `ATLAS_QWEN4_BATCHED_K5_LOG`: exact `"1"` enables batched-K5 step logs.
pub(super) fn parse_batched_k5_log_enabled(raw: Option<&str>) -> bool {
    raw == Some("1")
}

/// Cached `ATLAS_QWEN4_BATCHED_K5_LOG=1` observability gate (read once instead
/// of twice per verify step on the scheduler hot path).
#[inline]
fn batched_k5_log_enabled() -> bool {
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
pub(super) fn batched_k5_policy_required(a: &ActiveSeq, think: &ThinkSpecCtx<'_>) -> bool {
    (a.grammar_state.is_some() && grammar_serial_verify()) || (a.inside_thinking && !think.enabled)
}

/// [`plan_batched_k5_with_max_rows`] restricted to one verify width: the width
/// held by the most candidates (ties go to the wider K).
pub(super) fn plan_batched_uniform_with_max_rows(
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
pub(super) fn plan_batched_uniform(candidates: &[(usize, Option<usize>)]) -> BatchedK5Plan {
    plan_batched_uniform_with_max_rows(
        candidates,
        spark_model::layers::qwen4_verify_lanes::adaptive_max_frame_rows(),
    )
}

/// Bin-pack heterogeneous eligible `(index, width)` candidates (`width in 1..=8`)
/// into minimal ragged frames satisfying `2 <= frame.len() <= 64` and
/// `sum(k_s) <= max_frame_rows`.
pub(super) fn plan_ragged_cohort_with_max_rows(
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
pub(super) fn plan_ragged_cohort(candidates: &[(usize, Option<usize>)]) -> BatchedK5Plan {
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
pub(super) enum DraftDepthPolicy {
    PerSequence,
    CohortUniform,
}

impl DraftDepthPolicy {
    #[inline]
    pub(super) const fn for_cohort(batched_armed: bool, live_seqs: usize) -> Self {
        if batched_armed && live_seqs >= 2 {
            Self::CohortUniform
        } else {
            Self::PerSequence
        }
    }

    #[inline]
    pub(super) fn resolve(self, depth_key: u64, num_drafts: usize) -> usize {
        match self {
            Self::CohortUniform => num_drafts,
            Self::PerSequence => super::adaptive_depth::seq_next_draft_depth(depth_key, num_drafts)
                .unwrap_or(num_drafts),
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
pub(super) fn choose_harmonized_width(candidates: &[(usize, Option<usize>)]) -> Option<usize> {
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
fn batched_verify_proposer_eligible(model: &dyn Model) -> bool {
    batched_verify_proposer_eligible_with_policy(
        model,
        super::stochastic_verify_walk::spec_verify_policy(),
    )
}

#[inline]
fn batched_verify_proposer_eligible_with_policy(
    model: &dyn Model,
    policy: super::stochastic_verify_walk::SpecVerifyPolicy,
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
                    || (policy == super::stochastic_verify_walk::SpecVerifyPolicy::Rejection
                        && model.gpu_rejection_verify_available()))
        }
        _ => false,
    }
}

/// Physical KV sequence-length ceiling (`max_blocks_per_seq * block_size`)
/// exposed by models with resident paged KV caches, or `None` when absent.
#[inline]
fn model_kv_max_seq_len(model: &dyn Model) -> Option<usize> {
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
pub(super) fn clamp_drafts_to_max_seq_len_count(
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
pub(super) fn clamp_pending_drafts_for_max_seq_len(
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
            super::stochastic_verify_walk::truncate_seq_rejection_q(a, max_drafts);
        }
    }
}

/// Apply [`clamp_pending_drafts_for_max_seq_len`] using `model`'s physical KV ceiling.
pub(super) fn clamp_pending_drafts_to_model_max_seq_len(
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
pub(super) fn harmonize_cohort_widths(
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
                super::stochastic_verify_walk::truncate_seq_rejection_q(a, target_drafts);
            }
        }
    }
}

/// Build the plan for the current verify cohort.
pub(super) fn plan_for_cohort(
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
fn indices_distinct_in_range(idxs: &[usize], len: usize) -> bool {
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

/// Run every planned frame under one physical-lane reservation. A read-only
/// refusal leaves all proposals untouched and permits scalar fallback. Once
/// begin is attempted, every failure is terminal for the whole cohort.
pub(super) fn step_verify_qwen4_batched(
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

pub(super) fn batch_sequences_mut<'a>(
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
        super::stochastic_verify_walk::invalidate_seq_rejection_q(&active[idx]);
        fail_speculative_step(&mut active[idx], stage, &error);
    }
    crate::metrics::compat_metrics::set_kv_blocks_free(model.num_free_blocks());
}

fn verify_frame(
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
    let policy = super::stochastic_verify_walk::spec_verify_policy();
    let (seq_in_graph_gumbel, seq_in_graph_rejection) =
        if policy == super::stochastic_verify_walk::SpecVerifyPolicy::Rejection {
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
        super::stochastic_verify_walk::invalidate_seq_rejection_q(&active[idx]);
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
        super::adaptive_depth::ActiveDepthSessionGuard::new(a.depth_controller_key());
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
        super::spec_accept_log::CommittedFrameReceipt {
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
        super::adaptive_depth::ActiveDepthSessionGuard::new(a.depth_controller_key());
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
        super::spec_accept_log::CommittedFrameReceipt {
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
            super::spec_policy_accept::run_dflash_thinking_rejection_accept_at_row(
                model,
                a,
                row_base,
                drafts,
                verified,
                think,
                presampled_rejection,
            )
        } else {
            super::think_spec_accept::run_dflash_thinking_accept_at_row_with_presampled(
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
            super::spec_policy_accept::run_dflash_content_accept_at_row_with_rejection(
                model,
                a,
                row_base,
                drafts,
                verified,
                think,
                presampled_rejection,
            )
        } else {
            super::spec_policy_accept::run_dflash_content_accept_at_row_with_presampled(
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
            thinking_frame && super::think_spec_accept::outcome_ended_thinking(a, &outcome),
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

/// Next proposal for one committed sequence: the retrieval cascade first,
/// then the native MTP head, followed by zero-GPU-cost history fallback or
/// suffix grafting when remaining slots up to `num_drafts` are open.
fn propose_next(
    model: &dyn Model,
    a: &mut ActiveSeq,
    num_drafts: usize,
    depth_policy: DraftDepthPolicy,
) {
    super::mtp_step::history_ring::sync_one_validated_tokens(a.session_hash, &a.output_tokens);
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
        let _ = super::mtp_step::reconcile_dflash_before_external_draft(model, a);
        super::stochastic_verify_walk::invalidate_seq_rejection_q(a);
        return;
    }
    let _depth_session =
        super::adaptive_depth::ActiveDepthSessionGuard::new(a.depth_controller_key());
    super::stochastic_verify_walk::publish_seq_gumbel_draft_spec(a);
    let token = a.last_token;
    let grammar_masked = a.grammar_state.is_some();
    let allow_history_graft = depth_policy == DraftDepthPolicy::PerSequence;
    let mut retrieved = super::think_budget_nudge::full_forced_draft_chain(a, seq_drafts)
        .map(|chain| (DraftOrigin::Proposer, chain))
        .or_else(|| super::mtp_step::retrieval_chain(a, token, num_drafts))
        .or_else(|| {
            (allow_history_graft && !grammar_masked)
                .then(|| {
                    super::mtp_step::history_ring::try_explicit_preempt(
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
    if !should_propose && !super::mtp_step::reconcile_dflash_before_external_draft(model, a) {
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
                super::stochastic_verify_walk::invalidate_seq_rejection_q(a);
                *proposal = Ok(chain);
                a.draft_origin = origin;
            } else if !should_propose {
                notify_qwen4_external_draft(a, &[]);
                super::stochastic_verify_walk::invalidate_seq_rejection_q(a);
            }
            match proposal {
                Ok(drafts) if !drafts.is_empty() => {
                    let before_splice = drafts.clone();
                    cfg_jf_splice_drafts(a, drafts);
                    super::think_budget_nudge::splice_forced_drafts(a, drafts);
                    splice_grammar_drafts(a, drafts, vocab_size);
                    super::stochastic_verify_walk::truncate_seq_rejection_q_after_splice(
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
        super::mtp_step::apply_history_fallback_or_graft(
            model,
            a,
            token,
            num_drafts,
            grammar_masked,
        );
    }
}

pub(super) fn propose_next_batch(
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
        && super::stochastic_verify_walk::spec_verify_policy()
            == super::stochastic_verify_walk::SpecVerifyPolicy::Rejection;
    super::mtp_step::history_ring::sync_active_validated_tokens(active);
    let mut mtp_idxs: Vec<(usize, usize)> = Vec::with_capacity(propose_idxs.len());
    let vocab_size = model.vocab_size();
    for &idx in propose_idxs {
        let a = &mut active[idx];
        if dflash_grammar_skip_propose(model, a) {
            let _ = super::mtp_step::reconcile_dflash_before_external_draft(model, a);
            super::stochastic_verify_walk::invalidate_seq_rejection_q(a);
            continue;
        }
        // Every sequence in a multi-sequence batched cohort proposes the uniform
        // cohort width `num_drafts` so the next tick verifies all sequences in
        // one batched target forward instead of splintering into serial forwards.
        let seq_drafts = num_drafts;
        let token = a.last_token;
        let preempt = super::think_budget_nudge::full_forced_draft_chain(a, seq_drafts)
            .map(|chain| (DraftOrigin::Proposer, chain))
            .or_else(|| super::mtp_step::retrieval_chain(a, token, num_drafts));
        if let Some((origin, chain)) = preempt {
            if !super::mtp_step::reconcile_dflash_before_external_draft(model, a) {
                super::stochastic_verify_walk::invalidate_seq_rejection_q(a);
                continue;
            }
            let _depth_session =
                super::adaptive_depth::ActiveDepthSessionGuard::new(a.depth_controller_key());
            let (result, ()) = proposal_lifecycle::propose_and_install_with(
                model,
                a,
                token,
                num_drafts,
                None,
                false,
                |a, proposal| {
                    notify_qwen4_external_draft(a, &chain);
                    super::stochastic_verify_walk::invalidate_seq_rejection_q(a);
                    *proposal = Ok(chain);
                    a.draft_origin = origin;
                    if let Ok(drafts) = proposal
                        && !drafts.is_empty()
                    {
                        cfg_jf_splice_drafts(a, drafts);
                        super::think_budget_nudge::splice_forced_drafts(a, drafts);
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
                super::stochastic_verify_walk::invalidate_seq_rejection_q(a);
                super::stochastic_verify_walk::publish_seq_gumbel_draft_spec(a);
                a.pending_drafts.clear();
                a.pending_tree_payload = None;
                a.pending_dflash_sampled_proposal = None;
                a.draft_origin = super::DraftOrigin::Proposer;
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
                        if drafts.len() > seq_drafts {
                            drafts.truncate(seq_drafts);
                            super::stochastic_verify_walk::truncate_seq_rejection_q(
                                a,
                                drafts.len(),
                            );
                        }
                        if !drafts.is_empty() {
                            let before_splice = drafts.clone();
                            cfg_jf_splice_drafts(a, &mut drafts);
                            super::think_budget_nudge::splice_forced_drafts(a, &mut drafts);
                            splice_grammar_drafts(a, &mut drafts, vocab_size);
                            super::stochastic_verify_walk::truncate_seq_rejection_q_after_splice(
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
                        if !use_upward_rejection_pcum {
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
    if use_upward_rejection_pcum {
        apply_dflash_pcum_and_upward_bucket_quantize_cohort(
            active,
            propose_idxs,
            model.proposer_is_dflash(),
        );
    } else if model.proposer_is_dflash() && ragged {
        quantize_dflash_ragged_drafts_to_graph_bucket(active, propose_idxs);
    }
    if propose_idxs.len() < 2 {
        warm_ple_draft_ahead_for_seqs(model, active, propose_idxs);
    } else {
        warm_ple_verify_prefetch_for_seqs(model, active, propose_idxs, ragged);
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
pub(super) fn warm_ple_verify_prefetch_for_seqs(
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
pub(super) fn warm_ple_draft_ahead_for_seqs(
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
pub(super) fn warm_ple_draft_ahead_for_seq(model: &dyn Model, a: &ActiveSeq) {
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

/// Opt-in per-concurrency draft-width ladder, e.g. `2:4,4:3,8:2,32:1`: up
/// to 2 speculating sequences propose 4 drafts (K5), up to 4 propose 3, up
/// to 8 propose 2, beyond that 1. Unset keeps the configured width at every
/// concurrency. Draft width changes only acceptance and cost, never output.
pub(super) const LADDER_ENV: &str = "ATLAS_QWEN4_BATCHED_K5_LADDER";

/// Parse `width:drafts` rungs (strictly ascending widths, drafts >= 1).
pub(super) fn parse_ladder(raw: &str) -> std::result::Result<Vec<(usize, usize)>, String> {
    let mut rungs: Vec<(usize, usize)> = Vec::new();
    for rung in raw.split(',') {
        let (width, drafts) = rung
            .split_once(':')
            .ok_or_else(|| format!("{LADDER_ENV} rung {rung:?} is not width:drafts"))?;
        let width: usize = width
            .parse()
            .map_err(|_| format!("{LADDER_ENV} width {width:?} is not an integer"))?;
        let drafts: usize = drafts
            .parse()
            .map_err(|_| format!("{LADDER_ENV} drafts {drafts:?} is not an integer"))?;
        if width == 0 || drafts == 0 {
            return Err(format!("{LADDER_ENV} rung {rung:?} must be positive"));
        }
        if rungs.last().is_some_and(|&(last, _)| width <= last) {
            return Err(format!("{LADDER_ENV} widths must strictly ascend"));
        }
        rungs.push((width, drafts));
    }
    if rungs.is_empty() {
        return Err(format!("{LADDER_ENV} is empty"));
    }
    Ok(rungs)
}

/// Drafts per sequence at `concurrency`, never above `configured`.
pub(super) fn ladder_drafts(
    ladder: &[(usize, usize)],
    concurrency: usize,
    configured: usize,
) -> usize {
    let drafts = ladder
        .iter()
        .find(|&&(width, _)| concurrency <= width)
        .or(ladder.last())
        .map_or(configured, |&(_, drafts)| drafts);
    drafts.min(configured)
}

fn ladder() -> Option<&'static [(usize, usize)]> {
    static LADDER: std::sync::OnceLock<Option<Vec<(usize, usize)>>> = std::sync::OnceLock::new();
    LADDER
        .get_or_init(|| {
            let raw = std::env::var(LADDER_ENV).ok()?;
            match parse_ladder(&raw) {
                Ok(rungs) => {
                    tracing::info!(ladder = raw, "QWEN4_BATCHED_K5_LADDER_ENGAGED");
                    Some(rungs)
                }
                Err(error) => {
                    tracing::warn!("{error}; keeping the configured draft width");
                    None
                }
            }
        })
        .as_deref()
}

struct CohortAcceptState {
    acc_ewma: f32,
    trials_ewma: f32,
    held_k: usize,
    dwell_steps: u32,
    down_streak: u32,
    up_streak: u32,
    last_live: usize,
    observed_frames: u64,
    evaluated_frames: u64,
    last_activity: Option<Instant>,
}

impl CohortAcceptState {
    const INITIAL_P_STEP: f32 = 0.80;
    const INITIAL_TRIALS_PER_SEQ: f32 = 2.0;
    const INITIAL_ACC_PER_SEQ: f32 = Self::INITIAL_P_STEP * Self::INITIAL_TRIALS_PER_SEQ;
    const EWMA_ALPHA: f32 = 0.15;
    /// Minimum settled batched frames held at the current width `K` before any
    /// upward or downward transition may fire (stabilizes batched CUDA graphs).
    const MIN_DWELL_STEPS: u32 = 6;
    /// Consecutive frames favoring a narrower `K` (by at least
    /// `HYSTERESIS_RATE_MARGIN_DOWN`) required before stepping down.
    const CONFIRM_STREAK_DOWN: u32 = 4;
    /// Consecutive frames favoring a wider `K` (by at least
    /// `HYSTERESIS_RATE_MARGIN_UP`) required before stepping up.
    const CONFIRM_STREAK_UP: u32 = 3;
    /// Asymmetric hysteresis margins (`+4%` down / `+2%` up = `6%` deadband):
    /// stepping down pays CUDA graph warmup/capture and risks under-drafting
    /// on transient dips across `K = 2..=8`, while stepping up recovers
    /// promptly when acceptance rises.
    const HYSTERESIS_RATE_MARGIN_DOWN: f64 = 0.04;
    const HYSTERESIS_RATE_MARGIN_UP: f64 = 0.02;
    /// Idle gap between concurrent cohort steps after which a new burst
    /// warm-starts at `INITIAL_P_STEP` and full cohort `max_k`.
    const IDLE_WARM_START_GAP: std::time::Duration = std::time::Duration::from_secs(2);

    const fn new() -> Self {
        Self {
            acc_ewma: Self::INITIAL_ACC_PER_SEQ,
            trials_ewma: Self::INITIAL_TRIALS_PER_SEQ,
            held_k: spark_model::layers::qwen4_verify_lanes::ADAPTIVE_MAX_K,
            dwell_steps: 0,
            down_streak: 0,
            up_streak: 0,
            last_live: 0,
            observed_frames: 0,
            evaluated_frames: 0,
            last_activity: None,
        }
    }

    #[inline]
    fn ewma_p_step(&self) -> f32 {
        if !self.trials_ewma.is_finite() || self.trials_ewma <= 0.0 || !self.acc_ewma.is_finite() {
            return Self::INITIAL_P_STEP;
        }
        (self.acc_ewma / self.trials_ewma).clamp(0.0, 1.0)
    }

    fn warm_start(&mut self, live: usize, max_k: usize) {
        self.acc_ewma = Self::INITIAL_ACC_PER_SEQ;
        self.trials_ewma = Self::INITIAL_TRIALS_PER_SEQ;
        self.held_k = max_k;
        self.dwell_steps = 0;
        self.down_streak = 0;
        self.up_streak = 0;
        self.last_live = live;
        self.evaluated_frames = self.observed_frames;
    }

    fn observe_frame_seqs(
        &mut self,
        drafted_seqs: usize,
        total_accepted: usize,
        total_drafted: usize,
        rejected_seqs: usize,
    ) {
        if drafted_seqs == 0 || total_drafted == 0 {
            return;
        }
        let trials = total_accepted.saturating_add(rejected_seqs);
        if trials == 0 {
            return;
        }
        let seqs_f = drafted_seqs as f32;
        let acc_obs = total_accepted as f32 / seqs_f;
        let trials_obs = trials as f32 / seqs_f;
        let prev_acc = if self.acc_ewma.is_finite() && self.acc_ewma >= 0.0 {
            self.acc_ewma
        } else {
            Self::INITIAL_ACC_PER_SEQ
        };
        let prev_trials = if self.trials_ewma.is_finite() && self.trials_ewma > 0.0 {
            self.trials_ewma
        } else {
            Self::INITIAL_TRIALS_PER_SEQ
        };
        let keep = 1.0 - Self::EWMA_ALPHA;
        self.acc_ewma = (Self::EWMA_ALPHA * acc_obs + keep * prev_acc).max(0.0);
        self.trials_ewma = (Self::EWMA_ALPHA * trials_obs + keep * prev_trials).max(1e-6);
        self.observed_frames = self.observed_frames.wrapping_add(1);
    }

    fn select_k_inner(&mut self, live: usize, max_k: usize, idle_expired: bool) -> usize {
        let min_k = spark_model::layers::qwen4_verify_lanes::ADAPTIVE_MIN_K;
        let upper = max_k.clamp(
            min_k,
            spark_model::layers::qwen4_verify_lanes::ADAPTIVE_MAX_K,
        );
        if live <= 1 {
            self.last_live = live;
            return upper;
        }
        let is_new_burst = idle_expired
            || self.last_live <= 1
            || live >= self.last_live.saturating_mul(2)
            || live >= self.last_live.saturating_add(4);
        if is_new_burst {
            self.warm_start(live, upper);
            return self.held_k;
        }
        self.last_live = live;
        self.held_k = self.held_k.clamp(min_k, upper);
        if self.observed_frames != self.evaluated_frames {
            self.evaluated_frames = self.observed_frames;
            self.dwell_steps = self.dwell_steps.saturating_add(1);
            let p = self.ewma_p_step();
            let cand = spark_model::layers::qwen4_verify_lanes::optimal_cohort_k_for_max_rows(
                live,
                p,
                upper,
                spark_model::layers::qwen4_verify_lanes::adaptive_max_frame_rows(),
            );
            let held_rate =
                spark_model::layers::qwen4_verify_lanes::cohort_expected_rate(live, p, self.held_k);
            let cand_rate =
                spark_model::layers::qwen4_verify_lanes::cohort_expected_rate(live, p, cand);
            if cand < self.held_k
                && cand_rate > held_rate * (1.0 + Self::HYSTERESIS_RATE_MARGIN_DOWN)
            {
                self.down_streak = self.down_streak.saturating_add(1);
                self.up_streak = 0;
                if self.dwell_steps >= Self::MIN_DWELL_STEPS
                    && self.down_streak >= Self::CONFIRM_STREAK_DOWN
                {
                    self.held_k = cand;
                    self.dwell_steps = 0;
                    self.down_streak = 0;
                }
            } else if cand > self.held_k
                && cand_rate > held_rate * (1.0 + Self::HYSTERESIS_RATE_MARGIN_UP)
            {
                self.up_streak = self.up_streak.saturating_add(1);
                self.down_streak = 0;
                if self.dwell_steps >= Self::MIN_DWELL_STEPS
                    && self.up_streak >= Self::CONFIRM_STREAK_UP
                {
                    self.held_k = cand;
                    self.dwell_steps = 0;
                    self.up_streak = 0;
                }
            } else {
                self.down_streak = 0;
                self.up_streak = 0;
            }
        }
        self.held_k
    }
}

/// Online cohort acceptance estimator and hysteresis-stabilized verify-width
/// controller for batched native-MTP cohorts (`live >= 2`).
///
/// Tracks exponentially-weighted per-sequence accepted drafts and truncated-
/// geometric trials (`acc_ewma / trials_ewma`, `alpha = 0.15`) rather than an
/// EWMA of per-frame ratios, eliminating Jensen downward bias on small cohorts
/// (`c = 2..4`) while remaining K-invariant across `K = 2..=ADAPTIVE_MAX_K`.
/// Transitions require a minimum dwell (`>= 6` settled frames) plus directional
/// confirmation (`4` frames with `>= +4%` predicted throughput gain to step
/// down, `3` frames with `>= +2%` gain to step up) to prevent under-drafting
/// and keep batched CUDA graph captures minimal. New concurrent bursts
/// (`live <= 1 -> >= 2`, sharp ramp-ups, or `>= 2s` idle gaps) warm-start at
/// `p_step = 0.80` and full single-frame cohort `max_k` (e.g. `K = 7` with
/// `--num-drafts 6` for `c <= 16`, narrowed automatically by `128 / live` at
/// higher concurrency).
pub(super) struct CohortAcceptEwma {
    state: parking_lot::Mutex<CohortAcceptState>,
}

impl CohortAcceptEwma {
    pub(super) const fn new() -> Self {
        Self {
            state: parking_lot::Mutex::new(CohortAcceptState::new()),
        }
    }

    #[cfg(test)]
    #[inline]
    #[must_use]
    pub(super) fn ewma_p_step(&self) -> f32 {
        self.state.lock().ewma_p_step()
    }

    /// Fold one settled batched verify frame into the cohort EWMA using the
    /// unbiased ratio-of-EWMAs truncated-geometric estimator
    /// `ewma(accepted / seqs) / ewma((accepted + rejected_seqs) / seqs)`.
    pub(super) fn observe_frame_seqs(
        &self,
        drafted_seqs: usize,
        total_accepted: usize,
        total_drafted: usize,
        rejected_seqs: usize,
    ) {
        self.state.lock().observe_frame_seqs(
            drafted_seqs,
            total_accepted,
            total_drafted,
            rejected_seqs,
        );
    }

    /// Record a solo/idle step (`live <= 1`) so the next concurrent burst
    /// warm-starts cleanly at `INITIAL_P_STEP` and full cohort `max_k`.
    pub(super) fn mark_solo(&self, live: usize) {
        let mut guard = self.state.lock();
        guard.last_live = live;
        guard.last_activity = None;
    }

    /// Select the hysteresis-stabilized verify width `K` for `live` concurrent
    /// sequences (bounded by `max_k`), warm-starting on new concurrent bursts
    /// or after an idle gap (`>= 2s`).
    pub(super) fn select_k(&self, live: usize, max_k: usize) -> usize {
        let now = Instant::now();
        let mut guard = self.state.lock();
        let idle_expired = live >= 2
            && guard.last_activity.is_some_and(|prev| {
                now.saturating_duration_since(prev) >= CohortAcceptState::IDLE_WARM_START_GAP
            });
        if live >= 2 {
            guard.last_activity = Some(now);
        } else {
            guard.last_activity = None;
        }
        guard.select_k_inner(live, max_k, idle_expired)
    }

    /// Deterministic version of [`Self::select_k`] without wall-clock checks,
    /// used by unit tests to verify dwell, hysteresis, and warm-start transitions.
    #[cfg(test)]
    pub(super) fn select_k_deterministic(
        &self,
        live: usize,
        max_k: usize,
        idle_expired: bool,
    ) -> usize {
        self.state.lock().select_k_inner(live, max_k, idle_expired)
    }
}

static COHORT_ACCEPT_EWMA: CohortAcceptEwma = CohortAcceptEwma::new();

fn cohort_adaptive_k_enabled() -> bool {
    let read_live = || match spark_model::layers::qwen4_verify_lanes::cohort_adaptive_k_selected() {
        Ok(on) => on,
        Err(error) => {
            tracing::warn!("{error:#}; keeping cohort adaptive K enabled");
            true
        }
    };
    #[cfg(not(test))]
    {
        static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ENABLED.get_or_init(read_live)
    }
    #[cfg(test)]
    {
        read_live()
    }
}

/// Fixed batched verify draft count (`K = 5`) when `ATLAS_QWEN4_BATCHED_K5=1`
/// runs without `ATLAS_QWEN4_BATCHED_ADAPTIVE_K=1` (`verify_batch_owner` admits
/// `k == 5` only in non-adaptive mode).
const FIXED_BATCHED_K5_DRAFTS: usize = 4;

/// Resolve the cohort width ceiling (`drafts` per sequence) for `live` active
/// speculating sequences before single-sequence stochastic depth is considered,
/// using the stateless GB10 roofline model directly (for unit-test coverage of
/// overrides, ladders, and caps).
#[cfg(test)]
pub(super) fn resolve_cohort_width_capped(
    batched_k5_armed: bool,
    adaptive_k: bool,
    cohort_adaptive_k: bool,
    explicit_ladder: Option<&[(usize, usize)]>,
    live: usize,
    configured: usize,
    ewma_p_step: f32,
) -> usize {
    if !batched_k5_armed || live <= 1 || configured == 0 {
        return configured;
    }
    let adaptive_max_k = spark_model::layers::qwen4_verify_lanes::ADAPTIVE_MAX_K;
    let batched_configured = configured.min(adaptive_max_k.saturating_sub(1));
    if let Some(rungs) = explicit_ladder {
        return ladder_drafts(rungs, live, batched_configured);
    }
    if !adaptive_k {
        return batched_configured.min(FIXED_BATCHED_K5_DRAFTS);
    }
    let profile_drafts = adaptive_drafts(live, batched_configured);
    if profile_drafts == 0 || !cohort_adaptive_k {
        return profile_drafts;
    }
    let max_k = (profile_drafts + 1).min(adaptive_max_k);
    let optimal_k = spark_model::layers::qwen4_verify_lanes::optimal_cohort_k_for_max_rows(
        live,
        ewma_p_step,
        max_k,
        spark_model::layers::qwen4_verify_lanes::adaptive_max_frame_rows(),
    );
    optimal_k.saturating_sub(1).clamp(1, profile_drafts)
}

/// Resolve the cohort width ceiling (`drafts` per sequence) for `live` active
/// speculating sequences using `controller`'s hysteresis-stabilized GB10
/// roofline width selection and burst warm-start.
///
/// - Solo `live <= 1` keeps the full `configured` draft width (e.g. `6` -> `K=7`)
///   and marks the cohort idle so the next concurrent burst warm-starts cleanly.
/// - Batched `live >= 2` is bounded by `<= ADAPTIVE_MAX_K - 1` (up to `K=8`
///   under adaptive-K, or `K=5` when adaptive-K is off) and the single-frame
///   `128 / live` row cap (`live = 2..=16` allows up to `K=7` with
///   `--num-drafts 6`, `16 * 7 = 112 <= 128`).
/// - Explicit `ATLAS_QWEN4_BATCHED_K5_LADDER` overrides take precedence.
/// - Otherwise, when `ATLAS_QWEN4_BATCHED_ADAPTIVE_K=1` and
///   `ATLAS_QWEN4_COHORT_ADAPTIVE_K` (default ON) are both active, selects `K`
///   via `controller.select_k(live, max_k)`.
pub(super) fn resolve_cohort_width_with_controller(
    batched_k5_armed: bool,
    adaptive_k: bool,
    cohort_adaptive_k: bool,
    explicit_ladder: Option<&[(usize, usize)]>,
    live: usize,
    configured: usize,
    controller: &CohortAcceptEwma,
) -> usize {
    if !batched_k5_armed {
        return configured;
    }
    if live <= 1 {
        controller.mark_solo(live);
        return configured;
    }
    if configured == 0 {
        return 0;
    }
    let adaptive_max_k = spark_model::layers::qwen4_verify_lanes::ADAPTIVE_MAX_K;
    let batched_configured = configured.min(adaptive_max_k.saturating_sub(1));
    if let Some(rungs) = explicit_ladder {
        return ladder_drafts(rungs, live, batched_configured);
    }
    if !adaptive_k {
        return batched_configured.min(FIXED_BATCHED_K5_DRAFTS);
    }
    let profile_drafts = adaptive_drafts(live, batched_configured);
    if profile_drafts == 0 || !cohort_adaptive_k {
        return profile_drafts;
    }
    let max_k = (profile_drafts + 1).min(adaptive_max_k);
    let selected_k = controller.select_k(live, max_k);
    selected_k.saturating_sub(1).clamp(1, profile_drafts)
}

fn resolve_effective_depth(
    proposer_is_dflash: bool,
    batched_k5_armed: bool,
    _adaptive_k: bool,
    live: usize,
    width_capped: usize,
    stochastic_depth: Option<usize>,
) -> usize {
    let depth = if proposer_is_dflash {
        width_capped
    } else {
        stochastic_depth.unwrap_or(width_capped)
    };
    // Batched cohort mode (`live >= 2`) is governed uniformly by the cohort
    // roofline / ladder / fixed-K5 width (`width_capped`), never by
    // single-sequence `adaptive_depth::stochastic_depth`, while solo `live == 1`
    // retains full single-sequence adaptive-depth parity.
    if batched_k5_armed && live >= 2 {
        width_capped
    } else {
        depth
    }
}

/// True when the current scheduler step's active cohort includes at least one
/// sampled sequence (`temperature > 0.0`). Defaults to `true` so standalone
/// calls and unit tests retain ragged width behavior; greedy-only cohorts
/// (`temperature == 0.0`) route through zero-sync uniform `argmax_bf16_rows`
/// + `CohortAcceptEwma` widths.
static STEP_COHORT_SAMPLED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(true);

#[inline]
pub(super) fn set_step_cohort_sampled(any_sampled: bool) {
    STEP_COHORT_SAMPLED.store(any_sampled, std::sync::atomic::Ordering::Relaxed);
}

#[inline]
fn step_cohort_sampled() -> bool {
    STEP_COHORT_SAMPLED.load(std::sync::atomic::Ordering::Relaxed)
}

/// Draft width for this scheduler step: the online-EWMA roofline adaptive width
/// (or explicit ladder rung) for `live >= 2` when batched native MTP is armed,
/// and full single-sequence configured + adaptive depth for solo `live <= 1`.
pub(super) fn effective_num_drafts(model: &dyn Model, live: usize, configured: usize) -> usize {
    super::adaptive_depth::set_step_live_seqs(live);
    // FLUX_SPEC_VERIFY=stochastic adaptive depth applies on every route,
    // including C=1 (not armed): C=1 dispatch picks K2/K3/K4/generic from
    // the actual draft count, so a shorter chain is always verifiable.
    let batched_armed = model.qwen4_batched_k5_armed();
    let adaptive_k = adaptive_selected();
    let proposer_is_dflash = model.proposer_is_dflash();
    // When ragged per-sequence `p_cum` compaction is active (TP=1, sampled
    // cohorts), let the cohort draft ceiling stay at `profile_drafts` (up to
    // `max_k - 1`, e.g. 6 drafts -> `K_s` up to 7 across `C=2..16`) so
    // confident streams expand automatically while uncertain streams stop
    // early inside `propose_batch`. Greedy-only cohorts (`T=0`) and explicit
    // `ATLAS_QWEN4_BATCHED_RAGGED_MTP=0` use `CohortAcceptEwma` + zero-sync
    // uniform `argmax_bf16_rows`. DFlash2 emits its full γ block in one
    // forward pass, so bypass `CohortAcceptEwma` narrowing and keep full
    // `min(configured, adaptive_verify_width(live) - 1)` (γ=7, K=8 across
    // C=2..=16 -> 16 × 8 = 128 rows).
    let ragged_active =
        adaptive_k && batched_ragged_enabled() && !model.is_ep() && step_cohort_sampled();
    let width_capped = resolve_cohort_width_with_controller(
        batched_armed,
        adaptive_k,
        cohort_adaptive_k_enabled() && !ragged_active && !proposer_is_dflash,
        ladder(),
        live,
        configured,
        &COHORT_ACCEPT_EWMA,
    );
    // FLUX_SPEC_VERIFY=stochastic: acceptance-driven depth (None when off).
    // DFlash emits an entire block in a single forward pass, so bypass MTP
    // adaptive_depth truncation and keep the full configured block width.
    let stochastic = if proposer_is_dflash {
        None
    } else {
        super::adaptive_depth::stochastic_depth(width_capped)
    };
    resolve_effective_depth(
        proposer_is_dflash,
        batched_armed,
        adaptive_k,
        live,
        width_capped,
        stochastic,
    )
}

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
fn stage_batched_gumbel_frame_if_eligible(
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
        super::stochastic_verify_walk::spec_verify_policy(),
    )
}

#[allow(clippy::too_many_arguments)]
fn stage_batched_gumbel_frame_if_eligible_with_policy(
    model: &dyn Model,
    active: &[ActiveSeq],
    idxs: &[usize],
    k_per_seq: &[usize],
    row_bases: &[usize],
    total_rows: usize,
    think: &ThinkSpecCtx<'_>,
    policy: super::stochastic_verify_walk::SpecVerifyPolicy,
) -> Option<Vec<bool>> {
    use super::spec_policy_accept::{
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
struct PresampledGumbelFrame {
    tokens: Vec<u32>,
    seq_offsets: Vec<Option<usize>>,
}

/// Consume per-sequence in-graph GPU Gumbel-Max tokens from the unified verify
/// tail (zero extra launches or syncs, even in mixed cohorts where some
/// sequences ran `temp == 0` greedy or `temp == -1` host-policy skip), or fall
/// back to ONE batched `Model::sample_gumbel_verify_rows_batched` call over the
/// eligible sequences when dedup or an unsupported backend bypassed in-graph
/// sampling.
#[allow(clippy::too_many_arguments)]
fn presample_batched_gumbel_frame_ragged(
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
        super::stochastic_verify_walk::spec_verify_policy(),
    )
}

#[cfg(test)]
fn presample_batched_gumbel_frame_with_policy(
    model: &dyn Model,
    active: &[ActiveSeq],
    idxs: &[usize],
    k_verify: usize,
    think: &ThinkSpecCtx<'_>,
    policy: super::stochastic_verify_walk::SpecVerifyPolicy,
) -> anyhow::Result<Option<PresampledGumbelFrame>> {
    let k_per_seq = vec![k_verify; idxs.len()];
    let row_bases: Vec<usize> = (0..idxs.len()).map(|s| s * k_verify).collect();
    presample_batched_gumbel_frame_ragged_with_policy(
        model, active, idxs, &k_per_seq, &row_bases, think, None, policy,
    )
}

#[allow(clippy::too_many_arguments)]
fn presample_batched_gumbel_frame_ragged_with_policy(
    model: &dyn Model,
    active: &[ActiveSeq],
    idxs: &[usize],
    k_per_seq: &[usize],
    row_bases: &[usize],
    think: &ThinkSpecCtx<'_>,
    seq_in_graph_gumbel: Option<&[bool]>,
    policy: super::stochastic_verify_walk::SpecVerifyPolicy,
) -> anyhow::Result<Option<PresampledGumbelFrame>> {
    if policy != super::stochastic_verify_walk::SpecVerifyPolicy::Gumbel {
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
        if super::spec_policy_accept::batched_gumbel_seq_eligible(&active[idx], k_s, think) {
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
            super::spec_policy_accept::log_batched_gumbel_engaged_once(
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
        let seed = super::stochastic_verify_walk::resolve_seq_gumbel_seed(a);
        let base_pos = super::stochastic_verify_walk::seq_abs_pos(a);
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
    super::spec_policy_accept::log_batched_gumbel_engaged_once(reqs.len(), eligible_seqs);
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
fn stage_batched_rejection_frame_if_eligible(
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
        super::stochastic_verify_walk::spec_verify_policy(),
    )
}

#[allow(clippy::too_many_arguments)]
fn stage_batched_rejection_frame_if_eligible_with_policy(
    model: &dyn Model,
    active: &[ActiveSeq],
    idxs: &[usize],
    drafts_per_seq: &[Vec<u32>],
    k_per_seq: &[usize],
    row_bases: &[usize],
    total_rows: usize,
    think: &ThinkSpecCtx<'_>,
    policy: super::stochastic_verify_walk::SpecVerifyPolicy,
) -> Option<Vec<bool>> {
    use super::spec_policy_accept::{
        RejectionTailRowMode, classify_rejection_tail_row_mode, push_rejection_tail_row_specs,
    };
    if policy != super::stochastic_verify_walk::SpecVerifyPolicy::Rejection
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
struct PresampledRejectionFrame {
    words: Vec<spark_model::PackedRejectionWord>,
    seq_offsets: Vec<Option<usize>>,
}

/// Consume per-sequence in-graph [`spark_model::PackedRejectionWord`] words
/// from the unified verify suffix tail (zero extra launches or syncs), or fall
/// back to ONE batched [`Model::sample_rejection_verify_rows_batched`] launch
/// over ONLY the eligible sequences when eager verify bypassed in-graph capture.
#[allow(clippy::too_many_arguments)]
fn presample_batched_rejection_frame_ragged(
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
        super::stochastic_verify_walk::spec_verify_policy(),
    )
}

#[allow(clippy::too_many_arguments)]
fn presample_batched_rejection_frame_ragged_with_policy(
    model: &dyn Model,
    active: &[ActiveSeq],
    idxs: &[usize],
    drafts_per_seq: &[Vec<u32>],
    k_per_seq: &[usize],
    row_bases: &[usize],
    think: &ThinkSpecCtx<'_>,
    seq_in_graph_rejection: Option<&[bool]>,
    policy: super::stochastic_verify_walk::SpecVerifyPolicy,
) -> anyhow::Result<Option<PresampledRejectionFrame>> {
    // Always drain any in-graph buffer published by the immediately preceding
    // verify forward so packed words can never leak across steps.
    let in_graph_words = model.take_in_graph_rejection_verify_words();
    if policy != super::stochastic_verify_walk::SpecVerifyPolicy::Rejection
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
        if super::spec_policy_accept::batched_rejection_seq_eligible(&active[idx], k_s, think) {
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
            super::spec_policy_accept::log_batched_rejection_engaged_once(
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
        super::spec_policy_accept::push_rejection_tail_row_specs(
            &mut reqs,
            &active[idx],
            &drafts_per_seq[s],
            row_bases[s],
            k_per_seq[s],
            super::spec_policy_accept::RejectionTailRowMode::Rejection,
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
    super::spec_policy_accept::log_batched_rejection_engaged_once(reqs.len(), eligible_seqs);
    Ok(Some(PresampledRejectionFrame { words, seq_offsets }))
}

/// `ATLAS_QWEN4_BATCHED_ADAPTIVE_K=1`, parsed strictly once outside unit tests.
pub(super) fn adaptive_selected() -> bool {
    let read_live = || match spark_model::layers::qwen4_verify_lanes::adaptive_k_selected() {
        Ok(true) => {
            match spark_model::layers::qwen4_verify_lanes::adaptive_verify_profile() {
                    Ok(spark_model::layers::qwen4_verify_lanes::AdaptiveVerifyProfile::K4C7C16) => {
                        tracing::info!(
                            "QWEN4_BATCHED_ADAPTIVE_K_ENGAGED profile=k4_c7_c16 widths=C1-C6:K5,C7-C32:K4,C33-C42:K3,C43-C128:K2"
                        );
                    }
                    Ok(
                        spark_model::layers::qwen4_verify_lanes::AdaptiveVerifyProfile::SingleSweepDefault,
                    ) => {
                        tracing::info!(
                            max_k = spark_model::layers::qwen4_verify_lanes::ADAPTIVE_MAX_K,
                            max_frame_rows =
                                spark_model::layers::qwen4_verify_lanes::adaptive_max_frame_rows(),
                            "QWEN4_BATCHED_ADAPTIVE_K_ENGAGED profile=single_frame_adaptive"
                        );
                    }
                    Err(error) => {
                        tracing::warn!(
                            "{error:#}; falling back to default adaptive verify width"
                        );
                    }
                }
            true
        }
        Ok(false) => false,
        Err(error) => {
            tracing::warn!("{error:#}; keeping the fixed verify width");
            false
        }
    };
    #[cfg(not(test))]
    {
        static ADAPTIVE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ADAPTIVE.get_or_init(read_live)
    }
    #[cfg(test)]
    {
        read_live()
    }
}

/// Cached [`AdaptiveVerifyProfile`] so `adaptive_drafts` avoids an uncached
/// `std::env::var("ATLAS_QWEN4_ADAPTIVE_K4_C7_C16")` syscall on every step.
fn adaptive_profile_cached()
-> Option<spark_model::layers::qwen4_verify_lanes::AdaptiveVerifyProfile> {
    let read_live = || match spark_model::layers::qwen4_verify_lanes::adaptive_verify_profile() {
        Ok(profile) => Some(profile),
        Err(error) => {
            tracing::warn!("{error:#}; falling back to default adaptive verify width");
            None
        }
    };
    #[cfg(not(test))]
    {
        static PROFILE: std::sync::OnceLock<
            Option<spark_model::layers::qwen4_verify_lanes::AdaptiveVerifyProfile>,
        > = std::sync::OnceLock::new();
        *PROFILE.get_or_init(read_live)
    }
    #[cfg(test)]
    {
        read_live()
    }
}

/// Drafts per sequence at `live` under an explicit single-frame row budget
/// `max_frame_rows`, never above `configured`.
pub(super) fn adaptive_drafts_with_max_rows(
    live: usize,
    configured: usize,
    max_frame_rows: usize,
) -> usize {
    let width = match adaptive_profile_cached() {
        Some(spark_model::layers::qwen4_verify_lanes::AdaptiveVerifyProfile::K4C7C16) => {
            spark_model::layers::qwen4_verify_lanes::adaptive_verify_width_for_profile_and_max_rows(
                live,
                spark_model::layers::qwen4_verify_lanes::AdaptiveVerifyProfile::K4C7C16,
                max_frame_rows,
            )
        }
        Some(
            spark_model::layers::qwen4_verify_lanes::AdaptiveVerifyProfile::SingleSweepDefault,
        )
        | None => spark_model::layers::qwen4_verify_lanes::adaptive_verify_width_for_max_rows(
            live,
            max_frame_rows,
        ),
    };
    width
        .map_or(1, |k| k.saturating_sub(1).max(1))
        .min(configured)
}

/// Drafts per sequence at `live` under the active GPU's single-frame adaptive
/// ceiling (`128 / live` on GB10 `<= 64` SMs; `256 / live` on `> 64`-SM GPUs,
/// clamped to `ADAPTIVE_MIN_K..=ADAPTIVE_MAX_K`), or the explicit opt-in
/// `ATLAS_QWEN4_ADAPTIVE_K4_C7_C16=1` (`K4C7C16`) profile, never above
/// `configured`. C129+ has no admitted verify width; startup refuses adaptive
/// servers wider than C128, so the one-draft floor is only defensive.
pub(super) fn adaptive_drafts(live: usize, configured: usize) -> usize {
    adaptive_drafts_with_max_rows(
        live,
        configured,
        spark_model::layers::qwen4_verify_lanes::adaptive_max_frame_rows(),
    )
}

/// Shorten every flat pending chain longer than `num_drafts` to the current
/// width. The proposer already wrote rows for the whole chain; dropping its
/// tail is the same outcome as an MTP confidence-gate truncation, and
/// `after_verify` trims from the rows actually computed. Trees and sampled
/// proposals are left untouched (they never batch).
pub(super) fn narrow_pending_drafts(active: &mut [ActiveSeq], idxs: &[usize], num_drafts: usize) {
    for &idx in idxs {
        let a = &mut active[idx];
        if a.pending_tree_payload.is_none()
            && a.pending_dflash_sampled_proposal.is_none()
            && a.pending_drafts.len() > num_drafts
        {
            let target = num_drafts.max(1);
            a.pending_drafts.truncate(target);
            super::stochastic_verify_walk::truncate_seq_rejection_q(a, target);
        }
    }
}

/// Optional override for non-greedy (`T > 0`) DFlash2 draft width under
/// `FLUX_SPEC_VERIFY=gumbel|rejection`: `1..=7` caps `num_drafts` to
/// `configured.min(v)`, `0` keeps full `configured` without narrowing, and
/// unset uses the measured GB10 cohort/policy defaults (`7` for `C <= 8`, `5`
/// for `C > 8` under `Rejection`; `6` for `C <= 2`, `5` for `C <= 4`, `4` for
/// `C > 4` under `Gumbel`).
const DFLASH_GUMBEL_GAMMA_ENV: &str = "ATLAS_QWEN4_DFLASH_GUMBEL_GAMMA";

/// Maximum override value admitted for [`DFLASH_GUMBEL_GAMMA_ENV`] (block-8
/// DFlash2 carries at most 7 draft slots).
const DFLASH_GUMBEL_GAMMA_MAX: usize = 7;

/// Parse [`DFLASH_GUMBEL_GAMMA_ENV`]: `Some(0)` disables narrowing,
/// `Some(1..=7)` sets an explicit cap, and `None` selects the default profile.
pub(super) fn parse_dflash_gumbel_gamma_override(raw: Option<&str>) -> Option<usize> {
    let raw = raw?.trim();
    if raw.is_empty() {
        return None;
    }
    match raw.parse::<usize>() {
        Ok(gamma) if gamma <= DFLASH_GUMBEL_GAMMA_MAX => Some(gamma),
        _ => {
            tracing::warn!(
                "{DFLASH_GUMBEL_GAMMA_ENV}={raw:?} invalid (want 0 or 1..={DFLASH_GUMBEL_GAMMA_MAX}); using default"
            );
            None
        }
    }
}

fn dflash_gumbel_gamma_override() -> Option<usize> {
    static OVERRIDE: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
    *OVERRIDE.get_or_init(|| {
        parse_dflash_gumbel_gamma_override(std::env::var(DFLASH_GUMBEL_GAMMA_ENV).ok().as_deref())
    })
}

/// SM count threshold above which DFlash2 draft-width, `p_cum`, and ragged
/// bucket defaults target high-SM discrete Blackwell GPUs (SM120/SM103) while
/// preserving exact GB10 (`sm_count <= 64`) policies.
const DFLASH_HIGH_SM_THRESHOLD: u32 = 64;

/// Pure SM-aware resolver for DFlash2 `T > 0` Gumbel / GPU rejection draft width.
///
/// Narrows ONLY when `proposer_is_dflash`, `policy` is `Gumbel` or `Rejection`,
/// and `all_non_greedy` are all true; any greedy sequence in the cohort (or any
/// non-DFlash / Argmax / Stochastic route) keeps full `configured`.
///
/// Under `SpecVerifyPolicy::Rejection`:
/// - GB10 (`sm_count <= 64`): keeps `gamma = 7` for `C <= 8` and caps `C > 8` at `gamma = 5`.
/// - High-SM Blackwell (`sm_count > 64`): preserves full `gamma = 7` (`K = 8`)
///   through `C <= 32` (`32 * 8 = 256 <= QWEN4_CROSS_SEQ_WIDE_MAX_ROWS`).
#[must_use]
pub(super) const fn resolve_dflash_gumbel_num_drafts_for_sm(
    proposer_is_dflash: bool,
    policy: super::stochastic_verify_walk::SpecVerifyPolicy,
    all_non_greedy: bool,
    live: usize,
    configured: usize,
    env_override: Option<usize>,
    sm_count: u32,
) -> usize {
    if !proposer_is_dflash
        || !matches!(
            policy,
            super::stochastic_verify_walk::SpecVerifyPolicy::Gumbel
                | super::stochastic_verify_walk::SpecVerifyPolicy::Rejection
        )
        || !all_non_greedy
        || configured == 0
    {
        return configured;
    }
    match env_override {
        Some(0) => configured,
        Some(gamma) => {
            if configured < gamma {
                configured
            } else {
                gamma
            }
        }
        None => match policy {
            super::stochastic_verify_walk::SpecVerifyPolicy::Rejection => {
                let cap = if sm_count > DFLASH_HIGH_SM_THRESHOLD {
                    match live {
                        0..=32 => 7,
                        _ => 5,
                    }
                } else {
                    match live {
                        0..=8 => 7,
                        _ => 5,
                    }
                };
                if configured < cap { configured } else { cap }
            }
            // Coupled Gumbel-Max peaks at wider `gamma = 6..5` on small cohorts
            // (`C <= 4`) and narrows to `gamma = 4` for `C > 4` to avoid
            // bandwidth-bound MoE expert waste.
            super::stochastic_verify_walk::SpecVerifyPolicy::Gumbel => {
                let cap = match live {
                    0..=2 => 6,
                    3..=4 => 5,
                    _ => 4,
                };
                if configured < cap { configured } else { cap }
            }
            _ => configured,
        },
    }
}

/// Pure resolver for DFlash2 `T > 0` Gumbel / GPU rejection draft width on the
/// active device's SM count.
#[must_use]
pub(super) fn resolve_dflash_gumbel_num_drafts(
    proposer_is_dflash: bool,
    policy: super::stochastic_verify_walk::SpecVerifyPolicy,
    all_non_greedy: bool,
    live: usize,
    configured: usize,
    env_override: Option<usize>,
) -> usize {
    resolve_dflash_gumbel_num_drafts_for_sm(
        proposer_is_dflash,
        policy,
        all_non_greedy,
        live,
        configured,
        env_override,
        atlas_core::device::active_sm_count(),
    )
}

/// Resolve DFlash2 `T > 0` Gumbel draft width for `live` active sequences
/// (used at both `C = 1` in `verify_dflash_step` and `C >= 2` here).
#[must_use]
pub(super) fn dflash_gumbel_num_drafts(
    model: &dyn Model,
    all_non_greedy: bool,
    live: usize,
    configured: usize,
) -> usize {
    resolve_dflash_gumbel_num_drafts(
        model.proposer_is_dflash(),
        super::stochastic_verify_walk::spec_verify_policy(),
        all_non_greedy,
        live,
        configured,
        dflash_gumbel_gamma_override(),
    )
}

/// Count non-finished active sequences and report whether every live sequence
/// is non-greedy (`temperature > 0.0` and finite).
#[must_use]
fn cohort_live_and_all_non_greedy(active: &[ActiveSeq]) -> (usize, bool) {
    let mut live = 0usize;
    let mut all_non_greedy = true;
    for a in active {
        if !a.finished {
            live += 1;
            if !a.temperature.is_finite() || a.temperature <= 0.0 {
                all_non_greedy = false;
            }
        }
    }
    (live, live > 0 && all_non_greedy)
}

/// Resolve the cohort draft width for `active`, keeping full `configured`
/// whenever any live sequence is greedy (`temperature <= 0.0`).
#[must_use]
pub(super) fn dflash_gumbel_cohort_num_drafts(
    model: &dyn Model,
    active: &[ActiveSeq],
    configured: usize,
) -> usize {
    let (live, all_non_greedy) = cohort_live_and_all_non_greedy(active);
    dflash_gumbel_num_drafts(model, all_non_greedy, live, configured)
}

/// Optional cumulative selector-probability truncation threshold for DFlash2
/// (`ATLAS_QWEN4_DFLASH_PCUM_THRESH`): `<= 0.0` disables truncation;
/// `(0.0, 1.0]` sets an explicit threshold; unset uses policy- and
/// cohort-calibrated defaults (`0.55/0.58/0.58/0.58` across `C <= 2/4/8/>8`
/// for lossless `max_z q_t(z)` `Rejection` `T > 0`; `0.05/0.06/0.10/0.13`
/// across `C <= 2/4/8/>8` for `Gumbel` `T > 0`; `0.25` for greedy `T == 0`).
const DFLASH_PCUM_THRESH_ENV: &str = "ATLAS_QWEN4_DFLASH_PCUM_THRESH";
/// Calibrated default cumulative selector thresholds for non-greedy (`T > 0`)
/// DFlash2 frames under `SpecVerifyPolicy::Rejection` by cohort size
/// (`C <= 2`, `C <= 4`, `C <= 8`, `C > 8`) on GB10 (`sm_count <= 64`),
/// calibrated for prefix-only `out_max_prob[t] = max_z q_t(z | x_{<t})`
/// (`TV = 0` lossless maximal coupling) with upward bucket alignment.
const DFLASH_PCUM_DEFAULT_REJECTION_C2: f32 = 0.55;
const DFLASH_PCUM_DEFAULT_REJECTION_C4: f32 = 0.58;
const DFLASH_PCUM_DEFAULT_REJECTION_C8: f32 = 0.58;
const DFLASH_PCUM_DEFAULT_REJECTION_HIGH_C: f32 = 0.58;
/// Default `SpecVerifyPolicy::Rejection` (`T > 0`) `p_cum` thresholds on
/// high-SM discrete Blackwell GPUs (`sm_count > 64`): disable truncation
/// (`0.0`, exact fixed `K = 8`, single steady CUDA graph, `TV = 0`) through
/// `C <= 24`, and apply light `0.15` truncation above `C > 24`.
const DFLASH_PCUM_DEFAULT_REJECTION_HIGH_SM_C24: f32 = 0.0;
const DFLASH_PCUM_DEFAULT_REJECTION_HIGH_SM_WIDE: f32 = 0.15;
/// Calibrated default cumulative selector thresholds for non-greedy (`T > 0`)
/// DFlash2 frames under `SpecVerifyPolicy::Gumbel` by cohort size
/// (`C <= 2`, `C <= 4`, `C <= 8`, `C > 8`), avoiding excessive draft truncation
/// while bounding high-concurrency MoE expert waste (`0-ULP` exact output).
const DFLASH_PCUM_DEFAULT_GUMBEL_C2: f32 = 0.05;
const DFLASH_PCUM_DEFAULT_GUMBEL_C4: f32 = 0.06;
const DFLASH_PCUM_DEFAULT_GUMBEL_C8: f32 = 0.10;
const DFLASH_PCUM_DEFAULT_GUMBEL_HIGH_C: f32 = 0.13;
/// Calibrated default cumulative selector threshold for greedy (`T == 0`)
/// DFlash2 frames (keeps `K = 8` on high-acceptance code, narrows uncertain prose).
const DFLASH_PCUM_DEFAULT_GREEDY: f32 = 0.25;

#[must_use]
const fn dflash_pcum_default_for_cohort_with_policy_for_sm(
    temperature: f32,
    cohort_size: usize,
    policy: super::stochastic_verify_walk::SpecVerifyPolicy,
    sm_count: u32,
) -> f32 {
    if temperature.is_finite() && temperature > 0.0 {
        match policy {
            super::stochastic_verify_walk::SpecVerifyPolicy::Rejection => {
                if sm_count > DFLASH_HIGH_SM_THRESHOLD {
                    match cohort_size {
                        0..=24 => DFLASH_PCUM_DEFAULT_REJECTION_HIGH_SM_C24,
                        _ => DFLASH_PCUM_DEFAULT_REJECTION_HIGH_SM_WIDE,
                    }
                } else {
                    match cohort_size {
                        0..=2 => DFLASH_PCUM_DEFAULT_REJECTION_C2,
                        3..=4 => DFLASH_PCUM_DEFAULT_REJECTION_C4,
                        5..=8 => DFLASH_PCUM_DEFAULT_REJECTION_C8,
                        _ => DFLASH_PCUM_DEFAULT_REJECTION_HIGH_C,
                    }
                }
            }
            _ => match cohort_size {
                0..=2 => DFLASH_PCUM_DEFAULT_GUMBEL_C2,
                3..=4 => DFLASH_PCUM_DEFAULT_GUMBEL_C4,
                5..=8 => DFLASH_PCUM_DEFAULT_GUMBEL_C8,
                _ => DFLASH_PCUM_DEFAULT_GUMBEL_HIGH_C,
            },
        }
    } else {
        DFLASH_PCUM_DEFAULT_GREEDY
    }
}

#[must_use]
fn dflash_pcum_default_for_cohort_with_policy(
    temperature: f32,
    cohort_size: usize,
    policy: super::stochastic_verify_walk::SpecVerifyPolicy,
) -> f32 {
    dflash_pcum_default_for_cohort_with_policy_for_sm(
        temperature,
        cohort_size,
        policy,
        atlas_core::device::active_sm_count(),
    )
}

/// Parse [`DFLASH_PCUM_THRESH_ENV`]: `Some(0.0)` disables truncation,
/// `Some(tau)` in `(0.0, 1.0]` overrides the default threshold, and `None`
/// selects the temperature-calibrated default.
pub(super) fn parse_dflash_pcum_thresh_override(raw: Option<&str>) -> Option<f32> {
    let raw = raw?.trim();
    if raw.is_empty() {
        return None;
    }
    match raw.parse::<f32>() {
        Ok(tau) if tau.is_finite() && tau <= 1.0 => Some(tau.max(0.0)),
        _ => {
            tracing::warn!(
                "{DFLASH_PCUM_THRESH_ENV}={raw:?} invalid (want finite float <= 1.0, 0 disables); using default"
            );
            None
        }
    }
}

fn dflash_pcum_thresh_override() -> Option<f32> {
    static OVERRIDE: std::sync::OnceLock<Option<f32>> = std::sync::OnceLock::new();
    *OVERRIDE.get_or_init(|| {
        parse_dflash_pcum_thresh_override(std::env::var(DFLASH_PCUM_THRESH_ENV).ok().as_deref())
    })
}

/// Pure kept-draft count (`1..=draft_len`) and stopping-frontier cumulative
/// confidence `frontier_pcum` under cumulative selector confidence
/// `p_cum = prod_{j=0..=i} probs[j]` against threshold `tau` resolved for
/// `(temperature, cohort_size, policy)`.
///
/// When `probs[j]` is prefix-only `max_z q_j(z | x_{<j})` (independent of the
/// sampled draft `x_j`), both `keep` and `frontier_pcum` depend solely on
/// `(x_0 .. x_{keep-1})` and NEVER inspect `x_{keep..}`. This makes forward
/// truncation and upward bucket extension strictly `F_t`-measurable (`TV = 0`
/// under maximal-coupling rejection sampling).
///
/// Draft `0` is always retained (`keep >= 1`, so `K_s >= 2`) to preserve
/// batched CUDA-graph and `KEEP_LATE` eligibility.
#[must_use]
pub(super) fn dflash_pcum_keep_len_and_frontier_for_cohort_with_policy(
    draft_len: usize,
    probs: &[f32],
    temperature: f32,
    cohort_size: usize,
    policy: super::stochastic_verify_walk::SpecVerifyPolicy,
    env_override: Option<f32>,
) -> (usize, f32) {
    if draft_len <= 1 || probs.is_empty() {
        return (draft_len, 1.0);
    }
    let tau = match env_override {
        Some(v) => v,
        None => dflash_pcum_default_for_cohort_with_policy(temperature, cohort_size, policy),
    };
    if !tau.is_finite() || tau <= 0.0 {
        return (draft_len, 1.0);
    }
    let n = draft_len.min(probs.len());
    let mut p_cum = 1.0f32;
    let mut keep = 1usize;
    for (i, &raw_p) in probs.iter().take(n).enumerate() {
        let p = if raw_p.is_finite() {
            raw_p.clamp(0.0, 1.0)
        } else {
            0.0
        };
        p_cum *= p;
        if i >= 1 {
            if p_cum >= tau {
                keep = i + 1;
            } else {
                return (keep, p_cum);
            }
        }
    }
    (keep, p_cum)
}

/// Pure kept-draft count (`1..=draft_len`) under cumulative selector confidence
/// `p_cum = prod_{j=0..=i} probs[j]` against threshold `tau` resolved for
/// `(temperature, cohort_size, policy)`.
///
/// Draft `0` is always retained (`keep >= 1`, so `K_s >= 2`) to preserve
/// batched CUDA-graph and `KEEP_LATE` eligibility.
#[must_use]
pub(super) fn dflash_pcum_keep_len_for_cohort_with_policy(
    draft_len: usize,
    probs: &[f32],
    temperature: f32,
    cohort_size: usize,
    policy: super::stochastic_verify_walk::SpecVerifyPolicy,
    env_override: Option<f32>,
) -> usize {
    dflash_pcum_keep_len_and_frontier_for_cohort_with_policy(
        draft_len,
        probs,
        temperature,
        cohort_size,
        policy,
        env_override,
    )
    .0
}

#[cfg_attr(not(test), allow(dead_code))]
#[must_use]
pub(super) fn dflash_pcum_keep_len_for_cohort(
    draft_len: usize,
    probs: &[f32],
    temperature: f32,
    cohort_size: usize,
    env_override: Option<f32>,
) -> usize {
    dflash_pcum_keep_len_for_cohort_with_policy(
        draft_len,
        probs,
        temperature,
        cohort_size,
        super::stochastic_verify_walk::SpecVerifyPolicy::Gumbel,
        env_override,
    )
}

#[cfg_attr(not(test), allow(dead_code))]
#[must_use]
pub(super) fn dflash_pcum_keep_len(
    draft_len: usize,
    probs: &[f32],
    temperature: f32,
    env_override: Option<f32>,
) -> usize {
    dflash_pcum_keep_len_for_cohort(draft_len, probs, temperature, 1, env_override)
}

#[must_use]
fn dflash_pcum_target_len_and_frontier(
    a: &ActiveSeq,
    is_dflash: bool,
    probs: &[f32],
    cohort_size: usize,
    policy: super::stochastic_verify_walk::SpecVerifyPolicy,
    env_override: Option<f32>,
) -> (usize, f32) {
    let before = a.pending_drafts.len();
    if !is_dflash
        || before <= 1
        || probs.is_empty()
        || a.draft_origin != DraftOrigin::Proposer
        || a.pending_tree_payload.is_some()
        || a.pending_dflash_sampled_proposal.is_some()
        || super::think_budget_nudge::close_allowed(a)
        // Under uniform (non-ragged) cohorts, `harmonize_batched_verify_cohort_widths`
        // trims streams down to `min_k`, which would retroactively drop a draft
        // below another stream's stopping point and bias stochastic Rejection.
        || (policy == super::stochastic_verify_walk::SpecVerifyPolicy::Rejection
            && a.temperature.is_finite()
            && a.temperature > 0.0
            && cohort_size >= 2
            && !batched_ragged_enabled())
    {
        return (before, 1.0);
    }
    dflash_pcum_keep_len_and_frontier_for_cohort_with_policy(
        before,
        probs,
        a.temperature,
        cohort_size,
        policy,
        env_override,
    )
}

#[must_use]
fn dflash_pcum_target_len(
    a: &ActiveSeq,
    is_dflash: bool,
    probs: &[f32],
    cohort_size: usize,
    policy: super::stochastic_verify_walk::SpecVerifyPolicy,
    env_override: Option<f32>,
) -> usize {
    dflash_pcum_target_len_and_frontier(a, is_dflash, probs, cohort_size, policy, env_override).0
}

fn log_dflash_pcum_truncation_once(before: usize, after: usize, temperature: f32) {
    static ENGAGED: std::sync::Once = std::sync::Once::new();
    ENGAGED.call_once(|| {
        tracing::info!(
            before,
            after,
            temperature,
            "DFLASH2_PCUM_TRUNCATION_ENGAGED"
        );
    });
}

/// Truncate freshly-proposed DFlash2 flat drafts using per-draft conditional
/// confidence probabilities from `DflashProposerState::last_draft_probs()`.
///
/// Lossless (`0-ULP`): target verification guarantees bit-identical output
/// tokens for any draft length `1..=7`. Never truncates retrieval chains,
/// forced-nudge chains, tree payloads, or probabilistic sampled-q frames.
pub(super) fn apply_dflash_pcum_truncation(a: &mut ActiveSeq, is_dflash: bool) {
    apply_dflash_pcum_truncation_for_cohort(a, is_dflash, 1);
}

pub(super) fn apply_dflash_pcum_truncation_for_cohort(
    a: &mut ActiveSeq,
    is_dflash: bool,
    cohort_size: usize,
) {
    if !is_dflash || a.pending_drafts.len() <= 1 {
        return;
    }
    let keep = {
        let a_ref: &ActiveSeq = a;
        let Some(probs) = a_ref
            .seq
            .proposer_state
            .as_ref()
            .and_then(|s| {
                s.as_any()
                    .downcast_ref::<spark_model::layers::dflash_head::DflashProposerState>()
            })
            .map(spark_model::layers::dflash_head::DflashProposerState::last_draft_probs)
        else {
            return;
        };
        dflash_pcum_target_len(
            a_ref,
            is_dflash,
            probs,
            cohort_size,
            super::stochastic_verify_walk::spec_verify_policy(),
            dflash_pcum_thresh_override(),
        )
    };
    let before = a.pending_drafts.len();
    if keep < before {
        a.pending_drafts.truncate(keep);
        super::stochastic_verify_walk::truncate_seq_rejection_q(a, keep);
        log_dflash_pcum_truncation_once(before, keep, a.temperature);
    }
}

#[cfg(test)]
fn apply_dflash_pcum_truncation_with_probs(
    a: &mut ActiveSeq,
    is_dflash: bool,
    probs: &[f32],
    env_override: Option<f32>,
) {
    let before = a.pending_drafts.len();
    let keep = dflash_pcum_target_len(
        a,
        is_dflash,
        probs,
        1,
        super::stochastic_verify_walk::SpecVerifyPolicy::Gumbel,
        env_override,
    );
    if keep < before {
        a.pending_drafts.truncate(keep);
    }
}

/// Graph-bucket row granularity for ragged DFlash2 batched-verify frames of
/// `b` sequences on a GPU with `sm_count` SMs.
///
/// GB10 (`sm_count <= 64`) preserves exact `2/4/8/16` buckets matching
/// `qwen4_mtp::ragged_row_bucket_step`. High-SM Blackwell (`sm_count > 64`)
/// widens coarse buckets at `b >= 12` (`16` for `12..=23`, `32` for `>= 24`)
/// so ragged `p_cum > 0` runs keep at most 3-4 suffix graphs while staying a
/// multiple of `verify_qwen4_batched::batched_verify_graph_capture_row_admitted`.
#[inline]
#[must_use]
const fn dflash_ragged_row_bucket_step_for_sm(b: usize, sm_count: u32) -> usize {
    if sm_count > DFLASH_HIGH_SM_THRESHOLD {
        match b {
            0..=1 => 1,
            2 => 2,
            3..=5 => 4,
            6..=11 => 8,
            12..=23 => 16,
            _ => 32,
        }
    } else {
        match b {
            0..=1 => 1,
            2 => 2,
            3..=5 => 4,
            6..=16 => 8,
            _ => 16,
        }
    }
}

/// Graph-bucket row granularity for ragged DFlash2 batched-verify frames of
/// `b` sequences on the active device.
#[inline]
#[must_use]
fn dflash_ragged_row_bucket_step(b: usize) -> usize {
    dflash_ragged_row_bucket_step_for_sm(b, atlas_core::device::active_sm_count())
}

/// Remaining-token lead margin below the cohort median (`rem < median - 8`)
/// that marks a stream as ahead for zero-cost bucket-rounding tie-breaking.
const DFLASH_BUCKET_TRIM_AHEAD_MARGIN: usize = 8;

/// Row-count tile cliff (`r = 56` is not a clean 16/32-row tile boundary for
/// `m32`/`m64` kernels and measured slower than `r = 64` at `C = 16`): when
/// `b >= 12` and bucket quantization lands on `56`, snap down by 8 rows to `48`
/// provided every lane preserves its `len >= 1` (`K >= 2`) floor.
const DFLASH_RAGGED_CLIFF_ROWS: usize = 56;
const DFLASH_RAGGED_CLIFF_MIN_SEQS: usize = 12;
const DFLASH_RAGGED_CLIFF_DROP_ROWS: usize = 8;

/// Compute how many draft rows to trim from a ragged cohort (`b` sequences,
/// `total_rows = sum(len + 1)`, `trimmable = sum(len.saturating_sub(1))`)
/// to hit the verify CUDA-graph bucket grid while skipping the `r = 56` cliff
/// at `b >= 12` (`56 -> 48`).
#[inline]
#[must_use]
fn ragged_bucket_trim_rows(b: usize, total_rows: usize, trimmable: usize) -> usize {
    let step = dflash_ragged_row_bucket_step(b);
    if step <= 1 {
        return 0;
    }
    let mut rem = total_rows % step;
    let quantized = total_rows - rem;
    if b >= DFLASH_RAGGED_CLIFF_MIN_SEQS
        && quantized == DFLASH_RAGGED_CLIFF_ROWS
        && trimmable >= rem + DFLASH_RAGGED_CLIFF_DROP_ROWS
    {
        rem += DFLASH_RAGGED_CLIFF_DROP_ROWS;
    }
    rem
}

/// Snap ragged draft lengths `lens` (`b >= 2`, `min != max`) so
/// `sum(lens[s] + 1)` aligns to [`dflash_ragged_row_bucket_step`] (and skips
/// the `r = 56` cliff down to `48` for `b >= 12`) by trimming the lowest
/// tail-`p_cum` extra draft token(s) first while keeping every `lens[s] >= 1`
/// (`K_s >= 2`). Uniform cohorts (`min == max`) are left untouched because
/// uniform frames are always graph-admitted.
pub(super) fn quantize_ragged_lens_to_graph_bucket(lens: &mut [usize], pcum_chains: &[Vec<f32>]) {
    quantize_ragged_lens_to_graph_bucket_with_remaining(lens, pcum_chains, None);
}

/// [`quantize_ragged_lens_to_graph_bucket`] with optional per-sequence
/// remaining token budgets for `D_bucket_trim_ahead_first` straggler-friendly
/// tie-breaking: when rounding down `1..step-1` rows to hit a graph bucket
/// boundary (or dropping 8 additional rows off the `r = 56` cliff at
/// `b >= 12`), streams ahead of the cohort median
/// (`rem < median.saturating_sub(8)`) are trimmed first (lowest `tail_p` among
/// ahead streams), falling back to lowest `tail_p` across all trimmable streams
/// while strictly preserving the `lens[s] >= 1` (`K_s >= 2`) floor.
pub(super) fn quantize_ragged_lens_to_graph_bucket_with_remaining(
    lens: &mut [usize],
    pcum_chains: &[Vec<f32>],
    remaining_tokens: Option<&[usize]>,
) {
    let b = lens.len();
    let step = dflash_ragged_row_bucket_step(b);
    if step <= 1 || b != pcum_chains.len() {
        return;
    }
    let min_len = lens.iter().copied().min().unwrap_or(0);
    let max_len = lens.iter().copied().max().unwrap_or(0);
    if min_len == max_len {
        return;
    }
    let total_rows: usize = lens.iter().map(|&l| l + 1).sum();
    let trimmable: usize = lens.iter().map(|&l| l.saturating_sub(1)).sum();
    let mut rem = ragged_bucket_trim_rows(b, total_rows, trimmable);
    if rem == 0 {
        return;
    }
    let ahead_threshold = remaining_tokens.filter(|rems| rems.len() == b).map(|rems| {
        let mut sorted = rems.to_vec();
        sorted.sort_unstable();
        sorted[b / 2].saturating_sub(DFLASH_BUCKET_TRIM_AHEAD_MARGIN)
    });
    while rem > 0 {
        let mut victim: Option<(usize, bool, f32, usize)> = None;
        for (s, &len) in lens.iter().enumerate() {
            if len <= 1 {
                continue;
            }
            let is_ahead = match (ahead_threshold, remaining_tokens) {
                (Some(threshold), Some(rems)) => rems[s] < threshold,
                _ => false,
            };
            let tail_p = pcum_chains[s].get(len - 1).copied().unwrap_or(1.0);
            let better = match victim {
                None => true,
                Some((best_s, best_ahead, best_p, best_len)) => {
                    (is_ahead && !best_ahead)
                        || (is_ahead == best_ahead
                            && (tail_p < best_p
                                || (tail_p == best_p
                                    && (len > best_len || (len == best_len && s > best_s)))))
                }
            };
            if better {
                victim = Some((s, is_ahead, tail_p, len));
            }
        }
        let Some((s, _, _, _)) = victim else {
            break;
        };
        lens[s] -= 1;
        rem -= 1;
    }
}

/// Compute how many draft rows to extend UPWARD from a ragged cohort (`b`
/// sequences, `total_rows = sum(len + 1)`, `slack = sum(max_len - len)`) to
/// hit the verify CUDA-graph bucket grid without adding a wasteful `+8` jump
/// off `r = 56 -> 64` (unlike downward `56 -> 48` which removes rows, at
/// `gamma = 5` on `C = 16` almost every step lands in `49..=56` rows so
/// jumping `+8` to `64` adds 8 low-confidence tail rows per round). Returns
/// `0` if `total_rows` is already aligned or `slack` cannot cover the upward
/// snap (never trims downward).
#[inline]
#[must_use]
fn ragged_bucket_extend_rows(b: usize, total_rows: usize, slack: usize) -> usize {
    let step = dflash_ragged_row_bucket_step(b);
    if step <= 1 {
        return 0;
    }
    let rem = total_rows % step;
    let add = if rem == 0 { 0 } else { step - rem };
    if slack < add { 0 } else { add }
}

/// Snap ragged `p_cum`-stopped draft lengths `lens` UPWARD (`lens[s] <= max_lens[s]`)
/// so `sum(lens[s] + 1)` aligns to [`dflash_ragged_row_bucket_step`] (without
/// adding a wasteful `+8` jump off `r = 56 -> 64`) using ONLY the
/// `F_{s, L_s}`-measurable stopping lengths `lens`, stopping-frontier
/// `frontier_pcums`, and pre-proposal `remaining_tokens`.
///
/// Because this helper NEVER trims below `lens[s]` and NEVER inspects any draft
/// token or probability at index `>= orig_lens[s]`, the event `{final_len[s] > t}`
/// stays strictly `F_{s, t}`-measurable for every stream `s` and draft slot `t`,
/// preserving exact `TV = 0` maximal-coupling rejection sampling at `C >= 2`.
pub(super) fn quantize_ragged_lens_upward_to_graph_bucket_with_remaining(
    lens: &mut [usize],
    max_lens: &[usize],
    frontier_pcums: &[f32],
    remaining_tokens: Option<&[usize]>,
) {
    let b = lens.len();
    let step = dflash_ragged_row_bucket_step(b);
    if step <= 1 || b != max_lens.len() || b != frontier_pcums.len() {
        return;
    }
    let min_len = lens.iter().copied().min().unwrap_or(0);
    let max_len = lens.iter().copied().max().unwrap_or(0);
    if min_len == max_len {
        return;
    }
    let total_rows: usize = lens.iter().map(|&l| l + 1).sum();
    let slack: usize = lens
        .iter()
        .zip(max_lens.iter())
        .map(|(&l, &cap)| cap.saturating_sub(l))
        .sum();
    let mut need_add = ragged_bucket_extend_rows(b, total_rows, slack);
    if need_add == 0 {
        return;
    }
    let ahead_threshold = remaining_tokens.filter(|rems| rems.len() == b).map(|rems| {
        let mut sorted = rems.to_vec();
        sorted.sort_unstable();
        sorted[b / 2].saturating_sub(DFLASH_BUCKET_TRIM_AHEAD_MARGIN)
    });
    let orig_lens = lens.to_vec();
    while need_add > 0 {
        let mut pick: Option<(usize, bool, usize, f32, usize)> = None;
        for (s, (&len, &cap)) in lens.iter().zip(max_lens.iter()).enumerate() {
            if len >= cap {
                continue;
            }
            // Prefer stragglers / behind streams (`!is_ahead`) so extra draft
            // rows go where remaining output work is highest.
            let is_behind = match (ahead_threshold, remaining_tokens) {
                (Some(threshold), Some(rems)) => rems[s] >= threshold,
                _ => false,
            };
            let added = len - orig_lens[s];
            let fp = frontier_pcums[s];
            let frontier = if fp.is_finite() { fp } else { 0.0 };
            let better = match pick {
                None => true,
                Some((best_s, best_behind, best_added, best_frontier, best_len)) => {
                    (is_behind && !best_behind)
                        || (is_behind == best_behind
                            && (added < best_added
                                || (added == best_added
                                    && (frontier > best_frontier
                                        || (frontier == best_frontier
                                            && (len < best_len
                                                || (len == best_len && s < best_s)))))))
                }
            };
            if better {
                pick = Some((s, is_behind, added, frontier, len));
            }
        }
        let Some((s, _, _, _, _)) = pick else {
            break;
        };
        lens[s] += 1;
        need_add -= 1;
    }
}

/// Apply `p_cum` stopping AND lossless upward graph-bucket quantization in a
/// single pass over freshly proposed `propose_idxs` under `SpecVerifyPolicy::Rejection`,
/// truncating `pending_drafts` and GPU rejection `q` buffers only once final
/// bucket-aligned lengths are known.
fn apply_dflash_pcum_and_upward_bucket_quantize_cohort(
    active: &mut [ActiveSeq],
    propose_idxs: &[usize],
    is_dflash: bool,
) {
    if !is_dflash {
        return;
    }
    let policy = super::stochastic_verify_walk::spec_verify_policy();
    let env_override = dflash_pcum_thresh_override();
    let mut lane_idxs: Vec<usize> = Vec::with_capacity(propose_idxs.len());
    let mut max_lens: Vec<usize> = Vec::with_capacity(propose_idxs.len());
    let mut lens: Vec<usize> = Vec::with_capacity(propose_idxs.len());
    let mut frontier_pcums: Vec<f32> = Vec::with_capacity(propose_idxs.len());
    let mut remaining_tokens: Vec<usize> = Vec::with_capacity(propose_idxs.len());
    let cohort_size = propose_idxs
        .iter()
        .filter(|&&idx| {
            active.get(idx).is_some_and(|a| {
                !a.finished
                    && !a.pending_drafts.is_empty()
                    && a.pending_tree_payload.is_none()
                    && a.pending_dflash_sampled_proposal.is_none()
            })
        })
        .count();
    for &idx in propose_idxs {
        let Some(a) = active.get(idx) else {
            continue;
        };
        if a.finished
            || a.pending_drafts.is_empty()
            || a.pending_tree_payload.is_some()
            || a.pending_dflash_sampled_proposal.is_some()
        {
            continue;
        }
        let before = a.pending_drafts.len();
        let (keep, frontier) = a
            .seq
            .proposer_state
            .as_ref()
            .and_then(|s| {
                s.as_any()
                    .downcast_ref::<spark_model::layers::dflash_head::DflashProposerState>()
            })
            .map(spark_model::layers::dflash_head::DflashProposerState::last_draft_probs)
            .map_or((before, 1.0), |probs| {
                dflash_pcum_target_len_and_frontier(
                    a,
                    is_dflash,
                    probs,
                    cohort_size,
                    policy,
                    env_override,
                )
            });
        let rem_budget = if a.max_output_tokens > 0 {
            a.remaining
                .min(a.max_output_tokens.saturating_sub(a.output_tokens.len()))
        } else {
            a.remaining
        };
        lane_idxs.push(idx);
        max_lens.push(before);
        lens.push(keep);
        frontier_pcums.push(frontier);
        remaining_tokens.push(rem_budget);
    }
    quantize_ragged_lens_upward_to_graph_bucket_with_remaining(
        &mut lens,
        &max_lens,
        &frontier_pcums,
        Some(&remaining_tokens),
    );
    for (&idx, &target_len) in lane_idxs.iter().zip(lens.iter()) {
        let a = &mut active[idx];
        let before = a.pending_drafts.len();
        if target_len < before {
            a.pending_drafts.truncate(target_len);
            super::stochastic_verify_walk::truncate_seq_rejection_q(a, target_len);
            log_dflash_pcum_truncation_once(before, target_len, a.temperature);
        }
    }
}

/// Quantize ragged DFlash2 `pending_drafts` across `idxs` onto the
/// `BATCHED_VERIFY_GRAPH` row bucket grid so suffix graphs never thrash.
fn quantize_dflash_ragged_drafts_to_graph_bucket(active: &mut [ActiveSeq], idxs: &[usize]) {
    if super::stochastic_verify_walk::spec_verify_policy()
        == super::stochastic_verify_walk::SpecVerifyPolicy::Rejection
        && idxs.iter().any(|&idx| {
            active.get(idx).is_some_and(|a| {
                !a.finished
                    && !a.pending_drafts.is_empty()
                    && a.temperature.is_finite()
                    && a.temperature > 0.0
            })
        })
    {
        // Under stochastic Rejection, steady-state cohorts are already aligned
        // UPWARD in `apply_dflash_pcum_and_upward_bucket_quantize_cohort` prior
        // to truncation. Never retroactively trim below a stream's `p_cum`
        // stopping point on join/exit ticks (`TV = 0` invariant).
        return;
    }
    let mut lane_stack = [0usize; spark_model::layer::QWEN4_CROSS_SEQ_MAX_ROWS];
    let mut lens_stack = [0usize; spark_model::layer::QWEN4_CROSS_SEQ_MAX_ROWS];
    let lane_heap;
    let mut lens_heap;
    let mut min_len = usize::MAX;
    let mut max_len = 0usize;
    let mut total_rows = 0usize;
    let (lane_idxs, lens): (&[usize], &mut [usize]) = if idxs.len() <= lane_stack.len() {
        let mut b = 0usize;
        for &idx in idxs {
            let Some(a) = active.get(idx) else {
                continue;
            };
            if a.finished
                || a.pending_drafts.is_empty()
                || a.pending_tree_payload.is_some()
                || a.pending_dflash_sampled_proposal.is_some()
            {
                continue;
            }
            let len = a.pending_drafts.len();
            min_len = min_len.min(len);
            max_len = max_len.max(len);
            total_rows += len + 1;
            lane_stack[b] = idx;
            lens_stack[b] = len;
            b += 1;
        }
        (&lane_stack[..b], &mut lens_stack[..b])
    } else {
        let mut lh = Vec::with_capacity(idxs.len());
        let mut nh = Vec::with_capacity(idxs.len());
        for &idx in idxs {
            let Some(a) = active.get(idx) else {
                continue;
            };
            if a.finished
                || a.pending_drafts.is_empty()
                || a.pending_tree_payload.is_some()
                || a.pending_dflash_sampled_proposal.is_some()
            {
                continue;
            }
            let len = a.pending_drafts.len();
            min_len = min_len.min(len);
            max_len = max_len.max(len);
            total_rows += len + 1;
            lh.push(idx);
            nh.push(len);
        }
        lane_heap = lh;
        lens_heap = nh;
        (&lane_heap, &mut lens_heap)
    };
    let b = lane_idxs.len();
    let trimmable: usize = lens.iter().map(|&l| l.saturating_sub(1)).sum();
    if min_len == max_len || ragged_bucket_trim_rows(b, total_rows, trimmable) == 0 {
        return;
    }
    let mut pcum_chains: Vec<Vec<f32>> = Vec::with_capacity(b);
    let mut remaining_tokens: Vec<usize> = Vec::with_capacity(b);
    for (&idx, &len) in lane_idxs.iter().zip(lens.iter()) {
        let a = &active[idx];
        let rem_budget = if a.max_output_tokens > 0 {
            a.remaining
                .min(a.max_output_tokens.saturating_sub(a.output_tokens.len()))
        } else {
            a.remaining
        };
        remaining_tokens.push(rem_budget);
        let mut chain = Vec::with_capacity(len);
        if a.draft_origin == DraftOrigin::Proposer
            && !super::think_budget_nudge::close_allowed(a)
            && let Some(probs) = a
                .seq
                .proposer_state
                .as_ref()
                .and_then(|s| {
                    s.as_any()
                        .downcast_ref::<spark_model::layers::dflash_head::DflashProposerState>()
                })
                .map(spark_model::layers::dflash_head::DflashProposerState::last_draft_probs)
            && !probs.is_empty()
        {
            let mut p_cum = 1.0f32;
            for i in 0..len {
                let raw_p = probs.get(i).copied().unwrap_or(1.0);
                let p = if raw_p.is_finite() {
                    raw_p.clamp(0.0, 1.0)
                } else {
                    0.0
                };
                p_cum *= p;
                chain.push(p_cum);
            }
        } else {
            chain.resize(len, 1.0);
        }
        pcum_chains.push(chain);
    }
    quantize_ragged_lens_to_graph_bucket_with_remaining(
        lens,
        &pcum_chains,
        Some(&remaining_tokens),
    );
    for (&idx, &target_len) in lane_idxs.iter().zip(lens.iter()) {
        if active[idx].pending_drafts.len() > target_len {
            active[idx].pending_drafts.truncate(target_len);
            super::stochastic_verify_walk::truncate_seq_rejection_q(&active[idx], target_len);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        BatchedK5Candidate, BatchedK5Plan, CohortAcceptEwma, DraftDepthPolicy, adaptive_drafts,
        apply_dflash_pcum_truncation_with_probs, batched_k5_policy_required, batched_k5_width,
        batched_k5_width_for, choose_harmonized_width, cohort_live_and_all_non_greedy,
        dflash_pcum_keep_len, dflash_ragged_row_bucket_step, ladder_drafts, narrow_pending_drafts,
        parse_dflash_gumbel_gamma_override, parse_dflash_pcum_thresh_override, parse_ladder,
        plan_batched_k5, plan_batched_uniform, plan_ragged_cohort,
        quantize_ragged_lens_to_graph_bucket, quantize_ragged_lens_to_graph_bucket_with_remaining,
        resolve_cohort_width_capped, resolve_cohort_width_with_controller,
        resolve_dflash_gumbel_num_drafts, resolve_effective_depth,
    };

    #[test]
    fn non_thinking_qwen4_sequence_with_think_ended_is_eligible_for_batched_k5() {
        let mut a = crate::scheduler::test_fixtures::active_seq(
            crate::scheduler::ResponseSink::Blocking(None),
            1,
        );
        a.inside_thinking = false;
        a.think_ended = true;
        a.think_just_ended = false;
        let think = crate::scheduler::ThinkSpecCtx {
            enabled: false,
            code_fence_token: None,
            reflection_suppress_ids: &[],
            adaptive_sampling: false,
            honor_eos_inside_thinking: Some(false),
        };
        assert!(!batched_k5_policy_required(&a, &think));

        a.think_just_ended = true;
        assert!(!batched_k5_policy_required(&a, &think));
        a.think_just_ended = false;

        a.inside_thinking = true;
        assert!(batched_k5_policy_required(&a, &think));
        let think_enabled = crate::scheduler::ThinkSpecCtx {
            enabled: true,
            ..think
        };
        assert!(!batched_k5_policy_required(&a, &think_enabled));
    }

    #[test]
    fn ladder_parses_ascending_positive_rungs_only() {
        assert_eq!(
            parse_ladder("2:4,4:3,8:2,32:1").unwrap(),
            vec![(2, 4), (4, 3), (8, 2), (32, 1)]
        );
        for bad in [
            "", "2", "2:0", "0:4", "4:3,2:4", "4:3,4:2", "a:1", "2:b", "2:4,",
        ] {
            assert!(parse_ladder(bad).is_err(), "{bad:?}");
        }
    }

    #[test]
    fn ladder_picks_the_first_covering_rung_and_never_exceeds_configured() {
        let ladder = parse_ladder("2:4,4:3,8:2,32:1").unwrap();
        let at = |c| ladder_drafts(&ladder, c, 4);
        assert_eq!(
            (at(1), at(2), at(3), at(4), at(5), at(8), at(9), at(64)),
            (4, 4, 3, 3, 2, 2, 1, 1)
        );
        assert_eq!(
            ladder_drafts(&ladder, 1, 2),
            2,
            "configured width is a ceiling"
        );
    }

    fn neutral(draft_len: usize) -> BatchedK5Candidate {
        BatchedK5Candidate {
            draft_len,
            configured_num_drafts: 4,
            grammar_active: false,
            finished: false,
            policy_required: false,
            has_tree: false,
            has_sampled: false,
        }
    }

    #[test]
    fn neutral_flat_chains_batch_at_their_own_width() {
        assert_eq!(batched_k5_width(neutral(4)), Some(5));
        assert_eq!(batched_k5_width(neutral(3)), Some(4));
        assert_eq!(batched_k5_width(neutral(1)), Some(2));
    }

    #[test]
    fn policy_tree_sampled_finished_and_empty_frames_stay_per_sequence_while_grammar_batches() {
        assert_eq!(
            batched_k5_width(BatchedK5Candidate {
                grammar_active: true,
                ..neutral(4)
            }),
            Some(5),
            "active grammar sequences share batched Qwen4 K5 verify"
        );
        let cases = [
            BatchedK5Candidate {
                finished: true,
                ..neutral(4)
            },
            BatchedK5Candidate {
                policy_required: true,
                ..neutral(4)
            },
            BatchedK5Candidate {
                has_tree: true,
                ..neutral(4)
            },
            BatchedK5Candidate {
                has_sampled: true,
                ..neutral(4)
            },
            neutral(0),
            neutral(5),
        ];
        for case in cases {
            assert_eq!(batched_k5_width(case), None, "{case:?}");
        }
    }

    #[test]
    fn wider_than_k9_never_batches() {
        let wide = BatchedK5Candidate {
            configured_num_drafts: 15,
            ..neutral(15)
        };
        assert_eq!(batched_k5_width(wide), None);
        let k9 = BatchedK5Candidate {
            configured_num_drafts: 8,
            ..neutral(8)
        };
        assert_eq!(batched_k5_width(k9), Some(9));
    }

    #[test]
    fn ragged_width_admits_k1_anchor_only_through_k8_and_plan_ragged_cohort_packs_mixed_widths() {
        // Under ragged mode (allow_k1 = true), draft_len=0 maps to Some(1) (K=1 anchor-only)
        // and widths 1..=8 are admitted; K=9 is excluded from ragged bin packing.
        assert_eq!(batched_k5_width_for(neutral(0), false), None);
        assert_eq!(batched_k5_width_for(neutral(0), true), Some(1));
        for drafts in 0..=7usize {
            let cand = BatchedK5Candidate {
                configured_num_drafts: 7,
                ..neutral(drafts)
            };
            assert_eq!(batched_k5_width_for(cand, true), Some(drafts + 1));
        }
        let k9 = BatchedK5Candidate {
            configured_num_drafts: 8,
            ..neutral(8)
        };
        assert_eq!(batched_k5_width_for(k9, true), None);

        // Mixed [7, 3, 1, None, 5, 8, 2] packs into ONE ragged frame (26 rows <= 128).
        let mixed = [
            (0, Some(7)),
            (1, Some(3)),
            (2, Some(1)),
            (3, None),
            (4, Some(5)),
            (5, Some(8)),
            (6, Some(2)),
        ];
        assert_eq!(
            plan_ragged_cohort(&mixed),
            BatchedK5Plan {
                frames: vec![vec![0, 1, 2, 4, 5, 6]],
                leftover: vec![3],
            }
        );

        // Single eligible sequence stays in leftover (no singleton frames).
        assert_eq!(
            plan_ragged_cohort(&[(0, Some(1)), (1, None)]),
            BatchedK5Plan {
                frames: Vec::new(),
                leftover: vec![0, 1],
            }
        );

        // 17 K=8 sequences (> 128 rows) split into 2 valid frames with zero singletons.
        let over_rows: Vec<(usize, Option<usize>)> = (0..17).map(|i| (i, Some(8))).collect();
        let plan_over = plan_ragged_cohort(&over_rows);
        assert_eq!(plan_over.frames.len(), 2);
        assert!(plan_over.leftover.is_empty());
        assert!(
            plan_over
                .frames
                .iter()
                .all(|f| f.len() >= 2 && f.len() * 8 <= 128)
        );

        // 65 K=1 anchor-only sequences (> 64 seqs) split into 2 valid frames (63 + 2).
        let over_seqs: Vec<(usize, Option<usize>)> = (0..65).map(|i| (i, Some(1))).collect();
        let plan_seqs = plan_ragged_cohort(&over_seqs);
        assert_eq!(plan_seqs.frames.len(), 2);
        assert!(plan_seqs.leftover.is_empty());
        assert!(plan_seqs.frames.iter().all(|f| (2..=64).contains(&f.len())));

        // With ragged compaction active (cohort_adaptive_k = false), C=2..16 keeps
        // configured 6 drafts (K_s up to 7); turning ragged off (cohort_adaptive_k = true)
        // restores CohortAcceptEwma uniform throttling on low acceptance.
        let ctrl = CohortAcceptEwma::new();
        for live in [2usize, 4, 8, 16] {
            assert_eq!(
                resolve_cohort_width_with_controller(true, true, false, None, live, 6, &ctrl),
                6,
                "ragged mode drafts up to K=7 ceiling at C={live}"
            );
        }
    }

    #[test]
    fn plan_groups_by_width_and_returns_singletons() {
        let candidates = [
            (0, Some(5)),
            (1, Some(5)),
            (2, Some(4)),
            (3, None),
            (4, Some(5)),
            (5, Some(3)),
            (6, Some(3)),
        ];
        let plan = plan_batched_k5(&candidates);
        assert_eq!(
            plan,
            BatchedK5Plan {
                frames: vec![vec![5, 6], vec![0, 1, 4]],
                leftover: vec![2, 3],
            }
        );
    }

    #[test]
    fn plan_splits_c26_k5_into_two_balanced_frames_while_c8_k5_is_one() {
        let c8: Vec<_> = (0..8).map(|idx| (idx, Some(5))).collect();
        let plan_c8 = plan_batched_k5(&c8);
        assert_eq!(plan_c8.frames, vec![(0..8).collect::<Vec<_>>()]);
        assert!(plan_c8.leftover.is_empty());

        let c26: Vec<_> = (0..26).map(|idx| (idx, Some(5))).collect();
        let plan_c26 = plan_batched_k5(&c26);
        assert_eq!(
            plan_c26.frames,
            vec![(0..13).collect::<Vec<_>>(), (13..26).collect::<Vec<_>>()]
        );
        assert!(plan_c26.leftover.is_empty());
    }

    #[test]
    fn plan_covers_every_index_exactly_once() {
        for n in 0..80usize {
            let candidates: Vec<_> = (0..n)
                .map(|idx| (idx, [Some(5), Some(4), None, Some(5), Some(3)][idx % 5]))
                .collect();
            let plan = plan_batched_k5(&candidates);
            let mut seen: Vec<usize> = plan.frames.iter().flatten().copied().collect();
            seen.extend(&plan.leftover);
            seen.sort_unstable();
            assert_eq!(seen, (0..n).collect::<Vec<_>>(), "n={n}");
            assert!(plan.frames.iter().all(|frame| frame.len() >= 2));
        }
    }

    #[test]
    fn adaptive_drafts_per_concurrency() {
        use spark_model::layers::qwen4_verify_lanes::{
            ADAPTIVE_MAX_FRAME_ROWS, ADAPTIVE_MAX_K, AdaptiveVerifyProfile,
            adaptive_verify_profile, adaptive_verify_width, adaptive_verify_width_for_profile,
        };
        let max_drafts = ADAPTIVE_MAX_K - 1;
        let use_k4_profile = matches!(
            adaptive_verify_profile(),
            Ok(AdaptiveVerifyProfile::K4C7C16)
        );
        for c in 0..=160 {
            let expected_width = if use_k4_profile {
                adaptive_verify_width_for_profile(c, AdaptiveVerifyProfile::K4C7C16)
            } else {
                adaptive_verify_width(c)
            };
            for configured in [2usize, 4, 6, max_drafts] {
                let expected = expected_width
                    .map_or(1, |k| k.saturating_sub(1).max(1))
                    .min(configured);
                assert_eq!(
                    adaptive_drafts(c, configured),
                    expected,
                    "C{c} configured={configured}"
                );
                if (1..=64).contains(&c) {
                    assert!(
                        c * (expected + 1) <= ADAPTIVE_MAX_FRAME_ROWS,
                        "C{c} configured={configured} drafts={expected} exceeds 128-row frame"
                    );
                }
            }
        }
        if !use_k4_profile {
            for c in 1..=16usize {
                assert_eq!(
                    adaptive_drafts(c, 6),
                    max_drafts.min(6),
                    "C{c} with --num-drafts 6 must not clamp below single-frame adaptive ceiling"
                );
            }
        }
        assert_eq!(adaptive_drafts(2, 2), 2, "configured width is a ceiling");
        assert_eq!(
            adaptive_drafts(2, 0),
            0,
            "zero configured drafts stays zero"
        );
    }

    /// Every C2..=C64 cohort plans one <=128-row frame, every C65..=C128 cohort
    /// plans two balanced <=128-row K2 sub-frames (up to 256 verify rows/cycle),
    /// and C129+ has no admitted adaptive width.
    #[test]
    fn adaptive_plans_one_frame_to_c64_and_two_balanced_frames_to_c128() {
        use spark_model::layers::qwen4_verify_lanes::{
            ADAPTIVE_MAX_FRAME_ROWS, ADAPTIVE_MAX_K, AdaptiveVerifyProfile,
            adaptive_verify_profile, adaptive_verify_width,
        };
        let max_drafts = ADAPTIVE_MAX_K - 1;
        let use_k4_profile = matches!(
            adaptive_verify_profile(),
            Ok(AdaptiveVerifyProfile::K4C7C16)
        );
        for c in 1..=140usize {
            let Some(k) = adaptive_verify_width(c) else {
                assert!(c >= 129, "C{c} must speculate");
                assert_eq!(adaptive_drafts(c, max_drafts), 1, "C{c} defensive floor");
                continue;
            };
            let drafts = adaptive_drafts(c, max_drafts);
            let k_eff = drafts + 1;
            if !use_k4_profile {
                assert_eq!(k_eff, k, "C{c}");
            } else {
                assert!(k_eff <= k, "C{c}: k_eff={k_eff} > k={k}");
            }
            let width = batched_k5_width(BatchedK5Candidate {
                configured_num_drafts: max_drafts,
                ..neutral(drafts)
            });
            assert_eq!(width, Some(k_eff), "C{c}");
            let candidates: Vec<_> = (0..c).map(|idx| (idx, width)).collect();
            let plan = plan_batched_uniform(&candidates);
            if c == 1 {
                assert!(plan.frames.is_empty() && plan.leftover == vec![0]);
                continue;
            }
            if c <= 64 {
                assert_eq!(plan.frames.len(), 1, "C{c} K{k_eff}: {plan:?}");
                assert_eq!(plan.frames[0], (0..c).collect::<Vec<_>>(), "C{c}");
                assert!(
                    plan.frames[0].len() * k_eff <= ADAPTIVE_MAX_FRAME_ROWS,
                    "C{c} K{k_eff}"
                );
            } else {
                assert_eq!(k_eff, 2, "C{c}");
                assert_eq!(plan.frames.len(), 2, "C{c} K{k_eff}: {plan:?}");
                let mid = c.div_ceil(2);
                assert_eq!(plan.frames[0], (0..mid).collect::<Vec<_>>(), "C{c}");
                assert_eq!(plan.frames[1], (mid..c).collect::<Vec<_>>(), "C{c}");
                assert!(
                    plan.frames[0].len() * k_eff <= ADAPTIVE_MAX_FRAME_ROWS,
                    "C{c}"
                );
                assert!(
                    plan.frames[1].len() * k_eff <= ADAPTIVE_MAX_FRAME_ROWS,
                    "C{c}"
                );
            }
            assert!(plan.leftover.is_empty(), "C{c}");
        }
    }

    #[test]
    fn fixed_k5_c26_needs_two_frames_and_adaptive_k4_needs_one() {
        let k5: Vec<_> = (0..26).map(|idx| (idx, Some(5))).collect();
        assert_eq!(plan_batched_k5(&k5).frames.len(), 2);
        let k4: Vec<_> = (0..26).map(|idx| (idx, Some(4))).collect();
        assert_eq!(
            plan_batched_uniform(&k4).frames,
            vec![(0..26).collect::<Vec<_>>()]
        );
    }

    #[test]
    fn uniform_plan_keeps_the_dominant_width_and_leaves_the_rest() {
        // A transition step: two sequences still carry a shorter retrieval
        // chain, one is ineligible.
        let candidates = [
            (0, Some(3)),
            (1, Some(3)),
            (2, Some(2)),
            (3, Some(3)),
            (4, None),
            (5, Some(2)),
            (6, Some(3)),
        ];
        let plan = plan_batched_uniform(&candidates);
        assert_eq!(
            plan,
            BatchedK5Plan {
                frames: vec![vec![0, 1, 3, 6]],
                leftover: vec![2, 4, 5],
            }
        );
        // Ties go to the wider width; singletons never form a cohort.
        let tie = [
            (0, Some(2)),
            (1, Some(2)),
            (2, Some(4)),
            (3, Some(4)),
            (4, Some(5)),
        ];
        assert_eq!(plan_batched_uniform(&tie).frames, vec![vec![2, 3]]);
        let singles = [(0, Some(2)), (1, Some(3)), (2, None)];
        let plan = plan_batched_uniform(&singles);
        assert!(plan.frames.is_empty());
        assert_eq!(plan.leftover, vec![0, 1, 2]);
    }

    #[test]
    fn uniform_plan_covers_every_index_exactly_once_with_one_width() {
        for n in 0..40usize {
            let candidates: Vec<_> = (0..n)
                .map(|idx| (idx, [Some(2), Some(3), None, Some(2), Some(5)][idx % 5]))
                .collect();
            let plan = plan_batched_uniform(&candidates);
            let mut seen: Vec<usize> = plan.frames.iter().flatten().copied().collect();
            seen.extend(&plan.leftover);
            seen.sort_unstable();
            assert_eq!(seen, (0..n).collect::<Vec<_>>(), "n={n}");
            let widths: std::collections::BTreeSet<_> = plan
                .frames
                .iter()
                .flatten()
                .map(|&idx| candidates[idx].1)
                .collect();
            assert!(widths.len() <= 1, "n={n} {widths:?}");
        }
    }

    #[test]
    fn narrowing_shortens_only_longer_flat_chains() {
        let mut active: Vec<_> = (0..4)
            .map(|slot| {
                crate::scheduler::test_fixtures::active_seq(
                    crate::scheduler::ResponseSink::Blocking(None),
                    slot,
                )
            })
            .collect();
        active[0].pending_drafts = vec![11, 12, 13, 14];
        active[1].pending_drafts = vec![21, 22];
        active[2].pending_drafts = vec![31, 32, 33, 34];
        active[3].pending_drafts = vec![41, 42, 43, 44];
        narrow_pending_drafts(&mut active, &[0, 1, 2], 2);
        assert_eq!(active[0].pending_drafts, vec![11, 12], "prefix is kept");
        assert_eq!(active[1].pending_drafts, vec![21, 22]);
        assert_eq!(active[2].pending_drafts, vec![31, 32]);
        assert_eq!(
            active[3].pending_drafts,
            vec![41, 42, 43, 44],
            "not in the cohort"
        );
        narrow_pending_drafts(&mut active, &[0], 0);
        assert_eq!(
            active[0].pending_drafts,
            vec![11],
            "a chain never narrows to empty"
        );
    }

    #[test]
    fn hidden_saves_precede_every_proposal() {
        let source = include_str!("verify_qwen4_batched_step.rs");
        let step = source
            .split("pub(super) fn step_verify_qwen4_batched(")
            .nth(1)
            .unwrap()
            .split("fn batch_sequences_mut")
            .next()
            .unwrap();
        let phase1 = step.find("verify_frame(").unwrap();
        let finish = step.find("finish_verify_qwen4_batch(").unwrap();
        let kv_metric = step
            .find("crate::metrics::compat_metrics::set_kv_blocks_free(model.num_free_blocks());")
            .expect("batched verify must refresh kv_blocks_free metric");
        let phase2 = step.find("propose_next(model").unwrap();
        assert!(
            phase1 < finish && finish < kv_metric && kv_metric < phase2,
            "every frame must settle, refresh kv_blocks_free, and release its cohort before proposals"
        );
        let finalize = source.split("fn finalize_commit(").nth(1).unwrap();
        let finalize = finalize.split("fn propose_next(").next().unwrap();
        let commit = finalize.find("commit_verify_state_async(").unwrap();
        let save = finalize
            .find("save_hidden_for_mtp_sequence_in_frame(")
            .unwrap();
        assert!(commit < save);
        assert!(save < finalize.find("trim_proposer_state(").unwrap());
        assert!(!finalize.contains("propose_and_install"));
    }

    #[test]
    fn commits_use_the_slot_lane_and_cohort_abort_owns_state_aware_rollback() {
        let source = include_str!("verify_qwen4_batched_step.rs");
        let step = source
            .split("fn verify_frame(")
            .nth(1)
            .unwrap()
            .split("fn finalize_commit(")
            .next()
            .unwrap();
        let lane = step
            .find("let lane = LaneGuard::enter_for_slot(a.seq.slot_idx)")
            .unwrap();
        let commit = step.find("let propose = finalize_commit(").unwrap();
        let release = step.find("drop(lane);").unwrap();
        assert!(lane < commit && commit < release);
        let abort = source
            .split("fn abort_cohort(")
            .nth(1)
            .unwrap()
            .split("fn verify_frame(")
            .next()
            .unwrap();
        assert!(abort.contains("model.abort_verify_qwen4_batch(&mut seqs)"));
        assert!(!abort.contains("tokens.truncate") && !abort.contains("seq_len ="));
    }

    #[test]
    fn dflash_gumbel_gamma_override_parser_accepts_zero_and_one_through_seven() {
        assert_eq!(parse_dflash_gumbel_gamma_override(None), None);
        assert_eq!(parse_dflash_gumbel_gamma_override(Some("")), None);
        assert_eq!(parse_dflash_gumbel_gamma_override(Some("   ")), None);
        assert_eq!(parse_dflash_gumbel_gamma_override(Some("0")), Some(0));
        assert_eq!(parse_dflash_gumbel_gamma_override(Some(" 0 ")), Some(0));
        for v in 1..=7usize {
            assert_eq!(
                parse_dflash_gumbel_gamma_override(Some(&v.to_string())),
                Some(v),
                "gamma={v}"
            );
        }
        for bad in ["8", "15", "-1", "auto", "4.0"] {
            assert_eq!(
                parse_dflash_gumbel_gamma_override(Some(bad)),
                None,
                "invalid override {bad:?} must fall back to default"
            );
        }
    }

    #[test]
    fn dflash_gumbel_width_keeps_greedy_full_and_narrows_non_greedy_with_overrides() {
        use crate::scheduler::stochastic_verify_walk::SpecVerifyPolicy;

        // Greedy (`T == 0`) keeps full γ=7 across C=1, C=5, C=8, C=16.
        for live in [1usize, 2, 5, 8, 9, 16] {
            assert_eq!(
                resolve_dflash_gumbel_num_drafts(
                    true,
                    SpecVerifyPolicy::Gumbel,
                    false,
                    live,
                    7,
                    None
                ),
                7,
                "greedy cohort at live={live} must keep full gamma=7"
            );
        }

        // Non-greedy (`T > 0`) under Gumbel + DFlash follows the calibrated
        // cohort ladder: C<=2 -> 6, C<=4 -> 5, C>4 -> 4.
        for (live, expected) in [
            (1usize, 6usize),
            (2, 6),
            (3, 5),
            (4, 5),
            (5, 4),
            (8, 4),
            (12, 4),
            (16, 4),
        ] {
            assert_eq!(
                resolve_dflash_gumbel_num_drafts(
                    true,
                    SpecVerifyPolicy::Gumbel,
                    true,
                    live,
                    7,
                    None
                ),
                expected,
                "non-greedy DFlash2 Gumbel at live={live} must resolve to gamma={expected}"
            );
        }

        // Non-greedy (`T > 0`) under Rejection + DFlash keeps gamma=7 for C<=8 and
        // narrows C>8 (C=16) to gamma=5 to eliminate wasted high-C drafter rows.
        for (live, expected) in [(1usize, 7usize), (2, 7), (4, 7), (8, 7), (16, 5)] {
            assert_eq!(
                resolve_dflash_gumbel_num_drafts(
                    true,
                    SpecVerifyPolicy::Rejection,
                    true,
                    live,
                    7,
                    None
                ),
                expected,
                "non-greedy DFlash2 Rejection at live={live} must resolve to gamma={expected}"
            );
        }

        // Configured ceiling is respected when narrower than default.
        assert_eq!(
            resolve_dflash_gumbel_num_drafts(true, SpecVerifyPolicy::Gumbel, true, 1, 2, None),
            2
        );
        assert_eq!(
            resolve_dflash_gumbel_num_drafts(true, SpecVerifyPolicy::Gumbel, true, 16, 2, None),
            2
        );

        // Explicit `ATLAS_QWEN4_DFLASH_GUMBEL_GAMMA=0` disables narrowing.
        for live in [1usize, 4, 8, 16] {
            assert_eq!(
                resolve_dflash_gumbel_num_drafts(
                    true,
                    SpecVerifyPolicy::Gumbel,
                    true,
                    live,
                    7,
                    Some(0),
                ),
                7,
                "override=0 must keep full configured gamma=7 at live={live}"
            );
        }

        // Explicit `ATLAS_QWEN4_DFLASH_GUMBEL_GAMMA=5` overrides cleanly (capped by configured).
        for live in [1usize, 4, 16] {
            assert_eq!(
                resolve_dflash_gumbel_num_drafts(
                    true,
                    SpecVerifyPolicy::Gumbel,
                    true,
                    live,
                    7,
                    Some(5),
                ),
                5,
                "override=5 must cap to 5 at live={live}"
            );
            assert_eq!(
                resolve_dflash_gumbel_num_drafts(
                    true,
                    SpecVerifyPolicy::Gumbel,
                    true,
                    live,
                    4,
                    Some(5),
                ),
                4,
                "override=5 must not exceed configured=4 at live={live}"
            );
        }

        // Non-DFlash proposers and non-Gumbel policies stay untouched.
        assert_eq!(
            resolve_dflash_gumbel_num_drafts(false, SpecVerifyPolicy::Gumbel, true, 16, 7, None),
            7
        );
        assert_eq!(
            resolve_dflash_gumbel_num_drafts(true, SpecVerifyPolicy::Argmax, true, 16, 7, None),
            7
        );
        assert_eq!(
            resolve_dflash_gumbel_num_drafts(true, SpecVerifyPolicy::Stochastic, true, 16, 7, None),
            7
        );
    }

    #[test]
    fn dflash_gumbel_cohort_temperature_check_requires_all_live_seqs_non_greedy() {
        let mk = |slot: usize, temp: f32, finished: bool| {
            let mut a = crate::scheduler::test_fixtures::active_seq(
                crate::scheduler::ResponseSink::Blocking(None),
                slot,
            );
            a.temperature = temp;
            a.finished = finished;
            a
        };

        // Empty or all-finished -> not non-greedy.
        assert_eq!(cohort_live_and_all_non_greedy(&[]), (0, false));
        assert_eq!(
            cohort_live_and_all_non_greedy(&[mk(0, 0.8, true)]),
            (0, false)
        );

        // All live sequences T > 0 -> non-greedy; finished greedy seq is ignored.
        let non_greedy = [mk(0, 0.8, false), mk(1, 0.6, false), mk(2, 0.0, true)];
        assert_eq!(cohort_live_and_all_non_greedy(&non_greedy), (2, true));

        // Any live greedy (or NaN) sequence keeps cohort on full greedy width.
        let mixed = [mk(0, 0.8, false), mk(1, 0.0, false)];
        assert_eq!(cohort_live_and_all_non_greedy(&mixed), (2, false));

        let nan_temp = [mk(0, 0.8, false), mk(1, f32::NAN, false)];
        assert_eq!(cohort_live_and_all_non_greedy(&nan_temp), (2, false));
    }

    #[test]
    fn dflash_pcum_truncation_narrows_uncertain_keeps_high_prob_and_respects_guards() {
        // Parser handles unset, empty, disable (0 / <=0), valid overrides, and invalid strings.
        assert_eq!(parse_dflash_pcum_thresh_override(None), None);
        assert_eq!(parse_dflash_pcum_thresh_override(Some("")), None);
        assert_eq!(parse_dflash_pcum_thresh_override(Some("   ")), None);
        assert_eq!(parse_dflash_pcum_thresh_override(Some("0")), Some(0.0));
        assert_eq!(parse_dflash_pcum_thresh_override(Some("0.0")), Some(0.0));
        assert_eq!(parse_dflash_pcum_thresh_override(Some("-0.25")), Some(0.0));
        assert_eq!(parse_dflash_pcum_thresh_override(Some("0.32")), Some(0.32));
        assert_eq!(parse_dflash_pcum_thresh_override(Some("1.0")), Some(1.0));
        for bad in ["1.01", "2.0", "NaN", "inf", "auto"] {
            assert_eq!(
                parse_dflash_pcum_thresh_override(Some(bad)),
                None,
                "invalid threshold {bad:?} must fall back to default"
            );
        }

        use crate::scheduler::stochastic_verify_walk::SpecVerifyPolicy;

        // Policy- and cohort-aware defaults for T > 0 vs T == 0.
        for (c, want_rej) in [
            (1usize, 0.55f32),
            (2, 0.55),
            (4, 0.58),
            (8, 0.58),
            (16, 0.58),
        ] {
            assert_eq!(
                super::dflash_pcum_default_for_cohort_with_policy(
                    0.8,
                    c,
                    SpecVerifyPolicy::Rejection
                ),
                want_rej,
                "Rejection T>0 must match calibrated threshold across cohort_size={c}"
            );
            assert_eq!(
                super::dflash_pcum_default_for_cohort_with_policy(
                    0.0,
                    c,
                    SpecVerifyPolicy::Rejection
                ),
                0.25,
                "greedy T==0 must stay at 0.25 across cohort_size={c}"
            );
            assert_eq!(
                super::dflash_pcum_default_for_cohort_with_policy(0.0, c, SpecVerifyPolicy::Gumbel),
                0.25,
                "greedy T==0 under Gumbel must stay at 0.25 across cohort_size={c}"
            );
        }
        assert_eq!(
            super::dflash_pcum_default_for_cohort_with_policy(0.8, 1, SpecVerifyPolicy::Gumbel),
            0.05
        );
        assert_eq!(
            super::dflash_pcum_default_for_cohort_with_policy(0.8, 2, SpecVerifyPolicy::Gumbel),
            0.05
        );
        assert_eq!(
            super::dflash_pcum_default_for_cohort_with_policy(0.8, 4, SpecVerifyPolicy::Gumbel),
            0.06
        );
        assert_eq!(
            super::dflash_pcum_default_for_cohort_with_policy(0.8, 8, SpecVerifyPolicy::Gumbel),
            0.10
        );
        assert_eq!(
            super::dflash_pcum_default_for_cohort_with_policy(0.8, 16, SpecVerifyPolicy::Gumbel),
            0.13
        );

        // T > 0 (C=1 Gumbel default tau = 0.05): narrows low-prob tails, keeps >= 1 draft (K_s >= 2).
        let low_probs = [0.20f32, 0.20, 0.20, 0.15];
        // p_cum: [0.20, 0.04 (< 0.05)] -> keeps 1 draft (K = 2).
        assert_eq!(dflash_pcum_keep_len(4, &low_probs, 0.7, None), 1);
        let mid_probs = [0.60f32, 0.50, 0.35, 0.25];
        // p_cum: [0.60, 0.30, 0.105, 0.02625].
        assert_eq!(dflash_pcum_keep_len(4, &mid_probs, 0.7, None), 3);
        let high_probs = [0.80f32, 0.70, 0.55, 0.45];
        // p_cum: [0.80, 0.56, 0.308, 0.1386 (>= 0.13)] -> keeps all 4 drafts (K = 5).
        assert_eq!(dflash_pcum_keep_len(4, &high_probs, 0.7, None), 4);
        // Cohort scaling under Gumbel: C=4 (0.06) and C=8 (0.10) keep 3 on mid_probs (0.105);
        // C=16 (0.13) trims mid_probs to 2 while keeping high_probs (0.1386) at 4.
        assert_eq!(
            super::dflash_pcum_keep_len_for_cohort(4, &mid_probs, 1.0, 4, None),
            3
        );
        assert_eq!(
            super::dflash_pcum_keep_len_for_cohort(4, &mid_probs, 1.0, 8, None),
            3
        );
        assert_eq!(
            super::dflash_pcum_keep_len_for_cohort(4, &mid_probs, 1.0, 16, None),
            2
        );
        assert_eq!(
            super::dflash_pcum_keep_len_for_cohort(4, &high_probs, 1.0, 16, None),
            4
        );
        // Under Rejection (calibrated for prefix-only `max_z q_t(z)`), C<=2 uses 0.55
        // and C>=4 uses 0.58: `high_probs` [0.80, 0.70, 0.55, 0.45] has p_cum [0.80, 0.56,
        // 0.308, 0.1386] so C=2 keeps 2 (0.56 >= 0.55) while C=16 keeps 1 (0.56 < 0.58),
        // and confident code [0.95, 0.90, 0.85, 0.80] (p_cum 0.5814 >= 0.58) keeps all 4.
        assert_eq!(
            super::dflash_pcum_keep_len_for_cohort_with_policy(
                4,
                &high_probs,
                1.0,
                2,
                SpecVerifyPolicy::Rejection,
                None,
            ),
            2
        );
        assert_eq!(
            super::dflash_pcum_keep_len_for_cohort_with_policy(
                4,
                &high_probs,
                1.0,
                16,
                SpecVerifyPolicy::Rejection,
                None,
            ),
            1
        );
        let confident_probs = [0.95f32, 0.90, 0.85, 0.80];
        assert_eq!(
            super::dflash_pcum_keep_len_for_cohort_with_policy(
                4,
                &confident_probs,
                1.0,
                16,
                SpecVerifyPolicy::Rejection,
                None,
            ),
            4
        );
        // Explicit override takes precedence over policy/cohort default.
        assert_eq!(
            super::dflash_pcum_keep_len_for_cohort_with_policy(
                4,
                &high_probs,
                1.0,
                16,
                SpecVerifyPolicy::Rejection,
                Some(0.25),
            ),
            3
        );

        // T == 0 greedy (default tau = 0.25): keeps full K=8 on high-prob code, narrows prose.
        let code_probs = [0.95f32, 0.92, 0.90, 0.88, 0.86, 0.85, 0.84];
        assert_eq!(dflash_pcum_keep_len(7, &code_probs, 0.0, None), 7);
        let prose_probs = [0.55f32, 0.50, 0.48, 0.45, 0.40, 0.35, 0.30];
        // p_cum: [0.55, 0.275 (>= 0.25), 0.132 (< 0.25)] -> keeps 2 drafts (K = 3).
        assert_eq!(dflash_pcum_keep_len(7, &prose_probs, 0.0, None), 2);

        // Override = 0.0 disables truncation completely; empty probs are a no-op.
        assert_eq!(dflash_pcum_keep_len(7, &prose_probs, 0.0, Some(0.0)), 7);
        assert_eq!(dflash_pcum_keep_len(4, &low_probs, 0.7, Some(0.0)), 4);
        assert_eq!(dflash_pcum_keep_len(7, &[], 0.7, None), 7);

        // ActiveSeq guards: non-DFlash, retrieval chains, forced nudge, and length-1 stay untouched.
        let mk_seq = |temp: f32, origin: crate::scheduler::DraftOrigin| {
            let mut a = crate::scheduler::test_fixtures::active_seq(
                crate::scheduler::ResponseSink::Blocking(None),
                0,
            );
            a.temperature = temp;
            a.draft_origin = origin;
            a.pending_drafts = vec![10, 11, 12, 13, 14, 15, 16];
            a
        };

        let mut non_dflash = mk_seq(0.7, crate::scheduler::DraftOrigin::Proposer);
        apply_dflash_pcum_truncation_with_probs(&mut non_dflash, false, &prose_probs, None);
        assert_eq!(non_dflash.pending_drafts.len(), 7);

        for origin in [
            crate::scheduler::DraftOrigin::RestStore,
            crate::scheduler::DraftOrigin::SelfContext,
        ] {
            let mut retrieved = mk_seq(0.7, origin);
            apply_dflash_pcum_truncation_with_probs(&mut retrieved, true, &prose_probs, None);
            assert_eq!(
                retrieved.pending_drafts.len(),
                7,
                "retrieval origin {origin:?} must never be truncated"
            );
        }

        let mut forced = mk_seq(0.7, crate::scheduler::DraftOrigin::Proposer);
        forced.inside_thinking = true;
        forced.force_end_thinking = true;
        forced.in_code_fence = false;
        apply_dflash_pcum_truncation_with_probs(&mut forced, true, &prose_probs, None);
        assert_eq!(
            forced.pending_drafts.len(),
            7,
            "forced close chain must never be truncated"
        );

        let mut dflash_prose = mk_seq(0.0, crate::scheduler::DraftOrigin::Proposer);
        apply_dflash_pcum_truncation_with_probs(&mut dflash_prose, true, &prose_probs, None);
        assert_eq!(dflash_prose.pending_drafts, vec![10, 11]);

        let mut dflash_low = mk_seq(0.8, crate::scheduler::DraftOrigin::Proposer);
        dflash_low.pending_drafts = vec![20, 21, 22, 23];
        apply_dflash_pcum_truncation_with_probs(&mut dflash_low, true, &low_probs, None);
        assert_eq!(
            dflash_low.pending_drafts,
            vec![20],
            "must always keep >= 1 draft (K_s >= 2)"
        );
    }

    #[test]
    fn dflash_sm120_vs_gb10_gamma_pcum_and_ragged_bucket_parity() {
        use crate::scheduler::stochastic_verify_walk::SpecVerifyPolicy;

        // 1. Draft width under Rejection: GB10 (<=64 SMs) caps C>8 at gamma=5;
        //    SM120 (>64 SMs) preserves full gamma=7 (K=8) through C<=32.
        for gb10_sms in [48u32, 64u32] {
            for (live, want) in [(1, 7), (4, 7), (8, 7), (9, 5), (16, 5), (32, 5)] {
                assert_eq!(
                    super::resolve_dflash_gumbel_num_drafts_for_sm(
                        true,
                        SpecVerifyPolicy::Rejection,
                        true,
                        live,
                        7,
                        None,
                        gb10_sms,
                    ),
                    want,
                    "GB10 sm={gb10_sms} live={live}"
                );
            }
        }
        for sm120_sms in [65u32, 148u32, 188u32] {
            for live in [1usize, 2, 4, 8, 9, 16, 24, 32] {
                assert_eq!(
                    super::resolve_dflash_gumbel_num_drafts_for_sm(
                        true,
                        SpecVerifyPolicy::Rejection,
                        true,
                        live,
                        7,
                        None,
                        sm120_sms,
                    ),
                    7,
                    "SM120 sm={sm120_sms} live={live} must preserve gamma=7"
                );
            }
            assert_eq!(
                super::resolve_dflash_gumbel_num_drafts_for_sm(
                    true,
                    SpecVerifyPolicy::Rejection,
                    true,
                    33,
                    7,
                    None,
                    sm120_sms,
                ),
                5,
                "SM120 sm={sm120_sms} live=33 exceeds wide cap"
            );
            // Explicit env override still wins on SM120.
            assert_eq!(
                super::resolve_dflash_gumbel_num_drafts_for_sm(
                    true,
                    SpecVerifyPolicy::Rejection,
                    true,
                    16,
                    7,
                    Some(5),
                    sm120_sms,
                ),
                5
            );
        }

        // 2. Rejection p_cum defaults: GB10 keeps 0.55/0.58/0.58/0.58;
        //    SM120 sets 0.0 (disabled, steady uniform K=8) for C<=24 and 0.15 for C>24.
        for gb10_sms in [48u32, 64u32] {
            assert_eq!(
                super::dflash_pcum_default_for_cohort_with_policy_for_sm(
                    0.8,
                    2,
                    SpecVerifyPolicy::Rejection,
                    gb10_sms
                ),
                0.55
            );
            for c in [4usize, 8, 16, 24, 32] {
                assert_eq!(
                    super::dflash_pcum_default_for_cohort_with_policy_for_sm(
                        0.8,
                        c,
                        SpecVerifyPolicy::Rejection,
                        gb10_sms
                    ),
                    0.58,
                    "GB10 sm={gb10_sms} c={c}"
                );
            }
        }
        for sm120_sms in [65u32, 188u32] {
            for c in [1usize, 2, 4, 8, 16, 24] {
                assert_eq!(
                    super::dflash_pcum_default_for_cohort_with_policy_for_sm(
                        0.8,
                        c,
                        SpecVerifyPolicy::Rejection,
                        sm120_sms
                    ),
                    0.0,
                    "SM120 sm={sm120_sms} c={c} must default Rejection p_cum to 0.0"
                );
            }
            for c in [25usize, 32] {
                assert_eq!(
                    super::dflash_pcum_default_for_cohort_with_policy_for_sm(
                        0.8,
                        c,
                        SpecVerifyPolicy::Rejection,
                        sm120_sms
                    ),
                    0.15,
                    "SM120 sm={sm120_sms} c={c} must default Rejection p_cum to 0.15"
                );
            }
            // Greedy T==0 stays 0.25 on SM120.
            assert_eq!(
                super::dflash_pcum_default_for_cohort_with_policy_for_sm(
                    0.0,
                    16,
                    SpecVerifyPolicy::Rejection,
                    sm120_sms
                ),
                0.25
            );
        }

        // 3. Ragged row bucket granularity: GB10 keeps 8 at C=16 and 16 at C>=24;
        //    SM120 widens to 16 at C=12..23 and 32 at C>=24.
        assert_eq!(super::dflash_ragged_row_bucket_step_for_sm(8, 48), 8);
        assert_eq!(super::dflash_ragged_row_bucket_step_for_sm(16, 48), 8);
        assert_eq!(super::dflash_ragged_row_bucket_step_for_sm(24, 48), 16);
        assert_eq!(super::dflash_ragged_row_bucket_step_for_sm(32, 48), 16);

        assert_eq!(super::dflash_ragged_row_bucket_step_for_sm(2, 188), 2);
        assert_eq!(super::dflash_ragged_row_bucket_step_for_sm(4, 188), 4);
        assert_eq!(super::dflash_ragged_row_bucket_step_for_sm(8, 188), 8);
        assert_eq!(super::dflash_ragged_row_bucket_step_for_sm(12, 188), 16);
        assert_eq!(super::dflash_ragged_row_bucket_step_for_sm(16, 188), 16);
        assert_eq!(super::dflash_ragged_row_bucket_step_for_sm(23, 188), 16);
        assert_eq!(super::dflash_ragged_row_bucket_step_for_sm(24, 188), 32);
        assert_eq!(super::dflash_ragged_row_bucket_step_for_sm(32, 188), 32);
    }

    #[test]
    fn dflash_ragged_bucket_quantization_matches_verify_graph_grid_and_preserves_floor() {
        assert_eq!(dflash_ragged_row_bucket_step(1), 1);
        assert_eq!(dflash_ragged_row_bucket_step(2), 2);
        assert_eq!(dflash_ragged_row_bucket_step(3), 4);
        assert_eq!(dflash_ragged_row_bucket_step(4), 4);
        assert_eq!(dflash_ragged_row_bucket_step(5), 4);
        assert_eq!(dflash_ragged_row_bucket_step(6), 8);
        assert_eq!(dflash_ragged_row_bucket_step(8), 8);
        assert_eq!(dflash_ragged_row_bucket_step(16), 8);
        assert_eq!(dflash_ragged_row_bucket_step(24), 16);

        // Uniform cohorts are always graph-admitted and must NEVER be trimmed.
        let mut uniform3 = [4usize, 4, 4]; // total_rows = 15 (not % 4), uniform -> untouched
        let pcum3 = vec![vec![0.9, 0.8, 0.7, 0.6]; 3];
        quantize_ragged_lens_to_graph_bucket(&mut uniform3, &pcum3);
        assert_eq!(uniform3, [4, 4, 4]);

        // b = 2 (step = 2): ragged [4, 1] -> total_rows = 7 -> snaps to 6 ([3, 1]).
        let mut b2 = [4usize, 1];
        let pcum2 = vec![vec![0.9, 0.8, 0.7, 0.4], vec![0.2]];
        quantize_ragged_lens_to_graph_bucket(&mut b2, &pcum2);
        assert_eq!(b2, [3, 1]);
        assert_eq!(b2.iter().map(|&l| l + 1).sum::<usize>() % 2, 0);

        // b = 4 (step = 4): trims lowest tail-p_cum drafts first while keeping >= 1 per lane.
        let mut b4 = [4usize, 3, 2, 1]; // total_rows = 14 (rem = 2) -> snaps to 12.
        let pcum4 = vec![
            vec![0.95, 0.90, 0.85, 0.80], // tail 0.80 (high confidence -> preserved)
            vec![0.70, 0.55, 0.35],       // tail 0.35 (lowest -> trimmed first, then 0.55)
            vec![0.60, 0.40],             // tail 0.40 (second lowest -> trimmed second)
            vec![0.20],                   // at K=2 floor (len=1 -> protected)
        ];
        quantize_ragged_lens_to_graph_bucket(&mut b4, &pcum4);
        assert_eq!(b4, [4, 2, 1, 1]);
        assert_eq!(b4.iter().map(|&l| l + 1).sum::<usize>(), 12);

        // b = 8 (step = 8) and b = 16 (step = 8): total_rows aligns to multiple of 8.
        let mut b8 = [4usize, 4, 3, 3, 2, 2, 1, 1]; // total_rows = 28 (rem = 4) -> 24
        let pcum8: Vec<Vec<f32>> = b8
            .iter()
            .enumerate()
            .map(|(s, &l)| {
                (0..l)
                    .map(|i| 0.9 - 0.05 * s as f32 - 0.1 * i as f32)
                    .collect()
            })
            .collect();
        quantize_ragged_lens_to_graph_bucket(&mut b8, &pcum8);
        assert_eq!(b8.iter().map(|&l| l + 1).sum::<usize>(), 24);
        assert!(b8.iter().all(|&l| l >= 1));

        let mut b16 = [3usize, 3, 3, 3, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1]; // total_rows = 46 (rem = 6) -> 40
        let pcum16: Vec<Vec<f32>> = b16
            .iter()
            .map(|&l| (0..l).map(|i| 0.85 - 0.15 * i as f32).collect())
            .collect();
        quantize_ragged_lens_to_graph_bucket(&mut b16, &pcum16);
        assert_eq!(b16.iter().map(|&l| l + 1).sum::<usize>(), 40);
        assert!(b16.iter().all(|&l| l >= 1));

        // Floor case (b = 6, total_rows = 13, rem = 5): stops at K=2 floor (12 == 2 * 6).
        let mut floor6 = [2usize, 1, 1, 1, 1, 1];
        let pcum6: Vec<Vec<f32>> = floor6.iter().map(|&l| vec![0.5; l]).collect();
        quantize_ragged_lens_to_graph_bucket(&mut floor6, &pcum6);
        assert_eq!(floor6, [1, 1, 1, 1, 1, 1]);
    }

    #[test]
    fn dflash_ragged_bucket_quantization_skips_r56_cliff_for_large_cohorts() {
        // b = 16, exact r = 56 (ragged): must snap down by 8 rows to r = 48 while keeping len >= 1.
        let mut exact56 = [4usize, 4, 4, 4, 3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1];
        assert_eq!(exact56.iter().map(|&l| l + 1).sum::<usize>(), 56);
        let pcum56: Vec<Vec<f32>> = exact56
            .iter()
            .map(|&l| (0..l).map(|i| 0.90 - 0.15 * i as f32).collect())
            .collect();
        quantize_ragged_lens_to_graph_bucket(&mut exact56, &pcum56);
        assert_eq!(exact56.iter().map(|&l| l + 1).sum::<usize>(), 48);
        assert!(exact56.iter().all(|&l| l >= 1));

        // b = 16, r = 60 (quantizes to 56 -> snaps down to 48 by trimming 12 rows).
        let mut over56 = [4usize, 4, 4, 4, 4, 4, 3, 3, 2, 2, 2, 2, 2, 2, 1, 1];
        assert_eq!(over56.iter().map(|&l| l + 1).sum::<usize>(), 60);
        let pcum60: Vec<Vec<f32>> = over56
            .iter()
            .map(|&l| (0..l).map(|i| 0.90 - 0.15 * i as f32).collect())
            .collect();
        quantize_ragged_lens_to_graph_bucket(&mut over56, &pcum60);
        assert_eq!(over56.iter().map(|&l| l + 1).sum::<usize>(), 48);
        assert!(over56.iter().all(|&l| l >= 1));

        // b = 12 (threshold boundary), r = 56 -> snaps down to 48.
        let mut b12 = [5usize, 5, 5, 5, 4, 4, 4, 4, 2, 2, 2, 2];
        assert_eq!(b12.iter().map(|&l| l + 1).sum::<usize>(), 56);
        let pcum12: Vec<Vec<f32>> = b12
            .iter()
            .map(|&l| (0..l).map(|i| 0.85 - 0.10 * i as f32).collect())
            .collect();
        quantize_ragged_lens_to_graph_bucket(&mut b12, &pcum12);
        assert_eq!(b12.iter().map(|&l| l + 1).sum::<usize>(), 48);
        assert!(b12.iter().all(|&l| l >= 1));

        // b = 10 (< 12), r = 56 stays at 56; b = 16 at r = 64 or r = 48 stays untouched.
        let mut b10 = [6usize, 6, 6, 5, 5, 4, 4, 4, 3, 3];
        assert_eq!(b10.iter().map(|&l| l + 1).sum::<usize>(), 56);
        let pcum10: Vec<Vec<f32>> = b10.iter().map(|&l| vec![0.5; l]).collect();
        quantize_ragged_lens_to_graph_bucket(&mut b10, &pcum10);
        assert_eq!(b10.iter().map(|&l| l + 1).sum::<usize>(), 56);

        let mut b16_64 = [5usize, 5, 5, 5, 4, 4, 4, 4, 2, 2, 2, 2, 1, 1, 1, 1];
        assert_eq!(b16_64.iter().map(|&l| l + 1).sum::<usize>(), 64);
        let pcum64: Vec<Vec<f32>> = b16_64.iter().map(|&l| vec![0.5; l]).collect();
        quantize_ragged_lens_to_graph_bucket(&mut b16_64, &pcum64);
        assert_eq!(b16_64.iter().map(|&l| l + 1).sum::<usize>(), 64);

        // End-to-end `quantize_dflash_ragged_drafts_to_graph_bucket` must NOT
        // early-return when ragged `total_rows == 56` (`56 % 8 == 0`) at `b == 16`.
        let mut active: Vec<crate::scheduler::ActiveSeq> =
            [4usize, 4, 4, 4, 3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1]
                .into_iter()
                .enumerate()
                .map(|(slot, len)| {
                    let mut a = crate::scheduler::test_fixtures::active_seq(
                        crate::scheduler::ResponseSink::Blocking(None),
                        slot,
                    );
                    a.pending_drafts = (0..len as u32).map(|i| 100 + i).collect();
                    a
                })
                .collect();
        let idxs: Vec<usize> = (0..16).collect();
        super::quantize_dflash_ragged_drafts_to_graph_bucket(&mut active, &idxs);
        let after_rows: usize = active.iter().map(|a| a.pending_drafts.len() + 1).sum();
        assert_eq!(after_rows, 48);
        assert!(active.iter().all(|a| !a.pending_drafts.is_empty()));
    }

    #[test]
    fn dflash_ragged_bucket_quantization_prefers_ahead_streams_before_stragglers() {
        // b = 4 (step = 4): total_rows = 14 (rem = 2) -> must trim 2 rows to hit 12.
        // Stream 0 and 1 are ahead of cohort median remaining (`median = 120`, threshold `112`),
        // with higher `tail_p` than straggler stream 2 (`rem = 120`, `tail_p = 0.20`).
        // Ahead-first tie-break must trim ahead streams first (lowest `tail_p` among ahead)
        // while preserving straggler stream 2 and respecting the `len >= 1` (`K >= 2`) floor.
        let mut lens = [4usize, 2, 3, 1];
        let pcum = vec![
            vec![0.95, 0.90, 0.85, 0.80], // ahead (rem=40), tail_p=0.80 -> trimmed second (after lane 1 hits floor)
            vec![0.75, 0.65], // ahead (rem=50), tail_p=0.65 -> trimmed first (hits floor len=1)
            vec![0.40, 0.30, 0.20], // straggler (rem=120), tail_p=0.20 -> preserved!
            vec![0.10],       // straggler (rem=130), at K=2 floor
        ];
        let remaining = [40usize, 50, 120, 130];
        quantize_ragged_lens_to_graph_bucket_with_remaining(&mut lens, &pcum, Some(&remaining));
        assert_eq!(lens, [3, 1, 3, 1]);
        assert_eq!(lens.iter().map(|&l| l + 1).sum::<usize>(), 12);

        // Fallback when all ahead streams hit the `len == 1` (`K == 2`) floor before `rem == 0`:
        // b = 4 (step = 4), total_rows = 15 (rem = 3) -> snaps to 12.
        // Ahead stream 0 starts at len=2 (can only give 1 row before K=2 floor); the remaining
        // 2 trims fall back to lowest `tail_p` across non-ahead streams (lane 2 then lane 1).
        let mut fallback = [2usize, 4, 4, 1];
        let pcum_fb = vec![
            vec![0.90, 0.85],             // ahead (rem=30): gives 1 row -> len=1 (floor)
            vec![0.95, 0.90, 0.85, 0.80], // non-ahead (rem=108 >= 102), tail=0.80 -> preserved
            vec![0.70, 0.55, 0.40, 0.30], // straggler (rem=110), tails 0.30, 0.40 -> gives 2 rows
            vec![0.10],                   // straggler (rem=120), at floor
        ];
        let rem_fb = [30usize, 108, 110, 120];
        quantize_ragged_lens_to_graph_bucket_with_remaining(&mut fallback, &pcum_fb, Some(&rem_fb));
        assert_eq!(fallback, [1, 4, 2, 1]);
        assert_eq!(fallback.iter().map(|&l| l + 1).sum::<usize>(), 12);

        // When all streams have equal remaining tokens (none < median - 8), behaves identically
        // to pure lowest-`tail_p` trimming and never touches uniform cohorts.
        let mut equal_rem = [4usize, 3, 2, 1];
        let mut baseline = equal_rem;
        let pcum_eq = vec![
            vec![0.95, 0.90, 0.85, 0.80],
            vec![0.70, 0.55, 0.35],
            vec![0.60, 0.40],
            vec![0.20],
        ];
        let same = [100usize; 4];
        quantize_ragged_lens_to_graph_bucket(&mut baseline, &pcum_eq);
        quantize_ragged_lens_to_graph_bucket_with_remaining(&mut equal_rem, &pcum_eq, Some(&same));
        assert_eq!(equal_rem, baseline);

        let mut uniform = [3usize, 3, 3, 3];
        let pcum_u = vec![vec![0.8, 0.6, 0.4]; 4];
        quantize_ragged_lens_to_graph_bucket_with_remaining(
            &mut uniform,
            &pcum_u,
            Some(&[20, 100, 110, 120]),
        );
        assert_eq!(uniform, [3, 3, 3, 3]);
    }

    #[test]
    fn dflash_ragged_upward_bucket_quantization_aligns_grid_skips_r56_and_never_trims_down() {
        // b = 2 (step = 2): ragged [4, 1] -> total_rows = 7 -> snaps UP to 8 ([4, 2] or [5, 1]).
        // Stream 0 has higher stopping-frontier pcum (0.055 vs 0.020), so stream 0 extends to 5.
        let mut b2 = [4usize, 1];
        let max2 = [7usize, 7];
        let fp2 = [0.055f32, 0.020];
        super::quantize_ragged_lens_upward_to_graph_bucket_with_remaining(
            &mut b2, &max2, &fp2, None,
        );
        assert_eq!(b2, [5, 1]);
        assert_eq!(b2.iter().map(|&l| l + 1).sum::<usize>(), 8);

        // b = 4 (step = 4): prefers stragglers (remaining >= median - 8) first,
        // then smallest increment, then highest frontier_pcum.
        let mut b4 = [3usize, 2, 2, 1]; // total_rows = 12 (aligned) -> untouched
        let max4 = [7usize; 4];
        let fp4 = [0.04f32, 0.05, 0.03, 0.02];
        super::quantize_ragged_lens_upward_to_graph_bucket_with_remaining(
            &mut b4, &max4, &fp4, None,
        );
        assert_eq!(b4, [3, 2, 2, 1]);

        let mut b4_unaligned = [4usize, 2, 2, 1]; // total_rows = 13 -> needs +3 to hit 16
        let rem4 = [30usize, 115, 120, 130]; // sorted[2] = 120, threshold = 112: lane 0 ahead, lanes 1,2,3 behind
        super::quantize_ragged_lens_upward_to_graph_bucket_with_remaining(
            &mut b4_unaligned,
            &max4,
            &fp4,
            Some(&rem4),
        );
        // Behind lanes 1, 2, 3 each receive +1 (smallest added first), lane 0 untouched.
        assert_eq!(b4_unaligned, [4, 3, 3, 2]);
        assert_eq!(b4_unaligned.iter().map(|&l| l + 1).sum::<usize>(), 16);

        // b = 16, r = 56 already aligns to step = 8: upward quantization stays at 56
        // (never adds a wasteful +8 jump to 64 and never trims downward below p_cum stop).
        let mut exact56 = [4usize, 4, 4, 4, 3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1];
        let orig56 = exact56;
        let max16 = [5usize; 16];
        let fp16 = [0.08f32; 16];
        assert_eq!(exact56.iter().map(|&l| l + 1).sum::<usize>(), 56);
        super::quantize_ragged_lens_upward_to_graph_bucket_with_remaining(
            &mut exact56,
            &max16,
            &fp16,
            None,
        );
        assert_eq!(exact56.iter().map(|&l| l + 1).sum::<usize>(), 56);
        assert_eq!(exact56, orig56);

        // b = 16, r = 53 (gamma = 5 ceiling): snaps UP minimally by +3 to 56 (not +11 to 64).
        let mut r53 = [4usize, 4, 4, 3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1, 1];
        let orig53 = r53;
        assert_eq!(r53.iter().map(|&l| l + 1).sum::<usize>(), 53);
        super::quantize_ragged_lens_upward_to_graph_bucket_with_remaining(
            &mut r53, &max16, &fp16, None,
        );
        assert_eq!(r53.iter().map(|&l| l + 1).sum::<usize>(), 56);
        for (&after, &before) in r53.iter().zip(orig53.iter()) {
            assert!(
                after >= before,
                "upward quantization must never trim below p_cum stop"
            );
            assert!(after <= 5, "upward quantization must respect gamma=5 cap");
        }

        // Insufficient slack: leaves lens untouched instead of trimming downward.
        let mut tight = [4usize, 1];
        let max_tight = [4usize, 1];
        super::quantize_ragged_lens_upward_to_graph_bucket_with_remaining(
            &mut tight, &max_tight, &fp2, None,
        );
        assert_eq!(tight, [4, 1]);
    }

    #[test]
    fn dflash_rejection_upward_quantization_preserves_exact_markov_marginal_tv_zero() {
        // Non-i.i.d. 3-state Markov chain (spiky repetition-prone state 0,
        // flat state 1, medium state 2) with mismatched flattened drafter Q.
        // Verifies that prefix-only max_z q_t(z) + `dflash_pcum_keep_len_and_frontier_*`
        // + upward bucket quantization preserves exact target marginal (TV < 0.01).
        const P: [[f64; 3]; 3] = [[0.90, 0.05, 0.05], [0.35, 0.35, 0.30], [0.55, 0.25, 0.20]];
        const Q: [[f64; 3]; 3] = [[0.84, 0.08, 0.08], [0.34, 0.34, 0.32], [0.50, 0.27, 0.23]];
        let mut rng_state = 0x9E37_79B9_7F4A_7C15u64;
        let mut next_f64 = || -> f64 {
            rng_state = rng_state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = rng_state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            ((z >> 11) as f64) * (1.0 / ((1u64 << 53) as f64))
        };
        let sample_cat = |probs: [f64; 3], u: f64| -> usize {
            if u < probs[0] {
                0
            } else if u < probs[0] + probs[1] {
                1
            } else {
                2
            }
        };

        const B: usize = 4;
        const GAMMA: usize = 7;
        let mut states = [0usize; B];
        let mut counts = [[0u64; 3]; 3];
        for _ in 0..25_000 {
            let mut drafts = [[0usize; GAMMA]; B];
            let mut max_probs = [[0.0f32; GAMMA]; B];
            for s in 0..B {
                let mut cur = states[s];
                for t in 0..GAMMA {
                    let u = next_f64();
                    let x = sample_cat(Q[cur], u);
                    drafts[s][t] = x;
                    // Prefix-only max_z q_t(z | x_{<t}) (matches dflash2_selector.cu s_w[0]).
                    max_probs[s][t] = Q[cur][0].max(Q[cur][1]).max(Q[cur][2]) as f32;
                    cur = x;
                }
            }
            let mut lens = [GAMMA; B];
            let max_lens = [GAMMA; B];
            let mut frontiers = [1.0f32; B];
            for s in 0..B {
                let (keep, frontier) =
                    super::dflash_pcum_keep_len_and_frontier_for_cohort_with_policy(
                        GAMMA,
                        &max_probs[s],
                        1.0,
                        B,
                        crate::scheduler::stochastic_verify_walk::SpecVerifyPolicy::Rejection,
                        Some(0.15),
                    );
                lens[s] = keep;
                frontiers[s] = frontier;
            }
            super::quantize_ragged_lens_upward_to_graph_bucket_with_remaining(
                &mut lens, &max_lens, &frontiers, None,
            );
            for s in 0..B {
                let mut st = states[s];
                let mut rejected = false;
                for &x in &drafts[s][..lens[s]] {
                    let px = P[st][x];
                    let qx = Q[st][x];
                    if px >= qx || next_f64() * qx < px {
                        counts[st][x] += 1;
                        st = x;
                    } else {
                        let mut res = [0.0f64; 3];
                        let mut sum = 0.0f64;
                        for j in 0..3 {
                            res[j] = (P[st][j] - Q[st][j]).max(0.0);
                            sum += res[j];
                        }
                        for r in &mut res {
                            *r /= sum;
                        }
                        let y = sample_cat(res, next_f64());
                        counts[st][y] += 1;
                        st = y;
                        rejected = true;
                        break;
                    }
                }
                if !rejected {
                    let y = sample_cat(P[st], next_f64());
                    counts[st][y] += 1;
                    st = y;
                }
                states[s] = st;
            }
        }
        for i in 0..3 {
            let total = (counts[i][0] + counts[i][1] + counts[i][2]) as f64;
            let tv: f64 = 0.5
                * (0..3)
                    .map(|j| ((counts[i][j] as f64) / total - P[i][j]).abs())
                    .sum::<f64>();
            assert!(
                tv < 0.01,
                "row {i} marginal TV={tv:.5} exceeded 0.01 (counts={:?}, true={:?})",
                counts[i],
                P[i]
            );
        }
    }

    mod strict_grammar_batch_tests {
        use super::super::{
            BatchedK5Plan, batched_k5_policy_required, plan_for_cohort, step_verify_qwen4_batched,
            verify_frame, warm_ple_draft_ahead_for_seq, warm_ple_draft_ahead_for_seqs,
            warm_ple_verify_prefetch_for_seqs,
        };
        use crate::grammar::{GrammarEngine, GrammarState};
        use crate::scheduler::{ActiveSeq, ResponseSink, ThinkSpecCtx};
        use anyhow::Result;
        use spark_model::traits::{Model, SequenceState};
        use spark_runtime::gpu::DevicePtr;
        use std::sync::Mutex;
        use std::sync::atomic::{AtomicUsize, Ordering};

        const VOCAB: usize = 129;
        const EOS_ID: u32 = 128;
        const ILLEGAL: u32 = 1; // ASCII SOH — never allowed outside string or at JSON start
        const LOGITS_BASE: u64 = 0x10_0000;
        const SCHEMA: &str = r#"{"type":"object","additionalProperties":false,"properties":{"bridge":{"type":"string"}},"required":["bridge"]}"#;

        fn strict_json_grammar() -> GrammarState {
            let mut vocab: Vec<String> = (0u8..128).map(|b| String::from(b as char)).collect();
            vocab.push("<eos>".to_string());
            let mut engine = GrammarEngine::new(&vocab, &[EOS_ID as i32]).unwrap();
            let compiled = engine.compile_json_schema(SCHEMA).unwrap();
            GrammarState::new(&compiled, engine.vocab_size()).unwrap()
        }

        fn ascii_tokens(s: &str) -> Vec<u32> {
            s.bytes().map(u32::from).collect()
        }

        fn bf16_row(legal_top: u32, illegal_outscores: bool) -> Vec<u8> {
            let mut f32s = vec![-8.0f32; VOCAB];
            f32s[legal_top as usize] = 40.0;
            if illegal_outscores {
                f32s[ILLEGAL as usize] = 50.0;
            }
            f32s.into_iter()
                .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
                .collect()
        }

        fn raw_argmax_of_bf16_row(row: &[u8]) -> u32 {
            row.chunks_exact(2)
                .enumerate()
                .max_by(|(_, a), (_, b)| {
                    let fa = f32::from_bits((u16::from_le_bytes([a[0], a[1]]) as u32) << 16);
                    let fb = f32::from_bits((u16::from_le_bytes([b[0], b[1]]) as u32) << 16);
                    fa.total_cmp(&fb)
                })
                .map_or(0, |(id, _)| id as u32)
        }

        fn default_think_ctx() -> ThinkSpecCtx<'static> {
            ThinkSpecCtx {
                enabled: false,
                code_fence_token: None,
                reflection_suppress_ids: &[],
                adaptive_sampling: false,
                honor_eos_inside_thinking: Some(false),
            }
        }

        fn active_with_drafts(
            slot: usize,
            last_token: u32,
            drafts: Vec<u32>,
            grammar: Option<GrammarState>,
        ) -> ActiveSeq {
            let mut a =
                crate::scheduler::test_fixtures::active_seq(ResponseSink::Blocking(None), slot);
            a.seq.prompt_len = 2;
            a.seq.tokens = vec![65, last_token];
            a.seq.seq_len = 2;
            a.last_token = last_token;
            a.output_tokens = vec![last_token];
            a.max_output_tokens = 64;
            a.remaining = 63;
            a.eos_tokens = vec![EOS_ID];
            a.inside_thinking = false;
            a.think_ended = true;
            a.think_just_ended = false;
            a.repetition_penalty = 1.0;
            a.presence_penalty = 0.0;
            a.frequency_penalty = 0.0;
            a.dry_multiplier = 0.0;
            a.lz_penalty = 0.0;
            a.pending_drafts = drafts;
            a.grammar_state = grammar;
            a
        }

        struct BatchedGrammarMockModel {
            inner: Box<dyn Model>,
            dflash: bool,
            dflash_sample_method: spark_model::dflash_sampling::DflashDraftSampleMethod,
            frame_rows_bf16: Vec<Vec<u8>>,
            next_proposals: Vec<Vec<u32>>,
            batched_verify_calls: AtomicUsize,
            batched_propose_calls: AtomicUsize,
            committed: Mutex<Vec<(usize, usize, usize)>>,
            saved_hiddens: Mutex<Vec<(usize, usize, usize)>>,
            retired_dflash: Mutex<Vec<(usize, usize, usize, usize)>>,
            trimmed: Mutex<Vec<(usize, usize)>>,
            ple_warmed_seqs: Mutex<Vec<Vec<(Vec<u32>, Vec<u32>)>>>,
        }

        impl BatchedGrammarMockModel {
            fn new(frame_rows_bf16: Vec<Vec<u8>>, next_proposals: Vec<Vec<u32>>) -> Self {
                Self::with_proposer(
                    false,
                    spark_model::dflash_sampling::DflashDraftSampleMethod::Greedy,
                    frame_rows_bf16,
                    next_proposals,
                )
            }

            fn new_dflash(
                frame_rows_bf16: Vec<Vec<u8>>,
                next_proposals: Vec<Vec<u32>>,
                dflash_sample_method: spark_model::dflash_sampling::DflashDraftSampleMethod,
            ) -> Self {
                Self::with_proposer(true, dflash_sample_method, frame_rows_bf16, next_proposals)
            }

            fn with_proposer(
                dflash: bool,
                dflash_sample_method: spark_model::dflash_sampling::DflashDraftSampleMethod,
                frame_rows_bf16: Vec<Vec<u8>>,
                next_proposals: Vec<Vec<u32>>,
            ) -> Self {
                let inner = crate::scheduler::test_fixtures::model::recording();
                inner.batched_k5.store(true, Ordering::SeqCst);
                Self {
                    inner: Box::new(inner),
                    dflash,
                    dflash_sample_method,
                    frame_rows_bf16,
                    next_proposals,
                    batched_verify_calls: AtomicUsize::new(0),
                    batched_propose_calls: AtomicUsize::new(0),
                    committed: Mutex::new(Vec::new()),
                    saved_hiddens: Mutex::new(Vec::new()),
                    retired_dflash: Mutex::new(Vec::new()),
                    trimmed: Mutex::new(Vec::new()),
                    ple_warmed_seqs: Mutex::new(Vec::new()),
                }
            }
        }

        impl Model for BatchedGrammarMockModel {
            fn prefill(&self, t: &[u32], s: &mut SequenceState, st: u64) -> Result<DevicePtr> {
                self.inner.prefill(t, s, st)
            }
            fn prefill_chunk(
                &self,
                t: &[u32],
                s: &mut SequenceState,
                o: usize,
                l: usize,
                last: bool,
                st: u64,
            ) -> Result<DevicePtr> {
                self.inner.prefill_chunk(t, s, o, l, last, st)
            }
            fn decode(&self, t: u32, s: &mut SequenceState, st: u64) -> Result<DevicePtr> {
                self.inner.decode(t, s, st)
            }
            fn decode_batch(
                &self,
                t: &[u32],
                s: &mut [&mut SequenceState],
                st: u64,
            ) -> Result<DevicePtr> {
                self.inner.decode_batch(t, s, st)
            }
            fn vocab_size(&self) -> usize {
                VOCAB
            }
            fn bind_gpu_to_thread(&self) -> Result<()> {
                Ok(())
            }
            fn synchronize_stream(&self, _: u64) -> Result<()> {
                Ok(())
            }
            fn alloc_sequence(&self) -> Result<SequenceState> {
                self.inner.alloc_sequence()
            }
            fn free_sequence(&self, s: &mut SequenceState) -> Result<()> {
                self.inner.free_sequence(s)
            }
            fn compact_sequence(&self, s: &mut SequenceState, k: usize) -> Result<bool> {
                self.inner.compact_sequence(s, k)
            }
            fn cache_sequence(&self, s: &SequenceState) {
                self.inner.cache_sequence(s)
            }
            fn logits_buffer_ptr(&self) -> DevicePtr {
                DevicePtr(LOGITS_BASE)
            }
            fn copy_logits_to_host(&self, ptr: DevicePtr, out: &mut [u8]) -> Result<()> {
                let row_bytes = VOCAB * 2;
                let byte_off = (ptr.0 - LOGITS_BASE) as usize;
                assert_eq!(byte_off % row_bytes, 0);
                assert_eq!(out.len() % row_bytes, 0);
                let start_row = byte_off / row_bytes;
                let count_rows = out.len() / row_bytes;
                for r in 0..count_rows {
                    let src = &self.frame_rows_bf16[start_row + r];
                    out[r * row_bytes..(r + 1) * row_bytes].copy_from_slice(src);
                }
                Ok(())
            }
            fn argmax_on_device(&self, p: DevicePtr, st: u64) -> Result<u32> {
                self.inner.argmax_on_device(p, st)
            }
            fn argmax_batch(&self, p: DevicePtr, n: usize, st: u64) -> Result<Vec<u32>> {
                self.inner.argmax_batch(p, n, st)
            }
            fn hidden_after_norm(&self) -> DevicePtr {
                DevicePtr::NULL
            }
            fn decode_verify(&self, t: &[u32], s: &mut SequenceState, st: u64) -> Result<Vec<u32>> {
                self.inner.decode_verify(t, s, st)
            }
            fn decode_verify_graphed(
                &self,
                t: &[u32; 2],
                s: &mut SequenceState,
                st: u64,
            ) -> Result<[u32; 2]> {
                self.inner.decode_verify_graphed(t, s, st)
            }
            fn decode_verify_graphed_k3(
                &self,
                t: &[u32; 3],
                s: &mut SequenceState,
                st: u64,
            ) -> Result<[u32; 3]> {
                self.inner.decode_verify_graphed_k3(t, s, st)
            }
            fn decode_verify_graphed_k4(
                &self,
                t: &[u32; 4],
                s: &mut SequenceState,
                st: u64,
            ) -> Result<[u32; 4]> {
                self.inner.decode_verify_graphed_k4(t, s, st)
            }
            fn checkpoint_ssm_states(&self, s: &mut SequenceState) -> Result<()> {
                self.inner.checkpoint_ssm_states(s)
            }
            fn rollback_ssm_states(&self, s: &mut SequenceState, n: usize) -> Result<()> {
                self.inner.rollback_ssm_states(s, n)
            }
            fn has_ssm_layers(&self) -> bool {
                true
            }
            fn generate_speculative(
                &self,
                p: &[u32],
                sp: &spark_runtime::sampler::SamplingParams,
                nd: usize,
            ) -> Result<spark_model::engine::GenerateResult> {
                self.inner.generate_speculative(p, sp, nd)
            }
            fn has_proposer(&self) -> bool {
                true
            }
            fn has_self_speculative(&self) -> bool {
                false
            }
            fn decode_draft(&self, t: u32, s: &mut SequenceState, st: u64) -> Result<DevicePtr> {
                self.inner.decode_draft(t, s, st)
            }
            fn save_hidden_for_mtp(&self, r: usize, st: u64) -> Result<()> {
                self.inner.save_hidden_for_mtp(r, st)
            }
            fn run_mtp_propose(
                &self,
                t: u32,
                p: usize,
                s: &mut SequenceState,
                st: u64,
            ) -> Result<Option<u32>> {
                self.inner.run_mtp_propose(t, p, s, st)
            }
            fn qwen4_batched_k5_armed(&self) -> bool {
                true
            }
            fn proposer_is_dflash(&self) -> bool {
                self.dflash
            }
            fn dflash_draft_sample_method(
                &self,
            ) -> spark_model::dflash_sampling::DflashDraftSampleMethod {
                self.dflash_sample_method
            }
            fn proposer_arm_kind(
                &self,
                _: u8,
            ) -> Option<spark_model::speculative::DraftProposerKind> {
                if self.dflash {
                    Some(spark_model::speculative::DraftProposerKind::DFlash)
                } else {
                    Some(spark_model::speculative::DraftProposerKind::Qwen4NativeMtp)
                }
            }
            fn sync_secondary(&self) -> Result<()> {
                Ok(())
            }
            fn preflight_verify_qwen4_batched(
                &self,
                _: &[Vec<u32>],
                _: &[&SequenceState],
            ) -> Result<()> {
                Ok(())
            }
            fn begin_verify_qwen4_batch(
                &self,
                _: &mut [&mut SequenceState],
                _: usize,
            ) -> Result<()> {
                Ok(())
            }
            fn end_verify_qwen4_batch_frame(&self, _: &[&SequenceState]) -> Result<()> {
                Ok(())
            }
            fn finish_verify_qwen4_batch(&self, _: &[&SequenceState]) -> Result<()> {
                Ok(())
            }
            fn abort_verify_qwen4_batch(&self, _: &mut [&mut SequenceState]) -> Result<()> {
                Ok(())
            }
            fn retire_verify_qwen4_batch_terminal(&self, _: &SequenceState) -> Result<()> {
                Ok(())
            }
            fn decode_verify_qwen4_batched(
                &self,
                tokens_per_seq: &[Vec<u32>],
                seqs: &mut [&mut SequenceState],
            ) -> Result<Vec<Vec<u32>>> {
                self.batched_verify_calls.fetch_add(1, Ordering::SeqCst);
                let mut row_cursor = 0usize;
                let mut out = Vec::with_capacity(seqs.len());
                for (window, seq) in tokens_per_seq.iter().zip(seqs.iter_mut()) {
                    seq.tokens.extend_from_slice(window);
                    seq.seq_len += window.len();
                    let verified: Vec<u32> = (0..window.len())
                        .map(|i| raw_argmax_of_bf16_row(&self.frame_rows_bf16[row_cursor + i]))
                        .collect();
                    row_cursor += window.len();
                    out.push(verified);
                }
                Ok(out)
            }
            fn commit_verify_state_async(
                &self,
                seq: &mut SequenceState,
                accepted: usize,
                k: usize,
            ) -> Result<()> {
                self.committed
                    .lock()
                    .unwrap()
                    .push((seq.slot_idx, accepted, k));
                Ok(())
            }
            fn save_hidden_for_mtp_sequence_in_frame(
                &self,
                row_base: usize,
                num_accepted: usize,
                seq: &SequenceState,
                _: u64,
            ) -> Result<()> {
                self.saved_hiddens
                    .lock()
                    .unwrap()
                    .push((seq.slot_idx, row_base, num_accepted));
                Ok(())
            }
            fn retire_dflash_frame_captures(
                &self,
                row_base: usize,
                num_accepted: usize,
                seq: &mut SequenceState,
                _: u64,
            ) -> Result<()> {
                self.retired_dflash.lock().unwrap().push((
                    seq.slot_idx,
                    row_base,
                    num_accepted,
                    seq.seq_len,
                ));
                Ok(())
            }
            fn trim_proposer_state(
                &self,
                seq: &mut SequenceState,
                num_accepted: usize,
                _: u64,
            ) -> Result<()> {
                self.trimmed
                    .lock()
                    .unwrap()
                    .push((seq.slot_idx, num_accepted));
                Ok(())
            }
            fn run_mtp_propose_multi(
                &self,
                _: u32,
                _: usize,
                _: usize,
                seq: &mut SequenceState,
                _: u64,
                _: Option<&[i32]>,
            ) -> Result<Vec<u32>> {
                Ok(self
                    .next_proposals
                    .get(seq.slot_idx)
                    .cloned()
                    .unwrap_or_default())
            }
            fn run_mtp_propose_multi_batched(
                &self,
                _: &[u32],
                _: &[usize],
                _: usize,
                seqs: &mut [&mut SequenceState],
                _: u64,
            ) -> Result<Vec<Vec<u32>>> {
                self.batched_propose_calls.fetch_add(1, Ordering::SeqCst);
                Ok(seqs
                    .iter()
                    .map(|s| {
                        self.next_proposals
                            .get(s.slot_idx)
                            .cloned()
                            .unwrap_or_default()
                    })
                    .collect())
            }
            fn warm_ple_draft_ahead_seqs(&self, seqs: &[(&[u32], &[u32])]) {
                self.ple_warmed_seqs.lock().unwrap().push(
                    seqs.iter()
                        .map(|(window, prior)| (window.to_vec(), prior.to_vec()))
                        .collect(),
                );
            }
        }

        #[test]
        fn grammar_and_mixed_cohorts_batch_together_in_plan_for_cohort() {
            let think = default_think_ctx();
            let model = BatchedGrammarMockModel::new(Vec::new(), Vec::new());

            // Pure grammar cohort (C = 2, K = 5)
            let pure_grammar = vec![
                active_with_drafts(0, 65, ascii_tokens(r#"{"br"#), Some(strict_json_grammar())),
                active_with_drafts(1, 66, ascii_tokens(r#"{"ok"#), Some(strict_json_grammar())),
            ];
            for a in &pure_grammar {
                assert!(!batched_k5_policy_required(a, &think));
            }
            let plan = plan_for_cohort(&model, &pure_grammar, &[0, 1], 4, &think);
            assert_eq!(
                plan,
                BatchedK5Plan {
                    frames: vec![vec![0, 1]],
                    leftover: vec![],
                }
            );

            // Mixed grammar + plain cohort (C = 4, K = 5)
            let mixed = vec![
                active_with_drafts(0, 65, ascii_tokens(r#"{"br"#), Some(strict_json_grammar())),
                active_with_drafts(1, 66, vec![70, 71, 72, 73], None),
                active_with_drafts(2, 67, ascii_tokens(r#"{"hi"#), Some(strict_json_grammar())),
                active_with_drafts(3, 68, vec![80, 81, 82, 83], None),
            ];
            let mixed_plan = plan_for_cohort(&model, &mixed, &[0, 1, 2, 3], 4, &think);
            assert_eq!(
                mixed_plan,
                BatchedK5Plan {
                    frames: vec![vec![0, 1, 2, 3]],
                    leftover: vec![],
                }
            );
        }

        #[test]
        fn batched_verify_advances_strict_grammar_only_on_accepted_prefix_and_masked_bonus() {
            // Seq 0 (K = 4, 3 legal drafts `{"b` + legal bonus `r`): accepts all 3 + bonus -> `{"br`.
            // Seq 1 (K = 4, `temperature = 0.8`, drafts = `[b'{', ILLEGAL, b'Z']`):
            //   - row 0: legal `{` matches draft[0] -> accepted!
            //   - row 1: raw argmax is ILLEGAL (50.0 > 40.0), legal top is `"`; grammar bitmask
            //     suppresses ILLEGAL so target is `"`, rejecting draft[1]=ILLEGAL and stopping!
            //   - row 2..3 (`b'Z'`): unaccepted tail must NEVER advance `grammar_state`!
            //   - next batched proposal for Seq 1 returns `[ILLEGAL, ILLEGAL, ILLEGAL]`:
            //     `splice_grammar_drafts` coalesces forced `"bridge":` prefix into `pending_drafts`
            //     while leaving `grammar_state` rolled back to exact post-`{"` state!
            let mut frame_rows = Vec::new();
            for ch in ascii_tokens(r#"{"br"#) {
                frame_rows.push(bf16_row(ch, false));
            }
            frame_rows.push(bf16_row(u32::from(b'{'), false));
            frame_rows.push(bf16_row(u32::from(b'"'), true)); // raw argmax is ILLEGAL!
            frame_rows.push(bf16_row(u32::from(b'Z'), true));
            frame_rows.push(bf16_row(u32::from(b'Z'), true));

            let next_proposals = vec![ascii_tokens("idge"), vec![ILLEGAL, ILLEGAL, ILLEGAL]];
            let model = BatchedGrammarMockModel::new(frame_rows, next_proposals);
            let mut active = vec![
                active_with_drafts(0, 65, ascii_tokens(r#"{"b"#), Some(strict_json_grammar())),
                active_with_drafts(
                    1,
                    66,
                    vec![u32::from(b'{'), ILLEGAL, u32::from(b'Z')],
                    Some(strict_json_grammar()),
                ),
            ];
            active[1].temperature = 0.8;
            active[1].seed = Some(0xCAFE_BABE);

            let think = default_think_ctx();
            let plan = plan_for_cohort(&model, &active, &[0, 1], 3, &think);
            assert_eq!(plan.frames, vec![vec![0, 1]]);
            assert!(step_verify_qwen4_batched(
                &model,
                &mut active,
                &plan.frames,
                3,
                &think
            ));

            assert_eq!(model.batched_verify_calls.load(Ordering::SeqCst), 1);
            assert_eq!(model.batched_propose_calls.load(Ordering::SeqCst), 1);
            assert_eq!(active[0].output_tokens[1..], ascii_tokens(r#"{"br"#));
            assert_eq!(active[1].output_tokens[1..], ascii_tokens(r#"{""#));
            assert_eq!(
                *model.committed.lock().unwrap(),
                vec![(0, 4, 4), (1, 2, 4)],
                "seq 0 commits 3+1=4 inputs; seq 1 commits 1+1=2 inputs"
            );
            assert_eq!(*model.trimmed.lock().unwrap(), vec![(0, 3), (1, 1)]);
            // Host sequence extents rolled back to pre_verify_len (2) + num_accepted + 1.
            assert_eq!(active[0].seq.seq_len, 6);
            assert_eq!(active[0].seq.tokens.len(), 6);
            assert_eq!(active[1].seq.seq_len, 4);
            assert_eq!(active[1].seq.tokens.len(), 4);

            // Seq 1's illegal proposal `[ILLEGAL, ILLEGAL, ILLEGAL]` after `{"` was coalesced
            // to the forced schema key `"bridge"` prefix (`b`, `r`, `i`) without mutating gs!
            assert_eq!(active[1].pending_drafts, ascii_tokens("bri"));

            // Batched cohorts (`propose_idxs.len() >= 2`) immediately submit an
            // exact-order reusable PLE verify prefetch for the proposed cohort so
            // NVMe reads overlap with tick-end host work, KV alloc, and Layer 0,
            // while solo `C=1` (`propose_idxs.len() == 1`) continues warming PLE.
            {
                let warmed = model.ple_warmed_seqs.lock().unwrap();
                assert_eq!(
                    warmed.len(),
                    1,
                    "batched cohorts (propose_idxs.len() >= 2) must submit PLE verify prefetch"
                );
                let mut expected_win_0 = vec![active[0].last_token];
                expected_win_0.extend_from_slice(&active[0].pending_drafts);
                let mut expected_win_1 = vec![active[1].last_token];
                expected_win_1.extend_from_slice(&active[1].pending_drafts);
                assert_eq!(
                    warmed[0],
                    vec![
                        (expected_win_0, active[0].seq.tokens.clone()),
                        (expected_win_1, active[1].seq.tokens.clone()),
                    ]
                );
            }
            super::super::propose_next_batch(&model, &mut active, &[0], 3);
            {
                let warmed = model.ple_warmed_seqs.lock().unwrap();
                assert_eq!(
                    warmed.len(),
                    2,
                    "solo C=1 propose_next_batch must still warm PLE draft-ahead"
                );
                let mut expected_win_0 = vec![active[0].last_token];
                expected_win_0.extend_from_slice(&active[0].pending_drafts);
                assert_eq!(
                    warmed[1],
                    vec![(expected_win_0, active[0].seq.tokens.clone())]
                );
            }

            // Verify each sequence's GrammarState advanced ONLY by its emitted prefix:
            // Seq 0 emitted `{"br` -> `i` is legal next, `Z` is illegal!
            let gs0 = active[0].grammar_state.as_mut().unwrap();
            assert!(!gs0.accept_token(u32::from(b'Z')));
            assert!(gs0.accept_token(u32::from(b'i')));

            // Seq 1 emitted `{"` -> `b` is legal next, `Z` is illegal!
            let gs1 = active[1].grammar_state.as_mut().unwrap();
            assert!(!gs1.accept_token(u32::from(b'Z')));
            assert!(gs1.accept_token(u32::from(b'b')));
        }

        #[test]
        fn mixed_grammar_and_plain_cohort_shares_one_batched_verify_forward() {
            // Seq 0: strict JSON grammar, drafts `[b'{', ILLEGAL, b'"']` -> accepts `{`, masked bonus `"`.
            // Seq 1: grammarless neutral greedy, drafts `[70, 71, 99]`, verified `[70, 71, 72, 73]` -> accepts 2 + bonus 72.
            let frame_rows = vec![
                bf16_row(u32::from(b'{'), false),
                bf16_row(u32::from(b'"'), true),
                bf16_row(u32::from(b'b'), false),
                bf16_row(u32::from(b'r'), false),
                bf16_row(70, false),
                bf16_row(71, false),
                bf16_row(72, false),
                bf16_row(73, false),
            ];
            let model = BatchedGrammarMockModel::new(
                frame_rows,
                vec![ascii_tokens("bri"), vec![73, 74, 75]],
            );
            let mut active = vec![
                active_with_drafts(
                    0,
                    65,
                    vec![u32::from(b'{'), ILLEGAL, u32::from(b'"')],
                    Some(strict_json_grammar()),
                ),
                active_with_drafts(1, 66, vec![70, 71, 99], None),
            ];
            // Seq 1 is pure neutral non-post-thinking greedy (fast non-policy branch in finalize_commit).
            active[1].think_ended = false;

            let think = default_think_ctx();
            let plan = plan_for_cohort(&model, &active, &[0, 1], 3, &think);
            assert_eq!(plan.frames, vec![vec![0, 1]]);
            assert!(step_verify_qwen4_batched(
                &model,
                &mut active,
                &plan.frames,
                3,
                &think
            ));

            assert_eq!(model.batched_verify_calls.load(Ordering::SeqCst), 1);
            assert_eq!(active[0].output_tokens[1..], ascii_tokens(r#"{""#));
            assert_eq!(active[1].output_tokens[1..], [70, 71, 72]);
            assert_eq!(active[1].last_token, 72);
            let gs0 = active[0].grammar_state.as_mut().unwrap();
            assert!(gs0.accept_token(u32::from(b'b')));
        }

        #[test]
        fn ragged_verify_frame_commits_mixed_k7_k3_k1_k5_with_exact_row_bases() {
            // 4 sequences in ONE ragged frame:
            //   seq 0 (K=7, row_base=0):  6 drafts [10..=15], target agrees on all 6 + bonus 16 -> accept 6
            //   seq 1 (K=3, row_base=7):  2 drafts [20, 21],  target agrees on [20], rejects 21->29 -> accept 1
            //   seq 2 (K=1, row_base=10): 0 drafts [] (anchor-only!), target emits bonus 30 -> accept 0
            //   seq 3 (K=5, row_base=11): 4 drafts [40..=43], target agrees on [40, 41], bonus 49 -> accept 2
            let target_tokens: Vec<u32> = vec![
                10, 11, 12, 13, 14, 15, 16, // seq 0: rows 0..7
                20, 29, 99, // seq 1: rows 7..10
                30, // seq 2: row 10 (K=1 anchor-only)
                40, 41, 49, 99, 99, // seq 3: rows 11..16
            ];
            let frame_rows: Vec<Vec<u8>> = target_tokens
                .iter()
                .map(|&tok| bf16_row(tok, false))
                .collect();
            let model = BatchedGrammarMockModel::new(frame_rows, Vec::new());

            let mut active = vec![
                active_with_drafts(0, 9, vec![10, 11, 12, 13, 14, 15], None),
                active_with_drafts(1, 19, vec![20, 21], None),
                active_with_drafts(2, 29, Vec::new(), None),
                active_with_drafts(3, 39, vec![40, 41, 42, 43], None),
            ];
            for a in &mut active {
                a.think_ended = false;
                a.tool_call_end_token = None;
            }
            let tokens_per_seq: Vec<Vec<u32>> = active
                .iter()
                .map(|a| {
                    let mut w = Vec::with_capacity(1 + a.pending_drafts.len());
                    w.push(a.last_token);
                    w.extend_from_slice(&a.pending_drafts);
                    w
                })
                .collect();
            let think = default_think_ctx();
            let propose_ready = verify_frame(
                &model,
                &mut active,
                &[0, 1, 2, 3],
                &tokens_per_seq,
                4,
                &think,
            )
            .expect("ragged verify_frame succeeds");

            assert_eq!(propose_ready, vec![0, 1, 2, 3]);
            assert_eq!(model.batched_verify_calls.load(Ordering::SeqCst), 1);
            assert_eq!(active[0].output_tokens[1..], [10, 11, 12, 13, 14, 15, 16]);
            assert_eq!(active[1].output_tokens[1..], [20, 29]);
            assert_eq!(active[2].output_tokens[1..], [30]);
            assert_eq!(active[3].output_tokens[1..], [40, 41, 49]);
            assert_eq!(active[2].last_token, 30);

            // Commit records (slot, accepted_inputs = num_accepted + 1, k_verify).
            assert_eq!(
                *model.committed.lock().unwrap(),
                vec![(0, 7, 7), (1, 2, 3), (2, 1, 1), (3, 3, 5)]
            );
            // Hidden handoffs MUST use cumulative row bases (0, 7, 10, 11), NOT s * k!
            assert_eq!(
                *model.saved_hiddens.lock().unwrap(),
                vec![(0, 0, 6), (1, 7, 1), (2, 10, 0), (3, 11, 2)]
            );
            assert_eq!(
                *model.trimmed.lock().unwrap(),
                vec![(0, 6), (1, 1), (2, 0), (3, 2)]
            );
            // Sequence extents rolled back to pre_verify_len (2) + num_accepted + 1.
            assert_eq!(active[0].seq.seq_len, 9);
            assert_eq!(active[1].seq.seq_len, 4);
            assert_eq!(active[2].seq.seq_len, 3);
            assert_eq!(active[3].seq.seq_len, 5);
        }

        #[test]
        fn warm_ple_draft_ahead_packs_dedups_and_skips_finished_or_empty_seqs() {
            let model = BatchedGrammarMockModel::new(Vec::new(), Vec::new());
            let mut s0 = active_with_drafts(0, 70, vec![71, 72, 73], None);
            s0.seq.tokens = vec![10, 20, 30];
            // Identical 2-token prior tail `[20, 30]`, last_token `70`, and drafts -> deduplicated.
            let mut s1 = active_with_drafts(1, 70, vec![71, 72, 73], None);
            s1.seq.tokens = vec![99, 20, 30];
            // Distinct prior tail `[21, 30]` -> kept because trigram/bigram hashes differ.
            let mut s2 = active_with_drafts(2, 70, vec![71, 72, 73], None);
            s2.seq.tokens = vec![21, 30];
            // Finished or empty pending_drafts -> skipped.
            let mut s3 = active_with_drafts(3, 80, vec![81, 82], None);
            s3.finished = true;
            let s4 = active_with_drafts(4, 90, Vec::new(), None);
            let active = [s0, s1, s2, s3, s4];

            warm_ple_draft_ahead_for_seqs(&model, &active, &[0, 1, 2, 3, 4]);
            {
                let warmed = model.ple_warmed_seqs.lock().unwrap();
                assert_eq!(warmed.len(), 1);
                assert_eq!(
                    warmed[0],
                    vec![
                        (vec![70, 71, 72, 73], vec![10, 20, 30]),
                        (vec![70, 71, 72, 73], vec![21, 30]),
                    ]
                );
            }

            // Single-sequence C=1 helper skips finished/empty and submits exact `(window, prior)`.
            warm_ple_draft_ahead_for_seq(&model, &active[3]);
            warm_ple_draft_ahead_for_seq(&model, &active[4]);
            assert_eq!(model.ple_warmed_seqs.lock().unwrap().len(), 1);

            warm_ple_draft_ahead_for_seq(&model, &active[2]);
            {
                let warmed = model.ple_warmed_seqs.lock().unwrap();
                assert_eq!(warmed.len(), 2);
                assert_eq!(warmed[1], vec![(vec![70, 71, 72, 73], vec![21, 30])]);
            }

            // Cross-sequence verify prefetch preserves exact cohort order and duplicates
            // (so un-deduplicated DFlash2 verify frames hit exact key match), skips finished
            // sequences, and admits K=1 anchor-only lanes only when `allow_k1` is enabled.
            warm_ple_verify_prefetch_for_seqs(&model, &active, &[0, 1, 2, 3, 4], false);
            warm_ple_verify_prefetch_for_seqs(&model, &active, &[0, 1, 2, 3, 4], true);
            let warmed = model.ple_warmed_seqs.lock().unwrap();
            assert_eq!(warmed.len(), 4);
            assert_eq!(
                warmed[2],
                vec![
                    (vec![70, 71, 72, 73], vec![10, 20, 30]),
                    (vec![70, 71, 72, 73], vec![99, 20, 30]),
                    (vec![70, 71, 72, 73], vec![21, 30]),
                ]
            );
            assert_eq!(
                warmed[3],
                vec![
                    (vec![70, 71, 72, 73], vec![10, 20, 30]),
                    (vec![70, 71, 72, 73], vec![99, 20, 30]),
                    (vec![70, 71, 72, 73], vec![21, 30]),
                    (vec![90], vec![65, 90]),
                ]
            );
        }

        #[test]
        fn dflash_batched_admits_greedy_k8_uniform_refuses_probabilistic_and_retires_captures() {
            use spark_model::dflash_sampling::DflashDraftSampleMethod;

            let think = default_think_ctx();
            // Seq 0 (K=8, row_base=0): drafts [10..=16], target agrees on 3 drafts + bonus 90 -> accept 3, seq_len = 2 + 4 = 6.
            // Seq 1 (K=8, row_base=8): drafts [20..=26], target agrees on all 7 drafts + bonus 27 -> accept 7, seq_len = 2 + 8 = 10.
            let target_tokens: Vec<u32> = vec![
                10, 11, 12, 90, 99, 99, 99, 99, // seq 0: rows 0..8
                20, 21, 22, 23, 24, 25, 26, 27, // seq 1: rows 8..16
            ];
            let frame_rows: Vec<Vec<u8>> = target_tokens
                .iter()
                .map(|&tok| bf16_row(tok, false))
                .collect();
            let next_proposals = vec![
                vec![30, 31, 32, 33, 34, 35, 36],
                vec![40, 41, 42, 43, 44, 45, 46],
            ];
            let model = BatchedGrammarMockModel::new_dflash(
                frame_rows,
                next_proposals,
                DflashDraftSampleMethod::Greedy,
            );
            let mut active = vec![
                active_with_drafts(0, 9, vec![10, 11, 12, 13, 14, 15, 16], None),
                active_with_drafts(1, 19, vec![20, 21, 22, 23, 24, 25, 26], None),
            ];
            for a in &mut active {
                a.think_ended = false;
                a.tool_call_end_token = None;
            }
            let plan = plan_for_cohort(&model, &active, &[0, 1], 7, &think);
            assert_eq!(
                plan,
                BatchedK5Plan {
                    frames: vec![vec![0, 1]],
                    leftover: Vec::new(),
                }
            );
            assert!(step_verify_qwen4_batched(
                &model,
                &mut active,
                &plan.frames,
                7,
                &think
            ));
            assert_eq!(model.batched_verify_calls.load(Ordering::SeqCst), 1);
            assert_eq!(model.batched_propose_calls.load(Ordering::SeqCst), 1);
            assert!(
                model.saved_hiddens.lock().unwrap().is_empty(),
                "DFlash2 must retire DFlash captures rather than saving native-MTP hiddens"
            );
            assert_eq!(
                *model.retired_dflash.lock().unwrap(),
                vec![(0, 0, 3, 6), (1, 8, 7, 10)],
                "retire_dflash_frame_captures sees exact (slot, row_base, num_accepted, seq_len = pre_verify_len + num_accepted + 1)"
            );
            assert_eq!(*model.trimmed.lock().unwrap(), vec![(0, 3), (1, 7)]);

            // Probabilistic DFlash stays excluded from batched Qwen4 verify.
            let prob_model = BatchedGrammarMockModel::new_dflash(
                Vec::new(),
                Vec::new(),
                DflashDraftSampleMethod::Probabilistic,
            );
            let prob_plan = plan_for_cohort(&prob_model, &active, &[0, 1], 7, &think);
            assert!(prob_plan.frames.is_empty());
            assert_eq!(prob_plan.leftover, vec![0, 1]);
            assert!(!step_verify_qwen4_batched(
                &prob_model,
                &mut active,
                &[vec![0, 1]],
                7,
                &think
            ));

            // Bypassing CohortAcceptEwma (cohort_adaptive_k = false) preserves full gamma=7 (K=8)
            // across all C=2..=16 even after sustained zero-accept frames.
            let ctrl = super::CohortAcceptEwma::new();
            for _ in 0..16 {
                ctrl.observe_frame_seqs(8, 0, 56, 8);
                let _ = ctrl.select_k_deterministic(8, 8, false);
            }
            for live in 2..=16usize {
                assert_eq!(
                    super::resolve_cohort_width_with_controller(
                        true, true, false, None, live, 7, &ctrl
                    ),
                    7,
                    "DFlash2 must keep full gamma=7 (K=8) at C={live}"
                );
            }
        }
    }

    #[test]
    fn harmonize_merges_singleton_k4_and_k5_into_one_k4_frame() {
        let candidates = [(0, Some(4)), (1, Some(5))];
        assert_eq!(
            choose_harmonized_width(&candidates),
            Some(4),
            "1xK4 + 1xK5 (2 serial fwds, 4.5 rows/fwd) must harmonize to 2xK4 (1 batched fwd, 8.0 rows/fwd)"
        );
    }

    #[test]
    fn harmonize_merges_balanced_k4_k5_split_into_one_k4_frame() {
        let candidates = [
            (0, Some(5)),
            (1, Some(5)),
            (2, Some(5)),
            (3, Some(4)),
            (4, Some(4)),
            (5, Some(4)),
        ];
        assert_eq!(
            choose_harmonized_width(&candidates),
            Some(4),
            "3xK5 + 3xK4 (4 fwds, 6.75 rows/fwd) must harmonize to 6xK4 (1 fwd, 24.0 rows/fwd)"
        );
    }

    #[test]
    fn harmonize_keeps_k5_when_single_k2_outlier_would_hurt_throughput() {
        let candidates = [
            (0, Some(5)),
            (1, Some(5)),
            (2, Some(5)),
            (3, Some(5)),
            (4, Some(5)),
            (5, Some(2)),
        ];
        assert_eq!(
            choose_harmonized_width(&candidates),
            None,
            "5xK5 + 1xK2 (2 fwds, 13.5 rows/fwd) beats 6xK2 (1 fwd, 12.0 rows/fwd)"
        );
    }

    #[test]
    fn harmonize_is_noop_when_cohort_already_uniform_or_singleton() {
        assert_eq!(choose_harmonized_width(&[(0, Some(5)), (1, Some(5))]), None);
        assert_eq!(choose_harmonized_width(&[(0, Some(4)), (1, None)]), None);
    }

    #[test]
    fn draft_depth_policy_is_uniform_only_when_batched_and_concurrent() {
        assert_eq!(
            DraftDepthPolicy::for_cohort(true, 2),
            DraftDepthPolicy::CohortUniform
        );
        assert_eq!(
            DraftDepthPolicy::for_cohort(true, 6),
            DraftDepthPolicy::CohortUniform
        );
        assert_eq!(
            DraftDepthPolicy::for_cohort(true, 1),
            DraftDepthPolicy::PerSequence
        );
        assert_eq!(
            DraftDepthPolicy::for_cohort(false, 6),
            DraftDepthPolicy::PerSequence
        );
        assert_eq!(DraftDepthPolicy::CohortUniform.resolve(0x1234, 4), 4);
    }

    #[test]
    fn dflash_preserves_gamma7_under_gumbel_while_native_mtp_uses_stochastic_depth() {
        // Under FLUX_SPEC_VERIFY=gumbel, adaptive_depth::stochastic_depth may
        // recommend a truncated depth (e.g. 3 or 1). DFlash emits its full
        // block in one pass and must keep gamma=7, whereas native MTP must
        // respect stochastic_depth at C=1 and fixed width at C>=2 fixed-K5.
        assert_eq!(
            resolve_effective_depth(true, false, false, 1, 7, Some(3)),
            7
        );
        assert_eq!(
            resolve_effective_depth(true, false, false, 1, 7, Some(1)),
            7
        );
        assert_eq!(
            resolve_effective_depth(false, false, false, 1, 7, Some(3)),
            3
        );
        assert_eq!(
            resolve_effective_depth(false, true, false, 1, 4, Some(2)),
            2
        );
        assert_eq!(
            resolve_effective_depth(false, true, false, 2, 4, Some(2)),
            4
        );
        assert_eq!(resolve_effective_depth(false, true, true, 2, 4, Some(2)), 4);
        assert_eq!(resolve_effective_depth(false, false, false, 1, 7, None), 7);
    }

    #[test]
    fn cohort_ewma_and_roofline_width_controller_adapt_by_acceptance_and_respect_overrides() {
        use spark_model::layers::qwen4_verify_lanes::{
            ADAPTIVE_MAX_FRAME_ROWS, ADAPTIVE_MAX_K, optimal_cohort_k,
        };
        let max_batched_drafts = ADAPTIVE_MAX_K - 1;
        let k7_or_cap_drafts = 6usize.min(max_batched_drafts);
        let ewma = CohortAcceptEwma::new();
        assert!((ewma.ewma_p_step() - 0.80).abs() < 1e-6);

        // Solo C=1 keeps configured width (e.g. 6 drafts -> K=7, or 10 -> K=11), even on a batched server.
        assert_eq!(
            resolve_cohort_width_capped(true, true, true, None, 1, 6, 0.55),
            6
        );
        assert_eq!(
            resolve_cohort_width_capped(true, true, true, None, 1, 10, 0.55),
            10
        );
        // Zero configured drafts returns 0 cleanly without panicking.
        assert_eq!(
            resolve_cohort_width_capped(true, true, true, None, 8, 0, 0.80),
            0
        );

        // High-acceptance cohort with configured=6 reaches K=7 (6 drafts, bounded by ADAPTIVE_MAX_K-1)
        // across c=2..16, and wider solo configs (e.g. 10) cap to ADAPTIVE_MAX_K-1 (K=8 -> 7 drafts).
        for live in [2usize, 4, 8, 16] {
            let expected_c = adaptive_drafts(live, k7_or_cap_drafts);
            assert_eq!(
                resolve_cohort_width_capped(true, true, true, None, live, 6, 1.0),
                expected_c,
                "high-accept configured=6 live={live}"
            );
            let expected_wide = adaptive_drafts(live, max_batched_drafts);
            assert_eq!(
                resolve_cohort_width_capped(true, true, true, None, live, 10, 1.0),
                expected_wide,
                "high-accept configured=10 caps to ADAPTIVE_MAX_K-1 live={live}"
            );
        }

        // Feed genuinely low-acceptance frames (8 seqs, 2 accepted, 8 rejections -> p=0.20).
        for _ in 0..16 {
            ewma.observe_frame_seqs(8, 2, 48, 8);
        }
        let p_low = ewma.ewma_p_step();
        assert!(
            p_low < 0.30,
            "ewma should converge below 0.30 on low-accept frames, got {p_low}"
        );
        for live in [2usize, 4, 8, 16] {
            assert!(
                resolve_cohort_width_capped(true, true, true, None, live, 6, p_low) <= 2,
                "low-accept p={p_low} live={live} must narrow cleanly"
            );
        }

        // Exact roofline delegation & single-frame <=128-row invariant across concurrency and p.
        for live in [2usize, 4, 8, 16, 24, 36, 64] {
            for configured in [4usize, 6] {
                let profile_drafts = adaptive_drafts(live, configured.min(max_batched_drafts));
                let max_k = (profile_drafts + 1).min(ADAPTIVE_MAX_K);
                for p in [0.25_f32, 0.60, 0.80, 0.98] {
                    let got =
                        resolve_cohort_width_capped(true, true, true, None, live, configured, p);
                    let want = optimal_cohort_k(live, p, max_k)
                        .saturating_sub(1)
                        .clamp(1, profile_drafts);
                    assert_eq!(got, want, "live={live} configured={configured} p={p}");
                    assert!(
                        live * (got + 1) <= ADAPTIVE_MAX_FRAME_ROWS,
                        "live={live} K={} exceeds 128-row single frame",
                        got + 1
                    );
                }
            }
        }

        // Disabling ATLAS_QWEN4_COHORT_ADAPTIVE_K restores profile ceiling for A/B runs.
        assert_eq!(
            resolve_cohort_width_capped(true, true, false, None, 8, 6, p_low),
            adaptive_drafts(8, k7_or_cap_drafts)
        );
        // Fixed-K5 mode without ATLAS_QWEN4_BATCHED_ADAPTIVE_K=1 caps batched cohorts to K=5 (4 drafts).
        assert_eq!(
            resolve_cohort_width_capped(true, false, true, None, 8, 6, 0.95),
            4
        );
        assert_eq!(
            resolve_cohort_width_capped(true, false, true, None, 8, 4, p_low),
            4
        );
        // Explicit ATLAS_QWEN4_BATCHED_K5_LADDER always overrides the roofline controller.
        let ladder = [(4usize, 3usize), (16usize, 1usize)];
        assert_eq!(
            resolve_cohort_width_capped(true, true, true, Some(&ladder), 4, 6, 0.90),
            3
        );
        assert_eq!(
            resolve_cohort_width_capped(true, true, true, Some(&ladder), 16, 6, 0.90),
            1
        );
    }

    #[test]
    fn cohort_controller_enforces_dwell_hysteresis_and_unbiased_ewma() {
        let max_k = spark_model::layers::qwen4_verify_lanes::ADAPTIVE_MAX_K.min(7);
        let ctrl = CohortAcceptEwma::new();
        // Warm-start at c=8 opens at full cohort max_k (K=7 when ADAPTIVE_MAX_K=8).
        assert_eq!(ctrl.select_k_deterministic(8, max_k, false), max_k);
        // Repeated calls without a new settled frame are idempotent.
        assert_eq!(ctrl.select_k_deterministic(8, max_k, false), max_k);

        // Sustained 0-accept frames must NOT prematurely drop K before MIN_DWELL_STEPS (6).
        let mut cur_k = max_k;
        for step in 1..=5 {
            let d = cur_k - 1;
            ctrl.observe_frame_seqs(8, 0, 8 * d, 8);
            cur_k = ctrl.select_k_deterministic(8, max_k, false);
            assert_eq!(cur_k, max_k, "dwell must hold K={max_k} on step {step}");
        }
        // Continued 0-accept frames collapse cleanly to K=2 with minimal transitions.
        let mut down_transitions = 0usize;
        for _ in 6..=16 {
            let d = cur_k - 1;
            ctrl.observe_frame_seqs(8, 0, 8 * d, 8);
            let next_k = ctrl.select_k_deterministic(8, max_k, false);
            if next_k != cur_k {
                down_transitions += 1;
                cur_k = next_k;
            }
        }
        assert_eq!(cur_k, 2, "sustained zero accepts must collapse to K=2");
        assert!(
            down_transitions <= 2,
            "collapse should avoid intermediate graph churn, got {down_transitions} transitions"
        );

        // Sustained full-accept frames recover smoothly all the way back up to max_k (K=7).
        let mut recovered = cur_k;
        let mut up_transitions = 0usize;
        for _ in 0..24 {
            let d = recovered - 1;
            ctrl.observe_frame_seqs(8, 8 * d, 8 * d, 0);
            let next_k = ctrl.select_k_deterministic(8, max_k, false);
            if next_k != recovered {
                up_transitions += 1;
                recovered = next_k;
            }
        }
        assert_eq!(
            recovered, max_k,
            "sustained full accepts must recover back to K={max_k}"
        );
        assert!(
            up_transitions <= 3,
            "recovery should keep CUDA graph transitions minimal, got {up_transitions}"
        );

        // Unbiased ratio-of-EWMAs avoids Jensen downward bias on small cohorts (c=2):
        // alternating 0/2 (1 trial/seq) and 2*d/2*d (d trials/seq) has true MLE d/(d+1)
        // (0.80 at K=5, 0.857 at K=7), whereas naive per-frame ratio averaging collapses to 0.50.
        for d in [4usize, 6] {
            let unbiased = CohortAcceptEwma::new();
            for _ in 0..40 {
                unbiased.observe_frame_seqs(2, 0, 2 * d, 2);
                unbiased.observe_frame_seqs(2, 2 * d, 2 * d, 0);
            }
            let true_mle = d as f32 / (d + 1) as f32;
            assert!(
                unbiased.ewma_p_step() > true_mle - 0.025,
                "d={d}: ratio-of-EWMAs must stay near true MLE {true_mle:.3}, got {}",
                unbiased.ewma_p_step()
            );
        }
    }

    #[test]
    fn cohort_controller_warm_starts_on_new_concurrency_bursts_and_idle_gaps() {
        let max_k = spark_model::layers::qwen4_verify_lanes::ADAPTIVE_MAX_K.min(7);
        let ctrl = CohortAcceptEwma::new();
        // Drive c=2 down to K=2 on sustained zero-accept frames.
        assert_eq!(ctrl.select_k_deterministic(2, max_k, false), max_k);
        for _ in 0..16 {
            ctrl.observe_frame_seqs(2, 0, 2 * (max_k - 1), 2);
            let _ = ctrl.select_k_deterministic(2, max_k, false);
        }
        assert_eq!(ctrl.select_k_deterministic(2, max_k, false), 2);
        assert!(ctrl.ewma_p_step() < 0.30);

        // 1. Sharp concurrency ramp (c=2 -> c=16) warm-starts at max_k, p=0.80 instead of K=2.
        assert_eq!(ctrl.select_k_deterministic(16, max_k, false), max_k);
        assert!((ctrl.ewma_p_step() - 0.80).abs() < 1e-6);

        // Drive c=16 down to K=2, then drain 16 -> 15 -> 14: drain keeps learned K without reset.
        for _ in 0..16 {
            ctrl.observe_frame_seqs(16, 0, 16 * (max_k - 1), 16);
            let _ = ctrl.select_k_deterministic(16, max_k, false);
        }
        assert_eq!(ctrl.select_k_deterministic(16, max_k, false), 2);
        assert_eq!(ctrl.select_k_deterministic(15, max_k, false), 2);
        assert_eq!(ctrl.select_k_deterministic(14, max_k, false), 2);

        // 2. Idle gap (`idle_expired = true`) warm-starts a fresh burst back to max_k.
        assert_eq!(ctrl.select_k_deterministic(14, max_k, true), max_k);
        assert!((ctrl.ewma_p_step() - 0.80).abs() < 1e-6);

        // 3. Dropping through solo C=1 resets the next concurrent burst (e.g. C=24 -> K=5) warm.
        for _ in 0..16 {
            ctrl.observe_frame_seqs(14, 0, 14 * (max_k - 1), 14);
            let _ = ctrl.select_k_deterministic(14, max_k, false);
        }
        assert_eq!(ctrl.select_k_deterministic(14, max_k, false), 2);
        assert_eq!(
            resolve_cohort_width_with_controller(true, true, true, None, 1, 6, &ctrl),
            6
        );
        assert_eq!(ctrl.select_k_deterministic(24, 5, false), 5);
        assert!((ctrl.ewma_p_step() - 0.80).abs() < 1e-6);
    }

    mod batched_gumbel_verify_tests {
        use super::super::{
            presample_batched_gumbel_frame_ragged_with_policy,
            presample_batched_gumbel_frame_with_policy,
            stage_batched_gumbel_frame_if_eligible_with_policy,
        };
        use crate::scheduler::spec_policy_accept::{
            batched_gumbel_seq_eligible, presample_single_seq_gumbel_with_policy,
            presampled_content_row_ok, run_dflash_content_accept_at_row_with_presampled,
            stage_single_seq_gumbel_if_eligible_with_policy,
        };
        use crate::scheduler::stochastic_verify_walk::{
            SpecVerifyPolicy, run_gumbel_thinking_accept_at_row,
        };
        use crate::scheduler::think_spec_accept::{
            presampled_thinking_row_ok, run_dflash_thinking_accept_at_row_with_presampled,
        };
        use crate::scheduler::{ActiveSeq, ResponseSink, ThinkSpecCtx};
        use anyhow::Result;
        use spark_model::traits::{Model, SequenceState};
        use spark_runtime::gpu::DevicePtr;
        use spark_runtime::sampler::device_topk::GumbelVerifyRowSpec;
        use std::sync::Mutex;
        use std::sync::atomic::{AtomicUsize, Ordering};

        const VOCAB: usize = 128;
        const THINK_START: u32 = 4;
        const THINK_END: u32 = 5;
        const TOOL_CALL: u32 = 6;
        const EOS_ID: u32 = 127;
        const LOGITS_BASE: u64 = 0x20_0000;

        fn think_ctx<'a>(enabled: bool, suppress: &'a [u32], adaptive: bool) -> ThinkSpecCtx<'a> {
            ThinkSpecCtx {
                enabled,
                code_fence_token: None,
                reflection_suppress_ids: suppress,
                adaptive_sampling: adaptive,
                honor_eos_inside_thinking: Some(false),
            }
        }

        fn make_gumbel_seq(slot: usize, inside_thinking: bool, seed: u64) -> ActiveSeq {
            let mut a =
                crate::scheduler::test_fixtures::active_seq(ResponseSink::Blocking(None), slot);
            a.seq.prompt_len = 8;
            a.seq.seq_len = 8;
            a.output_tokens.clear();
            a.temperature = 0.8;
            a.top_k = 20;
            a.top_p = 0.95;
            a.min_p = 0.0;
            a.top_n_sigma = 0.0;
            a.repetition_penalty = 1.0;
            a.presence_penalty = 0.0;
            a.frequency_penalty = 0.0;
            a.lz_penalty = 0.0;
            a.dry_multiplier = 0.0;
            a.logit_bias.clear();
            a.seed = Some(seed);
            a.max_output_tokens = 200;
            a.remaining = 200;
            a.eos_tokens = vec![EOS_ID];
            a.inside_thinking = inside_thinking;
            a.enable_thinking = inside_thinking;
            a.think_ended = !inside_thinking;
            a.think_just_ended = false;
            a.think_start_token = Some(THINK_START);
            a.think_end_token = Some(THINK_END);
            a.tool_call_start_token = Some(TOOL_CALL);
            a
        }

        fn bf16_row_with_masked_leader(masked: u32, fallback_tok: u32) -> Vec<u8> {
            let mut f32s = vec![-8.0f32; VOCAB];
            f32s[masked as usize] = 50.0;
            f32s[fallback_tok as usize] = 40.0;
            f32s.into_iter()
                .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
                .collect()
        }

        struct MockGumbelModel {
            inner: Box<dyn Model>,
            rows_bf16: Vec<Vec<u8>>,
            gumbel_calls: AtomicUsize,
            recorded_reqs: Mutex<Vec<Vec<GumbelVerifyRowSpec>>>,
            staged_reqs: Mutex<Vec<Vec<GumbelVerifyRowSpec>>>,
            in_graph_tokens: Mutex<Option<Vec<u32>>>,
            d2h_calls: AtomicUsize,
            return_none: bool,
        }

        impl MockGumbelModel {
            fn new(rows_bf16: Vec<Vec<u8>>, return_none: bool) -> Self {
                Self {
                    inner: Box::new(crate::scheduler::test_fixtures::model::recording()),
                    rows_bf16,
                    gumbel_calls: AtomicUsize::new(0),
                    recorded_reqs: Mutex::new(Vec::new()),
                    staged_reqs: Mutex::new(Vec::new()),
                    in_graph_tokens: Mutex::new(None),
                    d2h_calls: AtomicUsize::new(0),
                    return_none,
                }
            }
        }

        impl Model for MockGumbelModel {
            fn prefill(&self, t: &[u32], s: &mut SequenceState, st: u64) -> Result<DevicePtr> {
                self.inner.prefill(t, s, st)
            }
            fn prefill_chunk(
                &self,
                t: &[u32],
                s: &mut SequenceState,
                o: usize,
                l: usize,
                last: bool,
                st: u64,
            ) -> Result<DevicePtr> {
                self.inner.prefill_chunk(t, s, o, l, last, st)
            }
            fn decode(&self, t: u32, s: &mut SequenceState, st: u64) -> Result<DevicePtr> {
                self.inner.decode(t, s, st)
            }
            fn decode_batch(
                &self,
                t: &[u32],
                s: &mut [&mut SequenceState],
                st: u64,
            ) -> Result<DevicePtr> {
                self.inner.decode_batch(t, s, st)
            }
            fn vocab_size(&self) -> usize {
                VOCAB
            }
            fn bind_gpu_to_thread(&self) -> Result<()> {
                Ok(())
            }
            fn synchronize_stream(&self, _: u64) -> Result<()> {
                Ok(())
            }
            fn alloc_sequence(&self) -> Result<SequenceState> {
                self.inner.alloc_sequence()
            }
            fn free_sequence(&self, s: &mut SequenceState) -> Result<()> {
                self.inner.free_sequence(s)
            }
            fn compact_sequence(&self, s: &mut SequenceState, k: usize) -> Result<bool> {
                self.inner.compact_sequence(s, k)
            }
            fn cache_sequence(&self, s: &SequenceState) {
                self.inner.cache_sequence(s)
            }
            fn logits_buffer_ptr(&self) -> DevicePtr {
                DevicePtr(LOGITS_BASE)
            }
            fn copy_logits_to_host(&self, ptr: DevicePtr, out: &mut [u8]) -> Result<()> {
                self.d2h_calls.fetch_add(1, Ordering::SeqCst);
                let row_bytes = VOCAB * 2;
                let byte_off = (ptr.0 - LOGITS_BASE) as usize;
                let start_row = byte_off / row_bytes;
                let count_rows = out.len() / row_bytes;
                for r in 0..count_rows {
                    let src = &self.rows_bf16[start_row + r];
                    out[r * row_bytes..(r + 1) * row_bytes].copy_from_slice(src);
                }
                Ok(())
            }
            fn argmax_on_device(&self, p: DevicePtr, st: u64) -> Result<u32> {
                self.inner.argmax_on_device(p, st)
            }
            fn argmax_batch(&self, p: DevicePtr, n: usize, st: u64) -> Result<Vec<u32>> {
                self.inner.argmax_batch(p, n, st)
            }
            fn sample_gumbel_verify_rows_batched(
                &self,
                logits_base: DevicePtr,
                vocab_stride: usize,
                reqs: &[GumbelVerifyRowSpec],
            ) -> Result<Option<Vec<u32>>> {
                assert_eq!(logits_base, DevicePtr(LOGITS_BASE));
                assert_eq!(vocab_stride, VOCAB);
                self.gumbel_calls.fetch_add(1, Ordering::SeqCst);
                self.recorded_reqs.lock().unwrap().push(reqs.to_vec());
                if self.return_none {
                    return Ok(None);
                }
                Ok(Some(
                    reqs.iter().map(|r| 20 + (r.row_idx as u32 % 50)).collect(),
                ))
            }
            fn stage_next_verify_gumbel_rows(&self, reqs: &[GumbelVerifyRowSpec]) {
                self.staged_reqs.lock().unwrap().push(reqs.to_vec());
            }
            fn take_in_graph_gumbel_verify_tokens(&self) -> Option<Vec<u32>> {
                self.in_graph_tokens.lock().unwrap().take()
            }
            fn hidden_after_norm(&self) -> DevicePtr {
                DevicePtr::NULL
            }
            fn decode_verify(&self, t: &[u32], s: &mut SequenceState, st: u64) -> Result<Vec<u32>> {
                self.inner.decode_verify(t, s, st)
            }
            fn run_mtp_propose_multi(
                &self,
                tok: u32,
                pos: usize,
                k: usize,
                s: &mut SequenceState,
                st: u64,
                mask: Option<&[i32]>,
            ) -> Result<Vec<u32>> {
                Model::run_mtp_propose_multi(&*self.inner, tok, pos, k, s, st, mask)
            }
            fn trim_proposer_state(&self, s: &mut SequenceState, n: usize, st: u64) -> Result<()> {
                self.inner.trim_proposer_state(s, n, st)
            }
            fn decode_verify_graphed(
                &self,
                t: &[u32; 2],
                s: &mut SequenceState,
                st: u64,
            ) -> Result<[u32; 2]> {
                self.inner.decode_verify_graphed(t, s, st)
            }
            fn decode_verify_graphed_k3(
                &self,
                t: &[u32; 3],
                s: &mut SequenceState,
                st: u64,
            ) -> Result<[u32; 3]> {
                self.inner.decode_verify_graphed_k3(t, s, st)
            }
            fn decode_verify_graphed_k4(
                &self,
                t: &[u32; 4],
                s: &mut SequenceState,
                st: u64,
            ) -> Result<[u32; 4]> {
                self.inner.decode_verify_graphed_k4(t, s, st)
            }
            fn checkpoint_ssm_states(&self, s: &mut SequenceState) -> Result<()> {
                self.inner.checkpoint_ssm_states(s)
            }
            fn rollback_ssm_states(&self, s: &mut SequenceState, n: usize) -> Result<()> {
                self.inner.rollback_ssm_states(s, n)
            }
            fn has_ssm_layers(&self) -> bool {
                false
            }
            fn generate_speculative(
                &self,
                p: &[u32],
                sp: &spark_runtime::sampler::SamplingParams,
                nd: usize,
            ) -> Result<spark_model::engine::GenerateResult> {
                self.inner.generate_speculative(p, sp, nd)
            }
            fn has_proposer(&self) -> bool {
                true
            }
            fn has_self_speculative(&self) -> bool {
                false
            }
            fn decode_draft(&self, t: u32, s: &mut SequenceState, st: u64) -> Result<DevicePtr> {
                self.inner.decode_draft(t, s, st)
            }
            fn save_hidden_for_mtp(&self, r: usize, st: u64) -> Result<()> {
                self.inner.save_hidden_for_mtp(r, st)
            }
            fn run_mtp_propose(
                &self,
                t: u32,
                p: usize,
                s: &mut SequenceState,
                st: u64,
            ) -> Result<Option<u32>> {
                self.inner.run_mtp_propose(t, p, s, st)
            }
        }

        #[test]
        fn batched_gumbel_eligibility_enforces_every_lossless_guard() {
            let ctx = think_ctx(true, &[], false);
            let thinking = make_gumbel_seq(0, true, 100);
            let content = make_gumbel_seq(1, false, 200);
            assert!(batched_gumbel_seq_eligible(&thinking, 5, &ctx));
            assert!(batched_gumbel_seq_eligible(&content, 5, &ctx));

            // Full-vocab with top_p >= 1.0 is allowed; top_k==0 with top_p < 1.0 is not.
            let mut full_vocab = make_gumbel_seq(1, false, 200);
            full_vocab.top_k = 0;
            full_vocab.top_p = 1.0;
            assert!(batched_gumbel_seq_eligible(&full_vocab, 5, &ctx));
            full_vocab.top_p = 0.95;
            assert!(!batched_gumbel_seq_eligible(&full_vocab, 5, &ctx));

            // Every single lossless exclusion must reject in both content and thinking.
            let check_ineligible = |mutate: fn(&mut ActiveSeq), label: &str| {
                let mut sc = make_gumbel_seq(1, false, 200);
                mutate(&mut sc);
                assert!(
                    !batched_gumbel_seq_eligible(&sc, 5, &ctx),
                    "expected ineligible in content: {label}"
                );
                let mut st = make_gumbel_seq(0, true, 100);
                mutate(&mut st);
                assert!(
                    !batched_gumbel_seq_eligible(&st, 5, &ctx),
                    "expected ineligible in thinking: {label}"
                );
            };
            check_ineligible(|s| s.temperature = 0.0, "greedy");
            check_ineligible(|s| s.temperature = f32::NAN, "nan_temp");
            check_ineligible(|s| s.min_p = 0.05, "min_p");
            check_ineligible(|s| s.top_n_sigma = 1.5, "top_n_sigma");
            check_ineligible(|s| s.presence_penalty = 0.2, "presence_penalty");
            check_ineligible(|s| s.frequency_penalty = 0.1, "frequency_penalty");
            check_ineligible(|s| s.repetition_penalty = 1.1, "repetition_penalty");
            check_ineligible(|s| s.lz_penalty = 0.3, "lz_penalty");
            check_ineligible(|s| s.dry_multiplier = 0.5, "dry_multiplier");
            check_ineligible(|s| s.logit_bias = vec![(10, -2.0)], "logit_bias");
            check_ineligible(|s| s.top_logprobs = Some(2), "logprobs");
            check_ineligible(|s| s.top_p = 0.0, "top_p_zero");
            check_ineligible(|s| s.top_k = 257, "top_k_over_256");

            // Context-level guards.
            assert!(!batched_gumbel_seq_eligible(
                &thinking,
                5,
                &think_ctx(true, &[9], false)
            ));
            assert!(!batched_gumbel_seq_eligible(
                &thinking,
                5,
                &think_ctx(true, &[], true)
            ));
            assert!(!batched_gumbel_seq_eligible(
                &thinking,
                5,
                &think_ctx(false, &[], false)
            ));

            // Thinking budget exact boundary: `thinking_tokens + k_verify <= budget` is ok; `>` is not.
            let mut budgeted = make_gumbel_seq(0, true, 100);
            budgeted.thinking_budget = Some(100);
            budgeted.thinking_tokens = 95;
            assert!(batched_gumbel_seq_eligible(&budgeted, 5, &ctx));
            budgeted.thinking_tokens = 96;
            assert!(!batched_gumbel_seq_eligible(&budgeted, 5, &ctx));

            // Thinking answer-runway exact boundary (`max_output_tokens = 200` -> runway 100):
            // frames approaching cutoff stay eligible (mid-frame crossing is caught row-by-row
            // by `emit_step` + `presampled_thinking_row_ok`), while row 0 at/past cutoff rejects.
            let mut runway = make_gumbel_seq(0, true, 100);
            runway.thinking_budget = None;
            runway.max_output_tokens = 200;
            runway.output_tokens = vec![10; 99];
            assert!(batched_gumbel_seq_eligible(&runway, 5, &ctx));
            runway.output_tokens = vec![10; 100];
            assert!(!batched_gumbel_seq_eligible(&runway, 5, &ctx));

            // Content phase-specific guards.
            let mut c = make_gumbel_seq(1, false, 200);
            c.think_just_ended = true;
            assert!(!batched_gumbel_seq_eligible(&c, 5, &ctx));
            c.think_just_ended = false;
            c.suppress_tool_call = true;
            assert!(!batched_gumbel_seq_eligible(&c, 5, &ctx));
            c.suppress_tool_call = false;
            c.require_tool_call = true;
            assert!(!batched_gumbel_seq_eligible(&c, 5, &ctx));
            c.require_tool_call = false;
            c.min_tokens = 10;
            assert!(!batched_gumbel_seq_eligible(&c, 5, &ctx));
        }

        #[test]
        fn presampled_dynamic_guards_invalidate_on_state_transition_and_masked_tokens() {
            let ctx = think_ctx(true, &[], false);
            let mut t = make_gumbel_seq(0, true, 10);
            let pos = crate::scheduler::stochastic_verify_walk::seq_abs_pos(&t);
            assert!(presampled_thinking_row_ok(&t, 12, VOCAB, pos, &ctx));
            assert!(!presampled_thinking_row_ok(&t, TOOL_CALL, VOCAB, pos, &ctx));
            assert!(!presampled_thinking_row_ok(
                &t,
                VOCAB as u32,
                VOCAB,
                pos,
                &ctx
            ));
            assert!(!presampled_thinking_row_ok(&t, 12, VOCAB, pos + 1, &ctx));
            t.force_end_thinking = true;
            assert!(!presampled_thinking_row_ok(&t, 12, VOCAB, pos, &ctx));
            t.force_end_thinking = false;
            t.inside_thinking = false;
            assert!(!presampled_thinking_row_ok(&t, 12, VOCAB, pos, &ctx));

            let mut c = make_gumbel_seq(1, false, 20);
            let cpos = crate::scheduler::stochastic_verify_walk::seq_abs_pos(&c);
            assert!(presampled_content_row_ok(&c, 12, VOCAB, cpos, &ctx));
            assert!(!presampled_content_row_ok(
                &c,
                THINK_START,
                VOCAB,
                cpos,
                &ctx
            ));
            assert!(!presampled_content_row_ok(&c, THINK_END, VOCAB, cpos, &ctx));
            assert!(!presampled_content_row_ok(
                &c,
                VOCAB as u32,
                VOCAB,
                cpos,
                &ctx
            ));
            assert!(!presampled_content_row_ok(&c, 12, VOCAB, cpos + 1, &ctx));
            c.suppress_tool_call = true;
            assert!(!presampled_content_row_ok(&c, 12, VOCAB, cpos, &ctx));
        }

        #[test]
        fn presample_batched_gumbel_frame_collects_eligible_rows_in_one_call() {
            let ctx = think_ctx(true, &[], false);
            let s0 = make_gumbel_seq(0, true, 111);
            let mut s1 = make_gumbel_seq(1, false, 222);
            s1.presence_penalty = 0.5; // ineligible
            let s2 = make_gumbel_seq(2, false, 333);
            let active = [s0, s1, s2];
            let idxs = [0usize, 1, 2];
            let model = MockGumbelModel::new(Vec::new(), false);

            // Non-Gumbel policy does zero GPU calls.
            assert!(
                presample_batched_gumbel_frame_with_policy(
                    &model,
                    &active,
                    &idxs,
                    5,
                    &ctx,
                    SpecVerifyPolicy::Argmax,
                )
                .unwrap()
                .is_none()
            );
            assert_eq!(model.gumbel_calls.load(Ordering::SeqCst), 0);

            // Gumbel policy launches ONE call with 10 rows: seq 0 (`0..5`) + seq 2 (`10..15`).
            let frame = presample_batched_gumbel_frame_with_policy(
                &model,
                &active,
                &idxs,
                5,
                &ctx,
                SpecVerifyPolicy::Gumbel,
            )
            .unwrap()
            .expect("batched gumbel frame must engage");
            assert_eq!(model.gumbel_calls.load(Ordering::SeqCst), 1);
            assert_eq!(frame.seq_offsets, vec![Some(0), None, Some(5)]);
            assert_eq!(frame.tokens.len(), 10);
            let recorded = model.recorded_reqs.lock().unwrap();
            let row_idxs: Vec<usize> = recorded[0].iter().map(|r| r.row_idx).collect();
            assert_eq!(row_idxs, vec![0, 1, 2, 3, 4, 10, 11, 12, 13, 14]);
            assert_eq!(recorded[0][0].abs_pos, 8);
            assert_eq!(recorded[0][4].abs_pos, 12);
            assert_eq!(recorded[0][5].abs_pos, 8);
            assert_eq!(recorded[0][9].abs_pos, 12);

            // Unsupported model (`Ok(None)`) returns `Ok(None)` cleanly.
            let unsupported = MockGumbelModel::new(Vec::new(), true);
            assert!(
                presample_batched_gumbel_frame_with_policy(
                    &unsupported,
                    &active,
                    &idxs,
                    5,
                    &ctx,
                    SpecVerifyPolicy::Gumbel,
                )
                .unwrap()
                .is_none()
            );
        }

        #[test]
        fn presample_batched_gumbel_frame_ragged_indexes_cumulative_row_bases_and_k1() {
            let ctx = think_ctx(true, &[], false);
            let s0 = make_gumbel_seq(0, true, 111); // K=7, row_base=0 (eligible)
            let mut s1 = make_gumbel_seq(1, false, 222);
            s1.presence_penalty = 0.5; // K=3, row_base=7 (ineligible -> skipped)
            let s2 = make_gumbel_seq(2, false, 333); // K=1, row_base=10 (anchor-only eligible)
            let s3 = make_gumbel_seq(3, false, 444); // K=5, row_base=11 (eligible)
            let active = [s0, s1, s2, s3];
            let idxs = [0usize, 1, 2, 3];
            let k_per_seq = [7usize, 3, 1, 5];
            let row_bases = [0usize, 7, 10, 11];
            let model = MockGumbelModel::new(Vec::new(), false);

            let frame = presample_batched_gumbel_frame_ragged_with_policy(
                &model,
                &active,
                &idxs,
                &k_per_seq,
                &row_bases,
                &ctx,
                None,
                SpecVerifyPolicy::Gumbel,
            )
            .unwrap()
            .expect("ragged batched gumbel frame must engage");

            assert_eq!(model.gumbel_calls.load(Ordering::SeqCst), 1);
            assert_eq!(frame.seq_offsets, vec![Some(0), None, Some(7), Some(8)]);
            assert_eq!(frame.tokens.len(), 7 + 1 + 5);
            let recorded = model.recorded_reqs.lock().unwrap();
            let row_idxs: Vec<usize> = recorded[0].iter().map(|r| r.row_idx).collect();
            assert_eq!(row_idxs, vec![0, 1, 2, 3, 4, 5, 6, 10, 11, 12, 13, 14, 15]);
            // Verify abs_pos resets cleanly for each sequence (K=1 has single abs_pos=8).
            assert_eq!(recorded[0][0].abs_pos, 8);
            assert_eq!(recorded[0][6].abs_pos, 14);
            assert_eq!(recorded[0][7].abs_pos, 8);
            assert_eq!(recorded[0][8].abs_pos, 8);
            assert_eq!(recorded[0][12].abs_pos, 12);
        }

        #[test]
        fn presampled_walk_uses_zero_d2h_and_falls_back_cleanly_on_boundary_or_mask() {
            let ctx = think_ctx(true, &[], false);

            // A. Normal thinking walk accepting 2 drafts + bonus -> 0 logits D2H!
            {
                let model = MockGumbelModel::new(Vec::new(), false);
                let mut a = make_gumbel_seq(0, true, 10);
                let out = run_dflash_thinking_accept_at_row_with_presampled(
                    &model,
                    &mut a,
                    0,
                    &[10, 11],
                    &[99, 99, 99],
                    &ctx,
                    Some(&[10, 11, 12]),
                )
                .unwrap();
                assert_eq!(model.d2h_calls.load(Ordering::SeqCst), 0);
                assert_eq!(
                    (out.num_accepted, out.bonus, out.consumed_rows),
                    (2, Some(12), 3)
                );
                assert_eq!(a.output_tokens, vec![10, 11, 12]);
                assert!(a.inside_thinking);
            }

            // B. `</think>` boundary ends thinking cleanly on that row with 0 logits D2H.
            {
                let model = MockGumbelModel::new(Vec::new(), false);
                let mut a = make_gumbel_seq(0, true, 11);
                let out = run_dflash_thinking_accept_at_row_with_presampled(
                    &model,
                    &mut a,
                    0,
                    &[10, THINK_END],
                    &[99, 99, 99],
                    &ctx,
                    Some(&[10, THINK_END, 25]),
                )
                .unwrap();
                assert_eq!(model.d2h_calls.load(Ordering::SeqCst), 0);
                assert_eq!(
                    (out.num_accepted, out.bonus, out.consumed_rows),
                    (1, Some(THINK_END), 2)
                );
                assert!(!a.inside_thinking);
                assert!(a.think_ended);
                assert_eq!(a.output_tokens, vec![10, THINK_END]);
            }

            // C. Masked `<tool_call>` in pre-sampled row 1 invalidates and falls back to D2H.
            {
                let rows = vec![
                    bf16_row_with_masked_leader(TOOL_CALL, 31),
                    bf16_row_with_masked_leader(TOOL_CALL, 33),
                    bf16_row_with_masked_leader(TOOL_CALL, 35),
                ];
                let model = MockGumbelModel::new(rows, false);
                let mut a = make_gumbel_seq(0, true, 12);
                let out = run_dflash_thinking_accept_at_row_with_presampled(
                    &model,
                    &mut a,
                    0,
                    &[10, 15],
                    &[99, 99, 99],
                    &ctx,
                    Some(&[10, TOOL_CALL, 20]),
                )
                .unwrap();
                assert_eq!(model.d2h_calls.load(Ordering::SeqCst), 1);
                assert_eq!(
                    (out.num_accepted, out.bonus, out.consumed_rows),
                    (1, Some(33), 2)
                );
                assert_eq!(a.output_tokens, vec![10, 33]);
            }

            // D. Masked `</think>` in post-think content row 0 invalidates and falls back to D2H.
            {
                let rows = vec![
                    bf16_row_with_masked_leader(THINK_END, 44),
                    bf16_row_with_masked_leader(THINK_END, 45),
                ];
                let model = MockGumbelModel::new(rows, false);
                let mut a = make_gumbel_seq(1, false, 13);
                let out = run_dflash_content_accept_at_row_with_presampled(
                    &model,
                    &mut a,
                    0,
                    &[25],
                    &[99, 99],
                    &ctx,
                    Some(&[THINK_END, 25]),
                )
                .unwrap();
                assert_eq!(model.d2h_calls.load(Ordering::SeqCst), 1);
                assert_eq!(
                    (out.num_accepted, out.bonus, out.consumed_rows),
                    (0, Some(44), 1)
                );
                assert_eq!(a.output_tokens, vec![44]);
            }
        }

        #[test]
        fn in_graph_gumbel_staging_consumes_tokens_with_zero_extra_launch_and_falls_back_cleanly() {
            let ctx = think_ctx(true, &[], false);

            // 1. Batched ragged cohort: all eligible -> stages once, consumes in-graph with 0 extra launches.
            {
                let s0 = make_gumbel_seq(0, true, 111);
                let s1 = make_gumbel_seq(1, false, 222);
                let active = [s0, s1];
                let idxs = [0usize, 1];
                let k_per_seq = [4usize, 2];
                let row_bases = [0usize, 4];
                let model = MockGumbelModel::new(Vec::new(), false);

                let staged_mask = stage_batched_gumbel_frame_if_eligible_with_policy(
                    &model,
                    &active,
                    &idxs,
                    &k_per_seq,
                    &row_bases,
                    6,
                    &ctx,
                    SpecVerifyPolicy::Gumbel,
                );
                assert_eq!(staged_mask, Some(vec![true, true]));
                {
                    let staged_all = model.staged_reqs.lock().unwrap();
                    assert_eq!(staged_all.len(), 1);
                    let staged = &staged_all[0];
                    assert_eq!(staged.len(), 6);
                    assert_eq!(
                        staged.iter().map(|r| r.row_idx).collect::<Vec<_>>(),
                        vec![0, 1, 2, 3, 4, 5]
                    );
                }

                *model.in_graph_tokens.lock().unwrap() = Some(vec![70, 71, 72, 73, 80, 81]);
                let frame = presample_batched_gumbel_frame_ragged_with_policy(
                    &model,
                    &active,
                    &idxs,
                    &k_per_seq,
                    &row_bases,
                    &ctx,
                    staged_mask.as_deref(),
                    SpecVerifyPolicy::Gumbel,
                )
                .unwrap()
                .expect("in-graph tokens must be consumed");
                assert_eq!(model.gumbel_calls.load(Ordering::SeqCst), 0);
                assert_eq!(frame.tokens, vec![70, 71, 72, 73, 80, 81]);
                assert_eq!(frame.seq_offsets, vec![Some(0), Some(4)]);
            }

            // 2. Mixed cohort (Gumbel + host-policy/budget_nudge skip + greedy T==0):
            //    stages per-seq `temp > 0`, `temp == -1.0`, `temp == 0.0` in ONE unified
            //    frame, consuming in-graph tokens for eligible seqs with 0 extra GPU launches!
            {
                let s0 = make_gumbel_seq(0, true, 111); // K=4, row_base=0 (Gumbel)
                let mut s1 = make_gumbel_seq(1, true, 222);
                s1.force_end_thinking = true; // K=2, row_base=4 (HostPolicySkip -> temp=-1.0)
                let mut s2 = make_gumbel_seq(2, false, 333);
                s2.temperature = 0.0; // K=3, row_base=6 (GreedyArgmax -> temp=0.0)
                let s3 = make_gumbel_seq(3, false, 444); // K=2, row_base=9 (Gumbel)
                let active = [s0, s1, s2, s3];
                let idxs = [0usize, 1, 2, 3];
                let k_per_seq = [4usize, 2, 3, 2];
                let row_bases = [0usize, 4, 6, 9];
                let model = MockGumbelModel::new(Vec::new(), false);

                let staged_mask = stage_batched_gumbel_frame_if_eligible_with_policy(
                    &model,
                    &active,
                    &idxs,
                    &k_per_seq,
                    &row_bases,
                    11,
                    &ctx,
                    SpecVerifyPolicy::Gumbel,
                );
                assert_eq!(staged_mask, Some(vec![true, false, false, true]));
                {
                    let staged_all = model.staged_reqs.lock().unwrap();
                    assert_eq!(staged_all.len(), 1);
                    let staged = &staged_all[0];
                    assert_eq!(staged.len(), 11);
                    assert_eq!(
                        staged.iter().map(|r| r.row_idx).collect::<Vec<_>>(),
                        (0..11).collect::<Vec<_>>()
                    );
                    assert!(staged[0..4].iter().all(|r| r.temperature > 0.0));
                    assert!(staged[4..6].iter().all(|r| r.temperature == -1.0));
                    assert!(staged[6..9].iter().all(|r| r.temperature == 0.0));
                    assert!(staged[9..11].iter().all(|r| r.temperature > 0.0));
                }

                // Simulate unified GPU kernel output: valid tokens for Gumbel & greedy rows,
                // `u32::MAX` sentinel for the skipped `force_end_thinking` rows (4..6).
                *model.in_graph_tokens.lock().unwrap() =
                    Some(vec![10, 11, 12, 13, u32::MAX, u32::MAX, 30, 31, 32, 40, 41]);
                let frame = presample_batched_gumbel_frame_ragged_with_policy(
                    &model,
                    &active,
                    &idxs,
                    &k_per_seq,
                    &row_bases,
                    &ctx,
                    staged_mask.as_deref(),
                    SpecVerifyPolicy::Gumbel,
                )
                .unwrap()
                .expect("mixed cohort must consume in-graph Gumbel tokens for eligible seqs");
                assert_eq!(
                    model.gumbel_calls.load(Ordering::SeqCst),
                    0,
                    "mixed cohort must consume in-graph tokens with zero extra GPU launches"
                );
                assert_eq!(frame.seq_offsets, vec![Some(0), None, None, Some(9)]);
                assert_eq!(&frame.tokens[0..4], &[10, 11, 12, 13]);
                assert_eq!(&frame.tokens[9..11], &[40, 41]);
            }

            // 2b. Pure greedy cohort -> skips heap staging (verify_qwen4_batched auto-stages temp=0.0).
            {
                let mut g0 = make_gumbel_seq(0, false, 111);
                g0.temperature = 0.0;
                let mut g1 = make_gumbel_seq(1, false, 222);
                g1.temperature = 0.0;
                let active = [g0, g1];
                let model = MockGumbelModel::new(Vec::new(), false);
                let staged_mask = stage_batched_gumbel_frame_if_eligible_with_policy(
                    &model,
                    &active,
                    &[0, 1],
                    &[4, 2],
                    &[0, 4],
                    6,
                    &ctx,
                    SpecVerifyPolicy::Gumbel,
                );
                assert_eq!(staged_mask, None);
                assert!(model.staged_reqs.lock().unwrap().is_empty());
            }

            // 3. C=1 single-sequence staging (eligible Gumbel + budget-nudge skip) &
            //    zero-extra-launch consumption (row_base == 0 only).
            {
                let a = make_gumbel_seq(0, false, 333);
                let model = MockGumbelModel::new(Vec::new(), false);
                stage_single_seq_gumbel_if_eligible_with_policy(
                    &model,
                    &a,
                    5,
                    &ctx,
                    SpecVerifyPolicy::Gumbel,
                );
                assert_eq!(model.staged_reqs.lock().unwrap().len(), 1);
                assert_eq!(model.staged_reqs.lock().unwrap()[0].len(), 5);
                assert!(
                    model.staged_reqs.lock().unwrap()[0]
                        .iter()
                        .all(|r| r.temperature > 0.0)
                );

                // Ineligible T>0 C=1 (e.g. force_end_thinking) stages temp=-1.0 skip so the
                // C=1 CUDA graph key stays constant across token 128 budget nudge.
                let mut nudged = make_gumbel_seq(0, true, 333);
                nudged.force_end_thinking = true;
                stage_single_seq_gumbel_if_eligible_with_policy(
                    &model,
                    &nudged,
                    5,
                    &ctx,
                    SpecVerifyPolicy::Gumbel,
                );
                {
                    let staged_all = model.staged_reqs.lock().unwrap();
                    assert_eq!(staged_all.len(), 2);
                    assert_eq!(staged_all[1].len(), 5);
                    assert!(staged_all[1].iter().all(|r| r.temperature == -1.0));
                }

                *model.in_graph_tokens.lock().unwrap() = Some(vec![41, 42, 43, 44, 45]);
                let tokens = presample_single_seq_gumbel_with_policy(
                    &model,
                    &a,
                    0,
                    5,
                    &ctx,
                    SpecVerifyPolicy::Gumbel,
                )
                .unwrap()
                .expect("C=1 in-graph tokens must be consumed");
                assert_eq!(model.gumbel_calls.load(Ordering::SeqCst), 0);
                assert_eq!(tokens, vec![41, 42, 43, 44, 45]);

                // When eager fallback yields None from take_in_graph_gumbel_verify_tokens,
                // presample_single_seq_gumbel falls back to sample_gumbel_verify_rows_batched.
                let fallback = presample_single_seq_gumbel_with_policy(
                    &model,
                    &a,
                    0,
                    5,
                    &ctx,
                    SpecVerifyPolicy::Gumbel,
                )
                .unwrap()
                .expect("fallback batched launch must succeed");
                assert_eq!(model.gumbel_calls.load(Ordering::SeqCst), 1);
                assert_eq!(fallback.len(), 5);
            }
        }

        #[test]
        fn run_gumbel_thinking_accept_at_row_performs_lazy_per_row_d2h() {
            let ctx = think_ctx(true, &[], false);

            // A. Forced budget nudge + armed `</think>` across 5 verify rows: ZERO D2H calls!
            {
                let model = MockGumbelModel::new(Vec::new(), false);
                let mut a = make_gumbel_seq(0, true, 55);
                a.max_output_tokens = 64;
                a.remaining = 64;
                a.force_end_thinking = true;
                a.budget_nudge = crate::scheduler::think_budget_nudge::BudgetNudge::plan(
                    Some(std::sync::Arc::from([30u32, 31, 32])),
                    0,
                    64,
                );
                let out = run_gumbel_thinking_accept_at_row(
                    &model,
                    &mut a,
                    0,
                    &[30, 31, 32, THINK_END],
                    5,
                    &ctx,
                );
                assert_eq!(model.d2h_calls.load(Ordering::SeqCst), 0);
                assert_eq!(
                    (out.num_accepted, out.bonus, out.consumed_rows),
                    (3, Some(THINK_END), 4)
                );
                assert_eq!(a.output_tokens, vec![30, 31, 32, THINK_END]);
            }

            // B. Unforced row 0 mismatch over a 5-row verify frame: copies ONLY row 0 (1 D2H).
            //    `rows_bf16` only supplies row 0, proving rows 1..5 are never downloaded.
            {
                let rows = vec![bf16_row_with_masked_leader(TOOL_CALL, 31)];
                let model = MockGumbelModel::new(rows, false);
                let mut a = make_gumbel_seq(0, true, 56);
                a.max_output_tokens = 64;
                a.remaining = 64;
                let out = run_gumbel_thinking_accept_at_row(
                    &model,
                    &mut a,
                    0,
                    &[99, 98, 97, 96],
                    5,
                    &ctx,
                );
                assert_eq!(model.d2h_calls.load(Ordering::SeqCst), 1);
                assert_eq!(
                    (out.num_accepted, out.bonus, out.consumed_rows),
                    (0, Some(31), 1)
                );
                assert_eq!(a.output_tokens, vec![31]);
            }
        }
    }

    mod batched_rejection_verify_tests {
        use super::super::{
            batched_verify_proposer_eligible_with_policy, clamp_drafts_to_max_seq_len_count,
            clamp_pending_drafts_for_max_seq_len, clamp_pending_drafts_to_model_max_seq_len,
            presample_batched_rejection_frame_ragged_with_policy, resolve_dflash_gumbel_num_drafts,
            stage_batched_rejection_frame_if_eligible_with_policy,
        };
        use crate::scheduler::stochastic_verify_walk::SpecVerifyPolicy;
        use crate::scheduler::{ActiveSeq, ResponseSink, ThinkSpecCtx};
        use anyhow::Result;
        use spark_model::dflash_sampling::DflashDraftSampleMethod;
        use spark_model::speculative::DraftProposerKind;
        use spark_model::traits::{Model, SequenceState};
        use spark_model::{
            HOST_POLICY_FALLBACK_TOKEN, PackedRejectionWord, REJECTION_BONUS_Q_ROW_IDX,
            REJECTION_NO_MASK_TOKEN, RejectionVerifyRowSpec,
        };
        use spark_runtime::gpu::DevicePtr;
        use std::sync::Mutex;
        use std::sync::atomic::{AtomicUsize, Ordering};

        const VOCAB: usize = 128;
        const THINK_START: u32 = 4;
        const THINK_END: u32 = 5;
        const TOOL_CALL: u32 = 6;
        const EOS_ID: u32 = 127;
        const LOGITS_BASE: u64 = 0x30_0000;

        fn think_ctx(enabled: bool) -> ThinkSpecCtx<'static> {
            ThinkSpecCtx {
                enabled,
                code_fence_token: None,
                reflection_suppress_ids: &[],
                adaptive_sampling: false,
                honor_eos_inside_thinking: Some(false),
            }
        }

        fn make_rejection_seq(slot: usize, inside_thinking: bool, seed: u64) -> ActiveSeq {
            let mut a =
                crate::scheduler::test_fixtures::active_seq(ResponseSink::Blocking(None), slot);
            a.seq.prompt_len = 8;
            a.seq.seq_len = 8;
            a.output_tokens.clear();
            a.temperature = 0.8;
            a.top_k = 20;
            a.top_p = 0.95;
            a.min_p = 0.05;
            a.top_n_sigma = 0.0;
            a.repetition_penalty = 1.0;
            a.presence_penalty = 0.0;
            a.frequency_penalty = 0.0;
            a.lz_penalty = 0.0;
            a.dry_multiplier = 0.0;
            a.logit_bias.clear();
            a.seed = Some(seed);
            a.max_output_tokens = 200;
            a.remaining = 200;
            a.eos_tokens = vec![EOS_ID];
            a.inside_thinking = inside_thinking;
            a.enable_thinking = inside_thinking;
            a.think_ended = !inside_thinking;
            a.think_just_ended = false;
            a.think_start_token = Some(THINK_START);
            a.think_end_token = Some(THINK_END);
            a.tool_call_start_token = Some(TOOL_CALL);
            a
        }

        struct MockRejectionModel {
            inner: Box<dyn Model>,
            rejection_available: bool,
            sample_method: DflashDraftSampleMethod,
            return_none: bool,
            swap_dims: Option<spark_storage::ModelDims>,
            rejection_calls: AtomicUsize,
            recorded_reqs: Mutex<Vec<Vec<RejectionVerifyRowSpec>>>,
            staged_reqs: Mutex<Vec<Vec<RejectionVerifyRowSpec>>>,
            in_graph_words: Mutex<Option<Vec<PackedRejectionWord>>>,
        }

        impl MockRejectionModel {
            fn new(rejection_available: bool, return_none: bool) -> Self {
                Self {
                    inner: Box::new(crate::scheduler::test_fixtures::model::recording()),
                    rejection_available,
                    sample_method: DflashDraftSampleMethod::Greedy,
                    return_none,
                    swap_dims: None,
                    rejection_calls: AtomicUsize::new(0),
                    recorded_reqs: Mutex::new(Vec::new()),
                    staged_reqs: Mutex::new(Vec::new()),
                    in_graph_words: Mutex::new(None),
                }
            }
        }

        impl Model for MockRejectionModel {
            fn prefill(&self, t: &[u32], s: &mut SequenceState, st: u64) -> Result<DevicePtr> {
                self.inner.prefill(t, s, st)
            }
            fn prefill_chunk(
                &self,
                t: &[u32],
                s: &mut SequenceState,
                o: usize,
                l: usize,
                last: bool,
                st: u64,
            ) -> Result<DevicePtr> {
                self.inner.prefill_chunk(t, s, o, l, last, st)
            }
            fn decode(&self, t: u32, s: &mut SequenceState, st: u64) -> Result<DevicePtr> {
                self.inner.decode(t, s, st)
            }
            fn decode_batch(
                &self,
                t: &[u32],
                s: &mut [&mut SequenceState],
                st: u64,
            ) -> Result<DevicePtr> {
                self.inner.decode_batch(t, s, st)
            }
            fn vocab_size(&self) -> usize {
                VOCAB
            }
            fn high_speed_swap_dims(&self) -> Option<spark_storage::ModelDims> {
                self.swap_dims
            }
            fn bind_gpu_to_thread(&self) -> Result<()> {
                Ok(())
            }
            fn synchronize_stream(&self, _: u64) -> Result<()> {
                Ok(())
            }
            fn alloc_sequence(&self) -> Result<SequenceState> {
                self.inner.alloc_sequence()
            }
            fn free_sequence(&self, s: &mut SequenceState) -> Result<()> {
                self.inner.free_sequence(s)
            }
            fn compact_sequence(&self, s: &mut SequenceState, k: usize) -> Result<bool> {
                self.inner.compact_sequence(s, k)
            }
            fn cache_sequence(&self, s: &SequenceState) {
                self.inner.cache_sequence(s)
            }
            fn logits_buffer_ptr(&self) -> DevicePtr {
                DevicePtr(LOGITS_BASE)
            }
            fn copy_logits_to_host(&self, p: DevicePtr, out: &mut [u8]) -> Result<()> {
                self.inner.copy_logits_to_host(p, out)
            }
            fn argmax_on_device(&self, p: DevicePtr, st: u64) -> Result<u32> {
                self.inner.argmax_on_device(p, st)
            }
            fn argmax_batch(&self, p: DevicePtr, n: usize, st: u64) -> Result<Vec<u32>> {
                self.inner.argmax_batch(p, n, st)
            }
            fn proposer_is_dflash(&self) -> bool {
                true
            }
            fn proposer_arm(&self) -> u8 {
                0
            }
            fn proposer_arm_kind(&self, arm: u8) -> Option<DraftProposerKind> {
                (arm == 0).then_some(DraftProposerKind::DFlash)
            }
            fn dflash_draft_sample_method(&self) -> DflashDraftSampleMethod {
                self.sample_method
            }
            fn gpu_rejection_verify_available(&self) -> bool {
                self.rejection_available
            }
            fn sample_rejection_verify_rows_batched(
                &self,
                logits_base: DevicePtr,
                vocab_stride: usize,
                reqs: &[RejectionVerifyRowSpec],
            ) -> Result<Option<Vec<PackedRejectionWord>>> {
                assert_eq!(logits_base, DevicePtr(LOGITS_BASE));
                assert_eq!(vocab_stride, VOCAB);
                self.rejection_calls.fetch_add(1, Ordering::SeqCst);
                self.recorded_reqs.lock().unwrap().push(reqs.to_vec());
                if self.return_none {
                    return Ok(None);
                }
                Ok(Some(
                    reqs.iter()
                        .map(|r| PackedRejectionWord(20 + (r.row_idx as u32 % 50)))
                        .collect(),
                ))
            }
            fn stage_next_verify_rejection_rows(&self, reqs: &[RejectionVerifyRowSpec]) {
                self.staged_reqs.lock().unwrap().push(reqs.to_vec());
            }
            fn take_in_graph_rejection_verify_words(&self) -> Option<Vec<PackedRejectionWord>> {
                self.in_graph_words.lock().unwrap().take()
            }
            fn hidden_after_norm(&self) -> DevicePtr {
                DevicePtr::NULL
            }
            fn decode_verify(&self, t: &[u32], s: &mut SequenceState, st: u64) -> Result<Vec<u32>> {
                self.inner.decode_verify(t, s, st)
            }
            fn run_mtp_propose_multi(
                &self,
                tok: u32,
                pos: usize,
                k: usize,
                s: &mut SequenceState,
                st: u64,
                mask: Option<&[i32]>,
            ) -> Result<Vec<u32>> {
                Model::run_mtp_propose_multi(&*self.inner, tok, pos, k, s, st, mask)
            }
            fn trim_proposer_state(&self, s: &mut SequenceState, n: usize, st: u64) -> Result<()> {
                self.inner.trim_proposer_state(s, n, st)
            }
            fn decode_verify_graphed(
                &self,
                t: &[u32; 2],
                s: &mut SequenceState,
                st: u64,
            ) -> Result<[u32; 2]> {
                self.inner.decode_verify_graphed(t, s, st)
            }
            fn decode_verify_graphed_k3(
                &self,
                t: &[u32; 3],
                s: &mut SequenceState,
                st: u64,
            ) -> Result<[u32; 3]> {
                self.inner.decode_verify_graphed_k3(t, s, st)
            }
            fn decode_verify_graphed_k4(
                &self,
                t: &[u32; 4],
                s: &mut SequenceState,
                st: u64,
            ) -> Result<[u32; 4]> {
                self.inner.decode_verify_graphed_k4(t, s, st)
            }
            fn checkpoint_ssm_states(&self, s: &mut SequenceState) -> Result<()> {
                self.inner.checkpoint_ssm_states(s)
            }
            fn rollback_ssm_states(&self, s: &mut SequenceState, n: usize) -> Result<()> {
                self.inner.rollback_ssm_states(s, n)
            }
            fn has_ssm_layers(&self) -> bool {
                false
            }
            fn generate_speculative(
                &self,
                p: &[u32],
                sp: &spark_runtime::sampler::SamplingParams,
                nd: usize,
            ) -> Result<spark_model::engine::GenerateResult> {
                self.inner.generate_speculative(p, sp, nd)
            }
            fn has_proposer(&self) -> bool {
                true
            }
            fn has_self_speculative(&self) -> bool {
                false
            }
            fn decode_draft(&self, t: u32, s: &mut SequenceState, st: u64) -> Result<DevicePtr> {
                self.inner.decode_draft(t, s, st)
            }
            fn save_hidden_for_mtp(&self, r: usize, st: u64) -> Result<()> {
                self.inner.save_hidden_for_mtp(r, st)
            }
            fn run_mtp_propose(
                &self,
                t: u32,
                p: usize,
                s: &mut SequenceState,
                st: u64,
            ) -> Result<Option<u32>> {
                self.inner.run_mtp_propose(t, p, s, st)
            }
        }

        #[test]
        fn rejection_policy_eligibility_and_draft_width_follow_kernel_availability() {
            let mut m = MockRejectionModel::new(true, false);
            m.sample_method = DflashDraftSampleMethod::Probabilistic;
            assert!(batched_verify_proposer_eligible_with_policy(
                &m,
                SpecVerifyPolicy::Rejection
            ));
            assert!(!batched_verify_proposer_eligible_with_policy(
                &m,
                SpecVerifyPolicy::Gumbel
            ));
            m.rejection_available = false;
            assert!(!batched_verify_proposer_eligible_with_policy(
                &m,
                SpecVerifyPolicy::Rejection
            ));

            // Non-greedy DFlash under Rejection policy keeps gamma=7 for C<=8 and
            // narrows C>8 (C=16) to gamma=5 to eliminate wasted drafter rows.
            for (live, want_non_greedy) in [(1usize, 7usize), (4, 7), (8, 7), (16, 5)] {
                assert_eq!(
                    resolve_dflash_gumbel_num_drafts(
                        true,
                        SpecVerifyPolicy::Rejection,
                        true,
                        live,
                        7,
                        None
                    ),
                    want_non_greedy,
                    "non-greedy Rejection at live={live} must resolve to gamma={want_non_greedy}"
                );
                assert_eq!(
                    resolve_dflash_gumbel_num_drafts(
                        true,
                        SpecVerifyPolicy::Rejection,
                        false,
                        live,
                        7,
                        None
                    ),
                    7,
                    "greedy cohort at live={live} keeps gamma=7"
                );
            }
        }

        #[test]
        fn max_seq_len_clamp_trims_only_overshooting_flat_drafts() {
            assert_eq!(clamp_drafts_to_max_seq_len_count(100, 7, 1024), 7);
            assert_eq!(clamp_drafts_to_max_seq_len_count(1020, 7, 1024), 3);
            assert_eq!(clamp_drafts_to_max_seq_len_count(1023, 7, 1024), 0);
            assert_eq!(clamp_drafts_to_max_seq_len_count(1024, 7, 1024), 0);
            assert_eq!(clamp_drafts_to_max_seq_len_count(1020, 7, 0), 7);

            let mut s0 = make_rejection_seq(0, false, 10);
            s0.seq.seq_len = 252;
            s0.pending_drafts = vec![10, 11, 12, 13, 14, 15]; // 252 + 1 + 6 = 259 > 256 -> keep 3
            let mut s1 = make_rejection_seq(1, false, 11);
            s1.seq.seq_len = 64;
            s1.pending_drafts = vec![20, 21, 22, 23, 24, 25]; // fits easily
            let mut s2 = make_rejection_seq(2, false, 12);
            s2.seq.seq_len = 255;
            s2.pending_drafts = vec![30, 31]; // 255 + 1 + 2 = 258 > 256 -> clamp to 0 (K=1)
            let mut active = [s0, s1, s2];

            clamp_pending_drafts_for_max_seq_len(&mut active, &[0, 1, 2], 256);
            assert_eq!(active[0].pending_drafts, vec![10, 11, 12]);
            assert_eq!(active[1].pending_drafts, vec![20, 21, 22, 23, 24, 25]);
            assert!(active[2].pending_drafts.is_empty());

            // Model-driven helper derives ceiling from `high_speed_swap_dims()`.
            let mut model = MockRejectionModel::new(true, false);
            model.swap_dims = Some(spark_storage::ModelDims {
                num_layers: 4,
                max_blocks_per_layer: 16,
                num_q_heads: 4,
                num_kv_heads: 2,
                head_dim: 64,
                block_size: 16,
            }); // ceiling = 256
            active[0].pending_drafts = vec![10, 11, 12, 13, 14];
            clamp_pending_drafts_to_model_max_seq_len(&model, &mut active, &[0, 1]);
            assert_eq!(active[0].pending_drafts, vec![10, 11, 12]);
        }

        #[test]
        fn batched_rejection_staging_in_graph_consumption_and_eager_fallback() {
            let ctx = think_ctx(true);

            // 1. Mixed cohort (thinking Rejection + forced-end HostPolicySkip + greedy + post-think content Rejection)
            let s0 = make_rejection_seq(0, true, 111); // K=4, row_base=0 (thinking, no mask)
            let mut s1 = make_rejection_seq(1, true, 222);
            s1.force_end_thinking = true; // K=2, row_base=4 (HostPolicySkip -> temp=-1.0)
            let mut s2 = make_rejection_seq(2, false, 333);
            s2.temperature = 0.0; // K=3, row_base=6 (GreedyArgmax -> temp=0.0)
            let s3 = make_rejection_seq(3, false, 444); // K=2, row_base=9 (post-think -> mask think_start/end)
            let active = [s0, s1, s2, s3];
            let idxs = [0usize, 1, 2, 3];
            let drafts_per_seq = vec![vec![10, 11, 12], vec![20], vec![30, 31], vec![40]];
            let k_per_seq = [4usize, 2, 3, 2];
            let row_bases = [0usize, 4, 6, 9];
            let model = MockRejectionModel::new(true, false);

            let staged_mask = stage_batched_rejection_frame_if_eligible_with_policy(
                &model,
                &active,
                &idxs,
                &drafts_per_seq,
                &k_per_seq,
                &row_bases,
                11,
                &ctx,
                SpecVerifyPolicy::Rejection,
            );
            assert_eq!(staged_mask, Some(vec![true, false, false, true]));
            {
                let staged_all = model.staged_reqs.lock().unwrap();
                assert_eq!(staged_all.len(), 1);
                let staged = &staged_all[0];
                assert_eq!(staged.len(), 11);
                assert_eq!(
                    staged.iter().map(|r| r.row_idx).collect::<Vec<_>>(),
                    (0..11).collect::<Vec<_>>()
                );
                // Seq 0 (thinking): temp > 0, no think-tag mask.
                assert!(staged[0..4].iter().all(|r| r.temperature > 0.0
                    && r.mask_tok0 == REJECTION_NO_MASK_TOKEN
                    && r.mask_tok1 == REJECTION_NO_MASK_TOKEN));
                // Seq 1 (force_end_thinking): temp == -1.0, bonus q_row_idx.
                assert!(
                    staged[4..6]
                        .iter()
                        .all(|r| r.temperature == -1.0 && r.q_row_idx == REJECTION_BONUS_Q_ROW_IDX)
                );
                // Seq 2 (greedy): temp == 0.0, draft rows carry slot*8+r, bonus row carries 0xFFFF.
                assert!(staged[6..9].iter().all(|r| r.temperature == 0.0));
                assert_eq!(staged[6].q_row_idx, 2 * 8);
                assert_eq!(staged[6].draft_token, 30);
                assert_eq!(staged[7].q_row_idx, 2 * 8 + 1);
                assert_eq!(staged[7].draft_token, 31);
                assert_eq!(staged[8].q_row_idx, REJECTION_BONUS_Q_ROW_IDX);
                // Seq 3 (post-think content): temp > 0 and masks (THINK_START, THINK_END).
                assert!(staged[9..11].iter().all(|r| r.temperature > 0.0
                    && r.mask_tok0 == THINK_START
                    && r.mask_tok1 == THINK_END));
            }

            // In-graph consumption: 0 eager GPU launches.
            *model.in_graph_words.lock().unwrap() = Some(vec![
                PackedRejectionWord((1 << 31) | 10),
                PackedRejectionWord((1 << 31) | 11),
                PackedRejectionWord(19),
                PackedRejectionWord(21),
                PackedRejectionWord(HOST_POLICY_FALLBACK_TOKEN),
                PackedRejectionWord(HOST_POLICY_FALLBACK_TOKEN),
                PackedRejectionWord((1 << 31) | 30),
                PackedRejectionWord(39),
                PackedRejectionWord(41),
                PackedRejectionWord((1 << 31) | 40),
                PackedRejectionWord(45),
            ]);
            let frame = presample_batched_rejection_frame_ragged_with_policy(
                &model,
                &active,
                &idxs,
                &drafts_per_seq,
                &k_per_seq,
                &row_bases,
                &ctx,
                staged_mask.as_deref(),
                SpecVerifyPolicy::Rejection,
            )
            .unwrap()
            .expect("mixed cohort must consume in-graph rejection words");
            assert_eq!(model.rejection_calls.load(Ordering::SeqCst), 0);
            assert_eq!(frame.seq_offsets, vec![Some(0), None, None, Some(9)]);
            assert_eq!(frame.words[0], PackedRejectionWord((1 << 31) | 10));
            assert_eq!(frame.words[9], PackedRejectionWord((1 << 31) | 40));

            // 2. Eager fallback (in-graph words None): calls sample_rejection_verify_rows_batched
            //    once over ONLY eligible sequences (seq 0 rows 0..4 and seq 3 rows 9..11).
            let fallback = presample_batched_rejection_frame_ragged_with_policy(
                &model,
                &active,
                &idxs,
                &drafts_per_seq,
                &k_per_seq,
                &row_bases,
                &ctx,
                staged_mask.as_deref(),
                SpecVerifyPolicy::Rejection,
            )
            .unwrap()
            .expect("eager fallback must launch batched rejection verify");
            assert_eq!(model.rejection_calls.load(Ordering::SeqCst), 1);
            assert_eq!(fallback.seq_offsets, vec![Some(0), None, None, Some(4)]);
            assert_eq!(fallback.words.len(), 6);
            {
                let recorded = model.recorded_reqs.lock().unwrap();
                assert_eq!(recorded.len(), 1);
                let row_idxs: Vec<usize> = recorded[0].iter().map(|r| r.row_idx).collect();
                assert_eq!(row_idxs, vec![0, 1, 2, 3, 9, 10]);
                assert_eq!(recorded[0][4].mask_tok0, THINK_START);
                assert_eq!(recorded[0][4].mask_tok1, THINK_END);
            }

            // 3. Non-Rejection policy or unavailable kernel clears staging with `&[]` and returns None.
            let cleared = stage_batched_rejection_frame_if_eligible_with_policy(
                &model,
                &active,
                &idxs,
                &drafts_per_seq,
                &k_per_seq,
                &row_bases,
                11,
                &ctx,
                SpecVerifyPolicy::Gumbel,
            );
            assert_eq!(cleared, None);
            assert!(model.staged_reqs.lock().unwrap().last().unwrap().is_empty());

            let unavailable = MockRejectionModel::new(false, false);
            assert_eq!(
                stage_batched_rejection_frame_if_eligible_with_policy(
                    &unavailable,
                    &active,
                    &idxs,
                    &drafts_per_seq,
                    &k_per_seq,
                    &row_bases,
                    11,
                    &ctx,
                    SpecVerifyPolicy::Rejection,
                ),
                None
            );
            assert!(
                unavailable
                    .staged_reqs
                    .lock()
                    .unwrap()
                    .last()
                    .unwrap()
                    .is_empty()
            );
            assert!(
                presample_batched_rejection_frame_ragged_with_policy(
                    &unavailable,
                    &active,
                    &idxs,
                    &drafts_per_seq,
                    &k_per_seq,
                    &row_bases,
                    &ctx,
                    None,
                    SpecVerifyPolicy::Rejection,
                )
                .unwrap()
                .is_none()
            );
        }
    }

    #[test]
    fn scheduler_env_parsers_handle_enabled_disabled_and_invalid_values() {
        use super::{
            parse_batched_k5_log_enabled, parse_dflash_gumbel_gamma_override,
            parse_dflash_pcum_thresh_override, parse_grammar_serial_verify,
        };

        assert!(!parse_grammar_serial_verify(None));
        assert!(parse_grammar_serial_verify(Some("1")));
        assert!(parse_grammar_serial_verify(Some(" 1 ")));
        assert!(!parse_grammar_serial_verify(Some("0")));
        assert!(!parse_grammar_serial_verify(Some("true")));

        assert!(!parse_batched_k5_log_enabled(None));
        assert!(parse_batched_k5_log_enabled(Some("1")));
        assert!(!parse_batched_k5_log_enabled(Some("0")));
        assert!(!parse_batched_k5_log_enabled(Some(" 1 ")));

        assert_eq!(parse_dflash_gumbel_gamma_override(None), None);
        assert_eq!(parse_dflash_gumbel_gamma_override(Some("5")), Some(5));
        assert_eq!(parse_dflash_gumbel_gamma_override(Some("0")), Some(0));
        assert_eq!(parse_dflash_gumbel_gamma_override(Some("99")), None);

        assert_eq!(parse_dflash_pcum_thresh_override(None), None);
        assert_eq!(parse_dflash_pcum_thresh_override(Some("0.25")), Some(0.25));
        assert_eq!(parse_dflash_pcum_thresh_override(Some("0")), Some(0.0));
        assert_eq!(parse_dflash_pcum_thresh_override(Some("2.5")), None);
    }

    #[test]
    fn sm120_256_row_frame_budget_unlocks_c48_c64_k4_and_c24_c32_dflash2_k8() {
        use super::{
            adaptive_drafts_with_max_rows, plan_batched_uniform_with_max_rows,
            plan_ragged_cohort_with_max_rows,
        };

        // Native MTP C=48/64 under 256-row budget: K=4 (3 drafts) fits in ONE frame (192/256 rows).
        assert_eq!(adaptive_drafts_with_max_rows(48, 3, 256), 3);
        assert_eq!(adaptive_drafts_with_max_rows(48, 6, 256), 4);
        assert_eq!(adaptive_drafts_with_max_rows(64, 6, 256), 3);
        let c64_k4: Vec<_> = (0..64).map(|i| (i, Some(4))).collect();
        let plan_c64 = plan_batched_uniform_with_max_rows(&c64_k4, 256);
        assert_eq!(plan_c64.frames, vec![(0..64).collect::<Vec<_>>()]);
        assert!(plan_c64.leftover.is_empty());

        // DFlash2 C=24/32 under 256-row budget: full gamma=7, K=8 fits in ONE frame (192/256 rows).
        assert_eq!(adaptive_drafts_with_max_rows(24, 7, 256), 7);
        assert_eq!(adaptive_drafts_with_max_rows(32, 7, 256), 7);
        let c32_k8: Vec<_> = (0..32).map(|i| (i, Some(8))).collect();
        let plan_c32 = plan_batched_uniform_with_max_rows(&c32_k8, 256);
        assert_eq!(plan_c32.frames, vec![(0..32).collect::<Vec<_>>()]);
        assert!(plan_c32.leftover.is_empty());

        // Ragged packing: 32 x K=8 packs into 1 frame (256 rows); 33 x K=8 splits into 2 frames.
        let plan_r32 = plan_ragged_cohort_with_max_rows(&c32_k8, 256);
        assert_eq!(plan_r32.frames.len(), 1);
        assert_eq!(plan_r32.frames[0].len(), 32);
        let c33_k8: Vec<_> = (0..33).map(|i| (i, Some(8))).collect();
        let plan_r33 = plan_ragged_cohort_with_max_rows(&c33_k8, 256);
        assert_eq!(plan_r33.frames.len(), 2);
        assert!(plan_r33.leftover.is_empty());
        assert!(
            plan_r33
                .frames
                .iter()
                .all(|f| f.len() >= 2 && f.len() * 8 <= 256)
        );

        // GB10 128-row budget keeps C=64 at K=2 (1 draft) and C=32 at K=4 (3 drafts).
        assert_eq!(adaptive_drafts_with_max_rows(64, 6, 128), 1);
        assert_eq!(adaptive_drafts_with_max_rows(32, 7, 128), 3);
    }
}
