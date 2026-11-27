// SPDX-License-Identifier: AGPL-3.0-only

//! Cross-sequence batched DFlash2 proposal (`DraftProposer::propose_batch`).
//!
//! Batches `B` sequences (`2 <= B <= 64`, `M = B * m_rows <= 256`,
//! `M_out = B * gamma_eff <= 256`) through one unified two-stage GPU pipeline:
//!
//! - **Stage 1 (Batched Incremental Context Append):** gathers newly captured
//!   target-hidden rows across all `B` sequences in chunks of `<= 1024` rows
//!   via 1 coalesced `dflash_stage1_gather_ctx_bf16` launch (backed by a
//!   single rotating pinned-H2D metadata slab), runs 1 batched `fc` GEMM +
//!   1 batched `hidden_norm`, then per drafter layer runs 1 batched
//!   `k_proj` / `v_proj` + 1 batched `k_norm` + 1 fused
//!   `dflash_rope_yarn_scaled_scatter_ring_bf16` launch writing PostRope K/V
//!   directly into each sequence's circular `ctx_{k,v}_cache[l]` at physical
//!   slot `abs_pos % ctx_window`.
//! - **Stage 2 (Batched Noise Block):** packs `[B, m_rows]` noise embeddings,
//!   pre-uploads all `num_layers * B` context K/V cache pointers once before
//!   layer 0 (zero H2D copies inside the layer loop), runs all drafter layers
//!   with `dflash2_conv_{prepare,finish}_batched` and
//!   `inferspark_prefill_split_ctx_h128_batched`, compacts the `B * gamma_eff`
//!   output rows, projects the shared LM head directly into `ctx.buffers.logits()`
//!   (zero extra logits VRAM), runs batched top-K + `hidden_projection` +
//!   `dflash2_selector_walk_batched` (or `argmax_bf16_rows`), and reads back
//!   all `B * gamma_eff` draft token IDs in one `copy_d2h_on_stream`.

use std::collections::HashMap;
use std::mem::size_of;

use anyhow::{Context, Result};
use spark_runtime::gpu::{
    DevicePtr, GpuBackend, GraphHandle, HostToDeviceCopy, KernelHandle, PinnedHostBuffer,
};
use spark_runtime::kernel_args::KernelLaunch;

use super::noise_pass::{
    CtxKvCacheRepr, LM_HEAD_PATH_TAG_BF16, LmHeadNvfp4Mode, NoiseGraphCaptureGuard,
    dflash_noise_graph_enabled, invalidate_dstate_ctx_kv_ranges, noise_graph_attn_kgamma_disable_o,
    noise_graph_swa_enabled, resolve_dflash2_selector_topk, resolve_lm_head_path_tag,
    validate_selector_shortlist,
};
use super::{
    BlockDiffusionDraftHead, DFLASH_BATCH_MAX_SEL_ROWS, DFLASH_BATCH_MAX_SEQS,
    DFLASH2_GUMBEL_SHORTLIST_K, DflashLayer, DflashLayerFp8Weights, DflashLayerNvfp4,
    DflashLayerQuantWeights, DflashProposerState, DflashQuantization, DraftRowLayout, ring_window,
};
use crate::layer::ForwardContext;
use crate::layers::ops;
use crate::speculative::ProposerState;

const STAGE1_MAX_CHUNK_ROWS: usize = 1024;
const STAGE1_META_PINNED_SLOTS: usize = 4;
const BATCHED_MAX_NOISE_ROWS: usize = 256;
const MAX_CACHED_BATCHED_NOISE_GRAPHS: usize = 32;
const BATCHED_META_PINNED_SLOTS: usize = 4;

/// Fixed 256-byte-aligned byte offsets for one Stage-2 batched DFlash2
/// metadata slab, shared identically by the rotating page-locked host
/// staging slots (`meta_pinned`) and the persistent single-allocation device
/// slab (`meta_dev`).
///
/// Layout places all 7 always-active base segments first, followed by the 6
/// Gumbel segments, followed by `q_row_bases`:
/// - Base only: `[0 .. ctx_v_ptrs_off + num_layers * b * 8)`
/// - Base + Gumbel: `[0 .. gumbel_min_ps_off + b * 4)`
/// - Base + Gumbel + `q_row_bases`: `[0 .. q_row_bases_off + b * 4)`
///
/// Every offset depends solely on `num_layers` (fixed for the head's lifetime)
/// and is a multiple of 256 bytes (strictly 8- and 16-byte aligned), keeping
/// every device sub-pointer inside a captured Stage-2 CUDA graph immutable
/// across rounds while ONE contiguous `copy_h2d_pinned_async` updates the
/// active prefix.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Stage2MetaLayout {
    num_layers: usize,
    token_ids_off: usize,
    pos_ids_off: usize,
    eff_ctx_off: usize,
    ring_start_off: usize,
    anchor_ids_off: usize,
    ctx_k_ptrs_off: usize,
    ctx_v_ptrs_off: usize,
    gumbel_seeds_off: usize,
    gumbel_step0_off: usize,
    gumbel_temps_off: usize,
    gumbel_top_ks_off: usize,
    gumbel_top_ps_off: usize,
    gumbel_min_ps_off: usize,
    q_row_bases_off: usize,
    total_bytes: usize,
}

impl Stage2MetaLayout {
    fn new(num_layers: usize) -> Result<Self> {
        anyhow::ensure!(
            num_layers > 0,
            "Stage-2 metadata layout requires num_layers > 0"
        );
        let ptrs_cap = num_layers
            .checked_mul(DFLASH_BATCH_MAX_SEQS)
            .and_then(|n| n.checked_mul(size_of::<u64>()))
            .context("Stage-2 ctx ptrs capacity overflow")?;

        let token_ids_off = 0usize;
        let pos_ids_off = token_ids_off + BATCHED_MAX_NOISE_ROWS * size_of::<u32>(); // 1024
        let eff_ctx_off = pos_ids_off + BATCHED_MAX_NOISE_ROWS * size_of::<u32>(); // 2048
        let ring_start_off = eff_ctx_off + DFLASH_BATCH_MAX_SEQS * size_of::<u32>(); // 2304
        let anchor_ids_off = ring_start_off + DFLASH_BATCH_MAX_SEQS * size_of::<u32>(); // 2560
        let ctx_k_ptrs_off = anchor_ids_off + DFLASH_BATCH_MAX_SEQS * size_of::<u32>(); // 2816
        let ctx_v_ptrs_off = ctx_k_ptrs_off
            .checked_add(ptrs_cap)
            .context("Stage-2 ctx_v_ptrs offset overflow")?;
        let gumbel_seeds_off = ctx_v_ptrs_off
            .checked_add(ptrs_cap)
            .context("Stage-2 gumbel_seeds offset overflow")?;
        let gumbel_step0_off = gumbel_seeds_off
            .checked_add(DFLASH_BATCH_MAX_SEQS * size_of::<u64>())
            .context("Stage-2 gumbel_step0 offset overflow")?;
        let gumbel_temps_off = gumbel_step0_off
            .checked_add(DFLASH_BATCH_MAX_SEQS * size_of::<u64>())
            .context("Stage-2 gumbel_temps offset overflow")?;
        let gumbel_top_ks_off = gumbel_temps_off
            .checked_add(DFLASH_BATCH_MAX_SEQS * size_of::<f32>())
            .context("Stage-2 gumbel_top_ks offset overflow")?;
        let gumbel_top_ps_off = gumbel_top_ks_off
            .checked_add(DFLASH_BATCH_MAX_SEQS * size_of::<u32>())
            .context("Stage-2 gumbel_top_ps offset overflow")?;
        let gumbel_min_ps_off = gumbel_top_ps_off
            .checked_add(DFLASH_BATCH_MAX_SEQS * size_of::<f32>())
            .context("Stage-2 gumbel_min_ps offset overflow")?;
        let q_row_bases_off = gumbel_min_ps_off
            .checked_add(DFLASH_BATCH_MAX_SEQS * size_of::<f32>())
            .context("Stage-2 q_row_bases offset overflow")?;
        let total_bytes = q_row_bases_off
            .checked_add(DFLASH_BATCH_MAX_SEQS * size_of::<u32>())
            .context("Stage-2 total_bytes overflow")?;

        Ok(Self {
            num_layers,
            token_ids_off,
            pos_ids_off,
            eff_ctx_off,
            ring_start_off,
            anchor_ids_off,
            ctx_k_ptrs_off,
            ctx_v_ptrs_off,
            gumbel_seeds_off,
            gumbel_step0_off,
            gumbel_temps_off,
            gumbel_top_ks_off,
            gumbel_top_ps_off,
            gumbel_min_ps_off,
            q_row_bases_off,
            total_bytes,
        })
    }

    /// Active contiguous byte prefix `[0 .. len)` covering all segments written
    /// for `b` sequences and the active optional tails.
    #[inline]
    fn active_upload_bytes(&self, b: usize, has_gumbel: bool, has_q_row_bases: bool) -> usize {
        if has_q_row_bases {
            self.q_row_bases_off + b * size_of::<u32>()
        } else if has_gumbel {
            self.gumbel_min_ps_off + b * size_of::<u32>()
        } else {
            self.ctx_v_ptrs_off + self.num_layers * b * size_of::<u64>()
        }
    }

    #[inline]
    fn bind_device(&self, base: DevicePtr) -> BatchedStage2MetaPtrs {
        BatchedStage2MetaPtrs {
            base,
            draft_tokens_dev: base.offset(self.token_ids_off),
            position_ids: base.offset(self.pos_ids_off),
            batch_eff_ctx: base.offset(self.eff_ctx_off),
            batch_ring_start: base.offset(self.ring_start_off),
            batch_anchor_ids: base.offset(self.anchor_ids_off),
            batch_ctx_k_ptrs: base.offset(self.ctx_k_ptrs_off),
            batch_ctx_v_ptrs: base.offset(self.ctx_v_ptrs_off),
            batch_gumbel_seeds: base.offset(self.gumbel_seeds_off),
            batch_gumbel_step0_pos: base.offset(self.gumbel_step0_off),
            batch_gumbel_temps: base.offset(self.gumbel_temps_off),
            batch_gumbel_top_ks: base.offset(self.gumbel_top_ks_off),
            batch_gumbel_top_ps: base.offset(self.gumbel_top_ps_off),
            batch_gumbel_min_ps: base.offset(self.gumbel_min_ps_off),
            q_row_bases: base.offset(self.q_row_bases_off),
        }
    }
}

/// Per-sequence context window and incremental Stage-1 append plan.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct SeqContextPlan {
    eff_ctx: usize,
    needed_start: usize,
    needed_end: usize,
    ring_start: u32,
    new_start: usize,
    new_count: usize,
    invalidate_prev: bool,
}

/// One sequence's slice inside a packed Stage-1 context-projection chunk.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Stage1Slice {
    seq_idx: usize,
    abs_start: usize,
    count: usize,
    chunk_offset: usize,
}

/// Plan one sequence's circular PostRope context window and minimal newly
/// required suffix `[new_start..needed_end)`.
#[allow(clippy::too_many_arguments)]
#[cfg_attr(not(test), allow(dead_code))]
fn plan_seq_context_window(
    resident_len: usize,
    absolute_len: usize,
    ctx_capacity: usize,
    ctx_window: usize,
    ctx_kv_repr: Option<CtxKvCacheRepr>,
    cache_k_start: &[usize],
    cache_k_end: &[usize],
    cache_v_start: &[usize],
    cache_v_end: &[usize],
) -> SeqContextPlan {
    plan_seq_context_window_with_valid_from(
        resident_len,
        absolute_len,
        ctx_capacity,
        ctx_window,
        0,
        ctx_kv_repr,
        cache_k_start,
        cache_k_end,
        cache_v_start,
        cache_v_end,
    )
}

/// Plan one sequence's circular PostRope context window clamped to
/// `ctx_valid_from_abs` (for Marconi prefix-cache hits whose skipped prefix
/// `[0..ctx_valid_from_abs)` has no captured target hidden states).
#[allow(clippy::too_many_arguments)]
fn plan_seq_context_window_with_valid_from(
    resident_len: usize,
    absolute_len: usize,
    ctx_capacity: usize,
    ctx_window: usize,
    ctx_valid_from_abs: usize,
    ctx_kv_repr: Option<CtxKvCacheRepr>,
    cache_k_start: &[usize],
    cache_k_end: &[usize],
    cache_v_start: &[usize],
    cache_v_end: &[usize],
) -> SeqContextPlan {
    let raw_eff_ctx = resident_len.min(ctx_window).min(ctx_capacity);
    let needed_end = absolute_len;
    let needed_start = needed_end
        .saturating_sub(raw_eff_ctx)
        .max(ctx_valid_from_abs)
        .min(needed_end);
    let eff_ctx = needed_end.saturating_sub(needed_start);
    let ring_start = if ctx_window > 0 {
        (needed_start % ctx_window) as u32
    } else {
        0
    };
    let invalidate_prev = ctx_kv_repr != Some(CtxKvCacheRepr::PostRope);
    if eff_ctx == 0 {
        return SeqContextPlan {
            eff_ctx,
            needed_start,
            needed_end,
            ring_start,
            new_start: needed_end,
            new_count: 0,
            invalidate_prev,
        };
    }
    if invalidate_prev {
        return SeqContextPlan {
            eff_ctx,
            needed_start,
            needed_end,
            ring_start,
            new_start: needed_start,
            new_count: eff_ctx,
            invalidate_prev: true,
        };
    }

    let all_valid_start = cache_k_start
        .iter()
        .chain(cache_v_start.iter())
        .copied()
        .max()
        .unwrap_or(0);
    let all_valid_end = cache_k_end
        .iter()
        .chain(cache_v_end.iter())
        .copied()
        .min()
        .unwrap_or(0);
    let new_start = if all_valid_start <= needed_start
        && all_valid_end > needed_start
        && all_valid_end <= needed_end
        && (all_valid_end - all_valid_start) <= ctx_window
    {
        all_valid_end
    } else {
        needed_start
    };
    let new_count = needed_end.saturating_sub(new_start);
    SeqContextPlan {
        eff_ctx,
        needed_start,
        needed_end,
        ring_start,
        new_start,
        new_count,
        invalidate_prev: false,
    }
}

/// Partition all sequences' Stage-1 incremental appends into packed chunks of
/// at most `max_chunk` rows.
fn plan_stage1_chunks(plans: &[SeqContextPlan], max_chunk: usize) -> Vec<Vec<Stage1Slice>> {
    debug_assert!(max_chunk > 0);
    if plans.iter().all(|p| p.new_count == 0) {
        return Vec::new();
    }
    let mut cursors: Vec<usize> = plans.iter().map(|p| p.new_start).collect();
    let mut chunks = Vec::new();
    loop {
        let mut chunk = Vec::new();
        let mut chunk_rows = 0usize;
        for (seq_idx, plan) in plans.iter().enumerate() {
            let rem = plan.needed_end.saturating_sub(cursors[seq_idx]);
            if rem == 0 {
                continue;
            }
            let space = max_chunk.saturating_sub(chunk_rows);
            if space == 0 {
                break;
            }
            let take = rem.min(space);
            chunk.push(Stage1Slice {
                seq_idx,
                abs_start: cursors[seq_idx],
                count: take,
                chunk_offset: chunk_rows,
            });
            cursors[seq_idx] += take;
            chunk_rows += take;
            if chunk_rows == max_chunk {
                break;
            }
        }
        if chunk.is_empty() {
            break;
        }
        chunks.push(chunk);
    }
    chunks
}

fn conv_geometry_compatible(
    conv: &crate::weight_loader::DflashConvWeights,
    hidden_size: usize,
) -> bool {
    conv.groups > 0
        && conv.kernel_size == 2
        && conv.groups.checked_mul(16) == Some(hidden_size)
        && conv
            .kernel_size
            .checked_mul(conv.groups)
            .is_some_and(|half_dyn| half_dyn <= hidden_size)
}

/// Tail selector/argmax execution mode baked into a batched Stage-2 CUDA graph.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum BatchedSelectorMode {
    ProbabilisticRejection { use_radix_topk: bool },
    GreedyNgramBatched { use_radix_topk: bool },
    GumbelK32 { use_radix_topk: bool },
    GumbelK16 { use_radix_topk: bool },
    GreedyBatched { use_radix_topk: bool },
    ArgmaxRows,
    ArgmaxLoop,
}

/// Pure inputs determining the batched Stage-2 selector route and shortlist width.
#[derive(Clone, Copy, Debug)]
struct BatchedSelectorPlanInput {
    rejection_env: bool,
    any_coupled_gumbel: bool,
    sel_top_k: Option<usize>,
    rejection_selector_ready: bool,
    ngram_bonus_ready: bool,
    shared_q_stride: usize,
    shared_q_row_cap: usize,
    gamma_eff: usize,
    batch_size: usize,
    lm_stride: usize,
    has_topk_radix: bool,
    gumbel_k32_ready: bool,
    gumbel_k16_ready: bool,
    has_argmax_rows: bool,
}

/// Resolved batched Stage-2 selector dispatch plan.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct BatchedSelectorPlan {
    rejection_mode: bool,
    use_gumbel_selector: bool,
    shortlist_k: usize,
    selector_mode: BatchedSelectorMode,
}

/// Resolve the batched Stage-2 selector route and shortlist width for a cohort.
///
/// Pure-greedy (`T = 0.0`, `!any_coupled_gumbel`) cohorts bypass
/// `ProbabilisticRejection` even when rejection sampling is enabled globally so
/// they keep the exact `sel_top_k` greedy/k32 walk with softmax `out_probs` and
/// skip unused `q` buffer staging; stochastic or mixed cohorts with at least
/// one active `temperature > 0.0` sequence keep `ProbabilisticRejection`.
#[must_use]
fn resolve_batched_selector_plan(input: &BatchedSelectorPlanInput) -> BatchedSelectorPlan {
    let slot_stride = super::noise_pass::REJECTION_SLOT_STRIDE;
    let total_q_slots = input.shared_q_row_cap / slot_stride;
    let rejection_mode = input.rejection_env
        && input.any_coupled_gumbel
        && input.sel_top_k.is_some()
        && input.rejection_selector_ready
        && input.shared_q_stride == crate::model::trait_impl::device_topk::REJECTION_Q_STRIDE
        && input.gamma_eff <= slot_stride
        && total_q_slots >= input.batch_size;
    let ngram_mode = !rejection_mode
        && !input.any_coupled_gumbel
        && input.ngram_bonus_ready
        && input.sel_top_k.is_some();
    let has_k32 =
        !rejection_mode && !ngram_mode && input.sel_top_k.is_some() && input.gumbel_k32_ready;
    let use_gumbel_selector = !rejection_mode
        && !ngram_mode
        && input.sel_top_k.is_some()
        && (has_k32 || input.gumbel_k16_ready);
    let (shortlist_k, selector_mode) = if rejection_mode {
        let (k, use_radix_topk) = super::noise_pass::resolve_rejection_shortlist_k(
            input.lm_stride,
            input.has_topk_radix,
            input.shared_q_stride,
        );
        (
            k,
            BatchedSelectorMode::ProbabilisticRejection { use_radix_topk },
        )
    } else if ngram_mode && let Some(sel_top_k) = input.sel_top_k {
        let k = sel_top_k.min(input.lm_stride);
        (
            k,
            BatchedSelectorMode::GreedyNgramBatched {
                use_radix_topk: input.has_topk_radix,
            },
        )
    } else if let Some(sel_top_k) = input.sel_top_k {
        let (k, use_radix_topk, use_k32) = resolve_dflash2_selector_topk(
            sel_top_k,
            input.lm_stride,
            input.any_coupled_gumbel && !input.rejection_env,
            input.has_topk_radix,
            has_k32,
        );
        let mode = if use_k32 {
            BatchedSelectorMode::GumbelK32 { use_radix_topk }
        } else if use_gumbel_selector {
            BatchedSelectorMode::GumbelK16 { use_radix_topk }
        } else {
            BatchedSelectorMode::GreedyBatched { use_radix_topk }
        };
        (k, mode)
    } else if input.has_argmax_rows {
        (0, BatchedSelectorMode::ArgmaxRows)
    } else {
        (0, BatchedSelectorMode::ArgmaxLoop)
    };
    BatchedSelectorPlan {
        rejection_mode,
        use_gumbel_selector,
        shortlist_k,
        selector_mode,
    }
}

/// Bucket `batch_size` for Stage-2 batched DFlash2 CUDA graph capture to
/// `{1, 2, 4, 8, 16, 24, 32, 48, 64}` (`> 64` stays unpadded).
///
/// Eliminates mid-benchmark CUDA graph capture spikes when `retrieval_chain`
/// or draining cohorts temporarily shrink the batched propose count to odd
/// sizes (`3, 5..=7, 9..=15, 17..=23, 25..=31, 33..=47, 49..=63`).
#[inline]
#[must_use]
const fn batched_noise_graph_batch_bucket(actual_b: usize) -> usize {
    match actual_b {
        0..=1 => actual_b,
        2 => 2,
        3..=4 => 4,
        5..=8 => 8,
        9..=16 => 16,
        17..=24 => 24,
        25..=32 => 32,
        33..=48 => 48,
        49..=64 => 64,
        _ => actual_b,
    }
}

/// Count free, unallocatable `shared_q` sink slots (`slot >= configured_lanes`)
/// that cannot be owned by any live sequence in the process (including
/// sequences outside the current propose batch, such as `retrieval_chain`
/// preemptions).
///
/// Fails closed (`0`) when `configured_lanes <= 1` (unregistered or non-batched
/// lane table) or when any active batch slot lies outside `0..configured_lanes`.
#[must_use]
fn count_dedicated_rejection_q_sinks(
    configured_lanes: usize,
    total_q_slots: usize,
    slot_stride: usize,
    gamma_eff: usize,
    shared_q_row_cap: usize,
    used_slots: &[bool; DFLASH_BATCH_MAX_SEQS],
    active_slots: &[Option<usize>],
) -> usize {
    if configured_lanes <= 1
        || slot_stride == 0
        || active_slots
            .iter()
            .any(|slot| slot.is_some_and(|s| s >= configured_lanes))
    {
        return 0;
    }
    let cap = total_q_slots.min(DFLASH_BATCH_MAX_SEQS);
    let mut count = 0usize;
    for (slot, &in_use) in used_slots
        .iter()
        .enumerate()
        .take(cap)
        .skip(configured_lanes)
    {
        if !in_use
            && slot
                .checked_mul(slot_stride)
                .and_then(|base| base.checked_add(gamma_eff))
                .is_some_and(|end| end <= shared_q_row_cap)
        {
            count += 1;
        }
    }
    count
}

/// Assign strictly distinct, unallocatable `shared_q` sink slots
/// (`slot >= configured_lanes`) to padded Stage-2 lanes `actual_b..padded_b` so
/// `dflash2_selector_walk_probabilistic_batched` never races on shared memory
/// or clobbers any live sequence's rejection `q` buffer.
#[allow(clippy::too_many_arguments)]
#[must_use]
fn assign_padded_rejection_q_sinks(
    actual_b: usize,
    padded_b: usize,
    configured_lanes: usize,
    total_q_slots: usize,
    slot_stride: usize,
    gamma_eff: usize,
    shared_q_row_cap: usize,
    used_slots: &mut [bool; DFLASH_BATCH_MAX_SEQS],
    q_row_bases_buf: &mut [u32; DFLASH_BATCH_MAX_SEQS],
) -> bool {
    if padded_b <= actual_b {
        return true;
    }
    if configured_lanes <= 1 || slot_stride == 0 || padded_b > DFLASH_BATCH_MAX_SEQS {
        return false;
    }
    let cap = total_q_slots.min(DFLASH_BATCH_MAX_SEQS);
    let mut next_sink = cap;
    for base_slot in q_row_bases_buf.iter_mut().take(padded_b).skip(actual_b) {
        let mut assigned = false;
        while next_sink > configured_lanes {
            next_sink -= 1;
            if !used_slots[next_sink]
                && next_sink
                    .checked_mul(slot_stride)
                    .and_then(|base| base.checked_add(gamma_eff))
                    .is_some_and(|end| end <= shared_q_row_cap)
            {
                used_slots[next_sink] = true;
                *base_slot = (next_sink * slot_stride) as u32;
                assigned = true;
                break;
            }
        }
        if !assigned {
            return false;
        }
    }
    true
}

/// Pure capacity & safety inputs for bucketing Stage-2 noise graph `batch_size`.
#[derive(Clone, Copy, Debug)]
struct PaddedNoiseBatchInput {
    actual_b: usize,
    graph_enabled: bool,
    m_rows: usize,
    gamma_eff: usize,
    n_attn_scratch: usize,
    max_sel_rows: usize,
    lm_stride: usize,
    logits_bytes: usize,
    rejection_mode: bool,
    available_rejection_sinks: usize,
}

/// Resolve the Stage-2 noise batch width (`actual_b` or bucketed in
/// `{1, 2, 4, 8, 16, 24, 32, 48, 64}`), failing closed to `actual_b` if any scratch, logits,
/// or dedicated rejection sink bound is not provably satisfied.
#[must_use]
fn resolve_padded_noise_batch_size(input: &PaddedNoiseBatchInput) -> usize {
    if !input.graph_enabled {
        return input.actual_b;
    }
    let bucket_b = batched_noise_graph_batch_bucket(input.actual_b);
    if bucket_b <= input.actual_b || bucket_b > DFLASH_BATCH_MAX_SEQS {
        return input.actual_b;
    }
    let Some(padded_m_total) = bucket_b.checked_mul(input.m_rows) else {
        return input.actual_b;
    };
    let Some(padded_m_out) = bucket_b.checked_mul(input.gamma_eff) else {
        return input.actual_b;
    };
    let Some(padded_logits_bytes) = padded_m_out
        .checked_mul(input.lm_stride)
        .and_then(|elems| elems.checked_mul(2))
    else {
        return input.actual_b;
    };
    if padded_m_total > BATCHED_MAX_NOISE_ROWS.min(input.n_attn_scratch)
        || padded_m_out > BATCHED_MAX_NOISE_ROWS.min(input.max_sel_rows)
        || padded_logits_bytes > input.logits_bytes
    {
        return input.actual_b;
    }
    if input.rejection_mode && input.available_rejection_sinks < bucket_b - input.actual_b {
        return input.actual_b;
    }
    bucket_b
}

/// SM count threshold above which batched DFlash2 routes NVFP4 GEMMs (`m >= verify_min_m`)
/// through native cuBLASLt NVFP4 (`qwen4_fast_proj::nvfp4`) while keeping GB10 (`<= 64` SMs)
/// on coalesced W4A16.
const BATCHED_DRAFTER_NVFP4_HIGH_SM_THRESHOLD: u32 = 64;

#[inline]
#[must_use]
const fn batched_drafter_nvfp4_tc_admits(
    sm_count: u32,
    m: usize,
    verify_min_m: usize,
    mode: crate::layers::qwen4_fast_proj::nvfp4::Nvfp4FastProjMode,
) -> bool {
    sm_count > BATCHED_DRAFTER_NVFP4_HIGH_SM_THRESHOLD && mode.is_enabled() && m >= verify_min_m
}

#[inline]
#[must_use]
fn batched_drafter_nvfp4_tc_active(m: usize) -> bool {
    batched_drafter_nvfp4_tc_admits(
        atlas_core::device::active_sm_count(),
        m,
        crate::layers::qwen4_fast_proj::nvfp4::verify_min_m(),
        crate::layers::qwen4_fast_proj::nvfp4::mode_or_off(),
    )
}

/// Complete key isolating distinct Stage-2 batched DFlash2 CUDA graph captures.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct BatchedNoiseGraphKey {
    batch_size: usize,
    gamma_eff: usize,
    m_rows: usize,
    shortlist_k: usize,
    selector_mode: BatchedSelectorMode,
    lm_vocab: u32,
    lm_head_path_tag: u8,
    split_kv_splits: u8,
    output_slot: u8,
    logits_out_ptr: u64,
    meta_dev_ptr: u64,
    stream: u64,
}

/// Active Split-KV H=128 launch plan for batched Stage-2 context attention
/// when `batch_size * num_q_heads < sm_count` (`C=2..4` on high-SM GPUs).
#[derive(Clone, Copy, Debug)]
struct BatchedSplitKvH128Plan {
    split_kernel: KernelHandle,
    reduce_kernel: KernelHandle,
    workspace: DevicePtr,
    num_splits: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct BatchedStage2MetaPtrs {
    base: DevicePtr,
    draft_tokens_dev: DevicePtr,
    position_ids: DevicePtr,
    batch_eff_ctx: DevicePtr,
    batch_ring_start: DevicePtr,
    batch_anchor_ids: DevicePtr,
    batch_ctx_k_ptrs: DevicePtr,
    batch_ctx_v_ptrs: DevicePtr,
    batch_gumbel_seeds: DevicePtr,
    batch_gumbel_step0_pos: DevicePtr,
    batch_gumbel_temps: DevicePtr,
    batch_gumbel_top_ks: DevicePtr,
    batch_gumbel_top_ps: DevicePtr,
    batch_gumbel_min_ps: DevicePtr,
    q_row_bases: DevicePtr,
}

/// Tightly packed byte offsets for one Stage-1 chunk's device metadata slab
/// (`chunk_rows <= STAGE1_MAX_CHUNK_ROWS`, `num_layers >= 1`).
///
/// Every `u64` pointer array precedes every `u32` index/position array so all
/// `u64` slices remain 8-byte aligned and all `u32` slices remain 4-byte
/// aligned for arbitrary `chunk_rows` (including odd counts).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Stage1MetaLayout {
    chunk_rows: usize,
    num_layers: usize,
    gather_src_ptrs_off: usize,
    scatter_dst_k_ptrs_off: usize,
    scatter_dst_v_ptrs_off: usize,
    gather_src_row_idx_off: usize,
    scatter_dst_slot_off: usize,
    scatter_row_pos_off: usize,
    total_bytes: usize,
}

impl Stage1MetaLayout {
    fn new(chunk_rows: usize, num_layers: usize) -> Result<Self> {
        anyhow::ensure!(
            chunk_rows > 0 && chunk_rows <= STAGE1_MAX_CHUNK_ROWS && num_layers > 0,
            "Stage-1 metadata layout out of bounds: chunk_rows={chunk_rows} num_layers={num_layers}"
        );
        let u64_row_bytes = chunk_rows
            .checked_mul(size_of::<u64>())
            .context("Stage-1 u64 row bytes overflow")?;
        let layer_ptr_bytes = num_layers
            .checked_mul(u64_row_bytes)
            .context("Stage-1 layer ptr bytes overflow")?;
        let u32_row_bytes = chunk_rows
            .checked_mul(size_of::<u32>())
            .context("Stage-1 u32 row bytes overflow")?;

        let gather_src_ptrs_off = 0usize;
        let scatter_dst_k_ptrs_off = gather_src_ptrs_off
            .checked_add(u64_row_bytes)
            .context("Stage-1 k ptrs offset overflow")?;
        let scatter_dst_v_ptrs_off = scatter_dst_k_ptrs_off
            .checked_add(layer_ptr_bytes)
            .context("Stage-1 v ptrs offset overflow")?;
        let gather_src_row_idx_off = scatter_dst_v_ptrs_off
            .checked_add(layer_ptr_bytes)
            .context("Stage-1 row idx offset overflow")?;
        let scatter_dst_slot_off = gather_src_row_idx_off
            .checked_add(u32_row_bytes)
            .context("Stage-1 dst slot offset overflow")?;
        let scatter_row_pos_off = scatter_dst_slot_off
            .checked_add(u32_row_bytes)
            .context("Stage-1 row pos offset overflow")?;
        let total_bytes = scatter_row_pos_off
            .checked_add(u32_row_bytes)
            .context("Stage-1 total bytes overflow")?;

        Ok(Self {
            chunk_rows,
            num_layers,
            gather_src_ptrs_off,
            scatter_dst_k_ptrs_off,
            scatter_dst_v_ptrs_off,
            gather_src_row_idx_off,
            scatter_dst_slot_off,
            scatter_row_pos_off,
            total_bytes,
        })
    }

    #[inline]
    fn layer_ptr_stride_bytes(&self) -> usize {
        self.chunk_rows * size_of::<u64>()
    }

    fn bind_device(&self, base: DevicePtr) -> Stage1MetaDevicePtrs {
        Stage1MetaDevicePtrs {
            gather_src_ptrs: base.offset(self.gather_src_ptrs_off),
            scatter_dst_k_ptrs_base: base.offset(self.scatter_dst_k_ptrs_off),
            scatter_dst_v_ptrs_base: base.offset(self.scatter_dst_v_ptrs_off),
            gather_src_row_idx: base.offset(self.gather_src_row_idx_off),
            scatter_dst_slot: base.offset(self.scatter_dst_slot_off),
            scatter_row_pos: base.offset(self.scatter_row_pos_off),
            layer_ptr_stride_bytes: self.layer_ptr_stride_bytes(),
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct Stage1MetaDevicePtrs {
    gather_src_ptrs: DevicePtr,
    scatter_dst_k_ptrs_base: DevicePtr,
    scatter_dst_v_ptrs_base: DevicePtr,
    gather_src_row_idx: DevicePtr,
    scatter_dst_slot: DevicePtr,
    scatter_row_pos: DevicePtr,
    layer_ptr_stride_bytes: usize,
}

impl Stage1MetaDevicePtrs {
    #[inline]
    fn layer_dst_k_ptrs(&self, layer_idx: usize) -> DevicePtr {
        self.scatter_dst_k_ptrs_base
            .offset(layer_idx * self.layer_ptr_stride_bytes)
    }

    #[inline]
    fn layer_dst_v_ptrs(&self, layer_idx: usize) -> DevicePtr {
        self.scatter_dst_v_ptrs_base
            .offset(layer_idx * self.layer_ptr_stride_bytes)
    }
}

/// Pack one Stage-1 chunk's per-row gather pointers/indices and per-layer
/// circular KV-ring scatter pointers/slots/positions into a single contiguous
/// host staging buffer (zero heap allocation).
#[allow(clippy::too_many_arguments)]
fn pack_stage1_meta_into(
    layout: &Stage1MetaLayout,
    slices: &[Stage1Slice],
    ctx_window: usize,
    seq_hidden_acc: impl Fn(usize) -> u64,
    seq_ring_state: impl Fn(usize) -> Result<ring_window::RingState>,
    seq_ctx_k_ptr: impl Fn(usize, usize) -> u64,
    seq_ctx_v_ptr: impl Fn(usize, usize) -> u64,
    dst: &mut [u8],
) -> Result<()> {
    anyhow::ensure!(
        ctx_window > 0 && dst.len() >= layout.total_bytes,
        "Stage-1 metadata staging invalid: ctx_window={ctx_window} dst_len={} need={}",
        dst.len(),
        layout.total_bytes
    );
    let mut written_rows = 0usize;
    let layer_stride = layout.layer_ptr_stride_bytes();
    for sl in slices {
        anyhow::ensure!(
            sl.count <= ctx_window
                && sl
                    .chunk_offset
                    .checked_add(sl.count)
                    .is_some_and(|end| end <= layout.chunk_rows),
            "Stage-1 slice out of bounds: offset={} count={} chunk_rows={} ctx_window={ctx_window}",
            sl.chunk_offset,
            sl.count,
            layout.chunk_rows
        );
        let ring = seq_ring_state(sl.seq_idx)?;
        let gather = ring.plan_gather(sl.abs_start, sl.count)?;
        let write_end = sl
            .abs_start
            .checked_add(sl.count)
            .context("Stage-1 slice absolute end overflow")?;
        let _write_plan = ring_window::plan_write_chunk(
            sl.abs_start,
            sl.count,
            write_end.max(ctx_window),
            ctx_window,
        )?;
        let hidden_bytes = seq_hidden_acc(sl.seq_idx).to_le_bytes();

        for span in gather.spans() {
            for j in 0..span.slot_count {
                let row = sl.chunk_offset + span.dst_slot + j;
                anyhow::ensure!(
                    row < layout.chunk_rows,
                    "Stage-1 metadata row {row} out of bounds for chunk_rows={}",
                    layout.chunk_rows
                );
                let src_row_u32 = u32::try_from(span.src_slot + j)
                    .context("Stage-1 gather src_row_idx exceeds u32")?;
                let abs_pos = sl.abs_start + span.dst_slot + j;
                let abs_pos_u32 = u32::try_from(abs_pos).context("Stage-1 row_pos exceeds u32")?;
                let dst_slot_u32 =
                    u32::try_from(abs_pos % ctx_window).context("Stage-1 dst_slot exceeds u32")?;

                let u64_off = row * size_of::<u64>();
                let u32_off = row * size_of::<u32>();
                dst[layout.gather_src_ptrs_off + u64_off..layout.gather_src_ptrs_off + u64_off + 8]
                    .copy_from_slice(&hidden_bytes);
                dst[layout.gather_src_row_idx_off + u32_off
                    ..layout.gather_src_row_idx_off + u32_off + 4]
                    .copy_from_slice(&src_row_u32.to_le_bytes());
                dst[layout.scatter_dst_slot_off + u32_off
                    ..layout.scatter_dst_slot_off + u32_off + 4]
                    .copy_from_slice(&dst_slot_u32.to_le_bytes());
                dst[layout.scatter_row_pos_off + u32_off..layout.scatter_row_pos_off + u32_off + 4]
                    .copy_from_slice(&abs_pos_u32.to_le_bytes());
                written_rows += 1;
            }
        }

        for layer_idx in 0..layout.num_layers {
            let k_bytes = seq_ctx_k_ptr(sl.seq_idx, layer_idx).to_le_bytes();
            let v_bytes = seq_ctx_v_ptr(sl.seq_idx, layer_idx).to_le_bytes();
            let k_layer_base = layout.scatter_dst_k_ptrs_off + layer_idx * layer_stride;
            let v_layer_base = layout.scatter_dst_v_ptrs_off + layer_idx * layer_stride;
            for i in 0..sl.count {
                let row = sl.chunk_offset + i;
                let u64_off = row * size_of::<u64>();
                dst[k_layer_base + u64_off..k_layer_base + u64_off + 8].copy_from_slice(&k_bytes);
                dst[v_layer_base + u64_off..v_layer_base + u64_off + 8].copy_from_slice(&v_bytes);
            }
        }
    }
    anyhow::ensure!(
        written_rows == layout.chunk_rows,
        "Stage-1 metadata packed {written_rows} rows, expected {}",
        layout.chunk_rows
    );
    Ok(())
}

/// Per-head Stage-2 batched DFlash2 CUDA graph cache and rotating page-locked
/// host metadata staging for both Stage 1 and Stage 2 (zero per-step `Vec`
/// allocations, zero inter-stage `cuStreamSynchronize`).
pub(super) struct BatchedNoiseGraphCache {
    meta_dev: DevicePtr,
    meta_dev_bytes: usize,
    meta_pinned: Option<PinnedHostBuffer>,
    meta_slot: usize,
    meta_slot_bytes: usize,
    stage1_gather_kernel: Option<KernelHandle>,
    stage1_rope_scatter_kernel: Option<KernelHandle>,
    stage1_meta_dev: DevicePtr,
    stage1_meta_pinned: Option<PinnedHostBuffer>,
    stage1_meta_slot: usize,
    stage1_meta_slot_bytes: usize,
    split_ctx_batched_splitkv_kernel: Option<KernelHandle>,
    split_ctx_splitkv_reduce_kernel: Option<KernelHandle>,
    split_kv_ws_dev: DevicePtr,
    split_kv_ws_cap: usize,
    ngram_upload_pinned: Option<PinnedHostBuffer>,
    ngram_upload_slot: usize,
    graphs: HashMap<BatchedNoiseGraphKey, GraphHandle>,
    graph_order: Vec<BatchedNoiseGraphKey>,
}

impl BatchedNoiseGraphCache {
    pub(super) fn new() -> Self {
        Self {
            meta_dev: DevicePtr::NULL,
            meta_dev_bytes: 0,
            meta_pinned: None,
            meta_slot: 0,
            meta_slot_bytes: 0,
            stage1_gather_kernel: None,
            stage1_rope_scatter_kernel: None,
            stage1_meta_dev: DevicePtr::NULL,
            stage1_meta_pinned: None,
            stage1_meta_slot: 0,
            stage1_meta_slot_bytes: 0,
            split_ctx_batched_splitkv_kernel: None,
            split_ctx_splitkv_reduce_kernel: None,
            split_kv_ws_dev: DevicePtr::NULL,
            split_kv_ws_cap: 0,
            ngram_upload_pinned: None,
            ngram_upload_slot: 0,
            graphs: HashMap::new(),
            graph_order: Vec::new(),
        }
    }

    fn resolve_batched_split_kv_h128_kernels(
        &mut self,
        gpu: &dyn GpuBackend,
    ) -> Option<(KernelHandle, KernelHandle)> {
        let split_k = *self
            .split_ctx_batched_splitkv_kernel
            .get_or_insert_with(|| {
                crate::layers::try_kernel(
                    gpu,
                    "inferspark_prefill_h128",
                    "inferspark_prefill_split_ctx_h128_batched_splitkv",
                )
            });
        let reduce_k = *self.split_ctx_splitkv_reduce_kernel.get_or_insert_with(|| {
            crate::layers::try_kernel(
                gpu,
                "inferspark_prefill_h128",
                "inferspark_prefill_split_ctx_h128_splitkv_reduce",
            )
        });
        (split_k.0 != 0 && reduce_k.0 != 0).then_some((split_k, reduce_k))
    }

    fn ensure_split_kv_workspace(
        &mut self,
        gpu: &dyn GpuBackend,
        alloc_bytes: usize,
        needed_bytes: usize,
    ) -> Result<Option<DevicePtr>> {
        if needed_bytes == 0 {
            return Ok(None);
        }
        if self.split_kv_ws_dev.is_null() {
            let bytes = alloc_bytes.max(needed_bytes);
            self.split_kv_ws_dev = gpu.alloc(bytes)?;
            self.split_kv_ws_cap = bytes;
        }
        if self.split_kv_ws_cap >= needed_bytes && self.split_kv_ws_dev.0.is_multiple_of(16) {
            Ok(Some(self.split_kv_ws_dev))
        } else {
            Ok(None)
        }
    }

    fn resolve_stage1_kernels(&mut self, gpu: &dyn GpuBackend) -> (KernelHandle, KernelHandle) {
        let gather = *self.stage1_gather_kernel.get_or_insert_with(|| {
            crate::layers::try_kernel(gpu, "dflash2_selector", "dflash_stage1_gather_ctx_bf16")
        });
        let rope_scatter = *self.stage1_rope_scatter_kernel.get_or_insert_with(|| {
            crate::layers::try_kernel(
                gpu,
                "dflash2_selector",
                "dflash_rope_yarn_scaled_scatter_ring_bf16",
            )
        });
        (gather, rope_scatter)
    }

    /// Pack one Stage-1 chunk's gather + per-layer RoPE/KV-ring scatter
    /// metadata into a rotating page-locked slot and issue a single async H2D
    /// into the persistent `stage1_meta_dev` slab.
    #[allow(clippy::too_many_arguments)]
    fn upload_stage1_meta(
        &mut self,
        gpu: &dyn GpuBackend,
        slices: &[Stage1Slice],
        chunk_rows: usize,
        num_layers: usize,
        ctx_window: usize,
        seq_hidden_acc: impl Fn(usize) -> u64,
        seq_ring_state: impl Fn(usize) -> Result<ring_window::RingState>,
        seq_ctx_k_ptr: impl Fn(usize, usize) -> u64,
        seq_ctx_v_ptr: impl Fn(usize, usize) -> u64,
        stream: u64,
    ) -> Result<Stage1MetaDevicePtrs> {
        let layout = Stage1MetaLayout::new(chunk_rows, num_layers)?;
        let max_slot_bytes = Stage1MetaLayout::new(STAGE1_MAX_CHUNK_ROWS, num_layers)?.total_bytes;
        if self.stage1_meta_dev.is_null() {
            self.stage1_meta_dev = gpu.alloc(max_slot_bytes)?;
            self.stage1_meta_slot_bytes = max_slot_bytes;
        }
        if self.stage1_meta_pinned.is_none() {
            self.stage1_meta_pinned =
                Some(gpu.alloc_host_pinned(max_slot_bytes * STAGE1_META_PINNED_SLOTS)?);
            self.stage1_meta_slot_bytes = max_slot_bytes;
        }
        anyhow::ensure!(
            layout.total_bytes <= self.stage1_meta_slot_bytes,
            "batched Stage-1 metadata slab too small: need {}, have {}",
            layout.total_bytes,
            self.stage1_meta_slot_bytes
        );

        let slot = self.stage1_meta_slot;
        self.stage1_meta_slot = (self.stage1_meta_slot + 1) % STAGE1_META_PINNED_SLOTS;
        let base = slot * self.stage1_meta_slot_bytes;

        let pinned = self
            .stage1_meta_pinned
            .as_mut()
            .context("batched Stage-1 pinned metadata buffer missing")?;
        pack_stage1_meta_into(
            &layout,
            slices,
            ctx_window,
            seq_hidden_acc,
            seq_ring_state,
            seq_ctx_k_ptr,
            seq_ctx_v_ptr,
            &mut pinned.as_mut_slice()[base..base + layout.total_bytes],
        )?;
        let slice = pinned.pinned_slice_range(base, layout.total_bytes)?;
        // SAFETY: `stage1_meta_pinned` is page-locked, owned by the head's
        // `BatchedNoiseGraphCache`, and rotates across `STAGE1_META_PINNED_SLOTS`
        // disjoint slots. Multi-chunk calls drain `stream` before wrapping a slot
        // within the same call, and `execute_batched_dflash2` drains `stream` via
        // D2H before returning, so no in-flight H2D source is ever overwritten.
        unsafe {
            gpu.copy_h2d_pinned_async(slice, self.stage1_meta_dev, stream)?;
        }
        Ok(layout.bind_device(self.stage1_meta_dev))
    }

    fn insert_graph(
        &mut self,
        gpu: &dyn GpuBackend,
        key: BatchedNoiseGraphKey,
        graph: GraphHandle,
    ) {
        if let Some(old) = self.graphs.insert(key, graph)
            && old.0 != 0
        {
            let _ = gpu.destroy_graph(old);
        }
        self.graph_order.retain(|existing| existing != &key);
        self.graph_order.push(key);
        while self.graph_order.len() > MAX_CACHED_BATCHED_NOISE_GRAPHS {
            let evicted = self.graph_order.remove(0);
            if let Some(old) = self.graphs.remove(&evicted)
                && old.0 != 0
            {
                let _ = gpu.destroy_graph(old);
            }
        }
    }

    pub(super) fn clear(&mut self, gpu: &dyn GpuBackend) {
        self.graph_order.clear();
        for (_, graph) in self.graphs.drain() {
            if graph.0 != 0 {
                let _ = gpu.destroy_graph(graph);
            }
        }
        if !self.stage1_meta_dev.is_null() {
            let _ = gpu.free(self.stage1_meta_dev);
            self.stage1_meta_dev = DevicePtr::NULL;
        }
        self.stage1_meta_pinned = None;
        self.stage1_meta_slot = 0;
        self.stage1_meta_slot_bytes = 0;
        if !self.meta_dev.is_null() {
            let _ = gpu.free(self.meta_dev);
            self.meta_dev = DevicePtr::NULL;
            self.meta_dev_bytes = 0;
        }
        if !self.split_kv_ws_dev.is_null() {
            let _ = gpu.free(self.split_kv_ws_dev);
            self.split_kv_ws_dev = DevicePtr::NULL;
            self.split_kv_ws_cap = 0;
        }
        self.ngram_upload_pinned = None;
        self.ngram_upload_slot = 0;
    }

    /// Build per-sequence Gate-4 anti-loop n-gram tables for `actual_b` active
    /// sequences (zero-padding trailing bucketed lanes `actual_b..padded_b`)
    /// and upload them asynchronously on `stream` via rotating pinned slots.
    #[expect(
        clippy::too_many_arguments,
        reason = "mirrors batched Stage-2 pinned upload helper signature"
    )]
    fn upload_ngram_tables_batched(
        &mut self,
        gpu: &dyn GpuBackend,
        dstates: &[&mut DflashProposerState],
        last_tokens: &[u32],
        padded_b: usize,
        tables_dev: DevicePtr,
        prev3_dev: DevicePtr,
        stream: u64,
    ) -> Result<()> {
        use super::ngram_bonus::{
            NGRAM_EMPTY_TOK, NGRAM_PREV3_BYTES, NGRAM_TABLE_BYTES, NgramBonusConfig,
            build_sequence_ngram_table,
        };
        const MAX_TABLES_BYTES: usize = DFLASH_BATCH_MAX_SEQS * NGRAM_TABLE_BYTES;
        const MAX_PREV3_BYTES: usize = DFLASH_BATCH_MAX_SEQS * NGRAM_PREV3_BYTES;
        const SLOT_BYTES: usize = MAX_TABLES_BYTES + MAX_PREV3_BYTES;

        let actual_b = last_tokens.len();
        anyhow::ensure!(
            actual_b > 0
                && actual_b == dstates.len()
                && (actual_b..=DFLASH_BATCH_MAX_SEQS).contains(&padded_b),
            "batched n-gram upload batch bounds invalid: actual_b={actual_b} padded_b={padded_b}"
        );
        if self.ngram_upload_pinned.is_none() {
            self.ngram_upload_pinned =
                Some(gpu.alloc_host_pinned(SLOT_BYTES * BATCHED_META_PINNED_SLOTS)?);
        }
        let slot = self.ngram_upload_slot;
        self.ngram_upload_slot = (self.ngram_upload_slot + 1) % BATCHED_META_PINNED_SLOTS;
        let base = slot * SLOT_BYTES;
        let active_tables_bytes = padded_b * NGRAM_TABLE_BYTES;
        let active_prev3_bytes = padded_b * NGRAM_PREV3_BYTES;
        let cfg = NgramBonusConfig::default();

        let pinned = self
            .ngram_upload_pinned
            .as_mut()
            .context("batched n-gram pinned staging missing")?;
        let dst = &mut pinned.as_mut_slice()[base..base + SLOT_BYTES];
        for s in 0..actual_b {
            let (table, prev3) =
                build_sequence_ngram_table(&dstates[s].pld_tokens, last_tokens[s], cfg);
            let tbl_off = s * NGRAM_TABLE_BYTES;
            table.write_le_bytes_into(&mut dst[tbl_off..tbl_off + NGRAM_TABLE_BYTES])?;
            let p3_off = MAX_TABLES_BYTES + s * NGRAM_PREV3_BYTES;
            for (i, &tok) in prev3.iter().enumerate() {
                dst[p3_off + i * 4..p3_off + i * 4 + 4].copy_from_slice(&tok.to_le_bytes());
            }
        }
        for s in actual_b..padded_b {
            let tbl_off = s * NGRAM_TABLE_BYTES;
            dst[tbl_off..tbl_off + NGRAM_TABLE_BYTES].fill(0);
            let p3_off = MAX_TABLES_BYTES + s * NGRAM_PREV3_BYTES;
            for i in 0..3 {
                dst[p3_off + i * 4..p3_off + i * 4 + 4]
                    .copy_from_slice(&NGRAM_EMPTY_TOK.to_le_bytes());
            }
        }

        let tbl_slice = pinned.pinned_slice_range(base, active_tables_bytes)?;
        let prev3_slice = pinned.pinned_slice_range(base + MAX_TABLES_BYTES, active_prev3_bytes)?;
        // SAFETY: `ngram_upload_pinned` is page-locked and owned by
        // `BatchedNoiseGraphCache`; rotating across `BATCHED_META_PINNED_SLOTS`
        // disjoint slots ensures no in-flight async H2D transfer is overwritten.
        unsafe {
            gpu.copy_h2d_pinned_async(tbl_slice, tables_dev, stream)?;
            gpu.copy_h2d_pinned_async(prev3_slice, prev3_dev, stream)?;
        }
        Ok(())
    }

    /// Pack all Stage-2 per-step dynamic metadata into one rotating page-locked
    /// host slot mirroring [`Stage2MetaLayout`] and issue ONE stream-ordered
    /// async H2D copy directly into the persistent contiguous `self.meta_dev`
    /// slab (14 -> 1 H2D coalescing, zero heap allocation, immutable device
    /// sub-pointers across captured CUDA graph replays).
    ///
    /// Returns `(coupled_seqs, meta_ptrs)` where `coupled_seqs` is the count
    /// of coupled Gumbel/rejection sequences (`temperature > 0.0`).
    #[cfg_attr(not(test), allow(dead_code))]
    #[allow(clippy::too_many_arguments)]
    fn upload_stage2_meta(
        &mut self,
        gpu: &dyn GpuBackend,
        num_layers: usize,
        last_tokens: &[u32],
        positions: &[usize],
        plans: &[SeqContextPlan],
        m_rows: usize,
        mask_id: u32,
        ctx_k_ptr: impl Fn(usize, usize) -> u64,
        ctx_v_ptr: impl Fn(usize, usize) -> u64,
        gumbel_specs: Option<&[Option<spark_runtime::sampler::gumbel::GumbelDraftSpec>]>,
        q_row_bases: Option<&[u32]>,
        stream: u64,
    ) -> Result<(usize, BatchedStage2MetaPtrs)> {
        self.upload_stage2_meta_padded(
            gpu,
            num_layers,
            last_tokens,
            positions,
            plans,
            last_tokens.len(),
            m_rows,
            mask_id,
            ctx_k_ptr,
            ctx_v_ptr,
            gumbel_specs,
            q_row_bases,
            stream,
        )
    }

    /// Pack Stage-2 per-step dynamic metadata for `actual_b = last_tokens.len()`
    /// active sequences padded to `padded_b` (`actual_b <= padded_b <= DFLASH_BATCH_MAX_SEQS`)
    /// so bucketed Stage-2 CUDA graphs (`{1, 2, 4, 8, 16, 24, 32, 48, 64}`) replay cleanly.
    ///
    /// Trailing padded lanes (`actual_b..padded_b`) receive:
    /// - valid in-vocab `last_tokens[0]` anchor + `mask_id` noise tokens and
    ///   small non-negative positions `0..m_rows` (valid embed/RoPE/codebook rows),
    /// - `eff_ctx = 0, ring_start = 0` so `inferspark_prefill_split_ctx_h128_batched`
    ///   never dereferences `K_ctx`/`V_ctx` at all (while still passing valid
    ///   read-only `ctx_{k,v}_ptr(l, 0)` at stride `padded_b`),
    /// - `None` Gumbel spec (`temp = 0.0`, does not increment `coupled_seqs`),
    /// - caller-assigned distinct unallocatable `q_row_bases` sinks in rejection mode.
    #[allow(clippy::too_many_arguments)]
    fn upload_stage2_meta_padded(
        &mut self,
        gpu: &dyn GpuBackend,
        num_layers: usize,
        last_tokens: &[u32],
        positions: &[usize],
        plans: &[SeqContextPlan],
        padded_b: usize,
        m_rows: usize,
        mask_id: u32,
        ctx_k_ptr: impl Fn(usize, usize) -> u64,
        ctx_v_ptr: impl Fn(usize, usize) -> u64,
        gumbel_specs: Option<&[Option<spark_runtime::sampler::gumbel::GumbelDraftSpec>]>,
        q_row_bases: Option<&[u32]>,
        stream: u64,
    ) -> Result<(usize, BatchedStage2MetaPtrs)> {
        let actual_b = last_tokens.len();
        let m_total = padded_b
            .checked_mul(m_rows)
            .context("batched Stage-2 m_total overflow")?;
        anyhow::ensure!(
            actual_b > 0
                && (actual_b..=DFLASH_BATCH_MAX_SEQS).contains(&padded_b)
                && m_rows > 0
                && m_total <= BATCHED_MAX_NOISE_ROWS
                && num_layers > 0
                && positions.len() == actual_b
                && plans.len() == actual_b,
            "batched Stage-2 metadata geometry out of bounds: actual_b={actual_b} padded_b={padded_b} m_rows={m_rows} m_total={m_total} layers={num_layers}"
        );
        if let Some(specs) = gumbel_specs {
            anyhow::ensure!(
                specs.len() == padded_b,
                "batched Stage-2 gumbel_specs len {} != padded_b {padded_b}",
                specs.len()
            );
            anyhow::ensure!(
                specs[actual_b..padded_b].iter().all(Option::is_none),
                "batched Stage-2 padded lanes must have None Gumbel specs"
            );
        }
        if let Some(bases) = q_row_bases {
            anyhow::ensure!(
                bases.len() == padded_b,
                "batched Stage-2 q_row_bases len {} != padded_b {padded_b}",
                bases.len()
            );
        }
        let layout = Stage2MetaLayout::new(num_layers)?;
        if self.meta_dev.is_null() {
            let dev = gpu.alloc(layout.total_bytes)?;
            if let Err(err) = gpu.memset_async(dev, 0, layout.total_bytes, stream) {
                let _ = gpu.free(dev);
                return Err(err);
            }
            self.meta_dev = dev;
            self.meta_dev_bytes = layout.total_bytes;
        }
        if self.meta_pinned.is_none() {
            let mut pinned =
                gpu.alloc_host_pinned(layout.total_bytes * BATCHED_META_PINNED_SLOTS)?;
            pinned.as_mut_slice().fill(0);
            self.meta_pinned = Some(pinned);
            self.meta_slot_bytes = layout.total_bytes;
        }
        anyhow::ensure!(
            layout.total_bytes <= self.meta_dev_bytes && layout.total_bytes <= self.meta_slot_bytes,
            "batched Stage-2 metadata slab too small: need {}, dev={}, pinned={}",
            layout.total_bytes,
            self.meta_dev_bytes,
            self.meta_slot_bytes
        );

        let slot = self.meta_slot;
        self.meta_slot = (self.meta_slot + 1) % BATCHED_META_PINNED_SLOTS;
        let base = slot * self.meta_slot_bytes;

        let pinned = self
            .meta_pinned
            .as_mut()
            .context("batched Stage-2 pinned metadata buffer missing")?;
        let coupled_seqs = {
            let dst = &mut pinned.as_mut_slice()[base..base + layout.total_bytes];
            for s in 0..actual_b {
                let tok_row_base = layout.token_ids_off + s * m_rows * size_of::<u32>();
                dst[tok_row_base..tok_row_base + 4].copy_from_slice(&last_tokens[s].to_le_bytes());
                for r in 1..m_rows {
                    let off = tok_row_base + r * size_of::<u32>();
                    dst[off..off + 4].copy_from_slice(&mask_id.to_le_bytes());
                }
                let pos_row_base = layout.pos_ids_off + s * m_rows * size_of::<u32>();
                for r in 0..m_rows {
                    let off = pos_row_base + r * size_of::<u32>();
                    dst[off..off + 4].copy_from_slice(&((positions[s] + r) as u32).to_le_bytes());
                }
                let seq_u32_off = s * size_of::<u32>();
                dst[layout.eff_ctx_off + seq_u32_off..layout.eff_ctx_off + seq_u32_off + 4]
                    .copy_from_slice(&(plans[s].eff_ctx as u32).to_le_bytes());
                dst[layout.ring_start_off + seq_u32_off..layout.ring_start_off + seq_u32_off + 4]
                    .copy_from_slice(&plans[s].ring_start.to_le_bytes());
                dst[layout.anchor_ids_off + seq_u32_off..layout.anchor_ids_off + seq_u32_off + 4]
                    .copy_from_slice(&last_tokens[s].to_le_bytes());
            }
            let pad_anchor = last_tokens[0];
            for s in actual_b..padded_b {
                let tok_row_base = layout.token_ids_off + s * m_rows * size_of::<u32>();
                dst[tok_row_base..tok_row_base + 4].copy_from_slice(&pad_anchor.to_le_bytes());
                for r in 1..m_rows {
                    let off = tok_row_base + r * size_of::<u32>();
                    dst[off..off + 4].copy_from_slice(&mask_id.to_le_bytes());
                }
                let pos_row_base = layout.pos_ids_off + s * m_rows * size_of::<u32>();
                for r in 0..m_rows {
                    let off = pos_row_base + r * size_of::<u32>();
                    dst[off..off + 4].copy_from_slice(&(r as u32).to_le_bytes());
                }
                let seq_u32_off = s * size_of::<u32>();
                dst[layout.eff_ctx_off + seq_u32_off..layout.eff_ctx_off + seq_u32_off + 4]
                    .copy_from_slice(&0u32.to_le_bytes());
                dst[layout.ring_start_off + seq_u32_off..layout.ring_start_off + seq_u32_off + 4]
                    .copy_from_slice(&0u32.to_le_bytes());
                dst[layout.anchor_ids_off + seq_u32_off..layout.anchor_ids_off + seq_u32_off + 4]
                    .copy_from_slice(&pad_anchor.to_le_bytes());
            }
            for l in 0..num_layers {
                for s in 0..padded_b {
                    let src_s = if s < actual_b { s } else { 0 };
                    let off = (l * padded_b + s) * size_of::<u64>();
                    dst[layout.ctx_k_ptrs_off + off..layout.ctx_k_ptrs_off + off + 8]
                        .copy_from_slice(&ctx_k_ptr(l, src_s).to_le_bytes());
                    dst[layout.ctx_v_ptrs_off + off..layout.ctx_v_ptrs_off + off + 8]
                        .copy_from_slice(&ctx_v_ptr(l, src_s).to_le_bytes());
                }
            }
            if let Some(bases) = q_row_bases {
                for (s, &row_base) in bases.iter().enumerate() {
                    let off = layout.q_row_bases_off + s * size_of::<u32>();
                    dst[off..off + 4].copy_from_slice(&row_base.to_le_bytes());
                }
            }
            if let Some(specs) = gumbel_specs {
                write_batched_gumbel_meta_into(&layout, specs, dst)
            } else {
                0
            }
        };

        let upload_bytes =
            layout.active_upload_bytes(padded_b, gumbel_specs.is_some(), q_row_bases.is_some());
        let slice = pinned.pinned_slice_range(base, upload_bytes)?;
        // SAFETY: `meta_pinned` is page-locked, owned by the head's
        // `BatchedNoiseGraphCache`, and rotates across 4 disjoint slots
        // (with `execute_batched_dflash2` draining `stream` on D2H before
        // returning), so no in-flight H2D source is ever overwritten.
        unsafe {
            gpu.copy_h2d_pinned_async(slice, self.meta_dev, stream)?;
        }
        Ok((coupled_seqs, layout.bind_device(self.meta_dev)))
    }
}

/// Write per-sequence Gumbel metadata directly into the Gumbel segment offsets
/// of a Stage-2 pinned slot (zero allocation). Returns `coupled_seqs`.
fn write_batched_gumbel_meta_into(
    layout: &Stage2MetaLayout,
    specs: &[Option<spark_runtime::sampler::gumbel::GumbelDraftSpec>],
    slot_dst: &mut [u8],
) -> usize {
    let mut coupled_seqs = 0usize;
    for (s, &spec) in specs.iter().enumerate() {
        if spec.is_some_and(|g| g.is_active()) {
            coupled_seqs += 1;
        }
        let (seed, step0, temp, top_k, top_p, min_p) = super::noise_pass::gumbel_spec_fields(spec);
        let u64_off = s * size_of::<u64>();
        let u32_off = s * size_of::<u32>();
        slot_dst[layout.gumbel_seeds_off + u64_off..layout.gumbel_seeds_off + u64_off + 8]
            .copy_from_slice(&seed.to_le_bytes());
        slot_dst[layout.gumbel_step0_off + u64_off..layout.gumbel_step0_off + u64_off + 8]
            .copy_from_slice(&step0.to_le_bytes());
        slot_dst[layout.gumbel_temps_off + u32_off..layout.gumbel_temps_off + u32_off + 4]
            .copy_from_slice(&temp.to_le_bytes());
        slot_dst[layout.gumbel_top_ks_off + u32_off..layout.gumbel_top_ks_off + u32_off + 4]
            .copy_from_slice(&top_k.to_le_bytes());
        slot_dst[layout.gumbel_top_ps_off + u32_off..layout.gumbel_top_ps_off + u32_off + 4]
            .copy_from_slice(&top_p.to_le_bytes());
        slot_dst[layout.gumbel_min_ps_off + u32_off..layout.gumbel_min_ps_off + u32_off + 4]
            .copy_from_slice(&min_p.to_le_bytes());
    }
    coupled_seqs
}

/// Graph-safe single-launch row-compaction parameters for `ops::strided_copy_rows`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct StridedCompactParams {
    src_offset: usize,
    rows: u32,
    row_bytes: u32,
    src_stride: u64,
    dst_stride: u64,
}

fn compute_strided_compact_params(
    b: usize,
    m_rows: usize,
    gamma_eff: usize,
    output_start: usize,
    hidden_size: usize,
    stream_buf: DevicePtr,
    stream_acc: DevicePtr,
) -> Option<StridedCompactParams> {
    let row_bytes = hidden_size.checked_mul(2)?;
    let seq_out_bytes = gamma_eff.checked_mul(row_bytes)?;
    let src_stride = m_rows.checked_mul(row_bytes)?;
    let src_offset = output_start.checked_mul(row_bytes)?;
    let src_ptr = stream_buf.0.checked_add(src_offset as u64)?;
    if seq_out_bytes == 0
        || !seq_out_bytes.is_multiple_of(16)
        || !src_stride.is_multiple_of(16)
        || !src_ptr.is_multiple_of(16)
        || !stream_acc.0.is_multiple_of(16)
    {
        return None;
    }
    Some(StridedCompactParams {
        src_offset,
        rows: u32::try_from(b).ok()?,
        row_bytes: u32::try_from(seq_out_bytes).ok()?,
        src_stride: u64::try_from(src_stride).ok()?,
        dst_stride: u64::try_from(seq_out_bytes).ok()?,
    })
}

#[cfg(test)]
struct BatchedGumbelMetaBytes {
    seeds: Vec<u8>,
    step0_pos: Vec<u8>,
    temps: Vec<u8>,
    top_ks: Vec<u8>,
    top_ps: Vec<u8>,
    min_ps: Vec<u8>,
    coupled_seqs: usize,
}

#[cfg(test)]
fn pack_batched_gumbel_meta(
    specs: &[Option<spark_runtime::sampler::gumbel::GumbelDraftSpec>],
) -> BatchedGumbelMetaBytes {
    let b = specs.len();
    let u32_bytes = b.saturating_mul(size_of::<u32>());
    let u64_bytes = b.saturating_mul(size_of::<u64>());
    let layout = Stage2MetaLayout::new(1).expect("single-layer test layout");
    let mut slot = vec![0u8; layout.total_bytes];
    let coupled_seqs = write_batched_gumbel_meta_into(&layout, specs, &mut slot);
    BatchedGumbelMetaBytes {
        seeds: slot[layout.gumbel_seeds_off..layout.gumbel_seeds_off + u64_bytes].to_vec(),
        step0_pos: slot[layout.gumbel_step0_off..layout.gumbel_step0_off + u64_bytes].to_vec(),
        temps: slot[layout.gumbel_temps_off..layout.gumbel_temps_off + u32_bytes].to_vec(),
        top_ks: slot[layout.gumbel_top_ks_off..layout.gumbel_top_ks_off + u32_bytes].to_vec(),
        top_ps: slot[layout.gumbel_top_ps_off..layout.gumbel_top_ps_off + u32_bytes].to_vec(),
        min_ps: slot[layout.gumbel_min_ps_off..layout.gumbel_min_ps_off + u32_bytes].to_vec(),
        coupled_seqs,
    }
}

impl BlockDiffusionDraftHead {
    /// True cross-sequence batched DFlash2 proposal for `B >= 2` sequences.
    ///
    /// Returns `Ok(None)` without mutating any per-sequence state whenever the
    /// cohort or active runtime policy is ineligible, letting callers fall back
    /// seamlessly to per-sequence [`DraftProposer::propose`].
    #[allow(clippy::too_many_arguments)]
    pub(super) fn propose_batch_dflash2(
        &self,
        last_tokens: &[u32],
        target_hiddens: &[DevicePtr],
        positions: &[usize],
        num_drafts: usize,
        states: &mut [&mut dyn ProposerState],
        ctx: &ForwardContext,
        stream: u64,
        target_hidden_stacks: &[Option<DevicePtr>],
    ) -> Result<Option<Vec<Vec<u32>>>> {
        let b = last_tokens.len();
        if !(2..=DFLASH_BATCH_MAX_SEQS).contains(&b)
            || target_hiddens.len() != b
            || positions.len() != b
            || states.len() != b
            || target_hidden_stacks.len() != b
        {
            return Ok(None);
        }

        // Architectural, kernel, and scratch invariants.
        if self.head_dim != 128
            || self.kernels.split_ctx_h128_batched.0 == 0
            || self.layers.is_empty()
            || self.layers.len() != self.num_layers
            || self.num_q_heads == 0
            || self.num_kv_heads == 0
            || !self.num_q_heads.is_multiple_of(self.num_kv_heads)
            || self.ctx_window == 0
            || self.draft_sample_method
                == crate::dflash_sampling::DflashDraftSampleMethod::Probabilistic
            || self.markov.is_some()
            || self.dspine.is_some()
            || self.w4a4_shadow.is_some()
            || super::kernel_profile_enabled()
        {
            return Ok(None);
        }
        if self.scratch.draft_tokens_dev.is_null()
            || self.scratch.batch_ctx_k_ptrs.is_null()
            || self.scratch.batch_ctx_v_ptrs.is_null()
            || self.scratch.batch_eff_ctx.is_null()
            || self.scratch.batch_ring_start.is_null()
            || self.scratch.batch_anchor_ids.is_null()
        {
            return Ok(None);
        }

        let has_conv = self.layers_have_dflash2_conv();
        let has_batched_gumbel_selector = self.gumbel_selector_k32_ready()
            || self.rejection_selector_ready()
            || (self.kernels.dflash2_selector_walk_gumbel_batched.0 != 0
                && self.gumbel_selector_scratch_ready());
        if self.selector.is_some()
            && (self.kernels.dflash2_conv_prepare_batched.0 == 0
                || self.kernels.dflash2_conv_finish_batched.0 == 0
                || (!has_batched_gumbel_selector
                    && self.kernels.dflash2_selector_walk_batched.0 == 0)
                || self.scratch.selector_hidden.is_null())
        {
            return Ok(None);
        }
        if has_conv {
            if self.kernels.dflash2_conv_prepare_batched.0 == 0
                || self.kernels.dflash2_conv_finish_batched.0 == 0
                || self.scratch.conv_dyn.is_null()
                || self.scratch.conv_dyn1.is_null()
                || self.scratch.conv_out.is_null()
            {
                return Ok(None);
            }
            for layer in &self.layers {
                let (attn_conv, mlp_conv) = match layer {
                    DflashLayerQuantWeights::Bf16(l) => (&l.attention_conv, &l.mlp_conv),
                    DflashLayerQuantWeights::Fp8(l) => (&l.attention_conv, &l.mlp_conv),
                    DflashLayerQuantWeights::Nvfp4(l) => (&l.attention_conv, &l.mlp_conv),
                };
                if attn_conv
                    .as_ref()
                    .is_some_and(|c| !conv_geometry_compatible(c, self.hidden_size))
                    || mlp_conv
                        .as_ref()
                        .is_some_and(|c| !conv_geometry_compatible(c, self.hidden_size))
                {
                    return Ok(None);
                }
            }
        }

        // Single-sequence-only debug/policy modes fall back cleanly to `propose`.
        let policy = self
            .proposal_policy
            .get_or_init(super::proposal_policy::ProposalPolicy::from_env);
        if policy.force_no_ctx
            || policy.force_ctx_used.is_some()
            || policy.debug_dump
            || policy.zero_late > 0
            || policy.hf_override.is_some()
            || policy.force_pattern
            || policy.force_noise_pattern
            || policy.debug_dump_full
            || policy.dump_all_layers
            || policy.fc_layernorm
            || policy.denoise_steps != 1
            || policy.margin_gate > 0.0
            || policy.adaptive_gamma
            || policy.tps_router
            || policy.climbdrop_mode
            || policy.pld
            || policy.portfolio
            || policy.corroborate
            || policy.recycle
            || policy.accept_fallback
            || policy.caterpillar
            || policy.free_slots > 0
            || policy.branch
            || policy.ddtree
            || super::echo::EchoConfig::from_env().is_some()
        {
            return Ok(None);
        }

        // Draft width, row counts, scratch capacities, and target logits buffer.
        let Ok(budget) = super::draft_budget::DflashDraftBudget::new(
            num_drafts,
            super::effective_draft_width(self.gamma),
            self.physical_verify_k,
        ) else {
            return Ok(None);
        };
        let gamma_eff = budget.flat;
        if gamma_eff == 0
            || gamma_eff > self.gamma
            || super::retrieval::RetrievalConfig::from_env(gamma_eff).is_some()
        {
            return Ok(None);
        }
        let row_layout = DraftRowLayout::for_family(self.checkpoint_family, gamma_eff);
        let m_rows = row_layout.query_rows;
        let Some(m_total) = b.checked_mul(m_rows) else {
            return Ok(None);
        };
        let Some(m_out) = b.checked_mul(gamma_eff) else {
            return Ok(None);
        };
        if m_total == 0
            || m_total > BATCHED_MAX_NOISE_ROWS
            || m_out == 0
            || m_out > BATCHED_MAX_NOISE_ROWS
        {
            return Ok(None);
        }

        let max_row_layout = DraftRowLayout::for_family(self.checkpoint_family, self.gamma);
        let n_attn_scratch = max_row_layout.query_rows + self.ctx_window;
        let max_sel_rows = self.gamma.max(DFLASH_BATCH_MAX_SEL_ROWS);
        if m_total > n_attn_scratch || m_out > max_sel_rows {
            return Ok(None);
        }

        let target_hidden_dim = self.target_layer_ids.len() * self.target_hidden_size;
        let bf16 = 2usize;
        if target_hidden_dim == 0 {
            return Ok(None);
        }
        let ctx_slot_bytes = target_hidden_dim * bf16;
        let staging_rows = (n_attn_scratch * self.intermediate_size) / target_hidden_dim;
        let stage1_max_chunk = STAGE1_MAX_CHUNK_ROWS
            .min(self.ctx_window)
            .min(n_attn_scratch)
            .min(staging_rows);
        if stage1_max_chunk == 0 {
            return Ok(None);
        }

        let lm_stride = self.target_vocab_size.min(self.vocab_size);
        if lm_stride == 0 || lm_stride > u32::MAX as usize {
            return Ok(None);
        }
        let Some(needed_logits_bytes) = m_out
            .checked_mul(lm_stride)
            .and_then(|elems| elems.checked_mul(bf16))
        else {
            return Ok(None);
        };
        if ctx.buffers.logits().is_null() || ctx.buffers.sizes().logits < needed_logits_bytes {
            return Ok(None);
        }
        if !self.lm_head_path_valid(policy.lm_head_nvfp4) {
            return Ok(None);
        }
        if let Some(sel) = &self.selector
            && (sel.top_k != super::DDTREE_TOP_K_MAX
                || super::forward_block::checked_topk_layout(
                    m_out,
                    max_sel_rows,
                    lm_stride,
                    sel.top_k,
                )
                .is_err())
        {
            return Ok(None);
        }

        // Read-only per-sequence validation before mutating any state.
        let mut plans = Vec::with_capacity(b);
        for (s, state) in states.iter().enumerate() {
            let Some(dstate) = state.as_any().downcast_ref::<DflashProposerState>() else {
                return Ok(None);
            };
            if dstate.next_sampling_request.is_some() || dstate.active_sampling_request.is_some() {
                return Ok(None);
            }
            let Ok(ring) = dstate.ctx_ring_state() else {
                return Ok(None);
            };
            if dstate.ctx_len != positions[s] {
                return Ok(None);
            }
            if dstate.ctx_valid_from_abs > 0 && dstate.ctx_len <= dstate.ctx_valid_from_abs {
                return Ok(None);
            }
            if positions[s]
                .checked_add(m_rows)
                .is_none_or(|end| end > i32::MAX as usize)
            {
                return Ok(None);
            }
            if dstate.ctx_k_cache.len() != self.num_layers
                || dstate.ctx_v_cache.len() != self.num_layers
                || dstate.cache_k_start.len() != self.num_layers
                || dstate.cache_k_end.len() != self.num_layers
                || dstate.cache_v_start.len() != self.num_layers
                || dstate.cache_v_end.len() != self.num_layers
                || dstate.ctx_k_cache.iter().any(|p| p.is_null())
                || dstate.ctx_v_cache.iter().any(|p| p.is_null())
            {
                return Ok(None);
            }
            if ring.resident_len > 0
                && (dstate.ctx_hidden_acc.is_null() || dstate.ctx_slot_bytes != ctx_slot_bytes)
            {
                return Ok(None);
            }
            plans.push(plan_seq_context_window_with_valid_from(
                ring.resident_len,
                ring.absolute_len,
                dstate.ctx_capacity,
                self.ctx_window,
                dstate.ctx_valid_from_abs,
                dstate.ctx_kv_repr,
                &dstate.cache_k_start,
                &dstate.cache_k_end,
                &dstate.cache_v_start,
                &dstate.cache_v_end,
            ));
        }

        // All guards passed — resolve any stale in-flight single/batched async
        // handle before mutating shared scratch, then downcast mutably and run
        // Stage 1 + Stage 2 (asynchronously on `pstream` with alternating
        // double-buffered output scratch when `ATLAS_DFLASH_ASYNC=1` is active
        // and eligible, or synchronously on `stream` otherwise).
        self.resolve_async_inflight_impl(ctx.gpu, None)?;

        let mut dstates: Vec<&mut DflashProposerState> = Vec::with_capacity(b);
        for state in states.iter_mut() {
            let dstate = state
                .as_any_mut()
                .downcast_mut::<DflashProposerState>()
                .ok_or_else(|| anyhow::anyhow!("Invalid DFlash proposer state"))?;
            dstates.push(dstate);
        }

        let exec = self.execute_batched_dflash2(
            last_tokens,
            positions,
            &plans,
            gamma_eff,
            row_layout,
            budget,
            policy,
            stage1_max_chunk,
            target_hidden_dim,
            lm_stride,
            &mut dstates,
            ctx,
            stream,
        );
        match exec {
            Ok(drafts) => Ok(Some(drafts)),
            Err(error) => {
                // Drain both default and propose streams before returning so any
                // async H2D from `stage1_meta_pinned` / `meta_pinned` finishes
                // before the caller can reuse or drop the pinned staging slabs.
                let _ = ctx.gpu.synchronize(stream);
                if let Some(&pstream) = self.async_propose_stream.get()
                    && pstream != 0
                    && pstream != stream
                {
                    let _ = ctx.gpu.synchronize(pstream);
                }
                for dstate in dstates {
                    invalidate_dstate_ctx_kv_ranges(dstate);
                    dstate.cache_fc_start = 0;
                    dstate.cache_fc_end = 0;
                    dstate.last_draft_probs.clear();
                    dstate.pending_draft_probs_gamma = 0;
                    dstate.async_placeholder = false;
                    dstate.invalidate_rejection_q();
                    super::draft_budget::clear_proposal_outputs(dstate);
                }
                Err(error)
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn execute_batched_dflash2(
        &self,
        last_tokens: &[u32],
        positions: &[usize],
        plans: &[SeqContextPlan],
        gamma_eff: usize,
        row_layout: DraftRowLayout,
        budget: super::draft_budget::DflashDraftBudget,
        policy: &super::proposal_policy::ProposalPolicy,
        stage1_max_chunk: usize,
        target_hidden_dim: usize,
        lm_stride: usize,
        dstates: &mut [&mut DflashProposerState],
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<Vec<Vec<u32>>> {
        let gpu = ctx.gpu;
        let b = last_tokens.len();
        let h = self.hidden_size as u32;
        let q_dim = (self.num_q_heads * self.head_dim) as u32;
        let kv_dim = (self.num_kv_heads * self.head_dim) as u32;
        let inter = self.intermediate_size as u32;
        let inv_sqrt_d = 1.0f32 / (self.head_dim as f32).sqrt();
        let m_rows = row_layout.query_rows;
        let m_rows_u32 = m_rows as u32;
        let m_total = b * m_rows;
        let m_out = b * gamma_eff;
        let mask_id = policy.mask_override.unwrap_or(self.mask_token_id);

        static BATCHED_SEEN: std::sync::Once = std::sync::Once::new();
        BATCHED_SEEN.call_once(|| {
            tracing::info!(
                batch_size = b,
                m_rows,
                gamma_eff,
                has_selector = self.selector.is_some(),
                has_conv = self.layers_have_dflash2_conv(),
                "DFLASH2_BATCHED_PROPOSE_ENGAGED"
            );
        });

        // Clear any stale draft probabilities and rejection q stamps before
        // proposing, then consume each cohort sequence's pending spec + slot
        // only after every read-only eligibility guard has passed (so returning
        // Ok(None) preserves pending specs/slots for serial fallback).
        let mut gumbel_specs_buf = [None; DFLASH_BATCH_MAX_SEQS];
        let mut rejection_slots_buf = [None; DFLASH_BATCH_MAX_SEQS];
        let mut any_coupled_gumbel = false;
        for s in 0..b {
            dstates[s].last_draft_probs.clear();
            dstates[s].pending_draft_probs_gamma = 0;
            dstates[s].invalidate_rejection_q();
            let spec = dstates[s].take_gumbel_spec(positions[s]);
            if spec.is_some_and(|g| g.is_active()) {
                any_coupled_gumbel = true;
            }
            gumbel_specs_buf[s] = spec;
            rejection_slots_buf[s] = dstates[s].take_pending_rejection_slot();
        }

        let rejection_env = spark_runtime::sampler::gumbel::rejection_sampling_enabled();
        let slot_stride = super::noise_pass::REJECTION_SLOT_STRIDE;
        let total_q_slots = self.scratch.shared_q_row_cap / slot_stride;
        let BatchedSelectorPlan {
            rejection_mode,
            use_gumbel_selector,
            shortlist_k,
            selector_mode,
        } = resolve_batched_selector_plan(&BatchedSelectorPlanInput {
            rejection_env,
            any_coupled_gumbel,
            sel_top_k: self.selector.as_ref().map(|sel| sel.top_k),
            rejection_selector_ready: self.rejection_selector_ready(),
            ngram_bonus_ready: super::ngram_bonus::ngram_bonus_enabled()
                && !any_coupled_gumbel
                && self.ngram_selector_ready(),
            shared_q_stride: self.scratch.shared_q_stride,
            shared_q_row_cap: self.scratch.shared_q_row_cap,
            gamma_eff,
            batch_size: b,
            lm_stride,
            has_topk_radix: self.kernels.topk_bf16_radix.0 != 0,
            gumbel_k32_ready: self.gumbel_selector_k32_ready(),
            gumbel_k16_ready: self.kernels.dflash2_selector_walk_gumbel_batched.0 != 0
                && self.gumbel_selector_scratch_ready(),
            has_argmax_rows: self.kernels.argmax_rows.0 != 0,
        });

        let mut staged_specs_buf = [None; DFLASH_BATCH_MAX_SEQS];
        let mut q_row_bases_buf = [0u32; DFLASH_BATCH_MAX_SEQS];
        let mut valid_q_records: [Option<(usize, u64, usize)>; DFLASH_BATCH_MAX_SEQS] =
            [None; DFLASH_BATCH_MAX_SEQS];
        let mut used_slots = [false; DFLASH_BATCH_MAX_SEQS];
        if rejection_mode {
            let temp_scale = spark_runtime::sampler::gumbel::rejection_draft_temp_scale();
            let mut needs_sink = [false; DFLASH_BATCH_MAX_SEQS];
            for s in 0..b {
                let slot_opt = rejection_slots_buf[s];
                let slot_fits = slot_opt.is_some_and(|slot| {
                    slot < total_q_slots
                        && slot < DFLASH_BATCH_MAX_SEQS
                        && !used_slots[slot]
                        && slot * slot_stride + gamma_eff <= self.scratch.shared_q_row_cap
                });
                if let Some(slot) = slot_opt
                    && slot_fits
                {
                    let (eff_spec, row_base, record) =
                        super::noise_pass::resolve_rejection_seq_staging(
                            gumbel_specs_buf[s],
                            Some(slot),
                            gamma_eff,
                            self.scratch.shared_q_row_cap,
                            temp_scale,
                        );
                    staged_specs_buf[s] = eff_spec;
                    q_row_bases_buf[s] = if eff_spec.is_some() {
                        row_base
                    } else {
                        (slot * slot_stride) as u32
                    };
                    valid_q_records[s] = record;
                    used_slots[slot] = true;
                } else {
                    staged_specs_buf[s] = None;
                    valid_q_records[s] = None;
                    needs_sink[s] = true;
                }
            }
            let mut next_sink = total_q_slots.min(DFLASH_BATCH_MAX_SEQS);
            for s in 0..b {
                if needs_sink[s] {
                    while next_sink > 0 {
                        next_sink -= 1;
                        if !used_slots[next_sink] {
                            used_slots[next_sink] = true;
                            q_row_bases_buf[s] = (next_sink * slot_stride) as u32;
                            break;
                        }
                    }
                }
            }
        } else if !rejection_env {
            staged_specs_buf[..b].copy_from_slice(&gumbel_specs_buf[..b]);
        }

        let writes_out_probs = matches!(
            selector_mode,
            BatchedSelectorMode::GumbelK32 { .. }
                | BatchedSelectorMode::ProbabilisticRejection { .. }
                | BatchedSelectorMode::GreedyNgramBatched { .. }
        );

        // Select execution stream and double-buffered output slot.
        // When `ATLAS_DFLASH_ASYNC=1` is active and eligible (and secondary
        // scratch slots are allocated), run Stage 1 + Stage 2 on `pstream`
        // writing to alternating `Primary`/`Secondary` output buffers and defer
        // sync + D2H to `collect_async_drafts` at the top of the next step.
        let async_batch_slot = if super::async_propose::dflash_async_enabled()
            && super::async_propose::async_batched_launch_eligible(
                super::async_propose::AsyncEnvEligibility::cached(),
                self.markov.is_some(),
            )
            && !self.scratch.draft_tokens_dev_alt.is_null()
            && (!writes_out_probs || !self.scratch.batch_out_probs_alt.is_null())
        {
            let pstream = self.propose_stream_lazy(gpu);
            if pstream != 0 {
                self.order_pstream_for_batched_launch(gpu, stream, pstream)?;
                Some((pstream, self.next_async_batch_slot()))
            } else {
                None
            }
        } else {
            None
        };
        let (exec_stream, output_slot) = match async_batch_slot {
            Some((pstream, slot)) => (pstream, slot),
            None => (stream, super::async_propose::AsyncScratchSlot::Primary),
        };
        let draft_tokens_out = self.scratch.draft_tokens_dev_for_slot(output_slot);
        let batch_out_probs_out = self.scratch.batch_out_probs_for_slot(output_slot);

        // ── Stage 1: Batched incremental PostRope context append ──
        for (s, plan) in plans.iter().enumerate() {
            if plan.invalidate_prev {
                invalidate_dstate_ctx_kv_ranges(dstates[s]);
                dstates[s].ctx_kv_repr = Some(CtxKvCacheRepr::PostRope);
            }
        }
        let stage1_chunks = plan_stage1_chunks(plans, stage1_max_chunk);
        if !stage1_chunks.is_empty() {
            self.run_stage1_batched_ctx_append(
                &stage1_chunks,
                target_hidden_dim,
                h,
                kv_dim,
                policy.prefill_pipe,
                dstates,
                gpu,
                exec_stream,
            )?;
        }
        for (s, plan) in plans.iter().enumerate() {
            for l in 0..self.num_layers {
                dstates[s].cache_k_start[l] = plan.needed_start;
                dstates[s].cache_k_end[l] = plan.needed_end;
                dstates[s].cache_v_start[l] = plan.needed_start;
                dstates[s].cache_v_end[l] = plan.needed_end;
            }
            dstates[s].ctx_kv_repr = Some(CtxKvCacheRepr::PostRope);
        }

        // ── Stage 2: Batched noise block (M = B * m_rows, M_out = B * gamma_eff) ──
        static PROPOSE_BATCH_RECEIPT: std::sync::Once = std::sync::Once::new();
        PROPOSE_BATCH_RECEIPT.call_once(|| {
            tracing::info!(
                batch_size = b,
                m_rows,
                gamma = gamma_eff,
                m_total,
                m_out,
                shortlist_k,
                quant = ?self.quant,
                "DFLASH_PROPOSE_BATCH_ENGAGED"
            );
        });

        let logits_out = ctx.buffers.logits();
        let lm_head_tag = resolve_lm_head_path_tag(
            policy.lm_head_nvfp4,
            self.lm_head_shared_t.is_some(),
            self.kernels.w4a16_gemm_t_m32_n64.0 != 0,
            self.lm_head_shared_fp8.is_some(),
        )?;

        // 1. Bucket Stage-2 noise graph `batch_size` to `{1, 2, 4, 8, 16, 24, 32, 48, 64}` when
        //    graphs are enabled and all scratch/logits/rejection-sink bounds
        //    hold, then pack & upload all Stage-2 dynamic metadata into
        //    rotating pinned host staging via ONE coalesced H2D copy into the
        //    persistent contiguous `meta_dev` slab (zero per-step Vec
        //    allocations, zero stream sync), and check the Stage-2 graph cache.
        let graph_enabled = dflash_noise_graph_enabled() && exec_stream != 0;
        let max_row_layout = DraftRowLayout::for_family(self.checkpoint_family, self.gamma);
        let n_attn_scratch = max_row_layout.query_rows + self.ctx_window;
        let max_sel_rows = self.gamma.max(DFLASH_BATCH_MAX_SEL_ROWS);
        let configured_lanes = crate::layers::qwen4_verify_lanes::lanes();
        let available_rejection_sinks = if rejection_mode {
            count_dedicated_rejection_q_sinks(
                configured_lanes,
                total_q_slots,
                slot_stride,
                gamma_eff,
                self.scratch.shared_q_row_cap,
                &used_slots,
                &rejection_slots_buf[..b],
            )
        } else {
            0
        };
        let mut padded_b = resolve_padded_noise_batch_size(&PaddedNoiseBatchInput {
            actual_b: b,
            graph_enabled,
            m_rows,
            gamma_eff,
            n_attn_scratch,
            max_sel_rows,
            lm_stride,
            logits_bytes: ctx.buffers.sizes().logits,
            rejection_mode,
            available_rejection_sinks,
        });
        if rejection_mode
            && padded_b > b
            && !assign_padded_rejection_q_sinks(
                b,
                padded_b,
                configured_lanes,
                total_q_slots,
                slot_stride,
                gamma_eff,
                self.scratch.shared_q_row_cap,
                &mut used_slots,
                &mut q_row_bases_buf,
            )
        {
            padded_b = b;
        }
        let padded_m_total = padded_b * m_rows;
        let padded_m_total_u32 = padded_m_total as u32;
        let padded_m_out = padded_b * gamma_eff;

        let sm_count = atlas_core::device::active_sm_count();
        let max_eff_ctx = plans.iter().map(|p| p.eff_ctx).max().unwrap_or(0);
        let desired_split_kv = super::noise_pass::resolve_dflash_split_kv_splits_for_sm(
            self.head_dim,
            max_eff_ctx,
            self.num_q_heads,
            padded_b,
            sm_count,
        );
        let max_split_kv_batch = if self.num_q_heads > 0 {
            ((sm_count as usize) / self.num_q_heads).clamp(padded_b, DFLASH_BATCH_MAX_SEQS)
        } else {
            padded_b
        };
        let ((coupled_seqs, meta_ptrs), split_kv_plan, graph_key, cached_graph) = {
            let mut cache = self.batched_noise_graph.lock();
            let (coupled, ptrs) = cache.upload_stage2_meta_padded(
                gpu,
                self.num_layers,
                last_tokens,
                positions,
                plans,
                padded_b,
                m_rows,
                mask_id,
                |l, s| dstates[s].ctx_k_cache[l].0,
                |l, s| dstates[s].ctx_v_cache[l].0,
                (use_gumbel_selector || rejection_mode).then_some(&staged_specs_buf[..padded_b]),
                rejection_mode.then_some(&q_row_bases_buf[..padded_b]),
                exec_stream,
            )?;
            if matches!(
                selector_mode,
                BatchedSelectorMode::GreedyNgramBatched { .. }
            ) {
                cache.upload_ngram_tables_batched(
                    gpu,
                    dstates,
                    last_tokens,
                    padded_b,
                    self.scratch.ngram_tables_dev,
                    self.scratch.ngram_prev3_dev,
                    exec_stream,
                )?;
            }
            let split_kv_plan = if desired_split_kv > 1
                && let Some((split_kernel, reduce_kernel)) =
                    cache.resolve_batched_split_kv_h128_kernels(gpu)
                && let Some(needed_bytes) = super::noise_pass::dflash_split_kv_h128_workspace_bytes(
                    padded_b,
                    desired_split_kv,
                    self.num_q_heads,
                    m_rows,
                )
                && let Some(alloc_bytes) = super::noise_pass::dflash_split_kv_h128_workspace_bytes(
                    max_split_kv_batch,
                    8,
                    self.num_q_heads,
                    max_row_layout.query_rows.max(m_rows),
                )
                && let Some(workspace) =
                    cache.ensure_split_kv_workspace(gpu, alloc_bytes, needed_bytes)?
            {
                Some(BatchedSplitKvH128Plan {
                    split_kernel,
                    reduce_kernel,
                    workspace,
                    num_splits: desired_split_kv,
                })
            } else {
                None
            };
            let key = BatchedNoiseGraphKey {
                batch_size: padded_b,
                gamma_eff,
                m_rows,
                shortlist_k,
                selector_mode,
                lm_vocab: lm_stride as u32,
                lm_head_path_tag: lm_head_tag,
                split_kv_splits: split_kv_plan.map_or(1u8, |p| p.num_splits as u8),
                output_slot: output_slot.as_u8(),
                logits_out_ptr: logits_out.0,
                meta_dev_ptr: ptrs.base.0,
                stream: exec_stream,
            };
            let cached = if graph_enabled {
                cache.graphs.get(&key).copied()
            } else {
                None
            };
            ((coupled, ptrs), split_kv_plan, key, cached)
        };
        if let Some(plan) = split_kv_plan {
            static BATCHED_SPLIT_KV_RECEIPT: std::sync::Once = std::sync::Once::new();
            BATCHED_SPLIT_KV_RECEIPT.call_once(|| {
                tracing::info!(
                    batch_size = padded_b,
                    num_q_heads = self.num_q_heads,
                    num_splits = plan.num_splits,
                    max_eff_ctx,
                    sm_count,
                    "DFLASH_BATCHED_SPLIT_KV_H128_ENGAGED"
                );
            });
        }

        // 2–8. Fixed-shape Stage-2 noise block: CUDA graph replay/capture when
        //      noise graph is enabled and stream != 0, with fail-closed
        //      rollback to eager execution on capture error.
        if let Some(graph) = cached_graph {
            gpu.launch_graph(graph, exec_stream)?;
        } else if graph_enabled {
            crate::layers::qwen4_fast_proj::warm_coalesced_nvfp4_kernels(gpu);
            let bf16_lm_head_fallback = (lm_head_tag == LM_HEAD_PATH_TAG_BF16
                && self.lm_head_shared_nvfp4_exact.is_none())
            .then_some((logits_out, lm_stride as u32));
            self.prewarm_stage2_cublas(
                padded_m_total_u32,
                padded_m_out as u32,
                h,
                q_dim,
                kv_dim,
                inter,
                bf16_lm_head_fallback,
                exec_stream,
            )?;
            self.prewarm_batched_stage2_nvfp4(
                gpu,
                padded_m_total_u32,
                h,
                q_dim,
                kv_dim,
                inter,
                exec_stream,
            )?;
            let captured = (|| -> Result<GraphHandle> {
                let capture = NoiseGraphCaptureGuard::begin(gpu, exec_stream)?;
                self.launch_batched_stage2_body(
                    meta_ptrs,
                    padded_b,
                    m_rows_u32,
                    padded_m_total_u32,
                    padded_m_out,
                    gamma_eff,
                    row_layout,
                    h,
                    q_dim,
                    kv_dim,
                    inter,
                    inv_sqrt_d,
                    lm_stride,
                    logits_out,
                    draft_tokens_out,
                    batch_out_probs_out,
                    policy.lm_head_nvfp4,
                    shortlist_k,
                    selector_mode,
                    split_kv_plan,
                    gpu,
                    exec_stream,
                )?;
                capture.end()
            })();
            match captured {
                Ok(graph) if graph.0 != 0 => {
                    self.batched_noise_graph
                        .lock()
                        .insert_graph(gpu, graph_key, graph);
                    gpu.launch_graph(graph, exec_stream)?;
                    static BATCHED_GRAPH_RECEIPT: std::sync::Once = std::sync::Once::new();
                    BATCHED_GRAPH_RECEIPT.call_once(|| {
                        tracing::info!(
                            batch_size = padded_b,
                            actual_batch_size = b,
                            gamma = gamma_eff,
                            m_rows,
                            shortlist_k,
                            selector_mode = ?selector_mode,
                            lm_vocab = lm_stride,
                            "DFLASH_BATCHED_NOISE_GRAPH_ENGAGED"
                        );
                    });
                }
                Ok(_) => {
                    // Mock/no-op capture backends execute kernels directly inside
                    // `begin_capture`..`end_capture` and return `GraphHandle(0)`,
                    // so Stage 2 has already run once on `exec_stream`.
                }
                Err(capture_err) => {
                    static CAPTURE_FAIL_WARN: std::sync::Once = std::sync::Once::new();
                    CAPTURE_FAIL_WARN.call_once(|| {
                        tracing::warn!(
                            batch_size = padded_b,
                            actual_batch_size = b,
                            gamma = gamma_eff,
                            m_rows,
                            error = %capture_err,
                            "DFLASH_BATCHED_NOISE_GRAPH capture failed; falling back to eager"
                        );
                    });
                    self.launch_batched_stage2_body(
                        meta_ptrs,
                        padded_b,
                        m_rows_u32,
                        padded_m_total_u32,
                        padded_m_out,
                        gamma_eff,
                        row_layout,
                        h,
                        q_dim,
                        kv_dim,
                        inter,
                        inv_sqrt_d,
                        lm_stride,
                        logits_out,
                        draft_tokens_out,
                        batch_out_probs_out,
                        policy.lm_head_nvfp4,
                        shortlist_k,
                        selector_mode,
                        split_kv_plan,
                        gpu,
                        exec_stream,
                    )?;
                }
            }
        } else {
            self.launch_batched_stage2_body(
                meta_ptrs,
                padded_b,
                m_rows_u32,
                padded_m_total_u32,
                padded_m_out,
                gamma_eff,
                row_layout,
                h,
                q_dim,
                kv_dim,
                inter,
                inv_sqrt_d,
                lm_stride,
                logits_out,
                draft_tokens_out,
                batch_out_probs_out,
                policy.lm_head_nvfp4,
                shortlist_k,
                selector_mode,
                split_kv_plan,
                gpu,
                exec_stream,
            )?;
        }

        if use_gumbel_selector && coupled_seqs > 0 {
            static BATCHED_GUMBEL_RECEIPT: std::sync::Once = std::sync::Once::new();
            BATCHED_GUMBEL_RECEIPT.call_once(|| {
                tracing::info!(
                    batch_size = b,
                    coupled_seqs,
                    shortlist_k,
                    "DFLASH2_COUPLED_GUMBEL_ENGAGED mode=batched"
                );
            });
        }

        // Async batched overlap (`ATLAS_DFLASH_ASYNC=1`): defer stream sync +
        // D2H readback to the top of the next scheduler step, registering the
        // cohort under `AsyncBatchedInflight` with double-buffered output slot
        // `output_slot` and returning lossless MASK placeholder chains now.
        if let Some((pstream, slot)) = async_batch_slot {
            let mut owners = Vec::with_capacity(b);
            let mut results = Vec::with_capacity(b);
            for dstate in dstates.iter_mut() {
                owners.push(super::async_propose::dstate_id(dstate));
                dstate.draft_budget = Some(budget);
                dstate.last_num_drafted = gamma_eff;
                dstate.first_propose_done = true;
                dstate.pending_tree_payload = None;
                dstate.pending_sampled_proposal = None;
                dstate.active_sampling_request = None;
                dstate.async_placeholder = true;
                dstate.last_draft_probs.clear();
                dstate.pending_draft_probs_gamma = 0;
                dstate.invalidate_rejection_q();
                results.push(super::async_propose::placeholder_drafts(gamma_eff, mask_id));
            }
            self.register_async_batched_inflight(
                slot,
                owners,
                budget,
                gamma_eff,
                pstream,
                writes_out_probs,
                rejection_mode,
                valid_q_records[..b].to_vec(),
            );
            return Ok(results);
        }

        // 9. Single stream-ordered sync + D2H readback of draft tokens (and
        //    optional `batch_out_probs` via `copy_d2h_pair_on_stream`, avoiding
        //    a redundant second stream synchronization) using stack buffers
        //    (zero alloc), then split into B chains.
        let out_bytes = m_out * size_of::<u32>();
        let mut tok_host_buf = [0u8; BATCHED_MAX_NOISE_ROWS * size_of::<u32>()];
        let mut probs_host_buf = [0u8; BATCHED_MAX_NOISE_ROWS * size_of::<f32>()];
        if writes_out_probs {
            gpu.copy_d2h_pair_on_stream(
                draft_tokens_out,
                &mut tok_host_buf[..out_bytes],
                batch_out_probs_out,
                &mut probs_host_buf[..out_bytes],
                exec_stream,
            )?;
        } else {
            gpu.copy_d2h_on_stream(
                draft_tokens_out,
                &mut tok_host_buf[..out_bytes],
                exec_stream,
            )?;
        }
        let flat_tokens =
            super::proposal_inputs::decode_token_ids(&tok_host_buf[..out_bytes], m_out)?;

        let mut results = Vec::with_capacity(b);
        for s in 0..b {
            let start = s * gamma_eff;
            let chain = flat_tokens[start..start + gamma_eff].to_vec();
            let dstate: &mut DflashProposerState = dstates[s];
            dstate.draft_budget = Some(budget);
            dstate.last_num_drafted = chain.len();
            dstate.first_propose_done = true;
            dstate.pending_tree_payload = None;
            dstate.pending_sampled_proposal = None;
            dstate.active_sampling_request = None;
            dstate.async_placeholder = false;
            if rejection_mode {
                dstate.set_last_rejection_q(valid_q_records[s]);
            } else {
                dstate.invalidate_rejection_q();
            }
            if writes_out_probs {
                let prob_slice = &probs_host_buf
                    [start * size_of::<f32>()..(start + gamma_eff) * size_of::<f32>()];
                dstate.last_draft_probs.clear();
                dstate.last_draft_probs.extend(
                    prob_slice
                        .chunks_exact(size_of::<f32>())
                        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])),
                );
            }
            results.push(chain);
        }
        Ok(results)
    }

    /// Pre-swizzle all batched Stage-2 NVFP4 drafter weights and warm cuBLASLt
    /// NVFP4 activation scratch + plans strictly before `NoiseGraphCaptureGuard::begin`
    /// so Stage-2 CUDA graph capture is 100% allocation-free.
    #[allow(clippy::too_many_arguments)]
    fn prewarm_batched_stage2_nvfp4(
        &self,
        gpu: &dyn GpuBackend,
        m_total: u32,
        h: u32,
        q_dim: u32,
        kv_dim: u32,
        inter: u32,
        stream: u64,
    ) -> Result<()> {
        let m = m_total as usize;
        if !batched_drafter_nvfp4_tc_active(m) {
            return Ok(());
        }
        let h_u = h as usize;
        let q_u = q_dim as usize;
        let kv_u = kv_dim as usize;
        let inter_u = inter as usize;
        let mut warmed_plans = false;
        for layer in &self.layers {
            let DflashLayerQuantWeights::Nvfp4(l) = layer else {
                continue;
            };
            for (weight, n, k) in [
                (&l.q_proj, q_u, h_u),
                (&l.k_proj, kv_u, h_u),
                (&l.v_proj, kv_u, h_u),
                (&l.o_proj, h_u, q_u),
                (&l.gate_proj, inter_u, h_u),
                (&l.up_proj, inter_u, h_u),
                (&l.down_proj, h_u, inter_u),
            ] {
                let _ = crate::layers::qwen4_fast_proj::nvfp4::prewarm_weight_and_scratch(
                    gpu, weight, m, n, k, None, stream,
                )?;
            }
            if !warmed_plans {
                // Warm one cuBLASLt NVFP4 GEMM per distinct Stage-2 `(m, n, k)` shape
                // into dead scratch before capture starts.
                let _ = self.try_batched_drafter_nvfp4_gemm(
                    gpu,
                    self.scratch.norm_buf,
                    &l.q_proj,
                    self.scratch.q_buf,
                    m_total,
                    q_dim,
                    h,
                    stream,
                )?;
                let _ = self.try_batched_drafter_nvfp4_pair(
                    gpu,
                    self.scratch.norm_buf,
                    &l.k_proj,
                    self.scratch.k_buf,
                    &l.v_proj,
                    self.scratch.v_buf,
                    m_total,
                    kv_dim,
                    h,
                    stream,
                )?;
                let _ = self.try_batched_drafter_nvfp4_gemm(
                    gpu,
                    self.scratch.attn_out,
                    &l.o_proj,
                    self.scratch.stream_acc,
                    m_total,
                    h,
                    q_dim,
                    stream,
                )?;
                let _ = self.try_batched_drafter_nvfp4_pair(
                    gpu,
                    self.scratch.norm_buf,
                    &l.gate_proj,
                    self.scratch.mlp_intermediate,
                    &l.up_proj,
                    self.scratch.mlp_up,
                    m_total,
                    inter,
                    h,
                    stream,
                )?;
                let _ = self.try_batched_drafter_nvfp4_gemm(
                    gpu,
                    self.scratch.mlp_intermediate,
                    &l.down_proj,
                    self.scratch.stream_acc,
                    m_total,
                    h,
                    inter,
                    stream,
                )?;
                warmed_plans = true;
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn try_batched_drafter_nvfp4_gemm(
        &self,
        gpu: &dyn GpuBackend,
        act: DevicePtr,
        weight: &crate::weight_map::QuantizedWeight,
        out: DevicePtr,
        m: u32,
        n: u32,
        k: u32,
        stream: u64,
    ) -> Result<bool> {
        let m_u = m as usize;
        if !batched_drafter_nvfp4_tc_active(m_u) {
            return Ok(false);
        }
        let n_u = n as usize;
        let k_u = k as usize;
        let ran = crate::layers::qwen4_fast_proj::nvfp4::try_gemm_bf16(
            gpu, act, weight, out, m_u, n_u, k_u, n_u, None, true, stream,
        )?;
        if ran {
            static LOGGED: std::sync::Once = std::sync::Once::new();
            LOGGED.call_once(|| {
                tracing::info!(
                    m = m_u,
                    n = n_u,
                    k = k_u,
                    "DFLASH_BATCHED_DRAFTER_NVFP4_TC_ENGAGED"
                );
            });
        }
        Ok(ran)
    }

    /// Dispatch a shared-activation NVFP4 pair (`[k_proj, v_proj]` or
    /// `[gate_proj, up_proj]`) with a single BF16 -> NVFP4 activation quant pass.
    #[allow(clippy::too_many_arguments)]
    fn try_batched_drafter_nvfp4_pair(
        &self,
        gpu: &dyn GpuBackend,
        act: DevicePtr,
        w0: &crate::weight_map::QuantizedWeight,
        out0: DevicePtr,
        w1: &crate::weight_map::QuantizedWeight,
        out1: DevicePtr,
        m: u32,
        n: u32,
        k: u32,
        stream: u64,
    ) -> Result<bool> {
        let m_u = m as usize;
        if !batched_drafter_nvfp4_tc_active(m_u) {
            return Ok(false);
        }
        let n_u = n as usize;
        let k_u = k as usize;
        let _tc_guard = crate::layers::qwen4_hyper::BatchedVerifyTcDenseGuard::enter(true);
        let outs = [
            crate::layers::qwen4_fast_proj::SharedActOut {
                weight: w0,
                out: out0,
                n: n_u,
                qg: None,
            },
            crate::layers::qwen4_fast_proj::SharedActOut {
                weight: w1,
                out: out1,
                n: n_u,
                qg: None,
            },
        ];
        let outcome = crate::layers::qwen4_fast_proj::nvfp4::try_gemm_shared_act_all(
            gpu, act, &outs, m_u, k_u, n_u, stream,
        )?;
        Ok(outcome == crate::layers::qwen4_fast_proj::nvfp4::SharedActNvfp4Outcome::HandledAll)
    }

    /// Fixed-shape, 100% CUDA-graph-capturable Stage-2 body for cross-sequence
    /// batched DFlash2: every per-step dynamic input is read from the fixed
    /// contiguous `meta_ptrs` device slab uploaded prior to launch via ONE
    /// coalesced H2D copy, while `self.scratch.draft_tokens_dev` holds only
    /// the output draft token IDs for post-graph D2H readback.
    #[allow(clippy::too_many_arguments)]
    fn launch_batched_stage2_body(
        &self,
        meta_ptrs: BatchedStage2MetaPtrs,
        b: usize,
        m_rows_u32: u32,
        m_total_u32: u32,
        m_out: usize,
        gamma_eff: usize,
        row_layout: DraftRowLayout,
        h: u32,
        q_dim: u32,
        kv_dim: u32,
        inter: u32,
        inv_sqrt_d: f32,
        lm_stride: usize,
        logits_out: DevicePtr,
        draft_tokens_out: DevicePtr,
        batch_out_probs_out: DevicePtr,
        lm_head_nvfp4: bool,
        shortlist_k: usize,
        selector_mode: BatchedSelectorMode,
        split_kv: Option<BatchedSplitKvH128Plan>,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        let bf16 = 2usize;
        let m_rows = row_layout.query_rows;

        // 2. 1 batched embed (`m_total` rows) -> `scratch.stream_buf`.
        ops::batched_embed(
            gpu,
            self.kernels.batched_embed,
            meta_ptrs.draft_tokens_dev,
            self.embed_tokens_shared,
            self.scratch.stream_buf,
            m_total_u32,
            h,
            stream,
        )?;

        // 3. Drafter transformer layers across all `m_total = B * m_rows` rows.
        let swa_on = noise_graph_swa_enabled();
        let ring_cap = self.ctx_window as u32;
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let layer_window = if swa_on {
                self.layer_window_sizes.get(layer_idx).copied().unwrap_or(0)
            } else {
                0
            };
            let layer_causal = if swa_on {
                self.layer_causal.get(layer_idx).copied().unwrap_or(false)
            } else {
                false
            };
            let ptrs_off = layer_idx * b * size_of::<u64>();
            let k_ctx_ptrs = meta_ptrs.batch_ctx_k_ptrs.offset(ptrs_off);
            let v_ctx_ptrs = meta_ptrs.batch_ctx_v_ptrs.offset(ptrs_off);

            match layer {
                DflashLayerQuantWeights::Bf16(l) => self.batched_layer_bf16(
                    l,
                    meta_ptrs,
                    k_ctx_ptrs,
                    v_ctx_ptrs,
                    ring_cap,
                    b as u32,
                    m_rows_u32,
                    m_total_u32,
                    h,
                    q_dim,
                    kv_dim,
                    inter,
                    inv_sqrt_d,
                    layer_causal,
                    layer_window,
                    split_kv,
                    gpu,
                    stream,
                )?,
                DflashLayerQuantWeights::Fp8(l) => self.batched_layer_fp8(
                    l,
                    meta_ptrs,
                    k_ctx_ptrs,
                    v_ctx_ptrs,
                    ring_cap,
                    b as u32,
                    m_rows_u32,
                    m_total_u32,
                    h,
                    q_dim,
                    kv_dim,
                    inter,
                    inv_sqrt_d,
                    layer_causal,
                    layer_window,
                    split_kv,
                    gpu,
                    stream,
                )?,
                DflashLayerQuantWeights::Nvfp4(l) => self.batched_layer_nvfp4(
                    l,
                    meta_ptrs,
                    k_ctx_ptrs,
                    v_ctx_ptrs,
                    ring_cap,
                    b as u32,
                    m_rows_u32,
                    m_total_u32,
                    h,
                    q_dim,
                    kv_dim,
                    inter,
                    inv_sqrt_d,
                    layer_causal,
                    layer_window,
                    split_kv,
                    gpu,
                    stream,
                )?,
            }
        }

        // 4. Compact `M_out = B * gamma_eff` output rows into contiguous
        //    `scratch.stream_acc[0..M_out]` in a single graph-safe strided-copy
        //    kernel launch (no copy needed when output covers every query row,
        //    i.e. DSpark layout `output_start == 0`).
        let compact_stream = if row_layout.output_start == 0 && m_rows == gamma_eff {
            self.scratch.stream_buf
        } else {
            if self.kernels.strided_copy_rows.0 != 0
                && let Some(params) = compute_strided_compact_params(
                    b,
                    m_rows,
                    gamma_eff,
                    row_layout.output_start,
                    self.hidden_size,
                    self.scratch.stream_buf,
                    self.scratch.stream_acc,
                )
            {
                ops::strided_copy_rows(
                    gpu,
                    self.kernels.strided_copy_rows,
                    self.scratch.stream_buf.offset(params.src_offset),
                    self.scratch.stream_acc,
                    params.rows,
                    params.row_bytes,
                    params.src_stride,
                    params.dst_stride,
                    stream,
                )?;
            } else {
                let row_bytes = self.hidden_size * bf16;
                let seq_out_bytes = gamma_eff * row_bytes;
                for s in 0..b {
                    let src = self
                        .scratch
                        .stream_buf
                        .offset((s * m_rows + row_layout.output_start) * row_bytes);
                    let dst = self.scratch.stream_acc.offset(s * seq_out_bytes);
                    gpu.copy_d2d_async(src, dst, seq_out_bytes, stream)?;
                }
            }
            self.scratch.stream_acc
        };

        // 5–6. Final RMSNorm + shared LM head (Tensor Core coalesced NVFP4
        //      W4A16 GEMM across all C>=2 cohorts, including m_out > 32)
        //      directly into `ctx.buffers.logits()` (zero extra VRAM),
        //      restoring unscaled `norm_buf` when FP8 feeds the selector.
        self.run_final_norm_and_lm_head(
            gpu,
            compact_stream,
            self.scratch.norm_buf,
            logits_out,
            m_out,
            lm_stride as u32,
            h,
            lm_head_nvfp4,
            LmHeadNvfp4Mode::CoalescedTc,
            stream,
        )?;

        // 7–8. Candidate selector walk (batched K<=32 / K=16) OR row-parallel argmax.
        if let Some(sel) = &self.selector {
            let max_sel_rows = self.gamma.max(DFLASH_BATCH_MAX_SEL_ROWS);
            let use_radix_topk = matches!(
                selector_mode,
                BatchedSelectorMode::ProbabilisticRejection {
                    use_radix_topk: true
                } | BatchedSelectorMode::GreedyNgramBatched {
                    use_radix_topk: true
                } | BatchedSelectorMode::GumbelK32 {
                    use_radix_topk: true
                } | BatchedSelectorMode::GumbelK16 {
                    use_radix_topk: true
                } | BatchedSelectorMode::GreedyBatched {
                    use_radix_topk: true
                }
            );
            if use_radix_topk {
                let (num_rows, vocab_u32, k_u32) = validate_selector_shortlist(
                    m_out,
                    max_sel_rows,
                    lm_stride,
                    shortlist_k,
                    DFLASH2_GUMBEL_SHORTLIST_K,
                )?;
                ops::topk_bf16_fast(
                    gpu,
                    self.kernels.topk_bf16_radix,
                    logits_out,
                    self.scratch.topk_tokens_dev,
                    self.scratch.topk_logits_dev,
                    num_rows,
                    vocab_u32,
                    k_u32,
                    stream,
                )?;
            } else {
                let topk_layout = super::forward_block::checked_topk_layout(
                    m_out,
                    max_sel_rows,
                    lm_stride,
                    shortlist_k,
                )?;
                ops::topk_bf16(
                    gpu,
                    self.kernels.topk,
                    logits_out,
                    self.scratch.topk_tokens_dev,
                    self.scratch.topk_logits_dev,
                    topk_layout.num_rows,
                    topk_layout.vocab,
                    topk_layout.k as u32,
                    stream,
                )?;
            }
            self.project_bf16_dense(
                gpu,
                self.scratch.norm_buf,
                &sel.hidden_projection,
                self.scratch.selector_hidden,
                m_out as u32,
                sel.rank as u32,
                h,
                stream,
            )?;
            match selector_mode {
                BatchedSelectorMode::ProbabilisticRejection { .. } => {
                    ops::dflash2_selector_walk_probabilistic_batched(
                        gpu,
                        self.kernels.dflash2_selector_walk_probabilistic_batched,
                        self.scratch.topk_logits_dev,
                        self.scratch.topk_tokens_dev,
                        self.scratch.selector_hidden,
                        sel.predecessor_codebook.weight,
                        sel.successor_codebook.weight,
                        draft_tokens_out,
                        batch_out_probs_out,
                        self.scratch.shared_q_ids,
                        self.scratch.shared_q_probs,
                        Some(meta_ptrs.q_row_bases),
                        meta_ptrs.batch_anchor_ids,
                        meta_ptrs.batch_gumbel_seeds,
                        meta_ptrs.batch_gumbel_step0_pos,
                        meta_ptrs.batch_gumbel_temps,
                        meta_ptrs.batch_gumbel_top_ks,
                        meta_ptrs.batch_gumbel_top_ps,
                        meta_ptrs.batch_gumbel_min_ps,
                        shortlist_k,
                        self.scratch.shared_q_stride,
                        gamma_eff,
                        sel.rank,
                        b,
                        stream,
                    )?;
                }
                BatchedSelectorMode::GreedyNgramBatched { .. } => {
                    let cfg = super::ngram_bonus::NgramBonusConfig::default();
                    ops::dflash2_selector_walk_ngram_batched(
                        gpu,
                        self.kernels.dflash2_selector_walk_ngram_batched,
                        &ops::Dflash2NgramWalkArgs {
                            unary: self.scratch.topk_logits_dev,
                            candidates: self.scratch.topk_tokens_dev,
                            hidden: self.scratch.selector_hidden,
                            pred: sel.predecessor_codebook.weight,
                            succ: sel.successor_codebook.weight,
                            out_tokens: draft_tokens_out,
                            out_probs: Some(batch_out_probs_out),
                            anchor_ids: meta_ptrs.batch_anchor_ids,
                            prev3_tokens: Some(self.scratch.ngram_prev3_dev),
                            ngram_tables: Some(self.scratch.ngram_tables_dev),
                            temperatures: None,
                            margin_thresh: cfg.margin_thresh,
                            max_bonus: cfg.effective_max_bonus(),
                            top_k_shortlist: shortlist_k,
                            gamma: gamma_eff,
                            rank: sel.rank,
                            batch_size: b,
                        },
                        stream,
                    )?;
                }
                BatchedSelectorMode::GumbelK32 { .. } => {
                    ops::dflash2_selector_walk_gumbel_batched_k32(
                        gpu,
                        self.kernels.dflash2_selector_walk_gumbel_batched_k32,
                        self.scratch.topk_logits_dev,
                        self.scratch.topk_tokens_dev,
                        self.scratch.selector_hidden,
                        sel.predecessor_codebook.weight,
                        sel.successor_codebook.weight,
                        draft_tokens_out,
                        batch_out_probs_out,
                        meta_ptrs.batch_anchor_ids,
                        meta_ptrs.batch_gumbel_seeds,
                        meta_ptrs.batch_gumbel_step0_pos,
                        meta_ptrs.batch_gumbel_temps,
                        meta_ptrs.batch_gumbel_top_ks,
                        meta_ptrs.batch_gumbel_top_ps,
                        meta_ptrs.batch_gumbel_min_ps,
                        shortlist_k,
                        gamma_eff,
                        sel.rank,
                        b,
                        stream,
                    )?;
                }
                BatchedSelectorMode::GumbelK16 { .. } => {
                    ops::dflash2_selector_walk_gumbel_batched(
                        gpu,
                        self.kernels.dflash2_selector_walk_gumbel_batched,
                        self.scratch.topk_logits_dev,
                        self.scratch.topk_tokens_dev,
                        self.scratch.selector_hidden,
                        sel.predecessor_codebook.weight,
                        sel.successor_codebook.weight,
                        draft_tokens_out,
                        meta_ptrs.batch_anchor_ids,
                        meta_ptrs.batch_gumbel_seeds,
                        meta_ptrs.batch_gumbel_step0_pos,
                        meta_ptrs.batch_gumbel_temps,
                        meta_ptrs.batch_gumbel_top_ks,
                        meta_ptrs.batch_gumbel_top_ps,
                        meta_ptrs.batch_gumbel_min_ps,
                        gamma_eff,
                        sel.rank,
                        b,
                        stream,
                    )?;
                }
                BatchedSelectorMode::GreedyBatched { .. }
                | BatchedSelectorMode::ArgmaxRows
                | BatchedSelectorMode::ArgmaxLoop => {
                    ops::dflash2_selector_walk_batched(
                        gpu,
                        self.kernels.dflash2_selector_walk_batched,
                        self.scratch.topk_logits_dev,
                        self.scratch.topk_tokens_dev,
                        self.scratch.selector_hidden,
                        sel.predecessor_codebook.weight,
                        sel.successor_codebook.weight,
                        draft_tokens_out,
                        meta_ptrs.batch_anchor_ids,
                        gamma_eff as i32,
                        sel.rank as i32,
                        b as i32,
                        stream,
                    )?;
                }
            }
        } else if matches!(selector_mode, BatchedSelectorMode::ArgmaxRows) {
            ops::argmax_bf16_rows(
                gpu,
                self.kernels.argmax_rows,
                logits_out,
                draft_tokens_out,
                lm_stride as u32,
                lm_stride as u32,
                m_out as u32,
                stream,
            )?;
        } else {
            let logit_row_bytes = lm_stride * bf16;
            for i in 0..m_out {
                ops::argmax_bf16(
                    gpu,
                    self.kernels.argmax,
                    logits_out.offset(i * logit_row_bytes),
                    draft_tokens_out.offset(i * size_of::<u32>()),
                    lm_stride as u32,
                    stream,
                )?;
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn run_stage1_batched_ctx_append(
        &self,
        chunks: &[Vec<Stage1Slice>],
        target_hidden_dim: usize,
        h: u32,
        kv_dim: u32,
        prefill_pipe_requested: bool,
        dstates: &[&mut DflashProposerState],
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        let bf16 = 2usize;
        let ctx_slot_bytes = target_hidden_dim * bf16;
        let kv_slot_bytes = kv_dim as usize * bf16;

        let (gather_kernel, rope_scatter_kernel) =
            self.batched_noise_graph.lock().resolve_stage1_kernels(gpu);
        let use_fused_gather = gather_kernel.0 != 0;
        let use_fused_rope_scatter = rope_scatter_kernel.0 != 0
            && !self.yarn_inv_freq.is_null()
            && self.rotary_dim > 0
            && self.rotary_dim <= self.head_dim
            && self.rotary_dim.is_multiple_of(2)
            && self.rope_attention_factor.is_finite();
        let mut uploaded_chunks = 0usize;

        for slices in chunks {
            let chunk_rows: usize = slices.iter().map(|sl| sl.count).sum();
            if chunk_rows == 0 {
                continue;
            }

            let stage1_meta = if use_fused_gather || use_fused_rope_scatter {
                // Prevent overwriting a rotating pinned host slot while an
                // earlier chunk's async H2D from the same call is still in
                // flight (steady-state decode is 1 chunk, so zero syncs).
                if uploaded_chunks >= STAGE1_META_PINNED_SLOTS
                    && uploaded_chunks.is_multiple_of(STAGE1_META_PINNED_SLOTS)
                {
                    gpu.synchronize(stream)?;
                }
                let meta = self.batched_noise_graph.lock().upload_stage1_meta(
                    gpu,
                    slices,
                    chunk_rows,
                    self.num_layers,
                    self.ctx_window,
                    |seq_idx| dstates[seq_idx].ctx_hidden_acc.0,
                    |seq_idx| dstates[seq_idx].ctx_ring_state(),
                    |seq_idx, layer_idx| dstates[seq_idx].ctx_k_cache[layer_idx].0,
                    |seq_idx, layer_idx| dstates[seq_idx].ctx_v_cache[layer_idx].0,
                    stream,
                )?;
                uploaded_chunks += 1;
                static STAGE1_FUSED_RECEIPT: std::sync::Once = std::sync::Once::new();
                STAGE1_FUSED_RECEIPT.call_once(|| {
                    tracing::info!(
                        fused_gather = use_fused_gather,
                        fused_rope_scatter = use_fused_rope_scatter,
                        chunk_rows,
                        num_layers = self.num_layers,
                        "DFLASH_STAGE1_FUSED_IO_ENGAGED"
                    );
                });
                Some(meta)
            } else {
                None
            };

            // 1. Gather each sequence's new `ctx_hidden_acc` rows into
            //    `scratch.mlp_intermediate` via 1 coalesced kernel launch
            //    (falling back to per-sequence ring D2Ds only if PTX is absent).
            if let Some(meta) = stage1_meta.filter(|_| use_fused_gather) {
                ops::dflash_stage1_gather_ctx_bf16(
                    gpu,
                    gather_kernel,
                    self.scratch.mlp_intermediate,
                    meta.gather_src_ptrs,
                    meta.gather_src_row_idx,
                    chunk_rows,
                    target_hidden_dim,
                    stream,
                )?;
            } else {
                for sl in slices {
                    let dstate: &DflashProposerState = dstates[sl.seq_idx];
                    let ring = dstate.ctx_ring_state()?;
                    let gather = ring.plan_gather(sl.abs_start, sl.count)?;
                    for span in gather.spans() {
                        let src = dstate.ctx_hidden_acc.offset(span.src_slot * ctx_slot_bytes);
                        let dst = self
                            .scratch
                            .mlp_intermediate
                            .offset((sl.chunk_offset + span.dst_slot) * ctx_slot_bytes);
                        gpu.copy_d2d_async(src, dst, span.slot_count * ctx_slot_bytes, stream)?;
                    }
                }
            }

            // 2. 1 batched `fc` GEMM (`mlp_intermediate -> fc_proj`) + 1 batched
            //    `hidden_norm` in-place.
            match (self.quant, self.fc_nvfp4.as_ref(), self.fc_fp8.as_ref()) {
                (DflashQuantization::Nvfp4, Some(fc_q), _) => {
                    if self.try_batched_drafter_nvfp4_gemm(
                        gpu,
                        self.scratch.mlp_intermediate,
                        fc_q,
                        self.scratch.fc_proj,
                        chunk_rows as u32,
                        h,
                        target_hidden_dim as u32,
                        stream,
                    )? {
                        // Dispatched via cuBLASLt NVFP4 on high-SM Blackwell GPUs.
                    } else if super::proposal_policy::use_prefill_pipe(
                        prefill_pipe_requested,
                        self.kernels.w4a16_gemm_pipe.0 != 0,
                        chunk_rows,
                        target_hidden_dim,
                    )
                    .map_err(anyhow::Error::msg)?
                    {
                        ops::w4a16_gemm_pipe(
                            gpu,
                            self.kernels.w4a16_gemm_pipe,
                            self.scratch.mlp_intermediate,
                            fc_q,
                            self.scratch.fc_proj,
                            chunk_rows as u32,
                            h,
                            target_hidden_dim as u32,
                            stream,
                        )?;
                    } else {
                        self.nvfp4_project_coalesced_or_gemm(
                            gpu,
                            self.scratch.mlp_intermediate,
                            fc_q,
                            self.scratch.fc_proj,
                            chunk_rows as u32,
                            h,
                            target_hidden_dim as u32,
                            stream,
                        )?;
                    }
                }
                (DflashQuantization::Fp8, _, Some(fc_q)) => {
                    self.stage_fp8_project(
                        gpu,
                        self.scratch.mlp_intermediate,
                        fc_q,
                        self.scratch.fc_proj,
                        chunk_rows as u32,
                        h,
                        target_hidden_dim as u32,
                        stream,
                    )?;
                }
                _ => {
                    self.project_bf16_dense(
                        gpu,
                        self.scratch.mlp_intermediate,
                        &self.fc,
                        self.scratch.fc_proj,
                        chunk_rows as u32,
                        h,
                        target_hidden_dim as u32,
                        stream,
                    )?;
                }
            }

            ops::rms_norm(
                gpu,
                self.kernels.rms_norm,
                self.scratch.fc_proj,
                &self.hidden_norm,
                self.scratch.fc_proj,
                chunk_rows as u32,
                h,
                self.rms_norm_eps,
                stream,
            )?;

            // Stage 1 leaves each sequence's `ctx_fc_cache` and
            // `[cache_fc_start..cache_fc_end)` untouched: batched steps only
            // consume PostRope `ctx_{k,v}_cache` (populated to `plan.needed_end`
            // below), and leaving `cache_fc_{start,end}` unchanged keeps every
            // previously cached `ctx_fc_cache` slot exact if a sequence later
            // drains to C=1 `forward_block`.

            // 3. Upload packed absolute positions `[chunk_rows]` u32 only on
            //    the legacy fallback path (fused RoPE+scatter reads `row_pos`
            //    from the single packed Stage-1 metadata H2D above).
            if !use_fused_rope_scatter {
                let mut pos_bytes = Vec::with_capacity(chunk_rows * size_of::<u32>());
                for sl in slices {
                    for i in 0..sl.count {
                        pos_bytes.extend_from_slice(&((sl.abs_start + i) as u32).to_le_bytes());
                    }
                }
                gpu.copy_h2d_group_on_stream(
                    &[HostToDeviceCopy::new(&pos_bytes, self.scratch.position_ids)],
                    stream,
                )?;
            }

            // 4. Per layer: 1 batched K/V GEMM pair + 1 batched k_norm +
            //    1 fused YaRN-scaled RoPE & circular KV-ring scatter launch
            //    (or legacy RoPE + per-sequence D2Ds when PTX is absent).
            for (layer_idx, layer) in self.layers.iter().enumerate() {
                let k_norm = match layer {
                    DflashLayerQuantWeights::Bf16(l) => {
                        self.project_bf16_dense(
                            gpu,
                            self.scratch.fc_proj,
                            &l.k_proj,
                            self.scratch.k_buf,
                            chunk_rows as u32,
                            kv_dim,
                            h,
                            stream,
                        )?;
                        self.project_bf16_dense(
                            gpu,
                            self.scratch.fc_proj,
                            &l.v_proj,
                            self.scratch.v_buf,
                            chunk_rows as u32,
                            kv_dim,
                            h,
                            stream,
                        )?;
                        &l.k_norm
                    }
                    DflashLayerQuantWeights::Fp8(l) => {
                        self.stage_fp8_project(
                            gpu,
                            self.scratch.fc_proj,
                            &l.k_proj,
                            self.scratch.k_buf,
                            chunk_rows as u32,
                            kv_dim,
                            h,
                            stream,
                        )?;
                        self.stage_fp8_project(
                            gpu,
                            self.scratch.fc_proj,
                            &l.v_proj,
                            self.scratch.v_buf,
                            chunk_rows as u32,
                            kv_dim,
                            h,
                            stream,
                        )?;
                        &l.k_norm
                    }
                    DflashLayerQuantWeights::Nvfp4(l) => {
                        if self.try_batched_drafter_nvfp4_pair(
                            gpu,
                            self.scratch.fc_proj,
                            &l.k_proj,
                            self.scratch.k_buf,
                            &l.v_proj,
                            self.scratch.v_buf,
                            chunk_rows as u32,
                            kv_dim,
                            h,
                            stream,
                        )? {
                            // Dispatched both K and V via shared-activation cuBLASLt NVFP4.
                        } else if super::proposal_policy::use_prefill_dual_pipe(
                            prefill_pipe_requested,
                            self.kernels.w4a16_gemm_pipe_dual.0 != 0,
                            chunk_rows,
                            h as usize,
                        )
                        .map_err(anyhow::Error::msg)?
                        {
                            ops::w4a16_gemm_pipe_dual(
                                gpu,
                                self.kernels.w4a16_gemm_pipe_dual,
                                self.scratch.fc_proj,
                                &l.k_proj,
                                &l.v_proj,
                                self.scratch.k_buf,
                                self.scratch.v_buf,
                                false,
                                chunk_rows as u32,
                                kv_dim,
                                h,
                                stream,
                            )?;
                        } else if let (Some(k_t), Some(v_t)) =
                            (l.k_proj_t.as_ref(), l.v_proj_t.as_ref())
                            && l.q_proj_t.is_some()
                            && l.o_proj_t.is_some()
                            && self.kernels.w4a16_gemm_t_m16.0 != 0
                        {
                            let allow_m32 = chunk_rows <= 32
                                || (chunk_rows <= 64 && atlas_core::device::active_sm_count() > 64);
                            self.draft_gemm_t(
                                gpu,
                                self.scratch.fc_proj,
                                k_t,
                                self.scratch.k_buf,
                                chunk_rows as u32,
                                kv_dim,
                                h,
                                allow_m32,
                                stream,
                            )?;
                            self.draft_gemm_t(
                                gpu,
                                self.scratch.fc_proj,
                                v_t,
                                self.scratch.v_buf,
                                chunk_rows as u32,
                                kv_dim,
                                h,
                                allow_m32,
                                stream,
                            )?;
                        } else {
                            self.nvfp4_project_coalesced_or_gemm(
                                gpu,
                                self.scratch.fc_proj,
                                &l.k_proj,
                                self.scratch.k_buf,
                                chunk_rows as u32,
                                kv_dim,
                                h,
                                stream,
                            )?;
                            self.nvfp4_project_coalesced_or_gemm(
                                gpu,
                                self.scratch.fc_proj,
                                &l.v_proj,
                                self.scratch.v_buf,
                                chunk_rows as u32,
                                kv_dim,
                                h,
                                stream,
                            )?;
                        }
                        &l.k_norm
                    }
                };

                ops::rms_norm(
                    gpu,
                    self.kernels.rms_norm,
                    self.scratch.k_buf,
                    k_norm,
                    self.scratch.k_buf,
                    chunk_rows as u32 * self.num_kv_heads as u32,
                    self.head_dim as u32,
                    self.rms_norm_eps,
                    stream,
                )?;
                if let Some(meta) = stage1_meta.filter(|_| use_fused_rope_scatter) {
                    ops::dflash_rope_yarn_scaled_scatter_ring_bf16(
                        gpu,
                        rope_scatter_kernel,
                        self.scratch.k_buf,
                        self.scratch.v_buf,
                        meta.layer_dst_k_ptrs(layer_idx),
                        meta.layer_dst_v_ptrs(layer_idx),
                        meta.scatter_dst_slot,
                        meta.scatter_row_pos,
                        self.yarn_inv_freq,
                        self.rope_attention_factor,
                        chunk_rows,
                        self.num_kv_heads,
                        self.head_dim,
                        self.rotary_dim,
                        stream,
                    )?;
                } else {
                    ops::rope_yarn_scaled(
                        gpu,
                        self.kernels.rope_qwen3,
                        DevicePtr::NULL,
                        self.scratch.k_buf,
                        self.scratch.position_ids,
                        chunk_rows as u32,
                        0,
                        self.num_kv_heads as u32,
                        self.head_dim as u32,
                        self.rotary_dim as u32,
                        self.yarn_inv_freq,
                        self.rope_theta,
                        self.rope_attention_factor,
                        stream,
                    )?;

                    for sl in slices {
                        let write_end = sl.abs_start + sl.count;
                        let plan = ring_window::plan_write_chunk(
                            sl.abs_start,
                            sl.count,
                            write_end.max(self.ctx_window),
                            self.ctx_window,
                        )?;
                        let src_k = self.scratch.k_buf.offset(sl.chunk_offset * kv_slot_bytes);
                        let src_v = self.scratch.v_buf.offset(sl.chunk_offset * kv_slot_bytes);
                        let dst_k_base = dstates[sl.seq_idx].ctx_k_cache[layer_idx];
                        let dst_v_base = dstates[sl.seq_idx].ctx_v_cache[layer_idx];
                        for span in plan.spans() {
                            let linear_off = span.linear_slot * kv_slot_bytes;
                            let phys_off = span.physical_slot * kv_slot_bytes;
                            let bytes = span.slot_count * kv_slot_bytes;
                            gpu.copy_d2d_async(
                                src_k.offset(linear_off),
                                dst_k_base.offset(phys_off),
                                bytes,
                                stream,
                            )?;
                            gpu.copy_d2d_async(
                                src_v.offset(linear_off),
                                dst_v_base.offset(phys_off),
                                bytes,
                                stream,
                            )?;
                        }
                    }
                }
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn batched_conv_prepare(
        &self,
        conv: &crate::weight_loader::DflashConvWeights,
        m_total: u32,
        block_len: u32,
        h: u32,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<DevicePtr> {
        let dyn_width = (2 * conv.kernel_size * conv.groups) as u32;
        self.project_bf16_dense(
            gpu,
            self.scratch.norm_buf,
            &conv.kernel_projection,
            self.scratch.conv_dyn,
            m_total,
            dyn_width,
            h,
            stream,
        )?;
        ops::dflash2_conv_prepare_batched(
            gpu,
            self.kernels.dflash2_conv_prepare_batched,
            self.scratch.norm_buf,
            self.scratch.conv_dyn,
            conv.base_kernel.weight,
            self.scratch.conv_out,
            self.scratch.conv_dyn1,
            m_total,
            conv.groups as u32,
            block_len,
            stream,
        )?;
        Ok(self.scratch.conv_out)
    }

    #[allow(clippy::too_many_arguments)]
    fn batched_conv_finish(
        &self,
        conv: &crate::weight_loader::DflashConvWeights,
        m_total: u32,
        block_len: u32,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<DevicePtr> {
        ops::dflash2_conv_finish_batched(
            gpu,
            self.kernels.dflash2_conv_finish_batched,
            self.scratch.stream_acc,
            self.scratch.conv_dyn1,
            conv.base_kernel.weight,
            self.scratch.conv_out,
            m_total,
            conv.groups as u32,
            block_len,
            stream,
        )?;
        Ok(self.scratch.conv_out)
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_split_ctx_attention_h128_batched(
        &self,
        meta_ptrs: BatchedStage2MetaPtrs,
        k_ctx_ptrs: DevicePtr,
        v_ctx_ptrs: DevicePtr,
        ring_cap: u32,
        batch_size: u32,
        m_rows: u32,
        inv_sqrt_d: f32,
        causal: bool,
        sliding_window: u32,
        split_kv: Option<BatchedSplitKvH128Plan>,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        if let Some(plan) = split_kv {
            KernelLaunch::new(gpu, plan.split_kernel)
                .grid([
                    self.num_q_heads as u32,
                    m_rows.div_ceil(32),
                    batch_size * plan.num_splits,
                ])
                .block([128, 1, 1])
                .arg_ptr(self.scratch.q_buf)
                .arg_ptr(k_ctx_ptrs)
                .arg_ptr(v_ctx_ptrs)
                .arg_ptr(self.scratch.k_buf)
                .arg_ptr(self.scratch.v_buf)
                .arg_ptr(plan.workspace)
                .arg_ptr(meta_ptrs.batch_eff_ctx)
                .arg_ptr(meta_ptrs.batch_ring_start)
                .arg_u32(ring_cap)
                .arg_u32(m_rows)
                .arg_u32(self.num_q_heads as u32)
                .arg_u32(self.num_kv_heads as u32)
                .arg_u32(batch_size)
                .arg_u32(plan.num_splits)
                .arg_f32(inv_sqrt_d)
                .arg_u32(u32::from(causal))
                .arg_u32(sliding_window)
                .launch(stream)?;
            KernelLaunch::new(gpu, plan.reduce_kernel)
                .grid([self.num_q_heads as u32, m_rows, batch_size])
                .block([64, 1, 1])
                .arg_ptr(plan.workspace)
                .arg_ptr(self.scratch.attn_out)
                .arg_u32(m_rows)
                .arg_u32(self.num_q_heads as u32)
                .arg_u32(batch_size)
                .arg_u32(plan.num_splits)
                .launch(stream)
        } else {
            KernelLaunch::new(gpu, self.kernels.split_ctx_h128_batched)
                .grid([self.num_q_heads as u32, m_rows.div_ceil(32), batch_size])
                .block([128, 1, 1])
                .arg_ptr(self.scratch.q_buf)
                .arg_ptr(k_ctx_ptrs)
                .arg_ptr(v_ctx_ptrs)
                .arg_ptr(self.scratch.k_buf)
                .arg_ptr(self.scratch.v_buf)
                .arg_ptr(self.scratch.attn_out)
                .arg_ptr(meta_ptrs.batch_eff_ctx)
                .arg_ptr(meta_ptrs.batch_ring_start)
                .arg_u32(ring_cap)
                .arg_u32(m_rows)
                .arg_u32(self.num_q_heads as u32)
                .arg_u32(self.num_kv_heads as u32)
                .arg_f32(inv_sqrt_d)
                .arg_u32(u32::from(causal))
                .arg_u32(sliding_window)
                .launch(stream)
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn batched_qk_norm_rope_and_attn(
        &self,
        q_norm: &crate::weight_map::DenseWeight,
        k_norm: &crate::weight_map::DenseWeight,
        meta_ptrs: BatchedStage2MetaPtrs,
        k_ctx_ptrs: DevicePtr,
        v_ctx_ptrs: DevicePtr,
        ring_cap: u32,
        batch_size: u32,
        m_rows: u32,
        m_total: u32,
        inv_sqrt_d: f32,
        causal: bool,
        sliding_window: u32,
        split_kv: Option<BatchedSplitKvH128Plan>,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        ops::rms_norm(
            gpu,
            self.kernels.rms_norm,
            self.scratch.q_buf,
            q_norm,
            self.scratch.q_buf,
            m_total * self.num_q_heads as u32,
            self.head_dim as u32,
            self.rms_norm_eps,
            stream,
        )?;
        ops::rms_norm(
            gpu,
            self.kernels.rms_norm,
            self.scratch.k_buf,
            k_norm,
            self.scratch.k_buf,
            m_total * self.num_kv_heads as u32,
            self.head_dim as u32,
            self.rms_norm_eps,
            stream,
        )?;
        ops::rope_yarn_scaled(
            gpu,
            self.kernels.rope_qwen3,
            self.scratch.q_buf,
            self.scratch.k_buf,
            meta_ptrs.position_ids,
            m_total,
            self.num_q_heads as u32,
            self.num_kv_heads as u32,
            self.head_dim as u32,
            self.rotary_dim as u32,
            self.yarn_inv_freq,
            self.rope_theta,
            self.rope_attention_factor,
            stream,
        )?;
        self.launch_split_ctx_attention_h128_batched(
            meta_ptrs,
            k_ctx_ptrs,
            v_ctx_ptrs,
            ring_cap,
            batch_size,
            m_rows,
            inv_sqrt_d,
            causal,
            sliding_window,
            split_kv,
            gpu,
            stream,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn batched_layer_bf16(
        &self,
        layer: &DflashLayer,
        meta_ptrs: BatchedStage2MetaPtrs,
        k_ctx_ptrs: DevicePtr,
        v_ctx_ptrs: DevicePtr,
        ring_cap: u32,
        batch_size: u32,
        m_rows: u32,
        m_total: u32,
        h: u32,
        q_dim: u32,
        kv_dim: u32,
        inter: u32,
        inv_sqrt_d: f32,
        causal: bool,
        sliding_window: u32,
        split_kv: Option<BatchedSplitKvH128Plan>,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        ops::rms_norm(
            gpu,
            self.kernels.rms_norm,
            self.scratch.stream_buf,
            &layer.input_layernorm,
            self.scratch.norm_buf,
            m_total,
            h,
            self.rms_norm_eps,
            stream,
        )?;
        let attn_in = if let Some(conv) = &layer.attention_conv {
            self.batched_conv_prepare(conv, m_total, m_rows, h, gpu, stream)?
        } else {
            self.scratch.norm_buf
        };
        self.project_bf16_dense(
            gpu,
            attn_in,
            &layer.q_proj,
            self.scratch.q_buf,
            m_total,
            q_dim,
            h,
            stream,
        )?;
        self.project_bf16_dense(
            gpu,
            attn_in,
            &layer.k_proj,
            self.scratch.k_buf,
            m_total,
            kv_dim,
            h,
            stream,
        )?;
        self.project_bf16_dense(
            gpu,
            attn_in,
            &layer.v_proj,
            self.scratch.v_buf,
            m_total,
            kv_dim,
            h,
            stream,
        )?;
        self.batched_qk_norm_rope_and_attn(
            &layer.q_norm,
            &layer.k_norm,
            meta_ptrs,
            k_ctx_ptrs,
            v_ctx_ptrs,
            ring_cap,
            batch_size,
            m_rows,
            m_total,
            inv_sqrt_d,
            causal,
            sliding_window,
            split_kv,
            gpu,
            stream,
        )?;
        self.project_bf16_dense(
            gpu,
            self.scratch.attn_out,
            &layer.o_proj,
            self.scratch.stream_acc,
            m_total,
            h,
            q_dim,
            stream,
        )?;
        let attn_residual = if let Some(conv) = &layer.attention_conv {
            self.batched_conv_finish(conv, m_total, m_rows, gpu, stream)?
        } else {
            self.scratch.stream_acc
        };
        ops::residual_add(
            gpu,
            self.kernels.residual_add,
            self.scratch.stream_buf,
            attn_residual,
            m_total * h,
            stream,
        )?;
        ops::rms_norm(
            gpu,
            self.kernels.rms_norm,
            self.scratch.stream_buf,
            &layer.post_attention_layernorm,
            self.scratch.norm_buf,
            m_total,
            h,
            self.rms_norm_eps,
            stream,
        )?;
        let mlp_in = if let Some(conv) = &layer.mlp_conv {
            self.batched_conv_prepare(conv, m_total, m_rows, h, gpu, stream)?
        } else {
            self.scratch.norm_buf
        };
        self.project_bf16_dense(
            gpu,
            mlp_in,
            &layer.gate_proj,
            self.scratch.mlp_intermediate,
            m_total,
            inter,
            h,
            stream,
        )?;
        self.project_bf16_dense(
            gpu,
            mlp_in,
            &layer.up_proj,
            self.scratch.mlp_up,
            m_total,
            inter,
            h,
            stream,
        )?;
        ops::silu_mul(
            gpu,
            self.kernels.silu_mul,
            self.scratch.mlp_intermediate,
            self.scratch.mlp_up,
            self.scratch.mlp_intermediate,
            m_total * inter,
            stream,
        )?;
        self.project_bf16_dense(
            gpu,
            self.scratch.mlp_intermediate,
            &layer.down_proj,
            self.scratch.stream_acc,
            m_total,
            h,
            inter,
            stream,
        )?;
        let mlp_residual = if let Some(conv) = &layer.mlp_conv {
            self.batched_conv_finish(conv, m_total, m_rows, gpu, stream)?
        } else {
            self.scratch.stream_acc
        };
        ops::residual_add(
            gpu,
            self.kernels.residual_add,
            self.scratch.stream_buf,
            mlp_residual,
            m_total * h,
            stream,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn batched_layer_fp8(
        &self,
        layer: &DflashLayerFp8Weights,
        meta_ptrs: BatchedStage2MetaPtrs,
        k_ctx_ptrs: DevicePtr,
        v_ctx_ptrs: DevicePtr,
        ring_cap: u32,
        batch_size: u32,
        m_rows: u32,
        m_total: u32,
        h: u32,
        q_dim: u32,
        kv_dim: u32,
        inter: u32,
        inv_sqrt_d: f32,
        causal: bool,
        sliding_window: u32,
        split_kv: Option<BatchedSplitKvH128Plan>,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        ops::rms_norm(
            gpu,
            self.kernels.rms_norm,
            self.scratch.stream_buf,
            &layer.input_layernorm,
            self.scratch.norm_buf,
            m_total,
            h,
            self.rms_norm_eps,
            stream,
        )?;
        let attn_in = if let Some(conv) = &layer.attention_conv {
            self.batched_conv_prepare(conv, m_total, m_rows, h, gpu, stream)?
        } else {
            self.scratch.norm_buf
        };
        self.stage_fp8_project(
            gpu,
            attn_in,
            &layer.q_proj,
            self.scratch.q_buf,
            m_total,
            q_dim,
            h,
            stream,
        )?;
        self.stage_fp8_project(
            gpu,
            attn_in,
            &layer.k_proj,
            self.scratch.k_buf,
            m_total,
            kv_dim,
            h,
            stream,
        )?;
        self.stage_fp8_project(
            gpu,
            attn_in,
            &layer.v_proj,
            self.scratch.v_buf,
            m_total,
            kv_dim,
            h,
            stream,
        )?;
        self.batched_qk_norm_rope_and_attn(
            &layer.q_norm,
            &layer.k_norm,
            meta_ptrs,
            k_ctx_ptrs,
            v_ctx_ptrs,
            ring_cap,
            batch_size,
            m_rows,
            m_total,
            inv_sqrt_d,
            causal,
            sliding_window,
            split_kv,
            gpu,
            stream,
        )?;
        self.stage_fp8_project(
            gpu,
            self.scratch.attn_out,
            &layer.o_proj,
            self.scratch.stream_acc,
            m_total,
            h,
            q_dim,
            stream,
        )?;
        let attn_residual = if let Some(conv) = &layer.attention_conv {
            self.batched_conv_finish(conv, m_total, m_rows, gpu, stream)?
        } else {
            self.scratch.stream_acc
        };
        ops::residual_add(
            gpu,
            self.kernels.residual_add,
            self.scratch.stream_buf,
            attn_residual,
            m_total * h,
            stream,
        )?;
        ops::rms_norm(
            gpu,
            self.kernels.rms_norm,
            self.scratch.stream_buf,
            &layer.post_attention_layernorm,
            self.scratch.norm_buf,
            m_total,
            h,
            self.rms_norm_eps,
            stream,
        )?;
        let mlp_in = if let Some(conv) = &layer.mlp_conv {
            self.batched_conv_prepare(conv, m_total, m_rows, h, gpu, stream)?
        } else {
            self.scratch.norm_buf
        };
        self.stage_fp8_project(
            gpu,
            mlp_in,
            &layer.gate_proj,
            self.scratch.mlp_intermediate,
            m_total,
            inter,
            h,
            stream,
        )?;
        self.stage_fp8_project(
            gpu,
            mlp_in,
            &layer.up_proj,
            self.scratch.mlp_up,
            m_total,
            inter,
            h,
            stream,
        )?;
        ops::silu_mul(
            gpu,
            self.kernels.silu_mul,
            self.scratch.mlp_intermediate,
            self.scratch.mlp_up,
            self.scratch.mlp_intermediate,
            m_total * inter,
            stream,
        )?;
        self.stage_fp8_project(
            gpu,
            self.scratch.mlp_intermediate,
            &layer.down_proj,
            self.scratch.stream_acc,
            m_total,
            h,
            inter,
            stream,
        )?;
        let mlp_residual = if let Some(conv) = &layer.mlp_conv {
            self.batched_conv_finish(conv, m_total, m_rows, gpu, stream)?
        } else {
            self.scratch.stream_acc
        };
        ops::residual_add(
            gpu,
            self.kernels.residual_add,
            self.scratch.stream_buf,
            mlp_residual,
            m_total * h,
            stream,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn batched_layer_nvfp4(
        &self,
        layer: &DflashLayerNvfp4,
        meta_ptrs: BatchedStage2MetaPtrs,
        k_ctx_ptrs: DevicePtr,
        v_ctx_ptrs: DevicePtr,
        ring_cap: u32,
        batch_size: u32,
        m_rows: u32,
        m_total: u32,
        h: u32,
        q_dim: u32,
        kv_dim: u32,
        inter: u32,
        inv_sqrt_d: f32,
        causal: bool,
        sliding_window: u32,
        split_kv: Option<BatchedSplitKvH128Plan>,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        let attn_t = if self.kernels.w4a16_gemm_t_m16.0 != 0 {
            match (
                layer.q_proj_t.as_ref(),
                layer.k_proj_t.as_ref(),
                layer.v_proj_t.as_ref(),
                layer.o_proj_t.as_ref(),
            ) {
                (Some(q), Some(k), Some(v), Some(o)) => Some((q, k, v, o)),
                _ => None,
            }
        } else {
            None
        };

        ops::rms_norm(
            gpu,
            self.kernels.rms_norm,
            self.scratch.stream_buf,
            &layer.input_layernorm,
            self.scratch.norm_buf,
            m_total,
            h,
            self.rms_norm_eps,
            stream,
        )?;
        let attn_in = if let Some(conv) = &layer.attention_conv {
            self.batched_conv_prepare(conv, m_total, m_rows, h, gpu, stream)?
        } else {
            self.scratch.norm_buf
        };

        let q_nvfp4_ok = self.try_batched_drafter_nvfp4_gemm(
            gpu,
            attn_in,
            &layer.q_proj,
            self.scratch.q_buf,
            m_total,
            q_dim,
            h,
            stream,
        )?;
        if !q_nvfp4_ok {
            if let Some((q_t, _, _, _)) = attn_t {
                self.draft_gemm_t(
                    gpu,
                    attn_in,
                    q_t,
                    self.scratch.q_buf,
                    m_total,
                    q_dim,
                    h,
                    true,
                    stream,
                )?;
            } else {
                self.nvfp4_project_coalesced_or_gemm(
                    gpu,
                    attn_in,
                    &layer.q_proj,
                    self.scratch.q_buf,
                    m_total,
                    q_dim,
                    h,
                    stream,
                )?;
            }
        }
        let kv_nvfp4_ok = self.try_batched_drafter_nvfp4_pair(
            gpu,
            attn_in,
            &layer.k_proj,
            self.scratch.k_buf,
            &layer.v_proj,
            self.scratch.v_buf,
            m_total,
            kv_dim,
            h,
            stream,
        )?;
        if !kv_nvfp4_ok {
            if let Some((_, k_t, v_t, _)) = attn_t {
                self.draft_gemm_t(
                    gpu,
                    attn_in,
                    k_t,
                    self.scratch.k_buf,
                    m_total,
                    kv_dim,
                    h,
                    true,
                    stream,
                )?;
                self.draft_gemm_t(
                    gpu,
                    attn_in,
                    v_t,
                    self.scratch.v_buf,
                    m_total,
                    kv_dim,
                    h,
                    true,
                    stream,
                )?;
            } else {
                self.nvfp4_project_coalesced_or_gemm(
                    gpu,
                    attn_in,
                    &layer.k_proj,
                    self.scratch.k_buf,
                    m_total,
                    kv_dim,
                    h,
                    stream,
                )?;
                self.nvfp4_project_coalesced_or_gemm(
                    gpu,
                    attn_in,
                    &layer.v_proj,
                    self.scratch.v_buf,
                    m_total,
                    kv_dim,
                    h,
                    stream,
                )?;
            }
        }

        self.batched_qk_norm_rope_and_attn(
            &layer.q_norm,
            &layer.k_norm,
            meta_ptrs,
            k_ctx_ptrs,
            v_ctx_ptrs,
            ring_cap,
            batch_size,
            m_rows,
            m_total,
            inv_sqrt_d,
            causal,
            sliding_window,
            split_kv,
            gpu,
            stream,
        )?;

        let disable_o = noise_graph_attn_kgamma_disable_o();
        if self.try_batched_drafter_nvfp4_gemm(
            gpu,
            self.scratch.attn_out,
            &layer.o_proj,
            self.scratch.stream_acc,
            m_total,
            h,
            q_dim,
            stream,
        )? {
            // Dispatched O projection via cuBLASLt NVFP4 on high-SM Blackwell GPUs.
        } else if let Some((_, _, _, o_t)) = attn_t
            && !disable_o
        {
            self.draft_gemm_t(
                gpu,
                self.scratch.attn_out,
                o_t,
                self.scratch.stream_acc,
                m_total,
                h,
                q_dim,
                true,
                stream,
            )?;
        } else {
            self.nvfp4_project_coalesced_or_gemm(
                gpu,
                self.scratch.attn_out,
                &layer.o_proj,
                self.scratch.stream_acc,
                m_total,
                h,
                q_dim,
                stream,
            )?;
        }

        let attn_residual = if let Some(conv) = &layer.attention_conv {
            self.batched_conv_finish(conv, m_total, m_rows, gpu, stream)?
        } else {
            self.scratch.stream_acc
        };
        ops::residual_add(
            gpu,
            self.kernels.residual_add,
            self.scratch.stream_buf,
            attn_residual,
            m_total * h,
            stream,
        )?;
        ops::rms_norm(
            gpu,
            self.kernels.rms_norm,
            self.scratch.stream_buf,
            &layer.post_attention_layernorm,
            self.scratch.norm_buf,
            m_total,
            h,
            self.rms_norm_eps,
            stream,
        )?;
        let mlp_in = if let Some(conv) = &layer.mlp_conv {
            self.batched_conv_prepare(conv, m_total, m_rows, h, gpu, stream)?
        } else {
            self.scratch.norm_buf
        };

        let ffn_t = if self.kernels.w4a16_gemm_t_m16.0 != 0 {
            match (
                layer.gate_proj_t.as_ref(),
                layer.up_proj_t.as_ref(),
                layer.down_proj_t.as_ref(),
            ) {
                (Some(g), Some(u), Some(d)) => Some((g, u, d)),
                _ => None,
            }
        } else {
            None
        };
        let fused_gateup = ffn_t.is_some()
            && self.kernels.w4a16_gemm_t_m32_n64_gateup_silu.0 != 0
            && self.kernels.w4a16_gemm_t_m32_n64.0 != 0;

        let mut ran_fused_gateup = false;
        if self.try_batched_drafter_nvfp4_pair(
            gpu,
            mlp_in,
            &layer.gate_proj,
            self.scratch.mlp_intermediate,
            &layer.up_proj,
            self.scratch.mlp_up,
            m_total,
            inter,
            h,
            stream,
        )? {
            // Dispatched both gate and up via shared-activation cuBLASLt NVFP4.
        } else if let Some((gate_t, up_t, _)) = ffn_t
            && fused_gateup
        {
            ops::w4a16_gemm_n64_m32_gateup_silu(
                gpu,
                self.kernels.w4a16_gemm_t_m32_n64_gateup_silu,
                mlp_in,
                gate_t,
                up_t,
                self.scratch.mlp_intermediate,
                m_total,
                inter,
                h,
                stream,
            )?;
            ran_fused_gateup = true;
        } else if let Some((gate_t, up_t, _)) = ffn_t {
            self.draft_gemm_t(
                gpu,
                mlp_in,
                gate_t,
                self.scratch.mlp_intermediate,
                m_total,
                inter,
                h,
                true,
                stream,
            )?;
            self.draft_gemm_t(
                gpu,
                mlp_in,
                up_t,
                self.scratch.mlp_up,
                m_total,
                inter,
                h,
                true,
                stream,
            )?;
        } else {
            self.nvfp4_project_coalesced_or_gemm(
                gpu,
                mlp_in,
                &layer.gate_proj,
                self.scratch.mlp_intermediate,
                m_total,
                inter,
                h,
                stream,
            )?;
            self.nvfp4_project_coalesced_or_gemm(
                gpu,
                mlp_in,
                &layer.up_proj,
                self.scratch.mlp_up,
                m_total,
                inter,
                h,
                stream,
            )?;
        }

        if !ran_fused_gateup {
            ops::silu_mul(
                gpu,
                self.kernels.silu_mul,
                self.scratch.mlp_intermediate,
                self.scratch.mlp_up,
                self.scratch.mlp_intermediate,
                m_total * inter,
                stream,
            )?;
        }

        if self.try_batched_drafter_nvfp4_gemm(
            gpu,
            self.scratch.mlp_intermediate,
            &layer.down_proj,
            self.scratch.stream_acc,
            m_total,
            h,
            inter,
            stream,
        )? {
            // Dispatched down projection via cuBLASLt NVFP4 on high-SM Blackwell GPUs.
        } else if let Some((_, _, down_t)) = ffn_t {
            self.draft_gemm_t(
                gpu,
                self.scratch.mlp_intermediate,
                down_t,
                self.scratch.stream_acc,
                m_total,
                h,
                inter,
                true,
                stream,
            )?;
        } else {
            self.nvfp4_project_coalesced_or_gemm(
                gpu,
                self.scratch.mlp_intermediate,
                &layer.down_proj,
                self.scratch.stream_acc,
                m_total,
                h,
                inter,
                stream,
            )?;
        }

        let mlp_residual = if let Some(conv) = &layer.mlp_conv {
            self.batched_conv_finish(conv, m_total, m_rows, gpu, stream)?
        } else {
            self.scratch.stream_acc
        };
        ops::residual_add(
            gpu,
            self.kernels.residual_add,
            self.scratch.stream_buf,
            mlp_residual,
            m_total * h,
            stream,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::{
        CtxKvCacheRepr, SeqContextPlan, Stage1Slice, plan_seq_context_window, plan_stage1_chunks,
    };

    #[test]
    fn context_window_planner_handles_cold_start_incremental_wrap_and_repr_switch() {
        // Cold start (None repr): projects full [0..128).
        let cold = plan_seq_context_window(
            128, 128, 4096, 4096, None, &[0; 5], &[0; 5], &[0; 5], &[0; 5],
        );
        assert_eq!(
            cold,
            SeqContextPlan {
                eff_ctx: 128,
                needed_start: 0,
                needed_end: 128,
                ring_start: 0,
                new_start: 0,
                new_count: 128,
                invalidate_prev: true,
            }
        );

        // Steady unwrapped PostRope append: only projects [128..134).
        let inc = plan_seq_context_window(
            134,
            134,
            4096,
            4096,
            Some(CtxKvCacheRepr::PostRope),
            &[0; 5],
            &[128; 5],
            &[0; 5],
            &[128; 5],
        );
        assert_eq!(
            inc,
            SeqContextPlan {
                eff_ctx: 134,
                needed_start: 0,
                needed_end: 134,
                ring_start: 0,
                new_start: 128,
                new_count: 6,
                invalidate_prev: false,
            }
        );

        // Wrapped circular PostRope append: needed [904..5000), cached [900..4996) -> only 4 new!
        let wrapped = plan_seq_context_window(
            4096,
            5000,
            4096,
            4096,
            Some(CtxKvCacheRepr::PostRope),
            &[900; 5],
            &[4996; 5],
            &[900; 5],
            &[4996; 5],
        );
        assert_eq!(
            wrapped,
            SeqContextPlan {
                eff_ctx: 4096,
                needed_start: 904,
                needed_end: 5000,
                ring_start: 904,
                new_start: 4996,
                new_count: 4,
                invalidate_prev: false,
            }
        );

        // Transition from eager PreRope forces full PostRope recomputation.
        let switch = plan_seq_context_window(
            4096,
            5000,
            4096,
            4096,
            Some(CtxKvCacheRepr::PreRope),
            &[904; 5],
            &[4996; 5],
            &[904; 5],
            &[4996; 5],
        );
        assert_eq!(switch.new_start, 904);
        assert_eq!(switch.new_count, 4096);
        assert!(switch.invalidate_prev);
        assert_eq!(switch.ring_start, 904);
    }

    #[test]
    fn stage1_chunker_packs_heterogeneous_lengths_within_bound() {
        let plans = [
            SeqContextPlan {
                eff_ctx: 1500,
                needed_start: 0,
                needed_end: 1500,
                ring_start: 0,
                new_start: 0,
                new_count: 1500,
                invalidate_prev: true,
            },
            SeqContextPlan {
                eff_ctx: 100,
                needed_start: 0,
                needed_end: 100,
                ring_start: 0,
                new_start: 100,
                new_count: 0,
                invalidate_prev: false,
            },
            SeqContextPlan {
                eff_ctx: 600,
                needed_start: 0,
                needed_end: 600,
                ring_start: 0,
                new_start: 0,
                new_count: 600,
                invalidate_prev: true,
            },
        ];
        let chunks = plan_stage1_chunks(&plans, 1024);
        assert_eq!(chunks.len(), 3);
        assert_eq!(
            chunks[0],
            vec![Stage1Slice {
                seq_idx: 0,
                abs_start: 0,
                count: 1024,
                chunk_offset: 0,
            }]
        );
        assert_eq!(
            chunks[1],
            vec![
                Stage1Slice {
                    seq_idx: 0,
                    abs_start: 1024,
                    count: 476,
                    chunk_offset: 0,
                },
                Stage1Slice {
                    seq_idx: 2,
                    abs_start: 0,
                    count: 548,
                    chunk_offset: 476,
                },
            ]
        );
        assert_eq!(
            chunks[2],
            vec![Stage1Slice {
                seq_idx: 2,
                abs_start: 548,
                count: 52,
                chunk_offset: 0,
            }]
        );
    }

    #[test]
    fn batched_gumbel_meta_packs_mixed_coupled_and_greedy_cohort() {
        use super::pack_batched_gumbel_meta;
        use spark_runtime::sampler::gumbel::GumbelDraftSpec;

        let spec_a = GumbelDraftSpec {
            seed: 0x1111_2222_3333_4444,
            temperature: 0.7,
            top_k: 20,
            top_p: 0.95,
            min_p: 0.05,
            step0_abs_pos: 100,
        };
        let spec_b = GumbelDraftSpec {
            seed: 0x5555_6666_7777_8888,
            temperature: 1.0,
            top_k: 0,
            top_p: 0.9,
            min_p: 0.0,
            step0_abs_pos: 512,
        };
        let inactive = GumbelDraftSpec {
            temperature: 0.0,
            ..spec_a
        };

        let packed = pack_batched_gumbel_meta(&[Some(spec_a), None, Some(spec_b), Some(inactive)]);
        assert_eq!(packed.coupled_seqs, 2);
        assert_eq!(packed.seeds.len(), 4 * 8);
        assert_eq!(packed.step0_pos.len(), 4 * 8);
        assert_eq!(packed.temps.len(), 4 * 4);
        assert_eq!(packed.top_ks.len(), 4 * 4);
        assert_eq!(packed.top_ps.len(), 4 * 4);
        assert_eq!(packed.min_ps.len(), 4 * 4);

        let read_u64 = |buf: &[u8], idx: usize| {
            u64::from_le_bytes(buf[idx * 8..(idx + 1) * 8].try_into().unwrap())
        };
        let read_u32 = |buf: &[u8], idx: usize| {
            u32::from_le_bytes(buf[idx * 4..(idx + 1) * 4].try_into().unwrap())
        };
        let read_f32 = |buf: &[u8], idx: usize| {
            f32::from_le_bytes(buf[idx * 4..(idx + 1) * 4].try_into().unwrap())
        };

        assert_eq!(read_u64(&packed.seeds, 0), spec_a.seed);
        assert_eq!(read_u64(&packed.step0_pos, 0), 100);
        assert_eq!(read_f32(&packed.temps, 0), 0.7);
        assert_eq!(read_u32(&packed.top_ks, 0), 20);
        assert_eq!(read_f32(&packed.top_ps, 0), 0.95);
        assert_eq!(read_f32(&packed.min_ps, 0), 0.05);

        assert_eq!(read_u64(&packed.seeds, 1), 0);
        assert_eq!(read_f32(&packed.temps, 1), 0.0);
        assert_eq!(read_f32(&packed.top_ps, 1), 1.0);

        assert_eq!(read_u64(&packed.seeds, 2), spec_b.seed);
        assert_eq!(read_u64(&packed.step0_pos, 2), 512);
        assert_eq!(read_f32(&packed.temps, 2), 1.0);
        assert_eq!(read_u32(&packed.top_ks, 2), 0);
        assert_eq!(read_f32(&packed.top_ps, 2), 0.9);
        assert_eq!(read_f32(&packed.min_ps, 2), 0.0);

        assert_eq!(read_u64(&packed.seeds, 3), 0);
        assert_eq!(read_f32(&packed.temps, 3), 0.0);
    }

    #[test]
    fn batched_stage2_graph_key_isolates_all_capture_dimensions() {
        use std::collections::HashSet;

        use super::{BatchedNoiseGraphKey, BatchedSelectorMode};

        let base = BatchedNoiseGraphKey {
            batch_size: 16,
            gamma_eff: 3,
            m_rows: 4,
            shortlist_k: 32,
            selector_mode: BatchedSelectorMode::GumbelK32 {
                use_radix_topk: true,
            },
            lm_vocab: 248_320,
            lm_head_path_tag: 2,
            split_kv_splits: 1,
            output_slot: 0,
            logits_out_ptr: 0x1000_0000,
            meta_dev_ptr: 0x3000_0000,
            stream: 7,
        };
        let variants = [
            base,
            BatchedNoiseGraphKey {
                batch_size: 8,
                ..base
            },
            BatchedNoiseGraphKey {
                gamma_eff: 7,
                ..base
            },
            BatchedNoiseGraphKey { m_rows: 8, ..base },
            BatchedNoiseGraphKey {
                shortlist_k: 16,
                ..base
            },
            BatchedNoiseGraphKey {
                selector_mode: BatchedSelectorMode::ProbabilisticRejection {
                    use_radix_topk: true,
                },
                ..base
            },
            BatchedNoiseGraphKey {
                selector_mode: BatchedSelectorMode::GumbelK32 {
                    use_radix_topk: false,
                },
                ..base
            },
            BatchedNoiseGraphKey {
                selector_mode: BatchedSelectorMode::GumbelK16 {
                    use_radix_topk: true,
                },
                ..base
            },
            BatchedNoiseGraphKey {
                selector_mode: BatchedSelectorMode::GreedyBatched {
                    use_radix_topk: false,
                },
                ..base
            },
            BatchedNoiseGraphKey {
                selector_mode: BatchedSelectorMode::ArgmaxRows,
                ..base
            },
            BatchedNoiseGraphKey {
                lm_vocab: 248_077,
                ..base
            },
            BatchedNoiseGraphKey {
                lm_head_path_tag: 0,
                ..base
            },
            BatchedNoiseGraphKey {
                split_kv_splits: 4,
                ..base
            },
            BatchedNoiseGraphKey {
                output_slot: 1,
                ..base
            },
            BatchedNoiseGraphKey {
                logits_out_ptr: 0x2000_0000,
                ..base
            },
            BatchedNoiseGraphKey {
                meta_dev_ptr: 0x4000_0000,
                ..base
            },
            BatchedNoiseGraphKey { stream: 9, ..base },
        ];
        let set: HashSet<BatchedNoiseGraphKey> = variants.into_iter().collect();
        assert_eq!(set.len(), variants.len());
    }

    #[test]
    fn batched_drafter_nvfp4_tc_and_splitkv_gate_on_high_sm_only() {
        use crate::layers::qwen4_fast_proj::nvfp4::Nvfp4FastProjMode;

        // GB10 (<=64 SMs): never routes batched drafter GEMMs to cuBLASLt NVFP4.
        for gb10_sms in [48u32, 64u32] {
            for m in [8usize, 16, 32, 64, 128] {
                assert!(
                    !super::batched_drafter_nvfp4_tc_admits(
                        gb10_sms,
                        m,
                        32,
                        Nvfp4FastProjMode::All
                    ),
                    "GB10 sm={gb10_sms} m={m} must stay on W4A16"
                );
            }
        }
        // SM120 (>64 SMs): admits m >= 32 when mode is enabled; refuses Off or m < 32.
        for sm120_sms in [65u32, 188u32] {
            assert!(!super::batched_drafter_nvfp4_tc_admits(
                sm120_sms,
                16,
                32,
                Nvfp4FastProjMode::All
            ));
            assert!(!super::batched_drafter_nvfp4_tc_admits(
                sm120_sms,
                31,
                32,
                Nvfp4FastProjMode::All
            ));
            assert!(super::batched_drafter_nvfp4_tc_admits(
                sm120_sms,
                32,
                32,
                Nvfp4FastProjMode::All
            ));
            assert!(super::batched_drafter_nvfp4_tc_admits(
                sm120_sms,
                128,
                32,
                Nvfp4FastProjMode::All
            ));
            assert!(!super::batched_drafter_nvfp4_tc_admits(
                sm120_sms,
                128,
                32,
                Nvfp4FastProjMode::Off
            ));
        }
    }

    #[test]
    fn batched_graph_cache_eviction_and_clear_destroy_graphs() {
        use super::{
            BatchedNoiseGraphCache, BatchedNoiseGraphKey, BatchedSelectorMode,
            MAX_CACHED_BATCHED_NOISE_GRAPHS,
        };
        use spark_runtime::gpu::{
            DevicePtr, GpuBackend, GraphHandle, KernelHandle, PinnedHostBuffer,
            mock::MockGpuBackend,
        };

        struct RecordingGpu {
            inner: MockGpuBackend,
            destroyed: std::sync::Mutex<Vec<u64>>,
        }

        impl GpuBackend for RecordingGpu {
            fn alloc(&self, bytes: usize) -> anyhow::Result<DevicePtr> {
                self.inner.alloc(bytes)
            }
            fn alloc_managed(&self, bytes: usize) -> anyhow::Result<DevicePtr> {
                self.inner.alloc_managed(bytes)
            }
            fn alloc_host_pinned(&self, bytes: usize) -> anyhow::Result<PinnedHostBuffer> {
                self.inner.alloc_host_pinned(bytes)
            }
            fn free(&self, ptr: DevicePtr) -> anyhow::Result<()> {
                self.inner.free(ptr)
            }
            fn copy_h2d(&self, src: &[u8], dst: DevicePtr) -> anyhow::Result<()> {
                self.inner.copy_h2d(src, dst)
            }
            fn copy_d2h(&self, src: DevicePtr, dst: &mut [u8]) -> anyhow::Result<()> {
                self.inner.copy_d2h(src, dst)
            }
            fn copy_d2d(&self, src: DevicePtr, dst: DevicePtr, bytes: usize) -> anyhow::Result<()> {
                self.inner.copy_d2d(src, dst, bytes)
            }
            fn launch(
                &self,
                kernel: KernelHandle,
                grid: [u32; 3],
                block: [u32; 3],
                shared: u32,
                stream: u64,
                params: &mut [*mut std::ffi::c_void],
            ) -> anyhow::Result<()> {
                self.inner
                    .launch(kernel, grid, block, shared, stream, params)
            }
            fn synchronize(&self, stream: u64) -> anyhow::Result<()> {
                self.inner.synchronize(stream)
            }
            fn default_stream(&self) -> u64 {
                self.inner.default_stream()
            }
            fn kernel(&self, module: &str, name: &str) -> anyhow::Result<KernelHandle> {
                self.inner.kernel(module, name)
            }
            fn memset(&self, ptr: DevicePtr, value: u8, bytes: usize) -> anyhow::Result<()> {
                self.inner.memset(ptr, value, bytes)
            }
            fn memset_async(
                &self,
                ptr: DevicePtr,
                value: u8,
                bytes: usize,
                stream: u64,
            ) -> anyhow::Result<()> {
                self.inner.memset_async(ptr, value, bytes, stream)
            }
            fn total_memory(&self) -> anyhow::Result<usize> {
                self.inner.total_memory()
            }
            fn free_memory(&self) -> anyhow::Result<usize> {
                self.inner.free_memory()
            }
            fn destroy_graph(&self, graph: GraphHandle) -> anyhow::Result<()> {
                self.destroyed.lock().unwrap().push(graph.0);
                Ok(())
            }
        }

        let gpu = RecordingGpu {
            inner: MockGpuBackend::new(),
            destroyed: std::sync::Mutex::new(Vec::new()),
        };
        let mut cache = BatchedNoiseGraphCache::new();
        let make_key = |batch_size: usize| BatchedNoiseGraphKey {
            batch_size,
            gamma_eff: 3,
            m_rows: 4,
            shortlist_k: 32,
            selector_mode: BatchedSelectorMode::GumbelK32 {
                use_radix_topk: true,
            },
            lm_vocab: 248_320,
            lm_head_path_tag: 2,
            split_kv_splits: 1,
            output_slot: 0,
            logits_out_ptr: 0x1000,
            meta_dev_ptr: 0x3000,
            stream: 7,
        };
        // Insert past FIFO cap to trigger eviction of graph 1.
        for i in 1..=(MAX_CACHED_BATCHED_NOISE_GRAPHS + 2) {
            cache.insert_graph(&gpu, make_key(i), GraphHandle(i as u64));
        }
        assert_eq!(cache.graphs.len(), MAX_CACHED_BATCHED_NOISE_GRAPHS);
        assert_eq!(*gpu.destroyed.lock().unwrap(), vec![1u64, 2u64]);

        // Clearing destroys all remaining cached graphs.
        cache.clear(&gpu);
        assert!(cache.graphs.is_empty());
        assert_eq!(
            gpu.destroyed.lock().unwrap().len(),
            MAX_CACHED_BATCHED_NOISE_GRAPHS + 2
        );
    }

    #[test]
    fn batched_stage2_meta_layout_alignment_bounds_and_active_prefixes() {
        use super::{BATCHED_MAX_NOISE_ROWS, DFLASH_BATCH_MAX_SEQS, Stage2MetaLayout};
        use spark_runtime::gpu::DevicePtr;

        assert!(Stage2MetaLayout::new(0).is_err());
        for num_layers in [1usize, 2, 5, 8] {
            let layout = Stage2MetaLayout::new(num_layers).unwrap();
            let ptrs_cap = num_layers * DFLASH_BATCH_MAX_SEQS * 8;
            let segments = [
                (layout.token_ids_off, BATCHED_MAX_NOISE_ROWS * 4),
                (layout.pos_ids_off, BATCHED_MAX_NOISE_ROWS * 4),
                (layout.eff_ctx_off, DFLASH_BATCH_MAX_SEQS * 4),
                (layout.ring_start_off, DFLASH_BATCH_MAX_SEQS * 4),
                (layout.anchor_ids_off, DFLASH_BATCH_MAX_SEQS * 4),
                (layout.ctx_k_ptrs_off, ptrs_cap),
                (layout.ctx_v_ptrs_off, ptrs_cap),
                (layout.gumbel_seeds_off, DFLASH_BATCH_MAX_SEQS * 8),
                (layout.gumbel_step0_off, DFLASH_BATCH_MAX_SEQS * 8),
                (layout.gumbel_temps_off, DFLASH_BATCH_MAX_SEQS * 4),
                (layout.gumbel_top_ks_off, DFLASH_BATCH_MAX_SEQS * 4),
                (layout.gumbel_top_ps_off, DFLASH_BATCH_MAX_SEQS * 4),
                (layout.gumbel_min_ps_off, DFLASH_BATCH_MAX_SEQS * 4),
                (layout.q_row_bases_off, DFLASH_BATCH_MAX_SEQS * 4),
            ];
            let mut cursor = 0usize;
            for (off, size) in segments {
                assert_eq!(off, cursor);
                assert!(off.is_multiple_of(256));
                assert!(off.is_multiple_of(16));
                assert!(off.is_multiple_of(8));
                cursor = off + size;
            }
            assert_eq!(cursor, layout.total_bytes);
            assert!(layout.total_bytes.is_multiple_of(256));

            // Active contiguous prefix strictly grows Base < Base+Gumbel < Base+Gumbel+Rejection
            // and never exceeds `total_bytes`.
            for b in [1usize, 2, 16, DFLASH_BATCH_MAX_SEQS] {
                let base_only = layout.active_upload_bytes(b, false, false);
                let with_gumbel = layout.active_upload_bytes(b, true, false);
                let with_rej = layout.active_upload_bytes(b, true, true);
                assert_eq!(base_only, layout.ctx_v_ptrs_off + num_layers * b * 8);
                assert_eq!(with_gumbel, layout.gumbel_min_ps_off + b * 4);
                assert_eq!(with_rej, layout.q_row_bases_off + b * 4);
                assert!(base_only < with_gumbel);
                assert!(with_gumbel < with_rej);
                assert!(with_rej <= layout.total_bytes);
            }

            // Device binding reproduces exact offsets from `base`.
            let base = DevicePtr(0x1000_0000);
            let ptrs = layout.bind_device(base);
            assert_eq!(ptrs.base, base);
            assert_eq!(
                ptrs.draft_tokens_dev.0 - base.0,
                layout.token_ids_off as u64
            );
            assert_eq!(ptrs.position_ids.0 - base.0, layout.pos_ids_off as u64);
            assert_eq!(ptrs.batch_eff_ctx.0 - base.0, layout.eff_ctx_off as u64);
            assert_eq!(
                ptrs.batch_ring_start.0 - base.0,
                layout.ring_start_off as u64
            );
            assert_eq!(
                ptrs.batch_anchor_ids.0 - base.0,
                layout.anchor_ids_off as u64
            );
            assert_eq!(
                ptrs.batch_ctx_k_ptrs.0 - base.0,
                layout.ctx_k_ptrs_off as u64
            );
            assert_eq!(
                ptrs.batch_ctx_v_ptrs.0 - base.0,
                layout.ctx_v_ptrs_off as u64
            );
            assert_eq!(
                ptrs.batch_gumbel_seeds.0 - base.0,
                layout.gumbel_seeds_off as u64
            );
            assert_eq!(
                ptrs.batch_gumbel_step0_pos.0 - base.0,
                layout.gumbel_step0_off as u64
            );
            assert_eq!(
                ptrs.batch_gumbel_temps.0 - base.0,
                layout.gumbel_temps_off as u64
            );
            assert_eq!(
                ptrs.batch_gumbel_top_ks.0 - base.0,
                layout.gumbel_top_ks_off as u64
            );
            assert_eq!(
                ptrs.batch_gumbel_top_ps.0 - base.0,
                layout.gumbel_top_ps_off as u64
            );
            assert_eq!(
                ptrs.batch_gumbel_min_ps.0 - base.0,
                layout.gumbel_min_ps_off as u64
            );
            assert_eq!(ptrs.q_row_bases.0 - base.0, layout.q_row_bases_off as u64);
        }
    }

    #[test]
    fn batched_pinned_stage2_meta_upload_packs_all_buffers_across_rotating_slots() {
        use super::{BatchedNoiseGraphCache, Stage2MetaLayout};
        use spark_runtime::gpu::mock::MockGpuBackend;
        use spark_runtime::gpu::{
            DevicePtr, GpuBackend, KernelHandle, PinnedHostBuffer, PinnedHostSlice,
        };
        use spark_runtime::sampler::gumbel::GumbelDraftSpec;

        struct CountingH2dGpu {
            inner: MockGpuBackend,
            pinned_h2d_calls: std::sync::Mutex<Vec<(u64, usize)>>,
        }

        impl GpuBackend for CountingH2dGpu {
            fn alloc(&self, bytes: usize) -> anyhow::Result<DevicePtr> {
                self.inner.alloc(bytes)
            }
            fn alloc_managed(&self, bytes: usize) -> anyhow::Result<DevicePtr> {
                self.inner.alloc_managed(bytes)
            }
            fn alloc_host_pinned(&self, bytes: usize) -> anyhow::Result<PinnedHostBuffer> {
                self.inner.alloc_host_pinned(bytes)
            }
            fn free(&self, ptr: DevicePtr) -> anyhow::Result<()> {
                self.inner.free(ptr)
            }
            fn copy_h2d(&self, src: &[u8], dst: DevicePtr) -> anyhow::Result<()> {
                self.inner.copy_h2d(src, dst)
            }
            unsafe fn copy_h2d_pinned_async(
                &self,
                src: PinnedHostSlice<'_>,
                dst: DevicePtr,
                stream: u64,
            ) -> anyhow::Result<()> {
                self.pinned_h2d_calls
                    .lock()
                    .unwrap()
                    .push((dst.0, src.len()));
                // SAFETY: delegates caller's page-locked slice contract directly to `MockGpuBackend`.
                unsafe { self.inner.copy_h2d_pinned_async(src, dst, stream) }
            }
            fn copy_d2h(&self, src: DevicePtr, dst: &mut [u8]) -> anyhow::Result<()> {
                self.inner.copy_d2h(src, dst)
            }
            fn copy_d2d(&self, src: DevicePtr, dst: DevicePtr, bytes: usize) -> anyhow::Result<()> {
                self.inner.copy_d2d(src, dst, bytes)
            }
            fn launch(
                &self,
                kernel: KernelHandle,
                grid: [u32; 3],
                block: [u32; 3],
                shared: u32,
                stream: u64,
                params: &mut [*mut std::ffi::c_void],
            ) -> anyhow::Result<()> {
                self.inner
                    .launch(kernel, grid, block, shared, stream, params)
            }
            fn synchronize(&self, stream: u64) -> anyhow::Result<()> {
                self.inner.synchronize(stream)
            }
            fn default_stream(&self) -> u64 {
                self.inner.default_stream()
            }
            fn kernel(&self, module: &str, name: &str) -> anyhow::Result<KernelHandle> {
                self.inner.kernel(module, name)
            }
            fn memset(&self, ptr: DevicePtr, value: u8, bytes: usize) -> anyhow::Result<()> {
                self.inner.memset(ptr, value, bytes)
            }
            fn memset_async(
                &self,
                ptr: DevicePtr,
                value: u8,
                bytes: usize,
                stream: u64,
            ) -> anyhow::Result<()> {
                self.inner.memset_async(ptr, value, bytes, stream)
            }
            fn total_memory(&self) -> anyhow::Result<usize> {
                self.inner.total_memory()
            }
            fn free_memory(&self) -> anyhow::Result<usize> {
                self.inner.free_memory()
            }
        }

        let gpu = CountingH2dGpu {
            inner: MockGpuBackend::new(),
            pinned_h2d_calls: std::sync::Mutex::new(Vec::new()),
        };
        let num_layers = 2usize;
        let layout = Stage2MetaLayout::new(num_layers).unwrap();
        let plans = [
            SeqContextPlan {
                eff_ctx: 100,
                needed_start: 0,
                needed_end: 100,
                ring_start: 0,
                new_start: 100,
                new_count: 0,
                invalidate_prev: false,
            },
            SeqContextPlan {
                eff_ctx: 200,
                needed_start: 16,
                needed_end: 216,
                ring_start: 16,
                new_start: 216,
                new_count: 0,
                invalidate_prev: false,
            },
        ];
        let spec0 = GumbelDraftSpec {
            seed: 0xAA01,
            temperature: 0.7,
            top_k: 20,
            top_p: 0.95,
            min_p: 0.05,
            step0_abs_pos: 100,
        };
        let mut cache = BatchedNoiseGraphCache::new();

        // Fail-closed bounds checks before allocation/upload.
        assert!(
            cache
                .upload_stage2_meta(
                    &gpu,
                    num_layers,
                    &[],
                    &[],
                    &[],
                    4,
                    999,
                    |_, _| 0,
                    |_, _| 0,
                    None,
                    None,
                    0,
                )
                .is_err()
        );
        assert!(
            cache
                .upload_stage2_meta(
                    &gpu,
                    num_layers,
                    &[10, 20],
                    &[100],
                    &plans,
                    4,
                    999,
                    |_, _| 0,
                    |_, _| 0,
                    None,
                    None,
                    0,
                )
                .is_err()
        );
        assert!(
            cache
                .upload_stage2_meta(
                    &gpu,
                    num_layers,
                    &[10, 20],
                    &[100, 216],
                    &plans,
                    4,
                    999,
                    |_, _| 0,
                    |_, _| 0,
                    Some(&[Some(spec0)]),
                    None,
                    0,
                )
                .is_err()
        );
        assert!(gpu.pinned_h2d_calls.lock().unwrap().is_empty());

        // 1. Base-only upload -> exactly 1 H2D of `active_upload_bytes(2, false, false)`.
        let (coupled_base, ptrs_base) = cache
            .upload_stage2_meta(
                &gpu,
                num_layers,
                &[10, 20],
                &[100, 216],
                &plans,
                4,
                999,
                |l, s| 0x1000 + (l as u64) * 0x100 + (s as u64) * 0x10,
                |l, s| 0x2000 + (l as u64) * 0x100 + (s as u64) * 0x10,
                None,
                None,
                0,
            )
            .unwrap();
        assert_eq!(coupled_base, 0);
        {
            let mut calls = gpu.pinned_h2d_calls.lock().unwrap();
            assert_eq!(
                *calls,
                vec![(
                    ptrs_base.base.0,
                    layout.active_upload_bytes(2, false, false)
                )]
            );
            calls.clear();
        }

        // 2. Base + Gumbel upload -> exactly 1 H2D, identical device sub-pointers.
        let (coupled_gumbel, ptrs_gumbel) = cache
            .upload_stage2_meta(
                &gpu,
                num_layers,
                &[11, 21],
                &[100, 216],
                &plans,
                4,
                999,
                |l, s| 0x1000 + (l as u64) * 0x100 + (s as u64) * 0x10,
                |l, s| 0x2000 + (l as u64) * 0x100 + (s as u64) * 0x10,
                Some(&[Some(spec0), None]),
                None,
                0,
            )
            .unwrap();
        assert_eq!(coupled_gumbel, 1);
        assert_eq!(ptrs_gumbel, ptrs_base);
        {
            let mut calls = gpu.pinned_h2d_calls.lock().unwrap();
            assert_eq!(
                *calls,
                vec![(ptrs_base.base.0, layout.active_upload_bytes(2, true, false))]
            );
            calls.clear();
        }

        // 3. Rotate past BATCHED_META_PINNED_SLOTS (4) with full Base + Gumbel + Rejection:
        //    every step emits exactly 1 H2D and preserves immutable graph sub-pointers.
        let mut last_ptrs = ptrs_base;
        for step in 0..6u32 {
            let (coupled, ptrs) = cache
                .upload_stage2_meta(
                    &gpu,
                    num_layers,
                    &[10 + step, 20 + step],
                    &[100, 216],
                    &plans,
                    4,
                    999,
                    |l, s| 0x1000 + (l as u64) * 0x100 + (s as u64) * 0x10,
                    |l, s| 0x2000 + (l as u64) * 0x100 + (s as u64) * 0x10,
                    Some(&[Some(spec0), None]),
                    Some(&[0u32, 24u32]),
                    0,
                )
                .unwrap();
            assert_eq!(coupled, 1);
            assert_eq!(ptrs, ptrs_base);
            last_ptrs = ptrs;
        }
        {
            let calls = gpu.pinned_h2d_calls.lock().unwrap();
            assert_eq!(calls.len(), 6);
            let expected_bytes = layout.active_upload_bytes(2, true, true);
            assert!(
                calls
                    .iter()
                    .all(|&(dst, len)| dst == ptrs_base.base.0 && len == expected_bytes)
            );
        }

        let slab = gpu.inner.read_alloc(last_ptrs.base).unwrap();
        assert_eq!(slab.len(), layout.total_bytes);
        let read_u32 = |ptr: DevicePtr, idx: usize| {
            let off = (ptr.0 - last_ptrs.base.0) as usize + idx * 4;
            u32::from_le_bytes(slab[off..off + 4].try_into().unwrap())
        };
        let read_u64 = |ptr: DevicePtr, idx: usize| {
            let off = (ptr.0 - last_ptrs.base.0) as usize + idx * 8;
            u64::from_le_bytes(slab[off..off + 8].try_into().unwrap())
        };
        let read_f32 = |ptr: DevicePtr, idx: usize| {
            let off = (ptr.0 - last_ptrs.base.0) as usize + idx * 4;
            f32::from_le_bytes(slab[off..off + 4].try_into().unwrap())
        };

        // Last step was step=5 -> anchors [15, 25], masks = 999.
        assert_eq!(read_u32(last_ptrs.draft_tokens_dev, 0), 15);
        assert_eq!(read_u32(last_ptrs.draft_tokens_dev, 1), 999);
        assert_eq!(read_u32(last_ptrs.draft_tokens_dev, 4), 25);
        assert_eq!(read_u32(last_ptrs.draft_tokens_dev, 7), 999);
        assert_eq!(read_u32(last_ptrs.position_ids, 0), 100);
        assert_eq!(read_u32(last_ptrs.position_ids, 3), 103);
        assert_eq!(read_u32(last_ptrs.position_ids, 4), 216);
        assert_eq!(read_u32(last_ptrs.position_ids, 7), 219);
        assert_eq!(read_u32(last_ptrs.batch_eff_ctx, 0), 100);
        assert_eq!(read_u32(last_ptrs.batch_eff_ctx, 1), 200);
        assert_eq!(read_u32(last_ptrs.batch_ring_start, 0), 0);
        assert_eq!(read_u32(last_ptrs.batch_ring_start, 1), 16);
        assert_eq!(read_u32(last_ptrs.batch_anchor_ids, 0), 15);
        assert_eq!(read_u32(last_ptrs.batch_anchor_ids, 1), 25);
        assert_eq!(read_u64(last_ptrs.batch_ctx_k_ptrs, 0), 0x1000);
        assert_eq!(read_u64(last_ptrs.batch_ctx_k_ptrs, 1), 0x1010);
        assert_eq!(read_u64(last_ptrs.batch_ctx_k_ptrs, 2), 0x1100);
        assert_eq!(read_u64(last_ptrs.batch_ctx_v_ptrs, 3), 0x2110);
        assert_eq!(read_u64(last_ptrs.batch_gumbel_seeds, 0), 0xAA01);
        assert_eq!(read_u64(last_ptrs.batch_gumbel_seeds, 1), 0);
        assert_eq!(read_u64(last_ptrs.batch_gumbel_step0_pos, 0), 100);
        assert_eq!(read_f32(last_ptrs.batch_gumbel_temps, 0), 0.7);
        assert_eq!(read_f32(last_ptrs.batch_gumbel_temps, 1), 0.0);
        assert_eq!(read_u32(last_ptrs.batch_gumbel_top_ks, 0), 20);
        assert_eq!(read_f32(last_ptrs.batch_gumbel_top_ps, 0), 0.95);
        assert_eq!(read_f32(last_ptrs.batch_gumbel_top_ps, 1), 1.0);
        assert_eq!(read_f32(last_ptrs.batch_gumbel_min_ps, 0), 0.05);
        assert_eq!(read_u32(last_ptrs.q_row_bases, 0), 0);
        assert_eq!(read_u32(last_ptrs.q_row_bases, 1), 24);

        // Clearing releases `meta_dev` and lets a subsequent upload reallocate cleanly.
        cache.clear(&gpu);
        assert!(cache.meta_dev.is_null());
        assert_eq!(cache.meta_dev_bytes, 0);
        assert!(gpu.inner.read_alloc(last_ptrs.base).is_none());
        let (_, ptrs_after_clear) = cache
            .upload_stage2_meta(
                &gpu,
                num_layers,
                &[10, 20],
                &[100, 216],
                &plans,
                4,
                999,
                |l, s| 0x1000 + (l as u64) * 0x100 + (s as u64) * 0x10,
                |l, s| 0x2000 + (l as u64) * 0x100 + (s as u64) * 0x10,
                None,
                None,
                0,
            )
            .unwrap();
        assert!(!ptrs_after_clear.base.is_null());
        assert!(gpu.inner.read_alloc(ptrs_after_clear.base).is_some());
    }

    #[test]
    fn strided_compact_params_validates_geometry_and_alignment() {
        use super::{StridedCompactParams, compute_strided_compact_params};
        use spark_runtime::gpu::DevicePtr;

        // DFlash2 B=16, gamma=3, m_rows=4, output_start=1, hidden=2560 (BF16).
        let p = compute_strided_compact_params(
            16,
            4,
            3,
            1,
            2560,
            DevicePtr(0x1_0000),
            DevicePtr(0x2_0000),
        )
        .unwrap();
        assert_eq!(
            p,
            StridedCompactParams {
                src_offset: 5120,
                rows: 16,
                row_bytes: 3 * 5120,
                src_stride: 4 * 5120,
                dst_stride: 3 * 5120,
            }
        );

        // Misaligned device pointer falls back cleanly (returns None).
        assert!(
            compute_strided_compact_params(
                16,
                4,
                3,
                1,
                2560,
                DevicePtr(0x1_0008),
                DevicePtr(0x2_0000),
            )
            .is_none()
        );
    }

    #[test]
    fn stage1_meta_upload_packs_wrapped_gather_and_per_layer_kv_scatter_across_slots() {
        use super::{
            BatchedNoiseGraphCache, STAGE1_MAX_CHUNK_ROWS, Stage1MetaLayout, Stage1Slice,
            ring_window,
        };
        use spark_runtime::gpu::mock::MockGpuBackend;

        // Bounds and alignment validation (including odd chunk_rows = 7).
        assert!(Stage1MetaLayout::new(0, 5).is_err());
        assert!(Stage1MetaLayout::new(STAGE1_MAX_CHUNK_ROWS + 1, 5).is_err());
        assert!(Stage1MetaLayout::new(7, 0).is_err());
        let odd_layout = Stage1MetaLayout::new(7, 2).unwrap();
        assert!(odd_layout.gather_src_ptrs_off.is_multiple_of(8));
        assert!(odd_layout.scatter_dst_k_ptrs_off.is_multiple_of(8));
        assert!(odd_layout.scatter_dst_v_ptrs_off.is_multiple_of(8));
        assert!(odd_layout.gather_src_row_idx_off.is_multiple_of(4));
        assert!(odd_layout.scatter_dst_slot_off.is_multiple_of(4));
        assert!(odd_layout.scatter_row_pos_off.is_multiple_of(4));

        let gpu = MockGpuBackend::new();
        let mut cache = BatchedNoiseGraphCache::new();
        let ctx_window = 16usize;
        let num_layers = 2usize;

        // Seq 0: ctx_hidden_acc ring wraps across capacity 8 (abs [6..10) -> slots [6,7,0,1]).
        // Seq 1: KV ring wraps across ctx_window 16 (abs [14..17) -> dst_slot [14,15,0]).
        let rings = [
            ring_window::RingState::from_lengths(64, 8, 10, 8).unwrap(),
            ring_window::RingState::from_lengths(64, 16, 17, 16).unwrap(),
        ];
        let slices = [
            Stage1Slice {
                seq_idx: 0,
                abs_start: 6,
                count: 4,
                chunk_offset: 0,
            },
            Stage1Slice {
                seq_idx: 1,
                abs_start: 14,
                count: 3,
                chunk_offset: 4,
            },
        ];

        // Rotate past STAGE1_META_PINNED_SLOTS (4) and confirm single device slab reuse.
        let mut last_meta = None;
        let mut dev_base = None;
        for step in 0..6u64 {
            let meta = cache
                .upload_stage1_meta(
                    &gpu,
                    &slices,
                    7,
                    num_layers,
                    ctx_window,
                    |s| 0xA000 + step * 0x100 + (s as u64) * 0x10,
                    |s| Ok(rings[s]),
                    |s, l| 0xB000 + (l as u64) * 0x100 + (s as u64) * 0x10,
                    |s, l| 0xC000 + (l as u64) * 0x100 + (s as u64) * 0x10,
                    0,
                )
                .unwrap();
            if let Some(first_base) = dev_base {
                assert_eq!(cache.stage1_meta_dev, first_base);
            } else {
                dev_base = Some(cache.stage1_meta_dev);
            }
            last_meta = Some(meta);
        }

        let base = dev_base.unwrap();
        let meta = last_meta.unwrap();
        let slab = gpu.read_alloc(base).unwrap();

        let read_u64 = |ptr: spark_runtime::gpu::DevicePtr, idx: usize| {
            let off = (ptr.0 - base.0) as usize + idx * 8;
            u64::from_le_bytes(slab[off..off + 8].try_into().unwrap())
        };
        let read_u32 = |ptr: spark_runtime::gpu::DevicePtr, idx: usize| {
            let off = (ptr.0 - base.0) as usize + idx * 4;
            u32::from_le_bytes(slab[off..off + 4].try_into().unwrap())
        };

        // Last step = 5 -> hidden bases 0xA500 (seq 0) and 0xA510 (seq 1).
        let expected_hidden = [0xA500, 0xA500, 0xA500, 0xA500, 0xA510, 0xA510, 0xA510];
        let expected_src_row = [6u32, 7, 0, 1, 14, 15, 0];
        let expected_dst_slot = [6u32, 7, 8, 9, 14, 15, 0];
        let expected_row_pos = [6u32, 7, 8, 9, 14, 15, 16];
        for r in 0..7 {
            assert_eq!(read_u64(meta.gather_src_ptrs, r), expected_hidden[r]);
            assert_eq!(read_u32(meta.gather_src_row_idx, r), expected_src_row[r]);
            assert_eq!(read_u32(meta.scatter_dst_slot, r), expected_dst_slot[r]);
            assert_eq!(read_u32(meta.scatter_row_pos, r), expected_row_pos[r]);
            let seq = usize::from(r >= 4);
            for l in 0..num_layers {
                assert_eq!(
                    read_u64(meta.layer_dst_k_ptrs(l), r),
                    0xB000 + (l as u64) * 0x100 + (seq as u64) * 0x10
                );
                assert_eq!(
                    read_u64(meta.layer_dst_v_ptrs(l), r),
                    0xC000 + (l as u64) * 0x100 + (seq as u64) * 0x10
                );
            }
        }
    }

    #[test]
    fn batched_selector_plan_bypasses_rejection_for_pure_greedy_and_keeps_it_for_stochastic() {
        use super::super::noise_pass::{REJECTION_SLOT_STRIDE, resolve_rejection_shortlist_k};
        use super::{
            BatchedSelectorMode, BatchedSelectorPlan, BatchedSelectorPlanInput,
            resolve_batched_selector_plan,
        };
        use crate::model::trait_impl::device_topk::REJECTION_Q_STRIDE;
        use spark_runtime::sampler::gumbel::GumbelDraftSpec;

        let greedy_spec = GumbelDraftSpec {
            seed: 0x1111,
            temperature: 0.0,
            top_k: 20,
            top_p: 0.95,
            min_p: 0.0,
            step0_abs_pos: 64,
        };
        let stochastic_spec = GumbelDraftSpec {
            temperature: 0.7,
            ..greedy_spec
        };
        let pure_greedy_cohort = [None, Some(greedy_spec)];
        let mixed_cohort = [None, Some(stochastic_spec)];
        let pure_greedy_active = pure_greedy_cohort
            .iter()
            .any(|s| s.is_some_and(|g| g.is_active()));
        let mixed_active = mixed_cohort
            .iter()
            .any(|s| s.is_some_and(|g| g.is_active()));
        assert!(!pure_greedy_active);
        assert!(mixed_active);

        let base = BatchedSelectorPlanInput {
            rejection_env: true,
            any_coupled_gumbel: pure_greedy_active,
            sel_top_k: Some(16),
            rejection_selector_ready: true,
            ngram_bonus_ready: false,
            shared_q_stride: REJECTION_Q_STRIDE,
            shared_q_row_cap: 16 * REJECTION_SLOT_STRIDE,
            gamma_eff: 7,
            batch_size: 16,
            lm_stride: 248_320,
            has_topk_radix: true,
            gumbel_k32_ready: true,
            gumbel_k16_ready: true,
            has_argmax_rows: true,
        };

        // Pure-greedy T=0.0 cohort under Rejection policy takes GumbelK32 (exact sel_top_k=16
        // + softmax out_probs) instead of ProbabilisticRejection.
        let greedy_plan = resolve_batched_selector_plan(&base);
        assert_eq!(
            greedy_plan,
            BatchedSelectorPlan {
                rejection_mode: false,
                use_gumbel_selector: true,
                shortlist_k: 16,
                selector_mode: BatchedSelectorMode::GumbelK32 {
                    use_radix_topk: true,
                },
            }
        );

        // Without k32/k16 gumbel selector readiness, pure-greedy falls back to GreedyBatched.
        let greedy_fallback = resolve_batched_selector_plan(&BatchedSelectorPlanInput {
            gumbel_k32_ready: false,
            gumbel_k16_ready: false,
            ..base
        });
        assert_eq!(
            greedy_fallback,
            BatchedSelectorPlan {
                rejection_mode: false,
                use_gumbel_selector: false,
                shortlist_k: 16,
                selector_mode: BatchedSelectorMode::GreedyBatched {
                    use_radix_topk: true,
                },
            }
        );

        // Mixed or stochastic T>0 cohort under Rejection policy keeps ProbabilisticRejection.
        let (expected_rej_k, expected_radix) =
            resolve_rejection_shortlist_k(base.lm_stride, true, REJECTION_Q_STRIDE);
        let stochastic_plan = resolve_batched_selector_plan(&BatchedSelectorPlanInput {
            any_coupled_gumbel: mixed_active,
            ..base
        });
        assert_eq!(
            stochastic_plan,
            BatchedSelectorPlan {
                rejection_mode: true,
                use_gumbel_selector: false,
                shortlist_k: expected_rej_k,
                selector_mode: BatchedSelectorMode::ProbabilisticRejection {
                    use_radix_topk: expected_radix,
                },
            }
        );

        // When rejection policy is off, stochastic cohorts stay on GumbelK32.
        let gumbel_plan = resolve_batched_selector_plan(&BatchedSelectorPlanInput {
            rejection_env: false,
            any_coupled_gumbel: mixed_active,
            ..base
        });
        assert_eq!(
            gumbel_plan,
            BatchedSelectorPlan {
                rejection_mode: false,
                use_gumbel_selector: true,
                shortlist_k: 16,
                selector_mode: BatchedSelectorMode::GumbelK32 {
                    use_radix_topk: true,
                },
            }
        );

        // Opt-in n-gram bonus engages ONLY for pure-greedy cohorts and is strictly
        // bypassed for stochastic / mixed cohorts (Gate 0).
        let ngram_greedy = resolve_batched_selector_plan(&BatchedSelectorPlanInput {
            ngram_bonus_ready: true,
            ..base
        });
        assert_eq!(
            ngram_greedy,
            BatchedSelectorPlan {
                rejection_mode: false,
                use_gumbel_selector: false,
                shortlist_k: 16,
                selector_mode: BatchedSelectorMode::GreedyNgramBatched {
                    use_radix_topk: true,
                },
            }
        );
        let ngram_blocked_on_stochastic =
            resolve_batched_selector_plan(&BatchedSelectorPlanInput {
                ngram_bonus_ready: true,
                any_coupled_gumbel: mixed_active,
                ..base
            });
        assert_eq!(ngram_blocked_on_stochastic, stochastic_plan);
    }

    #[test]
    fn batched_noise_graph_batch_bucket_and_padded_resolution_enforce_capacity_and_sinks() {
        use super::super::DFLASH_BATCH_MAX_SEQS;
        use super::super::noise_pass::REJECTION_SLOT_STRIDE;
        use super::{
            PaddedNoiseBatchInput, assign_padded_rejection_q_sinks,
            batched_noise_graph_batch_bucket, count_dedicated_rejection_q_sinks,
            resolve_padded_noise_batch_size,
        };

        // Bucketing into `{1, 2, 4, 8, 16, 24, 32, 48, 64}` with `> 64` unpadded.
        let expected = [
            (0, 0),
            (1, 1),
            (2, 2),
            (3, 4),
            (4, 4),
            (5, 8),
            (6, 8),
            (7, 8),
            (8, 8),
            (9, 16),
            (13, 16),
            (15, 16),
            (16, 16),
            (17, 24),
            (20, 24),
            (24, 24),
            (25, 32),
            (32, 32),
            (33, 48),
            (48, 48),
            (49, 64),
            (64, 64),
            (65, 65),
        ];
        for (actual, bucket) in expected {
            assert_eq!(batched_noise_graph_batch_bucket(actual), bucket);
        }

        // C=24 x K=8 (192 rows), C=32 x K=8 (256 rows), C=48 x K=5 (240 rows),
        // and C=64 x K=4 (256 rows) resolve cleanly under 256-row capacity.
        let wide_base = PaddedNoiseBatchInput {
            actual_b: 20,
            graph_enabled: true,
            m_rows: 8,
            gamma_eff: 7,
            n_attn_scratch: 2056,
            max_sel_rows: 256,
            lm_stride: 248_320,
            logits_bytes: 256 * 248_320 * 2,
            rejection_mode: true,
            available_rejection_sinks: 32,
        };
        assert_eq!(resolve_padded_noise_batch_size(&wide_base), 24);
        assert_eq!(
            resolve_padded_noise_batch_size(&PaddedNoiseBatchInput {
                actual_b: 29,
                ..wide_base
            }),
            32
        );
        assert_eq!(
            resolve_padded_noise_batch_size(&PaddedNoiseBatchInput {
                actual_b: 40,
                m_rows: 5,
                gamma_eff: 4,
                available_rejection_sinks: 16,
                ..wide_base
            }),
            48
        );
        assert_eq!(
            resolve_padded_noise_batch_size(&PaddedNoiseBatchInput {
                actual_b: 56,
                m_rows: 4,
                gamma_eff: 3,
                available_rejection_sinks: 16,
                ..wide_base
            }),
            64
        );

        let base = PaddedNoiseBatchInput {
            actual_b: 13,
            graph_enabled: true,
            m_rows: 8,
            gamma_eff: 7,
            n_attn_scratch: 2048,
            max_sel_rows: 128,
            lm_stride: 248_320,
            logits_bytes: 128 * 248_320 * 2,
            rejection_mode: true,
            available_rejection_sinks: 48,
        };
        assert_eq!(resolve_padded_noise_batch_size(&base), 16);
        // Eager (`!graph_enabled`) never pads.
        assert_eq!(
            resolve_padded_noise_batch_size(&PaddedNoiseBatchInput {
                graph_enabled: false,
                ..base
            }),
            13
        );
        // Scratch / logits overflow fails closed to `actual_b`.
        assert_eq!(
            resolve_padded_noise_batch_size(&PaddedNoiseBatchInput {
                n_attn_scratch: 120,
                ..base
            }),
            13
        );
        assert_eq!(
            resolve_padded_noise_batch_size(&PaddedNoiseBatchInput {
                max_sel_rows: 100,
                ..base
            }),
            13
        );
        assert_eq!(
            resolve_padded_noise_batch_size(&PaddedNoiseBatchInput {
                logits_bytes: 100 * 248_320 * 2,
                ..base
            }),
            13
        );
        // Rejection mode requires enough dedicated unallocatable sinks (`>= padded_b - actual_b`).
        assert_eq!(
            resolve_padded_noise_batch_size(&PaddedNoiseBatchInput {
                available_rejection_sinks: 2,
                ..base
            }),
            13
        );
        assert_eq!(
            resolve_padded_noise_batch_size(&PaddedNoiseBatchInput {
                available_rejection_sinks: 3,
                ..base
            }),
            16
        );
        // Non-rejection mode ignores `available_rejection_sinks`.
        assert_eq!(
            resolve_padded_noise_batch_size(&PaddedNoiseBatchInput {
                rejection_mode: false,
                available_rejection_sinks: 0,
                ..base
            }),
            16
        );

        // Dedicated sink allocation strictly uses slots `>= configured_lanes` and
        // never collides with active slots (`0..configured_lanes`) or already-used sinks.
        let total_q_slots = 64usize;
        let row_cap = total_q_slots * REJECTION_SLOT_STRIDE;
        let mut used_slots = [false; DFLASH_BATCH_MAX_SEQS];
        used_slots[..13].fill(true);
        // Pretend an active greedy sequence in a mixed batch already claimed top sink 63.
        used_slots[63] = true;
        let active_slots: Vec<Option<usize>> = (0..13).map(Some).collect();
        assert_eq!(
            count_dedicated_rejection_q_sinks(
                16,
                total_q_slots,
                REJECTION_SLOT_STRIDE,
                7,
                row_cap,
                &used_slots,
                &active_slots,
            ),
            47
        );
        // Unregistered lane table (`configured_lanes <= 1`) or full 64-lane table fails closed.
        assert_eq!(
            count_dedicated_rejection_q_sinks(
                1,
                total_q_slots,
                REJECTION_SLOT_STRIDE,
                7,
                row_cap,
                &used_slots,
                &active_slots,
            ),
            0
        );
        assert_eq!(
            count_dedicated_rejection_q_sinks(
                64,
                total_q_slots,
                REJECTION_SLOT_STRIDE,
                7,
                row_cap,
                &used_slots,
                &active_slots,
            ),
            0
        );
        let mut q_row_bases = [0u32; DFLASH_BATCH_MAX_SEQS];
        assert!(assign_padded_rejection_q_sinks(
            13,
            16,
            16,
            total_q_slots,
            REJECTION_SLOT_STRIDE,
            7,
            row_cap,
            &mut used_slots,
            &mut q_row_bases,
        ));
        assert_eq!(
            &q_row_bases[13..16],
            &[
                (62 * REJECTION_SLOT_STRIDE) as u32,
                (61 * REJECTION_SLOT_STRIDE) as u32,
                (60 * REJECTION_SLOT_STRIDE) as u32,
            ]
        );
    }

    #[test]
    fn upload_stage2_meta_padded_isolates_padded_lanes_and_preserves_padded_ptr_stride() {
        use super::{BatchedNoiseGraphCache, SeqContextPlan};
        use spark_runtime::gpu::mock::MockGpuBackend;
        use spark_runtime::sampler::gumbel::GumbelDraftSpec;

        let gpu = MockGpuBackend::new();
        let num_layers = 2usize;
        let actual_b = 3usize;
        let padded_b = 4usize;
        let m_rows = 8usize;
        let mask_id = 999u32;
        let last_tokens = [42u32, 43, 44];
        let positions = [100usize, 200, 300];
        let plans = [
            SeqContextPlan {
                eff_ctx: 64,
                needed_start: 36,
                needed_end: 100,
                ring_start: 36,
                new_start: 100,
                new_count: 0,
                invalidate_prev: false,
            },
            SeqContextPlan {
                eff_ctx: 128,
                needed_start: 72,
                needed_end: 200,
                ring_start: 72,
                new_start: 200,
                new_count: 0,
                invalidate_prev: false,
            },
            SeqContextPlan {
                eff_ctx: 96,
                needed_start: 204,
                needed_end: 300,
                ring_start: 204,
                new_start: 300,
                new_count: 0,
                invalidate_prev: false,
            },
        ];
        let active_spec = GumbelDraftSpec {
            seed: 0xBEEF,
            temperature: 0.7,
            top_k: 20,
            top_p: 0.95,
            min_p: 0.05,
            step0_abs_pos: 100,
        };
        let specs = [Some(active_spec), None, None, None];
        let q_row_bases = [0u32, 8, 16, 63 * 8];
        let mut cache = BatchedNoiseGraphCache::new();

        // Reject active Gumbel spec in any trailing padded lane.
        let bad_specs = [Some(active_spec), None, None, Some(active_spec)];
        assert!(
            cache
                .upload_stage2_meta_padded(
                    &gpu,
                    num_layers,
                    &last_tokens,
                    &positions,
                    &plans,
                    padded_b,
                    m_rows,
                    mask_id,
                    |l, s| 0x1000 + (l as u64) * 0x100 + (s as u64) * 0x10,
                    |l, s| 0x2000 + (l as u64) * 0x100 + (s as u64) * 0x10,
                    Some(&bad_specs),
                    Some(&q_row_bases),
                    0,
                )
                .is_err()
        );

        let (coupled_seqs, ptrs) = cache
            .upload_stage2_meta_padded(
                &gpu,
                num_layers,
                &last_tokens,
                &positions,
                &plans,
                padded_b,
                m_rows,
                mask_id,
                |l, s| 0x1000 + (l as u64) * 0x100 + (s as u64) * 0x10,
                |l, s| 0x2000 + (l as u64) * 0x100 + (s as u64) * 0x10,
                Some(&specs),
                Some(&q_row_bases),
                0,
            )
            .unwrap();
        assert_eq!(coupled_seqs, 1);

        let slab = gpu.read_alloc(ptrs.base).unwrap();
        let read_u32 = |ptr: spark_runtime::gpu::DevicePtr, idx: usize| {
            let off = (ptr.0 - ptrs.base.0) as usize + idx * 4;
            u32::from_le_bytes(slab[off..off + 4].try_into().unwrap())
        };
        let read_f32 = |ptr: spark_runtime::gpu::DevicePtr, idx: usize| {
            let off = (ptr.0 - ptrs.base.0) as usize + idx * 4;
            f32::from_le_bytes(slab[off..off + 4].try_into().unwrap())
        };
        let read_u64 = |ptr: spark_runtime::gpu::DevicePtr, idx: usize| {
            let off = (ptr.0 - ptrs.base.0) as usize + idx * 8;
            u64::from_le_bytes(slab[off..off + 8].try_into().unwrap())
        };

        // Padded lane 3 must have `eff_ctx == 0` and `ring_start == 0` so
        // `inferspark_prefill_split_ctx_h128_batched` never reads `K_ctx`/`V_ctx`,
        // valid in-vocab anchor/tokens, small positions `0..m_rows`, temp == 0.0,
        // and its dedicated rejection sink `q_row_base`.
        assert_eq!(read_u32(ptrs.batch_eff_ctx, 3), 0);
        assert_eq!(read_u32(ptrs.batch_ring_start, 3), 0);
        assert_eq!(read_u32(ptrs.batch_anchor_ids, 3), last_tokens[0]);
        assert_eq!(read_u32(ptrs.draft_tokens_dev, 3 * m_rows), last_tokens[0]);
        for r in 1..m_rows {
            assert_eq!(read_u32(ptrs.draft_tokens_dev, 3 * m_rows + r), mask_id);
        }
        for r in 0..m_rows {
            assert_eq!(read_u32(ptrs.position_ids, 3 * m_rows + r), r as u32);
        }
        assert_eq!(read_f32(ptrs.batch_gumbel_temps, 3), 0.0);
        assert_eq!(read_u32(ptrs.q_row_bases, 3), 63 * 8);

        // `ctx_{k,v}_ptrs` stride must match `padded_b` across every layer.
        for l in 0..num_layers {
            for s in 0..padded_b {
                let src_s = if s < actual_b { s } else { 0 };
                assert_eq!(
                    read_u64(ptrs.batch_ctx_k_ptrs, l * padded_b + s),
                    0x1000 + (l as u64) * 0x100 + (src_s as u64) * 0x10
                );
                assert_eq!(
                    read_u64(ptrs.batch_ctx_v_ptrs, l * padded_b + s),
                    0x2000 + (l as u64) * 0x100 + (src_s as u64) * 0x10
                );
            }
        }
    }

    #[test]
    fn shared_nvfp4_lm_head_dispatches_coalesced_v2_for_bucketed_propose_shapes() {
        use crate::layers::qwen4_fast_proj::select_coalesced_nvfp4_tile_for_shape_v2;

        // `batched_propose.rs` invokes `run_final_norm_and_lm_head(..., LmHeadNvfp4Mode::CoalescedTc)`
        // -> `try_coalesced_nvfp4_w4a16_gemm`, which already dispatches the v2 2-CTA/SM
        // coalesced kernels (`k = 2560` is a multiple of 64, `n = 248320`) across all
        // bucketed `{2, 4, 8, 16}` DFlash2 propose shapes (`m_out = B * gamma`).
        let expected = [
            (14usize, "w4a16_gemm_coalesced_v2_n64_m16"),
            (28, "w4a16_gemm_coalesced_v2_n64_m32"),
            (56, "w4a16_gemm_coalesced_v2_n64_m64"),
            (64, "w4a16_gemm_coalesced_v2_n64_m64"),
            (80, "w4a16_gemm_coalesced_v2_n64_m80"),
            (112, "w4a16_gemm_coalesced_v2_n64_m128"),
        ];
        for (m_out, sym) in expected {
            let tile = select_coalesced_nvfp4_tile_for_shape_v2(m_out, 248_320, 2_560)
                .expect("bucketed DFlash2 LM head shape must select a coalesced v2 tile");
            assert_eq!(tile.dispatch_symbol(true), sym);
        }
    }
}
