// SPDX-License-Identifier: AGPL-3.0-only

//! Policy, routing, and ring/split-KV planning helpers for the DFlash noise pass.

use std::sync::OnceLock;

use anyhow::{Context, Result, ensure};
use spark_runtime::gpu::DevicePtr;

use super::graph_state::{DFLASH_NOISE_GRAPH_SELECTOR, DflashLayerQuantMode};
use super::lm_head_path::{LmHeadPath, resolve_lm_head_path};

/// Controls whether the shared untransposed NVFP4 LM head stays on the exact
/// scalar multi-row GEMV path (`Exact`, preserving 0-ULP C=1 T=0 greedy parity)
/// or dispatches through Tensor Core coalesced NVFP4 W4A16 GEMM (`CoalescedTc`,
/// used for C=1 T>0, C>=2 batched cohorts, and always when `rows > 32`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(in super::super) enum LmHeadNvfp4Mode {
    Exact,
    CoalescedTc,
}

/// SM count threshold above which `C=1` DFlash2 draft `lm_head` engages Tensor
/// Core coalesced NVFP4 (`CoalescedTc`) even in greedy mode (GB10 `<= 64` SMs
/// stays on `Exact` in greedy mode). Target verification strictly validates
/// every drafted token, so draft `lm_head` rounding never alters committed output.
const DFLASH_C1_TC_LM_HEAD_HIGH_SM_THRESHOLD: u32 = 64;

/// Resolve `C=1` DFlash2 draft `lm_head` NVFP4 mode for a given GPU SM count:
/// - Sampled (`sampled == true`): always [`LmHeadNvfp4Mode::CoalescedTc`].
/// - Greedy (`sampled == false`): [`LmHeadNvfp4Mode::CoalescedTc`] on high-SM
///   Blackwell GPUs (`sm_count > 64`, e.g. SM120/SM103); [`LmHeadNvfp4Mode::Exact`]
///   on GB10 (`sm_count <= 64`).
#[inline]
#[must_use]
pub(in super::super) const fn resolve_c1_lm_head_nvfp4_mode_for_sm(
    sampled: bool,
    sm_count: u32,
) -> LmHeadNvfp4Mode {
    if sampled || sm_count > DFLASH_C1_TC_LM_HEAD_HIGH_SM_THRESHOLD {
        LmHeadNvfp4Mode::CoalescedTc
    } else {
        LmHeadNvfp4Mode::Exact
    }
}

/// Resolve `C=1` DFlash2 draft `lm_head` NVFP4 mode for the active GPU.
#[inline]
#[must_use]
pub(in super::super) fn resolve_c1_lm_head_nvfp4_mode(sampled: bool) -> LmHeadNvfp4Mode {
    resolve_c1_lm_head_nvfp4_mode_for_sm(sampled, atlas_core::device::active_sm_count())
}

/// Whether `launch_lm_head_bf16_or_shared_nvfp4_exact` should try Tensor Core
/// coalesced NVFP4 W4A16 GEMM (`try_coalesced_nvfp4_w4a16_gemm`) before the
/// scalar `exact_kernels` / BF16 fallback.
///
/// - `LmHeadNvfp4Mode::Exact` keeps `rows <= 32` (including C=1 T=0 greedy) on
///   the 0-ULP exact scalar kernels, while still routing `rows > 32` (where
///   `exact_kernels` has no tier and previously fell back to 512 MB BF16 GEMM)
///   through coalesced NVFP4.
/// - `LmHeadNvfp4Mode::CoalescedTc` engages coalesced NVFP4 for all `rows >= 2`
///   (`rows == 1` remains on the single-row NVFP4 GEMV).
#[inline]
#[must_use]
pub(in super::super) const fn should_use_coalesced_nvfp4_lm_head(
    mode: LmHeadNvfp4Mode,
    rows: usize,
) -> bool {
    match mode {
        LmHeadNvfp4Mode::CoalescedTc => rows >= 2,
        LmHeadNvfp4Mode::Exact => rows > 32,
    }
}

/// Resolve `(shortlist_k, use_radix_topk, use_k32_selector)` for DFlash2
/// candidate selection.
///
/// - Keeps the checkpoint's trained `sel_top_k` (e.g. `16`) so the bilinear
///   Markov pair head never scores out-of-distribution tail candidates, while
///   still engaging the 23x faster 2-pass `topk_bf16_fast` kernel and the
///   `k32`-capable selector walk (which emits `batch_out_probs` for `p_cum`
///   confidence truncation on both greedy and CRN coupled Gumbel-Max paths).
#[inline]
#[must_use]
pub(in super::super) fn resolve_dflash2_selector_topk(
    sel_top_k: usize,
    lm_stride: usize,
    _widened_for_sampling: bool,
    has_topk_radix: bool,
    k32_selector_ready: bool,
) -> (usize, bool, bool) {
    let base_k = sel_top_k.min(lm_stride);
    (base_k, has_topk_radix, k32_selector_ready)
}

/// Per-slot row stride in the shared GPU `(q_cand_ids, q_probs)` buffer (`slot_idx * 8`).
pub(in super::super) const REJECTION_SLOT_STRIDE: usize = 8;

/// Resolve `(shortlist_k, use_radix_topk)` for maximal-coupling rejection
/// sampling (`dflash2_selector_walk_probabilistic_batched`), clamping
/// `rejection_shortlist_k()` to `1..=min(REJECTION_Q_STRIDE, shared_q_stride, max_topk, lm_stride)`.
#[inline]
#[must_use]
pub(in super::super) fn resolve_rejection_shortlist_k(
    lm_stride: usize,
    has_topk_radix: bool,
    shared_q_stride: usize,
) -> (usize, bool) {
    let max_topk = if has_topk_radix {
        super::super::DFLASH2_GUMBEL_SHORTLIST_K
    } else {
        super::super::DDTREE_TOP_K_MAX
    };
    let cap = crate::model::trait_impl::device_topk::REJECTION_Q_STRIDE
        .min(shared_q_stride)
        .min(max_topk)
        .min(lm_stride);
    let k = if cap == 0 {
        0
    } else {
        spark_runtime::sampler::gumbel::rejection_shortlist_k().clamp(1, cap)
    };
    (k, has_topk_radix)
}

/// Resolve per-sequence `(staged_spec, q_row_base, valid_q_record)` for
/// maximal-coupling rejection sampling:
/// - Scales active draft temperatures by `(spec.temperature * temp_scale).max(GUMBEL_MIN_TEMPERATURE)`.
/// - Validates `slot_idx` and `slot_idx * 8 + gamma_eff <= shared_q_row_cap` with `1 <= gamma_eff <= 8`.
/// - Returns `(None, 0, None)` whenever the sequence is greedy/inactive or the
///   slot/gamma bounds do not fit, so the kernel runs exact greedy (`temp = 0.0`)
///   and `last_rejection_q` is not marked valid.
#[must_use]
pub(in super::super) fn resolve_rejection_seq_staging(
    spec: Option<spark_runtime::sampler::gumbel::GumbelDraftSpec>,
    slot_idx: Option<usize>,
    gamma_eff: usize,
    shared_q_row_cap: usize,
    temp_scale: f32,
) -> (
    Option<spark_runtime::sampler::gumbel::GumbelDraftSpec>,
    u32,
    Option<(usize, u64, usize)>,
) {
    let Some(mut s) = spec.filter(|s| s.is_active()) else {
        return (None, 0, None);
    };
    let Some(slot) = slot_idx else {
        return (None, 0, None);
    };
    if gamma_eff == 0 || gamma_eff > REJECTION_SLOT_STRIDE {
        return (None, 0, None);
    }
    let Some(row_base) = slot.checked_mul(REJECTION_SLOT_STRIDE) else {
        return (None, 0, None);
    };
    let Some(row_end) = row_base.checked_add(gamma_eff) else {
        return (None, 0, None);
    };
    if row_end > shared_q_row_cap {
        return (None, 0, None);
    }
    let Ok(row_base_u32) = u32::try_from(row_base) else {
        return (None, 0, None);
    };
    let scale = if temp_scale.is_finite() && temp_scale > 0.0 {
        temp_scale
    } else {
        1.0
    };
    s.temperature =
        (s.temperature * scale).max(spark_runtime::sampler::gumbel::GUMBEL_MIN_TEMPERATURE);
    (
        Some(s),
        row_base_u32,
        Some((slot, s.step0_abs_pos, gamma_eff)),
    )
}

/// Validate `(num_rows, vocab, k)` against selector top-K scratch capacity
/// (`max_k` is `16` for legacy `topk_bf16` and `32` for `topk_bf16_radix`).
pub(in super::super) fn validate_selector_shortlist(
    num_rows: usize,
    scratch_rows: usize,
    vocab: usize,
    k: usize,
    max_k: usize,
) -> Result<(u32, u32, u32)> {
    ensure!(
        num_rows <= scratch_rows,
        "selector top-K rows {num_rows} exceed scratch capacity {scratch_rows}"
    );
    ensure!(vocab > 0, "selector top-K requires vocab > 0");
    let vocab_u32 = u32::try_from(vocab).context("selector top-K vocab exceeds u32")?;
    ensure!(k >= 1, "selector top-K requires k >= 1");
    ensure!(k <= max_k, "selector top-K requires k <= {max_k}, got {k}");
    ensure!(
        k <= vocab,
        "selector top-K requires k <= vocab ({vocab}), got {k}"
    );
    let rows_u32 = u32::try_from(num_rows).context("selector top-K rows exceed u32")?;
    Ok((rows_u32, vocab_u32, k as u32))
}

/// How Stage-2 split-context attention indexes `dstate.ctx_{k,v}_cache`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(in super::super) enum Stage2CtxAttnMode {
    /// Linear `[0..eff_ctx)` lookup via `inferspark_prefill_split_ctx_h{128,256}`.
    /// Preserves the exact `needed_start == 0` path bit-for-bit, and also serves
    /// non-SWA full-ring (`eff_ctx == ctx_window` or aligned `needed_start % ctx_window == 0`)
    /// models when no circular-ring kernel exists (e.g. `head_dim == 256`).
    Linear,
    /// Circular ring `[needed_start % ctx_window .. +eff_ctx)` lookup via
    /// `inferspark_prefill_split_ctx_h128_batched` (`batch_size = 1`), reading
    /// per-layer `ctx_{k,v}_cache` base pointers and `ring_start` from the head's
    /// fixed Stage-2 ring metadata slab. Exact chronological order for SWA,
    /// causal masking, partial rings (prefix-cache skips), and > 4K decode.
    CircularRingH128,
}

/// Resolve which Stage-2 split-context attention mode can safely execute for
/// `(head_dim, needed_start, eff_ctx, ctx_window, has_linear_kernel, has_ring_h128_kernel, any_layer_swa)`
/// on a GPU with `sm_count` streaming multiprocessors.
///
/// On high-SM GPUs (`sm_count > 64`, e.g. SM120), `CircularRingH128` is preferred
/// even when `needed_start == 0` so Stage-2 CUDA graphs read `ctx_{k,v}_cache`
/// pointers through the fixed ring metadata slab instead of baking per-request
/// pointers into `NoiseGraphRuntimeKey.ctx_kv_ptrs` (avoiding per-request graph
/// re-capture). On GB10 (`sm_count <= 64`), `needed_start == 0` preserves `Linear`
/// bit-for-bit.
#[must_use]
#[expect(
    clippy::too_many_arguments,
    reason = "pure Stage-2 attention mode geometry resolver mirroring resolve_stage2_ctx_attn_mode"
)]
pub(in super::super) fn resolve_stage2_ctx_attn_mode_for_sm(
    head_dim: usize,
    needed_start: usize,
    eff_ctx: usize,
    ctx_window: usize,
    has_linear_kernel: bool,
    has_ring_h128_kernel: bool,
    any_layer_swa: bool,
    sm_count: u32,
) -> Option<Stage2CtxAttnMode> {
    if ctx_window == 0 || eff_ctx > ctx_window || !matches!(head_dim, 128 | 256) {
        return None;
    }
    if needed_start == 0 {
        if sm_count > 64 && head_dim == 128 && has_ring_h128_kernel {
            return Some(Stage2CtxAttnMode::CircularRingH128);
        }
        if has_linear_kernel {
            return Some(Stage2CtxAttnMode::Linear);
        }
        if head_dim == 128 && has_ring_h128_kernel {
            return Some(Stage2CtxAttnMode::CircularRingH128);
        }
        return None;
    }
    if head_dim == 128 && has_ring_h128_kernel {
        return Some(Stage2CtxAttnMode::CircularRingH128);
    }
    if has_linear_kernel
        && (eff_ctx == 0
            || (!any_layer_swa
                && (eff_ctx == ctx_window || needed_start.is_multiple_of(ctx_window))))
    {
        return Some(Stage2CtxAttnMode::Linear);
    }
    None
}

/// Resolve which Stage-2 split-context attention mode can safely execute on the
/// default (<= 64 SM) path.
#[cfg_attr(not(test), allow(dead_code))]
#[must_use]
pub(in super::super) fn resolve_stage2_ctx_attn_mode(
    head_dim: usize,
    needed_start: usize,
    eff_ctx: usize,
    ctx_window: usize,
    has_linear_kernel: bool,
    has_ring_h128_kernel: bool,
    any_layer_swa: bool,
) -> Option<Stage2CtxAttnMode> {
    resolve_stage2_ctx_attn_mode_for_sm(
        head_dim,
        needed_start,
        eff_ctx,
        ctx_window,
        has_linear_kernel,
        has_ring_h128_kernel,
        any_layer_swa,
        0,
    )
}

/// One contiguous physical write span for Stage-1 incremental circular KV append.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(in super::super) struct Stage1RingSpan {
    /// Row index into `scratch.fc_proj` and `scratch.position_ids`.
    pub linear_row: usize,
    /// Physical slot index into `dstate.ctx_{k,v}_cache[l]` (`abs_pos % window`).
    pub physical_slot: usize,
    /// Number of contiguous rows in this span.
    pub count: usize,
}

/// Pure per-layer plan for Stage-1 incremental post-`k_norm`/post-RoPE circular
/// KV append over `[needed_start .. needed_start + eff_ctx)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in super::super) struct Stage1RingAppendPlan {
    /// Leading rows `[needed_start .. needed_start + old_ctx_count)` already
    /// valid in both `ctx_k_cache` and `ctx_v_cache`.
    pub old_ctx_count: usize,
    /// First absolute position to project (`needed_start + old_ctx_count`).
    pub new_abs_start: usize,
    /// Number of new rows to project, `k_norm`, RoPE, and write (`eff_ctx - old_ctx_count`).
    pub new_ctx_count: usize,
    spans: [Stage1RingSpan; 2],
    span_len: usize,
}

impl Stage1RingAppendPlan {
    #[inline]
    pub(in super::super) fn spans(&self) -> &[Stage1RingSpan] {
        &self.spans[..self.span_len]
    }
}

/// Plan one layer's Stage-1 circular `PostRope` KV append over
/// `[needed_start .. needed_start + eff_ctx)`.
///
/// Reuses the cached prefix only when BOTH K and V cover `[needed_start .. valid_end)`
/// without exceeding `window` and without `cache_end > needed_end` (rollback).
/// Splits newly appended rows into at most two contiguous physical ring spans
/// when `(new_abs_start % window) + new_ctx_count > window`.
pub(in super::super) fn plan_stage1_ring_append(
    needed_start: usize,
    eff_ctx: usize,
    window: usize,
    cache_k_start: usize,
    cache_k_end: usize,
    cache_v_start: usize,
    cache_v_end: usize,
) -> Result<Stage1RingAppendPlan> {
    ensure!(
        window > 0 && eff_ctx <= window,
        "DFlash Stage-1 ring append invalid geometry: eff_ctx={eff_ctx} window={window}"
    );
    let needed_end = needed_start
        .checked_add(eff_ctx)
        .context("DFlash Stage-1 ring append needed_end overflow")?;
    if eff_ctx == 0 {
        return Ok(Stage1RingAppendPlan {
            old_ctx_count: 0,
            new_abs_start: needed_end,
            new_ctx_count: 0,
            spans: [Stage1RingSpan::default(); 2],
            span_len: 0,
        });
    }
    let k_range_ok = cache_k_start <= cache_k_end && cache_k_end - cache_k_start <= window;
    let v_range_ok = cache_v_start <= cache_v_end && cache_v_end - cache_v_start <= window;
    let valid_start = cache_k_start.max(cache_v_start);
    let valid_end = cache_k_end.min(cache_v_end);
    let no_rollback = cache_k_end <= needed_end && cache_v_end <= needed_end;
    let old_ctx_count = if k_range_ok
        && v_range_ok
        && no_rollback
        && valid_start <= needed_start
        && valid_end > needed_start
    {
        valid_end - needed_start
    } else {
        0
    };
    let new_abs_start = needed_start + old_ctx_count;
    let new_ctx_count = eff_ctx - old_ctx_count;
    if new_ctx_count == 0 {
        return Ok(Stage1RingAppendPlan {
            old_ctx_count,
            new_abs_start,
            new_ctx_count: 0,
            spans: [Stage1RingSpan::default(); 2],
            span_len: 0,
        });
    }
    let phys_start = new_abs_start % window;
    let first_count = new_ctx_count.min(window - phys_start);
    let second_count = new_ctx_count - first_count;
    let spans = [
        Stage1RingSpan {
            linear_row: old_ctx_count,
            physical_slot: phys_start,
            count: first_count,
        },
        Stage1RingSpan {
            linear_row: old_ctx_count + first_count,
            physical_slot: 0,
            count: second_count,
        },
    ];
    let span_len = if second_count == 0 { 1 } else { 2 };
    Ok(Stage1RingAppendPlan {
        old_ctx_count,
        new_abs_start,
        new_ctx_count,
        spans,
        span_len,
    })
}

/// Return the largest leading prefix count `skip <= old_fc_count` of
/// `[needed_start .. needed_start + old_fc_count)` that EVERY layer's circular
/// `PostRope` K/V cache already covers without rollback or range errors.
///
/// Rows `[0..skip)` of `scratch.fc_proj` are never read by Stage 1, allowing
/// `forward_block` to skip copying `[needed_start .. needed_start + skip)` from
/// `ctx_fc_cache` even when `ATLAS_DFLASH_EARLY_STAGE1=1` pre-populated the
/// trailing `[skip .. old_fc_count)` rows in `ctx_fc_cache`.
#[allow(clippy::too_many_arguments)]
#[must_use]
pub(in super::super) fn min_layers_covered_old_fc_prefix(
    needed_start: usize,
    eff_ctx: usize,
    old_fc_count: usize,
    window: usize,
    cache_k_start: &[usize],
    cache_k_end: &[usize],
    cache_v_start: &[usize],
    cache_v_end: &[usize],
) -> usize {
    let num_layers = cache_k_start.len();
    if num_layers == 0
        || cache_k_end.len() != num_layers
        || cache_v_start.len() != num_layers
        || cache_v_end.len() != num_layers
        || old_fc_count == 0
        || old_fc_count > eff_ctx
    {
        return 0;
    }
    let mut min_covered = old_fc_count;
    for l in 0..num_layers {
        let Ok(plan) = plan_stage1_ring_append(
            needed_start,
            eff_ctx,
            window,
            cache_k_start[l],
            cache_k_end[l],
            cache_v_start[l],
            cache_v_end[l],
        ) else {
            return 0;
        };
        min_covered = min_covered.min(plan.old_ctx_count);
        if min_covered == 0 {
            return 0;
        }
    }
    min_covered
}

/// Whether every layer's circular `PostRope` K/V cache already covers at least
/// `[needed_start .. needed_start + old_fc_count)` without rollback or range
/// errors, guaranteeing `scratch.fc_proj[0..old_fc_count)` is never read by Stage 1.
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
#[must_use]
pub(in super::super) fn all_layers_cover_old_fc_prefix(
    needed_start: usize,
    eff_ctx: usize,
    old_fc_count: usize,
    window: usize,
    cache_k_start: &[usize],
    cache_k_end: &[usize],
    cache_v_start: &[usize],
    cache_v_end: &[usize],
) -> bool {
    if old_fc_count == 0 {
        let num_layers = cache_k_start.len();
        return num_layers > 0
            && cache_k_end.len() == num_layers
            && cache_v_start.len() == num_layers
            && cache_v_end.len() == num_layers;
    }
    min_layers_covered_old_fc_prefix(
        needed_start,
        eff_ctx,
        old_fc_count,
        window,
        cache_k_start,
        cache_k_end,
        cache_v_start,
        cache_v_end,
    ) == old_fc_count
}

/// Minimum `eff_ctx` for wide-SM DFlash2 Split-KV H=128 attention.
/// Below 256 rows (`<= 8` KV tiles), non-split single-launch attention avoids
/// reduce-pass overhead.
pub(in super::super) const DFLASH_SPLIT_KV_H128_MIN_EFF_CTX: usize = 256;
/// SM-count threshold above which DFlash2 Split-KV H=128 attention engages
/// (`<= 64` keeps 48-SM GB10 on the single-launch kernel).
pub(in super::super) const DFLASH_SPLIT_KV_H128_HIGH_SM_THRESHOLD: u32 = 64;
/// Minimum Split-KV partition count when Split-KV engages (`4..=8`).
pub(in super::super) const DFLASH_SPLIT_KV_H128_MIN_SPLITS: u32 = 4;
/// Maximum Split-KV partition count (`4..=8`, bounds per-head workspace to ~2 MiB).
pub(in super::super) const DFLASH_SPLIT_KV_H128_MAX_SPLITS: u32 = 8;
pub(in super::super) const DFLASH_SPLIT_KV_H128_LINEAR_KERNEL: &str =
    "inferspark_prefill_split_ctx_h128_splitkv";
pub(in super::super) const DFLASH_SPLIT_KV_H128_BATCHED_KERNEL: &str =
    "inferspark_prefill_split_ctx_h128_batched_splitkv";
pub(in super::super) const DFLASH_SPLIT_KV_H128_REDUCE_KERNEL: &str =
    "inferspark_prefill_split_ctx_h128_splitkv_reduce";

/// Compute Split-KV partition count for DFlash2 H=128 attention.
///
/// Engages `num_splits in 4..=8` only when `head_dim == 128`, `sm_count > 64`,
/// `eff_ctx >= 256`, and `num_q_heads * batch_size < sm_count` (lifting `C=1..4`
/// on 188-SM GPUs from 32–128 CTAs to 192–512 CTAs). Returns `1` otherwise.
#[inline]
#[must_use]
pub(in super::super) fn resolve_dflash_split_kv_splits_for_sm(
    head_dim: usize,
    eff_ctx: usize,
    num_q_heads: usize,
    batch_size: usize,
    sm_count: u32,
) -> u32 {
    if head_dim != 128
        || sm_count <= DFLASH_SPLIT_KV_H128_HIGH_SM_THRESHOLD
        || eff_ctx < DFLASH_SPLIT_KV_H128_MIN_EFF_CTX
        || num_q_heads == 0
        || batch_size == 0
    {
        return 1;
    }
    let base_ctas = num_q_heads.saturating_mul(batch_size);
    if base_ctas >= sm_count as usize {
        return 1;
    }
    (sm_count as usize).div_ceil(base_ctas).clamp(
        DFLASH_SPLIT_KV_H128_MIN_SPLITS as usize,
        DFLASH_SPLIT_KV_H128_MAX_SPLITS as usize,
    ) as u32
}

/// Workspace bytes required for Split-KV H=128 partial FP32 outputs and `(m, l)` metadata:
/// `batch_size * num_splits * num_q_heads * m_rows * (128 + 2) * sizeof(f32)`.
#[inline]
#[must_use]
pub(in super::super) fn dflash_split_kv_h128_workspace_bytes(
    batch_size: usize,
    num_splits: u32,
    num_q_heads: usize,
    m_rows: usize,
) -> Option<usize> {
    batch_size
        .checked_mul(num_splits as usize)?
        .checked_mul(num_q_heads)?
        .checked_mul(m_rows)?
        .checked_mul(128 + 2)?
        .checked_mul(std::mem::size_of::<f32>())
}

/// Logical 2-stage DFlash noise-block CUDA graph key.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DflashNoiseGraphKey {
    pub gamma_eff: usize,
    pub quant_mode: DflashLayerQuantMode,
    pub markov_active: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(in super::super) struct NoiseGraphRuntimeKey {
    pub(in super::super) logical: DflashNoiseGraphKey,
    pub(in super::super) attn_mode: Stage2CtxAttnMode,
    pub(in super::super) split_kv_splits: u8,
    pub(in super::super) lm_head_path_tag: u8,
    pub(in super::super) prefer_tc_lm_head: bool,
    pub(in super::super) rejection_mode: bool,
    pub(in super::super) ngram_mode: bool,
    pub(in super::super) shortlist_k: u8,
    pub(in super::super) mask_id: u32,
    pub(in super::super) stream: u64,
    pub(in super::super) ctx_kv_ptrs: Vec<(u64, u64)>,
}

pub(in super::super) const LM_HEAD_PATH_TAG_BF16: u8 = 2;

pub(in super::super) fn lm_head_path_tag(path: LmHeadPath) -> u8 {
    match path {
        LmHeadPath::Nvfp4 => 0,
        LmHeadPath::Fp8 => 1,
        LmHeadPath::Bf16 => LM_HEAD_PATH_TAG_BF16,
    }
}

/// Whether a dense BF16 projection `[M, K] @ [N, K]^T -> [M, N]` satisfies
/// cuBLASLt's non-null 16-byte BF16 alignment requirements (`N % 8 == 0`,
/// `K % 8 == 0`, and 16-byte aligned operand/output device pointers).
#[inline]
#[must_use]
pub(in super::super) fn bf16_proj_cublas_aligned(
    act: DevicePtr,
    weight: DevicePtr,
    out: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
) -> bool {
    m > 0
        && n > 0
        && k > 0
        && n.is_multiple_of(8)
        && k.is_multiple_of(8)
        && !act.is_null()
        && !weight.is_null()
        && !out.is_null()
        && act.0.is_multiple_of(16)
        && weight.0.is_multiple_of(16)
        && out.0.is_multiple_of(16)
}

pub(in super::super) fn resolve_lm_head_path_tag(
    nvfp4_requested: bool,
    nvfp4_weight_present: bool,
    nvfp4_kernel_present: bool,
    fp8_weight_present: bool,
) -> Result<u8> {
    let path = resolve_lm_head_path(
        nvfp4_requested,
        nvfp4_weight_present,
        nvfp4_kernel_present,
        fp8_weight_present,
    )
    .map_err(anyhow::Error::msg)?;
    Ok(lm_head_path_tag(path))
}

/// Representation currently stored in a sequence's `dstate.ctx_k_cache`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CtxKvCacheRepr {
    /// Eager `forward_block_layer` format: pre-`k_norm`, pre-RoPE ring buffer.
    PreRope,
    /// Stage-1 / batched DFlash2 format: circular `abs_pos % ctx_window`
    /// post-`k_norm`, post-RoPE ring buffer.
    PostRope,
}

/// Why a sequence's `ctx_{k,v}_cache` range counters must be invalidated before
/// reading cached entries.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CtxCacheInvalidationReason {
    None,
    RingWrapAround,
    ReprTransition {
        from: CtxKvCacheRepr,
        to: CtxKvCacheRepr,
    },
}

/// Pure transition planner for per-sequence `ctx_k_cache` representation and
/// invalidation.
///
/// Circular `PostRope` caches stay hot across the `ctx_window` wrap boundary
/// whenever `want_post_rope` remains true. Whenever a sequence transitions
/// between `PostRope` and eager `PreRope` (e.g. a non-ring-capable model
/// wrapping past `ctx_window` or an eager fallback), any non-empty cached rows
/// MUST be invalidated so eager `forward_block_layer` never applies `k_norm`
/// and RoPE twice and Stage 1 never mixes pre-RoPE rows into `PostRope` slots.
pub(crate) fn plan_ctx_kv_cache_transition(
    prev_repr: Option<CtxKvCacheRepr>,
    want_post_rope: bool,
    needed_start: usize,
    cache_end: usize,
) -> (CtxKvCacheRepr, CtxCacheInvalidationReason) {
    let target = if want_post_rope {
        CtxKvCacheRepr::PostRope
    } else {
        CtxKvCacheRepr::PreRope
    };
    let reason = match prev_repr {
        Some(from) if from != target && cache_end > 0 => {
            if from == CtxKvCacheRepr::PostRope && needed_start > 0 {
                CtxCacheInvalidationReason::RingWrapAround
            } else {
                CtxCacheInvalidationReason::ReprTransition { from, to: target }
            }
        }
        _ => CtxCacheInvalidationReason::None,
    };
    (target, reason)
}

/// SM-count threshold above which `ATLAS_DFLASH_NOISE_GRAPH` auto-enables when
/// unset (`<= 64` keeps 48-SM GB10 off by default). Allowed by release-flat
/// contract (`cli_dflash_environment_rules.rs`: `ZERO_OR_ONE`).
const AUTO_DFLASH_NOISE_GRAPH_SM_THRESHOLD: u32 = 64;

/// Strict parser for `ATLAS_DFLASH_NOISE_GRAPH` with SM-aware default.
///
/// Explicit `"0"` -> `Ok(false)`, `"1"` -> `Ok(true)`, any other value returns
/// an error (fail-closed off). When unset (`None`), auto-enables on wide GPUs
/// (`sm_count > 64`) and stays off on `<= 64`-SM GPUs (GB10).
pub fn parse_dflash_noise_graph_env_for_sm(raw: Option<&str>, sm_count: u32) -> Result<bool> {
    match raw.map(str::trim) {
        None => Ok(sm_count > AUTO_DFLASH_NOISE_GRAPH_SM_THRESHOLD),
        Some("0") => Ok(false),
        Some("1") => Ok(true),
        Some(other) => {
            anyhow::bail!(
                "invalid {DFLASH_NOISE_GRAPH_SELECTOR}={other:?}: expected \"0\" or \"1\""
            )
        }
    }
}

/// Strict parser for `ATLAS_DFLASH_NOISE_GRAPH`. Unset or `"0"` -> `Ok(false)`;
/// `"1"` -> `Ok(true)`; any other value returns an error (fail-closed off).
#[cfg(test)]
pub fn parse_dflash_noise_graph_env(raw: Option<&str>) -> Result<bool> {
    parse_dflash_noise_graph_env_for_sm(raw, 0)
}

pub(in super::super) fn dflash_noise_graph_enabled() -> bool {
    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED.get_or_init(|| {
        let raw = std::env::var(DFLASH_NOISE_GRAPH_SELECTOR).ok();
        let sm_count = atlas_core::device::active_sm_count();
        match parse_dflash_noise_graph_env_for_sm(raw.as_deref(), sm_count) {
            Ok(on) => {
                if on {
                    let source = if raw.is_some() { "env" } else { "auto_wide_sm" };
                    tracing::info!(
                        "DFLASH_NOISE_GRAPH_ENGAGED selector={DFLASH_NOISE_GRAPH_SELECTOR}=1 \
                         source={source} sm_count={sm_count}"
                    );
                }
                on
            }
            Err(err) => {
                tracing::warn!(
                    "DFLASH_NOISE_GRAPH_REFUSED reason=invalid_selector ({err}); falling back to eager"
                );
                false
            }
        }
    })
}

pub(in super::super) fn noise_graph_swa_enabled() -> bool {
    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED.get_or_init(|| {
        let raw = std::env::var("ATLAS_DFLASH_SWA").ok();
        !matches!(
            raw.as_deref(),
            Some("0" | "false" | "FALSE" | "off" | "OFF")
        )
    })
}

pub(in super::super) fn noise_graph_attn_kgamma_disable_o() -> bool {
    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED.get_or_init(|| {
        std::env::var("ATLAS_DFLASH_ATTN_KGAMMA_DISABLE_O")
            .ok()
            .as_deref()
            == Some("1")
    })
}

/// Environment selector for C=1 DFlash2 Stage-1 CUDA graph capture
/// (`5x {k_proj, v_proj, k_norm, rope}` bucketed by `new_ctx_count in 1..=8`).
/// Default ON unless `"0"` / `"off"` / `"false"`.
const DFLASH_STAGE1_GRAPH_ENV: &str = "ATLAS_DFLASH_STAGE1_GRAPH";
pub(in super::super) const STAGE1_GRAPH_MAX_ROWS: usize = 8;

#[must_use]
pub(in super::super) fn parse_dflash_stage1_graph_enabled(raw: Option<&str>) -> bool {
    !matches!(
        raw.map(str::trim),
        Some(v) if v == "0" || v.eq_ignore_ascii_case("off") || v.eq_ignore_ascii_case("false")
    )
}

#[must_use]
pub(in super::super) fn dflash_stage1_graph_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        parse_dflash_stage1_graph_enabled(std::env::var(DFLASH_STAGE1_GRAPH_ENV).ok().as_deref())
    })
}

/// Whether stochastic draft sampling runs inside the noise pass itself and
/// therefore requires bypassing the deterministic Stage-2 CUDA graph.
///
/// Greedy drafting paired with target Gumbel verification (`temperature > 0.0`
/// on the target request while `draft_sample_method == Greedy`) keeps the
/// noise pass completely deterministic, so Stage-2 CUDA graph capture/replay
/// remains valid.
#[inline]
#[must_use]
pub(in super::super) fn noise_graph_stochastic_draft_sampling_active(
    draft_sample_method: crate::dflash_sampling::DflashDraftSampleMethod,
    active_sampling_request: Option<&crate::dflash_sampling::DflashDraftSamplingRequest>,
) -> bool {
    draft_sample_method == crate::dflash_sampling::DflashDraftSampleMethod::Probabilistic
        && active_sampling_request.is_some_and(|request| request.request_sampler.temperature > 0.0)
}
