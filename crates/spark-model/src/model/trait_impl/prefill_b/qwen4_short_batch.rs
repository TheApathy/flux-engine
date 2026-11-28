// SPDX-License-Identifier: AGPL-3.0-only

//! Batched short-prefill orchestrator for Qwen4 (`2..=16` single-chunk prompts
//! sharing ONE unified MoE weight pass per layer).
//!
//! For concurrent short prompts (`chunk_start == 0`, `is_last_chunk == true`,
//! `chunk_len >= 2`, `sum(chunk_len) <= min(max_chunk_rows, max_batch_tokens, 2048)`),
//! running `N` sequential prefills streams the 512 NVFP4 routed experts `N`
//! times per layer. Packing all `N` prompts contiguously into `hidden` and
//! `residual` lets each layer run its per-prompt SSM/attention mixer under
//! [`ExpertMajorSliceGuard`] (skipping MoE) followed by **one**
//! [`TransformerLayer::qwen4_prefill_finish_moe`] pass across all `total_rows`,
//! then one batched terminal mix + norm + LM head over the `N` last-token rows.
//!
//! Controlled by `ATLAS_QWEN4_PREFILL_SHORT_BATCH` (default **ON**; set `=0`
//! to disable and fall back to sequential `prefill_chunk_dispatch`).

use std::sync::Once;
use std::sync::atomic::Ordering;

use anyhow::{Context, Result, ensure};
use atlas_core::config::DflashCaptureMode;
use spark_runtime::gpu::{DevicePtr, GpuBackend, HostToDeviceCopy};
use spark_runtime::kv_cache::PagedKvCache;

use crate::layer::{AttnMetadataDev, ForwardContext};
use crate::layers::qwen4_prefill_moe::ExpertMajorSliceGuard;
use crate::model::block_mgmt::{cache_acquires_disk_refs, ensure_blocks_through_prefill};
use crate::model::trait_impl::verify_qwen4_batched::{moe_routing_scratch_bytes, ranges_disjoint};
use crate::model::types::TransformerModel;
use crate::traits::{PrefillSlice, Qwen4PrefillLineage, SequenceState};

/// Runtime kill-switch for Qwen4 batched short-prefill (default ON; `0` disables).
pub(crate) const SHORT_BATCH_ENV: &str = "ATLAS_QWEN4_PREFILL_SHORT_BATCH";

/// Minimum concurrent prompts required to amortize the unified MoE pass.
pub(crate) const MIN_SHORT_BATCH_STREAMS: usize = 2;

/// Maximum concurrent prompts admitted into one batched short-prefill round.
pub(crate) const MAX_SHORT_BATCH_STREAMS: usize = 16;

/// Hard ceiling on total packed prompt rows across all streams in one round.
pub(crate) const MAX_SHORT_BATCH_TOTAL_ROWS: usize = 2048;

/// Alignment (bytes) separating MoE routing scratch and per-slice attention metadata.
const ATTN_META_ALIGN: usize = 256;

/// Parse `ATLAS_QWEN4_PREFILL_SHORT_BATCH`: default ON, explicit `"0"` disables.
#[must_use]
pub(super) fn parse_short_batch_enabled(raw: Option<&str>) -> bool {
    raw.map(str::trim) != Some("0")
}

/// Resolve Qwen4 batched short-prefill under `det_batch`:
/// `DeterministicBatchPolicy::Enabled` disables multi-prompt short-batch
/// packing so every prompt takes the canonical single-sequence prefill path.
#[inline]
#[must_use]
pub(super) const fn short_batch_enabled_with_policy(
    det_batch: crate::layers::deterministic_batch::DeterministicBatchPolicy,
    env_enabled: bool,
) -> bool {
    !det_batch.is_enabled() && env_enabled
}

#[must_use]
fn short_batch_enabled() -> bool {
    parse_short_batch_enabled(std::env::var(SHORT_BATCH_ENV).ok().as_deref())
}

/// Pure host-side summary of one [`PrefillSlice`] for eligibility gating.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct ShortBatchSliceSummary {
    pub prompt_len: usize,
    pub chunk_start: usize,
    pub chunk_len: usize,
    pub is_last_chunk: bool,
    pub seq_len: usize,
    pub tokens_empty: bool,
    pub block_table_empty: bool,
    pub chunked_meta_none: bool,
    pub lineage_unset: bool,
    pub rotary_identity: bool,
    pub has_vision: bool,
    pub has_layer_states: bool,
    pub slot_idx: usize,
}

/// Pure host-side model/runtime invariants required for batched short-prefill.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct ShortBatchGateInputs {
    pub env_enabled: bool,
    pub is_qwen4_exp: bool,
    pub fp32_residual: bool,
    pub fp32_logits: bool,
    pub single_rank: bool,
    pub pp_world_size: usize,
    pub moe_selected: bool,
    pub family_suppressed: bool,
    pub dflash_supported: bool,
    pub has_final_mixer: bool,
    pub max_chunk_rows: usize,
    pub max_batch_tokens: usize,
}

/// Pure eligibility predicate returning `Some(total_rows)` iff the batch meets
/// every Qwen4 short-prefill invariant before any GPU or sequence state changes.
#[must_use]
pub(super) fn check_short_batch_eligible(
    inputs: ShortBatchGateInputs,
    slices: &[ShortBatchSliceSummary],
) -> Option<usize> {
    if !inputs.env_enabled
        || !inputs.is_qwen4_exp
        || inputs.fp32_residual
        || inputs.fp32_logits
        || !inputs.single_rank
        || inputs.pp_world_size > 1
        || !inputs.moe_selected
        || inputs.family_suppressed
        || !inputs.dflash_supported
        || !inputs.has_final_mixer
    {
        return None;
    }
    if !(MIN_SHORT_BATCH_STREAMS..=MAX_SHORT_BATCH_STREAMS).contains(&slices.len()) {
        return None;
    }
    let row_cap = inputs
        .max_chunk_rows
        .min(inputs.max_batch_tokens)
        .min(MAX_SHORT_BATCH_TOTAL_ROWS);
    let mut total_rows = 0usize;
    for (idx, slice) in slices.iter().enumerate() {
        if slice.chunk_start != 0
            || !slice.is_last_chunk
            || slice.chunk_len < 2
            || slice.chunk_len != slice.prompt_len
            || slice.seq_len != 0
            || !slice.tokens_empty
            || !slice.block_table_empty
            || !slice.chunked_meta_none
            || !slice.lineage_unset
            || !slice.rotary_identity
            || slice.has_vision
            || !slice.has_layer_states
        {
            return None;
        }
        if slices[..idx]
            .iter()
            .any(|prior| prior.slot_idx == slice.slot_idx)
        {
            return None;
        }
        total_rows = total_rows.checked_add(slice.chunk_len)?;
    }
    (total_rows <= row_cap).then_some(total_rows)
}

/// Per-slice byte offsets inside the shared hidden/residual/scratch/pinned arenas.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct ShortBatchSliceLayout {
    pub row_offset: usize,
    pub len: usize,
    pub meta_offset_bytes: usize,
    pub slot_offset_bytes: usize,
    pub meta_bytes: usize,
    pub pinned_offset_bytes: usize,
    pub pinned_bytes: usize,
}

/// Arena capacity parameters passed to [`ShortBatchLayout::compute`].
#[derive(Clone, Copy, Debug)]
pub(super) struct ShortBatchLayoutParams {
    pub residual_width: usize,
    pub top_k: usize,
    pub use_mrope: bool,
    pub vocab_size: usize,
    pub arena_hidden_bytes: usize,
    pub arena_residual_bytes: usize,
    pub arena_scratch_bytes: usize,
    pub arena_logits_bytes: usize,
}

/// Verified non-aliasing buffer layout for one batched short-prefill round.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct ShortBatchLayout {
    pub total_rows: usize,
    pub row_bytes: usize,
    pub hidden_total_bytes: usize,
    pub moe_routing_scratch_bytes: usize,
    pub meta_region_start_bytes: usize,
    pub meta_region_end_bytes: usize,
    pub total_pinned_bytes: usize,
    pub vocab_row_bytes: usize,
    pub logits_total_bytes: usize,
    pub slices: Vec<ShortBatchSliceLayout>,
}

impl ShortBatchLayout {
    /// Compute and bounds-check disjoint offsets across all `N` prompts.
    pub(super) fn compute(lengths: &[usize], params: ShortBatchLayoutParams) -> Result<Self> {
        ensure!(
            (MIN_SHORT_BATCH_STREAMS..=MAX_SHORT_BATCH_STREAMS).contains(&lengths.len()),
            "Qwen4 short-batch requires {MIN_SHORT_BATCH_STREAMS}..={MAX_SHORT_BATCH_STREAMS} streams, got {}",
            lengths.len()
        );
        let row_bytes = params
            .residual_width
            .checked_mul(2)
            .context("short-batch residual row byte overflow")?;
        ensure!(row_bytes > 0, "short-batch residual width must be non-zero");

        let mut total_rows = 0usize;
        for &len in lengths {
            ensure!(len >= 2, "short-batch slice length {len} must be >= 2");
            total_rows = total_rows
                .checked_add(len)
                .context("short-batch total rows overflow")?;
        }
        ensure!(
            total_rows <= MAX_SHORT_BATCH_TOTAL_ROWS,
            "short-batch total_rows={total_rows} exceeds cap {MAX_SHORT_BATCH_TOTAL_ROWS}"
        );

        let hidden_total_bytes = total_rows
            .checked_mul(row_bytes)
            .context("short-batch hidden byte size overflow")?;
        ensure!(
            hidden_total_bytes <= params.arena_hidden_bytes
                && hidden_total_bytes <= params.arena_residual_bytes,
            "short-batch hidden/residual ({hidden_total_bytes} B) exceeds arena (hidden={}, residual={})",
            params.arena_hidden_bytes,
            params.arena_residual_bytes
        );

        // `ffn.forward_prefill` writes `[0 .. 2 * total_rows * top_k * 4)` of
        // `buffers.scratch()`. Place every slice's attention metadata strictly
        // after that routing prefix on 256-byte boundaries.
        let moe_routing_scratch_bytes = moe_routing_scratch_bytes(total_rows, params.top_k)
            .context("short-batch MoE routing scratch overflow")?;
        let meta_region_start_bytes = moe_routing_scratch_bytes
            .checked_next_multiple_of(ATTN_META_ALIGN)
            .context("short-batch metadata start alignment overflow")?;

        let pos_streams = if params.use_mrope { 3usize } else { 1usize };
        let mut row_cursor = 0usize;
        let mut meta_cursor = meta_region_start_bytes;
        let mut pinned_cursor = 0usize;
        let mut slices = Vec::with_capacity(lengths.len());

        for &len in lengths {
            let pos_bytes = len
                .checked_mul(4)
                .and_then(|b| b.checked_mul(pos_streams))
                .context("short-batch position metadata overflow")?;
            let slot_offset_bytes = pos_bytes
                .checked_next_multiple_of(8)
                .context("short-batch slot offset overflow")?;
            let slot_bytes = len
                .checked_mul(8)
                .context("short-batch slot metadata overflow")?;
            let meta_bytes = slot_offset_bytes
                .checked_add(slot_bytes)
                .context("short-batch slice metadata total overflow")?;
            let pinned_offset_bytes = pinned_cursor
                .checked_next_multiple_of(8)
                .context("short-batch pinned offset overflow")?;
            pinned_cursor = pinned_offset_bytes
                .checked_add(pos_bytes)
                .context("short-batch pinned total overflow")?;

            slices.push(ShortBatchSliceLayout {
                row_offset: row_cursor,
                len,
                meta_offset_bytes: meta_cursor,
                slot_offset_bytes,
                meta_bytes,
                pinned_offset_bytes,
                pinned_bytes: pos_bytes,
            });

            row_cursor = row_cursor
                .checked_add(len)
                .context("short-batch row cursor overflow")?;
            meta_cursor = meta_cursor
                .checked_add(meta_bytes)
                .and_then(|end| end.checked_next_multiple_of(ATTN_META_ALIGN))
                .context("short-batch metadata cursor overflow")?;
        }
        let meta_region_end_bytes = meta_cursor;
        ensure!(
            meta_region_end_bytes <= params.arena_scratch_bytes,
            "short-batch scratch footprint ({meta_region_end_bytes} B = moe {moe_routing_scratch_bytes} B + meta) exceeds scratch arena ({} B)",
            params.arena_scratch_bytes
        );

        let vocab_row_bytes = params
            .vocab_size
            .checked_mul(2)
            .context("short-batch vocab row byte size overflow")?;
        let logits_total_bytes = lengths
            .len()
            .checked_mul(vocab_row_bytes)
            .context("short-batch logits total bytes overflow")?;
        ensure!(
            logits_total_bytes <= params.arena_logits_bytes,
            "short-batch logits ({logits_total_bytes} B) exceed logits arena ({} B)",
            params.arena_logits_bytes
        );

        Ok(Self {
            total_rows,
            row_bytes,
            hidden_total_bytes,
            moe_routing_scratch_bytes,
            meta_region_start_bytes,
            meta_region_end_bytes,
            total_pinned_bytes: pinned_cursor,
            vocab_row_bytes,
            logits_total_bytes,
            slices,
        })
    }

    /// Prove that every live device slice in this batch is pairwise disjoint.
    pub(super) fn verify_device_non_aliasing(
        &self,
        hidden_base: DevicePtr,
        residual_base: DevicePtr,
        scratch_base: DevicePtr,
        logits_base: DevicePtr,
    ) -> Result<()> {
        ensure!(
            !hidden_base.is_null()
                && !residual_base.is_null()
                && !scratch_base.is_null()
                && !logits_base.is_null(),
            "short-batch requires non-null arena base pointers"
        );
        ensure!(
            self.meta_region_start_bytes >= self.moe_routing_scratch_bytes
                && self.meta_region_end_bytes >= self.meta_region_start_bytes,
            "short-batch metadata region [{}..{}) violates MoE routing scratch prefix {}",
            self.meta_region_start_bytes,
            self.meta_region_end_bytes,
            self.moe_routing_scratch_bytes
        );
        ensure!(
            self.logits_total_bytes == self.slices.len() * self.vocab_row_bytes,
            "short-batch logits total bytes {} disagrees with {} slices * {} B",
            self.logits_total_bytes,
            self.slices.len(),
            self.vocab_row_bytes
        );
        ensure!(
            ranges_disjoint(
                hidden_base.0,
                self.hidden_total_bytes,
                residual_base.0,
                self.hidden_total_bytes,
            ),
            "short-batch hidden and residual arenas overlap"
        );
        for (i, si) in self.slices.iter().enumerate() {
            ensure!(
                si.meta_offset_bytes >= self.moe_routing_scratch_bytes
                    && si.meta_offset_bytes.is_multiple_of(ATTN_META_ALIGN),
                "short-batch slice {i} metadata offset {} violates MoE routing/alignment bound",
                si.meta_offset_bytes
            );
            let meta_i = scratch_base.offset(si.meta_offset_bytes);
            ensure!(
                ranges_disjoint(
                    scratch_base.0,
                    self.moe_routing_scratch_bytes,
                    meta_i.0,
                    si.meta_bytes,
                ),
                "short-batch slice {i} metadata aliases MoE routing scratch"
            );
            let hid_i = hidden_base.offset(si.row_offset * self.row_bytes);
            let res_i = residual_base.offset(si.row_offset * self.row_bytes);
            let bytes_i = si.len * self.row_bytes;
            let logits_i = logits_base.offset(i * self.vocab_row_bytes);
            for (j, sj) in self.slices[..i].iter().enumerate() {
                let hid_j = hidden_base.offset(sj.row_offset * self.row_bytes);
                let res_j = residual_base.offset(sj.row_offset * self.row_bytes);
                let bytes_j = sj.len * self.row_bytes;
                ensure!(
                    ranges_disjoint(hid_i.0, bytes_i, hid_j.0, bytes_j)
                        && ranges_disjoint(res_i.0, bytes_i, res_j.0, bytes_j),
                    "short-batch hidden/residual slices {j} and {i} overlap"
                );
                let meta_j = scratch_base.offset(sj.meta_offset_bytes);
                ensure!(
                    ranges_disjoint(meta_i.0, si.meta_bytes, meta_j.0, sj.meta_bytes),
                    "short-batch metadata slices {j} and {i} overlap"
                );
                let logits_j = logits_base.offset(j * self.vocab_row_bytes);
                ensure!(
                    ranges_disjoint(
                        logits_i.0,
                        self.vocab_row_bytes,
                        logits_j.0,
                        self.vocab_row_bytes,
                    ),
                    "short-batch logits rows {j} and {i} overlap"
                );
            }
        }
        Ok(())
    }
}

/// Build the ascending `(src_row, dst_row)` compaction plan mapping each slice's
/// last token row `row_offset + len - 1` into contiguous row `b` (`0..N`).
///
/// Since every slice has `len >= 2`, `row_offset_b >= 2 * b`, so
/// `src_row >= 2 * b + 1 > b = dst_row`. Ascending copies therefore never
/// overwrite an unread source row and never alias `src == dst`.
pub(super) fn compact_last_token_rows_plan(
    slices: &[ShortBatchSliceLayout],
) -> Result<Vec<(usize, usize)>> {
    let mut plan = Vec::with_capacity(slices.len());
    for (dst_row, slice) in slices.iter().enumerate() {
        let src_row = slice
            .row_offset
            .checked_add(slice.len)
            .and_then(|end| end.checked_sub(1))
            .context("short-batch last-token row index overflow")?;
        ensure!(
            src_row > dst_row,
            "short-batch last-token compaction requires src_row ({src_row}) > dst_row ({dst_row})"
        );
        plan.push((src_row, dst_row));
    }
    Ok(plan)
}

/// Drains outstanding pinned-host reads if the upload loop exits early on error.
struct ShortBatchPinnedH2dGuard<'a> {
    gpu: &'a dyn GpuBackend,
    stream: u64,
    armed: bool,
}

impl Drop for ShortBatchPinnedH2dGuard<'_> {
    fn drop(&mut self) {
        if self.armed
            && let Err(error) = self.gpu.synchronize(self.stream)
        {
            tracing::error!(
                "short-batch pinned H2D error-path synchronization failed on stream {}: {error:#}",
                self.stream
            );
            std::process::abort();
        }
    }
}

/// Per-stream DFlash proposer state summary for read-only preflight validation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct ShortBatchDflashStreamSummary {
    pub chunk_len: usize,
    pub ctx_len: usize,
    pub ctx_resident_len: usize,
    pub ctx_capacity: usize,
    pub ctx_slot_bytes: usize,
    pub ctx_allocation_bytes: usize,
    pub ctx_hidden_acc: DevicePtr,
    pub ring_append_ok: bool,
}

/// Pure host-side model and buffer inputs for DFlash short-batch capture gating.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct ShortBatchDflashPreflightInputs {
    pub capture_layers_len: usize,
    pub capture_mode: DflashCaptureMode,
    pub has_hc_mean_kernel: bool,
    pub has_strided_copy_kernel: bool,
    pub capture_width: usize,
    pub residual_width: usize,
    pub hidden_base: DevicePtr,
    pub norm_output_base: DevicePtr,
    pub arena_norm_output_bytes: usize,
}

/// Verified DFlash capture parameters for one short-batch round.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct ShortBatchDflashCapture {
    pub capture_width: usize,
    pub capture_row_bytes: usize,
    pub ctx_slot_bytes: usize,
    pub ctx_hidden_accs: Vec<DevicePtr>,
}

/// Read-only preflight disposition for DFlash capture during batched short-prefill.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum ShortBatchDflashPlan {
    Inactive,
    Supported(ShortBatchDflashCapture),
    Unsupported,
}

/// Pure validator for batched short-prefill DFlash capture.
///
/// Admits active DFlash capture only when `Qwen4HcMean`, both capture kernels,
/// 16-byte strided-copy alignment, zero-wrap fresh prompt ring invariants
/// (`ctx_len == 0` and `chunk_len <= ctx_capacity`), exact slot geometry,
/// pairwise-disjoint accumulator buffers, and `norm_output` capacity all hold.
#[must_use]
pub(super) fn validate_short_batch_dflash_preflight(
    inputs: ShortBatchDflashPreflightInputs,
    stream_states: &[Option<ShortBatchDflashStreamSummary>],
) -> ShortBatchDflashPlan {
    if inputs.capture_layers_len == 0 {
        return ShortBatchDflashPlan::Inactive;
    }
    if inputs.capture_mode != DflashCaptureMode::Qwen4HcMean
        || !inputs.has_hc_mean_kernel
        || !inputs.has_strided_copy_kernel
        || inputs.capture_width == 0
        || inputs.residual_width != inputs.capture_width.saturating_mul(4)
        || i32::try_from(inputs.capture_width).is_err()
        || inputs.hidden_base.is_null()
        || inputs.norm_output_base.is_null()
        || !inputs.norm_output_base.0.is_multiple_of(16)
        || stream_states.is_empty()
    {
        return ShortBatchDflashPlan::Unsupported;
    }
    let Some(capture_row_bytes) = inputs.capture_width.checked_mul(2) else {
        return ShortBatchDflashPlan::Unsupported;
    };
    if !capture_row_bytes.is_multiple_of(16) || u32::try_from(capture_row_bytes).is_err() {
        return ShortBatchDflashPlan::Unsupported;
    }
    let Some(ctx_slot_bytes) = inputs.capture_layers_len.checked_mul(capture_row_bytes) else {
        return ShortBatchDflashPlan::Unsupported;
    };

    let mut total_rows = 0usize;
    let mut ctx_hidden_accs = Vec::with_capacity(stream_states.len());
    let mut ctx_spans: Vec<(u64, usize)> = Vec::with_capacity(stream_states.len());
    for entry in stream_states {
        let Some(s) = entry else {
            return ShortBatchDflashPlan::Unsupported;
        };
        if s.chunk_len < 2
            || u32::try_from(s.chunk_len).is_err()
            || s.ctx_len != 0
            || s.ctx_resident_len != 0
            || s.chunk_len > s.ctx_capacity
            || s.ctx_slot_bytes != ctx_slot_bytes
            || !s.ring_append_ok
            || s.ctx_hidden_acc.is_null()
            || !s.ctx_hidden_acc.0.is_multiple_of(16)
        {
            return ShortBatchDflashPlan::Unsupported;
        }
        let Some(needed_bytes) = s.chunk_len.checked_mul(ctx_slot_bytes) else {
            return ShortBatchDflashPlan::Unsupported;
        };
        if s.ctx_allocation_bytes < needed_bytes {
            return ShortBatchDflashPlan::Unsupported;
        }
        let span_bytes = s.ctx_allocation_bytes;
        if ctx_spans
            .iter()
            .any(|&(base, bytes)| !ranges_disjoint(base, bytes, s.ctx_hidden_acc.0, span_bytes))
        {
            return ShortBatchDflashPlan::Unsupported;
        }
        let Some(next_rows) = total_rows.checked_add(s.chunk_len) else {
            return ShortBatchDflashPlan::Unsupported;
        };
        total_rows = next_rows;
        ctx_hidden_accs.push(s.ctx_hidden_acc);
        ctx_spans.push((s.ctx_hidden_acc.0, span_bytes));
    }

    if i32::try_from(total_rows).is_err() || u32::try_from(total_rows).is_err() {
        return ShortBatchDflashPlan::Unsupported;
    }
    let Some(norm_bytes) = total_rows.checked_mul(capture_row_bytes) else {
        return ShortBatchDflashPlan::Unsupported;
    };
    let Some(hidden_bytes) = total_rows
        .checked_mul(inputs.residual_width)
        .and_then(|elems| elems.checked_mul(2))
    else {
        return ShortBatchDflashPlan::Unsupported;
    };
    if norm_bytes > inputs.arena_norm_output_bytes
        || !ranges_disjoint(
            inputs.hidden_base.0,
            hidden_bytes,
            inputs.norm_output_base.0,
            norm_bytes,
        )
        || ctx_spans.iter().any(|&(base, bytes)| {
            !ranges_disjoint(base, bytes, inputs.norm_output_base.0, norm_bytes)
        })
    {
        return ShortBatchDflashPlan::Unsupported;
    }

    ShortBatchDflashPlan::Supported(ShortBatchDflashCapture {
        capture_width: inputs.capture_width,
        capture_row_bytes,
        ctx_slot_bytes,
        ctx_hidden_accs,
    })
}

fn diagnostic_dump_armed() -> bool {
    let env_nonempty = |key: &str| std::env::var(key).is_ok_and(|v| !v.is_empty());
    env_nonempty("ATLAS_NEMO_DUMP")
        || env_nonempty("ATLAS_PREFILL_HIDDEN_DUMP")
        || crate::model::env_diag::dump_ctx_hidden_path().is_some()
        || super::super::qgate_dump::qgate_dump_armed()
}

impl TransformerModel {
    /// Attempt the Qwen4 batched short-prefill fast path (ONE unified MoE pass
    /// across `2..=16` single-chunk prompts).
    ///
    /// Returns `Ok(None)` without mutating any sequence state, KV cache refs,
    /// or GPU buffers whenever any eligibility, arena, admission, or
    /// prefix-cache check declines batching so the caller can fall back cleanly
    /// to sequential `prefill_chunk_dispatch`. Once committed past preflight,
    /// errors propagate fail-closed.
    pub(super) fn try_prefill_qwen4_short_batch(
        &self,
        streams: &mut [PrefillSlice<'_>],
        stream: u64,
    ) -> Result<Option<Vec<DevicePtr>>> {
        let det_batch = crate::layers::deterministic_batch::deterministic_batch_policy()?;
        if !short_batch_enabled_with_policy(det_batch, short_batch_enabled()) {
            return Ok(None);
        }
        if self.profile
            || self.profile_first_pending.load(Ordering::Relaxed)
            || diagnostic_dump_armed()
        {
            return Ok(None);
        }
        let moe_selected = crate::layers::qwen4_prefill_moe::selected()?;
        let family_suppressed = crate::layers::qwen4_prefill_moe::family_suppressed();
        let max_chunk_rows = crate::layers::qwen4_prefill_moe::max_chunk_rows();

        let summaries: Vec<ShortBatchSliceSummary> = streams
            .iter()
            .map(|s| ShortBatchSliceSummary {
                prompt_len: s.prompt_tokens.len(),
                chunk_start: s.chunk_start,
                chunk_len: s.chunk_len,
                is_last_chunk: s.is_last_chunk,
                seq_len: s.seq.seq_len,
                tokens_empty: s.seq.tokens.is_empty(),
                block_table_empty: s.seq.block_table.is_empty(),
                chunked_meta_none: s.seq.chunked_prefill_meta.is_none(),
                lineage_unset: matches!(s.seq.qwen4_prefill_lineage, Qwen4PrefillLineage::Unset),
                rotary_identity: s.seq.rotary_positions.is_identity(),
                has_vision: self.vision_prompt_present(s.prompt_tokens),
                has_layer_states: s.seq.layer_states.len() == self.layers.len(),
                slot_idx: s.seq.slot_idx,
            })
            .collect();

        let dflash_plan = self.plan_short_batch_dflash_capture(streams);
        let dflash_supported = !matches!(dflash_plan, ShortBatchDflashPlan::Unsupported);
        let dflash_capture = match dflash_plan {
            ShortBatchDflashPlan::Supported(plan) => Some(plan),
            ShortBatchDflashPlan::Inactive | ShortBatchDflashPlan::Unsupported => None,
        };

        let gate_inputs = ShortBatchGateInputs {
            env_enabled: true,
            is_qwen4_exp: self.config.is_qwen4_exp(),
            fp32_residual: self.config.use_fp32_residual(),
            fp32_logits: self.use_fp32_logits,
            single_rank: self.comm.is_none(),
            pp_world_size: self.config.pp_world_size,
            moe_selected,
            family_suppressed,
            dflash_supported,
            has_final_mixer: self.qwen4_final_mixer.is_some(),
            max_chunk_rows,
            max_batch_tokens: self.buffers.max_batch_tokens(),
        };
        let Some(total_rows) = check_short_batch_eligible(gate_inputs, &summaries) else {
            return Ok(None);
        };

        let n = streams.len();
        let Some(final_mixer) = self.qwen4_final_mixer.as_ref() else {
            return Ok(None);
        };
        if final_mixer
            .preflight_qwen4_native_batched(n, false)
            .is_err()
        {
            return Ok(None);
        }

        let lengths: Vec<usize> = streams.iter().map(|s| s.chunk_len).collect();
        let vocab = self.verify_lmhead_vocab() as usize;
        let arenas = self.buffers.sizes();
        let Ok(layout) = ShortBatchLayout::compute(
            &lengths,
            ShortBatchLayoutParams {
                residual_width: self.config.residual_width(),
                top_k: self.config.num_experts_per_tok,
                use_mrope: self.config.mrope_interleaved,
                vocab_size: vocab,
                arena_hidden_bytes: arenas.hidden_states,
                arena_residual_bytes: arenas.residual,
                arena_scratch_bytes: arenas.scratch,
                arena_logits_bytes: arenas.logits,
            },
        ) else {
            return Ok(None);
        };

        let hidden = self.buffers.hidden_states();
        let residual = self.buffers.residual();
        let scratch = self.buffers.scratch();
        let logits_base = self.buffers.logits();
        if layout.total_rows != total_rows
            || layout
                .verify_device_non_aliasing(hidden, residual, scratch, logits_base)
                .is_err()
        {
            return Ok(None);
        }

        // Read-only MoE/compact admission checks before mutating any state.
        let preflight_ctx = ForwardContext {
            buffers: &self.buffers,
            gpu: self.gpu.as_ref(),
            config: &self.config,
            attn_metadata: None,
            profile: false,
            comm: None,
            graph_capture: false,
            ddtree_parent_ids_dev: None,
            tree_aware_attn: None,
            ssm_multi_seq_ptr_table_override: None,
            self_spec_sparse_draft: None,
            ffn_defer: None,
        };
        if crate::layers::qwen4_prefill_moe::admit_request(&self.config, total_rows, 0).is_err()
            || crate::layers::moe::qwen4_prefill_compact::validate_context(
                &preflight_ctx,
                total_rows,
            )
            .is_err()
        {
            return Ok(None);
        }

        let mut kv_cache = self.kv_cache.lock();
        if kv_cache.config().cache_blocks_per_seq.is_some() {
            // High-speed swap is incompatible with the fast Qwen4 prefill path.
            return Ok(None);
        }
        let bs = kv_cache.block_size();
        if bs == 0 {
            return Ok(None);
        }

        // Non-mutating prefix-cache probe: if any slice has a prefix hit, release
        // its acquired ref immediately and fall back to sequential prefill while
        // every SequenceState is still completely untouched.
        if self.prefix_cache.is_active() {
            for slice in streams.iter() {
                let hit =
                    self.lookup_prefill_prefix(slice.prompt_tokens, bs, slice.seq.session_hash);
                if hit.matched_tokens > 0 {
                    self.prefix_cache
                        .release_matched(slice.prompt_tokens, bs, hit.matched_tokens);
                    return Ok(None);
                }
            }
        }

        // Read-only pinned staging capacity check before committing.
        // We perform the actual ring reservation after Step 3a embedding so
        // `embed_qwen4_batch`'s H2D copies never share the reserved ring span.
        let pinned_capacity_ok = {
            let stg = self.pinned_staging.lock();
            let ring_len = stg.buffer.len().saturating_sub(stg.h2d_ring_base);
            let fits_ring =
                Self::fast_host_h2d_enabled() && layout.total_pinned_bytes + 64 <= ring_len;
            let fits_legacy = layout.total_pinned_bytes <= stg.h2d_ring_base;
            fits_ring || fits_legacy
        };
        if !pinned_capacity_ok {
            return Ok(None);
        }

        // ── COMMITTED TO BATCHED SHORT-PREFILL ──
        if super::super::prefill_sync_diag() >= 1 {
            self.gpu.synchronize_device()?;
        }

        // 1. Zero essential arenas once (matching chunk_start == 0 in prefill_b.rs).
        self.zero_short_batch_arenas(stream)?;

        // 2. Kick off host PLE pre-reads for all N prompts before embedding & Layer 0.
        #[cfg(all(feature = "cuda", target_os = "linux"))]
        let ple_prefill_batch = std::env::var("ATLAS_QWEN4_PLE_PREFILL_BATCH")
            .ok()
            .as_deref()
            == Some("1");
        #[cfg(all(feature = "cuda", target_os = "linux"))]
        let mut ple_prefetches: Vec<Option<crate::layers::qwen4_ple::PlePrefillRead>> = {
            if let Some(ple) = self.qwen4_ple.as_ref() {
                let use_stream_batch =
                    ple_prefill_batch && crate::layers::qwen4_ple::prefill_stream_level() >= 2;
                {
                    let mut pending_map = self.qwen4_ple_prefetch.lock();
                    if !use_stream_batch
                        && let Some(stale) =
                            pending_map.remove(&super::super::QWEN4_PLE_ADMISSION_SLOT)
                        && let Err(error) = stale.finish_discard()
                    {
                        tracing::warn!(
                            target: "atlas::qwen4_ple",
                            error = %format!("{error:#}"),
                            "discarded stale Qwen4 PLE admission prefetch before short-batch prefill"
                        );
                    }
                    for slice in streams.iter() {
                        if let Some(stale) = pending_map.remove(&slice.seq.slot_idx)
                            && let Err(error) = stale.finish_discard()
                        {
                            tracing::warn!(
                                target: "atlas::qwen4_ple",
                                slot_idx = slice.seq.slot_idx,
                                error = %format!("{error:#}"),
                                "discarded stale Qwen4 PLE prefetch before short-batch prefill"
                            );
                        }
                    }
                }
                if use_stream_batch {
                    streams
                        .iter()
                        .enumerate()
                        .map(|(b, slice)| {
                            let admitted = if b == 0 {
                                self.take_ple_admission_prefetch(slice.prompt_tokens, &[])
                            } else {
                                None
                            };
                            Some(admitted.unwrap_or_else(|| {
                                ple.begin_prefill_read(slice.prompt_tokens, &[])
                            }))
                        })
                        .collect()
                } else {
                    (0..n).map(|_| None).collect()
                }
            } else {
                (0..n).map(|_| None).collect()
            }
        };

        // 3a. Embed all N prompts contiguously and allocate KV blocks BEFORE
        //     staging attention metadata so `embed_qwen4_batch`'s temporary
        //     `scratch[0..len*4]` usage and fast-H2D copies complete first.
        let row_bytes = layout.row_bytes;
        for (b, slice) in streams.iter_mut().enumerate() {
            let sl = layout.slices[b];
            let tokens = slice.prompt_tokens;
            let seq = &mut *slice.seq;
            seq.cached_prefix_tokens = 0;
            seq.marconi_skip_to = 0;
            seq.prompt_len = sl.len;

            self.prefill_b_embed_chunk_at(
                tokens,
                0,
                sl.len,
                hidden.offset(sl.row_offset * row_bytes),
                stream,
            )?;

            let blocks_needed = sl.len.div_ceil(bs);
            ensure_blocks_through_prefill(
                seq,
                blocks_needed - 1,
                &mut kv_cache,
                self.prefix_cache.as_ref(),
                self.gpu.as_ref(),
                stream,
            )?;
        }

        // 3b. Reserve pinned staging and upload disjoint per-slice attention
        //     metadata + paged block tables after the MoE routing prefix.
        let (pinned_base, needs_pinned_sync) = if Self::fast_host_h2d_enabled()
            && let Some(ring_base) = self.reserve_pinned_h2d_range(layout.total_pinned_bytes + 64)
        {
            (ring_base, false)
        } else {
            (0usize, true)
        };
        let mut attn_metas: Vec<AttnMetadataDev> = Vec::with_capacity(n);
        let mut pinned_guard = ShortBatchPinnedH2dGuard {
            gpu: self.gpu.as_ref(),
            stream,
            armed: false,
        };

        for (b, slice) in streams.iter_mut().enumerate() {
            let sl = layout.slices[b];
            let tokens = slice.prompt_tokens;
            let seq = &mut *slice.seq;

            let meta_base = scratch.offset(sl.meta_offset_bytes);
            if needs_pinned_sync {
                pinned_guard.armed = true;
            }
            let meta_layout = self.prefill_b_upload_meta_at(
                tokens,
                seq,
                0,
                sl.len,
                0,
                sl.len,
                0,
                &kv_cache,
                meta_base,
                pinned_base + sl.pinned_offset_bytes,
                stream,
            )?;
            ensure!(
                meta_layout.needs_paged,
                "Qwen4 short-batch requires paged attention metadata"
            );
            ensure!(
                meta_layout.slot_offset == sl.slot_offset_bytes,
                "Qwen4 short-batch slot offset mismatch: layout={} actual={}",
                sl.slot_offset_bytes,
                meta_layout.slot_offset
            );
            ensure!(
                meta_layout.pinned_bytes(sl.len)? == sl.pinned_bytes,
                "Qwen4 short-batch pinned metadata size mismatch: layout={} actual={}",
                sl.pinned_bytes,
                meta_layout.pinned_bytes(sl.len)?
            );
            self.prefill_b_upload_paged(
                seq,
                sl.len,
                0,
                sl.len,
                meta_base,
                meta_layout.slot_offset,
                &kv_cache,
                stream,
            )?;

            let page_meta = seq
                .chunked_prefill_meta
                .as_ref()
                .context("Qwen4 short-batch missing chunked_prefill_meta after upload")?;
            let (positions_h, positions_w) = if meta_layout.use_mrope {
                (
                    meta_base.offset(meta_layout.pos_stream_bytes),
                    meta_base.offset(meta_layout.pos_stream_bytes * 2),
                )
            } else {
                (meta_base, meta_base)
            };
            attn_metas.push(AttnMetadataDev {
                qwen4_qsa_required: seq.qwen4_qsa_required,
                positions: meta_base,
                positions_h,
                positions_w,
                slot: meta_base.offset(meta_layout.slot_offset),
                seq_len: page_meta.seq_len,
                block_table: page_meta.block_table,
                max_blocks_per_seq: u32::try_from(seq.block_table.len())
                    .context("short-batch block_table length overflow")?,
                num_seqs: 1,
            });
        }

        if needs_pinned_sync {
            self.gpu.synchronize(stream)?;
            pinned_guard.armed = false;
        }

        // 4. Preflight attention16 kernels & arm layer-loop tracking.
        if crate::layers::qwen4_prefill_moe::attn16::selected()? {
            for layer in &self.layers {
                layer.preflight_qwen4_attn16(&kv_cache, self.gpu.as_ref(), stream)?;
            }
        }
        crate::layers::qwen4_prefill_moe::last_layer_crop::take_engaged();
        if crate::layers::qwen4_hyper::pending::discard_stale() {
            tracing::warn!("dropped stale pending HC injection before Qwen4 short-batch prefill");
        }
        if crate::layers::moe::qwen4_moe_status::discard_stale() {
            tracing::warn!(
                "dropped stale deferred compact-MoE status words before Qwen4 short-batch prefill"
            );
        }
        crate::layers::qwen4_fp8_act_amax::oproj_chunk_begin();

        let unified_ctx = ForwardContext {
            attn_metadata: Some(attn_metas[0]),
            ..preflight_ctx
        };
        let prefill_receipt =
            crate::model::qwen4_prefill_engagement::begin(&self.config, total_rows)?;

        // 5. Layers 0..num_hidden_layers: Split Mixer -> ONE Unified MoE Pass.
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            #[cfg(all(feature = "cuda", target_os = "linux"))]
            if layer_idx == 1
                && let Some(ple) = self.qwen4_ple.as_ref()
            {
                crate::layers::qwen4_hyper::pending::flush(self.gpu.as_ref(), stream)?;
                for (b, slice) in streams.iter().enumerate() {
                    let sl = layout.slices[b];
                    let hidden_b = hidden.offset(sl.row_offset * row_bytes);
                    let tokens_b = slice.prompt_tokens;
                    let slot_idx = slice.seq.slot_idx;
                    if ple_prefill_batch {
                        ple.forward_prefill(
                            tokens_b,
                            &[],
                            hidden_b,
                            slot_idx,
                            true,
                            self.gpu.as_ref(),
                            stream,
                            ple_prefetches[b].take(),
                        )?;
                    } else {
                        let mut history = Vec::with_capacity(tokens_b.len());
                        for (row, &token) in tokens_b.iter().enumerate() {
                            ple.forward_token(
                                token,
                                &history,
                                hidden_b.offset(row * row_bytes),
                                slot_idx,
                                row == 0,
                                self.gpu.as_ref(),
                                stream,
                            )?;
                            history.push(token);
                        }
                    }
                }
            }

            // Sub-phase A: per-prompt mixer under ExpertMajorSliceGuard (MoE deferred).
            for (b, slice) in streams.iter_mut().enumerate() {
                let sl = layout.slices[b];
                let slice_ctx = ForwardContext {
                    attn_metadata: Some(attn_metas[b]),
                    ..unified_ctx
                };
                {
                    let _slice_guard = ExpertMajorSliceGuard::enter(max_chunk_rows);
                    layer
                        .prefill(
                            hidden.offset(sl.row_offset * row_bytes),
                            residual.offset(sl.row_offset * row_bytes),
                            sl.len,
                            slice.seq.layer_states[layer_idx].as_mut(),
                            &mut kv_cache,
                            0,
                            &mut slice.seq.block_table,
                            &mut slice.seq.disk_block_ids,
                            &mut slice.seq.disk_last_offloaded_per_layer,
                            0,
                            &slice_ctx,
                            stream,
                        )
                        .with_context(|| {
                            format!(
                                "Qwen4 short-batch layer {layer_idx} slice {b} (len={}) mixer failed",
                                sl.len
                            )
                        })?;
                }
                crate::layers::qwen4_hyper::pending::flush(self.gpu.as_ref(), stream)?;
            }

            // Sub-phase B: ONE unified MoE pass over all total_rows.
            layer
                .qwen4_prefill_finish_moe(hidden, residual, total_rows, &unified_ctx, stream)
                .with_context(|| {
                    format!(
                        "Qwen4 short-batch layer {layer_idx} unified MoE (total_rows={total_rows}) failed"
                    )
                })?;
            crate::layers::qwen4_hyper::pending::flush(self.gpu.as_ref(), stream)?;

            // Sub-phase C: DFlash post-layer hidden capture (Qwen4HcMean) over all
            // total_rows, matching single-sequence `forward_layers.rs` timing bit-for-bit.
            if let Some(ref dflash) = dflash_capture
                && let Some(cap_slot) = self
                    .dflash_capture_layers
                    .iter()
                    .position(|&l| l == layer_idx)
            {
                self.capture_short_batch_dflash_layer(
                    dflash, &layout, layer_idx, cap_slot, hidden, stream,
                )?;
            }

            if super::super::prefill_sync_diag() >= 2 {
                self.gpu.synchronize_device()?;
            }
        }

        crate::layers::qwen4_hyper::pending::flush(self.gpu.as_ref(), stream)?;
        crate::layers::moe::qwen4_moe_status::flush(self.gpu.as_ref(), stream)?;
        crate::layers::qwen4_fp8_act_amax::oproj_chunk_receipt(0, total_rows);
        if let Some(receipt) = prefill_receipt {
            receipt.finish()?;
        }

        // 6. Capture MTP last-K hidden rows per slice before mutating hidden[0..N].
        self.mtp_lastk_capture_short_batch(streams, &layout, stream)?;

        // 7. Update sequence state + insert into prefix cache & Marconi snapshot pool.
        for (b, slice) in streams.iter_mut().enumerate() {
            let sl = layout.slices[b];
            let tokens = slice.prompt_tokens;
            let seq = &mut *slice.seq;
            seq.tokens.extend_from_slice(tokens);
            seq.seq_len = sl.len;
            self.insert_short_batch_prefix_cache(tokens, seq, &mut kv_cache, bs, stream);
            self.update_dflash_ctx_len_after_prefill(seq, 0, sl.len)?;
        }
        drop(kv_cache);

        // 8. Run MTP last-K prompt replay BEFORE batched finalization when active,
        //    because `proposer.prefill_last_k` uses `buffers.hidden_states()`,
        //    `buffers.residual()`, and `buffers.logits()` as scratch.
        let run_mtp_lastk_replay = !self.speculation_forced_disabled
            && self.mtp_lastk_capture_active()
            && self.active_proposer().is_some()
            && self.mtp_lastk_buf.is_some();

        if run_mtp_lastk_replay {
            // `mtp_lastk_prefill_after_finalize_enabled` uploads each stream's
            // last-K hidden rows through the 8 MiB unsynchronized pinned H2D
            // staging ring (`copy_h2d_group_fast`). Synchronize the stream
            // before cumulative uploads exceed half the ring (4 MiB) so wide
            // short-batches (e.g. C=16 × 1.25 MiB = 20 MiB) can never wrap over
            // an in-flight GPU MTP replay transfer.
            const MTP_LASTK_RING_SYNC_BYTES: usize = 4 * 1024 * 1024;
            let mut uploaded_ring_bytes = 0usize;
            for (b, slice) in streams.iter_mut().enumerate() {
                let sl = layout.slices[b];
                let slice_h2d_bytes = slice.seq.mtp_lastk_host_filled.saturating_mul(row_bytes);
                if uploaded_ring_bytes > 0
                    && uploaded_ring_bytes.saturating_add(slice_h2d_bytes)
                        > MTP_LASTK_RING_SYNC_BYTES
                {
                    self.gpu.synchronize(stream)?;
                    uploaded_ring_bytes = 0;
                }
                self.mtp_lastk_prefill_after_finalize_enabled(
                    slice.prompt_tokens,
                    slice.seq,
                    0,
                    sl.len,
                    sl.len,
                    stream,
                )?;
                uploaded_ring_bytes = uploaded_ring_bytes.saturating_add(slice_h2d_bytes);
            }
            if uploaded_ring_bytes > 0 {
                self.gpu.synchronize(stream)?;
            }
            // Restore the N last-token hidden rows directly from each sequence's
            // already-captured `mtp_lastk_host_buf` tail row into `hidden[0..N)`.
            let mut packed_last_hidden = Vec::with_capacity(n * row_bytes);
            for (b, slice) in streams.iter().enumerate() {
                let filled = slice.seq.mtp_lastk_host_filled;
                let last_start = filled
                    .checked_sub(1)
                    .and_then(|idx| idx.checked_mul(row_bytes))
                    .context("short-batch MTP last-K host ring missing final row")?;
                let last_end = last_start
                    .checked_add(row_bytes)
                    .context("short-batch MTP last-K host row end overflow")?;
                ensure!(
                    last_end <= slice.seq.mtp_lastk_host_buf.len(),
                    "short-batch slice {b} MTP last-K host ring truncated ({} < {last_end})",
                    slice.seq.mtp_lastk_host_buf.len()
                );
                packed_last_hidden
                    .extend_from_slice(&slice.seq.mtp_lastk_host_buf[last_start..last_end]);
            }
            self.copy_h2d_group_fast(
                &[HostToDeviceCopy::new(&packed_last_hidden, hidden)],
                stream,
            )?;
        } else {
            // Compact the N last-token rows (offset_b + len_b - 1) of hidden &
            // residual into rows 0..N without any source/destination overlap.
            let compaction = compact_last_token_rows_plan(&layout.slices)?;
            for (src_row, dst_row) in compaction {
                self.gpu.copy_d2d_async(
                    hidden.offset(src_row * row_bytes),
                    hidden.offset(dst_row * row_bytes),
                    row_bytes,
                    stream,
                )?;
                self.gpu.copy_d2d_async(
                    residual.offset(src_row * row_bytes),
                    residual.offset(dst_row * row_bytes),
                    row_bytes,
                    stream,
                )?;
            }
        }

        // 9. ONE Batched Finalization: final_mixer -> final_norm -> lm_head over N rows.
        let n_u32 = u32::try_from(n).context("short-batch stream count exceeds u32")?;
        let normed = final_mixer.prepare_batched(
            hidden,
            residual,
            n,
            &self.buffers,
            self.gpu.as_ref(),
            self.config.rms_norm_eps as f32,
            stream,
        )?;
        self.final_norm_rows(normed, normed, n_u32, stream)?;
        let logits_dev = self.lm_head_qwen4_rows(normed, n_u32, stream)?;

        static ENGAGED: Once = Once::new();
        let dflash_active = dflash_capture.is_some();
        ENGAGED.call_once(|| {
            tracing::info!(
                "QWEN4_SHORT_BATCH_PREFILL_ENGAGED streams={n} total_rows={total_rows} max_chunk_rows={max_chunk_rows} dflash_capture={dflash_active}"
            );
        });

        let logits_out = (0..n)
            .map(|b| logits_dev.offset(b * layout.vocab_row_bytes))
            .collect();
        Ok(Some(logits_out))
    }

    fn plan_short_batch_dflash_capture(
        &self,
        streams: &[PrefillSlice<'_>],
    ) -> ShortBatchDflashPlan {
        if self.dflash_capture_layers.is_empty() {
            return ShortBatchDflashPlan::Inactive;
        }
        let stream_states: Vec<Option<ShortBatchDflashStreamSummary>> = streams
            .iter()
            .map(|slice| {
                let dstate = slice
                    .seq
                    .proposer_state
                    .as_ref()?
                    .as_any()
                    .downcast_ref::<crate::layers::DflashProposerState>()?;
                let ring_append_ok = dstate
                    .ctx_ring_state()
                    .is_ok_and(|ring| ring.plan_append_at(0, slice.chunk_len).is_ok());
                Some(ShortBatchDflashStreamSummary {
                    chunk_len: slice.chunk_len,
                    ctx_len: dstate.ctx_len,
                    ctx_resident_len: dstate.ctx_resident_len,
                    ctx_capacity: dstate.ctx_capacity,
                    ctx_slot_bytes: dstate.ctx_slot_bytes,
                    ctx_allocation_bytes: dstate.ctx_allocation_bytes,
                    ctx_hidden_acc: dstate.ctx_hidden_acc,
                    ring_append_ok,
                })
            })
            .collect();
        validate_short_batch_dflash_preflight(
            ShortBatchDflashPreflightInputs {
                capture_layers_len: self.dflash_capture_layers.len(),
                capture_mode: self.dflash_capture_mode,
                has_hc_mean_kernel: self.qwen4_hc_mean_rows_kernel.0 != 0,
                has_strided_copy_kernel: self.strided_copy_rows_kernel.0 != 0,
                capture_width: self.dflash_capture_width,
                residual_width: self.config.residual_width(),
                hidden_base: self.buffers.hidden_states(),
                norm_output_base: self.buffers.norm_output(),
                arena_norm_output_bytes: self.buffers.sizes().norm_output,
            },
            &stream_states,
        )
    }

    fn capture_short_batch_dflash_layer(
        &self,
        dflash: &ShortBatchDflashCapture,
        layout: &ShortBatchLayout,
        layer_idx: usize,
        cap_slot: usize,
        hidden: DevicePtr,
        stream: u64,
    ) -> Result<()> {
        let total_rows = layout.total_rows;
        let h = dflash.capture_width;
        let norm_output = self.buffers.norm_output();
        let rows_i32 =
            i32::try_from(total_rows).context("short-batch qwen4_hc_mean_rows rows overflow")?;
        let rows_u32 = u32::try_from(total_rows)
            .context("short-batch qwen4_hc_mean_rows grid rows overflow")?;
        let h_i32 = i32::try_from(h).context("short-batch qwen4_hc_mean_rows hidden overflow")?;
        let half_h_u32 = u32::try_from(h.div_ceil(2))
            .context("short-batch qwen4_hc_mean_rows grid_x overflow")?;
        let grid_x = spark_runtime::kernel_args::div_ceil(half_h_u32, 256);
        spark_runtime::kernel_args::KernelLaunch::new(
            self.gpu.as_ref(),
            self.qwen4_hc_mean_rows_kernel,
        )
        .grid([grid_x, rows_u32, 1])
        .block([256, 1, 1])
        .arg_ptr(hidden)
        .arg_ptr(norm_output)
        .arg_i32(rows_i32)
        .arg_i32(h_i32)
        .launch(stream)
        .with_context(|| {
            format!(
                "Qwen4 short-batch layer {layer_idx} DFlash HC mean (total_rows={total_rows}) failed"
            )
        })?;

        let cap_slot_offset = cap_slot
            .checked_mul(dflash.capture_row_bytes)
            .context("short-batch DFlash capture slot byte offset overflow")?;
        let row_bytes_u32 = u32::try_from(dflash.capture_row_bytes)
            .context("short-batch DFlash capture row bytes overflow")?;
        let src_stride_u64 = dflash.capture_row_bytes as u64;
        let dst_stride_u64 = dflash.ctx_slot_bytes as u64;
        for (b, sl) in layout.slices.iter().enumerate() {
            let src = norm_output.offset(sl.row_offset * dflash.capture_row_bytes);
            let dst = dflash.ctx_hidden_accs[b].offset(cap_slot_offset);
            let slice_rows =
                u32::try_from(sl.len).context("short-batch DFlash capture slice len overflow")?;
            crate::layers::ops::strided_copy_rows(
                self.gpu.as_ref(),
                self.strided_copy_rows_kernel,
                src,
                dst,
                slice_rows,
                row_bytes_u32,
                src_stride_u64,
                dst_stride_u64,
                stream,
            )
            .with_context(|| {
                format!(
                    "Qwen4 short-batch layer {layer_idx} slice {b} (len={}) DFlash capture copy failed",
                    sl.len
                )
            })?;
        }
        Ok(())
    }

    /// Clean sequential fallback for Qwen4 multi-stream `prefill_batch_chunk`
    /// when batched short-prefill declines (kill-switch, prefix hit, multi-chunk,
    /// unsupported DFlash capture, oversized total rows, etc.).
    ///
    /// Preserves distinct, non-aliased logits rows in `self.buffers.logits()`
    /// when two or more streams complete their final chunk in the same call.
    pub(super) fn prefill_qwen4_sequential_fallback(
        &self,
        streams: &mut [PrefillSlice<'_>],
        multi_chunk: bool,
        stream: u64,
    ) -> Result<Vec<DevicePtr>> {
        let n = streams.len();
        if multi_chunk {
            for slice in streams.iter_mut() {
                crate::layers::qwen4_prefill_moe::attn16::mark_ordinary_for_selected_route(
                    &mut slice.seq.qwen4_prefill_lineage,
                    slice.prompt_tokens.len(),
                );
            }
        }
        let completing_count = streams.iter().filter(|s| s.is_last_chunk).count();
        let last_completing = streams.iter().rposition(|s| s.is_last_chunk);
        let preserve_distinct = completing_count >= 2 && self.config.pp_rank == 0;
        let vocab_row_bytes = if preserve_distinct {
            self.validate_distinct_bf16_logits_rows(n, self.verify_lmhead_vocab() as usize)?
        } else {
            0
        };

        let mut saved_logits: Vec<Option<Vec<u8>>> = (0..n).map(|_| None).collect();
        let mut logits_out: Vec<DevicePtr> = Vec::with_capacity(n);

        for (b, slice) in streams.iter_mut().enumerate() {
            let ptr = self.prefill_chunk_dispatch(
                slice.prompt_tokens,
                slice.seq,
                slice.chunk_start,
                slice.chunk_len,
                slice.is_last_chunk,
                stream,
            )?;
            if preserve_distinct
                && slice.is_last_chunk
                && !ptr.is_null()
                && Some(b) != last_completing
            {
                self.gpu.synchronize(stream)?;
                let mut host_row = vec![0u8; vocab_row_bytes];
                self.gpu.copy_d2h(ptr, &mut host_row)?;
                saved_logits[b] = Some(host_row);
            }
            logits_out.push(ptr);
        }

        if preserve_distinct && let Some(last_idx) = last_completing {
            self.restore_batched_prefill_logits_rows(
                &mut logits_out,
                &saved_logits,
                last_idx,
                vocab_row_bytes,
                stream,
            )?;
        }

        Ok(logits_out)
    }

    fn zero_short_batch_arenas(&self, stream: u64) -> Result<()> {
        use crate::model::host_bucket::{ZERO_ALL_SELECTOR, first_time, zero_all_plan_for};
        let names = &spark_runtime::buffers::BufferArena::ZERO_ALL_BUFFERS;
        match zero_all_plan_for(names, true).map_err(anyhow::Error::msg)? {
            None => self.buffers.zero_all(self.gpu.as_ref(), stream)?,
            Some(plan) => {
                let fill: [Option<u8>; 19] = plan
                    .try_into()
                    .map_err(|_| anyhow::anyhow!("{ZERO_ALL_SELECTOR} plan length"))?;
                let (written, untouched) =
                    self.buffers
                        .zero_all_with(self.gpu.as_ref(), stream, &fill)?;
                if first_time("PREFILL_ZERO_ALL_PLAN_ENGAGED") {
                    tracing::info!(
                        "PREFILL_ZERO_ALL_PLAN_ENGAGED selector={ZERO_ALL_SELECTOR} plan={} written_bytes={written} untouched_bytes={untouched}",
                        std::env::var(ZERO_ALL_SELECTOR)
                            .ok()
                            .filter(|s| !s.is_empty())
                            .unwrap_or_else(|| "default_skip_data".to_string()),
                    );
                }
            }
        }
        Ok(())
    }

    /// Capture each short-batch slice's trailing `min(K, len_b)` hidden rows from
    /// `hidden.offset(row_offset_b * row_bytes)` into `seq.mtp_lastk_host_buf`,
    /// matching [`Self::mtp_lastk_capture_chunk`] semantics exactly.
    fn mtp_lastk_capture_short_batch(
        &self,
        streams: &mut [PrefillSlice<'_>],
        layout: &ShortBatchLayout,
        stream: u64,
    ) -> Result<()> {
        crate::model::speculation_control::run_lastk_if_enabled(
            self.speculation_forced_disabled,
            || {
                let capacity = self.mtp_lastk_capacity;
                if !self.mtp_lastk_capture_active() {
                    return Ok(());
                }
                if let Some(ref c) = self.comm
                    && c.rank() != 0
                {
                    return Ok(());
                }
                let row_bytes = layout.row_bytes;
                let ring_capacity_bytes = capacity
                    .checked_mul(row_bytes)
                    .context("short-batch MTP last-K ring size overflow")?;
                let hidden = self.buffers.hidden_states();

                // Synchronize once before reading all N slices' tail rows D2H.
                self.gpu.synchronize(stream)?;
                for (b, slice) in streams.iter_mut().enumerate() {
                    let sl = layout.slices[b];
                    let seq = &mut *slice.seq;
                    if seq.mtp_lastk_host_buf.len() != ring_capacity_bytes {
                        seq.mtp_lastk_host_buf = vec![0u8; ring_capacity_bytes];
                    }
                    let ingest = capacity.min(sl.len);
                    let src_first_row = sl.row_offset + (sl.len - ingest);
                    let src_bytes = ingest * row_bytes;
                    let src_ptr = hidden.offset(src_first_row * row_bytes);
                    self.gpu
                        .copy_d2h(src_ptr, &mut seq.mtp_lastk_host_buf[..src_bytes])?;
                    seq.mtp_lastk_host_filled = ingest;
                    seq.mtp_lastk_end_abs = sl.len - 1;
                }
                static CAPTURED: Once = Once::new();
                CAPTURED.call_once(|| {
                    tracing::info!(
                        "MTP_LASTK_CAPTURE_ENGAGED capacity={capacity} short_batch_streams={}",
                        streams.len()
                    );
                });
                Ok(())
            },
        )?;
        Ok(())
    }

    fn insert_short_batch_prefix_cache(
        &self,
        tokens: &[u32],
        seq: &SequenceState,
        kv_cache: &mut PagedKvCache,
        bs: usize,
        stream: u64,
    ) {
        if self.tokens_have_vision_pad(tokens) {
            return;
        }
        let noncausal_spans = seq.rotary_positions.noncausal_spans();
        let save_aligned_ssm = bs > 0 && tokens.len() >= bs && tokens.len().is_multiple_of(bs);
        let mut snap_saved = None;
        if self.prefix_cache.is_active() && self.ssm_snapshots.is_enabled() && save_aligned_ssm {
            let snap_result = match self.ssm_snapshots.save(
                seq.slot_idx,
                seq.session_hash,
                Self::seq_h_is_f16(seq),
                &self.ssm_pool,
                self.gpu.as_ref(),
                stream,
            ) {
                Ok(Some(id)) => Some(id),
                Ok(None) => {
                    tracing::debug!("Snapshot pool full, reclaiming...");
                    if self
                        .ssm_snapshots
                        .reclaim_from_cache(self.prefix_cache.as_ref(), kv_cache)
                    {
                        self.ssm_snapshots
                            .save(
                                seq.slot_idx,
                                seq.session_hash,
                                Self::seq_h_is_f16(seq),
                                &self.ssm_pool,
                                self.gpu.as_ref(),
                                stream,
                            )
                            .ok()
                            .flatten()
                    } else {
                        None
                    }
                }
                Err(e) => {
                    tracing::warn!("Qwen4 short-batch SSM snapshot save error: {e}");
                    None
                }
            };
            snap_saved = snap_result;
        }
        if let Some(snap_id) = snap_saved {
            let (displaced, acquired) = self.prefix_cache.insert_with_snapshot_and_noncausal_spans(
                tokens,
                &seq.block_table,
                &seq.disk_block_ids,
                bs,
                snap_id,
                seq.session_hash,
                seq.cached_prefix_tokens,
                &noncausal_spans,
            );
            cache_acquires_disk_refs(&acquired);
            if let Some(old) = displaced {
                self.ssm_snapshots.free(old);
            }
        } else {
            let acquired = self.prefix_cache.insert_with_noncausal_spans(
                tokens,
                &seq.block_table,
                &seq.disk_block_ids,
                bs,
                seq.cached_prefix_tokens,
                &noncausal_spans,
            );
            cache_acquires_disk_refs(&acquired);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_gate() -> ShortBatchGateInputs {
        ShortBatchGateInputs {
            env_enabled: true,
            is_qwen4_exp: true,
            fp32_residual: false,
            fp32_logits: false,
            single_rank: true,
            pp_world_size: 1,
            moe_selected: true,
            family_suppressed: false,
            dflash_supported: true,
            has_final_mixer: true,
            max_chunk_rows: 2048,
            max_batch_tokens: 4096,
        }
    }

    fn slice_summary(len: usize, slot_idx: usize) -> ShortBatchSliceSummary {
        ShortBatchSliceSummary {
            prompt_len: len,
            chunk_start: 0,
            chunk_len: len,
            is_last_chunk: true,
            seq_len: 0,
            tokens_empty: true,
            block_table_empty: true,
            chunked_meta_none: true,
            lineage_unset: true,
            rotary_identity: true,
            has_vision: false,
            has_layer_states: true,
            slot_idx,
        }
    }

    #[test]
    fn kill_switch_defaults_on_and_disables_only_on_zero() {
        assert!(parse_short_batch_enabled(None));
        assert!(parse_short_batch_enabled(Some("1")));
        assert!(parse_short_batch_enabled(Some("auto")));
        assert!(parse_short_batch_enabled(Some("")));
        assert!(!parse_short_batch_enabled(Some("0")));
        assert!(!parse_short_batch_enabled(Some(" 0 ")));
    }

    #[test]
    fn eligibility_admits_2_through_16_streams_and_rejects_out_of_range() {
        let gate = default_gate();
        assert_eq!(check_short_batch_eligible(gate, &[]), None);
        assert_eq!(
            check_short_batch_eligible(gate, &[slice_summary(64, 0)]),
            None
        );

        for n in MIN_SHORT_BATCH_STREAMS..=MAX_SHORT_BATCH_STREAMS {
            let slices: Vec<ShortBatchSliceSummary> =
                (0..n).map(|b| slice_summary(64, b)).collect();
            assert_eq!(
                check_short_batch_eligible(gate, &slices),
                Some(n * 64),
                "n={n} should be admitted"
            );
        }

        let too_many: Vec<ShortBatchSliceSummary> = (0..=MAX_SHORT_BATCH_STREAMS)
            .map(|b| slice_summary(64, b))
            .collect();
        assert_eq!(check_short_batch_eligible(gate, &too_many), None);
    }

    #[test]
    fn eligibility_enforces_total_rows_cap_and_min_chunk_len_and_kill_switch() {
        let gate = default_gate();

        // Exact 2048 ceiling passes; 2049 is rejected.
        let exact_2048: Vec<ShortBatchSliceSummary> =
            (0..16).map(|b| slice_summary(128, b)).collect();
        assert_eq!(check_short_batch_eligible(gate, &exact_2048), Some(2048));

        let mut over_2048 = exact_2048.clone();
        over_2048[0].prompt_len = 129;
        over_2048[0].chunk_len = 129;
        assert_eq!(check_short_batch_eligible(gate, &over_2048), None);

        // Even with wide max_chunk_rows=16384 and max_batch_tokens=16384, cap stays at 2048.
        let wide_gate = ShortBatchGateInputs {
            max_chunk_rows: 16384,
            max_batch_tokens: 16384,
            ..gate
        };
        assert_eq!(check_short_batch_eligible(wide_gate, &over_2048), None);

        // chunk_len < 2 is rejected; chunk_len == 2 is admitted.
        assert_eq!(
            check_short_batch_eligible(gate, &[slice_summary(2, 0), slice_summary(2, 1)]),
            Some(4)
        );
        assert_eq!(
            check_short_batch_eligible(gate, &[slice_summary(1, 0), slice_summary(64, 1)]),
            None
        );

        // Kill-switch =0 disables gating cleanly.
        let disabled = ShortBatchGateInputs {
            env_enabled: false,
            ..gate
        };
        assert_eq!(check_short_batch_eligible(disabled, &exact_2048), None);
    }

    #[test]
    fn eligibility_rejects_multi_chunk_vision_dflash_suppressed_tp_and_slot_alias() {
        let gate = default_gate();
        let base = vec![slice_summary(64, 0), slice_summary(64, 1)];
        assert_eq!(check_short_batch_eligible(gate, &base), Some(128));

        // Non-zero chunk_start or non-last chunk or partial prompt slice rejected.
        let mut bad_start = base.clone();
        bad_start[1].chunk_start = 32;
        assert_eq!(check_short_batch_eligible(gate, &bad_start), None);

        let mut bad_last = base.clone();
        bad_last[0].is_last_chunk = false;
        assert_eq!(check_short_batch_eligible(gate, &bad_last), None);

        let mut bad_vision = base.clone();
        bad_vision[1].has_vision = true;
        assert_eq!(check_short_batch_eligible(gate, &bad_vision), None);

        let mut aliased_slot = base.clone();
        aliased_slot[1].slot_idx = 0;
        assert_eq!(check_short_batch_eligible(gate, &aliased_slot), None);

        // Model/runtime gates.
        for rejected in [
            ShortBatchGateInputs {
                is_qwen4_exp: false,
                ..gate
            },
            ShortBatchGateInputs {
                single_rank: false,
                ..gate
            },
            ShortBatchGateInputs {
                moe_selected: false,
                ..gate
            },
            ShortBatchGateInputs {
                family_suppressed: true,
                ..gate
            },
            ShortBatchGateInputs {
                dflash_supported: false,
                ..gate
            },
            ShortBatchGateInputs {
                has_final_mixer: false,
                ..gate
            },
        ] {
            assert_eq!(check_short_batch_eligible(rejected, &base), None);
        }
    }

    #[test]
    fn dflash_preflight_admits_qwen4_hc_mean_and_rejects_unsupported_modes_or_states() {
        const H: usize = 2560;
        const N_CAP: usize = 8;
        const CAP_ROW_BYTES: usize = H * 2;
        const SLOT_BYTES: usize = N_CAP * CAP_ROW_BYTES;
        const CTX_CAP: usize = 4096;
        const ALLOC_BYTES: usize = CTX_CAP * SLOT_BYTES;

        let base_inputs = ShortBatchDflashPreflightInputs {
            capture_layers_len: N_CAP,
            capture_mode: DflashCaptureMode::Qwen4HcMean,
            has_hc_mean_kernel: true,
            has_strided_copy_kernel: true,
            capture_width: H,
            residual_width: 4 * H,
            hidden_base: DevicePtr(0x1_0000_0000),
            norm_output_base: DevicePtr(0x2_0000_0000),
            arena_norm_output_bytes: 2048 * CAP_ROW_BYTES,
        };
        let mk_stream = |idx: usize, chunk_len: usize| ShortBatchDflashStreamSummary {
            chunk_len,
            ctx_len: 0,
            ctx_resident_len: 0,
            ctx_capacity: CTX_CAP,
            ctx_slot_bytes: SLOT_BYTES,
            ctx_allocation_bytes: ALLOC_BYTES,
            ctx_hidden_acc: DevicePtr(0x10_0000_0000 + (idx as u64) * (ALLOC_BYTES as u64 + 4096)),
            ring_append_ok: true,
        };
        let streams = vec![Some(mk_stream(0, 64)), Some(mk_stream(1, 128))];

        // Inactive when no DFlash capture layers are configured.
        let inactive = ShortBatchDflashPreflightInputs {
            capture_layers_len: 0,
            ..base_inputs
        };
        assert_eq!(
            validate_short_batch_dflash_preflight(inactive, &[]),
            ShortBatchDflashPlan::Inactive
        );

        // Supported when Qwen4HcMean + both kernels + all fresh zero-wrap stream states pass.
        assert_eq!(
            validate_short_batch_dflash_preflight(base_inputs, &streams),
            ShortBatchDflashPlan::Supported(ShortBatchDflashCapture {
                capture_width: H,
                capture_row_bytes: CAP_ROW_BYTES,
                ctx_slot_bytes: SLOT_BYTES,
                ctx_hidden_accs: vec![
                    streams[0].unwrap().ctx_hidden_acc,
                    streams[1].unwrap().ctx_hidden_acc,
                ],
            })
        );

        // Unsupported capture modes and missing kernels must be rejected.
        for bad_inputs in [
            ShortBatchDflashPreflightInputs {
                capture_mode: DflashCaptureMode::ResidualSlice,
                ..base_inputs
            },
            ShortBatchDflashPreflightInputs {
                capture_mode: DflashCaptureMode::Qwen4HyperProjected,
                ..base_inputs
            },
            ShortBatchDflashPreflightInputs {
                capture_mode: DflashCaptureMode::Qwen4LayerAttnHc,
                ..base_inputs
            },
            ShortBatchDflashPreflightInputs {
                has_hc_mean_kernel: false,
                ..base_inputs
            },
            ShortBatchDflashPreflightInputs {
                has_strided_copy_kernel: false,
                ..base_inputs
            },
            ShortBatchDflashPreflightInputs {
                arena_norm_output_bytes: 191 * CAP_ROW_BYTES,
                ..base_inputs
            },
            ShortBatchDflashPreflightInputs {
                norm_output_base: DevicePtr(0x2_0000_0008),
                ..base_inputs
            },
            ShortBatchDflashPreflightInputs {
                norm_output_base: base_inputs.hidden_base,
                ..base_inputs
            },
        ] {
            assert_eq!(
                validate_short_batch_dflash_preflight(bad_inputs, &streams),
                ShortBatchDflashPlan::Unsupported
            );
        }

        // Per-stream state violations (missing DflashProposerState, non-zero ctx_len,
        // ring wrap needed, slot geometry mismatch, aliased accumulator) must be rejected.
        let mut missing_state = streams.clone();
        missing_state[1] = None;
        assert_eq!(
            validate_short_batch_dflash_preflight(base_inputs, &missing_state),
            ShortBatchDflashPlan::Unsupported
        );

        let mut nonzero_ctx = streams.clone();
        nonzero_ctx[0].as_mut().unwrap().ctx_len = 16;
        assert_eq!(
            validate_short_batch_dflash_preflight(base_inputs, &nonzero_ctx),
            ShortBatchDflashPlan::Unsupported
        );

        let mut wrap_needed = streams.clone();
        wrap_needed[1].as_mut().unwrap().ctx_capacity = 64;
        assert_eq!(
            validate_short_batch_dflash_preflight(base_inputs, &wrap_needed),
            ShortBatchDflashPlan::Unsupported
        );

        let mut bad_slot_bytes = streams.clone();
        bad_slot_bytes[0].as_mut().unwrap().ctx_slot_bytes = SLOT_BYTES - CAP_ROW_BYTES;
        assert_eq!(
            validate_short_batch_dflash_preflight(base_inputs, &bad_slot_bytes),
            ShortBatchDflashPlan::Unsupported
        );

        let mut aliased_acc = streams.clone();
        aliased_acc[1].as_mut().unwrap().ctx_hidden_acc =
            aliased_acc[0].as_ref().unwrap().ctx_hidden_acc;
        assert_eq!(
            validate_short_batch_dflash_preflight(base_inputs, &aliased_acc),
            ShortBatchDflashPlan::Unsupported
        );
    }

    #[test]
    fn disjoint_metadata_layout_bounds_and_compaction_are_strictly_non_aliasing() {
        const RESIDUAL: usize = 10_240;
        const TOP_K: usize = 10;
        const VOCAB: usize = 151_936;
        const MAX_ROWS: usize = 2048;

        let hidden_base = DevicePtr(0x1_0000_0000);
        let residual_base = DevicePtr(0x2_0000_0000);
        let scratch_base = DevicePtr(0x3_0000_0000);
        let logits_base = DevicePtr(0x4_0000_0000);

        let row_bytes = RESIDUAL * 2;
        let arena_hidden = MAX_ROWS * row_bytes;
        let arena_scratch = 2 * MAX_ROWS * TOP_K * 4 + MAX_ROWS * 32 + 65_536;
        let arena_logits = MAX_SHORT_BATCH_STREAMS * VOCAB * 2;

        for lengths in [
            vec![2usize, 2],
            vec![64, 128, 37, 91, 200],
            vec![128; 16],
            vec![1024, 1024],
        ] {
            for use_mrope in [false, true] {
                let layout = ShortBatchLayout::compute(
                    &lengths,
                    ShortBatchLayoutParams {
                        residual_width: RESIDUAL,
                        top_k: TOP_K,
                        use_mrope,
                        vocab_size: VOCAB,
                        arena_hidden_bytes: arena_hidden,
                        arena_residual_bytes: arena_hidden,
                        arena_scratch_bytes: arena_scratch,
                        arena_logits_bytes: arena_logits,
                    },
                )
                .unwrap();

                let expected_total: usize = lengths.iter().sum();
                assert_eq!(layout.total_rows, expected_total);
                assert_eq!(
                    layout.moe_routing_scratch_bytes,
                    2 * expected_total * TOP_K * 4
                );
                assert!(layout.meta_region_start_bytes >= layout.moe_routing_scratch_bytes);
                assert!(
                    layout
                        .meta_region_start_bytes
                        .is_multiple_of(ATTN_META_ALIGN)
                );
                assert!(layout.meta_region_end_bytes <= arena_scratch);

                for slice in &layout.slices {
                    assert!(slice.meta_offset_bytes >= layout.moe_routing_scratch_bytes);
                    assert!(slice.meta_offset_bytes.is_multiple_of(ATTN_META_ALIGN));
                }

                layout
                    .verify_device_non_aliasing(
                        hidden_base,
                        residual_base,
                        scratch_base,
                        logits_base,
                    )
                    .unwrap();

                // Verify ascending last-token compaction never overwrites an unread source row.
                let plan = compact_last_token_rows_plan(&layout.slices).unwrap();
                let mut rows: Vec<usize> = (0..layout.total_rows).collect();
                let expected_last: Vec<usize> = layout
                    .slices
                    .iter()
                    .map(|s| s.row_offset + s.len - 1)
                    .collect();
                for (src, dst) in plan {
                    assert!(src > dst, "src={src} must strictly exceed dst={dst}");
                    rows[dst] = rows[src];
                }
                assert_eq!(&rows[..lengths.len()], expected_last.as_slice());
            }
        }

        // Undersized scratch fails closed instead of aliasing MoE routing prefix.
        let tight_scratch = 2 * 256 * TOP_K * 4 + 64;
        assert!(
            ShortBatchLayout::compute(
                &[128, 128],
                ShortBatchLayoutParams {
                    residual_width: RESIDUAL,
                    top_k: TOP_K,
                    use_mrope: true,
                    vocab_size: VOCAB,
                    arena_hidden_bytes: arena_hidden,
                    arena_residual_bytes: arena_hidden,
                    arena_scratch_bytes: tight_scratch,
                    arena_logits_bytes: arena_logits,
                },
            )
            .is_err()
        );
    }

    #[test]
    fn deterministic_batch_disables_qwen4_short_batch_and_q12_kernel_batch() {
        use super::super::q12_batched_enabled_with_policy;
        use crate::layers::deterministic_batch::DeterministicBatchPolicy;

        assert!(!short_batch_enabled_with_policy(
            DeterministicBatchPolicy::Enabled,
            true
        ));
        assert!(short_batch_enabled_with_policy(
            DeterministicBatchPolicy::Disabled,
            true
        ));
        assert!(!short_batch_enabled_with_policy(
            DeterministicBatchPolicy::Disabled,
            false
        ));

        assert!(!q12_batched_enabled_with_policy(
            DeterministicBatchPolicy::Enabled,
            true
        ));
        assert!(q12_batched_enabled_with_policy(
            DeterministicBatchPolicy::Disabled,
            true
        ));
        assert!(!q12_batched_enabled_with_policy(
            DeterministicBatchPolicy::Disabled,
            false
        ));
    }
}
