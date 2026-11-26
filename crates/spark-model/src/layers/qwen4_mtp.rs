// SPDX-License-Identifier: AGPL-3.0-only

//! Native Qwen3.8-Flash-Next multi-token predictor.

use std::any::Any;
use std::collections::{HashMap, HashSet, VecDeque};

use anyhow::{Result, ensure};
use parking_lot::Mutex;
use spark_runtime::gpu::{DevicePtr, GpuBackend, GraphHandle, HostToDeviceCopy, KernelHandle};
use spark_runtime::kv_cache::{KvCacheConfig, KvCacheDtype, PagedKvCache};
use spark_runtime::weights::WeightStore;

use crate::layer::{AttnMetadataDev, ForwardContext, LayerState, TransformerLayer};
use crate::layers::qwen4_mtp_gate::{self, GatePolicy};
use crate::layers::qwen4_mtp_rows::{self, AfterVerifyPlan, RowPolicy};
use crate::layers::{Qwen4HyperConnection, ops};
use crate::speculative::{DraftProposer, ProposerState};
use crate::weight_map::{
    DenseWeight, QuantizedWeight, dense_auto, dense_f32_safe, quantize_to_nvfp4_cached,
};

#[path = "qwen4_mtp/gumbel_draft.rs"]
mod gumbel_draft;

pub struct Qwen4MtpState {
    /// Sequence-owned target rotary coordinates. Cache addresses and lengths
    /// remain physical even when an image prompt uses three-axis MRoPE.
    pub(crate) rotary_positions: crate::traits::RotaryPositions,
    layer_state: Box<dyn LayerState>,
    block_table: Vec<u32>,
    disk_block_ids: Vec<u32>,
    disk_last_offloaded_per_layer: Vec<u32>,
    seq_len: usize,
    last_num_drafted: usize,
    /// Every draft the last chain computed (`d_1..d_c`), for the K7
    /// committed-row rewrite (`ATLAS_QWEN4_MTP_RECONDITION`).
    last_drafts: Vec<u32>,
    /// Committed draft rows to rewrite from staged target hidden rows at the
    /// start of the next propose (set by `after_verify`).
    pending_replay: Option<PendingReplay>,
    /// The sequence's exact verify generation, bound before after_verify.
    verify_frame: Option<crate::traits::verify_owner::Ticket>,
    /// Owner-bound row publication; an unrelated request cannot satisfy it by count.
    staged_frame: Mutex<crate::traits::verify_owner::StagedFrame>,
    /// Sequence-scoped CRN Gumbel draft spec (`Some(Some(spec))` = coupled Gumbel,
    /// `Some(None)` = explicit greedy/non-Gumbel, `None` = fallback to ring).
    pending_gumbel_spec: Mutex<Option<Option<spark_runtime::sampler::gumbel::GumbelDraftSpec>>>,
}

/// A planned committed-row rewrite: the plan plus the frame's accepted count,
/// which locates the root row relative to the next chain's position. When
/// `after_verify` runs before `save_hidden_for_mtp` (`verify_k2/k3/k4_step`
/// partial/reject paths), `staged_at_trim` is false and `propose` upgrades
/// the fallback `KeepLast` plan once the staged hidden rows are noted.
#[derive(Clone, Copy, Debug)]
struct PendingReplay {
    owner: Option<crate::traits::verify_owner::Ticket>,
    row_policy: RowPolicy,
    computed: usize,
    staged_accepted: usize,
    plan: AfterVerifyPlan,
    accepted: usize,
    staged_at_trim: bool,
}

impl Qwen4MtpState {
    pub(crate) fn has_unconsumed_verify_frame(&self) -> bool {
        self.pending_replay.is_some() || self.staged_frame.lock().pending()
    }

    pub(crate) fn bind_verify_frame(
        &mut self,
        owner: Option<crate::traits::verify_owner::Ticket>,
    ) -> Result<()> {
        self.staged_frame
            .lock()
            .bind(owner)
            .map_err(|error| anyhow::anyhow!("MTP staged owner binding failed: {error:?}"))?;
        self.verify_frame = owner;
        Ok(())
    }

    pub(crate) fn publish_staged_frame(
        &self,
        owner: crate::traits::verify_owner::Ticket,
        rows: usize,
    ) -> Result<()> {
        self.staged_frame
            .lock()
            .publish(owner, rows)
            .map_err(|error| anyhow::anyhow!("MTP staged publication failed: {error:?}"))
    }

    pub(crate) fn check_staged_frame(
        &self,
        owner: crate::traits::verify_owner::Ticket,
        rows: usize,
    ) -> Result<()> {
        self.staged_frame
            .lock()
            .check_publish(owner, rows)
            .map_err(|error| anyhow::anyhow!("MTP staged publication preflight failed: {error:?}"))
    }

    fn take_staged_frame(
        &self,
        owner: Option<crate::traits::verify_owner::Ticket>,
        rows: usize,
    ) -> Result<bool> {
        self.staged_frame
            .lock()
            .take(owner, rows)
            .map_err(|error| anyhow::anyhow!("MTP staged consumption failed: {error:?}"))
    }

    pub(crate) fn discard_verify_frame(&mut self) {
        self.verify_frame = None;
        self.staged_frame.lock().clear();
        self.pending_replay = None;
        *self.pending_gumbel_spec.lock() = None;
    }

    /// Stamp this sequence's upcoming CRN Gumbel draft spec (`None` = greedy / uncoupled).
    pub fn set_gumbel_spec(&self, spec: Option<spark_runtime::sampler::gumbel::GumbelDraftSpec>) {
        *self.pending_gumbel_spec.lock() = Some(spec);
    }

    fn resolve_gumbel_spec(
        &self,
        position: usize,
    ) -> Option<spark_runtime::sampler::gumbel::GumbelDraftSpec> {
        if !gumbel_draft::coupled_drafting_enabled() {
            return None;
        }
        match *self.pending_gumbel_spec.lock() {
            Some(published) => published.filter(|s| s.is_active()),
            None => gumbel_draft::active_spec_for(position),
        }
    }

    fn clear_gumbel_spec(&self) {
        *self.pending_gumbel_spec.lock() = None;
    }

    pub(crate) fn snapshot_dedup_draft(&self) -> (usize, usize, Vec<u32>) {
        (
            self.seq_len,
            self.last_num_drafted,
            self.last_drafts.clone(),
        )
    }

    pub(crate) fn apply_dedup_draft(
        &mut self,
        seq_len: usize,
        last_num_drafted: usize,
        last_drafts: &[u32],
    ) {
        self.staged_frame.lock().clear();
        self.seq_len = seq_len;
        self.last_num_drafted = last_num_drafted;
        self.last_drafts.clear();
        self.last_drafts.extend_from_slice(last_drafts);
    }
}

impl ProposerState for Qwen4MtpState {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

impl Qwen4MtpState {
    /// Prepare state when an external zero-cost drafter (e.g. N-gram suffix
    /// copy-drafter) supplies the current cycle's drafts instead of launching
    /// the GPU MTP chain.
    pub fn notify_external_draft(&mut self) {
        if let Some(pending) = self.pending_replay.take()
            && pending.computed > 0
            && pending.staged_at_trim
        {
            let keep_last = (pending.accepted + 1).min(pending.computed);
            if keep_last > pending.plan.keep_rows {
                self.seq_len += keep_last - pending.plan.keep_rows;
            }
        }
        self.verify_frame = None;
        self.last_num_drafted = 0;
        self.last_drafts.clear();
        self.staged_frame.lock().clear();
        self.clear_gumbel_spec();
    }

    pub fn notify_external_draft_with_tokens(&mut self, drafts: &[u32]) {
        self.notify_external_draft();
        self.last_drafts.extend_from_slice(drafts);
    }

    /// Record suffix tokens grafted onto the tail of a partial neural MTP
    /// proposal without altering `last_num_drafted` (only the neural prefix
    /// wrote MTP KV cache rows).
    pub fn notify_grafted_suffix(&mut self, suffix: &[u32]) {
        self.last_drafts.extend_from_slice(suffix);
    }

    /// Synchronize `last_drafts` prefix when host grammar coalescence splices
    /// forced/legal tokens into a proposed chain before installation.
    pub fn sync_spliced_drafts(&mut self, drafts: &[u32]) {
        let n = drafts.len().min(self.last_drafts.len());
        self.last_drafts[..n].copy_from_slice(&drafts[..n]);
    }
}

/// Zero-cost CPU N-gram suffix copy-drafter (`ATLAS_QWEN4_MTP_NGRAM`, default
/// enabled when `ATLAS_QWEN4_VERIFY_FAST_HOST=1` unless `ATLAS_QWEN4_MTP_NGRAM=0`).
///
/// Searches the committed token history `(prefix, last_token)` in a single
/// backward pass over an 8K window for a `6..=12`-gram suffix match, or a
/// repeated `4..=5`-gram match in the last 128 tokens with agreeing 2-token
/// continuation. Returns `num_drafts` copied tokens in `< 2 us` on hit.
pub fn try_ngram_propose(prefix: &[u32], last_token: u32, num_drafts: usize) -> Option<Vec<u32>> {
    static NGRAM_MIN_N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    let min_n = *NGRAM_MIN_N.get_or_init(|| {
        if let Ok(v) = std::env::var("ATLAS_QWEN4_MTP_NGRAM") {
            let s = v.trim();
            if s == "0" || s.eq_ignore_ascii_case("off") {
                return 0;
            }
            if let Ok(n) = s.parse::<usize>()
                && (3..=16).contains(&n)
            {
                tracing::info!("QWEN4_MTP_NGRAM_ENGAGED min_n={n}");
                return n;
            }
        }
        if crate::layers::qwen4_verify_fast_host_selected() {
            tracing::info!("QWEN4_MTP_NGRAM_ENGAGED min_n=6 (with local n=4 repeated fallback)");
            6
        } else {
            0
        }
    });
    if min_n == 0 || num_drafts == 0 {
        return None;
    }
    let plen = prefix.len();
    let pat_prev = min_n - 1;
    // 1. Single backward pass for n >= min_n (default 6..=12) across an 8192-token window.
    if plen > pat_prev + num_drafts {
        let pat_slice = &prefix[plen - pat_prev..plen];
        let pat_first = pat_slice[0];
        let max_end = plen - num_drafts - 1;
        let start_end = plen.saturating_sub(8192).max(pat_prev);
        let mut best_len = 0usize;
        let mut best_start = 0usize;
        for idx_end in (start_end..=max_end).rev() {
            if prefix[idx_end] == last_token
                && prefix[idx_end - pat_prev] == pat_first
                && &prefix[idx_end - pat_prev..idx_end] == pat_slice
            {
                let mut mlen = min_n;
                while mlen < 12
                    && idx_end >= mlen
                    && plen >= mlen
                    && prefix[idx_end - mlen] == prefix[plen - mlen]
                {
                    mlen += 1;
                }
                if mlen > best_len {
                    best_len = mlen;
                    best_start = idx_end + 1;
                    if best_len == 12 {
                        break;
                    }
                }
            }
        }
        if best_len >= min_n {
            return Some(prefix[best_start..best_start + num_drafts].to_vec());
        }
    }
    // 2. Local 128-token fallback for repeated 4-gram suffix with agreeing 2-token continuation.
    if min_n <= 6 && plen > 3 + num_drafts && num_drafts >= 2 {
        let pat3 = &prefix[plen - 3..plen];
        let p0 = pat3[0];
        let max_end = plen - num_drafts - 1;
        let start_end = plen.saturating_sub(128).max(3);
        let mut first_cand: Option<&[u32]> = None;
        for idx_end in (start_end..=max_end).rev() {
            if prefix[idx_end] == last_token
                && prefix[idx_end - 3] == p0
                && &prefix[idx_end - 3..idx_end] == pat3
            {
                let cand = &prefix[idx_end + 1..idx_end + 1 + num_drafts];
                if let Some(first) = first_cand {
                    if cand[0] == first[0] && cand[1] == first[1] {
                        return Some(first.to_vec());
                    }
                } else {
                    first_cand = Some(cand);
                }
            }
        }
    }
    None
}

/// Opt-in draft block truncation (`ATLAS_QWEN4_MTP_BOUNDED_BLOCKS=1`).
/// Defaults to 0 / off so full state.block_table is preserved across long contexts.
pub fn mtp_bounded_blocks_selected() -> bool {
    static SELECTED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *SELECTED.get_or_init(|| {
        std::env::var("ATLAS_QWEN4_MTP_BOUNDED_BLOCKS")
            .ok()
            .and_then(|v| v.trim().parse::<u32>().ok())
            .map(|v| v != 0)
            .unwrap_or(false)
    })
}

/// Qwen4 MTP sparse-attention (QSA) mode for `mtp.layers.0`.
///
/// Controlled by `ATLAS_QWEN4_MTP_QSA` (`1` enables; unset/`0` -> `Disabled`)
/// and `ATLAS_QWEN4_MTP_DRAFT_SELECTION_REUSE` (default `1` when
/// `ATLAS_QWEN4_MTP_QSA=1`; `0` -> `PerStep`). Invalid values fail closed to
/// `Disabled`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MtpQsaMode {
    /// Default: `qwen4_qsa_required: false` (dense/legacy MTP path).
    Disabled,
    /// Full QSA indexer update + selection on every committed and draft step.
    PerStep,
    /// Full QSA selection on step 0 (snapshotted into `reused_draft_indices`),
    /// then draft steps `1..num_drafts` skip index/score/select and attend over
    /// `OUTPUT_WIDTH + step` indices with appended speculative draft positions.
    DraftSelectionReuse,
}

pub const MTP_QSA_SELECTOR: &str = "ATLAS_QWEN4_MTP_QSA";
pub const MTP_DRAFT_SELECTION_REUSE_SELECTOR: &str = "ATLAS_QWEN4_MTP_DRAFT_SELECTION_REUSE";

/// Strict parser for `ATLAS_QWEN4_MTP_QSA` + `ATLAS_QWEN4_MTP_DRAFT_SELECTION_REUSE`.
pub fn parse_mtp_qsa_mode(
    mtp_qsa: Option<&str>,
    draft_selection_reuse: Option<&str>,
) -> Result<MtpQsaMode> {
    let qsa_enabled = match mtp_qsa.map(str::trim) {
        None | Some("0") => false,
        Some("1") => true,
        Some(other) => {
            anyhow::bail!("invalid {MTP_QSA_SELECTOR}={other:?}: expected \"0\" or \"1\"");
        }
    };
    if !qsa_enabled {
        return Ok(MtpQsaMode::Disabled);
    }
    match draft_selection_reuse.map(str::trim) {
        None | Some("1") => Ok(MtpQsaMode::DraftSelectionReuse),
        Some("0") => Ok(MtpQsaMode::PerStep),
        Some(other) => {
            anyhow::bail!(
                "invalid {MTP_DRAFT_SELECTION_REUSE_SELECTOR}={other:?}: expected \"0\" or \"1\""
            );
        }
    }
}

/// Fail-closed resolver: returns `MtpQsaMode::Disabled` if either selector is invalid.
pub fn resolve_mtp_qsa_mode(
    mtp_qsa: Option<&str>,
    draft_selection_reuse: Option<&str>,
) -> MtpQsaMode {
    parse_mtp_qsa_mode(mtp_qsa, draft_selection_reuse).unwrap_or(MtpQsaMode::Disabled)
}

/// Process-wide cached `MtpQsaMode` read from environment variables.
pub fn mtp_qsa_mode() -> MtpQsaMode {
    static MODE: std::sync::OnceLock<MtpQsaMode> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| {
        let qsa_raw = std::env::var(MTP_QSA_SELECTOR).ok();
        let reuse_raw = std::env::var(MTP_DRAFT_SELECTION_REUSE_SELECTOR).ok();
        match parse_mtp_qsa_mode(qsa_raw.as_deref(), reuse_raw.as_deref()) {
            Ok(mode) => {
                if mode != MtpQsaMode::Disabled {
                    tracing::info!(
                        "QWEN4_MTP_QSA_ENGAGED mode={mode:?} \
                         {MTP_QSA_SELECTOR}={qsa_raw:?} \
                         {MTP_DRAFT_SELECTION_REUSE_SELECTOR}={reuse_raw:?}"
                    );
                }
                mode
            }
            Err(err) => {
                tracing::warn!(
                    "QWEN4_MTP_QSA_REFUSED reason=invalid_selector ({err}); failing closed to Disabled"
                );
                MtpQsaMode::Disabled
            }
        }
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DraftStepReuseAction {
    SnapshotStep0,
    ReuseStep {
        base_seq_len: usize,
        step_offset: usize,
        max_drafts: usize,
    },
}

/// Minimum sequence length at which Qwen4 switches from dense fallback to QSA.
const MTP_QSA_SPARSE_THRESHOLD: usize = 2048;

/// Decide `(qwen4_qsa_required, reuse_action)` for a single MTP draft step.
fn draft_step_qsa_plan(
    mode: MtpQsaMode,
    has_qsa_side_cache: bool,
    first_seq_len: usize,
    step: usize,
    num_drafts: usize,
) -> (bool, Option<DraftStepReuseAction>) {
    if !has_qsa_side_cache || mode == MtpQsaMode::Disabled {
        return (false, None);
    }
    if mode == MtpQsaMode::PerStep {
        return (true, None);
    }
    let max_drafts = num_drafts.saturating_sub(1);
    let reuse_eligible = first_seq_len >= MTP_QSA_SPARSE_THRESHOLD
        && (1..=crate::layers::qwen4_qsa::QSA_REUSE_MAX_DRAFTS).contains(&max_drafts);
    if step == 0 {
        (
            true,
            reuse_eligible.then_some(DraftStepReuseAction::SnapshotStep0),
        )
    } else if reuse_eligible && step <= max_drafts {
        (
            true,
            Some(DraftStepReuseAction::ReuseStep {
                base_seq_len: first_seq_len,
                step_offset: step,
                max_drafts,
            }),
        )
    } else if first_seq_len.saturating_add(num_drafts) <= MTP_QSA_SPARSE_THRESHOLD {
        // Pure dense prefix (< 2048): skip QSA side-cache writes on speculative drafts.
        (false, None)
    } else {
        // 2048-straddling boundary chain or wider-than-reuse chain: PerStep fallback.
        (true, None)
    }
}

/// Checked `OUTPUT_WIDTH + step` index width for an MTP draft step under
/// `DraftSelectionReuse`.
pub fn checked_mtp_reused_index_width(step: usize, max_drafts: usize) -> Result<usize> {
    crate::layers::qwen4_qsa::checked_reused_index_width(step, max_drafts)
}

/// Capacity of the MTP head's page-locked H2D staging ring (`8 MiB`), matching
/// the target model's fast-H2D ring so `propose_batch` / `propose_chain` upload
/// attention and Gumbel metadata via `cuMemcpyHtoDAsync` without draining the GPU.
const MTP_PINNED_H2D_RING_BYTES: usize = 8 * 1024 * 1024;

struct MtpPinnedH2dRing {
    buffer: spark_runtime::gpu::PinnedHostBuffer,
    cursor: usize,
    /// Per-half CUDA events fencing the latest async H2D DMA reading from
    /// half 0 (`[0..mid)`) and half 1 (`[mid..cap)`).
    half_events: [u64; 2],
    /// Whether `half_events[h]` tracks at least one async H2D DMA in half `h`
    /// that has not yet been confirmed complete via `poll_event`.
    half_in_flight: [bool; 2],
    /// Stream on which the latest in-flight DMA in half `h` was recorded.
    half_stream: [Option<u64>; 2],
    /// Set when the cursor wraps past half `h` while it still has in-flight
    /// DMAs from the prior pass; the next pass must confirm `half_events[h]`
    /// completed before overwriting any byte in half `h`.
    half_needs_fence: [bool; 2],
    /// Permanently disables the ring after an unfenceable stream/event failure
    /// or after [`Self::destroy`] releases its events.
    poisoned: bool,
}

impl MtpPinnedH2dRing {
    fn new(gpu: &dyn GpuBackend, bytes: usize) -> Option<Self> {
        let buffer = gpu.alloc_host_pinned(bytes).ok()?;
        let event0 = gpu.create_event().ok()?;
        let event1 = match gpu.create_event() {
            Ok(ev) => ev,
            Err(_) => {
                if event0 != 0 {
                    let _ = gpu.destroy_event(event0);
                }
                return None;
            }
        };
        Some(Self {
            buffer,
            cursor: 0,
            half_events: [event0, event1],
            half_in_flight: [false, false],
            half_stream: [None, None],
            half_needs_fence: [false, false],
            poisoned: false,
        })
    }

    fn destroy(&mut self, gpu: &dyn GpuBackend) {
        self.poisoned = true;
        for ev in &mut self.half_events {
            let event = *ev;
            if event != 0 {
                if let Err(error) = gpu.destroy_event(event) {
                    tracing::error!(
                        "MtpPinnedH2dRing::destroy: destroy_event({event}) failed: {error:#}"
                    );
                }
                *ev = 0;
            }
        }
        self.half_in_flight = [false, false];
        self.half_stream = [None, None];
        self.half_needs_fence = [false, false];
    }

    fn plan_range(&self, bytes: usize) -> Option<(usize, usize, bool)> {
        let cap = self.buffer.len();
        if bytes > cap {
            return None;
        }
        let aligned = self.cursor.checked_add(7)? & !7;
        let (start, wrapped) = if aligned.checked_add(bytes).is_some_and(|end| end <= cap) {
            (aligned, false)
        } else {
            (0, true)
        };
        let end = start.checked_add(bytes)?;
        if end > cap {
            return None;
        }
        Some((start, end, wrapped))
    }

    fn copy_h2d_group(
        &mut self,
        gpu: &dyn GpuBackend,
        copies: &[HostToDeviceCopy<'_>],
        stream: u64,
    ) -> Result<bool> {
        if copies.is_empty() {
            return Ok(true);
        }
        if self.poisoned || gpu.stream_is_capturing(stream) {
            return Ok(false);
        }
        let mut total_aligned = 0usize;
        for copy in copies {
            let Some(aligned) = copy.src().len().checked_add(7).map(|n| n & !7) else {
                return Ok(false);
            };
            match total_aligned.checked_add(aligned) {
                Some(sum) => total_aligned = sum,
                None => return Ok(false),
            }
        }
        if total_aligned == 0 {
            return Ok(true);
        }
        let Some((start, end, wrapped)) = self.plan_range(total_aligned) else {
            return Ok(false);
        };
        let cap = self.buffer.len();
        let mid = (cap / 2) & !7;
        let touches = [start < mid, end > mid];

        // Confirm every half we are about to reuse after wrap-around (or on a
        // different stream) has finished all prior async DMAs before writing.
        for (h, &touched) in touches.iter().enumerate() {
            if !touched {
                continue;
            }
            let must_confirm_complete = self.half_in_flight[h]
                && (wrapped || self.half_needs_fence[h] || self.half_stream[h] != Some(stream));
            if must_confirm_complete {
                if !gpu.poll_event(self.half_events[h])? {
                    return Ok(false);
                }
                self.half_in_flight[h] = false;
                self.half_needs_fence[h] = false;
                self.half_stream[h] = None;
            }
        }

        let mut cursor = start;
        let mut enqueued = false;
        let mut copy_err: Option<anyhow::Error> = None;
        for copy in copies {
            let src = copy.src();
            if src.is_empty() {
                continue;
            }
            let len = src.len();
            // SAFETY: `plan_range` validated `[start .. end) <= self.buffer.len()`,
            // so `[cursor .. cursor + len)` is within `self.buffer` and disjoint
            // from `src`. Any prior async DMA reading from the overlapped
            // half(ves) was confirmed complete via `poll_event` above.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    src.as_ptr(),
                    self.buffer.as_mut_ptr().add(cursor),
                    len,
                );
            }
            let pinned_slice = match self.buffer.pinned_slice_range(cursor, len) {
                Ok(slice) => slice,
                Err(err) => {
                    copy_err = Some(err);
                    break;
                }
            };
            // SAFETY: `self.buffer` is backend-owned page-locked memory held
            // until teardown synchronizes streams and calls
            // `release_rejection_resources`. Each touched half records its
            // CUDA event on `stream` below and is polled to completion before
            // wrap-around or cross-stream reuse can overwrite those bytes.
            if let Err(err) = unsafe { gpu.copy_h2d_pinned_async(pinned_slice, copy.dst(), stream) }
            {
                copy_err = Some(err);
                break;
            }
            enqueued = true;
            cursor = (cursor + len + 7) & !7;
        }

        if enqueued {
            if wrapped {
                for (h, &touched) in touches.iter().enumerate() {
                    if !touched && self.half_in_flight[h] {
                        self.half_needs_fence[h] = true;
                    }
                }
            }
            self.cursor = end;
            for (h, &touched) in touches.iter().enumerate() {
                if !touched {
                    continue;
                }
                if let Err(record_err) = gpu.record_event(self.half_events[h], stream) {
                    if gpu.synchronize(stream).is_err() {
                        self.poisoned = true;
                    } else {
                        self.half_in_flight[h] = false;
                        self.half_needs_fence[h] = false;
                        self.half_stream[h] = None;
                    }
                    return Err(copy_err.unwrap_or(record_err));
                }
                self.half_in_flight[h] = true;
                self.half_stream[h] = Some(stream);
                self.half_needs_fence[h] = false;
            }
        }

        if let Some(err) = copy_err {
            return Err(err);
        }
        Ok(true)
    }
}

/// Parse `ATLAS_QWEN4_MTP_LM_HEAD_TC`: opt-in (`"1"`) only.
///
/// `w4a16_gemm_pipe` issues strided 16-byte global loads across `N`, saturating
/// memory bandwidth at ~53 GB/s on GB10 (~2.0-3.5 ms per draft step at `b=16`).
/// Defaulting off routes `propose_batch` LM head through coalesced single-sweep
/// exact GEMV (`w4a16_gemv_batch_logits_exact_with`, ~225 GB/s, 0-ULP exact).
fn parse_mtp_lm_head_tc_pipe_env(raw: Option<&str>) -> bool {
    raw.map(str::trim) == Some("1")
}

fn mtp_lm_head_tc_pipe_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        parse_mtp_lm_head_tc_pipe_env(std::env::var("ATLAS_QWEN4_MTP_LM_HEAD_TC").ok().as_deref())
    })
}

#[inline]
fn should_use_mtp_lm_head_pipe(
    enabled: bool,
    batch_rows: usize,
    hidden: usize,
    pipe_available: bool,
    input: DevicePtr,
    weight: DevicePtr,
    output: DevicePtr,
) -> bool {
    enabled
        && batch_rows >= 8
        && pipe_available
        && hidden.is_multiple_of(32)
        && input.0.is_multiple_of(16)
        && weight.0.is_multiple_of(16)
        && output.0.is_multiple_of(16)
}

#[inline]
const fn should_use_mtp_lm_head_coalesced(
    tc_active: bool,
    batch_rows: usize,
    hidden: usize,
) -> bool {
    tc_active
        && batch_rows >= 6
        && batch_rows <= crate::layers::QWEN4_NATIVE_MAX_ROWS
        && hidden > 0
        && hidden.is_multiple_of(32)
}

/// Maximum `batch_rows * hc_count` admitted to coalesced NVFP4 MTP FC
/// (`batch_rows <= 64` at `hc_count == 4`), executed via
/// `grid.y = div_ceil(hc_rows, tile_m)` on `w4a16_gemm_coalesced*`.
const MTP_FC_COALESCED_MAX_HC_ROWS: usize = 256;

#[inline]
#[must_use]
const fn should_use_batched_argmax_prob(rows: usize, batched_available: bool) -> bool {
    rows > 1 && batched_available
}

#[inline]
const fn should_use_mtp_fc_coalesced(
    tc_dense_scope: bool,
    batch_rows: usize,
    hc_count: usize,
    hidden: usize,
    qkv_output_bytes: usize,
    broadcast_add_available: bool,
) -> bool {
    if !tc_dense_scope
        || batch_rows < 6
        || hc_count != 4
        || hidden == 0
        || !hidden.is_multiple_of(32)
        || !broadcast_add_available
    {
        return false;
    }
    let Some(hc_rows) = batch_rows.checked_mul(hc_count) else {
        return false;
    };
    if hc_rows > MTP_FC_COALESCED_MAX_HC_ROWS {
        return false;
    }
    let Some(embed_elems) = batch_rows.checked_mul(hidden) else {
        return false;
    };
    let Some(embed_bytes) = embed_elems.checked_mul(2) else {
        return false;
    };
    embed_bytes <= qkv_output_bytes
}

/// Environment selector for batched Native MTP proposer CUDA-graph capture
/// (`"0"` / `"false"` / `"off"` disables; `"1"` / `"true"` / `"on"` forces on;
/// unset / `"auto"` auto-enables on `active_sm_count() > 64` while keeping GB10
/// `<= 64` SMs default OFF).
pub const MTP_GRAPH_SELECTOR: &str = "ATLAS_QWEN4_MTP_GRAPH";

/// Minimum SM count for auto-enabling batched MTP CUDA-graph capture.
const MTP_GRAPH_AUTO_SM_THRESHOLD: u32 = 64;

/// Maximum cached CUDA graphs per [`Qwen4MtpHead`] before FIFO eviction.
const MTP_GRAPH_CACHE_CAP: usize = 32;

/// Minimum bucketed block-table stride (`max_blocks_per_seq`) for graphed MTP
/// launches so growing sequences hit one cached graph across block boundaries.
const MTP_GRAPH_MIN_BT_BUCKET: usize = 16;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MtpGraphPolicy {
    Disabled,
    Auto,
    Enabled,
}

#[must_use]
fn parse_mtp_graph_policy(raw: Option<&str>) -> MtpGraphPolicy {
    match raw.map(str::trim) {
        Some(v) if v == "0" || v.eq_ignore_ascii_case("false") || v.eq_ignore_ascii_case("off") => {
            MtpGraphPolicy::Disabled
        }
        Some(v) if v == "1" || v.eq_ignore_ascii_case("true") || v.eq_ignore_ascii_case("on") => {
            MtpGraphPolicy::Enabled
        }
        _ => MtpGraphPolicy::Auto,
    }
}

#[must_use]
const fn resolve_mtp_graph_enabled(policy: MtpGraphPolicy, sm_count: u32) -> bool {
    match policy {
        MtpGraphPolicy::Disabled => false,
        MtpGraphPolicy::Enabled => true,
        MtpGraphPolicy::Auto => sm_count > MTP_GRAPH_AUTO_SM_THRESHOLD,
    }
}

#[must_use]
fn mtp_graph_enabled() -> bool {
    static POLICY: std::sync::OnceLock<MtpGraphPolicy> = std::sync::OnceLock::new();
    let policy = *POLICY.get_or_init(|| {
        let raw = std::env::var(MTP_GRAPH_SELECTOR).ok();
        parse_mtp_graph_policy(raw.as_deref())
    });
    resolve_mtp_graph_enabled(policy, atlas_core::device::active_sm_count())
}

/// Bucket `max_mb` up to a power of two (`>= 16`) when the padded per-step
/// metadata slab still fits inside `scratch_bytes`; otherwise return `max_mb`.
#[must_use]
fn mtp_graph_bt_blocks_cap(
    max_mb: usize,
    batch_size: usize,
    step_count: usize,
    bt_off: usize,
    scratch_bytes: usize,
) -> usize {
    let bucket = max_mb.max(MTP_GRAPH_MIN_BT_BUCKET).next_power_of_two();
    let fits = bucket
        .checked_mul(4)
        .and_then(|row_bytes| batch_size.checked_mul(row_bytes))
        .and_then(|bt_bytes| bt_off.checked_add(bt_bytes))
        .and_then(|raw_stride| raw_stride.checked_next_multiple_of(256))
        .and_then(|stride| step_count.checked_mul(stride))
        .and_then(|meta_bytes| CHAIN_META_OFFSET.checked_add(meta_bytes))
        .is_some_and(|end| end <= scratch_bytes);
    if fits { bucket } else { max_mb }
}

/// Shape- and address-exact key for cached multi-step batched MTP CUDA graphs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct MtpGraphKey {
    batch_size: usize,
    step_count: usize,
    want_prob: bool,
    bt_blocks_cap: u32,
    vocab: usize,
    write_step_token_out: bool,
    tc_dense_scope: bool,
    coalesced_fc: bool,
    coalesced_lm_head: bool,
    scratch_ptr: u64,
    hidden_ptr: u64,
    residual_ptr: u64,
    logits_ptr: u64,
    k_pool_ptr: u64,
    v_pool_ptr: u64,
    stream: u64,
}

#[derive(Default)]
struct MtpGraphCache {
    warmed: HashSet<MtpGraphKey>,
    graphs: HashMap<MtpGraphKey, GraphHandle>,
    order: VecDeque<MtpGraphKey>,
    poisoned: HashSet<MtpGraphKey>,
}

impl MtpGraphCache {
    fn get(&self, key: &MtpGraphKey) -> Option<GraphHandle> {
        self.graphs.get(key).copied()
    }

    fn is_poisoned(&self, key: &MtpGraphKey) -> bool {
        self.poisoned.contains(key)
    }

    fn mark_warmed(&mut self, key: MtpGraphKey) -> bool {
        self.warmed.insert(key)
    }

    fn mark_poisoned(&mut self, key: MtpGraphKey) {
        self.poisoned.insert(key);
    }

    fn insert(&mut self, gpu: &dyn GpuBackend, key: MtpGraphKey, graph: GraphHandle) {
        if let Some(prev) = self.graphs.insert(key, graph) {
            if let Err(error) = gpu.destroy_graph(prev) {
                tracing::error!(
                    "Qwen4MtpHead: destroy_graph({prev:?}) on replace failed: {error:#}"
                );
            }
            return;
        }
        self.order.push_back(key);
        while self.graphs.len() > MTP_GRAPH_CACHE_CAP {
            let Some(oldest) = self.order.pop_front() else {
                break;
            };
            if let Some(evicted) = self.graphs.remove(&oldest)
                && let Err(error) = gpu.destroy_graph(evicted)
            {
                tracing::error!(
                    "Qwen4MtpHead: destroy_graph({evicted:?}) on eviction failed: {error:#}"
                );
            }
            self.warmed.remove(&oldest);
        }
    }

    fn clear(&mut self, gpu: &dyn GpuBackend) {
        for (_, graph) in self.graphs.drain() {
            if let Err(error) = gpu.destroy_graph(graph) {
                tracing::error!(
                    "Qwen4MtpHead::release_rejection_resources: destroy_graph({graph:?}) failed: {error:#}"
                );
            }
        }
        self.order.clear();
        self.warmed.clear();
        self.poisoned.clear();
    }
}

/// Static parameters for one capture-safe batched MTP multi-step replay.
struct MtpBatchedChainLaunch<'a> {
    batch_size: usize,
    step_count: usize,
    want_prob: bool,
    bt_blocks_cap: usize,
    stride: usize,
    last_tok_off: usize,
    slot_off: usize,
    seq_len_off: usize,
    hidden_ptrs_off: usize,
    bt_off: usize,
    vocab: usize,
    meta_base: DevicePtr,
    argmax_base: DevicePtr,
    prob_base: DevicePtr,
    token_out_base: Option<DevicePtr>,
    base_seq_lens: &'a [usize],
}

pub struct Qwen4MtpHead {
    pre_fc_norm_embedding: DenseWeight,
    pre_fc_norm_hidden: DenseWeight,
    fc_embedding: QuantizedWeight,
    fc_hidden: QuantizedWeight,
    layer: Box<dyn TransformerLayer>,
    final_mixer: Qwen4HyperConnection,
    embed_tokens: DenseWeight,
    lm_head_nvfp4: QuantizedWeight,
    mtp_vocab_size: u32,
    kv_cache: Mutex<PagedKvCache>,
    /// Reusable host staging buffer for `propose_batch` and `propose_chain`
    /// metadata packing, eliminating `0.8–2.0 MiB/step` heap churn.
    packed_meta_scratch: Mutex<Vec<u8>>,
    /// Dedicated page-locked ring for async stream-ordered metadata H2D copies.
    pinned_h2d_ring: Mutex<Option<MtpPinnedH2dRing>>,
    /// Per-`(batch_size, step_count, want_prob, ...)` CUDA graph cache for
    /// uniform batched MTP proposer chains (`ATLAS_QWEN4_MTP_GRAPH`).
    graph_cache: Mutex<MtpGraphCache>,
    rms_norm_k: KernelHandle,
    pre_fc_dual_norm_k: KernelHandle,
    embed_dual_rms_norm_fused_k: KernelHandle,
    embed_dual_rms_norm_rows_packed_k: KernelHandle,
    fc_dual_broadcast_rows_k: KernelHandle,
    w4a16_gemv_k: KernelHandle,
    w4a16_gemv_sw_k: KernelHandle,
    gemv_sw: bool,
    w4a16_gemv_exact_m4_k: KernelHandle,
    w4a16_exact_fc_kernels: ops::W4a16ExactLmHeadKernels,
    /// Opt-in (`ATLAS_QWEN4_MTP_LM_HEAD_TC=1`) `w4a16_gemm_pipe` kernel for A/B
    /// testing MTP LM head at `b >= 8` (default off in favor of coalesced exact GEMV).
    w4a16_gemm_pipe_k: KernelHandle,
    residual_add_k: KernelHandle,
    residual_add_broadcast_hc_k: KernelHandle,
    argmax_k: KernelHandle,
    argmax_rows_k: KernelHandle,
    argmax_prob_k: KernelHandle,
    argmax_prob_batched_k: KernelHandle,
    mtp_gumbel_argmax_k: KernelHandle,
    embed_from_argmax_k: KernelHandle,
    /// The exact non-YaRN MRoPE symbol used by the inner attention layer.
    /// That layer probes it as optional, so image admission must preserve and
    /// check availability before any proposer device write.
    mrope_kernel_available: bool,
}

/// Scratch layout of the sync-free draft chain. The single-token MoE router
/// writes the first bytes of scratch and the per-step attention metadata
/// starts at `CHAIN_META_OFFSET` (the same base the one-step path uses). That
/// base is the target's `STEP_META_SCRATCH_OFFSET`: every draft step and chain
/// uploads its metadata before launching, and the target re-uploads its own
/// before each decode/verify frame, so neither may read the other's.
const CHAIN_ARGMAX_OFFSET: usize = 16384;
const CHAIN_TOKEN_OFFSET: usize = CHAIN_ARGMAX_OFFSET + 1024;
/// Draft probabilities live in the upper half of the 1 KiB argmax span.
const CHAIN_PROB_OFFSET: usize = CHAIN_ARGMAX_OFFSET + 512;
const CHAIN_PROB_SPAN: usize = 512;
const CHAIN_META_OFFSET: usize = crate::layer::STEP_META_SCRATCH_OFFSET;
const POSITION_H_OFFSET: usize = 20;
const POSITION_W_OFFSET: usize = 24;
const MAX_ROTARY_METADATA_ROWS: usize = 1_048_576;

/// Where one MTP step takes its inputs from and leaves its draft.
#[derive(Clone, Copy)]
struct ChainStep {
    /// Pre-uploaded attention metadata for this step.
    meta: DevicePtr,
    /// Number of block-table entries uploaded in `meta`.
    max_blocks_per_seq: u32,
    /// Device u32 holding the previous step's argmax (None: host token).
    device_token: Option<DevicePtr>,
    /// Device u32 receiving this step's argmax; no host readback.
    argmax_out: DevicePtr,
    /// Device f32 receiving the argmax's softmax probability (draft confidence).
    prob_out: Option<DevicePtr>,
    /// Optional CRN Gumbel perturbation `(spec, abs_pos)` for this step.
    gumbel: Option<(spark_runtime::sampler::gumbel::GumbelDraftSpec, u64)>,
}

/// Number of drafts kept under the active gate policy (`0..=probs.len()`).
fn confident_prefix(probs: &[f32], threshold: f32) -> usize {
    qwen4_mtp_gate::confident_prefix_env(probs, threshold)
}

/// Number of drafts kept for a batched (`b >= 2`) cohort-uniform chain
/// (`1..=probs.len()` for non-empty `probs`, never ejecting a sequence to `0`).
#[cfg(test)]
fn confident_prefix_batch(probs: &[f32], threshold: f32) -> usize {
    qwen4_mtp_gate::confident_prefix_batch_env(probs, threshold)
}

/// Confidence-gating plan for one MTP draft chain.
#[derive(Clone, Copy, Debug, PartialEq)]
struct ChainConfidenceGate {
    /// Whether to compute and copy back per-step raw top-1 softmax probabilities.
    want_prob: bool,
    /// Optional mid-chain early-stop threshold (`Some` triggers a per-step D2H sync).
    early: Option<f32>,
    /// Whether to truncate the finished draft prefix by raw top-1 probability.
    apply_truncate: bool,
}

/// Resize and zero-fill a reusable host metadata staging buffer to `len` bytes
/// without releasing its existing capacity across draft steps.
fn prepare_packed_meta_scratch(buf: &mut Vec<u8>, len: usize) -> &mut [u8] {
    if buf.len() < len {
        buf.resize(len, 0);
    }
    let slice = &mut buf[..len];
    slice.fill(0);
    slice
}

/// Whether confidence gating runs for a single sequence (`b == 1`) or a
/// cohort-uniform batched proposal (`b >= 2`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ChainConfidenceScope {
    /// Single-sequence chain (`b == 1`): honors `ATLAS_QWEN4_MTP_CONF`
    /// early-stop syncs and post-chain prefix truncation under greedy drafting.
    SingleSequence,
    /// Cross-sequence batched cohort (`b >= 2`): never applies per-sequence
    /// `MTP_CONF` early-stop syncs or prefix truncation so every sequence in
    /// the batch preserves one uniform verify width `K`.
    BatchedCohort,
}

/// Resolve the confidence-gating plan for a draft chain under `scope`.
///
/// - Under [`ChainConfidenceScope::BatchedCohort`] (`b >= 2`), per-sequence
///   early-stop and truncation are always disabled so `C=1` can retain full
///   `MTP_CONF` early-stop speed on `C>1` servers without shattering `C>=2`
///   cohorts into ragged widths or disabling `argmax_bf16_rows`.
/// - Under non-greedy Gumbel-CRN (`gumbel_active == true`) on `SingleSequence`,
///   per-step greedy floors (`MTP_CONF` / `MTP_CONF_STEPS`) are replaced by a
///   `0.0` per-step floor plus running-product `MTP_CONF_CUM` over true
///   post-temperature `p_T(chosen_tok)` (default `0.38`; `ATLAS_QWEN4_MTP_CONF_CUM=0`
///   disables gating).
#[allow(clippy::too_many_arguments)]
fn resolve_chain_confidence_gate_with_cum(
    scope: ChainConfidenceScope,
    policy: GatePolicy,
    policy_wants_prob: bool,
    policy_early: Option<f32>,
    argmax_prob_available: bool,
    gumbel_active: bool,
    gumbel_cum: Option<f32>,
    early_explicitly_disabled: bool,
) -> ChainConfidenceGate {
    if matches!(scope, ChainConfidenceScope::BatchedCohort) {
        return ChainConfidenceGate {
            want_prob: policy.log && argmax_prob_available,
            early: None,
            apply_truncate: false,
        };
    }
    if gumbel_active {
        let cum_active = gumbel_cum.is_some();
        let early_enabled = cum_active && (policy.early_stop || !early_explicitly_disabled);
        return ChainConfidenceGate {
            want_prob: policy.log || cum_active,
            early: early_enabled.then_some(0.0),
            apply_truncate: cum_active,
        };
    }
    let want_prob = policy_wants_prob && argmax_prob_available;
    let early = policy_early.filter(|_| want_prob);
    ChainConfidenceGate {
        want_prob,
        early,
        apply_truncate: want_prob,
    }
}

fn resolve_chain_confidence_gate_for_scope(
    scope: ChainConfidenceScope,
    policy: GatePolicy,
    policy_wants_prob: bool,
    policy_early: Option<f32>,
    argmax_prob_available: bool,
    gumbel_active: bool,
) -> ChainConfidenceGate {
    resolve_chain_confidence_gate_with_cum(
        scope,
        policy,
        policy_wants_prob,
        policy_early,
        argmax_prob_available,
        gumbel_active,
        qwen4_mtp_gate::gumbel_cum_threshold(),
        qwen4_mtp_gate::early_stop_explicitly_disabled(),
    )
}

/// Resolve the confidence-gating plan for a single-sequence (`b == 1`) draft chain.
fn resolve_chain_confidence_gate(
    policy: GatePolicy,
    policy_wants_prob: bool,
    policy_early: Option<f32>,
    argmax_prob_available: bool,
    gumbel_active: bool,
) -> ChainConfidenceGate {
    resolve_chain_confidence_gate_for_scope(
        ChainConfidenceScope::SingleSequence,
        policy,
        policy_wants_prob,
        policy_early,
        argmax_prob_available,
        gumbel_active,
    )
}

/// Packed device metadata offsets (relative to `meta_base`) for a homogeneous
/// batched Gumbel-CRN launch: `[seeds: u64; b]` followed by step-major
/// `[abs_positions: u64; num_drafts * b]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct BatchGumbelMetaLayout {
    seeds_off: usize,
    pos_base_off: usize,
    total_bytes: usize,
}

impl BatchGumbelMetaLayout {
    fn compute(base_off: usize, b: usize, num_drafts: usize) -> Option<Self> {
        let seeds_off = base_off.checked_add(7)? & !7;
        let seeds_bytes = b.checked_mul(8)?;
        let pos_base_off = seeds_off.checked_add(seeds_bytes)?;
        let pos_bytes = num_drafts.checked_mul(b)?.checked_mul(8)?;
        let end = pos_base_off.checked_add(pos_bytes)?;
        let raw_span = end.checked_sub(base_off)?;
        let total_bytes = raw_span.checked_add(255)? & !255;
        Some(Self {
            seeds_off,
            pos_base_off,
            total_bytes,
        })
    }

    #[inline]
    fn step_pos_off(self, step: usize, b: usize) -> usize {
        self.pos_base_off + step * b * 8
    }
}

fn prompt_tail_start(base: usize, count: usize) -> Result<usize> {
    let end = base
        .checked_add(1)
        .ok_or_else(|| anyhow::anyhow!("Qwen4 MTP prompt position overflow"))?;
    ensure!(
        count <= end,
        "Qwen4 MTP prompt tail exceeds physical history"
    );
    end.checked_sub(count)
        .and_then(|start| start.checked_add(1))
        .ok_or_else(|| anyhow::anyhow!("Qwen4 MTP predict-into position overflow"))
}

fn rotary_axes(
    positions: &crate::traits::RotaryPositions,
    physical: usize,
    mrope_interleaved: bool,
    mrope_kernel_available: bool,
    yarn_factor: f32,
) -> Result<[u32; 3]> {
    if !positions.is_identity() {
        ensure!(
            mrope_interleaved,
            "Qwen4 native MTP image positions require MRoPE"
        );
        ensure!(
            mrope_kernel_available,
            "Qwen4 native MTP MRoPE kernel is unavailable"
        );
        ensure!(
            yarn_factor == 0.0,
            "Qwen4 native MTP image YaRN positions require separate numerical qualification"
        );
    }
    positions.position(physical)
}

fn validate_rotary_span(
    positions: &crate::traits::RotaryPositions,
    start: usize,
    count: usize,
    mrope_interleaved: bool,
    mrope_kernel_available: bool,
    yarn_factor: f32,
) -> Result<()> {
    ensure!(
        count <= MAX_ROTARY_METADATA_ROWS,
        "Qwen4 MTP rotary span exceeds metadata bound"
    );
    if count == 0 {
        return Ok(());
    }
    let last = start
        .checked_add(count - 1)
        .ok_or_else(|| anyhow::anyhow!("Qwen4 MTP rotary span overflow"))?;
    rotary_axes(
        positions,
        start,
        mrope_interleaved,
        mrope_kernel_available,
        yarn_factor,
    )?;
    rotary_axes(
        positions,
        last,
        mrope_interleaved,
        mrope_kernel_available,
        yarn_factor,
    )?;
    Ok(())
}

fn write_rotary_metadata(
    positions: &crate::traits::RotaryPositions,
    physical: usize,
    packed: &mut [u8],
) -> Result<()> {
    ensure!(
        packed.len() >= POSITION_W_OFFSET + 4,
        "Qwen4 MTP rotary metadata header is truncated"
    );
    let [t, h, w] = positions.position(physical)?;
    packed[0..4].copy_from_slice(&t.to_le_bytes());
    if !positions.is_identity() {
        packed[POSITION_H_OFFSET..POSITION_H_OFFSET + 4].copy_from_slice(&h.to_le_bytes());
        packed[POSITION_W_OFFSET..POSITION_W_OFFSET + 4].copy_from_slice(&w.to_le_bytes());
    }
    Ok(())
}

fn rotary_metadata_ptrs(
    meta_base: DevicePtr,
    positions: &crate::traits::RotaryPositions,
) -> (DevicePtr, DevicePtr) {
    if positions.is_identity() {
        (meta_base, meta_base)
    } else {
        (
            meta_base.offset(POSITION_H_OFFSET),
            meta_base.offset(POSITION_W_OFFSET),
        )
    }
}

impl Qwen4MtpHead {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        store: &WeightStore,
        config: &atlas_core::config::ModelConfig,
        layer: Box<dyn TransformerLayer>,
        final_mixer: Qwen4HyperConnection,
        embed_tokens: DenseWeight,
        lm_head_nvfp4: QuantizedWeight,
        gpu: &dyn GpuBackend,
        mtp_vocab_size: u32,
        max_seq_len: usize,
        max_batch_size: usize,
        // Token capacity of a shared target KV pool (`--kv-pool-tokens`);
        // `None` sizes the MTP cache per slot at max_seq_len as before.
        shared_target_pool_tokens: Option<usize>,
    ) -> Result<Self> {
        ensure!(
            config.is_qwen4_exp(),
            "Qwen4MtpHead requires qwen4_exp target"
        );
        let h = config.hidden_size;
        let stream = gpu.default_stream();
        let absmax = gpu.kernel("quantize_nvfp4", "nvfp4_global_absmax")?;
        let quantize = gpu
            .kernel("quantize_nvfp4", "quantize_bf16_to_nvfp4_weight_rne")
            .or_else(|_| gpu.kernel("quantize_nvfp4", "quantize_bf16_to_nvfp4"))?;
        let mrope_kernel_available = gpu
            .kernel("rope_mrope_interleaved", "rope_forward_mrope_interleaved")
            .is_ok();
        let fc_embedding_dense = dense_auto(store, "mtp.fc_embedding.weight", gpu)?;
        let fc_hidden_dense = dense_auto(store, "mtp.fc_hidden.weight", gpu)?;
        let fc_embedding = quantize_to_nvfp4_cached(
            &fc_embedding_dense,
            h,
            h,
            gpu,
            absmax,
            quantize,
            stream,
            "mtp.fc_embedding.nvfp4",
        )?;
        let fc_hidden = quantize_to_nvfp4_cached(
            &fc_hidden_dense,
            h,
            h,
            gpu,
            absmax,
            quantize,
            stream,
            "mtp.fc_hidden.nvfp4",
        )?;
        let kv_config = KvCacheConfig {
            block_size: 16,
            num_kv_heads: config.num_key_value_heads,
            head_dim: config.head_dim,
            num_layers: 1,
            dtype: KvCacheDtype::Bf16,
            layer_dtypes: vec![],
            layer_dims: vec![],
            cache_blocks_per_seq: None,
        };
        let num_blocks = crate::factory::kv_pool::mtp_kv_pool_blocks(
            max_seq_len,
            max_batch_size,
            kv_config.block_size,
            shared_target_pool_tokens,
        );
        let kv_cache = PagedKvCache::new(kv_config, num_blocks, gpu)?;
        let effective_vocab = if qwen4_mtp_gate::mtp_vocab_prune_selected() {
            (qwen4_mtp_gate::PRUNED_VOCAB_SIZE as usize).min(config.vocab_size)
        } else if mtp_vocab_size == 0 {
            config.vocab_size
        } else {
            (mtp_vocab_size as usize).min(config.vocab_size)
        };
        tracing::info!(
            hidden = h,
            residual = config.residual_width(),
            experts = config.num_experts,
            vocab = effective_vocab,
            "Qwen4 native MTP head constructed"
        );
        let w4a16_gemv_exact_m4_k = gpu.kernel("w4a16_gemv", "w4a16_gemv_batch_logits_exact_m4")?;
        let w4a16_exact_fc_kernels = ops::W4a16ExactLmHeadKernels::new(
            w4a16_gemv_exact_m4_k,
            crate::layers::try_kernel(gpu, "w4a16_gemv", "w4a16_gemv_batch_logits_exact_m8"),
            crate::layers::try_kernel(gpu, "w4a16_gemv", "w4a16_gemv_batch_logits_exact_m17"),
            crate::layers::try_kernel(gpu, "w4a16_gemv", "w4a16_gemv_batch_logits_exact_m32"),
        )
        .with_rt2(
            crate::layers::try_kernel(gpu, "w4a16_gemv_rt", ops::ExactLmHeadTier::M4.symbol_rt2()),
            crate::layers::try_kernel(gpu, "w4a16_gemv_rt", ops::ExactLmHeadTier::M8.symbol_rt2()),
            crate::layers::try_kernel(gpu, "w4a16_gemv_rt", ops::ExactLmHeadTier::M17.symbol_rt2()),
            crate::layers::try_kernel(gpu, "w4a16_gemv_rt", ops::ExactLmHeadTier::M32.symbol_rt2()),
        )
        .with_rowpar(crate::layers::try_kernel(
            gpu,
            "w4a16_gemv",
            "w4a16_gemv_rows",
        ))
        .with_small_k_sw(
            crate::layers::try_kernel(gpu, "w4a16_gemv", "w4a16_gemv_batch_logits_exact_sw_m4_o2"),
            crate::layers::try_kernel(gpu, "w4a16_gemv", "w4a16_gemv_batch_logits_exact_sw_m8_o1"),
        )
        .with_small_k_sw_m10(crate::layers::try_kernel(
            gpu,
            "w4a16_gemv",
            "w4a16_gemv_batch_logits_exact_sw_m10_o1",
        ))
        .with_wide_n_sw4(crate::layers::try_kernel(
            gpu,
            "w4a16_gemv",
            "w4a16_gemv_batch_logits_exact_sw_m8_o4w4",
        ))
        .with_wide_n_sw4_m10(crate::layers::try_kernel(
            gpu,
            "w4a16_gemv",
            "w4a16_gemv_batch_logits_exact_sw_m10_o4w4",
        ))
        .with_wide_n_sw4_m15_m25(
            crate::layers::try_kernel(
                gpu,
                "w4a16_gemv",
                "w4a16_gemv_batch_logits_exact_sw_m15_o2w4",
            ),
            crate::layers::try_kernel(
                gpu,
                "w4a16_gemv",
                "w4a16_gemv_batch_logits_exact_sw_m25_o1w4",
            ),
        )
        .with_single_sweep_m64_m80_m128(
            crate::layers::try_kernel(gpu, "w4a16_gemv", ops::ExactSingleSweepTier::M64.symbol()),
            crate::layers::try_kernel(gpu, "w4a16_gemv", ops::ExactSingleSweepTier::M80.symbol()),
            crate::layers::try_kernel(gpu, "w4a16_gemv", ops::ExactSingleSweepTier::M128.symbol()),
        );
        Ok(Self {
            pre_fc_norm_embedding: dense_f32_safe(store, "mtp.pre_fc_norm_embedding.weight", gpu)?,
            pre_fc_norm_hidden: dense_f32_safe(store, "mtp.pre_fc_norm_hidden.weight", gpu)?,
            fc_embedding,
            fc_hidden,
            layer,
            final_mixer,
            embed_tokens,
            lm_head_nvfp4,
            mtp_vocab_size,
            kv_cache: Mutex::new(kv_cache),
            packed_meta_scratch: Mutex::new(Vec::new()),
            pinned_h2d_ring: Mutex::new(MtpPinnedH2dRing::new(gpu, MTP_PINNED_H2D_RING_BYTES)),
            graph_cache: Mutex::new(MtpGraphCache::default()),
            rms_norm_k: gpu.kernel("norm", "rms_norm")?,
            pre_fc_dual_norm_k: crate::layers::try_kernel(
                gpu,
                "norm",
                "qwen4_mtp_pre_fc_dual_rms_norm_rows",
            ),
            embed_dual_rms_norm_fused_k: crate::layers::try_kernel(
                gpu,
                "qwen4_mtp_wave2",
                "qwen4_mtp_embed_dual_rms_norm_fused",
            ),
            embed_dual_rms_norm_rows_packed_k: crate::layers::try_kernel(
                gpu,
                "qwen4_mtp_wave2",
                "qwen4_mtp_embed_dual_rms_norm_rows_packed",
            ),
            fc_dual_broadcast_rows_k: crate::layers::try_kernel(
                gpu,
                "qwen4_mtp_wave2",
                "qwen4_mtp_fc_dual_broadcast_rows",
            ),
            w4a16_gemv_k: gpu.kernel("w4a16_gemv", "w4a16_gemv")?,
            w4a16_gemv_sw_k: crate::layers::try_kernel(gpu, "w4a16_gemv", "w4a16_gemv_sw"),
            gemv_sw: ops::gemv_sw_enabled(),
            w4a16_gemv_exact_m4_k,
            w4a16_exact_fc_kernels,
            w4a16_gemm_pipe_k: crate::layers::try_kernel(gpu, "w4a16", "w4a16_gemm_pipe"),
            residual_add_k: gpu.kernel("residual_add", "bf16_residual_add")?,
            residual_add_broadcast_hc_k: crate::layers::try_kernel(
                gpu,
                "residual_add",
                "bf16_residual_add_broadcast_hc",
            ),
            argmax_k: gpu.kernel("argmax", "argmax_bf16")?,
            argmax_rows_k: crate::layers::try_kernel(gpu, "argmax", "argmax_bf16_rows"),
            argmax_prob_k: crate::layers::try_kernel(gpu, "argmax", "argmax_prob_bf16"),
            argmax_prob_batched_k: crate::layers::try_kernel(
                gpu,
                "argmax",
                "argmax_prob_batched_bf16",
            ),
            mtp_gumbel_argmax_k: crate::layers::try_kernel(
                gpu,
                "sampler_topk_candidates",
                "mtp_gumbel_argmax_bf16",
            ),
            embed_from_argmax_k: gpu
                .kernel("embed_from_argmax", "embed_from_argmax")
                .unwrap_or(KernelHandle(0)),
            mrope_kernel_available,
        })
    }

    /// Launch coalesced NVFP4 `fc_hidden` for `hc_rows` up to
    /// [`MTP_FC_COALESCED_MAX_HC_ROWS`] (`256`, i.e. `batch_rows <= 64`) by
    /// delegating directly to `qwen4_fast_proj::try_coalesced_nvfp4_w4a16_gemm`,
    /// which handles `1..=256` rows directly via SM-aware `w4a16_gemm_coalesced*`
    /// tiles.
    #[allow(clippy::too_many_arguments)]
    fn try_coalesced_mtp_fc_hidden(
        &self,
        gpu: &dyn GpuBackend,
        normed_hidden: DevicePtr,
        hidden: DevicePtr,
        hc_rows: usize,
        h: usize,
        stream: u64,
    ) -> Result<bool> {
        if hc_rows > MTP_FC_COALESCED_MAX_HC_ROWS {
            return Ok(false);
        }
        crate::layers::qwen4_fast_proj::try_coalesced_nvfp4_w4a16_gemm(
            gpu,
            normed_hidden,
            &self.fc_hidden,
            hidden,
            hc_rows,
            h,
            h,
            stream,
        )
    }

    /// Launch batched 0-ULP `argmax_prob_batched_bf16` (`grid = [rows, 1, 1]`,
    /// `block = [1024, 1, 1]`) when loaded and `rows > 1`, falling back to
    /// sequential per-row `argmax_prob_bf16` launches otherwise.
    #[allow(clippy::too_many_arguments)]
    fn launch_argmax_prob_batch(
        &self,
        gpu: &dyn GpuBackend,
        logits: DevicePtr,
        out_ids: DevicePtr,
        out_probs: DevicePtr,
        rows: usize,
        vocab: usize,
        stream: u64,
    ) -> Result<()> {
        if should_use_batched_argmax_prob(rows, self.argmax_prob_batched_k.0 != 0) {
            spark_runtime::kernel_args::KernelLaunch::new(gpu, self.argmax_prob_batched_k)
                .grid([rows as u32, 1, 1])
                .block([1024, 1, 1])
                .arg_ptr(logits)
                .arg_ptr(out_ids)
                .arg_ptr(out_probs)
                .arg_u32(vocab as u32)
                .arg_u32(vocab as u32)
                .launch(stream)?;
        } else {
            for row in 0..rows {
                spark_runtime::kernel_args::KernelLaunch::new(gpu, self.argmax_prob_k)
                    .grid([1, 1, 1])
                    .block([1024, 1, 1])
                    .arg_ptr(logits.offset(row * vocab * 2))
                    .arg_ptr(out_ids.offset(row * 4))
                    .arg_ptr(out_probs.offset(row * 4))
                    .arg_u32(vocab as u32)
                    .launch(stream)?;
            }
        }
        Ok(())
    }

    /// Capture-safe multi-step batched MTP proposer body: reads all per-step
    /// inputs (`last_tokens`, `target_hiddens` pointer table, `positions`,
    /// `kv_slots`, `seq_lens`, `block_tables`) strictly from pre-staged device
    /// memory (`spec.meta_base`), executes all `spec.step_count` steps (dual
    /// norm -> FC -> MTP layer -> final mixer -> LM head -> batched argmax/prob)
    /// with zero `gpu.alloc()` and zero host-value kernel args, leaving results
    /// in `spec.argmax_base` (and `spec.prob_base`) for ONE trailing D2H copy.
    #[allow(clippy::too_many_arguments)]
    fn launch_batched_mtp_graph_body(
        &self,
        spec: &MtpBatchedChainLaunch<'_>,
        active_lanes: Option<&[usize]>,
        states: &mut [&mut dyn ProposerState],
        kv_cache: &mut PagedKvCache,
        ctx: &ForwardContext,
        stream: u64,
        capturing: bool,
    ) -> Result<()> {
        let b = spec.batch_size;
        let h = ctx.config.hidden_size;
        let r = ctx.config.residual_width();
        let hc = ctx.config.hc_count;
        let eps = ctx.config.rms_norm_eps as f32;
        let row_bytes = h * 2;

        let embed = ctx.buffers.ssm_qkvz();
        let normed_embed = ctx.buffers.ssm_deinterleaved();
        let normed_hidden = ctx.buffers.residual();
        let embed_proj = ctx.buffers.qkv_output();
        let hidden = ctx.buffers.hidden_states();
        let logits = ctx.buffers.logits();

        let tc_dense_scope =
            crate::layers::qwen4_hyper::batched_verify_tc_dense_active(b, capturing);
        let _tc_dense_guard =
            crate::layers::qwen4_hyper::BatchedVerifyTcDenseGuard::enter(tc_dense_scope);
        let coalesced_lm_head_eligible = should_use_mtp_lm_head_coalesced(
            crate::layers::qwen4_hyper::batched_verify_tc_component_active(
                crate::layers::qwen4_hyper::BatchedVerifyTcComponent::LmHead,
                b,
                capturing,
            ),
            b,
            h,
        );
        let coalesced_fc_eligible = should_use_mtp_fc_coalesced(
            tc_dense_scope,
            b,
            hc,
            h,
            ctx.buffers.sizes().qkv_output,
            self.residual_add_broadcast_hc_k.0 != 0,
        );

        let mut step_seq_lens = vec![0usize; b];
        for step in 0..spec.step_count {
            let step_meta = spec.meta_base.offset(step * spec.stride);
            if step == 0 {
                spark_runtime::kernel_args::KernelLaunch::new(ctx.gpu, self.embed_from_argmax_k)
                    .grid([(h as u32).div_ceil(256), b as u32, 1])
                    .block([256, 1, 1])
                    .arg_ptr(spec.meta_base.offset(spec.last_tok_off))
                    .arg_ptr(self.embed_tokens.weight)
                    .arg_ptr(embed)
                    .arg_ptr(DevicePtr(0))
                    .arg_u32(h as u32)
                    .launch(stream)?;
                spark_runtime::kernel_args::KernelLaunch::new(ctx.gpu, self.pre_fc_dual_norm_k)
                    .grid([b as u32, 2, 1])
                    .block([1024, 1, 1])
                    .arg_ptr(embed)
                    .arg_ptr(self.pre_fc_norm_embedding.weight)
                    .arg_ptr(normed_embed)
                    .arg_ptr(DevicePtr::NULL)
                    .arg_ptr(self.pre_fc_norm_hidden.weight)
                    .arg_ptr(normed_hidden)
                    .arg_u32(h as u32)
                    .arg_u32(r as u32)
                    .arg_f32(eps)
                    .arg_ptr(spec.meta_base.offset(spec.hidden_ptrs_off))
                    .launch(stream)?;
            } else {
                let prev_argmax = spec.argmax_base.offset((step - 1) * b * 4);
                let token_out = spec
                    .token_out_base
                    .map_or(DevicePtr(0), |base| base.offset((step - 1) * b * 4));
                spark_runtime::kernel_args::KernelLaunch::new(ctx.gpu, self.embed_from_argmax_k)
                    .grid([(h as u32).div_ceil(256), b as u32, 1])
                    .block([256, 1, 1])
                    .arg_ptr(prev_argmax)
                    .arg_ptr(self.embed_tokens.weight)
                    .arg_ptr(embed)
                    .arg_ptr(token_out)
                    .arg_u32(h as u32)
                    .launch(stream)?;
                spark_runtime::kernel_args::KernelLaunch::new(ctx.gpu, self.pre_fc_dual_norm_k)
                    .grid([b as u32, 2, 1])
                    .block([1024, 1, 1])
                    .arg_ptr(embed)
                    .arg_ptr(self.pre_fc_norm_embedding.weight)
                    .arg_ptr(normed_embed)
                    .arg_ptr(hidden)
                    .arg_ptr(self.pre_fc_norm_hidden.weight)
                    .arg_ptr(normed_hidden)
                    .arg_u32(h as u32)
                    .arg_u32(r as u32)
                    .arg_f32(eps)
                    .arg_ptr(DevicePtr::NULL)
                    .launch(stream)?;
            }

            let ran_coalesced_fc = if coalesced_fc_eligible
                && crate::layers::qwen4_fast_proj::try_coalesced_nvfp4_w4a16_gemm(
                    ctx.gpu,
                    normed_embed,
                    &self.fc_embedding,
                    embed_proj,
                    b,
                    h,
                    h,
                    stream,
                )?
                && self.try_coalesced_mtp_fc_hidden(
                    ctx.gpu,
                    normed_hidden,
                    hidden,
                    b * hc,
                    h,
                    stream,
                )? {
                spark_runtime::kernel_args::KernelLaunch::new(
                    ctx.gpu,
                    self.residual_add_broadcast_hc_k,
                )
                .grid([(h as u32).div_ceil(256), b as u32, 1])
                .block([256, 1, 1])
                .arg_ptr(hidden)
                .arg_ptr(embed_proj)
                .arg_u32(h as u32)
                .arg_u32(hc as u32)
                .launch(stream)?;
                true
            } else {
                false
            };
            if !ran_coalesced_fc {
                if self.fc_dual_broadcast_rows_k.0 != 0 {
                    spark_runtime::kernel_args::KernelLaunch::new(
                        ctx.gpu,
                        self.fc_dual_broadcast_rows_k,
                    )
                    .grid([(h as u32).div_ceil(4), b as u32, 1])
                    .block([256, 1, 1])
                    .arg_ptr(normed_embed)
                    .arg_ptr(self.fc_embedding.weight)
                    .arg_ptr(self.fc_embedding.weight_scale)
                    .arg_f32(self.fc_embedding.weight_scale_2)
                    .arg_ptr(normed_hidden)
                    .arg_ptr(self.fc_hidden.weight)
                    .arg_ptr(self.fc_hidden.weight_scale)
                    .arg_f32(self.fc_hidden.weight_scale_2)
                    .arg_ptr(hidden)
                    .arg_u32(h as u32)
                    .arg_u32(h as u32)
                    .launch(stream)?;
                } else {
                    ops::w4a16_gemv_batch_logits_exact_with(
                        ctx.gpu,
                        self.w4a16_exact_fc_kernels,
                        normed_embed,
                        &self.fc_embedding,
                        embed_proj,
                        b as u32,
                        h as u32,
                        h as u32,
                        stream,
                        false,
                    )?;
                    ops::w4a16_gemv_batch_logits_exact_with(
                        ctx.gpu,
                        self.w4a16_exact_fc_kernels,
                        normed_hidden,
                        &self.fc_hidden,
                        hidden,
                        (b * hc) as u32,
                        h as u32,
                        h as u32,
                        stream,
                        false,
                    )?;
                    if self.residual_add_broadcast_hc_k.0 != 0 {
                        spark_runtime::kernel_args::KernelLaunch::new(
                            ctx.gpu,
                            self.residual_add_broadcast_hc_k,
                        )
                        .grid([(h as u32).div_ceil(256), b as u32, 1])
                        .block([256, 1, 1])
                        .arg_ptr(hidden)
                        .arg_ptr(embed_proj)
                        .arg_u32(h as u32)
                        .arg_u32(hc as u32)
                        .launch(stream)?;
                    } else {
                        for j in 0..b {
                            let embed_row = embed_proj.offset(j * row_bytes);
                            for branch in 0..hc {
                                let hidden_branch = hidden.offset((j * hc + branch) * row_bytes);
                                ops::residual_add(
                                    ctx.gpu,
                                    self.residual_add_k,
                                    hidden_branch,
                                    embed_row,
                                    h as u32,
                                    stream,
                                )?;
                            }
                        }
                    }
                }
            }

            let metadata = AttnMetadataDev {
                qwen4_qsa_required: self.mtp_qsa_active(),
                positions: step_meta,
                positions_h: step_meta,
                positions_w: step_meta,
                slot: step_meta.offset(spec.slot_off),
                seq_len: step_meta.offset(spec.seq_len_off),
                block_table: step_meta.offset(spec.bt_off),
                max_blocks_per_seq: spec.bt_blocks_cap as u32,
                num_seqs: b as u32,
            };
            let layer_ctx = ForwardContext {
                attn_metadata: Some(metadata),
                comm: None,
                graph_capture: capturing,
                ..*ctx
            };
            for (dst, &base_len) in step_seq_lens.iter_mut().zip(spec.base_seq_lens.iter()) {
                *dst = base_len + step;
            }
            {
                let mut typed_states: Vec<&mut Qwen4MtpState> = Vec::with_capacity(b);
                if let Some(lanes) = active_lanes {
                    let mut cursor = 0usize;
                    for (orig_s, st) in states.iter_mut().enumerate() {
                        if cursor < lanes.len() && lanes[cursor] == orig_s {
                            typed_states.push(
                                st.as_any_mut()
                                    .downcast_mut::<Qwen4MtpState>()
                                    .expect("validated above"),
                            );
                            cursor += 1;
                        }
                    }
                } else {
                    for st in states.iter_mut() {
                        typed_states.push(
                            st.as_any_mut()
                                .downcast_mut::<Qwen4MtpState>()
                                .expect("validated above"),
                        );
                    }
                }
                let mut block_tables: Vec<&mut Vec<u32>> = Vec::with_capacity(b);
                let mut disk_block_ids: Vec<&mut Vec<u32>> = Vec::with_capacity(b);
                let mut disk_last_offloaded: Vec<&mut Vec<u32>> = Vec::with_capacity(b);
                for st in typed_states.iter_mut() {
                    let Qwen4MtpState {
                        block_table,
                        disk_block_ids: d_ids,
                        disk_last_offloaded_per_layer: d_off,
                        ..
                    } = &mut **st;
                    block_tables.push(block_table);
                    disk_block_ids.push(d_ids);
                    disk_last_offloaded.push(d_off);
                }
                self.layer.decode_qwen4_cross_seq_mtp(
                    hidden,
                    ctx.buffers.residual(),
                    &step_seq_lens,
                    &mut block_tables,
                    &mut disk_block_ids,
                    &mut disk_last_offloaded,
                    kv_cache,
                    &layer_ctx,
                    stream,
                )?;
            }

            let sample_hiddens = self.final_mixer.prepare_batched(
                hidden,
                ctx.buffers.residual(),
                b,
                ctx.buffers,
                ctx.gpu,
                eps,
                stream,
            )?;
            let ran_coalesced_lm = if coalesced_lm_head_eligible {
                crate::layers::qwen4_fast_proj::try_coalesced_nvfp4_w4a16_gemm(
                    ctx.gpu,
                    sample_hiddens,
                    &self.lm_head_nvfp4,
                    logits,
                    b,
                    spec.vocab,
                    h,
                    stream,
                )?
            } else {
                false
            };
            if !ran_coalesced_lm {
                if should_use_mtp_lm_head_pipe(
                    mtp_lm_head_tc_pipe_enabled(),
                    b,
                    h,
                    self.w4a16_gemm_pipe_k.0 != 0,
                    sample_hiddens,
                    self.lm_head_nvfp4.weight,
                    logits,
                ) {
                    ops::w4a16_gemm_pipe(
                        ctx.gpu,
                        self.w4a16_gemm_pipe_k,
                        sample_hiddens,
                        &self.lm_head_nvfp4,
                        logits,
                        b as u32,
                        spec.vocab as u32,
                        h as u32,
                        stream,
                    )?;
                } else {
                    ops::w4a16_gemv_batch_logits_exact_with(
                        ctx.gpu,
                        self.w4a16_exact_fc_kernels,
                        sample_hiddens,
                        &self.lm_head_nvfp4,
                        logits,
                        b as u32,
                        spec.vocab as u32,
                        h as u32,
                        stream,
                        ops::w4a16_exact_rt2_for_rows(
                            ops::w4a16_gemv_rt2_enabled(),
                            ops::w4a16_exact_rt2_min_rows(),
                            b as u32,
                        ),
                    )?;
                }
            }

            let step_argmax = spec.argmax_base.offset(step * b * 4);
            let step_prob = spec.prob_base.offset(step * b * 4);
            if spec.want_prob {
                self.launch_argmax_prob_batch(
                    ctx.gpu,
                    logits,
                    step_argmax,
                    step_prob,
                    b,
                    spec.vocab,
                    stream,
                )?;
            } else if self.argmax_rows_k.0 != 0 {
                ops::argmax_bf16_rows(
                    ctx.gpu,
                    self.argmax_rows_k,
                    logits,
                    step_argmax,
                    spec.vocab as u32,
                    spec.vocab as u32,
                    b as u32,
                    stream,
                )?;
            } else {
                for s in 0..b {
                    ops::argmax_bf16(
                        ctx.gpu,
                        self.argmax_k,
                        logits.offset(s * spec.vocab * 2),
                        step_argmax.offset(s * 4),
                        spec.vocab as u32,
                        stream,
                    )?;
                }
            }
        }
        Ok(())
    }

    /// Execute a uniform batched MTP multi-step chain via the cached CUDA graph
    /// (`ATLAS_QWEN4_MTP_GRAPH`), warming up lazily on hit 1, capturing on hit 2
    /// with fail-closed fallback to eager on any capture error, and replaying in
    /// 1 graph launch on hits 3+. Returns `Ok(true)` if the chain was executed.
    #[allow(clippy::too_many_arguments)]
    fn run_batched_mtp_chain_graphed(
        &self,
        spec: &MtpBatchedChainLaunch<'_>,
        active_lanes: Option<&[usize]>,
        states: &mut [&mut dyn ProposerState],
        kv_cache: &mut PagedKvCache,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<bool> {
        let b = spec.batch_size;
        let h = ctx.config.hidden_size;
        let hc = ctx.config.hc_count;
        let tc_dense_scope = crate::layers::qwen4_hyper::batched_verify_tc_dense_active(b, true);
        let coalesced_lm_head = should_use_mtp_lm_head_coalesced(
            crate::layers::qwen4_hyper::batched_verify_tc_component_active(
                crate::layers::qwen4_hyper::BatchedVerifyTcComponent::LmHead,
                b,
                true,
            ),
            b,
            h,
        );
        let coalesced_fc = should_use_mtp_fc_coalesced(
            tc_dense_scope,
            b,
            hc,
            h,
            ctx.buffers.sizes().qkv_output,
            self.residual_add_broadcast_hc_k.0 != 0,
        );
        let key = MtpGraphKey {
            batch_size: b,
            step_count: spec.step_count,
            want_prob: spec.want_prob,
            bt_blocks_cap: spec.bt_blocks_cap as u32,
            vocab: spec.vocab,
            write_step_token_out: spec.token_out_base.is_some(),
            tc_dense_scope,
            coalesced_fc,
            coalesced_lm_head,
            scratch_ptr: ctx.buffers.scratch().0,
            hidden_ptr: ctx.buffers.hidden_states().0,
            residual_ptr: ctx.buffers.residual().0,
            logits_ptr: ctx.buffers.logits().0,
            k_pool_ptr: kv_cache.k_pool_ptr(0).0,
            v_pool_ptr: kv_cache.v_pool_ptr(0).0,
            stream,
        };

        let (cached_graph, is_poisoned, is_warmed) = {
            let cache = self.graph_cache.lock();
            (
                cache.get(&key),
                cache.is_poisoned(&key),
                cache.warmed.contains(&key),
            )
        };
        if let Some(graph) = cached_graph {
            ctx.gpu.launch_graph(graph, stream)?;
            return Ok(true);
        }
        if is_poisoned {
            return Ok(false);
        }

        crate::layers::qwen4_fast_proj::warm_coalesced_nvfp4_kernels(ctx.gpu);
        if !is_warmed {
            self.launch_batched_mtp_graph_body(
                spec,
                active_lanes,
                states,
                kv_cache,
                ctx,
                stream,
                false,
            )?;
            self.graph_cache.lock().mark_warmed(key);
            return Ok(true);
        }

        let captured = (|| -> Result<GraphHandle> {
            let capture = crate::model::trait_impl::capture_guard::CaptureGuard::begin(
                ctx.gpu,
                stream,
                "qwen4_mtp_graph",
            )?;
            self.launch_batched_mtp_graph_body(
                spec,
                active_lanes,
                states,
                kv_cache,
                ctx,
                stream,
                true,
            )?;
            capture.end()
        })();
        match captured {
            Ok(graph) if graph.0 != 0 => {
                self.graph_cache.lock().insert(ctx.gpu, key, graph);
                ctx.gpu.launch_graph(graph, stream)?;
                static GRAPH_ENGAGED: std::sync::Once = std::sync::Once::new();
                GRAPH_ENGAGED.call_once(|| {
                    tracing::info!(
                        batch_size = b,
                        step_count = spec.step_count,
                        want_prob = spec.want_prob,
                        bt_blocks_cap = spec.bt_blocks_cap,
                        vocab = spec.vocab,
                        tc_dense_scope,
                        coalesced_fc,
                        coalesced_lm_head,
                        "QWEN4_MTP_GRAPH_ENGAGED"
                    );
                });
                Ok(true)
            }
            Ok(_) => {
                // Mock/no-op capture backends executed the body eagerly during capture.
                Ok(true)
            }
            Err(capture_err) => {
                self.graph_cache.lock().mark_poisoned(key);
                static CAPTURE_FAIL_WARN: std::sync::Once = std::sync::Once::new();
                CAPTURE_FAIL_WARN.call_once(|| {
                    tracing::warn!(
                        batch_size = b,
                        step_count = spec.step_count,
                        want_prob = spec.want_prob,
                        bt_blocks_cap = spec.bt_blocks_cap,
                        error = %capture_err,
                        "QWEN4_MTP_GRAPH capture failed; falling back to eager"
                    );
                });
                self.launch_batched_mtp_graph_body(
                    spec,
                    active_lanes,
                    states,
                    kv_cache,
                    ctx,
                    stream,
                    false,
                )?;
                Ok(true)
            }
        }
    }

    /// Stage `copies` into the MTP head's dedicated page-locked ring and enqueue
    /// async stream-ordered `copy_h2d_pinned_async` transfers with 0 stream drains.
    /// Falls back to `gpu.copy_h2d_group_on_stream` if disabled (`ATLAS_QWEN4_MTP_PINNED_H2D=0`)
    /// or if a payload exceeds the ring capacity.
    fn copy_h2d_group_fast(
        &self,
        gpu: &dyn GpuBackend,
        copies: &[HostToDeviceCopy<'_>],
        stream: u64,
    ) -> Result<()> {
        if copies.is_empty() {
            return Ok(());
        }
        static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let enabled = *ENABLED.get_or_init(|| {
            std::env::var("ATLAS_QWEN4_MTP_PINNED_H2D").ok().as_deref() != Some("0")
        });
        if enabled {
            let mut ring_guard = self.pinned_h2d_ring.lock();
            if let Some(ring) = ring_guard.as_mut()
                && ring.copy_h2d_group(gpu, copies, stream)?
            {
                return Ok(());
            }
        }
        gpu.copy_h2d_group_on_stream(copies, stream)
    }

    fn mtp_qsa_active(&self) -> bool {
        mtp_qsa_mode() != MtpQsaMode::Disabled && self.layer.has_qsa_side_cache()
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_one(
        &self,
        token: u32,
        target_hidden: DevicePtr,
        position: usize,
        state: &mut Qwen4MtpState,
        ctx: &ForwardContext,
        stream: u64,
        grammar_bitmask: Option<&[i32]>,
        emit_draft: bool,
        qsa_required: bool,
        chain: Option<ChainStep>,
    ) -> Result<u32> {
        // Fail before embeddings, cache allocation, metadata upload, or any
        // other device write. The metadata pack below repeats only the already
        // admitted immutable position lookup.
        rotary_axes(
            &state.rotary_positions,
            position,
            ctx.config.mrope_interleaved,
            self.mrope_kernel_available,
            ctx.config.yarn_factor,
        )?;
        let h = ctx.config.hidden_size;
        let r = ctx.config.residual_width();
        let hc = ctx.config.hc_count;
        let eps = ctx.config.rms_norm_eps as f32;
        let row_bytes = h * 2;

        let normed_embed = ctx.buffers.ssm_deinterleaved();
        let normed_hidden = ctx.buffers.residual();
        if self.embed_dual_rms_norm_fused_k.0 != 0 && h >= 1024 && r >= 1024 {
            let (tok_ptr, tok_out) = if let Some(device_token) = chain.and_then(|c| c.device_token)
            {
                (
                    device_token,
                    ctx.buffers.scratch().offset(CHAIN_TOKEN_OFFSET),
                )
            } else {
                (DevicePtr::NULL, DevicePtr::NULL)
            };
            spark_runtime::kernel_args::KernelLaunch::new(
                ctx.gpu,
                self.embed_dual_rms_norm_fused_k,
            )
            .grid([1, 2, 1])
            .block([1024, 1, 1])
            .arg_ptr(self.embed_tokens.weight)
            .arg_ptr(tok_ptr)
            .arg_ptr(tok_out)
            .arg_u32(token)
            .arg_u32(ctx.config.vocab_size as u32)
            .arg_ptr(self.pre_fc_norm_embedding.weight)
            .arg_ptr(normed_embed)
            .arg_ptr(target_hidden)
            .arg_ptr(self.pre_fc_norm_hidden.weight)
            .arg_ptr(normed_hidden)
            .arg_u32(h as u32)
            .arg_u32(r as u32)
            .arg_f32(eps)
            .launch(stream)?;
        } else {
            let embed = ctx.buffers.ssm_qkvz();
            if let Some(device_token) = chain.and_then(|c| c.device_token) {
                ops::embed_from_argmax(
                    ctx.gpu,
                    self.embed_from_argmax_k,
                    device_token,
                    self.embed_tokens.weight,
                    embed,
                    ctx.buffers.scratch().offset(CHAIN_TOKEN_OFFSET),
                    h as u32,
                    stream,
                )?;
            } else {
                let embed_src = self.embed_tokens.weight.offset(token as usize * row_bytes);
                ctx.gpu
                    .copy_d2d_async(embed_src, embed, row_bytes, stream)?;
            }
            if self.pre_fc_dual_norm_k.0 != 0 && h >= 1024 && r >= 1024 {
                spark_runtime::kernel_args::KernelLaunch::new(ctx.gpu, self.pre_fc_dual_norm_k)
                    .grid([1, 2, 1])
                    .block([1024, 1, 1])
                    .arg_ptr(embed)
                    .arg_ptr(self.pre_fc_norm_embedding.weight)
                    .arg_ptr(normed_embed)
                    .arg_ptr(target_hidden)
                    .arg_ptr(self.pre_fc_norm_hidden.weight)
                    .arg_ptr(normed_hidden)
                    .arg_u32(h as u32)
                    .arg_u32(r as u32)
                    .arg_f32(eps)
                    .arg_ptr(DevicePtr(0))
                    .launch(stream)?;
            } else {
                ops::rms_norm(
                    ctx.gpu,
                    self.rms_norm_k,
                    embed,
                    &self.pre_fc_norm_embedding,
                    normed_embed,
                    1,
                    h as u32,
                    eps,
                    stream,
                )?;
                ops::rms_norm(
                    ctx.gpu,
                    self.rms_norm_k,
                    target_hidden,
                    &self.pre_fc_norm_hidden,
                    normed_hidden,
                    1,
                    r as u32,
                    eps,
                    stream,
                )?;
            }
        }

        let embed_proj = ctx.buffers.qkv_output();
        let hidden = ctx.buffers.hidden_states();
        ensure!(hc == 4, "Qwen4 native MTP expects four hidden streams");

        if self.fc_dual_broadcast_rows_k.0 != 0 && mtp_step0_dual_gemv_selected() {
            spark_runtime::kernel_args::KernelLaunch::new(ctx.gpu, self.fc_dual_broadcast_rows_k)
                .grid([(h as u32).div_ceil(4), 1, 1])
                .block([256, 1, 1])
                .arg_ptr(normed_embed)
                .arg_ptr(self.fc_embedding.weight)
                .arg_ptr(self.fc_embedding.weight_scale)
                .arg_f32(self.fc_embedding.weight_scale_2)
                .arg_ptr(normed_hidden)
                .arg_ptr(self.fc_hidden.weight)
                .arg_ptr(self.fc_hidden.weight_scale)
                .arg_f32(self.fc_hidden.weight_scale_2)
                .arg_ptr(hidden)
                .arg_u32(h as u32)
                .arg_u32(h as u32)
                .launch(stream)?;
        } else {
            ops::w4a16_decode_gemv(
                ctx.gpu,
                self.w4a16_gemv_k,
                self.w4a16_gemv_sw_k,
                self.gemv_sw,
                normed_embed,
                &self.fc_embedding,
                embed_proj,
                h as u32,
                h as u32,
                stream,
            )?;
            ops::w4a16_gemv_batch_logits_exact_with(
                ctx.gpu,
                ops::W4a16ExactLmHeadKernels::new(
                    self.w4a16_gemv_exact_m4_k,
                    KernelHandle(0),
                    KernelHandle(0),
                    KernelHandle(0),
                ),
                normed_hidden,
                &self.fc_hidden,
                hidden,
                hc as u32,
                h as u32,
                h as u32,
                stream,
                false,
            )?;
            if self.residual_add_broadcast_hc_k.0 != 0 {
                spark_runtime::kernel_args::KernelLaunch::new(
                    ctx.gpu,
                    self.residual_add_broadcast_hc_k,
                )
                .grid([(h as u32).div_ceil(256), 1, 1])
                .block([256, 1, 1])
                .arg_ptr(hidden)
                .arg_ptr(embed_proj)
                .arg_u32(h as u32)
                .arg_u32(hc as u32)
                .launch(stream)?;
            } else {
                for branch in 0..hc {
                    let hidden_branch = hidden.offset(branch * row_bytes);
                    ops::residual_add(
                        ctx.gpu,
                        self.residual_add_k,
                        hidden_branch,
                        embed_proj,
                        h as u32,
                        stream,
                    )?;
                }
            }
        }

        let mut kv_cache = self.kv_cache.lock();
        let (meta_base, mb) = if let Some(c) = chain {
            (c.meta, c.max_blocks_per_seq)
        } else {
            let meta_base = ctx.buffers.scratch().offset(CHAIN_META_OFFSET);
            let packed = self.step_metadata(&mut kv_cache, state, state.seq_len, position)?;
            self.copy_h2d_group_fast(
                ctx.gpu,
                &[HostToDeviceCopy::new(&packed, meta_base)],
                stream,
            )?;
            let mb = ((packed.len() - 768) / 4) as u32;
            (meta_base, mb)
        };
        let (positions_h, positions_w) = rotary_metadata_ptrs(meta_base, &state.rotary_positions);
        let metadata = AttnMetadataDev {
            qwen4_qsa_required: qsa_required,
            positions: meta_base,
            positions_h,
            positions_w,
            slot: meta_base.offset(256),
            seq_len: meta_base.offset(512),
            block_table: meta_base.offset(768),
            max_blocks_per_seq: mb,
            num_seqs: 1,
        };
        let layer_ctx = ForwardContext {
            attn_metadata: Some(metadata),
            comm: None,
            graph_capture: false,
            ..*ctx
        };
        if !emit_draft && mtp_kv_only_replay_enabled() {
            self.layer.decode_kv_only(
                hidden,
                ctx.buffers.residual(),
                state.layer_state.as_mut(),
                &mut kv_cache,
                state.seq_len,
                &mut state.block_table,
                &mut state.disk_block_ids,
                &mut state.disk_last_offloaded_per_layer,
                &layer_ctx,
                stream,
            )?;
            state.seq_len += 1;
            return Ok(0);
        }
        self.layer.decode(
            hidden,
            ctx.buffers.residual(),
            state.layer_state.as_mut(),
            &mut kv_cache,
            state.seq_len,
            &mut state.block_table,
            &mut state.disk_block_ids,
            &mut state.disk_last_offloaded_per_layer,
            &layer_ctx,
            stream,
        )?;

        // Prompt replay only fills the MTP KV cache. It must not run the head:
        // `buffers.logits()` still holds the target's prefill logits, which the
        // scheduler samples the first output token from after this returns.
        if !emit_draft {
            state.seq_len += 1;
            return Ok(0);
        }

        let (sample_hidden, inject) = self.final_mixer.prepare_decode(
            hidden,
            ctx.buffers.residual(),
            ctx.buffers,
            ctx.gpu,
            eps,
            stream,
        )?;
        debug_assert!(inject.is_none());
        let sync_gumbel_spec = if chain.is_none() && grammar_bitmask.is_none() {
            state.resolve_gumbel_spec(position)
        } else {
            None
        };
        let gumbel_active = chain.and_then(|c| c.gumbel).is_some() || sync_gumbel_spec.is_some();
        let base_vocab = if qwen4_mtp_gate::mtp_vocab_prune_selected() {
            (qwen4_mtp_gate::PRUNED_VOCAB_SIZE as usize).min(ctx.config.vocab_size)
        } else if self.mtp_vocab_size == 0 {
            ctx.config.vocab_size
        } else {
            (self.mtp_vocab_size as usize).min(ctx.config.vocab_size)
        };
        let vocab = qwen4_mtp_gate::mtp_draft_vocab_override_for(gumbel_active)
            .map_or(base_vocab, |cap| base_vocab.min(cap));
        let logits = ctx.buffers.logits();
        ops::w4a16_decode_gemv(
            ctx.gpu,
            self.w4a16_gemv_k,
            self.w4a16_gemv_sw_k,
            self.gemv_sw,
            sample_hidden,
            &self.lm_head_nvfp4,
            logits,
            vocab as u32,
            h as u32,
            stream,
        )?;
        if let Some(c) = chain {
            if let Some((spec, abs_pos)) = c.gumbel {
                gumbel_draft::select_chain_gumbel_step(
                    ctx.gpu,
                    self.mtp_gumbel_argmax_k,
                    logits,
                    DevicePtr::NULL,
                    c.argmax_out,
                    c.prob_out,
                    vocab,
                    spec,
                    abs_pos,
                    stream,
                )?;
            } else if let Some(prob_out) = c.prob_out {
                spark_runtime::kernel_args::KernelLaunch::new(ctx.gpu, self.argmax_prob_k)
                    .grid([1, 1, 1])
                    .block([1024, 1, 1])
                    .arg_ptr(logits)
                    .arg_ptr(c.argmax_out)
                    .arg_ptr(prob_out)
                    .arg_u32(vocab as u32)
                    .launch(stream)?;
            } else {
                ops::argmax_bf16(
                    ctx.gpu,
                    self.argmax_k,
                    logits,
                    c.argmax_out,
                    vocab as u32,
                    stream,
                )?;
            }
            state.seq_len += 1;
            return Ok(0);
        }
        let out = ctx.buffers.scratch();
        let token_id = if let Some(mask) = grammar_bitmask {
            let mut bytes = vec![0u8; vocab * 2];
            ctx.gpu.copy_d2h_on_stream(logits, &mut bytes, stream)?;
            let mut best = None::<(u32, f32)>;
            for candidate in 0..vocab {
                if candidate / 32 >= mask.len()
                    || (mask[candidate / 32] & (1i32 << (candidate % 32))) == 0
                {
                    continue;
                }
                let bits = u16::from_le_bytes([bytes[candidate * 2], bytes[candidate * 2 + 1]]);
                let value = f32::from_bits((bits as u32) << 16);
                if best.is_none_or(|(_, old)| value > old) {
                    best = Some((candidate as u32, value));
                }
            }
            best.map_or(0, |(candidate, _)| candidate)
        } else if let Some(spec) = sync_gumbel_spec {
            gumbel_draft::select_sync_gumbel_step(
                ctx.gpu,
                self.mtp_gumbel_argmax_k,
                logits,
                out,
                vocab,
                spec,
                (position + 1) as u64,
                stream,
            )?
        } else {
            ops::argmax_bf16(ctx.gpu, self.argmax_k, logits, out, vocab as u32, stream)?;
            let mut bytes = [0u8; 4];
            ctx.gpu.copy_d2h_on_stream(out, &mut bytes, stream)?;
            u32::from_le_bytes(bytes)
        };
        state.seq_len += 1;
        Ok(token_id)
    }

    /// True when the exact multi-row FC and inner attention/MoE kernels cover `rows`.
    fn forward_rows_admits(&self, rows: usize) -> bool {
        (2..=8).contains(&rows)
            && self
                .w4a16_exact_fc_kernels
                .route_for_rows(rows as u32)
                .is_some()
            && self
                .w4a16_exact_fc_kernels
                .route_for_rows((rows * 4) as u32)
                .is_some()
            && crate::layers::qwen4_hybrid_rows(rows)
    }

    /// True when cross-sequence batched MTP (`2..=64` sequences) has exact
    /// FC/LM-head slabs, MoE route, and workspace capacity for `rows`.
    fn propose_batch_rows_admits(&self, rows: usize, ctx: &ForwardContext) -> bool {
        if !(2..=64).contains(&rows) {
            return false;
        }
        let fc_ok = self.fc_dual_broadcast_rows_k.0 != 0
            || (self
                .w4a16_exact_fc_kernels
                .route_for_rows(rows as u32)
                .is_some()
                && self
                    .w4a16_exact_fc_kernels
                    .route_for_rows((rows * 4) as u32)
                    .is_some());
        let lm_ok = self
            .w4a16_exact_fc_kernels
            .route_for_rows(rows as u32)
            .is_some()
            || (mtp_lm_head_tc_pipe_enabled() && rows >= 8 && self.w4a16_gemm_pipe_k.0 != 0);
        let moe_ok =
            rows > crate::layers::QWEN4_HYBRID_MAX_ROWS || crate::layers::qwen4_hybrid_rows(rows);
        let h_bytes = rows * ctx.config.hidden_size * 2;
        let r_bytes = rows * ctx.config.residual_width() * 2;
        let sizes = ctx.buffers.sizes();
        let arena_ok = h_bytes <= sizes.ssm_qkvz
            && h_bytes <= sizes.ssm_deinterleaved
            && h_bytes <= sizes.gate_logits
            && r_bytes <= sizes.residual
            && r_bytes <= sizes.hidden_states
            && (self.fc_dual_broadcast_rows_k.0 != 0 || h_bytes <= sizes.qkv_output);
        fc_ok && lm_ok && moe_ok && arena_ok
    }

    /// Batched multi-row MTP forward (`M in 2..=8`), bit-for-bit identical to
    /// `M` sequential `forward_one` calls: gathers `M` embeddings, runs exact
    /// multi-row `fc_embedding` and `fc_hidden`, executes `decode_qwen4_batched`
    /// with token-ordered causal attention and KV writes, and optionally runs
    /// `final_mixer` + `lm_head_nvfp4` + `argmax` on the last row `M - 1`.
    #[allow(clippy::too_many_arguments)]
    fn forward_rows(
        &self,
        tokens: &[u32],
        target_hiddens: &[DevicePtr],
        positions: &[usize],
        meta_base: DevicePtr,
        state: &mut Qwen4MtpState,
        ctx: &ForwardContext,
        stream: u64,
        grammar_bitmask: Option<&[i32]>,
        emit_last_draft: bool,
        chain_out: Option<(
            DevicePtr,
            Option<DevicePtr>,
            Option<(spark_runtime::sampler::gumbel::GumbelDraftSpec, u64)>,
        )>,
    ) -> Result<u32> {
        let rows = tokens.len();
        ensure!(
            (2..=8).contains(&rows) && target_hiddens.len() == rows && positions.len() == rows,
            "Qwen4 MTP forward_rows requires 2..=8 matched rows, got {rows}"
        );
        for &pos in positions {
            rotary_axes(
                &state.rotary_positions,
                pos,
                ctx.config.mrope_interleaved,
                self.mrope_kernel_available,
                ctx.config.yarn_factor,
            )?;
        }
        let h = ctx.config.hidden_size;
        let r = ctx.config.residual_width();
        let hc = ctx.config.hc_count;
        let eps = ctx.config.rms_norm_eps as f32;
        let row_bytes = h * 2;
        let r_bytes = r * 2;
        ensure!(hc == 4, "Qwen4 native MTP expects four hidden streams");

        let normed_embed = ctx.buffers.ssm_deinterleaved();
        let normed_hidden = ctx.buffers.residual();
        if self.embed_dual_rms_norm_rows_packed_k.0 != 0 && h >= 1024 && r >= 1024 && rows <= 8 {
            let mut run_start = 0usize;
            while run_start < rows {
                let mut run_len = 1usize;
                while run_start + run_len < rows
                    && target_hiddens[run_start + run_len]
                        == target_hiddens[run_start].offset(run_len * r_bytes)
                {
                    run_len += 1;
                }
                let tok_at = |idx: usize| -> u64 {
                    tokens
                        .get(run_start + idx)
                        .copied()
                        .filter(|_| idx < run_len)
                        .unwrap_or(0) as u64
                };
                let tok_pack01 = tok_at(0) | (tok_at(1) << 32);
                let tok_pack23 = tok_at(2) | (tok_at(3) << 32);
                let tok_pack45 = tok_at(4) | (tok_at(5) << 32);
                let tok_pack67 = tok_at(6) | (tok_at(7) << 32);
                spark_runtime::kernel_args::KernelLaunch::new(
                    ctx.gpu,
                    self.embed_dual_rms_norm_rows_packed_k,
                )
                .grid([run_len as u32, 2, 1])
                .block([1024, 1, 1])
                .arg_ptr(self.embed_tokens.weight)
                .arg_u64(tok_pack01)
                .arg_u64(tok_pack23)
                .arg_u64(tok_pack45)
                .arg_u64(tok_pack67)
                .arg_u32(ctx.config.vocab_size as u32)
                .arg_ptr(self.pre_fc_norm_embedding.weight)
                .arg_ptr(normed_embed.offset(run_start * row_bytes))
                .arg_ptr(target_hiddens[run_start])
                .arg_ptr(self.pre_fc_norm_hidden.weight)
                .arg_ptr(normed_hidden.offset(run_start * r_bytes))
                .arg_u32(h as u32)
                .arg_u32(r as u32)
                .arg_f32(eps)
                .arg_ptr(DevicePtr(0))
                .launch(stream)?;
                run_start += run_len;
            }
        } else {
            let embed = ctx.buffers.ssm_qkvz();
            for (j, &token) in tokens.iter().enumerate() {
                let embed_src = self.embed_tokens.weight.offset(token as usize * row_bytes);
                ctx.gpu.copy_d2d_async(
                    embed_src,
                    embed.offset(j * row_bytes),
                    row_bytes,
                    stream,
                )?;
            }
            if self.pre_fc_dual_norm_k.0 != 0 && h >= 1024 && r >= 1024 {
                let mut run_start = 0usize;
                while run_start < rows {
                    let mut run_len = 1usize;
                    while run_start + run_len < rows
                        && target_hiddens[run_start + run_len]
                            == target_hiddens[run_start].offset(run_len * r_bytes)
                    {
                        run_len += 1;
                    }
                    spark_runtime::kernel_args::KernelLaunch::new(ctx.gpu, self.pre_fc_dual_norm_k)
                        .grid([run_len as u32, 2, 1])
                        .block([1024, 1, 1])
                        .arg_ptr(embed.offset(run_start * row_bytes))
                        .arg_ptr(self.pre_fc_norm_embedding.weight)
                        .arg_ptr(normed_embed.offset(run_start * row_bytes))
                        .arg_ptr(target_hiddens[run_start])
                        .arg_ptr(self.pre_fc_norm_hidden.weight)
                        .arg_ptr(normed_hidden.offset(run_start * r_bytes))
                        .arg_u32(h as u32)
                        .arg_u32(r as u32)
                        .arg_f32(eps)
                        .arg_ptr(DevicePtr(0))
                        .launch(stream)?;
                    run_start += run_len;
                }
            } else {
                ops::rms_norm(
                    ctx.gpu,
                    self.rms_norm_k,
                    embed,
                    &self.pre_fc_norm_embedding,
                    normed_embed,
                    rows as u32,
                    h as u32,
                    eps,
                    stream,
                )?;
                let mut run_start = 0usize;
                while run_start < rows {
                    let mut run_len = 1usize;
                    while run_start + run_len < rows
                        && target_hiddens[run_start + run_len]
                            == target_hiddens[run_start].offset(run_len * r_bytes)
                    {
                        run_len += 1;
                    }
                    ops::rms_norm(
                        ctx.gpu,
                        self.rms_norm_k,
                        target_hiddens[run_start],
                        &self.pre_fc_norm_hidden,
                        normed_hidden.offset(run_start * r_bytes),
                        run_len as u32,
                        r as u32,
                        eps,
                        stream,
                    )?;
                    run_start += run_len;
                }
            }
        }

        let hidden = ctx.buffers.hidden_states();
        if self.fc_dual_broadcast_rows_k.0 != 0 {
            spark_runtime::kernel_args::KernelLaunch::new(ctx.gpu, self.fc_dual_broadcast_rows_k)
                .grid([(h as u32).div_ceil(4), rows as u32, 1])
                .block([256, 1, 1])
                .arg_ptr(normed_embed)
                .arg_ptr(self.fc_embedding.weight)
                .arg_ptr(self.fc_embedding.weight_scale)
                .arg_f32(self.fc_embedding.weight_scale_2)
                .arg_ptr(normed_hidden)
                .arg_ptr(self.fc_hidden.weight)
                .arg_ptr(self.fc_hidden.weight_scale)
                .arg_f32(self.fc_hidden.weight_scale_2)
                .arg_ptr(hidden)
                .arg_u32(h as u32)
                .arg_u32(h as u32)
                .launch(stream)?;
        } else {
            let embed_proj = ctx.buffers.qkv_output();
            ops::w4a16_gemv_batch_logits_exact_with(
                ctx.gpu,
                self.w4a16_exact_fc_kernels,
                normed_embed,
                &self.fc_embedding,
                embed_proj,
                rows as u32,
                h as u32,
                h as u32,
                stream,
                false,
            )?;
            ops::w4a16_gemv_batch_logits_exact_with(
                ctx.gpu,
                self.w4a16_exact_fc_kernels,
                normed_hidden,
                &self.fc_hidden,
                hidden,
                (rows * hc) as u32,
                h as u32,
                h as u32,
                stream,
                false,
            )?;
            if self.residual_add_broadcast_hc_k.0 != 0 {
                spark_runtime::kernel_args::KernelLaunch::new(
                    ctx.gpu,
                    self.residual_add_broadcast_hc_k,
                )
                .grid([(h as u32).div_ceil(256), rows as u32, 1])
                .block([256, 1, 1])
                .arg_ptr(hidden)
                .arg_ptr(embed_proj)
                .arg_u32(h as u32)
                .arg_u32(hc as u32)
                .launch(stream)?;
            } else {
                for j in 0..rows {
                    let embed_row = embed_proj.offset(j * row_bytes);
                    for branch in 0..hc {
                        let hidden_branch = hidden.offset((j * hc + branch) * row_bytes);
                        ops::residual_add(
                            ctx.gpu,
                            self.residual_add_k,
                            hidden_branch,
                            embed_row,
                            h as u32,
                            stream,
                        )?;
                    }
                }
            }
        }

        let mut kv_cache = self.kv_cache.lock();
        let packed = self.batch_metadata(&mut kv_cache, state, state.seq_len, positions)?;
        self.copy_h2d_group_fast(
            ctx.gpu,
            &[HostToDeviceCopy::new(&packed, meta_base)],
            stream,
        )?;
        let (positions_h, positions_w) = if state.rotary_positions.is_identity() {
            (meta_base, meta_base)
        } else {
            (meta_base.offset(64), meta_base.offset(128))
        };
        let max_blocks_per_seq = if rows > 0 {
            ((packed.len() - 768) / (rows * 4)) as u32
        } else {
            state.block_table.len() as u32
        };
        let metadata = AttnMetadataDev {
            qwen4_qsa_required: self.mtp_qsa_active(),
            positions: meta_base,
            positions_h,
            positions_w,
            slot: meta_base.offset(256),
            seq_len: meta_base.offset(512),
            block_table: meta_base.offset(768),
            max_blocks_per_seq,
            num_seqs: rows as u32,
        };
        let layer_ctx = ForwardContext {
            attn_metadata: Some(metadata),
            comm: None,
            graph_capture: false,
            ..*ctx
        };
        if mtp_kv_only_replay_enabled() {
            self.layer.decode_qwen4_batched_mtp(
                hidden,
                ctx.buffers.residual(),
                rows,
                emit_last_draft,
                state.layer_state.as_mut(),
                &mut kv_cache,
                state.seq_len,
                &mut state.block_table,
                &mut state.disk_block_ids,
                &mut state.disk_last_offloaded_per_layer,
                &layer_ctx,
                stream,
            )?;
        } else {
            self.layer.decode_qwen4_batched(
                hidden,
                ctx.buffers.residual(),
                rows,
                state.layer_state.as_mut(),
                &mut kv_cache,
                state.seq_len,
                &mut state.block_table,
                &mut state.disk_block_ids,
                &mut state.disk_last_offloaded_per_layer,
                DevicePtr::NULL,
                DevicePtr::NULL,
                0,
                0,
                &layer_ctx,
                stream,
            )?;
        }
        drop(kv_cache);

        if !emit_last_draft {
            state.seq_len += rows;
            return Ok(0);
        }

        let last_hidden = hidden.offset((rows - 1) * r_bytes);
        let (sample_hidden, inject) = self.final_mixer.prepare_decode(
            last_hidden,
            ctx.buffers.residual(),
            ctx.buffers,
            ctx.gpu,
            eps,
            stream,
        )?;
        debug_assert!(inject.is_none());
        let sync_gumbel_spec = if chain_out.is_none() && grammar_bitmask.is_none() {
            positions
                .last()
                .and_then(|&last_pos| state.resolve_gumbel_spec(last_pos))
        } else {
            None
        };
        let gumbel_active =
            chain_out.and_then(|(_, _, gumbel)| gumbel).is_some() || sync_gumbel_spec.is_some();
        let base_vocab = if qwen4_mtp_gate::mtp_vocab_prune_selected() {
            (qwen4_mtp_gate::PRUNED_VOCAB_SIZE as usize).min(ctx.config.vocab_size)
        } else if self.mtp_vocab_size == 0 {
            ctx.config.vocab_size
        } else {
            (self.mtp_vocab_size as usize).min(ctx.config.vocab_size)
        };
        let vocab = qwen4_mtp_gate::mtp_draft_vocab_override_for(gumbel_active)
            .map_or(base_vocab, |cap| base_vocab.min(cap));
        let logits = ctx.buffers.logits();
        ops::w4a16_decode_gemv(
            ctx.gpu,
            self.w4a16_gemv_k,
            self.w4a16_gemv_sw_k,
            self.gemv_sw,
            sample_hidden,
            &self.lm_head_nvfp4,
            logits,
            vocab as u32,
            h as u32,
            stream,
        )?;
        if let Some((argmax_out, prob_out, gumbel)) = chain_out {
            if let Some((spec, abs_pos)) = gumbel {
                gumbel_draft::select_chain_gumbel_step(
                    ctx.gpu,
                    self.mtp_gumbel_argmax_k,
                    logits,
                    DevicePtr::NULL,
                    argmax_out,
                    prob_out,
                    vocab,
                    spec,
                    abs_pos,
                    stream,
                )?;
            } else if let Some(prob_out) = prob_out {
                spark_runtime::kernel_args::KernelLaunch::new(ctx.gpu, self.argmax_prob_k)
                    .grid([1, 1, 1])
                    .block([1024, 1, 1])
                    .arg_ptr(logits)
                    .arg_ptr(argmax_out)
                    .arg_ptr(prob_out)
                    .arg_u32(vocab as u32)
                    .launch(stream)?;
            } else {
                ops::argmax_bf16(
                    ctx.gpu,
                    self.argmax_k,
                    logits,
                    argmax_out,
                    vocab as u32,
                    stream,
                )?;
            }
            state.seq_len += rows;
            return Ok(0);
        }
        let out = ctx.buffers.scratch();
        let token_id = if let Some(mask) = grammar_bitmask {
            let mut bytes = vec![0u8; vocab * 2];
            ctx.gpu.copy_d2h_on_stream(logits, &mut bytes, stream)?;
            let mut best = None::<(u32, f32)>;
            for candidate in 0..vocab {
                if candidate / 32 >= mask.len()
                    || (mask[candidate / 32] & (1i32 << (candidate % 32))) == 0
                {
                    continue;
                }
                let bits = u16::from_le_bytes([bytes[candidate * 2], bytes[candidate * 2 + 1]]);
                let value = f32::from_bits((bits as u32) << 16);
                if best.is_none_or(|(_, old)| value > old) {
                    best = Some((candidate as u32, value));
                }
            }
            best.map_or(0, |(candidate, _)| candidate)
        } else if let Some(spec) = sync_gumbel_spec {
            gumbel_draft::select_sync_gumbel_step(
                ctx.gpu,
                self.mtp_gumbel_argmax_k,
                logits,
                out,
                vocab,
                spec,
                spec.step0_abs_pos,
                stream,
            )?
        } else {
            ops::argmax_bf16(ctx.gpu, self.argmax_k, logits, out, vocab as u32, stream)?;
            let mut bytes = [0u8; 4];
            ctx.gpu.copy_d2h_on_stream(out, &mut bytes, stream)?;
            u32::from_le_bytes(bytes)
        };
        state.seq_len += rows;
        Ok(token_id)
    }

    /// Attention metadata for the MTP step writing KV row `seq_len` at rotary
    /// `position`, allocating its block if needed: position at +0, KV slot at
    /// +256, sequence length at +512, block table at +768.
    fn step_metadata(
        &self,
        kv_cache: &mut PagedKvCache,
        state: &mut Qwen4MtpState,
        seq_len: usize,
        position: usize,
    ) -> Result<Vec<u8>> {
        let bs = kv_cache.block_size();
        let blocks_needed = seq_len / bs + 1;
        while state.block_table.len() < blocks_needed {
            state.block_table.push(kv_cache.alloc_block()?);
        }
        let physical = state.block_table[seq_len / bs];
        let slot = physical as i64 * bs as i64 + (seq_len % bs) as i64;
        let draft_blocks: Vec<u32> = if mtp_bounded_blocks_selected()
            && !self.mtp_qsa_active()
            && state.block_table.len() > 192
        {
            let mut b = Vec::with_capacity(192);
            b.extend_from_slice(&state.block_table[..64]);
            b.extend_from_slice(&state.block_table[state.block_table.len() - 128..]);
            b
        } else {
            state.block_table.clone()
        };
        let bt_bytes = draft_blocks.len() * 4;
        let mut packed = vec![0u8; 768 + bt_bytes];
        write_rotary_metadata(&state.rotary_positions, position, &mut packed)?;
        packed[256..264].copy_from_slice(&slot.to_le_bytes());
        let effective_seq_len = (seq_len + 1).min(draft_blocks.len() * bs) as i32;
        packed[512..516].copy_from_slice(&effective_seq_len.to_le_bytes());
        for (i, block) in draft_blocks.iter().enumerate() {
            packed[768 + i * 4..772 + i * 4].copy_from_slice(&(*block as i32).to_le_bytes());
        }
        Ok(packed)
    }

    /// Multi-row attention metadata matching `Qwen3AttentionLayer::decode_qwen4_batched_inner`:
    /// `positions` at `+0..4*M`, `positions_h` at `+64..64+4*M` (when non-identity),
    /// `positions_w` at `+128..128+4*M` (when non-identity), `slot` at `+256..256+8*M`,
    /// `seq_len` at `+512..512+4*M`, and `M` copies of `state.block_table` at `+768`.
    fn batch_metadata(
        &self,
        kv_cache: &mut PagedKvCache,
        state: &mut Qwen4MtpState,
        seq_len: usize,
        positions: &[usize],
    ) -> Result<Vec<u8>> {
        let rows = positions.len();
        let bs = kv_cache.block_size();
        let last_seq_len = seq_len + rows - 1;
        let blocks_needed = last_seq_len / bs + 1;
        while state.block_table.len() < blocks_needed {
            state.block_table.push(kv_cache.alloc_block()?);
        }
        let draft_blocks: Vec<u32> = if mtp_bounded_blocks_selected()
            && !self.mtp_qsa_active()
            && state.block_table.len() > 192
        {
            let mut b = Vec::with_capacity(192);
            b.extend_from_slice(&state.block_table[..64]);
            b.extend_from_slice(&state.block_table[state.block_table.len() - 128..]);
            b
        } else {
            state.block_table.clone()
        };
        let mb = draft_blocks.len();
        let bt_row_bytes = mb * 4;
        let mut packed = vec![0u8; 768 + rows * bt_row_bytes];
        for (i, block) in draft_blocks.iter().enumerate() {
            packed[768 + i * 4..772 + i * 4].copy_from_slice(&(*block as i32).to_le_bytes());
        }
        for (row, &position) in positions.iter().enumerate() {
            let [t, h, w] = state.rotary_positions.position(position)?;
            packed[row * 4..row * 4 + 4].copy_from_slice(&t.to_le_bytes());
            if !state.rotary_positions.is_identity() {
                packed[64 + row * 4..68 + row * 4].copy_from_slice(&h.to_le_bytes());
                packed[128 + row * 4..132 + row * 4].copy_from_slice(&w.to_le_bytes());
            }
            let r_seq = seq_len + row;
            let physical = state.block_table[r_seq / bs];
            let slot = physical as i64 * bs as i64 + (r_seq % bs) as i64;
            packed[256 + row * 8..264 + row * 8].copy_from_slice(&slot.to_le_bytes());
            let effective_seq_len = (r_seq + 1).min(draft_blocks.len() * bs) as i32;
            packed[512 + row * 4..516 + row * 4].copy_from_slice(&effective_seq_len.to_le_bytes());
            if row > 0 {
                packed.copy_within(768..768 + bt_row_bytes, 768 + row * bt_row_bytes);
            }
        }
        Ok(packed)
    }

    /// Greedy draft chain with one metadata upload and one readback: each
    /// step's argmax stays on the device and feeds the next step's embedding
    /// gather, so the host never waits between drafts. Same arithmetic per
    /// step as the one-step path. With an early-stop gate the host reads each
    /// draft's probability before launching the next step and stops at the
    /// first one below the threshold. Returns (kept drafts, computed steps).
    #[allow(clippy::too_many_arguments)]
    fn propose_chain(
        &self,
        last_token: u32,
        target_hidden: DevicePtr,
        position: usize,
        num_drafts: usize,
        policy: GatePolicy,
        first_batch: Option<(&[u32], &[DevicePtr], &[usize])>,
        state: &mut Qwen4MtpState,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<(Vec<u32>, usize)> {
        static DEVICE_PROPOSE_LOGGED: std::sync::Once = std::sync::Once::new();
        DEVICE_PROPOSE_LOGGED.call_once(|| {
            tracing::info!(
                "mtp_propose: MTP_DEVICE_PROPOSE_CHAIN_ENGAGED chained_steps=1..3 host_roundtrips=0"
            );
        });
        validate_rotary_span(
            &state.rotary_positions,
            position,
            num_drafts,
            ctx.config.mrope_interleaved,
            self.mrope_kernel_available,
            ctx.config.yarn_factor,
        )?;
        let scratch = ctx.buffers.scratch();
        let replay_prefix = first_batch.map_or(0, |(t, _, _)| t.len() - 1);
        let first_seq_len = state.seq_len + replay_prefix;
        let bs = {
            let mut kv_cache = self.kv_cache.lock();
            let bs = kv_cache.block_size();
            let last = first_seq_len + num_drafts - 1;
            let blocks_needed = last / bs + 1;
            while state.block_table.len() < blocks_needed {
                state.block_table.push(kv_cache.alloc_block()?);
            }
            bs
        };
        let draft_blocks: Vec<u32> = if mtp_bounded_blocks_selected()
            && !self.mtp_qsa_active()
            && state.block_table.len() > 192
        {
            let mut b = Vec::with_capacity(192);
            b.extend_from_slice(&state.block_table[..64]);
            b.extend_from_slice(&state.block_table[state.block_table.len() - 128..]);
            b
        } else {
            state.block_table.clone()
        };
        let bt_bytes = draft_blocks.len() * 4;
        let stride = (768 + bt_bytes).next_multiple_of(256);
        let batch_meta_bytes = first_batch.map_or(0, |(t, _, _)| {
            (768 + t.len() * bt_bytes).next_multiple_of(256)
        });
        ensure!(
            CHAIN_META_OFFSET + num_drafts * stride + batch_meta_bytes
                <= ctx.buffers.sizes().scratch
                && CHAIN_ARGMAX_OFFSET + num_drafts * 4 <= CHAIN_TOKEN_OFFSET,
            "Qwen4 MTP draft chain of {num_drafts} steps does not fit the scratch layout"
        );
        let mut packed_guard = self.packed_meta_scratch.lock();
        let packed = prepare_packed_meta_scratch(&mut packed_guard, num_drafts * stride);
        for (i, block) in draft_blocks.iter().enumerate() {
            packed[768 + i * 4..772 + i * 4].copy_from_slice(&(*block as i32).to_le_bytes());
        }
        for step in 0..num_drafts {
            let s_off = step * stride;
            let s_seq = first_seq_len + step;
            let physical = state.block_table[s_seq / bs];
            let slot = physical as i64 * bs as i64 + (s_seq % bs) as i64;
            write_rotary_metadata(
                &state.rotary_positions,
                position + step,
                &mut packed[s_off..],
            )?;
            packed[s_off + 256..s_off + 264].copy_from_slice(&slot.to_le_bytes());
            let effective_seq_len = (s_seq + 1).min(draft_blocks.len() * bs) as i32;
            packed[s_off + 512..s_off + 516].copy_from_slice(&effective_seq_len.to_le_bytes());
            if step > 0 {
                packed.copy_within(768..768 + bt_bytes, s_off + 768);
            }
        }
        let meta_base = scratch.offset(CHAIN_META_OFFSET);
        self.copy_h2d_group_fast(ctx.gpu, &[HostToDeviceCopy::new(packed, meta_base)], stream)?;
        drop(packed_guard);

        let argmax_base = scratch.offset(CHAIN_ARGMAX_OFFSET);
        let gumbel_spec = state.resolve_gumbel_spec(position);
        let conf_gate = resolve_chain_confidence_gate(
            policy,
            policy.wants_probabilities(),
            policy.early_threshold(),
            self.argmax_prob_k.0 != 0,
            gumbel_spec.is_some(),
        );
        let want_prob = conf_gate.want_prob;
        ensure!(
            !want_prob || num_drafts * 4 <= CHAIN_PROB_SPAN,
            "Qwen4 MTP draft confidence supports at most {} drafts",
            CHAIN_PROB_SPAN / 4
        );
        let early = conf_gate.early;
        // A width floor (QSA context) lets the chain stop early only where it
        // would keep zero/one draft; otherwise every draft is computed and kept.
        let min_width = qwen4_mtp_gate::chain_min_width();
        let prob_base = scratch.offset(CHAIN_PROB_OFFSET);
        let r_bytes = ctx.config.residual_width() * 2;
        let mut hidden = target_hidden;
        let mut computed = 0usize;
        let mut p_cum = 1.0f32;
        let qsa_mode = mtp_qsa_mode();
        let has_qsa = self.layer.has_qsa_side_cache();
        let chain_max_blocks = draft_blocks.len() as u32;
        for step in 0..num_drafts {
            let is_last_step = step + 1 == num_drafts;
            let (qsa_required, reuse_action) =
                draft_step_qsa_plan(qsa_mode, has_qsa, first_seq_len, step, num_drafts);
            let _reuse_guard = reuse_action.map(|action| match action {
                DraftStepReuseAction::SnapshotStep0 => {
                    crate::layers::qwen4_qsa::DraftSelectionReuseGuard::snapshot_step0()
                }
                DraftStepReuseAction::ReuseStep {
                    base_seq_len,
                    step_offset,
                    max_drafts,
                } => crate::layers::qwen4_qsa::DraftSelectionReuseGuard::reuse_step(
                    base_seq_len,
                    step_offset,
                    max_drafts,
                ),
            });
            let chain = ChainStep {
                meta: meta_base.offset(step * stride),
                max_blocks_per_seq: chain_max_blocks,
                device_token: (step > 0).then(|| argmax_base.offset((step - 1) * 4)),
                argmax_out: argmax_base.offset(step * 4),
                prob_out: want_prob.then(|| prob_base.offset(step * 4)),
                gumbel: gumbel_spec.map(|spec| (spec, spec.step0_abs_pos + step as u64)),
            };
            if step == 0
                && let Some((b_tokens, b_hiddens, b_positions)) = first_batch
            {
                self.forward_rows(
                    b_tokens,
                    b_hiddens,
                    b_positions,
                    meta_base.offset(num_drafts * stride),
                    state,
                    ctx,
                    stream,
                    None,
                    true,
                    Some((chain.argmax_out, chain.prob_out, chain.gumbel)),
                )?;
                computed = 1;
                if !is_last_step {
                    hidden = ctx.buffers.hidden_states().offset(replay_prefix * r_bytes);
                }
            } else {
                self.forward_one(
                    last_token,
                    hidden,
                    position + step,
                    state,
                    ctx,
                    stream,
                    None,
                    true,
                    qsa_required,
                    Some(chain),
                )?;
                computed = step + 1;
                if !is_last_step {
                    hidden = ctx.buffers.hidden_states();
                }
            }
            static SYNC_MID_CHAIN_ENV: std::sync::OnceLock<Option<String>> =
                std::sync::OnceLock::new();
            let sync_mid_chain = should_sync_mid_chain(
                SYNC_MID_CHAIN_ENV
                    .get_or_init(|| std::env::var("ATLAS_QWEN4_MTP_CONF_EARLY_SYNC").ok())
                    .as_deref(),
                num_drafts,
            );
            if sync_mid_chain
                && !is_last_step
                && let Some(threshold) = early
            {
                let mut pb = [0u8; 4];
                ctx.gpu
                    .copy_d2h_on_stream(prob_base.offset(step * 4), &mut pb, stream)?;
                let p = f32::from_le_bytes(pb);
                p_cum *= p;
                let keep_opt = if gumbel_spec.is_some() {
                    qwen4_mtp_gate::early_stop_keep_step_gumbel(
                        step,
                        p,
                        p_cum,
                        qwen4_mtp_gate::gumbel_cum_threshold(),
                    )
                } else {
                    qwen4_mtp_gate::early_stop_keep_step(step, p, p_cum, threshold)
                };
                if let Some(keep) = keep_opt
                    && min_width.is_none_or(|w| keep < 2 || keep + 1 >= w)
                {
                    break;
                }
            }
        }
        let mut bytes = vec![0u8; computed * 4];
        let mut pbytes = vec![0u8; if want_prob { computed * 4 } else { 0 }];
        if want_prob {
            ctx.gpu.copy_d2h_pair_on_stream(
                argmax_base,
                &mut bytes,
                prob_base,
                &mut pbytes,
                stream,
            )?;
        } else {
            ctx.gpu
                .copy_d2h_on_stream(argmax_base, &mut bytes, stream)?;
        }
        let mut drafts: Vec<u32> = bytes
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        // Every computed draft (the gate may verify fewer): the committed-row
        // rewrite reads the accepted ones after the verify.
        state.last_drafts.clear();
        state.last_drafts.extend_from_slice(&drafts);
        if want_prob {
            let probs: Vec<f32> = pbytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect();
            let keep = if conf_gate.apply_truncate {
                let keep = if gumbel_spec.is_some() {
                    qwen4_mtp_gate::confident_prefix_gumbel(
                        &probs,
                        qwen4_mtp_gate::gumbel_cum_threshold(),
                    )
                } else {
                    policy
                        .effective_base_threshold()
                        .map_or(drafts.len(), |t| confident_prefix(&probs, t))
                };
                min_width.map_or(keep, |w| qwen4_mtp_gate::floor_keep(keep, computed, w))
            } else {
                drafts.len()
            };
            if policy.log {
                tracing::info!(
                    "MTP_DRAFT_CONF pos={position} n={num_drafts} computed={computed} \
                     drafts={drafts:?} p={probs:.4?} keep={keep}"
                );
            }
            if conf_gate.apply_truncate {
                drafts.truncate(keep);
            }
        }
        Ok((drafts, computed))
    }

    /// Collect the `(tokens, target_hiddens, positions)` tuples for a planned
    /// K7 committed-row rewrite (`ATLAS_QWEN4_MTP_RECONDITION`).
    fn collect_replay_rows(
        &self,
        pending: PendingReplay,
        stack: Option<DevicePtr>,
        bonus_position: usize,
        state: &Qwen4MtpState,
        ctx: &ForwardContext,
    ) -> Result<Option<(Vec<u32>, Vec<DevicePtr>, Vec<usize>)>> {
        let Some(stack) = stack else {
            static NO_STACK: std::sync::Once = std::sync::Once::new();
            NO_STACK.call_once(|| {
                tracing::warn!(
                    "QWEN4_MTP_RECONDITION_REFUSED reason=no_staged_rows: the target passed no \
                     staged hidden rows; the planned rows stay unwritten (drafts only)"
                );
            });
            return Ok(None);
        };
        let stride = ctx.config.residual_width() * 2;
        let staged_rows = qwen4_mtp_rows::staged_rows_capacity();
        let mut tokens = Vec::with_capacity(pending.plan.replay_rows + 1);
        let mut hiddens = Vec::with_capacity(pending.plan.replay_rows + 1);
        let mut positions = Vec::with_capacity(pending.plan.replay_rows + 1);
        for row in qwen4_mtp_rows::replay_rows(pending.plan) {
            let token = *state.last_drafts.get(row.draft).ok_or_else(|| {
                anyhow::anyhow!(
                    "Qwen4 MTP recondition: draft {} missing ({} recorded)",
                    row.draft,
                    state.last_drafts.len()
                )
            })?;
            let position =
                qwen4_mtp_rows::replay_position(bonus_position, pending.staged_accepted, row.index)
                    .ok_or_else(|| {
                        anyhow::anyhow!(
                            "Qwen4 MTP recondition: row {} has no position before {bonus_position}",
                            row.index
                        )
                    })?;
            ensure!(
                row.hidden_row < staged_rows,
                "Qwen4 MTP recondition: hidden row {} beyond the {staged_rows} staged rows",
                row.hidden_row
            );
            tokens.push(token);
            hiddens.push(stack.offset(row.hidden_row * stride));
            positions.push(position);
        }
        let rows = tokens.len();
        static ENGAGED: std::sync::Once = std::sync::Once::new();
        ENGAGED.call_once(|| {
            tracing::info!(
                "QWEN4_MTP_RECONDITION_ENGAGED selector={} first_rows={rows} accepted={} \
                 keep={} staged_rows={staged_rows}",
                qwen4_mtp_rows::RECONDITION_SELECTOR,
                pending.accepted,
                pending.plan.keep_rows,
            );
        });
        Ok(Some((tokens, hiddens, positions)))
    }
}

pub const MTP_STEP0_DUAL_GEMV_SELECTOR: &str = "ATLAS_QWEN4_MTP_STEP0_DUAL_GEMV";

pub fn mtp_step0_dual_gemv_selected() -> bool {
    static S: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *S.get_or_init(|| {
        let on = std::env::var(MTP_STEP0_DUAL_GEMV_SELECTOR).ok().as_deref() != Some("0");
        if on {
            static STEP0_LOGGED: std::sync::Once = std::sync::Once::new();
            STEP0_LOGGED.call_once(|| {
                tracing::info!(
                    "MTP_STEP0_OPTIM_ENGAGED dual_gemv=fc_hidden+fc_embedding hidden=2560"
                );
            });
        }
        on
    })
}

pub const MTP_DEVICE_PROPOSE_SELECTOR: &str = "ATLAS_QWEN4_MTP_DEVICE_PROPOSE";
pub const MTP_CHAIN_ALIAS: &str = "ATLAS_QWEN4_MTP_CHAIN";

pub fn mtp_device_propose_selected() -> bool {
    static S: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *S.get_or_init(|| {
        let dev = std::env::var(MTP_DEVICE_PROPOSE_SELECTOR).ok();
        let chain = std::env::var(MTP_CHAIN_ALIAS).ok();
        let on = dev.as_deref() != Some("0") && chain.as_deref() != Some("0");
        if on {
            static LOGGED: std::sync::Once = std::sync::Once::new();
            LOGGED.call_once(|| {
                tracing::info!("mtp_propose: MTP_DEVICE_PROPOSE_CHAIN_ENGAGED chained_steps=1..3 host_roundtrips=0");
            });
        }
        on
    })
}

/// Enabled by default; disabled when either `ATLAS_QWEN4_MTP_DEVICE_PROPOSE=0` or `ATLAS_QWEN4_MTP_CHAIN=0` is set.
fn mtp_chain_enabled() -> bool {
    mtp_device_propose_selected()
}

/// True when batched multi-row MTP execution (`forward_rows`) is enabled for
/// `ATLAS_QWEN4_MTP_RECONDITION` + `draft[0]` and `prefill_last_k`. Enabled by
/// default whenever `ATLAS_QWEN4_MTP_RECONDITION` or `ATLAS_QWEN4_VERIFY_FAST_HOST=1`
/// or `ATLAS_QWEN4_MTP_BATCH_REPLAY=1` is active unless explicitly disabled via
/// `ATLAS_QWEN4_MTP_BATCH_REPLAY=0`.
fn mtp_batch_rows_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let env = std::env::var("ATLAS_QWEN4_MTP_BATCH_REPLAY").ok();
        if env.as_deref() == Some("0") {
            return false;
        }
        let on = env.as_deref() == Some("1")
            || crate::layers::qwen4_verify_fast_host_selected()
            || qwen4_mtp_rows::recondition_engaged();
        if on {
            tracing::info!("QWEN4_MTP_BATCH_ROWS_ENGAGED selector=ATLAS_QWEN4_MTP_BATCH_REPLAY");
        }
        on
    })
}

/// True when KV-only MTP replay (`decode_kv_only` / `decode_qwen4_batched_mtp`)
/// is enabled for non-emitting replay rows. Default on unless
/// `ATLAS_QWEN4_MTP_KV_ONLY_REPLAY=0`.
fn mtp_kv_only_replay_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("ATLAS_QWEN4_MTP_KV_ONLY_REPLAY")
            .ok()
            .as_deref()
            != Some("0")
    })
}

/// Selector enabling per-sequence running-product `p_cum` early stop and
/// active-lane compaction inside cross-sequence batched MTP (`propose_batch`).
/// Default ON when `ATLAS_QWEN4_BATCHED_ADAPTIVE_K=1` unless explicitly set to `0`.
pub const BATCHED_RAGGED_MTP_ENV: &str = "ATLAS_QWEN4_BATCHED_RAGGED_MTP";

/// Cumulative probability threshold (`p_cum = prod p_i`) below which a sequence
/// stops drafting and is compacted out of subsequent batched MTP steps.
/// Default `0.30` (aligned with `C=1`'s `0.38` Gumbel cumulative gate while
/// preserving deeper drafts on high-confidence prose/code lanes); `0` disables early drop.
pub const MTP_RAGGED_PCUM_THRESH_ENV: &str = "ATLAS_QWEN4_MTP_RAGGED_PCUM_THRESH";
pub const DEFAULT_MTP_RAGGED_PCUM_THRESH: f32 = 0.30;

#[inline]
#[must_use]
fn parse_batched_ragged_mtp(raw: Option<&str>, adaptive_k: bool) -> bool {
    if !adaptive_k {
        return false;
    }
    !matches!(
        raw.map(str::trim),
        Some(v) if v == "0" || v.eq_ignore_ascii_case("false") || v.eq_ignore_ascii_case("off")
    )
}

/// True when ragged mixed-`K` batched verify and active-subset compacted MTP
/// drafting are enabled (default ON when `ATLAS_QWEN4_BATCHED_ADAPTIVE_K=1`;
/// set `ATLAS_QWEN4_BATCHED_RAGGED_MTP=0` to restore uniform cohort widths).
pub fn batched_ragged_mtp_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let adaptive = crate::layers::qwen4_verify_lanes::adaptive_k_selected().unwrap_or(false);
        let raw = std::env::var(BATCHED_RAGGED_MTP_ENV).ok();
        let on = parse_batched_ragged_mtp(raw.as_deref(), adaptive);
        if on {
            tracing::info!(
                selector = BATCHED_RAGGED_MTP_ENV,
                pcum_thresh = mtp_ragged_pcum_thresh(),
                "QWEN4_BATCHED_RAGGED_MTP_ENGAGED"
            );
        }
        on
    })
}

#[inline]
#[must_use]
fn parse_mtp_ragged_pcum_thresh(raw: Option<&str>) -> f32 {
    let Some(trimmed) = raw.map(str::trim).filter(|s| !s.is_empty()) else {
        return DEFAULT_MTP_RAGGED_PCUM_THRESH;
    };
    match trimmed.parse::<f32>() {
        Ok(v) if v.is_finite() && (0.0..1.0).contains(&v) => v,
        _ => DEFAULT_MTP_RAGGED_PCUM_THRESH,
    }
}

/// Cumulative confidence floor (`ATLAS_QWEN4_MTP_RAGGED_PCUM_THRESH`, default `0.30`).
pub fn mtp_ragged_pcum_thresh() -> f32 {
    static THRESH: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
    *THRESH.get_or_init(|| {
        parse_mtp_ragged_pcum_thresh(std::env::var(MTP_RAGGED_PCUM_THRESH_ENV).ok().as_deref())
    })
}

thread_local! {
    static RAGGED_MTP_TARGET_DRAFTS: std::cell::RefCell<Option<Vec<usize>>> =
        const { std::cell::RefCell::new(None) };
}

/// Optional per-sequence draft-depth ceiling for `propose_batch` under ragged
/// compaction (`target_drafts[s] == 0` excludes lane `s` from step 0).
pub struct RaggedMtpTargetDraftsGuard {
    prev: Option<Vec<usize>>,
}

impl RaggedMtpTargetDraftsGuard {
    #[must_use]
    pub fn enter(target_drafts: &[usize]) -> Self {
        let prev =
            RAGGED_MTP_TARGET_DRAFTS.with(|cell| cell.borrow_mut().replace(target_drafts.to_vec()));
        Self { prev }
    }
}

impl Drop for RaggedMtpTargetDraftsGuard {
    fn drop(&mut self) {
        let prev = self.prev.take();
        RAGGED_MTP_TARGET_DRAFTS.with(|cell| {
            *cell.borrow_mut() = prev;
        });
    }
}

fn current_ragged_mtp_target_drafts() -> Option<Vec<usize>> {
    RAGGED_MTP_TARGET_DRAFTS.with(|cell| cell.borrow().clone())
}

/// Whether `propose_chain` (C=1) performs mid-chain D2H probability syncs so
/// `MTP_CONF` stops drafting as soon as token probability drops below threshold,
/// honoring explicit `ATLAS_QWEN4_MTP_CONF_EARLY_SYNC=0|1`.
#[inline]
#[must_use]
fn should_sync_mid_chain(raw: Option<&str>, _num_drafts: usize) -> bool {
    match raw.map(str::trim) {
        Some("0") => false,
        Some("1") => true,
        _ => true,
    }
}

/// Keep draft token when cumulative confidence meets `pcum_thresh` (or early stop is disabled).
#[inline]
#[must_use]
fn ragged_mtp_keep_step(p_cum: f32, pcum_thresh: f32) -> bool {
    pcum_thresh <= 0.0 || p_cum >= pcum_thresh
}

/// Whether `propose_batch_ragged_compacted` can chain all `0..max_steps`
/// strictly on GPU (`embed_from_argmax_k` reading device-resident `argmax`
/// directly, ONE pre-loop H2D metadata upload, and ONE post-loop batched D2H
/// readback) with 0-ULP bit-identical results.
///
/// Requires:
/// - `pcum_thresh <= 0.0` (no lane can drop out early on host between steps),
/// - `max_steps > 1` (single-step chains already perform only one final D2H),
/// - every active lane targets the exact same `max_steps`,
/// - `[max_steps, active_lanes.len()]` `(token, prob)` spans and pre-packed
///   per-step metadata fit within the scratch arena.
#[inline]
#[must_use]
fn can_gpu_chain_ragged_steps(
    pcum_thresh: f32,
    active_lanes: &[usize],
    target_drafts: &[usize],
    max_steps: usize,
    argmax_span_bytes: usize,
    prob_span_bytes: usize,
    meta_end_bytes: usize,
    scratch_bytes: usize,
) -> bool {
    if pcum_thresh > 0.0 || max_steps <= 1 || active_lanes.is_empty() {
        return false;
    }
    if !active_lanes
        .iter()
        .all(|&orig_s| target_drafts.get(orig_s).copied() == Some(max_steps))
    {
        return false;
    }
    let Some(total_step_bytes) = max_steps
        .checked_mul(active_lanes.len())
        .and_then(|n| n.checked_mul(4))
    else {
        return false;
    };
    total_step_bytes <= argmax_span_bytes
        && total_step_bytes <= prob_span_bytes
        && meta_end_bytes <= scratch_bytes
}

/// Whether batched ragged MTP may emit `0` drafts (`K_s = 1`) on a hopeless
/// step-0 lane (`ATLAS_QWEN4_MTP_ALLOW_ZERO=1`). Default `false` keeps the
/// already-computed step-0 draft (`K_s = 2`) while still dropping the lane
/// from steps `1..`, avoiding ejecting the lane into a separate Phase-A
/// bootstrap pass in `mtp_step.rs` and preserving `min_k >= 2` KEEP_LATE
/// + CUDA graph capture/replay.
fn ragged_mtp_allow_zero_drafts() -> bool {
    static ALLOW: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ALLOW.get_or_init(|| {
        std::env::var("ATLAS_QWEN4_MTP_ALLOW_ZERO")
            .ok()
            .as_deref()
            .map(str::trim)
            == Some("1")
    })
}

/// Plan strictly non-overlapping left-to-right row copies `(src_row, dst_row, count)`
/// that compact `prev_active` rows into `next_active` rows in-place.
///
/// Every emitted tuple satisfies `dst_row + count <= src_row` so
/// `cudaMemcpyAsync` (device-to-device) never operates on overlapping spans.
fn plan_hidden_compaction_copies(
    prev_active: &[usize],
    next_active: &[usize],
) -> Vec<(usize, usize, usize)> {
    let mut old_indices = Vec::with_capacity(next_active.len());
    let mut prev_pos = 0usize;
    for &lane in next_active {
        while prev_pos < prev_active.len() && prev_active[prev_pos] != lane {
            prev_pos += 1;
        }
        if prev_pos < prev_active.len() {
            old_indices.push(prev_pos);
            prev_pos += 1;
        }
    }
    let mut copies = Vec::new();
    let mut new_i = 0usize;
    while new_i < old_indices.len() {
        let old_i = old_indices[new_i];
        if old_i == new_i {
            new_i += 1;
            continue;
        }
        let shift = old_i - new_i;
        let mut run = 1usize;
        while new_i + run < old_indices.len()
            && old_indices[new_i + run] == old_i + run
            && run < shift
        {
            run += 1;
        }
        copies.push((old_i, new_i, run));
        new_i += run;
    }
    copies
}

/// Advance one step of active-subset ragged MTP compaction on host after reading
/// back `[b_act]` `(token, prob)` pairs. Updates `steps_run`, `computed_drafts`,
/// `emitted_drafts`, `p_cum`, and `current_tokens`, returning the surviving
/// `next_active_lanes` for `step + 1`.
#[allow(clippy::too_many_arguments)]
fn advance_ragged_mtp_step(
    step: usize,
    max_steps: usize,
    pcum_thresh: f32,
    allow_zero: bool,
    active_lanes: &[usize],
    step_tokens: &[u32],
    step_probs: &[f32],
    target_drafts: &[usize],
    p_cum: &mut [f32],
    steps_run: &mut [usize],
    computed_drafts: &mut [Vec<u32>],
    emitted_drafts: &mut [Vec<u32>],
    current_tokens: &mut [u32],
    mut computed_probs: Option<&mut [Vec<f32>]>,
) -> Vec<usize> {
    let mut next_active = Vec::with_capacity(active_lanes.len());
    for (local_i, &orig_s) in active_lanes.iter().enumerate() {
        let tok = step_tokens[local_i];
        let prob = step_probs[local_i];
        steps_run[orig_s] += 1;
        computed_drafts[orig_s].push(tok);
        if let Some(probs_log) = computed_probs.as_deref_mut() {
            probs_log[orig_s].push(prob);
        }
        p_cum[orig_s] *= prob;
        let meets_thresh = ragged_mtp_keep_step(p_cum[orig_s], pcum_thresh);
        if meets_thresh {
            emitted_drafts[orig_s].push(tok);
            current_tokens[orig_s] = tok;
            if emitted_drafts[orig_s].len() < target_drafts[orig_s] && step + 1 < max_steps {
                next_active.push(orig_s);
            }
        } else if step == 0 && !allow_zero {
            // Keep the already-computed step-0 draft (`K_s = 2`) so the sequence
            // stays inside the batched verify cohort (no extra bootstrap forward)
            // while still stopping immediately after step 0 (zero work on steps 1..).
            emitted_drafts[orig_s].push(tok);
        }
    }
    next_active
}

/// Graph-bucket row granularity for ragged batched-verify frames of `b`
/// sequences so `BatchedVerifyGraphKey = (leader_slots, total_rows)` collapses
/// from ~45 distinct row counts per cohort down to ~3-5 cached CUDA graphs.
#[inline]
#[must_use]
const fn ragged_row_bucket_step(b: usize) -> usize {
    match b {
        0..=1 => 1,
        2 => 2,
        3..=5 => 4,
        6..=16 => 8,
        _ => 16,
    }
}

/// Stop ragged MTP drafting early at `C >= 4` once `step >= 2` (at least 3
/// steps run) and at most 25% of lanes survive, avoiding dragging an entire
/// wide cohort through extra MTP passes and D2H syncs for 1-4 stragglers.
#[inline]
#[must_use]
const fn ragged_mtp_should_stop_stragglers(
    cohort_b: usize,
    step: usize,
    next_active_len: usize,
) -> bool {
    cohort_b >= 4
        && step >= 2
        && next_active_len > 0
        && next_active_len.saturating_mul(4) <= cohort_b
}

/// Quantize ragged `emitted_drafts` total verify rows (`sum(drafts_s + 1)`)
/// down to a multiple of [`ragged_row_bucket_step`] by trimming the lowest
/// tail-`p_cum` draft(s) while preserving at least `min_drafts` per lane
/// (`1` when `!allow_zero`, so every lane retains `K_s >= 2`).
fn quantize_ragged_drafts_to_graph_bucket(
    emitted_drafts: &mut [Vec<u32>],
    emitted_pcum: &mut [Vec<f32>],
    allow_zero: bool,
) {
    let b = emitted_drafts.len();
    let step = ragged_row_bucket_step(b);
    if step <= 1 || b != emitted_pcum.len() {
        return;
    }
    let min_drafts = usize::from(!allow_zero);
    let total_rows: usize = emitted_drafts.iter().map(|d| d.len() + 1).sum();
    let mut rem = total_rows % step;
    while rem > 0 {
        let mut victim: Option<(usize, f32, usize)> = None;
        for s in 0..b {
            let len = emitted_drafts[s].len();
            if len <= min_drafts {
                continue;
            }
            let tail_p = emitted_pcum[s].last().copied().unwrap_or(1.0);
            let better = match victim {
                None => true,
                Some((best_s, best_p, best_len)) => {
                    tail_p < best_p
                        || (tail_p == best_p && (len > best_len || (len == best_len && s > best_s)))
                }
            };
            if better {
                victim = Some((s, tail_p, len));
            }
        }
        let Some((s, _, _)) = victim else {
            break;
        };
        emitted_drafts[s].pop();
        emitted_pcum[s].pop();
        rem -= 1;
    }
}

/// True when cross-sequence batched MTP draft proposals (`propose_batch`) are
/// enabled for C2..C8 cohorts. Default on unless `ATLAS_QWEN4_MTP_CROSS_SEQ=0`.
fn mtp_cross_seq_propose_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var("ATLAS_QWEN4_MTP_CROSS_SEQ").ok().as_deref() != Some("0");
        if on {
            tracing::info!("QWEN4_MTP_CROSS_SEQ_ENGAGED selector=ATLAS_QWEN4_MTP_CROSS_SEQ");
        }
        on
    })
}

/// The K7 head-row policy (`qwen4_mtp_rows`), its receipt logged once.
fn rows_policy_logged() -> RowPolicy {
    static LOGGED: std::sync::Once = std::sync::Once::new();
    let policy = qwen4_mtp_rows::policy();
    LOGGED.call_once(|| {
        if let Some(err) = qwen4_mtp_rows::policy_error() {
            tracing::warn!("QWEN4_MTP_ROWS_REFUSED reason=bad_selector ({err}); legacy head rows");
            return;
        }
        match policy {
            RowPolicy::Legacy => {}
            RowPolicy::KeepLast => tracing::info!(
                "QWEN4_MTP_KEEP_LAST_ROW_ENGAGED selector={}: the last committed row stays in the \
                 head cache",
                qwen4_mtp_rows::KEEP_LAST_ROW_SELECTOR
            ),
            RowPolicy::Recondition { max_rows } => {
                if qwen4_mtp_rows::recondition_engaged() {
                    tracing::info!(
                        "QWEN4_MTP_RECONDITION_REQUESTED selector={} max_rows={max_rows} \
                         staged_rows={}",
                        qwen4_mtp_rows::RECONDITION_SELECTOR,
                        qwen4_mtp_rows::staged_rows_capacity()
                    );
                } else {
                    tracing::warn!(
                        "QWEN4_MTP_RECONDITION_REFUSED reason=no_staging_buffer staged_rows={} \
                         need={}; KEEP_LAST_ROW semantics",
                        qwen4_mtp_rows::staged_rows_capacity(),
                        qwen4_mtp_rows::MAX_STAGED_ROWS
                    );
                }
            }
        }
    });
    policy
}

/// Running counts for the K7 head-row levers (frames planned, frames whose
/// last committed row is now present, rewritten rows, frames still leaving a
/// hole), logged at 64, 128, 256, ... frames.
fn record_row_stats(computed: usize, accepted: usize, plan: AfterVerifyPlan) {
    use std::sync::atomic::{AtomicU64, Ordering};
    static FRAMES: AtomicU64 = AtomicU64::new(0);
    static FILLED: AtomicU64 = AtomicU64::new(0);
    static REPLAYED: AtomicU64 = AtomicU64::new(0);
    static HOLES: AtomicU64 = AtomicU64::new(0);
    if computed == 0 {
        return;
    }
    let accepted = accepted.min(computed);
    let covered = plan.keep_rows + plan.replay_rows;
    let frames = FRAMES.fetch_add(1, Ordering::Relaxed) + 1;
    if covered > accepted {
        FILLED.fetch_add(1, Ordering::Relaxed);
    }
    if covered < accepted + 1 {
        HOLES.fetch_add(1, Ordering::Relaxed);
    }
    REPLAYED.fetch_add(plan.replay_rows as u64, Ordering::Relaxed);
    if frames >= 64 && frames.is_power_of_two() {
        tracing::info!(
            "QWEN4_MTP_ROWS_STATS frames={frames} filled={} replayed_rows={} holes_left={}",
            FILLED.load(Ordering::Relaxed),
            REPLAYED.load(Ordering::Relaxed),
            HOLES.load(Ordering::Relaxed),
        );
    }
}

impl Qwen4MtpHead {
    /// Active-subset compacted cross-sequence MTP drafting with per-sequence
    /// running-product `p_cum = prod p_i` early stop.
    ///
    /// After each draft step, ONE tiny `[b_act]` `(token, prob)` D2H pair
    /// readback evaluates `p_cum[s] *= p_step` on host and drops lanes where
    /// `p_cum[s] < pcum_thresh` or `emitted[s].len() >= target_drafts[s]`.
    /// Surviving lanes compact `hidden` in-place (strictly non-overlapping D2D
    /// spans) and pack metadata only for `b_act` rows, so stopped streams
    /// execute ZERO further MTP layers/MoE/LM-head work. Supports `b_act`
    /// shrinking all the way down to `1` (`decode_qwen4_cross_seq_mtp` 1..=64 +
    /// 1-row `final_mixer.prepare_decode` / `w4a16_decode_gemv`).
    #[allow(clippy::too_many_arguments)]
    fn propose_batch_ragged_compacted(
        &self,
        last_tokens: &[u32],
        target_hiddens: &[DevicePtr],
        positions: &[usize],
        num_drafts: usize,
        policy: qwen4_mtp_gate::GatePolicy,
        gumbel_specs: &[Option<spark_runtime::sampler::gumbel::GumbelDraftSpec>],
        any_gumbel: bool,
        states: &mut [&mut dyn ProposerState],
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<Option<Vec<Vec<u32>>>> {
        let b = last_tokens.len();
        let h = ctx.config.hidden_size;
        let r = ctx.config.residual_width();
        let hc = ctx.config.hc_count;
        let eps = ctx.config.rms_norm_eps as f32;
        let row_bytes = h * 2;
        let r_bytes = r * 2;
        let base_vocab = if qwen4_mtp_gate::mtp_vocab_prune_selected() {
            (qwen4_mtp_gate::PRUNED_VOCAB_SIZE as usize).min(ctx.config.vocab_size)
        } else if self.mtp_vocab_size == 0 {
            ctx.config.vocab_size
        } else {
            (self.mtp_vocab_size as usize).min(ctx.config.vocab_size)
        };
        let vocab = qwen4_mtp_gate::mtp_draft_vocab_override_for(any_gumbel)
            .map_or(base_vocab, |cap| base_vocab.min(cap));
        let single_step_bytes = b * 4;
        if hc != 4
            || b * vocab * 2 > ctx.buffers.sizes().logits
            || CHAIN_ARGMAX_OFFSET + single_step_bytes > CHAIN_PROB_OFFSET
            || single_step_bytes > CHAIN_PROB_SPAN
        {
            return Ok(None);
        }

        let target_drafts: Vec<usize> = match current_ragged_mtp_target_drafts() {
            Some(scoped) if scoped.len() == b => {
                scoped.into_iter().map(|d| d.min(num_drafts)).collect()
            }
            _ => vec![num_drafts; b],
        };
        let max_steps = target_drafts.iter().copied().max().unwrap_or(0);
        let mut active_lanes: Vec<usize> = (0..b).filter(|&s| target_drafts[s] > 0).collect();

        let mut all_identity = true;
        for (s, st) in states.iter().enumerate() {
            let Some(state) = st.as_any().downcast_ref::<Qwen4MtpState>() else {
                return Ok(None);
            };
            if state.pending_replay.is_some() {
                return Ok(None);
            }
            if target_drafts[s] > 0 {
                validate_rotary_span(
                    &state.rotary_positions,
                    positions[s],
                    target_drafts[s],
                    ctx.config.mrope_interleaved,
                    self.mrope_kernel_available,
                    ctx.config.yarn_factor,
                )?;
            }
            if !state.rotary_positions.is_identity() {
                all_identity = false;
            }
        }

        let mut kv_cache = self.kv_cache.lock();
        let bs = kv_cache.block_size();
        let mut total_new_blocks = 0usize;
        let mut max_mb = 1usize;
        for (s, st) in states.iter().enumerate() {
            let state = st
                .as_any()
                .downcast_ref::<Qwen4MtpState>()
                .expect("validated above");
            let effective_blocks = if target_drafts[s] > 0 {
                let last = state.seq_len + target_drafts[s] - 1;
                let blocks_needed = last / bs + 1;
                total_new_blocks += blocks_needed.saturating_sub(state.block_table.len());
                state.block_table.len().max(blocks_needed)
            } else {
                state.block_table.len()
            };
            max_mb = max_mb.max(effective_blocks);
        }
        if kv_cache.num_free_blocks() < total_new_blocks {
            return Ok(None);
        }
        let max_bt_off = if b <= 8 { 768 } else { 2560 };
        let bt_row_bytes = max_mb * 4;
        let max_step_stride = (max_bt_off + b * bt_row_bytes).next_multiple_of(256);
        let shared_gumbel_spec = gumbel_draft::uniform_batch_gumbel_spec(gumbel_specs);
        let max_gumbel_bytes = shared_gumbel_spec
            .and_then(|_| BatchGumbelMetaLayout::compute(max_step_stride, b, 1))
            .map_or(0, |layout| layout.total_bytes);
        if CHAIN_META_OFFSET + max_step_stride + max_gumbel_bytes > ctx.buffers.sizes().scratch {
            return Ok(None);
        }
        for st in states.iter() {
            if let Some(state) = st.as_any().downcast_ref::<Qwen4MtpState>() {
                state.clear_gumbel_spec();
            }
        }
        spark_runtime::sampler::gumbel::clear_gumbel_draft_specs();
        let mut base_seq_lens = Vec::with_capacity(b);
        for (s, st) in states.iter_mut().enumerate() {
            let state = st
                .as_any_mut()
                .downcast_mut::<Qwen4MtpState>()
                .expect("validated above");
            state.staged_frame.lock().clear();
            base_seq_lens.push(state.seq_len);
            if target_drafts[s] > 0 {
                let last = state.seq_len + target_drafts[s] - 1;
                let blocks_needed = last / bs + 1;
                while state.block_table.len() < blocks_needed {
                    state.block_table.push(kv_cache.alloc_block()?);
                }
            }
        }

        if active_lanes.is_empty() || max_steps == 0 {
            drop(kv_cache);
            for st in states.iter_mut() {
                let state = st
                    .as_any_mut()
                    .downcast_mut::<Qwen4MtpState>()
                    .expect("validated above");
                state.last_num_drafted = 0;
                state.last_drafts.clear();
            }
            return Ok(Some(vec![Vec::new(); b]));
        }

        let scratch = ctx.buffers.scratch();
        let meta_base = scratch.offset(CHAIN_META_OFFSET);
        let argmax_dev = scratch.offset(CHAIN_ARGMAX_OFFSET);
        let prob_dev = scratch.offset(CHAIN_PROB_OFFSET);
        let embed = ctx.buffers.ssm_qkvz();
        let normed_embed = ctx.buffers.ssm_deinterleaved();
        let normed_hidden = ctx.buffers.residual();
        let embed_proj = ctx.buffers.qkv_output();
        let hidden = ctx.buffers.hidden_states();
        let logits = ctx.buffers.logits();

        let pcum_thresh = mtp_ragged_pcum_thresh();
        let allow_zero = ragged_mtp_allow_zero_drafts();
        let mut p_cum = vec![1.0f32; b];
        let mut steps_run = vec![0usize; b];
        let mut computed_drafts: Vec<Vec<u32>> = vec![Vec::with_capacity(max_steps); b];
        let mut emitted_drafts: Vec<Vec<u32>> = vec![Vec::with_capacity(max_steps); b];
        let mut emitted_pcum: Vec<Vec<f32>> = vec![Vec::with_capacity(max_steps); b];
        let mut computed_probs: Option<Vec<Vec<f32>>> =
            policy.log.then(|| vec![Vec::with_capacity(max_steps); b]);
        let mut current_tokens: Vec<u32> = last_tokens.to_vec();
        let mut prev_active_lanes: Vec<usize> = Vec::new();
        let mut tok_bytes = [0u8; 64 * 4];
        let mut prob_bytes = [0u8; 64 * 4];
        let mut step_tokens = [0u32; 64];
        let mut step_probs = [0.0f32; 64];

        // When `pcum_thresh <= 0.0` and every active lane runs all `max_steps`
        // (homogeneous multi-step batch), no host-side compaction can occur
        // between steps: pre-pack all `max_steps` metadata in ONE H2D, chain
        // `embed_from_argmax_k` directly off device `argmax` on GPU, and do
        // ONE batched `(token, prob)` D2H readback after the loop (0-ULP
        // bit-identical to per-step drains).
        let b_init = active_lanes.len();
        let (chain_prob_offset, chain_prob_span) = if b_init <= 8 {
            (CHAIN_PROB_OFFSET, CHAIN_PROB_SPAN)
        } else {
            (CHAIN_ARGMAX_OFFSET + 2048, 2048)
        };
        let chain_argmax_span = chain_prob_offset.saturating_sub(CHAIN_ARGMAX_OFFSET);
        let chain_prob_fits_before_meta =
            chain_prob_offset.saturating_add(chain_prob_span) <= CHAIN_META_OFFSET;
        let chain_bt_off = if b_init <= 8 { 768 } else { 2560 };
        let chain_mb = max_mb;
        let chain_bt_row_bytes = chain_mb * 4;
        let chain_stride = (chain_bt_off + b_init * chain_bt_row_bytes).next_multiple_of(256);
        let chain_gumbel = shared_gumbel_spec.and_then(|shared_spec| {
            let meta_steps_bytes = max_steps.checked_mul(chain_stride)?;
            let layout = BatchGumbelMetaLayout::compute(meta_steps_bytes, b_init, max_steps)?;
            let meta_end = CHAIN_META_OFFSET
                .checked_add(meta_steps_bytes)?
                .checked_add(layout.total_bytes)?;
            if meta_end > ctx.buffers.sizes().scratch {
                return None;
            }
            let specs: Vec<spark_runtime::sampler::gumbel::GumbelDraftSpec> = active_lanes
                .iter()
                .map(|&orig_s| gumbel_specs[orig_s].expect("uniform gumbel"))
                .collect();
            Some((shared_spec, specs, layout))
        });
        let chain_gumbel_bytes = chain_gumbel
            .as_ref()
            .map_or(0, |(_, _, layout)| layout.total_bytes);
        let chain_meta_end = CHAIN_META_OFFSET
            .checked_add(max_steps.saturating_mul(chain_stride))
            .and_then(|end| end.checked_add(chain_gumbel_bytes))
            .unwrap_or(usize::MAX);
        let gpu_chained = chain_prob_fits_before_meta
            && can_gpu_chain_ragged_steps(
                pcum_thresh,
                &active_lanes,
                &target_drafts,
                max_steps,
                chain_argmax_span,
                chain_prob_span,
                chain_meta_end,
                ctx.buffers.sizes().scratch,
            );
        let chain_prob_dev = scratch.offset(chain_prob_offset);
        if gpu_chained {
            let (
                pos_h_off,
                pos_w_off,
                last_tok_off,
                slot_off,
                seq_len_off,
                hidden_ptrs_off,
                bt_off,
            ) = if b_init <= 8 {
                (64, 128, 192, 256, 512, 576, 768)
            } else {
                (256, 512, 768, 1024, 1536, 2048, 2560)
            };
            {
                let mut packed_guard = self.packed_meta_scratch.lock();
                let packed = prepare_packed_meta_scratch(
                    &mut packed_guard,
                    max_steps * chain_stride + chain_gumbel_bytes,
                );
                for (local_i, &orig_s) in active_lanes.iter().enumerate() {
                    let state = states[orig_s]
                        .as_any()
                        .downcast_ref::<Qwen4MtpState>()
                        .expect("validated above");
                    packed[last_tok_off + local_i * 4..last_tok_off + local_i * 4 + 4]
                        .copy_from_slice(&current_tokens[orig_s].to_le_bytes());
                    packed[hidden_ptrs_off + local_i * 8..hidden_ptrs_off + local_i * 8 + 8]
                        .copy_from_slice(&target_hiddens[orig_s].0.to_le_bytes());
                    for (bi, &block) in state.block_table.iter().enumerate() {
                        let off = bt_off + local_i * chain_bt_row_bytes + bi * 4;
                        packed[off..off + 4].copy_from_slice(&(block as i32).to_le_bytes());
                    }
                }
                for step in 0..max_steps {
                    let s_off = step * chain_stride;
                    if step > 0 {
                        packed.copy_within(
                            bt_off..bt_off + b_init * chain_bt_row_bytes,
                            s_off + bt_off,
                        );
                    }
                    for (local_i, &orig_s) in active_lanes.iter().enumerate() {
                        let state = states[orig_s]
                            .as_any()
                            .downcast_ref::<Qwen4MtpState>()
                            .expect("validated above");
                        let [t, rh, rw] =
                            state.rotary_positions.position(positions[orig_s] + step)?;
                        packed[s_off + local_i * 4..s_off + local_i * 4 + 4]
                            .copy_from_slice(&t.to_le_bytes());
                        if !all_identity {
                            packed[s_off + pos_h_off + local_i * 4
                                ..s_off + pos_h_off + local_i * 4 + 4]
                                .copy_from_slice(&rh.to_le_bytes());
                            packed[s_off + pos_w_off + local_i * 4
                                ..s_off + pos_w_off + local_i * 4 + 4]
                                .copy_from_slice(&rw.to_le_bytes());
                        }
                        let r_seq = base_seq_lens[orig_s] + step;
                        let physical = state.block_table[r_seq / bs];
                        let slot = physical as i64 * bs as i64 + (r_seq % bs) as i64;
                        packed[s_off + slot_off + local_i * 8..s_off + slot_off + local_i * 8 + 8]
                            .copy_from_slice(&slot.to_le_bytes());
                        packed[s_off + seq_len_off + local_i * 4
                            ..s_off + seq_len_off + local_i * 4 + 4]
                            .copy_from_slice(&((r_seq + 1) as i32).to_le_bytes());
                    }
                }
                if let Some((_, ref act_specs, layout)) = chain_gumbel {
                    for (local_i, spec) in act_specs.iter().enumerate() {
                        let seed_off = layout.seeds_off + local_i * 8;
                        packed[seed_off..seed_off + 8].copy_from_slice(&spec.seed.to_le_bytes());
                        for step in 0..max_steps {
                            let pos_off = layout.step_pos_off(step, b_init) + local_i * 8;
                            let abs_pos = spec.step0_abs_pos + step as u64;
                            packed[pos_off..pos_off + 8].copy_from_slice(&abs_pos.to_le_bytes());
                        }
                    }
                }
                self.copy_h2d_group_fast(
                    ctx.gpu,
                    &[HostToDeviceCopy::new(packed, meta_base)],
                    stream,
                )?;
            }
        }

        for step in 0..max_steps {
            let b_act = active_lanes.len();
            if b_act == 0 {
                break;
            }
            // Compact surviving hidden rows in-place when any lane dropped.
            if !gpu_chained && step > 0 && active_lanes != prev_active_lanes {
                for (src_row, dst_row, count) in
                    plan_hidden_compaction_copies(&prev_active_lanes, &active_lanes)
                {
                    ctx.gpu.copy_d2d_async(
                        hidden.offset(src_row * r_bytes),
                        hidden.offset(dst_row * r_bytes),
                        count * r_bytes,
                        stream,
                    )?;
                }
            }

            let (
                pos_h_off,
                pos_w_off,
                last_tok_off,
                slot_off,
                seq_len_off,
                hidden_ptrs_off,
                bt_off,
            ) = if b_act <= 8 {
                (64, 128, 192, 256, 512, 576, 768)
            } else {
                (256, 512, 768, 1024, 1536, 2048, 2560)
            };
            let act_stride = (bt_off + b_act * bt_row_bytes).next_multiple_of(256);
            let step_meta = if gpu_chained {
                meta_base.offset(step * chain_stride)
            } else {
                meta_base
            };
            let step_argmax = if gpu_chained {
                argmax_dev.offset(step * b_act * 4)
            } else {
                argmax_dev
            };
            let step_prob = if gpu_chained {
                chain_prob_dev.offset(step * b_act * 4)
            } else {
                prob_dev
            };
            let step_gumbel = if gpu_chained {
                None
            } else {
                shared_gumbel_spec.and_then(|shared_spec| {
                    let layout = BatchGumbelMetaLayout::compute(act_stride, b_act, 1)?;
                    let meta_end = CHAIN_META_OFFSET
                        .checked_add(act_stride)?
                        .checked_add(layout.total_bytes)?;
                    if meta_end > ctx.buffers.sizes().scratch {
                        return None;
                    }
                    let specs: Vec<spark_runtime::sampler::gumbel::GumbelDraftSpec> = active_lanes
                        .iter()
                        .map(|&orig_s| gumbel_specs[orig_s].expect("uniform gumbel"))
                        .collect();
                    Some((shared_spec, specs, layout))
                })
            };
            let gumbel_bytes = step_gumbel
                .as_ref()
                .map_or(0, |(_, _, layout)| layout.total_bytes);

            if !gpu_chained {
                let mut packed_guard = self.packed_meta_scratch.lock();
                let packed =
                    prepare_packed_meta_scratch(&mut packed_guard, act_stride + gumbel_bytes);
                for (local_i, &orig_s) in active_lanes.iter().enumerate() {
                    let state = states[orig_s]
                        .as_any()
                        .downcast_ref::<Qwen4MtpState>()
                        .expect("validated above");
                    packed[last_tok_off + local_i * 4..last_tok_off + local_i * 4 + 4]
                        .copy_from_slice(&current_tokens[orig_s].to_le_bytes());
                    if step == 0 {
                        packed[hidden_ptrs_off + local_i * 8..hidden_ptrs_off + local_i * 8 + 8]
                            .copy_from_slice(&target_hiddens[orig_s].0.to_le_bytes());
                    }
                    let [t, rh, rw] = state.rotary_positions.position(positions[orig_s] + step)?;
                    packed[local_i * 4..local_i * 4 + 4].copy_from_slice(&t.to_le_bytes());
                    if !all_identity {
                        packed[pos_h_off + local_i * 4..pos_h_off + local_i * 4 + 4]
                            .copy_from_slice(&rh.to_le_bytes());
                        packed[pos_w_off + local_i * 4..pos_w_off + local_i * 4 + 4]
                            .copy_from_slice(&rw.to_le_bytes());
                    }
                    let r_seq = base_seq_lens[orig_s] + step;
                    let physical = state.block_table[r_seq / bs];
                    let slot = physical as i64 * bs as i64 + (r_seq % bs) as i64;
                    packed[slot_off + local_i * 8..slot_off + local_i * 8 + 8]
                        .copy_from_slice(&slot.to_le_bytes());
                    packed[seq_len_off + local_i * 4..seq_len_off + local_i * 4 + 4]
                        .copy_from_slice(&((r_seq + 1) as i32).to_le_bytes());
                    for (bi, &block) in state.block_table.iter().enumerate() {
                        let off = bt_off + local_i * bt_row_bytes + bi * 4;
                        packed[off..off + 4].copy_from_slice(&(block as i32).to_le_bytes());
                    }
                }
                if let Some((_, ref act_specs, layout)) = step_gumbel {
                    for (local_i, spec) in act_specs.iter().enumerate() {
                        let seed_off = layout.seeds_off + local_i * 8;
                        packed[seed_off..seed_off + 8].copy_from_slice(&spec.seed.to_le_bytes());
                        let pos_off = layout.step_pos_off(0, b_act) + local_i * 8;
                        let abs_pos = spec.step0_abs_pos + step as u64;
                        packed[pos_off..pos_off + 8].copy_from_slice(&abs_pos.to_le_bytes());
                    }
                }
                self.copy_h2d_group_fast(
                    ctx.gpu,
                    &[HostToDeviceCopy::new(packed, step_meta)],
                    stream,
                )?;
            }

            let tc_dense_scope =
                crate::layers::qwen4_hyper::batched_verify_tc_dense_active(b_act, false);
            let _tc_dense_guard =
                crate::layers::qwen4_hyper::BatchedVerifyTcDenseGuard::enter(tc_dense_scope);
            let coalesced_lm_head_eligible = should_use_mtp_lm_head_coalesced(
                crate::layers::qwen4_hyper::batched_verify_tc_component_active(
                    crate::layers::qwen4_hyper::BatchedVerifyTcComponent::LmHead,
                    b_act,
                    false,
                ),
                b_act,
                h,
            );
            let coalesced_fc_eligible = should_use_mtp_fc_coalesced(
                tc_dense_scope,
                b_act,
                hc,
                h,
                ctx.buffers.sizes().qkv_output,
                self.residual_add_broadcast_hc_k.0 != 0,
            );

            if step == 0 {
                if self.embed_dual_rms_norm_rows_packed_k.0 != 0
                    && h >= 1024
                    && r >= 1024
                    && b_act <= 8
                {
                    let tok_at = |idx: usize| -> u64 {
                        active_lanes
                            .get(idx)
                            .map(|&orig_s| current_tokens[orig_s] as u64)
                            .unwrap_or(0)
                    };
                    let tok_pack01 = tok_at(0) | (tok_at(1) << 32);
                    let tok_pack23 = tok_at(2) | (tok_at(3) << 32);
                    let tok_pack45 = tok_at(4) | (tok_at(5) << 32);
                    let tok_pack67 = tok_at(6) | (tok_at(7) << 32);
                    spark_runtime::kernel_args::KernelLaunch::new(
                        ctx.gpu,
                        self.embed_dual_rms_norm_rows_packed_k,
                    )
                    .grid([b_act as u32, 2, 1])
                    .block([1024, 1, 1])
                    .arg_ptr(self.embed_tokens.weight)
                    .arg_u64(tok_pack01)
                    .arg_u64(tok_pack23)
                    .arg_u64(tok_pack45)
                    .arg_u64(tok_pack67)
                    .arg_u32(ctx.config.vocab_size as u32)
                    .arg_ptr(self.pre_fc_norm_embedding.weight)
                    .arg_ptr(normed_embed)
                    .arg_ptr(target_hiddens[active_lanes[0]])
                    .arg_ptr(self.pre_fc_norm_hidden.weight)
                    .arg_ptr(normed_hidden)
                    .arg_u32(h as u32)
                    .arg_u32(r as u32)
                    .arg_f32(eps)
                    .arg_ptr(step_meta.offset(hidden_ptrs_off))
                    .launch(stream)?;
                } else {
                    spark_runtime::kernel_args::KernelLaunch::new(
                        ctx.gpu,
                        self.embed_from_argmax_k,
                    )
                    .grid([(h as u32).div_ceil(256), b_act as u32, 1])
                    .block([256, 1, 1])
                    .arg_ptr(step_meta.offset(last_tok_off))
                    .arg_ptr(self.embed_tokens.weight)
                    .arg_ptr(embed)
                    .arg_ptr(DevicePtr(0))
                    .arg_u32(h as u32)
                    .launch(stream)?;
                    if self.pre_fc_dual_norm_k.0 != 0 && h >= 1024 && r >= 1024 {
                        spark_runtime::kernel_args::KernelLaunch::new(
                            ctx.gpu,
                            self.pre_fc_dual_norm_k,
                        )
                        .grid([b_act as u32, 2, 1])
                        .block([1024, 1, 1])
                        .arg_ptr(embed)
                        .arg_ptr(self.pre_fc_norm_embedding.weight)
                        .arg_ptr(normed_embed)
                        .arg_ptr(target_hiddens[active_lanes[0]])
                        .arg_ptr(self.pre_fc_norm_hidden.weight)
                        .arg_ptr(normed_hidden)
                        .arg_u32(h as u32)
                        .arg_u32(r as u32)
                        .arg_f32(eps)
                        .arg_ptr(step_meta.offset(hidden_ptrs_off))
                        .launch(stream)?;
                    } else {
                        ops::rms_norm(
                            ctx.gpu,
                            self.rms_norm_k,
                            embed,
                            &self.pre_fc_norm_embedding,
                            normed_embed,
                            b_act as u32,
                            h as u32,
                            eps,
                            stream,
                        )?;
                        for (local_i, &orig_s) in active_lanes.iter().enumerate() {
                            ops::rms_norm(
                                ctx.gpu,
                                self.rms_norm_k,
                                target_hiddens[orig_s],
                                &self.pre_fc_norm_hidden,
                                normed_hidden.offset(local_i * r_bytes),
                                1,
                                r as u32,
                                eps,
                                stream,
                            )?;
                        }
                    }
                }
            } else {
                let prev_tok_ptr = if gpu_chained {
                    argmax_dev.offset((step - 1) * b_act * 4)
                } else {
                    step_meta.offset(last_tok_off)
                };
                spark_runtime::kernel_args::KernelLaunch::new(ctx.gpu, self.embed_from_argmax_k)
                    .grid([(h as u32).div_ceil(256), b_act as u32, 1])
                    .block([256, 1, 1])
                    .arg_ptr(prev_tok_ptr)
                    .arg_ptr(self.embed_tokens.weight)
                    .arg_ptr(embed)
                    .arg_ptr(DevicePtr(0))
                    .arg_u32(h as u32)
                    .launch(stream)?;
                if self.pre_fc_dual_norm_k.0 != 0 && h >= 1024 && r >= 1024 {
                    spark_runtime::kernel_args::KernelLaunch::new(ctx.gpu, self.pre_fc_dual_norm_k)
                        .grid([b_act as u32, 2, 1])
                        .block([1024, 1, 1])
                        .arg_ptr(embed)
                        .arg_ptr(self.pre_fc_norm_embedding.weight)
                        .arg_ptr(normed_embed)
                        .arg_ptr(hidden)
                        .arg_ptr(self.pre_fc_norm_hidden.weight)
                        .arg_ptr(normed_hidden)
                        .arg_u32(h as u32)
                        .arg_u32(r as u32)
                        .arg_f32(eps)
                        .arg_ptr(DevicePtr(0))
                        .launch(stream)?;
                } else {
                    ops::rms_norm(
                        ctx.gpu,
                        self.rms_norm_k,
                        embed,
                        &self.pre_fc_norm_embedding,
                        normed_embed,
                        b_act as u32,
                        h as u32,
                        eps,
                        stream,
                    )?;
                    ops::rms_norm(
                        ctx.gpu,
                        self.rms_norm_k,
                        hidden,
                        &self.pre_fc_norm_hidden,
                        normed_hidden,
                        b_act as u32,
                        r as u32,
                        eps,
                        stream,
                    )?;
                }
            }

            let ran_coalesced_fc = if coalesced_fc_eligible
                && crate::layers::qwen4_fast_proj::try_coalesced_nvfp4_w4a16_gemm(
                    ctx.gpu,
                    normed_embed,
                    &self.fc_embedding,
                    embed_proj,
                    b_act,
                    h,
                    h,
                    stream,
                )?
                && self.try_coalesced_mtp_fc_hidden(
                    ctx.gpu,
                    normed_hidden,
                    hidden,
                    b_act * hc,
                    h,
                    stream,
                )? {
                spark_runtime::kernel_args::KernelLaunch::new(
                    ctx.gpu,
                    self.residual_add_broadcast_hc_k,
                )
                .grid([(h as u32).div_ceil(256), b_act as u32, 1])
                .block([256, 1, 1])
                .arg_ptr(hidden)
                .arg_ptr(embed_proj)
                .arg_u32(h as u32)
                .arg_u32(hc as u32)
                .launch(stream)?;
                true
            } else {
                false
            };
            if !ran_coalesced_fc {
                if self.fc_dual_broadcast_rows_k.0 != 0 {
                    spark_runtime::kernel_args::KernelLaunch::new(
                        ctx.gpu,
                        self.fc_dual_broadcast_rows_k,
                    )
                    .grid([(h as u32).div_ceil(4), b_act as u32, 1])
                    .block([256, 1, 1])
                    .arg_ptr(normed_embed)
                    .arg_ptr(self.fc_embedding.weight)
                    .arg_ptr(self.fc_embedding.weight_scale)
                    .arg_f32(self.fc_embedding.weight_scale_2)
                    .arg_ptr(normed_hidden)
                    .arg_ptr(self.fc_hidden.weight)
                    .arg_ptr(self.fc_hidden.weight_scale)
                    .arg_f32(self.fc_hidden.weight_scale_2)
                    .arg_ptr(hidden)
                    .arg_u32(h as u32)
                    .arg_u32(h as u32)
                    .launch(stream)?;
                } else {
                    if b_act == 1 {
                        ops::w4a16_decode_gemv(
                            ctx.gpu,
                            self.w4a16_gemv_k,
                            self.w4a16_gemv_sw_k,
                            self.gemv_sw,
                            normed_embed,
                            &self.fc_embedding,
                            embed_proj,
                            h as u32,
                            h as u32,
                            stream,
                        )?;
                    } else {
                        ops::w4a16_gemv_batch_logits_exact_with(
                            ctx.gpu,
                            self.w4a16_exact_fc_kernels,
                            normed_embed,
                            &self.fc_embedding,
                            embed_proj,
                            b_act as u32,
                            h as u32,
                            h as u32,
                            stream,
                            false,
                        )?;
                    }
                    ops::w4a16_gemv_batch_logits_exact_with(
                        ctx.gpu,
                        self.w4a16_exact_fc_kernels,
                        normed_hidden,
                        &self.fc_hidden,
                        hidden,
                        (b_act * hc) as u32,
                        h as u32,
                        h as u32,
                        stream,
                        false,
                    )?;
                    if self.residual_add_broadcast_hc_k.0 != 0 {
                        spark_runtime::kernel_args::KernelLaunch::new(
                            ctx.gpu,
                            self.residual_add_broadcast_hc_k,
                        )
                        .grid([(h as u32).div_ceil(256), b_act as u32, 1])
                        .block([256, 1, 1])
                        .arg_ptr(hidden)
                        .arg_ptr(embed_proj)
                        .arg_u32(h as u32)
                        .arg_u32(hc as u32)
                        .launch(stream)?;
                    } else {
                        for j in 0..b_act {
                            let embed_row = embed_proj.offset(j * row_bytes);
                            for branch in 0..hc {
                                let hidden_branch = hidden.offset((j * hc + branch) * row_bytes);
                                ops::residual_add(
                                    ctx.gpu,
                                    self.residual_add_k,
                                    hidden_branch,
                                    embed_row,
                                    h as u32,
                                    stream,
                                )?;
                            }
                        }
                    }
                }
            }

            let (positions_h, positions_w) = if all_identity {
                (step_meta, step_meta)
            } else {
                (step_meta.offset(pos_h_off), step_meta.offset(pos_w_off))
            };
            let metadata = AttnMetadataDev {
                qwen4_qsa_required: self.mtp_qsa_active(),
                positions: step_meta,
                positions_h,
                positions_w,
                slot: step_meta.offset(slot_off),
                seq_len: step_meta.offset(seq_len_off),
                block_table: step_meta.offset(bt_off),
                max_blocks_per_seq: if gpu_chained {
                    chain_mb as u32
                } else {
                    max_mb as u32
                },
                num_seqs: b_act as u32,
            };
            let layer_ctx = ForwardContext {
                attn_metadata: Some(metadata),
                comm: None,
                graph_capture: false,
                ..*ctx
            };
            let step_seq_lens: Vec<usize> = active_lanes
                .iter()
                .map(|&orig_s| base_seq_lens[orig_s] + step)
                .collect();
            {
                let mut typed_states: Vec<&mut Qwen4MtpState> = Vec::with_capacity(b_act);
                let mut cursor = 0usize;
                for (orig_s, st) in states.iter_mut().enumerate() {
                    if cursor < b_act && active_lanes[cursor] == orig_s {
                        typed_states.push(
                            st.as_any_mut()
                                .downcast_mut::<Qwen4MtpState>()
                                .expect("validated above"),
                        );
                        cursor += 1;
                    }
                }
                let mut block_tables: Vec<&mut Vec<u32>> = Vec::with_capacity(b_act);
                let mut disk_block_ids: Vec<&mut Vec<u32>> = Vec::with_capacity(b_act);
                let mut disk_last_offloaded: Vec<&mut Vec<u32>> = Vec::with_capacity(b_act);
                for st in typed_states.iter_mut() {
                    let Qwen4MtpState {
                        block_table,
                        disk_block_ids: d_ids,
                        disk_last_offloaded_per_layer: d_off,
                        ..
                    } = &mut **st;
                    block_tables.push(block_table);
                    disk_block_ids.push(d_ids);
                    disk_last_offloaded.push(d_off);
                }
                self.layer.decode_qwen4_cross_seq_mtp(
                    hidden,
                    ctx.buffers.residual(),
                    &step_seq_lens,
                    &mut block_tables,
                    &mut disk_block_ids,
                    &mut disk_last_offloaded,
                    &mut kv_cache,
                    &layer_ctx,
                    stream,
                )?;
            }

            let sample_hiddens = if b_act == 1 {
                let (mixed, inject) = self.final_mixer.prepare_decode(
                    hidden,
                    ctx.buffers.residual(),
                    ctx.buffers,
                    ctx.gpu,
                    eps,
                    stream,
                )?;
                debug_assert!(inject.is_none());
                mixed
            } else {
                self.final_mixer.prepare_batched(
                    hidden,
                    ctx.buffers.residual(),
                    b_act,
                    ctx.buffers,
                    ctx.gpu,
                    eps,
                    stream,
                )?
            };

            if b_act == 1 {
                ops::w4a16_decode_gemv(
                    ctx.gpu,
                    self.w4a16_gemv_k,
                    self.w4a16_gemv_sw_k,
                    self.gemv_sw,
                    sample_hiddens,
                    &self.lm_head_nvfp4,
                    logits,
                    vocab as u32,
                    h as u32,
                    stream,
                )?;
            } else {
                let ran_coalesced_lm = if coalesced_lm_head_eligible {
                    crate::layers::qwen4_fast_proj::try_coalesced_nvfp4_w4a16_gemm(
                        ctx.gpu,
                        sample_hiddens,
                        &self.lm_head_nvfp4,
                        logits,
                        b_act,
                        vocab,
                        h,
                        stream,
                    )?
                } else {
                    false
                };
                if !ran_coalesced_lm {
                    if should_use_mtp_lm_head_pipe(
                        mtp_lm_head_tc_pipe_enabled(),
                        b_act,
                        h,
                        self.w4a16_gemm_pipe_k.0 != 0,
                        sample_hiddens,
                        self.lm_head_nvfp4.weight,
                        logits,
                    ) {
                        ops::w4a16_gemm_pipe(
                            ctx.gpu,
                            self.w4a16_gemm_pipe_k,
                            sample_hiddens,
                            &self.lm_head_nvfp4,
                            logits,
                            b_act as u32,
                            vocab as u32,
                            h as u32,
                            stream,
                        )?;
                    } else {
                        ops::w4a16_gemv_batch_logits_exact_with(
                            ctx.gpu,
                            self.w4a16_exact_fc_kernels,
                            sample_hiddens,
                            &self.lm_head_nvfp4,
                            logits,
                            b_act as u32,
                            vocab as u32,
                            h as u32,
                            stream,
                            ops::w4a16_exact_rt2_for_rows(
                                ops::w4a16_gemv_rt2_enabled(),
                                ops::w4a16_exact_rt2_min_rows(),
                                b_act as u32,
                            ),
                        )?;
                    }
                }
            }

            let active_gumbel = if gpu_chained {
                chain_gumbel
                    .as_ref()
                    .map(|(s, specs, l)| (*s, specs, *l, step))
            } else {
                step_gumbel.as_ref().map(|(s, specs, l)| (*s, specs, *l, 0))
            };
            if let Some((shared_spec, act_specs, layout, pos_step)) = active_gumbel {
                gumbel_draft::select_chain_gumbel_batch(
                    ctx.gpu,
                    self.mtp_gumbel_argmax_k,
                    logits,
                    DevicePtr::NULL,
                    meta_base.offset(layout.seeds_off),
                    meta_base.offset(layout.step_pos_off(pos_step, b_act)),
                    step_argmax,
                    Some(step_prob),
                    b_act,
                    vocab,
                    shared_spec,
                    act_specs,
                    step as u64,
                    stream,
                )?;
            } else if any_gumbel {
                for (local_i, &orig_s) in active_lanes.iter().enumerate() {
                    if let Some(spec) = gumbel_specs[orig_s] {
                        gumbel_draft::select_chain_gumbel_step(
                            ctx.gpu,
                            self.mtp_gumbel_argmax_k,
                            logits.offset(local_i * vocab * 2),
                            DevicePtr::NULL,
                            step_argmax.offset(local_i * 4),
                            Some(step_prob.offset(local_i * 4)),
                            vocab,
                            spec,
                            spec.step0_abs_pos + step as u64,
                            stream,
                        )?;
                    } else {
                        spark_runtime::kernel_args::KernelLaunch::new(ctx.gpu, self.argmax_prob_k)
                            .grid([1, 1, 1])
                            .block([1024, 1, 1])
                            .arg_ptr(logits.offset(local_i * vocab * 2))
                            .arg_ptr(step_argmax.offset(local_i * 4))
                            .arg_ptr(step_prob.offset(local_i * 4))
                            .arg_u32(vocab as u32)
                            .launch(stream)?;
                    }
                }
            } else {
                self.launch_argmax_prob_batch(
                    ctx.gpu,
                    logits,
                    step_argmax,
                    step_prob,
                    b_act,
                    vocab,
                    stream,
                )?;
            }

            if !gpu_chained {
                // ONE single-sync tiny `[b_act]` (token, prob) pair readback per step.
                let n_bytes = b_act * 4;
                ctx.gpu.copy_d2h_pair_on_stream(
                    step_argmax,
                    &mut tok_bytes[..n_bytes],
                    step_prob,
                    &mut prob_bytes[..n_bytes],
                    stream,
                )?;
                for local_i in 0..b_act {
                    let off = local_i * 4;
                    step_tokens[local_i] = u32::from_le_bytes([
                        tok_bytes[off],
                        tok_bytes[off + 1],
                        tok_bytes[off + 2],
                        tok_bytes[off + 3],
                    ]);
                    step_probs[local_i] = f32::from_le_bytes([
                        prob_bytes[off],
                        prob_bytes[off + 1],
                        prob_bytes[off + 2],
                        prob_bytes[off + 3],
                    ]);
                }
                let mut next_active = advance_ragged_mtp_step(
                    step,
                    max_steps,
                    pcum_thresh,
                    allow_zero,
                    &active_lanes,
                    &step_tokens[..b_act],
                    &step_probs[..b_act],
                    &target_drafts,
                    &mut p_cum,
                    &mut steps_run,
                    &mut computed_drafts,
                    &mut emitted_drafts,
                    &mut current_tokens,
                    computed_probs.as_deref_mut(),
                );
                for &orig_s in &active_lanes {
                    if emitted_drafts[orig_s].len() > emitted_pcum[orig_s].len() {
                        emitted_pcum[orig_s].push(p_cum[orig_s]);
                    }
                }
                if ragged_mtp_should_stop_stragglers(b, step, next_active.len()) {
                    next_active.clear();
                }
                prev_active_lanes = active_lanes;
                active_lanes = next_active;
            }
            drop(_tc_dense_guard);
        }
        if gpu_chained {
            let total_bytes = max_steps * b_init * 4;
            let mut all_tok_bytes = vec![0u8; total_bytes];
            let mut all_prob_bytes = vec![0u8; total_bytes];
            ctx.gpu.copy_d2h_pair_on_stream(
                argmax_dev,
                &mut all_tok_bytes,
                chain_prob_dev,
                &mut all_prob_bytes,
                stream,
            )?;
            for step in 0..max_steps {
                let step_base = step * b_init * 4;
                for local_i in 0..b_init {
                    let off = step_base + local_i * 4;
                    step_tokens[local_i] = u32::from_le_bytes([
                        all_tok_bytes[off],
                        all_tok_bytes[off + 1],
                        all_tok_bytes[off + 2],
                        all_tok_bytes[off + 3],
                    ]);
                    step_probs[local_i] = f32::from_le_bytes([
                        all_prob_bytes[off],
                        all_prob_bytes[off + 1],
                        all_prob_bytes[off + 2],
                        all_prob_bytes[off + 3],
                    ]);
                }
                let _next = advance_ragged_mtp_step(
                    step,
                    max_steps,
                    pcum_thresh,
                    allow_zero,
                    &active_lanes,
                    &step_tokens[..b_init],
                    &step_probs[..b_init],
                    &target_drafts,
                    &mut p_cum,
                    &mut steps_run,
                    &mut computed_drafts,
                    &mut emitted_drafts,
                    &mut current_tokens,
                    computed_probs.as_deref_mut(),
                );
                for &orig_s in &active_lanes {
                    if emitted_drafts[orig_s].len() > emitted_pcum[orig_s].len() {
                        emitted_pcum[orig_s].push(p_cum[orig_s]);
                    }
                }
            }
        }
        drop(kv_cache);
        quantize_ragged_drafts_to_graph_bucket(&mut emitted_drafts, &mut emitted_pcum, allow_zero);

        for (s, st) in states.iter_mut().enumerate() {
            let state = st
                .as_any_mut()
                .downcast_mut::<Qwen4MtpState>()
                .expect("validated above");
            state.seq_len += steps_run[s];
            state.last_num_drafted = steps_run[s];
            state.last_drafts.clear();
            state.last_drafts.extend_from_slice(&computed_drafts[s]);
            if let Some(ref probs_log) = computed_probs {
                tracing::info!(
                    "MTP_DRAFT_RAGGED_BATCH seq={s} pos={} target={} computed={} kept={} \
                     drafts={:?} p={:.4?}",
                    positions[s],
                    target_drafts[s],
                    steps_run[s],
                    emitted_drafts[s].len(),
                    emitted_drafts[s],
                    probs_log[s]
                );
            }
        }
        Ok(Some(emitted_drafts))
    }
}

impl DraftProposer for Qwen4MtpHead {
    fn kind(&self) -> crate::speculative::DraftProposerKind {
        crate::speculative::DraftProposerKind::Qwen4NativeMtp
    }

    fn alloc_state(&self, gpu: &dyn GpuBackend) -> Result<Box<dyn ProposerState>> {
        Ok(Box::new(Qwen4MtpState {
            rotary_positions: crate::traits::RotaryPositions::identity(),
            layer_state: self.layer.alloc_state(gpu)?,
            block_table: Vec::new(),
            disk_block_ids: Vec::new(),
            disk_last_offloaded_per_layer: vec![u32::MAX],
            seq_len: 0,
            last_num_drafted: 0,
            last_drafts: Vec::new(),
            pending_replay: None,
            verify_frame: None,
            staged_frame: Mutex::new(Default::default()),
            pending_gumbel_spec: Mutex::new(None),
        }))
    }

    fn propose(
        &self,
        last_token: u32,
        target_hidden: DevicePtr,
        position: usize,
        num_drafts: usize,
        state: &mut dyn ProposerState,
        ctx: &ForwardContext,
        stream: u64,
        _draft_embed_target: Option<DevicePtr>,
        grammar_bitmask: Option<&[i32]>,
        target_hidden_stack: Option<DevicePtr>,
    ) -> Result<Vec<u32>> {
        let state = state
            .as_any_mut()
            .downcast_mut::<Qwen4MtpState>()
            .ok_or_else(|| anyhow::anyhow!("Invalid Qwen4 MTP proposer state"))?;

        // Feature 19: Out-of-pruned-range tokens fallback to base model
        if qwen4_mtp_gate::mtp_vocab_prune_selected()
            && last_token >= qwen4_mtp_gate::PRUNED_VOCAB_SIZE
        {
            state.last_num_drafted = 0;
            state.clear_gumbel_spec();
            spark_runtime::sampler::gumbel::clear_gumbel_draft_specs();
            return Ok(Vec::new());
        }

        let policy = qwen4_mtp_gate::current_policy();
        let num_drafts = policy.chain_len(num_drafts);
        // K7: rewrite the committed draft rows planned by `after_verify`
        // before this chain reads the head cache (drafts only; lossless).
        // In `verify_k2/k3/k4_step` partial/reject paths, `after_verify` runs
        // before `save_hidden_for_mtp`, so `staged_at_trim` is false and we
        // consume the staged frame token here before launching the chain.
        let mut replay_rows = if let Some(mut pending) = state.pending_replay.take() {
            if !pending.staged_at_trim {
                let staged = qwen4_mtp_rows::recondition_engaged()
                    && state.take_staged_frame(pending.owner, pending.staged_accepted)?;
                if staged {
                    if pending.computed == 0 {
                        let max_rows = match pending.row_policy {
                            RowPolicy::Recondition { max_rows } => max_rows,
                            _ => 0,
                        };
                        pending.plan.replay_rows = pending.accepted.min(max_rows);
                    } else {
                        let reconditioned = qwen4_mtp_rows::after_verify_plan(
                            pending.row_policy,
                            pending.computed,
                            pending.staged_accepted,
                            true,
                        );
                        let rewind = pending
                            .plan
                            .keep_rows
                            .saturating_sub(reconditioned.keep_rows);
                        state.seq_len = state.seq_len.saturating_sub(rewind);
                        pending.plan = reconditioned;
                    }
                }
                record_row_stats(pending.computed, pending.staged_accepted, pending.plan);
            }
            if pending.plan.replay_rows > 0 {
                self.collect_replay_rows(pending, target_hidden_stack, position, state, ctx)?
            } else {
                None
            }
        } else {
            None
        };
        state.staged_frame.lock().clear();
        validate_rotary_span(
            &state.rotary_positions,
            position,
            num_drafts,
            ctx.config.mrope_interleaved,
            self.mrope_kernel_available,
            ctx.config.yarn_factor,
        )?;
        // When `forward_rows` covers `m + 1` rows (`m` reconditioned committed
        // rows plus `draft[0]`), fuse them into a single batched MTP pass;
        // otherwise replay the `m` committed rows before `draft[0]`.
        let fuse_first_draft = replay_rows.as_ref().is_some_and(|(t, _, _)| {
            num_drafts > 0 && mtp_batch_rows_enabled() && self.forward_rows_admits(t.len() + 1)
        });
        let first_batch = if fuse_first_draft {
            let (mut tokens, mut hiddens, mut positions) = replay_rows.take().unwrap();
            tokens.push(last_token);
            hiddens.push(target_hidden);
            positions.push(position);
            Some((tokens, hiddens, positions))
        } else {
            if let Some((tokens, hiddens, positions)) = replay_rows.take() {
                let qsa_replay = self.mtp_qsa_active();
                for ((token, hidden), pos) in tokens.into_iter().zip(hiddens).zip(positions) {
                    self.forward_one(
                        token, hidden, pos, state, ctx, stream, None, false, qsa_replay, None,
                    )?;
                }
            }
            None
        };
        if grammar_bitmask.is_none()
            && num_drafts > 0
            && self.embed_from_argmax_k.0 != 0
            && mtp_chain_enabled()
        {
            let batch_refs = first_batch
                .as_ref()
                .map(|(t, h, p)| (t.as_slice(), h.as_slice(), p.as_slice()));
            let (drafts, computed) = self.propose_chain(
                last_token,
                target_hidden,
                position,
                num_drafts,
                policy,
                batch_refs,
                state,
                ctx,
                stream,
            )?;
            // The chain wrote `computed` MTP KV rows even when the confidence
            // gate returns fewer drafts; after_verify trims from this count.
            state.last_num_drafted = computed;
            state.clear_gumbel_spec();
            spark_runtime::sampler::gumbel::clear_gumbel_draft_specs();
            return Ok(drafts);
        }
        let mut drafts = Vec::with_capacity(num_drafts);
        let mut token = last_token;
        let mut hidden = target_hidden;
        let r_bytes = ctx.config.residual_width() * 2;
        let replay_prefix = first_batch.as_ref().map_or(0, |(t, _, _)| t.len() - 1);
        let first_seq_len = state.seq_len + replay_prefix;
        let qsa_mode = mtp_qsa_mode();
        let has_qsa = self.layer.has_qsa_side_cache();
        for step in 0..num_drafts {
            let (qsa_required, reuse_action) =
                draft_step_qsa_plan(qsa_mode, has_qsa, first_seq_len, step, num_drafts);
            let _reuse_guard = reuse_action.map(|action| match action {
                DraftStepReuseAction::SnapshotStep0 => {
                    crate::layers::qwen4_qsa::DraftSelectionReuseGuard::snapshot_step0()
                }
                DraftStepReuseAction::ReuseStep {
                    base_seq_len,
                    step_offset,
                    max_drafts,
                } => crate::layers::qwen4_qsa::DraftSelectionReuseGuard::reuse_step(
                    base_seq_len,
                    step_offset,
                    max_drafts,
                ),
            });
            if step == 0
                && let Some((ref b_tokens, ref b_hiddens, ref b_positions)) = first_batch
            {
                let replay_prefix = b_tokens.len() - 1;
                token = self.forward_rows(
                    b_tokens,
                    b_hiddens,
                    b_positions,
                    ctx.buffers.scratch().offset(CHAIN_META_OFFSET),
                    state,
                    ctx,
                    stream,
                    grammar_bitmask,
                    true,
                    None,
                )?;
                drafts.push(token);
                hidden = ctx.buffers.hidden_states().offset(replay_prefix * r_bytes);
                continue;
            }
            token = self.forward_one(
                token,
                hidden,
                position + step,
                state,
                ctx,
                stream,
                grammar_bitmask,
                true,
                qsa_required,
                None,
            )?;
            drafts.push(token);
            hidden = ctx.buffers.hidden_states();
        }
        state.last_num_drafted = drafts.len();
        state.last_drafts.clear();
        state.last_drafts.extend_from_slice(&drafts);
        state.clear_gumbel_spec();
        spark_runtime::sampler::gumbel::clear_gumbel_draft_specs();
        Ok(drafts)
    }

    fn propose_batch(
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
        if !(2..=64).contains(&b)
            || target_hiddens.len() != b
            || positions.len() != b
            || states.len() != b
            || target_hidden_stacks.len() != b
            || self.mtp_qsa_active()
            || !mtp_cross_seq_propose_enabled()
            || !mtp_chain_enabled()
            || self.embed_from_argmax_k.0 == 0
            || !self.propose_batch_rows_admits(b, ctx)
        {
            return Ok(None);
        }
        if qwen4_mtp_gate::mtp_vocab_prune_selected()
            && last_tokens
                .iter()
                .any(|&t| t >= qwen4_mtp_gate::PRUNED_VOCAB_SIZE)
        {
            return Ok(None);
        }
        let policy = qwen4_mtp_gate::current_policy();
        let num_drafts = policy.chain_len(num_drafts);
        if num_drafts == 0 {
            return Ok(None);
        }
        let mut gumbel_specs = Vec::with_capacity(b);
        for (s, st) in states.iter().enumerate() {
            let Some(state) = st.as_any().downcast_ref::<Qwen4MtpState>() else {
                return Ok(None);
            };
            gumbel_specs.push(state.resolve_gumbel_spec(positions[s]));
        }
        let any_gumbel = gumbel_specs.iter().any(Option::is_some);
        let all_gumbel = gumbel_specs.iter().all(Option::is_some);
        if any_gumbel && batched_ragged_mtp_enabled() && (self.argmax_prob_k.0 != 0 || all_gumbel) {
            return self.propose_batch_ragged_compacted(
                last_tokens,
                target_hiddens,
                positions,
                num_drafts,
                policy,
                &gumbel_specs,
                any_gumbel,
                states,
                ctx,
                stream,
            );
        }
        // Batched (b >= 2) cohort-uniform drafting never applies per-sequence
        // MTP_CONF early-stop syncs or prefix truncation so every sequence in
        // the cohort shares one verify width K and keeps `argmax_bf16_rows`
        // active, allowing C=1 (`propose_chain`) to retain `MTP_CONF=0.6` on
        // multi-concurrency servers without breaking C>=2 cohorts.
        let conf_gate = resolve_chain_confidence_gate_for_scope(
            ChainConfidenceScope::BatchedCohort,
            policy,
            policy.wants_probabilities_batch(),
            policy.early_threshold_batch(),
            self.argmax_prob_k.0 != 0,
            all_gumbel,
        );
        let want_prob = conf_gate.want_prob;
        debug_assert!(conf_gate.early.is_none() && !conf_gate.apply_truncate);
        let h = ctx.config.hidden_size;
        let r = ctx.config.residual_width();
        let hc = ctx.config.hc_count;
        let eps = ctx.config.rms_norm_eps as f32;
        let row_bytes = h * 2;
        let r_bytes = r * 2;
        let base_vocab = if qwen4_mtp_gate::mtp_vocab_prune_selected() {
            (qwen4_mtp_gate::PRUNED_VOCAB_SIZE as usize).min(ctx.config.vocab_size)
        } else if self.mtp_vocab_size == 0 {
            ctx.config.vocab_size
        } else {
            (self.mtp_vocab_size as usize).min(ctx.config.vocab_size)
        };
        let vocab = qwen4_mtp_gate::mtp_draft_vocab_override_for(any_gumbel)
            .map_or(base_vocab, |cap| base_vocab.min(cap));
        let (prob_offset, prob_span, token_offset) = if b <= 8 {
            (CHAIN_PROB_OFFSET, CHAIN_PROB_SPAN, CHAIN_TOKEN_OFFSET)
        } else {
            (CHAIN_ARGMAX_OFFSET + 2048, 2048, CHAIN_ARGMAX_OFFSET + 4096)
        };
        let argmax_limit = if want_prob { prob_offset } else { token_offset };
        if hc != 4
            || b * vocab * 2 > ctx.buffers.sizes().logits
            || CHAIN_ARGMAX_OFFSET + num_drafts * b * 4 > argmax_limit
            || (want_prob && num_drafts * b * 4 > prob_span)
            || token_offset + num_drafts * b * 4 > CHAIN_META_OFFSET
        {
            return Ok(None);
        }

        let mut all_identity = true;
        for (s, st) in states.iter().enumerate() {
            let Some(state) = st.as_any().downcast_ref::<Qwen4MtpState>() else {
                return Ok(None);
            };
            if state.pending_replay.is_some() {
                return Ok(None);
            }
            validate_rotary_span(
                &state.rotary_positions,
                positions[s],
                num_drafts,
                ctx.config.mrope_interleaved,
                self.mrope_kernel_available,
                ctx.config.yarn_factor,
            )?;
            if !state.rotary_positions.is_identity() {
                all_identity = false;
            }
        }

        let mut kv_cache = self.kv_cache.lock();
        let bs = kv_cache.block_size();
        let mut total_new_blocks = 0usize;
        let mut max_mb = 1usize;
        for st in states.iter() {
            let state = st
                .as_any()
                .downcast_ref::<Qwen4MtpState>()
                .expect("validated above");
            let last = state.seq_len + num_drafts - 1;
            let blocks_needed = last / bs + 1;
            total_new_blocks += blocks_needed.saturating_sub(state.block_table.len());
            max_mb = max_mb.max(state.block_table.len().max(blocks_needed));
        }
        if kv_cache.num_free_blocks() < total_new_blocks {
            return Ok(None);
        }
        let (pos_h_off, pos_w_off, last_tok_off, slot_off, seq_len_off, hidden_ptrs_off, bt_off) =
            if b <= 8 {
                (64, 128, 192, 256, 512, 576, 768)
            } else {
                (256, 512, 768, 1024, 1536, 2048, 2560)
            };
        let batch_graph_candidate = mtp_graph_enabled()
            && stream != 0
            && !ctx.graph_capture
            && !any_gumbel
            && all_identity
            && self.embed_from_argmax_k.0 != 0
            && self.pre_fc_dual_norm_k.0 != 0
            && h >= 1024
            && r >= 1024
            && kv_cache.num_layers() > 0;
        let effective_mb = if batch_graph_candidate {
            mtp_graph_bt_blocks_cap(max_mb, b, num_drafts, bt_off, ctx.buffers.sizes().scratch)
        } else {
            max_mb
        };
        let bt_row_bytes = effective_mb * 4;
        let stride = (bt_off + b * bt_row_bytes).next_multiple_of(256);
        if CHAIN_META_OFFSET + num_drafts * stride > ctx.buffers.sizes().scratch {
            return Ok(None);
        }
        for st in states.iter() {
            if let Some(state) = st.as_any().downcast_ref::<Qwen4MtpState>() {
                state.clear_gumbel_spec();
            }
        }
        spark_runtime::sampler::gumbel::clear_gumbel_draft_specs();
        for st in states.iter_mut() {
            let state = st
                .as_any_mut()
                .downcast_mut::<Qwen4MtpState>()
                .expect("validated above");
            let last = state.seq_len + num_drafts - 1;
            let blocks_needed = last / bs + 1;
            while state.block_table.len() < blocks_needed {
                state.block_table.push(kv_cache.alloc_block()?);
            }
        }
        let batch_gumbel =
            gumbel_draft::uniform_batch_gumbel_spec(&gumbel_specs).and_then(|shared_spec| {
                let layout = BatchGumbelMetaLayout::compute(num_drafts * stride, b, num_drafts)?;
                let meta_end = CHAIN_META_OFFSET
                    .checked_add(num_drafts * stride)?
                    .checked_add(layout.total_bytes)?;
                if meta_end > ctx.buffers.sizes().scratch {
                    return None;
                }
                let specs: Vec<spark_runtime::sampler::gumbel::GumbelDraftSpec> =
                    gumbel_specs.iter().flatten().copied().collect();
                Some((shared_spec, specs, layout))
            });
        let gumbel_meta_bytes = batch_gumbel
            .as_ref()
            .map_or(0, |(_, _, layout)| layout.total_bytes);

        let mut base_seq_lens = Vec::with_capacity(b);
        let packed_len = num_drafts * stride + gumbel_meta_bytes;
        let mut packed_guard = self.packed_meta_scratch.lock();
        let packed = prepare_packed_meta_scratch(&mut packed_guard, packed_len);
        for (row, st) in states.iter_mut().enumerate() {
            let state = st
                .as_any_mut()
                .downcast_mut::<Qwen4MtpState>()
                .expect("validated above");
            state.staged_frame.lock().clear();
            base_seq_lens.push(state.seq_len);
            packed[last_tok_off + row * 4..last_tok_off + row * 4 + 4]
                .copy_from_slice(&last_tokens[row].to_le_bytes());
            packed[hidden_ptrs_off + row * 8..hidden_ptrs_off + row * 8 + 8]
                .copy_from_slice(&target_hiddens[row].0.to_le_bytes());
            for (i, &block) in state.block_table.iter().enumerate() {
                let off = bt_off + row * bt_row_bytes + i * 4;
                packed[off..off + 4].copy_from_slice(&(block as i32).to_le_bytes());
            }
        }
        for step in 0..num_drafts {
            let s_off = step * stride;
            if step > 0 {
                packed.copy_within(bt_off..bt_off + b * bt_row_bytes, s_off + bt_off);
            }
            for (row, st) in states.iter().enumerate() {
                let state = st
                    .as_any()
                    .downcast_ref::<Qwen4MtpState>()
                    .expect("validated above");
                let [t, rh, rw] = state.rotary_positions.position(positions[row] + step)?;
                packed[s_off + row * 4..s_off + row * 4 + 4].copy_from_slice(&t.to_le_bytes());
                if !all_identity {
                    packed[s_off + pos_h_off + row * 4..s_off + pos_h_off + row * 4 + 4]
                        .copy_from_slice(&rh.to_le_bytes());
                    packed[s_off + pos_w_off + row * 4..s_off + pos_w_off + row * 4 + 4]
                        .copy_from_slice(&rw.to_le_bytes());
                }
                let r_seq = base_seq_lens[row] + step;
                let physical = state.block_table[r_seq / bs];
                let slot = physical as i64 * bs as i64 + (r_seq % bs) as i64;
                packed[s_off + slot_off + row * 8..s_off + slot_off + row * 8 + 8]
                    .copy_from_slice(&slot.to_le_bytes());
                packed[s_off + seq_len_off + row * 4..s_off + seq_len_off + row * 4 + 4]
                    .copy_from_slice(&((r_seq + 1) as i32).to_le_bytes());
            }
        }
        if let Some((_, batch_specs, layout)) = &batch_gumbel {
            for (row, spec) in batch_specs.iter().enumerate() {
                let seed_off = layout.seeds_off + row * 8;
                packed[seed_off..seed_off + 8].copy_from_slice(&spec.seed.to_le_bytes());
                for step in 0..num_drafts {
                    let pos_off = layout.step_pos_off(step, b) + row * 8;
                    let abs_pos = spec.step0_abs_pos + step as u64;
                    packed[pos_off..pos_off + 8].copy_from_slice(&abs_pos.to_le_bytes());
                }
            }
        }

        let scratch = ctx.buffers.scratch();
        let meta_base = scratch.offset(CHAIN_META_OFFSET);
        self.copy_h2d_group_fast(ctx.gpu, &[HostToDeviceCopy::new(packed, meta_base)], stream)?;
        drop(packed_guard);

        let argmax_base = scratch.offset(CHAIN_ARGMAX_OFFSET);
        let prob_base = scratch.offset(prob_offset);
        let embed = ctx.buffers.ssm_qkvz();
        let normed_embed = ctx.buffers.ssm_deinterleaved();
        let normed_hidden = ctx.buffers.residual();
        let embed_proj = ctx.buffers.qkv_output();
        let hidden = ctx.buffers.hidden_states();
        let logits = ctx.buffers.logits();

        let actual_computed = num_drafts;
        let ran_batch_graph = if batch_graph_candidate {
            let launch_spec = MtpBatchedChainLaunch {
                batch_size: b,
                step_count: num_drafts,
                want_prob,
                bt_blocks_cap: effective_mb,
                stride,
                last_tok_off,
                slot_off,
                seq_len_off,
                hidden_ptrs_off,
                bt_off,
                vocab,
                meta_base,
                argmax_base,
                prob_base,
                token_out_base: Some(scratch.offset(token_offset)),
                base_seq_lens: &base_seq_lens,
            };
            self.run_batched_mtp_chain_graphed(
                &launch_spec,
                None,
                states,
                &mut kv_cache,
                ctx,
                stream,
            )?
        } else {
            false
        };
        let tc_dense_scope = crate::layers::qwen4_hyper::batched_verify_tc_dense_active(b, false);
        let _tc_dense_guard =
            crate::layers::qwen4_hyper::BatchedVerifyTcDenseGuard::enter(tc_dense_scope);
        let coalesced_lm_head_eligible = should_use_mtp_lm_head_coalesced(
            crate::layers::qwen4_hyper::batched_verify_tc_component_active(
                crate::layers::qwen4_hyper::BatchedVerifyTcComponent::LmHead,
                b,
                false,
            ),
            b,
            h,
        );
        let coalesced_fc_eligible = should_use_mtp_fc_coalesced(
            tc_dense_scope,
            b,
            hc,
            h,
            ctx.buffers.sizes().qkv_output,
            self.residual_add_broadcast_hc_k.0 != 0,
        );

        for step in 0..if ran_batch_graph { 0 } else { num_drafts } {
            if step == 0 {
                if self.embed_dual_rms_norm_rows_packed_k.0 != 0 && h >= 1024 && r >= 1024 && b <= 8
                {
                    let tok_at = |idx: usize| -> u64 {
                        last_tokens
                            .get(idx)
                            .copied()
                            .filter(|_| idx < b)
                            .unwrap_or(0) as u64
                    };
                    let tok_pack01 = tok_at(0) | (tok_at(1) << 32);
                    let tok_pack23 = tok_at(2) | (tok_at(3) << 32);
                    let tok_pack45 = tok_at(4) | (tok_at(5) << 32);
                    let tok_pack67 = tok_at(6) | (tok_at(7) << 32);
                    spark_runtime::kernel_args::KernelLaunch::new(
                        ctx.gpu,
                        self.embed_dual_rms_norm_rows_packed_k,
                    )
                    .grid([b as u32, 2, 1])
                    .block([1024, 1, 1])
                    .arg_ptr(self.embed_tokens.weight)
                    .arg_u64(tok_pack01)
                    .arg_u64(tok_pack23)
                    .arg_u64(tok_pack45)
                    .arg_u64(tok_pack67)
                    .arg_u32(ctx.config.vocab_size as u32)
                    .arg_ptr(self.pre_fc_norm_embedding.weight)
                    .arg_ptr(normed_embed)
                    .arg_ptr(target_hiddens[0])
                    .arg_ptr(self.pre_fc_norm_hidden.weight)
                    .arg_ptr(normed_hidden)
                    .arg_u32(h as u32)
                    .arg_u32(r as u32)
                    .arg_f32(eps)
                    .arg_ptr(meta_base.offset(hidden_ptrs_off))
                    .launch(stream)?;
                } else {
                    if self.embed_from_argmax_k.0 != 0 {
                        spark_runtime::kernel_args::KernelLaunch::new(
                            ctx.gpu,
                            self.embed_from_argmax_k,
                        )
                        .grid([(h as u32).div_ceil(256), b as u32, 1])
                        .block([256, 1, 1])
                        .arg_ptr(meta_base.offset(last_tok_off))
                        .arg_ptr(self.embed_tokens.weight)
                        .arg_ptr(embed)
                        .arg_ptr(DevicePtr(0))
                        .arg_u32(h as u32)
                        .launch(stream)?;
                    } else {
                        for (row, &token) in last_tokens.iter().enumerate() {
                            let embed_src =
                                self.embed_tokens.weight.offset(token as usize * row_bytes);
                            ctx.gpu.copy_d2d_async(
                                embed_src,
                                embed.offset(row * row_bytes),
                                row_bytes,
                                stream,
                            )?;
                        }
                    }
                    if self.pre_fc_dual_norm_k.0 != 0 && h >= 1024 && r >= 1024 {
                        spark_runtime::kernel_args::KernelLaunch::new(
                            ctx.gpu,
                            self.pre_fc_dual_norm_k,
                        )
                        .grid([b as u32, 2, 1])
                        .block([1024, 1, 1])
                        .arg_ptr(embed)
                        .arg_ptr(self.pre_fc_norm_embedding.weight)
                        .arg_ptr(normed_embed)
                        .arg_ptr(target_hiddens[0])
                        .arg_ptr(self.pre_fc_norm_hidden.weight)
                        .arg_ptr(normed_hidden)
                        .arg_u32(h as u32)
                        .arg_u32(r as u32)
                        .arg_f32(eps)
                        .arg_ptr(meta_base.offset(hidden_ptrs_off))
                        .launch(stream)?;
                    } else {
                        ops::rms_norm(
                            ctx.gpu,
                            self.rms_norm_k,
                            embed,
                            &self.pre_fc_norm_embedding,
                            normed_embed,
                            b as u32,
                            h as u32,
                            eps,
                            stream,
                        )?;
                        let mut run_start = 0usize;
                        while run_start < b {
                            let mut run_len = 1usize;
                            while run_start + run_len < b
                                && target_hiddens[run_start + run_len]
                                    == target_hiddens[run_start].offset(run_len * r_bytes)
                            {
                                run_len += 1;
                            }
                            ops::rms_norm(
                                ctx.gpu,
                                self.rms_norm_k,
                                target_hiddens[run_start],
                                &self.pre_fc_norm_hidden,
                                normed_hidden.offset(run_start * r_bytes),
                                run_len as u32,
                                r as u32,
                                eps,
                                stream,
                            )?;
                            run_start += run_len;
                        }
                    }
                }
            } else {
                let prev_argmax = argmax_base.offset((step - 1) * b * 4);
                let token_out = scratch.offset(token_offset + (step - 1) * b * 4);
                spark_runtime::kernel_args::KernelLaunch::new(ctx.gpu, self.embed_from_argmax_k)
                    .grid([(h as u32).div_ceil(256), b as u32, 1])
                    .block([256, 1, 1])
                    .arg_ptr(prev_argmax)
                    .arg_ptr(self.embed_tokens.weight)
                    .arg_ptr(embed)
                    .arg_ptr(token_out)
                    .arg_u32(h as u32)
                    .launch(stream)?;
                if self.pre_fc_dual_norm_k.0 != 0 && h >= 1024 && r >= 1024 {
                    spark_runtime::kernel_args::KernelLaunch::new(ctx.gpu, self.pre_fc_dual_norm_k)
                        .grid([b as u32, 2, 1])
                        .block([1024, 1, 1])
                        .arg_ptr(embed)
                        .arg_ptr(self.pre_fc_norm_embedding.weight)
                        .arg_ptr(normed_embed)
                        .arg_ptr(hidden)
                        .arg_ptr(self.pre_fc_norm_hidden.weight)
                        .arg_ptr(normed_hidden)
                        .arg_u32(h as u32)
                        .arg_u32(r as u32)
                        .arg_f32(eps)
                        .arg_ptr(DevicePtr(0))
                        .launch(stream)?;
                } else {
                    ops::rms_norm(
                        ctx.gpu,
                        self.rms_norm_k,
                        embed,
                        &self.pre_fc_norm_embedding,
                        normed_embed,
                        b as u32,
                        h as u32,
                        eps,
                        stream,
                    )?;
                    ops::rms_norm(
                        ctx.gpu,
                        self.rms_norm_k,
                        hidden,
                        &self.pre_fc_norm_hidden,
                        normed_hidden,
                        b as u32,
                        r as u32,
                        eps,
                        stream,
                    )?;
                }
            }

            let ran_coalesced_fc = if coalesced_fc_eligible
                && crate::layers::qwen4_fast_proj::try_coalesced_nvfp4_w4a16_gemm(
                    ctx.gpu,
                    normed_embed,
                    &self.fc_embedding,
                    embed_proj,
                    b,
                    h,
                    h,
                    stream,
                )?
                && self.try_coalesced_mtp_fc_hidden(
                    ctx.gpu,
                    normed_hidden,
                    hidden,
                    b * hc,
                    h,
                    stream,
                )? {
                spark_runtime::kernel_args::KernelLaunch::new(
                    ctx.gpu,
                    self.residual_add_broadcast_hc_k,
                )
                .grid([(h as u32).div_ceil(256), b as u32, 1])
                .block([256, 1, 1])
                .arg_ptr(hidden)
                .arg_ptr(embed_proj)
                .arg_u32(h as u32)
                .arg_u32(hc as u32)
                .launch(stream)?;
                static FC_COALESCED_LOGGED: std::sync::Once = std::sync::Once::new();
                FC_COALESCED_LOGGED.call_once(|| {
                    tracing::info!(
                        rows = b,
                        hc_rows = b * hc,
                        hidden = h,
                        "QWEN4_MTP_FC_COALESCED_NVFP4_ENGAGED"
                    );
                });
                true
            } else {
                false
            };
            if !ran_coalesced_fc {
                if self.fc_dual_broadcast_rows_k.0 != 0 {
                    spark_runtime::kernel_args::KernelLaunch::new(
                        ctx.gpu,
                        self.fc_dual_broadcast_rows_k,
                    )
                    .grid([(h as u32).div_ceil(4), b as u32, 1])
                    .block([256, 1, 1])
                    .arg_ptr(normed_embed)
                    .arg_ptr(self.fc_embedding.weight)
                    .arg_ptr(self.fc_embedding.weight_scale)
                    .arg_f32(self.fc_embedding.weight_scale_2)
                    .arg_ptr(normed_hidden)
                    .arg_ptr(self.fc_hidden.weight)
                    .arg_ptr(self.fc_hidden.weight_scale)
                    .arg_f32(self.fc_hidden.weight_scale_2)
                    .arg_ptr(hidden)
                    .arg_u32(h as u32)
                    .arg_u32(h as u32)
                    .launch(stream)?;
                } else {
                    ops::w4a16_gemv_batch_logits_exact_with(
                        ctx.gpu,
                        self.w4a16_exact_fc_kernels,
                        normed_embed,
                        &self.fc_embedding,
                        embed_proj,
                        b as u32,
                        h as u32,
                        h as u32,
                        stream,
                        false,
                    )?;
                    ops::w4a16_gemv_batch_logits_exact_with(
                        ctx.gpu,
                        self.w4a16_exact_fc_kernels,
                        normed_hidden,
                        &self.fc_hidden,
                        hidden,
                        (b * hc) as u32,
                        h as u32,
                        h as u32,
                        stream,
                        false,
                    )?;
                    if self.residual_add_broadcast_hc_k.0 != 0 {
                        spark_runtime::kernel_args::KernelLaunch::new(
                            ctx.gpu,
                            self.residual_add_broadcast_hc_k,
                        )
                        .grid([(h as u32).div_ceil(256), b as u32, 1])
                        .block([256, 1, 1])
                        .arg_ptr(hidden)
                        .arg_ptr(embed_proj)
                        .arg_u32(h as u32)
                        .arg_u32(hc as u32)
                        .launch(stream)?;
                    } else {
                        for j in 0..b {
                            let embed_row = embed_proj.offset(j * row_bytes);
                            for branch in 0..hc {
                                let hidden_branch = hidden.offset((j * hc + branch) * row_bytes);
                                ops::residual_add(
                                    ctx.gpu,
                                    self.residual_add_k,
                                    hidden_branch,
                                    embed_row,
                                    h as u32,
                                    stream,
                                )?;
                            }
                        }
                    }
                }
            }

            let step_meta = meta_base.offset(step * stride);
            let (positions_h, positions_w) = if all_identity {
                (step_meta, step_meta)
            } else {
                (step_meta.offset(pos_h_off), step_meta.offset(pos_w_off))
            };
            let metadata = AttnMetadataDev {
                qwen4_qsa_required: self.mtp_qsa_active(),
                positions: step_meta,
                positions_h,
                positions_w,
                slot: step_meta.offset(slot_off),
                seq_len: step_meta.offset(seq_len_off),
                block_table: step_meta.offset(bt_off),
                max_blocks_per_seq: effective_mb as u32,
                num_seqs: b as u32,
            };
            let layer_ctx = ForwardContext {
                attn_metadata: Some(metadata),
                comm: None,
                graph_capture: false,
                ..*ctx
            };
            let step_seq_lens: Vec<usize> = base_seq_lens.iter().map(|&s| s + step).collect();
            {
                let mut typed_states: Vec<&mut Qwen4MtpState> = states
                    .iter_mut()
                    .map(|st| {
                        st.as_any_mut()
                            .downcast_mut::<Qwen4MtpState>()
                            .expect("validated above")
                    })
                    .collect();
                let mut block_tables: Vec<&mut Vec<u32>> = Vec::with_capacity(b);
                let mut disk_block_ids: Vec<&mut Vec<u32>> = Vec::with_capacity(b);
                let mut disk_last_offloaded: Vec<&mut Vec<u32>> = Vec::with_capacity(b);
                for st in typed_states.iter_mut() {
                    let Qwen4MtpState {
                        block_table,
                        disk_block_ids: d_ids,
                        disk_last_offloaded_per_layer: d_off,
                        ..
                    } = &mut **st;
                    block_tables.push(block_table);
                    disk_block_ids.push(d_ids);
                    disk_last_offloaded.push(d_off);
                }
                self.layer.decode_qwen4_cross_seq_mtp(
                    hidden,
                    ctx.buffers.residual(),
                    &step_seq_lens,
                    &mut block_tables,
                    &mut disk_block_ids,
                    &mut disk_last_offloaded,
                    &mut kv_cache,
                    &layer_ctx,
                    stream,
                )?;
            }

            let sample_hiddens = self.final_mixer.prepare_batched(
                hidden,
                ctx.buffers.residual(),
                b,
                ctx.buffers,
                ctx.gpu,
                eps,
                stream,
            )?;
            let ran_coalesced_lm = if coalesced_lm_head_eligible {
                let ran = crate::layers::qwen4_fast_proj::try_coalesced_nvfp4_w4a16_gemm(
                    ctx.gpu,
                    sample_hiddens,
                    &self.lm_head_nvfp4,
                    logits,
                    b,
                    vocab,
                    h,
                    stream,
                )?;
                if ran {
                    static COALESCED_LM_LOGGED: std::sync::Once = std::sync::Once::new();
                    COALESCED_LM_LOGGED.call_once(|| {
                        tracing::info!(
                            rows = b,
                            vocab,
                            hidden = h,
                            "QWEN4_MTP_LM_HEAD_COALESCED_NVFP4_ENGAGED"
                        );
                    });
                }
                ran
            } else {
                false
            };
            if !ran_coalesced_lm {
                if should_use_mtp_lm_head_pipe(
                    mtp_lm_head_tc_pipe_enabled(),
                    b,
                    h,
                    self.w4a16_gemm_pipe_k.0 != 0,
                    sample_hiddens,
                    self.lm_head_nvfp4.weight,
                    logits,
                ) {
                    ops::w4a16_gemm_pipe(
                        ctx.gpu,
                        self.w4a16_gemm_pipe_k,
                        sample_hiddens,
                        &self.lm_head_nvfp4,
                        logits,
                        b as u32,
                        vocab as u32,
                        h as u32,
                        stream,
                    )?;
                    static PIPE_LOGGED: std::sync::Once = std::sync::Once::new();
                    PIPE_LOGGED.call_once(|| {
                        tracing::info!(
                            rows = b,
                            vocab,
                            hidden = h,
                            "QWEN4_MTP_LM_HEAD_TC_PIPE_ENGAGED kernel=w4a16_gemm_pipe"
                        );
                    });
                } else {
                    ops::w4a16_gemv_batch_logits_exact_with(
                        ctx.gpu,
                        self.w4a16_exact_fc_kernels,
                        sample_hiddens,
                        &self.lm_head_nvfp4,
                        logits,
                        b as u32,
                        vocab as u32,
                        h as u32,
                        stream,
                        ops::w4a16_exact_rt2_for_rows(
                            ops::w4a16_gemv_rt2_enabled(),
                            ops::w4a16_exact_rt2_min_rows(),
                            b as u32,
                        ),
                    )?;
                }
            }
            let step_argmax = argmax_base.offset(step * b * 4);
            let step_prob = prob_base.offset(step * b * 4);
            if let Some((shared_spec, batch_specs, layout)) = &batch_gumbel {
                gumbel_draft::select_chain_gumbel_batch(
                    ctx.gpu,
                    self.mtp_gumbel_argmax_k,
                    logits,
                    DevicePtr::NULL,
                    meta_base.offset(layout.seeds_off),
                    meta_base.offset(layout.step_pos_off(step, b)),
                    step_argmax,
                    want_prob.then_some(step_prob),
                    b,
                    vocab,
                    *shared_spec,
                    batch_specs,
                    step as u64,
                    stream,
                )?;
            } else if any_gumbel {
                for s in 0..b {
                    if let Some(spec) = gumbel_specs[s] {
                        gumbel_draft::select_chain_gumbel_step(
                            ctx.gpu,
                            self.mtp_gumbel_argmax_k,
                            logits.offset(s * vocab * 2),
                            DevicePtr::NULL,
                            step_argmax.offset(s * 4),
                            want_prob.then(|| step_prob.offset(s * 4)),
                            vocab,
                            spec,
                            spec.step0_abs_pos + step as u64,
                            stream,
                        )?;
                    } else if want_prob {
                        spark_runtime::kernel_args::KernelLaunch::new(ctx.gpu, self.argmax_prob_k)
                            .grid([1, 1, 1])
                            .block([1024, 1, 1])
                            .arg_ptr(logits.offset(s * vocab * 2))
                            .arg_ptr(step_argmax.offset(s * 4))
                            .arg_ptr(step_prob.offset(s * 4))
                            .arg_u32(vocab as u32)
                            .launch(stream)?;
                    } else {
                        ops::argmax_bf16(
                            ctx.gpu,
                            self.argmax_k,
                            logits.offset(s * vocab * 2),
                            step_argmax.offset(s * 4),
                            vocab as u32,
                            stream,
                        )?;
                    }
                }
            } else if want_prob {
                self.launch_argmax_prob_batch(
                    ctx.gpu,
                    logits,
                    step_argmax,
                    step_prob,
                    b,
                    vocab,
                    stream,
                )?;
            } else if self.argmax_rows_k.0 != 0 {
                ops::argmax_bf16_rows(
                    ctx.gpu,
                    self.argmax_rows_k,
                    logits,
                    step_argmax,
                    vocab as u32,
                    vocab as u32,
                    b as u32,
                    stream,
                )?;
            } else {
                for s in 0..b {
                    ops::argmax_bf16(
                        ctx.gpu,
                        self.argmax_k,
                        logits.offset(s * vocab * 2),
                        step_argmax.offset(s * 4),
                        vocab as u32,
                        stream,
                    )?;
                }
            }
        }
        drop(_tc_dense_guard);
        drop(kv_cache);

        let total_items = actual_computed * b;
        let mut bytes = vec![0u8; total_items * 4];
        let mut pbytes = vec![0u8; if want_prob { total_items * 4 } else { 0 }];
        if want_prob {
            ctx.gpu.copy_d2h_pair_on_stream(
                argmax_base,
                &mut bytes,
                prob_base,
                &mut pbytes,
                stream,
            )?;
        } else {
            ctx.gpu
                .copy_d2h_on_stream(argmax_base, &mut bytes, stream)?;
        }
        let all_tokens: Vec<u32> = bytes
            .chunks_exact(4)
            .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();
        let all_probs: Vec<f32> = if want_prob {
            pbytes
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect()
        } else {
            Vec::new()
        };
        let mut out = Vec::with_capacity(b);
        for (s, st) in states.iter_mut().enumerate() {
            let state = st
                .as_any_mut()
                .downcast_mut::<Qwen4MtpState>()
                .expect("validated above");
            state.seq_len += actual_computed;
            state.last_num_drafted = actual_computed;
            let drafts: Vec<u32> = (0..actual_computed)
                .map(|step| all_tokens[step * b + s])
                .collect();
            state.last_drafts.clear();
            state.last_drafts.extend_from_slice(&drafts);
            if want_prob && policy.log {
                let probs: Vec<f32> = (0..actual_computed)
                    .map(|step| all_probs[step * b + s])
                    .collect();
                tracing::info!(
                    "MTP_DRAFT_CONF_BATCH seq={s} pos={} n={num_drafts} computed={actual_computed} \
                     drafts={drafts:?} p={probs:.4?} keep={}",
                    positions[s],
                    drafts.len()
                );
            }
            out.push(drafts);
        }
        Ok(Some(out))
    }

    fn after_verify(
        &self,
        num_accepted: usize,
        state: &mut dyn ProposerState,
        _stream: u64,
    ) -> Result<()> {
        let state = state
            .as_any_mut()
            .downcast_mut::<Qwen4MtpState>()
            .ok_or_else(|| anyhow::anyhow!("Invalid Qwen4 MTP proposer state"))?;
        let row_policy = rows_policy_logged();
        if row_policy == RowPolicy::Legacy {
            let trim = state.last_num_drafted.saturating_sub(num_accepted);
            state.seq_len = state.seq_len.saturating_sub(trim);
            state.last_num_drafted = 0;
            state.pending_replay = None;
            return Ok(());
        }
        // K7 (ATLAS_QWEN4_MTP_KEEP_LAST_ROW / _RECONDITION): keep the row of
        // the last committed token and optionally rewrite committed draft rows
        // from the verify's target hidden rows (see qwen4_mtp_rows).
        let computed = state.last_num_drafted;
        let recondition_active = qwen4_mtp_rows::recondition_engaged()
            && matches!(row_policy, RowPolicy::Recondition { .. });
        let staged =
            recondition_active && state.take_staged_frame(state.verify_frame, num_accepted)?;
        if computed == 0 && recondition_active && num_accepted > 0 && !state.last_drafts.is_empty()
        {
            let max_rows = match row_policy {
                RowPolicy::Recondition { max_rows } => max_rows,
                _ => 0,
            };
            let accepted = num_accepted.min(state.last_drafts.len());
            let replay = accepted.min(max_rows);
            state.pending_replay = (replay > 0).then_some(PendingReplay {
                owner: state.verify_frame,
                row_policy,
                computed: 0,
                staged_accepted: num_accepted,
                plan: AfterVerifyPlan {
                    keep_rows: 1,
                    replay_rows: if staged { replay } else { 0 },
                },
                accepted,
                staged_at_trim: staged,
            });
            return Ok(());
        }
        let plan = qwen4_mtp_rows::after_verify_plan(row_policy, computed, num_accepted, staged);
        state.seq_len = state.seq_len.saturating_sub(computed) + plan.keep_rows;
        let defer_staged_check = recondition_active && !staged && computed > 0 && num_accepted > 0;
        state.pending_replay = if staged {
            (plan.replay_rows > 0).then_some(PendingReplay {
                owner: state.verify_frame,
                row_policy,
                computed,
                staged_accepted: num_accepted,
                plan,
                accepted: num_accepted.min(computed),
                staged_at_trim: true,
            })
        } else if defer_staged_check {
            Some(PendingReplay {
                owner: state.verify_frame,
                row_policy,
                computed,
                staged_accepted: num_accepted,
                plan,
                accepted: num_accepted.min(computed),
                staged_at_trim: false,
            })
        } else {
            None
        };
        state.last_num_drafted = 0;
        if !defer_staged_check {
            record_row_stats(computed, num_accepted, plan);
        }
        Ok(())
    }

    fn prefill_last_k(
        &self,
        tokens: &[u32],
        target_hiddens: DevicePtr,
        base_position: usize,
        state: &mut dyn ProposerState,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let state = state
            .as_any_mut()
            .downcast_mut::<Qwen4MtpState>()
            .ok_or_else(|| anyhow::anyhow!("Invalid Qwen4 MTP proposer state"))?;
        if tokens.is_empty() {
            return Ok(());
        }
        ensure!(
            state.seq_len == 0,
            "Qwen4 MTP prompt replay requires an empty proposer cache"
        );
        let predict_start = prompt_tail_start(base_position, tokens.len())?;
        validate_rotary_span(
            &state.rotary_positions,
            predict_start,
            tokens.len(),
            ctx.config.mrope_interleaved,
            self.mrope_kernel_available,
            ctx.config.yarn_factor,
        )?;
        let stride = ctx.config.residual_width() * 2;
        let pairs: Vec<(usize, u32)> = crate::speculative::mtp_replay_pairs(
            tokens,
            crate::speculative::mtp_replay_next_token(),
        );
        let batch_enabled = mtp_batch_rows_enabled();
        let meta_base = ctx.buffers.scratch().offset(CHAIN_META_OFFSET);
        let mut cursor = 0usize;
        while cursor < pairs.len() {
            let rem = pairs.len() - cursor;
            // Avoid leaving a 1-row tail when `rem > 8 && rem % 8 == 1`.
            let chunk = if !batch_enabled || rem < 2 {
                1
            } else if rem > 8 && rem % 8 == 1 {
                5
            } else {
                rem.min(8)
            };
            if chunk >= 2 && self.forward_rows_admits(chunk) {
                let slice = &pairs[cursor..cursor + chunk];
                let b_tokens: Vec<u32> = slice.iter().map(|&(_, t)| t).collect();
                let b_hiddens: Vec<DevicePtr> = slice
                    .iter()
                    .map(|&(idx, _)| target_hiddens.offset(idx * stride))
                    .collect();
                let b_positions: Vec<usize> =
                    slice.iter().map(|&(idx, _)| predict_start + idx).collect();
                let _ = self.forward_rows(
                    &b_tokens,
                    &b_hiddens,
                    &b_positions,
                    meta_base,
                    state,
                    ctx,
                    stream,
                    None,
                    false,
                    None,
                )?;
                cursor += chunk;
            } else {
                let (index, token) = pairs[cursor];
                let _ = self.forward_one(
                    token,
                    target_hiddens.offset(index * stride),
                    predict_start + index,
                    state,
                    ctx,
                    stream,
                    None,
                    false,
                    self.mtp_qsa_active(),
                    None,
                )?;
                cursor += 1;
            }
        }
        state.last_num_drafted = 0;
        Ok(())
    }

    fn reserve_prompt_replay_blocks(
        &self,
        rows: usize,
        state: &mut dyn ProposerState,
    ) -> Result<usize> {
        let state = state
            .as_any_mut()
            .downcast_mut::<Qwen4MtpState>()
            .ok_or_else(|| anyhow::anyhow!("Invalid Qwen4 MTP proposer state"))?;
        ensure!(
            state.seq_len == 0,
            "Qwen4 MTP prompt replay requires an empty proposer cache"
        );
        // The allocations `step_metadata` makes for replay steps seq_len = 0..rows,
        // in the same order (each `alloc_block` pops the free list).
        let mut kv_cache = self.kv_cache.lock();
        let need = crate::model::host_bucket::mtp_replay_blocks(rows, kv_cache.block_size());
        let before = state.block_table.len();
        while state.block_table.len() < need {
            state.block_table.push(kv_cache.alloc_block()?);
        }
        Ok(state.block_table.len() - before)
    }

    fn free_state(&self, state: &mut dyn ProposerState) -> Result<()> {
        let state = state
            .as_any_mut()
            .downcast_mut::<Qwen4MtpState>()
            .ok_or_else(|| anyhow::anyhow!("Invalid Qwen4 MTP proposer state"))?;
        if !state.block_table.is_empty() {
            self.kv_cache.lock().free_blocks(&state.block_table);
            state.block_table.clear();
        }
        state.seq_len = 0;
        Ok(())
    }

    fn seqstate_export(
        &self,
        state: &dyn ProposerState,
    ) -> Result<Option<crate::model::seqstate::ProposerSeqState>> {
        let state = state
            .as_any()
            .downcast_ref::<Qwen4MtpState>()
            .ok_or_else(|| anyhow::anyhow!("Invalid Qwen4 MTP proposer state"))?;
        anyhow::ensure!(
            !state.has_unconsumed_verify_frame() && state.verify_frame.is_none(),
            "seqstate: native MTP has an unconsumed verify frame (not quiescent)"
        );
        anyhow::ensure!(
            state.rotary_positions.is_identity() && state.disk_block_ids.is_empty(),
            "seqstate: native MTP state with image rotary or disk blocks is not exportable"
        );
        let (seq_len, last_num_drafted, last_drafts) = state.snapshot_dedup_draft();
        let regions = crate::model::seqstate::engine::kv_block_regions(
            &self.kv_cache.lock(),
            &state.block_table,
            crate::model::seqstate::SectionKind::MtpKvK,
            crate::model::seqstate::SectionKind::MtpKvV,
        )?;
        Ok(Some(crate::model::seqstate::ProposerSeqState {
            meta: crate::model::seqstate::ProposerSeqMeta {
                seq_len,
                num_blocks: state.block_table.len(),
                last_num_drafted,
                last_drafts,
            },
            regions,
        }))
    }

    fn seqstate_import(
        &self,
        state: &mut dyn ProposerState,
        meta: &crate::model::seqstate::ProposerSeqMeta,
    ) -> Result<Option<Vec<crate::model::seqstate::StateRegion>>> {
        let state = state
            .as_any_mut()
            .downcast_mut::<Qwen4MtpState>()
            .ok_or_else(|| anyhow::anyhow!("Invalid Qwen4 MTP proposer state"))?;
        anyhow::ensure!(
            state.block_table.is_empty()
                && state.seq_len == 0
                && !state.has_unconsumed_verify_frame(),
            "seqstate: native MTP import needs a fresh proposer state"
        );
        let block_size = self.kv_cache.lock().block_size();
        anyhow::ensure!(
            meta.num_blocks >= meta.seq_len.div_ceil(block_size),
            "seqstate: {} MTP blocks cannot hold {} positions",
            meta.num_blocks,
            meta.seq_len
        );
        let mut kv = self.kv_cache.lock();
        let mut table = Vec::with_capacity(meta.num_blocks);
        for _ in 0..meta.num_blocks {
            match kv.alloc_block() {
                Ok(b) => table.push(b),
                Err(e) => {
                    kv.free_blocks(&table);
                    return Err(e.context("seqstate: native MTP block allocation"));
                }
            }
        }
        let regions = crate::model::seqstate::engine::kv_block_regions(
            &kv,
            &table,
            crate::model::seqstate::SectionKind::MtpKvK,
            crate::model::seqstate::SectionKind::MtpKvV,
        );
        drop(kv);
        let regions = match regions {
            Ok(r) => r,
            Err(e) => {
                self.kv_cache.lock().free_blocks(&table);
                return Err(e);
            }
        };
        state.block_table = table;
        state.rotary_positions = crate::traits::RotaryPositions::identity();
        state.apply_dedup_draft(meta.seq_len, meta.last_num_drafted, &meta.last_drafts);
        Ok(Some(regions))
    }

    fn release_rejection_resources(&self, gpu: &dyn GpuBackend) {
        self.graph_cache.lock().clear(gpu);
        if let Some(mut ring) = self.pinned_h2d_ring.lock().take() {
            ring.destroy(gpu);
        }
    }
}

#[cfg(test)]
mod draft_confidence_tests {
    use super::*;
    use crate::traits::{ImageSpan, RotaryPositions};

    #[test]
    fn stops_before_first_low_confidence_draft() {
        assert_eq!(confident_prefix(&[0.9, 0.8, 0.5, 0.95], 0.6), 2);
        assert_eq!(confident_prefix(&[0.9, 0.8, 0.7, 0.65], 0.6), 4);
    }

    #[test]
    fn always_keeps_one_draft() {
        assert_eq!(confident_prefix(&[0.1, 0.9], 0.6), 1);
        assert_eq!(confident_prefix(&[0.59], 0.6), 1);
    }

    #[test]
    fn threshold_is_inclusive() {
        assert_eq!(confident_prefix(&[0.6, 0.6], 0.6), 2);
    }

    #[test]
    fn cohort_uniform_batch_drafting_never_truncates_to_zero_under_gumbel() {
        // Batched (C >= 2) proposals always keep at least 1 draft even on hopeless
        // step-0 probabilities (< 0.12) and keep all drafts when base threshold is 0.
        assert_eq!(confident_prefix_batch(&[0.01, 0.9, 0.9, 0.9], 0.0), 4);
        assert_eq!(confident_prefix_batch(&[0.01, 0.9, 0.9, 0.9], 0.6), 1);
        assert_eq!(confident_prefix_batch(&[0.05], 0.85), 1);

        let conf_off = GatePolicy {
            threshold: None,
            max_drafts: None,
            early_stop: false,
            log: false,
        };
        assert!(!conf_off.wants_probabilities_batch());
        assert_eq!(conf_off.effective_base_threshold_batch(), None);
        assert_eq!(conf_off.early_threshold_batch(), None);
    }

    fn image_positions() -> RotaryPositions {
        RotaryPositions::from_image_spans(
            13,
            64,
            &[
                ImageSpan {
                    start: 1,
                    height: 2,
                    width: 2,
                },
                ImageSpan {
                    start: 6,
                    height: 2,
                    width: 3,
                },
            ],
        )
        .unwrap()
    }

    #[test]
    fn image_rotary_admission_is_fail_closed_before_metadata() {
        let positions = image_positions();
        assert_eq!(
            rotary_axes(&positions, 2, true, true, 0.0).unwrap(),
            [1, 1, 2]
        );
        assert!(rotary_axes(&positions, 2, false, true, 0.0).is_err());
        assert!(rotary_axes(&positions, 2, true, false, 0.0).is_err());
        assert!(rotary_axes(&positions, 2, true, true, 4.0).is_err());
        assert!(rotary_axes(&positions, 64, true, true, 0.0).is_err());
        assert!(validate_rotary_span(&positions, 62, 2, true, true, 0.0).is_ok());
        assert!(validate_rotary_span(&positions, 63, 2, true, true, 0.0).is_err());
        assert!(
            validate_rotary_span(&positions, 0, MAX_ROTARY_METADATA_ROWS + 1, true, true, 0.0)
                .is_err()
        );
        assert!(validate_rotary_span(&positions, usize::MAX, 2, true, true, 0.0).is_err());
    }

    #[test]
    fn metadata_preserves_physical_cache_fields_and_maps_all_rotary_axes() {
        let positions = image_positions();
        let mut packed = vec![0xa5; 32];
        write_rotary_metadata(&positions, 2, &mut packed).unwrap();
        assert_eq!(&packed[0..4], &1u32.to_le_bytes());
        assert_eq!(
            &packed[POSITION_H_OFFSET..POSITION_H_OFFSET + 4],
            &1u32.to_le_bytes()
        );
        assert_eq!(
            &packed[POSITION_W_OFFSET..POSITION_W_OFFSET + 4],
            &2u32.to_le_bytes()
        );
        assert!(
            packed[4..POSITION_H_OFFSET]
                .iter()
                .all(|byte| *byte == 0xa5)
        );

        let mut identity = vec![0xa5; 32];
        write_rotary_metadata(&RotaryPositions::identity(), 9, &mut identity).unwrap();
        assert_eq!(&identity[0..4], &9u32.to_le_bytes());
        assert!(
            identity[POSITION_H_OFFSET..]
                .iter()
                .all(|byte| *byte == 0xa5)
        );
        assert!(write_rotary_metadata(&positions, 2, &mut [0u8; 27]).is_err());
    }

    #[test]
    fn image_metadata_uses_distinct_axes_but_text_keeps_legacy_aliases() {
        let base = DevicePtr(4096);
        let image = image_positions();
        assert_eq!(
            rotary_metadata_ptrs(base, &image),
            (
                base.offset(POSITION_H_OFFSET),
                base.offset(POSITION_W_OFFSET)
            )
        );
        assert_eq!(
            rotary_metadata_ptrs(base, &RotaryPositions::identity()),
            (base, base)
        );
    }

    #[test]
    fn prompt_tail_maps_predict_into_positions_without_wrapping() {
        let positions = image_positions();
        let start = prompt_tail_start(12, 4).unwrap();
        assert_eq!(start, 10);
        assert_eq!(
            positions.range(start, 4).unwrap(),
            [[4, 5, 5], [4, 5, 6], [7; 3], [8; 3]]
        );
        assert!(prompt_tail_start(2, 4).is_err());
        assert!(prompt_tail_start(usize::MAX, 1).is_err());
    }

    #[test]
    fn propose_batch_metadata_and_scratch_spans_are_disjoint_up_to_c64() {
        for b in 2..=64usize {
            let (
                pos_h_off,
                pos_w_off,
                last_tok_off,
                slot_off,
                seq_len_off,
                hidden_ptrs_off,
                bt_off,
            ) = if b <= 8 {
                (64, 128, 192, 256, 512, 576, 768)
            } else {
                (256, 512, 768, 1024, 1536, 2048, 2560)
            };
            assert_eq!(slot_off % 8, 0, "b={b}");
            assert_eq!(hidden_ptrs_off % 8, 0, "b={b}");
            let bt_row_bytes = 64 * 4;
            let stride = (bt_off + b * bt_row_bytes).next_multiple_of(256);
            let spans = [
                (0usize, b * 4),
                (pos_h_off, pos_h_off + b * 4),
                (pos_w_off, pos_w_off + b * 4),
                (last_tok_off, last_tok_off + b * 4),
                (slot_off, slot_off + b * 8),
                (seq_len_off, seq_len_off + b * 4),
                (hidden_ptrs_off, hidden_ptrs_off + b * 8),
                (bt_off, bt_off + b * bt_row_bytes),
            ];
            for window in spans.windows(2) {
                assert!(window[0].1 <= window[1].0, "b={b}: {window:?}");
            }
            assert!(spans.last().unwrap().1 <= stride, "b={b}");

            let (prob_offset, prob_span, token_offset) = if b <= 8 {
                (CHAIN_PROB_OFFSET, CHAIN_PROB_SPAN, CHAIN_TOKEN_OFFSET)
            } else {
                (CHAIN_ARGMAX_OFFSET + 2048, 2048, CHAIN_ARGMAX_OFFSET + 4096)
            };
            for num_drafts in 1..=7usize {
                let bytes = num_drafts * b * 4;
                assert!(CHAIN_ARGMAX_OFFSET + bytes <= prob_offset, "b={b}");
                assert!(bytes <= prob_span, "b={b}");
                assert!(prob_offset + bytes <= token_offset, "b={b}");
                assert!(token_offset + bytes <= CHAIN_META_OFFSET, "b={b}");
            }
        }
    }

    #[test]
    fn mtp_qsa_mode_parsing_and_fail_closed_invalid_values() {
        assert_eq!(
            parse_mtp_qsa_mode(None, None).unwrap(),
            MtpQsaMode::Disabled
        );
        assert_eq!(
            parse_mtp_qsa_mode(Some("0"), None).unwrap(),
            MtpQsaMode::Disabled
        );
        assert_eq!(
            parse_mtp_qsa_mode(Some("0"), Some("1")).unwrap(),
            MtpQsaMode::Disabled
        );
        assert_eq!(
            parse_mtp_qsa_mode(Some("0"), Some("bogus")).unwrap(),
            MtpQsaMode::Disabled
        );
        assert_eq!(
            parse_mtp_qsa_mode(Some("1"), None).unwrap(),
            MtpQsaMode::DraftSelectionReuse
        );
        assert_eq!(
            parse_mtp_qsa_mode(Some(" 1 "), Some(" 1 ")).unwrap(),
            MtpQsaMode::DraftSelectionReuse
        );
        assert_eq!(
            parse_mtp_qsa_mode(Some("1"), Some("0")).unwrap(),
            MtpQsaMode::PerStep
        );
        for bad in ["", "2", "true", "yes", "on", "-1", "reuse"] {
            assert!(
                parse_mtp_qsa_mode(Some(bad), None).is_err(),
                "expected invalid ATLAS_QWEN4_MTP_QSA={bad:?} to error"
            );
            assert_eq!(
                resolve_mtp_qsa_mode(Some(bad), Some("1")),
                MtpQsaMode::Disabled,
                "invalid ATLAS_QWEN4_MTP_QSA={bad:?} must fail closed"
            );
            assert!(
                parse_mtp_qsa_mode(Some("1"), Some(bad)).is_err(),
                "expected invalid ATLAS_QWEN4_MTP_DRAFT_SELECTION_REUSE={bad:?} to error"
            );
            assert_eq!(
                resolve_mtp_qsa_mode(Some("1"), Some(bad)),
                MtpQsaMode::Disabled,
                "invalid ATLAS_QWEN4_MTP_DRAFT_SELECTION_REUSE={bad:?} must fail closed"
            );
        }
    }

    #[test]
    fn mtp_qsa_reused_index_width_bounds() {
        let base = crate::layers::qwen4_qsa::QSA_OUTPUT_WIDTH;
        assert_eq!(base, 2_051);
        for step in 0..=8 {
            let width = checked_mtp_reused_index_width(step, 8).unwrap();
            assert_eq!(width, base + step);
            assert!(width <= 2_059);
        }
        assert_eq!(checked_mtp_reused_index_width(8, 8).unwrap(), 2_059);
        assert_eq!(checked_mtp_reused_index_width(16, 16).unwrap(), 2_067);
        assert!(checked_mtp_reused_index_width(9, 8).is_err());
        assert!(checked_mtp_reused_index_width(1, 17).is_err());
        assert!(checked_mtp_reused_index_width(usize::MAX, 8).is_err());
    }

    #[test]
    fn mtp_draft_step_qsa_plan_routes_snapshot_reuse_and_dense_prefix() {
        // Disabled or no QSA side cache -> always off.
        assert_eq!(
            draft_step_qsa_plan(MtpQsaMode::Disabled, true, 4096, 0, 4),
            (false, None)
        );
        assert_eq!(
            draft_step_qsa_plan(MtpQsaMode::DraftSelectionReuse, false, 4096, 0, 4),
            (false, None)
        );
        // PerStep -> always QSA required without reuse guard.
        assert_eq!(
            draft_step_qsa_plan(MtpQsaMode::PerStep, true, 4096, 2, 4),
            (true, None)
        );
        // DraftSelectionReuse at long context (>= 2048): step 0 snapshots, steps 1..K-1 reuse.
        assert_eq!(
            draft_step_qsa_plan(MtpQsaMode::DraftSelectionReuse, true, 4096, 0, 8),
            (true, Some(DraftStepReuseAction::SnapshotStep0))
        );
        for step in 1..8 {
            assert_eq!(
                draft_step_qsa_plan(MtpQsaMode::DraftSelectionReuse, true, 4096, step, 8),
                (
                    true,
                    Some(DraftStepReuseAction::ReuseStep {
                        base_seq_len: 4096,
                        step_offset: step,
                        max_drafts: 7,
                    })
                )
            );
        }
        // Dense prefix (< 2048): step 0 updates side cache, speculative drafts skip side cache.
        assert_eq!(
            draft_step_qsa_plan(MtpQsaMode::DraftSelectionReuse, true, 512, 0, 4),
            (true, None)
        );
        assert_eq!(
            draft_step_qsa_plan(MtpQsaMode::DraftSelectionReuse, true, 512, 1, 4),
            (false, None)
        );
        // 2048-straddling boundary chain falls back to PerStep cleanly.
        assert_eq!(
            draft_step_qsa_plan(MtpQsaMode::DraftSelectionReuse, true, 2046, 2, 4),
            (true, None)
        );
    }

    #[test]
    fn chain_confidence_gate_preserves_greedy_and_enables_gumbel_running_product() {
        let active_policy = GatePolicy {
            threshold: Some(0.85),
            max_drafts: None,
            early_stop: true,
            log: false,
        };
        // Greedy (`gumbel_active = false`) keeps early-stop syncs and truncation intact.
        assert_eq!(
            resolve_chain_confidence_gate(active_policy, true, Some(0.85), true, false),
            ChainConfidenceGate {
                want_prob: true,
                early: Some(0.85),
                apply_truncate: true,
            }
        );
        // Greedy without argmax_prob kernel disables prob/early/truncate cleanly.
        assert_eq!(
            resolve_chain_confidence_gate(active_policy, true, Some(0.85), false, false),
            ChainConfidenceGate {
                want_prob: false,
                early: None,
                apply_truncate: false,
            }
        );
        // Non-greedy Gumbel-CRN (`gumbel_active = true`) enables running-product
        // MTP_CONF_CUM with 0.0 per-step floor so single T>0 tokens survive greedy
        // per-step floors while low-cumulative chains stop cleanly.
        assert_eq!(
            resolve_chain_confidence_gate_with_cum(
                ChainConfidenceScope::SingleSequence,
                active_policy,
                true,
                Some(0.85),
                true,
                true,
                Some(qwen4_mtp_gate::DEFAULT_GUMBEL_CUM_THRESHOLD),
                false,
            ),
            ChainConfidenceGate {
                want_prob: true,
                early: Some(0.0),
                apply_truncate: true,
            }
        );
        // Explicit `ATLAS_QWEN4_MTP_CONF_CUM=0` (`gumbel_cum = None`) disables gating.
        assert_eq!(
            resolve_chain_confidence_gate_with_cum(
                ChainConfidenceScope::SingleSequence,
                active_policy,
                true,
                Some(0.85),
                true,
                true,
                None,
                false,
            ),
            ChainConfidenceGate {
                want_prob: false,
                early: None,
                apply_truncate: false,
            }
        );
        // Diagnostic logging with `gumbel_cum = None` still collects probabilities without truncating.
        let logged_policy = GatePolicy {
            log: true,
            ..active_policy
        };
        assert_eq!(
            resolve_chain_confidence_gate_with_cum(
                ChainConfidenceScope::SingleSequence,
                logged_policy,
                true,
                Some(0.85),
                true,
                true,
                None,
                false,
            ),
            ChainConfidenceGate {
                want_prob: true,
                early: None,
                apply_truncate: false,
            }
        );
        // BatchedCohort (`b >= 2`) never enables early-stop or truncation even under Gumbel.
        assert_eq!(
            resolve_chain_confidence_gate_with_cum(
                ChainConfidenceScope::BatchedCohort,
                active_policy,
                true,
                Some(0.85),
                true,
                true,
                Some(qwen4_mtp_gate::DEFAULT_GUMBEL_CUM_THRESHOLD),
                false,
            ),
            ChainConfidenceGate {
                want_prob: false,
                early: None,
                apply_truncate: false,
            }
        );
    }

    #[test]
    fn mtp_pinned_h2d_ring_aligns_wraps_and_copies() {
        let gpu = spark_runtime::gpu::mock::MockGpuBackend::new();
        let mut ring = MtpPinnedH2dRing::new(&gpu, 64).expect("mock pinned ring");
        let dst_a = gpu.alloc(20).unwrap();
        let dst_b = gpu.alloc(36).unwrap();
        let payload_a = [0x5Au8; 20];
        let payload_b = [0xA5u8; 36];

        assert!(
            ring.copy_h2d_group(&gpu, &[HostToDeviceCopy::new(&payload_a, dst_a)], 0)
                .unwrap()
        );
        assert_eq!(gpu.read_alloc(dst_a).unwrap(), payload_a);

        // Next 36B copy aligns to 24 (24 + 40 = 64 <= 64).
        assert!(
            ring.copy_h2d_group(&gpu, &[HostToDeviceCopy::new(&payload_b, dst_b)], 0)
                .unwrap()
        );
        assert_eq!(gpu.read_alloc(dst_b).unwrap(), payload_b);

        // Next 20B copy wraps cleanly back to offset 0.
        assert!(
            ring.copy_h2d_group(&gpu, &[HostToDeviceCopy::new(&payload_a, dst_a)], 0)
                .unwrap()
        );
        // Oversized payload (> 64 B) returns Ok(false) to trigger fallback.
        let oversized = [0x11u8; 72];
        assert!(
            !ring
                .copy_h2d_group(&gpu, &[HostToDeviceCopy::new(&oversized, dst_a)], 0)
                .unwrap()
        );
        ring.destroy(&gpu);
        assert!(
            !ring
                .copy_h2d_group(&gpu, &[HostToDeviceCopy::new(&payload_a, dst_a)], 0)
                .unwrap()
        );
    }

    #[test]
    fn mtp_pinned_h2d_ring_fences_wrap_and_destroys_events() {
        use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};

        struct EventFenceGpu {
            inner: spark_runtime::gpu::mock::MockGpuBackend,
            next_event: AtomicU64,
            ready: AtomicBool,
            destroyed: AtomicUsize,
        }

        impl GpuBackend for EventFenceGpu {
            fn alloc(&self, bytes: usize) -> Result<DevicePtr> {
                self.inner.alloc(bytes)
            }
            fn alloc_managed(&self, bytes: usize) -> Result<DevicePtr> {
                self.inner.alloc_managed(bytes)
            }
            fn free(&self, ptr: DevicePtr) -> Result<()> {
                self.inner.free(ptr)
            }
            fn copy_h2d(&self, src: &[u8], dst: DevicePtr) -> Result<()> {
                self.inner.copy_h2d(src, dst)
            }
            fn copy_d2h(&self, src: DevicePtr, dst: &mut [u8]) -> Result<()> {
                self.inner.copy_d2h(src, dst)
            }
            fn copy_d2d(&self, src: DevicePtr, dst: DevicePtr, bytes: usize) -> Result<()> {
                self.inner.copy_d2d(src, dst, bytes)
            }
            fn launch(
                &self,
                func: KernelHandle,
                grid: [u32; 3],
                block: [u32; 3],
                shared_mem: u32,
                stream: u64,
                params: &mut [*mut std::ffi::c_void],
            ) -> Result<()> {
                self.inner
                    .launch(func, grid, block, shared_mem, stream, params)
            }
            fn synchronize(&self, stream: u64) -> Result<()> {
                self.inner.synchronize(stream)
            }
            fn default_stream(&self) -> u64 {
                self.inner.default_stream()
            }
            fn kernel(&self, module: &str, func_name: &str) -> Result<KernelHandle> {
                self.inner.kernel(module, func_name)
            }
            fn memset(&self, ptr: DevicePtr, value: u8, bytes: usize) -> Result<()> {
                self.inner.memset(ptr, value, bytes)
            }
            fn memset_async(
                &self,
                ptr: DevicePtr,
                value: u8,
                bytes: usize,
                stream: u64,
            ) -> Result<()> {
                self.inner.memset_async(ptr, value, bytes, stream)
            }
            fn total_memory(&self) -> Result<usize> {
                self.inner.total_memory()
            }
            fn free_memory(&self) -> Result<usize> {
                self.inner.free_memory()
            }
            fn create_event(&self) -> Result<u64> {
                Ok(self.next_event.fetch_add(1, Ordering::Relaxed))
            }
            fn record_event(&self, _event: u64, _stream: u64) -> Result<()> {
                Ok(())
            }
            fn poll_event(&self, _event: u64) -> Result<bool> {
                Ok(self.ready.load(Ordering::Acquire))
            }
            fn destroy_event(&self, _event: u64) -> Result<()> {
                self.destroyed.fetch_add(1, Ordering::Relaxed);
                Ok(())
            }
        }

        let gpu = EventFenceGpu {
            inner: spark_runtime::gpu::mock::MockGpuBackend::new(),
            next_event: AtomicU64::new(1),
            ready: AtomicBool::new(false),
            destroyed: AtomicUsize::new(0),
        };
        let mut ring = MtpPinnedH2dRing::new(&gpu, 64).expect("event-fenced ring");
        let dst = gpu.alloc(32).unwrap();
        let half0 = [0x11u8; 24];
        let half1 = [0x22u8; 32];
        let wrap = [0x33u8; 24];

        // Fill half 0 and half 1 on stream 1 while events remain busy.
        assert!(
            ring.copy_h2d_group(&gpu, &[HostToDeviceCopy::new(&half0, dst)], 1)
                .unwrap()
        );
        assert!(
            ring.copy_h2d_group(&gpu, &[HostToDeviceCopy::new(&half1, dst)], 1)
                .unwrap()
        );
        // Wrap back into busy half 0 must decline (fallback to synchronous H2D).
        assert!(
            !ring
                .copy_h2d_group(&gpu, &[HostToDeviceCopy::new(&wrap, dst)], 1)
                .unwrap()
        );
        // Once prior DMA completes, wrap succeeds and copies the new payload.
        gpu.ready.store(true, Ordering::Release);
        assert!(
            ring.copy_h2d_group(&gpu, &[HostToDeviceCopy::new(&wrap, dst)], 1)
                .unwrap()
        );
        assert_eq!(gpu.inner.read_alloc(dst).unwrap()[..24], wrap);

        ring.destroy(&gpu);
        assert_eq!(gpu.destroyed.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn mtp_lm_head_tc_pipe_defaults_off_and_admits_aligned_b_ge_8_only_on_opt_in() {
        assert!(!parse_mtp_lm_head_tc_pipe_env(None));
        assert!(!parse_mtp_lm_head_tc_pipe_env(Some("0")));
        assert!(!parse_mtp_lm_head_tc_pipe_env(Some("auto")));
        assert!(parse_mtp_lm_head_tc_pipe_env(Some("1")));
        assert!(parse_mtp_lm_head_tc_pipe_env(Some(" 1 ")));

        let p = DevicePtr(256);
        assert!(!should_use_mtp_lm_head_pipe(true, 4, 2560, true, p, p, p));
        assert!(!should_use_mtp_lm_head_pipe(true, 7, 2560, true, p, p, p));
        assert!(should_use_mtp_lm_head_pipe(true, 8, 2560, true, p, p, p));
        assert!(should_use_mtp_lm_head_pipe(true, 16, 2560, true, p, p, p));
        assert!(!should_use_mtp_lm_head_pipe(false, 16, 2560, true, p, p, p));
        assert!(!should_use_mtp_lm_head_pipe(true, 16, 2560, false, p, p, p));
        assert!(!should_use_mtp_lm_head_pipe(true, 16, 2544, true, p, p, p));
        assert!(!should_use_mtp_lm_head_pipe(
            true,
            16,
            2560,
            true,
            DevicePtr(264),
            p,
            p,
        ));
    }

    #[test]
    fn batched_cohort_confidence_gate_never_early_stops_or_truncates_per_sequence() {
        let active_policy = GatePolicy {
            threshold: Some(0.6),
            max_drafts: None,
            early_stop: true,
            log: false,
        };
        // Even with MTP_CONF=0.6 + MTP_CONF_EARLY=1 + greedy, b >= 2 cohorts never
        // enable prob readback, mid-chain early-stop syncs, or per-sequence truncation.
        assert_eq!(
            resolve_chain_confidence_gate_for_scope(
                ChainConfidenceScope::BatchedCohort,
                active_policy,
                true,
                Some(0.6),
                true,
                false,
            ),
            ChainConfidenceGate {
                want_prob: false,
                early: None,
                apply_truncate: false,
            }
        );
        // Diagnostic logging on b >= 2 still skips early-stop and truncation.
        let logged_policy = GatePolicy {
            log: true,
            ..active_policy
        };
        assert_eq!(
            resolve_chain_confidence_gate_for_scope(
                ChainConfidenceScope::BatchedCohort,
                logged_policy,
                true,
                Some(0.6),
                true,
                false,
            ),
            ChainConfidenceGate {
                want_prob: true,
                early: None,
                apply_truncate: false,
            }
        );
        // Missing argmax_prob_k disables want_prob even with log=true.
        assert_eq!(
            resolve_chain_confidence_gate_for_scope(
                ChainConfidenceScope::BatchedCohort,
                logged_policy,
                true,
                Some(0.6),
                false,
                false,
            ),
            ChainConfidenceGate {
                want_prob: false,
                early: None,
                apply_truncate: false,
            }
        );
    }

    #[test]
    fn packed_meta_scratch_reuses_capacity_and_zeroes_active_span() {
        let mut buf = Vec::new();
        let s1 = prepare_packed_meta_scratch(&mut buf, 4096);
        assert_eq!(s1.len(), 4096);
        assert!(s1.iter().all(|&b| b == 0));
        s1.fill(0xAB);
        let ptr_after_grow = buf.as_ptr();

        // Shrinking reuses the exact allocation and zeroes the active prefix.
        let s2 = prepare_packed_meta_scratch(&mut buf, 1024);
        assert_eq!(s2.len(), 1024);
        assert!(s2.iter().all(|&b| b == 0));
        assert_eq!(buf.as_ptr(), ptr_after_grow);

        // Growing back up to high-water mark also zeroes previously written tail bytes.
        let s3 = prepare_packed_meta_scratch(&mut buf, 4096);
        assert_eq!(s3.len(), 4096);
        assert!(s3.iter().all(|&b| b == 0));
        assert_eq!(buf.as_ptr(), ptr_after_grow);
    }

    #[test]
    fn batch_gumbel_meta_layout_is_aligned_and_disjoint_up_to_c64() {
        for b in 2..=64usize {
            let bt_off = if b <= 8 { 768 } else { 2560 };
            let stride = (bt_off + b * 64 * 4).next_multiple_of(256);
            for num_drafts in 1..=7usize {
                let base_off = num_drafts * stride;
                let layout = BatchGumbelMetaLayout::compute(base_off, b, num_drafts)
                    .expect("layout fits without overflow");
                assert_eq!(layout.seeds_off % 8, 0, "b={b} num_drafts={num_drafts}");
                assert!(layout.seeds_off >= base_off);
                assert_eq!(layout.pos_base_off, layout.seeds_off + b * 8);
                assert_eq!(layout.pos_base_off % 8, 0, "b={b} num_drafts={num_drafts}");
                assert_eq!(layout.total_bytes % 256, 0);
                for step in 0..num_drafts {
                    let step_off = layout.step_pos_off(step, b);
                    assert_eq!(step_off % 8, 0, "b={b} step={step}");
                    assert!(step_off + b * 8 <= base_off + layout.total_bytes);
                }
            }
        }
    }

    #[test]
    fn mtp_coalesced_nvfp4_lm_head_and_fc_gate_admits_b_ge_6_and_enforces_bounds() {
        // LM head coalesced NVFP4 engages for b in 6..=QWEN4_NATIVE_MAX_ROWS when TC dense is active.
        for b in [2usize, 4, 5] {
            assert!(!should_use_mtp_lm_head_coalesced(true, b, 2560), "b={b}");
        }
        for b in [
            6usize,
            7,
            8,
            16,
            32,
            64,
            128,
            crate::layers::QWEN4_NATIVE_MAX_ROWS,
        ] {
            assert!(should_use_mtp_lm_head_coalesced(true, b, 2560), "b={b}");
            assert!(!should_use_mtp_lm_head_coalesced(false, b, 2560), "b={b}");
        }
        assert!(!should_use_mtp_lm_head_coalesced(
            true,
            crate::layers::QWEN4_NATIVE_MAX_ROWS + 1,
            2560
        ));
        assert!(!should_use_mtp_lm_head_coalesced(true, 16, 0));
        assert!(!should_use_mtp_lm_head_coalesced(true, 16, 2544));

        // Pre-FC coalesced NVFP4 engages for b in 6..=64 (4*b <= 256) with adequate staging.
        let qkv_cap = 64 * 2560 * 2;
        assert!(!should_use_mtp_fc_coalesced(
            true, 5, 4, 2560, qkv_cap, true
        ));
        for b in [6usize, 7, 8, 16, 32, 33, 48, 64] {
            assert!(
                should_use_mtp_fc_coalesced(true, b, 4, 2560, qkv_cap, true),
                "b={b}"
            );
            assert!(
                !should_use_mtp_fc_coalesced(false, b, 4, 2560, qkv_cap, true),
                "b={b}"
            );
            assert!(
                !should_use_mtp_fc_coalesced(true, b, 4, 2560, qkv_cap, false),
                "b={b}"
            );
        }
        // b=65 -> 4*65 = 260 > 256 fails closed; insufficient qkv_output or bad hc fails closed.
        assert!(!should_use_mtp_fc_coalesced(
            true,
            65,
            4,
            2560,
            128 * 2560 * 2,
            true
        ));
        assert!(!should_use_mtp_fc_coalesced(
            true,
            64,
            4,
            2560,
            64 * 2560 * 2 - 2,
            true
        ));
        assert!(!should_use_mtp_fc_coalesced(
            true,
            16,
            4,
            2560,
            16 * 2560 * 2 - 2,
            true
        ));
        assert!(!should_use_mtp_fc_coalesced(
            true, 16, 2, 2560, qkv_cap, true
        ));
        assert!(!should_use_mtp_fc_coalesced(
            true,
            usize::MAX,
            4,
            2560,
            usize::MAX,
            true
        ));
    }

    #[test]
    fn batched_argmax_prob_and_ragged_gpu_chain_gates_enforce_invariants() {
        assert!(!should_use_batched_argmax_prob(0, true));
        assert!(!should_use_batched_argmax_prob(1, true));
        assert!(should_use_batched_argmax_prob(2, true));
        assert!(should_use_batched_argmax_prob(64, true));
        assert!(!should_use_batched_argmax_prob(64, false));

        let lanes_64: Vec<usize> = (0..64).collect();
        let targets_homog = vec![4usize; 64];
        assert!(can_gpu_chain_ragged_steps(
            0.0,
            &lanes_64,
            &targets_homog,
            4,
            2048,
            2048,
            262_144,
            524_288
        ));
        // Positive pcum_thresh requires host compaction between steps.
        assert!(!can_gpu_chain_ragged_steps(
            0.30,
            &lanes_64,
            &targets_homog,
            4,
            2048,
            2048,
            262_144,
            524_288
        ));
        // Single-step chain already performs one final D2H.
        assert!(!can_gpu_chain_ragged_steps(
            0.0,
            &lanes_64,
            &vec![1usize; 64],
            1,
            2048,
            2048,
            262_144,
            524_288
        ));
        // Heterogeneous active target depths require per-step host compaction.
        let mut targets_hetero = targets_homog.clone();
        targets_hetero[17] = 2;
        assert!(!can_gpu_chain_ragged_steps(
            0.0,
            &lanes_64,
            &targets_hetero,
            4,
            2048,
            2048,
            262_144,
            524_288
        ));
        // Inactive zero-target lane excluded from active_lanes still allows GPU chaining.
        let active_subset: Vec<usize> = (0..64).filter(|&i| i != 17).collect();
        targets_hetero[17] = 0;
        assert!(can_gpu_chain_ragged_steps(
            0.0,
            &active_subset,
            &targets_hetero,
            4,
            2048,
            2048,
            262_144,
            524_288
        ));
        // Scratch / span overflow fails closed to per-step path.
        assert!(!can_gpu_chain_ragged_steps(
            0.0,
            &lanes_64,
            &vec![9usize; 64],
            9,
            2048,
            2048,
            262_144,
            524_288
        ));
        assert!(!can_gpu_chain_ragged_steps(
            0.0,
            &lanes_64,
            &targets_homog,
            4,
            2048,
            2048,
            524_289,
            524_288
        ));
    }

    #[test]
    fn ragged_mtp_env_parsers_and_c1_sync_mid_chain_defaults() {
        // Ragged MTP defaults ON when adaptive_k=true, OFF when adaptive_k=false or =0.
        assert!(parse_batched_ragged_mtp(None, true));
        assert!(parse_batched_ragged_mtp(Some("1"), true));
        assert!(!parse_batched_ragged_mtp(Some("0"), true));
        assert!(!parse_batched_ragged_mtp(Some("false"), true));
        assert!(!parse_batched_ragged_mtp(Some("off"), true));
        assert!(!parse_batched_ragged_mtp(None, false));
        assert!(!parse_batched_ragged_mtp(Some("1"), false));

        // p_cum threshold defaults to 0.30, honors valid [0, 1), falls back on invalid.
        assert!((parse_mtp_ragged_pcum_thresh(None) - 0.30).abs() < 1e-6);
        assert!((parse_mtp_ragged_pcum_thresh(Some("0.25")) - 0.25).abs() < 1e-6);
        assert!((parse_mtp_ragged_pcum_thresh(Some("0")) - 0.0).abs() < 1e-6);
        assert!((parse_mtp_ragged_pcum_thresh(Some("1.0")) - 0.30).abs() < 1e-6);
        assert!((parse_mtp_ragged_pcum_thresh(Some("-0.1")) - 0.30).abs() < 1e-6);
        assert!((parse_mtp_ragged_pcum_thresh(Some("nan")) - 0.30).abs() < 1e-6);

        // C=1 sync_mid_chain defaults ON so MTP_CONF early-stops low-confidence drafts,
        // while honoring explicit =0/=1 overrides.
        for n in 1..=8 {
            assert!(should_sync_mid_chain(None, n), "n={n}");
            assert!(should_sync_mid_chain(Some("1"), n), "n={n}");
            assert!(!should_sync_mid_chain(Some("0"), n), "n={n}");
        }
    }

    #[test]
    fn hidden_compaction_copies_are_non_overlapping_and_preserve_exact_rows() {
        let cases: &[(&[usize], &[usize])] = &[
            (&[0, 1, 2, 3, 4, 5], &[0, 1, 2, 3, 4, 5]),
            (&[0, 1, 2, 3, 4, 5], &[1, 2, 3, 5]),
            (&[0, 1, 2, 3, 4, 5], &[3, 4, 5]),
            (&[0, 1, 2, 3, 4, 5], &[0, 1, 4, 5]),
            (&[0, 1, 2, 3, 4, 5], &[5]),
            (&[1, 3, 5, 7], &[7]),
            (&[0, 1, 2, 3], &[]),
        ];
        for &(prev, next) in cases {
            let copies = plan_hidden_compaction_copies(prev, next);
            let mut rows: Vec<usize> = prev.to_vec();
            for &(src, dst, count) in &copies {
                assert!(count > 0);
                assert!(
                    dst + count <= src,
                    "overlapping copy src={src} dst={dst} count={count} for {prev:?}->{next:?}"
                );
                let tmp: Vec<usize> = rows[src..src + count].to_vec();
                rows[dst..dst + count].copy_from_slice(&tmp);
            }
            assert_eq!(
                &rows[..next.len()],
                next,
                "compaction mismatch for {prev:?}->{next:?} copies={copies:?}"
            );
        }
    }

    #[test]
    fn active_lane_pcum_compaction_expands_confident_lanes_and_stops_uncertain_lanes() {
        // 4 lanes with target_drafts = [6, 6, 6, 3], threshold = 0.15:
        // - lane 0 (hopeless prose): p0 = 0.10 < 0.15 -> stops at step 0 with 0 drafts (K=1), steps_run=1
        // - lane 1 (medium prose):   p = [0.60, 0.50, 0.40] -> p_cum = [0.60, 0.30, 0.12 < 0.15] -> 2 drafts (K=3), steps_run=3
        // - lane 2 (confident code): p = [0.95; 6] -> 6 drafts (K=7), steps_run=6 (runs last 3 steps with b_act==1!)
        // - lane 3 (capped target):  p = [0.90; 3], target=3 -> 3 drafts (K=4), steps_run=3
        let target_drafts = [6usize, 6, 6, 3];
        let max_steps = 6usize;
        let thresh = 0.15f32;
        let probs_by_lane: [&[f32]; 4] = [
            &[0.10],
            &[0.60, 0.50, 0.40],
            &[0.95, 0.95, 0.95, 0.95, 0.95, 0.95],
            &[0.90, 0.90, 0.90],
        ];
        for allow_zero in [false, true] {
            let mut active = vec![0usize, 1, 2, 3];
            let mut p_cum = vec![1.0f32; 4];
            let mut steps_run = vec![0usize; 4];
            let mut computed = vec![Vec::new(); 4];
            let mut emitted = vec![Vec::new(); 4];
            let mut cur_tokens = vec![10u32, 20, 30, 40];
            let mut active_counts = Vec::new();

            for step in 0..max_steps {
                if active.is_empty() {
                    break;
                }
                active_counts.push(active.len());
                let step_toks: Vec<u32> = active
                    .iter()
                    .map(|&s| 100 * (s as u32 + 1) + step as u32)
                    .collect();
                let step_probs: Vec<f32> = active.iter().map(|&s| probs_by_lane[s][step]).collect();
                active = advance_ragged_mtp_step(
                    step,
                    max_steps,
                    thresh,
                    allow_zero,
                    &active,
                    &step_toks,
                    &step_probs,
                    &target_drafts,
                    &mut p_cum,
                    &mut steps_run,
                    &mut computed,
                    &mut emitted,
                    &mut cur_tokens,
                    None,
                );
            }

            // Compaction progression: 4 -> 3 -> 3 -> 1 -> 1 -> 1 (zero wasted work on stopped lanes!).
            assert_eq!(active_counts, vec![4, 3, 3, 1, 1, 1]);
            assert!(active.is_empty());
            assert_eq!(steps_run, vec![1, 3, 6, 3]);
            assert_eq!(
                emitted[0].len(),
                usize::from(!allow_zero),
                "lane 0 -> K=1 when allow_zero else K=2 without extra steps"
            );
            assert_eq!(emitted[1].len(), 2, "lane 1 -> K=3");
            assert_eq!(emitted[2].len(), 6, "lane 2 -> K=7");
            assert_eq!(emitted[3].len(), 3, "lane 3 -> K=4 (target cap)");
            assert_eq!(computed[0].len(), 1, "KV steps match last_num_drafted");
            assert_eq!(computed[1].len(), 3, "KV steps match last_num_drafted");
            assert_eq!(computed[2].len(), 6, "KV steps match last_num_drafted");
            assert_eq!(computed[3].len(), 3, "KV steps match last_num_drafted");
        }
    }

    #[test]
    fn ragged_graph_bucket_quantization_trims_lowest_tail_pcum_and_preserves_k2_floor() {
        assert_eq!(ragged_row_bucket_step(2), 2);
        assert_eq!(ragged_row_bucket_step(4), 4);
        assert_eq!(ragged_row_bucket_step(8), 8);
        assert_eq!(ragged_row_bucket_step(16), 8);
        assert_eq!(ragged_row_bucket_step(32), 16);

        // Straggler stop engages for b >= 4, step >= 2, and <= 25% surviving.
        assert!(!ragged_mtp_should_stop_stragglers(3, 3, 1));
        assert!(ragged_mtp_should_stop_stragglers(4, 2, 1));
        assert!(!ragged_mtp_should_stop_stragglers(4, 2, 2));
        assert!(!ragged_mtp_should_stop_stragglers(8, 1, 1));
        assert!(ragged_mtp_should_stop_stragglers(8, 2, 1));
        assert!(ragged_mtp_should_stop_stragglers(8, 2, 2));
        assert!(!ragged_mtp_should_stop_stragglers(8, 2, 3));
        assert!(ragged_mtp_should_stop_stragglers(16, 2, 4));
        assert!(!ragged_mtp_should_stop_stragglers(16, 2, 5));

        // C=4 (step=4): 19 total rows -> trims 3 lowest-tail-pcum drafts to 16 rows
        // without dropping lane 0 below 1 draft (K=2).
        let mut drafts = vec![
            vec![1],
            vec![2, 3, 4],
            vec![5, 6, 7, 8, 9],
            vec![10, 11, 12, 13, 14, 15],
        ];
        let mut pcums = vec![
            vec![0.05],
            vec![0.70, 0.40, 0.12],
            vec![0.80, 0.55, 0.35, 0.20, 0.10],
            vec![0.95, 0.90, 0.85, 0.80, 0.75, 0.70],
        ];
        quantize_ragged_drafts_to_graph_bucket(&mut drafts, &mut pcums, false);
        let total: usize = drafts.iter().map(|d| d.len() + 1).sum();
        assert_eq!(total, 16);
        assert_eq!(drafts[0].len(), 1, "lane 0 preserved at K=2 floor");
        assert_eq!(drafts[1].len(), 2, "lane 1 trimmed 0.12 tail");
        assert_eq!(drafts[2].len(), 3, "lane 2 trimmed 0.10 and 0.20 tails");
        assert_eq!(drafts[3].len(), 6, "lane 3 confident drafts untouched");
    }

    #[test]
    fn mtp_graph_policy_auto_enables_above_64_sms_and_keeps_gb10_off() {
        assert_eq!(parse_mtp_graph_policy(None), MtpGraphPolicy::Auto);
        assert_eq!(parse_mtp_graph_policy(Some("auto")), MtpGraphPolicy::Auto);
        assert_eq!(parse_mtp_graph_policy(Some("0")), MtpGraphPolicy::Disabled);
        assert_eq!(
            parse_mtp_graph_policy(Some("false")),
            MtpGraphPolicy::Disabled
        );
        assert_eq!(
            parse_mtp_graph_policy(Some("OFF")),
            MtpGraphPolicy::Disabled
        );
        assert_eq!(parse_mtp_graph_policy(Some("1")), MtpGraphPolicy::Enabled);
        assert_eq!(
            parse_mtp_graph_policy(Some("true")),
            MtpGraphPolicy::Enabled
        );
        assert_eq!(parse_mtp_graph_policy(Some("ON")), MtpGraphPolicy::Enabled);

        // GB10 (<= 64 SMs): default OFF under Auto; explicit "1" opts in.
        for sm in [48u32, 64u32] {
            assert!(!resolve_mtp_graph_enabled(MtpGraphPolicy::Auto, sm));
            assert!(!resolve_mtp_graph_enabled(MtpGraphPolicy::Disabled, sm));
            assert!(resolve_mtp_graph_enabled(MtpGraphPolicy::Enabled, sm));
        }
        // SM120 / SM103 (> 64 SMs): auto-ON under Auto; explicit "0" opts out.
        for sm in [128u32, 188u32] {
            assert!(resolve_mtp_graph_enabled(MtpGraphPolicy::Auto, sm));
            assert!(!resolve_mtp_graph_enabled(MtpGraphPolicy::Disabled, sm));
            assert!(resolve_mtp_graph_enabled(MtpGraphPolicy::Enabled, sm));
        }
    }

    #[test]
    fn mtp_graph_bt_bucket_and_cache_lifecycle() {
        let scratch = 16 * 1024 * 1024;
        assert_eq!(mtp_graph_bt_blocks_cap(1, 32, 3, 2560, scratch), 16);
        assert_eq!(mtp_graph_bt_blocks_cap(16, 32, 3, 2560, scratch), 16);
        assert_eq!(mtp_graph_bt_blocks_cap(17, 32, 3, 2560, scratch), 32);
        assert_eq!(mtp_graph_bt_blocks_cap(65, 64, 3, 2560, scratch), 128);
        // Tiny scratch fails closed to exact max_mb instead of overflowing.
        assert_eq!(
            mtp_graph_bt_blocks_cap(17, 64, 4, 2560, CHAIN_META_OFFSET + 32_768),
            17
        );

        let gpu = spark_runtime::gpu::mock::MockGpuBackend::new();
        let mut cache = MtpGraphCache::default();
        let make_key = |batch_size: usize, step_count: usize, want_prob: bool| MtpGraphKey {
            batch_size,
            step_count,
            want_prob,
            bt_blocks_cap: 16,
            vocab: 248_320,
            write_step_token_out: true,
            tc_dense_scope: true,
            coalesced_fc: true,
            coalesced_lm_head: true,
            scratch_ptr: 0x1000,
            hidden_ptr: 0x2000,
            residual_ptr: 0x3000,
            logits_ptr: 0x4000,
            k_pool_ptr: 0x5000,
            v_pool_ptr: 0x6000,
            stream: 1,
        };

        let k1 = make_key(32, 3, false);
        assert!(cache.get(&k1).is_none());
        assert!(cache.mark_warmed(k1));
        assert!(!cache.mark_warmed(k1));
        cache.insert(&gpu, k1, GraphHandle(42));
        assert_eq!(cache.get(&k1).map(|h| h.0), Some(42));

        let k_prob = make_key(32, 3, true);
        assert!(cache.get(&k_prob).is_none());
        cache.mark_poisoned(k_prob);
        assert!(cache.is_poisoned(&k_prob));

        // FIFO eviction enforces MTP_GRAPH_CACHE_CAP cleanly.
        for i in 0..(MTP_GRAPH_CACHE_CAP + 4) {
            let k = make_key(i + 2, 1, false);
            cache.insert(&gpu, k, GraphHandle((i + 100) as u64));
        }
        assert_eq!(cache.graphs.len(), MTP_GRAPH_CACHE_CAP);
        assert!(cache.get(&k1).is_none(), "oldest entry evicted at cap");

        cache.clear(&gpu);
        assert!(cache.graphs.is_empty());
        assert!(cache.order.is_empty());
        assert!(cache.warmed.is_empty());
        assert!(cache.poisoned.is_empty());
    }
}
