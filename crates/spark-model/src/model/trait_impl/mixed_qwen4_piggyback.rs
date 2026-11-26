// SPDX-License-Identifier: AGPL-3.0-only

//! Single-forward mixed prefill+decode piggybacking for Qwen4 (`0` extra MoE
//! weight bytes).
//!
//! Packs one prefill chunk (`[0..P)`) and one cross-sequence speculative
//! verify frame (`[P..P+D)`) contiguously in `hidden`/`residual`:
//!
//! - **Sub-phase A (Split Mixer)**: prefill rows `[0..P)` run their attention
//!   or SSM mixer under [`ExpertMajorSliceGuard`] (skipping MoE), followed by
//!   decode verify rows `[P..P+D)` running [`TransformerLayer::decode_qwen4_cross_seq_mixer_only`].
//! - **Sub-phase B (Unified MLP/MoE)**: all `N = P + D` rows pass through ONE
//!   [`TransformerLayer::qwen4_prefill_finish_moe`] call under
//!   [`MixedMoeRowsGuard`], streaming the 512 NVFP4 routed experts once for
//!   both prefill and decode.
//! - **Split Finalization**: prefill row `P - 1` (on `is_last_prefill_chunk`)
//!   writes `buffers.logits()[D]`, decode rows `[P..P+D)` write
//!   `buffers.logits()[0..D)` + argmax, and decode pre-final-norm hidden rows
//!   are compacted to `hidden[0..D)` for `save_hidden_for_mtp_sequence_in_frame`.
//!
//! Gated by `ATLAS_QWEN4_MIXED_PIGGYBACK` (default [`MixedPiggybackMode::ExactStandaloneOnly`]).

use anyhow::{Context, Result, bail, ensure};
use atlas_core::config::LayerType;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::super::block_mgmt::{ensure_blocks_through_decode, ensure_blocks_through_prefill};
use super::super::types::TransformerModel;
use super::verify_qwen4_batched::{
    ensure_verify_meta_disjoint_from_scratch, frame_metadata_bytes, moe_routing_scratch_bytes,
    ranges_disjoint, reset_dedup_slot,
};
use crate::layer::{
    AttnMetadataDev, QWEN4_CROSS_SEQ_MAX_ROWS, Qwen4CrossSeqFrame, Qwen4CrossSeqRows,
};
use crate::layers::ops;
use crate::layers::qwen4_prefill_moe::{ExpertMajorSliceGuard, MixedMoeRowsGuard};
use crate::traits::{Model, SequenceState};

/// Environment selector controlling single-forward mixed prefill+decode
/// piggybacking on Qwen4.
pub const MIXED_PIGGYBACK_ENV: &str = "ATLAS_QWEN4_MIXED_PIGGYBACK";

/// Execution policy for single-forward mixed prefill+decode piggybacking.
///
/// Defaults to [`Self::ExactStandaloneOnly`] until live parity-certified so
/// standalone prefill and C1/batched verify keep zero numerical drift.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MixedPiggybackMode {
    /// Fuse prefill `[0..P)` and decode verify `[P..P+D)` through one shared
    /// MoE weight pass per layer (`ATLAS_QWEN4_MIXED_PIGGYBACK=1` or `auto`).
    Auto,
    /// Keep exact standalone prefill and decode verify paths (default when
    /// unset or set to `exact` / `standalone` / `exact_standalone_only`).
    ExactStandaloneOnly,
    /// Explicitly disable mixed piggybacking (`0` / `off` / `disabled` / `false`).
    Disabled,
}

impl MixedPiggybackMode {
    /// Parse an explicit `ATLAS_QWEN4_MIXED_PIGGYBACK` value; rejects malformed
    /// spellings fail-closed.
    pub fn parse(value: Option<&str>) -> Result<Self> {
        match value {
            None => Ok(Self::ExactStandaloneOnly),
            Some("1" | "auto" | "AUTO") => Ok(Self::Auto),
            Some("exact" | "EXACT" | "standalone" | "STANDALONE" | "exact_standalone_only") => {
                Ok(Self::ExactStandaloneOnly)
            }
            Some("0" | "off" | "OFF" | "disabled" | "DISABLED" | "false" | "FALSE") => {
                Ok(Self::Disabled)
            }
            Some(other) => bail!(
                "{MIXED_PIGGYBACK_ENV}: unsupported value {other:?} \
                 (expected absent, 1/auto, exact/standalone, or 0/off/disabled)"
            ),
        }
    }

    /// Resolve the active policy from the process environment.
    pub fn from_env() -> Result<Self> {
        match std::env::var(MIXED_PIGGYBACK_ENV) {
            Ok(v) => Self::parse(Some(&v)),
            Err(std::env::VarError::NotPresent) => Self::parse(None),
            Err(err) => Err(err).with_context(|| format!("invalid {MIXED_PIGGYBACK_ENV}")),
        }
    }

    /// True only when single-forward mixed piggybacking is permitted.
    pub const fn allows_piggyback(self) -> bool {
        matches!(self, Self::Auto)
    }
}

/// Inputs for one single-forward mixed prefill+decode verify round on Qwen4.
pub struct Qwen4MixedRound<'a> {
    pub prefill_tokens: &'a [u32],
    pub prefill_seq: &'a mut SequenceState,
    pub prefill_chunk_start: usize,
    pub prefill_chunk_len: usize,
    pub is_last_prefill_chunk: bool,
    pub decode_tokens_per_seq: &'a [Vec<u32>],
    pub decode_seqs: &'a mut [&'a mut SequenceState],
    pub stream: u64,
}

/// Outputs produced by [`TransformerModel::forward_qwen4_mixed_prefill_verify`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Qwen4MixedRoundOutput {
    /// Per-sequence verified greedy argmax tokens (`decode_seqs.len()` × `K`).
    pub decode_verified: Vec<Vec<u32>>,
    /// Base pointer of decode verify BF16 logits `[0..D)` in `buffers.logits()`.
    pub decode_logits: DevicePtr,
    /// Pointer to prefill last-token BF16 logits at `buffers.logits()[D]` when
    /// `is_last_prefill_chunk` is true, or [`DevicePtr::NULL`] otherwise.
    pub prefill_logits: DevicePtr,
    /// Number of prefill rows (`P`) processed in `[0..P)`.
    pub prefill_rows: usize,
    /// Number of decode verify rows (`D`) processed in `[P..P+D)`.
    pub decode_rows: usize,
}

/// Pure buffer-offset and capacity plan for one mixed prefill (`[0..P)`) +
/// decode verify (`[P..P+D)`) round.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct MixedPiggybackLayout {
    pub prefill_rows: usize,
    pub decode_rows: usize,
    pub total_rows: usize,
    pub row_bytes: usize,
    pub hidden_decode_offset_bytes: usize,
    pub hidden_total_bytes: usize,
    pub moe_routing_scratch_bytes: usize,
    pub prefill_meta_offset_bytes: usize,
    pub prefill_meta_bytes: usize,
    pub decode_attn_meta_bytes: usize,
    pub decode_logits_bytes: usize,
    pub prefill_logits_offset_bytes: usize,
    pub prefill_logits_bytes: usize,
    pub logits_total_bytes: usize,
}

/// Arena geometry parameters passed to [`MixedPiggybackLayout::compute`].
#[derive(Clone, Copy, Debug)]
pub(crate) struct MixedLayoutParams {
    pub prefill_rows: usize,
    pub decode_rows: usize,
    pub max_prefill_chunk_rows: usize,
    pub max_batch_tokens: usize,
    pub hidden_size: usize,
    pub residual_width: usize,
    pub top_k: usize,
    pub use_mrope: bool,
    pub max_blocks_per_seq: usize,
    pub vocab_size: usize,
    pub is_last_prefill_chunk: bool,
    pub arena_hidden_bytes: usize,
    pub arena_residual_bytes: usize,
    pub arena_scratch_bytes: usize,
    pub arena_logits_bytes: usize,
}

impl MixedPiggybackLayout {
    /// Validate row bounds, arena capacities, and compute disjoint byte offsets.
    pub(crate) fn compute(params: MixedLayoutParams) -> Result<Self> {
        ensure!(
            (2..=params.max_prefill_chunk_rows).contains(&params.prefill_rows),
            "mixed prefill rows P={} must be in 2..={}",
            params.prefill_rows,
            params.max_prefill_chunk_rows
        );
        ensure!(
            (1..=QWEN4_CROSS_SEQ_MAX_ROWS).contains(&params.decode_rows),
            "mixed decode rows D={} must be in 1..={QWEN4_CROSS_SEQ_MAX_ROWS}",
            params.decode_rows
        );
        let total_rows = params
            .prefill_rows
            .checked_add(params.decode_rows)
            .context("mixed P + D row count overflow")?;
        ensure!(
            total_rows <= params.max_batch_tokens,
            "mixed rows P+D={total_rows} exceed buffers.max_batch_tokens()={}",
            params.max_batch_tokens
        );

        let row_bytes = params
            .residual_width
            .checked_mul(2)
            .context("mixed residual row byte size overflow")?;
        ensure!(
            params.hidden_size > 0 && params.hidden_size.checked_mul(2).is_some(),
            "mixed core hidden row byte size invalid"
        );
        let hidden_decode_offset_bytes = params
            .prefill_rows
            .checked_mul(row_bytes)
            .context("mixed hidden decode offset overflow")?;
        let hidden_total_bytes = total_rows
            .checked_mul(row_bytes)
            .context("mixed hidden total bytes overflow")?;
        ensure!(
            hidden_total_bytes <= params.arena_hidden_bytes
                && hidden_total_bytes <= params.arena_residual_bytes,
            "mixed hidden/residual footprint ({hidden_total_bytes} B) exceeds serving arena \
             (hidden={}, residual={})",
            params.arena_hidden_bytes,
            params.arena_residual_bytes
        );

        // `ffn.forward_prefill` writes `[0 .. 2 * (P + D) * top_k * 4)` of `buffers.scratch()`.
        // Place prefill positions/slots strictly after that MoE routing prefix.
        let moe_routing_scratch_bytes = moe_routing_scratch_bytes(total_rows, params.top_k)
            .context("mixed MoE routing scratch byte size overflow")?;
        let prefill_meta_offset_bytes = moe_routing_scratch_bytes
            .checked_next_multiple_of(8)
            .context("mixed prefill metadata alignment overflow")?;
        let pos_streams = if params.use_mrope { 3usize } else { 1usize };
        let prefill_pos_bytes = params
            .prefill_rows
            .checked_mul(4)
            .and_then(|b| b.checked_mul(pos_streams))
            .context("mixed prefill position metadata overflow")?;
        let prefill_slot_offset = prefill_pos_bytes
            .checked_next_multiple_of(8)
            .context("mixed prefill slot offset overflow")?;
        let prefill_slot_bytes = params
            .prefill_rows
            .checked_mul(8)
            .context("mixed prefill slot metadata overflow")?;
        let prefill_meta_bytes = prefill_slot_offset
            .checked_add(prefill_slot_bytes)
            .context("mixed prefill metadata total overflow")?;
        let prefill_meta_end = prefill_meta_offset_bytes
            .checked_add(prefill_meta_bytes)
            .context("mixed scratch end overflow")?;
        ensure!(
            prefill_meta_end <= params.arena_scratch_bytes,
            "mixed scratch footprint ({prefill_meta_end} B = moe {moe_routing_scratch_bytes} B + \
             prefill_meta {prefill_meta_bytes} B) exceeds arena scratch ({} B)",
            params.arena_scratch_bytes
        );

        let decode_attn_meta_bytes =
            frame_metadata_bytes(params.decode_rows, params.max_blocks_per_seq)
                .context("mixed decode verify metadata exceeds 128-row ABI")?;

        let vocab_row_bytes = params
            .vocab_size
            .checked_mul(2)
            .context("mixed vocab row byte size overflow")?;
        let decode_logits_bytes = params
            .decode_rows
            .checked_mul(vocab_row_bytes)
            .context("mixed decode logits bytes overflow")?;
        let prefill_logits_offset_bytes = decode_logits_bytes;
        let prefill_logits_bytes = if params.is_last_prefill_chunk {
            vocab_row_bytes
        } else {
            0
        };
        let logits_total_bytes = decode_logits_bytes
            .checked_add(prefill_logits_bytes)
            .context("mixed total logits bytes overflow")?;
        ensure!(
            logits_total_bytes <= params.arena_logits_bytes,
            "mixed logits footprint ({logits_total_bytes} B) exceeds logits arena ({} B)",
            params.arena_logits_bytes
        );

        Ok(Self {
            prefill_rows: params.prefill_rows,
            decode_rows: params.decode_rows,
            total_rows,
            row_bytes,
            hidden_decode_offset_bytes,
            hidden_total_bytes,
            moe_routing_scratch_bytes,
            prefill_meta_offset_bytes,
            prefill_meta_bytes,
            decode_attn_meta_bytes,
            decode_logits_bytes,
            prefill_logits_offset_bytes,
            prefill_logits_bytes,
            logits_total_bytes,
        })
    }

    /// Prove that every live device region in a mixed round is non-overlapping.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn verify_device_non_aliasing(
        self,
        hidden_base: DevicePtr,
        residual_base: DevicePtr,
        scratch_base: DevicePtr,
        scratch_arena_bytes: usize,
        decode_attn_meta_dev: DevicePtr,
        qsa_verify_meta_dev: DevicePtr,
        qsa_verify_meta_bytes: usize,
        logits_base: DevicePtr,
    ) -> Result<()> {
        let decode_hidden_bytes = self
            .decode_rows
            .checked_mul(self.row_bytes)
            .context("decode hidden bytes overflow")?;
        ensure!(
            self.hidden_decode_offset_bytes
                .checked_add(decode_hidden_bytes)
                == Some(self.hidden_total_bytes)
                && self
                    .decode_logits_bytes
                    .checked_add(self.prefill_logits_bytes)
                    == Some(self.logits_total_bytes),
            "mixed layout byte extent invariant mismatch"
        );
        ensure!(
            ranges_disjoint(
                hidden_base.0,
                self.hidden_decode_offset_bytes,
                hidden_base.offset(self.hidden_decode_offset_bytes).0,
                decode_hidden_bytes,
            ),
            "mixed hidden prefill [0..P) and decode [P..P+D) ranges overlap"
        );
        ensure!(
            ranges_disjoint(
                residual_base.0,
                self.hidden_decode_offset_bytes,
                residual_base.offset(self.hidden_decode_offset_bytes).0,
                decode_hidden_bytes,
            ),
            "mixed residual prefill [0..P) and decode [P..P+D) ranges overlap"
        );
        ensure!(
            ranges_disjoint(
                scratch_base.0,
                self.moe_routing_scratch_bytes,
                scratch_base.offset(self.prefill_meta_offset_bytes).0,
                self.prefill_meta_bytes,
            ),
            "mixed prefill metadata aliases MoE routing scratch [0..2*(P+D)*top_k*4)"
        );
        ensure_verify_meta_disjoint_from_scratch(
            decode_attn_meta_dev,
            self.decode_attn_meta_bytes,
            scratch_base,
            scratch_arena_bytes.max(self.moe_routing_scratch_bytes),
            "mixed_decode_attn_meta",
        )?;
        ensure_verify_meta_disjoint_from_scratch(
            qsa_verify_meta_dev,
            qsa_verify_meta_bytes,
            scratch_base,
            scratch_arena_bytes.max(self.moe_routing_scratch_bytes),
            "mixed_qsa_verify_meta",
        )?;
        if qsa_verify_meta_bytes > 0 {
            ensure!(
                ranges_disjoint(
                    decode_attn_meta_dev.0,
                    self.decode_attn_meta_bytes,
                    qsa_verify_meta_dev.0,
                    qsa_verify_meta_bytes,
                ),
                "mixed decode_attn_meta and qsa_verify_meta overlap"
            );
        }
        if self.prefill_logits_bytes > 0 {
            ensure!(
                ranges_disjoint(
                    logits_base.0,
                    self.decode_logits_bytes,
                    logits_base.offset(self.prefill_logits_offset_bytes).0,
                    self.prefill_logits_bytes,
                ),
                "mixed decode logits [0..D) and prefill logits [D] overlap"
            );
        }
        Ok(())
    }
}

/// Non-overlapping copy segments `(src_row, dst_row, count)` that relocate
/// `hidden[src_start..src_start + num_rows)` down to `hidden[0..num_rows)`
/// without ever overlapping source and destination within a single D2D copy.
pub(crate) fn plan_hidden_rows_shift_down(
    src_start: usize,
    num_rows: usize,
) -> Vec<(usize, usize, usize)> {
    if src_start == 0 || num_rows == 0 {
        return Vec::new();
    }
    let mut chunks = Vec::new();
    let mut moved = 0usize;
    while moved < num_rows {
        let count = src_start.min(num_rows - moved);
        chunks.push((src_start + moved, moved, count));
        moved += count;
    }
    chunks
}

fn copy_hidden_rows_down(
    gpu: &dyn GpuBackend,
    hidden: DevicePtr,
    src_start: usize,
    num_rows: usize,
    row_bytes: usize,
    stream: u64,
) -> Result<()> {
    for (src_row, dst_row, count) in plan_hidden_rows_shift_down(src_start, num_rows) {
        gpu.copy_d2d_async(
            hidden.offset(src_row * row_bytes),
            hidden.offset(dst_row * row_bytes),
            count * row_bytes,
            stream,
        )?;
    }
    Ok(())
}

impl TransformerModel {
    /// Execute one single-forward mixed prefill+decode round on Qwen4:
    /// prefill chunk `[0..P)` and cross-sequence decode verify `[P..P+D)` run
    /// separate mixers per layer and ONE unified MLP/MoE pass over all `P + D`
    /// rows (`0` extra MoE weight bytes).
    ///
    /// # Errors
    ///
    /// Fails closed before mutating any state when policy disables piggybacking,
    /// bounds/non-aliasing checks fail, or layer preflight refuses the frame.
    /// Any error after acquiring batch ownership aborts the verify cohort
    /// fail-closed.
    pub fn forward_qwen4_mixed_prefill_verify(
        &self,
        round: Qwen4MixedRound<'_>,
    ) -> Result<Qwen4MixedRoundOutput> {
        crate::model::verify_failure::check()?;
        self.ensure_verify_healthy()?;

        let mode = MixedPiggybackMode::from_env()?;
        ensure!(
            mode.allows_piggyback(),
            "single-forward mixed prefill+decode is not enabled by {MIXED_PIGGYBACK_ENV} \
             (mode={mode:?})"
        );
        ensure!(
            self.config.is_qwen4_exp()
                && !self.config.use_fp32_residual()
                && !self.use_fp32_logits
                && self.comm.is_none()
                && self.config.pp_world_size <= 1
                && self.dflash_capture_layers.is_empty(),
            "single-forward mixed prefill+decode requires single-rank BF16-residual Qwen4"
        );
        ensure!(
            crate::layers::qwen4_prefill_moe::selected()?
                && !crate::layers::qwen4_prefill_moe::family_suppressed(),
            "single-forward mixed prefill+decode requires ATLAS_QWEN4_PREFILL_MOE_BATCH=1 \
             without family suppression"
        );
        ensure!(
            !self.vision_prompt_present(round.prefill_tokens),
            "single-forward mixed prefill+decode does not support multimodal vision prompts"
        );

        let p_rows = round.prefill_chunk_len;
        let prefill_end = round
            .prefill_chunk_start
            .checked_add(p_rows)
            .context("mixed prefill chunk end overflow")?;
        ensure!(
            prefill_end <= round.prefill_tokens.len(),
            "mixed prefill chunk [{}..{}) exceeds prompt length {}",
            round.prefill_chunk_start,
            prefill_end,
            round.prefill_tokens.len()
        );
        ensure!(
            round.prefill_seq.seq_len == round.prefill_chunk_start
                && round.prefill_seq.tokens.len() == round.prefill_chunk_start,
            "mixed prefill sequence cursor (seq_len={}, tokens={}) != chunk_start={}",
            round.prefill_seq.seq_len,
            round.prefill_seq.tokens.len(),
            round.prefill_chunk_start
        );
        ensure!(
            round.prefill_seq.layer_states.len() == self.layers.len(),
            "mixed prefill sequence lacks per-layer states"
        );
        ensure!(
            round.prefill_seq.cached_prefix_tokens <= round.prefill_chunk_start,
            "single-forward mixed prefill+decode requires an uncached prefill chunk"
        );
        self.ensure_verify_sequence_idle(round.prefill_seq)?;
        ensure!(
            round
                .decode_seqs
                .iter()
                .all(|seq| seq.slot_idx != round.prefill_seq.slot_idx),
            "mixed prefill slot {} aliases a decode verify slot",
            round.prefill_seq.slot_idx
        );

        let decode_views: Vec<&SequenceState> =
            round.decode_seqs.iter().map(|seq| &**seq).collect();
        let frame = self.qwen4_cross_seq_admission(round.decode_tokens_per_seq, &decode_views)?;
        let k = frame.k();
        let d_rows = frame.rows();
        let max_prefill_chunk_rows = crate::layers::qwen4_prefill_moe::max_chunk_rows();
        let arenas = self.buffers.sizes();
        let layout = MixedPiggybackLayout::compute(MixedLayoutParams {
            prefill_rows: p_rows,
            decode_rows: d_rows,
            max_prefill_chunk_rows,
            max_batch_tokens: self.buffers.max_batch_tokens(),
            hidden_size: self.config.hidden_size,
            residual_width: self.config.residual_width(),
            top_k: self.config.num_experts_per_tok,
            use_mrope: self.config.mrope_interleaved,
            max_blocks_per_seq: self.max_blocks_per_seq as usize,
            vocab_size: self.config.vocab_size,
            is_last_prefill_chunk: round.is_last_prefill_chunk,
            arena_hidden_bytes: arenas.hidden_states,
            arena_residual_bytes: arenas.residual,
            arena_scratch_bytes: arenas.scratch,
            arena_logits_bytes: arenas.logits,
        })?;

        let staging = self
            .qwen4_cross_seq_staging
            .context("cross-sequence verify staging was not provisioned")?;
        let (decode_attn_meta_dev, _) =
            self.ensure_disjoint_verify_attn_meta(layout.decode_attn_meta_bytes)?;
        let hidden = self.buffers.hidden_states();
        let residual = self.buffers.residual();
        layout.verify_device_non_aliasing(
            hidden,
            residual,
            self.buffers.scratch(),
            arenas.scratch,
            decode_attn_meta_dev,
            staging.verify_meta_dev,
            staging.verify_meta_bytes,
            self.buffers.logits(),
        )?;

        // Read-only layer admission before acquiring batch ownership or mutating GPU state.
        crate::layers::qwen4_prefill_moe::admit_request(
            &self.config,
            p_rows,
            round.prefill_chunk_start,
        )?;
        let decode_ctx = self.qwen4_cross_seq_ctx(None);
        crate::layers::qwen4_prefill_moe::validate_mixed_moe(&decode_ctx, p_rows, d_rows)?;
        let qsa_required: Vec<bool> = decode_views
            .iter()
            .map(|seq| seq.qwen4_qsa_required)
            .collect();
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            ensure!(
                layer.qwen4_cross_seq_admits_frames(),
                "mixed verify layer {layer_idx} has a poisoned pointer table"
            );
            layer
                .preflight_qwen4_cross_seq_mixer_only(frame, &qsa_required, &decode_ctx)
                .with_context(|| {
                    format!("mixed verify layer {layer_idx} failed mixer-only preflight")
                })?;
        }

        // Acquire OwnedBatch ownership (unless caller already holds an active cohort).
        let already_owned = round
            .decode_seqs
            .iter()
            .any(|seq| seq.verify_frame.is_some_and(|f| f.batch().is_some()));
        if !already_owned {
            self.begin_owned_verify_batch(round.decode_seqs, k)?;
        }
        if let Err(err) = self.begin_owned_batch_forward(round.decode_seqs, k) {
            let _ = self.abort_owned_verify_batch(round.decode_seqs);
            return Err(err);
        }

        let exec_result = self.execute_qwen4_mixed_prefill_verify_owned(
            frame,
            layout,
            max_prefill_chunk_rows,
            staging,
            round.prefill_tokens,
            round.prefill_seq,
            round.prefill_chunk_start,
            round.is_last_prefill_chunk,
            round.decode_tokens_per_seq,
            round.decode_seqs,
            round.stream,
        );
        match exec_result {
            Ok(output) => {
                self.record_owned_batch_execution(round.decode_seqs);
                Ok(output)
            }
            Err(err) => {
                self.poison_owned_batch(round.decode_seqs);
                let _ = self.abort_owned_verify_batch(round.decode_seqs);
                Err(err)
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn execute_qwen4_mixed_prefill_verify_owned(
        &self,
        frame: Qwen4CrossSeqFrame,
        layout: MixedPiggybackLayout,
        max_prefill_chunk_rows: usize,
        staging: crate::layer::Qwen4CrossSeqStaging,
        prefill_tokens: &[u32],
        prefill_seq: &mut SequenceState,
        prefill_chunk_start: usize,
        is_last_prefill_chunk: bool,
        decode_tokens_per_seq: &[Vec<u32>],
        decode_seqs: &mut [&mut SequenceState],
        requested_stream: u64,
    ) -> Result<Qwen4MixedRoundOutput> {
        let stream = if requested_stream == 0 {
            self.gpu.default_stream()
        } else {
            requested_stream
        };
        let p_rows = layout.prefill_rows;
        let d_rows = layout.decode_rows;
        let total_rows = layout.total_rows;
        let k = frame.k();
        let row_bytes = layout.row_bytes;
        let hidden = self.buffers.hidden_states();
        let residual = self.buffers.residual();
        let hidden_decode = hidden.offset(layout.hidden_decode_offset_bytes);
        let residual_decode = residual.offset(layout.hidden_decode_offset_bytes);

        self.dflash_flat_tree_route
            .store(false, std::sync::atomic::Ordering::Release);
        self.ddtree_dfs_inv_perm.lock().clear();

        // ── Launch async PLE pre-reads for BOTH prefill and decode before embed ──
        #[cfg(all(feature = "cuda", target_os = "linux"))]
        let ple_prefill_batch = std::env::var("ATLAS_QWEN4_PLE_PREFILL_BATCH")
            .ok()
            .as_deref()
            == Some("1");
        #[cfg(all(feature = "cuda", target_os = "linux"))]
        let mut prefill_ple_prefetch = if ple_prefill_batch
            && crate::layers::qwen4_ple::prefill_stream_level() >= 2
            && let Some(ple) = self.qwen4_ple.as_ref()
        {
            let prior = &prefill_tokens[..prefill_chunk_start];
            let current = &prefill_tokens[prefill_chunk_start..prefill_chunk_start + p_rows];
            Some(ple.begin_prefill_read(current, prior))
        } else {
            None
        };
        #[cfg(all(feature = "cuda", target_os = "linux"))]
        let mut decode_ple_warm_ticket = if let Some(ple) = self.qwen4_ple.as_ref()
            && crate::layers::qwen4_ple::ple_verify_batch_selected()
        {
            let warm_seqs: Vec<crate::layers::qwen4_ple::PleWarmSeq> = decode_seqs
                .iter()
                .zip(decode_tokens_per_seq)
                .map(|(seq, window)| {
                    crate::layers::qwen4_ple::PleWarmSeq::from_slices(window, &seq.tokens)
                })
                .collect::<Result<_>>()?;
            Some(ple.submit_warm_cross_seq_verify(warm_seqs)?)
        } else {
            None
        };

        // ── Restore decode checkpoints & embed [0..P) + [P..P+D) contiguously ──
        let unique_leaders: Vec<usize> = (0..decode_seqs.len()).collect();
        let mut decode_flat_tokens = Vec::with_capacity(d_rows);
        for (s, seq) in decode_seqs.iter_mut().enumerate() {
            reset_dedup_slot(seq.slot_idx);
            self.pre_verify_copy_async(seq)?;
            decode_flat_tokens.extend_from_slice(&decode_tokens_per_seq[s]);
        }
        self.prefill_b_embed_chunk_at(prefill_tokens, prefill_chunk_start, p_rows, hidden, stream)?;
        self.embed_qwen4_batch(&decode_flat_tokens, hidden_decode, stream)?;

        // ── Allocate KV blocks & upload disjoint prefill + decode metadata ──
        let mut kv_cache = self.kv_cache.lock();
        ensure!(
            kv_cache.config().cache_blocks_per_seq.is_none(),
            "single-forward mixed prefill+decode requires resident non-HSS KV"
        );
        let block_size = kv_cache.block_size();
        let prefill_end = prefill_chunk_start + p_rows;
        let prefill_blocks_needed = prefill_end.div_ceil(block_size);
        ensure_blocks_through_prefill(
            prefill_seq,
            prefill_blocks_needed - 1,
            &mut kv_cache,
            self.prefix_cache.as_ref(),
            self.gpu.as_ref(),
            stream,
        )?;
        for seq in decode_seqs.iter_mut() {
            for t in 0..k {
                let pos = seq.seq_len + t;
                ensure_blocks_through_decode(
                    seq,
                    pos / block_size,
                    &mut kv_cache,
                    self.prefix_cache.as_ref(),
                    self.gpu.as_ref(),
                    stream,
                )?;
            }
        }

        let prefill_meta_base = self
            .buffers
            .scratch()
            .offset(layout.prefill_meta_offset_bytes);
        let pinned_offset = if Self::fast_host_h2d_enabled() {
            self.reserve_pinned_h2d_range(p_rows.saturating_mul(20).saturating_add(64))
                .unwrap_or(0)
        } else {
            0
        };
        let prefill_meta_layout = self.prefill_b_upload_meta_at(
            prefill_tokens,
            prefill_seq,
            prefill_chunk_start,
            p_rows,
            prefill_chunk_start,
            p_rows,
            prefill_chunk_start,
            &kv_cache,
            prefill_meta_base,
            pinned_offset,
            stream,
        )?;
        if prefill_meta_layout.needs_paged {
            self.prefill_b_upload_paged(
                prefill_seq,
                prefill_tokens.len(),
                prefill_chunk_start,
                p_rows,
                prefill_meta_layout.meta_base,
                prefill_meta_layout.slot_offset,
                &kv_cache,
                stream,
            )?;
        }
        let decode_metadata = self.qwen4_cross_seq_upload_metadata(
            frame,
            decode_seqs,
            &unique_leaders,
            staging,
            block_size,
            stream,
        )?;

        let (block_table_dev, seq_len_dev) = if prefill_meta_layout.needs_paged {
            let page_meta = prefill_seq
                .chunked_prefill_meta
                .as_ref()
                .context("mixed prefill paged metadata missing")?;
            (page_meta.block_table, page_meta.seq_len)
        } else {
            (DevicePtr::NULL, DevicePtr::NULL)
        };
        let (pos_h_dev, pos_w_dev) = if prefill_meta_layout.use_mrope {
            (
                prefill_meta_layout
                    .meta_base
                    .offset(prefill_meta_layout.pos_stream_bytes),
                prefill_meta_layout
                    .meta_base
                    .offset(prefill_meta_layout.pos_stream_bytes * 2),
            )
        } else {
            (prefill_meta_layout.meta_base, prefill_meta_layout.meta_base)
        };
        let prefill_attn_meta = AttnMetadataDev {
            qwen4_qsa_required: prefill_seq.qwen4_qsa_required,
            positions: prefill_meta_layout.meta_base,
            positions_h: pos_h_dev,
            positions_w: pos_w_dev,
            slot: prefill_meta_layout
                .meta_base
                .offset(prefill_meta_layout.slot_offset),
            seq_len: seq_len_dev,
            block_table: block_table_dev,
            max_blocks_per_seq: prefill_seq.block_table.len() as u32,
            num_seqs: 1,
        };
        let prefill_ctx = self.qwen4_cross_seq_ctx(Some(prefill_attn_meta));
        let decode_ctx = self.qwen4_cross_seq_ctx(None);
        let mixed_ctx = self.qwen4_cross_seq_ctx(Some(prefill_attn_meta));

        if crate::layers::qwen4_hyper::pending::discard_stale() {
            tracing::warn!("dropped stale pending HC injection before mixed prefill+decode round");
        }
        if prefill_chunk_start == 0 && crate::layers::moe::qwen4_moe_status::discard_stale() {
            tracing::warn!(
                "dropped stale deferred compact-MoE status words before mixed prefill+decode round"
            );
        }
        let prefill_receipt =
            crate::model::qwen4_prefill_engagement::begin(&self.config, total_rows)?;

        // ── Layers 0..num_hidden_layers: Split Mixer → Unified MLP/MoE ──
        let mut ssm_layer_idx = 0usize;
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            #[cfg(all(feature = "cuda", target_os = "linux"))]
            if layer_idx == 1
                && let Some(ple) = self.qwen4_ple.as_ref()
            {
                crate::layers::qwen4_hyper::pending::flush(self.gpu.as_ref(), stream)?;
                let prior = &prefill_tokens[..prefill_chunk_start];
                let current = &prefill_tokens[prefill_chunk_start..prefill_end];
                if ple_prefill_batch {
                    ple.forward_prefill(
                        current,
                        prior,
                        hidden,
                        prefill_seq.slot_idx,
                        prefill_chunk_start == 0,
                        self.gpu.as_ref(),
                        stream,
                        prefill_ple_prefetch.take(),
                    )?;
                } else {
                    let mut history = prior.to_vec();
                    for (row, &token) in current.iter().enumerate() {
                        ple.forward_token(
                            token,
                            &history,
                            hidden.offset(row * row_bytes),
                            prefill_seq.slot_idx,
                            prefill_chunk_start == 0 && row == 0,
                            self.gpu.as_ref(),
                            stream,
                        )?;
                        history.push(token);
                    }
                }

                let decode_pre_read = if let Some(ticket) = decode_ple_warm_ticket.take() {
                    Some(ticket.wait()?)
                } else {
                    None
                };
                if crate::layers::qwen4_ple::ple_verify_batch_selected() {
                    let ple_seqs: Vec<(&[u32], &[u32], usize)> = decode_seqs
                        .iter()
                        .zip(decode_tokens_per_seq)
                        .map(|(seq, window)| {
                            (window.as_slice(), seq.tokens.as_slice(), seq.slot_idx)
                        })
                        .collect();
                    ple.forward_cross_seq_verify(
                        &ple_seqs,
                        decode_pre_read,
                        hidden_decode,
                        row_bytes,
                        self.gpu.as_ref(),
                        stream,
                    )?;
                } else {
                    for (s, seq) in decode_seqs.iter().enumerate() {
                        self.qwen4_verify_ple(
                            &decode_tokens_per_seq[s],
                            seq,
                            hidden_decode.offset(frame.row_base(s) * row_bytes),
                            row_bytes,
                            stream,
                        )?;
                    }
                }
            }

            let linear = self.config.layer_type(layer_idx) == LayerType::LinearAttention;

            // Sub-phase A1: Prefill [0..P) mixer only (MoE deferred via ExpertMajorSliceGuard).
            crate::layers::qwen4_hyper::pending::flush(self.gpu.as_ref(), stream)?;
            {
                let _slice_guard = ExpertMajorSliceGuard::enter(max_prefill_chunk_rows);
                layer
                    .prefill(
                        hidden,
                        residual,
                        p_rows,
                        prefill_seq.layer_states[layer_idx].as_mut(),
                        &mut kv_cache,
                        prefill_chunk_start,
                        &mut prefill_seq.block_table,
                        &mut prefill_seq.disk_block_ids,
                        &mut prefill_seq.disk_last_offloaded_per_layer,
                        0,
                        &prefill_ctx,
                        stream,
                    )
                    .with_context(|| format!("mixed prefill mixer failed at layer {layer_idx}"))?;
            }
            crate::layers::qwen4_hyper::pending::flush(self.gpu.as_ref(), stream)?;

            // Sub-phase A2: Decode [P..P+D) mixer only.
            let mut views: Vec<Qwen4CrossSeqRows<'_>> = decode_seqs
                .iter_mut()
                .zip(decode_metadata.iter())
                .map(|(seq, &meta)| {
                    let slot_idx = seq.slot_idx;
                    let seq_len = seq.seq_len;
                    let (h_intermediate, conv_intermediate) = if linear {
                        (
                            self.ssm_pool.h_intermediate(ssm_layer_idx, slot_idx, 0),
                            self.ssm_pool.conv_intermediate(ssm_layer_idx, slot_idx, 0),
                        )
                    } else {
                        (DevicePtr::NULL, DevicePtr::NULL)
                    };
                    let SequenceState {
                        layer_states,
                        block_table,
                        disk_block_ids,
                        disk_last_offloaded_per_layer,
                        ..
                    } = &mut **seq;
                    Qwen4CrossSeqRows {
                        state: layer_states[layer_idx].as_mut(),
                        seq_len,
                        block_table,
                        disk_block_ids,
                        disk_last_offloaded_per_layer,
                        h_intermediate,
                        conv_intermediate,
                        h_intermediate_stride: self.ssm_pool.h_intermediate_stride(),
                        conv_intermediate_stride: self.ssm_pool.conv_bytes,
                        metadata: meta,
                        slot_idx,
                    }
                })
                .collect();
            layer
                .decode_qwen4_cross_seq_mixer_only(
                    hidden_decode,
                    residual_decode,
                    frame,
                    &mut views,
                    &mut kv_cache,
                    staging,
                    &decode_ctx,
                    stream,
                )
                .with_context(|| format!("mixed decode mixer failed at layer {layer_idx}"))?;

            // Sub-phase B: ONE unified MLP/MoE pass over ALL N = P + D rows (0 extra weight bytes).
            {
                let _mixed_moe_guard = MixedMoeRowsGuard::enter(p_rows, d_rows);
                layer
                    .qwen4_prefill_finish_moe(hidden, residual, total_rows, &mixed_ctx, stream)
                    .with_context(|| format!("mixed unified MoE failed at layer {layer_idx}"))?;
            }
            crate::layers::qwen4_hyper::pending::flush(self.gpu.as_ref(), stream)?;

            if linear {
                ssm_layer_idx += 1;
            }
        }

        crate::layers::qwen4_hyper::pending::flush(self.gpu.as_ref(), stream)?;
        crate::layers::moe::qwen4_moe_status::flush(self.gpu.as_ref(), stream)?;
        if let Some(receipt) = prefill_receipt {
            receipt.finish()?;
        }

        // Capture MTP last-K prefill hidden rows BEFORE relocating decode hidden rows.
        let window_start = prefill_tokens.len().saturating_sub(self.mtp_lastk_capacity);
        if is_last_prefill_chunk || prefill_end > window_start {
            self.mtp_lastk_capture_chunk(
                prefill_seq,
                prefill_chunk_start,
                p_rows,
                p_rows,
                prefill_tokens.len(),
                stream,
            )?;
        }

        prefill_seq
            .tokens
            .extend_from_slice(&prefill_tokens[prefill_chunk_start..prefill_end]);
        prefill_seq.seq_len = prefill_end;

        // ── Split Finalization ──
        // 1) Prefill row P - 1 → buffers.logits()[D] (before decode overwrites logits[0..D)).
        let vocab = self.verify_lmhead_vocab();
        let prefill_logits = if is_last_prefill_chunk {
            let last_hidden = hidden.offset((p_rows - 1) * row_bytes);
            let last_residual = residual.offset((p_rows - 1) * row_bytes);
            let normed = self.buffers.norm_output();
            let final_input = self
                .qwen4_final_hidden_nvfp4(last_hidden, last_residual, stream)?
                .unwrap_or(last_hidden);
            self.final_norm_rows(final_input, normed, 1, stream)?;
            let src_logits = self.lm_head(normed, stream)?;
            let dst_logits = self
                .buffers
                .logits()
                .offset(layout.prefill_logits_offset_bytes);
            self.gpu
                .copy_d2d_async(src_logits, dst_logits, layout.prefill_logits_bytes, stream)?;
            dst_logits
        } else {
            self.prefill_b_save_checkpoint(
                prefill_tokens,
                prefill_seq,
                &mut kv_cache,
                prefill_chunk_start,
                p_rows,
                stream,
            )?;
            DevicePtr::NULL
        };
        drop(kv_cache);

        // 2) Decode rows [P..P+D) → buffers.logits()[0..D) + argmax + MTP hidden handoff.
        let final_mixer = self
            .qwen4_final_mixer
            .as_ref()
            .context("mixed verify is missing the final hyperconnection mixer")?;
        let normed_decode = final_mixer.prepare_batched(
            hidden_decode,
            residual_decode,
            d_rows,
            &self.buffers,
            self.gpu.as_ref(),
            self.config.rms_norm_eps as f32,
            stream,
        )?;
        self.final_norm_rows(normed_decode, normed_decode, d_rows as u32, stream)?;
        self.lm_head_qwen4_rows(normed_decode, d_rows as u32, stream)?;

        let argmax_out = self.buffers.scratch();
        if super::verify_d::verify_argmax_rows_selected() && self.argmax_rows_kernel.0 != 0 {
            ops::argmax_bf16_rows(
                self.gpu.as_ref(),
                self.argmax_rows_kernel,
                self.buffers.logits(),
                argmax_out,
                vocab,
                vocab,
                d_rows as u32,
                stream,
            )?;
        } else {
            for row in 0..d_rows {
                ops::argmax_bf16(
                    self.gpu.as_ref(),
                    self.argmax_kernel,
                    self.buffers.logits().offset(row * vocab as usize * 2),
                    argmax_out.offset(row * 4),
                    vocab,
                    stream,
                )?;
            }
        }
        // Relocate raw pre-final-norm decode hidden states `[P..P+D)` down to
        // `[0..D)` so `save_hidden_for_mtp_sequence_in_frame(s * k, ..)` sees
        // the exact same row layout as standalone `decode_verify_qwen4_batched`.
        copy_hidden_rows_down(self.gpu.as_ref(), hidden, p_rows, d_rows, row_bytes, stream)?;

        let mut host_argmax = vec![0u8; d_rows * 4];
        self.gpu
            .copy_d2h_on_stream(argmax_out, &mut host_argmax, stream)?;
        let flat_argmax: Vec<u32> = host_argmax
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();

        for (seq, window) in decode_seqs.iter_mut().zip(decode_tokens_per_seq) {
            seq.tokens.extend_from_slice(window);
            seq.seq_len += k;
        }

        let decode_verified = (0..frame.seqs())
            .map(|s| {
                let base = frame.row_base(s);
                flat_argmax[base..base + k].to_vec()
            })
            .collect();

        Ok(Qwen4MixedRoundOutput {
            decode_verified,
            decode_logits: self.buffers.logits(),
            prefill_logits,
            prefill_rows: p_rows,
            decode_rows: d_rows,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{
        MixedLayoutParams, MixedPiggybackLayout, MixedPiggybackMode, plan_hidden_rows_shift_down,
    };
    use crate::layers::qwen4_prefill_moe::{
        MixedMoeRowsGuard, active_mixed_moe_rows, chunk_window_for_selection, max_chunk_rows,
    };
    use spark_runtime::gpu::DevicePtr;

    #[test]
    fn mixed_piggyback_policy_parsing_is_strict_and_fail_closed() {
        assert_eq!(
            MixedPiggybackMode::parse(None).unwrap(),
            MixedPiggybackMode::ExactStandaloneOnly
        );
        assert!(!MixedPiggybackMode::ExactStandaloneOnly.allows_piggyback());

        for val in ["1", "auto", "AUTO"] {
            let mode = MixedPiggybackMode::parse(Some(val)).unwrap();
            assert_eq!(mode, MixedPiggybackMode::Auto, "val={val}");
            assert!(mode.allows_piggyback());
        }
        for val in [
            "exact",
            "EXACT",
            "standalone",
            "STANDALONE",
            "exact_standalone_only",
        ] {
            let mode = MixedPiggybackMode::parse(Some(val)).unwrap();
            assert_eq!(mode, MixedPiggybackMode::ExactStandaloneOnly, "val={val}");
            assert!(!mode.allows_piggyback());
        }
        for val in ["0", "off", "OFF", "disabled", "DISABLED", "false", "FALSE"] {
            let mode = MixedPiggybackMode::parse(Some(val)).unwrap();
            assert_eq!(mode, MixedPiggybackMode::Disabled, "val={val}");
            assert!(!mode.allows_piggyback());
        }
        for invalid in ["", " ", "2", "true", "yes", " 1", "1 ", "Auto", "on"] {
            assert!(
                MixedPiggybackMode::parse(Some(invalid)).is_err(),
                "invalid spelling {invalid:?} must fail closed"
            );
        }
    }

    #[test]
    fn row_layout_and_byte_offsets_are_non_aliasing_across_p_and_d_grid() {
        const HIDDEN: usize = 2560;
        const RESIDUAL: usize = 10_240;
        const TOP_K: usize = 10;
        const MAX_BLOCKS: usize = 4096;
        const VOCAB: usize = 151_936;

        let hidden_base = DevicePtr(0x1_0000_0000);
        let residual_base = DevicePtr(0x2_0000_0000);
        let scratch_base = DevicePtr(0x3_0000_0000);
        let decode_attn_meta_dev = DevicePtr(0x4_0000_0000);
        let qsa_verify_meta_dev = DevicePtr(0x5_0000_0000);
        let logits_base = DevicePtr(0x6_0000_0000);

        for p in [512usize, 2048, 4096] {
            for d in [5usize, 40, 80, 128] {
                let total = p + d;
                let max_batch_tokens = 4096 + 128; // 4224
                let row_bytes = RESIDUAL * 2;
                let arena_hidden_bytes = max_batch_tokens * row_bytes;
                let arena_scratch_bytes =
                    max_batch_tokens * TOP_K * 8 + max_batch_tokens * 32 + 65536;
                let arena_logits_bytes = max_batch_tokens * VOCAB * 2;

                for is_last in [false, true] {
                    let layout = MixedPiggybackLayout::compute(MixedLayoutParams {
                        prefill_rows: p,
                        decode_rows: d,
                        max_prefill_chunk_rows: 4096,
                        max_batch_tokens,
                        hidden_size: HIDDEN,
                        residual_width: RESIDUAL,
                        top_k: TOP_K,
                        use_mrope: false,
                        max_blocks_per_seq: MAX_BLOCKS,
                        vocab_size: VOCAB,
                        is_last_prefill_chunk: is_last,
                        arena_hidden_bytes,
                        arena_residual_bytes: arena_hidden_bytes,
                        arena_scratch_bytes,
                        arena_logits_bytes,
                    })
                    .unwrap();

                    assert_eq!(layout.total_rows, total);
                    assert_eq!(layout.hidden_decode_offset_bytes, p * row_bytes);
                    assert_eq!(layout.hidden_total_bytes, total * row_bytes);
                    assert_eq!(layout.moe_routing_scratch_bytes, 2 * total * TOP_K * 4);
                    assert!(layout.prefill_meta_offset_bytes >= layout.moe_routing_scratch_bytes);
                    assert!(layout.prefill_meta_offset_bytes.is_multiple_of(8));
                    assert_eq!(layout.decode_logits_bytes, d * VOCAB * 2);
                    assert_eq!(layout.prefill_logits_offset_bytes, d * VOCAB * 2);
                    assert_eq!(
                        layout.prefill_logits_bytes,
                        if is_last { VOCAB * 2 } else { 0 }
                    );

                    layout
                        .verify_device_non_aliasing(
                            hidden_base,
                            residual_base,
                            scratch_base,
                            arena_scratch_bytes,
                            decode_attn_meta_dev,
                            qsa_verify_meta_dev,
                            2048,
                            logits_base,
                        )
                        .unwrap();

                    // Verify aliasing detection fails closed if decode metadata sits in scratch.
                    let aliased_meta = scratch_base.offset(crate::layer::STEP_META_SCRATCH_OFFSET);
                    assert!(
                        layout
                            .verify_device_non_aliasing(
                                hidden_base,
                                residual_base,
                                scratch_base,
                                arena_scratch_bytes,
                                aliased_meta,
                                qsa_verify_meta_dev,
                                2048,
                                logits_base,
                            )
                            .is_err()
                    );
                    // Also fail closed if QSA metadata aliases scratch.
                    assert!(
                        layout
                            .verify_device_non_aliasing(
                                hidden_base,
                                residual_base,
                                scratch_base,
                                arena_scratch_bytes,
                                decode_attn_meta_dev,
                                scratch_base,
                                2048,
                                logits_base,
                            )
                            .is_err()
                    );
                }
            }
        }
    }

    #[test]
    fn mixed_moe_rows_guard_admission_bounds_cover_p_times_d_matrix() {
        assert_eq!(active_mixed_moe_rows(), None);
        let max_p = max_chunk_rows();

        for d in [5usize, 40, 80, 128] {
            for p in [512usize.min(max_p), max_p] {
                let total = p + d;
                let _guard = MixedMoeRowsGuard::enter(p, d);
                assert!(
                    chunk_window_for_selection(total, 0, false),
                    "P={p}, D={d}, total={total} must be admitted under MixedMoeRowsGuard"
                );
                assert!(
                    !chunk_window_for_selection(total + 1, 0, false),
                    "mismatched row count must stay fail-closed"
                );
            }
            // P > max_p fails closed even under the guard.
            let _oversized = MixedMoeRowsGuard::enter(max_p + 1, d);
            assert!(
                !chunk_window_for_selection(max_p + 1 + d, 0, false),
                "P > max_chunk_rows must remain rejected"
            );
        }
        assert_eq!(active_mixed_moe_rows(), None);
    }

    #[test]
    fn hidden_shift_down_plan_is_exact_and_strictly_non_overlapping() {
        // P >= D: single contiguous D2D copy.
        assert_eq!(plan_hidden_rows_shift_down(512, 128), vec![(512, 0, 128)]);
        assert_eq!(plan_hidden_rows_shift_down(4096, 40), vec![(4096, 0, 40)]);
        // P < D: ascending chunks of size P, each strictly non-overlapping.
        let plan = plan_hidden_rows_shift_down(2, 5);
        assert_eq!(plan, vec![(2, 0, 2), (4, 2, 2), (6, 4, 1)]);
        let mut buf: Vec<usize> = (0..7).collect();
        for (src, dst, count) in plan {
            assert!(
                dst + count <= src,
                "source and destination must not overlap"
            );
            for i in 0..count {
                buf[dst + i] = buf[src + i];
            }
        }
        assert_eq!(&buf[..5], &[2, 3, 4, 5, 6]);
    }
}
