// SPDX-License-Identifier: AGPL-3.0-only

//! Complete Qwen4 native C2-C256 (`QWEN4_NATIVE_MAX_ROWS`) decode frame.
//!
//! Default C2+ route for Qwen4 (`crate::qwen4_batch_policy`; `=0` opts out),
//! still without a GPU parity receipt. It executes one active-row model
//! traversal, with no dummy padding and no C1 loop: batched embedding, per-row
//! metadata, native attention or recurrent state kernels for every layer, the
//! optional slot-indexed PLE boundary, one final hyperconnection collapse, and
//! one row-major LM-head call.

use std::collections::HashSet;

use anyhow::{Context, Result};
use atlas_core::config::LayerType;
use spark_runtime::gpu::DevicePtr;
use spark_runtime::kv_cache::KvCacheDtype;

use super::super::block_mgmt::{ensure_blocks_through_decode, extract_layer_refs};
use super::super::types::TransformerModel;
use crate::layer::ForwardContext;
use crate::layers::qwen4_ple::PleActiveDecodeRow;
use crate::layers::qwen4_qsa::native_batch::{self as qsa_native, NativeAttnRoute, NativeQsaMode};
use crate::traits::{Model, SequenceState};

const NATIVE_BATCH_ENV: &str = "ATLAS_QWEN4_NATIVE_BATCH";
const NATIVE_BATCH_RECEIPT_ENV: &str = "ATLAS_QWEN4_NATIVE_BATCH_RECEIPT";
const QSA_BOUNDARY: usize = 2_048;

const fn native_target_width(rows: usize) -> bool {
    rows >= 2 && rows <= crate::layers::QWEN4_NATIVE_MAX_ROWS
}

pub(super) const fn qwen4_native_batch_runtime_eligible(profile: bool) -> bool {
    !profile
}

/// Enable qualification-only per-frame route and ownership receipts.
///
/// Keep this separate from the native selector: a serialized selector-off
/// control needs the same receipt surface as the native candidate. Invalid
/// values fail closed instead of silently disabling the evidence required by
/// the direct-binary qualifier.
pub(super) fn qwen4_native_batch_receipt_enabled() -> Result<bool> {
    match std::env::var(NATIVE_BATCH_RECEIPT_ENV) {
        Err(std::env::VarError::NotPresent) => Ok(false),
        Ok(value) if value == "0" => Ok(false),
        Ok(value) if value == "1" => Ok(true),
        Ok(_) => anyhow::bail!("{NATIVE_BATCH_RECEIPT_ENV} must be exact value 0 or 1"),
        Err(std::env::VarError::NotUnicode(_)) => {
            anyhow::bail!("{NATIVE_BATCH_RECEIPT_ENV} must be valid UTF-8")
        }
    }
}

/// Whether native batching is *enabled*: unset (the default) or `1`. Whether
/// it is also *available* is the process plan (`Qwen4BatchPlan::native`,
/// `decode_qwen4_route.rs`); `0` is the explicit serial opt-out.
fn parse_native_batch_selector(raw: Option<&str>) -> std::result::Result<bool, &'static str> {
    crate::qwen4_batch_policy::NativeBatchSelector::parse(raw)
        .map(crate::qwen4_batch_policy::NativeBatchSelector::enabled)
}

fn select_native_target_width(
    selected: bool,
    rows: usize,
) -> std::result::Result<bool, &'static str> {
    if !selected || rows <= 1 {
        return Ok(false);
    }
    if native_target_width(rows) {
        return Ok(true);
    }
    Err("native Qwen4 batching supports active widths C2-C128 only")
}

fn validate_active_rows(
    rows: usize,
    seq_lens: &[usize],
    slot_ids: &[usize],
    qsa_required: &[bool],
) -> std::result::Result<(), &'static str> {
    if !native_target_width(rows) {
        return Err("native Qwen4 decode admits only C2-C256");
    }
    if seq_lens.len() != rows || slot_ids.len() != rows || qsa_required.len() != rows {
        return Err("native Qwen4 decode needs one metadata row per active request");
    }
    if qsa_required.iter().any(|&required| required) {
        return Err("native Qwen4 decode cannot maintain the QSA side index");
    }
    for &seq_len in seq_lens {
        let next = seq_len
            .checked_add(1)
            .ok_or("native Qwen4 decode sequence length overflow")?;
        if next >= QSA_BOUNDARY {
            return Err("native Qwen4 decode cannot reach the sparse-QSA boundary");
        }
    }
    let mut unique = HashSet::with_capacity(rows);
    if !slot_ids.iter().all(|&slot| unique.insert(slot)) {
        return Err("native Qwen4 decode requires one distinct state slot per row");
    }
    Ok(())
}

/// Validate every host-side KV fact needed by the native frame and return the
/// exact number of blocks that must be allocated for this decode step.
///
/// Native rows are independent requests, so sharing any physical block is not
/// an admissible prefix-sharing topology.  Checking this before the first
/// allocation also keeps a later row from discovering malformed ownership
/// after earlier rows have already grown their block tables.
fn validate_kv_rows_before_allocation(
    seq_lens: &[usize],
    block_tables: &[Vec<u32>],
    block_size: usize,
    max_blocks_per_seq: usize,
    num_blocks: usize,
) -> std::result::Result<usize, &'static str> {
    if block_size == 0 {
        return Err("native Qwen4 KV block size is zero");
    }
    if max_blocks_per_seq == 0 {
        return Err("native Qwen4 KV metadata row has zero block capacity");
    }
    if seq_lens.len() != block_tables.len() {
        return Err("native Qwen4 KV rows do not match active sequence rows");
    }

    let mut physical_blocks = HashSet::new();
    let mut blocks_needed = 0usize;
    for (&seq_len, table) in seq_lens.iter().zip(block_tables) {
        if table.len() > max_blocks_per_seq {
            return Err("native Qwen4 KV block table exceeds its metadata row stride");
        }
        let historical_blocks = if seq_len == 0 {
            0
        } else {
            (seq_len - 1) / block_size + 1
        };
        if table.len() < historical_blocks {
            return Err("native Qwen4 KV block table does not cover existing tokens");
        }
        let decode_block = seq_len / block_size;
        if decode_block >= max_blocks_per_seq {
            return Err("native Qwen4 decode exceeds the KV metadata row capacity");
        }
        let decode_blocks = decode_block
            .checked_add(1)
            .ok_or("native Qwen4 KV block extent overflow")?;
        blocks_needed = blocks_needed
            .checked_add(decode_blocks.saturating_sub(table.len()))
            .ok_or("native Qwen4 KV allocation extent overflow")?;

        for &physical in table {
            if physical as usize >= num_blocks {
                return Err("native Qwen4 KV block table contains an out-of-range block");
            }
            if !physical_blocks.insert(physical) {
                return Err("native Qwen4 independent rows alias a physical KV block");
            }
        }
    }
    Ok(blocks_needed)
}

/// Prove that every admitted table entry is a live, exclusively owned request
/// block. Active-row uniqueness alone is insufficient: a corrupted row could
/// otherwise target the reserved padding block or a block retained by an
/// inactive request.
fn validate_kv_block_ownership(
    block_tables: &[Vec<u32>],
    dummy_kv_block: u32,
    ref_count: impl Fn(u32) -> u32,
) -> std::result::Result<(), &'static str> {
    for &physical in block_tables.iter().flatten() {
        if physical == dummy_kv_block {
            return Err("native Qwen4 request row references the reserved dummy KV block");
        }
        if ref_count(physical) != 1 {
            return Err("native Qwen4 request row does not exclusively own its KV block");
        }
    }
    Ok(())
}

/// Prefix-cache-aware variant of [`validate_kv_rows_before_allocation`]:
/// read-only historical prefix blocks (`logical < seq_len / block_size`) may be
/// shared across concurrent requests, while writable decode blocks
/// (`logical >= seq_len / block_size`) must never alias each other or any
/// read-only historical block.
fn validate_kv_rows_before_allocation_prefix_cache(
    seq_lens: &[usize],
    block_tables: &[Vec<u32>],
    block_size: usize,
    max_blocks_per_seq: usize,
    num_blocks: usize,
) -> std::result::Result<usize, &'static str> {
    if block_size == 0 {
        return Err("native Qwen4 KV block size is zero");
    }
    if max_blocks_per_seq == 0 {
        return Err("native Qwen4 KV metadata row has zero block capacity");
    }
    if seq_lens.len() != block_tables.len() {
        return Err("native Qwen4 KV rows do not match active sequence rows");
    }

    let mut read_only_blocks = HashSet::new();
    let mut writable_blocks = HashSet::new();
    let mut blocks_needed = 0usize;
    for (&seq_len, table) in seq_lens.iter().zip(block_tables) {
        if table.len() > max_blocks_per_seq {
            return Err("native Qwen4 KV block table exceeds its metadata row stride");
        }
        let historical_blocks = if seq_len == 0 {
            0
        } else {
            (seq_len - 1) / block_size + 1
        };
        if table.len() < historical_blocks {
            return Err("native Qwen4 KV block table does not cover existing tokens");
        }
        let decode_block = seq_len / block_size;
        if decode_block >= max_blocks_per_seq {
            return Err("native Qwen4 decode exceeds the KV metadata row capacity");
        }
        let decode_blocks = decode_block
            .checked_add(1)
            .ok_or("native Qwen4 KV block extent overflow")?;
        blocks_needed = blocks_needed
            .checked_add(decode_blocks.saturating_sub(table.len()))
            .ok_or("native Qwen4 KV allocation extent overflow")?;

        for (logical, &physical) in table.iter().enumerate() {
            if physical as usize >= num_blocks {
                return Err("native Qwen4 KV block table contains an out-of-range block");
            }
            if logical < decode_block {
                read_only_blocks.insert(physical);
            } else if !writable_blocks.insert(physical) {
                return Err("native Qwen4 independent rows alias a physical KV block");
            }
        }
    }
    if writable_blocks
        .iter()
        .any(|physical| read_only_blocks.contains(physical))
    {
        return Err("native Qwen4 independent rows alias a physical KV block");
    }
    Ok(blocks_needed)
}

/// Prefix-cache-aware variant of [`validate_kv_block_ownership`]:
/// read-only historical prefix blocks (`logical < seq_len / block_size`) must
/// be live (`ref_count >= 1`, not dummy), while writable decode blocks
/// (`logical >= seq_len / block_size`) must remain exclusively owned
/// (`ref_count == 1`).
fn validate_kv_block_ownership_prefix_cache(
    seq_lens: &[usize],
    block_tables: &[Vec<u32>],
    block_size: usize,
    dummy_kv_block: u32,
    ref_count: impl Fn(u32) -> u32,
) -> std::result::Result<(), &'static str> {
    if block_size == 0 {
        return Err("native Qwen4 KV block size is zero");
    }
    if seq_lens.len() != block_tables.len() {
        return Err("native Qwen4 KV rows do not match active sequence rows");
    }
    for (&seq_len, table) in seq_lens.iter().zip(block_tables) {
        let decode_block = seq_len / block_size;
        for (logical, &physical) in table.iter().enumerate() {
            if physical == dummy_kv_block {
                return Err("native Qwen4 request row references the reserved dummy KV block");
            }
            let refs = ref_count(physical);
            if logical < decode_block {
                if refs == 0 {
                    return Err("native Qwen4 request row references an unowned prefix KV block");
                }
            } else if refs != 1 {
                return Err("native Qwen4 request row does not exclusively own its KV block");
            }
        }
    }
    Ok(())
}

/// The ordinary sequence allocator reserves one zero cursor per attention
/// layer even when high-speed swap is disabled. A zero-filled cursor vector
/// is therefore the canonical no-swap state; requiring an empty vector makes
/// every real Qwen4 sequence ineligible for native batching.
fn has_no_swap_lineage(
    disk_block_ids: &[u32],
    disk_last_offloaded_per_layer: &[u32],
    num_attention_layers: usize,
) -> bool {
    disk_block_ids.is_empty()
        && disk_last_offloaded_per_layer.len() == num_attention_layers
        && disk_last_offloaded_per_layer
            .iter()
            .all(|&cursor| cursor == 0)
}

pub(super) fn qwen4_native_batch_selected(
    plan: &crate::qwen4_batch_policy::Qwen4BatchPlan,
    rows: usize,
) -> Result<bool> {
    let raw = match std::env::var(NATIVE_BATCH_ENV) {
        Ok(value) => Some(value),
        Err(std::env::VarError::NotPresent) => None,
        Err(std::env::VarError::NotUnicode(_)) => {
            anyhow::bail!("{NATIVE_BATCH_ENV} must be valid UTF-8")
        }
    };
    // Effective native-on: enabled (unset or `1`) and available in this
    // process (`Qwen4BatchPlan::native`). `resolve_plan` has already refused
    // QSA rows on a serial plan, so the QSA mode resolves against this value.
    let selected =
        parse_native_batch_selector(raw.as_deref()).map_err(anyhow::Error::msg)? && plan.native();
    // Validate the QSA-row selector on the same fallible path, before the
    // infallible request-eligibility check reads it.
    let qsa_mode = qsa_native::native_batch_qsa_mode()?;
    qsa_native::resolve_native_batch_qsa_mode(selected, qsa_mode).map_err(anyhow::Error::msg)?;
    select_native_target_width(selected, rows).map_err(anyhow::Error::msg)
}

/// Requests that need QSA stay on the serialized path from their first
/// generated token so every attention layer maintains its side index, unless
/// `ATLAS_QWEN4_NATIVE_BATCH_QSA=1` admits them as per-row QSA rows. This
/// read-only check runs before native dispatch, making long-context fallback
/// safe rather than a recovery from partial native work.
pub(super) fn qwen4_native_batch_request_eligible(seqs: &[&mut SequenceState]) -> bool {
    // Every caller evaluates `qwen4_native_batch_selected` first, which
    // rejects an invalid QSA selector; an unreadable one here stays serial.
    let qsa_mode = qsa_native::native_batch_qsa_mode().unwrap_or(NativeQsaMode::Off);
    native_target_width(seqs.len())
        && seqs.iter().all(|seq| {
            (!seq.qwen4_qsa_required
                && seq
                    .seq_len
                    .checked_add(1)
                    .is_some_and(|next| next < QSA_BOUNDARY))
                || (seq.qwen4_qsa_required
                    && qsa_native::native_attn_route(qsa_mode, true, seq.seq_len).is_ok())
        })
}

/// Per-row QSA routes for a QSA-row frame, or `None` when the frame has no
/// row that keeps a side index (including every frame with the selector off,
/// which `validate_active_rows` then rejects exactly as before).
fn native_qsa_routes(
    mode: NativeQsaMode,
    qsa_required: &[bool],
    seq_lens: &[usize],
) -> std::result::Result<Option<Vec<NativeAttnRoute>>, &'static str> {
    if mode == NativeQsaMode::Off || !qsa_required.iter().any(|&required| required) {
        return Ok(None);
    }
    qsa_native::native_attn_routes(mode, qsa_required, seq_lens).map(Some)
}

impl TransformerModel {
    /// Whether this model admits native C2-C128 K=1 batching for post-prefill
    /// native-MTP bootstrap frames (retaining each row's post-layer hidden
    /// state in `qwen4_mtp_hidden_frames` at the end of the forward).
    fn qwen4_native_mtp_hidden_bootstrap_admitted(&self) -> bool {
        !self.self_speculative
            && !self.speculation_forced_disabled
            && self.qwen4_mtp_hidden_frames.is_some()
            && !crate::layers::qwen4_mtp_rows::recondition_engaged()
            && self.active_proposer().is_some_and(|proposer| {
                proposer.kind() == crate::speculative::DraftProposerKind::Qwen4NativeMtp
            })
    }

    /// Whether this model admits native C2-C128 K=1 batching for post-prefill
    /// DFlash bootstrap frames (capturing each row's tap hiddens via
    /// `try_dflash_capture_rows` and retiring row `row` into each sequence's
    /// `DflashProposerState` context ring).
    pub(super) fn qwen4_dflash_bootstrap_admitted(&self) -> bool {
        let max_native_rows = self
            .ssm_pool
            .max_slots
            .min(crate::layers::QWEN4_NATIVE_MAX_ROWS);
        !self.self_speculative
            && !self.speculation_forced_disabled
            && self.proposer_is_dflash()
            && self.dflash_hidden_save.is_some()
            && !self.dflash_capture_layers.is_empty()
            && self.dflash_hidden_save_rows >= max_native_rows
            && self.dflash_hidden_save_rows >= 2
    }

    /// Whether this model admits native C2-C128 K=1 batching for post-prefill
    /// speculative bootstrap frames (either native Qwen4 MTP or DFlash).
    ///
    /// Consulted by `decode_qwen4_capability.rs` so the cached per-width native
    /// capability gate admits both native-MTP and DFlash bootstrap cohorts.
    pub(super) fn qwen4_native_mtp_bootstrap_admitted(&self) -> bool {
        self.qwen4_native_mtp_hidden_bootstrap_admitted() || self.qwen4_dflash_bootstrap_admitted()
    }

    /// Run one complete native target forward for C2-C256 (`QWEN4_NATIVE_MAX_ROWS`) active requests.
    ///
    /// Every statically knowable admission check occurs before embedding, KV
    /// allocation, PLE I/O, or recurrent-state mutation. Operational GPU or
    /// storage errors can still fail the frame after submission; such a frame
    /// is never replayed through the serial decoder.
    pub(in crate::model) fn decode_qwen4_native_batch_dispatch(
        &self,
        tokens: &[u32],
        seqs: &mut [&mut SequenceState],
    ) -> Result<DevicePtr> {
        let rows = tokens.len();
        anyhow::ensure!(
            rows == seqs.len(),
            "native Qwen4 token and sequence row counts differ"
        );
        let receipt_enabled = qwen4_native_batch_receipt_enabled()?;
        anyhow::ensure!(
            tokens
                .iter()
                .all(|&token| (token as usize) < self.config.vocab_size),
            "native Qwen4 token ID is outside the embedding vocabulary"
        );
        anyhow::ensure!(
            qwen4_native_batch_runtime_eligible(self.profile),
            "profiled Qwen4 requests must use the generic profile-aware decoder"
        );
        let seq_lens: Vec<_> = seqs.iter().map(|seq| seq.seq_len).collect();
        let slot_ids: Vec<_> = seqs.iter().map(|seq| seq.slot_idx).collect();
        let qsa_required: Vec<_> = seqs.iter().map(|seq| seq.qwen4_qsa_required).collect();
        // `Some` only for a QSA-row frame (`ATLAS_QWEN4_NATIVE_BATCH_QSA=1`
        // with at least one QSA row); `None` keeps the dense lane unchanged.
        let qsa_mode = qsa_native::native_batch_qsa_mode()?;
        let qsa_routes =
            native_qsa_routes(qsa_mode, &qsa_required, &seq_lens).map_err(anyhow::Error::msg)?;
        let (policy_seq_lens, policy_qsa_required) = match &qsa_routes {
            None => (seq_lens.clone(), qsa_required.clone()),
            Some(routes) => (
                qsa_native::dense_policy_seq_lens(routes, &seq_lens).map_err(anyhow::Error::msg)?,
                vec![false; rows],
            ),
        };
        validate_active_rows(rows, &policy_seq_lens, &slot_ids, &policy_qsa_required)
            .map_err(anyhow::Error::msg)?;

        anyhow::ensure!(
            self.config.is_qwen4_exp()
                && self.config.hc_count == 4
                && self.config.hidden_size == 2_560
                && self.config.residual_width() == 10_240
                && !self.config.use_fp32_residual(),
            "native Qwen4 decode requires the canonical BF16 Flash-Next geometry"
        );
        let topology_ok = (self.config.tp_world_size == 1
            && self.config.ep_world_size == 1
            && ((self.config.pp_world_size == 1 && self.comm.is_none())
                || (self.config.pp_world_size >= 2 && self.comm.is_some())))
            || ((2..=4).contains(&self.config.tp_world_size)
                && (self.config.ep_world_size <= 1
                    || self.config.ep_world_size == self.config.tp_world_size)
                && self.config.pp_world_size == 1
                && self.comm.is_some());
        anyhow::ensure!(
            topology_ok,
            "native Qwen4 decode requires TP1..TP4 (+ matching EP) with PP1, or PP>=2 with TP1/EP1"
        );
        let target_only =
            !self.self_speculative && (!self.any_proposer() || self.speculation_forced_disabled);
        let native_mtp_bootstrap = self.qwen4_native_mtp_hidden_bootstrap_admitted();
        let dflash_bootstrap = self.qwen4_dflash_bootstrap_admitted();
        anyhow::ensure!(
            target_only || native_mtp_bootstrap || dflash_bootstrap,
            "native Qwen4 decode is target-only; disable speculative decoding"
        );
        anyhow::ensure!(
            !crate::layers::qwen3_ssm::ssm_h_fp16_enabled(),
            "native Qwen4 decode requires FP32 recurrent state; unset ATLAS_SSM_H_FP16"
        );
        let prefix_cache_active = self.prefix_cache.is_active();
        let prefix_cache_admitted = !prefix_cache_active
            || (self.kv_cache.lock().config().cache_blocks_per_seq.is_none()
                && seqs
                    .iter()
                    .all(|seq| seq.cached_prefix_tokens <= seq.seq_len));
        anyhow::ensure!(
            prefix_cache_admitted,
            "native Qwen4 decode does not admit prefix-cache ownership"
        );
        anyhow::ensure!(
            seqs.iter().all(|seq| {
                seq.rotary_positions.is_identity()
                    && seq.layer_states.len() == self.layers.len()
                    && seq.tokens.len() == seq.seq_len
                    && has_no_swap_lineage(
                        &seq.disk_block_ids,
                        &seq.disk_last_offloaded_per_layer,
                        self.config.num_attention_layers(),
                    )
            }),
            "native Qwen4 decode requires flat rows with complete state and no swap lineage"
        );
        anyhow::ensure!(
            self.layers.len() == self.config.num_hidden_layers
                && (0..self.layers.len()).all(|index| {
                    matches!(
                        self.config.layer_type(index),
                        LayerType::FullAttention | LayerType::LinearAttention
                    )
                }),
            "native Qwen4 decode found an unsupported or incomplete layer schedule"
        );
        if self.qwen4_ple.is_some() {
            anyhow::ensure!(
                self.layers.len() > 1 && self.config.ple_layer_ids.as_slice() == [2],
                "native Qwen4 PLE requires the canonical layer-2 injection boundary"
            );
        }
        if self.config.pp_rank == 0 {
            anyhow::ensure!(
                !self.use_fp32_logits
                    && self.verify_lmhead_vocab() as usize == self.config.vocab_size,
                "native Qwen4 decode requires full-vocabulary BF16 row-major logits"
            );
            anyhow::ensure!(
                self.lm_head_nvfp4.is_some(),
                "native Qwen4 decode requires an ordinary NVFP4 LM head"
            );
            anyhow::ensure!(
                !(super::super::impl_a3::lm_head_tc_enabled()
                    && self.lm_head_nvfp4_t.is_some()
                    && self.w4a16_gemm_t_m32_n64_kernel.0 != 0),
                "native Qwen4 decode requires the exact LM-head route; unset ATLAS_LM_HEAD_TC"
            );
            let lm_head_route = self
                .w4a16_exact_lm_head_kernels
                .route_for_rows(rows as u32)
                .context("native Qwen4 decode has no exact/K1 LM-head row tier")?;
            if matches!(
                lm_head_route,
                crate::layers::ops::ExactLmHeadRoute::SerialK1(_)
            ) {
                anyhow::ensure!(
                    crate::layers::ops::use_gemv_sw(self.gemv_sw, self.w4a16_gemv_sw_kernel)
                        || self.w4a16_gemv_kernel.0 != 0,
                    "native Qwen4 decode is missing the K1 LM-head fallback kernel"
                );
            }
            let final_mixer = self
                .qwen4_final_mixer
                .as_ref()
                .context("native Qwen4 decode is missing the final hyperconnection mixer")?;
            anyhow::ensure!(
                final_mixer.residual_width() == self.config.residual_width()
                    && final_mixer.inject.is_none(),
                "native Qwen4 final mixer geometry is not target-decode compatible"
            );
            final_mixer.preflight_qwen4_native_batched(rows, false)?;
        }

        let hyper_bytes = rows
            .checked_mul(self.config.residual_width())
            .and_then(|items| items.checked_mul(2))
            .context("native Qwen4 hyper-row extent overflow")?;
        let core_bytes = rows
            .checked_mul(self.config.hidden_size)
            .and_then(|items| items.checked_mul(2))
            .context("native Qwen4 core-row extent overflow")?;
        let logits_bytes = rows
            .checked_mul(self.config.vocab_size)
            .and_then(|items| items.checked_mul(2))
            .context("native Qwen4 logits extent overflow")?;
        let hyper_scratch = rows
            .checked_mul(self.config.hc_lowrank)
            .and_then(|elements| elements.checked_mul(2))
            .and_then(|down_bytes| {
                rows.checked_mul(self.config.hc_count)
                    .and_then(|elements| elements.checked_mul(2))
                    .and_then(|inject_bytes| down_bytes.checked_add(inject_bytes))
            })
            .context("native Qwen4 hyper scratch extent overflow")?;
        let arenas = self.buffers.sizes();
        let metadata_bytes = crate::layer::STEP_META_SCRATCH_OFFSET
            .checked_add(
                self.buffers
                    .decode_meta()
                    .meta_bytes(self.max_blocks_per_seq as usize),
            )
            .context("native Qwen4 metadata scratch extent overflow")?;
        anyhow::ensure!(
            rows <= self.buffers.decode_meta().rows()
                && rows <= self.buffers.max_batch_tokens()
                && hyper_bytes <= arenas.hidden_states
                && hyper_bytes <= arenas.residual
                && hyper_bytes <= arenas.qkv_output
                && core_bytes <= arenas.norm_output
                && core_bytes <= arenas.moe_output
                && logits_bytes <= arenas.logits
                && hyper_scratch <= arenas.ssm_ba
                && metadata_bytes <= arenas.scratch,
            "native Qwen4 active rows exceed the serving arenas"
        );
        self.preflight_embed_qwen4_batch(rows)?;
        if let Some(ple) = &self.qwen4_ple {
            ple.preflight_active_decode_rows(
                rows,
                self.config
                    .residual_width()
                    .checked_mul(2)
                    .context("native Qwen4 PLE row-stride overflow")?,
                &slot_ids,
            )?;
        }

        // Complete a read-only walk of every actual request-owned layer state
        // before KV allocation, embedding, metadata upload, or any GPU/state
        // mutation. A layer that cannot prove its exact C2-C256 execution route
        // fails here, before an earlier recurrent layer can be advanced.
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
        for (layer_index, layer) in self.layers.iter().enumerate() {
            let states: Vec<&(dyn crate::layer::LayerState + 'static)> = seqs
                .iter()
                .map(|seq| seq.layer_states[layer_index].as_ref())
                .collect();
            layer
                .preflight_qwen4_native_target(rows, &states, &preflight_ctx)
                .with_context(|| {
                    format!("native Qwen4 layer {layer_index} failed read-only preflight")
                })?;
        }

        if native_mtp_bootstrap {
            let (_, frames) = self.qwen4_mtp_hidden_frames.as_ref().ok_or_else(|| {
                anyhow::anyhow!("MTP bootstrap lacks sequence-owned target-hidden storage")
            })?;
            let locked_frames = frames.lock();
            for seq in seqs.iter() {
                self.ensure_verify_sequence_idle(seq)?;
                locked_frames
                    .slot_range(seq.slot_idx)
                    .map_err(anyhow::Error::msg)?;
            }
        }
        if dflash_bootstrap {
            anyhow::ensure!(
                rows <= self.dflash_hidden_save_rows,
                "DFlash bootstrap rows {rows} exceed dflash_hidden_save_rows {}",
                self.dflash_hidden_save_rows
            );
            for seq in seqs.iter() {
                self.ensure_verify_sequence_idle(seq)?;
                anyhow::ensure!(
                    seq.proposer_state.as_deref().is_some_and(|ps| {
                        ps.as_any()
                            .downcast_ref::<crate::layers::DflashProposerState>()
                            .is_some()
                    }),
                    "DFlash bootstrap sequence on slot {} lacks DflashProposerState",
                    seq.slot_idx
                );
            }
        }

        let stream = self.gpu.default_stream();
        for seq in seqs.iter() {
            self.flush_gdn_keep_late_slot(seq, false, stream)?;
        }
        let hidden = self.buffers.hidden_states();
        let residual = self.buffers.residual();
        let mut kv_cache = self.kv_cache.lock();
        anyhow::ensure!(
            kv_cache.config().cache_blocks_per_seq.is_none()
                && (0..kv_cache.num_layers()).all(|layer| {
                    matches!(
                        kv_cache.dtype_for_layer(layer),
                        KvCacheDtype::Bf16 | KvCacheDtype::Nvfp4
                    )
                }),
            "native Qwen4 decode requires resident BF16 KV (or NVFP4) without high-speed swap"
        );

        let block_size = kv_cache.block_size();
        let initial_block_tables: Vec<Vec<u32>> =
            seqs.iter().map(|seq| seq.block_table.clone()).collect();
        let blocks_needed = if prefix_cache_active {
            validate_kv_rows_before_allocation_prefix_cache(
                &seq_lens,
                &initial_block_tables,
                block_size,
                self.max_blocks_per_seq as usize,
                kv_cache.num_blocks(),
            )
            .map_err(anyhow::Error::msg)?
        } else {
            validate_kv_rows_before_allocation(
                &seq_lens,
                &initial_block_tables,
                block_size,
                self.max_blocks_per_seq as usize,
                kv_cache.num_blocks(),
            )
            .map_err(anyhow::Error::msg)?
        };
        if prefix_cache_active {
            validate_kv_block_ownership_prefix_cache(
                &seq_lens,
                &initial_block_tables,
                block_size,
                self.dummy_kv_block,
                |block| kv_cache.ref_count(block),
            )
            .map_err(anyhow::Error::msg)?;
        } else {
            validate_kv_block_ownership(&initial_block_tables, self.dummy_kv_block, |block| {
                kv_cache.ref_count(block)
            })
            .map_err(anyhow::Error::msg)?;
        }
        if prefix_cache_active && blocks_needed > kv_cache.num_free_blocks() {
            let deficit = blocks_needed - kv_cache.num_free_blocks();
            let evicted = self.prefix_cache.evict(deficit);
            super::super::block_mgmt::apply_evicted_blocks(evicted, &mut kv_cache);
        }
        anyhow::ensure!(
            blocks_needed <= kv_cache.num_free_blocks(),
            "native Qwen4 decode needs {blocks_needed} KV blocks but only {} are free",
            kv_cache.num_free_blocks()
        );
        if let Some(routes) = &qsa_routes {
            for (layer_index, layer) in self.layers.iter().enumerate() {
                if matches!(
                    self.config.layer_type(layer_index),
                    LayerType::FullAttention
                ) {
                    layer
                        .preflight_qwen4_native_target_qsa(routes, &seq_lens, &kv_cache)
                        .with_context(|| {
                            format!("native Qwen4 layer {layer_index} failed QSA preflight")
                        })?;
                }
            }
        }

        // Pre-effect admission is complete. Reconcile any pending single-seq
        // DFlash capture debt before the layer loop overwrites `dflash_hidden_save`,
        // then allocate one real KV slot per active request and upload metadata.
        if dflash_bootstrap {
            for seq in seqs.iter_mut() {
                self.reconcile_dflash_context_before_bootstrap_dispatch(seq, stream)?;
            }
        }
        for seq in seqs.iter_mut() {
            let block_index = seq.seq_len / block_size;
            ensure_blocks_through_decode(
                seq,
                block_index,
                &mut kv_cache,
                self.prefix_cache.as_ref(),
                self.gpu.as_ref(),
                stream,
            )?;
        }
        if self
            .comm
            .as_ref()
            .is_some_and(|c| c.world_size() > 1 && c.rank() == 0)
        {
            self.ep_broadcast_u32(crate::traits::EP_CMD_DECODE_QWEN4_BATCH)?;
            self.ep_broadcast_u32(rows as u32)?;
            let mut packed_tokens = Vec::with_capacity(rows * 3);
            for seq in seqs.iter() {
                let tail_len = seq.tokens.len().min(2);
                let meta = ((seq.slot_idx as u32) & 0xFFFF)
                    | (u32::from(seq.qwen4_qsa_required) << 16)
                    | ((tail_len as u32) << 24);
                self.ep_broadcast_u32(meta)?;
                self.ep_broadcast_u32(seq.seq_len as u32)?;
                if tail_len > 0 {
                    packed_tokens.extend_from_slice(&seq.tokens[seq.tokens.len() - tail_len..]);
                }
            }
            packed_tokens.extend_from_slice(tokens);
            let _ = self.ep_broadcast_tokens(&packed_tokens)?;
        }
        if self.config.pp_rank == 0 {
            self.embed_qwen4_batch(tokens, hidden, stream)?;
        }
        let metadata = self.upload_batch_metadata_fixed(seqs, rows, &mut kv_cache, stream)?;
        debug_assert_eq!(metadata.num_seqs as usize, rows);
        let block_tables: Vec<Vec<u32>> = seqs.iter().map(|seq| seq.block_table.clone()).collect();
        let ctx = ForwardContext {
            buffers: &self.buffers,
            gpu: self.gpu.as_ref(),
            config: &self.config,
            attn_metadata: Some(metadata),
            profile: false,
            comm: self.comm.as_deref(),
            graph_capture: false,
            ddtree_parent_ids_dev: None,
            tree_aware_attn: None,
            ssm_multi_seq_ptr_table_override: None,
            self_spec_sparse_draft: None,
            ffn_defer: None,
        };

        let tc_dense_active =
            crate::layers::qwen4_hyper::batched_verify_tc_dense_active(rows, false);
        let _tc_dense_guard =
            crate::layers::qwen4_hyper::BatchedVerifyTcDenseGuard::enter(tc_dense_active);
        let mut all_layer_states: Vec<Vec<Box<dyn crate::layer::LayerState>>> = seqs
            .iter_mut()
            .map(|seq| std::mem::take(&mut seq.layer_states))
            .collect();
        let forward = (|| -> Result<DevicePtr> {
            self.pp_stage_enter(hidden, rows, stream)?;
            for (layer_index, layer) in self.layers.iter().enumerate() {
                if layer_index == 1
                    && let Some(ple) = &self.qwen4_ple
                {
                    let ple_rows: Vec<_> = tokens
                        .iter()
                        .zip(seqs.iter())
                        .map(|(&current_token, seq)| PleActiveDecodeRow {
                            current_token,
                            prior_tokens: &seq.tokens,
                            slot_idx: seq.slot_idx,
                            reset_state: seq.seq_len == 0,
                        })
                        .collect();
                    ple.forward_active_decode_rows(
                        &ple_rows,
                        hidden,
                        self.config.residual_width() * 2,
                        self.gpu.as_ref(),
                        stream,
                    )?;
                }
                match self.config.layer_type(layer_index) {
                    LayerType::FullAttention => {
                        if let Some(routes) = &qsa_routes {
                            layer.decode_qwen4_native_target_attention_qsa(
                                hidden,
                                residual,
                                rows,
                                &mut kv_cache,
                                &seq_lens,
                                &block_tables,
                                routes,
                                &ctx,
                                stream,
                            )?;
                        } else {
                            layer.decode_qwen4_native_target_attention(
                                hidden,
                                residual,
                                rows,
                                &mut kv_cache,
                                &seq_lens,
                                &block_tables,
                                &ctx,
                                stream,
                            )?;
                        }
                        layer
                            .decode_qwen4_native_target_mlp(hidden, residual, rows, &ctx, stream)?;
                    }
                    LayerType::LinearAttention => {
                        let mut states = extract_layer_refs(&mut all_layer_states, layer_index);
                        layer.decode_qwen4_native_target_ssm(
                            hidden,
                            residual,
                            rows,
                            &mut states,
                            &ctx,
                            stream,
                        )?;
                    }
                    LayerType::Moe => unreachable!("layer schedule was checked before mutation"),
                }
                if dflash_bootstrap && !self.dflash_capture_layers.is_empty() {
                    self.try_dflash_capture_rows(layer_index, rows, stream)?;
                }
            }
            self.pp_stage_exit(hidden, rows, stream)?;
            if self.config.pp_rank > 0 || self.config.tp_rank > 0 {
                return Ok(self.buffers.logits());
            }

            let final_mixer = self
                .qwen4_final_mixer
                .as_ref()
                .context("native Qwen4 decode is missing the final hyperconnection mixer")?;
            let final_hidden = final_mixer.prepare_batched(
                hidden,
                residual,
                rows,
                &self.buffers,
                self.gpu.as_ref(),
                self.config.rms_norm_eps as f32,
                stream,
            )?;
            if tc_dense_active {
                self.lm_head_qwen4_batched_verify_rows(final_hidden, rows as u32, stream)
            } else {
                self.lm_head_qwen4_rows(final_hidden, rows as u32, stream)
            }
        })();

        // Layer-state ownership returns to every request on both success and
        // failure.  The caller never retries a partially executed frame.
        for (seq, states) in seqs.iter_mut().zip(all_layer_states) {
            seq.layer_states = states;
        }
        drop(kv_cache);
        let logits = forward?;
        for (row, seq) in seqs.iter_mut().enumerate() {
            seq.tokens.push(tokens[row]);
            seq.seq_len += 1;
        }
        if dflash_bootstrap
            && self.config.pp_rank == 0
            && self.config.tp_rank == 0
            && self.proposer_is_dflash()
        {
            for (row, seq) in seqs.iter_mut().enumerate() {
                self.capture_dflash_batched_decode_row(seq, row, stream)?;
            }
        }
        if native_mtp_bootstrap && self.config.pp_rank == 0 && self.config.tp_rank == 0 {
            let (storage, frames) = self.qwen4_mtp_hidden_frames.as_ref().ok_or_else(|| {
                anyhow::anyhow!("MTP bootstrap lacks sequence-owned target-hidden storage")
            })?;
            let row_bytes = self.config.residual_width() * 2;
            let hidden_arena_bytes = self.buffers.sizes().hidden_states;
            let mut locked_frames = frames.lock();
            for (row, seq) in seqs.iter().enumerate() {
                let source_offset = row
                    .checked_mul(row_bytes)
                    .ok_or_else(|| anyhow::anyhow!("MTP bootstrap hidden row offset overflow"))?;
                let source_bytes = hidden_arena_bytes
                    .checked_sub(source_offset)
                    .ok_or_else(|| anyhow::anyhow!("MTP bootstrap hidden row past arena"))?;
                let write = locked_frames
                    .begin_write(seq.slot_idx, seq.session_hash, seq.seq_len, 0, source_bytes)
                    .map_err(anyhow::Error::msg)?;
                self.gpu.copy_d2d_async(
                    hidden.offset(source_offset),
                    storage.offset(write.retained.selected.offset),
                    row_bytes,
                    stream,
                )?;
                locked_frames.publish(write).map_err(anyhow::Error::msg)?;
            }
            self.last_mtp_hidden_idx
                .store(0, std::sync::atomic::Ordering::Relaxed);
        }
        let target_only_frame = !dflash_bootstrap && !native_mtp_bootstrap;
        let qsa_active = qsa_routes.is_some();
        if receipt_enabled {
            let sessions = seqs
                .iter()
                .map(|seq| format!("{:016x}", seq.session_hash))
                .collect::<Vec<_>>()
                .join(",");
            let slots = slot_ids
                .iter()
                .map(usize::to_string)
                .collect::<Vec<_>>()
                .join(",");
            let pre_step_seq_lens = seq_lens
                .iter()
                .map(usize::to_string)
                .collect::<Vec<_>>()
                .join(",");
            if let Some(routes) = &qsa_routes {
                let routes = qsa_native::route_labels(routes);
                tracing::info!(
                    "QWEN4_NATIVE_BATCH_FRAME rows={rows} sessions={sessions} slots={slots} pre_step_seq_lens={pre_step_seq_lens} active_only=true target_only={target_only_frame} qsa=true routes={routes} graph=false"
                );
            } else if target_only_frame {
                tracing::info!(
                    "QWEN4_NATIVE_BATCH_FRAME rows={rows} sessions={sessions} slots={slots} pre_step_seq_lens={pre_step_seq_lens} active_only=true target_only=true qsa=false graph=false"
                );
            } else {
                tracing::info!(
                    "QWEN4_NATIVE_BATCH_FRAME rows={rows} sessions={sessions} slots={slots} pre_step_seq_lens={pre_step_seq_lens} active_only=true target_only=false qsa=false graph=false"
                );
            }
        }
        if let Some(routes) = &qsa_routes {
            static QSA_ENGAGED: std::sync::Once = std::sync::Once::new();
            QSA_ENGAGED.call_once(|| {
                tracing::info!(
                    "QWEN4_NATIVE_QSA_ENGAGED rows={rows} routes={} selector=ATLAS_QWEN4_NATIVE_BATCH_QSA mode=per_row graph=false",
                    qsa_native::route_labels(routes)
                );
            });
        }
        static ENGAGED: std::sync::Once = std::sync::Once::new();
        ENGAGED.call_once(|| {
            tracing::info!(
                "QWEN4_NATIVE_BATCH_ENGAGED rows={rows} active_only=true target_only={target_only_frame} qsa={qsa_active}"
            );
        });
        Ok(logits)
    }
}

#[cfg(test)]
#[path = "decode_qwen4_native_batch_tests.rs"]
mod tests;

#[cfg(test)]
mod prefix_cache_kv_tests {
    use super::{
        validate_kv_block_ownership_prefix_cache, validate_kv_rows_before_allocation_prefix_cache,
    };

    #[test]
    fn prefix_cache_allows_shared_readonly_prefix_blocks_with_exclusive_writable_tail() {
        // Two sequences share read-only prefix block 5 (`logical 0 < decode_block 1`),
        // and each holds its own exclusively-owned writable decode block (10 vs 11).
        let seq_lens = [20usize, 22];
        let tables = [vec![5u32, 10], vec![5u32, 11]];
        assert_eq!(
            validate_kv_rows_before_allocation_prefix_cache(&seq_lens, &tables, 16, 16, 64),
            Ok(0)
        );
        let refs = |block: u32| match block {
            5 => 2,
            10 | 11 => 1,
            _ => 0,
        };
        assert_eq!(
            validate_kv_block_ownership_prefix_cache(&seq_lens, &tables, 16, 99, refs),
            Ok(())
        );
    }

    #[test]
    fn prefix_cache_allows_aligned_boundary_needing_fresh_writable_blocks() {
        // Both sequences sit at an exact block boundary (seq_len = 16, decode_block = 1):
        // logical 0 (block 5) is shared read-only, and each row needs 1 fresh block.
        let seq_lens = [16usize, 16];
        let tables = [vec![5u32], vec![5u32]];
        assert_eq!(
            validate_kv_rows_before_allocation_prefix_cache(&seq_lens, &tables, 16, 16, 64),
            Ok(2)
        );
        assert_eq!(
            validate_kv_block_ownership_prefix_cache(&seq_lens, &tables, 16, 99, |_| 3),
            Ok(())
        );
    }

    #[test]
    fn prefix_cache_rejects_shared_or_aliased_writable_decode_block() {
        // Writable decode block 10 aliased across two rows must fail closed.
        let seq_lens = [20usize, 22];
        let aliased = [vec![5u32, 10], vec![5u32, 10]];
        assert!(
            validate_kv_rows_before_allocation_prefix_cache(&seq_lens, &aliased, 16, 16, 64)
                .unwrap_err()
                .contains("alias")
        );

        // A writable decode block aliasing another row's read-only prefix block must fail closed.
        let cross_alias = [vec![5u32, 10], vec![10u32, 11]];
        assert!(
            validate_kv_rows_before_allocation_prefix_cache(&seq_lens, &cross_alias, 16, 16, 64)
                .unwrap_err()
                .contains("alias")
        );

        // A writable decode block with ref_count != 1 must fail closed.
        let tables = [vec![5u32, 10], vec![5u32, 11]];
        assert!(
            validate_kv_block_ownership_prefix_cache(&seq_lens, &tables, 16, 99, |b| {
                if b == 10 { 2 } else { 1 }
            })
            .unwrap_err()
            .contains("exclusively")
        );

        // An unowned read-only prefix block (ref_count == 0) must fail closed.
        assert!(
            validate_kv_block_ownership_prefix_cache(&seq_lens, &tables, 16, 99, |b| {
                if b == 5 { 0 } else { 1 }
            })
            .is_err()
        );
    }
}
