// SPDX-License-Identifier: AGPL-3.0-only

#![allow(unused_imports, dead_code, clippy::too_many_arguments)]

use parking_lot::Mutex;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use anyhow::{Result, bail};
use atlas_core::config::{LayerType, ModelConfig};
use spark_runtime::buffers::BufferArena;
use spark_runtime::gpu::{DevicePtr, GpuBackend, GraphHandle, KernelHandle};
use spark_runtime::kv_cache::PagedKvCache;

use super::super::block_mgmt::{
    apply_evicted_blocks, ensure_blocks_through_decode, ensure_blocks_through_prefill,
    extract_layer_refs, reuse_prefix_match_disk_ids,
};
use super::super::ssm_pool::SsmStatePool;
use super::super::ssm_snapshot::SsmSnapshotPool;
use super::super::types::{PinnedMetaStaging, TransformerModel};
use crate::layer::{
    AttnMetadataDev, ForwardContext, GdnPrefillBuffers, LayerState, SsmLayerState, TransformerLayer,
};
use crate::layers::ops;
use crate::speculative::DraftProposer;
use crate::traits::{ChunkedPrefillPageMetadata, Model, SequenceState};
use crate::weight_map::{DenseWeight, MtpWeights, QuantizedWeight};

impl TransformerModel {
    pub(super) fn start_checkpoint_async_dispatch(&self, seq: &mut SequenceState) -> Result<()> {
        use crate::layer::SsmLayerState;

        let stream = self.secondary_stream;
        // Materialize any deferred KEEP_LATE rows before snapshotting live
        // state into checkpoints.
        self.flush_gdn_keep_late_slot(seq, false, stream)?;
        let mut allocated_ckpt = false;
        for (i, layer_state) in seq.layer_states.iter_mut().enumerate() {
            if self.config.layer_type(i) == LayerType::LinearAttention {
                let ssm = layer_state
                    .as_any_mut()
                    .downcast_mut::<SsmLayerState>()
                    .ok_or_else(|| anyhow::anyhow!("Expected SsmLayerState at layer {i}"))?;

                let nv = self.config.linear_num_value_heads;
                let vd = self.config.linear_value_head_dim;
                let nk = self.config.linear_num_key_heads;
                let kd = self.config.linear_key_head_dim;
                let h_bytes = nv * vd * kd * 4;
                let conv_dim = nk * kd * 2 + nv * vd;
                let d_conv = self.config.linear_conv_kernel_dim;
                let conv_bytes = conv_dim * d_conv * 4;

                if !self.ssm_pool.compact_lazy_h && ssm.h_state_checkpoint.is_none() {
                    ssm.h_state_checkpoint = Some(self.gpu.alloc(h_bytes)?);
                    allocated_ckpt = true;
                }
                if ssm.conv_state_checkpoint.is_none() {
                    ssm.conv_state_checkpoint = Some(self.gpu.alloc(conv_bytes)?);
                    allocated_ckpt = true;
                }

                if let Some(h_ckpt) = ssm.h_state_checkpoint {
                    self.gpu
                        .copy_d2d_async(ssm.h_state, h_ckpt, h_bytes, stream)?;
                }
                if let Some(conv_ckpt) = ssm.conv_state_checkpoint {
                    self.gpu
                        .copy_d2d_async(ssm.conv_state, conv_ckpt, conv_bytes, stream)?;
                }
            }
        }
        if allocated_ckpt {
            self.slot_commit_table.lock().invalidate_slot(seq.slot_idx);
        }
        if self.ssm_pool.compact_lazy_h {
            self.mark_h_ckpt_stale(seq.slot_idx);
        } else {
            self.clear_h_ckpt_stale(seq.slot_idx);
        }
        self.clear_conv_ckpt_stale(seq.slot_idx);
        self.mark_live_state_clean(seq.slot_idx);
        super::verify_qwen4_batched::reset_dedup_slot(seq.slot_idx);
        if let Some(ple) = &self.qwen4_ple {
            ple.checkpoint(seq.slot_idx, self.gpu.as_ref(), stream)?;
        }
        // Record event so default stream can wait (GPU-side, no CPU block).
        self.gpu.record_event(self.secondary_event, stream)?;
        Ok(())
    }

    pub(super) fn start_rollback_and_checkpoint_async_dispatch(
        &self,
        seq: &mut SequenceState,
        num_accepted: usize,
    ) -> Result<()> {
        use crate::layer::SsmLayerState;

        anyhow::ensure!(
            !self.ssm_pool.compact_lazy_h,
            "start_rollback_and_checkpoint_async_dispatch is unsupported on compact_lazy_h pools \
             (slot {}); Qwen4 native MTP commits via commit_verify_state_async_dispatch",
            seq.slot_idx,
        );

        if num_accepted == 0 {
            anyhow::ensure!(
                !self.is_h_ckpt_stale(seq.slot_idx),
                "start_rollback_and_checkpoint_async_dispatch(num_accepted=0): h_state_checkpoint \
                 is stale for slot {} under {QWEN4_GDN_COMMIT_NO_MIRROR_SELECTOR}=1",
                seq.slot_idx,
            );
        }

        let stream = self.secondary_stream;
        let mut ssm_layer_idx = 0usize;

        for (i, layer_state) in seq.layer_states.iter_mut().enumerate() {
            if self.config.layer_type(i) == LayerType::LinearAttention {
                let ssm = layer_state
                    .as_any_mut()
                    .downcast_mut::<SsmLayerState>()
                    .ok_or_else(|| anyhow::anyhow!("Expected SsmLayerState at layer {i}"))?;

                let nv = self.config.linear_num_value_heads;
                let vd = self.config.linear_value_head_dim;
                let nk = self.config.linear_num_key_heads;
                let kd = self.config.linear_key_head_dim;
                let h_bytes = nv * vd * kd * 4;
                let conv_dim = nk * kd * 2 + nv * vd;
                let d_conv = self.config.linear_conv_kernel_dim;
                let conv_bytes = conv_dim * d_conv * 4;

                // Rollback: restore h_state and conv_state from the appropriate source.
                if num_accepted == 0 {
                    // No tokens accepted: restore from checkpoint (pre-verify state).
                    if let Some(ckpt) = ssm.h_state_checkpoint {
                        self.gpu
                            .copy_d2d_async(ckpt, ssm.h_state, h_bytes, stream)?;
                    }
                    if let Some(ckpt) = ssm.conv_state_checkpoint {
                        self.gpu
                            .copy_d2d_async(ckpt, ssm.conv_state, conv_bytes, stream)?;
                    }
                } else {
                    // Partial acceptance: restore from intermediate[num_accepted - 1].
                    let slot = seq.slot_idx;
                    let inter_idx = num_accepted - 1;
                    let h_inter = self.ssm_pool.h_intermediate(ssm_layer_idx, slot, inter_idx);
                    let conv_inter =
                        self.ssm_pool
                            .conv_intermediate(ssm_layer_idx, slot, inter_idx);
                    self.gpu
                        .copy_d2d_async(h_inter, ssm.h_state, h_bytes, stream)?;
                    self.gpu
                        .copy_d2d_async(conv_inter, ssm.conv_state, conv_bytes, stream)?;
                }

                // Checkpoint the (now rolled-back) state for the next verify.
                if let Some(ckpt) = ssm.h_state_checkpoint {
                    self.gpu
                        .copy_d2d_async(ssm.h_state, ckpt, h_bytes, stream)?;
                }
                if let Some(ckpt) = ssm.conv_state_checkpoint {
                    self.gpu
                        .copy_d2d_async(ssm.conv_state, ckpt, conv_bytes, stream)?;
                }

                ssm_layer_idx += 1;
            }
        }
        self.clear_h_ckpt_stale(seq.slot_idx);
        self.mark_live_state_clean(seq.slot_idx);
        super::verify_qwen4_batched::reset_dedup_slot(seq.slot_idx);
        if let Some(ple) = &self.qwen4_ple {
            ple.rollback_and_checkpoint(seq.slot_idx, num_accepted, self.gpu.as_ref(), stream)?;
        }
        // Record event so default stream can wait (GPU-side, no CPU block).
        self.gpu.record_event(self.secondary_event, stream)?;
        Ok(())
    }

    pub(super) fn sync_secondary_dispatch(&self) -> Result<()> {
        // GPU-side event sync: make the default stream wait for the secondary
        // event. Zero CPU cost — the GPU scheduler handles the dependency.
        self.gpu
            .stream_wait_event(self.gpu.default_stream(), self.secondary_event)
    }

    pub(super) fn pre_verify_copy_async_dispatch(&self, seq: &mut SequenceState) -> Result<()> {
        use crate::layer::SsmLayerState;

        let stream = self.gpu.default_stream();
        if in_verify_abort() {
            // Error-only abort rollback: determine whether the GPU already
            // executed a staged KEEP_LATE verify before flushing any remaining
            // pending rows, then leave live + checkpoint conv state consistent.
            if self.has_gdn_keep_late_pending(seq.slot_idx) {
                self.gpu.synchronize(stream)?;
                for (i, layer) in self.layers.iter().enumerate() {
                    if self.config.layer_type(i) == LayerType::LinearAttention {
                        layer.gdn_keep_late_reconcile_abort(self.gpu.as_ref(), seq.slot_idx)?;
                    }
                }
                self.flush_gdn_keep_late_slot(seq, false, stream)?;
            } else {
                for (i, layer) in self.layers.iter().enumerate() {
                    if self.config.layer_type(i) == LayerType::LinearAttention {
                        layer.gdn_keep_late_clear_pending(seq.slot_idx);
                    }
                }
            }
            if self.is_conv_ckpt_stale(seq.slot_idx) {
                self.refresh_conv_ckpt_from_live(seq, stream)?;
                self.clear_conv_ckpt_stale(seq.slot_idx);
                if let Some(ple) = &self.qwen4_ple {
                    ple.restore_checkpoint(seq.slot_idx, self.gpu.as_ref(), stream)?;
                }
                return Ok(());
            }
        } else {
            // Mission 3 soundness guard: verify graphs retained across requests
            // on the same slot must match the new request's QSA and rotary regime.
            if !seq.rotary_positions.is_identity()
                || self.retained_verify_graph_qsa_mismatches(seq.slot_idx, seq.qwen4_qsa_required)
            {
                self.invalidate_verify_graphs_for_slot(seq.slot_idx);
            }

            let active_frame = seq
                .verify_frame
                .filter(|frame| frame.active())
                .map(|frame| (frame.ticket().owner().rows, frame.batch().is_some()));
            let prewarm_k = active_frame.map(|(k, _)| k).unwrap_or_else(|| {
                self.ssm_pool
                    .num_intermediates
                    .min(crate::layers::qwen3_ssm::GDN_LAZY_MAX_K)
            });
            if prewarm_k > 0 {
                self.prewarm_gdn_lazy_retain_pools(prewarm_k)?;
            }
            if let Some((k, is_batch)) = active_frame
                && self.can_stage_keep_late_for_frame(k, is_batch)
            {
                let live_clean = self.take_live_state_clean(seq.slot_idx);
                if !live_clean {
                    if !self.is_conv_ckpt_stale(seq.slot_idx)
                        && !self.has_gdn_keep_late_pending(seq.slot_idx)
                    {
                        self.restore_ssm_checkpoints_to_live(seq, stream)?;
                    }
                    if let Some(ple) = &self.qwen4_ple {
                        ple.restore_checkpoint(seq.slot_idx, self.gpu.as_ref(), stream)?;
                    }
                }
                if self.stage_keep_late_for_slot(seq, k, is_batch, stream)? {
                    return Ok(());
                }
                if live_clean {
                    self.mark_live_state_clean(seq.slot_idx);
                }
            }

            return self.pre_verify_copy_non_keep_late_slot(seq, stream);
        }

        self.restore_pre_verify_ssm_and_ple(seq, stream)
    }

    fn pre_verify_copy_non_keep_late_slot(
        &self,
        seq: &mut SequenceState,
        stream: u64,
    ) -> Result<()> {
        // Non-KEEP_LATE verify path: materialize any deferred rows first so
        // live h_state, conv_state, and conv_ckpt are canonical.
        if self.has_gdn_keep_late_pending(seq.slot_idx) {
            self.flush_gdn_keep_late_slot(seq, false, stream)?;
        } else {
            for (i, layer) in self.layers.iter().enumerate() {
                if self.config.layer_type(i) == LayerType::LinearAttention {
                    layer.gdn_keep_late_clear_pending(seq.slot_idx);
                }
            }
        }
        if self.take_live_state_clean(seq.slot_idx) {
            if self.is_conv_ckpt_stale(seq.slot_idx) {
                self.refresh_conv_ckpt_from_live(seq, stream)?;
                self.clear_conv_ckpt_stale(seq.slot_idx);
            }
            return Ok(());
        }
        if self.is_conv_ckpt_stale(seq.slot_idx) {
            self.refresh_conv_ckpt_from_live(seq, stream)?;
            self.clear_conv_ckpt_stale(seq.slot_idx);
            if let Some(ple) = &self.qwen4_ple {
                ple.restore_checkpoint(seq.slot_idx, self.gpu.as_ref(), stream)?;
            }
            return Ok(());
        }
        self.restore_pre_verify_ssm_and_ple(seq, stream)
    }

    fn restore_pre_verify_ssm_and_ple(&self, seq: &mut SequenceState, stream: u64) -> Result<()> {
        let skip_h_ckpt_restore =
            self.ssm_pool.compact_lazy_h || self.is_h_ckpt_stale(seq.slot_idx);
        for (i, layer_state) in seq.layer_states.iter_mut().enumerate() {
            if self.config.layer_type(i) == LayerType::LinearAttention {
                let ssm = layer_state
                    .as_any_mut()
                    .downcast_mut::<SsmLayerState>()
                    .ok_or_else(|| anyhow::anyhow!("Expected SsmLayerState at layer {i}"))?;

                // No-op if conv checkpoint isn't populated (non-MTP path).
                let Some(conv_ckpt) = ssm.conv_state_checkpoint else {
                    continue;
                };
                if !self.ssm_pool.compact_lazy_h && ssm.h_state_checkpoint.is_none() {
                    continue;
                }

                let nv = self.config.linear_num_value_heads;
                let vd = self.config.linear_value_head_dim;
                let nk = self.config.linear_num_key_heads;
                let kd = self.config.linear_key_head_dim;
                let h_bytes = nv * vd * kd * 4;
                let conv_dim = nk * kd * 2 + nv * vd;
                let d_conv = self.config.linear_conv_kernel_dim;
                let conv_bytes = conv_dim * d_conv * 4;

                // canonical → scratch (live → kernel input/output).
                // Under ATLAS_QWEN4_GDN_COMMIT_NO_MIRROR=1 or compact_lazy_h on
                // seq_lazy, commit_verify_state_async_dispatch leaves canonical
                // post-commit H directly in `ssm.h_state` (and lazyfinal verify
                // never mutates `ssm.h_state`), so only `conv_ckpt -> ssm.conv_state`
                // needs restoration here.
                if !skip_h_ckpt_restore && let Some(h_ckpt) = ssm.h_state_checkpoint {
                    self.gpu
                        .copy_d2d_async(h_ckpt, ssm.h_state, h_bytes, stream)?;
                }
                self.gpu
                    .copy_d2d_async(conv_ckpt, ssm.conv_state, conv_bytes, stream)?;
            }
        }
        if let Some(ple) = &self.qwen4_ple {
            ple.restore_checkpoint(seq.slot_idx, self.gpu.as_ref(), stream)?;
        }
        Ok(())
    }

    /// Batched cross-sequence pre-verify preparation across all `unique_leaders`.
    ///
    /// Semantically identical to calling [`Self::pre_verify_copy_async_dispatch`]
    /// per leader, but collects every `KEEP_LATE` slot's 32-byte control blocks
    /// across all SSM layers into ONE coalesced upload (`16 × 36 = 576` disjoint
    /// 32 B H2D copies -> 1 contiguous `18.4 KB` H2D copy for contiguous slots).
    pub(super) fn pre_verify_copy_async_batch_dispatch(
        &self,
        seqs: &mut [&mut SequenceState],
        unique_leaders: &[usize],
    ) -> Result<()> {
        if unique_leaders.is_empty() {
            return Ok(());
        }
        if in_verify_abort() || unique_leaders.len() == 1 {
            for &s in unique_leaders {
                self.pre_verify_copy_async_dispatch(seqs[s])?;
            }
            return Ok(());
        }
        let stream = self.gpu.default_stream();
        let mut max_prewarm_k = 0usize;
        let mut saw_k1_prewarm = false;
        for &s in unique_leaders {
            let prewarm_k = seqs[s]
                .verify_frame
                .filter(|frame| frame.active())
                .map(|frame| frame.ticket().owner().rows)
                .unwrap_or_else(|| {
                    self.ssm_pool
                        .num_intermediates
                        .min(crate::layers::qwen3_ssm::GDN_LAZY_MAX_K)
                });
            if prewarm_k == 1 {
                saw_k1_prewarm = true;
            }
            max_prewarm_k = max_prewarm_k.max(prewarm_k);
        }
        if max_prewarm_k > 0 {
            self.prewarm_gdn_lazy_retain_pools(max_prewarm_k)?;
        }
        if saw_k1_prewarm && max_prewarm_k != 1 {
            self.prewarm_gdn_lazy_retain_pools(1)?;
        }

        let mut can_stage_cache = [None::<bool>; crate::layers::qwen3_ssm::GDN_LAZY_MAX_K + 1];
        let expected_entries = unique_leaders.len() * self.ssm_pool.num_ssm_layers;
        let mut batch_ctrl_payloads = take_keep_late_entries_buf(expected_entries);
        let mut slot_ctrl_buf: Vec<(DevicePtr, [u8; 32])> =
            Vec::with_capacity(self.ssm_pool.num_ssm_layers);

        let batch_res: Result<()> = (|| {
            for &s in unique_leaders {
                let seq = &mut *seqs[s];
                if !seq.rotary_positions.is_identity()
                    || self
                        .retained_verify_graph_qsa_mismatches(seq.slot_idx, seq.qwen4_qsa_required)
                {
                    self.invalidate_verify_graphs_for_slot(seq.slot_idx);
                }
                let active_frame = seq
                    .verify_frame
                    .filter(|frame| frame.active())
                    .map(|frame| (frame.ticket().owner().rows, frame.batch().is_some()));
                match active_frame {
                    Some((k, true)) => {
                        let can_stage = if k < can_stage_cache.len() {
                            *can_stage_cache[k]
                                .get_or_insert_with(|| self.can_stage_keep_late_for_frame(k, true))
                        } else {
                            self.can_stage_keep_late_for_frame(k, true)
                        };
                        if can_stage {
                            let live_clean = self.take_live_state_clean(seq.slot_idx);
                            if !live_clean {
                                if !self.is_conv_ckpt_stale(seq.slot_idx)
                                    && !self.has_gdn_keep_late_pending(seq.slot_idx)
                                {
                                    self.restore_ssm_checkpoints_to_live(seq, stream)?;
                                }
                                if let Some(ple) = &self.qwen4_ple {
                                    ple.restore_checkpoint(
                                        seq.slot_idx,
                                        self.gpu.as_ref(),
                                        stream,
                                    )?;
                                }
                            }
                            slot_ctrl_buf.clear();
                            if self.collect_keep_late_ctrl_for_slot(seq, &mut slot_ctrl_buf)? {
                                batch_ctrl_payloads.extend_from_slice(&slot_ctrl_buf);
                                continue;
                            }
                            if live_clean {
                                self.mark_live_state_clean(seq.slot_idx);
                            }
                        }
                        self.pre_verify_copy_non_keep_late_slot(seq, stream)?;
                    }
                    Some((_, false)) => {
                        self.pre_verify_copy_async_dispatch(seq)?;
                    }
                    None => {
                        self.pre_verify_copy_non_keep_late_slot(seq, stream)?;
                    }
                }
            }
            if !batch_ctrl_payloads.is_empty() {
                self.upload_coalesced_keep_late_ctrl(&mut batch_ctrl_payloads, &[], stream)?;
            }
            Ok(())
        })();
        return_keep_late_entries_buf(batch_ctrl_payloads);
        batch_res
    }

    pub(in crate::model) fn commit_qwen4_qsa_k5_only_dispatch(
        &self,
        num_accepted: usize,
        k: usize,
    ) -> Result<()> {
        anyhow::ensure!(
            num_accepted <= k,
            "QSA verify commit accepted {num_accepted} rows from a K={k} frame"
        );
        if !crate::layers::qwen4_qsa::verify_journal_rows(k) {
            return Ok(());
        }
        let stream = self.gpu.default_stream();
        for layer in &self.layers {
            layer.commit_qwen4_qsa_rows(num_accepted, k, self.gpu.as_ref(), stream)?;
        }
        if self.qwen4_k5_cmulti_enforced() {
            // Every layer's commit or restore is enqueued: the journals are
            // consumed. A partial failure above leaves the owner armed, so
            // only that request can retry or abort.
            self.qwen4_qsa_journal_owner.lock().release();
        }
        Ok(())
    }

    /// Commit QSA verify prefix journals across a batch of `(slot_idx, num_accepted, k)`
    /// entries, collapsing `C × 12` per-sequence launches into 12 multi-sequence
    /// `qwen4_qsa_commit_verify_prefix_multi_seq` launches (with clean fallback
    /// to per-sequence commits when `num_seqs <= 1`, lanes repeat, staging is
    /// absent, or the multi-seq kernel handle is unavailable).
    fn commit_qwen4_qsa_batch_dispatch(&self, entries: &[(usize, usize, usize)]) -> Result<()> {
        if entries.is_empty() {
            return Ok(());
        }
        match crate::layers::qwen4_qsa::pack_qsa_multi_seq_commit_entries(entries)? {
            Some(pack) if pack.num_seqs == 0 => return Ok(()),
            Some(pack) if pack.num_seqs >= 2 => {
                if let Some(staging) = self.qwen4_cross_seq_staging
                    && !staging.verify_meta_dev.is_null()
                    && pack.bytes.len() <= staging.verify_meta_bytes
                {
                    let stream = self.gpu.default_stream();
                    let base = staging.verify_meta_dev;
                    self.copy_h2d_group_fast(
                        &[spark_runtime::gpu::HostToDeviceCopy::new(&pack.bytes, base)],
                        stream,
                    )?;
                    let _guard = crate::layers::qwen4_qsa::QsaMultiSeqCommitGuard::enter(
                        base,
                        base.offset(pack.accepted_offset),
                        base.offset(pack.rows_offset),
                        pack.num_seqs,
                        pack.max_lane,
                        pack.fallback_entries,
                    );
                    for layer in &self.layers {
                        layer.commit_qwen4_qsa_rows(0, 2, self.gpu.as_ref(), stream)?;
                    }
                    drop(_guard);
                    if self.qwen4_k5_cmulti_enforced() {
                        self.qwen4_qsa_journal_owner.lock().release();
                    }
                    return Ok(());
                }
            }
            Some(_) | None => {}
        }
        for &(slot_idx, num_accepted, k) in entries {
            let _lane = crate::layers::qwen4_verify_lanes::LaneGuard::enter_for_slot(slot_idx)?;
            self.commit_qwen4_qsa_k5_only_dispatch(num_accepted, k)?;
        }
        Ok(())
    }

    fn qwen4_k5_cmulti_enforced(&self) -> bool {
        self.commit_runtime_flags.k5_cmulti_enforced
    }

    /// `ATLAS_QWEN4_K5_CMULTI`: a request may commit or restore only a QSA
    /// journal it armed itself (an unarmed journal always passes).
    fn check_qsa_journal_commit(&self, seq: &SequenceState) -> Result<()> {
        if !self.qwen4_k5_cmulti_enforced() {
            return Ok(());
        }
        self.qwen4_qsa_journal_owner
            .lock()
            .check_commit(super::super::qwen4_k5_cmulti::QsaJournalOwner {
                slot: seq.slot_idx,
                session: seq.session_hash,
            })
            .map_err(anyhow::Error::msg)
    }

    /// Whether the cached single-launch batched GDN verify-commit kernel (or
    /// zero-replay `KEEP_LATE` batch commit under `compact_lazy_h`) is
    /// available and eligible under the model's resolved runtime flags.
    pub(crate) fn supports_verify_batch_cached(&self) -> bool {
        let keep_late_batch = self.ssm_pool.compact_lazy_h
            && self.config.is_qwen4_exp()
            && crate::layers::qwen3_ssm::gdn_keep_late_enabled();
        (self.gdn_batch_commit_kernel.0 != 0 || keep_late_batch)
            && self.ssm_pool.num_ssm_layers > 0
            && self.ssm_pool.has_mtp
            && (!self.ssm_pool.compact_lazy_h || keep_late_batch)
            && !self.commit_runtime_flags.skip_restore_control
            && !self.commit_runtime_flags.async_probe
            && !self.commit_runtime_flags.bootstrap_trace
    }

    fn collect_slot_commit_pointers(
        &self,
        seq: &SequenceState,
    ) -> Result<Vec<super::commit_cache::SsmLayerCommitPointers>> {
        let mut ptrs = Vec::with_capacity(self.ssm_pool.num_ssm_layers);
        let slot = seq.slot_idx;
        let _lane = crate::layers::qwen4_verify_lanes::LaneGuard::enter_for_slot(slot)?;
        let mut ssm_layer_idx = 0usize;
        for (i, layer_state) in seq.layer_states.iter().enumerate() {
            if self.config.layer_type(i) != LayerType::LinearAttention {
                continue;
            }
            let ssm = layer_state
                .as_any()
                .downcast_ref::<SsmLayerState>()
                .ok_or_else(|| anyhow::anyhow!("Expected SsmLayerState at layer {i}"))?;
            let h_inter_base = if self.ssm_pool.num_intermediates > 0 {
                self.ssm_pool.h_intermediate(ssm_layer_idx, slot, 0).0
            } else {
                0
            };
            let conv_inter_base = if self.ssm_pool.num_intermediates > 0 {
                self.ssm_pool.conv_intermediate(ssm_layer_idx, slot, 0).0
            } else {
                0
            };
            let retain_slot_base = self.layers[i].gdn_seq_lazy_retain().map_or(0, |ptr| ptr.0);
            ptrs.push(super::commit_cache::SsmLayerCommitPointers {
                h_state: ssm.h_state.0,
                h_ckpt: ssm.h_state_checkpoint.map_or(0, |ptr| ptr.0),
                conv_state: ssm.conv_state.0,
                conv_ckpt: ssm.conv_state_checkpoint.map_or(0, |ptr| ptr.0),
                h_inter_base,
                conv_inter_base,
                retain_slot_base,
                _pad: 0,
            });
            ssm_layer_idx += 1;
        }
        anyhow::ensure!(
            ssm_layer_idx == self.ssm_pool.num_ssm_layers,
            "commit cache SSM layer count mismatch: collected {ssm_layer_idx}, expected {}",
            self.ssm_pool.num_ssm_layers
        );
        Ok(ptrs)
    }

    /// Commit a batch of verified sequences `(seq, num_accepted, k)` through the
    /// cached per-slot pointer table (`SlotCommitTable`) and single-launch
    /// `gdn_batch_commit_copy_f32` kernel (1 H2D + 1 kernel launch + 1 event
    /// record across all `num_ssm_layers × batch_size` pairs).
    ///
    /// Under `compact_lazy_h`, eligible non-zero-accept sequences use this
    /// method's dedicated zero-replay `KEEP_LATE` cohort path. Sequences that
    /// require lazy replay, full-reject handling, tree mode, ineligible
    /// `compact_lazy_h` fallback, or probe diagnostics fall back cleanly to
    /// [`Self::commit_verify_state_async_dispatch`] for 100% numerical parity.
    ///
    /// # Errors
    ///
    /// Returns an error if validation fails (OOB slot/intermediate, QSA journal
    /// ownership mismatch) or any GPU copy/launch/event operation fails.
    pub fn commit_verify_batch_cached(
        &self,
        entries: &mut [(&mut SequenceState, usize, usize)],
        stream: u64,
    ) -> Result<()> {
        use super::commit_cache::{
            BATCH_COMMIT_FLAG_H_ALREADY_LIVE, BATCH_COMMIT_FLAG_HAS_CONV_INTER,
            BATCH_COMMIT_FLAG_MIRROR_CKPT, BatchCommitEntry, serialize_batch_entries,
        };

        if entries.is_empty() {
            return Ok(());
        }
        let commit_stream = if stream == 0 {
            self.secondary_stream
        } else {
            stream
        };
        let flat_tree_injected = self
            .dflash_flat_tree_route
            .load(std::sync::atomic::Ordering::Acquire);
        let was_tree_mode = super::commit_plan::commit_sees_tree_mode(
            self.ddtree_parent_ids_dev.lock().is_some(),
            flat_tree_injected,
        );

        let flags = self.commit_runtime_flags;
        let h_bytes = self.ssm_pool.h_bytes;
        let conv_bytes = self.ssm_pool.conv_bytes;
        let can_batch = self.supports_verify_batch_cached()
            && !was_tree_mode
            && h_bytes.is_multiple_of(4)
            && conv_bytes.is_multiple_of(4);
        if !can_batch {
            for (seq, num_accepted, k) in entries.iter_mut() {
                let _lane =
                    crate::layers::qwen4_verify_lanes::LaneGuard::enter_for_slot(seq.slot_idx)?;
                self.commit_verify_state_async_dispatch(
                    seq,
                    *num_accepted,
                    *k,
                    num_accepted.saturating_sub(1),
                )?;
            }
            return Ok(());
        }

        // Zero-replay KEEP_LATE cohort commit path (`compact_lazy_h` + Qwen4):
        // eliminates 36×C GDN replay launches and 36×2C conv D2D copies per round,
        // folding each slot's committed rows into the next fused verify kernel.
        if self.ssm_pool.compact_lazy_h {
            let mut keep_late_indices: Vec<usize> = Vec::with_capacity(entries.len());
            let mut fallback_indices: Vec<usize> = Vec::new();
            for (idx, (seq, num_accepted, k)) in entries.iter().enumerate() {
                let num_accepted = *num_accepted;
                let k = *k;
                self.note_gdn_keep_late_verify_completed(seq.slot_idx);
                self.check_qsa_journal_commit(seq)?;
                if num_accepted == 0 {
                    fallback_indices.push(idx);
                    continue;
                }
                let last_inter_slot = num_accepted - 1;
                if last_inter_slot >= self.ssm_pool.num_intermediates {
                    bail!(
                        "commit_verify_batch_cached: last_inter_slot={last_inter_slot} OOB \
                         (num_intermediates={}, k={k}, num_accepted={num_accepted})",
                        self.ssm_pool.num_intermediates,
                    );
                }
                if self.can_commit_keep_late_slot(seq, num_accepted, k, false) {
                    keep_late_indices.push(idx);
                } else {
                    fallback_indices.push(idx);
                }
            }
            if !keep_late_indices.is_empty() {
                for &idx in &keep_late_indices {
                    let (ref seq, num_accepted, _k) = entries[idx];
                    super::verify_qwen4_batched::reset_dedup_slot(seq.slot_idx);
                    if let Some(ple) = &self.qwen4_ple {
                        ple.rollback_and_checkpoint(
                            seq.slot_idx,
                            num_accepted,
                            self.gpu.as_ref(),
                            commit_stream,
                        )?;
                    }
                }
                self.gpu.record_event(self.secondary_event, commit_stream)?;
                if fallback_indices.is_empty() {
                    self.dflash_flat_tree_route
                        .store(false, std::sync::atomic::Ordering::Release);
                }
                let qsa_entries: Vec<(usize, usize, usize)> = keep_late_indices
                    .iter()
                    .map(|&idx| (entries[idx].0.slot_idx, entries[idx].1, entries[idx].2))
                    .collect();
                self.commit_qwen4_qsa_batch_dispatch(&qsa_entries)?;
                for &idx in &keep_late_indices {
                    let num_accepted = entries[idx].1;
                    self.record_keep_late_slot(entries[idx].0, num_accepted);
                }
            }
            for idx in fallback_indices {
                let (seq, num_accepted, k) = &mut entries[idx];
                let _lane =
                    crate::layers::qwen4_verify_lanes::LaneGuard::enter_for_slot(seq.slot_idx)?;
                self.commit_verify_state_async_dispatch(
                    seq,
                    *num_accepted,
                    *k,
                    num_accepted.saturating_sub(1),
                )?;
            }
            return Ok(());
        }

        let num_ssm_layers = self.ssm_pool.num_ssm_layers;
        let h_state_f32s = u32::try_from(h_bytes / 4)
            .map_err(|_| anyhow::anyhow!("commit cache h_state_f32s exceeds u32"))?;
        let conv_state_f32s = u32::try_from(conv_bytes / 4)
            .map_err(|_| anyhow::anyhow!("commit cache conv_state_f32s exceeds u32"))?;
        let h_inter_stride_bytes = self.ssm_pool.h_intermediate_stride() as u64;
        let conv_inter_stride_bytes = conv_bytes as u64;

        let mut gpu_entries: Vec<BatchCommitEntry> = Vec::with_capacity(entries.len());
        let mut fast_meta: Vec<(usize, usize, usize, bool)> = Vec::with_capacity(entries.len());
        let mut fallback_indices: Vec<usize> = Vec::new();

        for (idx, (seq, num_accepted, k)) in entries.iter().enumerate() {
            let num_accepted = *num_accepted;
            let k = *k;
            self.check_qsa_journal_commit(seq)?;
            if num_accepted == 0 {
                fallback_indices.push(idx);
                continue;
            }
            let last_inter_slot = num_accepted - 1;
            if last_inter_slot >= self.ssm_pool.num_intermediates {
                bail!(
                    "commit_verify_batch_cached: last_inter_slot={last_inter_slot} OOB \
                     (num_intermediates={}, k={k}, num_accepted={num_accepted})",
                    self.ssm_pool.num_intermediates,
                );
            }
            if seq.slot_idx >= self.ssm_pool.spec_slots {
                fallback_indices.push(idx);
                continue;
            }

            let mut all_seq_lazy = true;
            let mut any_seq_lazy = false;
            let mut any_wy17_replay = false;
            for (layer_i, layer) in self.layers.iter().enumerate() {
                if self.config.layer_type(layer_i) != LayerType::LinearAttention {
                    continue;
                }
                let sl = layer.gdn_seq_lazy_engaged(k);
                all_seq_lazy &= sl;
                any_seq_lazy |= sl;
                if num_accepted < k
                    && super::commit_plan::wy17_replay_allowed(
                        flags.lazy_commit_env,
                        flags.lazy_j,
                        false,
                        num_accepted,
                        k,
                        last_inter_slot,
                        layer.wy17_lazy_engaged(k),
                    )
                    && layer.wy17_replay_kernel().0 != 0
                {
                    any_wy17_replay = true;
                }
            }

            // Decide whether this entry fits the pure-copy kernel or needs
            // per-layer replay in `commit_verify_state_async_dispatch`.
            let (entry_flags, skipped_h_ckpt) = if num_accepted == k {
                if any_seq_lazy && !all_seq_lazy {
                    fallback_indices.push(idx);
                    continue;
                }
                let no_mirror = all_seq_lazy
                    && seq_lazy_no_mirror_enabled(
                        self.config.is_qwen4_exp(),
                        flags.qwen4_no_mirror_req,
                        ssm_gdn_commit_no_mirror_default_on(),
                    );
                let mut f = if all_seq_lazy {
                    0u32
                } else {
                    BATCH_COMMIT_FLAG_H_ALREADY_LIVE
                };
                if !no_mirror {
                    f |= BATCH_COMMIT_FLAG_MIRROR_CKPT;
                }
                (f, no_mirror)
            } else {
                if any_seq_lazy || any_wy17_replay {
                    fallback_indices.push(idx);
                    continue;
                }
                (
                    BATCH_COMMIT_FLAG_MIRROR_CKPT | BATCH_COMMIT_FLAG_HAS_CONV_INTER,
                    false,
                )
            };

            let cached = {
                let maybe_hit = self.slot_commit_table.lock().cached_slot(seq.slot_idx);
                match maybe_hit {
                    Some(hit) => hit,
                    None => {
                        let ptrs = self.collect_slot_commit_pointers(seq)?;
                        self.slot_commit_table.lock().ensure_slot_table(
                            seq.slot_idx,
                            &ptrs,
                            self.gpu.as_ref(),
                        )?
                    }
                }
            };
            if !cached.all_have_conv_ckpt || !cached.all_have_h_ckpt {
                fallback_indices.push(idx);
                continue;
            }
            gpu_entries.push(BatchCommitEntry {
                ssm_table_dev: cached.ssm_ptrs_dev.0,
                accepted_idx: u32::try_from(last_inter_slot)
                    .map_err(|_| anyhow::anyhow!("accepted_idx exceeds u32"))?,
                flags: entry_flags,
            });
            fast_meta.push((seq.slot_idx, num_accepted, k, skipped_h_ckpt));
        }

        if !gpu_entries.is_empty() {
            let entries_dev = self
                .slot_commit_table
                .lock()
                .ensure_entries_buf(gpu_entries.len(), self.gpu.as_ref())?;
            let entries_bytes = serialize_batch_entries(&gpu_entries);
            self.copy_h2d_group_fast(
                &[spark_runtime::gpu::HostToDeviceCopy::new(
                    &entries_bytes,
                    entries_dev,
                )],
                commit_stream,
            )?;
            spark_runtime::kernel_args::KernelLaunch::new(
                self.gpu.as_ref(),
                self.gdn_batch_commit_kernel,
            )
            .grid([num_ssm_layers as u32, gpu_entries.len() as u32, 1])
            .block([256, 1, 1])
            .arg_ptr(entries_dev)
            .arg_u32(num_ssm_layers as u32)
            .arg_u32(h_state_f32s)
            .arg_u32(conv_state_f32s)
            .arg_u64(h_inter_stride_bytes)
            .arg_u64(conv_inter_stride_bytes)
            .launch(commit_stream)?;

            for &(slot_idx, num_accepted, _k, skipped_h_ckpt) in &fast_meta {
                if skipped_h_ckpt {
                    self.mark_h_ckpt_stale(slot_idx);
                    log_qwen4_gdn_commit_no_mirror_engaged();
                } else {
                    self.clear_h_ckpt_stale(slot_idx);
                }
                self.mark_live_state_clean(slot_idx);
                super::verify_qwen4_batched::reset_dedup_slot(slot_idx);
                if let Some(ple) = &self.qwen4_ple {
                    ple.rollback_and_checkpoint(
                        slot_idx,
                        num_accepted,
                        self.gpu.as_ref(),
                        commit_stream,
                    )?;
                }
            }
            self.gpu.record_event(self.secondary_event, commit_stream)?;
            if fallback_indices.is_empty() {
                self.dflash_flat_tree_route
                    .store(false, std::sync::atomic::Ordering::Release);
            }
            let qsa_entries: Vec<(usize, usize, usize)> = fast_meta
                .iter()
                .map(|&(slot_idx, num_accepted, k, _)| (slot_idx, num_accepted, k))
                .collect();
            self.commit_qwen4_qsa_batch_dispatch(&qsa_entries)?;
        }

        for idx in fallback_indices {
            let (seq, num_accepted, k) = &mut entries[idx];
            let _lane = crate::layers::qwen4_verify_lanes::LaneGuard::enter_for_slot(seq.slot_idx)?;
            self.commit_verify_state_async_dispatch(
                seq,
                *num_accepted,
                *k,
                num_accepted.saturating_sub(1),
            )?;
        }

        Ok(())
    }

    /// Error-only transaction abort. A failed commit can leave work queued on
    /// the secondary stream before its event is recorded, so an event wait is
    /// insufficient here: drain that stream on the host, then restore QSA and
    /// recurrent/PLE state in default-stream FIFO order.
    pub(in crate::model) fn abort_verify_state_dispatch(
        &self,
        seq: &mut SequenceState,
        k: usize,
    ) -> Result<()> {
        // Normal commit consumes this one-shot route flag at its final QSA
        // boundary. Abort has no retry contract, so consume it before cleanup
        // even if cleanup reports an error; otherwise an unrelated next frame
        // can inherit stale tree-mode ownership.
        self.dflash_flat_tree_route
            .swap(false, std::sync::atomic::Ordering::Acquire);
        self.clear_live_state_clean(seq.slot_idx);
        let drain = self.gpu.synchronize(self.secondary_stream);
        let qsa = self
            .check_qsa_journal_commit(seq)
            .and_then(|()| self.commit_qwen4_qsa_k5_only_dispatch(0, k));
        let recurrent = {
            let _abort_guard = VerifyAbortGuard::enter();
            self.pre_verify_copy_async_dispatch(seq)
        };
        let mut failures = Vec::new();
        if let Err(error) = drain {
            failures.push(format!("secondary-stream drain failed: {error:#}"));
        }
        if let Err(error) = qsa {
            failures.push(format!("QSA rollback failed: {error:#}"));
        }
        if let Err(error) = recurrent {
            failures.push(format!("recurrent rollback failed: {error:#}"));
        }
        if !failures.is_empty() {
            crate::model::verify_failure::poison_abort(crate::model::verify_failure::AbortOwner {
                slot: seq.slot_idx,
                session: seq.session_hash,
                cursor: seq.seq_len,
                rows: k,
            });
            anyhow::bail!("{}; engine restart required", failures.join("; "));
        }
        Ok(())
    }

    pub(in crate::model) fn commit_verify_state_async_dispatch(
        &self,
        seq: &mut SequenceState,
        num_accepted: usize,
        k: usize,
        last_inter_slot: usize,
    ) -> Result<()> {
        use crate::layer::SsmLayerState;

        // Notify SSM layers that verify completed on GPU (covering CUDA-graph
        // replayed suffix layers whose host forward did not execute).
        self.note_gdn_keep_late_verify_completed(seq.slot_idx);

        // Read, but do not consume, flat-tree ownership yet. Every operation
        // below that precedes the final QSA journal commit may fail, and
        // sampled DFlash can retry this commit after output becomes
        // irreversible. A retry must still take the tree-state copy path.
        let flat_tree_injected = self
            .dflash_flat_tree_route
            .load(std::sync::atomic::Ordering::Acquire);
        // Refuse before any recurrent/PLE effect, so the owner can still abort.
        self.check_qsa_journal_commit(seq)?;

        if num_accepted == 0 {
            for (i, layer) in self.layers.iter().enumerate() {
                if self.config.layer_type(i) == LayerType::LinearAttention {
                    layer.gdn_keep_late_clear_pending(seq.slot_idx);
                }
            }
            self.clear_live_state_clean(seq.slot_idx);
            // Full reject: canonical state untouched — no commit needed.
            // Still record the event so sync_secondary has something to wait
            // on (defensive: ensures pre-verify ordering on next iteration).
            self.gpu
                .record_event(self.secondary_event, self.secondary_stream)?;
            // All recurrent/event work succeeded. Consume the one-shot route
            // immediately before the final QSA transaction operation.
            self.dflash_flat_tree_route
                .store(false, std::sync::atomic::Ordering::Release);
            // Keep QSA armed until every other fallible enqueue succeeds, so
            // the caller can still restore the full speculative frame on an
            // error. This default-stream launch is the final fallible action.
            self.commit_qwen4_qsa_k5_only_dispatch(num_accepted, k)?;
            return Ok(());
        }

        // Tree-mode verify runs the tree-aware GDN kernel, which only writes
        // h_state_intermediates[t] per token and leaves h_state untouched. The
        // full-accept fast path below (`num_accepted == k && !was_tree_mode &&
        // !self.ssm_pool.compact_lazy_h`) assumes wy_k semantics where h_state
        // was updated in-place to post-K state — wrong for tree mode. Force the
        // partial-accept copy path (copy intermediate[last_inter_slot] ->
        // h_state) regardless of num_accepted when tree mode was active.
        //
        // "Tree mode" has TWO producers, and both must be visible:
        //   * the scheduler-set payload stash (`ddtree_parent_ids_dev`), and
        //   * the verify's own graph-safe FLAT-CHAIN injection
        //     (`dflash_flat_tree_route`, set by verify_d.rs when
        //     `k == ddtree_parent_ids_capacity` with graphs on and
        //     ATLAS_DISABLE_TREE_WY unset).
        // Missing the second one made every FULL accept on an injected verify
        // (e.g. K=12 / γ=11 DSpark) commit the STALE pre-verify h_state.
        let was_tree_mode = super::commit_plan::commit_sees_tree_mode(
            self.ddtree_parent_ids_dev.lock().is_some(),
            flat_tree_injected,
        );

        // Defensive bounds check: `last_inter_slot` must be a valid pool slot.
        // The intermediate pool is sized to `num_intermediates = γ+2 = k+1`
        // (verified intermediate slots `0..self.ssm_pool.num_intermediates`).
        // Out-of-bounds indicates a caller bug — fail loudly rather than
        // corrupting an unrelated sequence's state pool.
        if last_inter_slot >= self.ssm_pool.num_intermediates {
            bail!(
                "commit_verify_state_async: last_inter_slot={} OOB (num_intermediates={}, k={}, num_accepted={}, tree_mode={})",
                last_inter_slot,
                self.ssm_pool.num_intermediates,
                k,
                num_accepted,
                was_tree_mode,
            );
        }

        // WY17 LAZY commit: reconstruct a skipped intermediate slot via the
        // replay kernel instead of the intermediate → h_state D2D copy. Active
        // only when the gate is on, lazy J>1, and the tree path was NOT used
        // (tree writes all slots; the flat wy17 path is the only lazy producer).
        //
        // Task #34 sibling hazard: the env gates alone are NOT sufficient — a
        // K≠17 verify runs the CHUNKED wy4/wy3/wy2 path, which persists ALL
        // intermediate slots and never populates the k/v/gate/beta retention
        // buffers the replay kernel reads. The per-layer
        // `wy17_lazy_engaged(k)` check below (shared with the dispatch)
        // guarantees replay fires only when the lazy wy17 kernel actually
        // produced this verify's intermediates.
        let flags = self.commit_runtime_flags;
        let lazy_j = flags.lazy_j;
        let lazy_commit_env = flags.lazy_commit_env;

        // ATLAS_DFLASH_ASYNC_PROBE=1: measure the TRUE GPU duration of this
        // commit tail (enqueue + secondary-stream drain), not just the CPU
        // enqueue time STEP_TIMING reports. Measurement-only.
        let probe = flags.async_probe;
        let t_probe = if probe {
            Some(std::time::Instant::now())
        } else {
            None
        };

        let no_mirror_req = seq_lazy_no_mirror_enabled(
            self.config.is_qwen4_exp(),
            flags.qwen4_no_mirror_req,
            ssm_gdn_commit_no_mirror_default_on(),
        );
        let skip_restore = flags.skip_restore_control;
        let bootstrap_trace = flags.bootstrap_trace;
        let nv = self.config.linear_num_value_heads;
        let vd = self.config.linear_value_head_dim;
        let nk = self.config.linear_num_key_heads;
        let kd = self.config.linear_key_head_dim;
        let h_bytes = nv * vd * kd * 4;
        let conv_dim = nk * kd * 2 + nv * vd;
        let d_conv = self.config.linear_conv_kernel_dim;
        let conv_bytes = conv_dim * d_conv * 4;
        let mut any_h_ckpt_skipped = false;
        let mut all_h_ckpt_written = true;

        let stream = self.secondary_stream;
        let keep_late_ready = self.can_commit_keep_late_slot(seq, num_accepted, k, was_tree_mode);
        if !keep_late_ready {
            let mut ssm_layer_idx = 0usize;

            for (i, layer_state) in seq.layer_states.iter_mut().enumerate() {
                if self.config.layer_type(i) == LayerType::LinearAttention {
                    // Snapshot the replay kernel handle + retention pointers BEFORE
                    // taking the &mut borrow of the layer state (avoids borrowing
                    // `self.layers` while `seq.layer_states` is mutably borrowed).
                    let replay_kernel = self.layers[i].wy17_replay_kernel();
                    // Did THIS layer's dispatch run the lazy wy17 kernel for a
                    // k-token verify? Pure function of k + env + kernel handles
                    // (graph-replay safe); shared with the dispatch. False for
                    // every K≠17 (chunked/fused) verify — replay must not fire
                    // there (task #34 sibling hazard).
                    let lazy_engaged = self.layers[i].wy17_lazy_engaged(k);
                    // GDN LAZY (ExactSequence): pure mirror of the dispatch
                    // decision — same k, same env, same handles (graph-safe).
                    let seq_lazy = self.layers[i].gdn_seq_lazy_engaged(k);
                    let seq_replay_kernel = self.layers[i].gdn_seq_replay_kernel();
                    let seq_retain = self.layers[i].gdn_seq_lazy_retain();

                    let ssm = layer_state
                        .as_any_mut()
                        .downcast_mut::<SsmLayerState>()
                        .ok_or_else(|| anyhow::anyhow!("Expected SsmLayerState at layer {i}"))?;

                    let h_ckpt_opt = ssm.h_state_checkpoint;
                    let Some(conv_ckpt) = ssm.conv_state_checkpoint else {
                        ssm_layer_idx += 1;
                        continue;
                    };
                    if h_ckpt_opt.is_none() && !self.ssm_pool.compact_lazy_h {
                        ssm_layer_idx += 1;
                        continue;
                    }

                    if self.ssm_pool.compact_lazy_h && (!seq_lazy || was_tree_mode) {
                        bail!(
                            "commit_verify_state_async: compact_lazy_h requires flat-chain GDN seq_lazy \
                         at ssm layer {ssm_layer_idx} (k={k}, tree={was_tree_mode})"
                        );
                    }

                    // Under `compact_lazy_h`, `h_intermediate` is a single shared
                    // write-sink overwritten across layers, so full accepts (`num_accepted == k`)
                    // reconstruct post-K `ssm.h_state` via the same exact `nosnap`
                    // replay kernel used by partial accepts (`last_inter_slot = k - 1`).
                    if num_accepted == k && !was_tree_mode && !self.ssm_pool.compact_lazy_h {
                        let no_mirror = seq_lazy && no_mirror_req;
                        if seq_lazy {
                            // lazyfinal left h_state at STEP-INITIAL; the true
                            // post-K state is in inter[k-1] (the one slot the
                            // lazy kernel writes). Restore it before the
                            // checkpoint mirror below.
                            let h_fin =
                                self.ssm_pool
                                    .h_intermediate(ssm_layer_idx, seq.slot_idx, k - 1);
                            self.gpu
                                .copy_d2d_async(h_fin, ssm.h_state, h_bytes, stream)?;
                        }
                        if no_mirror || h_ckpt_opt.is_none() {
                            any_h_ckpt_skipped = true;
                            all_h_ckpt_written = false;
                        } else if let Some(h_ckpt) = h_ckpt_opt {
                            self.gpu
                                .copy_d2d_async(ssm.h_state, h_ckpt, h_bytes, stream)?;
                        }
                        self.gpu
                            .copy_d2d_async(ssm.conv_state, conv_ckpt, conv_bytes, stream)?;
                    } else {
                        let slot = seq.slot_idx;
                        let conv_inter =
                            self.ssm_pool
                                .conv_intermediate(ssm_layer_idx, slot, last_inter_slot);

                        let use_replay = super::commit_plan::wy17_replay_allowed(
                            lazy_commit_env,
                            lazy_j,
                            was_tree_mode,
                            num_accepted,
                            k,
                            last_inter_slot,
                            lazy_engaged,
                        ) && replay_kernel.0 != 0
                            && h_ckpt_opt.is_some()
                            && ssm.wy17_kv_retain.is_some()
                            && ssm.wy17_gate_retain.is_some();

                        if ssm_layer_idx == 0 && bootstrap_trace {
                            tracing::info!(
                                "DFLASH_COMMIT_TRACE k={} accepted={} inter_slot={} tree={} lazy_j={} lazy_engaged={} replay_kernel={} retained={} use_replay={}",
                                k,
                                num_accepted,
                                last_inter_slot,
                                was_tree_mode,
                                lazy_j,
                                lazy_engaged,
                                replay_kernel.0 != 0,
                                ssm.wy17_kv_retain.is_some() && ssm.wy17_gate_retain.is_some(),
                                use_replay,
                            );
                        }

                        if use_replay {
                            anyhow::ensure!(
                                !self.is_h_ckpt_stale(seq.slot_idx),
                                "WY17 lazy replay requires a fresh h_state_checkpoint on slot {}, \
                             but it is stale under {QWEN4_GDN_COMMIT_NO_MIRROR_SELECTOR}=1",
                                seq.slot_idx,
                            );
                            let h_ckpt = h_ckpt_opt.ok_or_else(|| {
                            anyhow::anyhow!(
                                "WY17 lazy replay missing h_state_checkpoint at ssm layer {ssm_layer_idx}"
                            )
                        })?;
                            // Retained forward inputs (this layer's snapshot).
                            let kv_ret = ssm.wy17_kv_retain.ok_or_else(|| {
                            anyhow::anyhow!(
                                "WY17 lazy replay missing wy17_kv_retain at ssm layer {ssm_layer_idx}"
                            )
                        })?;
                            let gate_ret = ssm.wy17_gate_retain.ok_or_else(|| {
                            anyhow::anyhow!(
                                "WY17 lazy replay missing wy17_gate_retain at ssm layer {ssm_layer_idx}"
                            )
                        })?;
                            let bf16 = 2usize;
                            let fp32 = 4usize;
                            let key_dim = nk * kd;
                            let q_ptr = kv_ret;
                            let k_ptr = kv_ret.offset(key_dim * bf16);
                            let v_ptr = kv_ret.offset(key_dim * 2 * bf16);
                            let gate_ptr = gate_ret;
                            let beta_ptr = gate_ret.offset(nv * fp32);
                            // out_h = live h_state (read by next forward).
                            crate::layers::ops::gdn_wy17_replay(
                                self.gpu.as_ref(),
                                replay_kernel,
                                h_ckpt, // h_root == pre-verify ROOT state
                                q_ptr,
                                k_ptr,
                                v_ptr,
                                gate_ptr,
                                beta_ptr,
                                ssm.h_state, // out_h
                                0,           // ckpt_first_token (root replay)
                                last_inter_slot as u32,
                                1, // batch_size
                                nk as u32,
                                nv as u32,
                                kd as u32,
                                vd as u32,
                                conv_dim as u32, // qk_stride
                                conv_dim as u32, // v_stride
                                (nv * 2) as u32, // gb_stride
                                stream,
                            )?;
                            self.gpu
                                .copy_d2d_async(ssm.h_state, h_ckpt, h_bytes, stream)?;
                            self.gpu.copy_d2d_async(
                                conv_inter,
                                ssm.conv_state,
                                conv_bytes,
                                stream,
                            )?;
                            self.gpu
                                .copy_d2d_async(conv_inter, conv_ckpt, conv_bytes, stream)?;
                        } else if seq_lazy && !was_tree_mode {
                            // GDN LAZY commit (partial accept, or full accept under
                            // `compact_lazy_h`): `h_state` still holds the step-initial
                            // H — replay the identical FP32 recurrence (`nosnap` kernel)
                            // over the first `last_inter_slot + 1` retained tokens in
                            // place. Bit-exact: same kernel arithmetic, same order,
                            // same inputs.
                            let Some(retain) = seq_retain else {
                                bail!(
                                    "GDN lazy commit: retention buffer missing at ssm layer {ssm_layer_idx} \
                                 (dispatch/commit mismatch — refusing to copy garbage intermediates)"
                                );
                            };
                            if seq_replay_kernel.0 == 0 {
                                bail!(
                                    "GDN lazy commit: replay kernel missing (dispatch/commit mismatch)"
                                );
                            };
                            let fp32 = 4usize;
                            let qkvz = self.config.ssm_qkvz_size();
                            let max_k = crate::layers::qwen3_ssm::GDN_LAZY_MAX_K;
                            let key_dim = nk * kd;
                            let gates = retain.offset(max_k * qkvz * fp32);
                            // Dead output target: slot's 3 MiB scratch is unused
                            // during commit and comfortably larger than k*qkvz*4.
                            let out_scratch = self.ssm_pool.h_intermediate(ssm_layer_idx, slot, 0);
                            let splitv4_k = gdn_lazyfinal_splitv4_kernel(self.gpu.as_ref());
                            if gdn_seq_lazy_replay_splitv4_eligible(
                                crate::layers::qwen3_ssm::qwen4_gdn_split_v_enabled_for_single_seq(
                                ),
                                !ssm.h_is_f16,
                                kd,
                                vd,
                                splitv4_k.0 != 0,
                            ) {
                                // 192-CTA Split-V=4 replay on wide discrete GPUs
                                // (`sm_count > 64`): each thread owns a disjoint
                                // column `v_col = sv * 32 + tid`, loads `H_reg`
                                // from `ssm.h_state` before the token loop, and
                                // writes back once to `h_inter == ssm.h_state`
                                // (`inter_stride = 0`). 0-ULP bit-exact vs `nosnap`.
                                crate::layers::ops::gdn_decode_f32_sequence_persistent_splitv4(
                                    self.gpu.as_ref(),
                                    splitv4_k,
                                    ssm.h_state,
                                    retain,
                                    retain.offset(key_dim * fp32),
                                    retain.offset(key_dim * 2 * fp32),
                                    gates,
                                    gates.offset(nv * fp32),
                                    out_scratch,
                                    ssm.h_state,
                                    (last_inter_slot + 1) as u32,
                                    nk as u32,
                                    nv as u32,
                                    kd as u32,
                                    vd as u32,
                                    qkvz as u32,
                                    qkvz as u32,
                                    (nv * 2) as u32,
                                    qkvz as u32,
                                    0,
                                    stream,
                                )?;
                            } else {
                                crate::layers::ops::gdn_decode_f32_sequence(
                                    self.gpu.as_ref(),
                                    seq_replay_kernel,
                                    ssm.h_state, // in/out: starts at step-initial
                                    retain,
                                    retain.offset(key_dim * fp32),
                                    retain.offset(key_dim * 2 * fp32),
                                    gates,
                                    gates.offset(nv * fp32),
                                    out_scratch,
                                    out_scratch, // h_inter arg: nosnap never writes it
                                    (last_inter_slot + 1) as u32,
                                    nk as u32,
                                    nv as u32,
                                    kd as u32,
                                    vd as u32,
                                    qkvz as u32,
                                    qkvz as u32,
                                    (nv * 2) as u32,
                                    qkvz as u32,
                                    (h_bytes / fp32) as u32,
                                    stream,
                                )?;
                            }
                            if num_accepted < k {
                                self.gpu.copy_d2d_async(
                                    conv_inter,
                                    ssm.conv_state,
                                    conv_bytes,
                                    stream,
                                )?;
                            }
                            if no_mirror_req || h_ckpt_opt.is_none() {
                                any_h_ckpt_skipped = true;
                                all_h_ckpt_written = false;
                            } else if let Some(h_ckpt) = h_ckpt_opt {
                                self.gpu
                                    .copy_d2d_async(ssm.h_state, h_ckpt, h_bytes, stream)?;
                            }
                            let conv_src = if num_accepted < k {
                                conv_inter
                            } else {
                                ssm.conv_state
                            };
                            self.gpu
                                .copy_d2d_async(conv_src, conv_ckpt, conv_bytes, stream)?;
                        } else {
                            let h_ckpt = h_ckpt_opt.ok_or_else(|| {
                            anyhow::anyhow!(
                                "commit_verify_state_async: missing h_state_checkpoint at ssm layer {ssm_layer_idx}"
                            )
                        })?;
                            let h_inter =
                                self.ssm_pool
                                    .h_intermediate(ssm_layer_idx, slot, last_inter_slot);
                            if !skip_restore {
                                self.gpu
                                    .copy_d2d_async(h_inter, ssm.h_state, h_bytes, stream)?;
                                self.gpu.copy_d2d_async(
                                    conv_inter,
                                    ssm.conv_state,
                                    conv_bytes,
                                    stream,
                                )?;
                                self.gpu.copy_d2d_async(h_inter, h_ckpt, h_bytes, stream)?;
                                self.gpu
                                    .copy_d2d_async(conv_inter, conv_ckpt, conv_bytes, stream)?;
                            }
                        }
                    }

                    ssm_layer_idx += 1;
                }
            }

            if any_h_ckpt_skipped {
                self.mark_h_ckpt_stale(seq.slot_idx);
                log_qwen4_gdn_commit_no_mirror_engaged();
            } else if all_h_ckpt_written && !skip_restore {
                self.clear_h_ckpt_stale(seq.slot_idx);
            }
            if !skip_restore {
                self.clear_conv_ckpt_stale(seq.slot_idx);
                self.mark_live_state_clean(seq.slot_idx);
                super::verify_qwen4_batched::reset_dedup_slot(seq.slot_idx);
            }
        } else {
            super::verify_qwen4_batched::reset_dedup_slot(seq.slot_idx);
        }

        if let Some(ple) = &self.qwen4_ple {
            ple.rollback_and_checkpoint(seq.slot_idx, num_accepted, self.gpu.as_ref(), stream)?;
        }
        self.gpu.record_event(self.secondary_event, stream)?;
        if let Some(t0) = t_probe {
            let enqueue_us = t0.elapsed().as_micros();
            self.gpu.synchronize(stream)?;
            let total_us = t0.elapsed().as_micros();
            tracing::info!(
                "ASYNC_PROBE commit_tail: enqueue={enqueue_us}μs gpu_total={total_us}μs \
                 (num_accepted={num_accepted} k={k})",
            );
        }
        // QSA's physical cache is not visibility-bounded like ordinary KV:
        // five speculative writes alias its four-row raw ring and endpoint
        // pooling persists compressed keys. Consume its journal only after
        // every recurrent/PLE enqueue and validation has succeeded. Keeping
        // this as the final action lets scheduler error paths still issue a
        // full zero-accept rollback after any earlier failure.
        self.dflash_flat_tree_route
            .store(false, std::sync::atomic::Ordering::Release);
        self.commit_qwen4_qsa_k5_only_dispatch(num_accepted, k)?;
        if keep_late_ready {
            self.record_keep_late_slot(seq, num_accepted);
        }
        Ok(())
    }
}

const QWEN4_GDN_COMMIT_NO_MIRROR_SELECTOR: &str = "ATLAS_QWEN4_GDN_COMMIT_NO_MIRROR";
const SSM_GDN_COMMIT_NO_MIRROR_ENV: &str = "ATLAS_SSM_GDN_COMMIT_NO_MIRROR";

fn parse_ssm_gdn_no_mirror_env(raw: Option<&str>) -> bool {
    !matches!(
        raw.map(str::trim),
        Some("0" | "false" | "FALSE" | "off" | "OFF")
    )
}

fn ssm_gdn_commit_no_mirror_default_on() -> bool {
    static GATE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *GATE.get_or_init(|| {
        parse_ssm_gdn_no_mirror_env(std::env::var(SSM_GDN_COMMIT_NO_MIRROR_ENV).ok().as_deref())
    })
}

/// Whether `seq_lazy` flat-chain commits skip the dead `ssm.h_state -> h_ckpt`
/// mirror copy (144 MiB/step on 48-SSM-layer dense models; 108 MiB/step on
/// 36-SSM-layer models).
///
/// Under `seq_lazy`, the verify kernel only reads `ssm.h_state` and never
/// mutates it, so `ssm.h_state` already holds the canonical recurrent state.
/// Qwen4 models preserve their existing `qwen4_no_mirror_req` policy; all
/// other hybrid GDN models enable `no_mirror` by default unless
/// `ATLAS_SSM_GDN_COMMIT_NO_MIRROR=0`.
#[must_use]
const fn seq_lazy_no_mirror_enabled(
    is_qwen4: bool,
    qwen4_no_mirror_req: bool,
    non_qwen4_default_on: bool,
) -> bool {
    if is_qwen4 {
        qwen4_no_mirror_req
    } else {
        non_qwen4_default_on
    }
}

/// Whether `seq_lazy` commit replay uses the 192-CTA Split-V=4
/// `gated_delta_rule_decode_f32_sequence_lazyfinal_splitv4` kernel instead of
/// the 48-CTA `nosnap` kernel.
#[must_use]
const fn gdn_seq_lazy_replay_splitv4_eligible(
    split_v_enabled: bool,
    h_is_fp32: bool,
    kd: usize,
    vd: usize,
    has_splitv4_kernel: bool,
) -> bool {
    split_v_enabled && h_is_fp32 && kd == 128 && vd == 128 && has_splitv4_kernel
}

struct GdnLazyfinalSplitv4Cache {
    primary: std::sync::OnceLock<(usize, KernelHandle)>,
    fallback: std::sync::OnceLock<Mutex<HashMap<usize, KernelHandle>>>,
}

impl GdnLazyfinalSplitv4Cache {
    const fn new() -> Self {
        Self {
            primary: std::sync::OnceLock::new(),
            fallback: std::sync::OnceLock::new(),
        }
    }

    fn get_or_load(&self, gpu: &dyn GpuBackend) -> KernelHandle {
        let key = std::ptr::from_ref(gpu) as *const () as usize;
        if let Some(&(cached_key, cached_val)) = self.primary.get() {
            if cached_key == key {
                return cached_val;
            }
        } else {
            let val = crate::layers::try_kernel(
                gpu,
                "gated_delta_rule",
                "gated_delta_rule_decode_f32_sequence_lazyfinal_splitv4",
            );
            let &(stored_key, stored_val) = self.primary.get_or_init(|| (key, val));
            if stored_key == key {
                return stored_val;
            }
            let map = self.fallback.get_or_init(|| Mutex::new(HashMap::new()));
            return *map.lock().entry(key).or_insert(val);
        }
        let map = self.fallback.get_or_init(|| Mutex::new(HashMap::new()));
        if let Some(val) = map.lock().get(&key).copied() {
            return val;
        }
        let val = crate::layers::try_kernel(
            gpu,
            "gated_delta_rule",
            "gated_delta_rule_decode_f32_sequence_lazyfinal_splitv4",
        );
        *map.lock().entry(key).or_insert(val)
    }
}

fn gdn_lazyfinal_splitv4_kernel(gpu: &dyn GpuBackend) -> KernelHandle {
    static CACHE: GdnLazyfinalSplitv4Cache = GdnLazyfinalSplitv4Cache::new();
    CACHE.get_or_load(gpu)
}

fn log_qwen4_gdn_commit_no_mirror_engaged() {
    static ENGAGED: std::sync::Once = std::sync::Once::new();
    ENGAGED.call_once(|| {
        tracing::info!(
            "QWEN4_GDN_COMMIT_NO_MIRROR_ENGAGED selector={QWEN4_GDN_COMMIT_NO_MIRROR_SELECTOR} mode=seq_lazy"
        );
    });
}

/// Per-`TransformerModel` slot-tracking state keyed by `TransformerModel::instance_key()`.
///
/// Using dynamic sets/maps avoids the 256-slot ceiling of fixed bitmasks and
/// prevents aliasing across multiple `TransformerModel` instances in one process.
#[derive(Default)]
struct ModelSlotTracking {
    h_ckpt_stale: HashSet<usize>,
    live_state_clean: HashSet<usize>,
    conv_ckpt_stale: HashSet<usize>,
    /// Per-slot retained verify-graph QSA regime (`false = dense`, `true = sparse`).
    /// Prevents a verify graph retained across requests on the same slot from
    /// replaying under a mismatched `qwen4_qsa_required` regime.
    verify_graph_qsa_regime: HashMap<usize, bool>,
}

static MODEL_SLOT_TRACKING: std::sync::OnceLock<Mutex<HashMap<usize, ModelSlotTracking>>> =
    std::sync::OnceLock::new();

fn with_model_slot_tracking<R>(
    instance_key: usize,
    f: impl FnOnce(&mut ModelSlotTracking) -> R,
) -> R {
    let mut map = MODEL_SLOT_TRACKING
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock();
    f(map.entry(instance_key).or_default())
}

fn read_model_slot_tracking<R: Default>(
    instance_key: usize,
    f: impl FnOnce(&ModelSlotTracking) -> R,
) -> R {
    let Some(mutex) = MODEL_SLOT_TRACKING.get() else {
        return R::default();
    };
    mutex.lock().get(&instance_key).map(f).unwrap_or_default()
}

impl TransformerModel {
    pub(crate) fn clear_model_slot_tracking(&self) {
        if let Some(mutex) = MODEL_SLOT_TRACKING.get() {
            mutex.lock().remove(&self.instance_key());
        }
    }

    fn mark_h_ckpt_stale(&self, slot_idx: usize) {
        with_model_slot_tracking(self.instance_key(), |state| {
            state.h_ckpt_stale.insert(slot_idx);
        });
    }

    fn clear_h_ckpt_stale(&self, slot_idx: usize) {
        with_model_slot_tracking(self.instance_key(), |state| {
            state.h_ckpt_stale.remove(&slot_idx);
        });
    }

    fn is_h_ckpt_stale(&self, slot_idx: usize) -> bool {
        read_model_slot_tracking(self.instance_key(), |state| {
            state.h_ckpt_stale.contains(&slot_idx)
        })
    }

    fn mark_live_state_clean(&self, slot_idx: usize) {
        with_model_slot_tracking(self.instance_key(), |state| {
            state.live_state_clean.insert(slot_idx);
        });
    }

    fn clear_live_state_clean(&self, slot_idx: usize) {
        with_model_slot_tracking(self.instance_key(), |state| {
            state.live_state_clean.remove(&slot_idx);
        });
    }

    fn take_live_state_clean(&self, slot_idx: usize) -> bool {
        with_model_slot_tracking(self.instance_key(), |state| {
            state.live_state_clean.remove(&slot_idx)
        })
    }

    fn mark_conv_ckpt_stale(&self, slot_idx: usize) {
        with_model_slot_tracking(self.instance_key(), |state| {
            state.conv_ckpt_stale.insert(slot_idx);
        });
    }

    fn clear_conv_ckpt_stale(&self, slot_idx: usize) {
        with_model_slot_tracking(self.instance_key(), |state| {
            state.conv_ckpt_stale.remove(&slot_idx);
        });
    }

    fn is_conv_ckpt_stale(&self, slot_idx: usize) -> bool {
        read_model_slot_tracking(self.instance_key(), |state| {
            state.conv_ckpt_stale.contains(&slot_idx)
        })
    }

    pub(super) fn note_verify_graph_retained_qsa(&self, slot_idx: usize, qsa_required: bool) {
        with_model_slot_tracking(self.instance_key(), |state| {
            state.verify_graph_qsa_regime.insert(slot_idx, qsa_required);
        });
    }

    pub(super) fn clear_verify_graph_qsa_regime(&self, slot_idx: usize) {
        with_model_slot_tracking(self.instance_key(), |state| {
            state.verify_graph_qsa_regime.remove(&slot_idx);
        });
    }

    fn retained_verify_graph_qsa_mismatches(&self, slot_idx: usize, qsa_required: bool) -> bool {
        read_model_slot_tracking(self.instance_key(), |state| {
            state
                .verify_graph_qsa_regime
                .get(&slot_idx)
                .is_some_and(|&recorded| recorded != qsa_required)
        })
    }
}

thread_local! {
    static IN_VERIFY_ABORT: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

struct VerifyAbortGuard(bool);

impl VerifyAbortGuard {
    fn enter() -> Self {
        let prev = IN_VERIFY_ABORT.with(|flag| flag.replace(true));
        Self(prev)
    }
}

impl Drop for VerifyAbortGuard {
    fn drop(&mut self) {
        IN_VERIFY_ABORT.with(|flag| flag.set(self.0));
    }
}

fn in_verify_abort() -> bool {
    IN_VERIFY_ABORT.with(std::cell::Cell::get)
}

fn log_qwen4_gdn_keep_late_engaged() {
    static ENGAGED: std::sync::Once = std::sync::Once::new();
    ENGAGED.call_once(|| {
        tracing::info!(
            "QWEN4_GDN_KEEP_LATE_ENGAGED selector={} zero_replay=true ring=compact_staggered flush_period={}",
            crate::layers::qwen3_ssm::GDN_KEEP_LATE_ENV,
            crate::layers::qwen3_ssm::gdn_keep_late_flush_period()
        );
    });
}

impl TransformerModel {
    /// Eagerly pre-allocate every SSM layer's lazy-retain pool (and Qwen4
    /// cross-sequence descriptor table) outside CUDA graph capture so
    /// `verify_d` and batched verify never trigger lazy `gpu.alloc` inside
    /// `CaptureGuard`.
    fn prewarm_gdn_lazy_retain_pools(&self, k_hint: usize) -> Result<()> {
        if k_hint == 0
            || !self.ssm_pool.has_mtp
            || spark_runtime::gpu_capture_region::active_capture_region().is_some()
            || self.gpu.stream_is_capturing(self.gpu.default_stream())
        {
            return Ok(());
        }
        for (i, layer) in self.layers.iter().enumerate() {
            if self.config.layer_type(i) != LayerType::LinearAttention {
                continue;
            }
            if layer.gdn_seq_lazy_engaged(k_hint) {
                layer.ensure_qwen4_cross_seq_fused_descs_ready(
                    self.gpu.as_ref(),
                    &self.config,
                    k_hint,
                )?;
            }
        }
        Ok(())
    }

    pub(super) fn has_gdn_keep_late_pending(&self, slot_idx: usize) -> bool {
        self.layers.iter().enumerate().any(|(i, layer)| {
            self.config.layer_type(i) == LayerType::LinearAttention
                && layer.gdn_keep_late_has_pending(slot_idx)
        })
    }

    pub(super) fn reset_gdn_keep_late_slot(&self, slot_idx: usize) {
        self.clear_conv_ckpt_stale(slot_idx);
        invalidate_keep_late_desc_cache_for_model(std::ptr::from_ref(self) as usize);
        for (i, layer) in self.layers.iter().enumerate() {
            if self.config.layer_type(i) == LayerType::LinearAttention {
                layer.gdn_keep_late_reset_slot(slot_idx);
            }
        }
    }

    pub(super) fn flush_gdn_keep_late_slot(
        &self,
        seq: &SequenceState,
        mirror_conv_ckpt: bool,
        stream: u64,
    ) -> Result<()> {
        if !self.has_gdn_keep_late_pending(seq.slot_idx) {
            return Ok(());
        }
        invalidate_keep_late_desc_cache_for_model(std::ptr::from_ref(self) as usize);
        let mut ssm_layer_idx = 0usize;
        for (i, layer_state) in seq.layer_states.iter().enumerate() {
            if self.config.layer_type(i) != LayerType::LinearAttention {
                continue;
            }
            let ssm = layer_state
                .as_any()
                .downcast_ref::<SsmLayerState>()
                .ok_or_else(|| anyhow::anyhow!("Expected SsmLayerState at layer {i}"))?;
            let conv_inter_base = self
                .ssm_pool
                .conv_intermediate(ssm_layer_idx, seq.slot_idx, 0);
            let conv_inter_stride = self.ssm_pool.conv_bytes;
            let h_scratch = self.ssm_pool.h_intermediate(ssm_layer_idx, seq.slot_idx, 0);
            let conv_ckpt = if mirror_conv_ckpt {
                ssm.conv_state_checkpoint
            } else {
                None
            };
            self.layers[i].gdn_keep_late_flush_slot(
                self.gpu.as_ref(),
                &self.config,
                seq.slot_idx,
                ssm.h_state,
                ssm.conv_state,
                conv_ckpt,
                conv_inter_base,
                conv_inter_stride,
                h_scratch,
                stream,
            )?;
            ssm_layer_idx += 1;
        }
        if mirror_conv_ckpt {
            self.clear_conv_ckpt_stale(seq.slot_idx);
        }
        Ok(())
    }

    fn note_gdn_keep_late_verify_completed(&self, slot_idx: usize) {
        for (i, layer) in self.layers.iter().enumerate() {
            if self.config.layer_type(i) == LayerType::LinearAttention {
                layer.gdn_keep_late_note_verify_completed(slot_idx);
            }
        }
    }

    fn can_stage_keep_late_for_frame(&self, k: usize, is_batch: bool) -> bool {
        let min_k = if is_batch { 1 } else { 2 };
        if !self.config.is_qwen4_exp()
            || !self.ssm_pool.compact_lazy_h
            || !crate::layers::qwen3_ssm::gdn_keep_late_enabled()
            || !(min_k..=crate::layers::qwen3_ssm::GDN_LAZY_MAX_K).contains(&k)
            || self.commit_runtime_flags.skip_restore_control
            || self.commit_runtime_flags.async_probe
            || self.commit_runtime_flags.bootstrap_trace
            || self.ddtree_parent_ids_dev.lock().is_some()
        {
            return false;
        }
        if !is_batch {
            let legacy_k5 = (2..=crate::layers::QWEN4_HYBRID_MAX_ROWS).contains(&k)
                && crate::layers::qwen4_hybrid_rows(k)
                && std::env::var("ATLAS_QWEN4_K5_BATCH_SSM").ok().as_deref() == Some("1");
            let native_k16 = k == 16
                && std::env::var("ATLAS_QWEN4_K16_BATCHED_VERIFY")
                    .ok()
                    .as_deref()
                    == Some("1")
                && std::env::var("ATLAS_QWEN4_K16_EXACT").ok().as_deref() == Some("1");
            if !legacy_k5 && !native_k16 {
                return false;
            }
        }
        let mut saw_ssm = false;
        for (i, layer) in self.layers.iter().enumerate() {
            if self.config.layer_type(i) != LayerType::LinearAttention {
                continue;
            }
            saw_ssm = true;
            if !layer.gdn_keep_late_supported() || !layer.gdn_seq_lazy_engaged(k) {
                return false;
            }
        }
        saw_ssm
    }

    fn collect_keep_late_ctrl_for_slot(
        &self,
        seq: &SequenceState,
        out: &mut Vec<(DevicePtr, [u8; 32])>,
    ) -> Result<bool> {
        let start_len = out.len();
        for (i, layer_state) in seq.layer_states.iter().enumerate() {
            if self.config.layer_type(i) != LayerType::LinearAttention {
                continue;
            }
            let ssm = layer_state
                .as_any()
                .downcast_ref::<SsmLayerState>()
                .ok_or_else(|| anyhow::anyhow!("Expected SsmLayerState at layer {i}"))?;
            let Some(ctrl) = self.layers[i].stage_qwen4_keep_late_ctrl_bytes(
                self.gpu.as_ref(),
                &self.config,
                seq.slot_idx,
                ssm.h_state,
            )?
            else {
                out.truncate(start_len);
                return Ok(false);
            };
            out.push(ctrl);
        }
        Ok(out.len() > start_len)
    }

    fn upload_coalesced_keep_late_ctrl(
        &self,
        ctrl_payloads: &mut [(DevicePtr, [u8; 32])],
        desc_payloads: &[(DevicePtr, [u8; 64])],
        stream: u64,
    ) -> Result<()> {
        let model_id = std::ptr::from_ref(self) as usize;
        let mut scratch = KEEP_LATE_H2D_HOST_SCRATCH
            .get_or_init(|| Mutex::new(KeepLateH2dHostScratch::default()))
            .lock();
        let KeepLateH2dHostScratch {
            packed,
            runs,
            desc_dirty,
            desc_packed,
            desc_runs,
            last_uploaded_desc,
            ..
        } = &mut *scratch;
        filter_dirty_keep_late_descs(
            model_id,
            ctrl_payloads,
            desc_payloads,
            last_uploaded_desc,
            desc_dirty,
        );
        coalesce_keep_late_ctrl_runs(ctrl_payloads, packed, runs);
        coalesce_keep_late_desc_runs(desc_dirty, desc_packed, desc_runs);
        let mut copies = Vec::with_capacity(runs.len() + desc_runs.len());
        for &(dst, start, len) in runs.iter() {
            copies.push(spark_runtime::gpu::HostToDeviceCopy::new(
                &packed[start..start + len],
                dst,
            ));
        }
        for &(dst, start, len) in desc_runs.iter() {
            copies.push(spark_runtime::gpu::HostToDeviceCopy::new(
                &desc_packed[start..start + len],
                dst,
            ));
        }
        self.copy_h2d_group_fast(&copies, stream)?;
        // Record uploaded descriptors ONLY after `copy_h2d_group_fast` succeeds so
        // a failed H2D never marks an un-uploaded descriptor as resident.
        for &(dst, bytes) in desc_dirty.iter() {
            last_uploaded_desc.insert((model_id, dst.0), bytes);
        }
        Ok(())
    }

    fn stage_keep_late_for_slot(
        &self,
        seq: &SequenceState,
        k: usize,
        is_batch: bool,
        stream: u64,
    ) -> Result<bool> {
        // Single-seq Split-V fused GDN operates exclusively on canonical FP32
        // `h_state`. If `seq` just dropped out of a `C >= 2` cohort with an
        // in-place Scaled-FP16 `h_state`, unpack it via the non-split cold
        // flush once before staging the `C = 1` descriptor so even a
        // pre-captured `C = 1` Split-V CUDA graph never observes `LOAD_SCALED_FP16`.
        if !is_batch
            && crate::layers::qwen3_ssm::qwen4_gdn_split_v_enabled_for_single_seq()
            && self.layers.iter().enumerate().any(|(i, layer)| {
                self.config.layer_type(i) == LayerType::LinearAttention
                    && layer.gdn_keep_late_has_scaled_fp16(seq.slot_idx)
            })
        {
            self.flush_gdn_keep_late_slot(seq, false, stream)?;
        }
        let mut ctrl_payloads: Vec<(DevicePtr, [u8; 32])> =
            Vec::with_capacity(self.ssm_pool.num_ssm_layers);
        let mut desc_payloads: Vec<(DevicePtr, [u8; 64])> = if is_batch {
            Vec::new()
        } else {
            Vec::with_capacity(self.ssm_pool.num_ssm_layers)
        };
        let mut ssm_layer_idx = 0usize;
        for (i, layer_state) in seq.layer_states.iter().enumerate() {
            if self.config.layer_type(i) != LayerType::LinearAttention {
                continue;
            }
            let ssm = layer_state
                .as_any()
                .downcast_ref::<SsmLayerState>()
                .ok_or_else(|| anyhow::anyhow!("Expected SsmLayerState at layer {i}"))?;
            let Some(ctrl) = self.layers[i].stage_qwen4_keep_late_ctrl_bytes(
                self.gpu.as_ref(),
                &self.config,
                seq.slot_idx,
                ssm.h_state,
            )?
            else {
                return Ok(false);
            };
            ctrl_payloads.push(ctrl);
            if !is_batch {
                let conv_inter_base =
                    self.ssm_pool
                        .conv_intermediate(ssm_layer_idx, seq.slot_idx, 0);
                let Some(desc) = self.layers[i].stage_qwen4_single_seq_fused_desc_bytes(
                    self.gpu.as_ref(),
                    &self.config,
                    seq.slot_idx,
                    k,
                    ssm.h_state,
                    ssm.conv_state,
                    conv_inter_base,
                    stream,
                )?
                else {
                    return Ok(false);
                };
                desc_payloads.push(desc);
            }
            ssm_layer_idx += 1;
        }
        if ctrl_payloads.is_empty() {
            return Ok(false);
        }
        self.upload_coalesced_keep_late_ctrl(&mut ctrl_payloads, &desc_payloads, stream)?;
        Ok(true)
    }

    fn can_commit_keep_late_slot(
        &self,
        seq: &SequenceState,
        num_accepted: usize,
        k: usize,
        was_tree_mode: bool,
    ) -> bool {
        if !self.ssm_pool.compact_lazy_h
            || !self.config.is_qwen4_exp()
            || !crate::layers::qwen3_ssm::gdn_keep_late_enabled()
            || was_tree_mode
            || num_accepted == 0
            || num_accepted > k
            || num_accepted > crate::layers::qwen3_ssm::GDN_LAZY_MAX_K
            || self.commit_runtime_flags.skip_restore_control
            || self.commit_runtime_flags.async_probe
            || self.commit_runtime_flags.bootstrap_trace
        {
            return false;
        }
        let mut saw_ssm = false;
        for (i, layer) in self.layers.iter().enumerate() {
            if self.config.layer_type(i) != LayerType::LinearAttention {
                continue;
            }
            saw_ssm = true;
            if !layer.gdn_keep_late_armed(seq.slot_idx) || !layer.gdn_seq_lazy_engaged(k) {
                return false;
            }
        }
        saw_ssm
    }

    fn record_keep_late_slot(&self, seq: &SequenceState, num_accepted: usize) {
        let mut ssm_layer_idx = 0usize;
        for (i, layer_state) in seq.layer_states.iter().enumerate() {
            if self.config.layer_type(i) != LayerType::LinearAttention {
                continue;
            }
            if let Some(ssm) = layer_state.as_any().downcast_ref::<SsmLayerState>() {
                let conv_inter_base =
                    self.ssm_pool
                        .conv_intermediate(ssm_layer_idx, seq.slot_idx, 0);
                let conv_inter_stride = self.ssm_pool.conv_bytes;
                let h_scratch = self.ssm_pool.h_intermediate(ssm_layer_idx, seq.slot_idx, 0);
                let recorded = self.layers[i].gdn_keep_late_record(
                    seq.slot_idx,
                    num_accepted,
                    ssm.h_state,
                    ssm.conv_state,
                    conv_inter_base,
                    conv_inter_stride,
                    h_scratch,
                );
                debug_assert!(
                    recorded,
                    "KEEP_LATE layer {i} armed check passed but record returned false"
                );
            }
            ssm_layer_idx += 1;
        }
        // Keep `self.decode_graph[seq.slot_idx]` cached across KEEP_LATE
        // commits: every K=1 `decode_graph` replay/capture site (`decode_a.rs`
        // and `decode_qwen4_ple_graph.rs`) unconditionally runs
        // `flush_gdn_keep_late_slot` before entering graph replay, so evicting
        // the 48-layer PLE suffix graph here only forced redundant recaptures.
        self.mark_h_ckpt_stale(seq.slot_idx);
        self.mark_conv_ckpt_stale(seq.slot_idx);
        self.mark_live_state_clean(seq.slot_idx);
        log_qwen4_gdn_keep_late_engaged();
    }

    fn restore_ssm_checkpoints_to_live(&self, seq: &SequenceState, stream: u64) -> Result<()> {
        let skip_h_ckpt_restore =
            self.ssm_pool.compact_lazy_h || self.is_h_ckpt_stale(seq.slot_idx);
        let h_bytes = self.ssm_pool.h_bytes;
        let conv_bytes = self.ssm_pool.conv_bytes;
        for (i, layer_state) in seq.layer_states.iter().enumerate() {
            if self.config.layer_type(i) != LayerType::LinearAttention {
                continue;
            }
            let ssm = layer_state
                .as_any()
                .downcast_ref::<SsmLayerState>()
                .ok_or_else(|| anyhow::anyhow!("Expected SsmLayerState at layer {i}"))?;
            let Some(conv_ckpt) = ssm.conv_state_checkpoint else {
                continue;
            };
            if !self.ssm_pool.compact_lazy_h && ssm.h_state_checkpoint.is_none() {
                continue;
            }
            if !skip_h_ckpt_restore && let Some(h_ckpt) = ssm.h_state_checkpoint {
                self.gpu
                    .copy_d2d_async(h_ckpt, ssm.h_state, h_bytes, stream)?;
            }
            self.gpu
                .copy_d2d_async(conv_ckpt, ssm.conv_state, conv_bytes, stream)?;
        }
        Ok(())
    }

    /// Pre-load `gated_delta_rule_decode_f32_sequence_lazyfinal_splitv4` once
    /// during model initialization so commit-side Split-V=4 replay never loads
    /// the module lazily on the hot path.
    pub(in crate::model) fn warm_gdn_lazyfinal_splitv4_kernel(gpu: &dyn GpuBackend) {
        let _ = gdn_lazyfinal_splitv4_kernel(gpu);
    }

    fn all_ssm_layers_seq_lazy_engaged(&self, k: usize) -> bool {
        let mut saw_ssm = false;
        for (i, layer) in self.layers.iter().enumerate() {
            if self.config.layer_type(i) != LayerType::LinearAttention {
                continue;
            }
            saw_ssm = true;
            if !layer.gdn_seq_lazy_engaged(k) {
                return false;
            }
        }
        saw_ssm
    }

    fn refresh_h_ckpt_from_live(&self, seq: &SequenceState, stream: u64) -> Result<()> {
        let h_bytes = self.ssm_pool.h_bytes;
        for (i, layer_state) in seq.layer_states.iter().enumerate() {
            if self.config.layer_type(i) != LayerType::LinearAttention {
                continue;
            }
            let ssm = layer_state
                .as_any()
                .downcast_ref::<SsmLayerState>()
                .ok_or_else(|| anyhow::anyhow!("Expected SsmLayerState at layer {i}"))?;
            if let Some(h_ckpt) = ssm.h_state_checkpoint {
                self.gpu
                    .copy_d2d_async(ssm.h_state, h_ckpt, h_bytes, stream)?;
            }
        }
        Ok(())
    }

    /// Guard for `no_mirror` slots transitioning to a non-`seq_lazy` verify
    /// (e.g., tree mode or `k > GDN_LAZY_MAX_K` WY17).
    ///
    /// While a slot stays on flat-chain `seq_lazy`, `ssm.h_state` holds the
    /// canonical pre-verify state and `lazyfinal` never mutates it, so
    /// skipping the `144 MiB/step` (`48 × 3 MiB`, or `108 MiB/step` on 36-SSM-layer
    /// models) `ssm.h_state -> h_ckpt` mirror is exact.
    /// Before any verify frame that might mutate `ssm.h_state` in place or
    /// read `h_ckpt` (tree mode / non-`seq_lazy`), re-mirror `ssm.h_state`
    /// into `h_ckpt` and clear the stale flag.
    pub(super) fn ensure_h_ckpt_fresh_if_non_lazy(
        &self,
        seq: &SequenceState,
        k: usize,
        tree_mode: bool,
        stream: u64,
    ) -> Result<()> {
        if self.ssm_pool.compact_lazy_h || !self.is_h_ckpt_stale(seq.slot_idx) {
            return Ok(());
        }
        let stays_seq_lazy = !tree_mode
            && self.ddtree_parent_ids_dev.lock().is_none()
            && self.all_ssm_layers_seq_lazy_engaged(k);
        if stays_seq_lazy {
            return Ok(());
        }
        self.refresh_h_ckpt_from_live(seq, stream)?;
        self.clear_h_ckpt_stale(seq.slot_idx);
        Ok(())
    }

    fn refresh_conv_ckpt_from_live(&self, seq: &SequenceState, stream: u64) -> Result<()> {
        let conv_bytes = self.ssm_pool.conv_bytes;
        for (i, layer_state) in seq.layer_states.iter().enumerate() {
            if self.config.layer_type(i) != LayerType::LinearAttention {
                continue;
            }
            let ssm = layer_state
                .as_any()
                .downcast_ref::<SsmLayerState>()
                .ok_or_else(|| anyhow::anyhow!("Expected SsmLayerState at layer {i}"))?;
            if let Some(conv_ckpt) = ssm.conv_state_checkpoint {
                self.gpu
                    .copy_d2d_async(ssm.conv_state, conv_ckpt, conv_bytes, stream)?;
            }
        }
        Ok(())
    }

    fn invalidate_verify_graphs_for_slot(&self, slot_idx: usize) {
        self.clear_verify_graph_qsa_regime(slot_idx);
        for graph_mutex in [
            &self.verify2_graph,
            &self.verify3_graph,
            &self.verify4_graph,
        ] {
            if let Some(graph) = graph_mutex.lock().remove(&slot_idx)
                && let Err(e) = self.gpu.destroy_graph(graph)
            {
                tracing::error!("qsa_regime_change: destroy_graph(verify[{slot_idx}]): {e:#}");
            }
        }
        let mut cache = self.verify_kgamma_graph.lock();
        let stale_keys =
            super::sequence_graph_cleanup::verify_kgamma_keys_for_slots(&cache, &[slot_idx]);
        super::verify_graph_receipts::reset_slot(slot_idx);
        for key in stale_keys {
            if let Some(graph) = cache.remove(&key)
                && let Err(e) = self.gpu.destroy_graph(graph)
            {
                tracing::error!(
                    "qsa_regime_change: destroy_graph(verify_kgamma_graph[{key:?}]): {e:#}"
                );
            }
        }
    }
}

/// Reusable host staging buffers for coalesced `KEEP_LATE` control + descriptor H2D uploads.
#[derive(Default)]
struct KeepLateH2dHostScratch {
    entries_pool: Vec<(DevicePtr, [u8; 32])>,
    packed: Vec<u8>,
    runs: Vec<(DevicePtr, usize, usize)>,
    desc_dirty: Vec<(DevicePtr, [u8; 64])>,
    desc_packed: Vec<u8>,
    desc_runs: Vec<(DevicePtr, usize, usize)>,
    /// Last successfully uploaded 64-byte `GdnFusedMultiSeqDesc` per `(model_id, desc_dst)`.
    /// Steady-state C=1 rounds keep `pending_rows > 0` with identical descriptor bytes;
    /// skipping unchanged uploads eliminates 36 per-layer H2D calls per round, while
    /// cold flushes (`flush_gdn_keep_late_slot`), slot resets, `pending_rows == 0`, or
    /// any byte change (`k`, `h_format`, pointers) always force a fresh upload.
    last_uploaded_desc: std::collections::HashMap<(usize, u64), [u8; 64]>,
}

static KEEP_LATE_H2D_HOST_SCRATCH: std::sync::OnceLock<Mutex<KeepLateH2dHostScratch>> =
    std::sync::OnceLock::new();

fn invalidate_keep_late_desc_cache_for_model(model_id: usize) {
    if let Some(lock) = KEEP_LATE_H2D_HOST_SCRATCH.get() {
        let mut scratch = lock.lock();
        scratch
            .last_uploaded_desc
            .retain(|&(cached_model, _), _| cached_model != model_id);
    }
}

/// Filter `desc_payloads` to only descriptors that are dirty or may have been
/// overwritten by a cold flush (`pending_rows == 0` in the matching `ctrl[12..16]`).
fn filter_dirty_keep_late_descs(
    model_id: usize,
    ctrl_payloads: &[(DevicePtr, [u8; 32])],
    desc_payloads: &[(DevicePtr, [u8; 64])],
    cache: &std::collections::HashMap<(usize, u64), [u8; 64]>,
    out_dirty: &mut Vec<(DevicePtr, [u8; 64])>,
) {
    out_dirty.clear();
    if desc_payloads.is_empty() {
        return;
    }
    let aligned_ctrl = ctrl_payloads.len() == desc_payloads.len();
    for (idx, &(dst, bytes)) in desc_payloads.iter().enumerate() {
        let pending_rows = if aligned_ctrl {
            let c = &ctrl_payloads[idx].1;
            u32::from_le_bytes([c[12], c[13], c[14], c[15]])
        } else {
            0
        };
        let unchanged_live = pending_rows > 0 && cache.get(&(model_id, dst.0)) == Some(&bytes);
        if !unchanged_live {
            out_dirty.push((dst, bytes));
        }
    }
}

fn coalesce_keep_late_fixed_runs<const N: usize>(
    entries: &mut [(DevicePtr, [u8; N])],
    packed: &mut Vec<u8>,
    runs: &mut Vec<(DevicePtr, usize, usize)>,
) {
    packed.clear();
    runs.clear();
    if entries.is_empty() {
        return;
    }
    entries.sort_unstable_by_key(|(dst, _)| dst.0);
    packed.reserve(entries.len() * N);
    for &(dst, ref bytes) in entries.iter() {
        let start = packed.len();
        packed.extend_from_slice(bytes);
        if let Some(last) = runs.last_mut()
            && last.0.0.checked_add(last.2 as u64) == Some(dst.0)
        {
            last.2 += N;
        } else {
            runs.push((dst, start, N));
        }
    }
}

fn coalesce_keep_late_desc_runs(
    entries: &mut [(DevicePtr, [u8; 64])],
    packed: &mut Vec<u8>,
    runs: &mut Vec<(DevicePtr, usize, usize)>,
) {
    coalesce_keep_late_fixed_runs::<64>(entries, packed, runs);
}

fn take_keep_late_entries_buf(capacity: usize) -> Vec<(DevicePtr, [u8; 32])> {
    let mut buf = {
        let mut scratch = KEEP_LATE_H2D_HOST_SCRATCH
            .get_or_init(|| Mutex::new(KeepLateH2dHostScratch::default()))
            .lock();
        std::mem::take(&mut scratch.entries_pool)
    };
    buf.clear();
    buf.reserve(capacity);
    buf
}

fn return_keep_late_entries_buf(mut buf: Vec<(DevicePtr, [u8; 32])>) {
    buf.clear();
    let mut scratch = KEEP_LATE_H2D_HOST_SCRATCH
        .get_or_init(|| Mutex::new(KeepLateH2dHostScratch::default()))
        .lock();
    if buf.capacity() > scratch.entries_pool.capacity() {
        scratch.entries_pool = buf;
    }
}

/// Sort `entries` by destination GPU address and coalesce exact contiguous
/// 32-byte control blocks into `packed` + `runs: Vec<(dst, byte_start, byte_len)>`.
///
/// On the shared slot-major `[lanes][num_ssm_layers][32 B]` slab:
/// - `C=1` (any single slot) collapses `36 -> 1` H2D (`1,152 B`).
/// - Contiguous `C=2..16` slots (in any permutation) collapse `36*C -> 1` H2D
///   (up to `18,432 B` at `C=16`).
/// - Active slot sets with holes (e.g. `{0, 1, 4, 5}`) emit one contiguous span
///   per occupied run, never overwriting inactive slots' GPU memory.
fn coalesce_keep_late_ctrl_runs(
    entries: &mut [(DevicePtr, [u8; 32])],
    packed: &mut Vec<u8>,
    runs: &mut Vec<(DevicePtr, usize, usize)>,
) {
    coalesce_keep_late_fixed_runs::<32>(entries, packed, runs);
}

#[cfg(test)]
mod keep_late_batch_h2d_tests {
    use super::coalesce_keep_late_ctrl_runs;
    use crate::layers::qwen3_ssm::keep_late_ctrl_slab_offset;
    use spark_runtime::gpu::DevicePtr;

    fn make_ctrl(slot: usize, layer: usize) -> [u8; 32] {
        let mut bytes = [0u8; 32];
        bytes[0..8].copy_from_slice(&((slot as u64) << 16 | layer as u64).to_le_bytes());
        bytes[28..32].copy_from_slice(&1u32.to_le_bytes());
        bytes
    }

    #[test]
    fn coalesce_keep_late_collapses_c16_x_36_into_one_h2d_and_preserves_holes() {
        let slab = DevicePtr(0x1_0000_0000);
        let num_layers = 36usize;

        // 1. C=16 contiguous slots (0..16) presented in shuffled order -> 1 H2D (18,432 B).
        let shuffled_slots = [15, 0, 7, 2, 14, 1, 9, 3, 13, 4, 8, 5, 12, 6, 11, 10];
        let mut entries = Vec::with_capacity(16 * num_layers);
        for &slot in &shuffled_slots {
            for layer in 0..num_layers {
                let off = keep_late_ctrl_slab_offset(slot, layer, num_layers).unwrap();
                entries.push((slab.offset(off), make_ctrl(slot, layer)));
            }
        }
        let mut packed = Vec::new();
        let mut runs = Vec::new();
        coalesce_keep_late_ctrl_runs(&mut entries, &mut packed, &mut runs);
        assert_eq!(runs, vec![(slab, 0, 16 * num_layers * 32)]);
        assert_eq!(packed.len(), 18_432);
        for slot in 0..16usize {
            for layer in 0..num_layers {
                let off = keep_late_ctrl_slab_offset(slot, layer, num_layers).unwrap();
                assert_eq!(&packed[off..off + 32], &make_ctrl(slot, layer));
            }
        }

        // 2. Non-contiguous active slots with holes ({0, 1, 4, 5, 10}) -> 3 disjoint spans;
        //    hole slots (2, 3, 6, 7, 8, 9) are completely excluded (zero writes).
        let mut hole_entries = Vec::new();
        for &slot in &[5usize, 0, 10, 1, 4] {
            for layer in 0..num_layers {
                let off = keep_late_ctrl_slab_offset(slot, layer, num_layers).unwrap();
                hole_entries.push((slab.offset(off), make_ctrl(slot, layer)));
            }
        }
        coalesce_keep_late_ctrl_runs(&mut hole_entries, &mut packed, &mut runs);
        let slot_bytes = num_layers * 32;
        assert_eq!(
            runs,
            vec![
                (slab.offset(0), 0, 2 * slot_bytes),
                (slab.offset(4 * slot_bytes), 2 * slot_bytes, 2 * slot_bytes),
                (slab.offset(10 * slot_bytes), 4 * slot_bytes, slot_bytes),
            ]
        );
        // Verify simulated device slab retains sentinel bytes in all hole slots.
        let mut sim_gpu = vec![0xAAu8; 16 * slot_bytes];
        for &(dst, start, len) in &runs {
            let gpu_off = (dst.0 - slab.0) as usize;
            sim_gpu[gpu_off..gpu_off + len].copy_from_slice(&packed[start..start + len]);
        }
        for slot in [2usize, 3, 6, 7, 8, 9, 11, 12, 13, 14, 15] {
            let off = slot * slot_bytes;
            assert!(
                sim_gpu[off..off + slot_bytes].iter().all(|&b| b == 0xAA),
                "inactive hole slot {slot} must remain untouched on GPU"
            );
        }
        for slot in [0usize, 1, 4, 5, 10] {
            for layer in 0..num_layers {
                let off = keep_late_ctrl_slab_offset(slot, layer, num_layers).unwrap();
                assert_eq!(&sim_gpu[off..off + 32], &make_ctrl(slot, layer));
            }
        }

        // 3. Single slot C=1 (slot 7) -> 1 H2D (1,152 B).
        let mut c1_entries = (0..num_layers)
            .map(|layer| {
                let off = keep_late_ctrl_slab_offset(7, layer, num_layers).unwrap();
                (slab.offset(off), make_ctrl(7, layer))
            })
            .collect::<Vec<_>>();
        coalesce_keep_late_ctrl_runs(&mut c1_entries, &mut packed, &mut runs);
        assert_eq!(runs, vec![(slab.offset(7 * slot_bytes), 0, slot_bytes)]);
    }

    #[test]
    fn seq_lazy_no_mirror_enabled_for_non_qwen4_and_preserves_qwen4_policy() {
        use super::{
            gdn_seq_lazy_replay_splitv4_eligible, parse_ssm_gdn_no_mirror_env,
            seq_lazy_no_mirror_enabled,
        };

        assert!(parse_ssm_gdn_no_mirror_env(None));
        assert!(parse_ssm_gdn_no_mirror_env(Some("1")));
        assert!(!parse_ssm_gdn_no_mirror_env(Some("0")));
        assert!(!parse_ssm_gdn_no_mirror_env(Some("false")));
        assert!(!parse_ssm_gdn_no_mirror_env(Some("OFF")));

        // Non-Qwen4 (e.g. Qwen3.8-27B): enabled by default on seq_lazy, opt-outable.
        assert!(seq_lazy_no_mirror_enabled(false, false, true));
        assert!(!seq_lazy_no_mirror_enabled(false, true, false));
        // Qwen4 preserves its existing `qwen4_no_mirror_req` policy verbatim.
        assert!(!seq_lazy_no_mirror_enabled(true, false, true));
        assert!(seq_lazy_no_mirror_enabled(true, true, false));

        // Split-V=4 commit replay requires split_v policy, FP32 H, kd==vd==128, kernel handle.
        assert!(gdn_seq_lazy_replay_splitv4_eligible(
            true, true, 128, 128, true
        ));
        assert!(!gdn_seq_lazy_replay_splitv4_eligible(
            false, true, 128, 128, true
        ));
        assert!(!gdn_seq_lazy_replay_splitv4_eligible(
            true, false, 128, 128, true
        ));
        assert!(!gdn_seq_lazy_replay_splitv4_eligible(
            true, true, 64, 128, true
        ));
        assert!(!gdn_seq_lazy_replay_splitv4_eligible(
            true, true, 128, 128, false
        ));
    }

    #[test]
    fn keep_late_desc_filter_skips_steady_state_uploads_and_reuploads_on_change_or_flush() {
        use super::{coalesce_keep_late_desc_runs, filter_dirty_keep_late_descs};
        use std::collections::HashMap;

        let model_id = 0xDEAD_BEEFusize;
        let num_layers = 36usize;
        let desc_base = DevicePtr(0x2_0000_0000);

        let make_ctrl_pending = |pending_rows: u32, layer: usize| -> (DevicePtr, [u8; 32]) {
            let mut b = [0u8; 32];
            b[12..16].copy_from_slice(&pending_rows.to_le_bytes());
            (DevicePtr(0x1_0000_0000 + (layer as u64) * 32), b)
        };
        let make_desc = |k_tag: u8, layer: usize| -> (DevicePtr, [u8; 64]) {
            let mut b = [k_tag; 64];
            b[0] = layer as u8;
            (desc_base.offset(layer * 64), b)
        };

        let mut cache: HashMap<(usize, u64), [u8; 64]> = HashMap::new();
        let mut dirty = Vec::new();
        let mut packed = Vec::new();
        let mut runs = Vec::new();

        // Round 0 (pending_rows == 0): uploads all 36 descs; contiguous slab coalesces to 1 run (2,304 B).
        let ctrls0: Vec<_> = (0..num_layers).map(|l| make_ctrl_pending(0, l)).collect();
        let descs_k8: Vec<_> = (0..num_layers).map(|l| make_desc(8, l)).collect();
        filter_dirty_keep_late_descs(model_id, &ctrls0, &descs_k8, &cache, &mut dirty);
        assert_eq!(dirty.len(), 36);
        coalesce_keep_late_desc_runs(&mut dirty, &mut packed, &mut runs);
        assert_eq!(runs, vec![(desc_base, 0, 36 * 64)]);
        assert_eq!(packed.len(), 2_304);
        for &(dst, bytes) in &dirty {
            cache.insert((model_id, dst.0), bytes);
        }

        // Steady-state round (pending_rows > 0, same k=8 descs): skips ALL 36 desc H2Ds!
        let ctrls_live: Vec<_> = (0..num_layers).map(|l| make_ctrl_pending(5, l)).collect();
        filter_dirty_keep_late_descs(model_id, &ctrls_live, &descs_k8, &cache, &mut dirty);
        assert!(
            dirty.is_empty(),
            "steady-state C=1 round must skip all 36 unchanged descs"
        );

        // Width change (k=8 -> k=5) with pending_rows > 0: re-uploads all 36 descs!
        let descs_k5: Vec<_> = (0..num_layers).map(|l| make_desc(5, l)).collect();
        filter_dirty_keep_late_descs(model_id, &ctrls_live, &descs_k5, &cache, &mut dirty);
        assert_eq!(dirty.len(), 36);
        for &(dst, bytes) in &dirty {
            cache.insert((model_id, dst.0), bytes);
        }

        // Cold flush resets pending_rows == 0 -> forces full re-upload even with identical bytes!
        filter_dirty_keep_late_descs(model_id, &ctrls0, &descs_k5, &cache, &mut dirty);
        assert_eq!(dirty.len(), 36);

        // Disjoint per-layer allocations coalesce adjacent pairs without corrupting gaps.
        let mut disjoint_descs = vec![
            (DevicePtr(0x4000_0000), [1u8; 64]),
            (DevicePtr(0x4000_0040), [2u8; 64]),
            (DevicePtr(0x4000_1000), [3u8; 64]),
        ];
        coalesce_keep_late_desc_runs(&mut disjoint_descs, &mut packed, &mut runs);
        assert_eq!(
            runs,
            vec![
                (DevicePtr(0x4000_0000), 0, 128),
                (DevicePtr(0x4000_1000), 128, 64),
            ]
        );
    }
}
