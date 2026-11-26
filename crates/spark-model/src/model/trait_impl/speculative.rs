// SPDX-License-Identifier: AGPL-3.0-only

#![allow(unused_imports, dead_code, clippy::too_many_arguments)]

use parking_lot::Mutex;
use std::collections::HashMap;
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
    pub(super) fn save_hidden_for_mtp_sequence_dispatch(
        &self,
        token_idx: usize,
        seq: &SequenceState,
        stream: u64,
    ) -> Result<()> {
        let native = self.active_proposer().is_some_and(|proposer| {
            proposer.kind() == crate::speculative::DraftProposerKind::Qwen4NativeMtp
        });
        if !native {
            return self.save_hidden_for_mtp_dispatch(token_idx, stream);
        }
        self.ensure_verify_sequence_idle(seq)?;
        anyhow::ensure!(
            stream == 0 || stream == self.gpu.default_stream(),
            "save_hidden_for_mtp_sequence requires the default stream (got {stream})"
        );
        let native_state = seq
            .proposer_state
            .as_ref()
            .and_then(|state| {
                state
                    .as_any()
                    .downcast_ref::<crate::layers::qwen4_mtp::Qwen4MtpState>()
            })
            .ok_or_else(|| anyhow::anyhow!("native MTP has no sequence-owned proposer state"))?;
        if let Some(frame) = seq.verify_frame {
            anyhow::ensure!(
                frame.accepted() == token_idx.checked_add(1),
                "MTP hidden save does not match the committed verify prefix"
            );
            if crate::layers::qwen4_mtp_rows::recondition_engaged() {
                native_state.check_staged_frame(frame.ticket(), token_idx)?;
            }
        }
        let (storage, frames) = self.qwen4_mtp_hidden_frames.as_ref().ok_or_else(|| {
            anyhow::anyhow!("native MTP has no sequence-owned target-hidden storage")
        })?;
        let mut frames = frames.lock();
        let write = frames
            .begin_write(
                seq.slot_idx,
                seq.session_hash,
                seq.seq_len,
                token_idx,
                self.buffers.sizes().hidden_states,
            )
            .map_err(anyhow::Error::msg)?;
        // Keep legacy EP metadata and optional committed-row recondition
        // bookkeeping synchronized. Native propose reads only the
        // sequence-owned storage below.
        self.save_hidden_for_mtp_dispatch(token_idx, stream)?;
        self.gpu.copy_d2d_async(
            self.buffers.hidden_states(),
            storage.offset(write.retained.prefix.offset),
            write.retained.prefix.bytes,
            self.gpu.default_stream(),
        )?;
        frames.publish(write).map_err(anyhow::Error::msg)?;
        if let Some(frame) = seq.verify_frame
            && crate::layers::qwen4_mtp_rows::recondition_engaged()
        {
            native_state.publish_staged_frame(frame.ticket(), token_idx)?;
        }
        Ok(())
    }

    /// [`Self::save_hidden_for_mtp_sequence_dispatch`] for a sequence of a
    /// cross-sequence verify frame whose rows start at `row_base`.
    pub(super) fn save_hidden_for_mtp_sequence_in_frame_dispatch(
        &self,
        row_base: usize,
        token_idx: usize,
        seq: &SequenceState,
        stream: u64,
    ) -> Result<()> {
        let rows = Self::batch_lane_rows(seq)
            .ok_or_else(|| anyhow::anyhow!("batch hidden publication has no lane ticket"))?;
        let lane = self
            .batch_lane(seq, rows, false)?
            .ok_or_else(|| anyhow::anyhow!("batch hidden publication has no lane ticket"))?;
        let accepted = token_idx
            .checked_add(1)
            .ok_or_else(|| anyhow::anyhow!("batch hidden accepted prefix overflow"))?;
        {
            let batch = self.verify_batch_transactions.lock();
            anyhow::ensure!(seq.verify_frame.and_then(|frame| frame.accepted()) == Some(accepted)
                && batch.state(lane) == Some(crate::traits::verify_owner::batch::LaneState::CommittedAwaitingHidden { accepted })
                && batch.frame_row_base(lane) == Some(row_base),
                "batch hidden publication prefix or target row owner mismatch");
        }
        let result =
            self.save_hidden_for_mtp_sequence_in_frame_unowned(row_base, token_idx, seq, stream);
        let mut shared = self.verify_transactions.lock();
        let mut batch = self.verify_batch_transactions.lock();
        if result.is_ok() {
            batch
                .publish_hidden(&mut shared, lane, accepted)
                .map_err(|error| {
                    super::super::verify_frame::protocol_error(
                        "batch hidden publication failed",
                        error,
                    )
                })?;
        } else {
            let _ = batch.fail_execution::<()>(&mut shared, lane.reservation());
            crate::model::verify_failure::poison_owner();
        }
        result
    }

    fn save_hidden_for_mtp_sequence_in_frame_unowned(
        &self,
        row_base: usize,
        token_idx: usize,
        seq: &SequenceState,
        stream: u64,
    ) -> Result<()> {
        anyhow::ensure!(
            self.qwen4_cross_seq_staging.is_some(),
            "cross-sequence hidden handoff requires ATLAS_QWEN4_BATCHED_K5 lanes"
        );
        anyhow::ensure!(
            self.active_proposer().is_some_and(|proposer| {
                proposer.kind() == crate::speculative::DraftProposerKind::Qwen4NativeMtp
            }),
            "cross-sequence hidden handoff pairs only with the native Qwen4 MTP proposer"
        );
        anyhow::ensure!(
            !crate::layers::qwen4_mtp_rows::recondition_engaged(),
            "cross-sequence hidden handoff does not stage committed-row recondition rows"
        );
        let (storage, frames) = self.qwen4_mtp_hidden_frames.as_ref().ok_or_else(|| {
            anyhow::anyhow!("native MTP has no sequence-owned target-hidden storage")
        })?;
        let row_bytes = self.config.residual_width() * 2;
        let source_offset = row_base
            .checked_mul(row_bytes)
            .ok_or_else(|| anyhow::anyhow!("cross-sequence hidden row offset overflow"))?;
        let source_bytes = self
            .buffers
            .sizes()
            .hidden_states
            .checked_sub(source_offset)
            .ok_or_else(|| anyhow::anyhow!("cross-sequence hidden rows start past the arena"))?;
        let stream = if stream == 0 {
            self.gpu.default_stream()
        } else {
            stream
        };
        let mut frames = frames.lock();
        let write = frames
            .begin_write(
                seq.slot_idx,
                seq.session_hash,
                seq.seq_len,
                token_idx,
                source_bytes,
            )
            .map_err(anyhow::Error::msg)?;
        let source = self.buffers.hidden_states().offset(source_offset);
        self.last_mtp_hidden_idx
            .store(token_idx, std::sync::atomic::Ordering::Relaxed);
        self.gpu.copy_d2d_async(
            source.offset(token_idx * row_bytes),
            storage.offset(write.retained.selected.offset),
            row_bytes,
            stream,
        )?;
        frames.publish(write).map_err(anyhow::Error::msg)
    }

    pub(super) fn generate_speculative_dispatch(
        &self,
        prompt_tokens: &[u32],
        params: &spark_runtime::sampler::SamplingParams,
        num_drafts: usize,
    ) -> Result<crate::engine::GenerateResult> {
        anyhow::ensure!(
            !self.config.is_qwen4_exp()
                || (!self.self_speculative && self.active_proposer().is_none()),
            "Qwen4 speculative generation requires the sequence-owned serving scheduler"
        );
        // Self-speculative mode: draft via layer-skipping (no MTP weights needed)
        if self.self_speculative {
            let mut seq = self.alloc_sequence()?;
            let stream = self.gpu.default_stream();
            let result = self.generate_self_speculative_inner(
                prompt_tokens,
                params,
                num_drafts,
                &mut seq,
                stream,
            );
            self.free_sequence(&mut seq)?;
            return result;
        }

        let proposer = match self.active_proposer() {
            Some(p) => p.clone(),
            None => {
                // Fallback to regular generation
                return crate::engine::generate(self, prompt_tokens, params);
            }
        };

        let mut seq = self.alloc_sequence()?;
        let stream = self.gpu.default_stream();

        let result = self.generate_speculative_inner(
            prompt_tokens,
            params,
            num_drafts,
            &proposer,
            &mut seq,
            stream,
        );

        self.free_sequence(&mut seq)?;

        result
    }

    pub(super) fn has_proposer_dispatch(&self) -> bool {
        self.any_proposer() || self.self_speculative
    }

    pub(super) fn has_self_speculative_dispatch(&self) -> bool {
        // The draft path saves work only by skipping recurrent SSM/linear-
        // attention layers. A pure-attention model has no such layer to skip,
        // so advertising self-spec there runs a full target forward for every
        // draft and cannot satisfy the mode's construction contract.
        self.self_speculative && self.config.num_ssm_layers() > 0 && !self.config.is_qwen4_exp()
    }

    pub(super) fn decode_draft_dispatch(
        &self,
        token: u32,
        seq: &mut SequenceState,
        stream: u64,
    ) -> Result<DevicePtr> {
        TransformerModel::decode_draft(self, token, seq, stream)
    }

    pub(super) fn decode_draft_sparse_dispatch(
        &self,
        token: u32,
        thresh_frac: f32,
        seq: &mut SequenceState,
        stream: u64,
    ) -> Result<DevicePtr> {
        TransformerModel::decode_draft_sparse(self, token, thresh_frac, seq, stream)
    }

    pub(super) fn save_hidden_for_mtp_dispatch(&self, token_idx: usize, stream: u64) -> Result<()> {
        let stream = if stream == 0 {
            self.gpu.default_stream()
        } else {
            stream
        };
        let h = self.config.hidden_size;
        let fp32 = if self.config.use_fp32_residual() {
            4usize
        } else {
            2usize
        };
        // Save the RAW hidden state (before final_norm), not norm_output.
        // The MTP head applies its own pre_fc_norm_hidden — passing norm_output
        // would double-normalize and degrade prediction accuracy.
        let (row_width, element_bytes) = if self.config.is_qwen4_exp() {
            (self.config.residual_width(), 2usize)
        } else {
            (h, fp32)
        };
        let src = self
            .buffers
            .hidden_states()
            .offset(token_idx * row_width * element_bytes);
        self.gpu
            .copy_d2d_async(src, self.mtp_hidden_save, row_width * element_bytes, stream)?;
        // Committed-row recondition rewrite: also stage rows 0..token_idx (the
        // target hidden after the prefix plus 0..token_idx-1 accepted drafts)
        // for the native MTP head's next propose. Default off.
        if self.config.is_qwen4_exp()
            && crate::layers::qwen4_mtp_rows::recondition_engaged()
            && token_idx <= crate::layers::qwen4_mtp_rows::staged_rows_capacity()
            && let Some(staging) = self.mtp_recondition_staging_buf()
            && token_idx > 0
        {
            self.gpu.copy_d2d_async(
                self.buffers.hidden_states(),
                staging,
                token_idx * row_width * element_bytes,
                stream,
            )?;
        }
        // The sequence wrapper publishes the generation-bound staging note.
        self.last_mtp_hidden_idx
            .store(token_idx, std::sync::atomic::Ordering::Relaxed);
        Ok(())
    }

    pub(super) fn run_mtp_propose_dispatch(
        &self,
        token: u32,
        position: usize,
        seq: &mut SequenceState,
        _stream: u64,
    ) -> Result<Option<u32>> {
        let drafts = self.run_mtp_propose_multi(token, position, 1, seq, 0, None)?;
        Ok(drafts.into_iter().next())
    }

    pub(super) fn run_mtp_propose_multi_dispatch(
        &self,
        token: u32,
        position: usize,
        num_drafts: usize,
        seq: &mut SequenceState,
        _stream: u64,
        grammar_bitmask: Option<&[i32]>,
    ) -> Result<Vec<u32>> {
        // MTP loads ALL experts on every rank — no EP all_reduce needed.
        // Rank 1 does not participate in MTP propose.
        self.run_mtp_propose_inner(token, position, num_drafts, seq, grammar_bitmask)
    }

    pub(super) fn run_mtp_propose_multi_batched_dispatch(
        &self,
        tokens: &[u32],
        positions: &[usize],
        num_drafts: usize,
        seqs: &mut [&mut SequenceState],
        _stream: u64,
    ) -> Result<Vec<Vec<u32>>> {
        self.run_mtp_propose_multi_batched_inner(tokens, positions, num_drafts, seqs)
    }

    pub(super) fn stage_dflash_sampling_request_dispatch(
        &self,
        seq: &mut SequenceState,
        request: crate::dflash_sampling::DflashDraftSamplingRequest,
    ) -> Result<()> {
        let proposer = self
            .active_proposer()
            .ok_or_else(|| anyhow::anyhow!("no active proposer for sampled DFlash request"))?;
        let state = seq
            .proposer_state
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("sequence has no proposer state"))?;
        proposer.stage_dflash_sampling_request(state.as_mut(), request)
    }

    pub(super) fn dflash_sampled_proposal_width_dispatch(&self, requested: usize) -> Result<usize> {
        let proposer = self
            .active_proposer()
            .ok_or_else(|| anyhow::anyhow!("no active proposer for sampled DFlash width"))?;
        anyhow::ensure!(proposer.is_dflash(), "active proposer is not DFlash");
        proposer.dflash_sampled_proposal_width(requested)
    }

    pub(super) fn take_pending_dflash_sampled_proposal_dispatch(
        &self,
        seq: &mut SequenceState,
    ) -> Option<crate::dflash_sampling::DflashSampledProposalFrame> {
        let proposer = self.active_proposer()?;
        let state = seq.proposer_state.as_mut()?;
        proposer.take_pending_sampled_proposal(state.as_mut())
    }

    pub(super) fn read_deferred_draft_token_dispatch(&self) -> Result<u32> {
        let proposer = match self.active_proposer() {
            Some(p) => p.as_ref(),
            None => return Ok(0),
        };
        proposer.read_deferred_draft_token(self.gpu.as_ref())
    }

    /// ATLAS_DFLASH_ASYNC: collect a deferred async propose's drafts for
    /// this sequence (no-op `None` when nothing is pending). See
    /// `dflash_head/async_propose.rs`.
    pub(super) fn dflash_collect_async_drafts_dispatch(
        &self,
        seq: &mut SequenceState,
    ) -> Result<Option<Vec<u32>>> {
        let Some(proposer) = self.active_proposer() else {
            return Ok(None);
        };
        let Some(ref mut state) = seq.proposer_state else {
            return Ok(None);
        };
        proposer.collect_async_drafts(self.gpu.as_ref(), state.as_mut())
    }

    pub(in crate::model) fn trim_proposer_state_dispatch(
        &self,
        seq: &mut SequenceState,
        num_accepted: usize,
        stream: u64,
    ) -> Result<()> {
        let proposer = match self.active_proposer() {
            Some(p) => p.as_ref(),
            None => return Ok(()),
        };
        let stream = if stream == 0 {
            self.gpu.default_stream()
        } else {
            stream
        };
        let owner = seq.verify_frame.map(|frame| frame.ticket());
        if let Some(ref mut state) = seq.proposer_state {
            if let Some(native) = state
                .as_any_mut()
                .downcast_mut::<crate::layers::qwen4_mtp::Qwen4MtpState>()
            {
                native.bind_verify_frame(owner)?;
            }
            proposer.after_verify(num_accepted, state.as_mut(), stream)?;
        }
        Ok(())
    }

    /// `ATLAS_DFLASH_TREE_COMMIT`: stamp the accepted-path capture rows
    /// onto the DFlash proposer state so the next propose's ctx-hidden append
    /// reads the scattered (fork) capture rows instead of the contiguous
    /// prefix. Must be called AFTER `trim_proposer_state` (`after_verify`
    /// clears the path). No-op for non-DFlash proposers.
    ///
    /// Frame mapping: `accepted_compact` is in the COMPACT frame, but
    /// `dflash_hidden_save` (which these rows index) was written by
    /// `try_dflash_capture(layer, t)` in the KERNEL (DFS-reordered) frame —
    /// capture row `t` holds kernel slot `t`. So compact index `c`'s captured
    /// hidden lives at row `dfs_inv_perm[c]`. We map here through the SAME
    /// `ddtree_dfs_inv_perm` the KV gather and SSM commit use, then store the
    /// KERNEL-frame rows the append will read directly. Empty inv_perm (chain
    /// mode / DFS off) ⇒ identity ⇒ flat path unchanged.
    pub(super) fn set_dflash_accepted_compact_dispatch(
        &self,
        seq: &mut SequenceState,
        accepted_compact: &[usize],
    ) {
        let Some(ref mut state) = seq.proposer_state else {
            return;
        };
        if let Some(ds) = state
            .as_any_mut()
            .downcast_mut::<crate::layers::DflashProposerState>()
        {
            use crate::layers::dflash_head::ddtree::map_compact_path_to_kernel;
            let inv_perm = self.ddtree_dfs_inv_perm.lock();
            let kernel_rows = map_compact_path_to_kernel(accepted_compact, &inv_perm);
            drop(inv_perm);
            ds.last_accepted_compact.clear();
            ds.last_accepted_compact.extend_from_slice(&kernel_rows);
        }
    }

    /// Retire accepted batched-verify DFlash hidden captures from
    /// `dflash_hidden_save[row_base..=row_base + num_accepted]` into `seq`'s
    /// private `DflashProposerState` ring and close the verify batch
    /// transaction's hidden-publication lane.
    ///
    /// Mirrors [`Self::save_hidden_for_mtp_sequence_in_frame_dispatch`]:
    /// `commit_verify_state_async` / `commit_verify_batch_cached` committed
    /// `accepted = num_accepted + 1` rows (anchor + accepted drafts), leaving
    /// the lane in `CommittedAwaitingHidden { accepted }`.
    pub(crate) fn retire_dflash_frame_captures_impl(
        &self,
        row_base: usize,
        num_accepted: usize,
        seq: &mut SequenceState,
        stream: u64,
    ) -> Result<()> {
        let rows = Self::batch_lane_rows(seq)
            .ok_or_else(|| anyhow::anyhow!("DFlash batch capture retirement has no lane ticket"))?;
        let lane = self
            .batch_lane(seq, rows, false)?
            .ok_or_else(|| anyhow::anyhow!("DFlash batch capture retirement has no lane ticket"))?;
        let accepted = num_accepted
            .checked_add(1)
            .ok_or_else(|| anyhow::anyhow!("DFlash batch capture accepted prefix overflow"))?;
        {
            let batch = self.verify_batch_transactions.lock();
            anyhow::ensure!(
                seq.verify_frame.and_then(|frame| frame.accepted()) == Some(accepted)
                    && batch.state(lane)
                        == Some(
                            crate::traits::verify_owner::batch::LaneState::CommittedAwaitingHidden {
                                accepted,
                            }
                        )
                    && batch.frame_row_base(lane) == Some(row_base),
                "DFlash batch capture retirement prefix or target row owner mismatch"
            );
        }
        let result =
            self.retire_dflash_frame_captures_from_save(row_base, num_accepted, seq, stream);
        let mut shared = self.verify_transactions.lock();
        let mut batch = self.verify_batch_transactions.lock();
        if result.is_ok() {
            batch
                .publish_hidden(&mut shared, lane, accepted)
                .map_err(|error| {
                    super::super::verify_frame::protocol_error(
                        "DFlash batch hidden publication failed",
                        error,
                    )
                })?;
        } else {
            let _ = batch.fail_execution::<()>(&mut shared, lane.reservation());
            crate::model::verify_failure::poison_owner();
        }
        result
    }
}
