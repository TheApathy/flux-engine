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
    pub(super) fn ep_worker_step_dispatch(&self, seq: &mut SequenceState) -> Result<bool> {
        self.ep_worker_step_impl(seq)
    }

    pub(super) fn is_ep_dispatch(&self) -> bool {
        self.comm.as_ref().is_some_and(|c| c.world_size() > 1)
            || self.config.ep_world_size > 1
            || self.config.tp_world_size > 1
            || self.config.pp_world_size > 1
    }

    pub(in crate::model) fn qwen4_tp_batched_verify_supported_dispatch(&self) -> bool {
        self.config.is_qwen4_exp()
            && (((2..=4).contains(&self.config.tp_world_size)
                && (self.config.ep_world_size <= 1
                    || self.config.ep_world_size == self.config.tp_world_size)
                && self.config.pp_world_size <= 1)
                || (self.config.tp_world_size <= 1
                    && self.config.ep_world_size <= 1
                    && self.config.pp_world_size >= 2))
    }

    pub(super) fn is_mla_dispatch(&self) -> bool {
        // MLA models have a kv_lora_rank > 0 (kv-compression dimension).
        // Same detection as the existing prefill-vs-decode-detail check at
        // line 2185 (`ctx.config.kv_lora_rank > 0`).
        self.config.kv_lora_rank > 0
    }

    pub(super) fn decode_logits_fp32_dispatch(&self) -> bool {
        // Forward to the inherent method. Active when
        // ATLAS_GEMMA4_FP32_LMHEAD=1 + Gemma-4 dense + FP32 residual.
        TransformerModel::decode_logits_fp32(self)
    }

    pub(super) fn decode_logits_ptr_dispatch(&self) -> DevicePtr {
        TransformerModel::decode_logits_ptr(self)
    }

    #[inline]
    pub(in crate::model) const fn pack_ep_slot_meta(
        slot_idx: usize,
        qsa_required: bool,
        fresh_if_dirty: bool,
    ) -> u32 {
        ((slot_idx as u32) & 0x3FFF_FFFF)
            | (if fresh_if_dirty { 0x4000_0000 } else { 0 })
            | (if qsa_required { 0x8000_0000 } else { 0 })
    }

    #[inline]
    pub(in crate::model) const fn unpack_ep_slot_meta(packed: u32) -> (usize, bool, bool) {
        let slot_idx = (packed & 0x3FFF_FFFF) as usize;
        let fresh_if_dirty = (packed & 0x4000_0000) != 0;
        let qsa_required = (packed & 0x8000_0000) != 0;
        (slot_idx, qsa_required, fresh_if_dirty)
    }

    pub(in crate::model) fn ep_head_sync_for_prefill(
        &self,
        seq: &SequenceState,
        chunk_start: usize,
    ) -> Result<()> {
        use super::super::types::EpHeadWireState;
        if !self
            .comm
            .as_ref()
            .is_some_and(|c| c.world_size() > 1 && c.rank() == 0)
        {
            return Ok(());
        }
        let should_send = {
            let mut state = self.ep_wire_state.lock();
            if *state == EpHeadWireState::PrefillReady {
                *state = EpHeadWireState::Idle;
                true
            } else {
                false
            }
        };
        if should_send {
            let fresh_if_dirty = seq.seq_len == 0 && chunk_start == 0;
            let packed =
                Self::pack_ep_slot_meta(seq.slot_idx, seq.qwen4_qsa_required, fresh_if_dirty);
            self.ep_broadcast_u32(packed)?;
        }
        Ok(())
    }

    pub(in crate::model) fn ep_head_sync_for_decode(&self, seq: &SequenceState) -> Result<()> {
        use super::super::types::EpHeadWireState;
        if !self
            .comm
            .as_ref()
            .is_some_and(|c| c.world_size() > 1 && c.rank() == 0)
        {
            return Ok(());
        }
        let should_send = {
            let mut state = self.ep_wire_state.lock();
            if *state == EpHeadWireState::DecodeReady {
                *state = EpHeadWireState::Idle;
                true
            } else {
                false
            }
        };
        if should_send {
            let packed = Self::pack_ep_slot_meta(seq.slot_idx, seq.qwen4_qsa_required, false);
            self.ep_broadcast_u32(packed)?;
        }
        Ok(())
    }

    pub(in crate::model) fn ep_head_sync_for_verify(&self, seq: &SequenceState) -> Result<()> {
        use super::super::types::EpHeadWireState;
        if !self
            .comm
            .as_ref()
            .is_some_and(|c| c.world_size() > 1 && c.rank() == 0)
        {
            return Ok(());
        }
        let should_send = {
            let mut state = self.ep_wire_state.lock();
            if matches!(
                *state,
                EpHeadWireState::VerifyFixedReady | EpHeadWireState::VerifyKgammaReady
            ) {
                *state = EpHeadWireState::VerifyResultPending;
                true
            } else {
                false
            }
        };
        if should_send {
            let packed = Self::pack_ep_slot_meta(seq.slot_idx, seq.qwen4_qsa_required, false);
            if let Err(err) = self.ep_broadcast_u32(packed) {
                *self.ep_wire_state.lock() = EpHeadWireState::Idle;
                return Err(err);
            }
        }
        Ok(())
    }

    pub(in crate::model) fn ep_broadcast_cmd_dispatch(&self, cmd: u32) -> Result<()> {
        use super::super::types::EpHeadWireState;
        let Some(comm) = self.comm.as_ref().filter(|c| c.world_size() > 1) else {
            return Ok(());
        };
        if comm.rank() != 0 {
            self.ep_broadcast_u32(cmd)?;
            return Ok(());
        }
        let prev_state = *self.ep_wire_state.lock();
        let res: Result<()> = match (prev_state, cmd) {
            (_, 0xFFFFFFF1) => {
                *self.ep_wire_state.lock() = EpHeadWireState::Idle;
                (|| -> Result<()> {
                    self.ep_broadcast_u32(0xFFFFFFF1)?;
                    let slots: Vec<u32> = self
                        .pending_freed_slots
                        .lock()
                        .drain(..)
                        .map(|s| s as u32)
                        .collect();
                    if slots.is_empty() {
                        self.ep_broadcast_u32(u32::MAX)?;
                    } else {
                        self.ep_broadcast_u32(slots.len() as u32)?;
                        for s in slots {
                            self.ep_broadcast_u32(s)?;
                        }
                    }
                    Ok(())
                })()
            }
            (EpHeadWireState::VerifyResultPending, _) => {
                // Must precede `(_, 0xFFFFFFFF)` because `EP_VERIFY_KGAMMA_ABORT == 0xFFFFFFFF`.
                *self.ep_wire_state.lock() = EpHeadWireState::Idle;
                self.ep_broadcast_u32(cmd).map(|_| ())
            }
            (_, 0xFFFFFFFF) => {
                *self.ep_wire_state.lock() = EpHeadWireState::Idle;
                self.ep_broadcast_u32(0xFFFFFFFF).map(|_| ())
            }
            (EpHeadWireState::Idle, 0xFFFFFFF0) => {
                *self.ep_wire_state.lock() = EpHeadWireState::PrefillArgs { remaining_u32: 3 };
                self.ep_broadcast_u32(cmd).map(|_| ())
            }
            (EpHeadWireState::PrefillArgs { remaining_u32 }, _) => {
                *self.ep_wire_state.lock() = if remaining_u32 <= 1 {
                    EpHeadWireState::PrefillTokensPending
                } else {
                    EpHeadWireState::PrefillArgs {
                        remaining_u32: remaining_u32 - 1,
                    }
                };
                self.ep_broadcast_u32(cmd).map(|_| ())
            }
            (EpHeadWireState::Idle, 0xFFFFFFF2) => {
                *self.ep_wire_state.lock() = EpHeadWireState::VerifyFixedArgs { remaining_u32: 2 };
                self.ep_broadcast_u32(cmd).map(|_| ())
            }
            (EpHeadWireState::Idle, 0xFFFFFFF3) => {
                *self.ep_wire_state.lock() = EpHeadWireState::VerifyFixedArgs { remaining_u32: 3 };
                self.ep_broadcast_u32(cmd).map(|_| ())
            }
            (EpHeadWireState::Idle, 0xFFFFFFF4) => {
                *self.ep_wire_state.lock() = EpHeadWireState::VerifyFixedArgs { remaining_u32: 4 };
                self.ep_broadcast_u32(cmd).map(|_| ())
            }
            (EpHeadWireState::VerifyFixedArgs { remaining_u32 }, _) => {
                *self.ep_wire_state.lock() = if remaining_u32 <= 1 {
                    EpHeadWireState::VerifyFixedReady
                } else {
                    EpHeadWireState::VerifyFixedArgs {
                        remaining_u32: remaining_u32 - 1,
                    }
                };
                self.ep_broadcast_u32(cmd).map(|_| ())
            }
            (EpHeadWireState::Idle, crate::traits::EP_CMD_VERIFY_KGAMMA) => {
                *self.ep_wire_state.lock() = EpHeadWireState::VerifyKgammaLen;
                self.ep_broadcast_u32(cmd).map(|_| ())
            }
            (EpHeadWireState::VerifyKgammaLen, _) => {
                *self.ep_wire_state.lock() = EpHeadWireState::VerifyKgammaTokens;
                self.ep_broadcast_u32(cmd).map(|_| ())
            }
            (EpHeadWireState::Idle, token) if token < 0xFFFFFFF0 => {
                *self.ep_wire_state.lock() = EpHeadWireState::DecodeReady;
                self.ep_broadcast_u32(token).map(|_| ())
            }
            (_, other) => {
                tracing::warn!(
                    prev_state = ?prev_state,
                    cmd = format_args!("{other:#x}"),
                    "EP_WIRE_STATE_UNEXPECTED_PASSTHROUGH"
                );
                self.ep_broadcast_u32(other).map(|_| ())
            }
        };
        if res.is_err() {
            *self.ep_wire_state.lock() = EpHeadWireState::Idle;
        }
        res
    }

    pub(super) fn ep_broadcast_tokens_dispatch(&self, tokens: &[u32]) -> Result<Vec<u32>> {
        TransformerModel::ep_broadcast_tokens(self, tokens)
    }

    pub(super) fn synchronize_stream_dispatch(&self, stream: u64) -> Result<()> {
        self.gpu.synchronize(stream)
    }

    pub(super) fn default_stream_dispatch(&self) -> u64 {
        self.gpu.default_stream()
    }

    pub(super) fn create_stream_dispatch(&self) -> Result<u64> {
        self.gpu.create_stream()
    }

    pub(super) fn create_event_dispatch(&self) -> Result<u64> {
        self.gpu.create_event()
    }

    pub(super) fn record_event_dispatch(&self, event: u64, stream: u64) -> Result<()> {
        self.gpu.record_event(event, stream)
    }

    pub(super) fn stream_wait_event_dispatch(&self, stream: u64, event: u64) -> Result<()> {
        self.gpu.stream_wait_event(stream, event)
    }
}

#[path = "../ep_control_frame.rs"]
pub(crate) mod ep_control_frame;

#[cfg(test)]
#[path = "ep_misc_tests.rs"]
mod tests;
