// SPDX-License-Identifier: AGPL-3.0-only

#![allow(unused_imports, dead_code)]

use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Result, bail};
use atlas_core::config::{LayerType, ModelConfig};
use spark_runtime::buffers::BufferArena;
use spark_runtime::gpu::{DevicePtr, GpuBackend, GraphHandle, KernelHandle};
use spark_runtime::kv_cache::PagedKvCache;

use super::block_mgmt::{
    apply_evicted_blocks, ensure_blocks_through_decode, ensure_blocks_through_prefill,
    extract_layer_refs, reuse_prefix_match_disk_ids,
};
use super::ssm_pool::SsmStatePool;
use super::ssm_snapshot::SsmSnapshotPool;
use super::types::{PinnedMetaStaging, TransformerModel};
use crate::layer::{
    AttnMetadataDev, ForwardContext, GdnPrefillBuffers, LayerState, SsmLayerState, TransformerLayer,
};
use crate::layers::ops;
use crate::speculative::DraftProposer;
use crate::traits::{ChunkedPrefillPageMetadata, Model, SequenceState};
use crate::traits::{
    EP_CMD_ABORT_VERIFY_SLOT, EP_CMD_COMMIT_VERIFY_SLOT, EP_CMD_DECODE_QWEN4_BATCH,
    EP_CMD_VERIFY_KGAMMA, EP_CMD_VERIFY_QWEN4_BATCHED, EP_VERIFY_KGAMMA_ABORT,
};
use crate::weight_map::{DenseWeight, MtpWeights, QuantizedWeight};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct EpVerifyKgCommitPlan {
    num_accepted: usize,
    total_accepted: usize,
    drop_tokens: usize,
}

/// Decode the rank-0 result frame into the worker's sequence-state action.
/// `k` includes the prefix input, while `num_accepted` counts drafts only.
fn ep_verify_kgamma_commit_plan(k: usize, result: u32) -> Result<Option<EpVerifyKgCommitPlan>> {
    if result == EP_VERIFY_KGAMMA_ABORT {
        return Ok(None);
    }
    if k == 0 {
        bail!("EP K=gamma verify received an empty token frame");
    }
    let num_accepted = result as usize;
    if num_accepted >= k {
        bail!(
            "EP K=gamma verify result {num_accepted} exceeds the {} available drafts",
            k - 1
        );
    }
    let total_accepted = num_accepted + 1;
    Ok(Some(EpVerifyKgCommitPlan {
        num_accepted,
        total_accepted,
        drop_tokens: k - total_accepted,
    }))
}

impl TransformerModel {
    pub(super) fn comm_ref(&self) -> Option<&dyn spark_comm::CommBackend> {
        self.comm.as_deref()
    }

    /// Per-vision-pad-token-id helper. Vision prompts splice ViT
    /// embeddings into placeholder `<|image_pad|>` token positions; the
    /// hashed token-ID stream therefore looks identical for two
    /// distinct images of the same prompt, and naive prefix-cache reuse
    /// would resurrect the FIRST image's KV/SSM blocks for the SECOND
    /// image. Skip cache lookup AND insert whenever any `image_pad`
    /// token is present in the prefill window.
    pub(super) fn tokens_have_vision_pad(&self, tokens: &[u32]) -> bool {
        let pad_id = match self.config.vision.as_ref().map(|v| v.image_pad_token_id) {
            Some(id) if id != 0 => id,
            _ => crate::layers::vision_encoder::IMAGE_PAD_TOKEN_ID,
        };
        tokens.contains(&pad_id)
    }

    pub(super) fn ensure_chunked_prefill_meta<'a>(
        &self,
        seq: &'a mut SequenceState,
        total_tokens: usize,
        block_size: usize,
    ) -> Result<&'a mut ChunkedPrefillPageMetadata> {
        let required_blocks = total_tokens.saturating_sub(1) / block_size + 1;
        if seq.chunked_prefill_meta.is_none() {
            seq.chunked_prefill_meta = Some(ChunkedPrefillPageMetadata {
                block_table: self.gpu.alloc(required_blocks.max(1) * 4)?,
                seq_len: self.gpu.alloc(std::mem::size_of::<u32>())?,
                block_capacity: required_blocks,
                uploaded_blocks: 0,
            });
        }

        let meta = seq.chunked_prefill_meta.as_mut().unwrap();
        if meta.block_capacity < required_blocks {
            bail!(
                "chunked prefill metadata capacity {} < required {} blocks",
                meta.block_capacity,
                required_blocks,
            );
        }
        Ok(meta)
    }

    pub(super) fn free_chunked_prefill_meta(&self, seq: &mut SequenceState) -> Result<()> {
        if let Some(meta) = seq.chunked_prefill_meta.take() {
            if !meta.block_table.is_null() {
                self.gpu.free(meta.block_table)?;
            }
            if !meta.seq_len.is_null() {
                self.gpu.free(meta.seq_len)?;
            }
        }
        Ok(())
    }

    /// Bulk broadcast: send an array of u32 tokens from rank 0 to all ranks.
    ///
    /// Uses a single NCCL broadcast instead of per-token broadcasts.
    /// Per-token broadcasting causes NCCL deadlocks on prompts >4K tokens.
    pub(super) fn ep_broadcast_tokens(&self, tokens: &[u32]) -> Result<Vec<u32>> {
        let n = tokens.len();
        if n == 0 || self.comm.is_none() {
            return Ok(tokens.to_vec());
        }
        let comm = self.comm.as_ref().unwrap();
        let stream = self.gpu.default_stream();

        // Use scratch buffer as device staging, chunked to its capacity so
        // 1M-token prompts cannot overrun smaller chunk-budgeted arenas.
        let dev_buf = self.buffers.scratch();
        let max_chunk_tokens = (self.buffers.sizes().scratch / 4).max(1);
        let mut result = if comm.rank() != 0 {
            vec![0u32; n]
        } else {
            tokens.to_vec()
        };
        let mut offset = 0usize;
        let send_res = (|| -> Result<()> {
            while offset < n {
                let chunk_n = (n - offset).min(max_chunk_tokens);
                let byte_len = chunk_n * 4;
                if comm.rank() == 0 {
                    let token_slice = &tokens[offset..offset + chunk_n];
                    // SAFETY: `token_slice` is a valid, initialized `&[u32]` slice,
                    // `u8` has alignment 1 and no invalid bit patterns, and
                    // `size_of_val(token_slice) == byte_len` stays within the slice.
                    let token_bytes: &[u8] = unsafe {
                        std::slice::from_raw_parts(
                            token_slice.as_ptr().cast::<u8>(),
                            std::mem::size_of_val(token_slice),
                        )
                    };
                    self.gpu.copy_h2d(token_bytes, dev_buf)?;
                }
                comm.broadcast(dev_buf.0, byte_len, 0)?;
                if comm.rank() != 0 {
                    self.gpu.synchronize(stream)?;
                    let result_slice = &mut result[offset..offset + chunk_n];
                    // SAFETY: `result_slice` is an exclusively borrowed, initialized
                    // `&mut [u32]` slice, `u8` has alignment 1, and every written byte
                    // pattern is a valid `u32` within `size_of_val(result_slice)`.
                    let result_bytes = unsafe {
                        std::slice::from_raw_parts_mut(
                            result_slice.as_mut_ptr().cast::<u8>(),
                            std::mem::size_of_val(result_slice),
                        )
                    };
                    self.gpu.copy_d2h(dev_buf, result_bytes)?;
                } else if offset + chunk_n < n {
                    self.gpu.synchronize(stream)?;
                }
                offset += chunk_n;
            }
            Ok(())
        })();
        if comm.world_size() > 1 && comm.rank() == 0 {
            use super::types::EpHeadWireState;
            let mut state = self.ep_wire_state.lock();
            if send_res.is_err() {
                *state = EpHeadWireState::Idle;
            } else {
                match *state {
                    EpHeadWireState::PrefillTokensPending => {
                        *state = EpHeadWireState::PrefillReady;
                    }
                    EpHeadWireState::VerifyKgammaTokens => {
                        *state = EpHeadWireState::VerifyKgammaReady;
                    }
                    _ => {}
                }
            }
        }
        send_res?;
        Ok(result)
    }

    pub(super) fn pp_stage_rank(&self, pp_stage: usize) -> usize {
        let tp_ep = if self.config.tp_world_size == self.config.ep_world_size
            && self.config.tp_world_size > 1
        {
            self.config.tp_world_size
        } else {
            self.config.tp_world_size.max(1) * self.config.ep_world_size.max(1)
        };
        let local_lane = if self.config.tp_world_size == self.config.ep_world_size
            && self.config.tp_world_size > 1
        {
            self.config.tp_rank
        } else {
            self.config.ep_rank * self.config.tp_world_size.max(1) + self.config.tp_rank
        };
        pp_stage * tp_ep + local_lane
    }

    /// Pipeline-stage entry: non-zero stages receive `[rows, residual_width]`
    /// activations into `hidden` from `pp_rank - 1` on `stream`.
    pub(super) fn pp_stage_enter(&self, hidden: DevicePtr, rows: usize, stream: u64) -> Result<()> {
        let pp_size = self.config.pp_world_size.max(1);
        if pp_size <= 1 || self.config.pp_rank == 0 || rows == 0 {
            return Ok(());
        }
        let comm = self.comm.as_ref().ok_or_else(|| {
            anyhow::anyhow!("PP stage {} missing communicator", self.config.pp_rank)
        })?;
        let elem_bytes = if self.config.use_fp32_residual() {
            4
        } else {
            2
        };
        let bytes = rows * self.config.residual_width() * elem_bytes;
        let prev = self.pp_stage_rank(self.config.pp_rank - 1);
        comm.recv_from(hidden.0, bytes, prev, stream)
    }

    /// Pipeline-stage exit:
    /// - Stage `0`: sends `hidden` to stage `1`, then receives final-stage `hidden`
    ///   back from stage `pp_size - 1` so final mixer, LM head, prefix cache, and
    ///   native MTP all run locally on stage `0`.
    /// - Intermediate stages: send `hidden` to `pp_rank + 1`.
    /// - Last stage (`pp_size - 1`): sends `hidden` back to stage `0`.
    pub(super) fn pp_stage_exit(&self, hidden: DevicePtr, rows: usize, stream: u64) -> Result<()> {
        let pp_size = self.config.pp_world_size.max(1);
        if pp_size <= 1 || rows == 0 {
            return Ok(());
        }
        let comm = self.comm.as_ref().ok_or_else(|| {
            anyhow::anyhow!("PP stage {} missing communicator", self.config.pp_rank)
        })?;
        let elem_bytes = if self.config.use_fp32_residual() {
            4
        } else {
            2
        };
        let bytes = rows * self.config.residual_width() * elem_bytes;
        let r = self.config.pp_rank;
        if r == 0 {
            let next = self.pp_stage_rank(1);
            comm.send_to(hidden.0, bytes, next, stream)?;
            let tail = self.pp_stage_rank(pp_size - 1);
            comm.recv_from(hidden.0, bytes, tail, stream)
        } else {
            let next_stage = if r + 1 == pp_size { 0 } else { r + 1 };
            let dst = self.pp_stage_rank(next_stage);
            comm.send_to(hidden.0, bytes, dst, stream)
        }
    }

    /// F83 (2026-04-30): all-reduce-min on a single u32 across all
    /// participating ranks.
    pub(super) fn ep_min_u32(&self, val: u32) -> Result<u32> {
        let Some(comm) = self.comm.as_ref() else {
            return Ok(val);
        };
        let stream = self.gpu.default_stream();
        let world = comm
            .world_size()
            .max(self.config.ep_world_size)
            .max(self.config.tp_world_size)
            .max(self.config.pp_world_size);
        let mut min_val = val;
        for root in 0..world {
            let v = if comm.rank() == root {
                self.gpu.copy_h2d(&val.to_le_bytes(), self.ep_cmd_buf)?;
                comm.broadcast(self.ep_cmd_buf.0, 4, root)?;
                val
            } else {
                comm.broadcast(self.ep_cmd_buf.0, 4, root)?;
                self.gpu.synchronize(stream)?;
                let mut buf = [0u8; 4];
                self.gpu.copy_d2h(self.ep_cmd_buf, &mut buf)?;
                u32::from_le_bytes(buf)
            };
            min_val = min_val.min(v);
        }
        Ok(min_val)
    }

    /// Broadcast a u32 command from rank 0 to all ranks.
    /// Rank 0 writes `val` to GPU buffer and broadcasts.
    /// Other ranks receive the value and return it.
    pub(super) fn ep_broadcast_u32(&self, val: u32) -> Result<u32> {
        let comm = self.comm.as_ref().expect("ep_broadcast_u32 without comm");
        let stream = self.gpu.default_stream();
        if comm.rank() == 0 {
            // Sender: H2D + broadcast. Stream ordering ensures completion
            // before next GPU operation on the same stream. No sync needed.
            self.gpu.copy_h2d(&val.to_le_bytes(), self.ep_cmd_buf)?;
            comm.broadcast(self.ep_cmd_buf.0, 4, 0)?;
            Ok(val)
        } else {
            // Receiver: broadcast + sync + D2H to read the received value.
            comm.broadcast(self.ep_cmd_buf.0, 4, 0)?;
            self.gpu.synchronize(stream)?;
            let mut buf = [0u8; 4];
            self.gpu.copy_d2h(self.ep_cmd_buf, &mut buf)?;
            Ok(u32::from_le_bytes(buf))
        }
    }

    fn ensure_worker_slot_allocated(
        &self,
        primary: &mut SequenceState,
        wanted_slot: usize,
    ) -> Result<()> {
        if wanted_slot == primary.slot_idx
            || self.worker_extra_slots.lock().contains_key(&wanted_slot)
        {
            return Ok(());
        }
        anyhow::ensure!(
            wanted_slot < self.ssm_pool.max_slots,
            "worker requested out-of-range slot {wanted_slot} (max_slots={}, primary={}, extra={:?}, claimed={:?})",
            self.ssm_pool.max_slots,
            primary.slot_idx,
            self.worker_extra_slots
                .lock()
                .keys()
                .copied()
                .collect::<Vec<_>>(),
            self.ssm_pool.claimed_slots(),
        );
        let mut temp: Vec<SequenceState> = Vec::new();
        let max_iters = self.ssm_pool.max_slots.max(256);
        let found = loop {
            let s = match self.alloc_sequence() {
                Ok(s) => s,
                Err(err) => {
                    for mut tmp in temp {
                        let _ = self.free_sequence(&mut tmp);
                    }
                    return Err(err);
                }
            };
            if s.slot_idx == wanted_slot {
                break s;
            }
            temp.push(s);
            if temp.len() > max_iters {
                for mut tmp in temp {
                    let _ = self.free_sequence(&mut tmp);
                }
                bail!("worker failed to reach requested slot {wanted_slot}");
            }
        };
        for mut tmp in temp {
            let _ = self.free_sequence(&mut tmp);
        }
        if primary.slot_idx == usize::MAX {
            *primary = found;
        } else {
            self.worker_extra_slots.lock().insert(wanted_slot, found);
        }
        Ok(())
    }

    fn free_worker_slot(&self, primary: &mut SequenceState, target_slot: usize) -> Result<()> {
        if primary.slot_idx == target_slot {
            let mut old = std::mem::replace(primary, SequenceState::empty_placeholder(usize::MAX));
            self.free_sequence(&mut old)?;
        } else if let Some(mut s) = self.worker_extra_slots.lock().remove(&target_slot) {
            self.free_sequence(&mut s)?;
        }
        Ok(())
    }

    fn with_worker_slot<T>(
        &self,
        primary: &mut SequenceState,
        wanted_slot: usize,
        fresh_if_dirty: bool,
        qsa_required: bool,
        f: impl FnOnce(&mut SequenceState) -> Result<T>,
    ) -> Result<T> {
        self.ensure_worker_slot_allocated(primary, wanted_slot)?;
        let mut s = if wanted_slot == primary.slot_idx {
            std::mem::replace(primary, SequenceState::empty_placeholder(usize::MAX))
        } else {
            self.worker_extra_slots
                .lock()
                .remove(&wanted_slot)
                .ok_or_else(|| anyhow::anyhow!("missing worker slot {wanted_slot}"))?
        };
        if fresh_if_dirty && (s.seq_len > 0 || !s.block_table.is_empty()) {
            let _ = self.free_sequence(&mut s);
            self.ensure_worker_slot_allocated(primary, wanted_slot)?;
            s = if wanted_slot == primary.slot_idx {
                std::mem::replace(primary, SequenceState::empty_placeholder(usize::MAX))
            } else {
                self.worker_extra_slots
                    .lock()
                    .remove(&wanted_slot)
                    .ok_or_else(|| anyhow::anyhow!("missing refreshed worker slot {wanted_slot}"))?
            };
        }
        s.qwen4_qsa_required = qsa_required;
        let out = f(&mut s);
        if primary.slot_idx == usize::MAX {
            *primary = s;
        } else {
            self.worker_extra_slots.lock().insert(wanted_slot, s);
        }
        out
    }

    /// EP worker step: receive a command from rank 0 and execute it.
    pub(super) fn ep_worker_step_impl(&self, seq: &mut SequenceState) -> Result<bool> {
        thread_local! {
            static LAST_CMDS: std::cell::RefCell<std::collections::VecDeque<u32>> =
                std::cell::RefCell::new(std::collections::VecDeque::with_capacity(16));
        }
        let cmd = self.ep_broadcast_u32(0)?;
        LAST_CMDS.with(|q| {
            let mut q = q.borrow_mut();
            if q.len() >= 16 {
                q.pop_front();
            }
            q.push_back(cmd);
        });
        self.ep_worker_step_exec(cmd, seq).map_err(|e| {
            let hist: Vec<String> =
                LAST_CMDS.with(|q| q.borrow().iter().map(|c| format!("{c:#x}")).collect());
            e.context(format!("ep_worker_step cmd={cmd:#x} recent={hist:?}"))
        })
    }

    fn ep_worker_step_exec(&self, cmd: u32, seq: &mut SequenceState) -> Result<bool> {
        let stream = self.gpu.default_stream();

        match cmd {
            0xFFFFFFFF => {
                let extra_all: Vec<SequenceState> = self
                    .worker_extra_slots
                    .lock()
                    .drain()
                    .map(|(_, s)| s)
                    .collect();
                for mut s in extra_all {
                    let _ = self.free_sequence(&mut s);
                }
                return Ok(false);
            }
            0xFFFFFFF1 => {
                let count_or_sentinel = self.ep_broadcast_u32(0)?;
                if count_or_sentinel == u32::MAX {
                    let extra_all: Vec<SequenceState> = self
                        .worker_extra_slots
                        .lock()
                        .drain()
                        .map(|(_, s)| s)
                        .collect();
                    for mut s in extra_all {
                        self.free_sequence(&mut s)?;
                    }
                    if seq.slot_idx < self.ssm_pool.max_slots {
                        self.free_sequence(seq)?;
                    }
                    *seq = self.alloc_sequence()?;
                } else {
                    for _ in 0..count_or_sentinel {
                        let target_slot = self.ep_broadcast_u32(0)? as usize;
                        self.free_worker_slot(seq, target_slot)?;
                    }
                }
            }
            0xFFFFFFF0 => {
                let chunk_len = self.ep_broadcast_u32(0)? as usize;
                let chunk_start = self.ep_broadcast_u32(0)? as usize;
                let full_len = self.ep_broadcast_u32(0)? as usize;
                let full_tokens = self.ep_broadcast_tokens(&vec![0u32; full_len])?;
                let packed = self.ep_broadcast_u32(0)?;
                let (slot_idx, qsa_req, fresh_if_dirty) = Self::unpack_ep_slot_meta(packed);
                let is_last = chunk_start + chunk_len >= full_len;
                self.sync_secondary()?;
                self.with_worker_slot(seq, slot_idx, fresh_if_dirty, qsa_req, |s| {
                    let _ = self.prefill_chunk(
                        &full_tokens,
                        s,
                        chunk_start,
                        chunk_len,
                        is_last,
                        stream,
                    )?;
                    if let Err(e) = self.normalize_ssm_states(s, stream) {
                        tracing::warn!("Worker SSM state normalization failed: {e:#}");
                    }
                    if is_last && self.ssm_pool.has_mtp {
                        self.gpu.synchronize(stream)?;
                        self.start_checkpoint_async(s)?;
                    }
                    Ok(())
                })?;
            }
            0xFFFFFFF2 => {
                let t0 = self.ep_broadcast_u32(0)?;
                let t1 = self.ep_broadcast_u32(0)?;
                let packed = self.ep_broadcast_u32(0)?;
                let (slot_idx, qsa_req, _) = Self::unpack_ep_slot_meta(packed);
                self.sync_secondary()?;
                self.with_worker_slot(seq, slot_idx, false, qsa_req, |s| {
                    self.decode_verify_graphed(&[t0, t1], s, stream)?;
                    let accepted = self.ep_broadcast_u32(0)?;
                    if self.config.is_qwen4_exp() {
                        self.commit_ep_short_owned(s, accepted as usize, 2)?;
                    } else if accepted == 1 {
                        self.start_checkpoint_async(s)?;
                        self.trim_proposer_state(s, 1, 0)?;
                    } else {
                        s.seq_len -= 1;
                        s.tokens.pop();
                        self.trim_proposer_state(s, 0, 0)?;
                        self.start_rollback_and_checkpoint_async(s, 1)?;
                    }
                    Ok(())
                })?;
            }
            0xFFFFFFF3 => {
                let t0 = self.ep_broadcast_u32(0)?;
                let t1 = self.ep_broadcast_u32(0)?;
                let t2 = self.ep_broadcast_u32(0)?;
                let packed = self.ep_broadcast_u32(0)?;
                let (slot_idx, qsa_req, _) = Self::unpack_ep_slot_meta(packed);
                self.sync_secondary()?;
                self.with_worker_slot(seq, slot_idx, false, qsa_req, |s| {
                    self.decode_verify_graphed_k3(&[t0, t1, t2], s, stream)?;
                    let num_accepted = self.ep_broadcast_u32(0)?;
                    if self.config.is_qwen4_exp() {
                        self.commit_ep_short_owned(s, num_accepted as usize, 3)?;
                    } else {
                        self.trim_proposer_state(s, num_accepted as usize, 0)?;
                        match num_accepted {
                            2 => {
                                self.start_checkpoint_async(s)?;
                            }
                            1 => {
                                s.seq_len -= 1;
                                s.tokens.pop();
                                self.start_rollback_and_checkpoint_async(s, 2)?;
                            }
                            _ => {
                                s.seq_len -= 2;
                                s.tokens.pop();
                                s.tokens.pop();
                                self.start_rollback_and_checkpoint_async(s, 1)?;
                            }
                        }
                    }
                    Ok(())
                })?;
            }
            0xFFFFFFF4 => {
                let t0 = self.ep_broadcast_u32(0)?;
                let t1 = self.ep_broadcast_u32(0)?;
                let t2 = self.ep_broadcast_u32(0)?;
                let t3 = self.ep_broadcast_u32(0)?;
                let packed = self.ep_broadcast_u32(0)?;
                let (slot_idx, qsa_req, _) = Self::unpack_ep_slot_meta(packed);
                self.sync_secondary()?;
                self.with_worker_slot(seq, slot_idx, false, qsa_req, |s| {
                    self.decode_verify_graphed_k4(&[t0, t1, t2, t3], s, stream)?;
                    let num_accepted = self.ep_broadcast_u32(0)?;
                    if self.config.is_qwen4_exp() {
                        self.commit_ep_short_owned(s, num_accepted as usize, 4)?;
                    } else {
                        self.trim_proposer_state(s, num_accepted as usize, 0)?;
                        match num_accepted {
                            3 => {
                                self.start_checkpoint_async(s)?;
                            }
                            2 => {
                                s.seq_len -= 1;
                                s.tokens.pop();
                                self.start_rollback_and_checkpoint_async(s, 3)?;
                            }
                            1 => {
                                s.seq_len -= 2;
                                s.tokens.pop();
                                s.tokens.pop();
                                self.start_rollback_and_checkpoint_async(s, 2)?;
                            }
                            _ => {
                                s.seq_len -= 3;
                                s.tokens.pop();
                                s.tokens.pop();
                                s.tokens.pop();
                                self.start_rollback_and_checkpoint_async(s, 1)?;
                            }
                        }
                    }
                    Ok(())
                })?;
            }
            EP_CMD_VERIFY_KGAMMA => {
                let k = self.ep_broadcast_u32(0)? as usize;
                if k == 0 {
                    bail!("EP K=gamma verify received k=0");
                }
                let tokens = self.ep_broadcast_tokens(&vec![0u32; k])?;
                let packed = self.ep_broadcast_u32(0)?;
                let (slot_idx, qsa_req, _) = Self::unpack_ep_slot_meta(packed);
                self.sync_secondary()?;
                self.with_worker_slot(seq, slot_idx, false, qsa_req, |seq| {
                    let pre_verify_len = seq.seq_len;
                    let abort_verify = |seq: &mut SequenceState| -> Result<()> {
                        seq.seq_len = pre_verify_len;
                        seq.tokens.truncate(pre_verify_len);
                        let rollback = <Self as Model>::abort_verify_state(self, seq, k);
                        self.clear_ddtree_parent_ids();
                        rollback
                    };
                    if let Err(error) = self.decode_verify_dflash(&tokens, seq, stream) {
                        abort_verify(seq)?;
                        return Err(error);
                    }
                    let result = match self.ep_broadcast_u32(0) {
                        Ok(result) => result,
                        Err(error) => {
                            abort_verify(seq)?;
                            return Err(error);
                        }
                    };
                    let plan = match ep_verify_kgamma_commit_plan(k, result) {
                        Ok(plan) => plan,
                        Err(error) => {
                            abort_verify(seq)?;
                            return Err(error);
                        }
                    };
                    let Some(plan) = plan else {
                        abort_verify(seq)?;
                        return Ok(());
                    };

                    let target_seq_len = pre_verify_len + plan.total_accepted;
                    if seq.seq_len < target_seq_len || seq.tokens.len() < target_seq_len {
                        let error = anyhow::anyhow!(
                            "EP K=gamma verify produced short sequence state: seq_len={} tokens={} target={target_seq_len}",
                            seq.seq_len,
                            seq.tokens.len()
                        );
                        abort_verify(seq)?;
                        return Err(error);
                    }
                    seq.seq_len = target_seq_len;
                    seq.tokens.truncate(target_seq_len);
                    if let Err(error) = self.commit_verify_state_async(seq, plan.total_accepted, k)
                    {
                        abort_verify(seq)?;
                        return Err(error);
                    }
                    if let Err(error) = self.trim_proposer_state(seq, plan.num_accepted, 0) {
                        self.clear_ddtree_parent_ids();
                        return Err(error);
                    }
                    self.clear_ddtree_parent_ids();
                    Ok(())
                })?;
            }
            EP_CMD_VERIFY_QWEN4_BATCHED => {
                let n_seqs = self.ep_broadcast_u32(0)? as usize;
                let k = self.ep_broadcast_u32(0)? as usize;
                let mut metas = Vec::with_capacity(n_seqs);
                let mut ks = Vec::with_capacity(n_seqs);
                let mut total_tail_tokens = 0usize;
                let mut total_window_tokens = 0usize;
                for _ in 0..n_seqs {
                    let meta = self.ep_broadcast_u32(0)?;
                    let slot_idx = (meta & 0xFFFF) as usize;
                    let qsa_req = ((meta >> 16) & 1) != 0;
                    let seq_k_raw = ((meta >> 17) & 0x7F) as usize;
                    let seq_k = if seq_k_raw == 0 { k } else { seq_k_raw };
                    let tail_len = ((meta >> 24) & 0xFF) as usize;
                    let seq_len = self.ep_broadcast_u32(0)? as usize;
                    self.ensure_worker_slot_allocated(seq, slot_idx)?;
                    total_tail_tokens += tail_len;
                    total_window_tokens += seq_k;
                    ks.push(seq_k);
                    metas.push((slot_idx, seq_len, qsa_req, tail_len, seq_k));
                }
                let frame = crate::layer::Qwen4CrossSeqFrame::new_ragged(&ks)
                    .map_err(anyhow::Error::msg)?;
                let packed_tokens =
                    self.ep_broadcast_tokens(&vec![0u32; total_tail_tokens + total_window_tokens])?;
                let (all_tails, flat_windows) = packed_tokens.split_at(total_tail_tokens);
                let mut tokens_per_seq: Vec<Vec<u32>> = Vec::with_capacity(n_seqs);
                let mut win_cursor = 0usize;
                for &seq_k in &ks {
                    tokens_per_seq.push(flat_windows[win_cursor..win_cursor + seq_k].to_vec());
                    win_cursor += seq_k;
                }
                self.sync_secondary()?;
                let mut taken: Vec<(usize, SequenceState)> = Vec::with_capacity(n_seqs);
                let primary_slot = seq.slot_idx;
                {
                    let mut extra = self.worker_extra_slots.lock();
                    let mut tail_cursor = 0usize;
                    for (slot_idx, seq_len, qsa_req, tail_len, _seq_k) in metas {
                        let tail = &all_tails[tail_cursor..tail_cursor + tail_len];
                        tail_cursor += tail_len;
                        let mut s = if slot_idx == primary_slot {
                            std::mem::replace(seq, SequenceState::empty_placeholder(primary_slot))
                        } else {
                            extra
                                .remove(&slot_idx)
                                .ok_or_else(|| anyhow::anyhow!("missing worker slot {slot_idx}"))?
                        };
                        s.seq_len = seq_len;
                        s.qwen4_qsa_required = qsa_req;
                        s.tokens.resize(seq_len, 0);
                        if tail_len > 0 && tail_len <= seq_len {
                            s.tokens[seq_len - tail_len..seq_len].copy_from_slice(tail);
                        }
                        taken.push((slot_idx, s));
                    }
                }
                let run_res = {
                    let mut refs: Vec<&mut SequenceState> =
                        taken.iter_mut().map(|(_, s)| s).collect();
                    self.decode_verify_qwen4_batched_unowned(frame, &tokens_per_seq, &mut refs)
                };
                {
                    let mut extra = self.worker_extra_slots.lock();
                    for (slot_idx, s) in taken {
                        if slot_idx == primary_slot {
                            *seq = s;
                        } else {
                            extra.insert(slot_idx, s);
                        }
                    }
                }
                let _ = run_res?;
            }
            EP_CMD_DECODE_QWEN4_BATCH => {
                let n_seqs = self.ep_broadcast_u32(0)? as usize;
                let mut metas = Vec::with_capacity(n_seqs);
                let mut total_tail_tokens = 0usize;
                for _ in 0..n_seqs {
                    let meta = self.ep_broadcast_u32(0)?;
                    let slot_idx = (meta & 0xFFFF) as usize;
                    let qsa_req = ((meta >> 16) & 1) != 0;
                    let tail_len = ((meta >> 24) & 0xFF) as usize;
                    let seq_len = self.ep_broadcast_u32(0)? as usize;
                    self.ensure_worker_slot_allocated(seq, slot_idx)?;
                    total_tail_tokens += tail_len;
                    metas.push((slot_idx, seq_len, qsa_req, tail_len));
                }
                let packed_tokens =
                    self.ep_broadcast_tokens(&vec![0u32; total_tail_tokens + n_seqs])?;
                let (all_tails, tokens) = packed_tokens.split_at(total_tail_tokens);
                self.sync_secondary()?;
                let mut taken: Vec<(usize, SequenceState)> = Vec::with_capacity(n_seqs);
                let primary_slot = seq.slot_idx;
                {
                    let mut extra = self.worker_extra_slots.lock();
                    let mut tail_cursor = 0usize;
                    for (slot_idx, seq_len, qsa_req, tail_len) in metas {
                        let tail = &all_tails[tail_cursor..tail_cursor + tail_len];
                        tail_cursor += tail_len;
                        let mut s = if slot_idx == primary_slot {
                            std::mem::replace(seq, SequenceState::empty_placeholder(primary_slot))
                        } else {
                            extra
                                .remove(&slot_idx)
                                .ok_or_else(|| anyhow::anyhow!("missing worker slot {slot_idx}"))?
                        };
                        s.seq_len = seq_len;
                        s.qwen4_qsa_required = qsa_req;
                        s.tokens.resize(seq_len, 0);
                        if tail_len > 0 && tail_len <= seq_len {
                            s.tokens[seq_len - tail_len..seq_len].copy_from_slice(tail);
                        }
                        taken.push((slot_idx, s));
                    }
                }
                let run_res = (|| -> Result<DevicePtr> {
                    let mut refs: Vec<&mut SequenceState> =
                        taken.iter_mut().map(|(_, s)| s).collect();
                    let ptr = self.decode_qwen4_native_batch_dispatch(tokens, &mut refs)?;
                    if self.ssm_pool.has_mtp {
                        self.gpu.synchronize(stream)?;
                        for s in &mut refs {
                            self.start_checkpoint_async(s)?;
                        }
                    }
                    Ok(ptr)
                })();
                {
                    let mut extra = self.worker_extra_slots.lock();
                    for (slot_idx, s) in taken {
                        if slot_idx == primary_slot {
                            *seq = s;
                        } else {
                            extra.insert(slot_idx, s);
                        }
                    }
                }
                let _ = run_res?;
            }
            EP_CMD_COMMIT_VERIFY_SLOT => {
                let slot_idx = self.ep_broadcast_u32(0)? as usize;
                let accepted = self.ep_broadcast_u32(0)? as usize;
                let rows = self.ep_broadcast_u32(0)? as usize;
                let last_slot = self.ep_broadcast_u32(0)? as usize;
                let _lane = crate::layers::qwen4_verify_lanes::LaneGuard::enter_for_slot(slot_idx)
                    .map_err(|e| {
                        e.context(format!(
                            "commit slot={slot_idx} accepted={accepted} rows={rows} last_slot={last_slot}"
                        ))
                    })?;
                if slot_idx == seq.slot_idx {
                    let pre_len = seq.seq_len.saturating_sub(rows);
                    let target = pre_len + accepted;
                    seq.seq_len = target;
                    seq.tokens.truncate(target);
                    self.commit_verify_state_async_dispatch(seq, accepted, rows, last_slot)?;
                } else {
                    let mut extra = self.worker_extra_slots.lock();
                    let s = extra
                        .get_mut(&slot_idx)
                        .ok_or_else(|| anyhow::anyhow!("missing worker slot {slot_idx}"))?;
                    let pre_len = s.seq_len.saturating_sub(rows);
                    let target = pre_len + accepted;
                    s.seq_len = target;
                    s.tokens.truncate(target);
                    self.commit_verify_state_async_dispatch(s, accepted, rows, last_slot)?;
                }
            }
            EP_CMD_ABORT_VERIFY_SLOT => {
                let slot_idx = self.ep_broadcast_u32(0)? as usize;
                let rows = self.ep_broadcast_u32(0)? as usize;
                let cursor = self.ep_broadcast_u32(0)? as usize;
                let _lane = crate::layers::qwen4_verify_lanes::LaneGuard::enter_for_slot(slot_idx)
                    .map_err(|e| {
                        e.context(format!("abort slot={slot_idx} rows={rows} cursor={cursor}"))
                    })?;
                if slot_idx == seq.slot_idx {
                    self.abort_verify_state_dispatch(seq, rows)?;
                    seq.seq_len = cursor;
                    seq.tokens.truncate(cursor);
                } else {
                    let mut extra = self.worker_extra_slots.lock();
                    let s = extra
                        .get_mut(&slot_idx)
                        .ok_or_else(|| anyhow::anyhow!("missing worker slot {slot_idx}"))?;
                    self.abort_verify_state_dispatch(s, rows)?;
                    s.seq_len = cursor;
                    s.tokens.truncate(cursor);
                }
            }
            token => {
                let packed = self.ep_broadcast_u32(0)?;
                let (slot_idx, qsa_req, _) = Self::unpack_ep_slot_meta(packed);
                self.sync_secondary()?;
                self.with_worker_slot(seq, slot_idx, false, qsa_req, |s| {
                    self.decode(token, s, stream)?;
                    if self.ssm_pool.has_mtp {
                        self.gpu.synchronize(stream)?;
                        self.start_checkpoint_async(s)?;
                    }
                    Ok(())
                })?;
            }
        }

        Ok(true)
    }
}

#[cfg(test)]
mod ep_verify_kgamma_tests {
    use super::{EpVerifyKgCommitPlan, ep_verify_kgamma_commit_plan};
    use crate::traits::EP_VERIFY_KGAMMA_ABORT;

    #[test]
    fn abort_discards_the_variable_width_verify() {
        assert_eq!(
            ep_verify_kgamma_commit_plan(17, EP_VERIFY_KGAMMA_ABORT).unwrap(),
            None
        );
    }

    #[test]
    fn every_accept_depth_has_symmetric_trim_and_commit_counts() {
        let k = 5;
        for num_accepted in 0..k {
            assert_eq!(
                ep_verify_kgamma_commit_plan(k, num_accepted as u32).unwrap(),
                Some(EpVerifyKgCommitPlan {
                    num_accepted,
                    total_accepted: num_accepted + 1,
                    drop_tokens: k - num_accepted - 1,
                })
            );
        }
    }

    #[test]
    fn malformed_variable_width_results_fail_closed() {
        assert!(ep_verify_kgamma_commit_plan(0, 0).is_err());
        assert!(ep_verify_kgamma_commit_plan(4, 4).is_err());
        assert!(ep_verify_kgamma_commit_plan(4, 99).is_err());
    }
}
