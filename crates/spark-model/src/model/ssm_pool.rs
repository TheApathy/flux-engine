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
use super::ssm_pool_geometry::{SsmPoolGeometryInput, checked_ssm_pool_geometry};
use super::ssm_snapshot::SsmSnapshotPool;
use super::types::{PinnedMetaStaging, TransformerModel};
use crate::layer::{
    AttnMetadataDev, ForwardContext, GdnPrefillBuffers, LayerState, SsmLayerState, TransformerLayer,
};
use crate::layers::ops;
use crate::speculative::DraftProposer;
use crate::traits::{ChunkedPrefillPageMetadata, Model, SequenceState};
use crate::weight_map::{DenseWeight, MtpWeights, QuantizedWeight};

/// Pre-allocated contiguous GPU memory pool for SSM layer states.
///
/// Each pool slot has fixed GPU addresses for h_state and conv_state across
/// all SSM layers. This enables CUDA graph capture at batch sizes > 1 because
/// the graph embeds memory addresses that remain stable across replays.
pub(crate) struct SsmStatePool {
    pub(super) allocation_geometry: super::ssm_pool_geometry::SsmPoolGeometry,
    pub(super) h_state_pools: Vec<DevicePtr>,
    pub(super) conv_state_pools: Vec<DevicePtr>,
    /// Per-slot intermediate state pools (only allocated when `has_mtp && !compact_lazy_h`).
    /// Layout: `[num_ssm_layers]`, each allocation is sized by the checked
    /// dummy-inclusive geometry authority.
    pub(super) h_intermediate_pools: Vec<DevicePtr>,
    pub(super) conv_intermediate_pools: Vec<DevicePtr>,
    /// Shared per-slot 1-copy `H` verify write-sink under `compact_lazy_h`
    /// (`spec_slots * h_bytes`, reused sequentially across SSM layers).
    pub(super) h_verify_scratch: DevicePtr,
    /// Per-slot SSM state checkpoint pools (`h_checkpoint_pools` omitted under `compact_lazy_h`).
    pub(super) h_checkpoint_pools: Vec<DevicePtr>,
    pub(super) conv_checkpoint_pools: Vec<DevicePtr>,
    /// WY17 LAZY-commit retention pools (only allocated when has_mtp AND the
    /// lazy-commit gate is on). Each holds one full per-verify k/q/v buffer
    /// (`[K, conv_dim]` BF16 = `kv_retain_bytes`) and gate/beta buffer
    /// (`[K, 2*nv]` FP32 = `gate_retain_bytes`) per slot per layer, so the
    /// commit path can feed the SAME inputs to `gated_delta_rule_wy17_replay`.
    /// Fixed addresses (like the intermediate pools) for CUDA-graph stability.
    pub(super) wy17_kv_retain_pools: Vec<DevicePtr>,
    pub(super) wy17_gate_retain_pools: Vec<DevicePtr>,
    /// Byte size of one retained k/q/v buffer (`K * conv_dim * 2`), 0 if unused.
    pub(super) kv_retain_bytes: usize,
    /// Byte size of one retained gate/beta buffer (`K * 2*nv * 4`), 0 if unused.
    pub(super) gate_retain_bytes: usize,
    pub(super) h_bytes: usize,
    pub(super) conv_bytes: usize,
    /// Number of CLAIMABLE slots (excludes the reserved dummy slot at
    /// index `max_slots`). All claim_slot/release_slot operations work
    /// in `[0, max_slots)`.
    pub(super) max_slots: usize,
    /// Slots whose intermediate and checkpoint state exists (`max_slots + 1`,
    /// or `max_slots` under `ATLAS_SSM_DUMMY_STATE_ONLY=1` / `compact_lazy_h`).
    pub(super) spec_slots: usize,
    pub(super) num_ssm_layers: usize,
    pub(super) has_mtp: bool,
    pub(super) compact_lazy_h: bool,
    pub(super) num_intermediates: usize,
    pub(super) free_slots: Mutex<Vec<usize>>,
}

impl SsmStatePool {
    pub(super) fn new(
        config: &ModelConfig,
        max_slots: usize,
        has_mtp: bool,
        num_intermediates: usize,
        gpu: &dyn GpuBackend,
    ) -> Result<Self> {
        Self::new_with_mode(config, max_slots, has_mtp, false, num_intermediates, gpu)
    }

    pub(super) fn new_with_mode(
        config: &ModelConfig,
        max_slots: usize,
        has_mtp: bool,
        native_qwen4_mtp: bool,
        num_intermediates: usize,
        gpu: &dyn GpuBackend,
    ) -> Result<Self> {
        let num_ssm_layers = config.num_ssm_layers();
        let lazy_commit = crate::layers::wy17_lazy_commit();
        let geometry = checked_ssm_pool_geometry(
            SsmPoolGeometryInput::from_config(
                config,
                max_slots,
                has_mtp,
                num_intermediates,
                lazy_commit,
            )?
            .with_qwen4_native_mtp(config, native_qwen4_mtp),
        )?;
        let total_slots = geometry.total_slots;
        let h_bytes = geometry.h_bytes;
        let conv_bytes = geometry.conv_bytes;

        let mut h_state_pools = Vec::with_capacity(num_ssm_layers);
        let mut conv_state_pools = Vec::with_capacity(num_ssm_layers);
        let mut h_intermediate_pools = Vec::new();
        let mut conv_intermediate_pools = Vec::new();
        let mut h_verify_scratch = DevicePtr::NULL;
        let mut h_checkpoint_pools = Vec::new();
        let mut conv_checkpoint_pools = Vec::new();
        let mut wy17_kv_retain_pools = Vec::new();
        let mut wy17_gate_retain_pools = Vec::new();

        let kv_retain_bytes = geometry.kv_retain_bytes;
        let gate_retain_bytes = geometry.gate_retain_bytes;

        // Predict the pool footprint BEFORE allocating any of it.
        {
            let per_slot = geometry.bytes_per_layer_slot * num_ssm_layers
                + if geometry.compact_lazy_h { h_bytes } else { 0 };
            let predicted = geometry.total_bytes;
            let free = gpu.free_memory().unwrap_or(0);
            tracing::info!(
                "SSM pool pre-flight: {} layers × {} slots × (ni={}, compact_h={}) = {:.1} GB predicted \
                 ({:.2} GB/slot), {:.1} GB free",
                num_ssm_layers,
                total_slots,
                num_intermediates,
                geometry.compact_lazy_h,
                predicted as f64 / (1024.0 * 1024.0 * 1024.0),
                per_slot as f64 / (1024.0 * 1024.0 * 1024.0),
                free as f64 / (1024.0 * 1024.0 * 1024.0),
            );
            if free > 0 && predicted > free {
                anyhow::bail!(
                    "SSM pools need {:.1} GB but only {:.1} GB is free. This scales \
                     linearly with --max-batch-size ({:.2} GB per slot) and with the \
                     verify window (--dflash-gamma {} → ni={}). Reduce --max-batch-size, \
                     lower --dflash-gamma, or disable MTP/DFlash. \
                     NOTE: on unified-memory parts (GB10) exceeding this global-OOMs the \
                     host rather than failing cleanly, so this is a hard error.",
                    predicted as f64 / (1024.0 * 1024.0 * 1024.0),
                    free as f64 / (1024.0 * 1024.0 * 1024.0),
                    per_slot as f64 / (1024.0 * 1024.0 * 1024.0),
                    num_intermediates.saturating_sub(1),
                    num_intermediates,
                );
            }
        }

        for _ in 0..num_ssm_layers {
            let h_pool = gpu.alloc(geometry.h_state_allocation_bytes)?;
            gpu.memset(h_pool, 0, geometry.h_state_allocation_bytes)?;
            h_state_pools.push(h_pool);

            let conv_pool = gpu.alloc(geometry.conv_state_allocation_bytes)?;
            gpu.memset(conv_pool, 0, geometry.conv_state_allocation_bytes)?;
            conv_state_pools.push(conv_pool);
        }

        if has_mtp && num_ssm_layers > 0 {
            let ni = num_intermediates;
            if geometry.compact_lazy_h && geometry.h_verify_scratch_bytes > 0 {
                let scratch = gpu.alloc(geometry.h_verify_scratch_bytes)?;
                gpu.memset(scratch, 0, geometry.h_verify_scratch_bytes)?;
                h_verify_scratch = scratch;
            }
            for _ in 0..num_ssm_layers {
                if !geometry.compact_lazy_h {
                    let h_inter = gpu.alloc(geometry.h_intermediate_allocation_bytes)?;
                    gpu.memset(h_inter, 0, geometry.h_intermediate_allocation_bytes)?;
                    h_intermediate_pools.push(h_inter);
                }

                let conv_inter = gpu.alloc(geometry.conv_intermediate_allocation_bytes)?;
                gpu.memset(conv_inter, 0, geometry.conv_intermediate_allocation_bytes)?;
                conv_intermediate_pools.push(conv_inter);

                if !geometry.compact_lazy_h {
                    // 1 checkpoint per speculative slot per layer
                    let h_ckpt = gpu.alloc(geometry.h_checkpoint_allocation_bytes)?;
                    gpu.memset(h_ckpt, 0, geometry.h_checkpoint_allocation_bytes)?;
                    h_checkpoint_pools.push(h_ckpt);
                }

                let conv_ckpt = gpu.alloc(geometry.conv_checkpoint_allocation_bytes)?;
                gpu.memset(conv_ckpt, 0, geometry.conv_checkpoint_allocation_bytes)?;
                conv_checkpoint_pools.push(conv_ckpt);

                if kv_retain_bytes > 0 {
                    let kv_ret = gpu.alloc(geometry.kv_retain_allocation_bytes)?;
                    gpu.memset(kv_ret, 0, geometry.kv_retain_allocation_bytes)?;
                    wy17_kv_retain_pools.push(kv_ret);

                    let gate_ret = gpu.alloc(geometry.gate_retain_allocation_bytes)?;
                    gpu.memset(gate_ret, 0, geometry.gate_retain_allocation_bytes)?;
                    wy17_gate_retain_pools.push(gate_ret);
                }
            }

            if kv_retain_bytes > 0 {
                let retain_mb = num_ssm_layers
                    * (geometry.kv_retain_allocation_bytes + geometry.gate_retain_allocation_bytes)
                    / (1024 * 1024);
                tracing::info!(
                    "SSM WY17 LAZY-commit retention pools (K={num_intermediates}): {retain_mb} MB"
                );
            }

            let mtp_mb = (geometry.h_verify_scratch_bytes
                + num_ssm_layers
                    * (geometry.h_intermediate_allocation_bytes
                        + geometry.conv_intermediate_allocation_bytes
                        + geometry.h_checkpoint_allocation_bytes
                        + geometry.conv_checkpoint_allocation_bytes))
                / (1024 * 1024);
            tracing::info!(
                "SSM MTP pools ({ni} intermediates + checkpoints, compact_h={}): {mtp_mb} MB",
                geometry.compact_lazy_h
            );
            if geometry.compact_lazy_h {
                let saved_mib = (num_ssm_layers * geometry.spec_slots * (ni + 1) * h_bytes)
                    .saturating_sub(geometry.h_verify_scratch_bytes)
                    / (1024 * 1024);
                tracing::info!(
                    "SSM_COMPACT_LAZY_H_ENGAGED selector=ATLAS_QWEN4_REPLAY_SSM \
                     spec_slots={} scratch_mib={} saved_mib={saved_mib}",
                    geometry.spec_slots,
                    geometry.h_verify_scratch_bytes / (1024 * 1024),
                );
            } else if geometry.spec_slots < geometry.total_slots {
                let saved_mib = num_ssm_layers
                    * (geometry.total_slots - geometry.spec_slots)
                    * (ni + 1)
                    * (h_bytes + conv_bytes)
                    / (1024 * 1024);
                tracing::info!(
                    "SSM_DUMMY_STATE_ONLY_ENGAGED selector=ATLAS_SSM_DUMMY_STATE_ONLY \
                     spec_slots={} saved_mib={saved_mib}",
                    geometry.spec_slots
                );
            }
        }

        // free_slots holds claimable indices only; the dummy at index
        // `max_slots` is permanently reserved. Attention-only models still need
        // one claimable index per sequence; with no SSM layers it owns no state.
        let free_slots: Vec<usize> = (0..max_slots).rev().collect();

        let total_mb = geometry.total_bytes / (1024 * 1024);
        tracing::info!(
            "SSM state pool: {max_slots} slots × {num_ssm_layers} layers = {total_mb} MB",
        );

        Ok(Self {
            allocation_geometry: geometry,
            h_state_pools,
            conv_state_pools,
            h_intermediate_pools,
            conv_intermediate_pools,
            h_verify_scratch,
            h_checkpoint_pools,
            conv_checkpoint_pools,
            wy17_kv_retain_pools,
            wy17_gate_retain_pools,
            kv_retain_bytes,
            gate_retain_bytes,
            h_bytes,
            conv_bytes,
            max_slots,
            spec_slots: geometry.spec_slots,
            num_ssm_layers,
            has_mtp,
            compact_lazy_h: geometry.compact_lazy_h,
            num_intermediates,
            free_slots: Mutex::new(free_slots),
        })
    }

    pub(super) fn claim_slot(&self) -> Result<usize> {
        let mut free = self.free_slots.lock();
        let (min_pos, _) = free
            .iter()
            .enumerate()
            .min_by_key(|&(_, &slot)| slot)
            .ok_or_else(|| {
                anyhow::anyhow!("SSM state pool exhausted (max {} slots)", self.max_slots)
            })?;
        Ok(free.remove(min_pos))
    }

    /// Slots currently held by live sequences; `None` without SSM layers (no
    /// slot census). Read by the `ATLAS_MTP_LASTK_DEFER` exactness guard.
    pub(super) fn claimed_slots(&self) -> Option<usize> {
        if self.num_ssm_layers == 0 {
            return None;
        }
        Some(self.max_slots.saturating_sub(self.free_slots.lock().len()))
    }

    pub(super) fn release_slot(&self, idx: usize) {
        let mut free = self.free_slots.lock();
        // UNCONDITIONAL (upstream 11e83f5b; the fork never even had the
        // `debug_assert!`): a slot pushed twice stays on the free list twice
        // for the life of the process, and `claim_slot` then hands it to two
        // sequences on every later overlap. Dropping the second push turns a
        // slot double-ownership bug into a one-request fault instead of a
        // process-lifetime one. `max_slots` is small, so the scan is free.
        if free.contains(&idx) {
            tracing::error!(
                "release_slot: slot {idx} already free — double release ignored \
                 (two sequences owned this slot; their SSM state was shared)"
            );
            return;
        }
        // Keep `free_slots` sorted descending so the lowest free `slot_idx`
        // always sits at the back and solo C=1 traffic after a multi-seq burst
        // lands back on `slot_idx = 0` (preserving the PLE suffix graph).
        let pos = free.partition_point(|&s| s > idx);
        free.insert(pos, idx);
    }

    /// Remove a SPECIFIC slot from the free list if present, returning whether
    /// it was. Used by `compact_sequence` to claim its migration target
    /// EXCLUSIVELY, so a slot is never simultaneously owned and free (an
    /// owned-and-free slot gets handed to two sequences by `claim_slot`,
    /// sharing GDN state → cross-stream corruption).
    ///
    /// `false` means the slot is OWNED by a live sequence (active, prefilling,
    /// anything): the caller must NOT copy onto it. The free list is the only
    /// source of truth for that — the scheduler's active list does not see
    /// sequences that are still prefilling.
    pub(super) fn claim_specific(&self, slot: usize) -> bool {
        let mut free = self.free_slots.lock();
        if let Some(pos) = free.iter().position(|&s| s == slot) {
            free.remove(pos);
            true
        } else {
            false
        }
    }

    /// Reserve a free destination before pre-copy validation. A failed
    /// preflight must leave device/hidden ownership untouched; return its
    /// reservation immediately so an error cannot strand an unowned slot.
    pub(super) fn claim_specific_with_preflight<T>(
        &self,
        slot: usize,
        prepare: impl FnOnce() -> Result<T>,
    ) -> Result<Option<T>> {
        if !self.claim_specific(slot) {
            return Ok(None);
        }
        match prepare() {
            Ok(prepared) => Ok(Some(prepared)),
            Err(error) => {
                self.release_slot(slot);
                Err(error)
            }
        }
    }

    /// Reserved pool slot used by `decode_batch` / `mixed_forward` padding.
    /// Never claimed by `claim_slot()`, never released. SSM kernels are
    /// free to read/write this slot's base state without affecting any
    /// active sequence; its intermediates and checkpoint exist only while
    /// `ATLAS_SSM_DUMMY_STATE_ONLY` is off.
    #[inline]
    pub(super) fn dummy_slot(&self) -> usize {
        self.max_slots
    }

    /// Zero h_state and conv_state for a slot across all SSM layers.
    /// Must be called on slot allocation to prevent stale SSM state
    /// from prior sequences from corrupting new prefill output.
    pub(super) fn zero_slot(&self, idx: usize, gpu: &dyn GpuBackend, stream: u64) -> Result<()> {
        for i in 0..self.num_ssm_layers {
            gpu.memset_async(self.h_state(i, idx), 0, self.h_bytes, stream)?;
            gpu.memset_async(self.conv_state(i, idx), 0, self.conv_bytes, stream)?;
        }
        Ok(())
    }

    pub(super) fn h_state(&self, ssm_layer_idx: usize, slot: usize) -> DevicePtr {
        self.h_state_pools[ssm_layer_idx].offset(slot * self.h_bytes)
    }

    pub(super) fn conv_state(&self, ssm_layer_idx: usize, slot: usize) -> DevicePtr {
        self.conv_state_pools[ssm_layer_idx].offset(slot * self.conv_bytes)
    }

    #[inline]
    pub(super) fn h_intermediate_stride(&self) -> usize {
        if self.compact_lazy_h { 0 } else { self.h_bytes }
    }

    /// Get fixed-address intermediate h_state for verify.
    /// Under `compact_lazy_h`, every token maps to the slot's 1-copy shared
    /// verify write-sink (`h_intermediate_stride() == 0`).
    pub(super) fn h_intermediate(
        &self,
        ssm_layer_idx: usize,
        slot: usize,
        token_idx: usize,
    ) -> DevicePtr {
        self.assert_spec_slot(slot);
        self.assert_intermediate_idx(token_idx);
        if self.compact_lazy_h {
            return self.h_verify_scratch.offset(slot * self.h_bytes);
        }
        let ni = self.num_intermediates;
        self.h_intermediate_pools[ssm_layer_idx].offset((slot * ni + token_idx) * self.h_bytes)
    }

    pub(super) fn conv_intermediate(
        &self,
        ssm_layer_idx: usize,
        slot: usize,
        token_idx: usize,
    ) -> DevicePtr {
        self.assert_spec_slot(slot);
        self.assert_intermediate_idx(token_idx);
        let ni = self.num_intermediates;
        self.conv_intermediate_pools[ssm_layer_idx]
            .offset((slot * ni + token_idx) * self.conv_bytes)
    }

    pub(super) fn h_checkpoint(&self, ssm_layer_idx: usize, slot: usize) -> DevicePtr {
        self.assert_spec_slot(slot);
        self.h_checkpoint_pools[ssm_layer_idx].offset(slot * self.h_bytes)
    }

    pub(super) fn h_checkpoint_opt(&self, ssm_layer_idx: usize, slot: usize) -> Option<DevicePtr> {
        self.assert_spec_slot(slot);
        if self.compact_lazy_h {
            None
        } else {
            Some(self.h_checkpoint_pools[ssm_layer_idx].offset(slot * self.h_bytes))
        }
    }

    pub(super) fn conv_checkpoint(&self, ssm_layer_idx: usize, slot: usize) -> DevicePtr {
        self.assert_spec_slot(slot);
        self.conv_checkpoint_pools[ssm_layer_idx].offset(slot * self.conv_bytes)
    }

    /// The speculative pools hold `spec_slots` slots; an address past them
    /// would alias other memory, so fail loudly instead.
    #[inline]
    fn assert_spec_slot(&self, slot: usize) {
        assert!(
            slot < self.spec_slots,
            "SSM slot {slot} has no intermediate/checkpoint state (spec_slots={}, \
             ATLAS_SSM_DUMMY_STATE_ONLY)",
            self.spec_slots
        );
    }

    /// Each speculative slot owns `num_intermediates` intermediate entries
    /// (`token_idx < num_intermediates`); an out-of-range index would silently
    /// alias into the next slot's intermediate state, so fail loudly instead.
    #[inline]
    fn assert_intermediate_idx(&self, token_idx: usize) {
        assert!(
            token_idx < self.num_intermediates,
            "SSM intermediate index {token_idx} out of bounds (num_intermediates={})",
            self.num_intermediates
        );
    }

    /// WY17 LAZY-commit k/q/v retention buffer for `(layer, slot)`. `None` when
    /// retention pools weren't allocated (lazy-commit gate off or `!has_mtp`).
    pub(super) fn wy17_kv_retain(&self, ssm_layer_idx: usize, slot: usize) -> Option<DevicePtr> {
        if self.wy17_kv_retain_pools.is_empty() {
            return None;
        }
        Some(self.wy17_kv_retain_pools[ssm_layer_idx].offset(slot * self.kv_retain_bytes))
    }

    /// WY17 LAZY-commit gate/beta retention buffer for `(layer, slot)`.
    pub(super) fn wy17_gate_retain(&self, ssm_layer_idx: usize, slot: usize) -> Option<DevicePtr> {
        if self.wy17_gate_retain_pools.is_empty() {
            return None;
        }
        Some(self.wy17_gate_retain_pools[ssm_layer_idx].offset(slot * self.gate_retain_bytes))
    }

    pub(super) fn reset_slot(&self, slot: usize, gpu: &dyn GpuBackend) -> Result<()> {
        if crate::layers::ssm_reset_async_enabled() {
            let stream = gpu.default_stream();
            for i in 0..self.num_ssm_layers {
                gpu.memset_async(self.h_state(i, slot), 0, self.h_bytes, stream)?;
                gpu.memset_async(self.conv_state(i, slot), 0, self.conv_bytes, stream)?;
                if self.has_mtp {
                    if self.num_intermediates > 0 {
                        if !self.compact_lazy_h {
                            gpu.memset_async(
                                self.h_intermediate(i, slot, 0),
                                0,
                                self.h_bytes * self.num_intermediates,
                                stream,
                            )?;
                        }
                        gpu.memset_async(
                            self.conv_intermediate(i, slot, 0),
                            0,
                            self.conv_bytes * self.num_intermediates,
                            stream,
                        )?;
                    }
                    if !self.compact_lazy_h {
                        gpu.memset_async(self.h_checkpoint(i, slot), 0, self.h_bytes, stream)?;
                    }
                    gpu.memset_async(self.conv_checkpoint(i, slot), 0, self.conv_bytes, stream)?;
                }
            }
            if self.compact_lazy_h && self.num_ssm_layers > 0 && slot < self.spec_slots {
                gpu.memset_async(
                    self.h_verify_scratch.offset(slot * self.h_bytes),
                    0,
                    self.h_bytes,
                    stream,
                )?;
            }
            gpu.synchronize(stream)?;
            return Ok(());
        }
        for i in 0..self.num_ssm_layers {
            gpu.memset(self.h_state(i, slot), 0, self.h_bytes)?;
            gpu.memset(self.conv_state(i, slot), 0, self.conv_bytes)?;
            if self.has_mtp {
                for t in 0..self.num_intermediates {
                    if !self.compact_lazy_h {
                        gpu.memset(self.h_intermediate(i, slot, t), 0, self.h_bytes)?;
                    }
                    gpu.memset(self.conv_intermediate(i, slot, t), 0, self.conv_bytes)?;
                }
                if !self.compact_lazy_h {
                    gpu.memset(self.h_checkpoint(i, slot), 0, self.h_bytes)?;
                }
                gpu.memset(self.conv_checkpoint(i, slot), 0, self.conv_bytes)?;
            }
        }
        if self.compact_lazy_h && self.num_ssm_layers > 0 && slot < self.spec_slots {
            gpu.memset(
                self.h_verify_scratch.offset(slot * self.h_bytes),
                0,
                self.h_bytes,
            )?;
        }
        Ok(())
    }

    /// The geometry [`super::slot_zero_plan`] needs.
    pub(super) fn slot_zero_geometry(&self) -> super::slot_zero_plan::SlotGeometry {
        super::slot_zero_plan::SlotGeometry {
            num_layers: self.num_ssm_layers,
            h_bytes: self.h_bytes,
            conv_bytes: self.conv_bytes,
            has_mtp: self.has_mtp,
            compact_lazy_h: self.compact_lazy_h,
            num_intermediates: self.num_intermediates,
        }
    }

    /// `ATLAS_ALLOC_SEQ_ASYNC`: every byte [`Self::zero_slot`] and
    /// [`Self::reset_slot`] zero for `slot`, each once
    /// ([`super::slot_zero_plan::async_plan`]), as stream-ordered memsets on
    /// `stream` and nothing else: no host sync. The caller orders every other
    /// stream after them with an event. Returns (memsets, bytes).
    #[allow(dead_code)]
    pub(super) fn zero_slot_once_async(
        &self,
        slot: usize,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<(usize, usize)> {
        use super::slot_zero_plan::{Pool, async_plan};
        if self.has_mtp {
            // The same guard the per-token accessors apply on the synchronous path.
            self.assert_spec_slot(slot);
        }
        let plan = async_plan(&self.slot_zero_geometry(), slot);
        let mut bytes = 0usize;
        for r in &plan {
            let base = match r.pool {
                Pool::HState => self.h_state_pools[r.layer],
                Pool::ConvState => self.conv_state_pools[r.layer],
                Pool::HIntermediate => self.h_intermediate_pools[r.layer],
                Pool::ConvIntermediate => self.conv_intermediate_pools[r.layer],
                Pool::HCheckpoint => self.h_checkpoint_pools[r.layer],
                Pool::ConvCheckpoint => self.conv_checkpoint_pools[r.layer],
            };
            gpu.memset_async(base.offset(r.offset), 0, r.bytes, stream)?;
            bytes += r.bytes;
        }
        let mut memsets = plan.len();
        if self.compact_lazy_h && self.num_ssm_layers > 0 {
            gpu.memset_async(
                self.h_verify_scratch.offset(slot * self.h_bytes),
                0,
                self.h_bytes,
                stream,
            )?;
            memsets += 1;
            bytes += self.h_bytes;
        }
        Ok((memsets, bytes))
    }

    /// `ATLAS_ALLOC_SEQ_ASYNC` split plan: enqueue the prefill-essential memsets
    /// (`HState` + `ConvState`) on `stream` first, record `slot_zero_event` on
    /// `stream` immediately after those prefill-essential memsets, and then
    /// enqueue the MTP speculative-decode-only memsets (`HIntermediate`,
    /// `ConvIntermediate`, `HCheckpoint`, `ConvCheckpoint`) on `stream` after
    /// `record_event`. Returns `(memsets, bytes)`.
    pub(super) fn zero_slot_split_async(
        &self,
        slot: usize,
        gpu: &dyn GpuBackend,
        stream: u64,
        slot_zero_event: u64,
    ) -> Result<(usize, usize)> {
        use super::slot_zero_plan::{Pool, async_split_plan};
        if self.has_mtp {
            self.assert_spec_slot(slot);
        }
        let (prefill_plan, mtp_plan) = async_split_plan(&self.slot_zero_geometry(), slot);
        let mut bytes = 0usize;
        for r in &prefill_plan {
            let base = match r.pool {
                Pool::HState => self.h_state_pools[r.layer],
                Pool::ConvState => self.conv_state_pools[r.layer],
                Pool::HIntermediate => self.h_intermediate_pools[r.layer],
                Pool::ConvIntermediate => self.conv_intermediate_pools[r.layer],
                Pool::HCheckpoint => self.h_checkpoint_pools[r.layer],
                Pool::ConvCheckpoint => self.conv_checkpoint_pools[r.layer],
            };
            gpu.memset_async(base.offset(r.offset), 0, r.bytes, stream)?;
            bytes += r.bytes;
        }
        gpu.record_event(slot_zero_event, stream)?;
        for r in &mtp_plan {
            let base = match r.pool {
                Pool::HState => self.h_state_pools[r.layer],
                Pool::ConvState => self.conv_state_pools[r.layer],
                Pool::HIntermediate => self.h_intermediate_pools[r.layer],
                Pool::ConvIntermediate => self.conv_intermediate_pools[r.layer],
                Pool::HCheckpoint => self.h_checkpoint_pools[r.layer],
                Pool::ConvCheckpoint => self.conv_checkpoint_pools[r.layer],
            };
            gpu.memset_async(base.offset(r.offset), 0, r.bytes, stream)?;
            bytes += r.bytes;
        }
        let mut memsets = prefill_plan.len() + mtp_plan.len();
        if self.compact_lazy_h && self.num_ssm_layers > 0 {
            gpu.memset_async(
                self.h_verify_scratch.offset(slot * self.h_bytes),
                0,
                self.h_bytes,
                stream,
            )?;
            memsets += 1;
            bytes += self.h_bytes;
        }
        Ok((memsets, bytes))
    }

    pub(super) fn copy_slot(
        &self,
        from: usize,
        to: usize,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        for i in 0..self.num_ssm_layers {
            gpu.copy_d2d_async(
                self.h_state(i, from),
                self.h_state(i, to),
                self.h_bytes,
                stream,
            )?;
            gpu.copy_d2d_async(
                self.conv_state(i, from),
                self.conv_state(i, to),
                self.conv_bytes,
                stream,
            )?;
            if self.has_mtp {
                for t in 0..self.num_intermediates {
                    if !self.compact_lazy_h {
                        gpu.copy_d2d_async(
                            self.h_intermediate(i, from, t),
                            self.h_intermediate(i, to, t),
                            self.h_bytes,
                            stream,
                        )?;
                    }
                    gpu.copy_d2d_async(
                        self.conv_intermediate(i, from, t),
                        self.conv_intermediate(i, to, t),
                        self.conv_bytes,
                        stream,
                    )?;
                }
                if !self.compact_lazy_h {
                    gpu.copy_d2d_async(
                        self.h_checkpoint(i, from),
                        self.h_checkpoint(i, to),
                        self.h_bytes,
                        stream,
                    )?;
                }
                gpu.copy_d2d_async(
                    self.conv_checkpoint(i, from),
                    self.conv_checkpoint(i, to),
                    self.conv_bytes,
                    stream,
                )?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "ssm_pool_capacity_tests.rs"]
mod capacity_tests;

#[cfg(test)]
#[path = "ssm_pool_slot_tests.rs"]
mod slot_tests;

#[cfg(test)]
mod lowest_slot_tests {
    use super::*;

    fn test_pool(max_slots: usize) -> SsmStatePool {
        SsmStatePool {
            allocation_geometry: Default::default(),
            h_state_pools: Vec::new(),
            conv_state_pools: Vec::new(),
            h_intermediate_pools: Vec::new(),
            conv_intermediate_pools: Vec::new(),
            h_verify_scratch: DevicePtr::NULL,
            h_checkpoint_pools: Vec::new(),
            conv_checkpoint_pools: Vec::new(),
            wy17_kv_retain_pools: Vec::new(),
            wy17_gate_retain_pools: Vec::new(),
            kv_retain_bytes: 0,
            gate_retain_bytes: 0,
            h_bytes: 0,
            conv_bytes: 0,
            max_slots,
            spec_slots: max_slots,
            num_ssm_layers: 0,
            has_mtp: false,
            compact_lazy_h: false,
            num_intermediates: 0,
            free_slots: Mutex::new((0..max_slots).rev().collect()),
        }
    }

    #[test]
    fn claim_always_returns_lowest_free_slot_after_multi_seq_burst() {
        let pool = test_pool(8);
        let claimed: Vec<usize> = (0..4).map(|_| pool.claim_slot().unwrap()).collect();
        assert_eq!(claimed, vec![0, 1, 2, 3]);
        // Release in ascending order (LIFO would hand out 3 next; lowest-slot must hand out 0).
        for slot in [0, 2, 1, 3] {
            pool.release_slot(slot);
        }
        assert_eq!(pool.claim_slot().unwrap(), 0);
        assert_eq!(pool.claim_slot().unwrap(), 1);
        pool.release_slot(0);
        assert_eq!(pool.claim_slot().unwrap(), 0);
    }

    #[test]
    fn wide_k7_and_k8_intermediates_at_c16_stay_disjoint_and_compact_in_ram() {
        use super::super::ssm_pool_geometry::{
            SsmPoolGeometryInput, checked_ssm_pool_geometry, checked_ssm_speculative_geometry,
        };
        use spark_runtime::gpu::mock::MockGpuBackend;

        // K=7 (num_drafts=6 -> ni=8) and K=8 (num_drafts=7 -> ni=9) at C=16.
        let h_bytes = 48 * 128 * 128 * 4; // 3 MiB per layer-slot
        let conv_bytes = (2 * 16 * 128 + 48 * 128) * 4 * 4; // 160 KiB per layer-slot
        for (num_drafts, k) in [(6usize, 7usize), (7usize, 8usize)] {
            let spec = checked_ssm_speculative_geometry(true, false, num_drafts, None).unwrap();
            assert!(spec.num_intermediates >= k);
            assert_eq!(spec.num_intermediates, num_drafts + 2);

            let compact = checked_ssm_pool_geometry(SsmPoolGeometryInput {
                max_slots: 16,
                num_ssm_layers: 36,
                h_bytes,
                conv_bytes,
                has_mtp: true,
                compact_lazy_h: true,
                num_intermediates: spec.num_intermediates,
                lazy_commit: false,
                num_key_heads: 16,
                key_head_dim: 128,
                num_value_heads: 48,
                value_head_dim: 128,
            })
            .unwrap();
            assert_eq!(compact.spec_slots, 16);
            assert_eq!(compact.h_intermediate_allocation_bytes, 0);
            assert_eq!(compact.h_verify_scratch_bytes, 16 * h_bytes);
            assert_eq!(
                compact.conv_intermediate_allocation_bytes,
                16 * spec.num_intermediates * conv_bytes
            );
            // Total C=16 pool at ni=8..9 stays compact (< 3.5 GiB across all 36 SSM layers).
            assert!(
                compact.total_bytes < 3_500_000_000,
                "k={k} compact total_bytes={} exceeded budget",
                compact.total_bytes
            );

            let gpu = MockGpuBackend::new();
            let mut config = ModelConfig::qwen3_next_80b_nvfp4();
            config.layer_types = vec![LayerType::LinearAttention; 2];
            config.linear_num_key_heads = 1;
            config.linear_key_head_dim = 1;
            config.linear_num_value_heads = 1;
            config.linear_value_head_dim = 1;
            config.linear_conv_kernel_dim = 2;
            let pool = SsmStatePool::new(&config, 16, true, spec.num_intermediates, &gpu).unwrap();
            assert_eq!(pool.num_intermediates, spec.num_intermediates);
            let conv_base = pool.conv_intermediate_pools[0].0;
            let conv_end =
                conv_base + pool.allocation_geometry.conv_intermediate_allocation_bytes as u64;
            let mut seen = std::collections::HashSet::new();
            for slot in 0..16 {
                for tok in 0..k {
                    let ptr = pool.conv_intermediate(0, slot, tok);
                    assert!(
                        ptr.0 >= conv_base && ptr.0 + pool.conv_bytes as u64 <= conv_end,
                        "k={k} slot={slot} tok={tok} out of conv_intermediate bounds"
                    );
                    assert!(
                        seen.insert(ptr.0),
                        "k={k} slot={slot} tok={tok} aliased another intermediate"
                    );
                }
            }
        }
    }

    #[test]
    #[should_panic(expected = "SSM intermediate index 6 out of bounds")]
    fn conv_intermediate_panics_on_oob_token_index_instead_of_aliasing_next_slot() {
        use spark_runtime::gpu::mock::MockGpuBackend;

        let gpu = MockGpuBackend::new();
        let mut config = ModelConfig::qwen3_next_80b_nvfp4();
        config.layer_types = vec![LayerType::LinearAttention; 1];
        config.linear_num_key_heads = 1;
        config.linear_key_head_dim = 1;
        config.linear_num_value_heads = 1;
        config.linear_value_head_dim = 1;
        config.linear_conv_kernel_dim = 2;
        let pool = SsmStatePool::new(&config, 4, true, 6, &gpu).unwrap();
        let _ = pool.conv_intermediate(0, 0, 6);
    }
}
