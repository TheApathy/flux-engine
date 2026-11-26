// SPDX-License-Identifier: AGPL-3.0-only

#![allow(unused_imports, dead_code)]

use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Result, anyhow, bail, ensure};
use atlas_core::config::{LayerType, ModelConfig};
use spark_runtime::buffers::BufferArena;
use spark_runtime::gpu::{DevicePtr, GpuBackend, GraphHandle, KernelHandle};
use spark_runtime::kv_cache::PagedKvCache;

use super::block_mgmt::{
    apply_evicted_blocks, ensure_blocks_through_decode, ensure_blocks_through_prefill,
    extract_layer_refs, reuse_prefix_match_disk_ids,
};
use super::ssm_pool::SsmStatePool;
use super::types::{PinnedMetaStaging, TransformerModel};
use crate::layer::{
    AttnMetadataDev, ForwardContext, GdnPrefillBuffers, LayerState, SsmLayerState, TransformerLayer,
};
use crate::layers::ops;
use crate::speculative::DraftProposer;
use crate::traits::{ChunkedPrefillPageMetadata, Model, SequenceState};
use crate::weight_map::{DenseWeight, MtpWeights, QuantizedWeight};

const SNAP_MARCONI: super::seqstate::slot_history::HistoryRegion =
    super::seqstate::slot_history::HistoryRegion::Marconi;
const SNAP_DECODE: super::seqstate::slot_history::HistoryRegion =
    super::seqstate::slot_history::HistoryRegion::Decode;

/// Device pointer binding for one PLE convolution history layer (`conv_state_dev`
/// and optional speculative `conv_checkpoint_dev`, strided by `[max_slots, ple_conv_bytes]`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PleConvLayerBinding {
    pub(crate) conv_state: DevicePtr,
    pub(crate) conv_checkpoint: DevicePtr,
    pub(crate) max_slots: usize,
}

/// Pre-allocated GPU memory pool for SSM + PLE state snapshots.
///
/// Each snapshot slot stores a copy of GDN `h_state` + `conv_state` across all
/// SSM layers AND PLE `conv_state` across all attached PLE layers at a specific
/// block-aligned boundary in a token sequence.
///
/// The pool serves **two** independent consumers from one set of GPU
/// allocations (SSOT — one snapshot mechanism, one D2D copy primitive):
///
/// 1. **Marconi prefix caching** — the LRU-managed `[0, num_slots)` slot
///    region, allocated/freed via [`save`](Self::save) / [`free`](Self::free)
///    against the `free_slots` list. When a prefix cache hit occurs the
///    snapshot is restored to skip SSM + PLE recompute for cached tokens.
///
/// 2. **Phase-C decode-time boundary rollback** — a *separate*,
///    deterministically-addressed `[0, decode_ring_slots)` region (per
///    active sequence). No free list: ring slot `r` for SSM-pool
///    sequence slot `s` lives at flat index `s * ring_slots + r`, so a
///    sequence's snapshots never collide with another's and never
///    contend with Marconi's LRU slots. Sized for `max_batch_size`
///    sequences so the watchdog rollback always has capacity.
pub(crate) struct SsmSnapshotPool {
    pub(super) h_snapshots: Vec<DevicePtr>,
    pub(super) conv_snapshots: Vec<DevicePtr>,
    pub(super) free_slots: Mutex<Vec<usize>>,
    pub(super) num_slots: usize,
    pub(super) h_bytes: usize,
    pub(super) conv_bytes: usize,
    pub(super) num_ssm_layers: usize,
    /// FP16 -> FP32 widener for the snapshot save path (`ATLAS_SSM_H_FP16`).
    ///
    /// Snapshots are ALWAYS stored FP32. A decoding slot may hold FP16, and
    /// `restore` only ever lands in a prefill that reads FP32 — so the widening
    /// happens here, on save, and every downstream consumer (restore, spill,
    /// fault-in, tier fingerprint, swap) stays dtype-agnostic.
    pub(super) h_f16_to_f32_k: KernelHandle,
    /// Maps snapshot_slot_id → session_hash for session-scoped isolation.
    /// When restoring, skip snapshots that belong to a different session.
    pub(super) session_tags: Mutex<std::collections::HashMap<usize, u64>>,
    /// Decode-rollback region: `h_snapshots` for the Phase-C ring.
    /// Layout per layer: `[max_batch_size * decode_ring_slots * h_bytes]`.
    /// Empty when `decode_ring_slots == 0`.
    pub(super) decode_h_snapshots: Vec<DevicePtr>,
    /// Decode-rollback region: `conv_snapshots` for the Phase-C ring.
    pub(super) decode_conv_snapshots: Vec<DevicePtr>,
    /// Number of decode-rollback ring slots reserved per active sequence.
    /// 0 disables the decode-rollback region entirely.
    pub(super) decode_ring_slots: usize,
    /// Number of active-sequence slots the decode region is sized for
    /// (equals `max_batch_size`). A sequence's SSM-pool `slot_idx` must
    /// be `< decode_max_seqs` to use the decode region.
    pub(super) decode_max_seqs: usize,
    /// Bound live PLE `conv_state_dev` layers saved/restored alongside GDN state.
    pub(super) ple_conv_layers: Vec<PleConvLayerBinding>,
    /// Marconi LRU region for PLE `conv_state` (`[num_slots * ple_conv_bytes]` per PLE layer).
    pub(super) ple_conv_snapshots: Vec<DevicePtr>,
    /// Phase-C decode-rollback region for PLE `conv_state`
    /// (`[decode_max_seqs * decode_ring_slots * ple_conv_bytes]` per PLE layer).
    pub(super) decode_ple_conv_snapshots: Vec<DevicePtr>,
    /// Byte stride per sequence slot for each PLE layer (`180 KiB` on Qwen4).
    pub(super) ple_conv_bytes: usize,
    /// Per-slot history outside `SsmStatePool` (Qwen4 PLE conv history),
    /// saved and restored with every Marconi and decode-ring snapshot.
    /// Attached only under `ATLAS_QWEN4_SEQSTATE=1`.
    pub(super) slot_history: Option<super::seqstate::slot_history::SlotHistory>,
}

#[derive(Debug, Default, PartialEq, Eq)]
struct SnapshotCapacity {
    marconi_h_bytes: usize,
    marconi_conv_bytes: usize,
    marconi_ple_conv_bytes: usize,
    marconi_total_bytes: usize,
    decode_region_slots: usize,
    decode_h_bytes: usize,
    decode_conv_bytes: usize,
    decode_ple_conv_bytes: usize,
    decode_total_bytes: usize,
}

fn checked_capacity_mul(left: usize, right: usize, label: &str) -> Result<usize> {
    left.checked_mul(right)
        .ok_or_else(|| anyhow!("SSM snapshot capacity overflow: {label} ({left} * {right})"))
}

fn checked_capacity_add(left: usize, right: usize, label: &str) -> Result<usize> {
    left.checked_add(right)
        .ok_or_else(|| anyhow!("SSM snapshot capacity overflow: {label} ({left} + {right})"))
}

fn checked_snapshot_capacity(
    num_slots: usize,
    h_bytes: usize,
    conv_bytes: usize,
    num_ssm_layers: usize,
    decode_ring_slots: usize,
    decode_max_seqs: usize,
) -> Result<SnapshotCapacity> {
    checked_snapshot_capacity_with_ple(
        num_slots,
        h_bytes,
        conv_bytes,
        num_ssm_layers,
        0,
        0,
        decode_ring_slots,
        decode_max_seqs,
    )
}

fn checked_snapshot_capacity_with_ple(
    num_slots: usize,
    h_bytes: usize,
    conv_bytes: usize,
    num_ssm_layers: usize,
    ple_conv_bytes: usize,
    num_ple_layers: usize,
    decode_ring_slots: usize,
    decode_max_seqs: usize,
) -> Result<SnapshotCapacity> {
    if num_ssm_layers == 0 && num_ple_layers == 0 {
        return Ok(SnapshotCapacity::default());
    }
    let marconi_enabled = num_slots > 0;
    let decode_enabled = decode_ring_slots > 0 && decode_max_seqs > 0;
    if !marconi_enabled && !decode_enabled {
        return Ok(SnapshotCapacity::default());
    }
    if num_ssm_layers > 0 {
        if h_bytes == 0 {
            bail!("SSM snapshot capacity: enabled h bytes must be positive");
        }
        if conv_bytes == 0 {
            bail!("SSM snapshot capacity: enabled conv bytes must be positive");
        }
    }
    if num_ple_layers > 0 && ple_conv_bytes == 0 {
        bail!("SSM snapshot capacity: enabled ple conv bytes must be positive");
    }

    let (marconi_h_bytes, marconi_conv_bytes, marconi_ssm_total) =
        if marconi_enabled && num_ssm_layers > 0 {
            let h = checked_capacity_mul(num_slots, h_bytes, "Marconi h bytes")?;
            let conv = checked_capacity_mul(num_slots, conv_bytes, "Marconi conv bytes")?;
            let per_layer = checked_capacity_add(h, conv, "Marconi per-layer bytes")?;
            let total = checked_capacity_mul(per_layer, num_ssm_layers, "Marconi total bytes")?;
            (h, conv, total)
        } else {
            (0, 0, 0)
        };

    let (marconi_ple_conv_bytes, marconi_ple_total) = if marconi_enabled && num_ple_layers > 0 {
        let ple = checked_capacity_mul(num_slots, ple_conv_bytes, "Marconi PLE conv bytes")?;
        let total = checked_capacity_mul(ple, num_ple_layers, "Marconi PLE total bytes")?;
        (ple, total)
    } else {
        (0, 0)
    };
    let marconi_total_bytes =
        checked_capacity_add(marconi_ssm_total, marconi_ple_total, "Marconi total bytes")?;

    let decode_region_slots = if decode_enabled {
        checked_capacity_mul(decode_max_seqs, decode_ring_slots, "decode region slots")?
    } else {
        0
    };

    let (decode_h_bytes, decode_conv_bytes, decode_ssm_total) =
        if decode_enabled && num_ssm_layers > 0 {
            let h = checked_capacity_mul(decode_region_slots, h_bytes, "decode h bytes")?;
            let conv = checked_capacity_mul(decode_region_slots, conv_bytes, "decode conv bytes")?;
            let per_layer = checked_capacity_add(h, conv, "decode per-layer bytes")?;
            let total = checked_capacity_mul(per_layer, num_ssm_layers, "decode total bytes")?;
            (h, conv, total)
        } else {
            (0, 0, 0)
        };

    let (decode_ple_conv_bytes, decode_ple_total) = if decode_enabled && num_ple_layers > 0 {
        let ple =
            checked_capacity_mul(decode_region_slots, ple_conv_bytes, "decode PLE conv bytes")?;
        let total = checked_capacity_mul(ple, num_ple_layers, "decode PLE total bytes")?;
        (ple, total)
    } else {
        (0, 0)
    };
    let decode_total_bytes =
        checked_capacity_add(decode_ssm_total, decode_ple_total, "decode total bytes")?;

    Ok(SnapshotCapacity {
        marconi_h_bytes,
        marconi_conv_bytes,
        marconi_ple_conv_bytes,
        marconi_total_bytes,
        decode_region_slots,
        decode_h_bytes,
        decode_conv_bytes,
        decode_ple_conv_bytes,
        decode_total_bytes,
    })
}

impl SsmSnapshotPool {
    /// Build the snapshot pool.
    ///
    /// `num_slots` sizes the Marconi LRU region; `decode_ring_slots` ×
    /// `decode_max_seqs` sizes the Phase-C decode-rollback region. A
    /// pool with `num_slots == 0` but `decode_ring_slots > 0` is valid
    /// (decode rollback enabled, Marconi caching disabled) and vice
    /// versa — the two regions are independent.
    pub(super) fn new(
        num_slots: usize,
        h_bytes: usize,
        conv_bytes: usize,
        num_ssm_layers: usize,
        decode_ring_slots: usize,
        decode_max_seqs: usize,
        gpu: &dyn GpuBackend,
    ) -> Result<Self> {
        let decode_ring_slots =
            if super::ssm_pool_geometry::ssm_decode_ring_enabled_in_scope(decode_max_seqs)? {
                decode_ring_slots
            } else {
                0
            };
        let decode_enabled = num_ssm_layers > 0 && decode_ring_slots > 0 && decode_max_seqs > 0;
        let marconi_enabled = num_ssm_layers > 0 && num_slots > 0;

        if !marconi_enabled && !decode_enabled {
            return Ok(Self {
                h_snapshots: Vec::new(),
                conv_snapshots: Vec::new(),
                free_slots: Mutex::new(Vec::new()),
                num_slots: 0,
                h_bytes,
                conv_bytes,
                num_ssm_layers,
                h_f16_to_f32_k: KernelHandle(0),
                session_tags: Mutex::new(std::collections::HashMap::new()),
                decode_h_snapshots: Vec::new(),
                decode_conv_snapshots: Vec::new(),
                decode_ring_slots: 0,
                decode_max_seqs: 0,
                ple_conv_layers: Vec::new(),
                ple_conv_snapshots: Vec::new(),
                decode_ple_conv_snapshots: Vec::new(),
                ple_conv_bytes: 0,
                slot_history: None,
            });
        }

        // Validate the complete geometry before the first allocation. An
        // overflow must never wrap into a smaller apparently valid buffer.
        let capacity = checked_snapshot_capacity(
            num_slots,
            h_bytes,
            conv_bytes,
            num_ssm_layers,
            decode_ring_slots,
            decode_max_seqs,
        )?;

        let mut owned = crate::device_allocations::DeviceAllocations::new(gpu);
        let mut h_snapshots = Vec::new();
        let mut conv_snapshots = Vec::new();
        if marconi_enabled {
            for _ in 0..num_ssm_layers {
                h_snapshots.push(owned.alloc(capacity.marconi_h_bytes)?);
                conv_snapshots.push(owned.alloc(capacity.marconi_conv_bytes)?);
            }
        }

        let mut decode_h_snapshots = Vec::new();
        let mut decode_conv_snapshots = Vec::new();
        if decode_enabled {
            for _ in 0..num_ssm_layers {
                decode_h_snapshots.push(owned.alloc(capacity.decode_h_bytes)?);
                decode_conv_snapshots.push(owned.alloc(capacity.decode_conv_bytes)?);
            }
        }
        owned.commit();

        let free_slots: Vec<usize> = if marconi_enabled {
            (0..num_slots).rev().collect()
        } else {
            Vec::new()
        };
        let marconi_mb = capacity.marconi_total_bytes / (1024 * 1024);
        let decode_mb = capacity.decode_total_bytes / (1024 * 1024);
        tracing::info!(
            "SSM snapshot pool: Marconi {num_slots} slots ({marconi_mb} MB), \
             decode-rollback {decode_ring_slots} slots × {decode_max_seqs} seqs \
             ({decode_mb} MB), {num_ssm_layers} layers",
        );

        Ok(Self {
            h_snapshots,
            conv_snapshots,
            free_slots: Mutex::new(free_slots),
            num_slots: if marconi_enabled { num_slots } else { 0 },
            h_bytes,
            conv_bytes,
            num_ssm_layers,
            h_f16_to_f32_k: crate::layers::try_kernel(gpu, "ssm_h_dtype", "ssm_h_state_f16_to_f32"),
            session_tags: Mutex::new(std::collections::HashMap::new()),
            decode_h_snapshots,
            decode_conv_snapshots,
            decode_ring_slots: if decode_enabled { decode_ring_slots } else { 0 },
            decode_max_seqs: if decode_enabled { decode_max_seqs } else { 0 },
            ple_conv_layers: Vec::new(),
            ple_conv_snapshots: Vec::new(),
            decode_ple_conv_snapshots: Vec::new(),
            ple_conv_bytes: 0,
            slot_history: None,
        })
    }

    /// Attach Qwen4 PLE `conv_state_dev` layers to this snapshot pool,
    /// allocating Marconi (`ple_conv_snapshots`) and Phase-C rollback
    /// (`decode_ple_conv_snapshots`) slabs via [`checked_snapshot_capacity_with_ple`].
    pub(super) fn attach_ple_conv_layers(
        &mut self,
        ple_layers: Vec<PleConvLayerBinding>,
        ple_conv_bytes: usize,
        decode_ring_slots: usize,
        decode_max_seqs: usize,
        marconi_slots: usize,
        gpu: &dyn GpuBackend,
    ) -> Result<()> {
        if ple_layers.is_empty() {
            return Ok(());
        }
        ensure!(
            self.ple_conv_layers.is_empty(),
            "PLE conv snapshot layers already attached"
        );
        let decode_ring_slots =
            if super::ssm_pool_geometry::ssm_decode_ring_enabled_in_scope(decode_max_seqs)? {
                decode_ring_slots
            } else {
                0
            };
        let marconi_enabled = marconi_slots > 0;
        let decode_enabled = decode_ring_slots > 0 && decode_max_seqs > 0;
        if !marconi_enabled && !decode_enabled {
            return Ok(());
        }

        let capacity = checked_snapshot_capacity_with_ple(
            marconi_slots,
            self.h_bytes,
            self.conv_bytes,
            self.num_ssm_layers,
            ple_conv_bytes,
            ple_layers.len(),
            decode_ring_slots,
            decode_max_seqs,
        )?;

        for binding in &ple_layers {
            ensure!(
                binding.conv_state.0 != 0,
                "PLE conv snapshot binding has null conv_state pointer"
            );
            if decode_enabled {
                ensure!(
                    decode_max_seqs <= binding.max_slots,
                    "PLE conv snapshot decode_max_seqs ({decode_max_seqs}) exceeds PLE layer capacity ({})",
                    binding.max_slots
                );
            }
        }

        if self.num_slots > 0 {
            ensure!(
                marconi_slots == self.num_slots,
                "PLE Marconi slots ({marconi_slots}) must match SSM pool num_slots ({})",
                self.num_slots
            );
        } else if marconi_enabled {
            ensure!(
                self.num_ssm_layers == 0 || !self.h_snapshots.is_empty(),
                "PLE Marconi slots ({marconi_slots}) cannot enable Marconi on a pool with {} SSM layers and unallocated SSM Marconi slabs",
                self.num_ssm_layers
            );
        }
        if self.decode_ring_slots > 0 {
            ensure!(
                decode_ring_slots == self.decode_ring_slots
                    && decode_max_seqs == self.decode_max_seqs,
                "PLE decode ring ({decode_ring_slots}x{decode_max_seqs}) must match SSM pool ({}x{})",
                self.decode_ring_slots,
                self.decode_max_seqs
            );
        } else if decode_enabled {
            ensure!(
                self.num_ssm_layers == 0 || !self.decode_h_snapshots.is_empty(),
                "PLE decode ring ({decode_ring_slots}x{decode_max_seqs}) cannot enable decode rollback on a pool with {} SSM layers and unallocated SSM decode slabs",
                self.num_ssm_layers
            );
        }

        let mut owned = crate::device_allocations::DeviceAllocations::new(gpu);
        let mut ple_conv_snapshots = Vec::new();
        if marconi_enabled {
            ple_conv_snapshots.reserve(ple_layers.len());
            for _ in 0..ple_layers.len() {
                ple_conv_snapshots.push(owned.alloc(capacity.marconi_ple_conv_bytes)?);
            }
        }

        let mut decode_ple_conv_snapshots = Vec::new();
        if decode_enabled {
            decode_ple_conv_snapshots.reserve(ple_layers.len());
            for _ in 0..ple_layers.len() {
                decode_ple_conv_snapshots.push(owned.alloc(capacity.decode_ple_conv_bytes)?);
            }
        }
        owned.commit();

        if marconi_enabled && self.num_slots == 0 {
            self.num_slots = marconi_slots;
            *self.free_slots.lock() = (0..marconi_slots).rev().collect();
        }
        if decode_enabled && self.decode_ring_slots == 0 {
            self.decode_ring_slots = decode_ring_slots;
            self.decode_max_seqs = decode_max_seqs;
        }

        let marconi_ple_kib = (capacity.marconi_ple_conv_bytes * ple_layers.len()) / 1024;
        let decode_ple_kib = (capacity.decode_ple_conv_bytes * ple_layers.len()) / 1024;
        tracing::info!(
            "PLE conv snapshot pool attached: Marconi {} slots ({marconi_ple_kib} KiB), \
             decode-rollback {} slots × {} seqs ({decode_ple_kib} KiB), {} PLE layer(s) ({} KiB/layer/slot)",
            self.num_slots,
            self.decode_ring_slots,
            self.decode_max_seqs,
            ple_layers.len(),
            ple_conv_bytes / 1024,
        );

        self.ple_conv_layers = ple_layers;
        self.ple_conv_snapshots = ple_conv_snapshots;
        self.decode_ple_conv_snapshots = decode_ple_conv_snapshots;
        self.ple_conv_bytes = ple_conv_bytes;
        Ok(())
    }

    /// Attach out-of-pool per-slot history (`SlotHistory`) sized for this
    /// pool's Marconi slots and decode ring.
    pub(super) fn attach_slot_history(
        &mut self,
        gpu: &dyn GpuBackend,
        live: DevicePtr,
        checkpoint: DevicePtr,
        stride: usize,
        slots: usize,
    ) -> Result<()> {
        let history = super::seqstate::slot_history::SlotHistory::new(
            gpu,
            live,
            checkpoint,
            stride,
            slots,
            self.num_slots,
            self.decode_max_seqs * self.decode_ring_slots,
        )?;
        tracing::info!(
            "QWEN4_SEQSTATE slot history attached: {} Marconi + {} decode-ring snapshots x {stride} B = {} B",
            self.num_slots,
            self.decode_max_seqs * self.decode_ring_slots,
            history.bytes()
        );
        self.slot_history = Some(history);
        Ok(())
    }

    /// Marconi prefix-cache region availability.
    pub(super) fn is_enabled(&self) -> bool {
        self.num_slots > 0
    }

    /// Phase-C decode-rollback region availability.
    pub(super) fn decode_rollback_enabled(&self) -> bool {
        let ssm_ring = self.decode_ring_slots > 0 && !self.decode_h_snapshots.is_empty();
        ssm_ring
            || (self.decode_ring_slots > 0
                && (!self.decode_ple_conv_snapshots.is_empty() || self.slot_history.is_some()))
    }

    /// Save the SSM + PLE state of pool slot `ssm_slot` into the decode-rollback
    /// ring slot `(ssm_slot, ring_slot)`. Deterministic addressing — no
    /// free list, no eviction. Errors if the decode region is disabled
    /// or the indices are out of the reserved range (fail fast — a
    /// silent skip would leave the watchdog rollback unable to undo SSM
    /// state, corrupting every subsequent decode).
    pub(super) fn save_decode(
        &self,
        ssm_slot: usize,
        ring_slot: usize,
        main_pool: &SsmStatePool,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        let flat = self.decode_flat_index(ssm_slot, ring_slot)?;
        let skip_h = crate::layers::qwen3_ssm::replay_ssm_enabled();
        for i in 0..self.num_ssm_layers {
            if !skip_h {
                gpu.copy_d2d_async(
                    main_pool.h_state(i, ssm_slot),
                    self.decode_h_snapshots[i].offset(flat * self.h_bytes),
                    self.h_bytes,
                    stream,
                )?;
            }
            gpu.copy_d2d_async(
                main_pool.conv_state(i, ssm_slot),
                self.decode_conv_snapshots[i].offset(flat * self.conv_bytes),
                self.conv_bytes,
                stream,
            )?;
        }
        for (ple_idx, binding) in self.ple_conv_layers.iter().enumerate() {
            ensure!(
                ssm_slot < binding.max_slots,
                "PLE decode snapshot save: slot {ssm_slot} >= PLE max_slots {}",
                binding.max_slots
            );
            gpu.copy_d2d_async(
                binding.conv_state.offset(ssm_slot * self.ple_conv_bytes),
                self.decode_ple_conv_snapshots[ple_idx].offset(flat * self.ple_conv_bytes),
                self.ple_conv_bytes,
                stream,
            )?;
        }
        if let Some(history) = &self.slot_history {
            history.save(gpu, ssm_slot, SNAP_DECODE, flat, stream)?;
        }
        Ok(())
    }

    /// Restore the SSM + PLE state of pool slot `ssm_slot` from the
    /// decode-rollback ring slot `(ssm_slot, ring_slot)`.
    pub(super) fn restore_decode(
        &self,
        ssm_slot: usize,
        ring_slot: usize,
        main_pool: &SsmStatePool,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        let flat = self.decode_flat_index(ssm_slot, ring_slot)?;
        let skip_h = crate::layers::qwen3_ssm::replay_ssm_enabled();
        for i in 0..self.num_ssm_layers {
            if !skip_h {
                gpu.copy_d2d_async(
                    self.decode_h_snapshots[i].offset(flat * self.h_bytes),
                    main_pool.h_state(i, ssm_slot),
                    self.h_bytes,
                    stream,
                )?;
            }
            gpu.copy_d2d_async(
                self.decode_conv_snapshots[i].offset(flat * self.conv_bytes),
                main_pool.conv_state(i, ssm_slot),
                self.conv_bytes,
                stream,
            )?;
        }
        for (ple_idx, binding) in self.ple_conv_layers.iter().enumerate() {
            ensure!(
                ssm_slot < binding.max_slots,
                "PLE decode snapshot restore: slot {ssm_slot} >= PLE max_slots {}",
                binding.max_slots
            );
            let live_dst = binding.conv_state.offset(ssm_slot * self.ple_conv_bytes);
            gpu.copy_d2d_async(
                self.decode_ple_conv_snapshots[ple_idx].offset(flat * self.ple_conv_bytes),
                live_dst,
                self.ple_conv_bytes,
                stream,
            )?;
            if binding.conv_checkpoint.0 != 0 {
                gpu.copy_d2d_async(
                    live_dst,
                    binding
                        .conv_checkpoint
                        .offset(ssm_slot * self.ple_conv_bytes),
                    self.ple_conv_bytes,
                    stream,
                )?;
            }
        }
        if let Some(history) = &self.slot_history {
            history.restore(gpu, SNAP_DECODE, flat, ssm_slot, stream)?;
        }
        Ok(())
    }

    /// Flat index into the decode-rollback region, with bounds checks.
    fn decode_flat_index(&self, ssm_slot: usize, ring_slot: usize) -> Result<usize> {
        if !self.decode_rollback_enabled() {
            bail!("SSM decode-rollback region not allocated");
        }
        if ssm_slot >= self.decode_max_seqs {
            bail!(
                "SSM decode-rollback: ssm_slot {ssm_slot} >= reserved {} seqs",
                self.decode_max_seqs
            );
        }
        if ring_slot >= self.decode_ring_slots {
            bail!(
                "SSM decode-rollback: ring_slot {ring_slot} >= reserved {} slots",
                self.decode_ring_slots
            );
        }
        Ok(ssm_slot * self.decode_ring_slots + ring_slot)
    }

    /// Save SSM + PLE state from active pool slot into a snapshot slot.
    /// Returns `None` if no free snapshot slots are available.
    /// Tags the snapshot with `session_hash` for session-scoped isolation.
    /// `h_is_f16` is the storage dtype of the SOURCE slot, not of the snapshot.
    /// Snapshots are always written FP32 (see `h_f16_to_f32_k`).
    pub(super) fn save(
        &self,
        ssm_slot: usize,
        session_hash: u64,
        h_is_f16: bool,
        main_pool: &SsmStatePool,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<Option<usize>> {
        if !self.is_enabled() {
            return Ok(None);
        }
        if h_is_f16 && self.h_f16_to_f32_k.0 == 0 {
            anyhow::bail!(
                "ATLAS_SSM_H_FP16: cannot widen a decode-produced snapshot — \
                 ssm_h_dtype::ssm_h_state_f16_to_f32 did not resolve on this target"
            );
        }
        let snap_slot = match self.free_slots.lock().pop() {
            Some(s) => s,
            None => return Ok(None),
        };
        let copy_result: Result<()> = (|| {
            for i in 0..self.num_ssm_layers {
                let dst = self.h_snapshots[i].offset(snap_slot * self.h_bytes);
                if h_is_f16 {
                    // Widen on the way in so the snapshot is FP32 like every
                    // other one. `n` is the FP32 ELEMENT count of the
                    // destination, derived from the pool's byte size rather
                    // than a duplicated shape literal.
                    crate::layers::ops::ssm_h_state_f16_to_f32(
                        gpu,
                        self.h_f16_to_f32_k,
                        main_pool.h_state(i, ssm_slot),
                        dst,
                        (self.h_bytes / 4) as u64,
                        stream,
                    )?;
                } else {
                    gpu.copy_d2d_async(main_pool.h_state(i, ssm_slot), dst, self.h_bytes, stream)?;
                }
                gpu.copy_d2d_async(
                    main_pool.conv_state(i, ssm_slot),
                    self.conv_snapshots[i].offset(snap_slot * self.conv_bytes),
                    self.conv_bytes,
                    stream,
                )?;
            }
            for (ple_idx, binding) in self.ple_conv_layers.iter().enumerate() {
                ensure!(
                    ssm_slot < binding.max_slots,
                    "PLE snapshot save: slot {ssm_slot} >= PLE max_slots {}",
                    binding.max_slots
                );
                gpu.copy_d2d_async(
                    binding.conv_state.offset(ssm_slot * self.ple_conv_bytes),
                    self.ple_conv_snapshots[ple_idx].offset(snap_slot * self.ple_conv_bytes),
                    self.ple_conv_bytes,
                    stream,
                )?;
            }
            if let Some(history) = &self.slot_history {
                history.save(gpu, ssm_slot, SNAP_MARCONI, snap_slot, stream)?;
            }
            Ok(())
        })();
        if let Err(error) = copy_result {
            self.free(snap_slot);
            return Err(error);
        }
        if session_hash != 0 {
            self.session_tags.lock().insert(snap_slot, session_hash);
        }
        Ok(Some(snap_slot))
    }

    /// Check if a snapshot belongs to the given session.
    /// Returns true if: session tracking is disabled (hash=0), no tag exists, or tags match.
    pub(super) fn session_matches(&self, snap_slot: usize, session_hash: u64) -> bool {
        if session_hash == 0 {
            return true;
        } // Legacy: no session tracking
        let tags = self.session_tags.lock();
        match tags.get(&snap_slot) {
            None => true, // Untagged snapshot (pre-session-manager) — allow
            Some(&tag) => tag == session_hash,
        }
    }

    /// Restore SSM + PLE state from a snapshot slot into an active pool slot.
    pub(super) fn restore(
        &self,
        snap_slot: usize,
        ssm_slot: usize,
        main_pool: &SsmStatePool,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        ensure!(
            snap_slot < self.num_slots,
            "SSM snapshot restore: snap_slot {snap_slot} >= num_slots {}",
            self.num_slots
        );
        for i in 0..self.num_ssm_layers {
            gpu.copy_d2d_async(
                self.h_snapshots[i].offset(snap_slot * self.h_bytes),
                main_pool.h_state(i, ssm_slot),
                self.h_bytes,
                stream,
            )?;
            gpu.copy_d2d_async(
                self.conv_snapshots[i].offset(snap_slot * self.conv_bytes),
                main_pool.conv_state(i, ssm_slot),
                self.conv_bytes,
                stream,
            )?;
        }
        for (ple_idx, binding) in self.ple_conv_layers.iter().enumerate() {
            ensure!(
                ssm_slot < binding.max_slots,
                "PLE snapshot restore: slot {ssm_slot} >= PLE max_slots {}",
                binding.max_slots
            );
            let live_dst = binding.conv_state.offset(ssm_slot * self.ple_conv_bytes);
            gpu.copy_d2d_async(
                self.ple_conv_snapshots[ple_idx].offset(snap_slot * self.ple_conv_bytes),
                live_dst,
                self.ple_conv_bytes,
                stream,
            )?;
            if binding.conv_checkpoint.0 != 0 {
                gpu.copy_d2d_async(
                    live_dst,
                    binding
                        .conv_checkpoint
                        .offset(ssm_slot * self.ple_conv_bytes),
                    self.ple_conv_bytes,
                    stream,
                )?;
            }
        }
        if let Some(history) = &self.slot_history {
            history.restore(gpu, SNAP_MARCONI, snap_slot, ssm_slot, stream)?;
        }
        Ok(())
    }

    /// Return a snapshot slot to the free list.
    pub(super) fn free(&self, snap_slot: usize) {
        // A free slot has no owner. Leaving the old tag behind makes a later
        // untagged/legacy save look as if it belongs to the previous session.
        self.session_tags.lock().remove(&snap_slot);
        self.free_slots.lock().push(snap_slot);
    }

    /// Try to reclaim a snapshot slot by evicting the LRU snapshot from the
    /// prefix cache's snapshot index. Snapshots are decoupled from tree nodes,
    /// so this directly frees a snapshot without needing to evict KV blocks.
    pub(super) fn reclaim_from_cache(
        &self,
        prefix_cache: &dyn spark_runtime::prefix_cache::PrefixCache,
        _kv_cache: &mut PagedKvCache,
    ) -> bool {
        if let Some(snap) = prefix_cache.evict_snapshot_lru() {
            self.free(snap);
            true
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn freeing_snapshot_clears_session_owner() {
        let pool = SsmSnapshotPool {
            h_snapshots: Vec::new(),
            conv_snapshots: Vec::new(),
            free_slots: Mutex::new(Vec::new()),
            num_slots: 1,
            h_bytes: 0,
            conv_bytes: 0,
            num_ssm_layers: 0,
            h_f16_to_f32_k: KernelHandle(0),
            session_tags: Mutex::new(HashMap::from([(0, 0xAABB)])),
            decode_h_snapshots: Vec::new(),
            decode_conv_snapshots: Vec::new(),
            decode_ring_slots: 0,
            decode_max_seqs: 0,
            ple_conv_layers: Vec::new(),
            ple_conv_snapshots: Vec::new(),
            decode_ple_conv_snapshots: Vec::new(),
            ple_conv_bytes: 0,
            slot_history: None,
        };

        pool.free(0);

        assert!(pool.session_tags.lock().is_empty());
        assert_eq!(*pool.free_slots.lock(), vec![0]);
    }
}

#[cfg(test)]
#[path = "ssm_snapshot_failure_tests.rs"]
mod failure_tests;

#[cfg(test)]
#[path = "ssm_snapshot_capacity_tests.rs"]
mod capacity_tests;
