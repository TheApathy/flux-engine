// SPDX-License-Identifier: AGPL-3.0-only

//! Batched verify-commit pointer cache and runtime flag snapshot.
//!
//! Eliminates per-layer host `cuMemcpyDtoDAsync` loops and repeated environment
//! lookups on the speculative verify commit hot path by caching each physical
//! slot's per-SSM-layer device pointers (`SsmLayerCommitPointers[num_ssm_layers]`,
//! `2,304 B/slot` for 36 SSM layers) keyed by `(slot_idx, generation)` and
//! dispatching all `(num_ssm_layers × batch_size)` copies through a single
//! `gdn_batch_commit_copy_f32` kernel launch.

use anyhow::{Context, Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};

/// Mirror committed H state into `h_ckpt` when `h_ckpt != 0`.
pub const BATCH_COMMIT_FLAG_MIRROR_CKPT: u32 = 1 << 0;
/// Read conv state from `conv_inter_base[accepted_idx]` and write both
/// `conv_state` and `conv_ckpt`; when cleared, mirror live `conv_state` to
/// `conv_ckpt` only.
pub const BATCH_COMMIT_FLAG_HAS_CONV_INTER: u32 = 1 << 1;
/// Live `h_state` already holds the post-K recurrent state (full accept on a
/// non-lazy verifier); only mirror `h_state -> h_ckpt` when
/// [`BATCH_COMMIT_FLAG_MIRROR_CKPT`] is set.
pub const BATCH_COMMIT_FLAG_H_ALREADY_LIVE: u32 = 1 << 2;

/// Per-SSM-layer device pointers for one physical sequence slot (`64 B`, 16 B aligned).
///
/// ABI layout matches `struct alignas(16) SsmLayerCommitPointers` in
/// `kernels/gb10/common/gdn_batch_commit.cu`.
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SsmLayerCommitPointers {
    pub h_state: u64,
    pub h_ckpt: u64,
    pub conv_state: u64,
    pub conv_ckpt: u64,
    pub h_inter_base: u64,
    pub conv_inter_base: u64,
    pub retain_slot_base: u64,
    pub _pad: u64,
}

impl SsmLayerCommitPointers {
    /// Exact serialized byte width (`64 B`).
    pub const SIZE_BYTES: usize = 64;

    /// Serialize this descriptor into exact little-endian ABI bytes.
    #[must_use]
    pub const fn to_le_bytes(self) -> [u8; Self::SIZE_BYTES] {
        let h_state = self.h_state.to_le_bytes();
        let h_ckpt = self.h_ckpt.to_le_bytes();
        let conv_state = self.conv_state.to_le_bytes();
        let conv_ckpt = self.conv_ckpt.to_le_bytes();
        let h_inter_base = self.h_inter_base.to_le_bytes();
        let conv_inter_base = self.conv_inter_base.to_le_bytes();
        let retain_slot_base = self.retain_slot_base.to_le_bytes();
        let pad = self._pad.to_le_bytes();
        [
            h_state[0],
            h_state[1],
            h_state[2],
            h_state[3],
            h_state[4],
            h_state[5],
            h_state[6],
            h_state[7],
            h_ckpt[0],
            h_ckpt[1],
            h_ckpt[2],
            h_ckpt[3],
            h_ckpt[4],
            h_ckpt[5],
            h_ckpt[6],
            h_ckpt[7],
            conv_state[0],
            conv_state[1],
            conv_state[2],
            conv_state[3],
            conv_state[4],
            conv_state[5],
            conv_state[6],
            conv_state[7],
            conv_ckpt[0],
            conv_ckpt[1],
            conv_ckpt[2],
            conv_ckpt[3],
            conv_ckpt[4],
            conv_ckpt[5],
            conv_ckpt[6],
            conv_ckpt[7],
            h_inter_base[0],
            h_inter_base[1],
            h_inter_base[2],
            h_inter_base[3],
            h_inter_base[4],
            h_inter_base[5],
            h_inter_base[6],
            h_inter_base[7],
            conv_inter_base[0],
            conv_inter_base[1],
            conv_inter_base[2],
            conv_inter_base[3],
            conv_inter_base[4],
            conv_inter_base[5],
            conv_inter_base[6],
            conv_inter_base[7],
            retain_slot_base[0],
            retain_slot_base[1],
            retain_slot_base[2],
            retain_slot_base[3],
            retain_slot_base[4],
            retain_slot_base[5],
            retain_slot_base[6],
            retain_slot_base[7],
            pad[0],
            pad[1],
            pad[2],
            pad[3],
            pad[4],
            pad[5],
            pad[6],
            pad[7],
        ]
    }

    /// Write this descriptor's little-endian ABI bytes into `dst`.
    ///
    /// # Errors
    ///
    /// Returns an error if `dst.len() != Self::SIZE_BYTES`.
    pub fn write_le_bytes(&self, dst: &mut [u8]) -> Result<()> {
        ensure!(
            dst.len() == Self::SIZE_BYTES,
            "SsmLayerCommitPointers destination slice must be {} bytes, got {}",
            Self::SIZE_BYTES,
            dst.len()
        );
        dst.copy_from_slice(&self.to_le_bytes());
        Ok(())
    }
}

/// Per-sequence batched verify commit descriptor (`16 B`, 16 B aligned).
///
/// ABI layout matches `struct alignas(16) BatchCommitEntry` in
/// `kernels/gb10/common/gdn_batch_commit.cu`.
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BatchCommitEntry {
    pub ssm_table_dev: u64,
    pub accepted_idx: u32,
    pub flags: u32,
}

impl BatchCommitEntry {
    /// Exact serialized byte width (`16 B`).
    pub const SIZE_BYTES: usize = 16;

    /// Serialize this entry into exact little-endian ABI bytes.
    #[must_use]
    pub const fn to_le_bytes(self) -> [u8; Self::SIZE_BYTES] {
        let table = self.ssm_table_dev.to_le_bytes();
        let idx = self.accepted_idx.to_le_bytes();
        let flags = self.flags.to_le_bytes();
        [
            table[0], table[1], table[2], table[3], table[4], table[5], table[6], table[7], idx[0],
            idx[1], idx[2], idx[3], flags[0], flags[1], flags[2], flags[3],
        ]
    }

    /// Write this entry's little-endian ABI bytes into `dst`.
    ///
    /// # Errors
    ///
    /// Returns an error if `dst.len() != Self::SIZE_BYTES`.
    pub fn write_le_bytes(&self, dst: &mut [u8]) -> Result<()> {
        ensure!(
            dst.len() == Self::SIZE_BYTES,
            "BatchCommitEntry destination slice must be {} bytes, got {}",
            Self::SIZE_BYTES,
            dst.len()
        );
        dst.copy_from_slice(&self.to_le_bytes());
        Ok(())
    }
}

const _: () = {
    assert!(std::mem::size_of::<SsmLayerCommitPointers>() == SsmLayerCommitPointers::SIZE_BYTES);
    assert!(std::mem::align_of::<SsmLayerCommitPointers>() == 16);
    assert!(std::mem::size_of::<BatchCommitEntry>() == BatchCommitEntry::SIZE_BYTES);
    assert!(std::mem::align_of::<BatchCommitEntry>() == 16);
    const fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<SsmLayerCommitPointers>();
    assert_send_sync::<BatchCommitEntry>();
    assert_send_sync::<CommitRuntimeFlags>();
    assert_send_sync::<SlotCommitTable>();
};

/// Commit-path runtime environment flags resolved once at model construction.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CommitRuntimeFlags {
    pub lazy_j: u32,
    pub lazy_commit_env: bool,
    pub async_probe: bool,
    pub bootstrap_trace: bool,
    pub skip_restore_control: bool,
    pub qwen4_no_mirror_req: bool,
    pub k5_cmulti_enforced: bool,
}

impl CommitRuntimeFlags {
    /// Snapshot every commit-path environment selector once at model init.
    #[must_use]
    pub fn from_env(is_qwen4_exp: bool) -> Self {
        let skip_restore_control = std::env::var("ATLAS_SPEC_SKIP_SSM_RESTORE_CONTROL")
            .ok()
            .as_deref()
            == Some("1");
        if skip_restore_control {
            tracing::warn!(
                "ATLAS_SPEC_SKIP_SSM_RESTORE_CONTROL=1: recurrent restore DISABLED (negative control)"
            );
        }
        let qwen4_no_mirror_env = Self::parse_truthy_flag(
            std::env::var("ATLAS_QWEN4_GDN_COMMIT_NO_MIRROR")
                .ok()
                .as_deref(),
        );
        let qwen4_no_mirror_req =
            is_qwen4_exp && (crate::layers::qwen3_ssm::replay_ssm_enabled() || qwen4_no_mirror_env);
        let k5_cmulti_enforced =
            is_qwen4_exp && crate::model::qwen4_k5_cmulti::k5_cmulti_enforced();
        Self {
            lazy_j: crate::layers::wy17_lazy(),
            lazy_commit_env: crate::layers::wy17_lazy_commit(),
            async_probe: crate::layers::dflash_async_probe(),
            bootstrap_trace: std::env::var("ATLAS_SPEC_BOOTSTRAP_TRACE").ok().as_deref()
                == Some("1"),
            skip_restore_control,
            qwen4_no_mirror_req,
            k5_cmulti_enforced,
        }
    }

    #[must_use]
    pub(crate) fn parse_truthy_flag(raw: Option<&str>) -> bool {
        matches!(raw, Some("1" | "true" | "TRUE" | "on" | "ON"))
    }
}

/// Validated, cached per-slot commit table metadata.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CachedSlotTable {
    pub ssm_ptrs_dev: DevicePtr,
    pub generation: u64,
    pub all_have_h_ckpt: bool,
    pub all_have_conv_ckpt: bool,
}

#[derive(Clone, Copy, Debug)]
struct SlotTableRecord {
    generation: u64,
    cached_generation: Option<u64>,
    ssm_ptrs_dev: DevicePtr,
    all_have_h_ckpt: bool,
    all_have_conv_ckpt: bool,
}

impl Default for SlotTableRecord {
    fn default() -> Self {
        Self {
            generation: 0,
            cached_generation: None,
            ssm_ptrs_dev: DevicePtr::NULL,
            all_have_h_ckpt: false,
            all_have_conv_ckpt: false,
        }
    }
}

/// Per-slot lazy device cache of `SsmLayerCommitPointers[num_ssm_layers]` plus
/// the shared `BatchCommitEntry` upload buffer.
///
/// Keyed by `(slot_idx, generation)`: freeing or compacting a slot bumps its
/// generation and invalidates the cached table while retaining the fixed-size
/// device buffer (`2,304 B` for 36 SSM layers) for zero-allocation rebuild on
/// reuse.
#[derive(Debug)]
pub struct SlotCommitTable {
    num_ssm_layers: usize,
    slots: Vec<SlotTableRecord>,
    entries_dev: DevicePtr,
    entries_capacity: usize,
}

impl SlotCommitTable {
    /// Create an empty lazy commit table sized for `max_slots` physical slots
    /// and `num_ssm_layers` recurrent layers.
    #[must_use]
    pub fn new(max_slots: usize, num_ssm_layers: usize) -> Self {
        Self {
            num_ssm_layers,
            slots: vec![SlotTableRecord::default(); max_slots],
            entries_dev: DevicePtr::NULL,
            entries_capacity: 0,
        }
    }

    /// Number of SSM layers represented in each slot table.
    #[must_use]
    pub const fn num_ssm_layers(&self) -> usize {
        self.num_ssm_layers
    }

    /// Byte size of one slot's device pointer table (`num_ssm_layers * 64`).
    ///
    /// # Errors
    ///
    /// Returns an error on arithmetic overflow.
    pub fn slot_table_bytes(&self) -> Result<usize> {
        self.num_ssm_layers
            .checked_mul(SsmLayerCommitPointers::SIZE_BYTES)
            .context("slot commit table byte size overflow")
    }

    /// Current generation counter for `slot_idx`.
    #[must_use]
    pub fn slot_generation(&self, slot_idx: usize) -> u64 {
        self.slots.get(slot_idx).map_or(0, |rec| rec.generation)
    }

    /// Return the cached slot table when valid for the slot's current generation.
    #[must_use]
    pub fn cached_slot(&self, slot_idx: usize) -> Option<CachedSlotTable> {
        let rec = self.slots.get(slot_idx)?;
        if rec.cached_generation == Some(rec.generation) && !rec.ssm_ptrs_dev.is_null() {
            Some(CachedSlotTable {
                ssm_ptrs_dev: rec.ssm_ptrs_dev,
                generation: rec.generation,
                all_have_h_ckpt: rec.all_have_h_ckpt,
                all_have_conv_ckpt: rec.all_have_conv_ckpt,
            })
        } else {
            None
        }
    }

    /// Invalidate `slot_idx` and bump its generation counter.
    ///
    /// Keeps any previously allocated `ssm_ptrs_dev` buffer so the next
    /// generation overwrites it in place without `cuMemAlloc`/`cuMemFree` churn.
    pub fn invalidate_slot(&mut self, slot_idx: usize) {
        if let Some(rec) = self.slots.get_mut(slot_idx) {
            rec.generation = rec.generation.wrapping_add(1);
            rec.cached_generation = None;
            rec.all_have_h_ckpt = false;
            rec.all_have_conv_ckpt = false;
        }
    }

    /// Lazily build and upload `slot_idx`'s pointer table if stale or absent.
    ///
    /// Uses a one-time synchronous H2D copy per slot generation so the uploaded
    /// table is visible to every CUDA stream for the remainder of the sequence's
    /// lifetime.
    ///
    /// # Errors
    ///
    /// Returns an error if `ptrs.len() != self.num_ssm_layers`, if `num_ssm_layers == 0`,
    /// or if GPU allocation/upload fails.
    pub fn ensure_slot_table(
        &mut self,
        slot_idx: usize,
        ptrs: &[SsmLayerCommitPointers],
        gpu: &dyn GpuBackend,
    ) -> Result<CachedSlotTable> {
        ensure!(
            self.num_ssm_layers > 0 && ptrs.len() == self.num_ssm_layers,
            "slot commit table expects {} SSM layers, got {}",
            self.num_ssm_layers,
            ptrs.len()
        );
        if slot_idx >= self.slots.len() {
            self.slots.resize(slot_idx + 1, SlotTableRecord::default());
        }
        if let Some(cached) = self.cached_slot(slot_idx) {
            return Ok(cached);
        }
        let table_bytes = self.slot_table_bytes()?;
        let rec = &mut self.slots[slot_idx];
        if rec.ssm_ptrs_dev.is_null() {
            rec.ssm_ptrs_dev = gpu.alloc(table_bytes)?;
        }
        let serialized = serialize_layer_pointers(ptrs);
        gpu.copy_h2d(&serialized, rec.ssm_ptrs_dev)?;
        rec.all_have_h_ckpt = ptrs.iter().all(|p| p.h_ckpt != 0);
        rec.all_have_conv_ckpt = ptrs.iter().all(|p| p.conv_ckpt != 0);
        rec.cached_generation = Some(rec.generation);
        Ok(CachedSlotTable {
            ssm_ptrs_dev: rec.ssm_ptrs_dev,
            generation: rec.generation,
            all_have_h_ckpt: rec.all_have_h_ckpt,
            all_have_conv_ckpt: rec.all_have_conv_ckpt,
        })
    }

    /// Lazily allocate (or grow) the device buffer holding `BatchCommitEntry[batch_size]`.
    ///
    /// Grow-only so an earlier in-flight GPU launch can never observe a freed pointer.
    ///
    /// # Errors
    ///
    /// Returns an error if `batch_size == 0`, on byte overflow, or if allocation fails.
    pub fn ensure_entries_buf(
        &mut self,
        batch_size: usize,
        gpu: &dyn GpuBackend,
    ) -> Result<DevicePtr> {
        ensure!(batch_size > 0, "batch commit requires at least one entry");
        if self.entries_dev.is_null() || self.entries_capacity < batch_size {
            let cap = batch_size.max(64);
            let bytes = cap
                .checked_mul(BatchCommitEntry::SIZE_BYTES)
                .context("batch commit entry buffer overflow")?;
            self.entries_dev = gpu.alloc(bytes)?;
            self.entries_capacity = cap;
        }
        Ok(self.entries_dev)
    }

    #[cfg(test)]
    fn seed_slot_for_test(
        &mut self,
        slot_idx: usize,
        dev: DevicePtr,
        all_have_h_ckpt: bool,
        all_have_conv_ckpt: bool,
    ) {
        if slot_idx >= self.slots.len() {
            self.slots.resize(slot_idx + 1, SlotTableRecord::default());
        }
        let rec = &mut self.slots[slot_idx];
        rec.ssm_ptrs_dev = dev;
        rec.cached_generation = Some(rec.generation);
        rec.all_have_h_ckpt = all_have_h_ckpt;
        rec.all_have_conv_ckpt = all_have_conv_ckpt;
    }
}

/// Serialize a slice of [`SsmLayerCommitPointers`] into contiguous little-endian bytes.
#[must_use]
pub fn serialize_layer_pointers(ptrs: &[SsmLayerCommitPointers]) -> Vec<u8> {
    let mut out = Vec::with_capacity(ptrs.len() * SsmLayerCommitPointers::SIZE_BYTES);
    for ptr in ptrs {
        out.extend_from_slice(&ptr.to_le_bytes());
    }
    out
}

/// Serialize a slice of [`BatchCommitEntry`] into contiguous little-endian bytes.
#[must_use]
pub fn serialize_batch_entries(entries: &[BatchCommitEntry]) -> Vec<u8> {
    let mut out = Vec::with_capacity(entries.len() * BatchCommitEntry::SIZE_BYTES);
    for entry in entries {
        out.extend_from_slice(&entry.to_le_bytes());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn abi_sizes_and_alignments_match_cuda_structs() {
        assert_eq!(std::mem::size_of::<SsmLayerCommitPointers>(), 64);
        assert_eq!(std::mem::align_of::<SsmLayerCommitPointers>(), 16);
        assert_eq!(std::mem::size_of::<BatchCommitEntry>(), 16);
        assert_eq!(std::mem::align_of::<BatchCommitEntry>(), 16);
        // 36 SSM layers × 64 B = 2,304 B/slot on Flash-Next.
        let table = SlotCommitTable::new(8, 36);
        assert_eq!(table.slot_table_bytes().expect("36 * 64 fits"), 2_304);
    }

    #[test]
    fn layer_pointers_and_batch_entries_serialize_exact_little_endian() {
        let ptrs = SsmLayerCommitPointers {
            h_state: 0x0102_0304_0506_0708,
            h_ckpt: 0x1112_1314_1516_1718,
            conv_state: 0x2122_2324_2526_2728,
            conv_ckpt: 0x3132_3334_3536_3738,
            h_inter_base: 0x4142_4344_4546_4748,
            conv_inter_base: 0x5152_5354_5556_5758,
            retain_slot_base: 0x6162_6364_6566_6768,
            _pad: 0,
        };
        let bytes = ptrs.to_le_bytes();
        assert_eq!(&bytes[0..8], &ptrs.h_state.to_le_bytes());
        assert_eq!(&bytes[8..16], &ptrs.h_ckpt.to_le_bytes());
        assert_eq!(&bytes[16..24], &ptrs.conv_state.to_le_bytes());
        assert_eq!(&bytes[24..32], &ptrs.conv_ckpt.to_le_bytes());
        assert_eq!(&bytes[32..40], &ptrs.h_inter_base.to_le_bytes());
        assert_eq!(&bytes[40..48], &ptrs.conv_inter_base.to_le_bytes());
        assert_eq!(&bytes[48..56], &ptrs.retain_slot_base.to_le_bytes());
        assert_eq!(&bytes[56..64], &0u64.to_le_bytes());

        let mut dst = [0u8; 64];
        ptrs.write_le_bytes(&mut dst).expect("exact 64 B slice");
        assert_eq!(dst, bytes);
        assert!(ptrs.write_le_bytes(&mut dst[..63]).is_err());

        let entry = BatchCommitEntry {
            ssm_table_dev: 0xDEAD_BEEF_CAFE_BABE,
            accepted_idx: 3,
            flags: BATCH_COMMIT_FLAG_MIRROR_CKPT | BATCH_COMMIT_FLAG_HAS_CONV_INTER,
        };
        let ebytes = entry.to_le_bytes();
        assert_eq!(&ebytes[0..8], &entry.ssm_table_dev.to_le_bytes());
        assert_eq!(&ebytes[8..12], &3u32.to_le_bytes());
        assert_eq!(&ebytes[12..16], &0x3u32.to_le_bytes());
        let mut edst = [0u8; 16];
        entry.write_le_bytes(&mut edst).expect("exact 16 B slice");
        assert_eq!(edst, ebytes);
        assert!(entry.write_le_bytes(&mut edst[..15]).is_err());

        let packed = serialize_batch_entries(&[entry, entry]);
        assert_eq!(packed.len(), 32);
        assert_eq!(&packed[..16], &ebytes);
        assert_eq!(&packed[16..32], &ebytes);
    }

    #[test]
    fn slot_generation_invalidation_clears_only_target_slot_and_preserves_allocation() {
        let mut table = SlotCommitTable::new(4, 36);
        assert_eq!(table.slot_generation(1), 0);
        assert!(table.cached_slot(1).is_none());

        table.seed_slot_for_test(1, DevicePtr(0x1000), true, true);
        table.seed_slot_for_test(2, DevicePtr(0x2000), true, true);

        let hit1 = table.cached_slot(1).expect("slot 1 cached at gen 0");
        assert_eq!(hit1.ssm_ptrs_dev, DevicePtr(0x1000));
        assert_eq!(hit1.generation, 0);
        assert!(hit1.all_have_h_ckpt && hit1.all_have_conv_ckpt);

        // Invalidate slot 1 (free_sequence / compact_sequence).
        table.invalidate_slot(1);
        assert_eq!(table.slot_generation(1), 1);
        assert!(
            table.cached_slot(1).is_none(),
            "invalidated slot must miss cache until rebuilt"
        );
        // Sibling slot 2 stays cached and untouched.
        let hit2 = table.cached_slot(2).expect("slot 2 unaffected");
        assert_eq!(hit2.ssm_ptrs_dev, DevicePtr(0x2000));
        assert_eq!(hit2.generation, 0);

        // Re-seeding slot 1 records the new generation 1 and reuses allocation.
        table.seed_slot_for_test(1, DevicePtr(0x1000), false, true);
        let rebuilt = table.cached_slot(1).expect("slot 1 rebuilt at gen 1");
        assert_eq!(rebuilt.generation, 1);
        assert!(!rebuilt.all_have_h_ckpt);
        assert!(rebuilt.all_have_conv_ckpt);
    }

    #[test]
    fn truthy_flag_parser_matches_no_mirror_contract() {
        for truthy in ["1", "true", "TRUE", "on", "ON"] {
            assert!(CommitRuntimeFlags::parse_truthy_flag(Some(truthy)));
        }
        for falsy in [
            None,
            Some(""),
            Some("0"),
            Some("false"),
            Some("off"),
            Some("2"),
        ] {
            assert!(!CommitRuntimeFlags::parse_truthy_flag(falsy));
        }
    }
}
