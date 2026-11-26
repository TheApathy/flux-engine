// SPDX-License-Identifier: AGPL-3.0-only

//! Per-slot recurrent history that lives outside `SsmStatePool` (Qwen4's PLE
//! convolution history), carried by `SsmSnapshotPool` alongside GDN h/conv.
//!
//! Without it a Marconi prefix restore or a decode-rollback restore rewinds
//! GDN while the PLE history still describes the tokens it skipped or dropped
//! (the reason Qwen4 refuses prefix caching, qwen4_concurrency.rs). Restore
//! writes the live history and the speculative checkpoint, so the slot is in
//! the same state as one that reached the position in place.

use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};

pub(crate) struct SlotHistory {
    live: DevicePtr,
    checkpoint: DevicePtr,
    stride: usize,
    slots: usize,
    marconi: DevicePtr,
    marconi_slots: usize,
    decode: DevicePtr,
    decode_slots: usize,
}

/// Which snapshot region an index refers to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum HistoryRegion {
    Marconi,
    Decode,
}

impl SlotHistory {
    /// `live` / `checkpoint`: slot 0 of the source buffers, `stride` bytes per
    /// slot, `slots` slots. Allocates one stride per snapshot index.
    pub(crate) fn new(
        gpu: &dyn GpuBackend,
        live: DevicePtr,
        checkpoint: DevicePtr,
        stride: usize,
        slots: usize,
        marconi_slots: usize,
        decode_slots: usize,
    ) -> Result<Self> {
        ensure!(stride > 0 && slots > 0, "slot history: empty geometry");
        let alloc = |n: usize| -> Result<DevicePtr> {
            if n == 0 {
                return Ok(DevicePtr::NULL);
            }
            let bytes = n
                .checked_mul(stride)
                .ok_or_else(|| anyhow::anyhow!("slot history: {n} x {stride} overflows"))?;
            let ptr = gpu.alloc(bytes)?;
            if let Err(err) = gpu.memset(ptr, 0, bytes) {
                let _ = gpu.free(ptr);
                return Err(err);
            }
            Ok(ptr)
        };
        let marconi = alloc(marconi_slots)?;
        let decode = match alloc(decode_slots) {
            Ok(ptr) => ptr,
            Err(err) => {
                if !marconi.is_null() {
                    let _ = gpu.free(marconi);
                }
                return Err(err);
            }
        };
        Ok(Self {
            live,
            checkpoint,
            stride,
            slots,
            marconi,
            marconi_slots,
            decode,
            decode_slots,
        })
    }

    pub(crate) fn bytes(&self) -> usize {
        (self.marconi_slots + self.decode_slots) * self.stride
    }

    fn snapshot(&self, region: HistoryRegion, index: usize) -> Result<DevicePtr> {
        let (base, n) = match region {
            HistoryRegion::Marconi => (self.marconi, self.marconi_slots),
            HistoryRegion::Decode => (self.decode, self.decode_slots),
        };
        ensure!(index < n, "slot history: {region:?} index {index} >= {n}");
        Ok(base.offset(index * self.stride))
    }

    fn source(&self, slot: usize) -> Result<usize> {
        ensure!(
            slot < self.slots,
            "slot history: slot {slot} >= {}",
            self.slots
        );
        Ok(slot * self.stride)
    }

    pub(crate) fn save(
        &self,
        gpu: &dyn GpuBackend,
        slot: usize,
        region: HistoryRegion,
        index: usize,
        stream: u64,
    ) -> Result<()> {
        let src = self.live.offset(self.source(slot)?);
        gpu.copy_d2d_async(src, self.snapshot(region, index)?, self.stride, stream)
    }

    pub(crate) fn restore(
        &self,
        gpu: &dyn GpuBackend,
        region: HistoryRegion,
        index: usize,
        slot: usize,
        stream: u64,
    ) -> Result<()> {
        let src = self.snapshot(region, index)?;
        let offset = self.source(slot)?;
        gpu.copy_d2d_async(src, self.live.offset(offset), self.stride, stream)?;
        gpu.copy_d2d_async(src, self.checkpoint.offset(offset), self.stride, stream)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use spark_runtime::gpu::mock::MockGpuBackend;

    fn read(gpu: &MockGpuBackend, ptr: DevicePtr, n: usize) -> Vec<u8> {
        let mut out = vec![0u8; n];
        gpu.copy_d2h(ptr, &mut out).unwrap();
        out
    }

    #[test]
    fn restore_rewinds_live_and_checkpoint_of_the_target_slot_only() {
        let gpu = MockGpuBackend::new();
        let stride = 24;
        let live = gpu.alloc(3 * stride).unwrap();
        let checkpoint = gpu.alloc(3 * stride).unwrap();
        let h = SlotHistory::new(&gpu, live, checkpoint, stride, 3, 2, 4).unwrap();
        assert_eq!(h.bytes(), 6 * stride);

        let at_boundary: Vec<u8> = (0..stride as u8).collect();
        gpu.copy_h2d(&at_boundary, live.offset(stride)).unwrap();
        h.save(&gpu, 1, HistoryRegion::Decode, 3, 0).unwrap();
        // Decoding past the boundary changes the live history.
        gpu.copy_h2d(&vec![0xAA; stride], live.offset(stride))
            .unwrap();
        gpu.copy_h2d(&vec![0xBB; stride], checkpoint.offset(stride))
            .unwrap();
        gpu.copy_h2d(&vec![0xCC; stride], live.offset(2 * stride))
            .unwrap();

        h.restore(&gpu, HistoryRegion::Decode, 3, 1, 0).unwrap();
        assert_eq!(read(&gpu, live.offset(stride), stride), at_boundary);
        assert_eq!(read(&gpu, checkpoint.offset(stride), stride), at_boundary);
        assert_eq!(
            read(&gpu, live.offset(2 * stride), stride),
            vec![0xCC; stride]
        );
    }

    #[test]
    fn regions_and_slots_are_bounds_checked() {
        let gpu = MockGpuBackend::new();
        let live = gpu.alloc(16).unwrap();
        let checkpoint = gpu.alloc(16).unwrap();
        let h = SlotHistory::new(&gpu, live, checkpoint, 8, 2, 1, 0).unwrap();
        assert!(h.save(&gpu, 2, HistoryRegion::Marconi, 0, 0).is_err());
        assert!(h.save(&gpu, 0, HistoryRegion::Marconi, 1, 0).is_err());
        assert!(h.save(&gpu, 0, HistoryRegion::Decode, 0, 0).is_err());
        h.save(&gpu, 1, HistoryRegion::Marconi, 0, 0).unwrap();
    }
}
