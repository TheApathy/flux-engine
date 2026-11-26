// SPDX-License-Identifier: AGPL-3.0-only

//! One split-K FP32 workspace for every layer of a model
//! (`ATLAS_MEMLEAN_SHARED_SCRATCH=1`). Without it each FFN, SSM and attention
//! layer allocates its own `[k_splits <= 8, 32, N]` FP32 workspace: on
//! Qwen3.8-27B that is 64 x 17 MiB + 48 x 8 MiB + 16 x 1 MiB. A split-K
//! projection writes its partials and reduces them on the same stream before
//! the next projection starts (gate and up already share one per-layer
//! workspace on that argument), and a captured graph replays the layers in
//! order, so one buffer sized for the widest projection serves all of them.
//! The buffer is allocated once at load and never moves, so graph capture
//! sees one constant address.

use std::sync::Mutex;

use anyhow::{Result, anyhow, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};

pub struct SharedSplitkWorkspace {
    capacity: usize,
    slot: Mutex<Option<DevicePtr>>,
}

impl SharedSplitkWorkspace {
    /// Covers every split-K projection of a model whose widest split-K output
    /// is `max_n`: `8` is the split clamp shared by every `*_SPLITK` selector
    /// and `32` the split-K kernel's M tile.
    pub fn for_max_n(max_n: usize) -> Self {
        Self {
            capacity: 8 * 32 * max_n * std::mem::size_of::<f32>(),
            slot: Mutex::new(None),
        }
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// The shared workspace for a layer that needs `bytes`, allocated at full
    /// capacity on the first lease.
    pub fn lease(&self, gpu: &dyn GpuBackend, bytes: usize) -> Result<DevicePtr> {
        ensure!(
            bytes <= self.capacity,
            "split-K workspace request of {bytes} bytes exceeds the shared {} bytes",
            self.capacity
        );
        let mut slot = self
            .slot
            .lock()
            .map_err(|_| anyhow!("shared split-K workspace lock poisoned"))?;
        if let Some(ptr) = *slot {
            return Ok(ptr);
        }
        let ptr = gpu.alloc(self.capacity)?;
        *slot = Some(ptr);
        Ok(ptr)
    }
}

#[cfg(test)]
mod tests {
    use spark_runtime::gpu::mock::MockGpuBackend;

    use super::SharedSplitkWorkspace;

    #[test]
    fn qwen38_27b_capacity_is_the_widest_ffn_projection() {
        // max(hidden 5120, intermediate 17408, qkvz 16384, kv 1024).
        let shared = SharedSplitkWorkspace::for_max_n(17_408);
        assert_eq!(shared.capacity(), 8 * 32 * 17_408 * 4);
        assert_eq!(shared.capacity(), 17_825_792);
    }

    #[test]
    fn every_lease_returns_the_one_allocation() {
        let gpu = MockGpuBackend::new();
        let shared = SharedSplitkWorkspace::for_max_n(17_408);
        let ffn = shared.lease(&gpu, 8 * 32 * 17_408 * 4).unwrap();
        let ssm = shared.lease(&gpu, 4 * 32 * 16_384 * 4).unwrap();
        let attention = shared.lease(&gpu, 8 * 32 * 1_024 * 4).unwrap();
        assert_eq!(ffn, ssm);
        assert_eq!(ffn, attention);
        assert_eq!(gpu.alloc_count(), 1);
    }

    #[test]
    fn an_oversized_request_fails_before_allocating() {
        let gpu = MockGpuBackend::new();
        let shared = SharedSplitkWorkspace::for_max_n(1_024);
        assert!(shared.lease(&gpu, shared.capacity() + 1).is_err());
        assert_eq!(gpu.alloc_count(), 0);
    }
}
