// SPDX-License-Identifier: AGPL-3.0-only

//! e4m3 activation arena of the cuBLASLt SSM projection route
//! (`ATLAS_SSM_PROJ_CUBLASLT=1`). Private to one layer by default; with
//! `ATLAS_MEMLEAN_SHARED_SCRATCH=1` the dense loader hands every SSM layer of
//! a model the same arena. Each projection rewrites the rows it reads (cast,
//! then GEMM, on one stream), and the layers run one after another, so one
//! arena per stream serves all of them. A shared arena is keyed by stream, so
//! a second stream gets its own allocation instead of rewriting bytes the
//! first stream's GEMM may still read.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use anyhow::{Result, anyhow, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};

/// The handle the loader gives every SSM layer that shares one arena.
#[derive(Clone, Default)]
pub struct SharedActE4m3Scratch {
    per_stream: Arc<Mutex<HashMap<u64, (DevicePtr, usize)>>>,
}

#[derive(Default)]
pub(super) struct ActE4m3Arena {
    /// `(pointer, capacity)` once allocated. Never replaced while work is
    /// live: an unexpectedly wide shape takes the handwritten path instead.
    private: Mutex<Option<(DevicePtr, usize)>>,
    /// Set when the layer adopted a shared arena.
    shared: Option<Arc<Mutex<HashMap<u64, (DevicePtr, usize)>>>>,
}

impl ActE4m3Arena {
    pub(super) fn adopt(&mut self, shared: &SharedActE4m3Scratch) -> Result<()> {
        ensure!(
            self.private
                .lock()
                .map_err(|_| anyhow!("SSM e4m3 activation scratch lock poisoned"))?
                .is_none(),
            "a shared SSM e4m3 activation scratch must be adopted before the layer allocates its own"
        );
        self.shared = Some(Arc::clone(&shared.per_stream));
        Ok(())
    }

    /// Scratch that can hold `bytes`, allocating `capacity` bytes on first
    /// use (per stream when shared). `None` wherever the private arena has
    /// always fallen back to the handwritten kernel: lock poisoned, no
    /// capacity, too small, allocation failure. The second element is the
    /// capacity when this call allocated.
    pub(super) fn get(
        &self,
        gpu: &dyn GpuBackend,
        bytes: usize,
        capacity: Option<usize>,
        stream: u64,
    ) -> Option<(DevicePtr, Option<usize>)> {
        let allocate = || -> Option<(DevicePtr, usize)> {
            let capacity = capacity.filter(|&capacity| capacity >= bytes)?;
            Some((gpu.alloc(capacity).ok()?, capacity))
        };
        match &self.shared {
            None => {
                let mut slot = self.private.lock().ok()?;
                if let Some((ptr, cap)) = *slot {
                    return (cap >= bytes).then_some((ptr, None));
                }
                let (ptr, capacity) = allocate()?;
                *slot = Some((ptr, capacity));
                Some((ptr, Some(capacity)))
            }
            Some(per_stream) => {
                let mut arenas = per_stream.lock().ok()?;
                if let Some(&(ptr, cap)) = arenas.get(&stream) {
                    return (cap >= bytes).then_some((ptr, None));
                }
                let (ptr, capacity) = allocate()?;
                arenas.insert(stream, (ptr, capacity));
                Some((ptr, Some(capacity)))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use spark_runtime::gpu::mock::MockGpuBackend;

    use super::{ActE4m3Arena, SharedActE4m3Scratch};

    #[test]
    fn a_private_arena_allocates_once_and_never_grows() {
        let gpu = MockGpuBackend::new();
        let arena = ActE4m3Arena::default();
        let (ptr, allocated) = arena.get(&gpu, 100, Some(1_000), 7).unwrap();
        assert_eq!(allocated, Some(1_000));
        assert_eq!(arena.get(&gpu, 1_000, Some(1_000), 9), Some((ptr, None)));
        assert_eq!(arena.get(&gpu, 1_001, Some(4_000), 7), None);
        assert_eq!(gpu.alloc_count(), 1);
    }

    #[test]
    fn a_private_arena_falls_back_without_capacity() {
        let gpu = MockGpuBackend::new();
        let arena = ActE4m3Arena::default();
        assert_eq!(arena.get(&gpu, 100, None, 7), None);
        assert_eq!(arena.get(&gpu, 100, Some(99), 7), None);
        assert_eq!(gpu.alloc_count(), 0);
    }

    #[test]
    fn adopters_share_one_allocation_per_stream() {
        let gpu = MockGpuBackend::new();
        let shared = SharedActE4m3Scratch::default();
        let mut layers: Vec<ActE4m3Arena> = (0..48).map(|_| ActE4m3Arena::default()).collect();
        for arena in &mut layers {
            arena.adopt(&shared).unwrap();
        }
        // Qwen3.8-27B: 8193 arena rows x max(5120, 6144).
        let capacity = 8_193 * 6_144;
        let (ptr, allocated) = layers[0].get(&gpu, 4_096, Some(capacity), 7).unwrap();
        assert_eq!(allocated, Some(capacity));
        for arena in &layers[1..] {
            assert_eq!(
                arena.get(&gpu, capacity, Some(capacity), 7),
                Some((ptr, None))
            );
        }
        assert_eq!(gpu.alloc_count(), 1);
        assert_eq!(layers[0].get(&gpu, capacity + 1, Some(capacity), 7), None);
    }

    #[test]
    fn a_second_stream_gets_its_own_shared_arena() {
        let gpu = MockGpuBackend::new();
        let shared = SharedActE4m3Scratch::default();
        let mut first = ActE4m3Arena::default();
        let mut second = ActE4m3Arena::default();
        first.adopt(&shared).unwrap();
        second.adopt(&shared).unwrap();
        let (on_seven, _) = first.get(&gpu, 64, Some(64), 7).unwrap();
        let (on_nine, allocated) = second.get(&gpu, 64, Some(64), 9).unwrap();
        assert_ne!(on_seven, on_nine);
        assert_eq!(allocated, Some(64));
        assert_eq!(second.get(&gpu, 64, Some(64), 7), Some((on_seven, None)));
        assert_eq!(gpu.alloc_count(), 2);
    }

    #[test]
    fn adoption_after_a_private_allocation_is_refused() {
        let gpu = MockGpuBackend::new();
        let mut arena = ActE4m3Arena::default();
        arena.get(&gpu, 64, Some(64), 7).unwrap();
        let error = arena
            .adopt(&SharedActE4m3Scratch::default())
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("before the layer allocates its own"),
            "{error}"
        );
    }
}
