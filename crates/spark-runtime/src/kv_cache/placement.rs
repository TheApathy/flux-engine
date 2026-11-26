// SPDX-License-Identifier: AGPL-3.0-only

//! Which KV buffers live in the backend's carveout
//! ([`GpuBackend::alloc_carveout`], `gpu::carveout`).
//!
//! Each layer's pools are separate allocations, so whole buffers move: a
//! pool never straddles the carveout and system memory. A placement is
//! planned once, at the largest block count a rank sizes for, and then kept:
//! any agreed block count no larger only shrinks every buffer, so the chosen
//! set still fits the carveout and the buffers left in system memory never
//! grow past what the rank sized for.

use std::collections::BTreeSet;

use anyhow::Result;

use super::{KvCacheConfig, PagedKvCache};
use crate::gpu::carveout::CarveoutArena;
use crate::gpu::{DevicePtr, GpuBackend};

/// One per-layer device buffer of a [`PagedKvCache`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum KvBuffer {
    K(usize),
    V(usize),
    IndexValues(usize),
    IndexScales(usize),
    IndexTail(usize),
    /// The slot map of lent index tails (a `u32` per block).
    TailMap,
}

/// The buffers allocated from the carveout; every other one comes from
/// system memory. The default places nothing.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct KvPlacement {
    carveout: BTreeSet<KvBuffer>,
}

/// Which buffers may move to the carveout (`ATLAS_KV_CARVEOUT_ORDER`).
///
/// K/V pools first because the carveout is sized to hold whole attention
/// pools, and keeping auxiliary index tables in system memory preserves
/// uniform access patterns when `Size` is not explicitly requested.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CarveoutOrder {
    /// Any buffer, largest first.
    Size,
    /// K/V (or MLA latent) pools only, largest first.
    Latent,
}

impl CarveoutOrder {
    #[must_use]
    pub fn from_env() -> Self {
        let raw = std::env::var("ATLAS_KV_CARVEOUT_ORDER")
            .or_else(|_| std::env::var("FLUX_KV_CARVEOUT_ORDER"));
        match raw.as_deref() {
            Ok("size") => Self::Size,
            _ => Self::Latent,
        }
    }

    fn takes(self, buffer: KvBuffer) -> bool {
        self == Self::Size || matches!(buffer, KvBuffer::K(_) | KvBuffer::V(_))
    }
}

impl KvPlacement {
    /// [`Self::plan_ordered`] in the order `ATLAS_KV_CARVEOUT_ORDER` selects.
    #[must_use]
    pub fn plan(buffers: &[(KvBuffer, usize)], capacity: usize) -> Self {
        Self::plan_ordered(buffers, capacity, CarveoutOrder::from_env())
    }

    /// The buffers `which` takes, largest first, each taken while its
    /// footprint still fits `capacity`. Ties go to the lower [`KvBuffer`],
    /// so every rank with the same sizes plans the same set.
    #[must_use]
    pub fn plan_ordered(
        buffers: &[(KvBuffer, usize)],
        capacity: usize,
        which: CarveoutOrder,
    ) -> Self {
        let mut order: Vec<_> = buffers
            .iter()
            .filter(|(buffer, bytes)| *bytes > 0 && which.takes(*buffer))
            .collect();
        order.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        let mut left = capacity;
        let mut carveout = BTreeSet::new();
        for &&(buffer, bytes) in &order {
            let need = CarveoutArena::footprint(bytes);
            if need <= left {
                left -= need;
                carveout.insert(buffer);
            }
        }
        Self { carveout }
    }

    #[must_use]
    pub fn contains(&self, buffer: KvBuffer) -> bool {
        self.carveout.contains(&buffer)
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.carveout.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.carveout.is_empty()
    }

    /// Bytes of `buffers` this placement keeps out of system memory.
    #[must_use]
    pub fn carved_bytes(&self, buffers: &[(KvBuffer, usize)]) -> usize {
        buffers
            .iter()
            .filter(|(buffer, _)| self.contains(*buffer))
            .map(|(_, bytes)| bytes)
            .sum()
    }

    pub(super) fn alloc(
        &self,
        gpu: &dyn GpuBackend,
        buffer: KvBuffer,
        bytes: usize,
    ) -> Result<DevicePtr> {
        if self.contains(buffer) {
            gpu.alloc_carveout(bytes)
        } else {
            gpu.alloc(bytes)
        }
    }
}

impl PagedKvCache {
    /// The per-layer buffers [`Self::new_placed`] allocates for `num_blocks`,
    /// with their byte sizes.
    #[must_use]
    pub fn buffer_sizes(
        config: &KvCacheConfig,
        num_blocks: usize,
        v_aliases_k: bool,
    ) -> Vec<(KvBuffer, usize)> {
        Self::buffer_sizes_with_k_slots(config, num_blocks, num_blocks, v_aliases_k)
    }

    /// [`Self::buffer_sizes`] with `k_slots` K-pool block slots per layer.
    #[must_use]
    pub fn buffer_sizes_with_k_slots(
        config: &KvCacheConfig,
        num_blocks: usize,
        k_slots: usize,
        v_aliases_k: bool,
    ) -> Vec<(KvBuffer, usize)> {
        let mut out = Vec::with_capacity(config.num_layers * if v_aliases_k { 1 } else { 2 });
        for layer in 0..config.num_layers {
            out.push((
                KvBuffer::K(layer),
                k_slots * config.k_block_bytes_for_layer(layer),
            ));
            if !v_aliases_k {
                out.push((
                    KvBuffer::V(layer),
                    num_blocks * config.v_block_bytes_for_layer(layer),
                ));
            }
        }
        out
    }
}

#[cfg(test)]
#[path = "placement_tests.rs"]
mod tests;
