// SPDX-License-Identifier: AGPL-3.0-only

//! KV blocks in the GPU backend's carveout (`spark_runtime::gpu::carveout`,
//! the GB10 display carveout under `spark display-carveout`).
//!
//! The carveout is extra memory outside Linux's `MemAvailable` pool, so it
//! adds blocks without taking any from system memory: the pool grows to the
//! most blocks whose buffers left in system memory still fit the bytes the
//! budgeted pool would have used, with whole per-layer buffers moved into the
//! carveout.

use spark_runtime::gpu::GpuBackend;
use spark_runtime::kv_cache::{KvBuffer, KvCacheConfig, KvPlacement, PagedKvCache};

/// The cache geometry the factory allocates (`PagedKvCache::new_placed`).
#[derive(Clone, Copy)]
pub(super) struct KvShape<'a> {
    pub(super) config: &'a KvCacheConfig,
    pub(super) v_aliases_k: bool,
    /// Per-block auxiliary bytes that remain in system memory even when K/V
    /// pools move to the display carveout (e.g. Qwen4 QSA side caches and
    /// optional E4M3 K/V shadows).
    pub(super) auxiliary_block_bytes: usize,
}

impl KvShape<'_> {
    /// Per-token aux bytes grow linearly with `block_size` per added block.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(super) fn with_per_token_aux(self, bytes: usize) -> anyhow::Result<Self> {
        let block_aux = bytes
            .checked_mul(self.config.block_size)
            .ok_or_else(|| anyhow::anyhow!("KV auxiliary block budget overflow"))?;
        self.with_per_block_aux(block_aux)
    }

    /// Per-block aux bytes that remain in system memory for every added block.
    pub(super) fn with_per_block_aux(mut self, bytes: usize) -> anyhow::Result<Self> {
        self.auxiliary_block_bytes = self
            .auxiliary_block_bytes
            .checked_add(bytes)
            .ok_or_else(|| anyhow::anyhow!("KV auxiliary block budget overflow"))?;
        Ok(self)
    }

    fn buffers(self, blocks: usize) -> Vec<(KvBuffer, usize)> {
        PagedKvCache::buffer_sizes(self.config, blocks, self.v_aliases_k)
    }

    /// System-memory bytes outside the per-layer KV pools that depend on `blocks`.
    fn side_bytes(self, blocks: usize) -> usize {
        blocks.saturating_mul(self.auxiliary_block_bytes)
    }

    /// System-memory bytes of a `blocks` pool with `placement`'s buffers in
    /// the carveout.
    fn system_bytes(self, blocks: usize, placement: &KvPlacement) -> usize {
        let buffers = self.buffers(blocks);
        total(&buffers) - placement.carved_bytes(&buffers) + self.side_bytes(blocks)
    }
}

fn total(buffers: &[(KvBuffer, usize)]) -> usize {
    buffers.iter().map(|(_, bytes)| bytes).sum()
}

/// The block count and placement for a pool budgeted at `blocks`, given a
/// carveout of `capacity` bytes. Without one: `blocks`, nothing placed.
pub(super) fn plan(shape: KvShape, blocks: usize, capacity: usize) -> (usize, KvPlacement) {
    if capacity == 0 || blocks == 0 {
        return (blocks, KvPlacement::default());
    }
    let system = shape.system_bytes(blocks, &KvPlacement::default());
    let per_block = (total(&shape.buffers(blocks + 2)) - total(&shape.buffers(blocks))) / 2
        + shape.auxiliary_block_bytes;
    let index_max_blocks = ((i32::MAX as usize) / shape.config.block_size.max(1)).max(blocks);
    // A block's bytes all in the carveout bound the gain; walk down from
    // there to the first count whose system-memory share fits. `blocks` always does.
    let ceiling = blocks
        .saturating_add(capacity / per_block.max(1))
        .saturating_add(1)
        .min(index_max_blocks);
    for n in (blocks..=ceiling).rev() {
        let placement = KvPlacement::plan(&shape.buffers(n), capacity);
        if shape.system_bytes(n, &placement) <= system {
            return (n, placement);
        }
    }
    unreachable!("the budgeted count fits its own system-memory bytes")
}

/// [`plan`] against `gpu`'s carveout, logged.
pub(super) fn extend(gpu: &dyn GpuBackend, shape: KvShape, blocks: usize) -> (usize, KvPlacement) {
    let capacity = gpu.carveout_capacity();
    let (n, placement) = plan(shape, blocks, capacity);
    if !placement.is_empty() {
        let carved = placement.carved_bytes(&shape.buffers(n));
        tracing::info!(
            "KV cache: display carveout {:.0} MiB holds {} buffers ({:.0} MiB) → {} blocks \
             (+{}, +{:.1}%) for the same system memory",
            capacity as f64 / (1 << 20) as f64,
            placement.len(),
            carved as f64 / (1 << 20) as f64,
            n,
            n - blocks,
            (n - blocks) as f64 * 100.0 / blocks as f64,
        );
    }
    (n, placement)
}

#[cfg(test)]
#[path = "kv_carveout_tests.rs"]
mod tests;
