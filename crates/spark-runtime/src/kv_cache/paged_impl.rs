// SPDX-License-Identifier: AGPL-3.0-only
//
// `PagedKvCache` impl block. Split from `kv_cache.rs` so the parent file
// keeps the public types small enough to read at a glance. The struct
// definition lives in the parent.

use anyhow::{Result, bail};

use super::{
    KvBuffer, KvCacheConfig, KvCacheDtype, KvPlacement, LayerPool, PagedKvCache,
    next_paged_kv_cache_id,
};
use crate::gpu::{DevicePtr, GpuBackend};

impl PagedKvCache {
    /// Allocate the KV cache pool on the GPU.
    pub fn new(config: KvCacheConfig, num_blocks: usize, gpu: &dyn GpuBackend) -> Result<Self> {
        Self::new_with_v_alias(config, num_blocks, gpu, false)
    }

    /// [`Self::new`], optionally with every layer's V side aliasing its K
    /// storage: for caches whose writers store one latent on both sides,
    /// which halves the pool.
    pub fn new_with_v_alias(
        config: KvCacheConfig,
        num_blocks: usize,
        gpu: &dyn GpuBackend,
        v_aliases_k: bool,
    ) -> Result<Self> {
        Self::new_placed(config, num_blocks, gpu, v_aliases_k, KvPlacement::default())
    }

    /// [`Self::new_with_v_alias`] with the pools `placement` names allocated
    /// from the backend's carveout. Allocation is transactional: any failure
    /// releases all previously allocated system and carveout pools.
    pub fn new_placed(
        config: KvCacheConfig,
        num_blocks: usize,
        gpu: &dyn GpuBackend,
        v_aliases_k: bool,
        placement: KvPlacement,
    ) -> Result<Self> {
        let cache_id = next_paged_kv_cache_id()?;
        let mut layers: Vec<LayerPool> = Vec::with_capacity(config.num_layers);
        let mut total_bytes: usize = 0;
        let mut initialised = 0usize;
        let release = |layers: &[LayerPool]| {
            for layer in layers {
                let _ = gpu.free(layer.k_pool);
                if layer.v_pool != layer.k_pool {
                    let _ = gpu.free(layer.v_pool);
                }
            }
        };
        for i in 0..config.num_layers {
            // Per-side block_bytes: for symmetric dtypes both are equal; for
            // asymmetric (e.g. Bf16KTurbo3V) the K pool is allocated bf16-sized
            // and the V pool is allocated turbo3-sized — avoids the 4× V over-
            // allocation that would result from a single MAX-sized stride.
            let k_block_bytes = config.k_block_bytes_for_layer(i);
            let k_pool_bytes = num_blocks * k_block_bytes;
            let k_pool = match placement.alloc(gpu, KvBuffer::K(i), k_pool_bytes) {
                Ok(ptr) => ptr,
                Err(error) => {
                    release(&layers);
                    return Err(error);
                }
            };
            let (v_pool, v_block_bytes, v_pool_bytes) = if v_aliases_k {
                (k_pool, k_block_bytes, 0)
            } else {
                let v_block_bytes = config.v_block_bytes_for_layer(i);
                let v_pool_bytes = num_blocks * v_block_bytes;
                let v_pool = match placement.alloc(gpu, KvBuffer::V(i), v_pool_bytes) {
                    Ok(ptr) => ptr,
                    Err(error) => {
                        let _ = gpu.free(k_pool);
                        release(&layers);
                        return Err(error);
                    }
                };
                (v_pool, v_block_bytes, v_pool_bytes)
            };
            // ATLAS_PREFILL_SCRATCH_INIT=poison|arena-zero: never-written pages read as NaN / zero.
            let init_res =
                crate::scratch_init::init_arena(gpu, k_pool, k_pool_bytes).and_then(|k_init| {
                    let v_init = if v_aliases_k {
                        false
                    } else {
                        crate::scratch_init::init_arena(gpu, v_pool, v_pool_bytes)?
                    };
                    Ok(k_init | v_init)
                });
            match init_res {
                Ok(true) => initialised += k_pool_bytes + v_pool_bytes,
                Ok(false) => {}
                Err(error) => {
                    let _ = gpu.free(k_pool);
                    if !v_aliases_k {
                        let _ = gpu.free(v_pool);
                    }
                    release(&layers);
                    return Err(error);
                }
            }
            total_bytes += k_pool_bytes + v_pool_bytes;
            layers.push(LayerPool {
                k_pool,
                v_pool,
                k_block_stride: k_block_bytes,
                v_block_stride: v_block_bytes,
                dtype: config.dtype_for_layer(i),
            });
        }

        if initialised > 0 {
            tracing::info!(
                "PREFILL_SCRATCH_INIT selector={} site=kv_pool bytes={initialised}",
                crate::scratch_init::SELECTOR
            );
        }

        let free_blocks: Vec<u32> = (0..num_blocks as u32).rev().collect();
        let block_ref_counts = vec![0u32; num_blocks];

        let has_mixed = !config.layer_dtypes.is_empty()
            && config.layer_dtypes.iter().any(|d| *d != config.dtype);
        if has_mixed {
            let hp_count = config
                .layer_dtypes
                .iter()
                .filter(|d| **d != config.dtype)
                .count();
            tracing::info!(
                "KV cache: {} blocks × {} layers ({} high-precision) = {:.1} GB total (mixed dtype)",
                num_blocks,
                config.num_layers,
                hp_count,
                total_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
            );
        } else {
            let bytes_per_block = if v_aliases_k {
                config.k_block_bytes_dims(config.dtype, config.num_kv_heads, config.head_dim)
            } else {
                config.block_bytes_kv()
            };
            tracing::info!(
                "KV cache: {} blocks × {} layers × {} bytes/block = {:.1} GB total{}",
                num_blocks,
                config.num_layers,
                bytes_per_block,
                total_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
                if v_aliases_k { " (V aliases K)" } else { "" },
            );
        }

        Ok(Self {
            cache_id,
            layers,
            num_blocks,
            free_blocks,
            block_ref_counts,
            config,
            placement,
        })
    }

    /// Which buffers of this cache were placed in the backend's carveout.
    pub fn placement(&self) -> &KvPlacement {
        &self.placement
    }

    /// Explicitly release every per-layer device pool (both system and carveout).
    pub fn release(&mut self, gpu: &dyn GpuBackend) -> Result<()> {
        let mut first_err: Option<anyhow::Error> = None;
        for layer in &mut self.layers {
            let aliased = layer.v_pool == layer.k_pool;
            if !layer.k_pool.is_null() {
                match gpu.free(layer.k_pool) {
                    Ok(()) => {
                        layer.k_pool = DevicePtr::NULL;
                        if aliased {
                            layer.v_pool = DevicePtr::NULL;
                        }
                    }
                    Err(e) if first_err.is_none() => first_err = Some(e),
                    Err(_) => {}
                }
            }
            if !aliased && !layer.v_pool.is_null() {
                match gpu.free(layer.v_pool) {
                    Ok(()) => layer.v_pool = DevicePtr::NULL,
                    Err(e) if first_err.is_none() => first_err = Some(e),
                    Err(_) => {}
                }
            }
        }
        match first_err {
            Some(err) => Err(err),
            None => Ok(()),
        }
    }

    /// Allocate a free block. Returns block index.
    pub fn alloc_block(&mut self) -> Result<u32> {
        let idx = self
            .free_blocks
            .pop()
            .ok_or_else(|| anyhow::anyhow!("KV cache exhausted: no free blocks"))?;
        self.block_ref_counts[idx as usize] = 1;
        Ok(idx)
    }

    /// Zero all KV data in a block across all layers.
    /// Prevents stale KV data from previous sequences from leaking into
    /// new sequences via paged attention reads beyond the current seq_len.
    pub fn zero_block(
        &self,
        block_idx: u32,
        gpu: &dyn crate::gpu::GpuBackend,
        stream: u64,
    ) -> anyhow::Result<()> {
        for layer in &self.layers {
            let k_offset = block_idx as usize * layer.k_block_stride;
            gpu.memset_async(
                layer.k_pool.offset(k_offset),
                0,
                layer.k_block_stride,
                stream,
            )?;
            if layer.v_pool != layer.k_pool {
                let v_offset = block_idx as usize * layer.v_block_stride;
                gpu.memset_async(
                    layer.v_pool.offset(v_offset),
                    0,
                    layer.v_block_stride,
                    stream,
                )?;
            }
        }
        Ok(())
    }

    /// Zero all KV data of `blocks` across all layers with one memset per
    /// (layer, side, run of consecutive block indices) instead of
    /// [`Self::zero_block`]'s two memsets per layer per block. Same bytes, same
    /// value, same stream: the device state afterwards is identical
    /// (`zero_runs` tests; micro 100). Returns `(runs, memsets)`.
    /// `ATLAS_PREFILL_KV_ZERO_COALESCE`, K8 item A.
    pub fn zero_blocks(
        &self,
        blocks: &[u32],
        gpu: &dyn crate::gpu::GpuBackend,
        stream: u64,
    ) -> anyhow::Result<(usize, usize)> {
        let runs = super::zero_runs::block_runs(blocks);
        if let Some(last) = runs.last() {
            anyhow::ensure!(
                last.end() <= self.num_blocks as u64,
                "zero_blocks: block {} out of range (num_blocks {})",
                last.end() - 1,
                self.num_blocks
            );
        }
        let strides: Vec<(usize, usize)> = self
            .layers
            .iter()
            .map(|l| (l.k_block_stride, l.v_block_stride))
            .collect();
        let plan = super::zero_runs::coalesced_plan(&runs, &strides)
            .ok_or_else(|| anyhow::anyhow!("zero_blocks: byte offset overflow"))?;
        let mut memsets = 0usize;
        for span in &plan {
            let layer = &self.layers[span.pool / 2];
            let pool = if span.pool % 2 == 0 {
                layer.k_pool
            } else if layer.v_pool == layer.k_pool {
                // V aliases K (e.g. MLA/DSA): K-side span already zeroes the shared pool.
                continue;
            } else {
                layer.v_pool
            };
            gpu.memset_async(pool.offset(span.offset), 0, span.bytes, stream)?;
            memsets += 1;
        }
        Ok((runs.len(), memsets))
    }

    /// Try to allocate a free block without failing. Returns None if exhausted.
    pub fn try_alloc_block(&mut self) -> Option<u32> {
        let idx = self.free_blocks.pop()?;
        self.block_ref_counts[idx as usize] = 1;
        Some(idx)
    }

    /// Increment reference count on a block (for prefix cache sharing).
    pub fn inc_ref(&mut self, block_idx: u32) {
        debug_assert!((block_idx as usize) < self.num_blocks);
        self.block_ref_counts[block_idx as usize] += 1;
    }

    /// Decrement reference count. Returns true if block was freed (count hit 0).
    pub fn dec_ref(&mut self, block_idx: u32) -> bool {
        let idx = block_idx as usize;
        debug_assert!(idx < self.num_blocks);
        debug_assert!(
            self.block_ref_counts[idx] > 0,
            "dec_ref on block with 0 refs"
        );
        self.block_ref_counts[idx] -= 1;
        if self.block_ref_counts[idx] == 0 {
            self.free_blocks.push(block_idx);
            true
        } else {
            false
        }
    }

    /// Free a previously allocated block (decrements ref, frees if count hits 0).
    pub fn free_block(&mut self, block_idx: u32) {
        self.dec_ref(block_idx);
    }

    /// Free all blocks in a block table.
    pub fn free_blocks(&mut self, block_table: &[u32]) {
        for &idx in block_table {
            self.free_block(idx);
        }
    }

    /// Return a block to the free pool directly, bypassing ref counting.
    /// Used by eviction: the radix tree already removed its reference.
    pub fn return_evicted_block(&mut self, block_idx: u32) {
        debug_assert!((block_idx as usize) < self.num_blocks);
        self.block_ref_counts[block_idx as usize] = 0;
        self.free_blocks.push(block_idx);
    }

    /// Current reference count for a block.
    pub fn ref_count(&self, block_idx: u32) -> u32 {
        self.block_ref_counts[block_idx as usize]
    }

    /// Number of free blocks.
    pub fn num_free_blocks(&self) -> usize {
        self.free_blocks.len()
    }

    /// Get K cache pointer for a layer and block.
    pub fn k_cache_ptr(&self, layer_idx: usize, block_idx: u32) -> DevicePtr {
        let layer = &self.layers[layer_idx];
        layer
            .k_pool
            .offset(block_idx as usize * layer.k_block_stride)
    }

    /// Get V cache pointer for a layer and block.
    pub fn v_cache_ptr(&self, layer_idx: usize, block_idx: u32) -> DevicePtr {
        let layer = &self.layers[layer_idx];
        layer
            .v_pool
            .offset(block_idx as usize * layer.v_block_stride)
    }

    /// Get the full K cache pool pointer for a layer (for paged decode kernel).
    pub fn k_pool_ptr(&self, layer_idx: usize) -> DevicePtr {
        self.layers[layer_idx].k_pool
    }

    /// Get the full V cache pool pointer for a layer.
    pub fn v_pool_ptr(&self, layer_idx: usize) -> DevicePtr {
        self.layers[layer_idx].v_pool
    }

    /// Cache stride in elements (for FP8/BF16 kernels that need explicit stride).
    /// Same for all layers (element count is dtype-independent).
    pub fn cache_stride(&self) -> usize {
        self.config.cache_stride_elements()
    }

    /// Block stride in bytes (for NVFP4 kernels), using the uniform dtype.
    pub fn block_stride_bytes(&self) -> usize {
        self.config.block_bytes()
    }

    /// Block stride in bytes for a specific attention layer.
    /// For symmetric dtypes returns the K stride (which equals V).
    /// For asymmetric dtypes returns the K-side stride; use
    /// `v_block_stride_bytes_for_layer` for the V-side stride explicitly.
    pub fn block_stride_bytes_for_layer(&self, layer_idx: usize) -> usize {
        self.layers[layer_idx].k_block_stride
    }

    /// K-side block stride in bytes for a specific attention layer.
    /// Same as `block_stride_bytes_for_layer`; named for clarity in asym call sites.
    pub fn k_block_stride_bytes_for_layer(&self, layer_idx: usize) -> usize {
        self.layers[layer_idx].k_block_stride
    }

    /// V-side block stride in bytes for a specific attention layer.
    /// Differs from K-side only for asymmetric KV cache dtypes.
    pub fn v_block_stride_bytes_for_layer(&self, layer_idx: usize) -> usize {
        self.layers[layer_idx].v_block_stride
    }

    /// NVFP4 data section size in bytes per block (uniform).
    pub fn nvfp4_data_bytes(&self) -> usize {
        self.config.nvfp4_data_bytes()
    }

    /// Turbo4 data section bytes (same layout as NVFP4: 4-bit packed).
    pub fn turbo4_data_bytes(&self) -> usize {
        self.config.turbo4_data_bytes()
    }

    /// Turbo3 data section bytes (3-bit packed).
    pub fn turbo3_data_bytes(&self) -> usize {
        self.config.turbo3_data_bytes()
    }

    /// Turbo2 data section bytes (2-bit packed).
    pub fn turbo2_data_bytes(&self) -> usize {
        self.config.turbo2_data_bytes()
    }

    /// Turbo8 data section bytes (FP8 E4M3 per element).
    pub fn turbo8_data_bytes(&self) -> usize {
        self.config.turbo8_data_bytes()
    }

    /// Turbo4 scale section bytes (same layout as NVFP4: FP8 per-group).
    pub fn turbo4_scale_bytes(&self) -> usize {
        self.config.turbo4_scale_bytes()
    }

    /// Cache configuration (read-only). Used by attention layers to query
    /// the `--high-speed-swap` HBM-shrink cap (`cache_blocks_per_seq`).
    pub fn config(&self) -> &KvCacheConfig {
        &self.config
    }

    /// Effective KV cache dtype for a specific attention layer.
    pub fn dtype_for_layer(&self, layer_idx: usize) -> KvCacheDtype {
        self.layers[layer_idx].dtype
    }

    pub fn block_size(&self) -> usize {
        self.config.block_size
    }

    pub fn num_blocks(&self) -> usize {
        self.num_blocks
    }

    /// Process-unique identity of this exact paged-KV cache instance.
    ///
    /// Target-side auxiliary caches use this to avoid aliasing a native-MTP
    /// cache (or a later model) that happens to use the same layer index.
    pub fn cache_id(&self) -> u64 {
        self.cache_id
    }

    pub fn dtype(&self) -> KvCacheDtype {
        self.config.dtype
    }

    /// Number of attention layers.
    pub fn num_layers(&self) -> usize {
        self.config.num_layers
    }

    /// Read K and V data for one block at one layer from GPU to host.
    ///
    /// Returns `(k_data, v_data)` sized to each side's block stride
    /// (which may differ for asymmetric dtypes).
    pub fn read_block(
        &self,
        layer_idx: usize,
        block_idx: u32,
        gpu: &dyn GpuBackend,
    ) -> Result<(Vec<u8>, Vec<u8>)> {
        let k_stride = self.layers[layer_idx].k_block_stride;
        let v_stride = self.layers[layer_idx].v_block_stride;
        let k_ptr = self.k_cache_ptr(layer_idx, block_idx);
        let v_ptr = self.v_cache_ptr(layer_idx, block_idx);

        let mut k_data = vec![0u8; k_stride];
        let mut v_data = vec![0u8; v_stride];
        gpu.copy_d2h(k_ptr, &mut k_data)?;
        gpu.copy_d2h(v_ptr, &mut v_data)?;

        Ok((k_data, v_data))
    }

    /// Write K and V data for one block at one layer from host to GPU.
    pub fn write_block(
        &self,
        layer_idx: usize,
        block_idx: u32,
        k_data: &[u8],
        v_data: &[u8],
        gpu: &dyn GpuBackend,
    ) -> Result<()> {
        let k_ptr = self.k_cache_ptr(layer_idx, block_idx);
        let v_ptr = self.v_cache_ptr(layer_idx, block_idx);
        gpu.copy_h2d(k_data, k_ptr)?;
        gpu.copy_h2d(v_data, v_ptr)?;
        Ok(())
    }

    /// Compute how many blocks can fit given available GPU memory.
    /// Accounts for mixed dtypes when layer_dtypes is set.
    pub fn compute_num_blocks(config: &KvCacheConfig, available_bytes: usize) -> Result<usize> {
        let bytes_per_block = config.block_bytes_kv_all_layers();
        if bytes_per_block == 0 {
            bail!("KV cache block size is zero");
        }
        Ok(available_bytes / bytes_per_block)
    }
}
