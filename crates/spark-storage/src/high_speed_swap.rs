// SPDX-License-Identifier: AGPL-3.0-only
//
// `HighSpeedSwap` orchestrator: combines Predictor + ScratchPool +
// IoUringBackend + TiledAttention + EvictionPolicy behind a two-method API
// (`offload_block`, `attend_layer`). Designed to be the primitive a future
// scheduler integration in `spark-model` plugs into.

use anyhow::{Context, Result};

use crate::backend::IoUringBackend;
use crate::config::HighSpeedSwapConfig;
use crate::cuda_min::{CudaCtx, DeviceBuffer};
use crate::eviction::EvictionPolicy;
use crate::group::GroupLayout;
use crate::layout::Layout;
use crate::predictor::{Predictor, PredictorDims};
use crate::scratch_pool::{ScratchDims, ScratchPool};
use crate::tiled_attention::{TiledAttention, TiledAttentionDims};

// `ModelDims` lives in `crate::model_dims` so it stays available on
// non-cuda builds where the swap orchestrator below isn't compiled.
pub use crate::model_dims::ModelDims;

pub struct HighSpeedSwap {
    cfg: HighSpeedSwapConfig,
    model: ModelDims,
    predictor: Predictor,
    // Must drop before `pool`: it may own pending H2D transfers targeting
    // scratch-pool device allocations and drains them in its Drop.
    backend: IoUringBackend,
    pool: ScratchPool,
    attn: TiledAttention,
    eviction: EvictionPolicy,
    // Reusable scratch buffers.
    q_proj: DeviceBuffer,
    block_scores_dev: DeviceBuffer, // [max_blocks] f32
    block_table_dev: DeviceBuffer,  // [tile_capacity] i32
    counts_dev: DeviceBuffer,       // [1] i32 (single seq)
    score_host_buf: Vec<f32>,
    /// Tracks which `(layer, block)` slots have been populated by `project_kv_block`
    /// (`layer * max_blocks_per_layer + block`). Unpredicted blocks (e.g. offloaded
    /// via `offload_block_no_predict_on_stream`) keep score `0.0` (LRU-only).
    has_projection: Vec<bool>,
    // Disk-block-ID allocator (Phase 6.1.a, refactored). One global
    // allocator: a `disk_block_id` indexes the SAME logical position
    // across every layer's file, so allocation, refcount, and free list
    // are layer-agnostic. Each layer's file independently stores its
    // K/V at `offset(layer, disk_block_id)`.
    disk_state: DiskState,
}

#[derive(Debug)]
struct DiskState {
    next_id: u32,
    free_list: Vec<u32>,
    refcount: Vec<u32>,
}

impl DiskState {
    fn new() -> Self {
        Self {
            next_id: 0,
            free_list: Vec::new(),
            refcount: Vec::new(),
        }
    }
}

impl HighSpeedSwap {
    pub fn new(ctx: &CudaCtx, cfg: HighSpeedSwapConfig, model: ModelDims) -> Result<Self> {
        Self::new_on_stream(ctx.stream, cfg, model)
    }

    /// Stream-only constructor for production callers that already own a
    /// CUDA context (spark-model). The provided `stream` is used only for
    /// init-time copies (uploading the projection matrix P); subsequent
    /// per-step calls take their own stream argument.
    pub fn new_on_stream(stream: u64, cfg: HighSpeedSwapConfig, model: ModelDims) -> Result<Self> {
        cfg.validate_and_prepare()?;
        let group_layout = GroupLayout::new(
            model.num_layers,
            model.max_blocks_per_layer,
            model.num_kv_heads,
            model.block_size as u32,
            model.head_dim as u32,
            2, // BF16
            4096,
        );
        let required_bytes = group_layout.checked_total_bytes().ok_or_else(|| {
            anyhow::anyhow!(
                "high-speed-swap disk layout byte count overflowed u64 \
                 (num_layers={}, max_blocks_per_layer={}, num_kv_heads={}, group_stride={})",
                group_layout.num_layers,
                group_layout.num_blocks,
                group_layout.num_kv_heads,
                group_layout.group_stride,
            )
        })?;
        if required_bytes > cfg.bytes {
            anyhow::bail!(
                "--high-speed-swap-bytes ({} bytes, {:.2} GiB) cannot fit layout: \
                 requires {} bytes ({:.2} GiB) for {} layers × {} bytes/layer \
                 (max_blocks_per_layer={}, num_kv_heads={}, group_stride={}). \
                 Increase --high-speed-swap-bytes or reduce --max-seq-len.",
                cfg.bytes,
                cfg.bytes as f64 / (1u64 << 30) as f64,
                required_bytes,
                required_bytes as f64 / (1u64 << 30) as f64,
                group_layout.num_layers,
                group_layout.bytes_per_layer(),
                group_layout.num_blocks,
                group_layout.num_kv_heads,
                group_layout.group_stride,
            );
        }
        if cfg.graph {
            tracing::debug!(
                "HighSpeedSwap: --high-speed-swap-graph is enabled in config, but \
                 HighSpeedSwap currently executes tiled attention eagerly (standalone \
                 CUDA graph capture is available via `spark_storage::cuda_graph::CapturedStep`)"
            );
        }
        let layout = Layout::create(&cfg.dir, group_layout).context("create layout")?;
        let backend = IoUringBackend::new(layout, cfg.qd as usize)?;
        let pool = ScratchPool::new(ScratchDims {
            num_slots: cfg.resident_blocks,
            num_kv_heads: model.num_kv_heads,
            group_stride: group_layout.group_stride,
        })?;
        let predictor = Predictor::new_on_stream(
            stream,
            PredictorDims {
                num_layers: model.num_layers as usize,
                num_q_heads: model.num_q_heads as usize,
                num_kv_heads: model.num_kv_heads as usize,
                head_dim: model.head_dim as usize,
                r: cfg.rank as usize,
                block_size: model.block_size as usize,
                max_blocks: model.max_blocks_per_layer as usize,
            },
            cfg.projection_seed,
        )?;
        let attn = TiledAttention::new(TiledAttentionDims {
            max_seqs: 1, // single-seq for the orchestrator's first iteration
            num_q_heads: model.num_q_heads as usize,
            num_kv_heads: model.num_kv_heads as usize,
            head_dim: model.head_dim as usize,
            block_size: model.block_size as usize,
            tile_capacity: cfg.resident_blocks as usize,
        })?;
        let eviction = EvictionPolicy::new(cfg.resident_blocks);
        let q_proj = DeviceBuffer::new(model.num_q_heads as usize * cfg.rank as usize * 2)?;
        let block_scores_dev = DeviceBuffer::new(model.max_blocks_per_layer as usize * 4)?;
        let block_table_dev = DeviceBuffer::new(cfg.resident_blocks as usize * 4)?;
        let counts_dev = DeviceBuffer::new(4)?;
        let score_host_buf = vec![0.0_f32; model.max_blocks_per_layer as usize];
        let total_block_slots = (model.num_layers as usize)
            .checked_mul(model.max_blocks_per_layer as usize)
            .ok_or_else(|| anyhow::anyhow!("num_layers * max_blocks_per_layer overflow"))?;
        let has_projection = vec![false; total_block_slots];
        let disk_state = DiskState::new();
        Ok(Self {
            cfg,
            model,
            predictor,
            backend,
            pool,
            attn,
            eviction,
            q_proj,
            block_scores_dev,
            block_table_dev,
            counts_dev,
            score_host_buf,
            has_projection,
            disk_state,
        })
    }

    // ── Disk-block-ID allocator (Phase 6.1.a) ─────────────────────────
    // Global ID space shared across all layers. Capacity == max_blocks_per_layer.
    // alloc / free list / refcount semantics:
    //   - alloc_disk_block_id() -> Some(id) if room, else None
    //   - inc_disk_ref(id) increments (panics if id is unallocated)
    //   - dec_disk_ref(id) -> new refcount; on 0 returns id to free list

    pub fn alloc_disk_block_id(&mut self) -> Option<u32> {
        let st = &mut self.disk_state;
        if let Some(id) = st.free_list.pop() {
            st.refcount[id as usize] = 1;
            return Some(id);
        }
        if st.next_id >= self.model.max_blocks_per_layer {
            return None; // capacity exhausted
        }
        let id = st.next_id;
        st.next_id += 1;
        st.refcount.push(1);
        Some(id)
    }

    pub fn inc_disk_ref(&mut self, id: u32) {
        let rc = &mut self.disk_state.refcount[id as usize];
        if *rc == 0 {
            panic!("inc_disk_ref on freed disk_block_id {id}; caller must hold a live ref");
        }
        *rc += 1;
    }

    pub fn dec_disk_ref(&mut self, id: u32) -> u32 {
        let st = &mut self.disk_state;
        let rc = &mut st.refcount[id as usize];
        assert!(
            *rc > 0,
            "dec_disk_ref on already-freed disk_block_id {id}; caller must hold a live ref"
        );
        *rc -= 1;
        let new_rc = *rc;
        if new_rc == 0 {
            st.free_list.push(id);
            let max_blocks = self.model.max_blocks_per_layer as usize;
            let block_idx = id as usize;
            if block_idx < max_blocks {
                for layer in 0..self.model.num_layers as usize {
                    self.has_projection[layer * max_blocks + block_idx] = false;
                }
            }
        }
        new_rc
    }

    pub fn disk_refcount(&self, id: u32) -> u32 {
        self.disk_state.refcount[id as usize]
    }

    pub fn disk_free_count(&self) -> usize {
        let st = &self.disk_state;
        st.free_list.len() + (self.model.max_blocks_per_layer - st.next_id) as usize
    }

    /// Aggregated diagnostic summary across all layers (Phase 6.1.j).
    /// Use to log periodic state during long-running decode loops; the
    /// scheduler can call this once per N steps to verify HBM-shrink
    /// behavior is on track.
    pub fn diagnostic_summary(&self) -> HighSpeedSwapDiagnostic {
        let st = &self.disk_state;
        let active = st.next_id.saturating_sub(st.free_list.len() as u32);
        HighSpeedSwapDiagnostic {
            num_layers: self.model.num_layers,
            active_disk_blocks: active,
            disk_block_capacity: self.model.max_blocks_per_layer,
            scratch_pool_resident: self.pool.dims().num_slots,
            scratch_pool_free: self.pool.free_count(),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct HighSpeedSwapDiagnostic {
    pub num_layers: u32,
    pub active_disk_blocks: u32,
    pub disk_block_capacity: u32,
    pub scratch_pool_resident: u32,
    pub scratch_pool_free: u32,
}

#[cfg(test)]
mod disk_id_tests;

mod impl_more;

// ── Thread-local installation for production callers (spark-model) ──
//
// The scheduler thread, after `bind_gpu_to_thread`, calls `install_local`
// to register the orchestrator. Per-layer attention code in spark-model
// then accesses it via `with_local`. The orchestrator's HBM allocations
// live as long as the thread; cleanup happens on thread exit (or
// explicit drop via `take_local`).

use std::cell::RefCell;
thread_local! {
    static LOCAL: RefCell<Option<HighSpeedSwap>> = const { RefCell::new(None) };
}

/// Install the orchestrator on the current thread. Idempotent (overwrites
/// any prior installation, dropping it).
pub fn install_local(stream: u64, cfg: HighSpeedSwapConfig, model: ModelDims) -> Result<()> {
    let hss = HighSpeedSwap::new_on_stream(stream, cfg, model)?;
    LOCAL.with(|cell| {
        *cell.borrow_mut() = Some(hss);
    });
    Ok(())
}

/// True iff `install_local` has populated this thread's slot.
pub fn local_installed() -> bool {
    LOCAL.with(|cell| cell.borrow().is_some())
}

/// Run `f` with a `&mut HighSpeedSwap` if installed; returns `None` if not.
pub fn with_local<R>(f: impl FnOnce(&mut HighSpeedSwap) -> Result<R>) -> Option<Result<R>> {
    LOCAL.with(|cell| cell.borrow_mut().as_mut().map(f))
}
