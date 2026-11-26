// SPDX-License-Identifier: AGPL-3.0-only

//! Device metadata, ring/bundle buffers, and CUDA graph cache state for DFlash noise passes.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use anyhow::{Context, Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend, GraphHandle, KernelHandle, PinnedHostBuffer};

use super::super::{DflashLayerQuantWeights, DflashProposerState, DflashScratch};
use super::policy::{
    DFLASH_SPLIT_KV_H128_BATCHED_KERNEL, DFLASH_SPLIT_KV_H128_LINEAR_KERNEL,
    DFLASH_SPLIT_KV_H128_REDUCE_KERNEL, NoiseGraphRuntimeKey, STAGE1_GRAPH_MAX_ROWS,
    Stage2CtxAttnMode,
};

/// Environment selector enabling the 2-stage CUDA-graphed DFlash2 noise-block
/// path (`Stage 1`: eager incremental post-`k_norm`/post-RoPE KV append + 80 B
/// metadata H2D; `Stage 2`: fixed-shape `m_rows` graphed noise pass using
/// `inferspark_prefill_split_ctx_h128` / `inferspark_prefill_split_ctx_h256`
/// and `enqueue_markov_sequential_dev` / `dflash2_selector_walk_batched`).
pub const DFLASH_NOISE_GRAPH_SELECTOR: &str = "ATLAS_DFLASH_NOISE_GRAPH";

pub(in super::super) const MAX_CACHED_NOISE_GRAPHS: usize = 32;
pub(in super::super) const NOISE_META_PINNED_SLOTS: usize = 4;
pub(in super::super) const GUMBEL_META_C1_SLOT_BYTES: usize = 40;
pub(in super::super) const GUMBEL_META_SEED_OFF: usize = 0;
pub(in super::super) const GUMBEL_META_STEP0_OFF: usize = 8;
pub(in super::super) const GUMBEL_META_TEMP_OFF: usize = 16;
pub(in super::super) const GUMBEL_META_TOPK_OFF: usize = 20;
pub(in super::super) const GUMBEL_META_TOPP_OFF: usize = 24;
pub(in super::super) const GUMBEL_META_MINP_OFF: usize = 28;
pub(in super::super) const GUMBEL_META_ANCHOR_OFF: usize = 32;
pub(in super::super) const GUMBEL_META_Q_ROW_BASE_OFF: usize = 36;

/// Canonical per-sequence scalar tuple `(seed, step0_abs_pos, temperature, top_k, top_p, min_p)`
/// uploaded to `dflash2_selector_walk_gumbel_batched`. Greedy / inactive specs emit
/// `temperature = 0.0` so the kernel takes its exact greedy argmax branch.
pub(in super::super) fn gumbel_spec_fields(
    spec: Option<spark_runtime::sampler::gumbel::GumbelDraftSpec>,
) -> (u64, u64, f32, u32, f32, f32) {
    match spec {
        Some(s) if s.is_active() => (
            s.seed,
            s.step0_abs_pos,
            s.temperature,
            s.top_k,
            s.top_p,
            s.min_p,
        ),
        _ => (0u64, 0u64, 0.0f32, 0u32, 1.0f32, 0.0f32),
    }
}

#[derive(Clone, Copy)]
pub(in super::super) struct GumbelScratchPtrs {
    pub(in super::super) seeds: DevicePtr,
    pub(in super::super) step0_pos: DevicePtr,
    pub(in super::super) temps: DevicePtr,
    pub(in super::super) top_ks: DevicePtr,
    pub(in super::super) top_ps: DevicePtr,
    pub(in super::super) min_ps: DevicePtr,
}

impl GumbelScratchPtrs {
    pub(in super::super) fn from_scratch(scratch: &DflashScratch) -> Self {
        Self {
            seeds: scratch.batch_gumbel_seeds,
            step0_pos: scratch.batch_gumbel_step0_pos,
            temps: scratch.batch_gumbel_temps,
            top_ks: scratch.batch_gumbel_top_ks,
            top_ps: scratch.batch_gumbel_top_ps,
            min_ps: scratch.batch_gumbel_min_ps,
        }
    }
}

/// Fixed-layout 80-byte device metadata uploaded once per draft step in
/// Stage 1 of the 2-stage CUDA-graphed DFlash2 noise pass.
///
/// Stage 2 kernels read `eff_ctx` (`+0`), `last_token` (`+4`), and
/// `noise_positions[0..noise_len]` (`+16..+80`) directly from device memory
/// so the captured CUDA graph contains zero per-step host scalars.
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DflashNoiseGraphMeta {
    pub eff_ctx: u32,
    pub last_token: u32,
    pub noise_len: u32,
    pub _pad: u32,
    pub noise_positions: [u32; 16],
}

impl DflashNoiseGraphMeta {
    pub const SIZE_BYTES: usize = 80;
    pub const MAX_NOISE_POSITIONS: usize = 16;
    pub const EFF_CTX_OFFSET: usize = 0;
    pub const LAST_TOKEN_OFFSET: usize = 4;
    #[cfg(test)]
    pub const NOISE_LEN_OFFSET: usize = 8;
    pub const NOISE_POSITIONS_OFFSET: usize = 16;

    /// Validate `eff_ctx` and `positions` bounds and build a zero-padded
    /// 80-byte metadata block ready for little-endian H2D serialization.
    pub fn try_new(
        eff_ctx: usize,
        last_token: u32,
        positions: &[u32],
        max_ctx_window: usize,
    ) -> Result<Self> {
        ensure!(
            eff_ctx <= max_ctx_window,
            "DFlash noise graph eff_ctx={eff_ctx} exceeds ctx_window={max_ctx_window}"
        );
        let eff_ctx_u32 =
            u32::try_from(eff_ctx).context("DFlash noise graph eff_ctx exceeds u32")?;
        ensure!(
            !positions.is_empty() && positions.len() <= Self::MAX_NOISE_POSITIONS,
            "DFlash noise graph noise_len={} outside 1..={}",
            positions.len(),
            Self::MAX_NOISE_POSITIONS
        );
        ensure!(
            positions[0] >= eff_ctx_u32,
            "DFlash noise graph first noise position {} precedes eff_ctx={eff_ctx_u32}",
            positions[0]
        );
        for pair in positions.windows(2) {
            ensure!(
                pair[1] > pair[0],
                "DFlash noise graph noise_positions must be strictly increasing, got {} then {}",
                pair[0],
                pair[1]
            );
        }
        let mut noise_positions = [0u32; Self::MAX_NOISE_POSITIONS];
        noise_positions[..positions.len()].copy_from_slice(positions);
        Ok(Self {
            eff_ctx: eff_ctx_u32,
            last_token,
            noise_len: positions.len() as u32,
            _pad: 0,
            noise_positions,
        })
    }

    /// Serialize into its exact 80-byte little-endian device wire layout.
    pub fn to_le_bytes(self) -> [u8; Self::SIZE_BYTES] {
        let mut out = [0u8; Self::SIZE_BYTES];
        out[0..4].copy_from_slice(&self.eff_ctx.to_le_bytes());
        out[4..8].copy_from_slice(&self.last_token.to_le_bytes());
        out[8..12].copy_from_slice(&self.noise_len.to_le_bytes());
        out[12..16].copy_from_slice(&self._pad.to_le_bytes());
        for (i, pos) in self.noise_positions.iter().enumerate() {
            let off = Self::NOISE_POSITIONS_OFFSET + i * 4;
            out[off..off + 4].copy_from_slice(&pos.to_le_bytes());
        }
        out
    }

    /// Write the 80-byte little-endian layout into `dst[..80]`.
    pub fn write_le_bytes(&self, dst: &mut [u8]) -> Result<usize> {
        ensure!(
            dst.len() >= Self::SIZE_BYTES,
            "DFlashNoiseGraphMeta staging slice too small: {} < {}",
            dst.len(),
            Self::SIZE_BYTES
        );
        let encoded = self.to_le_bytes();
        dst[..Self::SIZE_BYTES].copy_from_slice(&encoded);
        Ok(Self::SIZE_BYTES)
    }

    /// Decode an 80-byte little-endian buffer back into [`DflashNoiseGraphMeta`].
    #[cfg(test)]
    pub fn from_le_bytes(src: &[u8]) -> Result<Self> {
        ensure!(
            src.len() == Self::SIZE_BYTES,
            "DFlashNoiseGraphMeta requires exactly {} bytes, got {}",
            Self::SIZE_BYTES,
            src.len()
        );
        let read_u32 = |off: usize| -> u32 {
            u32::from_le_bytes([src[off], src[off + 1], src[off + 2], src[off + 3]])
        };
        let mut noise_positions = [0u32; Self::MAX_NOISE_POSITIONS];
        for (i, slot) in noise_positions.iter_mut().enumerate() {
            *slot = read_u32(Self::NOISE_POSITIONS_OFFSET + i * 4);
        }
        Ok(Self {
            eff_ctx: read_u32(0),
            last_token: read_u32(4),
            noise_len: read_u32(8),
            _pad: read_u32(12),
            noise_positions,
        })
    }
}

/// Precision class of the drafter transformer layers for graph-key separation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DflashLayerQuantMode {
    Bf16,
    Fp8,
    Nvfp4,
    Mixed,
}

impl DflashLayerQuantMode {
    pub(in super::super) fn from_layers(layers: &[DflashLayerQuantWeights]) -> Self {
        let mut saw_bf16 = false;
        let mut saw_fp8 = false;
        let mut saw_nvfp4 = false;
        for layer in layers {
            match layer {
                DflashLayerQuantWeights::Bf16(_) => saw_bf16 = true,
                DflashLayerQuantWeights::Fp8(_) => saw_fp8 = true,
                DflashLayerQuantWeights::Nvfp4(_) => saw_nvfp4 = true,
            }
        }
        match (saw_bf16, saw_fp8, saw_nvfp4) {
            (true, false, false) => Self::Bf16,
            (false, true, false) => Self::Fp8,
            (false, false, true) => Self::Nvfp4,
            _ => Self::Mixed,
        }
    }
}

/// Fixed byte offsets for the C=1 `CircularRingH128` Stage-2 device metadata
/// slab (`[ring_start: u32, _pad: u32, ctx_k_ptrs: [u64; L], ctx_v_ptrs: [u64; L]]`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in super::super) struct RingMetaLayout {
    pub(in super::super) num_layers: usize,
    pub(in super::super) ring_start_off: usize,
    pub(in super::super) ctx_k_ptrs_off: usize,
    pub(in super::super) ctx_v_ptrs_off: usize,
    pub(in super::super) total_bytes: usize,
}

impl RingMetaLayout {
    pub(in super::super) fn new(num_layers: usize) -> Result<Self> {
        ensure!(
            num_layers > 0,
            "DFlash C=1 ring metadata requires num_layers > 0"
        );
        let ptr_bytes = num_layers
            .checked_mul(std::mem::size_of::<u64>())
            .context("DFlash C=1 ring metadata pointer bytes overflow")?;
        let ring_start_off = 0usize;
        // Keep u64 pointer arrays 8-byte aligned right after ring_start (4B) + pad (4B).
        let ctx_k_ptrs_off = 8usize;
        let ctx_v_ptrs_off = ctx_k_ptrs_off
            .checked_add(ptr_bytes)
            .context("DFlash C=1 ring metadata V pointer offset overflow")?;
        let total_bytes = ctx_v_ptrs_off
            .checked_add(ptr_bytes)
            .context("DFlash C=1 ring metadata total bytes overflow")?;
        Ok(Self {
            num_layers,
            ring_start_off,
            ctx_k_ptrs_off,
            ctx_v_ptrs_off,
            total_bytes,
        })
    }

    #[inline]
    pub(in super::super) fn bind_device(&self, base: DevicePtr) -> RingMetaDevicePtrs {
        RingMetaDevicePtrs {
            base,
            ring_start_dev: base.offset(self.ring_start_off),
            ctx_k_ptrs_base: base.offset(self.ctx_k_ptrs_off),
            ctx_v_ptrs_base: base.offset(self.ctx_v_ptrs_off),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in super::super) struct RingMetaDevicePtrs {
    pub(in super::super) base: DevicePtr,
    pub(in super::super) ring_start_dev: DevicePtr,
    pub(in super::super) ctx_k_ptrs_base: DevicePtr,
    pub(in super::super) ctx_v_ptrs_base: DevicePtr,
}

impl RingMetaDevicePtrs {
    #[inline]
    pub(in super::super) fn layer_k_ptr(&self, layer_idx: usize) -> DevicePtr {
        self.ctx_k_ptrs_base
            .offset(layer_idx * std::mem::size_of::<u64>())
    }

    #[inline]
    pub(in super::super) fn layer_v_ptr(&self, layer_idx: usize) -> DevicePtr {
        self.ctx_v_ptrs_base
            .offset(layer_idx * std::mem::size_of::<u64>())
    }
}

/// Contiguous C=1 Stage-2 metadata bundle layout (`216 B` for 5 layers) collapsing
/// `DflashNoiseGraphMeta` (80 B) + 6 Gumbel scalars + `q_row_base` + `RingMeta`
/// from 9 separate H2D calls into 1 single pinned H2D per round:
/// - `[0..80)`: `DflashNoiseGraphMeta`
/// - `[80..128)`: Gumbel scalars (`seeds:8, step0_pos:8, temps:4, top_ks:4, top_ps:4, min_ps:4, q_row_base:4`, padded to 16-byte alignment)
/// - `[128..128+ring_bytes)`: `RingMetaLayout` (`ring_start` + layer K/V pointers)
pub(in super::super) const C1_BUNDLE_META_OFF: usize = 0;
pub(in super::super) const C1_BUNDLE_GUMBEL_OFF: usize = 80;
pub(in super::super) const C1_BUNDLE_RING_OFF: usize = 128;

const _: () = {
    assert!(DflashNoiseGraphMeta::SIZE_BYTES == C1_BUNDLE_GUMBEL_OFF);
    assert!(C1_BUNDLE_GUMBEL_OFF.is_multiple_of(16));
    assert!(C1_BUNDLE_RING_OFF.is_multiple_of(16));
    assert!(C1_BUNDLE_GUMBEL_OFF + 36 <= C1_BUNDLE_RING_OFF);
};

#[derive(Clone, Copy)]
pub(in super::super) struct C1Stage2BundlePtrs {
    pub(in super::super) meta_dev: DevicePtr,
    pub(in super::super) mask_dev: DevicePtr,
    pub(in super::super) gumbel_ptrs: GumbelScratchPtrs,
    pub(in super::super) q_row_base_dev: DevicePtr,
    pub(in super::super) ring_meta_ptrs: Option<RingMetaDevicePtrs>,
}

pub(in super::super) struct NoiseGraphHeadState {
    pub(in super::super) split_ctx_kernel: Option<KernelHandle>,
    pub(in super::super) split_ctx_splitkv_kernel: Option<KernelHandle>,
    pub(in super::super) split_ctx_batched_splitkv_kernel: Option<KernelHandle>,
    pub(in super::super) split_ctx_splitkv_reduce_kernel: Option<KernelHandle>,
    pub(in super::super) split_kv_ws_dev: DevicePtr,
    pub(in super::super) split_kv_ws_cap: usize,
    pub(in super::super) meta_dev: DevicePtr,
    pub(in super::super) meta_pinned: Option<PinnedHostBuffer>,
    pub(in super::super) meta_slot: usize,
    #[cfg(test)]
    pub(in super::super) ring_meta_dev: DevicePtr,
    #[cfg(test)]
    pub(in super::super) ring_meta_pinned: Option<PinnedHostBuffer>,
    #[cfg(test)]
    pub(in super::super) ring_meta_slot: usize,
    #[cfg(test)]
    pub(in super::super) ring_meta_layers: usize,
    #[cfg(test)]
    pub(in super::super) ring_meta_slot_bytes: usize,
    /// Coalesced C=1 Stage-2 metadata bundle (`meta + gumbel + q_row_base + ring_meta`).
    pub(in super::super) c1_bundle_dev: DevicePtr,
    pub(in super::super) c1_bundle_pinned: Option<PinnedHostBuffer>,
    pub(in super::super) c1_bundle_slot: usize,
    pub(in super::super) c1_bundle_layers: usize,
    pub(in super::super) c1_bundle_slot_bytes: usize,
    pub(in super::super) gumbel_pinned: Option<PinnedHostBuffer>,
    pub(in super::super) gumbel_slot: usize,
    pub(in super::super) mask_tokens_dev: DevicePtr,
    pub(in super::super) mask_tokens_pinned: Option<PinnedHostBuffer>,
    pub(in super::super) uploaded_mask_id: Option<u32>,
    /// Fixed staging buffers + bucketed CUDA graphs (`new_ctx_count in 1..=8`)
    /// for Stage 1 (`5x {k_proj, v_proj, k_norm, rope}`).
    pub(in super::super) stage1_fc_dev: DevicePtr,
    pub(in super::super) stage1_pos_dev: DevicePtr,
    pub(in super::super) stage1_pos_pinned: Option<PinnedHostBuffer>,
    pub(in super::super) stage1_pos_slot: usize,
    pub(in super::super) stage1_k_dev: DevicePtr,
    pub(in super::super) stage1_v_dev: DevicePtr,
    pub(in super::super) stage1_cfg: Option<(usize, usize, usize)>,
    pub(in super::super) stage1_graphs: HashMap<(usize, u64), GraphHandle>,
    pub(in super::super) stage1_capture_poisoned: bool,
    pub(in super::super) graphs: HashMap<NoiseGraphRuntimeKey, GraphHandle>,
    pub(in super::super) graph_order: Vec<NoiseGraphRuntimeKey>,
}

impl NoiseGraphHeadState {
    pub(in super::super) fn new() -> Self {
        Self {
            split_ctx_kernel: None,
            split_ctx_splitkv_kernel: None,
            split_ctx_batched_splitkv_kernel: None,
            split_ctx_splitkv_reduce_kernel: None,
            split_kv_ws_dev: DevicePtr::NULL,
            split_kv_ws_cap: 0,
            meta_dev: DevicePtr::NULL,
            meta_pinned: None,
            meta_slot: 0,
            #[cfg(test)]
            ring_meta_dev: DevicePtr::NULL,
            #[cfg(test)]
            ring_meta_pinned: None,
            #[cfg(test)]
            ring_meta_slot: 0,
            #[cfg(test)]
            ring_meta_layers: 0,
            #[cfg(test)]
            ring_meta_slot_bytes: 0,
            c1_bundle_dev: DevicePtr::NULL,
            c1_bundle_pinned: None,
            c1_bundle_slot: 0,
            c1_bundle_layers: 0,
            c1_bundle_slot_bytes: 0,
            gumbel_pinned: None,
            gumbel_slot: 0,
            mask_tokens_dev: DevicePtr::NULL,
            mask_tokens_pinned: None,
            uploaded_mask_id: None,
            stage1_fc_dev: DevicePtr::NULL,
            stage1_pos_dev: DevicePtr::NULL,
            stage1_pos_pinned: None,
            stage1_pos_slot: 0,
            stage1_k_dev: DevicePtr::NULL,
            stage1_v_dev: DevicePtr::NULL,
            stage1_cfg: None,
            stage1_graphs: HashMap::new(),
            stage1_capture_poisoned: false,
            graphs: HashMap::new(),
            graph_order: Vec::new(),
        }
    }

    pub(in super::super) fn resolve_split_ctx_kernel(
        &mut self,
        gpu: &dyn GpuBackend,
        head_dim: usize,
    ) -> KernelHandle {
        if let Some(handle) = self.split_ctx_kernel {
            return handle;
        }
        let handle = match head_dim {
            128 => crate::layers::try_kernel(
                gpu,
                "inferspark_prefill_h128",
                "inferspark_prefill_split_ctx_h128",
            ),
            256 => crate::layers::try_kernel(
                gpu,
                "inferspark_prefill",
                "inferspark_prefill_split_ctx_h256",
            ),
            _ => KernelHandle(0),
        };
        self.split_ctx_kernel = Some(handle);
        handle
    }

    pub(in super::super) fn resolve_split_kv_h128_kernels(
        &mut self,
        gpu: &dyn GpuBackend,
        mode: Stage2CtxAttnMode,
    ) -> Option<(KernelHandle, KernelHandle)> {
        let split_k = match mode {
            Stage2CtxAttnMode::Linear => *self.split_ctx_splitkv_kernel.get_or_insert_with(|| {
                crate::layers::try_kernel(
                    gpu,
                    "inferspark_prefill_h128",
                    DFLASH_SPLIT_KV_H128_LINEAR_KERNEL,
                )
            }),
            Stage2CtxAttnMode::CircularRingH128 => *self
                .split_ctx_batched_splitkv_kernel
                .get_or_insert_with(|| {
                    crate::layers::try_kernel(
                        gpu,
                        "inferspark_prefill_h128",
                        DFLASH_SPLIT_KV_H128_BATCHED_KERNEL,
                    )
                }),
        };
        let reduce_k = *self.split_ctx_splitkv_reduce_kernel.get_or_insert_with(|| {
            crate::layers::try_kernel(
                gpu,
                "inferspark_prefill_h128",
                DFLASH_SPLIT_KV_H128_REDUCE_KERNEL,
            )
        });
        (split_k.0 != 0 && reduce_k.0 != 0).then_some((split_k, reduce_k))
    }

    pub(in super::super) fn ensure_split_kv_workspace(
        &mut self,
        gpu: &dyn GpuBackend,
        alloc_bytes: usize,
        needed_bytes: usize,
    ) -> Result<Option<DevicePtr>> {
        if needed_bytes == 0 {
            return Ok(None);
        }
        if self.split_kv_ws_dev.is_null() {
            let bytes = alloc_bytes.max(needed_bytes);
            self.split_kv_ws_dev = gpu.alloc(bytes)?;
            self.split_kv_ws_cap = bytes;
        }
        if self.split_kv_ws_cap >= needed_bytes && self.split_kv_ws_dev.0.is_multiple_of(16) {
            Ok(Some(self.split_kv_ws_dev))
        } else {
            Ok(None)
        }
    }

    pub(in super::super) fn ensure_device_buffers(
        &mut self,
        gpu: &dyn GpuBackend,
    ) -> Result<(DevicePtr, DevicePtr)> {
        if self.meta_dev.is_null() {
            self.meta_dev = gpu.alloc(DflashNoiseGraphMeta::SIZE_BYTES)?;
        }
        if self.meta_pinned.is_none() {
            self.meta_pinned = Some(
                gpu.alloc_host_pinned(DflashNoiseGraphMeta::SIZE_BYTES * NOISE_META_PINNED_SLOTS)?,
            );
        }
        if self.mask_tokens_dev.is_null() {
            self.mask_tokens_dev =
                gpu.alloc(DflashNoiseGraphMeta::MAX_NOISE_POSITIONS * std::mem::size_of::<u32>())?;
        }
        if self.mask_tokens_pinned.is_none() {
            self.mask_tokens_pinned = Some(gpu.alloc_host_pinned(
                DflashNoiseGraphMeta::MAX_NOISE_POSITIONS * std::mem::size_of::<u32>(),
            )?);
        }
        Ok((self.meta_dev, self.mask_tokens_dev))
    }

    #[cfg(test)]
    pub(in super::super) fn ensure_ring_meta_buffers(
        &mut self,
        gpu: &dyn GpuBackend,
        num_layers: usize,
    ) -> Result<RingMetaLayout> {
        let layout = RingMetaLayout::new(num_layers)?;
        if self.ring_meta_dev.is_null() {
            self.ring_meta_dev = gpu.alloc(layout.total_bytes)?;
            self.ring_meta_layers = layout.num_layers;
            self.ring_meta_slot_bytes = layout.total_bytes;
        } else {
            ensure!(
                self.ring_meta_layers == layout.num_layers
                    && layout.total_bytes <= self.ring_meta_slot_bytes,
                "DFlash C=1 ring metadata layer count changed: expected {}, got {}",
                self.ring_meta_layers,
                layout.num_layers
            );
        }
        if self.ring_meta_pinned.is_none() {
            self.ring_meta_pinned =
                Some(gpu.alloc_host_pinned(self.ring_meta_slot_bytes * NOISE_META_PINNED_SLOTS)?);
        }
        Ok(layout)
    }

    #[cfg(test)]
    pub(in super::super) fn upload_ring_meta(
        &mut self,
        gpu: &dyn GpuBackend,
        ring_start: u32,
        ctx_k_cache: &[DevicePtr],
        ctx_v_cache: &[DevicePtr],
        stream: u64,
    ) -> Result<RingMetaDevicePtrs> {
        ensure!(
            !ctx_k_cache.is_empty() && ctx_k_cache.len() == ctx_v_cache.len(),
            "DFlash C=1 ring metadata KV cache length mismatch: k={} v={}",
            ctx_k_cache.len(),
            ctx_v_cache.len()
        );
        let layout = self.ensure_ring_meta_buffers(gpu, ctx_k_cache.len())?;
        let slot = self.ring_meta_slot;
        self.ring_meta_slot = (self.ring_meta_slot + 1) % NOISE_META_PINNED_SLOTS;
        let offset = slot * self.ring_meta_slot_bytes;
        let pinned = self
            .ring_meta_pinned
            .as_mut()
            .context("DFlash C=1 ring metadata staging buffer missing")?;
        let dst = &mut pinned.as_mut_slice()[offset..offset + layout.total_bytes];
        dst[layout.ring_start_off..layout.ring_start_off + 4]
            .copy_from_slice(&ring_start.to_le_bytes());
        dst[layout.ring_start_off + 4..layout.ctx_k_ptrs_off].copy_from_slice(&0u32.to_le_bytes());
        for (l, (&k_ptr, &v_ptr)) in ctx_k_cache.iter().zip(ctx_v_cache.iter()).enumerate() {
            let k_off = layout.ctx_k_ptrs_off + l * std::mem::size_of::<u64>();
            let v_off = layout.ctx_v_ptrs_off + l * std::mem::size_of::<u64>();
            dst[k_off..k_off + 8].copy_from_slice(&k_ptr.0.to_le_bytes());
            dst[v_off..v_off + 8].copy_from_slice(&v_ptr.0.to_le_bytes());
        }
        let slice = pinned.pinned_slice_range(offset, layout.total_bytes)?;
        // SAFETY: `ring_meta_pinned` is page-locked and owned by the static
        // per-head state; rotating across `NOISE_META_PINNED_SLOTS` disjoint
        // slots ensures no in-flight async H2D transfer is overwritten.
        unsafe {
            gpu.copy_h2d_pinned_async(slice, self.ring_meta_dev, stream)?;
        }
        Ok(layout.bind_device(self.ring_meta_dev))
    }

    #[allow(dead_code)]
    pub(in super::super) fn upload_meta_and_masks(
        &mut self,
        gpu: &dyn GpuBackend,
        meta: &DflashNoiseGraphMeta,
        mask_id: u32,
        stream: u64,
    ) -> Result<(DevicePtr, DevicePtr)> {
        let (meta_dev, mask_dev) = self.ensure_device_buffers(gpu)?;
        self.ensure_mask_tokens_uploaded(gpu, mask_dev, mask_id, stream)?;
        let slot = self.meta_slot;
        self.meta_slot = (self.meta_slot + 1) % NOISE_META_PINNED_SLOTS;
        let offset = slot * DflashNoiseGraphMeta::SIZE_BYTES;
        let meta_pinned = self
            .meta_pinned
            .as_mut()
            .context("DFlash noise graph meta staging buffer missing")?;
        meta.write_le_bytes(
            &mut meta_pinned.as_mut_slice()[offset..offset + DflashNoiseGraphMeta::SIZE_BYTES],
        )?;
        let slice = meta_pinned.pinned_slice_range(offset, DflashNoiseGraphMeta::SIZE_BYTES)?;
        // SAFETY: `meta_pinned` is page-locked and owned by the static per-head
        // state; rotating across `NOISE_META_PINNED_SLOTS` disjoint 80-byte
        // ranges ensures no in-flight async H2D transfer is overwritten.
        unsafe {
            gpu.copy_h2d_pinned_async(slice, meta_dev, stream)?;
        }
        Ok((meta_dev, mask_dev))
    }

    pub(in super::super) fn ensure_mask_tokens_uploaded(
        &mut self,
        gpu: &dyn GpuBackend,
        mask_dev: DevicePtr,
        mask_id: u32,
        stream: u64,
    ) -> Result<()> {
        if self.uploaded_mask_id != Some(mask_id) {
            if self.uploaded_mask_id.is_some() {
                // Single-slot page-locked mask buffer: drain prior async H2D
                // transfers across streams before mutating `mask_tokens_pinned`
                // in place for a new `mask_id`.
                gpu.synchronize_device()?;
            }
            let mask_pinned = self
                .mask_tokens_pinned
                .as_mut()
                .context("DFlash noise graph mask staging buffer missing")?;
            let mask_bytes = DflashNoiseGraphMeta::MAX_NOISE_POSITIONS * std::mem::size_of::<u32>();
            for chunk in mask_pinned.as_mut_slice()[..mask_bytes].chunks_exact_mut(4) {
                chunk.copy_from_slice(&mask_id.to_le_bytes());
            }
            let slice = mask_pinned.pinned_slice(mask_bytes)?;
            // SAFETY: `mask_tokens_pinned` is page-locked, owned by the static
            // per-head state, and any prior H2D copy was drained above before
            // mutating the single-slot buffer.
            unsafe {
                gpu.copy_h2d_pinned_async(slice, mask_dev, stream)?;
            }
            self.uploaded_mask_id = Some(mask_id);
        }
        Ok(())
    }

    pub(in super::super) fn ensure_c1_bundle_buffers(
        &mut self,
        gpu: &dyn GpuBackend,
        num_layers: usize,
    ) -> Result<(DevicePtr, DevicePtr, RingMetaLayout)> {
        let (_, mask_dev) = self.ensure_device_buffers(gpu)?;
        let ring_layout = RingMetaLayout::new(num_layers.max(1))?;
        let slot_bytes = C1_BUNDLE_RING_OFF
            .checked_add(ring_layout.total_bytes)
            .context("DFlash C=1 metadata bundle size overflow")?;
        if self.c1_bundle_dev.is_null() {
            self.c1_bundle_dev = gpu.alloc(slot_bytes)?;
            self.c1_bundle_layers = num_layers.max(1);
            self.c1_bundle_slot_bytes = slot_bytes;
        } else {
            ensure!(
                self.c1_bundle_layers == num_layers.max(1)
                    && slot_bytes <= self.c1_bundle_slot_bytes,
                "DFlash C=1 metadata bundle layer count changed: expected {}, got {}",
                self.c1_bundle_layers,
                num_layers.max(1)
            );
        }
        if self.c1_bundle_pinned.is_none() {
            self.c1_bundle_pinned =
                Some(gpu.alloc_host_pinned(self.c1_bundle_slot_bytes * NOISE_META_PINNED_SLOTS)?);
        }
        Ok((self.c1_bundle_dev, mask_dev, ring_layout))
    }

    /// Pack all per-round C=1 Stage-2 metadata (`DflashNoiseGraphMeta`, 6 Gumbel
    /// scalars, `q_row_base`, and optional `RingMeta` start + K/V pointers) into
    /// ONE contiguous rotating pinned slot and issue ONE single `copy_h2d_pinned_async`.
    #[allow(clippy::too_many_arguments)]
    pub(in super::super) fn upload_c1_stage2_bundle(
        &mut self,
        gpu: &dyn GpuBackend,
        meta: &DflashNoiseGraphMeta,
        mask_id: u32,
        ring_upload: Option<(u32, &[DevicePtr], &[DevicePtr])>,
        gumbel_spec: Option<spark_runtime::sampler::gumbel::GumbelDraftSpec>,
        q_row_base: u32,
        num_layers: usize,
        stream: u64,
    ) -> Result<C1Stage2BundlePtrs> {
        let (bundle_dev, mask_dev, ring_layout) = self.ensure_c1_bundle_buffers(gpu, num_layers)?;
        self.ensure_mask_tokens_uploaded(gpu, mask_dev, mask_id, stream)?;

        let slot = self.c1_bundle_slot;
        self.c1_bundle_slot = (self.c1_bundle_slot + 1) % NOISE_META_PINNED_SLOTS;
        let slot_off = slot * self.c1_bundle_slot_bytes;
        let (seed, step0_abs_pos, temp, top_k, top_p, min_p) = gumbel_spec_fields(gumbel_spec);

        let upload_bytes = if ring_upload.is_some() {
            C1_BUNDLE_RING_OFF + ring_layout.total_bytes
        } else {
            C1_BUNDLE_RING_OFF
        };
        let pinned = self
            .c1_bundle_pinned
            .as_mut()
            .context("DFlash C=1 metadata bundle pinned staging missing")?;
        let dst = &mut pinned.as_mut_slice()[slot_off..slot_off + upload_bytes];
        meta.write_le_bytes(
            &mut dst[C1_BUNDLE_META_OFF..C1_BUNDLE_META_OFF + DflashNoiseGraphMeta::SIZE_BYTES],
        )?;
        let g = &mut dst[C1_BUNDLE_GUMBEL_OFF..C1_BUNDLE_RING_OFF];
        g[0..8].copy_from_slice(&seed.to_le_bytes());
        g[8..16].copy_from_slice(&step0_abs_pos.to_le_bytes());
        g[16..20].copy_from_slice(&temp.to_le_bytes());
        g[20..24].copy_from_slice(&top_k.to_le_bytes());
        g[24..28].copy_from_slice(&top_p.to_le_bytes());
        g[28..32].copy_from_slice(&min_p.to_le_bytes());
        g[32..36].copy_from_slice(&q_row_base.to_le_bytes());
        g[36..48].fill(0);

        let ring_meta_ptrs = if let Some((ring_start, ctx_k_cache, ctx_v_cache)) = ring_upload {
            ensure!(
                !ctx_k_cache.is_empty()
                    && ctx_k_cache.len() == ctx_v_cache.len()
                    && ctx_k_cache.len() == ring_layout.num_layers,
                "DFlash C=1 bundle ring KV length mismatch: k={} v={} layout={}",
                ctx_k_cache.len(),
                ctx_v_cache.len(),
                ring_layout.num_layers
            );
            let r = &mut dst[C1_BUNDLE_RING_OFF..C1_BUNDLE_RING_OFF + ring_layout.total_bytes];
            r[ring_layout.ring_start_off..ring_layout.ring_start_off + 4]
                .copy_from_slice(&ring_start.to_le_bytes());
            r[ring_layout.ring_start_off + 4..ring_layout.ctx_k_ptrs_off]
                .copy_from_slice(&0u32.to_le_bytes());
            for (l, (&k_ptr, &v_ptr)) in ctx_k_cache.iter().zip(ctx_v_cache.iter()).enumerate() {
                let k_off = ring_layout.ctx_k_ptrs_off + l * std::mem::size_of::<u64>();
                let v_off = ring_layout.ctx_v_ptrs_off + l * std::mem::size_of::<u64>();
                r[k_off..k_off + 8].copy_from_slice(&k_ptr.0.to_le_bytes());
                r[v_off..v_off + 8].copy_from_slice(&v_ptr.0.to_le_bytes());
            }
            Some(ring_layout.bind_device(bundle_dev.offset(C1_BUNDLE_RING_OFF)))
        } else {
            None
        };

        let slice = pinned.pinned_slice_range(slot_off, upload_bytes)?;
        // SAFETY: `c1_bundle_pinned` is page-locked and owned by the static
        // per-head state; rotating across `NOISE_META_PINNED_SLOTS` disjoint
        // ranges ensures no in-flight async H2D transfer is overwritten.
        unsafe {
            gpu.copy_h2d_pinned_async(slice, bundle_dev, stream)?;
        }
        let g_base = bundle_dev.offset(C1_BUNDLE_GUMBEL_OFF);
        Ok(C1Stage2BundlePtrs {
            meta_dev: bundle_dev.offset(C1_BUNDLE_META_OFF),
            mask_dev,
            gumbel_ptrs: GumbelScratchPtrs {
                seeds: g_base,
                step0_pos: g_base.offset(8),
                temps: g_base.offset(16),
                top_ks: g_base.offset(20),
                top_ps: g_base.offset(24),
                min_ps: g_base.offset(28),
            },
            q_row_base_dev: g_base.offset(32),
            ring_meta_ptrs,
        })
    }

    pub(in super::super) fn ensure_stage1_buffers(
        &mut self,
        gpu: &dyn GpuBackend,
        num_layers: usize,
        hidden: usize,
        kv_dim: usize,
    ) -> Result<(DevicePtr, DevicePtr, DevicePtr, DevicePtr)> {
        let cfg = (num_layers, hidden, kv_dim);
        if let Some(prev) = self.stage1_cfg {
            ensure!(
                prev == cfg,
                "DFlash Stage-1 buffer geometry changed: expected {prev:?}, got {cfg:?}"
            );
            return Ok((
                self.stage1_fc_dev,
                self.stage1_pos_dev,
                self.stage1_k_dev,
                self.stage1_v_dev,
            ));
        }
        let bf16 = 2usize;
        let fc_bytes = STAGE1_GRAPH_MAX_ROWS * hidden * bf16;
        let pos_bytes = STAGE1_GRAPH_MAX_ROWS * std::mem::size_of::<u32>();
        let kv_bytes = num_layers * STAGE1_GRAPH_MAX_ROWS * kv_dim * bf16;
        let fc_dev = gpu.alloc(fc_bytes)?;
        let pos_dev = match gpu.alloc(pos_bytes) {
            Ok(p) => p,
            Err(e) => {
                let _ = gpu.free(fc_dev);
                return Err(e);
            }
        };
        let k_dev = match gpu.alloc(kv_bytes) {
            Ok(p) => p,
            Err(e) => {
                let _ = gpu.free(pos_dev);
                let _ = gpu.free(fc_dev);
                return Err(e);
            }
        };
        let v_dev = match gpu.alloc(kv_bytes) {
            Ok(p) => p,
            Err(e) => {
                let _ = gpu.free(k_dev);
                let _ = gpu.free(pos_dev);
                let _ = gpu.free(fc_dev);
                return Err(e);
            }
        };
        let pos_pinned = match gpu.alloc_host_pinned(pos_bytes * NOISE_META_PINNED_SLOTS) {
            Ok(p) => p,
            Err(e) => {
                let _ = gpu.free(v_dev);
                let _ = gpu.free(k_dev);
                let _ = gpu.free(pos_dev);
                let _ = gpu.free(fc_dev);
                return Err(e);
            }
        };
        self.stage1_fc_dev = fc_dev;
        self.stage1_pos_dev = pos_dev;
        self.stage1_pos_pinned = Some(pos_pinned);
        self.stage1_k_dev = k_dev;
        self.stage1_v_dev = v_dev;
        self.stage1_cfg = Some(cfg);
        Ok((fc_dev, pos_dev, k_dev, v_dev))
    }

    pub(in super::super) fn upload_stage1_positions(
        &mut self,
        gpu: &dyn GpuBackend,
        start_abs_pos: usize,
        count: usize,
        stream: u64,
    ) -> Result<()> {
        ensure!(
            (1..=STAGE1_GRAPH_MAX_ROWS).contains(&count),
            "DFlash Stage-1 position count {count} outside 1..={STAGE1_GRAPH_MAX_ROWS}"
        );
        let slot_bytes = STAGE1_GRAPH_MAX_ROWS * std::mem::size_of::<u32>();
        let slot = self.stage1_pos_slot;
        self.stage1_pos_slot = (self.stage1_pos_slot + 1) % NOISE_META_PINNED_SLOTS;
        let slot_off = slot * slot_bytes;
        let bytes = count * std::mem::size_of::<u32>();
        let pinned = self
            .stage1_pos_pinned
            .as_mut()
            .context("DFlash Stage-1 position staging buffer missing")?;
        let dst = &mut pinned.as_mut_slice()[slot_off..slot_off + bytes];
        for i in 0..count {
            let pos = u32::try_from(start_abs_pos + i)
                .context("DFlash Stage-1 absolute position overflow u32")?;
            dst[i * 4..i * 4 + 4].copy_from_slice(&pos.to_le_bytes());
        }
        let slice = pinned.pinned_slice_range(slot_off, bytes)?;
        // SAFETY: `stage1_pos_pinned` is page-locked and owned by the static per-head
        // state; rotating across `NOISE_META_PINNED_SLOTS` disjoint slots ensures no
        // in-flight async H2D transfer is overwritten.
        unsafe {
            gpu.copy_h2d_pinned_async(slice, self.stage1_pos_dev, stream)?;
        }
        Ok(())
    }

    pub(in super::super) fn upload_gumbel_meta_c1(
        &mut self,
        gpu: &dyn GpuBackend,
        spec: Option<spark_runtime::sampler::gumbel::GumbelDraftSpec>,
        ptrs: GumbelScratchPtrs,
        anchor_upload: Option<(u32, DevicePtr)>,
        stream: u64,
    ) -> Result<()> {
        self.upload_gumbel_meta_c1_with_q_base(gpu, spec, ptrs, anchor_upload, None, stream)
    }

    pub(in super::super) fn upload_gumbel_meta_c1_with_q_base(
        &mut self,
        gpu: &dyn GpuBackend,
        spec: Option<spark_runtime::sampler::gumbel::GumbelDraftSpec>,
        ptrs: GumbelScratchPtrs,
        anchor_upload: Option<(u32, DevicePtr)>,
        q_row_base_upload: Option<(u32, DevicePtr)>,
        stream: u64,
    ) -> Result<()> {
        if self.gumbel_pinned.is_none() {
            self.gumbel_pinned =
                Some(gpu.alloc_host_pinned(GUMBEL_META_C1_SLOT_BYTES * NOISE_META_PINNED_SLOTS)?);
        }
        let slot = self.gumbel_slot;
        self.gumbel_slot = (self.gumbel_slot + 1) % NOISE_META_PINNED_SLOTS;
        let base = slot * GUMBEL_META_C1_SLOT_BYTES;
        let (seed, step0_abs_pos, temp, top_k, top_p, min_p) = gumbel_spec_fields(spec);
        let anchor_id = anchor_upload.map_or(0u32, |(tok, _)| tok);
        let q_row_base = q_row_base_upload.map_or(0u32, |(row_base, _)| row_base);

        let gumbel_pinned = self
            .gumbel_pinned
            .as_mut()
            .context("DFlash C=1 Gumbel metadata staging buffer missing")?;
        let dst = &mut gumbel_pinned.as_mut_slice()[base..base + GUMBEL_META_C1_SLOT_BYTES];
        dst[GUMBEL_META_SEED_OFF..GUMBEL_META_SEED_OFF + 8].copy_from_slice(&seed.to_le_bytes());
        dst[GUMBEL_META_STEP0_OFF..GUMBEL_META_STEP0_OFF + 8]
            .copy_from_slice(&step0_abs_pos.to_le_bytes());
        dst[GUMBEL_META_TEMP_OFF..GUMBEL_META_TEMP_OFF + 4].copy_from_slice(&temp.to_le_bytes());
        dst[GUMBEL_META_TOPK_OFF..GUMBEL_META_TOPK_OFF + 4].copy_from_slice(&top_k.to_le_bytes());
        dst[GUMBEL_META_TOPP_OFF..GUMBEL_META_TOPP_OFF + 4].copy_from_slice(&top_p.to_le_bytes());
        dst[GUMBEL_META_MINP_OFF..GUMBEL_META_MINP_OFF + 4].copy_from_slice(&min_p.to_le_bytes());
        dst[GUMBEL_META_ANCHOR_OFF..GUMBEL_META_ANCHOR_OFF + 4]
            .copy_from_slice(&anchor_id.to_le_bytes());
        dst[GUMBEL_META_Q_ROW_BASE_OFF..GUMBEL_META_Q_ROW_BASE_OFF + 4]
            .copy_from_slice(&q_row_base.to_le_bytes());

        for (field_off, len, dev_ptr) in [
            (GUMBEL_META_SEED_OFF, 8usize, ptrs.seeds),
            (GUMBEL_META_STEP0_OFF, 8usize, ptrs.step0_pos),
            (GUMBEL_META_TEMP_OFF, 4usize, ptrs.temps),
            (GUMBEL_META_TOPK_OFF, 4usize, ptrs.top_ks),
            (GUMBEL_META_TOPP_OFF, 4usize, ptrs.top_ps),
            (GUMBEL_META_MINP_OFF, 4usize, ptrs.min_ps),
        ] {
            let slice = gumbel_pinned.pinned_slice_range(base + field_off, len)?;
            // SAFETY: `gumbel_pinned` is page-locked and owned by the static
            // per-head state; rotating across `NOISE_META_PINNED_SLOTS` disjoint
            // slots ensures no in-flight async H2D transfer is overwritten.
            unsafe {
                gpu.copy_h2d_pinned_async(slice, dev_ptr, stream)?;
            }
        }
        if let Some((_, anchor_dev)) = anchor_upload {
            let slice = gumbel_pinned.pinned_slice_range(base + GUMBEL_META_ANCHOR_OFF, 4)?;
            // SAFETY: same rotating page-locked slot invariant as above.
            unsafe {
                gpu.copy_h2d_pinned_async(slice, anchor_dev, stream)?;
            }
        }
        if let Some((_, q_row_base_dev)) = q_row_base_upload {
            let slice = gumbel_pinned.pinned_slice_range(base + GUMBEL_META_Q_ROW_BASE_OFF, 4)?;
            // SAFETY: same rotating page-locked slot invariant as above.
            unsafe {
                gpu.copy_h2d_pinned_async(slice, q_row_base_dev, stream)?;
            }
        }
        Ok(())
    }

    pub(in super::super) fn insert_graph(
        &mut self,
        gpu: &dyn GpuBackend,
        key: NoiseGraphRuntimeKey,
        graph: GraphHandle,
    ) {
        if let Some(old) = self.graphs.insert(key.clone(), graph)
            && old.0 != 0
        {
            let _ = gpu.destroy_graph(old);
        }
        self.graph_order.retain(|existing| existing != &key);
        self.graph_order.push(key);
        while self.graph_order.len() > MAX_CACHED_NOISE_GRAPHS {
            let evicted = self.graph_order.remove(0);
            if let Some(old) = self.graphs.remove(&evicted)
                && old.0 != 0
            {
                let _ = gpu.destroy_graph(old);
            }
        }
    }
}

type NoiseGraphRegistry = Mutex<HashMap<usize, Arc<Mutex<NoiseGraphHeadState>>>>;

static NOISE_GRAPH_REGISTRY: OnceLock<NoiseGraphRegistry> = OnceLock::new();

pub(in super::super) fn noise_graph_head_state(head_ptr: usize) -> Arc<Mutex<NoiseGraphHeadState>> {
    let map = NOISE_GRAPH_REGISTRY.get_or_init(|| Mutex::new(HashMap::new()));
    let mut guard = map
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    guard
        .entry(head_ptr)
        .or_insert_with(|| Arc::new(Mutex::new(NoiseGraphHeadState::new())))
        .clone()
}

/// Destroy all cached C=1 Stage-2 CUDA graphs and device metadata buffers for
/// `head_ptr` before the parent model frees shared rejection buffers in `Drop`.
pub(in super::super) fn clear_noise_graph_head_state(head_ptr: usize, gpu: &dyn GpuBackend) {
    let Some(map) = NOISE_GRAPH_REGISTRY.get() else {
        return;
    };
    let entry = map
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .remove(&head_ptr);
    if let Some(head_state_arc) = entry {
        let mut state = head_state_arc
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        state.graph_order.clear();
        for (_, graph) in state.graphs.drain() {
            if graph.0 != 0 {
                let _ = gpu.destroy_graph(graph);
            }
        }
        for (_, graph) in state.stage1_graphs.drain() {
            if graph.0 != 0 {
                let _ = gpu.destroy_graph(graph);
            }
        }
        if !state.c1_bundle_dev.is_null() {
            let _ = gpu.free(state.c1_bundle_dev);
            state.c1_bundle_dev = DevicePtr::NULL;
        }
        state.c1_bundle_pinned = None;
        state.c1_bundle_layers = 0;
        state.c1_bundle_slot_bytes = 0;
        let s = &mut *state;
        for ptr in [
            &mut s.stage1_fc_dev,
            &mut s.stage1_pos_dev,
            &mut s.stage1_k_dev,
            &mut s.stage1_v_dev,
        ] {
            if !ptr.is_null() {
                let _ = gpu.free(*ptr);
                *ptr = DevicePtr::NULL;
            }
        }
        state.stage1_pos_pinned = None;
        state.stage1_cfg = None;
        state.stage1_capture_poisoned = false;
        if !state.meta_dev.is_null() {
            let _ = gpu.free(state.meta_dev);
            state.meta_dev = DevicePtr::NULL;
        }
        #[cfg(test)]
        {
            if !state.ring_meta_dev.is_null() {
                let _ = gpu.free(state.ring_meta_dev);
                state.ring_meta_dev = DevicePtr::NULL;
            }
            state.ring_meta_pinned = None;
            state.ring_meta_layers = 0;
            state.ring_meta_slot_bytes = 0;
        }
        if !state.mask_tokens_dev.is_null() {
            let _ = gpu.free(state.mask_tokens_dev);
            state.mask_tokens_dev = DevicePtr::NULL;
        }
        if !state.split_kv_ws_dev.is_null() {
            let _ = gpu.free(state.split_kv_ws_dev);
            state.split_kv_ws_dev = DevicePtr::NULL;
            state.split_kv_ws_cap = 0;
        }
    }
}

/// Remove any remaining `NOISE_GRAPH_REGISTRY` entry for `head_ptr` when
/// [`BlockDiffusionDraftHead`] drops, preventing stale pointer-keyed entries
/// across address reuse if the caller dropped the head without invoking
/// `release_rejection_resources`.
///
/// Returns `true` if a stale entry was present and removed.
pub(in super::super) fn drop_noise_graph_head_entry(head_ptr: usize) -> bool {
    let Some(map) = NOISE_GRAPH_REGISTRY.get() else {
        return false;
    };
    let removed = map
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .remove(&head_ptr);
    removed.is_some()
}

pub(in super::super) struct NoiseGraphCaptureGuard<'a>(
    crate::model::trait_impl::capture_guard::CaptureGuard<'a>,
);

impl<'a> NoiseGraphCaptureGuard<'a> {
    pub(in super::super) fn begin(gpu: &'a dyn GpuBackend, stream: u64) -> Result<Self> {
        crate::model::trait_impl::capture_guard::CaptureGuard::begin(
            gpu,
            stream,
            "dflash_noise_graph",
        )
        .map(Self)
    }

    pub(in super::super) fn end(self) -> Result<GraphHandle> {
        self.0.end()
    }
}

pub(in super::super) fn invalidate_dstate_ctx_kv_ranges(dstate: &mut DflashProposerState) {
    dstate.cache_k_start.fill(0);
    dstate.cache_k_end.fill(0);
    dstate.cache_v_start.fill(0);
    dstate.cache_v_end.fill(0);
}

pub(in super::super) fn read_noise_positions_from_pinned(
    pos_pinned: &PinnedHostBuffer,
    eff_ctx: usize,
    noise_rows: usize,
) -> Result<Vec<u32>> {
    let start_byte = eff_ctx
        .checked_mul(std::mem::size_of::<u32>())
        .context("DFlash noise_positions byte offset overflow")?;
    let byte_len = noise_rows
        .checked_mul(std::mem::size_of::<u32>())
        .context("DFlash noise_positions byte length overflow")?;
    let end_byte = start_byte
        .checked_add(byte_len)
        .context("DFlash noise_positions byte end overflow")?;
    ensure!(
        end_byte <= pos_pinned.len(),
        "DFlash pinned position buffer too small: need {end_byte}, have {}",
        pos_pinned.len()
    );
    // SAFETY: `pos_pinned` is owned by `dstate`, `end_byte <= pos_pinned.len()`
    // was verified above, and this read-only borrow never mutates bytes being
    // read by the pending stream-ordered H2D transfer.
    let bytes =
        unsafe { std::slice::from_raw_parts(pos_pinned.as_ptr().add(start_byte), byte_len) };
    let mut positions = Vec::with_capacity(noise_rows);
    for chunk in bytes.chunks_exact(4) {
        positions.push(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(positions)
}
