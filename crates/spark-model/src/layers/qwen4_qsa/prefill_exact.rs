// SPDX-License-Identifier: AGPL-3.0-only

//! Exact four-row QSA history transaction for dense-window prefill.

use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;
use spark_runtime::kv_cache::PagedKvCache;

use super::{COMPRESS_RATIO, INDEX_DIM, INDEX_HEADS, Qwen4QsaIndexer};
use crate::layer::AttnMetadataDev;
use crate::layers::ops;
use crate::weight_map::DenseWeight;

/// Largest row count one exact QSA tile call accepts: a 32-row attention tile,
/// or a whole prefill chunk (`ATLAS_QWEN4_PREFILL_ATTN_FULLROW`) of up to the
/// family's chunk cap (2048, or W under `ATLAS_QWEN4_PREFILL_WIDE`).
fn tile_max_rows() -> usize {
    crate::layers::qwen4_prefill_moe::max_chunk_rows().max(2048)
}

/// `ATLAS_QWEN4_PREFILL_QSA_KEY_GEMM=1`: project the whole-chunk QSA index
/// keys with one cuBLASLt GEMM (default off; not bit-exact).
fn key_gemm_selected() -> bool {
    static SEL: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *SEL.get_or_init(|| {
        std::env::var("ATLAS_QWEN4_PREFILL_QSA_KEY_GEMM")
            .ok()
            .as_deref()
            == Some("1")
    })
}

impl Qwen4QsaIndexer {
    pub(crate) fn validate_exact_group4(&self) -> Result<()> {
        ensure!(
            self.dense_gemv_batchn.0 != 0,
            "exact QSA group4 requires dense_gemv_bf16_batchn"
        );
        ensure!(
            !self.index_qk_proj.weight.is_null() && !self.k_norm.weight.is_null(),
            "exact QSA group4 requires key projection and normalization weights"
        );
        Ok(())
    }

    pub(crate) fn prepare_exact_group4(
        &self,
        gpu: &dyn GpuBackend,
        kv_cache: &PagedKvCache,
    ) -> Result<()> {
        self.validate_exact_group4()?;
        let _ = self.cache(gpu, kv_cache)?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn update_prefill_exact_group4(
        &self,
        hidden: DevicePtr,
        group_start: usize,
        kv_cache: &PagedKvCache,
        meta: AttnMetadataDev,
        hidden_size: u32,
        eps: f32,
        rope_theta: f32,
        rotary_dim: u32,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        self.validate_exact_group4()?;
        ensure!(
            group_start.is_multiple_of(COMPRESS_RATIO),
            "exact QSA group4 requires an aligned complete group"
        );
        ensure!(meta.num_seqs == 1, "exact QSA group4 requires C1");
        ensure!(
            !hidden.is_null() && !meta.positions.is_null() && !meta.slot.is_null(),
            "exact QSA group4 requires complete inputs"
        );
        let cache = self.cache(gpu, kv_cache)?;
        ensure!(
            group_start
                .checked_add(COMPRESS_RATIO)
                .is_some_and(|end| end / COMPRESS_RATIO <= cache.max_compressed_groups),
            "exact QSA group4 exceeds side-cache capacity"
        );
        let key_weight = DenseWeight {
            weight: self
                .index_qk_proj
                .weight
                .offset(INDEX_HEADS as usize * INDEX_DIM as usize * hidden_size as usize * 2),
        };
        ops::dense_gemv_batchn(
            gpu,
            self.dense_gemv_batchn,
            hidden,
            &key_weight,
            cache.projected_qk,
            COMPRESS_RATIO as u32,
            INDEX_DIM,
            hidden_size,
            stream,
        )?;
        KernelLaunch::new(gpu, self.stage_prefill_raw)
            .grid([COMPRESS_RATIO as u32, 1, 1])
            .block([INDEX_DIM, 1, 1])
            .arg_ptr(cache.projected_qk)
            .arg_ptr(cache.raw_ring)
            .arg_ptr(meta.slot)
            .arg_ptr(meta.positions)
            .arg_u32(COMPRESS_RATIO as u32)
            .arg_u32(cache.block_size as u32)
            .launch(stream)?;
        KernelLaunch::new(gpu, self.pool_prefill)
            .grid([1, 1, 1])
            .block([INDEX_DIM, 1, 1])
            .arg_ptr(cache.raw_ring)
            .arg_ptr(cache.pooled_key)
            .arg_ptr(cache.first_position)
            .arg_ptr(meta.slot)
            .arg_ptr(meta.positions)
            .arg_u32((COMPRESS_RATIO - 1) as u32)
            .arg_u32(1)
            .arg_u32(cache.block_size as u32)
            .launch(stream)?;
        ops::rms_norm(
            gpu,
            self.rms_norm,
            cache.pooled_key,
            &self.k_norm,
            cache.pooled_key,
            1,
            INDEX_DIM,
            eps,
            stream,
        )?;
        self.apply_rope_n(
            cache.pooled_key,
            cache.first_position,
            1,
            1,
            rotary_dim,
            rope_theta,
            gpu,
            stream,
        )?;
        KernelLaunch::new(gpu, self.store_prefill_compressed)
            .grid([1, 1, 1])
            .block([INDEX_DIM, 1, 1])
            .arg_ptr(cache.pooled_key)
            .arg_ptr(cache.compressed_keys)
            .arg_ptr(meta.slot)
            .arg_u32((COMPRESS_RATIO - 1) as u32)
            .arg_u32(1)
            .arg_u32(cache.block_size as u32)
            .launch(stream)
    }
    /// Tile-batched twin of [`Self::update_prefill_exact_group4`]: one
    /// projection GEMV over `rows` tokens, one pool/norm/rope/store launch
    /// over `rows / 4` groups, and the ring staged only for the last group
    /// (the only ring state observable after the tile). Every per-row and
    /// per-group kernel is the same kernel with a larger grid, so results are
    /// bit-identical to the per-group sequence.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn update_prefill_exact_tile(
        &self,
        hidden: DevicePtr,
        tile_start: usize,
        rows: usize,
        kv_cache: &PagedKvCache,
        meta: AttnMetadataDev,
        hidden_size: u32,
        eps: f32,
        rope_theta: f32,
        rotary_dim: u32,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        self.validate_exact_group4()?;
        ensure!(
            tile_start.is_multiple_of(COMPRESS_RATIO) && rows > 0 && rows <= tile_max_rows(),
            "exact QSA tile requires aligned start and 1..={} rows",
            tile_max_rows()
        );
        ensure!(meta.num_seqs == 1, "exact QSA tile requires C1");
        let cache = self.cache(gpu, kv_cache)?;
        let groups = rows / COMPRESS_RATIO;
        let tail_rows = rows % COMPRESS_RATIO;
        let aligned_rows = groups * COMPRESS_RATIO;
        ensure!(
            tile_start
                .checked_add(rows)
                .is_some_and(|end| end.div_ceil(COMPRESS_RATIO) <= cache.max_compressed_groups),
            "exact QSA tile exceeds side-cache capacity"
        );
        static SCRATCH: std::sync::Mutex<Option<(DevicePtr, DevicePtr, DevicePtr, usize)>> =
            std::sync::Mutex::new(None);
        static POOL_K: std::sync::OnceLock<Result<KernelHandle, String>> =
            std::sync::OnceLock::new();
        let pool_k = POOL_K
            .get_or_init(|| {
                gpu.kernel("qwen4_qsa_tile", "qwen4_qsa_pool_prefill_from_raw")
                    .map_err(|e| e.to_string())
            })
            .clone()
            .map_err(|e| anyhow::anyhow!("QSA tile pool kernel unavailable: {e}"))?;
        let (projected, pooled, first_pos) = {
            let mut g = SCRATCH
                .lock()
                .map_err(|_| anyhow::anyhow!("QSA exact tile scratch lock poisoned"))?;
            let need_rows = crate::layers::qwen4_prefill_moe::configured_max_chunk_rows()
                .max(tile_max_rows())
                .max(rows)
                .next_multiple_of(COMPRESS_RATIO);
            if g.is_none_or(|(_, _, _, cap_rows)| cap_rows < need_rows) {
                // Deliberately retain any smaller prior allocation for in-flight stream safety.
                *g = Some((
                    gpu.alloc(need_rows * INDEX_DIM as usize * 2)?,
                    gpu.alloc(need_rows / COMPRESS_RATIO * INDEX_DIM as usize * 2)?,
                    gpu.alloc(need_rows / COMPRESS_RATIO * std::mem::size_of::<u32>())?,
                    need_rows,
                ));
            }
            let (proj, pool, fpos, _) = g.expect("QSA exact tile scratch initialized");
            (proj, pool, fpos)
        };
        let key_weight = DenseWeight {
            weight: self
                .index_qk_proj
                .weight
                .offset(INDEX_HEADS as usize * INDEX_DIM as usize * hidden_size as usize * 2),
        };
        if key_gemm_selected() {
            // One cuBLASLt GEMM instead of a GEMV block per row (which re-reads
            // the 128x2560 key weight once per row). Not bit-exact.
            spark_runtime::cublaslt::gemm_act_weight_t_typed_pinned(
                hidden.0,
                hidden_size,
                key_weight.weight.0,
                projected.0,
                INDEX_DIM,
                rows as u32,
                INDEX_DIM,
                hidden_size,
                spark_runtime::cublaslt::GemmDtype::Bf16,
                spark_runtime::cublaslt::GemmDtype::Bf16,
                false,
                rows as u32,
                stream,
            )?;
        } else {
            ops::dense_gemv_batchn(
                gpu,
                self.dense_gemv_batchn,
                hidden,
                &key_weight,
                projected,
                rows as u32,
                INDEX_DIM,
                hidden_size,
                stream,
            )?;
        }
        if groups == 0 {
            KernelLaunch::new(gpu, self.stage_prefill_raw)
                .grid([tail_rows as u32, 1, 1])
                .block([INDEX_DIM, 1, 1])
                .arg_ptr(projected)
                .arg_ptr(cache.raw_ring)
                .arg_ptr(meta.slot)
                .arg_ptr(meta.positions)
                .arg_u32(tail_rows as u32)
                .arg_u32(cache.block_size as u32)
                .launch(stream)?;
            return Ok(());
        }
        if super::prefill_sparse::prep_fused_selected()?
            && self.yarn_inv_freq.is_null()
            && rotary_dim == 64
            && INDEX_DIM == 128
        {
            static POOL_FUSED_K: std::sync::OnceLock<Result<KernelHandle, String>> =
                std::sync::OnceLock::new();
            let pool_fused_k = POOL_FUSED_K
                .get_or_init(|| {
                    gpu.kernel("qwen4_qsa_tile", "qwen4_qsa_pool_norm_rope_store_fused")
                        .map_err(|e| e.to_string())
                })
                .clone()
                .map_err(|e| anyhow::anyhow!("QSA fused pool kernel unavailable: {e}"))?;
            KernelLaunch::new(gpu, pool_fused_k)
                .grid([groups.div_ceil(4) as u32, 1, 1])
                .block([128, 1, 1])
                .arg_ptr(projected)
                .arg_ptr(self.k_norm.weight)
                .arg_ptr(cache.compressed_keys)
                .arg_ptr(cache.raw_ring)
                .arg_ptr(meta.positions)
                .arg_ptr(meta.slot)
                .arg_u32(groups as u32)
                .arg_u32(cache.block_size as u32)
                .arg_f32(eps)
                .arg_f32(rope_theta)
                .launch(stream)?;
            if tail_rows > 0 {
                KernelLaunch::new(gpu, self.stage_prefill_raw)
                    .grid([tail_rows as u32, 1, 1])
                    .block([INDEX_DIM, 1, 1])
                    .arg_ptr(projected.offset(aligned_rows * INDEX_DIM as usize * 2))
                    .arg_ptr(cache.raw_ring)
                    .arg_ptr(meta.slot.offset(aligned_rows * 8))
                    .arg_ptr(meta.positions.offset(aligned_rows * 4))
                    .arg_u32(tail_rows as u32)
                    .arg_u32(cache.block_size as u32)
                    .launch(stream)?;
            }
            static FUSED_LOGGED: std::sync::Once = std::sync::Once::new();
            FUSED_LOGGED.call_once(|| {
                tracing::info!(
                    "QSA_PREFILL_PREP_FUSED_ENGAGED selector={} rows={rows}",
                    super::prefill_sparse::PREP_FUSED_SELECTOR
                )
            });
            return Ok(());
        }
        KernelLaunch::new(gpu, pool_k)
            .grid([groups as u32, 1, 1])
            .block([INDEX_DIM, 1, 1])
            .arg_ptr(projected)
            .arg_ptr(pooled)
            .arg_ptr(first_pos)
            .arg_ptr(meta.positions)
            .arg_u32(groups as u32)
            .launch(stream)?;
        ops::rms_norm(
            gpu,
            self.rms_norm,
            pooled,
            &self.k_norm,
            pooled,
            groups as u32,
            INDEX_DIM,
            eps,
            stream,
        )?;
        self.apply_rope_n(
            pooled, first_pos, groups, 1, rotary_dim, rope_theta, gpu, stream,
        )?;
        KernelLaunch::new(gpu, self.store_prefill_compressed)
            .grid([groups as u32, 1, 1])
            .block([INDEX_DIM, 1, 1])
            .arg_ptr(pooled)
            .arg_ptr(cache.compressed_keys)
            .arg_ptr(meta.slot)
            .arg_u32((COMPRESS_RATIO - 1) as u32)
            .arg_u32(groups as u32)
            .arg_u32(cache.block_size as u32)
            .launch(stream)?;
        // Ring: only the last group's rows (plus any open tail rows) are observable after the tile.
        let (ring_start, ring_rows) = if tail_rows > 0 {
            (aligned_rows, tail_rows)
        } else {
            (rows - COMPRESS_RATIO, COMPRESS_RATIO)
        };
        KernelLaunch::new(gpu, self.stage_prefill_raw)
            .grid([ring_rows as u32, 1, 1])
            .block([INDEX_DIM, 1, 1])
            .arg_ptr(projected.offset(ring_start * INDEX_DIM as usize * 2))
            .arg_ptr(cache.raw_ring)
            .arg_ptr(meta.slot.offset(ring_start * 8))
            .arg_ptr(meta.positions.offset(ring_start * 4))
            .arg_u32(ring_rows as u32)
            .arg_u32(cache.block_size as u32)
            .launch(stream)?;
        static LOGGED: std::sync::Once = std::sync::Once::new();
        LOGGED.call_once(|| {
            tracing::info!(
                "QSA_PREFILL_TILE_ENGAGED selector=ATLAS_QWEN4_PREFILL_QSA_TILE rows={rows}"
            )
        });
        Ok(())
    }
}
