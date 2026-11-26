// SPDX-License-Identifier: AGPL-3.0-only
//! Device-metadata QSA entry used by the exact K5 candidate-eager path.

use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kernel_args::KernelLaunch;
use spark_runtime::kv_cache::PagedKvCache;

use super::*;
use crate::layers::qwen3_attention::Qwen4DeviceAttentionRow;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct Qwen4QsaDevicePointers {
    pub projected_qk: DevicePtr,
    pub pooled_key: DevicePtr,
    pub first_position: DevicePtr,
    pub logits: DevicePtr,
    pub token_indices: DevicePtr,
}

impl Qwen4QsaIndexer {
    pub(crate) fn prepare_device_attention(
        &self,
        kv_cache: &PagedKvCache,
        _meta: AttnMetadataDev,
        required_score_groups: u32,
        gpu: &dyn GpuBackend,
    ) -> Result<Qwen4QsaDevicePointers> {
        ensure!(
            self.score_device.0 != 0 && self.select_expand_device.0 != 0,
            "Qwen4 K5 device-attention QSA kernels are unavailable"
        );
        let cache = self.cache(gpu, kv_cache)?;
        ensure!(
            required_score_groups as usize <= cache.max_compressed_groups,
            "Qwen4 K5 device-attention exceeds QSA score capacity"
        );
        let p = Qwen4QsaDevicePointers {
            projected_qk: cache.projected_qk,
            pooled_key: cache.pooled_key,
            first_position: cache.first_position,
            logits: cache.logits,
            token_indices: cache.token_indices,
        };
        let vals = [
            p.projected_qk.0,
            p.pooled_key.0,
            p.first_position.0,
            p.logits.0,
            p.token_indices.0,
        ];
        ensure!(
            !vals.contains(&0) && vals.iter().enumerate().all(|(i, v)| !vals[..i].contains(v)),
            "Qwen4 K5 device-attention QSA pointers are null or aliased"
        );
        Ok(p)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn update_and_select_device(
        &self,
        hidden: DevicePtr,
        row: Qwen4DeviceAttentionRow,
        kv_cache: &PagedKvCache,
        meta: AttnMetadataDev,
        hidden_size: u32,
        eps: f32,
        rope_theta: f32,
        rotary_dim: u32,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<DevicePtr> {
        ensure!(meta.num_seqs == 1, "device QSA requires one metadata row");
        let cache = self.cache(gpu, kv_cache)?;
        ops::dense_gemv(
            gpu,
            self.dense_gemv,
            hidden,
            &self.index_qk_proj,
            cache.projected_qk,
            INDEX_WIDTH,
            hidden_size,
            stream,
        )?;
        ops::rms_norm(
            gpu,
            self.rms_norm,
            cache.projected_qk,
            &self.q_norm,
            cache.projected_qk,
            INDEX_HEADS,
            INDEX_DIM,
            eps,
            stream,
        )?;
        KernelLaunch::new(gpu, self.stage_pool)
            .grid([1, 1, 1])
            .block([INDEX_DIM, 1, 1])
            .arg_ptr(cache.projected_qk)
            .arg_ptr(cache.raw_ring)
            .arg_ptr(cache.pooled_key)
            .arg_ptr(cache.first_position)
            .arg_ptr(meta.slot)
            .arg_ptr(meta.positions)
            .arg_u32(cache.block_size as u32)
            .launch(stream)?;
        self.apply_rope(
            cache.projected_qk,
            meta.positions,
            INDEX_HEADS,
            rotary_dim,
            rope_theta,
            gpu,
            stream,
        )?;
        if row.qsa_pool_endpoint {
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
            self.apply_rope(
                cache.pooled_key,
                cache.first_position,
                1,
                rotary_dim,
                rope_theta,
                gpu,
                stream,
            )?;
            KernelLaunch::new(gpu, self.store_compressed)
                .grid([1, 1, 1])
                .block([INDEX_DIM, 1, 1])
                .arg_ptr(cache.pooled_key)
                .arg_ptr(cache.compressed_keys)
                .arg_ptr(meta.slot)
                .arg_ptr(meta.positions)
                .arg_u32(cache.block_size as u32)
                .launch(stream)?;
        }
        if row.qsa_score_grid_x != 0 {
            let (score_k, grid_x) = if self.score_device_v2.0 != 0 {
                (
                    self.score_device_v2,
                    row.qsa_score_grid_x
                        .div_ceil((super::SCORE_V2_GROUPS_PER_BLOCK / 8) as u32),
                )
            } else {
                (self.score_device, row.qsa_score_grid_x)
            };
            KernelLaunch::new(gpu, score_k)
                .grid([grid_x, 1, 1])
                .block([256, 1, 1])
                .arg_ptr(cache.projected_qk)
                .arg_ptr(cache.compressed_keys)
                .arg_ptr(meta.block_table)
                .arg_ptr(cache.logits)
                .arg_ptr(meta.seq_len)
                .arg_u32(cache.block_size as u32)
                .launch(stream)?;
        }
        let (select_kernel, select_threads) = if self.select_expand_device_par.0 != 0 {
            (self.select_expand_device_par, super::SELECT_PAR_THREADS)
        } else {
            (self.select_expand_device, 1)
        };
        KernelLaunch::new(gpu, select_kernel)
            .grid([1, 1, 1])
            .block([select_threads, 1, 1])
            .arg_ptr(cache.logits)
            .arg_ptr(cache.token_indices)
            .arg_ptr(meta.positions)
            .arg_ptr(meta.seq_len)
            .launch(stream)?;
        Ok(cache.token_indices)
    }

    /// Graph-safe plain-decode QSA (`ATLAS_QWEN4_QSA_GRAPH_SAFE=1`): reads all
    /// frame-varying metadata from device pointers so a captured decode graph
    /// stays valid on replay.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn update_and_select_graph_safe(
        &self,
        hidden: DevicePtr,
        kv_cache: &PagedKvCache,
        meta: AttnMetadataDev,
        hidden_size: u32,
        eps: f32,
        rope_theta: f32,
        rotary_dim: u32,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<DevicePtr> {
        ensure!(meta.num_seqs == 1, "graph-safe QSA requires 1 row");
        ensure!(
            self.score_device.0 != 0 && self.select_expand_device.0 != 0,
            "graph-safe QSA needs the device-metadata score/select kernels"
        );
        let cache = self.cache(gpu, kv_cache)?;
        ops::dense_gemv(
            gpu,
            self.dense_gemv,
            hidden,
            &self.index_qk_proj,
            cache.projected_qk,
            INDEX_WIDTH,
            hidden_size,
            stream,
        )?;
        ops::rms_norm(
            gpu,
            self.rms_norm,
            cache.projected_qk,
            &self.q_norm,
            cache.projected_qk,
            INDEX_HEADS,
            INDEX_DIM,
            eps,
            stream,
        )?;
        KernelLaunch::new(gpu, self.stage_pool)
            .grid([1, 1, 1])
            .block([INDEX_DIM, 1, 1])
            .arg_ptr(cache.projected_qk)
            .arg_ptr(cache.raw_ring)
            .arg_ptr(cache.pooled_key)
            .arg_ptr(cache.first_position)
            .arg_ptr(meta.slot)
            .arg_ptr(meta.positions)
            .arg_u32(cache.block_size as u32)
            .launch(stream)?;
        self.apply_rope(
            cache.projected_qk,
            meta.positions,
            INDEX_HEADS,
            rotary_dim,
            rope_theta,
            gpu,
            stream,
        )?;
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
        self.apply_rope(
            cache.pooled_key,
            cache.first_position,
            1,
            rotary_dim,
            rope_theta,
            gpu,
            stream,
        )?;
        KernelLaunch::new(gpu, self.store_compressed)
            .grid([1, 1, 1])
            .block([INDEX_DIM, 1, 1])
            .arg_ptr(cache.pooled_key)
            .arg_ptr(cache.compressed_keys)
            .arg_ptr(meta.slot)
            .arg_ptr(meta.positions)
            .arg_u32(cache.block_size as u32)
            .launch(stream)?;
        let (score_kernel, groups_per_block) = if self.score_device_v2.0 != 0 {
            (self.score_device_v2, super::SCORE_V2_GROUPS_PER_BLOCK)
        } else {
            (self.score_device, 8)
        };
        let score_grid = u32::try_from(cache.max_compressed_groups.div_ceil(groups_per_block))?;
        KernelLaunch::new(gpu, score_kernel)
            .grid([score_grid.max(1), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(cache.projected_qk)
            .arg_ptr(cache.compressed_keys)
            .arg_ptr(meta.block_table)
            .arg_ptr(cache.logits)
            .arg_ptr(meta.seq_len)
            .arg_u32(cache.block_size as u32)
            .launch(stream)?;
        let (select_kernel, select_threads) = if self.select_expand_device_par.0 != 0 {
            (self.select_expand_device_par, super::SELECT_PAR_THREADS)
        } else {
            (self.select_expand_device, 1)
        };
        KernelLaunch::new(gpu, select_kernel)
            .grid([1, 1, 1])
            .block([select_threads, 1, 1])
            .arg_ptr(cache.logits)
            .arg_ptr(cache.token_indices)
            .arg_ptr(meta.positions)
            .arg_ptr(meta.seq_len)
            .launch(stream)?;
        Ok(cache.token_indices)
    }
}
