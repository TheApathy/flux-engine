// SPDX-License-Identifier: AGPL-3.0-only
//! Call site of the fused full-row attention prep
//! (`ATLAS_QWEN4_PREFILL_ATTN_PREP_FUSED=1`, see `layers::qwen4_attn_prep`).

use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kernel_args::KernelLaunch;
use spark_runtime::kv_cache::{KvCacheDtype, PagedKvCache};

use super::super::Qwen3AttentionLayer;
use crate::layer::AttnMetadataDev;
use crate::layers::qwen4_attn_prep::{HEAD_DIM, SELECTOR, freq_table, kernels};
use crate::layers::qwen4_qsa::prefill_fp8;

pub(super) use crate::layers::qwen4_attn_prep::selected;

impl Qwen3AttentionLayer {
    /// Q/K norms + MRoPE in place on `rows` fused QKV rows (`[q | gate | k | v]`,
    /// `row_stride` bf16 elements) and the paged K/V write, as one launch.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn prefill_attn_prep_fused(
        &self,
        qkv: DevicePtr,
        row_stride: u32,
        rows: usize,
        start: usize,
        kv_cache: &PagedKvCache,
        metadata: AttnMetadataDev,
        nq: u32,
        nkv: u32,
        hd: u32,
        rotary_dim: u32,
        theta: f32,
        eps: f32,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        ensure!(
            hd as usize == HEAD_DIM && row_stride == 2 * nq * hd + 2 * nkv * hd,
            "{SELECTOR} requires head_dim 256 and fused [q | gate | k | v] rows"
        );
        ensure!(
            !self.attn.q_norm.weight.is_null() && !self.attn.k_norm.weight.is_null(),
            "{SELECTOR} requires per-head Q and K norms"
        );
        let layer = self.attn_layer_idx;
        let block_size = kv_cache.block_size();
        let layer_kv_dtype = kv_cache.dtype_for_layer(layer);
        let k_stride_bytes = kv_cache.k_block_stride_bytes_for_layer(layer);
        ensure!(
            k_stride_bytes == kv_cache.v_block_stride_bytes_for_layer(layer)
                && match layer_kv_dtype {
                    KvCacheDtype::Bf16 =>
                        k_stride_bytes == block_size * nkv as usize * HEAD_DIM * 2,
                    KvCacheDtype::Nvfp4 => {
                        let elems = block_size * nkv as usize * HEAD_DIM;
                        k_stride_bytes == elems / 2 + elems / 16
                    }
                    _ => false,
                },
            "{SELECTOR} requires a dense BF16 or NVFP4 paged cache (got {layer_kv_dtype})"
        );
        let yarn_active = !self.qwen4_yarn_inv_freq.is_null();
        let freq = if yarn_active {
            ensure!(
                (self.qwen4_yarn_attention_factor - 1.0).abs() <= f32::EPSILON,
                "{SELECTOR} requires unit YaRN attention_factor (1.0), got {}",
                self.qwen4_yarn_attention_factor
            );
            self.qwen4_yarn_inv_freq
        } else {
            freq_table(gpu, rotary_dim, theta, stream)?
        };
        let k_offset = 2 * nq * hd;
        if layer_kv_dtype == KvCacheDtype::Bf16
            && !yarn_active
            && nq == 24
            && nkv == 2
            && let Some(mode) = prefill_fp8::mode()?
        {
            return self.prefill_attn_prep_fused_fp8(
                mode, qkv, row_stride, rows, start, kv_cache, metadata, nq, nkv, freq, eps,
                rotary_dim, gpu, stream,
            );
        }
        let k = kernels(gpu)?;
        match layer_kv_dtype {
            KvCacheDtype::Nvfp4 => {
                ensure!(
                    self.active_bitfp4_kv(hd, u32::try_from(block_size)?)
                        .is_none(),
                    "ATLAS_BITFP4_KV=1 is incompatible with qwen4_attn_prep_fused_nvfp4 \
                     (legacy token-major NVFP4 KV writer, layer {}); unset ATLAS_BITFP4_KV",
                    layer
                );
                ensure!(
                    k.fused_nvfp4.0 != 0,
                    "{SELECTOR}: qwen4_attn_prep_fused_nvfp4 kernel unavailable"
                );
                KernelLaunch::new(gpu, k.fused_nvfp4)
                    .grid([u32::try_from(rows)? * (nq + nkv), 1, 1])
                    .block([HEAD_DIM as u32, 1, 1])
                    .arg_ptr(qkv)
                    .arg_ptr(self.attn.q_norm.weight)
                    .arg_ptr(self.attn.k_norm.weight)
                    .arg_ptr(metadata.positions)
                    .arg_ptr(metadata.positions_h)
                    .arg_ptr(metadata.positions_w)
                    .arg_ptr(freq)
                    .arg_ptr(kv_cache.k_pool_ptr(layer))
                    .arg_ptr(kv_cache.v_pool_ptr(layer))
                    .arg_ptr(metadata.slot)
                    .arg_u32(nq)
                    .arg_u32(nkv)
                    .arg_u32(row_stride)
                    .arg_u32(k_offset)
                    .arg_u32(k_offset + nkv * hd)
                    .arg_u32(rotary_dim)
                    .arg_u32(block_size as u32)
                    .arg_u64(k_stride_bytes as u64)
                    .arg_u64(kv_cache.nvfp4_data_bytes() as u64)
                    .arg_f32(eps)
                    .arg_f32(if yarn_active {
                        self.qwen4_yarn_attention_factor
                    } else {
                        1.0
                    })
                    .launch(stream)?;
            }
            KvCacheDtype::Bf16 if yarn_active => {
                ensure!(
                    k.fused_yarn.0 != 0,
                    "{SELECTOR}: qwen4_attn_prep_fused_yarn kernel unavailable"
                );
                KernelLaunch::new(gpu, k.fused_yarn)
                    .grid([u32::try_from(rows)? * (nq + nkv), 1, 1])
                    .block([HEAD_DIM as u32, 1, 1])
                    .arg_ptr(qkv)
                    .arg_ptr(self.attn.q_norm.weight)
                    .arg_ptr(self.attn.k_norm.weight)
                    .arg_ptr(metadata.positions)
                    .arg_ptr(metadata.positions_h)
                    .arg_ptr(metadata.positions_w)
                    .arg_ptr(freq)
                    .arg_ptr(kv_cache.k_pool_ptr(layer))
                    .arg_ptr(kv_cache.v_pool_ptr(layer))
                    .arg_ptr(metadata.slot)
                    .arg_u32(nq)
                    .arg_u32(nkv)
                    .arg_u32(row_stride)
                    .arg_u32(k_offset)
                    .arg_u32(k_offset + nkv * hd)
                    .arg_u32(rotary_dim)
                    .arg_u32(block_size as u32)
                    .arg_f32(eps)
                    .arg_f32(self.qwen4_yarn_attention_factor)
                    .launch(stream)?;
            }
            KvCacheDtype::Bf16 => {
                KernelLaunch::new(gpu, k.fused)
                    .grid([u32::try_from(rows)? * (nq + nkv), 1, 1])
                    .block([HEAD_DIM as u32, 1, 1])
                    .arg_ptr(qkv)
                    .arg_ptr(self.attn.q_norm.weight)
                    .arg_ptr(self.attn.k_norm.weight)
                    .arg_ptr(metadata.positions)
                    .arg_ptr(metadata.positions_h)
                    .arg_ptr(metadata.positions_w)
                    .arg_ptr(freq)
                    .arg_ptr(kv_cache.k_pool_ptr(layer))
                    .arg_ptr(kv_cache.v_pool_ptr(layer))
                    .arg_ptr(metadata.slot)
                    .arg_u32(nq)
                    .arg_u32(nkv)
                    .arg_u32(row_stride)
                    .arg_u32(k_offset)
                    .arg_u32(k_offset + nkv * hd)
                    .arg_u32(rotary_dim)
                    .arg_u32(block_size as u32)
                    .arg_f32(eps)
                    .launch(stream)?;
            }
            _ => unreachable!("dtype checked above"),
        }
        static ENGAGED: std::sync::Once = std::sync::Once::new();
        ENGAGED.call_once(|| {
            tracing::warn!(
                "ENGAGED {SELECTOR}: fused Q/K norm + MRoPE + {layer_kv_dtype} KV write rows={rows} yarn={yarn_active}"
            )
        });
        Ok(())
    }

    /// The fused prep plus the E4M3 operands of the FP8 QSA core
    /// (`ATLAS_QWEN4_PREFILL_QSA_FP8`): Q8 rows into the chunk scratch and K8/V8
    /// into this layer's shadow pages; then the shadow refill for [0, start)
    /// when this chunk does not continue the layer's previous prepped chunk.
    #[allow(clippy::too_many_arguments)]
    fn prefill_attn_prep_fused_fp8(
        &self,
        mode: prefill_fp8::Mode,
        qkv: DevicePtr,
        row_stride: u32,
        rows: usize,
        start: usize,
        kv_cache: &PagedKvCache,
        metadata: AttnMetadataDev,
        nq: u32,
        nkv: u32,
        freq: DevicePtr,
        eps: f32,
        rotary_dim: u32,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        ensure!(
            nq as usize * HEAD_DIM == prefill_fp8::q8_bytes(1) && nkv == 2,
            "{}: requires 24 query and 2 KV heads",
            prefill_fp8::SELECTOR
        );
        ensure!(
            metadata.num_seqs == 1 && !metadata.block_table.is_null(),
            "{}: requires C1 paged metadata",
            prefill_fp8::SELECTOR
        );
        let layer = self.attn_layer_idx;
        let hd = HEAD_DIM as u32;
        let k_offset = 2 * nq * hd;
        let sc = prefill_fp8::layer_scales(
            gpu,
            layer,
            self.attn.q_norm.weight,
            self.attn.k_norm.weight,
        )?;
        let q8 = prefill_fp8::q8_scratch(gpu, rows)?;
        let (k8, v8) = prefill_fp8::shadow(gpu, layer, kv_cache)?;
        let rows_u32 = u32::try_from(rows)?;
        let (prep_kernel, prep_grid) = if rotary_dim == 64
            && let Some(pf) = prefill_fp8::prep_fused_kernels(gpu)?
        {
            let use_v2 = self.qsa_sparse_prefill_selected(start, rows)?
                && !crate::layers::qwen4_qsa::prefill_sparse::dump_enabled();
            let k = match (mode, use_v2) {
                (prefill_fp8::Mode::Qk, true) => pf.prep_v2_k8,
                (prefill_fp8::Mode::Qk, false) => pf.prep_vec_k8,
                (prefill_fp8::Mode::QkPv, true) => pf.prep_v2,
                (prefill_fp8::Mode::QkPv, false) => pf.prep_vec,
            };
            (k, [rows_u32, 1, 1])
        } else {
            (
                prefill_fp8::kernels(gpu)?.prep,
                [rows_u32 * (nq + nkv), 1, 1],
            )
        };
        KernelLaunch::new(gpu, prep_kernel)
            .grid(prep_grid)
            .block([HEAD_DIM as u32, 1, 1])
            .arg_ptr(qkv)
            .arg_ptr(self.attn.q_norm.weight)
            .arg_ptr(self.attn.k_norm.weight)
            .arg_ptr(metadata.positions)
            .arg_ptr(metadata.positions_h)
            .arg_ptr(metadata.positions_w)
            .arg_ptr(freq)
            .arg_ptr(kv_cache.k_pool_ptr(layer))
            .arg_ptr(kv_cache.v_pool_ptr(layer))
            .arg_ptr(metadata.slot)
            .arg_u32(nq)
            .arg_u32(nkv)
            .arg_u32(row_stride)
            .arg_u32(k_offset)
            .arg_u32(k_offset + nkv * hd)
            .arg_u32(rotary_dim)
            .arg_u32(kv_cache.block_size() as u32)
            .arg_f32(eps)
            .arg_ptr(q8)
            .arg_ptr(k8)
            .arg_ptr(v8)
            .arg_f32(sc.q[0])
            .arg_f32(sc.q[1])
            .arg_f32(sc.k[0])
            .arg_f32(sc.k[1])
            .arg_f32(sc.v[0])
            .arg_f32(sc.v[1])
            .launch(stream)?;
        let refill = prefill_fp8::note_prepped(kv_cache, layer, start, rows)?;
        prefill_fp8::refill_shadow(gpu, kv_cache, layer, metadata.block_table, refill, stream)?;
        static ENGAGED: std::sync::Once = std::sync::Once::new();
        ENGAGED.call_once(|| {
            tracing::warn!(
                "ENGAGED {SELECTOR} + {}={}: fused Q/K norm + MRoPE + KV write + E4M3 Q/K/V rows={rows} layer={layer} prescale q={:?} k={:?} v={:?}",
                prefill_fp8::SELECTOR,
                mode.name(),
                sc.q,
                sc.k,
                sc.v
            )
        });
        Ok(())
    }
}
