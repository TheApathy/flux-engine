// SPDX-License-Identifier: AGPL-3.0-only

//! Plain-decode QSA indexer route on a caller-chosen stream, so
//! `ATLAS_QWEN4_QSA_OVERLAP=1` can run it beside the QKV projection.

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;
use spark_runtime::kv_cache::PagedKvCache;

use super::super::Qwen3AttentionLayer;
use crate::layer::ForwardContext;
use crate::layers::qwen3_attention::Qwen4DeviceAttentionRow;

impl Qwen3AttentionLayer {
    /// QSA top-k token list for this decode step, launched on `stream`; None
    /// when the request can never reach the sparse path. QSA's indexer consumes
    /// the same hyperconnection-mixed attention input as the QKV projection. It
    /// must run for every prefix token so the compressed side cache is complete
    /// before position 2048 switches from dense fallback to sparse attention.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::layers::qwen3_attention) fn qwen4_qsa_indices(
        &self,
        normed: DevicePtr,
        host_seq_len: Option<usize>,
        device_row: Option<Qwen4DeviceAttentionRow>,
        kv_cache: &PagedKvCache,
        ctx: &ForwardContext,
        h: u32,
        eps: f32,
        stream: u64,
    ) -> Result<Option<DevicePtr>> {
        let meta = ctx
            .attn_metadata
            .expect("attention layer requires pre-uploaded metadata");
        let Some(qsa) = self.qwen4_qsa.as_ref() else {
            return Ok(None);
        };
        if !meta.qwen4_qsa_required {
            return Ok(None);
        }
        if let Some(indices) = qsa.take_precomputed_verify_row() {
            return Ok(Some(indices));
        }
        let rope_theta = ctx.config.rope_theta as f32;
        let rotary_dim = ctx.config.rotary_dim() as u32;
        Ok(Some(if let Some(row) = device_row {
            qsa.update_and_select_device(
                normed, row, kv_cache, meta, h, eps, rope_theta, rotary_dim, ctx.gpu, stream,
            )?
        } else if crate::layers::qwen4_qsa::graph_safe_enabled() {
            // Device-metadata route: correct under per-slot graph replay.
            qsa.update_and_select_graph_safe(
                normed, kv_cache, meta, h, eps, rope_theta, rotary_dim, ctx.gpu, stream,
            )?
        } else {
            let seq_len = host_seq_len.expect("host QSA route needs the host sequence length");
            qsa.update_and_select(
                normed,
                seq_len,
                seq_len + 1,
                kv_cache,
                meta,
                h,
                eps,
                rope_theta,
                rotary_dim,
                ctx.gpu,
                stream,
            )?
        }))
    }

    #[allow(clippy::too_many_arguments)]
    pub(in super::super) fn try_qsa_sparse_attention_rows(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        raw_rows: DevicePtr,
        num_tokens: usize,
        seq_len: usize,
        exact_qkv: Option<(DevicePtr, usize)>,
        device_attention_plan: Option<crate::layers::qwen3_attention::Qwen4K5DeviceAttentionPlan>,
        oproj_rows: bool,
        attn_row_bytes: usize,
        row_bytes: usize,
        kv_cache: &mut PagedKvCache,
        metadata: crate::layer::AttnMetadataDev,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<Option<crate::layers::qwen4_qsa::VerifyAttentionRoute>> {
        let Some((qkv, qkv_row_bytes)) = exact_qkv else {
            return Ok(None);
        };
        let Some(qsa) = self.qwen4_qsa.as_ref() else {
            return Ok(None);
        };
        let Some(attn_hyper) = self.qwen4_attn_hyper.as_ref() else {
            return Ok(None);
        };
        let layer_kv_dtype = kv_cache.dtype_for_layer(self.attn_layer_idx);
        if !qsa.has_sparse_attention_rows_for_dtype(layer_kv_dtype)
            || self.mla.is_some()
            || !matches!(
                self.kv_dtype,
                spark_runtime::kv_cache::KvCacheDtype::Bf16
                    | spark_runtime::kv_cache::KvCacheDtype::Nvfp4
            )
            || layer_kv_dtype != self.kv_dtype
            || self.high_speed_swap_engaged(kv_cache)
            || num_tokens * attn_row_bytes > ctx.buffers.sizes().ssm_qkvz
        {
            return Ok(None);
        }
        let all_sparse_qsa = if let Some(plan) = device_attention_plan {
            let mut ok = true;
            for r in 0..num_tokens {
                if !matches!(
                    plan.row(r)?.route,
                    crate::layers::qwen3_attention::Qwen4DeviceAttentionRoute::SparseQsa
                ) {
                    ok = false;
                    break;
                }
            }
            ok
        } else {
            ctx.config.is_qwen4_exp() && seq_len >= 2048
        };
        if !all_sparse_qsa {
            return Ok(None);
        }

        let h = ctx.config.hidden_size as u32;
        let nq = self
            .num_q_heads_override
            .unwrap_or(ctx.config.num_attention_heads) as u32;
        let nkv = self
            .num_kv_heads_override
            .unwrap_or(ctx.config.num_key_value_heads) as u32;
        let hd = self.head_dim_override.unwrap_or(ctx.config.head_dim) as u32;
        let bs = kv_cache.block_size() as u32;
        if self.active_bitfp4_kv(hd, bs).is_some() {
            anyhow::bail!(
                "ATLAS_BITFP4_KV=1 is incompatible with Qwen4 QSA NVFP4 attention \
                 (layer {}); unset ATLAS_BITFP4_KV or disable --qwen4-qsa",
                self.attn_layer_idx
            );
        }
        let q_dim = nq * hd;
        let q_proj_dim = if self.gated { q_dim * 2 } else { q_dim };
        let q_proj_bytes = q_proj_dim as usize * 2;
        let kv_bytes = (nkv * hd) as usize * 2;
        let qkv_row_stride_elems = (qkv_row_bytes / 2) as u32;
        let rotary_dim = self
            .rotary_dim_override
            .unwrap_or(ctx.config.rotary_dim() as u32);
        let rope_theta = self
            .rope_theta_override
            .unwrap_or(ctx.config.rope_theta as f32);

        let k_offset_bf16 = (q_proj_bytes / 2) as u32;
        if !self.qwen4_yarn_inv_freq.is_null() && self.rope_strided_yarn_scaled_k.0 != 0 {
            let block_x = (rotary_dim / 2).next_power_of_two().clamp(32, 128);
            spark_runtime::kernel_args::KernelLaunch::new(ctx.gpu, self.rope_strided_yarn_scaled_k)
                .grid([nq + nkv, num_tokens as u32, 1])
                .block([block_x, 1, 1])
                .arg_ptr(qkv)
                .arg_ptr(metadata.positions)
                .arg_u32(num_tokens as u32)
                .arg_u32(qkv_row_stride_elems)
                .arg_u32(k_offset_bf16)
                .arg_u32(nq)
                .arg_u32(nkv)
                .arg_u32(hd)
                .arg_u32(rotary_dim)
                .arg_ptr(self.qwen4_yarn_inv_freq)
                .arg_f32(rope_theta)
                .arg_f32(self.qwen4_yarn_attention_factor)
                .launch(stream)?;
        } else if self.qwen4_yarn_inv_freq.is_null()
            && !self.rope_proportional
            && !self.mrope_interleaved
            && self.rope_strided_b3_k.0 != 0
        {
            crate::layers::ops::rope_strided_b3(
                ctx.gpu,
                self.rope_strided_b3_k,
                qkv,
                metadata.positions,
                num_tokens as u32,
                qkv_row_stride_elems,
                k_offset_bf16,
                nq,
                nkv,
                hd,
                rotary_dim,
                rope_theta,
                stream,
            )?;
        } else {
            for row in 0..num_tokens {
                let q_out = qkv.offset(row * qkv_row_bytes);
                let k_out = q_out.offset(q_proj_bytes);
                let row_pos = metadata.positions.offset(row * 4);
                let row_pos_h = metadata.positions_h.offset(row * 4);
                let row_pos_w = metadata.positions_w.offset(row * 4);

                if !self.qwen4_yarn_inv_freq.is_null() {
                    crate::layers::ops::rope_yarn_scaled(
                        ctx.gpu,
                        self.rope_yarn_scaled_k,
                        q_out,
                        k_out,
                        row_pos,
                        1,
                        nq,
                        nkv,
                        hd,
                        rotary_dim,
                        self.qwen4_yarn_inv_freq,
                        rope_theta,
                        self.qwen4_yarn_attention_factor,
                        stream,
                    )?;
                } else if self.rope_proportional && self.rope_proportional_k.0 != 0 {
                    crate::layers::ops::rope_proportional(
                        ctx.gpu,
                        self.rope_proportional_k,
                        q_out,
                        k_out,
                        row_pos,
                        1,
                        nq,
                        nkv,
                        hd,
                        rotary_dim,
                        rope_theta,
                        stream,
                    )?;
                } else if self.mrope_interleaved {
                    self.apply_mrope(
                        ctx.gpu, q_out, k_out, row_pos, row_pos_h, row_pos_w, 1, nq, nkv, hd,
                        rotary_dim, rope_theta, stream,
                    )?;
                } else {
                    crate::layers::ops::rope(
                        ctx.gpu,
                        self.rope_k,
                        q_out,
                        k_out,
                        row_pos,
                        1,
                        nq,
                        nkv,
                        hd,
                        rotary_dim,
                        rope_theta,
                        stream,
                    )?;
                }
            }
        }

        let k_base = qkv.offset(q_proj_bytes);
        let v_base = k_base.offset(kv_bytes);
        self.write_kv_cache(
            ctx.gpu,
            k_base,
            v_base,
            kv_cache,
            metadata.slot,
            num_tokens as u32,
            nkv,
            hd,
            bs,
            qkv_row_stride_elems,
            qkv_row_stride_elems,
            stream,
            ctx.graph_capture,
        )?;

        spark_runtime::cublaslt::stage_timer::mark("qkv_kvw", stream);
        let inv_sqrt_d = self.effective_attn_scale(hd);
        let fuse_sparse_gate = self.gated && qsa.can_fuse_sparse_rows_gate();
        let sparse_rows_kernel = qsa.sparse_attention_bf16_rows(
            qkv,
            qkv_row_stride_elems,
            raw_rows,
            num_tokens,
            kv_cache,
            metadata,
            self.attn_layer_idx,
            nq,
            nkv,
            hd,
            inv_sqrt_d,
            fuse_sparse_gate.then(|| (qkv.offset(q_dim as usize * 2), qkv_row_stride_elems)),
            ctx.gpu,
            stream,
        )?;
        spark_runtime::cublaslt::stage_timer::mark("core", stream);

        if self.gated && !fuse_sparse_gate {
            if self.sigmoid_gate_mul_batched_k.0 != 0 && attn_row_bytes == (q_dim as usize) * 2 {
                crate::layers::ops::sigmoid_gate_mul_batched(
                    ctx.gpu,
                    self.sigmoid_gate_mul_batched_k,
                    raw_rows,
                    qkv.offset(q_dim as usize * 2),
                    raw_rows,
                    q_dim,
                    qkv_row_stride_elems,
                    num_tokens as u32,
                    stream,
                )?;
            } else {
                for row in 0..num_tokens {
                    let row_attn = raw_rows.offset(row * attn_row_bytes);
                    let gate_ptr = qkv.offset(row * qkv_row_bytes + q_dim as usize * 2);
                    crate::layers::ops::sigmoid_gate_mul(
                        ctx.gpu,
                        self.sigmoid_gate_mul_k,
                        row_attn,
                        gate_ptr,
                        row_attn,
                        q_dim,
                        stream,
                    )?;
                }
            }
        }
        if !oproj_rows {
            for row in 0..num_tokens {
                let row_attn = raw_rows.offset(row * attn_row_bytes);
                let attn_out = self.attention_forward_oproj(row_attn, nq, hd, h, ctx, stream)?;
                let hidden_row = hidden.offset(row * row_bytes);
                let residual_row = residual.offset(row * row_bytes);
                attn_hyper.inject_decode(
                    hidden_row,
                    attn_out,
                    attn_hyper.saved_inject(residual_row),
                    ctx.gpu,
                    stream,
                )?;
            }
        }
        Ok(Some(match sparse_rows_kernel {
            crate::layers::qwen4_qsa::SparseRowsKernel::Legacy => {
                crate::layers::qwen4_qsa::VerifyAttentionRoute::SparseRowsLegacy
            }
            crate::layers::qwen4_qsa::SparseRowsKernel::V2 => {
                crate::layers::qwen4_qsa::VerifyAttentionRoute::SparseRowsV2
            }
        }))
    }

    #[allow(clippy::too_many_arguments)]
    pub(in super::super) fn try_dense_paged_attention_rows(
        &self,
        raw_rows: DevicePtr,
        num_tokens: usize,
        seq_len: usize,
        qkv: DevicePtr,
        qkv_row_bytes: usize,
        attn_row_bytes: usize,
        qsa_journal: bool,
        qsa_rows_precomputed: bool,
        kv_cache: &mut PagedKvCache,
        metadata: crate::layer::AttnMetadataDev,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<Option<crate::layers::qwen4_qsa::VerifyAttentionRoute>> {
        if num_tokens <= 1
            || self.mla.is_some()
            || !matches!(
                self.kv_dtype,
                spark_runtime::kv_cache::KvCacheDtype::Bf16
                    | spark_runtime::kv_cache::KvCacheDtype::Nvfp4
            )
            || kv_cache.dtype_for_layer(self.attn_layer_idx) != self.kv_dtype
            || self.high_speed_swap_engaged(kv_cache)
            || (qsa_journal && !qsa_rows_precomputed)
            || (ctx.config.is_qwen4_exp()
                && metadata.qwen4_qsa_required
                && seq_len.saturating_add(num_tokens) > 2048)
        {
            return Ok(None);
        }

        let nq = self
            .num_q_heads_override
            .unwrap_or(ctx.config.num_attention_heads) as u32;
        let nkv = self
            .num_kv_heads_override
            .unwrap_or(ctx.config.num_key_value_heads) as u32;
        let hd = self.head_dim_override.unwrap_or(ctx.config.head_dim) as u32;
        let bs = kv_cache.block_size() as u32;
        if self.active_bitfp4_kv(hd, bs).is_some() {
            anyhow::bail!(
                "ATLAS_BITFP4_KV=1 is incompatible with Qwen4 QSA NVFP4 attention \
                 (layer {}); unset ATLAS_BITFP4_KV or disable --qwen4-qsa",
                self.attn_layer_idx
            );
        }
        let q_dim = nq * hd;
        let q_proj_dim = if self.gated { q_dim * 2 } else { q_dim };
        let q_proj_bytes = q_proj_dim as usize * 2;
        let kv_bytes = (nkv * hd) as usize * 2;
        let qkv_row_stride_elems = (qkv_row_bytes / 2) as u32;
        let rotary_dim = self
            .rotary_dim_override
            .unwrap_or(ctx.config.rotary_dim() as u32);
        let rope_theta = self
            .rope_theta_override
            .unwrap_or(ctx.config.rope_theta as f32);

        let k_offset_bf16 = (q_proj_bytes / 2) as u32;
        if !self.qwen4_yarn_inv_freq.is_null() && self.rope_strided_yarn_scaled_k.0 != 0 {
            let block_x = (rotary_dim / 2).next_power_of_two().clamp(32, 128);
            spark_runtime::kernel_args::KernelLaunch::new(ctx.gpu, self.rope_strided_yarn_scaled_k)
                .grid([nq + nkv, num_tokens as u32, 1])
                .block([block_x, 1, 1])
                .arg_ptr(qkv)
                .arg_ptr(metadata.positions)
                .arg_u32(num_tokens as u32)
                .arg_u32(qkv_row_stride_elems)
                .arg_u32(k_offset_bf16)
                .arg_u32(nq)
                .arg_u32(nkv)
                .arg_u32(hd)
                .arg_u32(rotary_dim)
                .arg_ptr(self.qwen4_yarn_inv_freq)
                .arg_f32(rope_theta)
                .arg_f32(self.qwen4_yarn_attention_factor)
                .launch(stream)?;
        } else if self.qwen4_yarn_inv_freq.is_null()
            && !self.rope_proportional
            && !self.mrope_interleaved
            && self.rope_strided_b3_k.0 != 0
        {
            crate::layers::ops::rope_strided_b3(
                ctx.gpu,
                self.rope_strided_b3_k,
                qkv,
                metadata.positions,
                num_tokens as u32,
                qkv_row_stride_elems,
                k_offset_bf16,
                nq,
                nkv,
                hd,
                rotary_dim,
                rope_theta,
                stream,
            )?;
        } else {
            for row in 0..num_tokens {
                let q_out = qkv.offset(row * qkv_row_bytes);
                let k_out = q_out.offset(q_proj_bytes);
                let row_pos = metadata.positions.offset(row * 4);
                let row_pos_h = metadata.positions_h.offset(row * 4);
                let row_pos_w = metadata.positions_w.offset(row * 4);

                if !self.qwen4_yarn_inv_freq.is_null() {
                    crate::layers::ops::rope_yarn_scaled(
                        ctx.gpu,
                        self.rope_yarn_scaled_k,
                        q_out,
                        k_out,
                        row_pos,
                        1,
                        nq,
                        nkv,
                        hd,
                        rotary_dim,
                        self.qwen4_yarn_inv_freq,
                        rope_theta,
                        self.qwen4_yarn_attention_factor,
                        stream,
                    )?;
                } else if self.rope_proportional && self.rope_proportional_k.0 != 0 {
                    crate::layers::ops::rope_proportional(
                        ctx.gpu,
                        self.rope_proportional_k,
                        q_out,
                        k_out,
                        row_pos,
                        1,
                        nq,
                        nkv,
                        hd,
                        rotary_dim,
                        rope_theta,
                        stream,
                    )?;
                } else if self.mrope_interleaved {
                    self.apply_mrope(
                        ctx.gpu, q_out, k_out, row_pos, row_pos_h, row_pos_w, 1, nq, nkv, hd,
                        rotary_dim, rope_theta, stream,
                    )?;
                } else {
                    crate::layers::ops::rope(
                        ctx.gpu,
                        self.rope_k,
                        q_out,
                        k_out,
                        row_pos,
                        1,
                        nq,
                        nkv,
                        hd,
                        rotary_dim,
                        rope_theta,
                        stream,
                    )?;
                }
            }
        }

        let k_base = qkv.offset(q_proj_bytes);
        let v_base = k_base.offset(kv_bytes);
        self.write_kv_cache(
            ctx.gpu,
            k_base,
            v_base,
            kv_cache,
            metadata.slot,
            num_tokens as u32,
            nkv,
            hd,
            bs,
            qkv_row_stride_elems,
            qkv_row_stride_elems,
            stream,
            ctx.graph_capture,
        )?;

        spark_runtime::cublaslt::stage_timer::mark("qkv_kvw", stream);
        let inv_sqrt_d = self.effective_attn_scale(hd);
        let fused_dense_gate = self.run_paged_decode_with_gate(
            ctx.gpu,
            qkv,
            kv_cache,
            raw_rows,
            metadata.block_table,
            metadata.seq_len,
            metadata.max_blocks_per_seq,
            num_tokens as u32,
            nq,
            nkv,
            hd,
            bs,
            inv_sqrt_d,
            qkv_row_stride_elems,
            ctx.buffers.splitk_workspace(),
            DevicePtr::NULL,
            DevicePtr::NULL,
            0,
            (seq_len + num_tokens) as u32,
            self.gated
                .then(|| (qkv.offset(q_dim as usize * 2), qkv_row_stride_elems)),
            stream,
        )?;
        spark_runtime::cublaslt::stage_timer::mark("core", stream);

        if self.gated && !fused_dense_gate {
            if self.sigmoid_gate_mul_batched_k.0 != 0 && attn_row_bytes == (q_dim as usize) * 2 {
                crate::layers::ops::sigmoid_gate_mul_batched(
                    ctx.gpu,
                    self.sigmoid_gate_mul_batched_k,
                    raw_rows,
                    qkv.offset(q_dim as usize * 2),
                    raw_rows,
                    q_dim,
                    qkv_row_stride_elems,
                    num_tokens as u32,
                    stream,
                )?;
            } else {
                for row in 0..num_tokens {
                    let row_attn = raw_rows.offset(row * attn_row_bytes);
                    let gate_ptr = qkv.offset(row * qkv_row_bytes + q_dim as usize * 2);
                    crate::layers::ops::sigmoid_gate_mul(
                        ctx.gpu,
                        self.sigmoid_gate_mul_k,
                        row_attn,
                        gate_ptr,
                        row_attn,
                        q_dim,
                        stream,
                    )?;
                }
            }
        }

        let mut counts = crate::layers::qwen4_qsa::VerifyRowRouteCounts::default();
        for _ in 0..num_tokens {
            counts.record(crate::layers::qwen4_qsa::VerifyRowAttentionRoute::DensePaged);
        }
        Ok(Some(
            crate::layers::qwen4_qsa::VerifyAttentionRoute::RowSerial(counts),
        ))
    }
}
