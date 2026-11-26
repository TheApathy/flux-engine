// SPDX-License-Identifier: AGPL-3.0-only

//! Per-quantization Stage-2 drafter layer passes (BF16, FP8, NVFP4).

use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::super::{BlockDiffusionDraftHead, DflashLayer, DflashLayerFp8Weights, DflashLayerNvfp4};
use super::policy::noise_graph_attn_kgamma_disable_o;
use super::stage1::Stage2LayerCtxAttn;

impl BlockDiffusionDraftHead {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn stage2_layer_bf16(
        &self,
        layer: &DflashLayer,
        ctx_attn: Stage2LayerCtxAttn,
        noise_positions_dev: DevicePtr,
        m_rows: u32,
        h: u32,
        q_dim: u32,
        kv_dim: u32,
        inter: u32,
        inv_sqrt_d: f32,
        causal: bool,
        sliding_window: u32,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        use crate::layers::ops;

        ops::rms_norm(
            gpu,
            self.kernels.rms_norm,
            self.scratch.stream_buf,
            &layer.input_layernorm,
            self.scratch.norm_buf,
            m_rows,
            h,
            self.rms_norm_eps,
            stream,
        )?;
        let attn_in = if let Some(conv) = &layer.attention_conv {
            self.stage2_conv_prepare_noise(conv, m_rows, h, gpu, stream)?
        } else {
            self.scratch.norm_buf
        };
        self.project_bf16_dense(
            gpu,
            attn_in,
            &layer.q_proj,
            self.scratch.q_buf,
            m_rows,
            q_dim,
            h,
            stream,
        )?;
        self.project_bf16_dense(
            gpu,
            attn_in,
            &layer.k_proj,
            self.scratch.k_buf,
            m_rows,
            kv_dim,
            h,
            stream,
        )?;
        self.project_bf16_dense(
            gpu,
            attn_in,
            &layer.v_proj,
            self.scratch.v_buf,
            m_rows,
            kv_dim,
            h,
            stream,
        )?;
        self.stage2_qk_norm_rope_and_attn(
            &layer.q_norm,
            &layer.k_norm,
            ctx_attn,
            noise_positions_dev,
            m_rows,
            inv_sqrt_d,
            causal,
            sliding_window,
            gpu,
            stream,
        )?;
        self.project_bf16_dense(
            gpu,
            self.scratch.attn_out,
            &layer.o_proj,
            self.scratch.stream_acc,
            m_rows,
            h,
            q_dim,
            stream,
        )?;
        let attn_residual = if let Some(conv) = &layer.attention_conv {
            self.stage2_conv_finish_noise(conv, m_rows, gpu, stream)?
        } else {
            self.scratch.stream_acc
        };
        ops::residual_add(
            gpu,
            self.kernels.residual_add,
            self.scratch.stream_buf,
            attn_residual,
            m_rows * h,
            stream,
        )?;
        ops::rms_norm(
            gpu,
            self.kernels.rms_norm,
            self.scratch.stream_buf,
            &layer.post_attention_layernorm,
            self.scratch.norm_buf,
            m_rows,
            h,
            self.rms_norm_eps,
            stream,
        )?;
        let mlp_in = if let Some(conv) = &layer.mlp_conv {
            self.stage2_conv_prepare_noise(conv, m_rows, h, gpu, stream)?
        } else {
            self.scratch.norm_buf
        };
        self.project_bf16_dense(
            gpu,
            mlp_in,
            &layer.gate_proj,
            self.scratch.mlp_intermediate,
            m_rows,
            inter,
            h,
            stream,
        )?;
        self.project_bf16_dense(
            gpu,
            mlp_in,
            &layer.up_proj,
            self.scratch.mlp_up,
            m_rows,
            inter,
            h,
            stream,
        )?;
        ops::silu_mul(
            gpu,
            self.kernels.silu_mul,
            self.scratch.mlp_intermediate,
            self.scratch.mlp_up,
            self.scratch.mlp_intermediate,
            m_rows * inter,
            stream,
        )?;
        self.project_bf16_dense(
            gpu,
            self.scratch.mlp_intermediate,
            &layer.down_proj,
            self.scratch.stream_acc,
            m_rows,
            h,
            inter,
            stream,
        )?;
        let mlp_residual = if let Some(conv) = &layer.mlp_conv {
            self.stage2_conv_finish_noise(conv, m_rows, gpu, stream)?
        } else {
            self.scratch.stream_acc
        };
        ops::residual_add(
            gpu,
            self.kernels.residual_add,
            self.scratch.stream_buf,
            mlp_residual,
            m_rows * h,
            stream,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn stage2_layer_fp8(
        &self,
        layer: &DflashLayerFp8Weights,
        ctx_attn: Stage2LayerCtxAttn,
        noise_positions_dev: DevicePtr,
        m_rows: u32,
        h: u32,
        q_dim: u32,
        kv_dim: u32,
        inter: u32,
        inv_sqrt_d: f32,
        causal: bool,
        sliding_window: u32,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        use crate::layers::ops;

        ops::rms_norm(
            gpu,
            self.kernels.rms_norm,
            self.scratch.stream_buf,
            &layer.input_layernorm,
            self.scratch.norm_buf,
            m_rows,
            h,
            self.rms_norm_eps,
            stream,
        )?;
        let attn_in = if let Some(conv) = &layer.attention_conv {
            self.stage2_conv_prepare_noise(conv, m_rows, h, gpu, stream)?
        } else {
            self.scratch.norm_buf
        };
        self.stage_fp8_project(
            gpu,
            attn_in,
            &layer.q_proj,
            self.scratch.q_buf,
            m_rows,
            q_dim,
            h,
            stream,
        )?;
        self.stage_fp8_project(
            gpu,
            attn_in,
            &layer.k_proj,
            self.scratch.k_buf,
            m_rows,
            kv_dim,
            h,
            stream,
        )?;
        self.stage_fp8_project(
            gpu,
            attn_in,
            &layer.v_proj,
            self.scratch.v_buf,
            m_rows,
            kv_dim,
            h,
            stream,
        )?;
        self.stage2_qk_norm_rope_and_attn(
            &layer.q_norm,
            &layer.k_norm,
            ctx_attn,
            noise_positions_dev,
            m_rows,
            inv_sqrt_d,
            causal,
            sliding_window,
            gpu,
            stream,
        )?;
        self.stage_fp8_project(
            gpu,
            self.scratch.attn_out,
            &layer.o_proj,
            self.scratch.stream_acc,
            m_rows,
            h,
            q_dim,
            stream,
        )?;
        let attn_residual = if let Some(conv) = &layer.attention_conv {
            self.stage2_conv_finish_noise(conv, m_rows, gpu, stream)?
        } else {
            self.scratch.stream_acc
        };
        ops::residual_add(
            gpu,
            self.kernels.residual_add,
            self.scratch.stream_buf,
            attn_residual,
            m_rows * h,
            stream,
        )?;
        ops::rms_norm(
            gpu,
            self.kernels.rms_norm,
            self.scratch.stream_buf,
            &layer.post_attention_layernorm,
            self.scratch.norm_buf,
            m_rows,
            h,
            self.rms_norm_eps,
            stream,
        )?;
        let mlp_in = if let Some(conv) = &layer.mlp_conv {
            self.stage2_conv_prepare_noise(conv, m_rows, h, gpu, stream)?
        } else {
            self.scratch.norm_buf
        };
        self.stage_fp8_project(
            gpu,
            mlp_in,
            &layer.gate_proj,
            self.scratch.mlp_intermediate,
            m_rows,
            inter,
            h,
            stream,
        )?;
        self.stage_fp8_project(
            gpu,
            mlp_in,
            &layer.up_proj,
            self.scratch.mlp_up,
            m_rows,
            inter,
            h,
            stream,
        )?;
        ops::silu_mul(
            gpu,
            self.kernels.silu_mul,
            self.scratch.mlp_intermediate,
            self.scratch.mlp_up,
            self.scratch.mlp_intermediate,
            m_rows * inter,
            stream,
        )?;
        self.stage_fp8_project(
            gpu,
            self.scratch.mlp_intermediate,
            &layer.down_proj,
            self.scratch.stream_acc,
            m_rows,
            h,
            inter,
            stream,
        )?;
        let mlp_residual = if let Some(conv) = &layer.mlp_conv {
            self.stage2_conv_finish_noise(conv, m_rows, gpu, stream)?
        } else {
            self.scratch.stream_acc
        };
        ops::residual_add(
            gpu,
            self.kernels.residual_add,
            self.scratch.stream_buf,
            mlp_residual,
            m_rows * h,
            stream,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn stage2_layer_nvfp4(
        &self,
        layer: &DflashLayerNvfp4,
        ctx_attn: Stage2LayerCtxAttn,
        noise_positions_dev: DevicePtr,
        m_rows: u32,
        h: u32,
        q_dim: u32,
        kv_dim: u32,
        inter: u32,
        inv_sqrt_d: f32,
        causal: bool,
        sliding_window: u32,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        use crate::layers::ops;

        let w4a4 = self.w4a4_shadow.as_ref();
        let w4a4_scales = layer.input_scales;
        ensure!(
            w4a4.is_some() == w4a4_scales.is_some(),
            "DFlash2 W4A4 runtime state and calibrated layer receipt disagree"
        );
        let attn_t = if self.kernels.w4a16_gemm_t_m16.0 != 0 {
            match (
                layer.q_proj_t.as_ref(),
                layer.k_proj_t.as_ref(),
                layer.v_proj_t.as_ref(),
                layer.o_proj_t.as_ref(),
            ) {
                (Some(q), Some(k), Some(v), Some(o)) => Some((q, k, v, o)),
                _ => None,
            }
        } else {
            None
        };

        ops::rms_norm(
            gpu,
            self.kernels.rms_norm,
            self.scratch.stream_buf,
            &layer.input_layernorm,
            self.scratch.norm_buf,
            m_rows,
            h,
            self.rms_norm_eps,
            stream,
        )?;
        let attn_in = if let Some(conv) = &layer.attention_conv {
            self.stage2_conv_prepare_noise(conv, m_rows, h, gpu, stream)?
        } else {
            self.scratch.norm_buf
        };

        if let (Some(w4a4), Some(scales)) = (w4a4, w4a4_scales) {
            w4a4.quantize(gpu, attn_in, scales.qkv(), m_rows, h, stream)?;
            w4a4.project_prepared(
                gpu,
                scales.qkv(),
                &layer.q_proj,
                self.scratch.q_buf,
                m_rows,
                q_dim,
                h,
                stream,
            )?;
            w4a4.project_prepared(
                gpu,
                scales.qkv(),
                &layer.k_proj,
                self.scratch.k_buf,
                m_rows,
                kv_dim,
                h,
                stream,
            )?;
            w4a4.project_prepared(
                gpu,
                scales.qkv(),
                &layer.v_proj,
                self.scratch.v_buf,
                m_rows,
                kv_dim,
                h,
                stream,
            )?;
        } else if m_rows <= 16
            && crate::layers::dense_ffn::try_coalesced_nvfp4_sm120_small_m(
                gpu,
                attn_in,
                &layer.q_proj,
                self.scratch.q_buf,
                m_rows,
                q_dim,
                h,
                16,
                stream,
            )?
        {
            if !crate::layers::dense_ffn::try_coalesced_nvfp4_sm120_small_m(
                gpu,
                attn_in,
                &layer.k_proj,
                self.scratch.k_buf,
                m_rows,
                kv_dim,
                h,
                16,
                stream,
            )? {
                self.nvfp4_project_coalesced_or_gemm(
                    gpu,
                    attn_in,
                    &layer.k_proj,
                    self.scratch.k_buf,
                    m_rows,
                    kv_dim,
                    h,
                    stream,
                )?;
            }
            if !crate::layers::dense_ffn::try_coalesced_nvfp4_sm120_small_m(
                gpu,
                attn_in,
                &layer.v_proj,
                self.scratch.v_buf,
                m_rows,
                kv_dim,
                h,
                16,
                stream,
            )? {
                self.nvfp4_project_coalesced_or_gemm(
                    gpu,
                    attn_in,
                    &layer.v_proj,
                    self.scratch.v_buf,
                    m_rows,
                    kv_dim,
                    h,
                    stream,
                )?;
            }
        } else if let Some((q_t, k_t, v_t, _)) = attn_t {
            self.draft_gemm_t(
                gpu,
                attn_in,
                q_t,
                self.scratch.q_buf,
                m_rows,
                q_dim,
                h,
                true,
                stream,
            )?;
            self.draft_gemm_t(
                gpu,
                attn_in,
                k_t,
                self.scratch.k_buf,
                m_rows,
                kv_dim,
                h,
                true,
                stream,
            )?;
            self.draft_gemm_t(
                gpu,
                attn_in,
                v_t,
                self.scratch.v_buf,
                m_rows,
                kv_dim,
                h,
                true,
                stream,
            )?;
        } else {
            self.nvfp4_project_coalesced_or_gemm(
                gpu,
                attn_in,
                &layer.q_proj,
                self.scratch.q_buf,
                m_rows,
                q_dim,
                h,
                stream,
            )?;
            self.nvfp4_project_coalesced_or_gemm(
                gpu,
                attn_in,
                &layer.k_proj,
                self.scratch.k_buf,
                m_rows,
                kv_dim,
                h,
                stream,
            )?;
            self.nvfp4_project_coalesced_or_gemm(
                gpu,
                attn_in,
                &layer.v_proj,
                self.scratch.v_buf,
                m_rows,
                kv_dim,
                h,
                stream,
            )?;
        }

        self.stage2_qk_norm_rope_and_attn(
            &layer.q_norm,
            &layer.k_norm,
            ctx_attn,
            noise_positions_dev,
            m_rows,
            inv_sqrt_d,
            causal,
            sliding_window,
            gpu,
            stream,
        )?;

        let disable_o = noise_graph_attn_kgamma_disable_o();
        if let (Some(w4a4), Some(scales)) = (w4a4, w4a4_scales) {
            w4a4.quantize(
                gpu,
                self.scratch.attn_out,
                scales.output(),
                m_rows,
                q_dim,
                stream,
            )?;
            w4a4.project_prepared(
                gpu,
                scales.output(),
                &layer.o_proj,
                self.scratch.stream_acc,
                m_rows,
                h,
                q_dim,
                stream,
            )?;
        } else if m_rows <= 16
            && crate::layers::dense_ffn::try_coalesced_nvfp4_sm120_small_m(
                gpu,
                self.scratch.attn_out,
                &layer.o_proj,
                self.scratch.stream_acc,
                m_rows,
                h,
                q_dim,
                16,
                stream,
            )?
        {
            // Dispatched via SM120 BF16xBF16 MMA coalesced warp-split-K.
        } else if let Some((_, _, _, o_t)) = attn_t
            && !disable_o
        {
            self.draft_gemm_t(
                gpu,
                self.scratch.attn_out,
                o_t,
                self.scratch.stream_acc,
                m_rows,
                h,
                q_dim,
                true,
                stream,
            )?;
        } else {
            self.nvfp4_project_coalesced_or_gemm(
                gpu,
                self.scratch.attn_out,
                &layer.o_proj,
                self.scratch.stream_acc,
                m_rows,
                h,
                q_dim,
                stream,
            )?;
        }

        let attn_residual = if let Some(conv) = &layer.attention_conv {
            self.stage2_conv_finish_noise(conv, m_rows, gpu, stream)?
        } else {
            self.scratch.stream_acc
        };
        ops::residual_add(
            gpu,
            self.kernels.residual_add,
            self.scratch.stream_buf,
            attn_residual,
            m_rows * h,
            stream,
        )?;
        ops::rms_norm(
            gpu,
            self.kernels.rms_norm,
            self.scratch.stream_buf,
            &layer.post_attention_layernorm,
            self.scratch.norm_buf,
            m_rows,
            h,
            self.rms_norm_eps,
            stream,
        )?;
        let mlp_in = if let Some(conv) = &layer.mlp_conv {
            self.stage2_conv_prepare_noise(conv, m_rows, h, gpu, stream)?
        } else {
            self.scratch.norm_buf
        };

        let ffn_t = if self.kernels.w4a16_gemm_t_m16.0 != 0 {
            match (
                layer.gate_proj_t.as_ref(),
                layer.up_proj_t.as_ref(),
                layer.down_proj_t.as_ref(),
            ) {
                (Some(g), Some(u), Some(d)) => Some((g, u, d)),
                _ => None,
            }
        } else {
            None
        };
        let mut fused_gateup = w4a4.is_none()
            && ffn_t.is_some()
            && self.kernels.w4a16_gemm_t_m32_n64_gateup_silu.0 != 0
            && self.kernels.w4a16_gemm_t_m32_n64.0 != 0;

        if let (Some(w4a4), Some(scales)) = (w4a4, w4a4_scales) {
            w4a4.quantize(gpu, mlp_in, scales.gate_up(), m_rows, h, stream)?;
            w4a4.project_prepared(
                gpu,
                scales.gate_up(),
                &layer.gate_proj,
                self.scratch.mlp_intermediate,
                m_rows,
                inter,
                h,
                stream,
            )?;
            w4a4.project_prepared(
                gpu,
                scales.gate_up(),
                &layer.up_proj,
                self.scratch.mlp_up,
                m_rows,
                inter,
                h,
                stream,
            )?;
        } else if m_rows <= 16
            && crate::layers::dense_ffn::try_coalesced_nvfp4_sm120_small_m_gateup_silu(
                gpu,
                mlp_in,
                &layer.gate_proj,
                &layer.up_proj,
                self.scratch.mlp_intermediate,
                m_rows,
                inter,
                h,
                16,
                stream,
            )?
        {
            fused_gateup = true;
        } else if let Some((gate_t, up_t, _)) = ffn_t
            && fused_gateup
        {
            if atlas_core::device::active_sm_count() > 64
                && self.kernels.w4a16_gemm_t_m32_n32_gateup_silu_pipe_k64.0 != 0
                && h.is_multiple_of(64)
                && inter.is_multiple_of(32)
            {
                ops::w4a16_gemm_n32_m32_gateup_silu(
                    gpu,
                    self.kernels.w4a16_gemm_t_m32_n32_gateup_silu_pipe_k64,
                    mlp_in,
                    gate_t,
                    up_t,
                    self.scratch.mlp_intermediate,
                    m_rows,
                    inter,
                    h,
                    stream,
                )?;
            } else {
                ops::w4a16_gemm_n64_m32_gateup_silu(
                    gpu,
                    self.kernels.w4a16_gemm_t_m32_n64_gateup_silu,
                    mlp_in,
                    gate_t,
                    up_t,
                    self.scratch.mlp_intermediate,
                    m_rows,
                    inter,
                    h,
                    stream,
                )?;
            }
        } else if let Some((gate_t, up_t, _)) = ffn_t {
            self.draft_gemm_t(
                gpu,
                mlp_in,
                gate_t,
                self.scratch.mlp_intermediate,
                m_rows,
                inter,
                h,
                true,
                stream,
            )?;
            self.draft_gemm_t(
                gpu,
                mlp_in,
                up_t,
                self.scratch.mlp_up,
                m_rows,
                inter,
                h,
                true,
                stream,
            )?;
        } else {
            self.nvfp4_project_coalesced_or_gemm(
                gpu,
                mlp_in,
                &layer.gate_proj,
                self.scratch.mlp_intermediate,
                m_rows,
                inter,
                h,
                stream,
            )?;
            self.nvfp4_project_coalesced_or_gemm(
                gpu,
                mlp_in,
                &layer.up_proj,
                self.scratch.mlp_up,
                m_rows,
                inter,
                h,
                stream,
            )?;
        }

        if !fused_gateup {
            ops::silu_mul(
                gpu,
                self.kernels.silu_mul,
                self.scratch.mlp_intermediate,
                self.scratch.mlp_up,
                self.scratch.mlp_intermediate,
                m_rows * inter,
                stream,
            )?;
        }

        if let (Some(w4a4), Some(scales)) = (w4a4, w4a4_scales) {
            w4a4.quantize(
                gpu,
                self.scratch.mlp_intermediate,
                scales.down(),
                m_rows,
                inter,
                stream,
            )?;
            w4a4.project_prepared(
                gpu,
                scales.down(),
                &layer.down_proj,
                self.scratch.stream_acc,
                m_rows,
                h,
                inter,
                stream,
            )?;
        } else if m_rows <= 16
            && crate::layers::dense_ffn::try_coalesced_nvfp4_sm120_small_m(
                gpu,
                self.scratch.mlp_intermediate,
                &layer.down_proj,
                self.scratch.stream_acc,
                m_rows,
                h,
                inter,
                16,
                stream,
            )?
        {
            // Dispatched via SM120 BF16xBF16 MMA coalesced warp-split-K.
        } else if let Some((_, _, down_t)) = ffn_t {
            self.draft_gemm_t(
                gpu,
                self.scratch.mlp_intermediate,
                down_t,
                self.scratch.stream_acc,
                m_rows,
                h,
                inter,
                true,
                stream,
            )?;
        } else {
            self.nvfp4_project_coalesced_or_gemm(
                gpu,
                self.scratch.mlp_intermediate,
                &layer.down_proj,
                self.scratch.stream_acc,
                m_rows,
                h,
                inter,
                stream,
            )?;
        }

        let mlp_residual = if let Some(conv) = &layer.mlp_conv {
            self.stage2_conv_finish_noise(conv, m_rows, gpu, stream)?
        } else {
            self.scratch.stream_acc
        };
        ops::residual_add(
            gpu,
            self.kernels.residual_add,
            self.scratch.stream_buf,
            mlp_residual,
            m_rows * h,
            stream,
        )
    }
}
