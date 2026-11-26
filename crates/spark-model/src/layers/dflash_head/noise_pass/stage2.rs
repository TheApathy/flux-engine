// SPDX-License-Identifier: AGPL-3.0-only

//! Stage-2 graphed noise-block orchestration, dense projections, split-ctx attention, and cuBLAS prewarm.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kernel_args::KernelLaunch;

use super::super::{
    BlockDiffusionDraftHead, DflashLayerQuantWeights, DflashProposerState, DraftRowLayout,
    forward_block,
};
use super::graph_state::{DflashNoiseGraphMeta, GumbelScratchPtrs};
use super::lm_head_path::LmHeadPath;
use super::policy::{
    LmHeadNvfp4Mode, bf16_proj_cublas_aligned, noise_graph_swa_enabled, validate_selector_shortlist,
};
use super::stage1::{Stage2CtxAttnDispatch, Stage2LayerCtxAttn};
use crate::layer::ForwardContext;

impl BlockDiffusionDraftHead {
    #[allow(clippy::too_many_arguments)]
    pub(in super::super) fn stage_fp8_project(
        &self,
        gpu: &dyn GpuBackend,
        input: DevicePtr,
        weight: &crate::weight_map::Fp8DenseWeight,
        output: DevicePtr,
        m: u32,
        n: u32,
        k: u32,
        stream: u64,
    ) -> Result<()> {
        use crate::layers::ops;
        if m == 1 && self.kernels.dense_gemv_fp8w.0 != 0 {
            ops::dense_gemv_fp8w(
                gpu,
                self.kernels.dense_gemv_fp8w,
                input,
                weight,
                output,
                n,
                k,
                stream,
            )
        } else {
            ops::fp8_gemm_dflash_rowscale(
                gpu,
                self.kernels.fp8_gemm_dflash,
                input,
                weight,
                output,
                m,
                n,
                k,
                stream,
            )
        }
    }

    /// Execute a dense BF16 projection `out[M, N] = act[M, K] @ weight[N, K]^T`
    /// via cuBLASLt (`ops::cublas_bf16_proj_dense`), falling back to scalar
    /// `ops::dense_gemm` only when dimensions or pointers violate cuBLASLt's
    /// 16-byte BF16 alignment contract (e.g. odd vocabulary sizes).
    #[allow(clippy::too_many_arguments)]
    pub(in super::super) fn project_bf16_dense(
        &self,
        gpu: &dyn GpuBackend,
        act: DevicePtr,
        weight: &crate::weight_map::DenseWeight,
        out: DevicePtr,
        m: u32,
        n: u32,
        k: u32,
        stream: u64,
    ) -> Result<()> {
        if bf16_proj_cublas_aligned(act, weight.weight, out, m, n, k) {
            crate::layers::ops::cublas_bf16_proj_dense(act, weight, out, m, n, k, stream)
        } else {
            crate::layers::ops::dense_gemm(
                gpu,
                self.kernels.dense_gemm,
                act,
                weight,
                out,
                m,
                n,
                k,
                stream,
            )
        }
    }

    /// Pre-warm cuBLASLt handle, 64 MB workspace, tuned plans (when
    /// `ATLAS_CUBLAS_TUNED=1`), and lazy kernel cubins for every BF16
    /// projection shape executed inside Stage-2 CUDA graph capture.
    ///
    /// Must run before `NoiseGraphCaptureGuard::begin` so `cublasLtCreate`,
    /// workspace `cuMemAlloc_v2`, and any first-use plan tuning never execute
    /// during stream capture. Every scratch output written here is dead prior
    /// to Stage 2 and overwritten before any Stage-2 consumer reads it.
    #[allow(clippy::too_many_arguments)]
    pub(in super::super) fn prewarm_stage2_cublas(
        &self,
        m_layer: u32,
        m_out: u32,
        h: u32,
        q_dim: u32,
        kv_dim: u32,
        inter: u32,
        bf16_lm_head_fallback: Option<(DevicePtr, u32)>,
        stream: u64,
    ) -> Result<()> {
        let mut warmed_conv_widths = Vec::new();
        let mut warmed_bf16_layer = false;
        for layer in &self.layers {
            let (attn_conv, mlp_conv) = match layer {
                DflashLayerQuantWeights::Bf16(l) => {
                    if !warmed_bf16_layer {
                        for (act, weight, out, n, k) in [
                            (
                                self.scratch.norm_buf,
                                &l.q_proj,
                                self.scratch.q_buf,
                                q_dim,
                                h,
                            ),
                            (
                                self.scratch.norm_buf,
                                &l.k_proj,
                                self.scratch.k_buf,
                                kv_dim,
                                h,
                            ),
                            (
                                self.scratch.attn_out,
                                &l.o_proj,
                                self.scratch.stream_acc,
                                h,
                                q_dim,
                            ),
                            (
                                self.scratch.norm_buf,
                                &l.gate_proj,
                                self.scratch.mlp_intermediate,
                                inter,
                                h,
                            ),
                            (
                                self.scratch.mlp_intermediate,
                                &l.down_proj,
                                self.scratch.stream_acc,
                                h,
                                inter,
                            ),
                        ] {
                            if bf16_proj_cublas_aligned(act, weight.weight, out, m_layer, n, k) {
                                crate::layers::ops::cublas_bf16_proj_dense(
                                    act, weight, out, m_layer, n, k, stream,
                                )?;
                            }
                        }
                        warmed_bf16_layer = true;
                    }
                    (&l.attention_conv, &l.mlp_conv)
                }
                DflashLayerQuantWeights::Fp8(l) => (&l.attention_conv, &l.mlp_conv),
                DflashLayerQuantWeights::Nvfp4(l) => (&l.attention_conv, &l.mlp_conv),
            };
            for conv in [attn_conv.as_ref(), mlp_conv.as_ref()]
                .into_iter()
                .flatten()
            {
                let dyn_width = (2 * conv.kernel_size * conv.groups) as u32;
                if !warmed_conv_widths.contains(&dyn_width) {
                    if bf16_proj_cublas_aligned(
                        self.scratch.norm_buf,
                        conv.kernel_projection.weight,
                        self.scratch.conv_dyn,
                        m_layer,
                        dyn_width,
                        h,
                    ) {
                        crate::layers::ops::cublas_bf16_proj_dense(
                            self.scratch.norm_buf,
                            &conv.kernel_projection,
                            self.scratch.conv_dyn,
                            m_layer,
                            dyn_width,
                            h,
                            stream,
                        )?;
                    }
                    warmed_conv_widths.push(dyn_width);
                }
            }
        }
        if let Some(sel) = &self.selector {
            let rank = sel.rank as u32;
            if bf16_proj_cublas_aligned(
                self.scratch.norm_buf,
                sel.hidden_projection.weight,
                self.scratch.selector_hidden,
                m_out,
                rank,
                h,
            ) {
                crate::layers::ops::cublas_bf16_proj_dense(
                    self.scratch.norm_buf,
                    &sel.hidden_projection,
                    self.scratch.selector_hidden,
                    m_out,
                    rank,
                    h,
                    stream,
                )?;
            }
        }
        if let Some((logits_out, lm_vocab)) = bf16_lm_head_fallback
            && bf16_proj_cublas_aligned(
                self.scratch.norm_buf,
                self.lm_head_shared,
                logits_out,
                m_out,
                lm_vocab,
                h,
            )
        {
            let lm_head = crate::weight_map::DenseWeight {
                weight: self.lm_head_shared,
            };
            crate::layers::ops::cublas_bf16_proj_dense(
                self.scratch.norm_buf,
                &lm_head,
                logits_out,
                m_out,
                lm_vocab,
                h,
                stream,
            )?;
        }
        Ok(())
    }

    pub(super) fn stage2_conv_prepare_noise(
        &self,
        conv: &crate::weight_loader::DflashConvWeights,
        noise_count: u32,
        h: u32,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<DevicePtr> {
        use crate::layers::ops;
        let dyn_width = (2 * conv.kernel_size * conv.groups) as u32;
        self.project_bf16_dense(
            gpu,
            self.scratch.norm_buf,
            &conv.kernel_projection,
            self.scratch.conv_dyn,
            noise_count,
            dyn_width,
            h,
            stream,
        )?;
        ops::dflash2_conv_prepare(
            gpu,
            self.kernels.dflash2_conv_prepare,
            self.scratch.norm_buf,
            self.scratch.conv_dyn,
            conv.base_kernel.weight,
            self.scratch.conv_out,
            self.scratch.conv_dyn1,
            noise_count,
            conv.groups as u32,
            stream,
        )?;
        Ok(self.scratch.conv_out)
    }

    pub(super) fn stage2_conv_finish_noise(
        &self,
        conv: &crate::weight_loader::DflashConvWeights,
        noise_count: u32,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<DevicePtr> {
        use crate::layers::ops;
        ops::dflash2_conv_finish(
            gpu,
            self.kernels.dflash2_conv_finish,
            self.scratch.stream_acc,
            self.scratch.conv_dyn1,
            conv.base_kernel.weight,
            self.scratch.conv_out,
            noise_count,
            conv.groups as u32,
            stream,
        )?;
        Ok(self.scratch.conv_out)
    }

    /// Stage 2: fixed-shape `m_rows` noise-block forward (CUDA-graph-capture-safe:
    /// zero host scalars that vary per step, zero `eff_ctx`-dependent offsets).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn launch_stage2_noise_block(
        &self,
        ctx_attn_dispatch: Stage2CtxAttnDispatch,
        meta_dev: DevicePtr,
        mask_dev: DevicePtr,
        gumbel_ptrs: GumbelScratchPtrs,
        q_row_base_dev: DevicePtr,
        gamma_eff: usize,
        row_layout: DraftRowLayout,
        lm_head_path: LmHeadPath,
        prefer_tc_lm_head: bool,
        shortlist_k: usize,
        use_radix_topk: bool,
        use_k32_selector: bool,
        rejection_mode: bool,
        dstate: &DflashProposerState,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        use crate::layers::ops;

        let gpu = ctx.gpu;
        let bf16 = 2usize;
        let m_rows = row_layout.query_rows as u32;
        let h = self.hidden_size as u32;
        let q_dim = (self.num_q_heads * self.head_dim) as u32;
        let kv_dim = (self.num_kv_heads * self.head_dim) as u32;
        let inter = self.intermediate_size as u32;
        let inv_sqrt_d = 1.0f32 / (self.head_dim as f32).sqrt();
        let eff_ctx_dev = meta_dev.offset(DflashNoiseGraphMeta::EFF_CTX_OFFSET);
        let last_token_dev = meta_dev.offset(DflashNoiseGraphMeta::LAST_TOKEN_OFFSET);
        let noise_positions_dev = meta_dev.offset(DflashNoiseGraphMeta::NOISE_POSITIONS_OFFSET);
        let swa_on = noise_graph_swa_enabled();

        // 1. Noise embeddings at offset 0 of `stream_buf`:
        //    row 0 = embed[meta.last_token], rows 1..m_rows = embed[mask_id].
        ops::batched_embed(
            gpu,
            self.kernels.batched_embed,
            last_token_dev,
            self.embed_tokens_shared,
            self.scratch.stream_buf,
            1,
            h,
            stream,
        )?;
        if m_rows > 1 {
            ops::batched_embed(
                gpu,
                self.kernels.batched_embed,
                mask_dev,
                self.embed_tokens_shared,
                self.scratch.stream_buf.offset(self.hidden_size * bf16),
                m_rows - 1,
                h,
                stream,
            )?;
        }

        // 2. Fixed-shape m_rows drafter layers with split-KV attention.
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let layer_window = if swa_on {
                self.layer_window_sizes.get(layer_idx).copied().unwrap_or(0)
            } else {
                0
            };
            let layer_causal = if swa_on {
                self.layer_causal.get(layer_idx).copied().unwrap_or(false)
            } else {
                false
            };
            let layer_ctx_attn = match ctx_attn_dispatch {
                Stage2CtxAttnDispatch::Linear { kernel, split_kv } => Stage2LayerCtxAttn::Linear {
                    kernel,
                    ctx_k: dstate.ctx_k_cache[layer_idx],
                    ctx_v: dstate.ctx_v_cache[layer_idx],
                    eff_ctx_dev,
                    split_kv,
                },
                Stage2CtxAttnDispatch::CircularRingH128 {
                    kernel,
                    ring_meta,
                    ring_cap,
                    split_kv,
                } => Stage2LayerCtxAttn::CircularRingH128 {
                    kernel,
                    k_ctx_ptr_dev: ring_meta.layer_k_ptr(layer_idx),
                    v_ctx_ptr_dev: ring_meta.layer_v_ptr(layer_idx),
                    eff_ctx_dev,
                    ring_start_dev: ring_meta.ring_start_dev,
                    ring_cap,
                    split_kv,
                },
            };
            match layer {
                DflashLayerQuantWeights::Bf16(l) => self.stage2_layer_bf16(
                    l,
                    layer_ctx_attn,
                    noise_positions_dev,
                    m_rows,
                    h,
                    q_dim,
                    kv_dim,
                    inter,
                    inv_sqrt_d,
                    layer_causal,
                    layer_window,
                    gpu,
                    stream,
                )?,
                DflashLayerQuantWeights::Fp8(l) => self.stage2_layer_fp8(
                    l,
                    layer_ctx_attn,
                    noise_positions_dev,
                    m_rows,
                    h,
                    q_dim,
                    kv_dim,
                    inter,
                    inv_sqrt_d,
                    layer_causal,
                    layer_window,
                    gpu,
                    stream,
                )?,
                DflashLayerQuantWeights::Nvfp4(l) => self.stage2_layer_nvfp4(
                    l,
                    layer_ctx_attn,
                    noise_positions_dev,
                    m_rows,
                    h,
                    q_dim,
                    kv_dim,
                    inter,
                    inv_sqrt_d,
                    layer_causal,
                    layer_window,
                    gpu,
                    stream,
                )?,
            }
        }

        // 3. Final RMSNorm + LM head over compact noise output slice.
        let output_byte_offset = row_layout.output_start * self.hidden_size * bf16;
        let stream_noise = self.scratch.stream_buf.offset(output_byte_offset);
        let norm_noise = self.scratch.norm_buf.offset(output_byte_offset);
        let lm_head_input_norm = match lm_head_path {
            LmHeadPath::Fp8 => {
                &self
                    .lm_head_shared_fp8
                    .as_ref()
                    .ok_or_else(|| anyhow::anyhow!("DFlash FP8 path lost its paired input norm"))?
                    .input_norm
            }
            LmHeadPath::Nvfp4 | LmHeadPath::Bf16 => &self.norm,
        };
        ops::rms_norm(
            gpu,
            self.kernels.rms_norm,
            stream_noise,
            lm_head_input_norm,
            norm_noise,
            gamma_eff as u32,
            h,
            self.rms_norm_eps,
            stream,
        )?;
        let lm_vocab = self.target_vocab_size.min(self.vocab_size) as u32;
        match lm_head_path {
            LmHeadPath::Nvfp4 => {
                self.launch_nvfp4_lm_head(
                    gpu,
                    norm_noise,
                    self.scratch.logits,
                    gamma_eff as u32,
                    lm_vocab,
                    h,
                    stream,
                )?;
            }
            LmHeadPath::Fp8 => {
                let fp8 = self
                    .lm_head_shared_fp8
                    .as_ref()
                    .ok_or_else(|| anyhow::anyhow!("DFlash FP8 path lost its weight/norm pair"))?;
                ops::fp8_gemm_n128(
                    gpu,
                    self.kernels.fp8_gemm_t,
                    norm_noise,
                    fp8.weight,
                    self.scratch.logits,
                    gamma_eff as u32,
                    lm_vocab,
                    h,
                    stream,
                )?;
            }
            LmHeadPath::Bf16 => {
                self.launch_lm_head_bf16_or_shared_nvfp4_exact(
                    gpu,
                    norm_noise,
                    self.scratch.logits,
                    gamma_eff,
                    lm_vocab,
                    h,
                    if prefer_tc_lm_head {
                        LmHeadNvfp4Mode::CoalescedTc
                    } else {
                        LmHeadNvfp4Mode::Exact
                    },
                    stream,
                )?;
            }
        }
        if lm_head_path.restores_unscaled_norm_for_selector(self.selector.is_some()) {
            ops::rms_norm(
                gpu,
                self.kernels.rms_norm,
                stream_noise,
                &self.norm,
                norm_noise,
                gamma_eff as u32,
                h,
                self.rms_norm_eps,
                stream,
            )?;
        }

        // 4. Tail: selector topk + hidden_projection (+ graphed walk when
        //    available), device-pointer Markov, or per-row argmax.
        let lm_stride = self.target_vocab_size.min(self.vocab_size);
        if let Some(sel) = &self.selector {
            if use_radix_topk {
                let (num_rows, vocab_u32, k_u32) = validate_selector_shortlist(
                    gamma_eff,
                    self.gamma,
                    lm_stride,
                    shortlist_k,
                    super::super::DFLASH2_GUMBEL_SHORTLIST_K,
                )?;
                ops::topk_bf16_fast(
                    gpu,
                    self.kernels.topk_bf16_radix,
                    self.scratch.logits,
                    self.scratch.topk_tokens_dev,
                    self.scratch.topk_logits_dev,
                    num_rows,
                    vocab_u32,
                    k_u32,
                    stream,
                )?;
            } else {
                let layout = forward_block::checked_topk_layout(
                    gamma_eff,
                    self.gamma,
                    lm_stride,
                    shortlist_k,
                )?;
                ops::topk_bf16(
                    gpu,
                    self.kernels.topk,
                    self.scratch.logits,
                    self.scratch.topk_tokens_dev,
                    self.scratch.topk_logits_dev,
                    layout.num_rows,
                    layout.vocab,
                    layout.k as u32,
                    stream,
                )?;
            }
            self.project_bf16_dense(
                gpu,
                norm_noise,
                &sel.hidden_projection,
                self.scratch.selector_hidden,
                gamma_eff as u32,
                sel.rank as u32,
                h,
                stream,
            )?;
            if self.markov.is_none() && rejection_mode {
                ops::dflash2_selector_walk_probabilistic_batched(
                    gpu,
                    self.kernels.dflash2_selector_walk_probabilistic_batched,
                    self.scratch.topk_logits_dev,
                    self.scratch.topk_tokens_dev,
                    self.scratch.selector_hidden,
                    sel.predecessor_codebook.weight,
                    sel.successor_codebook.weight,
                    self.scratch.draft_tokens_dev,
                    self.scratch.batch_out_probs,
                    self.scratch.shared_q_ids,
                    self.scratch.shared_q_probs,
                    Some(q_row_base_dev),
                    last_token_dev,
                    gumbel_ptrs.seeds,
                    gumbel_ptrs.step0_pos,
                    gumbel_ptrs.temps,
                    gumbel_ptrs.top_ks,
                    gumbel_ptrs.top_ps,
                    gumbel_ptrs.min_ps,
                    shortlist_k,
                    self.scratch.shared_q_stride,
                    gamma_eff,
                    sel.rank,
                    1,
                    stream,
                )?;
            } else if self.markov.is_none() && use_k32_selector {
                ops::dflash2_selector_walk_gumbel_batched_k32(
                    gpu,
                    self.kernels.dflash2_selector_walk_gumbel_batched_k32,
                    self.scratch.topk_logits_dev,
                    self.scratch.topk_tokens_dev,
                    self.scratch.selector_hidden,
                    sel.predecessor_codebook.weight,
                    sel.successor_codebook.weight,
                    self.scratch.draft_tokens_dev,
                    self.scratch.batch_out_probs,
                    last_token_dev,
                    gumbel_ptrs.seeds,
                    gumbel_ptrs.step0_pos,
                    gumbel_ptrs.temps,
                    gumbel_ptrs.top_ks,
                    gumbel_ptrs.top_ps,
                    gumbel_ptrs.min_ps,
                    shortlist_k,
                    gamma_eff,
                    sel.rank,
                    1,
                    stream,
                )?;
            } else if self.markov.is_none()
                && self.kernels.dflash2_selector_walk_gumbel_batched.0 != 0
                && self.gumbel_selector_scratch_ready()
            {
                ops::dflash2_selector_walk_gumbel_batched(
                    gpu,
                    self.kernels.dflash2_selector_walk_gumbel_batched,
                    self.scratch.topk_logits_dev,
                    self.scratch.topk_tokens_dev,
                    self.scratch.selector_hidden,
                    sel.predecessor_codebook.weight,
                    sel.successor_codebook.weight,
                    self.scratch.draft_tokens_dev,
                    last_token_dev,
                    gumbel_ptrs.seeds,
                    gumbel_ptrs.step0_pos,
                    gumbel_ptrs.temps,
                    gumbel_ptrs.top_ks,
                    gumbel_ptrs.top_ps,
                    gumbel_ptrs.min_ps,
                    gamma_eff,
                    sel.rank,
                    1,
                    stream,
                )?;
            } else if self.markov.is_none() && self.kernels.dflash2_selector_walk_batched.0 != 0 {
                ops::dflash2_selector_walk_batched(
                    gpu,
                    self.kernels.dflash2_selector_walk_batched,
                    self.scratch.topk_logits_dev,
                    self.scratch.topk_tokens_dev,
                    self.scratch.selector_hidden,
                    sel.predecessor_codebook.weight,
                    sel.successor_codebook.weight,
                    self.scratch.draft_tokens_dev,
                    last_token_dev,
                    gamma_eff as i32,
                    sel.rank as i32,
                    1,
                    stream,
                )?;
            }
        }
        if self.markov.is_some() {
            self.enqueue_markov_sequential_dev(gpu, stream, last_token_dev, gamma_eff)?;
        } else if self.selector.is_none() {
            self.launch_argmax_rows_or_loop(gpu, gamma_eff, lm_stride, stream)?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn launch_split_ctx_attention(
        &self,
        ctx_attn: Stage2LayerCtxAttn,
        m_rows: u32,
        inv_sqrt_d: f32,
        causal: bool,
        sliding_window: u32,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        match ctx_attn {
            Stage2LayerCtxAttn::Linear {
                kernel,
                ctx_k,
                ctx_v,
                eff_ctx_dev,
                split_kv: Some(plan),
            } => {
                let _ = kernel;
                KernelLaunch::new(gpu, plan.split_kernel)
                    .grid([
                        self.num_q_heads as u32,
                        m_rows.div_ceil(32),
                        plan.num_splits,
                    ])
                    .block([128, 1, 1])
                    .arg_ptr(self.scratch.q_buf)
                    .arg_ptr(ctx_k)
                    .arg_ptr(ctx_v)
                    .arg_ptr(self.scratch.k_buf)
                    .arg_ptr(self.scratch.v_buf)
                    .arg_ptr(plan.workspace)
                    .arg_ptr(eff_ctx_dev)
                    .arg_u32(m_rows)
                    .arg_u32(self.num_q_heads as u32)
                    .arg_u32(self.num_kv_heads as u32)
                    .arg_u32(plan.num_splits)
                    .arg_f32(inv_sqrt_d)
                    .arg_u32(u32::from(causal))
                    .arg_u32(sliding_window)
                    .launch(stream)?;
                KernelLaunch::new(gpu, plan.reduce_kernel)
                    .grid([self.num_q_heads as u32, m_rows, 1])
                    .block([64, 1, 1])
                    .arg_ptr(plan.workspace)
                    .arg_ptr(self.scratch.attn_out)
                    .arg_u32(m_rows)
                    .arg_u32(self.num_q_heads as u32)
                    .arg_u32(1)
                    .arg_u32(plan.num_splits)
                    .launch(stream)
            }
            Stage2LayerCtxAttn::Linear {
                kernel,
                ctx_k,
                ctx_v,
                eff_ctx_dev,
                split_kv: None,
            } => KernelLaunch::new(gpu, kernel)
                .grid([self.num_q_heads as u32, m_rows.div_ceil(32), 1])
                .block([128, 1, 1])
                .arg_ptr(self.scratch.q_buf)
                .arg_ptr(ctx_k)
                .arg_ptr(ctx_v)
                .arg_ptr(self.scratch.k_buf)
                .arg_ptr(self.scratch.v_buf)
                .arg_ptr(self.scratch.attn_out)
                .arg_ptr(eff_ctx_dev)
                .arg_u32(m_rows)
                .arg_u32(self.num_q_heads as u32)
                .arg_u32(self.num_kv_heads as u32)
                .arg_f32(inv_sqrt_d)
                .arg_u32(u32::from(causal))
                .arg_u32(sliding_window)
                .launch(stream),
            Stage2LayerCtxAttn::CircularRingH128 {
                kernel,
                k_ctx_ptr_dev,
                v_ctx_ptr_dev,
                eff_ctx_dev,
                ring_start_dev,
                ring_cap,
                split_kv: Some(plan),
            } => {
                let _ = kernel;
                KernelLaunch::new(gpu, plan.split_kernel)
                    .grid([
                        self.num_q_heads as u32,
                        m_rows.div_ceil(32),
                        plan.num_splits,
                    ])
                    .block([128, 1, 1])
                    .arg_ptr(self.scratch.q_buf)
                    .arg_ptr(k_ctx_ptr_dev)
                    .arg_ptr(v_ctx_ptr_dev)
                    .arg_ptr(self.scratch.k_buf)
                    .arg_ptr(self.scratch.v_buf)
                    .arg_ptr(plan.workspace)
                    .arg_ptr(eff_ctx_dev)
                    .arg_ptr(ring_start_dev)
                    .arg_u32(ring_cap)
                    .arg_u32(m_rows)
                    .arg_u32(self.num_q_heads as u32)
                    .arg_u32(self.num_kv_heads as u32)
                    .arg_u32(1)
                    .arg_u32(plan.num_splits)
                    .arg_f32(inv_sqrt_d)
                    .arg_u32(u32::from(causal))
                    .arg_u32(sliding_window)
                    .launch(stream)?;
                KernelLaunch::new(gpu, plan.reduce_kernel)
                    .grid([self.num_q_heads as u32, m_rows, 1])
                    .block([64, 1, 1])
                    .arg_ptr(plan.workspace)
                    .arg_ptr(self.scratch.attn_out)
                    .arg_u32(m_rows)
                    .arg_u32(self.num_q_heads as u32)
                    .arg_u32(1)
                    .arg_u32(plan.num_splits)
                    .launch(stream)
            }
            Stage2LayerCtxAttn::CircularRingH128 {
                kernel,
                k_ctx_ptr_dev,
                v_ctx_ptr_dev,
                eff_ctx_dev,
                ring_start_dev,
                ring_cap,
                split_kv: None,
            } => KernelLaunch::new(gpu, kernel)
                .grid([self.num_q_heads as u32, m_rows.div_ceil(32), 1])
                .block([128, 1, 1])
                .arg_ptr(self.scratch.q_buf)
                .arg_ptr(k_ctx_ptr_dev)
                .arg_ptr(v_ctx_ptr_dev)
                .arg_ptr(self.scratch.k_buf)
                .arg_ptr(self.scratch.v_buf)
                .arg_ptr(self.scratch.attn_out)
                .arg_ptr(eff_ctx_dev)
                .arg_ptr(ring_start_dev)
                .arg_u32(ring_cap)
                .arg_u32(m_rows)
                .arg_u32(self.num_q_heads as u32)
                .arg_u32(self.num_kv_heads as u32)
                .arg_f32(inv_sqrt_d)
                .arg_u32(u32::from(causal))
                .arg_u32(sliding_window)
                .launch(stream),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn stage2_qk_norm_rope_and_attn(
        &self,
        q_norm: &crate::weight_map::DenseWeight,
        k_norm: &crate::weight_map::DenseWeight,
        ctx_attn: Stage2LayerCtxAttn,
        noise_positions_dev: DevicePtr,
        m_rows: u32,
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
            self.scratch.q_buf,
            q_norm,
            self.scratch.q_buf,
            m_rows * self.num_q_heads as u32,
            self.head_dim as u32,
            self.rms_norm_eps,
            stream,
        )?;
        ops::rms_norm(
            gpu,
            self.kernels.rms_norm,
            self.scratch.k_buf,
            k_norm,
            self.scratch.k_buf,
            m_rows * self.num_kv_heads as u32,
            self.head_dim as u32,
            self.rms_norm_eps,
            stream,
        )?;
        ops::rope_yarn_scaled(
            gpu,
            self.kernels.rope_qwen3,
            self.scratch.q_buf,
            self.scratch.k_buf,
            noise_positions_dev,
            m_rows,
            self.num_q_heads as u32,
            self.num_kv_heads as u32,
            self.head_dim as u32,
            self.rotary_dim as u32,
            self.yarn_inv_freq,
            self.rope_theta,
            self.rope_attention_factor,
            stream,
        )?;
        self.launch_split_ctx_attention(
            ctx_attn,
            m_rows,
            inv_sqrt_d,
            causal,
            sliding_window,
            gpu,
            stream,
        )
    }
}
