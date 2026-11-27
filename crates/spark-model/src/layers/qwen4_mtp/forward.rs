// SPDX-License-Identifier: AGPL-3.0-only

//! Single-row, multi-row, and device-chained forward helpers for `Qwen4MtpHead`.

use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, HostToDeviceCopy, KernelHandle};
use spark_runtime::kv_cache::PagedKvCache;

use super::gumbel_draft;
use super::init::Qwen4MtpHead;
use super::policy::{
    CHAIN_ARGMAX_OFFSET, CHAIN_META_OFFSET, CHAIN_PROB_OFFSET, CHAIN_PROB_SPAN, CHAIN_TOKEN_OFFSET,
    ChainStep, DraftStepReuseAction, MtpQsaMode, confident_prefix, draft_step_qsa_plan,
    mtp_bounded_blocks_selected, mtp_kv_only_replay_enabled, mtp_lm_head_tc_pipe_enabled,
    mtp_qsa_mode, mtp_step0_dual_gemv_selected, prepare_packed_meta_scratch,
    resolve_chain_confidence_gate, rotary_axes, rotary_metadata_ptrs, should_sync_mid_chain,
    validate_rotary_span, write_rotary_metadata,
};
use super::state::{PendingReplay, Qwen4MtpState};
use crate::layer::{AttnMetadataDev, ForwardContext};
use crate::layers::ops;
use crate::layers::qwen4_mtp_gate::{self, GatePolicy};
use crate::layers::qwen4_mtp_rows;

impl Qwen4MtpHead {
    pub(super) fn mtp_qsa_active(&self) -> bool {
        mtp_qsa_mode() != MtpQsaMode::Disabled && self.layer.has_qsa_side_cache()
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn forward_one(
        &self,
        token: u32,
        target_hidden: DevicePtr,
        position: usize,
        state: &mut Qwen4MtpState,
        ctx: &ForwardContext,
        stream: u64,
        grammar_bitmask: Option<&[i32]>,
        emit_draft: bool,
        qsa_required: bool,
        chain: Option<ChainStep>,
    ) -> Result<u32> {
        // Fail before embeddings, cache allocation, metadata upload, or any
        // other device write. The metadata pack below repeats only the already
        // admitted immutable position lookup.
        rotary_axes(
            &state.rotary_positions,
            position,
            ctx.config.mrope_interleaved,
            self.mrope_kernel_available,
            ctx.config.yarn_factor,
        )?;
        let h = ctx.config.hidden_size;
        let r = ctx.config.residual_width();
        let hc = ctx.config.hc_count;
        let eps = ctx.config.rms_norm_eps as f32;
        let row_bytes = h * 2;

        let normed_embed = ctx.buffers.ssm_deinterleaved();
        let normed_hidden = ctx.buffers.residual();
        if self.embed_dual_rms_norm_fused_k.0 != 0 && h >= 1024 && r >= 1024 {
            let (tok_ptr, tok_out) = if let Some(device_token) = chain.and_then(|c| c.device_token)
            {
                (
                    device_token,
                    ctx.buffers.scratch().offset(CHAIN_TOKEN_OFFSET),
                )
            } else {
                (DevicePtr::NULL, DevicePtr::NULL)
            };
            spark_runtime::kernel_args::KernelLaunch::new(
                ctx.gpu,
                self.embed_dual_rms_norm_fused_k,
            )
            .grid([1, 2, 1])
            .block([1024, 1, 1])
            .arg_ptr(self.embed_tokens.weight)
            .arg_ptr(tok_ptr)
            .arg_ptr(tok_out)
            .arg_u32(token)
            .arg_u32(ctx.config.vocab_size as u32)
            .arg_ptr(self.pre_fc_norm_embedding.weight)
            .arg_ptr(normed_embed)
            .arg_ptr(target_hidden)
            .arg_ptr(self.pre_fc_norm_hidden.weight)
            .arg_ptr(normed_hidden)
            .arg_u32(h as u32)
            .arg_u32(r as u32)
            .arg_f32(eps)
            .launch(stream)?;
        } else {
            let embed = ctx.buffers.ssm_qkvz();
            if let Some(device_token) = chain.and_then(|c| c.device_token) {
                ops::embed_from_argmax(
                    ctx.gpu,
                    self.embed_from_argmax_k,
                    device_token,
                    self.embed_tokens.weight,
                    embed,
                    ctx.buffers.scratch().offset(CHAIN_TOKEN_OFFSET),
                    h as u32,
                    stream,
                )?;
            } else {
                let embed_src = self.embed_tokens.weight.offset(token as usize * row_bytes);
                ctx.gpu
                    .copy_d2d_async(embed_src, embed, row_bytes, stream)?;
            }
            if self.pre_fc_dual_norm_k.0 != 0 && h >= 1024 && r >= 1024 {
                spark_runtime::kernel_args::KernelLaunch::new(ctx.gpu, self.pre_fc_dual_norm_k)
                    .grid([1, 2, 1])
                    .block([1024, 1, 1])
                    .arg_ptr(embed)
                    .arg_ptr(self.pre_fc_norm_embedding.weight)
                    .arg_ptr(normed_embed)
                    .arg_ptr(target_hidden)
                    .arg_ptr(self.pre_fc_norm_hidden.weight)
                    .arg_ptr(normed_hidden)
                    .arg_u32(h as u32)
                    .arg_u32(r as u32)
                    .arg_f32(eps)
                    .arg_ptr(DevicePtr(0))
                    .launch(stream)?;
            } else {
                ops::rms_norm(
                    ctx.gpu,
                    self.rms_norm_k,
                    embed,
                    &self.pre_fc_norm_embedding,
                    normed_embed,
                    1,
                    h as u32,
                    eps,
                    stream,
                )?;
                ops::rms_norm(
                    ctx.gpu,
                    self.rms_norm_k,
                    target_hidden,
                    &self.pre_fc_norm_hidden,
                    normed_hidden,
                    1,
                    r as u32,
                    eps,
                    stream,
                )?;
            }
        }

        let embed_proj = ctx.buffers.qkv_output();
        let hidden = ctx.buffers.hidden_states();
        ensure!(hc == 4, "Qwen4 native MTP expects four hidden streams");

        if self.fc_dual_broadcast_rows_k.0 != 0 && mtp_step0_dual_gemv_selected() {
            spark_runtime::kernel_args::KernelLaunch::new(ctx.gpu, self.fc_dual_broadcast_rows_k)
                .grid([(h as u32).div_ceil(4), 1, 1])
                .block([256, 1, 1])
                .arg_ptr(normed_embed)
                .arg_ptr(self.fc_embedding.weight)
                .arg_ptr(self.fc_embedding.weight_scale)
                .arg_f32(self.fc_embedding.weight_scale_2)
                .arg_ptr(normed_hidden)
                .arg_ptr(self.fc_hidden.weight)
                .arg_ptr(self.fc_hidden.weight_scale)
                .arg_f32(self.fc_hidden.weight_scale_2)
                .arg_ptr(hidden)
                .arg_u32(h as u32)
                .arg_u32(h as u32)
                .launch(stream)?;
        } else {
            ops::w4a16_decode_gemv(
                ctx.gpu,
                self.w4a16_gemv_k,
                self.w4a16_gemv_sw_k,
                self.gemv_sw,
                normed_embed,
                &self.fc_embedding,
                embed_proj,
                h as u32,
                h as u32,
                stream,
            )?;
            ops::w4a16_gemv_batch_logits_exact_with(
                ctx.gpu,
                ops::W4a16ExactLmHeadKernels::new(
                    self.w4a16_gemv_exact_m4_k,
                    KernelHandle(0),
                    KernelHandle(0),
                    KernelHandle(0),
                ),
                normed_hidden,
                &self.fc_hidden,
                hidden,
                hc as u32,
                h as u32,
                h as u32,
                stream,
                false,
            )?;
            if self.residual_add_broadcast_hc_k.0 != 0 {
                spark_runtime::kernel_args::KernelLaunch::new(
                    ctx.gpu,
                    self.residual_add_broadcast_hc_k,
                )
                .grid([(h as u32).div_ceil(256), 1, 1])
                .block([256, 1, 1])
                .arg_ptr(hidden)
                .arg_ptr(embed_proj)
                .arg_u32(h as u32)
                .arg_u32(hc as u32)
                .launch(stream)?;
            } else {
                for branch in 0..hc {
                    let hidden_branch = hidden.offset(branch * row_bytes);
                    ops::residual_add(
                        ctx.gpu,
                        self.residual_add_k,
                        hidden_branch,
                        embed_proj,
                        h as u32,
                        stream,
                    )?;
                }
            }
        }

        let mut kv_cache = self.kv_cache.lock();
        let (meta_base, mb) = if let Some(c) = chain {
            (c.meta, c.max_blocks_per_seq)
        } else {
            let meta_base = ctx.buffers.scratch().offset(CHAIN_META_OFFSET);
            let packed = self.step_metadata(&mut kv_cache, state, state.seq_len, position)?;
            self.copy_h2d_group_fast(
                ctx.gpu,
                &[HostToDeviceCopy::new(&packed, meta_base)],
                stream,
            )?;
            let mb = ((packed.len() - 768) / 4) as u32;
            (meta_base, mb)
        };
        let (positions_h, positions_w) = rotary_metadata_ptrs(meta_base, &state.rotary_positions);
        let metadata = AttnMetadataDev {
            qwen4_qsa_required: qsa_required,
            positions: meta_base,
            positions_h,
            positions_w,
            slot: meta_base.offset(256),
            seq_len: meta_base.offset(512),
            block_table: meta_base.offset(768),
            max_blocks_per_seq: mb,
            num_seqs: 1,
        };
        let layer_ctx = ForwardContext {
            attn_metadata: Some(metadata),
            comm: None,
            graph_capture: false,
            ..*ctx
        };
        if !emit_draft && mtp_kv_only_replay_enabled() {
            self.layer.decode_kv_only(
                hidden,
                ctx.buffers.residual(),
                state.layer_state.as_mut(),
                &mut kv_cache,
                state.seq_len,
                &mut state.block_table,
                &mut state.disk_block_ids,
                &mut state.disk_last_offloaded_per_layer,
                &layer_ctx,
                stream,
            )?;
            state.seq_len += 1;
            return Ok(0);
        }
        self.layer.decode(
            hidden,
            ctx.buffers.residual(),
            state.layer_state.as_mut(),
            &mut kv_cache,
            state.seq_len,
            &mut state.block_table,
            &mut state.disk_block_ids,
            &mut state.disk_last_offloaded_per_layer,
            &layer_ctx,
            stream,
        )?;

        // Prompt replay only fills the MTP KV cache. It must not run the head:
        // `buffers.logits()` still holds the target's prefill logits, which the
        // scheduler samples the first output token from after this returns.
        if !emit_draft {
            state.seq_len += 1;
            return Ok(0);
        }

        let (sample_hidden, inject) = self.final_mixer.prepare_decode(
            hidden,
            ctx.buffers.residual(),
            ctx.buffers,
            ctx.gpu,
            eps,
            stream,
        )?;
        debug_assert!(inject.is_none());
        let sync_gumbel_spec = if chain.is_none() && grammar_bitmask.is_none() {
            state.resolve_gumbel_spec(position)
        } else {
            None
        };
        let gumbel_active = chain.and_then(|c| c.gumbel).is_some() || sync_gumbel_spec.is_some();
        let base_vocab = if qwen4_mtp_gate::mtp_vocab_prune_selected() {
            (qwen4_mtp_gate::PRUNED_VOCAB_SIZE as usize).min(ctx.config.vocab_size)
        } else if self.mtp_vocab_size == 0 {
            ctx.config.vocab_size
        } else {
            (self.mtp_vocab_size as usize).min(ctx.config.vocab_size)
        };
        let vocab = qwen4_mtp_gate::mtp_draft_vocab_override_for(gumbel_active)
            .map_or(base_vocab, |cap| base_vocab.min(cap));
        let logits = ctx.buffers.logits();
        ops::w4a16_decode_gemv(
            ctx.gpu,
            self.w4a16_gemv_k,
            self.w4a16_gemv_sw_k,
            self.gemv_sw,
            sample_hidden,
            &self.lm_head_nvfp4,
            logits,
            vocab as u32,
            h as u32,
            stream,
        )?;
        if let Some(c) = chain {
            if let Some((spec, abs_pos)) = c.gumbel {
                gumbel_draft::select_chain_gumbel_step(
                    ctx.gpu,
                    self.mtp_gumbel_argmax_k,
                    logits,
                    DevicePtr::NULL,
                    c.argmax_out,
                    c.prob_out,
                    vocab,
                    spec,
                    abs_pos,
                    stream,
                )?;
            } else if let Some(prob_out) = c.prob_out {
                spark_runtime::kernel_args::KernelLaunch::new(ctx.gpu, self.argmax_prob_k)
                    .grid([1, 1, 1])
                    .block([1024, 1, 1])
                    .arg_ptr(logits)
                    .arg_ptr(c.argmax_out)
                    .arg_ptr(prob_out)
                    .arg_u32(vocab as u32)
                    .launch(stream)?;
            } else {
                ops::argmax_bf16(
                    ctx.gpu,
                    self.argmax_k,
                    logits,
                    c.argmax_out,
                    vocab as u32,
                    stream,
                )?;
            }
            state.seq_len += 1;
            return Ok(0);
        }
        let out = ctx.buffers.scratch();
        let token_id = if let Some(mask) = grammar_bitmask {
            let mut bytes = vec![0u8; vocab * 2];
            ctx.gpu.copy_d2h_on_stream(logits, &mut bytes, stream)?;
            let mut best = None::<(u32, f32)>;
            for candidate in 0..vocab {
                if candidate / 32 >= mask.len()
                    || (mask[candidate / 32] & (1i32 << (candidate % 32))) == 0
                {
                    continue;
                }
                let bits = u16::from_le_bytes([bytes[candidate * 2], bytes[candidate * 2 + 1]]);
                let value = f32::from_bits((bits as u32) << 16);
                if best.is_none_or(|(_, old)| value > old) {
                    best = Some((candidate as u32, value));
                }
            }
            best.map_or(0, |(candidate, _)| candidate)
        } else if let Some(spec) = sync_gumbel_spec {
            gumbel_draft::select_sync_gumbel_step(
                ctx.gpu,
                self.mtp_gumbel_argmax_k,
                logits,
                out,
                vocab,
                spec,
                (position + 1) as u64,
                stream,
            )?
        } else {
            ops::argmax_bf16(ctx.gpu, self.argmax_k, logits, out, vocab as u32, stream)?;
            let mut bytes = [0u8; 4];
            ctx.gpu.copy_d2h_on_stream(out, &mut bytes, stream)?;
            u32::from_le_bytes(bytes)
        };
        state.seq_len += 1;
        Ok(token_id)
    }

    /// True when the exact multi-row FC and inner attention/MoE kernels cover `rows`.
    pub(super) fn forward_rows_admits(&self, rows: usize) -> bool {
        (2..=8).contains(&rows)
            && self
                .w4a16_exact_fc_kernels
                .route_for_rows(rows as u32)
                .is_some()
            && self
                .w4a16_exact_fc_kernels
                .route_for_rows((rows * 4) as u32)
                .is_some()
            && crate::layers::qwen4_hybrid_rows(rows)
    }

    /// True when cross-sequence batched MTP (`2..=64` sequences) has exact
    /// FC/LM-head slabs, MoE route, and workspace capacity for `rows`.
    pub(super) fn propose_batch_rows_admits(&self, rows: usize, ctx: &ForwardContext) -> bool {
        if !(2..=64).contains(&rows) {
            return false;
        }
        let fc_ok = self.fc_dual_broadcast_rows_k.0 != 0
            || (self
                .w4a16_exact_fc_kernels
                .route_for_rows(rows as u32)
                .is_some()
                && self
                    .w4a16_exact_fc_kernels
                    .route_for_rows((rows * 4) as u32)
                    .is_some());
        let lm_ok = self
            .w4a16_exact_fc_kernels
            .route_for_rows(rows as u32)
            .is_some()
            || (mtp_lm_head_tc_pipe_enabled() && rows >= 8 && self.w4a16_gemm_pipe_k.0 != 0);
        let moe_ok =
            rows > crate::layers::QWEN4_HYBRID_MAX_ROWS || crate::layers::qwen4_hybrid_rows(rows);
        let h_bytes = rows * ctx.config.hidden_size * 2;
        let r_bytes = rows * ctx.config.residual_width() * 2;
        let sizes = ctx.buffers.sizes();
        let arena_ok = h_bytes <= sizes.ssm_qkvz
            && h_bytes <= sizes.ssm_deinterleaved
            && h_bytes <= sizes.gate_logits
            && r_bytes <= sizes.residual
            && r_bytes <= sizes.hidden_states
            && (self.fc_dual_broadcast_rows_k.0 != 0 || h_bytes <= sizes.qkv_output);
        fc_ok && lm_ok && moe_ok && arena_ok
    }

    /// Batched multi-row MTP forward (`M in 2..=8`), bit-for-bit identical to
    /// `M` sequential `forward_one` calls: gathers `M` embeddings, runs exact
    /// multi-row `fc_embedding` and `fc_hidden`, executes `decode_qwen4_batched`
    /// with token-ordered causal attention and KV writes, and optionally runs
    /// `final_mixer` + `lm_head_nvfp4` + `argmax` on the last row `M - 1`.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn forward_rows(
        &self,
        tokens: &[u32],
        target_hiddens: &[DevicePtr],
        positions: &[usize],
        meta_base: DevicePtr,
        state: &mut Qwen4MtpState,
        ctx: &ForwardContext,
        stream: u64,
        grammar_bitmask: Option<&[i32]>,
        emit_last_draft: bool,
        chain_out: Option<(
            DevicePtr,
            Option<DevicePtr>,
            Option<(spark_runtime::sampler::gumbel::GumbelDraftSpec, u64)>,
        )>,
    ) -> Result<u32> {
        let rows = tokens.len();
        ensure!(
            (2..=8).contains(&rows) && target_hiddens.len() == rows && positions.len() == rows,
            "Qwen4 MTP forward_rows requires 2..=8 matched rows, got {rows}"
        );
        for &pos in positions {
            rotary_axes(
                &state.rotary_positions,
                pos,
                ctx.config.mrope_interleaved,
                self.mrope_kernel_available,
                ctx.config.yarn_factor,
            )?;
        }
        let h = ctx.config.hidden_size;
        let r = ctx.config.residual_width();
        let hc = ctx.config.hc_count;
        let eps = ctx.config.rms_norm_eps as f32;
        let row_bytes = h * 2;
        let r_bytes = r * 2;
        ensure!(hc == 4, "Qwen4 native MTP expects four hidden streams");

        let normed_embed = ctx.buffers.ssm_deinterleaved();
        let normed_hidden = ctx.buffers.residual();
        if self.embed_dual_rms_norm_rows_packed_k.0 != 0 && h >= 1024 && r >= 1024 && rows <= 8 {
            let mut run_start = 0usize;
            while run_start < rows {
                let mut run_len = 1usize;
                while run_start + run_len < rows
                    && target_hiddens[run_start + run_len]
                        == target_hiddens[run_start].offset(run_len * r_bytes)
                {
                    run_len += 1;
                }
                let tok_at = |idx: usize| -> u64 {
                    tokens
                        .get(run_start + idx)
                        .copied()
                        .filter(|_| idx < run_len)
                        .unwrap_or(0) as u64
                };
                let tok_pack01 = tok_at(0) | (tok_at(1) << 32);
                let tok_pack23 = tok_at(2) | (tok_at(3) << 32);
                let tok_pack45 = tok_at(4) | (tok_at(5) << 32);
                let tok_pack67 = tok_at(6) | (tok_at(7) << 32);
                spark_runtime::kernel_args::KernelLaunch::new(
                    ctx.gpu,
                    self.embed_dual_rms_norm_rows_packed_k,
                )
                .grid([run_len as u32, 2, 1])
                .block([1024, 1, 1])
                .arg_ptr(self.embed_tokens.weight)
                .arg_u64(tok_pack01)
                .arg_u64(tok_pack23)
                .arg_u64(tok_pack45)
                .arg_u64(tok_pack67)
                .arg_u32(ctx.config.vocab_size as u32)
                .arg_ptr(self.pre_fc_norm_embedding.weight)
                .arg_ptr(normed_embed.offset(run_start * row_bytes))
                .arg_ptr(target_hiddens[run_start])
                .arg_ptr(self.pre_fc_norm_hidden.weight)
                .arg_ptr(normed_hidden.offset(run_start * r_bytes))
                .arg_u32(h as u32)
                .arg_u32(r as u32)
                .arg_f32(eps)
                .arg_ptr(DevicePtr(0))
                .launch(stream)?;
                run_start += run_len;
            }
        } else {
            let embed = ctx.buffers.ssm_qkvz();
            for (j, &token) in tokens.iter().enumerate() {
                let embed_src = self.embed_tokens.weight.offset(token as usize * row_bytes);
                ctx.gpu.copy_d2d_async(
                    embed_src,
                    embed.offset(j * row_bytes),
                    row_bytes,
                    stream,
                )?;
            }
            if self.pre_fc_dual_norm_k.0 != 0 && h >= 1024 && r >= 1024 {
                let mut run_start = 0usize;
                while run_start < rows {
                    let mut run_len = 1usize;
                    while run_start + run_len < rows
                        && target_hiddens[run_start + run_len]
                            == target_hiddens[run_start].offset(run_len * r_bytes)
                    {
                        run_len += 1;
                    }
                    spark_runtime::kernel_args::KernelLaunch::new(ctx.gpu, self.pre_fc_dual_norm_k)
                        .grid([run_len as u32, 2, 1])
                        .block([1024, 1, 1])
                        .arg_ptr(embed.offset(run_start * row_bytes))
                        .arg_ptr(self.pre_fc_norm_embedding.weight)
                        .arg_ptr(normed_embed.offset(run_start * row_bytes))
                        .arg_ptr(target_hiddens[run_start])
                        .arg_ptr(self.pre_fc_norm_hidden.weight)
                        .arg_ptr(normed_hidden.offset(run_start * r_bytes))
                        .arg_u32(h as u32)
                        .arg_u32(r as u32)
                        .arg_f32(eps)
                        .arg_ptr(DevicePtr(0))
                        .launch(stream)?;
                    run_start += run_len;
                }
            } else {
                ops::rms_norm(
                    ctx.gpu,
                    self.rms_norm_k,
                    embed,
                    &self.pre_fc_norm_embedding,
                    normed_embed,
                    rows as u32,
                    h as u32,
                    eps,
                    stream,
                )?;
                let mut run_start = 0usize;
                while run_start < rows {
                    let mut run_len = 1usize;
                    while run_start + run_len < rows
                        && target_hiddens[run_start + run_len]
                            == target_hiddens[run_start].offset(run_len * r_bytes)
                    {
                        run_len += 1;
                    }
                    ops::rms_norm(
                        ctx.gpu,
                        self.rms_norm_k,
                        target_hiddens[run_start],
                        &self.pre_fc_norm_hidden,
                        normed_hidden.offset(run_start * r_bytes),
                        run_len as u32,
                        r as u32,
                        eps,
                        stream,
                    )?;
                    run_start += run_len;
                }
            }
        }

        let hidden = ctx.buffers.hidden_states();
        if self.fc_dual_broadcast_rows_k.0 != 0 {
            spark_runtime::kernel_args::KernelLaunch::new(ctx.gpu, self.fc_dual_broadcast_rows_k)
                .grid([(h as u32).div_ceil(4), rows as u32, 1])
                .block([256, 1, 1])
                .arg_ptr(normed_embed)
                .arg_ptr(self.fc_embedding.weight)
                .arg_ptr(self.fc_embedding.weight_scale)
                .arg_f32(self.fc_embedding.weight_scale_2)
                .arg_ptr(normed_hidden)
                .arg_ptr(self.fc_hidden.weight)
                .arg_ptr(self.fc_hidden.weight_scale)
                .arg_f32(self.fc_hidden.weight_scale_2)
                .arg_ptr(hidden)
                .arg_u32(h as u32)
                .arg_u32(h as u32)
                .launch(stream)?;
        } else {
            let embed_proj = ctx.buffers.qkv_output();
            ops::w4a16_gemv_batch_logits_exact_with(
                ctx.gpu,
                self.w4a16_exact_fc_kernels,
                normed_embed,
                &self.fc_embedding,
                embed_proj,
                rows as u32,
                h as u32,
                h as u32,
                stream,
                false,
            )?;
            ops::w4a16_gemv_batch_logits_exact_with(
                ctx.gpu,
                self.w4a16_exact_fc_kernels,
                normed_hidden,
                &self.fc_hidden,
                hidden,
                (rows * hc) as u32,
                h as u32,
                h as u32,
                stream,
                false,
            )?;
            if self.residual_add_broadcast_hc_k.0 != 0 {
                spark_runtime::kernel_args::KernelLaunch::new(
                    ctx.gpu,
                    self.residual_add_broadcast_hc_k,
                )
                .grid([(h as u32).div_ceil(256), rows as u32, 1])
                .block([256, 1, 1])
                .arg_ptr(hidden)
                .arg_ptr(embed_proj)
                .arg_u32(h as u32)
                .arg_u32(hc as u32)
                .launch(stream)?;
            } else {
                for j in 0..rows {
                    let embed_row = embed_proj.offset(j * row_bytes);
                    for branch in 0..hc {
                        let hidden_branch = hidden.offset((j * hc + branch) * row_bytes);
                        ops::residual_add(
                            ctx.gpu,
                            self.residual_add_k,
                            hidden_branch,
                            embed_row,
                            h as u32,
                            stream,
                        )?;
                    }
                }
            }
        }

        let mut kv_cache = self.kv_cache.lock();
        let packed = self.batch_metadata(&mut kv_cache, state, state.seq_len, positions)?;
        self.copy_h2d_group_fast(
            ctx.gpu,
            &[HostToDeviceCopy::new(&packed, meta_base)],
            stream,
        )?;
        let (positions_h, positions_w) = if state.rotary_positions.is_identity() {
            (meta_base, meta_base)
        } else {
            (meta_base.offset(64), meta_base.offset(128))
        };
        let max_blocks_per_seq = if rows > 0 {
            ((packed.len() - 768) / (rows * 4)) as u32
        } else {
            state.block_table.len() as u32
        };
        let metadata = AttnMetadataDev {
            qwen4_qsa_required: self.mtp_qsa_active(),
            positions: meta_base,
            positions_h,
            positions_w,
            slot: meta_base.offset(256),
            seq_len: meta_base.offset(512),
            block_table: meta_base.offset(768),
            max_blocks_per_seq,
            num_seqs: rows as u32,
        };
        let layer_ctx = ForwardContext {
            attn_metadata: Some(metadata),
            comm: None,
            graph_capture: false,
            ..*ctx
        };
        if mtp_kv_only_replay_enabled() {
            self.layer.decode_qwen4_batched_mtp(
                hidden,
                ctx.buffers.residual(),
                rows,
                emit_last_draft,
                state.layer_state.as_mut(),
                &mut kv_cache,
                state.seq_len,
                &mut state.block_table,
                &mut state.disk_block_ids,
                &mut state.disk_last_offloaded_per_layer,
                &layer_ctx,
                stream,
            )?;
        } else {
            self.layer.decode_qwen4_batched(
                hidden,
                ctx.buffers.residual(),
                rows,
                state.layer_state.as_mut(),
                &mut kv_cache,
                state.seq_len,
                &mut state.block_table,
                &mut state.disk_block_ids,
                &mut state.disk_last_offloaded_per_layer,
                DevicePtr::NULL,
                DevicePtr::NULL,
                0,
                0,
                &layer_ctx,
                stream,
            )?;
        }
        drop(kv_cache);

        if !emit_last_draft {
            state.seq_len += rows;
            return Ok(0);
        }

        let last_hidden = hidden.offset((rows - 1) * r_bytes);
        let (sample_hidden, inject) = self.final_mixer.prepare_decode(
            last_hidden,
            ctx.buffers.residual(),
            ctx.buffers,
            ctx.gpu,
            eps,
            stream,
        )?;
        debug_assert!(inject.is_none());
        let sync_gumbel_spec = if chain_out.is_none() && grammar_bitmask.is_none() {
            positions
                .last()
                .and_then(|&last_pos| state.resolve_gumbel_spec(last_pos))
        } else {
            None
        };
        let gumbel_active =
            chain_out.and_then(|(_, _, gumbel)| gumbel).is_some() || sync_gumbel_spec.is_some();
        let base_vocab = if qwen4_mtp_gate::mtp_vocab_prune_selected() {
            (qwen4_mtp_gate::PRUNED_VOCAB_SIZE as usize).min(ctx.config.vocab_size)
        } else if self.mtp_vocab_size == 0 {
            ctx.config.vocab_size
        } else {
            (self.mtp_vocab_size as usize).min(ctx.config.vocab_size)
        };
        let vocab = qwen4_mtp_gate::mtp_draft_vocab_override_for(gumbel_active)
            .map_or(base_vocab, |cap| base_vocab.min(cap));
        let logits = ctx.buffers.logits();
        ops::w4a16_decode_gemv(
            ctx.gpu,
            self.w4a16_gemv_k,
            self.w4a16_gemv_sw_k,
            self.gemv_sw,
            sample_hidden,
            &self.lm_head_nvfp4,
            logits,
            vocab as u32,
            h as u32,
            stream,
        )?;
        if let Some((argmax_out, prob_out, gumbel)) = chain_out {
            if let Some((spec, abs_pos)) = gumbel {
                gumbel_draft::select_chain_gumbel_step(
                    ctx.gpu,
                    self.mtp_gumbel_argmax_k,
                    logits,
                    DevicePtr::NULL,
                    argmax_out,
                    prob_out,
                    vocab,
                    spec,
                    abs_pos,
                    stream,
                )?;
            } else if let Some(prob_out) = prob_out {
                spark_runtime::kernel_args::KernelLaunch::new(ctx.gpu, self.argmax_prob_k)
                    .grid([1, 1, 1])
                    .block([1024, 1, 1])
                    .arg_ptr(logits)
                    .arg_ptr(argmax_out)
                    .arg_ptr(prob_out)
                    .arg_u32(vocab as u32)
                    .launch(stream)?;
            } else {
                ops::argmax_bf16(
                    ctx.gpu,
                    self.argmax_k,
                    logits,
                    argmax_out,
                    vocab as u32,
                    stream,
                )?;
            }
            state.seq_len += rows;
            return Ok(0);
        }
        let out = ctx.buffers.scratch();
        let token_id = if let Some(mask) = grammar_bitmask {
            let mut bytes = vec![0u8; vocab * 2];
            ctx.gpu.copy_d2h_on_stream(logits, &mut bytes, stream)?;
            let mut best = None::<(u32, f32)>;
            for candidate in 0..vocab {
                if candidate / 32 >= mask.len()
                    || (mask[candidate / 32] & (1i32 << (candidate % 32))) == 0
                {
                    continue;
                }
                let bits = u16::from_le_bytes([bytes[candidate * 2], bytes[candidate * 2 + 1]]);
                let value = f32::from_bits((bits as u32) << 16);
                if best.is_none_or(|(_, old)| value > old) {
                    best = Some((candidate as u32, value));
                }
            }
            best.map_or(0, |(candidate, _)| candidate)
        } else if let Some(spec) = sync_gumbel_spec {
            gumbel_draft::select_sync_gumbel_step(
                ctx.gpu,
                self.mtp_gumbel_argmax_k,
                logits,
                out,
                vocab,
                spec,
                spec.step0_abs_pos,
                stream,
            )?
        } else {
            ops::argmax_bf16(ctx.gpu, self.argmax_k, logits, out, vocab as u32, stream)?;
            let mut bytes = [0u8; 4];
            ctx.gpu.copy_d2h_on_stream(out, &mut bytes, stream)?;
            u32::from_le_bytes(bytes)
        };
        state.seq_len += rows;
        Ok(token_id)
    }

    /// Attention metadata for the MTP step writing KV row `seq_len` at rotary
    /// `position`, allocating its block if needed: position at +0, KV slot at
    /// +256, sequence length at +512, block table at +768.
    pub(super) fn step_metadata(
        &self,
        kv_cache: &mut PagedKvCache,
        state: &mut Qwen4MtpState,
        seq_len: usize,
        position: usize,
    ) -> Result<Vec<u8>> {
        let bs = kv_cache.block_size();
        let blocks_needed = seq_len / bs + 1;
        while state.block_table.len() < blocks_needed {
            state.block_table.push(kv_cache.alloc_block()?);
        }
        let physical = state.block_table[seq_len / bs];
        let slot = physical as i64 * bs as i64 + (seq_len % bs) as i64;
        let draft_blocks: Vec<u32> = if mtp_bounded_blocks_selected()
            && !self.mtp_qsa_active()
            && state.block_table.len() > 192
        {
            let mut b = Vec::with_capacity(192);
            b.extend_from_slice(&state.block_table[..64]);
            b.extend_from_slice(&state.block_table[state.block_table.len() - 128..]);
            b
        } else {
            state.block_table.clone()
        };
        let bt_bytes = draft_blocks.len() * 4;
        let mut packed = vec![0u8; 768 + bt_bytes];
        write_rotary_metadata(&state.rotary_positions, position, &mut packed)?;
        packed[256..264].copy_from_slice(&slot.to_le_bytes());
        let effective_seq_len = (seq_len + 1).min(draft_blocks.len() * bs) as i32;
        packed[512..516].copy_from_slice(&effective_seq_len.to_le_bytes());
        for (i, block) in draft_blocks.iter().enumerate() {
            packed[768 + i * 4..772 + i * 4].copy_from_slice(&(*block as i32).to_le_bytes());
        }
        Ok(packed)
    }

    /// Multi-row attention metadata matching `Qwen3AttentionLayer::decode_qwen4_batched_inner`:
    /// `positions` at `+0..4*M`, `positions_h` at `+64..64+4*M` (when non-identity),
    /// `positions_w` at `+128..128+4*M` (when non-identity), `slot` at `+256..256+8*M`,
    /// `seq_len` at `+512..512+4*M`, and `M` copies of `state.block_table` at `+768`.
    pub(super) fn batch_metadata(
        &self,
        kv_cache: &mut PagedKvCache,
        state: &mut Qwen4MtpState,
        seq_len: usize,
        positions: &[usize],
    ) -> Result<Vec<u8>> {
        let rows = positions.len();
        let bs = kv_cache.block_size();
        let last_seq_len = seq_len + rows - 1;
        let blocks_needed = last_seq_len / bs + 1;
        while state.block_table.len() < blocks_needed {
            state.block_table.push(kv_cache.alloc_block()?);
        }
        let draft_blocks: Vec<u32> = if mtp_bounded_blocks_selected()
            && !self.mtp_qsa_active()
            && state.block_table.len() > 192
        {
            let mut b = Vec::with_capacity(192);
            b.extend_from_slice(&state.block_table[..64]);
            b.extend_from_slice(&state.block_table[state.block_table.len() - 128..]);
            b
        } else {
            state.block_table.clone()
        };
        let mb = draft_blocks.len();
        let bt_row_bytes = mb * 4;
        let mut packed = vec![0u8; 768 + rows * bt_row_bytes];
        for (i, block) in draft_blocks.iter().enumerate() {
            packed[768 + i * 4..772 + i * 4].copy_from_slice(&(*block as i32).to_le_bytes());
        }
        for (row, &position) in positions.iter().enumerate() {
            let [t, h, w] = state.rotary_positions.position(position)?;
            packed[row * 4..row * 4 + 4].copy_from_slice(&t.to_le_bytes());
            if !state.rotary_positions.is_identity() {
                packed[64 + row * 4..68 + row * 4].copy_from_slice(&h.to_le_bytes());
                packed[128 + row * 4..132 + row * 4].copy_from_slice(&w.to_le_bytes());
            }
            let r_seq = seq_len + row;
            let physical = state.block_table[r_seq / bs];
            let slot = physical as i64 * bs as i64 + (r_seq % bs) as i64;
            packed[256 + row * 8..264 + row * 8].copy_from_slice(&slot.to_le_bytes());
            let effective_seq_len = (r_seq + 1).min(draft_blocks.len() * bs) as i32;
            packed[512 + row * 4..516 + row * 4].copy_from_slice(&effective_seq_len.to_le_bytes());
            if row > 0 {
                packed.copy_within(768..768 + bt_row_bytes, 768 + row * bt_row_bytes);
            }
        }
        Ok(packed)
    }

    /// Greedy draft chain with one metadata upload and one readback: each
    /// step's argmax stays on the device and feeds the next step's embedding
    /// gather, so the host never waits between drafts. Same arithmetic per
    /// step as the one-step path. With an early-stop gate the host reads each
    /// draft's probability before launching the next step and stops at the
    /// first one below the threshold. Returns (kept drafts, computed steps).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn propose_chain(
        &self,
        last_token: u32,
        target_hidden: DevicePtr,
        position: usize,
        num_drafts: usize,
        policy: GatePolicy,
        first_batch: Option<(&[u32], &[DevicePtr], &[usize])>,
        state: &mut Qwen4MtpState,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<(Vec<u32>, usize)> {
        static DEVICE_PROPOSE_LOGGED: std::sync::Once = std::sync::Once::new();
        DEVICE_PROPOSE_LOGGED.call_once(|| {
            tracing::info!(
                "mtp_propose: MTP_DEVICE_PROPOSE_CHAIN_ENGAGED chained_steps=1..3 host_roundtrips=0"
            );
        });
        validate_rotary_span(
            &state.rotary_positions,
            position,
            num_drafts,
            ctx.config.mrope_interleaved,
            self.mrope_kernel_available,
            ctx.config.yarn_factor,
        )?;
        let scratch = ctx.buffers.scratch();
        let replay_prefix = first_batch.map_or(0, |(t, _, _)| t.len() - 1);
        let first_seq_len = state.seq_len + replay_prefix;
        let bs = {
            let mut kv_cache = self.kv_cache.lock();
            let bs = kv_cache.block_size();
            let last = first_seq_len + num_drafts - 1;
            let blocks_needed = last / bs + 1;
            while state.block_table.len() < blocks_needed {
                state.block_table.push(kv_cache.alloc_block()?);
            }
            bs
        };
        let draft_blocks: Vec<u32> = if mtp_bounded_blocks_selected()
            && !self.mtp_qsa_active()
            && state.block_table.len() > 192
        {
            let mut b = Vec::with_capacity(192);
            b.extend_from_slice(&state.block_table[..64]);
            b.extend_from_slice(&state.block_table[state.block_table.len() - 128..]);
            b
        } else {
            state.block_table.clone()
        };
        let bt_bytes = draft_blocks.len() * 4;
        let stride = (768 + bt_bytes).next_multiple_of(256);
        let batch_meta_bytes = first_batch.map_or(0, |(t, _, _)| {
            (768 + t.len() * bt_bytes).next_multiple_of(256)
        });
        ensure!(
            CHAIN_META_OFFSET + num_drafts * stride + batch_meta_bytes
                <= ctx.buffers.sizes().scratch
                && CHAIN_ARGMAX_OFFSET + num_drafts * 4 <= CHAIN_TOKEN_OFFSET,
            "Qwen4 MTP draft chain of {num_drafts} steps does not fit the scratch layout"
        );
        let mut packed_guard = self.packed_meta_scratch.lock();
        let packed = prepare_packed_meta_scratch(&mut packed_guard, num_drafts * stride);
        for (i, block) in draft_blocks.iter().enumerate() {
            packed[768 + i * 4..772 + i * 4].copy_from_slice(&(*block as i32).to_le_bytes());
        }
        for step in 0..num_drafts {
            let s_off = step * stride;
            let s_seq = first_seq_len + step;
            let physical = state.block_table[s_seq / bs];
            let slot = physical as i64 * bs as i64 + (s_seq % bs) as i64;
            write_rotary_metadata(
                &state.rotary_positions,
                position + step,
                &mut packed[s_off..],
            )?;
            packed[s_off + 256..s_off + 264].copy_from_slice(&slot.to_le_bytes());
            let effective_seq_len = (s_seq + 1).min(draft_blocks.len() * bs) as i32;
            packed[s_off + 512..s_off + 516].copy_from_slice(&effective_seq_len.to_le_bytes());
            if step > 0 {
                packed.copy_within(768..768 + bt_bytes, s_off + 768);
            }
        }
        let meta_base = scratch.offset(CHAIN_META_OFFSET);
        self.copy_h2d_group_fast(ctx.gpu, &[HostToDeviceCopy::new(packed, meta_base)], stream)?;
        drop(packed_guard);

        let argmax_base = scratch.offset(CHAIN_ARGMAX_OFFSET);
        let gumbel_spec = state.resolve_gumbel_spec(position);
        let conf_gate = resolve_chain_confidence_gate(
            policy,
            policy.wants_probabilities(),
            policy.early_threshold(),
            self.argmax_prob_k.0 != 0,
            gumbel_spec.is_some(),
        );
        // Zero-training Native-MTP DLoop (`ATLAS_MTP_DLOOP=1`, default OFF):
        // strictly dense-only (`num_experts <= 1`), `C <= 2`, extending
        // `K = 5 -> K = 9` when `c_1 = sum ln(p_i) >= -0.5`. MoE targets
        // (e.g. Flash-Next 512E) are refused fail-closed before any extra GPU work.
        let dloop_enabled = crate::layers::mtp_dloop::mtp_dloop_enabled();
        let dloop_topology =
            crate::layers::mtp_dloop::TargetFfnTopology::from_num_experts(ctx.config.num_experts);
        let dloop_cohort = crate::layers::mtp_dloop::active_cohort_context();
        let dloop_stage1_eligible = if dloop_enabled && gumbel_spec.is_none() {
            let precheck = match dloop_cohort {
                Some(cohort) => crate::layers::mtp_dloop::precheck_stage1_admission(
                    &crate::layers::mtp_dloop::MtpDloopStage1PrecheckInput {
                        enabled: true,
                        topology: dloop_topology,
                        live_seqs: cohort.live_seqs,
                        stage1_drafts: num_drafts,
                        max_verify_k: cohort.max_verify_k,
                    },
                ),
                None => Err(
                    crate::layers::mtp_dloop::MtpDloopRefusal::ConcurrencyTooHigh {
                        live_seqs: 0,
                        max_concurrency: crate::layers::mtp_dloop::MTP_DLOOP_MAX_CONCURRENCY,
                    },
                ),
            };
            if let Err(crate::layers::mtp_dloop::MtpDloopRefusal::MoeTargetRefused {
                num_experts,
            }) = precheck
            {
                static QWEN4_MOE_DLOOP_REFUSED: std::sync::Once = std::sync::Once::new();
                QWEN4_MOE_DLOOP_REFUSED.call_once(|| {
                    tracing::info!(
                        num_experts,
                        "QWEN4_MTP_DLOOP_REFUSED reason=moe_target (ATLAS_MTP_DLOOP=1 is dense-only)"
                    );
                });
            }
            precheck.is_ok() && self.argmax_prob_k.0 != 0
        } else {
            false
        };
        let want_prob = conf_gate.want_prob || dloop_stage1_eligible;
        ensure!(
            !want_prob || num_drafts * 4 <= CHAIN_PROB_SPAN,
            "Qwen4 MTP draft confidence supports at most {} drafts",
            CHAIN_PROB_SPAN / 4
        );
        let early = conf_gate.early;
        // A width floor (QSA context) lets the chain stop early only where it
        // would keep zero/one draft; otherwise every draft is computed and kept.
        let min_width = qwen4_mtp_gate::chain_min_width();
        let prob_base = scratch.offset(CHAIN_PROB_OFFSET);
        let r_bytes = ctx.config.residual_width() * 2;
        let mut hidden = target_hidden;
        let mut computed = 0usize;
        let mut p_cum = 1.0f32;
        let qsa_mode = mtp_qsa_mode();
        let has_qsa = self.layer.has_qsa_side_cache();
        let chain_max_blocks = draft_blocks.len() as u32;
        for step in 0..num_drafts {
            let is_last_step = step + 1 == num_drafts;
            let (qsa_required, reuse_action) =
                draft_step_qsa_plan(qsa_mode, has_qsa, first_seq_len, step, num_drafts);
            let _reuse_guard = reuse_action.map(|action| match action {
                DraftStepReuseAction::SnapshotStep0 => {
                    crate::layers::qwen4_qsa::DraftSelectionReuseGuard::snapshot_step0()
                }
                DraftStepReuseAction::ReuseStep {
                    base_seq_len,
                    step_offset,
                    max_drafts,
                } => crate::layers::qwen4_qsa::DraftSelectionReuseGuard::reuse_step(
                    base_seq_len,
                    step_offset,
                    max_drafts,
                ),
            });
            let chain = ChainStep {
                meta: meta_base.offset(step * stride),
                max_blocks_per_seq: chain_max_blocks,
                device_token: (step > 0).then(|| argmax_base.offset((step - 1) * 4)),
                argmax_out: argmax_base.offset(step * 4),
                prob_out: want_prob.then(|| prob_base.offset(step * 4)),
                gumbel: gumbel_spec.map(|spec| (spec, spec.step0_abs_pos + step as u64)),
            };
            if step == 0
                && let Some((b_tokens, b_hiddens, b_positions)) = first_batch
            {
                self.forward_rows(
                    b_tokens,
                    b_hiddens,
                    b_positions,
                    meta_base.offset(num_drafts * stride),
                    state,
                    ctx,
                    stream,
                    None,
                    true,
                    Some((chain.argmax_out, chain.prob_out, chain.gumbel)),
                )?;
                computed = 1;
                if !is_last_step {
                    hidden = ctx.buffers.hidden_states().offset(replay_prefix * r_bytes);
                }
            } else {
                self.forward_one(
                    last_token,
                    hidden,
                    position + step,
                    state,
                    ctx,
                    stream,
                    None,
                    true,
                    qsa_required,
                    Some(chain),
                )?;
                computed = step + 1;
                if !is_last_step {
                    hidden = ctx.buffers.hidden_states();
                }
            }
            static SYNC_MID_CHAIN_ENV: std::sync::OnceLock<Option<String>> =
                std::sync::OnceLock::new();
            let sync_mid_chain = should_sync_mid_chain(
                SYNC_MID_CHAIN_ENV
                    .get_or_init(|| std::env::var("ATLAS_QWEN4_MTP_CONF_EARLY_SYNC").ok())
                    .as_deref(),
                num_drafts,
            );
            if sync_mid_chain
                && !is_last_step
                && let Some(threshold) = early
            {
                let mut pb = [0u8; 4];
                ctx.gpu
                    .copy_d2h_on_stream(prob_base.offset(step * 4), &mut pb, stream)?;
                let p = f32::from_le_bytes(pb);
                p_cum *= p;
                let keep_opt = if gumbel_spec.is_some() {
                    qwen4_mtp_gate::early_stop_keep_step_gumbel(
                        step,
                        p,
                        p_cum,
                        qwen4_mtp_gate::gumbel_cum_threshold(),
                    )
                } else {
                    qwen4_mtp_gate::early_stop_keep_step(step, p, p_cum, threshold)
                };
                if let Some(keep) = keep_opt
                    && min_width.is_none_or(|w| keep < 2 || keep + 1 >= w)
                {
                    break;
                }
            }
        }
        let mut bytes = vec![0u8; computed * 4];
        let mut pbytes = vec![0u8; if want_prob { computed * 4 } else { 0 }];
        if want_prob {
            ctx.gpu.copy_d2h_pair_on_stream(
                argmax_base,
                &mut bytes,
                prob_base,
                &mut pbytes,
                stream,
            )?;
        } else {
            ctx.gpu
                .copy_d2h_on_stream(argmax_base, &mut bytes, stream)?;
        }
        let mut drafts: Vec<u32> = bytes
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        // Every computed draft (the gate may verify fewer): the committed-row
        // rewrite reads the accepted ones after the verify.
        state.last_drafts.clear();
        state.last_drafts.extend_from_slice(&drafts);
        if want_prob {
            let probs: Vec<f32> = pbytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect();
            let keep = if conf_gate.apply_truncate {
                let keep = if gumbel_spec.is_some() {
                    qwen4_mtp_gate::confident_prefix_gumbel(
                        &probs,
                        qwen4_mtp_gate::gumbel_cum_threshold(),
                    )
                } else {
                    policy
                        .effective_base_threshold()
                        .map_or(drafts.len(), |t| confident_prefix(&probs, t))
                };
                min_width.map_or(keep, |w| qwen4_mtp_gate::floor_keep(keep, computed, w))
            } else {
                drafts.len()
            };
            if policy.log {
                tracing::info!(
                    "MTP_DRAFT_CONF pos={position} n={num_drafts} computed={computed} \
                     drafts={drafts:?} p={probs:.4?} keep={keep}"
                );
            }
            if conf_gate.apply_truncate {
                drafts.truncate(keep);
            }
            if dloop_stage1_eligible
                && drafts.len() == num_drafts
                && computed == num_drafts
                && let Some(cohort) = dloop_cohort
                && let Some(&stage1_last_token) = drafts.last()
            {
                let decision = crate::layers::mtp_dloop::evaluate_stage2_extension(
                    &crate::layers::mtp_dloop::MtpDloopInput {
                        enabled: dloop_enabled,
                        topology: dloop_topology,
                        live_seqs: cohort.live_seqs,
                        stage1_drafts: num_drafts,
                        max_verify_k: cohort.max_verify_k,
                        stage1_probs: &probs,
                        threshold: crate::layers::mtp_dloop::mtp_dloop_thresh(),
                    },
                );
                if let crate::layers::mtp_dloop::MtpDloopDecision::ExtendStage2 {
                    extra_drafts,
                    total_drafts,
                } = decision
                {
                    static QWEN4_DLOOP_ENGAGED: std::sync::Once = std::sync::Once::new();
                    QWEN4_DLOOP_ENGAGED.call_once(|| {
                        tracing::info!(
                            stage1_drafts = num_drafts,
                            extra_drafts,
                            total_drafts,
                            live_seqs = cohort.live_seqs,
                            threshold = crate::layers::mtp_dloop::mtp_dloop_thresh(),
                            "QWEN4_MTP_DLOOP_STAGE2_ENGAGED"
                        );
                    });
                    let mut stage2_drafts = Vec::with_capacity(extra_drafts);
                    let mut stage2_tok = stage1_last_token;
                    let mut stage2_hidden = ctx.buffers.hidden_states();
                    let stage2_first_seq_len = state.seq_len;
                    for s2 in 0..extra_drafts {
                        let (qsa_required, reuse_action) = draft_step_qsa_plan(
                            qsa_mode,
                            has_qsa,
                            stage2_first_seq_len,
                            s2,
                            extra_drafts,
                        );
                        let _reuse_guard = reuse_action.map(|action| match action {
                            DraftStepReuseAction::SnapshotStep0 => {
                                crate::layers::qwen4_qsa::DraftSelectionReuseGuard::snapshot_step0()
                            }
                            DraftStepReuseAction::ReuseStep {
                                base_seq_len,
                                step_offset,
                                max_drafts,
                            } => crate::layers::qwen4_qsa::DraftSelectionReuseGuard::reuse_step(
                                base_seq_len,
                                step_offset,
                                max_drafts,
                            ),
                        });
                        stage2_tok = self.forward_one(
                            stage2_tok,
                            stage2_hidden,
                            position + num_drafts + s2,
                            state,
                            ctx,
                            stream,
                            None,
                            true,
                            qsa_required,
                            None,
                        )?;
                        stage2_drafts.push(stage2_tok);
                        stage2_hidden = ctx.buffers.hidden_states();
                    }
                    drafts.extend_from_slice(&stage2_drafts);
                    state.last_drafts.extend_from_slice(&stage2_drafts);
                    computed += extra_drafts;
                }
            }
        }
        Ok((drafts, computed))
    }

    /// Collect the `(tokens, target_hiddens, positions)` tuples for a planned
    /// K7 committed-row rewrite (`ATLAS_QWEN4_MTP_RECONDITION`).
    pub(super) fn collect_replay_rows(
        &self,
        pending: PendingReplay,
        stack: Option<DevicePtr>,
        bonus_position: usize,
        state: &Qwen4MtpState,
        ctx: &ForwardContext,
    ) -> Result<Option<(Vec<u32>, Vec<DevicePtr>, Vec<usize>)>> {
        let Some(stack) = stack else {
            static NO_STACK: std::sync::Once = std::sync::Once::new();
            NO_STACK.call_once(|| {
                tracing::warn!(
                    "QWEN4_MTP_RECONDITION_REFUSED reason=no_staged_rows: the target passed no \
                     staged hidden rows; the planned rows stay unwritten (drafts only)"
                );
            });
            return Ok(None);
        };
        let stride = ctx.config.residual_width() * 2;
        let staged_rows = qwen4_mtp_rows::staged_rows_capacity();
        let mut tokens = Vec::with_capacity(pending.plan.replay_rows + 1);
        let mut hiddens = Vec::with_capacity(pending.plan.replay_rows + 1);
        let mut positions = Vec::with_capacity(pending.plan.replay_rows + 1);
        for row in qwen4_mtp_rows::replay_rows(pending.plan) {
            let token = *state.last_drafts.get(row.draft).ok_or_else(|| {
                anyhow::anyhow!(
                    "Qwen4 MTP recondition: draft {} missing ({} recorded)",
                    row.draft,
                    state.last_drafts.len()
                )
            })?;
            let position =
                qwen4_mtp_rows::replay_position(bonus_position, pending.staged_accepted, row.index)
                    .ok_or_else(|| {
                        anyhow::anyhow!(
                            "Qwen4 MTP recondition: row {} has no position before {bonus_position}",
                            row.index
                        )
                    })?;
            ensure!(
                row.hidden_row < staged_rows,
                "Qwen4 MTP recondition: hidden row {} beyond the {staged_rows} staged rows",
                row.hidden_row
            );
            tokens.push(token);
            hiddens.push(stack.offset(row.hidden_row * stride));
            positions.push(position);
        }
        let rows = tokens.len();
        static ENGAGED: std::sync::Once = std::sync::Once::new();
        ENGAGED.call_once(|| {
            tracing::info!(
                "QWEN4_MTP_RECONDITION_ENGAGED selector={} first_rows={rows} accepted={} \
                 keep={} staged_rows={staged_rows}",
                qwen4_mtp_rows::RECONDITION_SELECTOR,
                pending.accepted,
                pending.plan.keep_rows,
            );
        });
        Ok(Some((tokens, hiddens, positions)))
    }
}
