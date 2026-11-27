// SPDX-License-Identifier: AGPL-3.0-only

//! Per-token MTP forward pass.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, HostToDeviceCopy};

use super::{MtpHead, MtpProposerState, MtpQuantization, ProjectionWeight};
use crate::layer::ForwardContext;
use crate::layers::ops;

impl MtpHead {
    /// Sequentially populate the MTP head's per-sequence KV cache from the
    /// last `k` prompt-tail positions. See [`crate::speculative::DraftProposer::prefill_last_k`]
    /// for full contract.
    ///
    /// Loops `forward_kv_only`, discarding draft predictions (we want the KV
    /// cache populated, not tokens).
    /// Updates `state.seq_len` to `k` (`k - 1` under `ATLAS_MTP_REPLAY_NEXT=1`),
    /// assuming it starts from 0.
    pub(crate) fn prefill_last_k_impl(
        &self,
        tokens: &[u32],
        target_hiddens: DevicePtr,
        base_position: usize,
        state: &mut MtpProposerState,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let k = tokens.len();
        if k == 0 {
            return Ok(());
        }
        let h = ctx.config.hidden_size;
        let fp32 = if ctx.config.use_fp32_residual() { 4 } else { 2 };
        let stride = h * fp32;
        let predict_start = super::rotary::prompt_tail_start(base_position, k)?;
        self.validate_rotary_span(state, predict_start, k, ctx)?;
        let pairs = crate::speculative::mtp_replay_pairs(
            tokens,
            crate::speculative::mtp_replay_next_token(),
        );
        for (i, token) in pairs {
            let target_hidden_i = target_hiddens.offset(i * stride);
            // Position passed to forward_kv_only is the position MTP is "predicting
            // into" — i.e. one past the hidden row's position.
            let position = predict_start + i;
            self.forward_kv_only(token, target_hidden_i, position, state, ctx, stream)?;
        }
        // We populated K KV entries but did NOT consume a draft step on the
        // caller's behalf. Reset `last_num_drafted` so the next `after_verify`
        // (which trims `last_num_drafted - num_accepted`) does not over-trim
        // the prefill entries.
        state.last_num_drafted = 0;
        Ok(())
    }

    /// MTP forward pass for a single token.
    ///
    /// When `draft_embed_target` is `Some(ptr)`, the draft token's embedding
    /// is written directly to `ptr` on GPU via `embed_from_argmax`, and the
    /// token ID is stored in `self.draft_token_id_dev` for deferred readback.
    /// This eliminates the D2H sync that was previously required.
    ///
    /// When `draft_embed_target` is `None`, falls back to D2H readback.
    ///
    /// Visible to sibling modules (`mtp_multi`) so the multi-module
    /// proposer can dispatch per-draft to a different `MtpHead` while
    /// reusing the same per-token forward code path.
    pub(crate) fn forward_one(
        &self,
        token: u32,
        target_hidden: DevicePtr,
        position: usize,
        state: &mut MtpProposerState,
        ctx: &ForwardContext,
        stream: u64,
        draft_embed_target: Option<DevicePtr>,
        grammar_bitmask: Option<&[i32]>,
    ) -> Result<u32> {
        let h = ctx.config.hidden_size as u32;
        let nq = ctx.config.num_attention_heads as u32;
        let nkv = ctx.config.num_key_value_heads as u32;
        let hd = ctx.config.head_dim as u32;
        let eps = ctx.config.rms_norm_eps as f32;
        let rotary = self.rotary_axes(state, position, ctx)?;

        // 1. Embed token
        let embed_out = ctx.buffers.ssm_qkvz(); // reuse scratch
        let row_bytes = h as usize * 2;
        let src = self.embed_tokens.weight.offset(token as usize * row_bytes);
        ctx.gpu.copy_d2d_async(src, embed_out, row_bytes, stream)?;

        // 2. RMSNorm embedding and hidden separately
        let normed_embed = ctx.buffers.ssm_deinterleaved();
        ops::rms_norm(
            ctx.gpu,
            self.rms_norm_k,
            embed_out,
            &self.pre_fc_norm_embedding,
            normed_embed,
            1,
            h,
            eps,
            stream,
        )?;

        let normed_hidden = ctx.buffers.ssm_gates();
        ops::rms_norm(
            ctx.gpu,
            self.rms_norm_k,
            target_hidden,
            &self.pre_fc_norm_hidden,
            normed_hidden,
            1,
            h,
            eps,
            stream,
        )?;

        // 3. Concatenate: [normed_embed | normed_hidden] → [2*h]
        let concat_out = ctx.buffers.ssm_ba();
        ops::bf16_concat(
            ctx.gpu,
            self.bf16_concat_k,
            normed_embed,
            normed_hidden,
            concat_out,
            h,
            stream,
        )?;

        // 4. FC projection: [2*h] → [h]
        let hidden = ctx.buffers.hidden_states();
        self.gemv(ctx.gpu, concat_out, &self.fc, hidden, h, h * 2, stream)?;

        // 5. Copy hidden to residual for residual stream
        let residual = ctx.buffers.residual();
        ctx.gpu
            .copy_d2d_async(hidden, residual, row_bytes, stream)?;

        // 6. Input layernorm
        let normed = ctx.buffers.norm_output();
        ops::rms_norm(
            ctx.gpu,
            self.rms_norm_k,
            hidden,
            &self.input_layernorm,
            normed,
            1,
            h,
            eps,
            stream,
        )?;

        // 7. Attention: Q+Gate and K+V projections
        let q_out = ctx.buffers.qkv_output();
        let q_dim = nq * hd;
        let qg_dim = q_dim * 2;
        let qg_bytes = qg_dim as usize * 2;

        match self.quant {
            MtpQuantization::Nvfp4 => {
                // Fused GEMV + deinterleave in one kernel
                if let ProjectionWeight::Nvfp4(ref w) = self.q_proj {
                    ops::w4a16_gemv_qg(
                        ctx.gpu,
                        self.w4a16_gemv_qg_k,
                        normed,
                        w,
                        q_out,
                        qg_dim,
                        h,
                        nq,
                        hd,
                        stream,
                    )?;
                }
            }
            MtpQuantization::Fp8 | MtpQuantization::Bf16 => {
                // Separate GEMV + deinterleave kernel
                let deinterleave_k = self
                    .deinterleave_qg_k
                    .ok_or_else(|| anyhow::anyhow!("MTP deinterleave_qg kernel not loaded"))?;
                self.gemv(ctx.gpu, normed, &self.q_proj, q_out, qg_dim, h, stream)?;
                ops::deinterleave_qg(
                    ctx.gpu,
                    deinterleave_k,
                    q_out,
                    1,
                    nq,
                    hd,
                    nq * hd * 2,
                    stream,
                )?;
            }
        }
        let gate_ptr = q_out.offset(q_dim as usize * 2);

        // K+V projections
        let k_out = q_out.offset(qg_bytes);
        let v_out = k_out.offset((nkv * hd) as usize * 2);

        match self.quant {
            MtpQuantization::Nvfp4 => {
                if let (ProjectionWeight::Nvfp4(kw), ProjectionWeight::Nvfp4(vw)) =
                    (&self.k_proj, &self.v_proj)
                {
                    ops::w4a16_gemv_dual(
                        ctx.gpu,
                        self.w4a16_gemv_dual_k,
                        normed,
                        kw,
                        k_out,
                        vw,
                        v_out,
                        nkv * hd,
                        h,
                        stream,
                    )?;
                }
            }
            MtpQuantization::Fp8 | MtpQuantization::Bf16 => {
                self.gemv(ctx.gpu, normed, &self.k_proj, k_out, nkv * hd, h, stream)?;
                self.gemv(ctx.gpu, normed, &self.v_proj, v_out, nkv * hd, h, stream)?;
            }
        }

        // Q/K norms
        ops::rms_norm(
            ctx.gpu,
            self.rms_norm_k,
            q_out,
            &self.q_norm,
            q_out,
            nq,
            hd,
            eps,
            stream,
        )?;
        ops::rms_norm(
            ctx.gpu,
            self.rms_norm_k,
            k_out,
            &self.k_norm,
            k_out,
            nkv,
            hd,
            eps,
            stream,
        )?;

        // 8. Upload attention metadata for MTP KV cache
        let mut kv_cache = self.kv_cache.lock();
        let bs = kv_cache.block_size();
        let blocks_needed = (state.seq_len / bs) + 1;
        while state.block_table.len() < blocks_needed {
            state.block_table.push(kv_cache.alloc_block()?);
        }

        let meta_base = ctx.buffers.scratch().offset(49152); // after target metadata
        let max_blocks = state.block_table.len() as u32;

        // Batch all metadata into a single H2D copy (saves 3 CUDA API calls).
        let block_idx = state.block_table[state.seq_len / bs];
        let global_slot = (block_idx as i64) * (bs as i64) + ((state.seq_len % bs) as i64);
        let actual_seq_len = (state.seq_len + 1) as i32;
        let bt_len = state.block_table.len() * 4;

        // Dynamic metadata buffer: 256 bytes header + block table.
        // Fixed 512-byte buffer overflows when seq_len > ~2000 (block table > 256 bytes).
        let meta_size = 256 + bt_len;
        let mut meta_buf = vec![0u8; meta_size];
        meta_buf[0..4].copy_from_slice(&rotary[0].to_le_bytes());
        if !state.rotary_positions.is_identity() {
            meta_buf[20..24].copy_from_slice(&rotary[1].to_le_bytes());
            meta_buf[24..28].copy_from_slice(&rotary[2].to_le_bytes());
        }
        meta_buf[8..16].copy_from_slice(&global_slot.to_le_bytes());
        meta_buf[16..20].copy_from_slice(&actual_seq_len.to_le_bytes());
        // Block table values are always < 2^31 (block indices), so u32 → i32 is lossless.
        // SAFETY: `state.block_table` is a valid `Vec<u32>` whose byte length is
        // `bt_len == state.block_table.len() * size_of::<u32>()`; `u32` has no
        // padding bytes, `u8` has alignment 1, and `bt_bytes` does not outlive
        // the borrow of `state.block_table`.
        let bt_bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(state.block_table.as_ptr() as *const u8, bt_len) };
        meta_buf[256..256 + bt_len].copy_from_slice(bt_bytes);
        ctx.gpu
            .copy_h2d_group_on_stream(&[HostToDeviceCopy::new(&meta_buf, meta_base)], stream)?;

        // RoPE
        self.apply_rotary(state, ctx, q_out, k_out, meta_base, nq, nkv, hd, stream)?;

        // Reshape + cache (FP8).
        //
        // Per-MTP-layer FP8 KV scale calibration (opt-in via `ATLAS_MTP_FP8_CALIB=1`,
        // default OFF per `mtp_fp8_calib_enabled()`): when enabled, observes
        // post-RoPE K/V magnitudes for `MTP_FP8_WARMUP_TOKENS` warmup positions
        // and periodically recalibrates `(k_scale, v_scale)`. When disabled,
        // `effective_mtp_fp8_scales()` returns `(1.0, 1.0)`. Mirrors
        // `qwen3_attention::decode::write_kv_cache`.
        if let Some(ref cal) = self.fp8_calibration {
            // 1 token per MTP forward (decode-only). nkv KV heads × hd dim.
            cal.observe(ctx.gpu, k_out, v_out, 1, nkv, hd, stream)?;
        }
        let (k_scale, v_scale) = self.effective_mtp_fp8_scales();

        let kv_stride = nkv * hd;
        ops::reshape_and_cache_fp8(
            ctx.gpu,
            self.reshape_cache_k,
            k_out,
            v_out,
            kv_cache.k_pool_ptr(self.attn_layer_idx),
            kv_cache.v_pool_ptr(self.attn_layer_idx),
            meta_base.offset(8), // slot
            1,
            nkv,
            hd,
            bs as u32,
            k_scale,
            v_scale,
            kv_stride,
            kv_stride,
            kv_cache.cache_stride() as u64,
            stream,
        )?;

        // Paged decode attention
        let attn_out = ctx.buffers.attn_output();
        let inv_sqrt_d = 1.0f32 / (hd as f32).sqrt();
        ops::paged_decode_attn_fp8(
            ctx.gpu,
            self.paged_decode_k,
            q_out,
            kv_cache.k_pool_ptr(self.attn_layer_idx),
            kv_cache.v_pool_ptr(self.attn_layer_idx),
            attn_out,
            meta_base.offset(256), // block_table
            meta_base.offset(16),  // seq_len
            max_blocks,
            1,
            nq,
            nkv,
            hd,
            bs as u32,
            inv_sqrt_d,
            k_scale,
            v_scale, // calibrated FP8 scales (matches reshape_and_cache_fp8 above)
            nq * hd,
            kv_cache.cache_stride() as u64,
            // No tree-aware indirection on MTP path.
            // CUDA graph fix: kv_indir_base is now a device-buffer ptr;
            // NULL deactivates the tree path entirely (kernel reads
            // `kv_indir_base` only when `kv_indirection != nullptr`).
            spark_runtime::gpu::DevicePtr::NULL,
            spark_runtime::gpu::DevicePtr::NULL,
            0,
            // No tree-pack on MTP path.
            spark_runtime::gpu::DevicePtr::NULL,
            spark_runtime::gpu::DevicePtr::NULL,
            0,
            stream,
        )?;

        // Sigmoid gate: attn_out = attn_out * sigmoid(gate)
        ops::sigmoid_gate_mul(
            ctx.gpu,
            self.sigmoid_gate_mul_k,
            attn_out,
            gate_ptr,
            attn_out,
            nq * hd,
            stream,
        )?;

        // O projection: [nq*hd] → [h]
        let o_out = ctx.buffers.norm_output();
        self.gemv(ctx.gpu, attn_out, &self.o_proj, o_out, h, nq * hd, stream)?;

        // 9. Residual + post-attention norm
        let normed2 = ctx.buffers.norm_output();
        ops::residual_add_rms_norm(
            ctx.gpu,
            self.residual_add_rms_norm_k,
            hidden,
            o_out,
            &self.post_attn_layernorm,
            normed2,
            residual,
            1,
            h,
            eps,
            stream,
        )?;

        // 10. MoE FFN — or dense MLP for Qwen3.5/3.6 27B-class.
        let moe_out = if self.is_dense_mlp {
            self.dense_mlp_forward(normed2, ctx, stream)?
        } else {
            match self.quant {
                MtpQuantization::Nvfp4 => self
                    .moe_nvfp4
                    .as_ref()
                    .unwrap()
                    .forward(normed2, ctx, stream)?,
                MtpQuantization::Fp8 | MtpQuantization::Bf16 => {
                    self.moe_forward_generic(normed2, ctx, stream)?
                }
            }
        };
        ops::residual_add(ctx.gpu, self.residual_add_k, hidden, moe_out, h, stream)?;

        // 11. Final norm
        let final_normed = ctx.buffers.norm_output();
        ops::rms_norm(
            ctx.gpu,
            self.rms_norm_k,
            hidden,
            &self.norm,
            final_normed,
            1,
            h,
            eps,
            stream,
        )?;

        // 12. LM head (shared NVFP4, reduced vocab for faster propose)
        let v = if self.mtp_vocab_size > 0 {
            self.mtp_vocab_size.min(ctx.config.vocab_size as u32)
        } else {
            ctx.config.vocab_size as u32
        };
        let logits = ctx.buffers.logits();
        ops::w4a16_gemv(
            ctx.gpu,
            self.w4a16_gemv_k,
            final_normed,
            &self.lm_head_nvfp4,
            logits,
            v,
            h,
            stream,
        )?;

        // 13. Argmax
        let out_ptr = ctx.buffers.scratch();

        let token_id = if let Some(bitmask) = grammar_bitmask {
            // Grammar-masked CPU argmax path — single fused pass.
            //
            // D2H the BF16 logits, then walk the vocab ONCE: for each token,
            // skip when the bitmask bit is clear, else convert in-register and
            // track the argmax. This replaces three sequential vocab-sized
            // passes (BF16→f32, mask apply, argmax) with one — roughly 3x less
            // memory traffic and zero scratch allocations beyond the D2H
            // buffer.
            //
            // We then H2D the chosen token id into `out_ptr` so the
            // downstream `embed_from_argmax` kernel can still gather the
            // embedding from the token table on GPU without a new kernel.
            //
            // Empirical: on the 248k-vocab aeon-ultimate this path went from
            // ~4 ms/draft (3 sequential passes) to ~0.5 ms/draft, a ~8x
            // improvement on grammar-constrained tool_call dispatch.
            let vocab = v as usize;
            let mut bf16_buf = vec![0u8; vocab * 2];
            ctx.gpu.copy_d2h(logits, &mut bf16_buf)?;

            // Shared first-wins greedy pick over the allowed ids (the device
            // argmax's tie rule); f32 comparison orders negative logits
            // numerically.
            let picked = spark_runtime::sampler::argmax_bf16_first_wins(&bf16_buf, |tok| {
                let (word, bit) = ((tok / 32) as usize, tok % 32);
                word < bitmask.len() && (bitmask[word] & (1i32 << bit)) != 0
            });
            let any_allowed = picked.is_some();
            let best_tok = picked.unwrap_or(0);

            if !any_allowed {
                tracing::warn!(
                    "MTP grammar mask allowed zero tokens at pos {position}; \
                     returning 0 as pad-draft (will be rejected at verify)."
                );
                0u32
            } else {
                // If caller wants the embedding staged on GPU, stage the
                // chosen token id into `out_ptr` (4 bytes) and reuse the
                // existing embed_from_argmax kernel — it reads the argmax
                // result from `out_ptr` and gathers the embedding on GPU.
                if let Some(embed_target) = draft_embed_target {
                    let tok_bytes = best_tok.to_le_bytes();
                    ctx.gpu.copy_h2d(&tok_bytes, out_ptr)?;
                    ops::embed_from_argmax(
                        ctx.gpu,
                        self.embed_from_argmax_k,
                        out_ptr,
                        self.embed_tokens.weight,
                        embed_target,
                        self.draft_token_id_dev,
                        h,
                        stream,
                    )?;
                }
                best_tok
            }
        } else {
            ops::argmax_bf16(ctx.gpu, self.argmax_k, logits, out_ptr, v, stream)?;

            // ATLAS_MTP_DEBUG=1: per-step diagnostic — emit drafter top-1 token,
            // top-1/top-2 confidence margin, and target_hidden bf16 abs-max so
            // long-context accept-rate drift can be correlated with logit
            // confidence + hidden-state magnitude growth.
            //
            // This is on the propose hot path so it forces a D2H sync — only
            // engage under env-gate when investigating accept-rate regressions.
            if std::env::var("ATLAS_MTP_DEBUG").ok().as_deref() == Some("1") {
                let vocab = v as usize;
                let mut bf16_buf = vec![0u8; vocab * 2];
                ctx.gpu.copy_d2h(logits, &mut bf16_buf)?;
                let mut top1_val = f32::NEG_INFINITY;
                let mut top2_val = f32::NEG_INFINITY;
                let mut top1_tok: u32 = 0;
                for i in 0..vocab {
                    let hi = u16::from_le_bytes([bf16_buf[2 * i], bf16_buf[2 * i + 1]]);
                    let val = f32::from_bits((hi as u32) << 16);
                    if val > top1_val {
                        top2_val = top1_val;
                        top1_val = val;
                        top1_tok = i as u32;
                    } else if val > top2_val {
                        top2_val = val;
                    }
                }
                let margin = top1_val - top2_val;
                // Sample target_hidden magnitude (bf16, h elements).
                let h_bytes = h as usize * 2;
                let mut th_buf = vec![0u8; h_bytes];
                ctx.gpu.copy_d2h(target_hidden, &mut th_buf)?;
                let mut th_absmax = 0.0f32;
                let mut th_sumsq = 0.0f64;
                for i in 0..(h as usize) {
                    let hi = u16::from_le_bytes([th_buf[2 * i], th_buf[2 * i + 1]]);
                    let val = f32::from_bits((hi as u32) << 16);
                    let abs = val.abs();
                    if abs > th_absmax {
                        th_absmax = abs;
                    }
                    th_sumsq += (val as f64) * (val as f64);
                }
                let th_rms = (th_sumsq / (h as f64)).sqrt();
                tracing::info!(
                    "ATLAS_MTP_DEBUG pos={position} mtp_seq={} top1={top1_tok} \
                     top1_val={top1_val:.3} margin={margin:.3} \
                     th_absmax={th_absmax:.3} th_rms={th_rms:.3}",
                    state.seq_len,
                );
            }

            if let Some(embed_target) = draft_embed_target {
                // GPU-side embedding: write draft embedding to verify input buffer
                // and token ID to deferred readback buffer. No D2H sync needed.
                ops::embed_from_argmax(
                    ctx.gpu,
                    self.embed_from_argmax_k,
                    out_ptr,
                    self.embed_tokens.weight,
                    embed_target,
                    self.draft_token_id_dev,
                    h,
                    stream,
                )?;
                // Return 0 as placeholder — caller reads actual ID later via
                // read_deferred_draft_token().
                0u32
            } else {
                // Fallback: synchronous D2H readback.
                let mut buf = [0u8; 4];
                ctx.gpu.copy_d2h(out_ptr, &mut buf)?;
                u32::from_le_bytes(buf)
            }
        };

        state.seq_len += 1;
        Ok(token_id)
    }

    /// Populate the single-layer MTP KV cache for one prompt-tail token without
    /// running paged decode attention, O-proj, MoE/dense FFN, final norm, LM head,
    /// argmax, or a synchronous D2H readback.
    pub(crate) fn forward_kv_only(
        &self,
        token: u32,
        target_hidden: DevicePtr,
        position: usize,
        state: &mut MtpProposerState,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let h = ctx.config.hidden_size as u32;
        let nq = ctx.config.num_attention_heads as u32;
        let nkv = ctx.config.num_key_value_heads as u32;
        let hd = ctx.config.head_dim as u32;
        let eps = ctx.config.rms_norm_eps as f32;
        let rotary = self.rotary_axes(state, position, ctx)?;

        let embed_out = ctx.buffers.ssm_qkvz();
        let row_bytes = h as usize * 2;
        let src = self.embed_tokens.weight.offset(token as usize * row_bytes);
        ctx.gpu.copy_d2d_async(src, embed_out, row_bytes, stream)?;

        let normed_embed = ctx.buffers.ssm_deinterleaved();
        ops::rms_norm(
            ctx.gpu,
            self.rms_norm_k,
            embed_out,
            &self.pre_fc_norm_embedding,
            normed_embed,
            1,
            h,
            eps,
            stream,
        )?;

        let normed_hidden = ctx.buffers.ssm_gates();
        ops::rms_norm(
            ctx.gpu,
            self.rms_norm_k,
            target_hidden,
            &self.pre_fc_norm_hidden,
            normed_hidden,
            1,
            h,
            eps,
            stream,
        )?;

        let concat_out = ctx.buffers.ssm_ba();
        ops::bf16_concat(
            ctx.gpu,
            self.bf16_concat_k,
            normed_embed,
            normed_hidden,
            concat_out,
            h,
            stream,
        )?;

        let hidden = ctx.buffers.hidden_states();
        self.gemv(ctx.gpu, concat_out, &self.fc, hidden, h, h * 2, stream)?;

        let normed = ctx.buffers.norm_output();
        ops::rms_norm(
            ctx.gpu,
            self.rms_norm_k,
            hidden,
            &self.input_layernorm,
            normed,
            1,
            h,
            eps,
            stream,
        )?;

        let q_out = ctx.buffers.qkv_output();
        let qg_bytes = (nq * hd * 2) as usize * 2;
        let k_out = q_out.offset(qg_bytes);
        let v_out = k_out.offset((nkv * hd) as usize * 2);

        match self.quant {
            MtpQuantization::Nvfp4 => {
                if let (ProjectionWeight::Nvfp4(kw), ProjectionWeight::Nvfp4(vw)) =
                    (&self.k_proj, &self.v_proj)
                {
                    ops::w4a16_gemv_dual(
                        ctx.gpu,
                        self.w4a16_gemv_dual_k,
                        normed,
                        kw,
                        k_out,
                        vw,
                        v_out,
                        nkv * hd,
                        h,
                        stream,
                    )?;
                }
            }
            MtpQuantization::Fp8 | MtpQuantization::Bf16 => {
                self.gemv(ctx.gpu, normed, &self.k_proj, k_out, nkv * hd, h, stream)?;
                self.gemv(ctx.gpu, normed, &self.v_proj, v_out, nkv * hd, h, stream)?;
            }
        }

        ops::rms_norm(
            ctx.gpu,
            self.rms_norm_k,
            k_out,
            &self.k_norm,
            k_out,
            nkv,
            hd,
            eps,
            stream,
        )?;

        let mut kv_cache = self.kv_cache.lock();
        let bs = kv_cache.block_size();
        let blocks_needed = (state.seq_len / bs) + 1;
        while state.block_table.len() < blocks_needed {
            state.block_table.push(kv_cache.alloc_block()?);
        }

        let meta_base = ctx.buffers.scratch().offset(49152);
        let block_idx = state.block_table[state.seq_len / bs];
        let global_slot = (block_idx as i64) * (bs as i64) + ((state.seq_len % bs) as i64);
        let mut meta_buf = [0u8; 32];
        meta_buf[0..4].copy_from_slice(&rotary[0].to_le_bytes());
        if !state.rotary_positions.is_identity() {
            meta_buf[20..24].copy_from_slice(&rotary[1].to_le_bytes());
            meta_buf[24..28].copy_from_slice(&rotary[2].to_le_bytes());
        }
        meta_buf[8..16].copy_from_slice(&global_slot.to_le_bytes());
        ctx.gpu
            .copy_h2d_group_on_stream(&[HostToDeviceCopy::new(&meta_buf, meta_base)], stream)?;

        self.apply_rotary(state, ctx, q_out, k_out, meta_base, 0, nkv, hd, stream)?;

        if let Some(ref cal) = self.fp8_calibration {
            cal.observe(ctx.gpu, k_out, v_out, 1, nkv, hd, stream)?;
        }
        let (k_scale, v_scale) = self.effective_mtp_fp8_scales();

        let kv_stride = nkv * hd;
        ops::reshape_and_cache_fp8(
            ctx.gpu,
            self.reshape_cache_k,
            k_out,
            v_out,
            kv_cache.k_pool_ptr(self.attn_layer_idx),
            kv_cache.v_pool_ptr(self.attn_layer_idx),
            meta_base.offset(8),
            1,
            nkv,
            hd,
            bs as u32,
            k_scale,
            v_scale,
            kv_stride,
            kv_stride,
            kv_cache.cache_stride() as u64,
            stream,
        )?;

        state.seq_len += 1;
        Ok(())
    }

    /// Device-chained multi-draft propose for `MtpHead`:
    /// - Packs all `num_drafts` step metadata blocks into a single H2D upload before step 0.
    /// - Chains intermediate steps `0..num_drafts - 2` on the GPU via `embed_from_argmax`
    ///   directly into `embed_out` (`ctx.buffers.ssm_qkvz()`) with zero intermediate D2H syncs.
    /// - Performs a single D2H readback on the final step `step == num_drafts - 1`.
    /// - Under opt-in `ATLAS_MTP_DLOOP=1` on dense targets (`!is_moe`, `C <= 2`), reads back
    ///   Stage-1 softmax probabilities and extends by Stage 2 (`K = 5 -> K = 9`) when
    ///   `c_1 = sum ln(p_i) >= -0.5`. Strictly fail-closed on MoE targets.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn propose_chain(
        &self,
        last_token: u32,
        target_hidden: DevicePtr,
        position: usize,
        num_drafts: usize,
        state: &mut MtpProposerState,
        ctx: &ForwardContext,
        stream: u64,
        draft_embed_target: Option<DevicePtr>,
    ) -> Result<Vec<u32>> {
        use crate::layers::mtp_dloop;

        let dloop_enabled = mtp_dloop::mtp_dloop_enabled();
        let topology = if self.is_dense_mlp {
            mtp_dloop::TargetFfnTopology::from_num_experts(ctx.config.num_experts)
        } else {
            mtp_dloop::TargetFfnTopology::Moe {
                num_experts: ctx.config.num_experts.max(2),
            }
        };
        let cohort_ctx = mtp_dloop::active_cohort_context();
        let dloop_precheck = if dloop_enabled && draft_embed_target.is_none() {
            match cohort_ctx {
                Some(cohort) => {
                    mtp_dloop::precheck_stage1_admission(&mtp_dloop::MtpDloopStage1PrecheckInput {
                        enabled: true,
                        topology,
                        live_seqs: cohort.live_seqs,
                        stage1_drafts: num_drafts,
                        max_verify_k: cohort.max_verify_k,
                    })
                }
                None => Err(mtp_dloop::MtpDloopRefusal::ConcurrencyTooHigh {
                    live_seqs: 0,
                    max_concurrency: mtp_dloop::MTP_DLOOP_MAX_CONCURRENCY,
                }),
            }
        } else {
            Err(mtp_dloop::MtpDloopRefusal::Disabled)
        };
        if let Err(mtp_dloop::MtpDloopRefusal::MoeTargetRefused { num_experts }) = dloop_precheck {
            static MOE_REFUSED_LOGGED: std::sync::Once = std::sync::Once::new();
            MOE_REFUSED_LOGGED.call_once(|| {
                tracing::info!(
                    num_experts,
                    "MTP_DLOOP_REFUSED reason=moe_target (ATLAS_MTP_DLOOP=1 is dense-only)"
                );
            });
        }
        let want_stage1_probs = dloop_precheck.is_ok() && self.argmax_prob_k.0 != 0;

        let (mut drafts, stage1_probs) = self.propose_chain_segment(
            last_token,
            target_hidden,
            position,
            num_drafts,
            state,
            ctx,
            stream,
            draft_embed_target,
            want_stage1_probs,
        )?;

        if want_stage1_probs
            && let Some(cohort) = cohort_ctx
            && let Some(&stage1_last_token) = drafts.last()
        {
            let decision = mtp_dloop::evaluate_stage2_extension(&mtp_dloop::MtpDloopInput {
                enabled: dloop_enabled,
                topology,
                live_seqs: cohort.live_seqs,
                stage1_drafts: num_drafts,
                max_verify_k: cohort.max_verify_k,
                stage1_probs: &stage1_probs,
                threshold: mtp_dloop::mtp_dloop_thresh(),
            });
            if let mtp_dloop::MtpDloopDecision::ExtendStage2 {
                extra_drafts,
                total_drafts,
            } = decision
            {
                static DLOOP_ENGAGED: std::sync::Once = std::sync::Once::new();
                DLOOP_ENGAGED.call_once(|| {
                    tracing::info!(
                        stage1_drafts = num_drafts,
                        extra_drafts,
                        total_drafts,
                        live_seqs = cohort.live_seqs,
                        threshold = mtp_dloop::mtp_dloop_thresh(),
                        "MTP_DLOOP_STAGE2_ENGAGED"
                    );
                });
                let (stage2_drafts, _) = self.propose_chain_segment(
                    stage1_last_token,
                    ctx.buffers.hidden_states(),
                    position + num_drafts,
                    extra_drafts,
                    state,
                    ctx,
                    stream,
                    None,
                    false,
                )?;
                drafts.extend_from_slice(&stage2_drafts);
            }
        }

        Ok(drafts)
    }

    #[allow(clippy::too_many_arguments)]
    fn propose_chain_segment(
        &self,
        last_token: u32,
        target_hidden: DevicePtr,
        position: usize,
        num_drafts: usize,
        state: &mut MtpProposerState,
        ctx: &ForwardContext,
        stream: u64,
        draft_embed_target: Option<DevicePtr>,
        capture_probs: bool,
    ) -> Result<(Vec<u32>, Vec<f32>)> {
        let h = ctx.config.hidden_size as u32;
        let nq = ctx.config.num_attention_heads as u32;
        let nkv = ctx.config.num_key_value_heads as u32;
        let hd = ctx.config.head_dim as u32;
        let eps = ctx.config.rms_norm_eps as f32;
        let row_bytes = h as usize * 2;
        let v = if self.mtp_vocab_size > 0 {
            self.mtp_vocab_size.min(ctx.config.vocab_size as u32)
        } else {
            ctx.config.vocab_size as u32
        };

        let first_seq_len = state.seq_len;
        let mut rotaries = Vec::with_capacity(num_drafts);
        for step in 0..num_drafts {
            rotaries.push(self.rotary_axes(state, position + step, ctx)?);
        }

        let mut kv_cache = self.kv_cache.lock();
        let bs = kv_cache.block_size();
        let last_seq_len = first_seq_len + num_drafts - 1;
        let blocks_needed = (last_seq_len / bs) + 1;
        while state.block_table.len() < blocks_needed {
            state.block_table.push(kv_cache.alloc_block()?);
        }
        let max_blocks = state.block_table.len() as u32;
        let bt_len = state.block_table.len() * 4;
        let step_meta_stride = (256 + bt_len).next_multiple_of(256);
        let mut packed_meta = vec![0u8; num_drafts * step_meta_stride];
        // SAFETY: `state.block_table` is a valid `Vec<u32>` whose byte length is
        // `bt_len == state.block_table.len() * size_of::<u32>()`; `u32` has no
        // padding bytes, `u8` has alignment 1, and `bt_bytes` does not outlive
        // the borrow of `state.block_table`.
        let bt_bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(state.block_table.as_ptr() as *const u8, bt_len) };
        for (step, rotary) in rotaries.iter().enumerate() {
            let seq_len_step = first_seq_len + step;
            let block_idx = state.block_table[seq_len_step / bs];
            let global_slot = (block_idx as i64) * (bs as i64) + ((seq_len_step % bs) as i64);
            let actual_seq_len = (seq_len_step + 1) as i32;
            let base = step * step_meta_stride;
            packed_meta[base..base + 4].copy_from_slice(&rotary[0].to_le_bytes());
            if !state.rotary_positions.is_identity() {
                packed_meta[base + 20..base + 24].copy_from_slice(&rotary[1].to_le_bytes());
                packed_meta[base + 24..base + 28].copy_from_slice(&rotary[2].to_le_bytes());
            }
            packed_meta[base + 8..16 + base].copy_from_slice(&global_slot.to_le_bytes());
            packed_meta[base + 16..base + 20].copy_from_slice(&actual_seq_len.to_le_bytes());
            packed_meta[base + 256..base + 256 + bt_len].copy_from_slice(bt_bytes);
        }
        let meta_base_all = ctx.buffers.scratch().offset(49152);
        ctx.gpu.copy_h2d_group_on_stream(
            &[HostToDeviceCopy::new(&packed_meta, meta_base_all)],
            stream,
        )?;

        let argmax_base = ctx.buffers.scratch();
        let prob_base = ctx.buffers.scratch().offset(256);
        let emit_probs = capture_probs && self.argmax_prob_k.0 != 0 && num_drafts <= 64;
        let embed_out = ctx.buffers.ssm_qkvz();
        let mut current_hidden = target_hidden;

        for step in 0..num_drafts {
            let is_last_step = step + 1 == num_drafts;
            let meta_base = meta_base_all.offset(step * step_meta_stride);
            let out_ptr = argmax_base.offset(step * 4);
            let prob_ptr = prob_base.offset(step * 4);

            // 1. Embed token (step 0 copies from host-known last_token; steps 1..num_drafts-1
            //    already had their embedding gathered into `embed_out` by the previous step's
            //    `embed_from_argmax`).
            if step == 0 {
                let src = self
                    .embed_tokens
                    .weight
                    .offset(last_token as usize * row_bytes);
                ctx.gpu.copy_d2d_async(src, embed_out, row_bytes, stream)?;
            }

            // 2. RMSNorm embedding and hidden separately
            let normed_embed = ctx.buffers.ssm_deinterleaved();
            ops::rms_norm(
                ctx.gpu,
                self.rms_norm_k,
                embed_out,
                &self.pre_fc_norm_embedding,
                normed_embed,
                1,
                h,
                eps,
                stream,
            )?;

            let normed_hidden = ctx.buffers.ssm_gates();
            ops::rms_norm(
                ctx.gpu,
                self.rms_norm_k,
                current_hidden,
                &self.pre_fc_norm_hidden,
                normed_hidden,
                1,
                h,
                eps,
                stream,
            )?;

            // 3. Concatenate: [normed_embed | normed_hidden] → [2*h]
            let concat_out = ctx.buffers.ssm_ba();
            ops::bf16_concat(
                ctx.gpu,
                self.bf16_concat_k,
                normed_embed,
                normed_hidden,
                concat_out,
                h,
                stream,
            )?;

            // 4. FC projection: [2*h] → [h]
            let hidden = ctx.buffers.hidden_states();
            self.gemv(ctx.gpu, concat_out, &self.fc, hidden, h, h * 2, stream)?;

            // 5. Copy hidden to residual for residual stream
            let residual = ctx.buffers.residual();
            ctx.gpu
                .copy_d2d_async(hidden, residual, row_bytes, stream)?;

            // 6. Input layernorm
            let normed = ctx.buffers.norm_output();
            ops::rms_norm(
                ctx.gpu,
                self.rms_norm_k,
                hidden,
                &self.input_layernorm,
                normed,
                1,
                h,
                eps,
                stream,
            )?;

            // 7. Attention: Q+Gate and K+V projections
            let q_out = ctx.buffers.qkv_output();
            let q_dim = nq * hd;
            let qg_dim = q_dim * 2;
            let qg_bytes = qg_dim as usize * 2;

            match self.quant {
                MtpQuantization::Nvfp4 => {
                    if let ProjectionWeight::Nvfp4(ref w) = self.q_proj {
                        ops::w4a16_gemv_qg(
                            ctx.gpu,
                            self.w4a16_gemv_qg_k,
                            normed,
                            w,
                            q_out,
                            qg_dim,
                            h,
                            nq,
                            hd,
                            stream,
                        )?;
                    }
                }
                MtpQuantization::Fp8 | MtpQuantization::Bf16 => {
                    let deinterleave_k = self
                        .deinterleave_qg_k
                        .ok_or_else(|| anyhow::anyhow!("MTP deinterleave_qg kernel not loaded"))?;
                    self.gemv(ctx.gpu, normed, &self.q_proj, q_out, qg_dim, h, stream)?;
                    ops::deinterleave_qg(
                        ctx.gpu,
                        deinterleave_k,
                        q_out,
                        1,
                        nq,
                        hd,
                        nq * hd * 2,
                        stream,
                    )?;
                }
            }
            let gate_ptr = q_out.offset(q_dim as usize * 2);
            let k_out = q_out.offset(qg_bytes);
            let v_out = k_out.offset((nkv * hd) as usize * 2);

            match self.quant {
                MtpQuantization::Nvfp4 => {
                    if let (ProjectionWeight::Nvfp4(kw), ProjectionWeight::Nvfp4(vw)) =
                        (&self.k_proj, &self.v_proj)
                    {
                        ops::w4a16_gemv_dual(
                            ctx.gpu,
                            self.w4a16_gemv_dual_k,
                            normed,
                            kw,
                            k_out,
                            vw,
                            v_out,
                            nkv * hd,
                            h,
                            stream,
                        )?;
                    }
                }
                MtpQuantization::Fp8 | MtpQuantization::Bf16 => {
                    self.gemv(ctx.gpu, normed, &self.k_proj, k_out, nkv * hd, h, stream)?;
                    self.gemv(ctx.gpu, normed, &self.v_proj, v_out, nkv * hd, h, stream)?;
                }
            }

            ops::rms_norm(
                ctx.gpu,
                self.rms_norm_k,
                q_out,
                &self.q_norm,
                q_out,
                nq,
                hd,
                eps,
                stream,
            )?;
            ops::rms_norm(
                ctx.gpu,
                self.rms_norm_k,
                k_out,
                &self.k_norm,
                k_out,
                nkv,
                hd,
                eps,
                stream,
            )?;

            // 8. RoPE + KV cache write + Paged decode attention
            self.apply_rotary(state, ctx, q_out, k_out, meta_base, nq, nkv, hd, stream)?;

            if let Some(ref cal) = self.fp8_calibration {
                cal.observe(ctx.gpu, k_out, v_out, 1, nkv, hd, stream)?;
            }
            let (k_scale, v_scale) = self.effective_mtp_fp8_scales();

            let kv_stride = nkv * hd;
            ops::reshape_and_cache_fp8(
                ctx.gpu,
                self.reshape_cache_k,
                k_out,
                v_out,
                kv_cache.k_pool_ptr(self.attn_layer_idx),
                kv_cache.v_pool_ptr(self.attn_layer_idx),
                meta_base.offset(8),
                1,
                nkv,
                hd,
                bs as u32,
                k_scale,
                v_scale,
                kv_stride,
                kv_stride,
                kv_cache.cache_stride() as u64,
                stream,
            )?;

            let attn_out = ctx.buffers.attn_output();
            let inv_sqrt_d = 1.0f32 / (hd as f32).sqrt();
            ops::paged_decode_attn_fp8(
                ctx.gpu,
                self.paged_decode_k,
                q_out,
                kv_cache.k_pool_ptr(self.attn_layer_idx),
                kv_cache.v_pool_ptr(self.attn_layer_idx),
                attn_out,
                meta_base.offset(256),
                meta_base.offset(16),
                max_blocks,
                1,
                nq,
                nkv,
                hd,
                bs as u32,
                inv_sqrt_d,
                k_scale,
                v_scale,
                nq * hd,
                kv_cache.cache_stride() as u64,
                spark_runtime::gpu::DevicePtr::NULL,
                spark_runtime::gpu::DevicePtr::NULL,
                0,
                spark_runtime::gpu::DevicePtr::NULL,
                spark_runtime::gpu::DevicePtr::NULL,
                0,
                stream,
            )?;

            ops::sigmoid_gate_mul(
                ctx.gpu,
                self.sigmoid_gate_mul_k,
                attn_out,
                gate_ptr,
                attn_out,
                nq * hd,
                stream,
            )?;

            let o_out = ctx.buffers.norm_output();
            self.gemv(ctx.gpu, attn_out, &self.o_proj, o_out, h, nq * hd, stream)?;

            // 9. Residual + post-attention norm
            let normed2 = ctx.buffers.norm_output();
            ops::residual_add_rms_norm(
                ctx.gpu,
                self.residual_add_rms_norm_k,
                hidden,
                o_out,
                &self.post_attn_layernorm,
                normed2,
                residual,
                1,
                h,
                eps,
                stream,
            )?;

            // 10. MoE / dense FFN
            let moe_out = if self.is_dense_mlp {
                self.dense_mlp_forward(normed2, ctx, stream)?
            } else {
                match self.quant {
                    MtpQuantization::Nvfp4 => self
                        .moe_nvfp4
                        .as_ref()
                        .unwrap()
                        .forward(normed2, ctx, stream)?,
                    MtpQuantization::Fp8 | MtpQuantization::Bf16 => {
                        self.moe_forward_generic(normed2, ctx, stream)?
                    }
                }
            };
            ops::residual_add(ctx.gpu, self.residual_add_k, hidden, moe_out, h, stream)?;

            // 11. Final norm + LM head + Argmax
            let final_normed = ctx.buffers.norm_output();
            ops::rms_norm(
                ctx.gpu,
                self.rms_norm_k,
                hidden,
                &self.norm,
                final_normed,
                1,
                h,
                eps,
                stream,
            )?;

            let logits = ctx.buffers.logits();
            ops::w4a16_gemv(
                ctx.gpu,
                self.w4a16_gemv_k,
                final_normed,
                &self.lm_head_nvfp4,
                logits,
                v,
                h,
                stream,
            )?;

            if emit_probs {
                spark_runtime::kernel_args::KernelLaunch::new(ctx.gpu, self.argmax_prob_k)
                    .grid([1, 1, 1])
                    .block([1024, 1, 1])
                    .arg_ptr(logits)
                    .arg_ptr(out_ptr)
                    .arg_ptr(prob_ptr)
                    .arg_u32(v)
                    .launch(stream)?;
            } else {
                ops::argmax_bf16(ctx.gpu, self.argmax_k, logits, out_ptr, v, stream)?;
            }

            if !is_last_step {
                // Chain token embedding directly on GPU into `embed_out` for `step + 1`
                // without any host D2H sync.
                ops::embed_from_argmax(
                    ctx.gpu,
                    self.embed_from_argmax_k,
                    out_ptr,
                    self.embed_tokens.weight,
                    embed_out,
                    out_ptr,
                    h,
                    stream,
                )?;
                current_hidden = hidden;
            } else if let Some(embed_target) = draft_embed_target {
                ops::embed_from_argmax(
                    ctx.gpu,
                    self.embed_from_argmax_k,
                    out_ptr,
                    self.embed_tokens.weight,
                    embed_target,
                    self.draft_token_id_dev,
                    h,
                    stream,
                )?;
            }

            state.seq_len += 1;
        }
        drop(kv_cache);

        // Single D2H readback on the final step `step == num_drafts - 1`.
        let read_count = if draft_embed_target.is_some() {
            num_drafts - 1
        } else {
            num_drafts
        };
        let mut bytes = vec![0u8; read_count * 4];
        let mut probs = Vec::new();
        if read_count > 0 {
            if emit_probs {
                let mut pbytes = vec![0u8; read_count * 4];
                ctx.gpu.copy_d2h_pair_on_stream(
                    argmax_base,
                    &mut bytes,
                    prob_base,
                    &mut pbytes,
                    stream,
                )?;
                probs = pbytes
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                    .collect();
            } else {
                ctx.gpu.copy_d2h(argmax_base, &mut bytes)?;
            }
        }
        let mut drafts: Vec<u32> = bytes
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        if draft_embed_target.is_some() {
            drafts.push(0u32);
        }
        Ok((drafts, probs))
    }
}
