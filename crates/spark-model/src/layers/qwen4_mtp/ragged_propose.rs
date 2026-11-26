// SPDX-License-Identifier: AGPL-3.0-only

//! Active-subset compacted ragged batched MTP drafting with per-sequence p_cum early stop.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, HostToDeviceCopy};

use super::gumbel_draft;
use super::init::Qwen4MtpHead;
use super::policy::{
    BatchGumbelMetaLayout, CHAIN_ARGMAX_OFFSET, CHAIN_META_OFFSET, CHAIN_PROB_OFFSET,
    CHAIN_PROB_SPAN, advance_ragged_mtp_step, can_gpu_chain_ragged_steps,
    current_ragged_mtp_target_drafts, mtp_lm_head_tc_pipe_enabled, mtp_ragged_pcum_thresh,
    plan_hidden_compaction_copies, prepare_packed_meta_scratch,
    quantize_ragged_drafts_to_graph_bucket, ragged_mtp_allow_zero_drafts,
    ragged_mtp_should_stop_stragglers, should_use_mtp_fc_coalesced,
    should_use_mtp_lm_head_coalesced, should_use_mtp_lm_head_pipe, validate_rotary_span,
};
use super::state::Qwen4MtpState;
use crate::layer::{AttnMetadataDev, ForwardContext};
use crate::layers::{ops, qwen4_mtp_gate};
use crate::speculative::ProposerState;

impl Qwen4MtpHead {
    /// Active-subset compacted cross-sequence MTP drafting with per-sequence
    /// running-product `p_cum = prod p_i` early stop.
    ///
    /// After each draft step, ONE tiny `[b_act]` `(token, prob)` D2H pair
    /// readback evaluates `p_cum[s] *= p_step` on host and drops lanes where
    /// `p_cum[s] < pcum_thresh` or `emitted[s].len() >= target_drafts[s]`.
    /// Surviving lanes compact `hidden` in-place (strictly non-overlapping D2D
    /// spans) and pack metadata only for `b_act` rows, so stopped streams
    /// execute ZERO further MTP layers/MoE/LM-head work. Supports `b_act`
    /// shrinking all the way down to `1` (`decode_qwen4_cross_seq_mtp` 1..=64 +
    /// 1-row `final_mixer.prepare_decode` / `w4a16_decode_gemv`).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn propose_batch_ragged_compacted(
        &self,
        last_tokens: &[u32],
        target_hiddens: &[DevicePtr],
        positions: &[usize],
        num_drafts: usize,
        policy: qwen4_mtp_gate::GatePolicy,
        gumbel_specs: &[Option<spark_runtime::sampler::gumbel::GumbelDraftSpec>],
        any_gumbel: bool,
        states: &mut [&mut dyn ProposerState],
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<Option<Vec<Vec<u32>>>> {
        let b = last_tokens.len();
        let h = ctx.config.hidden_size;
        let r = ctx.config.residual_width();
        let hc = ctx.config.hc_count;
        let eps = ctx.config.rms_norm_eps as f32;
        let row_bytes = h * 2;
        let r_bytes = r * 2;
        let base_vocab = if qwen4_mtp_gate::mtp_vocab_prune_selected() {
            (qwen4_mtp_gate::PRUNED_VOCAB_SIZE as usize).min(ctx.config.vocab_size)
        } else if self.mtp_vocab_size == 0 {
            ctx.config.vocab_size
        } else {
            (self.mtp_vocab_size as usize).min(ctx.config.vocab_size)
        };
        let vocab = qwen4_mtp_gate::mtp_draft_vocab_override_for(any_gumbel)
            .map_or(base_vocab, |cap| base_vocab.min(cap));
        let single_step_bytes = b * 4;
        if hc != 4
            || b * vocab * 2 > ctx.buffers.sizes().logits
            || CHAIN_ARGMAX_OFFSET + single_step_bytes > CHAIN_PROB_OFFSET
            || single_step_bytes > CHAIN_PROB_SPAN
        {
            return Ok(None);
        }

        let target_drafts: Vec<usize> = match current_ragged_mtp_target_drafts() {
            Some(scoped) if scoped.len() == b => {
                scoped.into_iter().map(|d| d.min(num_drafts)).collect()
            }
            _ => vec![num_drafts; b],
        };
        let max_steps = target_drafts.iter().copied().max().unwrap_or(0);
        let mut active_lanes: Vec<usize> = (0..b).filter(|&s| target_drafts[s] > 0).collect();

        let mut all_identity = true;
        for (s, st) in states.iter().enumerate() {
            let Some(state) = st.as_any().downcast_ref::<Qwen4MtpState>() else {
                return Ok(None);
            };
            if state.pending_replay.is_some() {
                return Ok(None);
            }
            if target_drafts[s] > 0 {
                validate_rotary_span(
                    &state.rotary_positions,
                    positions[s],
                    target_drafts[s],
                    ctx.config.mrope_interleaved,
                    self.mrope_kernel_available,
                    ctx.config.yarn_factor,
                )?;
            }
            if !state.rotary_positions.is_identity() {
                all_identity = false;
            }
        }

        let mut kv_cache = self.kv_cache.lock();
        let bs = kv_cache.block_size();
        let mut total_new_blocks = 0usize;
        let mut max_mb = 1usize;
        for (s, st) in states.iter().enumerate() {
            let state = st
                .as_any()
                .downcast_ref::<Qwen4MtpState>()
                .expect("validated above");
            let effective_blocks = if target_drafts[s] > 0 {
                let last = state.seq_len + target_drafts[s] - 1;
                let blocks_needed = last / bs + 1;
                total_new_blocks += blocks_needed.saturating_sub(state.block_table.len());
                state.block_table.len().max(blocks_needed)
            } else {
                state.block_table.len()
            };
            max_mb = max_mb.max(effective_blocks);
        }
        if kv_cache.num_free_blocks() < total_new_blocks {
            return Ok(None);
        }
        let max_bt_off = if b <= 8 { 768 } else { 2560 };
        let bt_row_bytes = max_mb * 4;
        let max_step_stride = (max_bt_off + b * bt_row_bytes).next_multiple_of(256);
        let shared_gumbel_spec = gumbel_draft::uniform_batch_gumbel_spec(gumbel_specs);
        let max_gumbel_bytes = shared_gumbel_spec
            .and_then(|_| BatchGumbelMetaLayout::compute(max_step_stride, b, 1))
            .map_or(0, |layout| layout.total_bytes);
        if CHAIN_META_OFFSET + max_step_stride + max_gumbel_bytes > ctx.buffers.sizes().scratch {
            return Ok(None);
        }
        for st in states.iter() {
            if let Some(state) = st.as_any().downcast_ref::<Qwen4MtpState>() {
                state.clear_gumbel_spec();
            }
        }
        spark_runtime::sampler::gumbel::clear_gumbel_draft_specs();
        let mut base_seq_lens = Vec::with_capacity(b);
        for (s, st) in states.iter_mut().enumerate() {
            let state = st
                .as_any_mut()
                .downcast_mut::<Qwen4MtpState>()
                .expect("validated above");
            state.staged_frame.lock().clear();
            base_seq_lens.push(state.seq_len);
            if target_drafts[s] > 0 {
                let last = state.seq_len + target_drafts[s] - 1;
                let blocks_needed = last / bs + 1;
                while state.block_table.len() < blocks_needed {
                    state.block_table.push(kv_cache.alloc_block()?);
                }
            }
        }

        if active_lanes.is_empty() || max_steps == 0 {
            drop(kv_cache);
            for st in states.iter_mut() {
                let state = st
                    .as_any_mut()
                    .downcast_mut::<Qwen4MtpState>()
                    .expect("validated above");
                state.last_num_drafted = 0;
                state.last_drafts.clear();
            }
            return Ok(Some(vec![Vec::new(); b]));
        }

        let scratch = ctx.buffers.scratch();
        let meta_base = scratch.offset(CHAIN_META_OFFSET);
        let argmax_dev = scratch.offset(CHAIN_ARGMAX_OFFSET);
        let prob_dev = scratch.offset(CHAIN_PROB_OFFSET);
        let embed = ctx.buffers.ssm_qkvz();
        let normed_embed = ctx.buffers.ssm_deinterleaved();
        let normed_hidden = ctx.buffers.residual();
        let embed_proj = ctx.buffers.qkv_output();
        let hidden = ctx.buffers.hidden_states();
        let logits = ctx.buffers.logits();

        let pcum_thresh = mtp_ragged_pcum_thresh();
        let allow_zero = ragged_mtp_allow_zero_drafts();
        let mut p_cum = vec![1.0f32; b];
        let mut steps_run = vec![0usize; b];
        let mut computed_drafts: Vec<Vec<u32>> = vec![Vec::with_capacity(max_steps); b];
        let mut emitted_drafts: Vec<Vec<u32>> = vec![Vec::with_capacity(max_steps); b];
        let mut emitted_pcum: Vec<Vec<f32>> = vec![Vec::with_capacity(max_steps); b];
        let mut computed_probs: Option<Vec<Vec<f32>>> =
            policy.log.then(|| vec![Vec::with_capacity(max_steps); b]);
        let mut current_tokens: Vec<u32> = last_tokens.to_vec();
        let mut prev_active_lanes: Vec<usize> = Vec::new();
        let mut tok_bytes = [0u8; 64 * 4];
        let mut prob_bytes = [0u8; 64 * 4];
        let mut step_tokens = [0u32; 64];
        let mut step_probs = [0.0f32; 64];

        // When `pcum_thresh <= 0.0` and every active lane runs all `max_steps`
        // (homogeneous multi-step batch), no host-side compaction can occur
        // between steps: pre-pack all `max_steps` metadata in ONE H2D, chain
        // `embed_from_argmax_k` directly off device `argmax` on GPU, and do
        // ONE batched `(token, prob)` D2H readback after the loop (0-ULP
        // bit-identical to per-step drains).
        let b_init = active_lanes.len();
        let (chain_prob_offset, chain_prob_span) = if b_init <= 8 {
            (CHAIN_PROB_OFFSET, CHAIN_PROB_SPAN)
        } else {
            (CHAIN_ARGMAX_OFFSET + 2048, 2048)
        };
        let chain_argmax_span = chain_prob_offset.saturating_sub(CHAIN_ARGMAX_OFFSET);
        let chain_prob_fits_before_meta =
            chain_prob_offset.saturating_add(chain_prob_span) <= CHAIN_META_OFFSET;
        let chain_bt_off = if b_init <= 8 { 768 } else { 2560 };
        let chain_mb = max_mb;
        let chain_bt_row_bytes = chain_mb * 4;
        let chain_stride = (chain_bt_off + b_init * chain_bt_row_bytes).next_multiple_of(256);
        let chain_gumbel = shared_gumbel_spec.and_then(|shared_spec| {
            let meta_steps_bytes = max_steps.checked_mul(chain_stride)?;
            let layout = BatchGumbelMetaLayout::compute(meta_steps_bytes, b_init, max_steps)?;
            let meta_end = CHAIN_META_OFFSET
                .checked_add(meta_steps_bytes)?
                .checked_add(layout.total_bytes)?;
            if meta_end > ctx.buffers.sizes().scratch {
                return None;
            }
            let specs: Vec<spark_runtime::sampler::gumbel::GumbelDraftSpec> = active_lanes
                .iter()
                .map(|&orig_s| gumbel_specs[orig_s].expect("uniform gumbel"))
                .collect();
            Some((shared_spec, specs, layout))
        });
        let chain_gumbel_bytes = chain_gumbel
            .as_ref()
            .map_or(0, |(_, _, layout)| layout.total_bytes);
        let chain_meta_end = CHAIN_META_OFFSET
            .checked_add(max_steps.saturating_mul(chain_stride))
            .and_then(|end| end.checked_add(chain_gumbel_bytes))
            .unwrap_or(usize::MAX);
        let gpu_chained = chain_prob_fits_before_meta
            && can_gpu_chain_ragged_steps(
                pcum_thresh,
                &active_lanes,
                &target_drafts,
                max_steps,
                chain_argmax_span,
                chain_prob_span,
                chain_meta_end,
                ctx.buffers.sizes().scratch,
            );
        let chain_prob_dev = scratch.offset(chain_prob_offset);
        if gpu_chained {
            let (
                pos_h_off,
                pos_w_off,
                last_tok_off,
                slot_off,
                seq_len_off,
                hidden_ptrs_off,
                bt_off,
            ) = if b_init <= 8 {
                (64, 128, 192, 256, 512, 576, 768)
            } else {
                (256, 512, 768, 1024, 1536, 2048, 2560)
            };
            {
                let mut packed_guard = self.packed_meta_scratch.lock();
                let packed = prepare_packed_meta_scratch(
                    &mut packed_guard,
                    max_steps * chain_stride + chain_gumbel_bytes,
                );
                for (local_i, &orig_s) in active_lanes.iter().enumerate() {
                    let state = states[orig_s]
                        .as_any()
                        .downcast_ref::<Qwen4MtpState>()
                        .expect("validated above");
                    packed[last_tok_off + local_i * 4..last_tok_off + local_i * 4 + 4]
                        .copy_from_slice(&current_tokens[orig_s].to_le_bytes());
                    packed[hidden_ptrs_off + local_i * 8..hidden_ptrs_off + local_i * 8 + 8]
                        .copy_from_slice(&target_hiddens[orig_s].0.to_le_bytes());
                    for (bi, &block) in state.block_table.iter().enumerate() {
                        let off = bt_off + local_i * chain_bt_row_bytes + bi * 4;
                        packed[off..off + 4].copy_from_slice(&(block as i32).to_le_bytes());
                    }
                }
                for step in 0..max_steps {
                    let s_off = step * chain_stride;
                    if step > 0 {
                        packed.copy_within(
                            bt_off..bt_off + b_init * chain_bt_row_bytes,
                            s_off + bt_off,
                        );
                    }
                    for (local_i, &orig_s) in active_lanes.iter().enumerate() {
                        let state = states[orig_s]
                            .as_any()
                            .downcast_ref::<Qwen4MtpState>()
                            .expect("validated above");
                        let [t, rh, rw] =
                            state.rotary_positions.position(positions[orig_s] + step)?;
                        packed[s_off + local_i * 4..s_off + local_i * 4 + 4]
                            .copy_from_slice(&t.to_le_bytes());
                        if !all_identity {
                            packed[s_off + pos_h_off + local_i * 4
                                ..s_off + pos_h_off + local_i * 4 + 4]
                                .copy_from_slice(&rh.to_le_bytes());
                            packed[s_off + pos_w_off + local_i * 4
                                ..s_off + pos_w_off + local_i * 4 + 4]
                                .copy_from_slice(&rw.to_le_bytes());
                        }
                        let r_seq = base_seq_lens[orig_s] + step;
                        let physical = state.block_table[r_seq / bs];
                        let slot = physical as i64 * bs as i64 + (r_seq % bs) as i64;
                        packed[s_off + slot_off + local_i * 8..s_off + slot_off + local_i * 8 + 8]
                            .copy_from_slice(&slot.to_le_bytes());
                        packed[s_off + seq_len_off + local_i * 4
                            ..s_off + seq_len_off + local_i * 4 + 4]
                            .copy_from_slice(&((r_seq + 1) as i32).to_le_bytes());
                    }
                }
                if let Some((_, ref act_specs, layout)) = chain_gumbel {
                    for (local_i, spec) in act_specs.iter().enumerate() {
                        let seed_off = layout.seeds_off + local_i * 8;
                        packed[seed_off..seed_off + 8].copy_from_slice(&spec.seed.to_le_bytes());
                        for step in 0..max_steps {
                            let pos_off = layout.step_pos_off(step, b_init) + local_i * 8;
                            let abs_pos = spec.step0_abs_pos + step as u64;
                            packed[pos_off..pos_off + 8].copy_from_slice(&abs_pos.to_le_bytes());
                        }
                    }
                }
                self.copy_h2d_group_fast(
                    ctx.gpu,
                    &[HostToDeviceCopy::new(packed, meta_base)],
                    stream,
                )?;
            }
        }

        for step in 0..max_steps {
            let b_act = active_lanes.len();
            if b_act == 0 {
                break;
            }
            // Compact surviving hidden rows in-place when any lane dropped.
            if !gpu_chained && step > 0 && active_lanes != prev_active_lanes {
                for (src_row, dst_row, count) in
                    plan_hidden_compaction_copies(&prev_active_lanes, &active_lanes)
                {
                    ctx.gpu.copy_d2d_async(
                        hidden.offset(src_row * r_bytes),
                        hidden.offset(dst_row * r_bytes),
                        count * r_bytes,
                        stream,
                    )?;
                }
            }

            let (
                pos_h_off,
                pos_w_off,
                last_tok_off,
                slot_off,
                seq_len_off,
                hidden_ptrs_off,
                bt_off,
            ) = if b_act <= 8 {
                (64, 128, 192, 256, 512, 576, 768)
            } else {
                (256, 512, 768, 1024, 1536, 2048, 2560)
            };
            let act_stride = (bt_off + b_act * bt_row_bytes).next_multiple_of(256);
            let step_meta = if gpu_chained {
                meta_base.offset(step * chain_stride)
            } else {
                meta_base
            };
            let step_argmax = if gpu_chained {
                argmax_dev.offset(step * b_act * 4)
            } else {
                argmax_dev
            };
            let step_prob = if gpu_chained {
                chain_prob_dev.offset(step * b_act * 4)
            } else {
                prob_dev
            };
            let step_gumbel = if gpu_chained {
                None
            } else {
                shared_gumbel_spec.and_then(|shared_spec| {
                    let layout = BatchGumbelMetaLayout::compute(act_stride, b_act, 1)?;
                    let meta_end = CHAIN_META_OFFSET
                        .checked_add(act_stride)?
                        .checked_add(layout.total_bytes)?;
                    if meta_end > ctx.buffers.sizes().scratch {
                        return None;
                    }
                    let specs: Vec<spark_runtime::sampler::gumbel::GumbelDraftSpec> = active_lanes
                        .iter()
                        .map(|&orig_s| gumbel_specs[orig_s].expect("uniform gumbel"))
                        .collect();
                    Some((shared_spec, specs, layout))
                })
            };
            let gumbel_bytes = step_gumbel
                .as_ref()
                .map_or(0, |(_, _, layout)| layout.total_bytes);

            if !gpu_chained {
                let mut packed_guard = self.packed_meta_scratch.lock();
                let packed =
                    prepare_packed_meta_scratch(&mut packed_guard, act_stride + gumbel_bytes);
                for (local_i, &orig_s) in active_lanes.iter().enumerate() {
                    let state = states[orig_s]
                        .as_any()
                        .downcast_ref::<Qwen4MtpState>()
                        .expect("validated above");
                    packed[last_tok_off + local_i * 4..last_tok_off + local_i * 4 + 4]
                        .copy_from_slice(&current_tokens[orig_s].to_le_bytes());
                    if step == 0 {
                        packed[hidden_ptrs_off + local_i * 8..hidden_ptrs_off + local_i * 8 + 8]
                            .copy_from_slice(&target_hiddens[orig_s].0.to_le_bytes());
                    }
                    let [t, rh, rw] = state.rotary_positions.position(positions[orig_s] + step)?;
                    packed[local_i * 4..local_i * 4 + 4].copy_from_slice(&t.to_le_bytes());
                    if !all_identity {
                        packed[pos_h_off + local_i * 4..pos_h_off + local_i * 4 + 4]
                            .copy_from_slice(&rh.to_le_bytes());
                        packed[pos_w_off + local_i * 4..pos_w_off + local_i * 4 + 4]
                            .copy_from_slice(&rw.to_le_bytes());
                    }
                    let r_seq = base_seq_lens[orig_s] + step;
                    let physical = state.block_table[r_seq / bs];
                    let slot = physical as i64 * bs as i64 + (r_seq % bs) as i64;
                    packed[slot_off + local_i * 8..slot_off + local_i * 8 + 8]
                        .copy_from_slice(&slot.to_le_bytes());
                    packed[seq_len_off + local_i * 4..seq_len_off + local_i * 4 + 4]
                        .copy_from_slice(&((r_seq + 1) as i32).to_le_bytes());
                    for (bi, &block) in state.block_table.iter().enumerate() {
                        let off = bt_off + local_i * bt_row_bytes + bi * 4;
                        packed[off..off + 4].copy_from_slice(&(block as i32).to_le_bytes());
                    }
                }
                if let Some((_, ref act_specs, layout)) = step_gumbel {
                    for (local_i, spec) in act_specs.iter().enumerate() {
                        let seed_off = layout.seeds_off + local_i * 8;
                        packed[seed_off..seed_off + 8].copy_from_slice(&spec.seed.to_le_bytes());
                        let pos_off = layout.step_pos_off(0, b_act) + local_i * 8;
                        let abs_pos = spec.step0_abs_pos + step as u64;
                        packed[pos_off..pos_off + 8].copy_from_slice(&abs_pos.to_le_bytes());
                    }
                }
                self.copy_h2d_group_fast(
                    ctx.gpu,
                    &[HostToDeviceCopy::new(packed, step_meta)],
                    stream,
                )?;
            }

            let tc_dense_scope =
                crate::layers::qwen4_hyper::batched_verify_tc_dense_active(b_act, false);
            let _tc_dense_guard =
                crate::layers::qwen4_hyper::BatchedVerifyTcDenseGuard::enter(tc_dense_scope);
            let coalesced_lm_head_eligible = should_use_mtp_lm_head_coalesced(
                crate::layers::qwen4_hyper::batched_verify_tc_component_active(
                    crate::layers::qwen4_hyper::BatchedVerifyTcComponent::LmHead,
                    b_act,
                    false,
                ),
                b_act,
                h,
            );
            let coalesced_fc_eligible = should_use_mtp_fc_coalesced(
                tc_dense_scope,
                b_act,
                hc,
                h,
                ctx.buffers.sizes().qkv_output,
                self.residual_add_broadcast_hc_k.0 != 0,
            );

            if step == 0 {
                if self.embed_dual_rms_norm_rows_packed_k.0 != 0
                    && h >= 1024
                    && r >= 1024
                    && b_act <= 8
                {
                    let tok_at = |idx: usize| -> u64 {
                        active_lanes
                            .get(idx)
                            .map(|&orig_s| current_tokens[orig_s] as u64)
                            .unwrap_or(0)
                    };
                    let tok_pack01 = tok_at(0) | (tok_at(1) << 32);
                    let tok_pack23 = tok_at(2) | (tok_at(3) << 32);
                    let tok_pack45 = tok_at(4) | (tok_at(5) << 32);
                    let tok_pack67 = tok_at(6) | (tok_at(7) << 32);
                    spark_runtime::kernel_args::KernelLaunch::new(
                        ctx.gpu,
                        self.embed_dual_rms_norm_rows_packed_k,
                    )
                    .grid([b_act as u32, 2, 1])
                    .block([1024, 1, 1])
                    .arg_ptr(self.embed_tokens.weight)
                    .arg_u64(tok_pack01)
                    .arg_u64(tok_pack23)
                    .arg_u64(tok_pack45)
                    .arg_u64(tok_pack67)
                    .arg_u32(ctx.config.vocab_size as u32)
                    .arg_ptr(self.pre_fc_norm_embedding.weight)
                    .arg_ptr(normed_embed)
                    .arg_ptr(target_hiddens[active_lanes[0]])
                    .arg_ptr(self.pre_fc_norm_hidden.weight)
                    .arg_ptr(normed_hidden)
                    .arg_u32(h as u32)
                    .arg_u32(r as u32)
                    .arg_f32(eps)
                    .arg_ptr(step_meta.offset(hidden_ptrs_off))
                    .launch(stream)?;
                } else {
                    spark_runtime::kernel_args::KernelLaunch::new(
                        ctx.gpu,
                        self.embed_from_argmax_k,
                    )
                    .grid([(h as u32).div_ceil(256), b_act as u32, 1])
                    .block([256, 1, 1])
                    .arg_ptr(step_meta.offset(last_tok_off))
                    .arg_ptr(self.embed_tokens.weight)
                    .arg_ptr(embed)
                    .arg_ptr(DevicePtr(0))
                    .arg_u32(h as u32)
                    .launch(stream)?;
                    if self.pre_fc_dual_norm_k.0 != 0 && h >= 1024 && r >= 1024 {
                        spark_runtime::kernel_args::KernelLaunch::new(
                            ctx.gpu,
                            self.pre_fc_dual_norm_k,
                        )
                        .grid([b_act as u32, 2, 1])
                        .block([1024, 1, 1])
                        .arg_ptr(embed)
                        .arg_ptr(self.pre_fc_norm_embedding.weight)
                        .arg_ptr(normed_embed)
                        .arg_ptr(target_hiddens[active_lanes[0]])
                        .arg_ptr(self.pre_fc_norm_hidden.weight)
                        .arg_ptr(normed_hidden)
                        .arg_u32(h as u32)
                        .arg_u32(r as u32)
                        .arg_f32(eps)
                        .arg_ptr(step_meta.offset(hidden_ptrs_off))
                        .launch(stream)?;
                    } else {
                        ops::rms_norm(
                            ctx.gpu,
                            self.rms_norm_k,
                            embed,
                            &self.pre_fc_norm_embedding,
                            normed_embed,
                            b_act as u32,
                            h as u32,
                            eps,
                            stream,
                        )?;
                        for (local_i, &orig_s) in active_lanes.iter().enumerate() {
                            ops::rms_norm(
                                ctx.gpu,
                                self.rms_norm_k,
                                target_hiddens[orig_s],
                                &self.pre_fc_norm_hidden,
                                normed_hidden.offset(local_i * r_bytes),
                                1,
                                r as u32,
                                eps,
                                stream,
                            )?;
                        }
                    }
                }
            } else {
                let prev_tok_ptr = if gpu_chained {
                    argmax_dev.offset((step - 1) * b_act * 4)
                } else {
                    step_meta.offset(last_tok_off)
                };
                spark_runtime::kernel_args::KernelLaunch::new(ctx.gpu, self.embed_from_argmax_k)
                    .grid([(h as u32).div_ceil(256), b_act as u32, 1])
                    .block([256, 1, 1])
                    .arg_ptr(prev_tok_ptr)
                    .arg_ptr(self.embed_tokens.weight)
                    .arg_ptr(embed)
                    .arg_ptr(DevicePtr(0))
                    .arg_u32(h as u32)
                    .launch(stream)?;
                if self.pre_fc_dual_norm_k.0 != 0 && h >= 1024 && r >= 1024 {
                    spark_runtime::kernel_args::KernelLaunch::new(ctx.gpu, self.pre_fc_dual_norm_k)
                        .grid([b_act as u32, 2, 1])
                        .block([1024, 1, 1])
                        .arg_ptr(embed)
                        .arg_ptr(self.pre_fc_norm_embedding.weight)
                        .arg_ptr(normed_embed)
                        .arg_ptr(hidden)
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
                        b_act as u32,
                        h as u32,
                        eps,
                        stream,
                    )?;
                    ops::rms_norm(
                        ctx.gpu,
                        self.rms_norm_k,
                        hidden,
                        &self.pre_fc_norm_hidden,
                        normed_hidden,
                        b_act as u32,
                        r as u32,
                        eps,
                        stream,
                    )?;
                }
            }

            let ran_coalesced_fc = if coalesced_fc_eligible
                && crate::layers::qwen4_fast_proj::try_coalesced_nvfp4_w4a16_gemm(
                    ctx.gpu,
                    normed_embed,
                    &self.fc_embedding,
                    embed_proj,
                    b_act,
                    h,
                    h,
                    stream,
                )?
                && self.try_coalesced_mtp_fc_hidden(
                    ctx.gpu,
                    normed_hidden,
                    hidden,
                    b_act * hc,
                    h,
                    stream,
                )? {
                spark_runtime::kernel_args::KernelLaunch::new(
                    ctx.gpu,
                    self.residual_add_broadcast_hc_k,
                )
                .grid([(h as u32).div_ceil(256), b_act as u32, 1])
                .block([256, 1, 1])
                .arg_ptr(hidden)
                .arg_ptr(embed_proj)
                .arg_u32(h as u32)
                .arg_u32(hc as u32)
                .launch(stream)?;
                true
            } else {
                false
            };
            if !ran_coalesced_fc {
                if self.fc_dual_broadcast_rows_k.0 != 0 {
                    spark_runtime::kernel_args::KernelLaunch::new(
                        ctx.gpu,
                        self.fc_dual_broadcast_rows_k,
                    )
                    .grid([(h as u32).div_ceil(4), b_act as u32, 1])
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
                    if b_act == 1 {
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
                    } else {
                        ops::w4a16_gemv_batch_logits_exact_with(
                            ctx.gpu,
                            self.w4a16_exact_fc_kernels,
                            normed_embed,
                            &self.fc_embedding,
                            embed_proj,
                            b_act as u32,
                            h as u32,
                            h as u32,
                            stream,
                            false,
                        )?;
                    }
                    ops::w4a16_gemv_batch_logits_exact_with(
                        ctx.gpu,
                        self.w4a16_exact_fc_kernels,
                        normed_hidden,
                        &self.fc_hidden,
                        hidden,
                        (b_act * hc) as u32,
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
                        .grid([(h as u32).div_ceil(256), b_act as u32, 1])
                        .block([256, 1, 1])
                        .arg_ptr(hidden)
                        .arg_ptr(embed_proj)
                        .arg_u32(h as u32)
                        .arg_u32(hc as u32)
                        .launch(stream)?;
                    } else {
                        for j in 0..b_act {
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
            }

            let (positions_h, positions_w) = if all_identity {
                (step_meta, step_meta)
            } else {
                (step_meta.offset(pos_h_off), step_meta.offset(pos_w_off))
            };
            let metadata = AttnMetadataDev {
                qwen4_qsa_required: self.mtp_qsa_active(),
                positions: step_meta,
                positions_h,
                positions_w,
                slot: step_meta.offset(slot_off),
                seq_len: step_meta.offset(seq_len_off),
                block_table: step_meta.offset(bt_off),
                max_blocks_per_seq: if gpu_chained {
                    chain_mb as u32
                } else {
                    max_mb as u32
                },
                num_seqs: b_act as u32,
            };
            let layer_ctx = ForwardContext {
                attn_metadata: Some(metadata),
                comm: None,
                graph_capture: false,
                ..*ctx
            };
            let step_seq_lens: Vec<usize> = active_lanes
                .iter()
                .map(|&orig_s| base_seq_lens[orig_s] + step)
                .collect();
            {
                let mut typed_states: Vec<&mut Qwen4MtpState> = Vec::with_capacity(b_act);
                let mut cursor = 0usize;
                for (orig_s, st) in states.iter_mut().enumerate() {
                    if cursor < b_act && active_lanes[cursor] == orig_s {
                        typed_states.push(
                            st.as_any_mut()
                                .downcast_mut::<Qwen4MtpState>()
                                .expect("validated above"),
                        );
                        cursor += 1;
                    }
                }
                let mut block_tables: Vec<&mut Vec<u32>> = Vec::with_capacity(b_act);
                let mut disk_block_ids: Vec<&mut Vec<u32>> = Vec::with_capacity(b_act);
                let mut disk_last_offloaded: Vec<&mut Vec<u32>> = Vec::with_capacity(b_act);
                for st in typed_states.iter_mut() {
                    let Qwen4MtpState {
                        block_table,
                        disk_block_ids: d_ids,
                        disk_last_offloaded_per_layer: d_off,
                        ..
                    } = &mut **st;
                    block_tables.push(block_table);
                    disk_block_ids.push(d_ids);
                    disk_last_offloaded.push(d_off);
                }
                self.layer.decode_qwen4_cross_seq_mtp(
                    hidden,
                    ctx.buffers.residual(),
                    &step_seq_lens,
                    &mut block_tables,
                    &mut disk_block_ids,
                    &mut disk_last_offloaded,
                    &mut kv_cache,
                    &layer_ctx,
                    stream,
                )?;
            }

            let sample_hiddens = if b_act == 1 {
                let (mixed, inject) = self.final_mixer.prepare_decode(
                    hidden,
                    ctx.buffers.residual(),
                    ctx.buffers,
                    ctx.gpu,
                    eps,
                    stream,
                )?;
                debug_assert!(inject.is_none());
                mixed
            } else {
                self.final_mixer.prepare_batched(
                    hidden,
                    ctx.buffers.residual(),
                    b_act,
                    ctx.buffers,
                    ctx.gpu,
                    eps,
                    stream,
                )?
            };

            if b_act == 1 {
                ops::w4a16_decode_gemv(
                    ctx.gpu,
                    self.w4a16_gemv_k,
                    self.w4a16_gemv_sw_k,
                    self.gemv_sw,
                    sample_hiddens,
                    &self.lm_head_nvfp4,
                    logits,
                    vocab as u32,
                    h as u32,
                    stream,
                )?;
            } else {
                let ran_coalesced_lm = if coalesced_lm_head_eligible {
                    crate::layers::qwen4_fast_proj::try_coalesced_nvfp4_w4a16_gemm(
                        ctx.gpu,
                        sample_hiddens,
                        &self.lm_head_nvfp4,
                        logits,
                        b_act,
                        vocab,
                        h,
                        stream,
                    )?
                } else {
                    false
                };
                if !ran_coalesced_lm {
                    if should_use_mtp_lm_head_pipe(
                        mtp_lm_head_tc_pipe_enabled(),
                        b_act,
                        h,
                        self.w4a16_gemm_pipe_k.0 != 0,
                        sample_hiddens,
                        self.lm_head_nvfp4.weight,
                        logits,
                    ) {
                        ops::w4a16_gemm_pipe(
                            ctx.gpu,
                            self.w4a16_gemm_pipe_k,
                            sample_hiddens,
                            &self.lm_head_nvfp4,
                            logits,
                            b_act as u32,
                            vocab as u32,
                            h as u32,
                            stream,
                        )?;
                    } else {
                        ops::w4a16_gemv_batch_logits_exact_with(
                            ctx.gpu,
                            self.w4a16_exact_fc_kernels,
                            sample_hiddens,
                            &self.lm_head_nvfp4,
                            logits,
                            b_act as u32,
                            vocab as u32,
                            h as u32,
                            stream,
                            ops::w4a16_exact_rt2_for_rows(
                                ops::w4a16_gemv_rt2_enabled(),
                                ops::w4a16_exact_rt2_min_rows(),
                                b_act as u32,
                            ),
                        )?;
                    }
                }
            }

            let active_gumbel = if gpu_chained {
                chain_gumbel
                    .as_ref()
                    .map(|(s, specs, l)| (*s, specs, *l, step))
            } else {
                step_gumbel.as_ref().map(|(s, specs, l)| (*s, specs, *l, 0))
            };
            if let Some((shared_spec, act_specs, layout, pos_step)) = active_gumbel {
                gumbel_draft::select_chain_gumbel_batch(
                    ctx.gpu,
                    self.mtp_gumbel_argmax_k,
                    logits,
                    DevicePtr::NULL,
                    meta_base.offset(layout.seeds_off),
                    meta_base.offset(layout.step_pos_off(pos_step, b_act)),
                    step_argmax,
                    Some(step_prob),
                    b_act,
                    vocab,
                    shared_spec,
                    act_specs,
                    step as u64,
                    stream,
                )?;
            } else if any_gumbel {
                for (local_i, &orig_s) in active_lanes.iter().enumerate() {
                    if let Some(spec) = gumbel_specs[orig_s] {
                        gumbel_draft::select_chain_gumbel_step(
                            ctx.gpu,
                            self.mtp_gumbel_argmax_k,
                            logits.offset(local_i * vocab * 2),
                            DevicePtr::NULL,
                            step_argmax.offset(local_i * 4),
                            Some(step_prob.offset(local_i * 4)),
                            vocab,
                            spec,
                            spec.step0_abs_pos + step as u64,
                            stream,
                        )?;
                    } else {
                        spark_runtime::kernel_args::KernelLaunch::new(ctx.gpu, self.argmax_prob_k)
                            .grid([1, 1, 1])
                            .block([1024, 1, 1])
                            .arg_ptr(logits.offset(local_i * vocab * 2))
                            .arg_ptr(step_argmax.offset(local_i * 4))
                            .arg_ptr(step_prob.offset(local_i * 4))
                            .arg_u32(vocab as u32)
                            .launch(stream)?;
                    }
                }
            } else {
                self.launch_argmax_prob_batch(
                    ctx.gpu,
                    logits,
                    step_argmax,
                    step_prob,
                    b_act,
                    vocab,
                    stream,
                )?;
            }

            if !gpu_chained {
                // ONE single-sync tiny `[b_act]` (token, prob) pair readback per step.
                let n_bytes = b_act * 4;
                ctx.gpu.copy_d2h_pair_on_stream(
                    step_argmax,
                    &mut tok_bytes[..n_bytes],
                    step_prob,
                    &mut prob_bytes[..n_bytes],
                    stream,
                )?;
                for local_i in 0..b_act {
                    let off = local_i * 4;
                    step_tokens[local_i] = u32::from_le_bytes([
                        tok_bytes[off],
                        tok_bytes[off + 1],
                        tok_bytes[off + 2],
                        tok_bytes[off + 3],
                    ]);
                    step_probs[local_i] = f32::from_le_bytes([
                        prob_bytes[off],
                        prob_bytes[off + 1],
                        prob_bytes[off + 2],
                        prob_bytes[off + 3],
                    ]);
                }
                let mut next_active = advance_ragged_mtp_step(
                    step,
                    max_steps,
                    pcum_thresh,
                    allow_zero,
                    &active_lanes,
                    &step_tokens[..b_act],
                    &step_probs[..b_act],
                    &target_drafts,
                    &mut p_cum,
                    &mut steps_run,
                    &mut computed_drafts,
                    &mut emitted_drafts,
                    &mut current_tokens,
                    computed_probs.as_deref_mut(),
                );
                for &orig_s in &active_lanes {
                    if emitted_drafts[orig_s].len() > emitted_pcum[orig_s].len() {
                        emitted_pcum[orig_s].push(p_cum[orig_s]);
                    }
                }
                if ragged_mtp_should_stop_stragglers(b, step, next_active.len()) {
                    next_active.clear();
                }
                prev_active_lanes = active_lanes;
                active_lanes = next_active;
            }
            drop(_tc_dense_guard);
        }
        if gpu_chained {
            let total_bytes = max_steps * b_init * 4;
            let mut all_tok_bytes = vec![0u8; total_bytes];
            let mut all_prob_bytes = vec![0u8; total_bytes];
            ctx.gpu.copy_d2h_pair_on_stream(
                argmax_dev,
                &mut all_tok_bytes,
                chain_prob_dev,
                &mut all_prob_bytes,
                stream,
            )?;
            for step in 0..max_steps {
                let step_base = step * b_init * 4;
                for local_i in 0..b_init {
                    let off = step_base + local_i * 4;
                    step_tokens[local_i] = u32::from_le_bytes([
                        all_tok_bytes[off],
                        all_tok_bytes[off + 1],
                        all_tok_bytes[off + 2],
                        all_tok_bytes[off + 3],
                    ]);
                    step_probs[local_i] = f32::from_le_bytes([
                        all_prob_bytes[off],
                        all_prob_bytes[off + 1],
                        all_prob_bytes[off + 2],
                        all_prob_bytes[off + 3],
                    ]);
                }
                let _next = advance_ragged_mtp_step(
                    step,
                    max_steps,
                    pcum_thresh,
                    allow_zero,
                    &active_lanes,
                    &step_tokens[..b_init],
                    &step_probs[..b_init],
                    &target_drafts,
                    &mut p_cum,
                    &mut steps_run,
                    &mut computed_drafts,
                    &mut emitted_drafts,
                    &mut current_tokens,
                    computed_probs.as_deref_mut(),
                );
                for &orig_s in &active_lanes {
                    if emitted_drafts[orig_s].len() > emitted_pcum[orig_s].len() {
                        emitted_pcum[orig_s].push(p_cum[orig_s]);
                    }
                }
            }
        }
        drop(kv_cache);
        quantize_ragged_drafts_to_graph_bucket(&mut emitted_drafts, &mut emitted_pcum, allow_zero);

        for (s, st) in states.iter_mut().enumerate() {
            let state = st
                .as_any_mut()
                .downcast_mut::<Qwen4MtpState>()
                .expect("validated above");
            state.seq_len += steps_run[s];
            state.last_num_drafted = steps_run[s];
            state.last_drafts.clear();
            state.last_drafts.extend_from_slice(&computed_drafts[s]);
            if let Some(ref probs_log) = computed_probs {
                tracing::info!(
                    "MTP_DRAFT_RAGGED_BATCH seq={s} pos={} target={} computed={} kept={} \
                     drafts={:?} p={:.4?}",
                    positions[s],
                    target_drafts[s],
                    steps_run[s],
                    emitted_drafts[s].len(),
                    emitted_drafts[s],
                    probs_log[s]
                );
            }
        }
        Ok(Some(emitted_drafts))
    }
}
