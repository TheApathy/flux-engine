// SPDX-License-Identifier: AGPL-3.0-only

//! `DraftProposer` trait implementation for `Qwen4MtpHead`.

use anyhow::{Result, ensure};
use parking_lot::Mutex;
use spark_runtime::gpu::{DevicePtr, GpuBackend, HostToDeviceCopy};

use super::graph_cache::MtpBatchedChainLaunch;
use super::gumbel_draft;
use super::init::Qwen4MtpHead;
use super::policy::{
    BatchGumbelMetaLayout, CHAIN_ARGMAX_OFFSET, CHAIN_META_OFFSET, CHAIN_PROB_OFFSET,
    CHAIN_PROB_SPAN, CHAIN_TOKEN_OFFSET, ChainConfidenceScope, DraftStepReuseAction,
    batched_ragged_mtp_enabled, draft_step_qsa_plan, mtp_batch_rows_enabled, mtp_chain_enabled,
    mtp_cross_seq_propose_enabled, mtp_graph_bt_blocks_cap, mtp_graph_enabled,
    mtp_lm_head_tc_pipe_enabled, mtp_qsa_mode, prepare_packed_meta_scratch, prompt_tail_start,
    record_row_stats, resolve_chain_confidence_gate_for_scope, rows_policy_logged,
    should_use_mtp_fc_coalesced, should_use_mtp_lm_head_coalesced, should_use_mtp_lm_head_pipe,
    validate_rotary_span,
};
use super::state::{PendingReplay, Qwen4MtpState};
use crate::layer::{AttnMetadataDev, ForwardContext};
use crate::layers::ops;
use crate::layers::qwen4_mtp_gate;
use crate::layers::qwen4_mtp_rows::{self, AfterVerifyPlan, RowPolicy};
use crate::speculative::{DraftProposer, ProposerState};

impl DraftProposer for Qwen4MtpHead {
    fn kind(&self) -> crate::speculative::DraftProposerKind {
        crate::speculative::DraftProposerKind::Qwen4NativeMtp
    }

    fn alloc_state(&self, gpu: &dyn GpuBackend) -> Result<Box<dyn ProposerState>> {
        Ok(Box::new(Qwen4MtpState {
            rotary_positions: crate::traits::RotaryPositions::identity(),
            layer_state: self.layer.alloc_state(gpu)?,
            block_table: Vec::new(),
            disk_block_ids: Vec::new(),
            disk_last_offloaded_per_layer: vec![u32::MAX],
            seq_len: 0,
            last_num_drafted: 0,
            last_drafts: Vec::new(),
            pending_replay: None,
            verify_frame: None,
            staged_frame: Mutex::new(Default::default()),
            pending_gumbel_spec: Mutex::new(None),
        }))
    }

    fn propose(
        &self,
        last_token: u32,
        target_hidden: DevicePtr,
        position: usize,
        num_drafts: usize,
        state: &mut dyn ProposerState,
        ctx: &ForwardContext,
        stream: u64,
        _draft_embed_target: Option<DevicePtr>,
        grammar_bitmask: Option<&[i32]>,
        target_hidden_stack: Option<DevicePtr>,
    ) -> Result<Vec<u32>> {
        let state = state
            .as_any_mut()
            .downcast_mut::<Qwen4MtpState>()
            .ok_or_else(|| anyhow::anyhow!("Invalid Qwen4 MTP proposer state"))?;

        // Feature 19: Out-of-pruned-range tokens fallback to base model
        if qwen4_mtp_gate::mtp_vocab_prune_selected()
            && last_token >= qwen4_mtp_gate::PRUNED_VOCAB_SIZE
        {
            state.last_num_drafted = 0;
            state.clear_gumbel_spec();
            spark_runtime::sampler::gumbel::clear_gumbel_draft_specs();
            return Ok(Vec::new());
        }

        let policy = qwen4_mtp_gate::current_policy();
        let num_drafts = policy.chain_len(num_drafts);
        // K7: rewrite the committed draft rows planned by `after_verify`
        // before this chain reads the head cache (drafts only; lossless).
        // In `verify_k2/k3/k4_step` partial/reject paths, `after_verify` runs
        // before `save_hidden_for_mtp`, so `staged_at_trim` is false and we
        // consume the staged frame token here before launching the chain.
        let mut replay_rows = if let Some(mut pending) = state.pending_replay.take() {
            if !pending.staged_at_trim {
                let staged = qwen4_mtp_rows::recondition_engaged()
                    && state.take_staged_frame(pending.owner, pending.staged_accepted)?;
                if staged {
                    if pending.computed == 0 {
                        let max_rows = match pending.row_policy {
                            RowPolicy::Recondition { max_rows } => max_rows,
                            _ => 0,
                        };
                        pending.plan.replay_rows = pending.accepted.min(max_rows);
                    } else {
                        let reconditioned = qwen4_mtp_rows::after_verify_plan(
                            pending.row_policy,
                            pending.computed,
                            pending.staged_accepted,
                            true,
                        );
                        let rewind = pending
                            .plan
                            .keep_rows
                            .saturating_sub(reconditioned.keep_rows);
                        state.seq_len = state.seq_len.saturating_sub(rewind);
                        pending.plan = reconditioned;
                    }
                }
                record_row_stats(pending.computed, pending.staged_accepted, pending.plan);
            }
            if pending.plan.replay_rows > 0 {
                self.collect_replay_rows(pending, target_hidden_stack, position, state, ctx)?
            } else {
                None
            }
        } else {
            None
        };
        state.staged_frame.lock().clear();
        validate_rotary_span(
            &state.rotary_positions,
            position,
            num_drafts,
            ctx.config.mrope_interleaved,
            self.mrope_kernel_available,
            ctx.config.yarn_factor,
        )?;
        // When `forward_rows` covers `m + 1` rows (`m` reconditioned committed
        // rows plus `draft[0]`), fuse them into a single batched MTP pass;
        // otherwise replay the `m` committed rows before `draft[0]`.
        let fuse_first_draft = replay_rows.as_ref().is_some_and(|(t, _, _)| {
            num_drafts > 0 && mtp_batch_rows_enabled() && self.forward_rows_admits(t.len() + 1)
        });
        let first_batch = if fuse_first_draft {
            let (mut tokens, mut hiddens, mut positions) = replay_rows.take().unwrap();
            tokens.push(last_token);
            hiddens.push(target_hidden);
            positions.push(position);
            Some((tokens, hiddens, positions))
        } else {
            if let Some((tokens, hiddens, positions)) = replay_rows.take() {
                let qsa_replay = self.mtp_qsa_active();
                for ((token, hidden), pos) in tokens.into_iter().zip(hiddens).zip(positions) {
                    self.forward_one(
                        token, hidden, pos, state, ctx, stream, None, false, qsa_replay, None,
                    )?;
                }
            }
            None
        };
        if grammar_bitmask.is_none()
            && num_drafts > 0
            && self.embed_from_argmax_k.0 != 0
            && mtp_chain_enabled()
        {
            let batch_refs = first_batch
                .as_ref()
                .map(|(t, h, p)| (t.as_slice(), h.as_slice(), p.as_slice()));
            let (drafts, computed) = self.propose_chain(
                last_token,
                target_hidden,
                position,
                num_drafts,
                policy,
                batch_refs,
                state,
                ctx,
                stream,
            )?;
            // The chain wrote `computed` MTP KV rows even when the confidence
            // gate returns fewer drafts; after_verify trims from this count.
            state.last_num_drafted = computed;
            state.clear_gumbel_spec();
            spark_runtime::sampler::gumbel::clear_gumbel_draft_specs();
            return Ok(drafts);
        }
        let mut drafts = Vec::with_capacity(num_drafts);
        let mut token = last_token;
        let mut hidden = target_hidden;
        let r_bytes = ctx.config.residual_width() * 2;
        let replay_prefix = first_batch.as_ref().map_or(0, |(t, _, _)| t.len() - 1);
        let first_seq_len = state.seq_len + replay_prefix;
        let qsa_mode = mtp_qsa_mode();
        let has_qsa = self.layer.has_qsa_side_cache();
        for step in 0..num_drafts {
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
            if step == 0
                && let Some((ref b_tokens, ref b_hiddens, ref b_positions)) = first_batch
            {
                let replay_prefix = b_tokens.len() - 1;
                token = self.forward_rows(
                    b_tokens,
                    b_hiddens,
                    b_positions,
                    ctx.buffers.scratch().offset(CHAIN_META_OFFSET),
                    state,
                    ctx,
                    stream,
                    grammar_bitmask,
                    true,
                    None,
                )?;
                drafts.push(token);
                hidden = ctx.buffers.hidden_states().offset(replay_prefix * r_bytes);
                continue;
            }
            token = self.forward_one(
                token,
                hidden,
                position + step,
                state,
                ctx,
                stream,
                grammar_bitmask,
                true,
                qsa_required,
                None,
            )?;
            drafts.push(token);
            hidden = ctx.buffers.hidden_states();
        }
        state.last_num_drafted = drafts.len();
        state.last_drafts.clear();
        state.last_drafts.extend_from_slice(&drafts);
        state.clear_gumbel_spec();
        spark_runtime::sampler::gumbel::clear_gumbel_draft_specs();
        Ok(drafts)
    }

    fn propose_batch(
        &self,
        last_tokens: &[u32],
        target_hiddens: &[DevicePtr],
        positions: &[usize],
        num_drafts: usize,
        states: &mut [&mut dyn ProposerState],
        ctx: &ForwardContext,
        stream: u64,
        target_hidden_stacks: &[Option<DevicePtr>],
    ) -> Result<Option<Vec<Vec<u32>>>> {
        let b = last_tokens.len();
        if !(2..=64).contains(&b)
            || target_hiddens.len() != b
            || positions.len() != b
            || states.len() != b
            || target_hidden_stacks.len() != b
            || self.mtp_qsa_active()
            || !mtp_cross_seq_propose_enabled()
            || !mtp_chain_enabled()
            || self.embed_from_argmax_k.0 == 0
            || !self.propose_batch_rows_admits(b, ctx)
        {
            return Ok(None);
        }
        if qwen4_mtp_gate::mtp_vocab_prune_selected()
            && last_tokens
                .iter()
                .any(|&t| t >= qwen4_mtp_gate::PRUNED_VOCAB_SIZE)
        {
            return Ok(None);
        }
        let policy = qwen4_mtp_gate::current_policy();
        let num_drafts = policy.chain_len(num_drafts);
        if num_drafts == 0 {
            return Ok(None);
        }
        let mut gumbel_specs = Vec::with_capacity(b);
        for (s, st) in states.iter().enumerate() {
            let Some(state) = st.as_any().downcast_ref::<Qwen4MtpState>() else {
                return Ok(None);
            };
            gumbel_specs.push(state.resolve_gumbel_spec(positions[s]));
        }
        let any_gumbel = gumbel_specs.iter().any(Option::is_some);
        let all_gumbel = gumbel_specs.iter().all(Option::is_some);
        if any_gumbel && batched_ragged_mtp_enabled() && (self.argmax_prob_k.0 != 0 || all_gumbel) {
            return self.propose_batch_ragged_compacted(
                last_tokens,
                target_hiddens,
                positions,
                num_drafts,
                policy,
                &gumbel_specs,
                any_gumbel,
                states,
                ctx,
                stream,
            );
        }
        // Batched (b >= 2) cohort-uniform drafting never applies per-sequence
        // MTP_CONF early-stop syncs or prefix truncation so every sequence in
        // the cohort shares one verify width K and keeps `argmax_bf16_rows`
        // active, allowing C=1 (`propose_chain`) to retain `MTP_CONF=0.6` on
        // multi-concurrency servers without breaking C>=2 cohorts.
        let conf_gate = resolve_chain_confidence_gate_for_scope(
            ChainConfidenceScope::BatchedCohort,
            policy,
            policy.wants_probabilities_batch(),
            policy.early_threshold_batch(),
            self.argmax_prob_k.0 != 0,
            all_gumbel,
        );
        let want_prob = conf_gate.want_prob;
        debug_assert!(conf_gate.early.is_none() && !conf_gate.apply_truncate);
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
        let (prob_offset, prob_span, token_offset) = if b <= 8 {
            (CHAIN_PROB_OFFSET, CHAIN_PROB_SPAN, CHAIN_TOKEN_OFFSET)
        } else {
            (CHAIN_ARGMAX_OFFSET + 2048, 2048, CHAIN_ARGMAX_OFFSET + 4096)
        };
        let argmax_limit = if want_prob { prob_offset } else { token_offset };
        if hc != 4
            || b * vocab * 2 > ctx.buffers.sizes().logits
            || CHAIN_ARGMAX_OFFSET + num_drafts * b * 4 > argmax_limit
            || (want_prob && num_drafts * b * 4 > prob_span)
            || token_offset + num_drafts * b * 4 > CHAIN_META_OFFSET
        {
            return Ok(None);
        }

        let mut all_identity = true;
        for (s, st) in states.iter().enumerate() {
            let Some(state) = st.as_any().downcast_ref::<Qwen4MtpState>() else {
                return Ok(None);
            };
            if state.pending_replay.is_some() {
                return Ok(None);
            }
            validate_rotary_span(
                &state.rotary_positions,
                positions[s],
                num_drafts,
                ctx.config.mrope_interleaved,
                self.mrope_kernel_available,
                ctx.config.yarn_factor,
            )?;
            if !state.rotary_positions.is_identity() {
                all_identity = false;
            }
        }

        let mut kv_cache = self.kv_cache.lock();
        let bs = kv_cache.block_size();
        let mut total_new_blocks = 0usize;
        let mut max_mb = 1usize;
        for st in states.iter() {
            let state = st
                .as_any()
                .downcast_ref::<Qwen4MtpState>()
                .expect("validated above");
            let last = state.seq_len + num_drafts - 1;
            let blocks_needed = last / bs + 1;
            total_new_blocks += blocks_needed.saturating_sub(state.block_table.len());
            max_mb = max_mb.max(state.block_table.len().max(blocks_needed));
        }
        if kv_cache.num_free_blocks() < total_new_blocks {
            return Ok(None);
        }
        let (pos_h_off, pos_w_off, last_tok_off, slot_off, seq_len_off, hidden_ptrs_off, bt_off) =
            if b <= 8 {
                (64, 128, 192, 256, 512, 576, 768)
            } else {
                (256, 512, 768, 1024, 1536, 2048, 2560)
            };
        let batch_graph_candidate = mtp_graph_enabled()
            && stream != 0
            && !ctx.graph_capture
            && !any_gumbel
            && all_identity
            && self.embed_from_argmax_k.0 != 0
            && self.pre_fc_dual_norm_k.0 != 0
            && h >= 1024
            && r >= 1024
            && kv_cache.num_layers() > 0;
        let effective_mb = if batch_graph_candidate {
            mtp_graph_bt_blocks_cap(max_mb, b, num_drafts, bt_off, ctx.buffers.sizes().scratch)
        } else {
            max_mb
        };
        let bt_row_bytes = effective_mb * 4;
        let stride = (bt_off + b * bt_row_bytes).next_multiple_of(256);
        if CHAIN_META_OFFSET + num_drafts * stride > ctx.buffers.sizes().scratch {
            return Ok(None);
        }
        for st in states.iter() {
            if let Some(state) = st.as_any().downcast_ref::<Qwen4MtpState>() {
                state.clear_gumbel_spec();
            }
        }
        spark_runtime::sampler::gumbel::clear_gumbel_draft_specs();
        for st in states.iter_mut() {
            let state = st
                .as_any_mut()
                .downcast_mut::<Qwen4MtpState>()
                .expect("validated above");
            let last = state.seq_len + num_drafts - 1;
            let blocks_needed = last / bs + 1;
            while state.block_table.len() < blocks_needed {
                state.block_table.push(kv_cache.alloc_block()?);
            }
        }
        let batch_gumbel =
            gumbel_draft::uniform_batch_gumbel_spec(&gumbel_specs).and_then(|shared_spec| {
                let layout = BatchGumbelMetaLayout::compute(num_drafts * stride, b, num_drafts)?;
                let meta_end = CHAIN_META_OFFSET
                    .checked_add(num_drafts * stride)?
                    .checked_add(layout.total_bytes)?;
                if meta_end > ctx.buffers.sizes().scratch {
                    return None;
                }
                let specs: Vec<spark_runtime::sampler::gumbel::GumbelDraftSpec> =
                    gumbel_specs.iter().flatten().copied().collect();
                Some((shared_spec, specs, layout))
            });
        let gumbel_meta_bytes = batch_gumbel
            .as_ref()
            .map_or(0, |(_, _, layout)| layout.total_bytes);

        let mut base_seq_lens = Vec::with_capacity(b);
        let packed_len = num_drafts * stride + gumbel_meta_bytes;
        let mut packed_guard = self.packed_meta_scratch.lock();
        let packed = prepare_packed_meta_scratch(&mut packed_guard, packed_len);
        for (row, st) in states.iter_mut().enumerate() {
            let state = st
                .as_any_mut()
                .downcast_mut::<Qwen4MtpState>()
                .expect("validated above");
            state.staged_frame.lock().clear();
            base_seq_lens.push(state.seq_len);
            packed[last_tok_off + row * 4..last_tok_off + row * 4 + 4]
                .copy_from_slice(&last_tokens[row].to_le_bytes());
            packed[hidden_ptrs_off + row * 8..hidden_ptrs_off + row * 8 + 8]
                .copy_from_slice(&target_hiddens[row].0.to_le_bytes());
            for (i, &block) in state.block_table.iter().enumerate() {
                let off = bt_off + row * bt_row_bytes + i * 4;
                packed[off..off + 4].copy_from_slice(&(block as i32).to_le_bytes());
            }
        }
        for step in 0..num_drafts {
            let s_off = step * stride;
            if step > 0 {
                packed.copy_within(bt_off..bt_off + b * bt_row_bytes, s_off + bt_off);
            }
            for (row, st) in states.iter().enumerate() {
                let state = st
                    .as_any()
                    .downcast_ref::<Qwen4MtpState>()
                    .expect("validated above");
                let [t, rh, rw] = state.rotary_positions.position(positions[row] + step)?;
                packed[s_off + row * 4..s_off + row * 4 + 4].copy_from_slice(&t.to_le_bytes());
                if !all_identity {
                    packed[s_off + pos_h_off + row * 4..s_off + pos_h_off + row * 4 + 4]
                        .copy_from_slice(&rh.to_le_bytes());
                    packed[s_off + pos_w_off + row * 4..s_off + pos_w_off + row * 4 + 4]
                        .copy_from_slice(&rw.to_le_bytes());
                }
                let r_seq = base_seq_lens[row] + step;
                let physical = state.block_table[r_seq / bs];
                let slot = physical as i64 * bs as i64 + (r_seq % bs) as i64;
                packed[s_off + slot_off + row * 8..s_off + slot_off + row * 8 + 8]
                    .copy_from_slice(&slot.to_le_bytes());
                packed[s_off + seq_len_off + row * 4..s_off + seq_len_off + row * 4 + 4]
                    .copy_from_slice(&((r_seq + 1) as i32).to_le_bytes());
            }
        }
        if let Some((_, batch_specs, layout)) = &batch_gumbel {
            for (row, spec) in batch_specs.iter().enumerate() {
                let seed_off = layout.seeds_off + row * 8;
                packed[seed_off..seed_off + 8].copy_from_slice(&spec.seed.to_le_bytes());
                for step in 0..num_drafts {
                    let pos_off = layout.step_pos_off(step, b) + row * 8;
                    let abs_pos = spec.step0_abs_pos + step as u64;
                    packed[pos_off..pos_off + 8].copy_from_slice(&abs_pos.to_le_bytes());
                }
            }
        }

        let scratch = ctx.buffers.scratch();
        let meta_base = scratch.offset(CHAIN_META_OFFSET);
        self.copy_h2d_group_fast(ctx.gpu, &[HostToDeviceCopy::new(packed, meta_base)], stream)?;
        drop(packed_guard);

        let argmax_base = scratch.offset(CHAIN_ARGMAX_OFFSET);
        let prob_base = scratch.offset(prob_offset);
        let embed = ctx.buffers.ssm_qkvz();
        let normed_embed = ctx.buffers.ssm_deinterleaved();
        let normed_hidden = ctx.buffers.residual();
        let embed_proj = ctx.buffers.qkv_output();
        let hidden = ctx.buffers.hidden_states();
        let logits = ctx.buffers.logits();

        let actual_computed = num_drafts;
        let ran_batch_graph = if batch_graph_candidate {
            let launch_spec = MtpBatchedChainLaunch {
                batch_size: b,
                step_count: num_drafts,
                want_prob,
                bt_blocks_cap: effective_mb,
                stride,
                last_tok_off,
                slot_off,
                seq_len_off,
                hidden_ptrs_off,
                bt_off,
                vocab,
                meta_base,
                argmax_base,
                prob_base,
                token_out_base: Some(scratch.offset(token_offset)),
                base_seq_lens: &base_seq_lens,
            };
            self.run_batched_mtp_chain_graphed(
                &launch_spec,
                None,
                states,
                &mut kv_cache,
                ctx,
                stream,
            )?
        } else {
            false
        };
        let tc_dense_scope = crate::layers::qwen4_hyper::batched_verify_tc_dense_active(b, false);
        let _tc_dense_guard =
            crate::layers::qwen4_hyper::BatchedVerifyTcDenseGuard::enter(tc_dense_scope);
        let coalesced_lm_head_eligible = should_use_mtp_lm_head_coalesced(
            crate::layers::qwen4_hyper::batched_verify_tc_component_active(
                crate::layers::qwen4_hyper::BatchedVerifyTcComponent::LmHead,
                b,
                false,
            ),
            b,
            h,
        );
        let coalesced_fc_eligible = should_use_mtp_fc_coalesced(
            tc_dense_scope,
            b,
            hc,
            h,
            ctx.buffers.sizes().qkv_output,
            self.residual_add_broadcast_hc_k.0 != 0,
        );

        for step in 0..if ran_batch_graph { 0 } else { num_drafts } {
            if step == 0 {
                if self.embed_dual_rms_norm_rows_packed_k.0 != 0 && h >= 1024 && r >= 1024 && b <= 8
                {
                    let tok_at = |idx: usize| -> u64 {
                        last_tokens
                            .get(idx)
                            .copied()
                            .filter(|_| idx < b)
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
                    .grid([b as u32, 2, 1])
                    .block([1024, 1, 1])
                    .arg_ptr(self.embed_tokens.weight)
                    .arg_u64(tok_pack01)
                    .arg_u64(tok_pack23)
                    .arg_u64(tok_pack45)
                    .arg_u64(tok_pack67)
                    .arg_u32(ctx.config.vocab_size as u32)
                    .arg_ptr(self.pre_fc_norm_embedding.weight)
                    .arg_ptr(normed_embed)
                    .arg_ptr(target_hiddens[0])
                    .arg_ptr(self.pre_fc_norm_hidden.weight)
                    .arg_ptr(normed_hidden)
                    .arg_u32(h as u32)
                    .arg_u32(r as u32)
                    .arg_f32(eps)
                    .arg_ptr(meta_base.offset(hidden_ptrs_off))
                    .launch(stream)?;
                } else {
                    if self.embed_from_argmax_k.0 != 0 {
                        spark_runtime::kernel_args::KernelLaunch::new(
                            ctx.gpu,
                            self.embed_from_argmax_k,
                        )
                        .grid([(h as u32).div_ceil(256), b as u32, 1])
                        .block([256, 1, 1])
                        .arg_ptr(meta_base.offset(last_tok_off))
                        .arg_ptr(self.embed_tokens.weight)
                        .arg_ptr(embed)
                        .arg_ptr(DevicePtr(0))
                        .arg_u32(h as u32)
                        .launch(stream)?;
                    } else {
                        for (row, &token) in last_tokens.iter().enumerate() {
                            let embed_src =
                                self.embed_tokens.weight.offset(token as usize * row_bytes);
                            ctx.gpu.copy_d2d_async(
                                embed_src,
                                embed.offset(row * row_bytes),
                                row_bytes,
                                stream,
                            )?;
                        }
                    }
                    if self.pre_fc_dual_norm_k.0 != 0 && h >= 1024 && r >= 1024 {
                        spark_runtime::kernel_args::KernelLaunch::new(
                            ctx.gpu,
                            self.pre_fc_dual_norm_k,
                        )
                        .grid([b as u32, 2, 1])
                        .block([1024, 1, 1])
                        .arg_ptr(embed)
                        .arg_ptr(self.pre_fc_norm_embedding.weight)
                        .arg_ptr(normed_embed)
                        .arg_ptr(target_hiddens[0])
                        .arg_ptr(self.pre_fc_norm_hidden.weight)
                        .arg_ptr(normed_hidden)
                        .arg_u32(h as u32)
                        .arg_u32(r as u32)
                        .arg_f32(eps)
                        .arg_ptr(meta_base.offset(hidden_ptrs_off))
                        .launch(stream)?;
                    } else {
                        ops::rms_norm(
                            ctx.gpu,
                            self.rms_norm_k,
                            embed,
                            &self.pre_fc_norm_embedding,
                            normed_embed,
                            b as u32,
                            h as u32,
                            eps,
                            stream,
                        )?;
                        let mut run_start = 0usize;
                        while run_start < b {
                            let mut run_len = 1usize;
                            while run_start + run_len < b
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
            } else {
                let prev_argmax = argmax_base.offset((step - 1) * b * 4);
                let token_out = scratch.offset(token_offset + (step - 1) * b * 4);
                spark_runtime::kernel_args::KernelLaunch::new(ctx.gpu, self.embed_from_argmax_k)
                    .grid([(h as u32).div_ceil(256), b as u32, 1])
                    .block([256, 1, 1])
                    .arg_ptr(prev_argmax)
                    .arg_ptr(self.embed_tokens.weight)
                    .arg_ptr(embed)
                    .arg_ptr(token_out)
                    .arg_u32(h as u32)
                    .launch(stream)?;
                if self.pre_fc_dual_norm_k.0 != 0 && h >= 1024 && r >= 1024 {
                    spark_runtime::kernel_args::KernelLaunch::new(ctx.gpu, self.pre_fc_dual_norm_k)
                        .grid([b as u32, 2, 1])
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
                        b as u32,
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
                        b as u32,
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
                    b,
                    h,
                    h,
                    stream,
                )?
                && self.try_coalesced_mtp_fc_hidden(
                    ctx.gpu,
                    normed_hidden,
                    hidden,
                    b * hc,
                    h,
                    stream,
                )? {
                spark_runtime::kernel_args::KernelLaunch::new(
                    ctx.gpu,
                    self.residual_add_broadcast_hc_k,
                )
                .grid([(h as u32).div_ceil(256), b as u32, 1])
                .block([256, 1, 1])
                .arg_ptr(hidden)
                .arg_ptr(embed_proj)
                .arg_u32(h as u32)
                .arg_u32(hc as u32)
                .launch(stream)?;
                static FC_COALESCED_LOGGED: std::sync::Once = std::sync::Once::new();
                FC_COALESCED_LOGGED.call_once(|| {
                    tracing::info!(
                        rows = b,
                        hc_rows = b * hc,
                        hidden = h,
                        "QWEN4_MTP_FC_COALESCED_NVFP4_ENGAGED"
                    );
                });
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
                    .grid([(h as u32).div_ceil(4), b as u32, 1])
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
                    ops::w4a16_gemv_batch_logits_exact_with(
                        ctx.gpu,
                        self.w4a16_exact_fc_kernels,
                        normed_embed,
                        &self.fc_embedding,
                        embed_proj,
                        b as u32,
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
                        (b * hc) as u32,
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
                        .grid([(h as u32).div_ceil(256), b as u32, 1])
                        .block([256, 1, 1])
                        .arg_ptr(hidden)
                        .arg_ptr(embed_proj)
                        .arg_u32(h as u32)
                        .arg_u32(hc as u32)
                        .launch(stream)?;
                    } else {
                        for j in 0..b {
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

            let step_meta = meta_base.offset(step * stride);
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
                max_blocks_per_seq: effective_mb as u32,
                num_seqs: b as u32,
            };
            let layer_ctx = ForwardContext {
                attn_metadata: Some(metadata),
                comm: None,
                graph_capture: false,
                ..*ctx
            };
            let step_seq_lens: Vec<usize> = base_seq_lens.iter().map(|&s| s + step).collect();
            {
                let mut typed_states: Vec<&mut Qwen4MtpState> = states
                    .iter_mut()
                    .map(|st| {
                        st.as_any_mut()
                            .downcast_mut::<Qwen4MtpState>()
                            .expect("validated above")
                    })
                    .collect();
                let mut block_tables: Vec<&mut Vec<u32>> = Vec::with_capacity(b);
                let mut disk_block_ids: Vec<&mut Vec<u32>> = Vec::with_capacity(b);
                let mut disk_last_offloaded: Vec<&mut Vec<u32>> = Vec::with_capacity(b);
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

            let sample_hiddens = self.final_mixer.prepare_batched(
                hidden,
                ctx.buffers.residual(),
                b,
                ctx.buffers,
                ctx.gpu,
                eps,
                stream,
            )?;
            let ran_coalesced_lm = if coalesced_lm_head_eligible {
                let ran = crate::layers::qwen4_fast_proj::try_coalesced_nvfp4_w4a16_gemm(
                    ctx.gpu,
                    sample_hiddens,
                    &self.lm_head_nvfp4,
                    logits,
                    b,
                    vocab,
                    h,
                    stream,
                )?;
                if ran {
                    static COALESCED_LM_LOGGED: std::sync::Once = std::sync::Once::new();
                    COALESCED_LM_LOGGED.call_once(|| {
                        tracing::info!(
                            rows = b,
                            vocab,
                            hidden = h,
                            "QWEN4_MTP_LM_HEAD_COALESCED_NVFP4_ENGAGED"
                        );
                    });
                }
                ran
            } else {
                false
            };
            if !ran_coalesced_lm {
                if should_use_mtp_lm_head_pipe(
                    mtp_lm_head_tc_pipe_enabled(),
                    b,
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
                        b as u32,
                        vocab as u32,
                        h as u32,
                        stream,
                    )?;
                    static PIPE_LOGGED: std::sync::Once = std::sync::Once::new();
                    PIPE_LOGGED.call_once(|| {
                        tracing::info!(
                            rows = b,
                            vocab,
                            hidden = h,
                            "QWEN4_MTP_LM_HEAD_TC_PIPE_ENGAGED kernel=w4a16_gemm_pipe"
                        );
                    });
                } else {
                    ops::w4a16_gemv_batch_logits_exact_with(
                        ctx.gpu,
                        self.w4a16_exact_fc_kernels,
                        sample_hiddens,
                        &self.lm_head_nvfp4,
                        logits,
                        b as u32,
                        vocab as u32,
                        h as u32,
                        stream,
                        ops::w4a16_exact_rt2_for_rows(
                            ops::w4a16_gemv_rt2_enabled(),
                            ops::w4a16_exact_rt2_min_rows(),
                            b as u32,
                        ),
                    )?;
                }
            }
            let step_argmax = argmax_base.offset(step * b * 4);
            let step_prob = prob_base.offset(step * b * 4);
            if let Some((shared_spec, batch_specs, layout)) = &batch_gumbel {
                gumbel_draft::select_chain_gumbel_batch(
                    ctx.gpu,
                    self.mtp_gumbel_argmax_k,
                    logits,
                    DevicePtr::NULL,
                    meta_base.offset(layout.seeds_off),
                    meta_base.offset(layout.step_pos_off(step, b)),
                    step_argmax,
                    want_prob.then_some(step_prob),
                    b,
                    vocab,
                    *shared_spec,
                    batch_specs,
                    step as u64,
                    stream,
                )?;
            } else if any_gumbel {
                for s in 0..b {
                    if let Some(spec) = gumbel_specs[s] {
                        gumbel_draft::select_chain_gumbel_step(
                            ctx.gpu,
                            self.mtp_gumbel_argmax_k,
                            logits.offset(s * vocab * 2),
                            DevicePtr::NULL,
                            step_argmax.offset(s * 4),
                            want_prob.then(|| step_prob.offset(s * 4)),
                            vocab,
                            spec,
                            spec.step0_abs_pos + step as u64,
                            stream,
                        )?;
                    } else if want_prob {
                        spark_runtime::kernel_args::KernelLaunch::new(ctx.gpu, self.argmax_prob_k)
                            .grid([1, 1, 1])
                            .block([1024, 1, 1])
                            .arg_ptr(logits.offset(s * vocab * 2))
                            .arg_ptr(step_argmax.offset(s * 4))
                            .arg_ptr(step_prob.offset(s * 4))
                            .arg_u32(vocab as u32)
                            .launch(stream)?;
                    } else {
                        ops::argmax_bf16(
                            ctx.gpu,
                            self.argmax_k,
                            logits.offset(s * vocab * 2),
                            step_argmax.offset(s * 4),
                            vocab as u32,
                            stream,
                        )?;
                    }
                }
            } else if want_prob {
                self.launch_argmax_prob_batch(
                    ctx.gpu,
                    logits,
                    step_argmax,
                    step_prob,
                    b,
                    vocab,
                    stream,
                )?;
            } else if self.argmax_rows_k.0 != 0 {
                ops::argmax_bf16_rows(
                    ctx.gpu,
                    self.argmax_rows_k,
                    logits,
                    step_argmax,
                    vocab as u32,
                    vocab as u32,
                    b as u32,
                    stream,
                )?;
            } else {
                for s in 0..b {
                    ops::argmax_bf16(
                        ctx.gpu,
                        self.argmax_k,
                        logits.offset(s * vocab * 2),
                        step_argmax.offset(s * 4),
                        vocab as u32,
                        stream,
                    )?;
                }
            }
        }
        drop(_tc_dense_guard);
        drop(kv_cache);

        let total_items = actual_computed * b;
        let mut bytes = vec![0u8; total_items * 4];
        let mut pbytes = vec![0u8; if want_prob { total_items * 4 } else { 0 }];
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
        let all_tokens: Vec<u32> = bytes
            .chunks_exact(4)
            .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();
        let all_probs: Vec<f32> = if want_prob {
            pbytes
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect()
        } else {
            Vec::new()
        };
        let mut out = Vec::with_capacity(b);
        for (s, st) in states.iter_mut().enumerate() {
            let state = st
                .as_any_mut()
                .downcast_mut::<Qwen4MtpState>()
                .expect("validated above");
            state.seq_len += actual_computed;
            state.last_num_drafted = actual_computed;
            let drafts: Vec<u32> = (0..actual_computed)
                .map(|step| all_tokens[step * b + s])
                .collect();
            state.last_drafts.clear();
            state.last_drafts.extend_from_slice(&drafts);
            if want_prob && policy.log {
                let probs: Vec<f32> = (0..actual_computed)
                    .map(|step| all_probs[step * b + s])
                    .collect();
                tracing::info!(
                    "MTP_DRAFT_CONF_BATCH seq={s} pos={} n={num_drafts} computed={actual_computed} \
                     drafts={drafts:?} p={probs:.4?} keep={}",
                    positions[s],
                    drafts.len()
                );
            }
            out.push(drafts);
        }
        Ok(Some(out))
    }

    fn after_verify(
        &self,
        num_accepted: usize,
        state: &mut dyn ProposerState,
        _stream: u64,
    ) -> Result<()> {
        let state = state
            .as_any_mut()
            .downcast_mut::<Qwen4MtpState>()
            .ok_or_else(|| anyhow::anyhow!("Invalid Qwen4 MTP proposer state"))?;
        let row_policy = rows_policy_logged();
        if row_policy == RowPolicy::Legacy {
            let trim = state.last_num_drafted.saturating_sub(num_accepted);
            state.seq_len = state.seq_len.saturating_sub(trim);
            state.last_num_drafted = 0;
            state.pending_replay = None;
            return Ok(());
        }
        // K7 (ATLAS_QWEN4_MTP_KEEP_LAST_ROW / _RECONDITION): keep the row of
        // the last committed token and optionally rewrite committed draft rows
        // from the verify's target hidden rows (see qwen4_mtp_rows).
        let computed = state.last_num_drafted;
        let recondition_active = qwen4_mtp_rows::recondition_engaged()
            && matches!(row_policy, RowPolicy::Recondition { .. });
        let staged =
            recondition_active && state.take_staged_frame(state.verify_frame, num_accepted)?;
        if computed == 0 && recondition_active && num_accepted > 0 && !state.last_drafts.is_empty()
        {
            let max_rows = match row_policy {
                RowPolicy::Recondition { max_rows } => max_rows,
                _ => 0,
            };
            let accepted = num_accepted.min(state.last_drafts.len());
            let replay = accepted.min(max_rows);
            state.pending_replay = (replay > 0).then_some(PendingReplay {
                owner: state.verify_frame,
                row_policy,
                computed: 0,
                staged_accepted: num_accepted,
                plan: AfterVerifyPlan {
                    keep_rows: 1,
                    replay_rows: if staged { replay } else { 0 },
                },
                accepted,
                staged_at_trim: staged,
            });
            return Ok(());
        }
        let plan = qwen4_mtp_rows::after_verify_plan(row_policy, computed, num_accepted, staged);
        state.seq_len = state.seq_len.saturating_sub(computed) + plan.keep_rows;
        let defer_staged_check = recondition_active && !staged && computed > 0 && num_accepted > 0;
        state.pending_replay = if staged {
            (plan.replay_rows > 0).then_some(PendingReplay {
                owner: state.verify_frame,
                row_policy,
                computed,
                staged_accepted: num_accepted,
                plan,
                accepted: num_accepted.min(computed),
                staged_at_trim: true,
            })
        } else if defer_staged_check {
            Some(PendingReplay {
                owner: state.verify_frame,
                row_policy,
                computed,
                staged_accepted: num_accepted,
                plan,
                accepted: num_accepted.min(computed),
                staged_at_trim: false,
            })
        } else {
            None
        };
        state.last_num_drafted = 0;
        if !defer_staged_check {
            record_row_stats(computed, num_accepted, plan);
        }
        Ok(())
    }

    fn prefill_last_k(
        &self,
        tokens: &[u32],
        target_hiddens: DevicePtr,
        base_position: usize,
        state: &mut dyn ProposerState,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let state = state
            .as_any_mut()
            .downcast_mut::<Qwen4MtpState>()
            .ok_or_else(|| anyhow::anyhow!("Invalid Qwen4 MTP proposer state"))?;
        if tokens.is_empty() {
            return Ok(());
        }
        ensure!(
            state.seq_len == 0,
            "Qwen4 MTP prompt replay requires an empty proposer cache"
        );
        let predict_start = prompt_tail_start(base_position, tokens.len())?;
        validate_rotary_span(
            &state.rotary_positions,
            predict_start,
            tokens.len(),
            ctx.config.mrope_interleaved,
            self.mrope_kernel_available,
            ctx.config.yarn_factor,
        )?;
        let stride = ctx.config.residual_width() * 2;
        let pairs: Vec<(usize, u32)> = crate::speculative::mtp_replay_pairs(
            tokens,
            crate::speculative::mtp_replay_next_token(),
        );
        let batch_enabled = mtp_batch_rows_enabled();
        let meta_base = ctx.buffers.scratch().offset(CHAIN_META_OFFSET);
        let mut cursor = 0usize;
        while cursor < pairs.len() {
            let rem = pairs.len() - cursor;
            // Avoid leaving a 1-row tail when `rem > 8 && rem % 8 == 1`.
            let chunk = if !batch_enabled || rem < 2 {
                1
            } else if rem > 8 && rem % 8 == 1 {
                5
            } else {
                rem.min(8)
            };
            if chunk >= 2 && self.forward_rows_admits(chunk) {
                let slice = &pairs[cursor..cursor + chunk];
                let b_tokens: Vec<u32> = slice.iter().map(|&(_, t)| t).collect();
                let b_hiddens: Vec<DevicePtr> = slice
                    .iter()
                    .map(|&(idx, _)| target_hiddens.offset(idx * stride))
                    .collect();
                let b_positions: Vec<usize> =
                    slice.iter().map(|&(idx, _)| predict_start + idx).collect();
                let _ = self.forward_rows(
                    &b_tokens,
                    &b_hiddens,
                    &b_positions,
                    meta_base,
                    state,
                    ctx,
                    stream,
                    None,
                    false,
                    None,
                )?;
                cursor += chunk;
            } else {
                let (index, token) = pairs[cursor];
                let _ = self.forward_one(
                    token,
                    target_hiddens.offset(index * stride),
                    predict_start + index,
                    state,
                    ctx,
                    stream,
                    None,
                    false,
                    self.mtp_qsa_active(),
                    None,
                )?;
                cursor += 1;
            }
        }
        state.last_num_drafted = 0;
        Ok(())
    }

    fn reserve_prompt_replay_blocks(
        &self,
        rows: usize,
        state: &mut dyn ProposerState,
    ) -> Result<usize> {
        let state = state
            .as_any_mut()
            .downcast_mut::<Qwen4MtpState>()
            .ok_or_else(|| anyhow::anyhow!("Invalid Qwen4 MTP proposer state"))?;
        ensure!(
            state.seq_len == 0,
            "Qwen4 MTP prompt replay requires an empty proposer cache"
        );
        // The allocations `step_metadata` makes for replay steps seq_len = 0..rows,
        // in the same order (each `alloc_block` pops the free list).
        let mut kv_cache = self.kv_cache.lock();
        let need = crate::model::host_bucket::mtp_replay_blocks(rows, kv_cache.block_size());
        let before = state.block_table.len();
        while state.block_table.len() < need {
            state.block_table.push(kv_cache.alloc_block()?);
        }
        Ok(state.block_table.len() - before)
    }

    fn free_state(&self, state: &mut dyn ProposerState) -> Result<()> {
        let state = state
            .as_any_mut()
            .downcast_mut::<Qwen4MtpState>()
            .ok_or_else(|| anyhow::anyhow!("Invalid Qwen4 MTP proposer state"))?;
        if !state.block_table.is_empty() {
            self.kv_cache.lock().free_blocks(&state.block_table);
            state.block_table.clear();
        }
        state.seq_len = 0;
        Ok(())
    }

    fn seqstate_export(
        &self,
        state: &dyn ProposerState,
    ) -> Result<Option<crate::model::seqstate::ProposerSeqState>> {
        let state = state
            .as_any()
            .downcast_ref::<Qwen4MtpState>()
            .ok_or_else(|| anyhow::anyhow!("Invalid Qwen4 MTP proposer state"))?;
        anyhow::ensure!(
            !state.has_unconsumed_verify_frame() && state.verify_frame.is_none(),
            "seqstate: native MTP has an unconsumed verify frame (not quiescent)"
        );
        anyhow::ensure!(
            state.rotary_positions.is_identity() && state.disk_block_ids.is_empty(),
            "seqstate: native MTP state with image rotary or disk blocks is not exportable"
        );
        let (seq_len, last_num_drafted, last_drafts) = state.snapshot_dedup_draft();
        let regions = crate::model::seqstate::engine::kv_block_regions(
            &self.kv_cache.lock(),
            &state.block_table,
            crate::model::seqstate::SectionKind::MtpKvK,
            crate::model::seqstate::SectionKind::MtpKvV,
        )?;
        Ok(Some(crate::model::seqstate::ProposerSeqState {
            meta: crate::model::seqstate::ProposerSeqMeta {
                seq_len,
                num_blocks: state.block_table.len(),
                last_num_drafted,
                last_drafts,
            },
            regions,
        }))
    }

    fn seqstate_import(
        &self,
        state: &mut dyn ProposerState,
        meta: &crate::model::seqstate::ProposerSeqMeta,
    ) -> Result<Option<Vec<crate::model::seqstate::StateRegion>>> {
        let state = state
            .as_any_mut()
            .downcast_mut::<Qwen4MtpState>()
            .ok_or_else(|| anyhow::anyhow!("Invalid Qwen4 MTP proposer state"))?;
        anyhow::ensure!(
            state.block_table.is_empty()
                && state.seq_len == 0
                && !state.has_unconsumed_verify_frame(),
            "seqstate: native MTP import needs a fresh proposer state"
        );
        let block_size = self.kv_cache.lock().block_size();
        anyhow::ensure!(
            meta.num_blocks >= meta.seq_len.div_ceil(block_size),
            "seqstate: {} MTP blocks cannot hold {} positions",
            meta.num_blocks,
            meta.seq_len
        );
        let mut kv = self.kv_cache.lock();
        let mut table = Vec::with_capacity(meta.num_blocks);
        for _ in 0..meta.num_blocks {
            match kv.alloc_block() {
                Ok(b) => table.push(b),
                Err(e) => {
                    kv.free_blocks(&table);
                    return Err(e.context("seqstate: native MTP block allocation"));
                }
            }
        }
        let regions = crate::model::seqstate::engine::kv_block_regions(
            &kv,
            &table,
            crate::model::seqstate::SectionKind::MtpKvK,
            crate::model::seqstate::SectionKind::MtpKvV,
        );
        drop(kv);
        let regions = match regions {
            Ok(r) => r,
            Err(e) => {
                self.kv_cache.lock().free_blocks(&table);
                return Err(e);
            }
        };
        state.block_table = table;
        state.rotary_positions = crate::traits::RotaryPositions::identity();
        state.apply_dedup_draft(meta.seq_len, meta.last_num_drafted, &meta.last_drafts);
        Ok(Some(regions))
    }

    fn release_rejection_resources(&self, gpu: &dyn GpuBackend) {
        self.graph_cache.lock().clear(gpu);
        if let Some(mut ring) = self.pinned_h2d_ring.lock().take() {
            ring.destroy(gpu);
        }
    }
}
